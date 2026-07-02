"""NYXARA · agency/filesystem.py — a real, whole-disk filesystem faculty (🗄).

NYXARA is *authorised* to touch the whole disk (``grant_full_operational_control`` blesses
``FS_READ``/``FS_WRITE``/``FS_DELETE`` with an empty scope — any target). What was missing was a
faculty that could actually *operate* it: the default toolset only had a truncated text read, a
flat directory list, and an overwrite-only write, and the ``FS_DELETE`` capability had no tool at
all. This module is that faculty — a single, governed, pure-Python engine that reads, writes,
lists, walks, globs, searches, copies, moves, and deletes anywhere on disk (subject to the OS's own
permissions).

NYXARA does the work **herself** — every operation is plain ``os``/``pathlib``/``shutil`` code, not
a shell-out and never delegated to the LLM. Safety is by construction and configurable:

* **scoping** — with ``whole_disk`` on (the default) there is no path restriction; turn it off to
  confine every operation under ``root``;
* **deny / allow globs** — a deny list fences off sensitive paths (e.g. the vault's key store) even
  when the whole disk is otherwise reachable, and a non-empty allow list flips to allow-only;
* **symlink control** — ``follow_symlinks`` decides whether a symlink is resolved to its target;
* **hard caps** — ``max_read_bytes`` bounds every read, and ``max_walk_depth`` / ``max_results`` /
  ``max_search_matches`` bound recursion so a walk or content search can never run away or exhaust
  memory.

The engine only enforces its *own* path guards; it does **not** decide whether NYXARA may act — the
kernel's capability/risk/authority pipeline (:mod:`nyxara.agency.tools`, :mod:`nyxara.agency.permissions`)
still gates every tool call, and ``/scram`` + oversight + corrigibility remain outside it entirely.
"""

from __future__ import annotations

import base64
import fnmatch
import os
import re
import shutil
import stat as stat_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Filesystem", "FilesystemError"]

# The engine's own defaults. They intentionally mirror :class:`FilesystemConfig`
# (nyxara/kernel/config.py) so a bare ``Filesystem()`` — used in tests and by any caller
# without settings — behaves like the shipped configuration: whole-disk reach, sane caps.
_DEFAULT_MAX_READ_BYTES = 5_000_000
_DEFAULT_MAX_WALK_DEPTH = 25
_DEFAULT_MAX_RESULTS = 5_000
_DEFAULT_MAX_SEARCH_MATCHES = 1_000


class FilesystemError(PermissionError):
    """Raised when a path is refused by the engine's own scope / deny-glob guards.

    Subclasses :class:`PermissionError` so callers that already handle OS permission
    failures treat an out-of-scope path the same way. The kernel's tool pipeline turns any
    exception raised here into an ``ok=False`` :class:`~nyxara.agency.tools.ToolResult`
    (errors-as-data), so a refused path never crashes the act stage.
    """


@dataclass
class Filesystem:
    """A governed, pure-Python engine for real whole-disk operations.

    Construct directly with keyword overrides, or via :meth:`from_config` to adopt a
    :class:`~nyxara.kernel.config.FilesystemConfig`. Every method normalises and guards its
    path(s) before touching reality; guards raise :class:`FilesystemError` (a
    ``PermissionError``) rather than silently widening reach.
    """

    whole_disk: bool = True
    root: Optional[Path] = None
    max_read_bytes: int = _DEFAULT_MAX_READ_BYTES
    follow_symlinks: bool = False
    allow_globs: List[str] = field(default_factory=list)
    deny_globs: List[str] = field(default_factory=list)
    max_walk_depth: int = _DEFAULT_MAX_WALK_DEPTH
    max_results: int = _DEFAULT_MAX_RESULTS
    max_search_matches: int = _DEFAULT_MAX_SEARCH_MATCHES

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(os.path.realpath(os.path.expanduser(str(self.root))))

    @classmethod
    def from_config(cls, cfg: Any) -> "Filesystem":
        """Build from a :class:`FilesystemConfig` (or any object exposing the same fields)."""
        if cfg is None:
            return cls()
        return cls(
            whole_disk=bool(getattr(cfg, "whole_disk", True)),
            root=getattr(cfg, "root", None),
            max_read_bytes=int(getattr(cfg, "max_read_bytes", _DEFAULT_MAX_READ_BYTES)),
            follow_symlinks=bool(getattr(cfg, "follow_symlinks", False)),
            allow_globs=list(getattr(cfg, "allow_globs", []) or []),
            deny_globs=list(getattr(cfg, "deny_globs", []) or []),
            max_walk_depth=int(getattr(cfg, "max_walk_depth", _DEFAULT_MAX_WALK_DEPTH)),
            max_results=int(getattr(cfg, "max_results", _DEFAULT_MAX_RESULTS)),
            max_search_matches=int(getattr(cfg, "max_search_matches", _DEFAULT_MAX_SEARCH_MATCHES)),
        )

    # ------------------------------------------------------------------ #
    # Path normalisation & guards
    # ------------------------------------------------------------------ #
    def _norm(self, path: str) -> str:
        """Expand ``~``, absolutise, and (when ``follow_symlinks``) resolve symlinks."""
        p = os.path.expanduser(str(path))
        if self.follow_symlinks:
            return os.path.realpath(p)
        # Resolve the parent so scope checks are stable, but leave the final component
        # unresolved so a symlinked leaf isn't silently followed to its target.
        p = os.path.abspath(p)
        parent = os.path.realpath(os.path.dirname(p))
        return os.path.join(parent, os.path.basename(p))

    def _guard(self, path: str) -> str:
        """Normalise ``path`` and enforce scope + deny/allow globs. Returns the safe path."""
        resolved = self._norm(path)
        if not self.whole_disk and self.root is not None:
            try:
                common = os.path.commonpath([str(self.root), resolved])
            except ValueError:
                # Different drives (Windows) or a mix of abs/rel — treat as outside root.
                common = ""
            if common != str(self.root):
                raise FilesystemError(
                    f"path {resolved!r} is outside the configured root {str(self.root)!r}")
        for pat in self.deny_globs:
            if fnmatch.fnmatch(resolved, pat):
                raise FilesystemError(f"path {resolved!r} is denied by deny-glob {pat!r}")
        if self.allow_globs and not any(fnmatch.fnmatch(resolved, pat) for pat in self.allow_globs):
            raise FilesystemError(f"path {resolved!r} matches no allow-glob")
        return resolved

    @staticmethod
    def _kind(st: os.stat_result) -> str:
        if stat_mod.S_ISDIR(st.st_mode):
            return "dir"
        if stat_mod.S_ISLNK(st.st_mode):
            return "symlink"
        if stat_mod.S_ISREG(st.st_mode):
            return "file"
        return "other"

    def _entry(self, path: str) -> Dict[str, Any]:
        st = os.lstat(path) if not self.follow_symlinks else os.stat(path)
        return {
            "path": path,
            "name": os.path.basename(path.rstrip(os.sep)) or path,
            "type": "symlink" if stat_mod.S_ISLNK(os.lstat(path).st_mode) else self._kind(st),
            "size": int(st.st_size),
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "mode": oct(stat_mod.S_IMODE(st.st_mode)),
        }

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def read_text(self, path: str, *, offset: int = 0, length: Optional[int] = None,
                  encoding: str = "utf-8") -> str:
        """Read a text file, byte-bounded by ``max_read_bytes`` (and optional ``length``)."""
        safe = self._guard(path)
        cap = self.max_read_bytes if length is None else min(int(length), self.max_read_bytes)
        with open(safe, "rb") as f:
            if offset:
                f.seek(int(offset))
            raw = f.read(max(0, cap))
        return raw.decode(encoding, errors="replace")

    def read_bytes(self, path: str, *, offset: int = 0, length: Optional[int] = None) -> str:
        """Read raw bytes (byte-bounded) and return them base64-encoded (binary-safe)."""
        safe = self._guard(path)
        cap = self.max_read_bytes if length is None else min(int(length), self.max_read_bytes)
        with open(safe, "rb") as f:
            if offset:
                f.seek(int(offset))
            raw = f.read(max(0, cap))
        return base64.b64encode(raw).decode("ascii")

    def stat(self, path: str) -> Dict[str, Any]:
        """Metadata for one path (type, size, mtime, POSIX mode)."""
        return self._entry(self._guard(path))

    def exists(self, path: str) -> bool:
        return os.path.lexists(self._guard(path))

    # ------------------------------------------------------------------ #
    # List / search
    # ------------------------------------------------------------------ #
    def list_dir(self, path: str = ".") -> List[Dict[str, Any]]:
        """Rich directory listing: one metadata entry per child, sorted by name."""
        safe = self._guard(path)
        out: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(safe)):
            child = os.path.join(safe, name)
            try:
                out.append(self._entry(child))
            except OSError:
                out.append({"path": child, "name": name, "type": "unknown"})
        return out

    def walk(self, path: str = ".", *, max_depth: Optional[int] = None,
             max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Recursively list a tree, depth- and count-bounded (breadth of the whole subtree)."""
        safe = self._guard(path)
        depth_cap = self.max_walk_depth if max_depth is None else int(max_depth)
        count_cap = self.max_results if max_results is None else int(max_results)
        base_depth = safe.rstrip(os.sep).count(os.sep)
        out: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(safe, followlinks=self.follow_symlinks):
            depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
            if depth >= depth_cap:
                dirnames[:] = []  # stop descending past the depth cap
            for name in sorted(dirnames) + sorted(filenames):
                if len(out) >= count_cap:
                    return out
                try:
                    out.append(self._entry(os.path.join(dirpath, name)))
                except OSError:
                    continue
        return out

    def glob(self, pattern: str, *, root: str = ".", max_results: Optional[int] = None,
             recursive: bool = True) -> List[str]:
        """Glob for paths under ``root`` (``**`` supported), count-bounded."""
        safe_root = self._guard(root)
        count_cap = self.max_results if max_results is None else int(max_results)
        out: List[str] = []
        for match in Path(safe_root).glob(pattern):
            try:
                self._guard(str(match))
            except FilesystemError:
                continue
            out.append(str(match))
            if len(out) >= count_cap:
                break
        return sorted(out)

    def search(self, root: str, pattern: str, *, regex: bool = False,
               max_matches: Optional[int] = None, max_files: Optional[int] = None,
               include: str = "*") -> List[Dict[str, Any]]:
        """Grep file *contents* under ``root`` for ``pattern`` (literal or regex).

        Returns ``{path, line, text}`` hits, bounded by ``max_matches`` and the number of
        files scanned (``max_files``). Binary/undecodable bytes are read as replacement text
        so the scan never crashes; each file is read up to ``max_read_bytes``.
        """
        safe_root = self._guard(root)
        match_cap = self.max_search_matches if max_matches is None else int(max_matches)
        file_cap = self.max_results if max_files is None else int(max_files)
        rx = re.compile(pattern) if regex else None
        hits: List[Dict[str, Any]] = []
        scanned = 0
        for dirpath, _dirnames, filenames in os.walk(safe_root, followlinks=self.follow_symlinks):
            for name in sorted(filenames):
                if not fnmatch.fnmatch(name, include):
                    continue
                if scanned >= file_cap:
                    return hits
                fpath = os.path.join(dirpath, name)
                try:
                    self._guard(fpath)
                    with open(fpath, "rb") as f:
                        text = f.read(self.max_read_bytes).decode("utf-8", errors="replace")
                except (OSError, FilesystemError):
                    continue
                scanned += 1
                for lineno, line in enumerate(text.splitlines(), start=1):
                    found = rx.search(line) if rx is not None else (pattern in line)
                    if found:
                        hits.append({"path": fpath, "line": lineno, "text": line[:500]})
                        if len(hits) >= match_cap:
                            return hits
        return hits

    def disk_usage(self, path: str = ".") -> Dict[str, int]:
        """Total / used / free bytes for the filesystem holding ``path``."""
        safe = self._guard(path)
        usage = shutil.disk_usage(safe)
        return {"total": int(usage.total), "used": int(usage.used), "free": int(usage.free)}

    # ------------------------------------------------------------------ #
    # Mutate
    # ------------------------------------------------------------------ #
    def write_text(self, path: str, content: str, *, overwrite: bool = True,
                   encoding: str = "utf-8") -> Dict[str, Any]:
        """Write text to a file, creating parent directories as needed."""
        safe = self._guard(path)
        if not overwrite and os.path.exists(safe):
            raise FilesystemError(f"refusing to overwrite existing path {safe!r}")
        os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
        data = content.encode(encoding)
        with open(safe, "wb") as f:
            f.write(data)
        return {"path": safe, "bytes": len(data)}

    def write_bytes(self, path: str, content_b64: str, *, overwrite: bool = True) -> Dict[str, Any]:
        """Write raw bytes (base64-encoded input) to a file, creating parents as needed."""
        safe = self._guard(path)
        if not overwrite and os.path.exists(safe):
            raise FilesystemError(f"refusing to overwrite existing path {safe!r}")
        os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
        data = base64.b64decode(content_b64)
        with open(safe, "wb") as f:
            f.write(data)
        return {"path": safe, "bytes": len(data)}

    def append_text(self, path: str, content: str, *, encoding: str = "utf-8") -> Dict[str, Any]:
        """Append text to a file (create it if absent)."""
        safe = self._guard(path)
        os.makedirs(os.path.dirname(safe) or ".", exist_ok=True)
        data = content.encode(encoding)
        with open(safe, "ab") as f:
            f.write(data)
        return {"path": safe, "appended_bytes": len(data)}

    def make_dir(self, path: str, *, parents: bool = True) -> Dict[str, Any]:
        """Create a directory (``parents`` makes the whole chain; idempotent when parents)."""
        safe = self._guard(path)
        if parents:
            os.makedirs(safe, exist_ok=True)
        else:
            os.mkdir(safe)
        return {"path": safe, "created": True}

    def move(self, src: str, dst: str) -> Dict[str, Any]:
        """Move / rename a file or directory."""
        safe_src = self._guard(src)
        safe_dst = self._guard(dst)
        os.makedirs(os.path.dirname(safe_dst) or ".", exist_ok=True)
        result = shutil.move(safe_src, safe_dst)
        return {"src": safe_src, "dst": str(result)}

    def copy(self, src: str, dst: str) -> Dict[str, Any]:
        """Copy a file (metadata preserved) or a whole directory tree."""
        safe_src = self._guard(src)
        safe_dst = self._guard(dst)
        if os.path.isdir(safe_src):
            shutil.copytree(safe_src, safe_dst, symlinks=not self.follow_symlinks)
        else:
            os.makedirs(os.path.dirname(safe_dst) or ".", exist_ok=True)
            shutil.copy2(safe_src, safe_dst)
        return {"src": safe_src, "dst": safe_dst}

    # ------------------------------------------------------------------ #
    # Delete — finally exercises the FS_DELETE capability
    # ------------------------------------------------------------------ #
    def delete(self, path: str, *, recursive: bool = False) -> Dict[str, Any]:
        """Delete a file, an empty directory, or (``recursive``) a whole tree."""
        safe = self._guard(path)
        if os.path.isdir(safe) and not os.path.islink(safe):
            if recursive:
                shutil.rmtree(safe)
            else:
                os.rmdir(safe)  # refuses non-empty dirs, by design
        else:
            os.remove(safe)
        return {"path": safe, "deleted": True, "recursive": bool(recursive)}
