"""NYXARA · senses/watch.py — filesystem watching: she notices files change (📂).

A passive sense, not an action tool (that's ``agency/filesystem.py``): a bounded
snapshot of the watched directory trees — path → (mtime, size) via ``os.scandir`` —
diffed against the previous snapshot to yield **created / modified / deleted** paths.
Pure standard library; no ``watchdog``, no inotify, no extra thread — the perception
loop's own cadence *is* the poll, which works identically on every OS and filesystem.

Honest by construction: the first ``diff()`` primes and reports nothing (a change
needs a before); unreadable roots and permission errors are skipped, never raised;
depth and entry caps keep a huge tree from stalling the senses; and NYXARA's own
churn (her data dir's journals, memory, caches) is filtered out so she doesn't
startle at her own footsteps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["FileWatcher"]

# Directory names that are churn, not signal.
_IGNORE_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
                "node_modules", ".venv", "venv", ".tox", ".cache"}
# File suffixes that are churn, not signal.
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".swp", ".swx", ".tmp", "~", ".lock")


class FileWatcher:
    """Snapshot-diff watcher over one or more root directories."""

    def __init__(self, roots: Sequence[os.PathLike | str], *, depth: int = 3,
                 max_entries: int = 2000,
                 ignore_paths: Iterable[os.PathLike | str] = ()) -> None:
        self.roots = [Path(r) for r in roots]
        self.depth = max(1, int(depth))
        self.max_entries = max(16, int(max_entries))
        self._ignore_prefixes = tuple(str(Path(p)) for p in ignore_paths)
        self._last: Optional[Dict[str, Tuple[float, int]]] = None
        self.truncated = False          # True when a snapshot hit max_entries

    # ------------------------------------------------------------------ #
    def _ignored(self, path: str, name: str) -> bool:
        if name.startswith(".") and name not in (".", ".."):
            return True
        if name in _IGNORE_DIRS or name.endswith(_IGNORE_SUFFIXES):
            return True
        return any(path.startswith(pref) for pref in self._ignore_prefixes)

    def snapshot(self) -> Dict[str, Tuple[float, int]]:
        """Current bounded view: path → (mtime, size). Never raises."""
        out: Dict[str, Tuple[float, int]] = {}
        self.truncated = False
        stack: List[Tuple[Path, int]] = [(r, 0) for r in self.roots]
        while stack:
            root, level = stack.pop()
            try:
                entries = os.scandir(root)
            except OSError:
                continue
            with entries:
                for entry in entries:
                    if len(out) >= self.max_entries:
                        self.truncated = True
                        return out
                    if self._ignored(entry.path, entry.name):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if level + 1 < self.depth:
                                stack.append((Path(entry.path), level + 1))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            out[entry.path] = (st.st_mtime, st.st_size)
                    except OSError:
                        continue
        return out

    def diff(self) -> Tuple[List[str], List[str], List[str]]:
        """``(created, modified, deleted)`` since the last call; first call primes."""
        cur = self.snapshot()
        prev, self._last = self._last, cur
        if prev is None:
            return [], [], []
        created = sorted(p for p in cur if p not in prev)
        deleted = sorted(p for p in prev if p not in cur)
        modified = sorted(p for p in cur if p in prev and cur[p] != prev[p])
        return created, modified, deleted


# --------------------------------------------------------------------------- #
# Self-test / demo — a real temp tree, really changed, really noticed
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA file-watch self-test (real files in a temp dir)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.txt").write_text("one")
        w = FileWatcher([root])
        assert w.diff() == ([], [], []), "first diff must prime, not report"

        (root / "b.txt").write_text("two")                    # create
        (root / "a.txt").write_text("one, but longer now")    # modify
        created, modified, deleted = w.diff()
        print(f"\ncreated             : {[Path(p).name for p in created]}")
        print(f"modified            : {[Path(p).name for p in modified]}")
        assert [Path(p).name for p in created] == ["b.txt"]
        assert [Path(p).name for p in modified] == ["a.txt"]

        (root / "b.txt").unlink()                             # delete
        created, modified, deleted = w.diff()
        print(f"deleted             : {[Path(p).name for p in deleted]}")
        assert [Path(p).name for p in deleted] == ["b.txt"]

        # churn is invisible: dotfiles, caches, ignored suffixes
        (root / ".hidden").write_text("x")
        (root / "junk.pyc").write_text("x")
        assert w.diff() == ([], [], []), "churn must be filtered out"

    print("\nALL SELF-TESTS PASSED ✓")
