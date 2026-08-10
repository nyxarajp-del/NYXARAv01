"""NYXARA · nyx/atlas.py — the map of her own body (🗺, NYX V.03).

Every faculty that edits code has, until now, worked from a file someone handed it. ``author.py``
writes one function into one module; ``omni.py`` lowers one function it was pointed at;
``self_debugger`` parses a traceback it was given. None of them can answer the question a
competent engineer answers first: **where does this thing live, and what breaks if I change it?**

This module is that answer. It builds a real index of a Python tree — parsed with :mod:`ast`,
**never imported** — and keeps it fresh incrementally:

* **Symbol table** — every function, class, method and module-level constant, with its
  ``file:line``, signature, decorators and docstring summary. :meth:`Atlas.find` resolves a bare
  name (``NyxBrain``) or a dotted one (``nyxara.nyx.brain.NyxBrain``).
* **Import graph** — who imports whom, module-level only, so the deliberate lazy imports this
  codebase uses to break cycles are not miscounted as dependencies.
* **Reverse call graph** — who calls this. Python is dynamic, so a call is indexed by the *name*
  invoked, and a name with several definitions returns **all** of them, marked ambiguous. It never
  guesses which one; :meth:`Atlas.callers` reports what it actually knows.
* **Blast radius** — :meth:`Atlas.affected_tests` walks the import graph backwards from a changed
  module to the test files that reach it, so a change can be checked by the tests that matter.

**Installed packages too.** :meth:`Atlas.index_package` resolves a distribution with
:func:`importlib.util.find_spec` and then walks its directory **on disk with ast**. The top-level
spec lookup is metadata-only (it does not execute the package), and no submodule is ever imported —
so she can answer "what is ``pydantic.BaseModel``?" from the source that is really installed on this
machine, not from a memory of it, and without running a line of third-party code.

Honest framing, enforced in code rather than described:

* **Lexical, not semantic.** This is an AST index, not a type checker. A call through a variable
  (``handler()``), a ``getattr`` dispatch, or a C-extension boundary is invisible to it. Every
  result carries how it was found, and :meth:`Atlas.find` on an unindexed name returns nothing
  rather than a guess.
* **It never imports what it indexes.** No target module's code runs — that is what makes indexing
  an untrusted or half-broken tree safe.
* **Incremental and deterministic.** A file is re-parsed only when ``(mtime, size, sha256)`` moved;
  the same tree yields the same index on every machine.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Symbol", "ModuleEntry", "CallSite", "Atlas", "atlas_home"]

# Directories that are never part of a source index.
_SKIP_DIRS = frozenset({
    "__pycache__", ".git", ".hg", ".svn", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "node_modules", ".eggs", "build", "dist",
})

_KIND_FUNCTION = "function"
_KIND_METHOD = "method"
_KIND_CLASS = "class"
_KIND_CONSTANT = "constant"


def atlas_home() -> Path:
    """Where the persisted index lives. Overridable for tests via ``NYXARA_HOME``."""
    base = os.environ.get("NYXARA_HOME") or os.path.join(os.path.expanduser("~"), ".nyxara")
    return Path(base) / "nyx" / "atlas"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Symbol:
    """One named thing defined in the tree, with where it is and what it looks like."""

    name: str                     # bare name, e.g. "NyxBrain"
    qualname: str                 # dotted within its module, e.g. "NyxBrain.think"
    kind: str                     # function | method | class | constant
    module: str                   # e.g. "nyxara.nyx.brain"
    path: str                     # absolute file path
    line: int                     # 1-indexed definition line
    end_line: int = 0
    signature: str = ""           # "(self, stimulus: str, *, remember: bool = True)"
    doc: str = ""                 # first line of the docstring, if any
    decorators: Tuple[str, ...] = ()
    parent: str = ""              # enclosing class qualname, for methods

    @property
    def dotted(self) -> str:
        """Fully-qualified name: ``nyxara.nyx.brain.NyxBrain.think``."""
        return f"{self.module}.{self.qualname}" if self.module else self.qualname

    @property
    def location(self) -> str:
        """``path:line`` — the clickable form."""
        return f"{self.path}:{self.line}"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "qualname": self.qualname, "kind": self.kind,
                "module": self.module, "path": self.path, "line": self.line,
                "end_line": self.end_line, "signature": self.signature, "doc": self.doc,
                "decorators": list(self.decorators), "parent": self.parent}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Symbol":
        return cls(name=d["name"], qualname=d["qualname"], kind=d["kind"], module=d["module"],
                   path=d["path"], line=int(d["line"]), end_line=int(d.get("end_line") or 0),
                   signature=d.get("signature", ""), doc=d.get("doc", ""),
                   decorators=tuple(d.get("decorators") or ()), parent=d.get("parent", ""))


@dataclass
class CallSite:
    """A call, indexed by the *name* invoked — never by a guessed target."""

    caller: str                   # qualname of the enclosing def, or "<module>"
    callee: str                   # the bare name invoked, e.g. "think" or "NyxBrain"
    module: str
    path: str
    line: int

    def to_dict(self) -> Dict[str, Any]:
        return {"caller": self.caller, "callee": self.callee, "module": self.module,
                "path": self.path, "line": self.line}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CallSite":
        return cls(caller=d["caller"], callee=d["callee"], module=d["module"],
                   path=d["path"], line=int(d["line"]))


@dataclass
class ModuleEntry:
    """One indexed module: its fingerprint, what it defines, and what it reaches for."""

    module: str
    path: str
    sha256: str = ""
    mtime: float = 0.0
    size: int = 0
    loc: int = 0
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)      # module-level import targets
    calls: List[CallSite] = field(default_factory=list)
    parse_error: str = ""                                  # non-empty ⇒ file did not parse

    def to_dict(self) -> Dict[str, Any]:
        return {"module": self.module, "path": self.path, "sha256": self.sha256,
                "mtime": self.mtime, "size": self.size, "loc": self.loc,
                "symbols": [s.to_dict() for s in self.symbols],
                "imports": list(self.imports),
                "calls": [c.to_dict() for c in self.calls],
                "parse_error": self.parse_error}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleEntry":
        return cls(module=d["module"], path=d["path"], sha256=d.get("sha256", ""),
                   mtime=float(d.get("mtime") or 0.0), size=int(d.get("size") or 0),
                   loc=int(d.get("loc") or 0),
                   symbols=[Symbol.from_dict(s) for s in d.get("symbols") or ()],
                   imports=list(d.get("imports") or ()),
                   calls=[CallSite.from_dict(c) for c in d.get("calls") or ()],
                   parse_error=d.get("parse_error", ""))


# --------------------------------------------------------------------------- #
# The per-file AST walk
# --------------------------------------------------------------------------- #
class _FileIndexer(ast.NodeVisitor):
    """Collect symbols, module-level imports and call sites from one parsed file.

    Written as an explicit visitor rather than ``ast.walk`` because both the *enclosing scope*
    (which def a call sits in) and *module-level-ness* (which imports form the dependency graph)
    are structural facts a flat walk throws away.
    """

    def __init__(self, module: str, path: str, pkg_parts: Sequence[str]) -> None:
        self.module = module
        self.path = path
        self.pkg_parts = list(pkg_parts)
        self.symbols: List[Symbol] = []
        self.imports: List[str] = []
        self.calls: List[CallSite] = []
        self._scope: List[str] = []      # enclosing def/class qualname parts
        self._class_stack: List[str] = []
        self._depth = 0                  # >0 ⇒ inside a def/class body

    # ---- helpers ---- #
    def _qual(self, name: str) -> str:
        return ".".join([*self._scope, name]) if self._scope else name

    def _here(self) -> str:
        return ".".join(self._scope) if self._scope else "<module>"

    @staticmethod
    def _doc_first_line(node: ast.AST) -> str:
        try:
            doc = ast.get_docstring(node)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return ""
        if not doc:
            return ""
        return doc.strip().splitlines()[0].strip()[:200]

    @staticmethod
    def _decorators(node: Any) -> Tuple[str, ...]:
        out: List[str] = []
        for dec in getattr(node, "decorator_list", ()) or ():
            try:
                out.append(ast.unparse(dec))
            except Exception:  # noqa: BLE001 — very old ast, or an exotic node
                out.append(getattr(dec, "id", "") or getattr(dec, "attr", "") or "?")
        return tuple(o for o in out if o)

    @staticmethod
    def _signature(node: Any) -> str:
        try:
            return f"({ast.unparse(node.args)})"
        except Exception:  # noqa: BLE001
            return "(...)"

    # ---- imports (module level only) ---- #
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        if self._depth == 0:
            for alias in node.names:
                self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if self._depth == 0:
            mod = node.module or ""
            if node.level:  # relative — resolve against this module's package
                base = self.pkg_parts[:-node.level] if node.level <= len(self.pkg_parts) else []
                mod = ".".join([*base, mod]) if mod else ".".join(base)
            if mod:
                self.imports.append(mod)
                for alias in node.names:
                    # `from pkg import sub` may name a submodule; offer the specific candidate too.
                    self.imports.append(f"{mod}.{alias.name}")
        self.generic_visit(node)

    # ---- definitions ---- #
    def _visit_def(self, node: Any, kind: str) -> None:
        name = node.name
        sym = Symbol(
            name=name,
            qualname=self._qual(name),
            kind=_KIND_METHOD if (kind == _KIND_FUNCTION and self._class_stack) else kind,
            module=self.module,
            path=self.path,
            line=int(getattr(node, "lineno", 0) or 0),
            end_line=int(getattr(node, "end_lineno", 0) or 0),
            signature=self._signature(node) if kind == _KIND_FUNCTION else "",
            doc=self._doc_first_line(node),
            decorators=self._decorators(node),
            parent=self._class_stack[-1] if self._class_stack else "",
        )
        self.symbols.append(sym)

        self._scope.append(name)
        if kind == _KIND_CLASS:
            self._class_stack.append(sym.qualname)
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1
        if kind == _KIND_CLASS:
            self._class_stack.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_def(node, _KIND_FUNCTION)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_def(node, _KIND_FUNCTION)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_def(node, _KIND_CLASS)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Module-level UPPER_CASE names are the constants worth being able to locate.
        if self._depth == 0:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.symbols.append(Symbol(
                        name=target.id, qualname=target.id, kind=_KIND_CONSTANT,
                        module=self.module, path=self.path,
                        line=int(getattr(node, "lineno", 0) or 0),
                        end_line=int(getattr(node, "end_lineno", 0) or 0)))
        self.generic_visit(node)

    # ---- calls ---- #
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        callee = _callee_name(node.func)
        if callee:
            self.calls.append(CallSite(
                caller=self._here(), callee=callee, module=self.module, path=self.path,
                line=int(getattr(node, "lineno", 0) or 0)))
        self.generic_visit(node)


def _callee_name(func: ast.AST) -> str:
    """The bare name being invoked, or "" when the target is an expression we cannot name.

    ``f()`` → ``f``; ``obj.method()`` → ``method``; ``a.b.c()`` → ``c``; ``funcs[i]()`` → ``""``.
    Returning "" for the dynamic cases is the honest answer — a name we cannot read is not
    a name we should invent.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
class Atlas:
    """An incremental, never-importing index of a Python tree.

    ``root`` is the package directory (``…/nyxara``); ``pkg`` is the dotted name it is rooted at.
    Everything read is required to sit under a registered root — an index is a read of a tree, and
    a traversal that escapes it is a bug, not a feature.
    """

    def __init__(self, root: Optional[Any] = None, *, pkg: str = "nyxara",
                 store: Optional[Any] = None, max_files: int = 20_000,
                 include_tests: bool = True) -> None:
        self.root = Path(root).resolve() if root is not None else _package_root()
        self.pkg = pkg
        self.max_files = int(max_files)
        self.store = Path(store) if store is not None else atlas_home()
        # Trees to index. More than one because the thing she most needs to find — the tests that
        # cover a change — lives *beside* the package, not inside it. An index of the package
        # alone can never answer "what tests reach this?", which is the whole point of
        # :meth:`affected_tests`, so a sibling ``tests/`` is picked up by default.
        self.trees: List[Tuple[Path, str]] = [(self.root, self.pkg)]
        if include_tests:
            sibling = self.root.parent / "tests"
            if sibling.is_dir():
                self.trees.append((sibling.resolve(), "tests"))
        self.modules: Dict[str, ModuleEntry] = {}
        # Derived indexes, rebuilt by _reindex()
        self._by_name: Dict[str, List[Symbol]] = {}
        self._callers: Dict[str, List[CallSite]] = {}
        self._importers: Dict[str, Set[str]] = {}
        self.last_refresh: float = 0.0
        self.last_scanned: int = 0
        self.last_parsed: int = 0

    # ------------------------------------------------------------------ #
    # Scanning
    # ------------------------------------------------------------------ #
    def add_tree(self, root: Any, pkg: str) -> "Atlas":
        """Index another tree (typically ``tests/``) under its own dotted prefix.

        Its modules keep that prefix, so they can never collide with the primary package's.
        Returns self so calls chain.
        """
        resolved = Path(root).resolve()
        if not any(r == resolved for r, _ in self.trees):
            self.trees.append((resolved, str(pkg)))
        return self

    def _tree_of(self, path: Path) -> Optional[Tuple[Path, str]]:
        """The registered tree containing ``path`` — the longest matching root wins, so a nested
        tree is attributed to itself rather than to its parent."""
        best: Optional[Tuple[Path, str]] = None
        try:
            resolved = path.resolve()
        except OSError:
            return None
        for root, pkg in self.trees:
            try:
                if os.path.commonpath([str(root), str(resolved)]) == str(root):
                    if best is None or len(str(root)) > len(str(best[0])):
                        best = (root, pkg)
            except (ValueError, OSError):
                continue
        return best

    def _within(self, path: Path) -> bool:
        """True when ``path`` really sits under one of the registered roots (no symlink escape)."""
        return self._tree_of(path) is not None

    def _module_name(self, path: Path) -> str:
        tree = self._tree_of(path)
        if tree is None:
            raise ValueError(f"{path} is outside every indexed tree")
        root, pkg = tree
        rel = path.resolve().relative_to(root).with_suffix("")
        parts = [pkg, *rel.parts] if pkg else list(rel.parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _source_files(self) -> List[Path]:
        out: List[Path] = []
        for root, _pkg in self.trees:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames
                               if d not in _SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if fn.endswith(".py"):
                        out.append(Path(dirpath) / fn)
                        if len(out) >= self.max_files:
                            return sorted(set(out))
        return sorted(set(out))

    @staticmethod
    def _fingerprint(path: Path) -> Tuple[float, int, str]:
        st = path.stat()
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return st.st_mtime, st.st_size, h.hexdigest()

    def refresh(self, *, force: bool = False) -> Dict[str, Any]:
        """Bring the index up to date. Re-parses only what actually moved."""
        started = time.time()
        scanned = parsed = 0
        seen: Set[str] = set()

        for path in self._source_files():
            if not self._within(path):
                continue
            scanned += 1
            try:
                module = self._module_name(path)
            except ValueError:
                continue
            seen.add(module)
            try:
                mtime, size, sha = self._fingerprint(path)
            except OSError:
                continue
            prior = self.modules.get(module)
            if (not force and prior is not None and prior.sha256 == sha
                    and prior.size == size and prior.path == str(path)):
                continue  # unchanged — keep the parsed entry
            self.modules[module] = self._index_file(path, module, mtime, size, sha)
            parsed += 1

        for gone in set(self.modules) - seen:
            del self.modules[gone]

        self._reindex()
        self.last_refresh = time.time()
        self.last_scanned, self.last_parsed = scanned, parsed
        return {"scanned": scanned, "parsed": parsed, "modules": len(self.modules),
                "symbols": sum(len(m.symbols) for m in self.modules.values()),
                "seconds": round(self.last_refresh - started, 3)}

    def _index_file(self, path: Path, module: str, mtime: float, size: int,
                    sha: str) -> ModuleEntry:
        entry = ModuleEntry(module=module, path=str(path), sha256=sha, mtime=mtime, size=size)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            entry.parse_error = f"unreadable: {exc}"
            return entry
        entry.loc = text.count("\n") + 1
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            # A file that does not parse is recorded as such, not silently dropped: a half-broken
            # tree is exactly when she most needs to know which file is the broken one.
            entry.parse_error = f"syntax error at line {exc.lineno}: {exc.msg}"
            return entry
        except Exception as exc:  # noqa: BLE001
            entry.parse_error = f"parse failed: {exc}"
            return entry

        indexer = _FileIndexer(module, str(path), module.split("."))
        try:
            indexer.visit(tree)
        except RecursionError:
            entry.parse_error = "ast too deeply nested"
            return entry
        entry.symbols = indexer.symbols
        entry.imports = sorted(set(indexer.imports))
        entry.calls = indexer.calls
        return entry

    def _reindex(self) -> None:
        by_name: Dict[str, List[Symbol]] = {}
        callers: Dict[str, List[CallSite]] = {}
        importers: Dict[str, Set[str]] = {}
        for entry in self.modules.values():
            for sym in entry.symbols:
                by_name.setdefault(sym.name, []).append(sym)
            for call in entry.calls:
                callers.setdefault(call.callee, []).append(call)
            for target in entry.imports:
                resolved = self._resolve_module(target)
                if resolved and resolved != entry.module:
                    importers.setdefault(resolved, set()).add(entry.module)
        self._by_name = by_name
        self._callers = callers
        self._importers = importers

    def _resolve_module(self, dotted: str) -> Optional[str]:
        """Map an imported dotted name to a known module (longest matching prefix)."""
        if dotted in self.modules:
            return dotted
        parts = dotted.split(".")
        while len(parts) > 1:
            parts = parts[:-1]
            cand = ".".join(parts)
            if cand in self.modules:
                return cand
        return None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def find(self, name: str, *, kind: str = "", limit: int = 25) -> List[Symbol]:
        """Resolve a bare name (``NyxBrain``), a qualname (``NyxBrain.think``) or a dotted path.

        Returns **every** match. An empty list means the name is not in the index — which is the
        honest answer, and the caller should say so rather than guess.
        """
        query = (name or "").strip()
        if not query:
            return []
        hits: List[Symbol] = list(self._by_name.get(query, ()))
        if not hits and "." in query:
            # Try qualname (Class.method) and fully-dotted (pkg.mod.Class.method) forms.
            tail = query.rsplit(".", 1)[-1]
            for sym in self._by_name.get(tail, ()):
                if sym.dotted == query or sym.qualname == query or sym.dotted.endswith("." + query):
                    hits.append(sym)
        if kind:
            hits = [s for s in hits if s.kind == kind]
        hits.sort(key=lambda s: (s.module, s.line))
        return hits[:limit]

    def definition(self, name: str) -> Optional[Symbol]:
        """The single definition of ``name``, or None when absent **or ambiguous**.

        Ambiguity returning None is deliberate: three functions called ``save`` means the question
        has no single answer, and picking one would be a guess wearing a fact's clothes.
        """
        hits = self.find(name)
        return hits[0] if len(hits) == 1 else None

    def callers(self, name: str, *, limit: int = 200) -> List[CallSite]:
        """Every site that invokes ``name``, by name. See the module docstring on why this is
        lexical: a dynamic dispatch is invisible, and a shared name returns all of its callers."""
        bare = (name or "").rsplit(".", 1)[-1].strip()
        return sorted(self._callers.get(bare, ()), key=lambda c: (c.module, c.line))[:limit]

    def callees(self, qualname: str, *, module: str = "", limit: int = 200) -> List[CallSite]:
        """Every call made *from inside* the given def."""
        out: List[CallSite] = []
        for entry in self.modules.values():
            if module and entry.module != module:
                continue
            out.extend(c for c in entry.calls if c.caller == qualname)
        return sorted(out, key=lambda c: (c.module, c.line))[:limit]

    def imports_of(self, module: str) -> List[str]:
        """The intra-tree modules ``module`` imports at load time."""
        entry = self.modules.get(module)
        if entry is None:
            return []
        out = {r for r in (self._resolve_module(t) for t in entry.imports) if r and r != module}
        return sorted(out)

    def importers_of(self, module: str) -> List[str]:
        """The modules that import ``module`` at load time."""
        return sorted(self._importers.get(module, ()))

    def dependents(self, module: str, *, depth: int = 6) -> Set[str]:
        """Everything that reaches ``module`` transitively — its blast radius."""
        seen: Set[str] = set()
        frontier = {module}
        for _ in range(max(1, depth)):
            nxt: Set[str] = set()
            for mod in frontier:
                for importer in self._importers.get(mod, ()):
                    if importer not in seen:
                        seen.add(importer)
                        nxt.add(importer)
            if not nxt:
                break
            frontier = nxt
        return seen

    def affected_tests(self, changed: Iterable[str], *, depth: int = 6,
                       limit: int = 200) -> List[str]:
        """Test *files* whose modules transitively import anything in ``changed``.

        Accepts module names or file paths. Returns paths, so the result can be handed straight
        to pytest. An empty result means nothing in the index reaches the change — which is a
        real answer, not a failure.
        """
        targets: Set[str] = set()
        for item in changed or ():
            text = str(item)
            if text in self.modules:
                targets.add(text)
                continue
            path = Path(text)
            for mod, entry in self.modules.items():
                if entry.path == str(path) or entry.path.endswith(os.sep + path.name):
                    targets.add(mod)
        reach: Set[str] = set(targets)
        for mod in targets:
            reach |= self.dependents(mod, depth=depth)
        out = {self.modules[m].path for m in reach
               if m in self.modules and _looks_like_test(self.modules[m].path)}
        return sorted(out)[:limit]

    def parse_errors(self) -> List[Tuple[str, str]]:
        """Files that did not parse, with why. Visible rather than swallowed."""
        return sorted((e.path, e.parse_error) for e in self.modules.values() if e.parse_error)

    def architecture(self) -> Any:
        """Cycles, layering violations and god-modules.

        Delegated wholesale to :class:`~nyxara.growth.architecture.ArchitectureAnalyzer`, which
        already implements Tarjan SCC, the declared layer order and the coupling threshold. There
        is no second copy of that judgement here.
        """
        from nyxara.growth.architecture import ArchitectureAnalyzer
        return ArchitectureAnalyzer(root=self.root, pkg=self.pkg).analyze()

    # ------------------------------------------------------------------ #
    # Installed packages
    # ------------------------------------------------------------------ #
    def index_package(self, name: str, *, max_files: int = 4_000) -> Dict[str, Any]:
        """Index an installed distribution from the source really on this machine.

        The top-level name is resolved with :func:`importlib.util.find_spec`, which consults the
        meta-path finders **without executing the module**; every submodule is then read from disk
        with :mod:`ast`. No third-party code runs at any point.
        """
        top = (name or "").split(".")[0].strip()
        if not top:
            return {"ok": False, "reason": "empty name"}
        try:
            import importlib.util
            spec = importlib.util.find_spec(top)
        except (ImportError, ValueError, ModuleNotFoundError) as exc:
            return {"ok": False, "package": top, "reason": f"not importable: {exc}"}
        except Exception as exc:  # noqa: BLE001 — a broken finder must not sink the call
            return {"ok": False, "package": top, "reason": f"finder failed: {exc}"}
        if spec is None:
            return {"ok": False, "package": top, "reason": "not installed"}

        locations = list(getattr(spec, "submodule_search_locations", None) or ())
        if locations:
            root = Path(locations[0]).resolve()
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            root = Path(spec.origin).resolve().parent
        else:
            return {"ok": False, "package": top,
                    "reason": f"no source on disk (origin={spec.origin!r})"}

        sub = Atlas(root=root, pkg=top, store=self.store, max_files=max_files,
                    include_tests=False)
        report = sub.refresh()
        # Merge the sub-index in. Third-party modules keep their own dotted names, so they can
        # never collide with hers, and `find` reaches both trees afterwards.
        self.modules.update(sub.modules)
        self._reindex()
        return {"ok": True, "package": top, "root": str(root), **report}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Optional[Any] = None) -> Path:
        """Write the index to disk atomically (temp file + replace)."""
        target = Path(path) if path is not None else (self.store / "index.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "pkg": self.pkg, "root": str(self.root),
                   "saved_at": time.time(),
                   "trees": [[str(r), p] for r, p in self.trees],
                   "modules": {m: e.to_dict() for m, e in self.modules.items()}}
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def load(self, path: Optional[Any] = None) -> bool:
        """Restore a saved index. Returns False when there is nothing usable to restore."""
        target = Path(path) if path is not None else (self.store / "index.json")
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return False
        try:
            self.modules = {m: ModuleEntry.from_dict(d)
                            for m, d in (payload.get("modules") or {}).items()}
        except (KeyError, TypeError, ValueError):
            self.modules = {}
            return False
        self.pkg = payload.get("pkg", self.pkg)
        trees = payload.get("trees") or []
        if trees:
            self.trees = [(Path(r), str(p)) for r, p in trees]
        self._reindex()
        return True

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for entry in self.modules.values():
            for sym in entry.symbols:
                kinds[sym.kind] = kinds.get(sym.kind, 0) + 1
        return {
            "root": str(self.root),
            "pkg": self.pkg,
            "trees": [[str(r), p] for r, p in self.trees],
            "modules": len(self.modules),
            "symbols": sum(len(e.symbols) for e in self.modules.values()),
            "kinds": kinds,
            "call_sites": sum(len(e.calls) for e in self.modules.values()),
            "distinct_callees": len(self._callers),
            "loc": sum(e.loc for e in self.modules.values()),
            "parse_errors": len(self.parse_errors()),
            "last_refresh": self.last_refresh,
            "last_scanned": self.last_scanned,
            "last_parsed": self.last_parsed,
            "note": ("lexical AST index, never imports the tree it reads; dynamic dispatch "
                     "(getattr, callables in variables, C extensions) is invisible to it, and a "
                     "name with several definitions is reported as all of them, never resolved"),
        }

    def summary(self) -> str:
        st = self.stats()
        lines = ["=" * 70, "NYXARA atlas", "=" * 70,
                 f"root      : {st['root']}",
                 f"modules   : {st['modules']}  ({st['loc']} lines)",
                 f"symbols   : {st['symbols']}  {st['kinds']}",
                 f"callsites : {st['call_sites']} over {st['distinct_callees']} distinct names"]
        errs = self.parse_errors()
        if errs:
            lines.append(f"unparsed  : {len(errs)}")
            for path, why in errs[:5]:
                lines.append(f"  ⚠ {path}: {why}")
        return "\n".join(lines)


def _looks_like_test(path: str) -> bool:
    name = os.path.basename(path)
    return name.startswith("test_") or name.endswith("_test.py") or f"{os.sep}tests{os.sep}" in path


def _package_root() -> Path:
    import nyxara
    return Path(nyxara.__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA atlas self-test")
    print("=" * 70)

    # 1) a synthetic tree indexes exactly what it defines
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "toy"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "a.py").write_text(
            "MAX_N = 10\n"
            "class Widget:\n"
            "    def save(self, x: int) -> bool:\n"
            "        '''Persist it.'''\n"
            "        return helper(x)\n"
            "def helper(x):\n"
            "    return x\n", encoding="utf-8")
        (pkg / "b.py").write_text("from toy import a\ndef go():\n    return a.helper(1)\n",
                                  encoding="utf-8")
        at = Atlas(root=pkg, pkg="toy", store=Path(d) / "store", include_tests=False)
        rep = at.refresh()
        print(f"\ntoy index      : {rep}")
        assert rep["parsed"] == 3, rep

        widget = at.find("Widget")
        assert len(widget) == 1 and widget[0].kind == "class", widget
        save = at.find("save")
        assert len(save) == 1 and save[0].kind == "method", save
        assert save[0].parent == "Widget", save[0]
        assert save[0].signature.startswith("(self, x: int"), save[0].signature
        assert save[0].doc == "Persist it.", save[0].doc
        assert at.find("MAX_N")[0].kind == "constant"

        callers = at.callers("helper")
        assert {c.caller for c in callers} == {"Widget.save", "go"}, callers

        assert at.importers_of("toy.a") == ["toy.b"], at.importers_of("toy.a")
        assert "toy.a" in at.imports_of("toy.b")

        again = at.refresh()
        assert again["parsed"] == 0, again
        (pkg / "a.py").write_text("def helper(x):\n    return x + 1\n", encoding="utf-8")
        after = at.refresh()
        assert after["parsed"] == 1, after
        assert at.find("Widget") == [], "removed symbol must leave the index"

        (pkg / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        at.refresh()
        assert any("bad.py" in p for p, _ in at.parse_errors()), at.parse_errors()

        saved = at.save()
        fresh = Atlas(root=pkg, pkg="toy", store=Path(d) / "store", include_tests=False)
        assert fresh.load(saved) is True
        assert fresh.find("helper"), "loaded index must answer queries"
        print("toy round-trip : ok")

    # 2) ambiguity is reported, never resolved
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "amb"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "x.py").write_text("def save():\n    pass\n", encoding="utf-8")
        (pkg / "y.py").write_text("def save():\n    pass\n", encoding="utf-8")
        at = Atlas(root=pkg, pkg="amb", store=Path(d) / "s", include_tests=False)
        at.refresh()
        assert len(at.find("save")) == 2
        assert at.definition("save") is None, "two definitions must not resolve to one"

    # 3) the real tree indexes and answers a real question
    real = Atlas()
    print(f"\nreal index     : {real.refresh()}")
    hits = real.find("NyxBrain")
    assert hits, "NyxBrain should be in her own index"
    print(f"NyxBrain       : {hits[0].location}")
    assert hits[0].path.endswith("nyx/brain.py"), hits[0].path
    tests = real.affected_tests(["nyxara.nyx.brain"])
    print(f"blast radius   : {len(tests)} test file(s) reach nyx/brain.py")
    assert tests, "a change to nyx/brain.py must reach real test files"
    assert any(t.endswith("test_brain.py") for t in tests), tests[:5]
    print("\n" + real.summary())

    print("\nALL SELF-TESTS PASSED ✓")
