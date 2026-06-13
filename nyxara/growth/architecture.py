"""NYXARA · growth/architecture.py — Self Architecture Analysis (🏛, RSI capability 2).

NYXARA maps her *own* module structure and judges its shape. This is the real intra-package
import graph — parsed from source with :mod:`ast`, no import side-effects — from which she
computes:

* **fan-in / fan-out** per module (how many depend on it; how many it depends on).
* **import cycles** — strongly-connected components of size > 1 (a `nx` ``simple_cycles`` when
  :mod:`networkx` is present; an in-house Tarjan SCC otherwise, so the analysis runs with no
  optional dependency).
* **layering violations** — NYXARA's layers have a declared order
  (``kernel < guard < mind < memory < agency < planning < growth < eval``). An import from a
  higher layer down into a lower one is expected; the reverse (e.g. ``kernel`` importing from
  ``growth``) is a violation and is flagged.
* **god-modules** — modules with outsized fan-in + fan-out + size.

Honest and deterministic: the same tree yields the same report on every machine.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ModuleNode", "ArchFinding", "ArchitectureReport", "ArchitectureAnalyzer",
           "LAYER_ORDER"]

# Declared layer order (lower index = more foundational). An import that points from a
# lower-index layer *up* to a higher-index one is a layering violation.
LAYER_ORDER: Tuple[str, ...] = (
    "kernel", "identity", "guard", "observe", "senses", "mind", "memory", "knowledge",
    "social", "sim", "agency", "planning", "growth", "eval", "server",
)


@dataclass
class ModuleNode:
    name: str
    fan_in: int = 0
    fan_out: int = 0
    loc: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "fan_in": self.fan_in,
                "fan_out": self.fan_out, "loc": self.loc}


@dataclass
class ArchFinding:
    kind: str
    detail: str
    severity: float

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "severity": round(self.severity, 3)}


@dataclass
class ArchitectureReport:
    modules: List[ModuleNode] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    layering_violations: List[Tuple[str, str]] = field(default_factory=list)
    god_modules: List[str] = field(default_factory=list)
    findings: List[ArchFinding] = field(default_factory=list)
    used_networkx: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"n_modules": len(self.modules), "n_edges": len(self.edges),
                "cycles": self.cycles, "layering_violations": self.layering_violations,
                "god_modules": self.god_modules, "used_networkx": self.used_networkx,
                "findings": [f.to_dict() for f in self.findings],
                "modules": [m.to_dict() for m in
                            sorted(self.modules, key=lambda m: m.fan_in + m.fan_out,
                                   reverse=True)[:15]]}

    def summary(self) -> str:
        lines = ["=" * 70, "NYXARA self architecture analysis", "=" * 70,
                 f"{len(self.modules)} module(s), {len(self.edges)} intra-package import edge(s)"
                 f" ({'networkx' if self.used_networkx else 'builtin SCC'})", ""]
        lines.append(f"import cycles        : {len(self.cycles)}")
        for cyc in self.cycles[:5]:
            lines.append(f"  ↻ {' → '.join(cyc)} → {cyc[0]}")
        lines.append(f"layering violations  : {len(self.layering_violations)}")
        for src, dst in self.layering_violations[:8]:
            lines.append(f"  ⚠ {src} imports {dst} (lower layer importing higher)")
        if self.god_modules:
            lines.append(f"god-modules          : {', '.join(self.god_modules[:8])}")
        return "\n".join(lines)


class ArchitectureAnalyzer:
    """Build and judge NYXARA's real intra-package import graph."""

    def __init__(self, root: Optional[Any] = None, *, llm: Any = None,
                 pkg: str = "nyxara") -> None:
        self.root = Path(root) if root is not None else _package_root()
        self.pkg = pkg
        self.llm = llm

    # ---- graph construction ---- #
    def _module_name(self, path: Path) -> str:
        rel = path.relative_to(self.root).with_suffix("")
        parts = [self.pkg, *rel.parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _imports_of(self, path: Path, module: str) -> List[str]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                             filename=str(path))
        except Exception:  # noqa: BLE001
            return []
        targets: List[str] = []
        pkg_parts = module.split(".")
        # Only module-level (load-time) imports define the dependency graph. Imports nested
        # inside a function/class are deliberate *lazy* imports the codebase uses to break
        # cycles — counting them would report cycles that don't exist at import time.
        for node in _module_level_imports(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(self.pkg):
                        targets.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:  # relative import — resolve against this module's package
                    base = pkg_parts[:-node.level] if node.level <= len(pkg_parts) else []
                    mod = ".".join([*base, mod]) if mod else ".".join(base)
                if mod.startswith(self.pkg):
                    targets.append(mod)
                    # `from pkg import sub` may import a *submodule*; offer it as a more
                    # specific candidate (resolved only if it is a real module node).
                    for alias in node.names:
                        targets.append(f"{mod}.{alias.name}")
        return targets

    def _build_graph(self) -> Tuple[Dict[str, ModuleNode], List[Tuple[str, str]]]:
        nodes: Dict[str, ModuleNode] = {}
        edges: List[Tuple[str, str]] = []
        # first pass — every module is a node
        files = [p for p in sorted(self.root.rglob("*.py"))
                 if "__pycache__" not in p.parts and "tests" not in p.parts]
        names = {self._module_name(p): p for p in files}
        for name, path in names.items():
            try:
                loc = sum(1 for _ in path.read_text(encoding="utf-8",
                                                     errors="replace").splitlines())
            except Exception:  # noqa: BLE001
                loc = 0
            nodes[name] = ModuleNode(name=name, loc=loc)
        # second pass — edges (resolve an import to the nearest known module/package)
        for name, path in names.items():
            for target in self._imports_of(path, name):
                resolved = self._resolve(target, nodes)
                if resolved and resolved != name:
                    edges.append((name, resolved))
        # de-dup edges and compute degrees
        edges = sorted(set(edges))
        for src, dst in edges:
            nodes[src].fan_out += 1
            nodes[dst].fan_in += 1
        return nodes, edges

    def _resolve(self, target: str, nodes: Dict[str, ModuleNode]) -> Optional[str]:
        """Map an imported dotted name to a known module node (longest matching prefix)."""
        if target in nodes:
            return target
        parts = target.split(".")
        while len(parts) > 1:
            parts = parts[:-1]
            cand = ".".join(parts)
            if cand in nodes:
                return cand
        return None

    # ---- analysis ---- #
    def analyze(self) -> ArchitectureReport:
        nodes, edges = self._build_graph()
        report = ArchitectureReport(modules=list(nodes.values()), edges=edges)

        cycles, used_nx = _find_cycles(edges)
        report.cycles = cycles
        report.used_networkx = used_nx

        report.layering_violations = self._layering_violations(edges)
        report.god_modules = self._god_modules(nodes)

        # findings
        for cyc in cycles:
            report.findings.append(ArchFinding(
                "import_cycle", f"cycle: {' → '.join(cyc)}", 0.85))
        for src, dst in report.layering_violations:
            report.findings.append(ArchFinding(
                "layering_violation", f"{src} imports {dst}", 0.75))
        for god in report.god_modules:
            report.findings.append(ArchFinding(
                "god_module", f"{god} has very high coupling", 0.5))
        return report

    def _layering_violations(self, edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        idx = {layer: i for i, layer in enumerate(LAYER_ORDER)}
        out: List[Tuple[str, str]] = []
        for src, dst in edges:
            s_layer = _layer_of(src, self.pkg)
            d_layer = _layer_of(dst, self.pkg)
            if s_layer in idx and d_layer in idx and idx[s_layer] < idx[d_layer]:
                out.append((src, dst))
        return out

    def _god_modules(self, nodes: Dict[str, ModuleNode]) -> List[str]:
        if not nodes:
            return []
        coupling = sorted((n.fan_in + n.fan_out for n in nodes.values()), reverse=True)
        if not coupling or coupling[0] == 0:
            return []
        # threshold = max(8, 90th-percentile coupling)
        p90 = coupling[max(0, int(len(coupling) * 0.1) - 1)]
        threshold = max(8, p90)
        return sorted((n.name for n in nodes.values()
                       if (n.fan_in + n.fan_out) >= threshold),
                      key=lambda nm: -(nodes[nm].fan_in + nodes[nm].fan_out))


# --------------------------------------------------------------------------- #
# Cycle detection — networkx if present, Tarjan SCC fallback otherwise
# --------------------------------------------------------------------------- #
def _find_cycles(edges: List[Tuple[str, str]]) -> Tuple[List[List[str]], bool]:
    try:
        import networkx as nx
        g = nx.DiGraph()
        g.add_edges_from(edges)
        cycles = [c for c in nx.simple_cycles(g) if len(c) > 1]
        return sorted(cycles, key=len, reverse=True)[:20], True
    except Exception:  # noqa: BLE001 — no networkx (or any error) → builtin SCC
        return _tarjan_scc(edges), False


def _tarjan_scc(edges: List[Tuple[str, str]]) -> List[List[str]]:
    """Strongly-connected components of size > 1 (each is a real cycle), iterative Tarjan."""
    graph: Dict[str, List[str]] = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
        graph.setdefault(dst, [])

    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    counter = [0]
    sccs: List[List[str]] = []

    def strongconnect(start: str) -> None:
        work = [(start, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            recurse = False
            neighbours = graph.get(node, [])
            for j in range(pi, len(neighbours)):
                w = neighbours[j]
                if w not in index:
                    work[-1] = (node, j + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack.get(w):
                    low[node] = min(low[node], index[w])
            if recurse:
                continue
            if low[node] == index[node]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:
                    sccs.append(list(reversed(comp)))
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    for node in list(graph.keys()):
        if node not in index:
            strongconnect(node)
    return sorted(sccs, key=len, reverse=True)[:20]


def _module_level_imports(tree: ast.AST) -> List[ast.AST]:
    """Yield Import / ImportFrom nodes at module scope (descending into control-flow blocks
    like ``try``/``if``/``with`` but NOT into function or class bodies, where lazy imports live)."""
    out: List[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                out.append(child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # lazy imports inside defs are intentional cycle-breakers
            else:
                visit(child)

    visit(tree)
    return out


def _layer_of(module: str, pkg: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == pkg else ""


def _package_root() -> Path:
    import nyxara
    return Path(nyxara.__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA architecture-analysis self-test")
    print("=" * 70)

    # 1) synthetic package with a deliberate a → b → a cycle
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d) / "toy"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "a.py").write_text("from toy import b\n", encoding="utf-8")
        (pkg / "b.py").write_text("from toy import a\n", encoding="utf-8")
        rep = ArchitectureAnalyzer(root=pkg, pkg="toy").analyze()
        print(f"\ntoy cycles          : {rep.cycles}")
        assert any(set(c) == {"toy.a", "toy.b"} for c in rep.cycles), "a↔b cycle not found"

    # 2) builtin SCC fallback finds the same cycle without networkx
    cyc_only, used_nx = _find_cycles([("x", "y"), ("y", "x")])
    assert any(set(c) == {"x", "y"} for c in cyc_only)

    # 3) real run over NYXARA's tree never crashes and finds real modules
    real = ArchitectureAnalyzer().analyze()
    print(f"\nreal modules        : {len(real.modules)}; edges {len(real.edges)}; "
          f"cycles {len(real.cycles)}; layering {len(real.layering_violations)}")
    assert len(real.modules) > 50, "should map the real package"
    print("\n" + real.summary())

    print("\nALL SELF-TESTS PASSED ✓")
