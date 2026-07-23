"""NYXARA · growth/arch_rewrite.py — DPO-style architecture graph rewriting (🧬, P).

NYXARA can see her whole codebase as a directed dependency (hyper)graph and, when the *shape* is
inefficient (import cycles, layering violations, god-modules), apply formal **Double-Pushout (DPO)**
rewrite rules — *match a subgraph → replace it with a restructured subgraph* (invert a dependency, extract
a module) — keeping a rewrite only when a structural metric strictly improves **and** the change is
behaviour-preserving (the caller's verify-or-rollback gauntlet says so).

This module owns the **graph model + rewrite rules + accept/reject decision**. The real import graph comes
from :class:`nyxara.growth.architecture.ArchitectureAnalyzer`; enacting the resulting multi-file edit goes
through the same reversible gauntlet as :mod:`nyxara.growth.self_optimize`, passed in as ``verify``.

Honest boundary: disciplined, bounded **graph surgery over the import graph**, verified for behaviour —
not unconstrained category-theory magic. A rewrite that does not improve the metric, or that cannot be
shown behaviour-preserving, is discarded. Pure standard library; no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

__all__ = ["DependencyGraph", "Rewrite", "ArchRewriteResult", "ArchRewriter"]


class DependencyGraph:
    """A directed graph of module → imported-modules, with structural weakness detection."""

    def __init__(self, edges: Optional[Dict[str, Set[str]]] = None) -> None:
        self._adj: Dict[str, Set[str]] = {}
        for src, dsts in (edges or {}).items():
            self._adj.setdefault(src, set())
            for d in dsts:
                self.add_edge(src, d)

    def add_edge(self, src: str, dst: str) -> None:
        self._adj.setdefault(src, set()).add(dst)
        self._adj.setdefault(dst, set())

    def remove_edge(self, src: str, dst: str) -> None:
        self._adj.get(src, set()).discard(dst)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._adj)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return sorted((s, d) for s, dsts in self._adj.items() for d in dsts)

    def copy(self) -> "DependencyGraph":
        return DependencyGraph({s: set(d) for s, d in self._adj.items()})

    # ---- weaknesses ---- #
    def cycles(self) -> List[List[str]]:
        """Return simple cycles found by DFS (bounded, deterministic)."""
        found: List[List[str]] = []
        seen_pairs: Set[frozenset] = set()

        def dfs(start: str, node: str, stack: List[str], visited: Set[str]) -> None:
            for nxt in sorted(self._adj.get(node, set())):
                if nxt == start and len(stack) >= 1:
                    key = frozenset(stack)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        found.append(list(stack))
                elif nxt not in visited and nxt not in stack:
                    dfs(start, nxt, stack + [nxt], visited)

        for n in self.nodes:
            dfs(n, n, [n], set())
        return found

    def god_modules(self, *, threshold: int = 8) -> List[str]:
        fan_in = {n: 0 for n in self._adj}
        for _, dsts in self._adj.items():
            for d in dsts:
                fan_in[d] = fan_in.get(d, 0) + 1
        return sorted(n for n in self._adj
                      if fan_in.get(n, 0) + len(self._adj.get(n, set())) >= threshold)

    def layering_violations(self, layers: Dict[str, int]) -> List[Tuple[str, str]]:
        """Edges where a lower layer imports a higher one (src layer < dst layer)."""
        out: List[Tuple[str, str]] = []
        for s, dsts in self._adj.items():
            for d in dsts:
                if s in layers and d in layers and layers[s] < layers[d]:
                    out.append((s, d))
        return sorted(out)

    def score(self, *, layers: Optional[Dict[str, int]] = None, god_threshold: int = 8) -> float:
        """A structural cost: lower is healthier. Cycles hurt most, then layering, then god-modules."""
        c = len(self.cycles())
        v = len(self.layering_violations(layers)) if layers else 0
        g = len(self.god_modules(threshold=god_threshold))
        return 10.0 * c + 5.0 * v + 1.0 * g


@dataclass
class Rewrite:
    kind: str                       # "invert_edge" | "extract_module"
    detail: str
    apply: Callable[[DependencyGraph], DependencyGraph]


@dataclass
class ArchRewriteResult:
    applied: bool
    kind: str = ""
    detail: str = ""
    before_score: float = 0.0
    after_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {"applied": self.applied, "kind": self.kind, "detail": self.detail,
                "before_score": self.before_score, "after_score": self.after_score,
                "reason": self.reason}


class ArchRewriter:
    """Propose DPO-style rewrites and keep one only if it improves the metric AND verifies."""

    def __init__(self, *, layers: Optional[Dict[str, int]] = None, god_threshold: int = 8) -> None:
        self.layers = layers
        self.god_threshold = god_threshold

    def _score(self, g: DependencyGraph) -> float:
        return g.score(layers=self.layers, god_threshold=self.god_threshold)

    def propose(self, g: DependencyGraph) -> List[Rewrite]:
        """Enumerate candidate rewrites for the current weaknesses."""
        rewrites: List[Rewrite] = []
        # DPO rule: break a cycle by inverting its weakest back-edge
        for cyc in g.cycles():
            if len(cyc) >= 2:
                def _break(graph: DependencyGraph, cyc=cyc) -> DependencyGraph:
                    ng = graph.copy()
                    ng.remove_edge(cyc[-1], cyc[0])   # drop the edge that closes the cycle
                    return ng

                rewrites.append(Rewrite(
                    "invert_edge", f"remove closing import {cyc[-1]} → {cyc[0]} to break cycle",
                    _break))
        # DPO rule: extract a helper out of a god-module (moves half its out-edges to a new node)
        for god in g.god_modules(threshold=self.god_threshold):
            def _extract(graph: DependencyGraph, god=god) -> DependencyGraph:
                ng = graph.copy()
                outs = sorted(ng._adj.get(god, set()))
                half = outs[len(outs) // 2:]
                helper = f"{god}._extracted"
                for d in half:
                    ng.remove_edge(god, d)
                    ng.add_edge(helper, d)
                ng.add_edge(god, helper)
                return ng
            rewrites.append(Rewrite(
                "extract_module", f"extract a helper module from god-module {god}", _extract))
        return rewrites

    def improve(self, g: DependencyGraph,
                verify: Callable[[DependencyGraph, DependencyGraph], bool]) -> ArchRewriteResult:
        """Apply the best rewrite that strictly lowers the score AND passes ``verify``. ``verify`` is
        the behaviour-equivalence gauntlet — only consulted on a candidate that already improves the
        metric, so the expensive check is never spent on a rewrite that could not be kept."""
        before = self._score(g)
        best: Optional[Tuple[Rewrite, DependencyGraph, float]] = None
        for rw in self.propose(g):
            cand = rw.apply(g)
            after = self._score(cand)
            if after < before and (best is None or after < best[2]):
                best = (rw, cand, after)
        if best is None:
            return ArchRewriteResult(False, before_score=before, after_score=before,
                                     reason="no rewrite improved the structural metric")
        rw, cand, after = best
        try:
            ok = bool(verify(g, cand))
        except Exception:  # noqa: BLE001 — a throwing gauntlet is a failed verification
            ok = False
        if not ok:
            return ArchRewriteResult(False, kind=rw.kind, detail=rw.detail,
                                     before_score=before, after_score=after,
                                     reason="rewrite not behaviour-preserving (gauntlet failed)")
        return ArchRewriteResult(True, kind=rw.kind, detail=rw.detail,
                                 before_score=before, after_score=after,
                                 reason="architecture improved + behaviour preserved")
