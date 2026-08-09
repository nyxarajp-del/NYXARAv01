"""NYXARA · nyx/meta_architecture.py — edges that mean something (🕸, NYX V.02).

In V.01's graph an edge is a **float**. `cat—warm` and `sparrow—bird` are the same kind of
thing, differing only in how strong they are. The graph can say *these two go together* and can
never say *how*.

:class:`GraphWeaver` gives the edge a **type**, and lets new types be *born from evidence*:

* **Typed edges** over the existing store — ``co-occurs``, ``is-a``, ``part-of``, ``causes``,
  ``precedes``, ``contradicts``, ``synonym-of`` — kept in their own layer, so the Hebbian weight
  underneath is never overwritten by a label.
* **New types at runtime.** A relational pattern she keeps meeting becomes a type of its own —
  but only after **minimum support**. A type minted from one sighting is an idea, not evidence,
  and this is where that line is drawn.
* **Node split.** A polysemous label ("bank") that co-occurs with two clean, disjoint
  neighbourhoods is split into two nodes, each keeping its own half of the structure.
* **Merge** is the graph's own (:meth:`~nyxara.nyx.graph.DynamicNeuralGraph.merge_nodes`),
  authorised only by the relational rung of :mod:`~nyxara.nyx.semantics`.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **This is bounded plasticity.** There is a per-tick rewire budget and a cap on how many types
  can exist, and every schema change is written to a **ledger** and is reversible. "Rewires
  itself like a biological brain every second" is an analogy; what is here is real *structural*
  plasticity on a budget, and not synaptogenesis.
* **A new edge type needs support, not an idea.** Below ``min_support`` sightings a pattern is
  reported as a *candidate* and nothing is minted.
* A split is only performed when the two neighbourhoods are genuinely **disjoint**; an ambiguous
  case is left alone, because splitting a node that was not polysemous destroys real structure
  to fix an imagined problem.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["EdgeType", "TypedEdge", "SchemaChange", "GraphWeaver", "BUILTIN_TYPES"]

#: The types she starts with. Everything else has to earn its place.
BUILTIN_TYPES: Tuple[str, ...] = ("co-occurs", "is-a", "part-of", "causes", "precedes",
                                  "contradicts", "synonym-of")

#: Surface patterns that name a relation. Each is a *candidate* until it has support.
_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bis\s+a\s+kind\s+of\b", "is-a"),
    (r"\bis\s+an?\b", "is-a"),
    (r"\bpart\s+of\b", "part-of"),
    (r"\bcontains\b", "part-of"),
    (r"\bcauses?\b", "causes"),
    (r"\bleads?\s+to\b", "causes"),
    (r"\bbefore\b", "precedes"),
    (r"\bthen\b", "precedes"),
    (r"\bbut\s+not\b", "contradicts"),
    (r"\bmeans\b", "synonym-of"),
)


@dataclass
class EdgeType:
    """One relation the graph can express, and where it came from."""

    name: str = ""
    builtin: bool = False
    support: int = 0
    minted_at: float = field(default_factory=time.time)
    example: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "builtin": self.builtin, "support": self.support,
                "minted_at": round(self.minted_at, 3), "example": self.example}


@dataclass
class TypedEdge:
    """A typed, directed relation laid over the graph's own symmetric weight."""

    a: str = ""
    b: str = ""
    type: str = "co-occurs"
    weight: float = 0.5
    directed: bool = True
    at: float = field(default_factory=time.time)

    def render(self) -> str:
        arrow = "→" if self.directed else "—"
        return f"{self.a} {arrow}[{self.type}]{arrow} {self.b}  ({self.weight:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "type": self.type,
                "weight": round(self.weight, 4), "directed": self.directed,
                "at": round(self.at, 3)}


@dataclass
class SchemaChange:
    """One structural change — in the ledger, and reversible."""

    kind: str = ""              # mint | split | merge | rewire
    detail: str = ""
    undo: Optional[Dict[str, Any]] = None
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "undo": self.undo,
                "at": round(self.at, 3)}


class GraphWeaver:
    """Typed edges, types born from evidence, and node splits — all on a budget, all ledgered."""

    def __init__(self, brain: Any = None, *, min_support: int = 3, max_types: int = 32,
                 rewire_budget: int = 16, max_edges: int = 8192) -> None:
        self.brain = brain
        # A pattern seen once is an idea. This is where "born from evidence" is enforced.
        self.min_support = max(2, int(min_support))
        self.max_types = max(len(BUILTIN_TYPES), int(max_types))
        self.rewire_budget = max(1, int(rewire_budget))
        self.max_edges = max(64, int(max_edges))

        self.types: Dict[str, EdgeType] = {
            name: EdgeType(name=name, builtin=True) for name in BUILTIN_TYPES}
        self.edges: Dict[Tuple[str, str, str], TypedEdge] = {}
        self.ledger: List[SchemaChange] = []
        self.candidates: Dict[str, int] = {}
        self.splits = 0
        self.minted = 0
        self._rewires_this_tick = 0

    # ---- typed edges ----------------------------------------------------------- #
    def relate(self, a: str, b: str, *, type: str = "co-occurs", weight: float = 0.5,
               directed: bool = True) -> Optional[TypedEdge]:
        """Lay a typed relation over the graph. The Hebbian weight underneath is untouched."""
        try:
            a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
            kind = str(type or "co-occurs")
            if not a or not b or a == b or kind not in self.types:
                return None
            if self._rewires_this_tick >= self.rewire_budget:
                return None
            edge = TypedEdge(a=a, b=b, type=kind,
                             weight=max(0.0, min(1.0, float(weight))), directed=directed)
            self.edges[(a, b, kind)] = edge
            self._rewires_this_tick += 1
            self.types[kind].support += 1
            self._enforce_cap()
            return edge
        except Exception:  # noqa: BLE001
            return None

    def _enforce_cap(self) -> None:
        over = len(self.edges) - self.max_edges
        if over <= 0:
            return
        weakest = sorted(self.edges.items(), key=lambda kv: kv[1].weight)[:over]
        for key, _edge in weakest:
            self.edges.pop(key, None)

    def edges_of(self, label: str, *, type: str = "") -> List[TypedEdge]:
        try:
            key = str(label or "").strip().lower()
            out = [e for e in self.edges.values()
                   if (e.a == key or (not e.directed and e.b == key))
                   and (not type or e.type == type)]
            out.sort(key=lambda e: -e.weight)
            return out
        except Exception:  # noqa: BLE001
            return []

    def tick(self) -> None:
        """A new tick: the rewire budget is restored. Bounded plasticity is bounded per tick."""
        self._rewires_this_tick = 0

    # ---- types born from evidence ------------------------------------------------- #
    def read(self, text: str) -> List[TypedEdge]:
        """Read typed relations out of a sentence, minting a new type only on real support."""
        made: List[TypedEdge] = []
        try:
            import re
            from nyxara.nyx.graph import concepts_in
            body = str(text or "")
            concepts = concepts_in(body, cap=8)
            if len(concepts) < 2:
                return made
            for pattern, kind in _PATTERNS:
                if not re.search(pattern, body, re.I):
                    continue
                edge = self.relate(concepts[0], concepts[1], type=kind, weight=0.6)
                if edge is not None:
                    made.append(edge)
                break
            return made
        except Exception:  # noqa: BLE001
            return made

    def propose_type(self, name: str, *, example: str = "") -> Optional[EdgeType]:
        """Note a candidate relation. It becomes a real type only at ``min_support`` sightings.

        This is the whole discipline of "new edge types are born from evidence, not from an
        idea": below the threshold nothing is minted and the candidate is simply counted.
        """
        try:
            key = str(name or "").strip().lower().replace(" ", "-")
            if not key or key in self.types:
                return self.types.get(key)
            self.candidates[key] = self.candidates.get(key, 0) + 1
            if self.candidates[key] < self.min_support:
                return None
            if len(self.types) >= self.max_types:
                return None
            minted = EdgeType(name=key, builtin=False,
                              support=self.candidates[key], example=str(example)[:120])
            self.types[key] = minted
            self.minted += 1
            self._record(SchemaChange(
                kind="mint",
                detail=f"minted edge type {key!r} after {self.candidates[key]} sightings",
                undo={"type": key}))
            return minted
        except Exception:  # noqa: BLE001
            return None

    # ---- node split --------------------------------------------------------------- #
    def split(self, label: str, *, min_side: int = 2) -> Optional[Tuple[str, str]]:
        """Split a polysemous node into two, each keeping its own half of the structure.

        Only when the two neighbourhoods are genuinely **disjoint**. Splitting a node that was
        not polysemous destroys real structure to fix an imagined problem, so an ambiguous case
        is left alone.
        """
        try:
            graph = getattr(self.brain, "graph", None)
            if graph is None:
                return None
            label = str(label or "").strip().lower()
            if label not in graph:
                return None
            neighbours = graph.neighbours(label, k=64)
            if len(neighbours) < 2 * max(1, int(min_side)):
                return None

            sides = self._disjoint_sides(graph, [n for n, _w in neighbours], min_side)
            if sides is None:
                return None
            left, right = sides

            a_label, b_label = f"{label}#1", f"{label}#2"
            weights = {n: graph.weight(label, n) for n, _w in neighbours}
            for new_label, side in ((a_label, left), (b_label, right)):
                graph.activate([new_label] + list(side), learn=True, decay=False)
            self.splits += 1
            self._record(SchemaChange(
                kind="split",
                detail=f"split {label!r} into {a_label} ({len(left)}) and "
                       f"{b_label} ({len(right)}) — two disjoint neighbourhoods",
                undo={"label": label, "parts": [a_label, b_label],
                      "weights": {k: round(v, 6) for k, v in weights.items()}}))
            return a_label, b_label
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _disjoint_sides(graph: Any, neighbours: Sequence[str],
                        min_side: int) -> Optional[Tuple[List[str], List[str]]]:
        """Two groups of neighbours that never co-occur with each other. Otherwise ``None``."""
        try:
            left: List[str] = [neighbours[0]]
            right: List[str] = []
            for candidate in neighbours[1:]:
                if all(graph.weight(candidate, other) <= 0.0 for other in left):
                    right.append(candidate)
                else:
                    left.append(candidate)
            if len(left) < min_side or len(right) < min_side:
                return None
            # The two sides must not touch each other at all — that is what "two senses" means.
            if any(graph.weight(a, b) > 0.0 for a in left for b in right):
                return None
            return left, right
        except Exception:  # noqa: BLE001
            return None

    # ---- reversibility --------------------------------------------------------------- #
    def _record(self, change: SchemaChange) -> None:
        self.ledger.append(change)
        if len(self.ledger) > 256:
            del self.ledger[: len(self.ledger) - 256]

    def undo_last(self) -> bool:
        """Reverse the most recent schema change. Structural change here is always reversible."""
        try:
            if not self.ledger:
                return False
            change = self.ledger.pop()
            undo = change.undo or {}
            if change.kind == "mint":
                name = str(undo.get("type", ""))
                self.types.pop(name, None)
                self.edges = {k: v for k, v in self.edges.items() if v.type != name}
                self.minted = max(0, self.minted - 1)
                return True
            if change.kind == "split":
                graph = getattr(self.brain, "graph", None)
                if graph is None:
                    return False
                for part in undo.get("parts", []) or []:
                    graph.nodes.pop(str(part), None)
                    for other in list(graph._adj.get(str(part), ())):  # noqa: SLF001
                        graph._cut(tuple(sorted((str(part), other))))  # noqa: SLF001
                    graph._adj.pop(str(part), None)  # noqa: SLF001
                self.splits = max(0, self.splits - 1)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    # ---- introspection ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            by_type: Dict[str, int] = {}
            for edge in self.edges.values():
                by_type[edge.type] = by_type.get(edge.type, 0) + 1
            return {
                "types": {n: t.to_dict() for n, t in self.types.items()},
                "minted": self.minted, "splits": self.splits,
                "typed_edges": len(self.edges), "by_type": by_type,
                "candidates": dict(self.candidates),
                "min_support": self.min_support,
                "rewire_budget": self.rewire_budget,
                "ledger": len(self.ledger),
                "note": ("bounded plasticity: a per-tick rewire budget, a cap on how many "
                         "types can exist, and every schema change in a ledger and reversible. "
                         "'Rewires itself like a biological brain every second' is an analogy "
                         "— this is real structural plasticity on a budget, not synaptogenesis. "
                         "A new edge type needs SUPPORT, not an idea: below min_support a "
                         "pattern is a counted candidate and nothing is minted. A node is "
                         "split only when its two neighbourhoods are genuinely disjoint"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        try:
            lines = [f"{len(self.types)} edge type(s), {len(self.edges)} typed edge(s):"]
            for name, kind in sorted(self.types.items()):
                mark = "builtin" if kind.builtin else f"minted ({kind.support} sightings)"
                lines.append(f"  {name:14} {mark}")
            if self.candidates:
                pending = ", ".join(f"{k}×{v}" for k, v in self.candidates.items()
                                    if k not in self.types)
                if pending:
                    lines.append(f"  candidates (not yet minted): {pending}")
            for change in self.ledger[-6:]:
                lines.append(f"  ledger: {change.kind} — {change.detail}")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence -------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"minted": self.minted, "splits": self.splits,
                "candidates": dict(self.candidates),
                "types": [t.to_dict() for t in self.types.values() if not t.builtin],
                "edges": [e.to_dict() for e in self.edges.values()],
                "ledger": [c.to_dict() for c in self.ledger[-64:]]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.minted = int(data.get("minted", 0))
            self.splits = int(data.get("splits", 0))
            self.candidates = {str(k): int(v) for k, v in (data.get("candidates") or {}).items()}
            for raw in data.get("types", []) or []:
                if isinstance(raw, dict) and raw.get("name"):
                    self.types[str(raw["name"])] = EdgeType(
                        name=str(raw["name"]), builtin=False,
                        support=int(raw.get("support", 0)),
                        example=str(raw.get("example", "")))
            self.edges = {}
            for raw in data.get("edges", []) or []:
                if not isinstance(raw, dict):
                    continue
                edge = TypedEdge(a=str(raw.get("a", "")), b=str(raw.get("b", "")),
                                 type=str(raw.get("type", "co-occurs")),
                                 weight=float(raw.get("weight", 0.5)),
                                 directed=bool(raw.get("directed", True)))
                if edge.a and edge.b and edge.type in self.types:
                    self.edges[(edge.a, edge.b, edge.type)] = edge
            self.ledger = [SchemaChange(kind=str(c.get("kind", "")),
                                        detail=str(c.get("detail", "")), undo=c.get("undo"))
                           for c in (data.get("ledger") or []) if isinstance(c, dict)]
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
