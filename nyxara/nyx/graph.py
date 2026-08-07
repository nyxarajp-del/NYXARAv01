"""NYXARA · nyx/graph.py — the Dynamic Neural Graph (🕸, NYX V.01 pillar 1).

A concept-level network that **rewires itself** as she lives. Nodes are concepts; edges are
weighted associations. Concepts that fire together in the same moment have their edge
*strengthened* (Hebb); edges nobody uses *decay* and, below a floor, are *pruned*; a concept
never seen before is *born* as a new node. Nothing about the topology is declared up front —
it is grown, and it changes every turn.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* This is **Hebbian co-activation bookkeeping over a bounded graph**, not biological
  synaptogenesis and not a trained neural network. There is no backpropagation here.
* Capacity is **bounded** (``max_nodes`` / ``max_edges``) with weakest-first eviction. It is
  not "infinite"; it forgets, on purpose and measurably.
* Rewiring per tick is **budgeted** (``rewire_budget``) so one strange turn cannot reshape
  the whole graph.

Pairs with :mod:`nyxara.nyx.holomem` (what the concepts *mean*) and
:mod:`nyxara.nyx5.topology` (the same idea one level down, at the synapse). Pure standard
library. Every public method is fail-soft: on any error it degrades to a null result rather
than breaking a turn.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["Concept", "Activation", "GraphStats", "DynamicNeuralGraph"]


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")

# Words that carry no conceptual weight — they would otherwise become the hubs of the graph,
# since they co-occur with everything. Deliberately small: this is a stop-list, not an ontology.
_STOP = frozenset("""
a an the and or but if then than that this these those there here of to in on at by for with
from as is am are was were be been being do does did done have has had having i you he she it
we they me him her us them my your his its our their not no yes so such very just only also
what which who whom whose when where why how can could shall should will would may might must
""".split())


def concepts_in(text: str, *, cap: int = 32) -> List[str]:
    """Extract candidate concept labels from ``text`` — lowercase content tokens, order kept.

    Deliberately dumb and deterministic: the *grounding* of a concept is
    :mod:`nyxara.nyx.ground`'s job, not this module's. Here we only need stable labels.
    """
    try:
        out: List[str] = []
        seen = set()
        for tok in _TOKEN_RE.findall((text or "").lower()):
            if len(tok) < 2 or tok in _STOP or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= cap:
                break
        return out
    except Exception:  # noqa: BLE001 — extraction never breaks a turn
        return []


@dataclass
class Concept:
    """One node: a concept the graph has met, and how much of a footprint it has."""

    label: str
    strength: float = 1.0            # how established this concept is (grows with use, decays)
    hits: int = 0                    # how many times it has been activated
    born_tick: int = 0
    last_tick: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "strength": round(self.strength, 6), "hits": self.hits,
                "born_tick": self.born_tick, "last_tick": self.last_tick}


@dataclass
class Activation:
    """The result of activating the graph with a set of concepts."""

    direct: List[str] = field(default_factory=list)      # concepts present in the stimulus
    spread: List[Tuple[str, float]] = field(default_factory=list)  # neighbours, by activation
    born: List[str] = field(default_factory=list)        # concepts that did not exist before
    strengthened: int = 0                                # edges reinforced this activation
    pruned: int = 0                                      # edges dropped this activation

    def to_dict(self) -> Dict[str, Any]:
        return {"direct": self.direct, "spread": [[n, round(w, 6)] for n, w in self.spread],
                "born": self.born, "strengthened": self.strengthened, "pruned": self.pruned}


@dataclass
class GraphStats:
    """A snapshot of the graph's shape — what actually got built."""

    nodes: int = 0
    edges: int = 0
    ticks: int = 0
    born_total: int = 0
    pruned_total: int = 0
    mean_degree: float = 0.0
    mean_weight: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "ticks": self.ticks,
                "born_total": self.born_total, "pruned_total": self.pruned_total,
                "mean_degree": round(self.mean_degree, 4),
                "mean_weight": round(self.mean_weight, 6)}


def _pair(a: str, b: str) -> Tuple[str, str]:
    """Undirected edge key — always ordered, so (a,b) and (b,a) are the same edge."""
    return (a, b) if a <= b else (b, a)


class DynamicNeuralGraph:
    """A self-reconfiguring concept graph: Hebbian growth, disuse decay, bounded capacity."""

    def __init__(self, *, max_nodes: int = 4096, max_edges: int = 32768,
                 hebbian_rate: float = 0.15, decay_rate: float = 0.01,
                 prune_threshold: float = 0.02, rewire_budget: int = 256,
                 spread_depth: int = 2, spread_falloff: float = 0.5) -> None:
        self.max_nodes = max(1, int(max_nodes))
        self.max_edges = max(1, int(max_edges))
        self.hebbian_rate = float(hebbian_rate)
        self.decay_rate = float(decay_rate)
        self.prune_threshold = float(prune_threshold)
        self.rewire_budget = max(0, int(rewire_budget))
        self.spread_depth = max(1, int(spread_depth))
        self.spread_falloff = float(spread_falloff)

        self.nodes: Dict[str, Concept] = {}
        self.edges: Dict[Tuple[str, str], float] = {}
        self._adj: Dict[str, set] = {}
        self.ticks = 0
        self.born_total = 0
        self.pruned_total = 0

    # ---- basic shape ----------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, label: str) -> bool:
        return str(label) in self.nodes

    def weight(self, a: str, b: str) -> float:
        """The current association strength between two concepts (0.0 when unrelated)."""
        return float(self.edges.get(_pair(str(a), str(b)), 0.0))

    def neighbours(self, label: str, *, k: int = 8) -> List[Tuple[str, float]]:
        """The ``k`` most strongly associated concepts, strongest first."""
        try:
            label = str(label)
            out = [(o, self.edges.get(_pair(label, o), 0.0)) for o in self._adj.get(label, ())]
            out.sort(key=lambda p: -p[1])
            return out[: max(0, int(k))]
        except Exception:  # noqa: BLE001
            return []

    # ---- growth ---------------------------------------------------------- #
    def _touch(self, label: str) -> bool:
        """Ensure a node exists; returns True when it was born on this call."""
        node = self.nodes.get(label)
        if node is not None:
            node.hits += 1
            node.last_tick = self.ticks
            node.strength = min(8.0, node.strength + self.hebbian_rate)
            return False
        self.nodes[label] = Concept(label=label, hits=1, born_tick=self.ticks,
                                    last_tick=self.ticks)
        self._adj[label] = set()
        self.born_total += 1
        self._enforce_node_cap()
        return True

    def _link(self, a: str, b: str) -> None:
        key = _pair(a, b)
        # Growth is saturating: an edge approaches 1.0 but never runs away, so a repeated pair
        # cannot dominate the graph the way an unbounded counter would.
        w = self.edges.get(key, 0.0)
        self.edges[key] = w + self.hebbian_rate * (1.0 - w)
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)

    def activate(self, concepts: Sequence[str] | str, *, learn: bool = True) -> Activation:
        """Co-activate ``concepts``: born-if-new, Hebbian-linked pairwise, then spread.

        This is the graph's whole learning rule. ``learn=False`` reads without rewiring, which
        is what a hypothetical ("what *would* relate to this?") needs.
        """
        act = Activation()
        try:
            labels = concepts_in(concepts) if isinstance(concepts, str) else \
                [str(c).strip().lower() for c in concepts if str(c).strip()]
            labels = [ln for ln in labels if ln][: self.rewire_budget or len(labels)]
            if not labels:
                return act

            if learn:
                self.ticks += 1
                for ln in labels:
                    if self._touch(ln):
                        act.born.append(ln)
                # Pairwise Hebbian binding, but budgeted: a very long turn must not spend the
                # whole tick building a clique. n*(n-1)/2 is capped by rewire_budget.
                budget = self.rewire_budget
                for i, a in enumerate(labels):
                    for b in labels[i + 1:]:
                        if budget <= 0:
                            break
                        self._link(a, b)
                        act.strengthened += 1
                        budget -= 1
                    if budget <= 0:
                        break
                act.pruned = self._decay_and_prune()
                self._enforce_edge_cap()

            act.direct = [ln for ln in labels if ln in self.nodes]
            act.spread = self._spread(act.direct)
            return act
        except Exception:  # noqa: BLE001 — activation is never allowed to break a turn
            return act

    def _spread(self, seeds: Sequence[str], *, k: int = 12) -> List[Tuple[str, float]]:
        """Spreading activation: neighbours of neighbours, attenuated by depth and edge weight."""
        try:
            scores: Dict[str, float] = {}
            frontier = {s: 1.0 for s in seeds}
            seen = set(seeds)
            for depth in range(self.spread_depth):
                nxt: Dict[str, float] = {}
                falloff = self.spread_falloff ** (depth + 1)
                for src, energy in frontier.items():
                    for dst in self._adj.get(src, ()):
                        w = self.edges.get(_pair(src, dst), 0.0)
                        gain = energy * w * falloff
                        if gain <= 0.0:
                            continue
                        scores[dst] = scores.get(dst, 0.0) + gain
                        if dst not in seen:
                            nxt[dst] = max(nxt.get(dst, 0.0), gain)
                seen.update(nxt)
                frontier = nxt
                if not frontier:
                    break
            for s in seeds:
                scores.pop(s, None)
            out = sorted(scores.items(), key=lambda p: -p[1])
            return out[: max(0, int(k))]
        except Exception:  # noqa: BLE001
            return []

    # ---- forgetting ------------------------------------------------------ #
    def _decay_and_prune(self) -> int:
        """Every edge fades a little each tick; the faded-out ones are cut. Returns cut count."""
        if self.decay_rate <= 0.0:
            return 0
        dead: List[Tuple[str, str]] = []
        keep = 1.0 - self.decay_rate
        for key, w in self.edges.items():
            w *= keep
            if w < self.prune_threshold:
                dead.append(key)
            else:
                self.edges[key] = w
        for key in dead:
            self._cut(key)
        self.pruned_total += len(dead)
        return len(dead)

    def _cut(self, key: Tuple[str, str]) -> None:
        self.edges.pop(key, None)
        a, b = key
        if a in self._adj:
            self._adj[a].discard(b)
        if b in self._adj:
            self._adj[b].discard(a)

    def _enforce_edge_cap(self) -> None:
        """Past ``max_edges`` the weakest associations go first — capacity is real."""
        over = len(self.edges) - self.max_edges
        if over <= 0:
            return
        weakest = sorted(self.edges.items(), key=lambda p: p[1])[:over]
        for key, _ in weakest:
            self._cut(key)
        self.pruned_total += len(weakest)

    def _enforce_node_cap(self) -> None:
        """Past ``max_nodes`` the least-established, least-recently-used concepts are dropped."""
        over = len(self.nodes) - self.max_nodes
        if over <= 0:
            return
        victims = sorted(self.nodes.values(),
                         key=lambda c: (c.strength, c.last_tick))[:over]
        for node in victims:
            for other in list(self._adj.get(node.label, ())):
                self._cut(_pair(node.label, other))
            self._adj.pop(node.label, None)
            self.nodes.pop(node.label, None)

    # ---- introspection --------------------------------------------------- #
    def gaps(self, *, k: int = 5) -> List[str]:
        """Concepts the graph knows *of* but barely knows — the honest edge of her knowledge.

        A node with many hits but few/weak edges is a word she keeps meeting without ever
        connecting to anything. That is exactly what :mod:`nyxara.nyx.episteme` should go
        after, and what :mod:`nyxara.nyx.ground` should try to ground.
        """
        try:
            scored: List[Tuple[float, str]] = []
            for label, node in self.nodes.items():
                degree = len(self._adj.get(label, ()))
                bound = sum(self.edges.get(_pair(label, o), 0.0)
                            for o in self._adj.get(label, ()))
                # high exposure, low connection = a real gap
                isolation = float(node.hits) / (1.0 + bound + degree)
                scored.append((isolation, label))
            scored.sort(reverse=True)
            return [label for _, label in scored[: max(0, int(k))]]
        except Exception:  # noqa: BLE001
            return []

    def stats(self) -> GraphStats:
        try:
            n, e = len(self.nodes), len(self.edges)
            total_w = math.fsum(self.edges.values()) if e else 0.0
            return GraphStats(nodes=n, edges=e, ticks=self.ticks, born_total=self.born_total,
                              pruned_total=self.pruned_total,
                              mean_degree=(2.0 * e / n) if n else 0.0,
                              mean_weight=(total_w / e) if e else 0.0)
        except Exception:  # noqa: BLE001
            return GraphStats()

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": {"max_nodes": self.max_nodes, "max_edges": self.max_edges,
                       "hebbian_rate": self.hebbian_rate, "decay_rate": self.decay_rate,
                       "prune_threshold": self.prune_threshold,
                       "rewire_budget": self.rewire_budget,
                       "spread_depth": self.spread_depth,
                       "spread_falloff": self.spread_falloff},
            "ticks": self.ticks, "born_total": self.born_total,
            "pruned_total": self.pruned_total,
            "nodes": [c.to_dict() for c in self.nodes.values()],
            "edges": [[a, b, round(w, 6)] for (a, b), w in self.edges.items()],
        }

    def load_dict(self, data: Any) -> bool:
        """Restore a saved graph. A corrupt or absent sidecar leaves an empty graph, never raises."""
        try:
            if not isinstance(data, dict):
                return False
            self.nodes, self.edges, self._adj = {}, {}, {}
            for raw in data.get("nodes", []) or []:
                label = str(raw.get("label", "")).strip()
                if not label:
                    continue
                self.nodes[label] = Concept(
                    label=label, strength=float(raw.get("strength", 1.0)),
                    hits=int(raw.get("hits", 0)), born_tick=int(raw.get("born_tick", 0)),
                    last_tick=int(raw.get("last_tick", 0)))
                self._adj[label] = set()
            for raw in data.get("edges", []) or []:
                if not (isinstance(raw, (list, tuple)) and len(raw) == 3):
                    continue
                a, b, w = str(raw[0]), str(raw[1]), float(raw[2])
                if a not in self.nodes or b not in self.nodes:
                    continue
                self.edges[_pair(a, b)] = w
                self._adj[a].add(b)
                self._adj[b].add(a)
            self.ticks = int(data.get("ticks", 0))
            self.born_total = int(data.get("born_total", len(self.nodes)))
            self.pruned_total = int(data.get("pruned_total", 0))
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
