"""NYXARA · nyx5/continuum.py — the sovereign narrative continuum (🌀, pillar 20).

Most models live inside a sliding context window and forget. NYX-5 weaves every turn into a persistent
**causal event graph** in hyperdimensional memory: each turn is a node, linked to the one before, and
recallable long after by content. Decisions the Master made weeks ago — a naming convention, an
architectural choice, a preference — surface naturally in later conversation, without any force-fed
prompt template.

Honest ceiling: recall is **best-effort associative retrieval**, not literally infinite, perfect memory.
"Zero context loss" is the aspiration; when a memory genuinely is not found she says so rather than
confabulating. Reuses the HDC algebra (:mod:`nyxara.cognition.hyper_dimensional_vectors`) for
content-addressable recall. Pure standard library on top of the HDC module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nyxara.cognition.hyper_dimensional_vectors import HyperSpace, ItemMemory

__all__ = ["EventNode", "NarrativeContinuum"]


@dataclass
class EventNode:
    """One causal event in the narrative: an id, its text, an optional decision, and its predecessor."""

    node_id: int
    text: str
    decision: Optional[str] = None
    prev: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "text": self.text, "decision": self.decision,
                "prev": self.prev}


class NarrativeContinuum:
    """A persistent causal event graph with hyperdimensional, content-addressable recall."""

    def __init__(self, *, dim: int = 10000, seed: int = 42, recall_threshold: float = 0.12) -> None:
        self.space = HyperSpace(dim=dim, seed=seed)
        self.toks = ItemMemory(self.space)               # token alphabet
        self.index = ItemMemory(self.space)              # node_id -> content vector
        self.recall_threshold = float(recall_threshold)
        self._nodes: List[EventNode] = []

    def __len__(self) -> int:
        return len(self._nodes)

    def _embed(self, text: str):
        toks = [t for t in text.lower().replace("-", " ").split() if t]
        if not toks:
            return self.space.zero()
        return self.space.bundle(*[self.toks.symbol(f"tok:{t}") for t in toks])

    def weave(self, text: str, *, decision: Optional[str] = None) -> EventNode:
        """Append a turn as a causal node, linked to its predecessor, indexed for recall."""
        node_id = len(self._nodes)
        prev = node_id - 1 if node_id > 0 else None
        node = EventNode(node_id=node_id, text=text, decision=decision, prev=prev)
        self._nodes.append(node)
        self.index.add(f"node:{node_id}", self._embed(f"{text} {decision or ''}"))
        return node

    def recall(self, query: str) -> Optional[EventNode]:
        """Find the most relevant past node for ``query``; return None if nothing is close enough."""
        if not self._nodes:
            return None
        result = self.index.cleanup(self._embed(query))
        if result.name is None or result.score < self.recall_threshold:
            return None
        return self._nodes[int(result.name[len("node:"):])]

    def decisions(self) -> List[str]:
        return [n.decision for n in self._nodes if n.decision]

    def chain(self, node_id: int) -> List[EventNode]:
        """Walk the causal predecessor chain back from ``node_id`` (most recent first)."""
        out: List[EventNode] = []
        cur: Optional[int] = node_id
        while cur is not None and 0 <= cur < len(self._nodes):
            node = self._nodes[cur]
            out.append(node)
            cur = node.prev
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"dim": self.space.dim, "seed": self.space.seed,
                "recall_threshold": self.recall_threshold,
                "nodes": [n.to_dict() for n in self._nodes]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.space = HyperSpace(dim=int(d.get("dim", 10000)), seed=int(d.get("seed", 42)))
        self.toks = ItemMemory(self.space)
        self.index = ItemMemory(self.space)
        self.recall_threshold = float(d.get("recall_threshold", self.recall_threshold))
        self._nodes = []
        for nd in d.get("nodes", []):
            node = EventNode(node_id=int(nd["node_id"]), text=nd["text"],
                             decision=nd.get("decision"), prev=nd.get("prev"))
            self._nodes.append(node)
            self.index.add(f"node:{node.node_id}", self._embed(f"{node.text} {node.decision or ''}"))
