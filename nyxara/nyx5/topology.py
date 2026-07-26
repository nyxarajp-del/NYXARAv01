"""NYXARA · nyx5/topology.py — the live, rewiring graph (⚡, pillar 2).

Human synapses are not static: learning **physically grows** new connections and unused ones are
**pruned** away. NYX-5's structure is therefore not a fixed weight matrix but a *live graph* — a
sparse dict-of-dicts adjacency that changes shape as she learns:

* **prune** — synapses whose weight has decayed below a floor and that have been silent for a while
  are removed (freeing capacity, the "use it or lose it" rule).
* **grow** — pairs of neurons that keep firing close together but are *not yet connected* get a new
  synapse, up to a hard ``max_synapses`` cap (structural potentiation).

This is dynamic structural plasticity: continuous, local, unsupervised, bounded. Pure standard
library. Depends on nyx5/synapse.py.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from nyxara.nyx5.synapse import Synapse

__all__ = ["LiveTopology"]


class LiveTopology:
    """A sparse, mutable set of synapses indexed by pre-neuron for fast fan-out.

    ``adj[pre][post]`` is the :class:`Synapse`. Growth and pruning keep the graph both plastic (new
    learning is possible) and bounded (it can never blow up).
    """

    def __init__(self, *, max_synapses: int = 8192, prune_threshold: float = 0.05,
                 prune_stale_ms: float = 1000.0) -> None:
        self.max_synapses = int(max_synapses)
        self.prune_threshold = float(prune_threshold)
        self.prune_stale_ms = float(prune_stale_ms)
        self._adj: Dict[int, Dict[int, Synapse]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def outgoing(self, pre: int) -> Iterable[Synapse]:
        return self._adj.get(pre, {}).values()

    def has(self, pre: int, post: int) -> bool:
        return post in self._adj.get(pre, {})

    def get(self, pre: int, post: int) -> Optional[Synapse]:
        return self._adj.get(pre, {}).get(post)

    def all_synapses(self) -> Iterator[Synapse]:
        for row in self._adj.values():
            yield from row.values()

    def connect(self, pre: int, post: int, weight: float = 0.5, delay: float = 1.0) -> Optional[Synapse]:
        """Add a synapse ``pre -> post``. Returns it, or None if the cap is reached (or a self-loop).

        An existing edge is returned unchanged (idempotent).
        """
        if pre == post:
            return None
        existing = self._adj.get(pre, {}).get(post)
        if existing is not None:
            return existing
        if self._count >= self.max_synapses:
            return None
        syn = Synapse(pre=pre, post=post, weight=weight, delay=delay)
        self._adj.setdefault(pre, {})[post] = syn
        self._count += 1
        return syn

    def disconnect(self, pre: int, post: int) -> bool:
        row = self._adj.get(pre)
        if row and post in row:
            del row[post]
            self._count -= 1
            if not row:
                del self._adj[pre]
            return True
        return False

    def prune(self, now: float) -> int:
        """Remove weak *and* stale synapses. Returns how many were pruned.

        A synapse is pruned only when it is both below the weight floor and has been silent longer
        than ``prune_stale_ms`` — so a weak-but-active connection (still learning) survives.
        """
        doomed: List[Tuple[int, int]] = []
        for syn in self.all_synapses():
            if syn.weight < self.prune_threshold and (now - syn.stale_since) > self.prune_stale_ms:
                doomed.append((syn.pre, syn.post))
        for pre, post in doomed:
            self.disconnect(pre, post)
        return len(doomed)

    def grow(self, correlated_pairs: Iterable[Tuple[int, int]], *, weight: float = 0.5) -> int:
        """Create synapses for co-active, not-yet-connected pairs, up to the cap. Returns count added."""
        added = 0
        for pre, post in correlated_pairs:
            if self._count >= self.max_synapses:
                break
            if pre != post and not self.has(pre, post):
                if self.connect(pre, post, weight=weight) is not None:
                    added += 1
        return added

    def snapshot_stats(self) -> Dict[str, Any]:
        weights = [s.weight for s in self.all_synapses()]
        mean_w = sum(weights) / len(weights) if weights else 0.0
        return {"synapses": self._count, "max_synapses": self.max_synapses,
                "mean_weight": round(mean_w, 6),
                "saturation": round(self._count / self.max_synapses, 4) if self.max_synapses else 0.0}

    def to_dict(self) -> Dict[str, Any]:
        return {"max_synapses": self.max_synapses, "prune_threshold": self.prune_threshold,
                "prune_stale_ms": self.prune_stale_ms,
                "synapses": [s.to_dict() for s in self.all_synapses()]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.max_synapses = int(d.get("max_synapses", self.max_synapses))
        self.prune_threshold = float(d.get("prune_threshold", self.prune_threshold))
        self.prune_stale_ms = float(d.get("prune_stale_ms", self.prune_stale_ms))
        self._adj.clear()
        self._count = 0
        for sd in d.get("synapses", []):
            syn = Synapse.from_dict(sd)
            self._adj.setdefault(syn.pre, {})[syn.post] = syn
            self._count += 1
