"""NYXARA · nyx5/mesh.py — the non-local entangled mesh (🌀, pillar 15).

When several NYX-5 nodes share a mind — the Master's phone, a laptop, a server — they keep their state in
step through a **conflict-free replicated data type (CRDT)**: each node emits deltas as it learns, and
merges the deltas it receives. Because the merge is commutative, associative and idempotent, every node
converges to the same state regardless of message order or duplication — no locks, no coordinator, no
lost writes.

Honest ceiling (non-negotiable): this is **low-latency eventual consistency**, propagating as fast as the
network allows. It is **not** literal zero-latency, and it is **not** quantum entanglement — entanglement
cannot transmit information faster than light. "Quantum-entangled / instant" is an analogy for a good
CRDT, never a measured fact. Reuses the transport spirit of :mod:`nyxara.agency.distributed`; the CRDT
here is a last-writer-wins map with a per-node logical clock. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["Delta", "EntangledMesh"]

# a versioned value: (value, timestamp, node_id) — the LWW register.
Stamp = Tuple[Any, int, str]


@dataclass
class Delta:
    """A batch of key updates to ship to peers. Bounded by the mesh's byte cap."""

    node_id: str
    updates: Dict[str, Stamp] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id,
                "updates": {k: list(v) for k, v in self.updates.items()}}


def _dominates(a: Stamp, b: Stamp) -> bool:
    """LWW ordering: higher timestamp wins; ties broken deterministically by node_id."""
    (_, ta, na), (_, tb, nb) = a, b
    return (ta, na) > (tb, nb)


class EntangledMesh:
    """A last-writer-wins CRDT map: local sets emit deltas; merges are conflict-free and idempotent."""

    def __init__(self, node_id: str, *, max_delta_bytes: int = 1_048_576) -> None:
        self.node_id = node_id
        self.max_delta_bytes = int(max_delta_bytes)
        self._state: Dict[str, Stamp] = {}
        self._clock = 0
        self._pending: Dict[str, Stamp] = {}

    def set(self, key: str, value: Any) -> None:
        """Locally set ``key`` and stage a delta for peers."""
        self._clock += 1
        stamp: Stamp = (value, self._clock, self.node_id)
        self._state[key] = stamp
        self._pending[key] = stamp

    def get(self, key: str, default: Any = None) -> Any:
        s = self._state.get(key)
        return s[0] if s is not None else default

    def keys(self) -> List[str]:
        return list(self._state)

    def emit(self) -> Delta:
        """Produce (and clear) the staged delta, respecting the byte cap (largest keys deferred)."""
        delta = Delta(node_id=self.node_id)
        size = 0
        for key, stamp in sorted(self._pending.items(), key=lambda kv: len(repr(kv[1]))):
            grow = len(key) + len(repr(stamp))
            if size + grow > self.max_delta_bytes:
                break                                 # cap reached; the rest re-emits next round
            delta.updates[key] = stamp
            size += grow
        for key in delta.updates:
            self._pending.pop(key, None)
        return delta

    def merge(self, delta: Delta) -> int:
        """Fold a peer's delta in. Conflict-free: LWW per key, commutative + idempotent. Returns
        the number of keys actually changed."""
        changed = 0
        for key, stamp in delta.updates.items():
            cur = self._state.get(key)
            incoming: Stamp = (stamp[0], int(stamp[1]), stamp[2])
            if cur is None or _dominates(incoming, cur):
                self._state[key] = incoming
                self._clock = max(self._clock, incoming[1])
                changed += 1
        return changed

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "clock": self._clock,
                "state": {k: list(v) for k, v in self._state.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.node_id = d.get("node_id", self.node_id)
        self._clock = int(d.get("clock", 0))
        self._state = {k: (v[0], int(v[1]), v[2]) for k, v in d.get("state", {}).items()}
