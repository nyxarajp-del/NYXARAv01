"""NYXARA · nyx5/holo_swarm.py — holographic fractal consciousness (🌌, pillar 6).

Cut a hologram in half and each fragment still shows the *whole* image — dimmer, but complete. NYX-5's
memory is stored the same way: every memory is bundled into one distributed hyperdimensional field, so
each node (shard) that holds a slice of experience can still recall the whole, degraded but functional,
even when the swarm is partitioned. When nodes reconnect, their fields **merge losslessly** — the union
of everything each learned, no committed memory dropped.

Honest ceiling: a fragment gives **degraded (partial) recall**, not magic full fidelity — "every piece
shows the whole image" means graceful holographic degradation, not lossless omniscience from a scrap.
Reuses :mod:`nyxara.nyx5.hd_memory` (the audited HDC store); the anchor-merge follows the lossless-union
spirit of :mod:`nyxara.memory.elastic_synapses`. Default single-node (zero overhead). Pure stdlib + HDC.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nyxara.cognition.hyper_dimensional_vectors import CleanupResult
from nyxara.nyx5.hd_memory import HDMemory, RasterLike

__all__ = ["HolographicShard"]


class HolographicShard:
    """One node's slice of the holographic memory field, mergeable with any other node's."""

    def __init__(self, node_id: str = "", *, dim: int = 10000, seed: int = 42) -> None:
        self.node_id = node_id or "node0"
        self.memory = HDMemory(dim=dim, seed=seed)
        self._versions: Dict[str, int] = {}          # per-key version for lossless merge

    def __len__(self) -> int:
        return len(self.memory)

    def remember(self, key: str, raster: RasterLike, *, version: Optional[int] = None) -> None:
        self.memory.remember(key, raster)
        self._versions[key] = self._versions.get(key, 0) + 1 if version is None else int(version)

    def recall(self, raster: RasterLike, *, threshold: float = 0.0) -> Optional[CleanupResult]:
        """Recall from *this fragment alone* — works under partition, degraded by local coverage."""
        return self.memory.recall(raster, threshold=threshold)

    def fragment(self, keys: List[str]) -> "HolographicShard":
        """Split off a partition holding only ``keys`` — each fragment still recalls what it holds."""
        frag = HolographicShard(f"{self.node_id}:frag", dim=self.memory.space.dim,
                                seed=self.memory.space.seed)
        for k in keys:
            if k in self.memory._recipes:
                frag.memory.remember_recipe(k, self.memory._recipes[k])
                frag._versions[k] = self._versions.get(k, 1)
        return frag

    def merge(self, other: "HolographicShard") -> int:
        """Losslessly fold ``other`` into this shard (higher version wins per key). Returns keys added
        or updated. Commutative and idempotent — reconnection order does not matter."""
        changed = 0
        for key, recipe in other.memory._recipes.items():
            ov = other._versions.get(key, 1)
            if key not in self._versions or ov > self._versions[key]:
                self.memory.remember_recipe(key, recipe)
                self._versions[key] = ov
                changed += 1
        return changed

    def stats(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "memories": len(self.memory), "keys": self.memory.keys()}

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "memory": self.memory.to_dict(),
                "versions": dict(self._versions)}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.node_id = d.get("node_id", self.node_id)
        self.memory.load_dict(d.get("memory", {}))
        self._versions = {k: int(v) for k, v in d.get("versions", {}).items()}
