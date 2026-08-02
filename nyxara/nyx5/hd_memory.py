"""NYXARA · nyx5/hd_memory.py — long-term memory in 10,000-D hypervectors (🧠, pillar 2).

A spike raster is fleeting — it vanishes the instant the cascade ends. To *remember* it without the
catastrophic forgetting that plagues gradient-trained nets, NYX-5 compresses each raster into a single
**10,000-dimensional hypervector** and files it in an associative cleanup memory. Because thousands of
random hypervectors are near-orthogonal, tens of thousands of memories coexist in the one space and any
measured similarity is signal — old memories do not decay when new ones are written.

This is a **bridge**, not a second implementation: it imports and reuses the audited HDC algebra
(``HyperSpace``, ``ItemMemory``, permute/bundle/cleanup) from
:mod:`nyxara.cognition.hyper_dimensional_vectors`. NYX-5 adds only the spike→vector encoding: each
firing neuron contributes its atomic symbol, permuted by a coarse time-bucket so *order* survives, and
the whole set is bundled into one holographic vector.

Pure standard library on top of the HDC module (numpy-accelerated when present). Deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from nyxara.cognition.hyper_dimensional_vectors import (
    CleanupResult,
    HyperSpace,
    Hypervector,
    ItemMemory,
)

__all__ = ["SpikeEncoder", "HDMemory"]

RasterLike = Union[Iterable[int], Iterable[Tuple[float, int]]]


class SpikeEncoder:
    """Encode a spike raster into one bundled hypervector (order stamped by time-bucket permutation)."""

    def __init__(self, space: HyperSpace, *, atoms: Optional[ItemMemory] = None,
                 time_bucket_ms: float = 5.0) -> None:
        self.space = space
        self.atoms = atoms if atoms is not None else ItemMemory(space)
        self.time_bucket_ms = float(time_bucket_ms)

    def recipe(self, raster: RasterLike) -> List[Tuple[int, int]]:
        """Normalise raw input to a compact, replayable list of (neuron_id, time_bucket)."""
        out: List[Tuple[int, int]] = []
        for item in raster:
            if isinstance(item, int):
                out.append((item, 0))
            else:
                t, nid = item  # type: ignore[misc]
                out.append((int(nid), int(float(t) // self.time_bucket_ms)))
        return out

    def encode_recipe(self, recipe: Sequence[Tuple[int, int]]) -> Hypervector:
        if not recipe:
            return self.space.zero()
        vecs = [self.space.permute(self.atoms.symbol(f"n{nid}"), bucket % self.space.dim)
                for nid, bucket in recipe]
        return self.space.bundle(*vecs)

    def encode(self, raster: RasterLike) -> Hypervector:
        return self.encode_recipe(self.recipe(raster))


class HDMemory:
    """Associative long-term store: remember a raster under a key, recall the nearest by similarity."""

    def __init__(self, *, dim: int = 10000, seed: int = 42, time_bucket_ms: float = 5.0) -> None:
        self.space = HyperSpace(dim=dim, seed=seed)
        self.encoder = SpikeEncoder(self.space, time_bucket_ms=time_bucket_ms)
        self.store = ItemMemory(self.space)
        self._recipes: Dict[str, List[Tuple[int, int]]] = {}

    def __len__(self) -> int:
        return len(self._recipes)

    def _put(self, key: str, vec: Hypervector) -> None:
        # ItemMemory.add is append-only; update in place so re-remembering a key refreshes it.
        self.store._items[key] = vec
        self.store._matrix = None

    def remember(self, key: str, raster: RasterLike) -> Hypervector:
        """File ``raster`` under ``key`` as a 10k-D vector (idempotent-updating per key)."""
        recipe = self.encoder.recipe(raster)
        vec = self.encoder.encode_recipe(recipe)
        self._recipes[key] = recipe
        self._put(key, vec)
        return vec

    def remember_recipe(self, key: str, recipe: Sequence[Tuple[int, int]]) -> Hypervector:
        """File an already-normalised ``(neuron_id, bucket)`` recipe directly (for merge/fragment copies)."""
        recipe = [(int(a), int(b)) for a, b in recipe]
        vec = self.encoder.encode_recipe(recipe)
        self._recipes[key] = recipe
        self._put(key, vec)
        return vec

    def recall(self, raster: RasterLike, *, threshold: float = 0.0) -> Optional[CleanupResult]:
        """Return the nearest stored memory to ``raster`` (or None if empty / below ``threshold``)."""
        if not self._recipes:
            return None
        result = self.store.cleanup(self.encoder.encode(raster))
        if result.name is None or result.score < threshold:
            return None
        return result

    def keys(self) -> List[str]:
        return list(self._recipes)

    def to_dict(self) -> Dict[str, Any]:
        # store recipes (compact + deterministic), not raw 10k-float vectors — vectors are regenerable.
        return {"dim": self.space.dim, "seed": self.space.seed,
                "time_bucket_ms": self.encoder.time_bucket_ms,
                "recipes": {k: [list(p) for p in v] for k, v in self._recipes.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.space = HyperSpace(dim=int(d.get("dim", 10000)), seed=int(d.get("seed", 42)))
        self.encoder = SpikeEncoder(self.space, time_bucket_ms=float(d.get("time_bucket_ms", 5.0)))
        self.store = ItemMemory(self.space)
        self._recipes = {}
        for key, recipe in d.get("recipes", {}).items():
            r = [(int(a), int(b)) for a, b in recipe]
            self._recipes[key] = r
            self._put(key, self.encoder.encode_recipe(r))
