"""Memory store — working / episodic / semantic / procedural over kv + vector.

A single :class:`MemoryStore` aggregates four subsystems that mirror human memory:

  * **working**  — a tiny, fast, capacity-bounded set (what's "in mind" right now);
  * **episodic** — a time-ordered log of experienced events (the autobiographical tape);
  * **semantic** — durable facts / beliefs, each provenance-tagged;
  * **procedural** — reusable skills / procedures keyed by name.

All items are embedded into a vector space for associative recall. To avoid a hard
dependency on a heavy embedding model, a deterministic hashing embedder is used by
default; a real embedder (sentence-transformers, an API, …) can be injected.
"""
from __future__ import annotations

import enum
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from nyxara.contracts import Belief, Provenance, new_id

Embedder = Callable[[str], np.ndarray]
_EMBED_DIM = 256


def _hash_embed(text: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """Deterministic bag-of-features embedding — no model required.

    Hashes whitespace tokens into a fixed-width vector and L2-normalises. Crude but real:
    it gives stable, comparable vectors so associative retrieval works out of the box.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class Modality(str, enum.Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryItem:
    """A single stored memory."""

    id: str
    modality: Modality
    content: Any
    text: str                                   # textual key used for embedding/recall
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0                        # decays over time, boosted on recall
    emotion: float = 0.0                         # valence tag, -1..1 (for indexing)
    context: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None

    def touch(self) -> None:
        self.last_access = time.time()
        self.access_count += 1
        self.strength = min(2.0, self.strength + 0.1)


class MemoryStore:
    """Aggregate of the four memory subsystems with a shared vector index."""

    def __init__(self, *, embedder: Embedder | None = None,
                 working_capacity: int = 7) -> None:
        self._embed = embedder or _hash_embed
        self._items: dict[str, MemoryItem] = {}
        self._working: deque[str] = deque(maxlen=working_capacity)
        self._by_modality: dict[Modality, list[str]] = {m: [] for m in Modality}
        self._kv: dict[str, Any] = {}

    # ── writing ──────────────────────────────────────────────────────────────────
    def remember(self, content: Any, *, modality: Modality, text: str | None = None,
                 emotion: float = 0.0, context: dict[str, Any] | None = None,
                 provenance: Provenance | None = None) -> MemoryItem:
        key_text = text if text is not None else str(content)
        item = MemoryItem(
            id=new_id("mem"), modality=modality, content=content, text=key_text,
            embedding=self._embed(key_text), emotion=emotion,
            context=context or {}, provenance=provenance,
        )
        self._items[item.id] = item
        self._by_modality[modality].append(item.id)
        if modality is Modality.WORKING:
            if len(self._working) == self._working.maxlen:
                evicted = self._working[0]
                # working items overflow to episodic rather than vanishing
                self._items[evicted].modality = Modality.EPISODIC
                self._by_modality[Modality.EPISODIC].append(evicted)
            self._working.append(item.id)
        return item

    def remember_episode(self, content: Any, **kw: Any) -> MemoryItem:
        return self.remember(content, modality=Modality.EPISODIC, **kw)

    def remember_fact(self, belief: Belief) -> MemoryItem:
        return self.remember(belief, modality=Modality.SEMANTIC, text=belief.claim,
                             provenance=belief.provenance)

    def remember_skill(self, name: str, procedure: Any) -> MemoryItem:
        return self.remember(procedure, modality=Modality.PROCEDURAL, text=name)

    # ── key/value side-channel ─────────────────────────────────────────────────────
    def put(self, key: str, value: Any) -> None:
        self._kv[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    # ── reading ────────────────────────────────────────────────────────────────────
    def by_id(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    def working_set(self) -> list[MemoryItem]:
        return [self._items[i] for i in self._working]

    def items(self, modality: Modality | None = None) -> list[MemoryItem]:
        if modality is None:
            return list(self._items.values())
        return [self._items[i] for i in self._by_modality[modality]]

    def search(self, query: str, *, k: int = 5,
               modality: Modality | None = None) -> list[tuple[float, MemoryItem]]:
        """Cosine nearest-neighbour recall over the chosen modality (or all)."""
        q = self._embed(query)
        pool = self.items(modality)
        scored: list[tuple[float, MemoryItem]] = []
        for item in pool:
            denom = (np.linalg.norm(q) * np.linalg.norm(item.embedding)) or 1.0
            sim = float(np.dot(q, item.embedding) / denom)
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, item in scored[:k]:
            item.touch()
        return scored[:k]

    # ── maintenance ─────────────────────────────────────────────────────────────────
    def decay(self, rate: float = 0.01) -> None:
        """Apply forgetting: reduce strength of all non-working items."""
        for item in self._items.values():
            if item.modality is not Modality.WORKING:
                item.strength = max(0.0, item.strength - rate)

    def forget_weak(self, threshold: float = 0.05) -> int:
        """Drop items whose strength fell below threshold. Returns count removed."""
        doomed = [i for i, it in self._items.items()
                  if it.strength < threshold and it.modality is not Modality.SEMANTIC]
        for i in doomed:
            it = self._items.pop(i)
            self._by_modality[it.modality].remove(i)
        return len(doomed)

    @property
    def stats(self) -> dict[str, int]:
        return {m.value: len(ids) for m, ids in self._by_modality.items()} | {
            "total": len(self._items), "kv": len(self._kv)}


__all__ = ["MemoryStore", "MemoryItem", "Modality", "Embedder"]
