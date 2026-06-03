"""Retrieval — associative spreading activation, emotional/context indexing, déjà-vu.

Human recall is not a database lookup; it is *spreading activation*. A cue lights up the
memories most similar to it, and that activation spreads to associated memories, decaying
with distance. Recency, emotional charge, and matching context all bias what surfaces.

"Déjà-vu" is modelled honestly as a *signal*, not magic: high global familiarity (the cue
strongly matches the semantic gist) combined with low episodic specificity (no concrete
episode matches) — i.e. "this feels familiar but I can't place it."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nyxara.memory.store import MemoryItem, MemoryStore, Modality


@dataclass
class RetrievalResult:
    item: MemoryItem
    activation: float           # final spread activation
    base_similarity: float      # raw cue similarity before spreading


@dataclass
class DejaVuSignal:
    familiarity: float          # gist-level match strength, 0..1
    specificity: float          # best concrete episodic match, 0..1
    is_deja_vu: bool            # familiar yet unplaceable


class AssociativeRetriever:
    """Spreading-activation retrieval over a :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore, *, decay: float = 0.6,
                 hops: int = 2, emotion_weight: float = 0.2,
                 recency_weight: float = 0.1) -> None:
        self._store = store
        self._decay = decay              # activation retained per associative hop
        self._hops = hops
        self._emotion_w = emotion_weight
        self._recency_w = recency_weight

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def _context_bonus(self, item: MemoryItem, context: dict[str, Any]) -> float:
        if not context or not item.context:
            return 0.0
        shared = sum(1 for k, v in context.items() if item.context.get(k) == v)
        return 0.1 * shared

    def recall(self, cue: str, *, k: int = 5, context: dict[str, Any] | None = None,
               mood: float = 0.0) -> list[RetrievalResult]:
        """Retrieve memories by spreading activation from a textual cue."""
        context = context or {}
        cue_vec = self._store._embed(cue)  # reuse the store's embedder
        pool = self._store.items()

        # Seed activation from direct cue similarity, biased by emotion/recency/context.
        activation: dict[str, float] = {}
        base: dict[str, float] = {}
        for item in pool:
            sim = self._similarity(cue_vec, item.embedding)
            base[item.id] = sim
            mood_match = 1.0 - abs(mood - item.emotion) / 2.0
            act = (sim
                   + self._emotion_w * mood_match * abs(item.emotion)
                   + self._recency_w * min(1.0, item.strength)
                   + self._context_bonus(item, context))
            activation[item.id] = act

        # Spread activation to associatively similar memories over a few hops.
        for _ in range(self._hops):
            spread: dict[str, float] = dict(activation)
            top = sorted(activation, key=activation.get, reverse=True)[:k]
            for src_id in top:
                src = self._store.by_id(src_id)
                if src is None:
                    continue
                for item in pool:
                    if item.id == src_id:
                        continue
                    edge = self._similarity(src.embedding, item.embedding)
                    spread[item.id] = spread.get(item.id, 0.0) + \
                        self._decay * activation[src_id] * max(0.0, edge - 0.3)
            activation = spread

        ranked = sorted(pool, key=lambda it: activation.get(it.id, 0.0), reverse=True)
        results = [RetrievalResult(item=it, activation=activation[it.id],
                                   base_similarity=base[it.id]) for it in ranked[:k]]
        for r in results:
            r.item.touch()
        return results

    def deja_vu(self, cue: str) -> DejaVuSignal:
        """Compute a déjà-vu signal for a cue: familiar gist, no specific episode."""
        semantic = self._store.search(cue, k=3, modality=Modality.SEMANTIC)
        episodic = self._store.search(cue, k=3, modality=Modality.EPISODIC)
        familiarity = max((s for s, _ in semantic), default=0.0)
        specificity = max((s for s, _ in episodic), default=0.0)
        return DejaVuSignal(
            familiarity=familiarity,
            specificity=specificity,
            is_deja_vu=familiarity > 0.5 and specificity < 0.35,
        )


__all__ = ["AssociativeRetriever", "RetrievalResult", "DejaVuSignal"]
