"""Schema induction — many episodes → reusable mental templates.

Experience repeats. After enough similar episodes ("owner asked for a summary, I produced
one, they were satisfied"), Nyxara should abstract a *schema*: a template with slots and a
typical script, so it can recognise and act in similar situations without reasoning from
scratch. This is the bridge from episodic specifics to semantic generality.

Clustering uses the same embedding space as the store; a schema is the centroid of a
cluster plus the slots/steps that recur across its members.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from nyxara.memory.store import MemoryItem, MemoryStore, Modality


@dataclass
class Schema:
    """An abstracted template induced from a cluster of episodes."""

    id: str
    label: str
    centroid: np.ndarray
    slots: dict[str, list[str]]          # recurring context keys → observed values
    script: list[str]                    # typical ordered steps / gist
    support: int                         # how many episodes back this schema
    confidence: float

    def matches(self, embedding: np.ndarray, threshold: float = 0.6) -> float:
        denom = (np.linalg.norm(self.centroid) * np.linalg.norm(embedding)) or 1.0
        sim = float(np.dot(self.centroid, embedding) / denom)
        return sim if sim >= threshold else 0.0


class SchemaInducer:
    """Greedy single-pass clustering + template extraction over episodic memory."""

    def __init__(self, store: MemoryStore, *, sim_threshold: float = 0.55,
                 min_support: int = 3) -> None:
        self._store = store
        self._threshold = sim_threshold
        self._min_support = min_support
        self._schemas: list[Schema] = []

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def induce(self) -> list[Schema]:
        """(Re)build schemas from the current episodic memory."""
        episodes = self._store.items(Modality.EPISODIC)
        clusters: list[list[MemoryItem]] = []
        centroids: list[np.ndarray] = []

        for ep in episodes:
            placed = False
            for idx, cen in enumerate(centroids):
                if self._cosine(cen, ep.embedding) >= self._threshold:
                    clusters[idx].append(ep)
                    members = clusters[idx]
                    centroids[idx] = np.mean([m.embedding for m in members], axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append([ep])
                centroids.append(ep.embedding.copy())

        self._schemas = []
        for idx, members in enumerate(clusters):
            if len(members) < self._min_support:
                continue
            self._schemas.append(self._build_schema(idx, members, centroids[idx]))
        return self._schemas

    def _build_schema(self, idx: int, members: list[MemoryItem],
                      centroid: np.ndarray) -> Schema:
        # Slots: context keys whose values recur across members.
        slot_values: dict[str, list[str]] = {}
        for m in members:
            for k, v in m.context.items():
                slot_values.setdefault(k, []).append(str(v))
        # Script: most common leading tokens across member texts (a crude gist).
        token_counts: Counter[str] = Counter()
        for m in members:
            token_counts.update(m.text.lower().split()[:6])
        script = [tok for tok, _ in token_counts.most_common(6)]
        label = " ".join(script[:3]) or f"schema-{idx}"
        return Schema(
            id=f"schema_{idx}", label=label, centroid=centroid,
            slots={k: sorted(set(v)) for k, v in slot_values.items()},
            script=script, support=len(members),
            confidence=min(1.0, len(members) / (self._min_support * 3)),
        )

    def recognise(self, text: str) -> tuple[Schema | None, float]:
        """Return the best-matching schema for new input, if any."""
        emb = self._store._embed(text)
        best: tuple[Schema | None, float] = (None, 0.0)
        for sch in self._schemas:
            score = sch.matches(emb)
            if score > best[1]:
                best = (sch, score)
        return best

    @property
    def schemas(self) -> list[Schema]:
        return list(self._schemas)


__all__ = ["Schema", "SchemaInducer"]
