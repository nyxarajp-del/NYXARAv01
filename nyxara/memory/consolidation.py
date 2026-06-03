"""Consolidation — nightly compress + dream replay + forgetting curve + reconsolidation.

Modelled on sleep. During a consolidation pass Nyxara:
  * **replays** salient/emotional episodes ("dreaming"), strengthening the ones worth
    keeping and weaving them into schemas;
  * **compresses** redundant episodes into induced schemas (gist over verbatim);
  * applies the **Ebbinghaus forgetting curve** so unrehearsed, low-value memories fade;
  * **reconsolidates** — when a memory is reactivated it becomes briefly labile and is
    re-written with updated context/strength, which is also how false memories form, so
    reconsolidation is provenance-aware and conservative.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from nyxara.memory.schema import Schema, SchemaInducer
from nyxara.memory.store import MemoryItem, MemoryStore, Modality


@dataclass
class ConsolidationReport:
    replayed: int = 0
    schematized: int = 0
    forgotten: int = 0
    reconsolidated: int = 0
    schemas: int = 0


def retention(strength: float, elapsed_s: float, stability: float = 86_400.0) -> float:
    """Ebbinghaus retention R = exp(-t/S), scaled by memory strength (rehearsal)."""
    s = max(stability * max(strength, 0.05), 1.0)
    return math.exp(-elapsed_s / s)


class Consolidator:
    """Runs the nightly maintenance cycle over a :class:`MemoryStore`."""

    def __init__(self, store: MemoryStore, *, inducer: SchemaInducer | None = None,
                 replay_top: int = 10, forget_threshold: float = 0.1) -> None:
        self._store = store
        self._inducer = inducer or SchemaInducer(store)
        self._replay_top = replay_top
        self._forget_threshold = forget_threshold

    def _dream_replay(self) -> int:
        """Replay the most salient (emotional × strong) episodes, strengthening them."""
        episodes = self._store.items(Modality.EPISODIC)
        ranked = sorted(
            episodes,
            key=lambda it: abs(it.emotion) * it.strength + it.access_count * 0.05,
            reverse=True,
        )
        for it in ranked[: self._replay_top]:
            it.touch()                  # reactivation
            it.strength = min(2.0, it.strength + 0.15)
        return min(self._replay_top, len(ranked))

    def _apply_forgetting(self, now: float) -> int:
        """Decay strength by the forgetting curve, then drop what fell through."""
        for it in self._store.items():
            if it.modality is Modality.SEMANTIC:
                continue  # facts are protected from passive decay
            elapsed = now - it.last_access
            it.strength = it.strength * retention(it.strength, elapsed)
        return self._store.forget_weak(self._forget_threshold)

    def reconsolidate(self, item_id: str, *, context: dict[str, Any] | None = None,
                      emotion: float | None = None) -> bool:
        """Reactivate a memory and rewrite it (labile window). Provenance-aware.

        Owner-sourced semantic facts are never silently rewritten — only their context is
        augmented — to avoid manufacturing false memories.
        """
        it = self._store.by_id(item_id)
        if it is None:
            return False
        it.touch()
        if context:
            it.context.update(context)
        if emotion is not None:
            protected = it.provenance is not None and it.provenance.source == "owner"
            if not protected:
                it.emotion = 0.5 * it.emotion + 0.5 * emotion
        return True

    def run(self) -> ConsolidationReport:
        now = time.time()
        report = ConsolidationReport()
        report.replayed = self._dream_replay()
        schemas: list[Schema] = self._inducer.induce()
        report.schemas = len(schemas)
        report.schematized = sum(s.support for s in schemas)
        report.forgotten = self._apply_forgetting(now)
        return report


__all__ = ["Consolidator", "ConsolidationReport", "retention"]
