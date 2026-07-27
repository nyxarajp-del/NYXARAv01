"""NYXARA · nyx5/intent.py — pre-cognitive intent symbiosis (🧠, pillar 8).

Typing is slow; a supreme mind reads the *shape* of what you are about to ask from the pattern of what
you have asked before. NYX-5 keeps a lightweight transition model over the Master's intents and, when
enabled, **speculatively pre-computes** the answer to the single most-likely next intent — so if that
intent arrives, the answer is already warm.

Honest framing: this is *prediction + speculative execution*, not telepathy. Latency drops when the
guess is right; a wrong guess is simply discarded (its cached work thrown away). The speculative cache
is bounded, and a cached answer — like every answer — still passes the kernel's gate before it is used.

Pure standard library. Depends on nothing else in NYX-5.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["IntentPrediction", "IntentSymbiote"]


@dataclass
class IntentPrediction:
    """A ranked guess at the next intent, with a normalised confidence."""

    label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


class IntentSymbiote:
    """A first-order transition model over intents, with a bounded speculative-answer cache."""

    def __init__(self, *, cache_size: int = 8, speculate: bool = True) -> None:
        self.cache_size = int(cache_size)
        self.speculate = bool(speculate)
        self._trans: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last: Optional[str] = None
        self._cache: Dict[str, Any] = {}
        self._order: List[str] = []

    def observe(self, intent: str) -> None:
        """Record that ``intent`` just happened (updates the transition counts)."""
        if self._last is not None:
            self._trans[self._last][intent] += 1
        self._last = intent

    def predict(self, top: int = 3) -> List[IntentPrediction]:
        """Rank the most likely next intents given the last observed one."""
        if self._last is None or self._last not in self._trans:
            return []
        counts = self._trans[self._last]
        total = sum(counts.values()) or 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [IntentPrediction(label=k, confidence=v / total) for k, v in ranked[:top]]

    def prefetch(self, solver: Callable[[str], Any]) -> Optional[str]:
        """Speculatively pre-compute the answer to the single most-likely next intent, and cache it."""
        if not self.speculate or self.cache_size <= 0:
            return None
        preds = self.predict(top=1)
        if not preds:
            return None
        label = preds[0].label
        if label not in self._cache:
            try:
                self._cache[label] = solver(label)
            except Exception:  # noqa: BLE001 — speculative work is best-effort; a failure is harmless
                return None
            self._order.append(label)
            while len(self._order) > self.cache_size:
                evicted = self._order.pop(0)
                self._cache.pop(evicted, None)
        return label

    def consume(self, intent: str) -> Tuple[bool, Any]:
        """If ``intent`` was speculatively pre-computed, return (True, cached result); else (False, None)."""
        if intent in self._cache:
            result = self._cache.pop(intent)
            if intent in self._order:
                self._order.remove(intent)
            return True, result
        return False, None

    def to_dict(self) -> Dict[str, Any]:
        return {"cache_size": self.cache_size, "speculate": self.speculate, "last": self._last,
                "trans": {k: dict(v) for k, v in self._trans.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.cache_size = int(d.get("cache_size", self.cache_size))
        self.speculate = bool(d.get("speculate", self.speculate))
        self._last = d.get("last")
        self._trans = defaultdict(lambda: defaultdict(int))
        for k, v in d.get("trans", {}).items():
            for k2, c in v.items():
                self._trans[k][k2] = int(c)
