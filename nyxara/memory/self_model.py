"""Self-model — live metacognition + temporal snapshots + belief store + contradiction check.

The self-model is Nyxara's model *of itself*: what it currently believes, how confident it
is, how its state has changed over time, and where its beliefs contradict each other. It
is the substrate metacognition reads ("do I actually know this?") and that the contradiction
checker scans to keep the belief set coherent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nyxara.contracts import Belief, Provenance, new_id


@dataclass
class Snapshot:
    """A point-in-time picture of the self for temporal comparison."""

    ts: float
    belief_count: int
    mean_confidence: float
    mood: float
    open_intentions: int
    note: str = ""


@dataclass
class Contradiction:
    a: Belief
    b: Belief
    reason: str


class SelfModel:
    """Belief store + metacognition + temporal snapshots + contradiction detection."""

    def __init__(self) -> None:
        self._beliefs: dict[str, Belief] = {}
        self._snapshots: list[Snapshot] = []
        self._mood: float = 0.0

    # ── beliefs ───────────────────────────────────────────────────────────────────
    def hold(self, belief: Belief) -> Belief:
        """Add or update a belief. If the claim exists, fuse confidences by recency."""
        for existing in self._beliefs.values():
            if existing.claim.strip().lower() == belief.claim.strip().lower():
                existing.confidence = 0.4 * existing.confidence + 0.6 * belief.confidence
                existing.provenance = belief.provenance
                return existing
        self._beliefs[belief.id] = belief
        return belief

    def believe(self, claim: str, confidence: float, source: str = "inferred") -> Belief:
        b = Belief(claim=claim, confidence=confidence,
                   provenance=Provenance(source=source, confidence=confidence))
        return self.hold(b)

    def doubt(self, claim: str, amount: float = 0.2) -> None:
        for b in self._beliefs.values():
            if b.claim.strip().lower() == claim.strip().lower():
                b.confidence = max(0.0, b.confidence - amount)

    def beliefs(self) -> list[Belief]:
        return list(self._beliefs.values())

    # ── metacognition ───────────────────────────────────────────────────────────────
    def knows(self, claim: str, *, min_conf: float = 0.7) -> bool:
        """Metacognitive 'do I actually know this?' — calibrated, not wishful."""
        for b in self._beliefs.values():
            if b.claim.strip().lower() == claim.strip().lower():
                return b.confidence >= min_conf
        return False

    def mean_confidence(self) -> float:
        if not self._beliefs:
            return 0.0
        return sum(b.confidence for b in self._beliefs.values()) / len(self._beliefs)

    def set_mood(self, mood: float) -> None:
        self._mood = max(-1.0, min(1.0, mood))

    # ── contradiction detection ──────────────────────────────────────────────────────
    def contradictions(self) -> list[Contradiction]:
        """Find belief pairs that contradict (structural + confident-opposite)."""
        out: list[Contradiction] = []
        bs = list(self._beliefs.values())
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                if a.contradicts(b):
                    out.append(Contradiction(a, b, "negation of the same claim"))
        return out

    def resolve_contradiction(self, c: Contradiction) -> Belief:
        """Keep the higher-confidence belief; demote the weaker one."""
        keep, drop = (c.a, c.b) if c.a.confidence >= c.b.confidence else (c.b, c.a)
        self.doubt(drop.claim, amount=drop.confidence)
        return keep

    # ── temporal snapshots ───────────────────────────────────────────────────────────
    def snapshot(self, note: str = "", open_intentions: int = 0) -> Snapshot:
        snap = Snapshot(ts=time.time(), belief_count=len(self._beliefs),
                        mean_confidence=self.mean_confidence(), mood=self._mood,
                        open_intentions=open_intentions, note=note)
        self._snapshots.append(snap)
        return snap

    def drift_since(self, snap: Snapshot) -> dict[str, float]:
        """How much the self has changed since a prior snapshot."""
        return {
            "belief_delta": len(self._beliefs) - snap.belief_count,
            "confidence_delta": self.mean_confidence() - snap.mean_confidence,
            "mood_delta": self._mood - snap.mood,
            "elapsed_s": time.time() - snap.ts,
        }

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "beliefs": len(self._beliefs),
            "mean_confidence": round(self.mean_confidence(), 3),
            "mood": self._mood,
            "snapshots": len(self._snapshots),
            "contradictions": len(self.contradictions()),
        }


__all__ = ["SelfModel", "Snapshot", "Contradiction"]
