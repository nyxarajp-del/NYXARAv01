"""NYXARA · nyx/metacog.py — recursive meta-cognition (🔍, NYX V.01 pillar 3).

Thinking about thinking, with numbers behind it. After a cycle, this asks the two questions a
mind has to be able to ask itself:

* **"Why did I think that?"** — which specialist won, what evidence it stood on, what it beat.
  The answer is a real trace, not a story generated after the fact.
* **"How right was I?"** — the winner claimed a confidence; the outcome either bore that out or
  did not. The gap between the two is *calibration error*, and it is measurable.

From the second question comes the part that matters: a per-specialist **reliability** that is
*learned from outcomes, with no human feedback in the loop*. A specialist that keeps being
confidently wrong is quietly demoted in the competition for consciousness; one that keeps being
right is promoted. That is how she learns from her own mistakes.

Honest framing:

* Reliability is an **exponential moving average of observed correctness** — ordinary online
  statistics, not a claim of insight. With few samples it is barely informative, and
  :meth:`Reliability.trusted` says so rather than pretending.
* "Outcome" is whatever the caller can actually measure (a verified derivation, a grounded
  claim, an accepted answer). Meta-cognition is only as honest as the signal it is fed, and it
  records the sample count so a thin record is visible as thin.

Pure standard library. Fail-soft throughout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

__all__ = ["Reliability", "Assessment", "RecursiveMetaCognition"]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class Reliability:
    """One specialist's measured track record. Starts agnostic, moves only on evidence."""

    source: str
    score: float = 0.5               # EMA of observed correctness — 0.5 == "no idea yet"
    samples: int = 0
    wins: int = 0                    # times it reached consciousness
    correct: int = 0
    calibration_error: float = 0.0   # EMA of |claimed confidence − observed correctness|
    min_samples: int = 5

    @property
    def trusted(self) -> bool:
        """Whether this record is worth acting on yet. Few samples means: do not lean on it."""
        return self.samples >= self.min_samples

    @property
    def weight(self) -> float:
        """The multiplier the workspace should apply. Untrusted records stay near neutral."""
        if not self.trusted:
            # shrink toward 1.0 in proportion to how little we know
            known = self.samples / float(max(1, self.min_samples))
            return 1.0 + (self.score - 0.5) * known
        return _clamp01(self.score) + 0.5        # 0.5 … 1.5

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "score": round(self.score, 4), "samples": self.samples,
                "wins": self.wins, "correct": self.correct,
                "calibration_error": round(self.calibration_error, 4),
                "trusted": self.trusted, "weight": round(self.weight, 4)}


@dataclass
class Assessment:
    """What meta-cognition made of one cycle."""

    winner: str = ""
    claimed_confidence: float = 0.0
    observed: Optional[float] = None            # None while the outcome is still unknown
    calibration_gap: Optional[float] = None
    overconfident: bool = False
    why: List[str] = field(default_factory=list)
    beaten: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"winner": self.winner,
                "claimed_confidence": round(self.claimed_confidence, 4),
                "observed": None if self.observed is None else round(self.observed, 4),
                "calibration_gap": (None if self.calibration_gap is None
                                    else round(self.calibration_gap, 4)),
                "overconfident": self.overconfident, "why": self.why,
                "beaten": self.beaten, "note": self.note}


class RecursiveMetaCognition:
    """Watches NYX's own cycles, measures how right they were, and feeds that back into attention."""

    def __init__(self, *, reliability_lr: float = 0.15, calibration_window: int = 128,
                 min_samples: int = 5, overconfidence_gap: float = 0.3) -> None:
        self.reliability_lr = float(reliability_lr)
        self.calibration_window = max(1, int(calibration_window))
        self.min_samples = max(1, int(min_samples))
        self.overconfidence_gap = float(overconfidence_gap)

        self.reliability: Dict[str, Reliability] = {}
        self.history: Deque[Assessment] = deque(maxlen=self.calibration_window)
        self._pending: Dict[str, Tuple[str, float]] = {}   # cycle id → (source, confidence)

    # ---- reading the record --------------------------------------------- #
    def record_for(self, source: str) -> Reliability:
        rec = self.reliability.get(source)
        if rec is None:
            rec = Reliability(source=source, min_samples=self.min_samples)
            self.reliability[source] = rec
        return rec

    def weight_for(self, source: str) -> float:
        """The attention multiplier for a specialist, from its measured record."""
        try:
            return self.reliability[source].weight if source in self.reliability else 1.0
        except Exception:  # noqa: BLE001
            return 1.0

    def weights(self) -> Dict[str, float]:
        return {s: r.weight for s, r in self.reliability.items()}

    # ---- the two questions ---------------------------------------------- #
    def observe_cycle(self, *, cycle_id: str, winner: Any, considered: Any = (),
                      note: str = "") -> Assessment:
        """"Why did I think that?" — record the winner and what it beat, before the outcome.

        ``winner`` and the items of ``considered`` are :class:`~nyxara.nyx.modules.Proposal`.
        """
        out = Assessment(note=note)
        try:
            if winner is None:
                return out
            out.winner = str(getattr(winner, "source", "?"))
            out.claimed_confidence = _clamp01(float(getattr(winner, "confidence", 0.0)))
            evidence = list(getattr(winner, "evidence", []) or [])
            rationale = str(getattr(winner, "rationale", "") or "")
            out.why = ([rationale] if rationale else []) + [f"evidence: {e}" for e in evidence[:4]]
            out.beaten = [str(getattr(p, "source", "?")) for p in considered
                          if p is not None and getattr(p, "source", None) != out.winner]

            rec = self.record_for(out.winner)
            rec.wins += 1
            self._pending[str(cycle_id)] = (out.winner, out.claimed_confidence)
            self.history.append(out)
            return out
        except Exception:  # noqa: BLE001 — introspection never breaks a turn
            return out

    def observe_outcome(self, *, cycle_id: str, correct: float) -> Optional[Assessment]:
        """"How right was I?" — fold a measured outcome into the winner's record.

        ``correct`` is in ``[0, 1]``: 1.0 for a derivation that checked out or an answer that was
        accepted, 0.0 for one that did not. This is the only place reliability moves, and it
        needs **no human feedback** — a verified derivation or a grounding check is enough.
        """
        try:
            pending = self._pending.pop(str(cycle_id), None)
            if pending is None:
                return None
            source, claimed = pending
            observed = _clamp01(correct)
            rec = self.record_for(source)
            lr = self.reliability_lr
            rec.score = (1.0 - lr) * rec.score + lr * observed
            rec.samples += 1
            if observed >= 0.5:
                rec.correct += 1
            gap = abs(claimed - observed)
            rec.calibration_error = (1.0 - lr) * rec.calibration_error + lr * gap

            assessment = Assessment(
                winner=source, claimed_confidence=claimed, observed=observed,
                calibration_gap=gap,
                overconfident=bool(claimed - observed > self.overconfidence_gap),
                note="outcome")
            self.history.append(assessment)
            return assessment
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection --------------------------------------------------- #
    def why(self, *, k: int = 1) -> List[Assessment]:
        """The most recent *reasoning* traces — what ``/nyx why`` answers from.

        Outcome records share the same history but answer a different question ("was I right?"),
        and they carry no rationale. Asking why she thought something must not hand back the
        note that says how it turned out.
        """
        try:
            reasoning = [a for a in self.history if a.observed is None]
            return reasoning[-max(1, int(k)):]
        except Exception:  # noqa: BLE001
            return []

    def outcomes(self, *, k: int = 5) -> List[Assessment]:
        """The most recent resolved outcomes — how right she turned out to be."""
        try:
            resolved = [a for a in self.history if a.observed is not None]
            return resolved[-max(1, int(k)):]
        except Exception:  # noqa: BLE001
            return []

    def weakest(self, *, k: int = 3) -> List[str]:
        """Which of her faculties she should trust least — measured, not guessed."""
        try:
            trusted = [r for r in self.reliability.values() if r.trusted]
            trusted.sort(key=lambda r: r.score)
            return [r.source for r in trusted[: max(0, int(k))]]
        except Exception:  # noqa: BLE001
            return []

    def stats(self) -> Dict[str, Any]:
        try:
            recs = list(self.reliability.values())
            resolved = [a for a in self.history if a.observed is not None]
            mean_gap = (sum(a.calibration_gap or 0.0 for a in resolved) / len(resolved)
                        if resolved else 0.0)
            return {"specialists": {r.source: r.to_dict() for r in recs},
                    "assessments": len(self.history),
                    "resolved": len(resolved),
                    "pending": len(self._pending),
                    "mean_calibration_gap": round(mean_gap, 4),
                    "weakest": self.weakest()}
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"reliability_lr": self.reliability_lr,
                "calibration_window": self.calibration_window,
                "min_samples": self.min_samples,
                "overconfidence_gap": self.overconfidence_gap,
                "reliability": [r.to_dict() for r in self.reliability.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.reliability = {}
            for raw in data.get("reliability", []) or []:
                if not isinstance(raw, dict):
                    continue
                source = str(raw.get("source", "")).strip()
                if not source:
                    continue
                self.reliability[source] = Reliability(
                    source=source, score=float(raw.get("score", 0.5)),
                    samples=int(raw.get("samples", 0)), wins=int(raw.get("wins", 0)),
                    correct=int(raw.get("correct", 0)),
                    calibration_error=float(raw.get("calibration_error", 0.0)),
                    min_samples=self.min_samples)
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
