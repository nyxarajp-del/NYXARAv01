"""NYXARA · mind/metacognition.py — the "do I actually know this?" gate (🪞, Phase 4).

A mind that always answers is not intelligent — it is merely fluent. This gate gives NYXARA
the metacognition to choose, before she speaks, *which* mind should answer and whether she
should answer at all: a verifiable faculty if one fits, her own model if she is calibrated-
confident, the teacher if she is not, and an honest **"I don't know"** when there is nothing
trustworthy to say and no one to consult. Bluffing is a failure mode here, not a default.

It composes signals NYXARA already has — a faculty's applicability, the own model's verifier
confidence (optionally recalibrated by :mod:`observe.honesty`'s calibrator), and, when given,
her self-belief store — into one auditable verdict. It decides; the router and reasoner act.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

__all__ = ["MetaDecision", "MetaVerdict", "MetaCognition", "HONEST_ABSTENTION"]

HONEST_ABSTENTION = "I don't know — I can't answer that reliably, Master."


class MetaDecision(str, Enum):
    USE_FACULTY = "use_faculty"        # a verifiable engine fits — compute/prove it
    ANSWER_SELF = "answer_self"        # her own model is calibrated-confident enough
    CONSULT_TEACHER = "consult_teacher"  # not confident — ask the external teacher
    ABSTAIN = "abstain"                # nothing trustworthy and no one to consult — say so


@dataclass
class MetaVerdict:
    decision: MetaDecision
    confidence: float
    reason: str

    def to_dict(self) -> dict:
        return {"decision": self.decision.value, "confidence": round(self.confidence, 4),
                "reason": self.reason}


class MetaCognition:
    """Decide who should answer — faculty / own model / teacher / no one (abstain)."""

    def __init__(self, *, answer_threshold: float = 0.6, abstain_below: float = 0.15,
                 calibrator: Any = None) -> None:
        self.answer_threshold = answer_threshold
        self.abstain_below = abstain_below
        self.calibrator = calibrator

    def _calibrated(self, confidence: float) -> float:
        """Recalibrate raw confidence against measured accuracy when a calibrator is given."""
        if self.calibrator is None:
            return confidence
        for attr in ("calibrate", "adjust", "apply"):
            fn = getattr(self.calibrator, attr, None)
            if callable(fn):
                try:
                    return max(0.0, min(1.0, float(fn(confidence))))
                except Exception:  # noqa: BLE001 — calibration is advisory, never fatal
                    return confidence
        return confidence

    def assess(self, prompt: str, *, own_answer: Optional[str] = None, own_conf: float = 0.0,
               faculty_available: bool = False, teacher_available: bool = False) -> MetaVerdict:
        """Choose the answering path for ``prompt`` given what each mind offers."""
        if faculty_available:
            return MetaVerdict(MetaDecision.USE_FACULTY, 1.0,
                               "a verifiable faculty fits — compute it exactly")

        conf = self._calibrated(own_conf)
        has_own = bool(own_answer and own_answer.strip())

        if has_own and conf >= self.answer_threshold:
            return MetaVerdict(MetaDecision.ANSWER_SELF, conf,
                               "own model is calibrated-confident enough to answer")
        if teacher_available:
            return MetaVerdict(MetaDecision.CONSULT_TEACHER, conf,
                               "not confident enough — consult the teacher")
        # no teacher to fall back on: speak a passable own answer, else admit the gap honestly
        if has_own and conf > self.abstain_below:
            return MetaVerdict(MetaDecision.ANSWER_SELF, conf,
                               "no teacher available — best-effort own answer")
        return MetaVerdict(MetaDecision.ABSTAIN, conf,
                           "nothing trustworthy to say and no one to consult — abstain")


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA metacognition self-test")
    print("=" * 70)

    mc = MetaCognition(answer_threshold=0.6, abstain_below=0.15)

    assert mc.assess("2+2?", faculty_available=True).decision is MetaDecision.USE_FACULTY
    assert mc.assess("hi", own_answer="Hello, Master.", own_conf=0.8).decision \
        is MetaDecision.ANSWER_SELF
    assert mc.assess("hard", own_answer="guess", own_conf=0.3,
                     teacher_available=True).decision is MetaDecision.CONSULT_TEACHER
    assert mc.assess("hard", own_answer="", own_conf=0.0).decision is MetaDecision.ABSTAIN
    assert mc.assess("hard", own_answer="weak", own_conf=0.05).decision is MetaDecision.ABSTAIN
    print("faculty / self / teacher / abstain : all chosen correctly ✓")
    print("\nALL SELF-TESTS PASSED ✓")
