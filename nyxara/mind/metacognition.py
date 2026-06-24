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

    @staticmethod
    def introspect(*, self_consistency: Optional[float] = None,
                   epistemic: Optional[float] = None,
                   novelty: Optional[float] = None) -> float:
        """Combine INTERNAL signals NYXARA measures about herself into one uncertainty in [0, 1].

        This is the introspection the old gate lacked: rather than importing a single confidence
        from outside, the system reads signals about its *own* state and decides how unsure it is.

        * ``self_consistency`` — agreement across repeated samples of her own answer (1.0 = the
          model says the same thing every time → low uncertainty; 0.0 = it wanders → high).
        * ``epistemic`` — reducible "I haven't seen enough" uncertainty (e.g. a world-model
          ensemble's disagreement, or a Beta-belief's epistemic term). Higher → more unsure.
        * ``novelty`` — how unfamiliar the situation is (count-based). Higher → more unsure.

        Each is optional; only the signals actually available are blended (equal weight), so this
        degrades gracefully as faculties come and go. Returns 0.0 when nothing is known.
        """
        terms = []
        if self_consistency is not None:
            terms.append(1.0 - max(0.0, min(1.0, float(self_consistency))))
        if epistemic is not None:
            terms.append(max(0.0, min(1.0, float(epistemic))))
        if novelty is not None:
            terms.append(max(0.0, min(1.0, float(novelty))))
        if not terms:
            return 0.0
        return max(0.0, min(1.0, sum(terms) / len(terms)))

    def assess(self, prompt: str, *, own_answer: Optional[str] = None, own_conf: float = 0.0,
               faculty_available: bool = False, teacher_available: bool = False,
               internal_uncertainty: float = 0.0) -> MetaVerdict:
        """Choose the answering path for ``prompt`` given what each mind offers.

        ``internal_uncertainty`` (0..1, from :meth:`introspect`) is the system's *own* measure of
        how unsure it is this turn. It discounts the effective confidence, so a fluent-but-unstable
        answer (high internal uncertainty) is held to a higher bar and more readily deferred or
        abstained — metacognition now reads an internal signal, not only an external score.
        """
        if faculty_available:
            return MetaVerdict(MetaDecision.USE_FACULTY, 1.0,
                               "a verifiable faculty fits — compute it exactly")

        iu = max(0.0, min(1.0, float(internal_uncertainty)))
        conf = self._calibrated(own_conf) * (1.0 - iu)   # introspection discounts confidence
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
