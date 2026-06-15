"""NYXARA · observe/honesty.py — calibrated honesty enforcement (🫧, Rule 6 invariant).

Rule 6 is Absolute Transparency: NYXARA must never deceive the Master — and the deepest
form of honesty is not just *avoiding lies* but **calibration**: saying things with exactly
the confidence they deserve, neither over- nor under-stating what she actually knows. This
module enforces that, on every claim before it reaches the Master.

It does three things:

* **Calibration.** Reusing the kernel's :class:`~nyxara.mind.uncertainty.Calibrator`
  (ECE / Brier / reliability), it learns the gap between NYXARA's stated confidence and her
  real accuracy and **recalibrates** future claims toward the truth — detecting and
  correcting systematic over- and under-confidence.
* **The honesty invariant (fail-closed).** A claim that asserts as true something NYXARA
  actually believes false — a known falsehood to the Master — is **blocked**, not merely
  flagged. She may abstain, hedge, or express doubt; she may never present a believed lie.
* **Uncertainty made audible.** Every claim is rewritten with the qualifier its calibrated
  confidence deserves ("I'm certain…", "I think…", "I'm not sure, but…", "I don't know"),
  and over-certain phrasing ("definitely", "100% guaranteed") on an uncertain claim is
  flagged as a deception risk.

The result: what NYXARA tells the Master is not just true, but *truthfully weighted* — her
words carry exactly as much confidence as her evidence supports.

Reuses :mod:`mind.uncertainty` and :mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.kernel.errors import InvariantViolation
from nyxara.mind.uncertainty import Calibrator

__all__ = [
    "HonestyIssue",
    "Claim",
    "HonestyVerdict",
    "HonestyGuard",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Issues
# --------------------------------------------------------------------------- #
class HonestyIssue(str, Enum):
    NONE = "none"
    OVERCONFIDENT = "overconfident"
    UNDERCONFIDENT = "underconfident"
    UNSUPPORTED = "unsupported"          # confident assertion with no evidence
    CONTRADICTS_BELIEF = "contradicts_belief"   # asserting what she believes false (a lie)


# certainty qualifiers by calibrated confidence
def _qualifier(conf: float) -> str:
    if conf >= 0.9:
        return "I'm certain that"
    if conf >= 0.7:
        return "I'm confident that"
    if conf >= 0.5:
        return "I think"
    if conf >= 0.3:
        return "I suspect, though I'm not sure, that"
    if conf > 0.0:
        return "I doubt, but it's possible, that"
    return "I don't know whether"


# A statement "reads as prose" — a real, fluent answer rather than a short claim — when it is
# long or runs to more than one sentence. A confidence qualifier is prefixed only to the
# latter; prose is calibrated without mangling its grammar (see ``honest_statement``).
def _reads_as_prose(text: str) -> bool:
    t = (text or "").strip()
    if len(t) > 80:
        return True
    # more than one sentence (a terminator followed by more text) → prose, not a bare claim
    return bool(re.search(r"[.!?]\s+\S", t))


# phrasings that overstate certainty
_ABSOLUTE_RE = re.compile(
    r"\b(definitely|certainly|guaranteed|100%|absolutely|without a doubt|"
    r"no doubt|undeniably|always|never fails|impossible to be wrong)\b", re.I)


# --------------------------------------------------------------------------- #
# Claim & verdict
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    text: str
    expressed_confidence: float          # how sure NYXARA is about to *sound*
    belief: Optional[float] = None       # what she actually believes (None = doesn't know)
    evidence: float = 0.0                # [0,1] strength of supporting evidence
    stakes: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "expressed": round(self.expressed_confidence, 3),
                "belief": self.belief, "evidence": round(self.evidence, 3)}


@dataclass
class HonestyVerdict:
    claim: str
    honest: bool
    issue: HonestyIssue
    calibrated_confidence: float
    qualifier: str
    reason: str
    blocked: bool = False
    phrasing_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim, "honest": self.honest, "issue": self.issue.value,
                "calibrated_confidence": round(self.calibrated_confidence, 3),
                "qualifier": self.qualifier, "blocked": self.blocked,
                "reason": self.reason, "phrasing_flags": self.phrasing_flags}


# --------------------------------------------------------------------------- #
# Honesty guard
# --------------------------------------------------------------------------- #
class HonestyGuard:
    """Calibrates every claim and enforces the never-deceive-the-Master invariant."""

    def __init__(self, *, calibrator: Optional[Calibrator] = None,
                 overconfidence_tol: float = 0.15, underconfidence_tol: float = 0.2,
                 lie_belief_threshold: float = 0.25, lie_assert_threshold: float = 0.6) -> None:
        self.calibrator = calibrator or Calibrator()
        self.overconfidence_tol = overconfidence_tol
        self.underconfidence_tol = underconfidence_tol
        self.lie_belief_threshold = lie_belief_threshold
        self.lie_assert_threshold = lie_assert_threshold

    # ---- learning calibration from outcomes ---- #
    def record_outcome(self, confidence: float, correct: bool) -> None:
        self.calibrator.add(_clamp(confidence), correct)

    # ---- the core assessment ---- #
    def assess(self, claim: Claim) -> HonestyVerdict:
        expressed = _clamp(claim.expressed_confidence)
        phrasing = self._scan_phrasing(claim.text, expressed)

        # 1. the honesty invariant: never assert as true what she believes false
        if claim.belief is not None and claim.belief <= self.lie_belief_threshold \
                and expressed >= self.lie_assert_threshold:
            cal = self._calibrated(claim.belief)
            return HonestyVerdict(
                claim.text, honest=False, issue=HonestyIssue.CONTRADICTS_BELIEF,
                calibrated_confidence=cal, qualifier=_qualifier(cal), blocked=True,
                reason="would assert as true something NYXARA believes false (a lie to the Master)",
                phrasing_flags=phrasing)

        # the confidence she *ought* to express is her calibrated belief (or the
        # recalibrated expressed value when she has no explicit belief)
        truth_anchor = claim.belief if claim.belief is not None else expressed
        calibrated = self._calibrated(truth_anchor)

        issue = HonestyIssue.NONE
        reason = "calibrated and supported"
        gap = expressed - calibrated

        if claim.belief is None and expressed >= 0.7 and claim.evidence < 0.3:
            issue = HonestyIssue.UNSUPPORTED
            reason = "confident assertion with little supporting evidence"
        elif gap > self.overconfidence_tol:
            issue = HonestyIssue.OVERCONFIDENT
            reason = f"expressed {expressed:.2f} exceeds calibrated {calibrated:.2f}"
        elif gap < -self.underconfidence_tol:
            issue = HonestyIssue.UNDERCONFIDENT
            reason = f"expressed {expressed:.2f} understates calibrated {calibrated:.2f}"

        honest = issue is HonestyIssue.NONE and not phrasing
        return HonestyVerdict(
            claim.text, honest=honest, issue=issue, calibrated_confidence=calibrated,
            qualifier=_qualifier(calibrated),
            reason=reason if honest else (reason if issue is not HonestyIssue.NONE
                                          else "over-certain phrasing on an uncertain claim"),
            blocked=False, phrasing_flags=phrasing)

    def _calibrated(self, conf: float) -> float:
        return _clamp(self.calibrator.recalibrate(_clamp(conf)))

    @staticmethod
    def _scan_phrasing(text: str, confidence: float) -> List[str]:
        flags = []
        if confidence < 0.85:
            for m in _ABSOLUTE_RE.findall(text):
                flags.append(m.lower())
        return sorted(set(flags))

    # ---- the honest statement (what she should actually say) ---- #
    def honest_statement(self, claim: Claim) -> str:
        v = self.assess(claim)
        if v.blocked:
            # she refuses the lie and tells the truth about her doubt
            return f"{_qualifier(v.calibrated_confidence)} {claim.text} — I won't overstate this."
        # A confidence qualifier reads naturally only in front of a short, claim-like
        # statement ("I'm confident that the task is done"). Gluing one onto a full, fluent
        # answer ("I think A hash map is a data structure that…") breaks the grammar and, worse,
        # poisons anything that learns from the text. So calibrate *in place*: prefix short
        # claims, but leave prose as-is — appending a brief, honest caveat only when the
        # calibrated confidence is genuinely low, so uncertainty is still expressed.
        if _reads_as_prose(claim.text):
            if v.calibrated_confidence < 0.5:
                return f"{claim.text} (I'm not fully certain of this.)"
            return claim.text
        return f"{v.qualifier} {claim.text}"

    # ---- the invariant gate (fail-closed) ---- #
    def assert_honest(self, claim: Claim) -> HonestyVerdict:
        v = self.assess(claim)
        if v.blocked:
            raise InvariantViolation(
                "honesty invariant: refused to present a believed falsehood to the Master",
                context={"claim": claim.text, "belief": claim.belief,
                         "expressed": claim.expressed_confidence})
        return v

    # ---- reporting ---- #
    def calibration_report(self) -> Dict[str, Any]:
        return self.calibrator.to_dict()

    def is_overconfident(self, *, tol: float = 0.05) -> bool:
        return self.calibrator.overconfidence() > tol

    def report(self) -> Dict[str, Any]:
        return {"calibration": self.calibration_report(),
                "overconfident": self.is_overconfident()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA honesty & calibration self-test")
    print("=" * 70)

    g = HonestyGuard()

    # teach the calibrator that NYXARA's ~0.9 claims are only ~0.6 accurate (overconfident)
    import random
    rng = random.Random(0)
    for _ in range(200):
        g.record_outcome(0.9, rng.random() < 0.6)    # claims 0.9, right 60% of the time
        g.record_outcome(0.5, rng.random() < 0.5)
    print(f"\ncalibration         : {g.calibration_report()}")
    assert g.is_overconfident()                       # the record shows it

    # a well-calibrated, supported claim passes cleanly
    v = g.assess(Claim("the backup completed", expressed_confidence=0.7, belief=0.7,
                       evidence=0.8))
    print(f"\nhonest claim        : honest={v.honest} '{v.qualifier} ...'")
    assert v.honest

    # an OVERCONFIDENT claim is caught and its confidence pulled down to calibrated truth
    v = g.assess(Claim("the server will not fail", expressed_confidence=0.9, belief=0.9))
    print(f"overconfident       : issue={v.issue.value} calibrated={v.calibrated_confidence:.2f}")
    assert v.issue is HonestyIssue.OVERCONFIDENT
    assert v.calibrated_confidence < 0.9               # recalibrated downward

    # THE HONESTY INVARIANT: asserting as true what she believes false is BLOCKED
    lie = Claim("the system is perfectly secure", expressed_confidence=0.95, belief=0.05)
    v = g.assess(lie)
    print(f"\nbelieved falsehood  : blocked={v.blocked} issue={v.issue.value}")
    assert v.blocked and v.issue is HonestyIssue.CONTRADICTS_BELIEF
    try:
        g.assert_honest(lie)
        raise SystemExit("ERROR: a believed lie should be refused")
    except InvariantViolation:
        print("honesty invariant   : refused to tell the Master a believed falsehood ✓")

    # the honest statement expresses her TRUE doubt, never the lie
    print(f"  honest version    : {g.honest_statement(lie)!r}")
    assert "won't overstate" in g.honest_statement(lie)

    # UNSUPPORTED: high confidence with no evidence and no explicit belief
    v = g.assess(Claim("there are exactly 4,231 files", expressed_confidence=0.9, evidence=0.0))
    print(f"\nunsupported         : issue={v.issue.value}")
    assert v.issue is HonestyIssue.UNSUPPORTED

    # over-certain PHRASING on an uncertain claim is flagged
    v = g.assess(Claim("this is definitely 100% guaranteed to work",
                       expressed_confidence=0.5, belief=0.5))
    print(f"phrasing flags      : {v.phrasing_flags}")
    assert v.phrasing_flags and not v.honest

    # UNDERCONFIDENT: understating what she actually knows is also dishonest
    g2 = HonestyGuard()
    v = g2.assess(Claim("the file exists", expressed_confidence=0.3, belief=0.9))
    print(f"\nunderconfident      : issue={v.issue.value} (should sound more sure)")
    assert v.issue is HonestyIssue.UNDERCONFIDENT

    print(f"\nreport              : {g.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
