"""P0-A — live calibration learning is wired into the turn loop.

The HonestyGuard owns a real Calibrator (ECE/Brier → recalibrate), but before this wiring
``record_outcome`` was never called from ``process``, so calibration only ever learned inside
unit tests — never from NYXARA's lived turns. These tests prove the loop is now closed:

* a *grounded* action turn (a tool ran and we know if it succeeded) feeds the calibrator;
* a *conversational* turn carries no ground truth and is never scored (recording it would
  teach false confidence — the very over-confidence calibration exists to cure);
* fed enough grounded outcomes, the guard detects systematic over-confidence and recalibrates
  a high claim downward (capabilities #23 Calibration, #46 Verification, #70 Honest Failure).
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore
from nyxara.observe.honesty import Claim, HonestyIssue


def _calc_candidate(conf: float = 0.9) -> Candidate:
    return Candidate(text="compute 2+2", kind="act", tool="calculate",
                     tool_args={"expression": "2+2"}, confidence=conf,
                     capability=Capability.TOOL_CALL, risk=RiskTier.LOW, reversible=True)


def _force(nyx: NyxaraCore, candidate: Candidate) -> None:
    """Pin the reasoner so the turn deterministically proposes ``candidate``."""
    nyx._invoke_reasoner = lambda *a, **k: candidate          # type: ignore[assignment]
    nyx._maybe_bootstrap = lambda s, f, c, a: c               # type: ignore[assignment]


# -------------------- grounded action turns are scored -------------------- #
def test_grounded_action_turn_feeds_calibrator():
    nyx = NyxaraCore()
    assert nyx.honesty.calibration_report()["samples"] == 0
    _force(nyx, _calc_candidate())
    result = nyx.process("compute 2+2", authority=Authority.OWNER)
    assert result.disposition is Disposition.ACT and result.tool == "calculate"
    # exactly one ground-truthed sample was recorded for this real, successful action
    assert nyx.honesty.calibration_report()["samples"] == 1


# -------------------- conversational turns are NOT scored -------------------- #
def test_conversational_turn_is_not_scored():
    # a plain reply has no verifiable outcome; scoring it as "correct" would inflate confidence.
    nyx = NyxaraCore()
    nyx.process("hello NYXARA", authority=Authority.OWNER)
    nyx.process("how are you?", authority=Authority.OWNER)
    assert nyx.honesty.calibration_report()["samples"] == 0


def test_record_calibration_excludes_toolless_candidates():
    nyx = NyxaraCore()
    nyx._record_calibration(Candidate(text="just a reply", kind="respond", confidence=0.9),
                            correct=True)
    assert nyx.honesty.calibration_report()["samples"] == 0   # no tool → no ground truth
    nyx._record_calibration(_calc_candidate(), correct=True)
    assert nyx.honesty.calibration_report()["samples"] == 1   # a real action is scored


# -------------------- the loop actually self-corrects over-confidence -------------------- #
def test_calibration_self_corrects_overconfidence():
    nyx = NyxaraCore()
    over = _calc_candidate(conf=0.9)
    # she keeps acting at 0.9 confidence but the action only really succeeds half the time
    for i in range(120):
        nyx._record_calibration(over, correct=(i % 2 == 0))
    assert nyx.honesty.calibration_report()["samples"] == 120
    assert nyx.honesty.is_overconfident()                     # the record shows the gap
    # a fresh 0.9 claim is now pulled down toward her real ~0.5 accuracy
    verdict = nyx.honesty.assess(Claim("the server will not fail",
                                       expressed_confidence=0.9, belief=0.9))
    assert verdict.calibrated_confidence < 0.9
    assert verdict.issue is HonestyIssue.OVERCONFIDENT
