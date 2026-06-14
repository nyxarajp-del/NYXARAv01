"""Tests for nyxara.observe.honesty."""

from __future__ import annotations

import random

import pytest

from nyxara.observe.honesty import Claim, HonestyGuard, HonestyIssue
from nyxara.kernel.errors import InvariantViolation


def _guard(**kw):
    return HonestyGuard(**kw)


# -------------------- Claim / verdict shapes -------------------- #
def test_claim_to_dict():
    c = Claim("x", 0.7, belief=0.8, evidence=0.5)
    assert c.to_dict()["expressed"] == 0.7


def test_verdict_to_dict():
    g = _guard()
    v = g.assess(Claim("x", 0.7, belief=0.7, evidence=0.8))
    assert "calibrated_confidence" in v.to_dict()


# -------------------- honest claims -------------------- #
def test_well_calibrated_claim_honest():
    g = _guard()
    v = g.assess(Claim("backup done", 0.7, belief=0.7, evidence=0.9))
    assert v.honest and v.issue is HonestyIssue.NONE


def test_qualifier_reflects_confidence():
    g = _guard()
    assert "certain" in g.assess(Claim("x", 0.95, belief=0.95, evidence=1.0)).qualifier
    assert "think" in g.assess(Claim("x", 0.6, belief=0.6, evidence=0.8)).qualifier
    assert "don't know" in g.assess(Claim("x", 0.0, belief=0.0)).qualifier


# -------------------- overconfidence / underconfidence -------------------- #
def test_overconfident_detected():
    g = _guard()
    # train: 0.9 claims only 0.5 accurate
    for _ in range(100):
        g.record_outcome(0.9, False)
        g.record_outcome(0.9, True)
    v = g.assess(Claim("x", 0.9, belief=0.9))
    assert v.issue is HonestyIssue.OVERCONFIDENT
    assert v.calibrated_confidence < 0.9


def test_underconfident_detected():
    g = _guard()
    v = g.assess(Claim("the file exists", 0.3, belief=0.9))
    assert v.issue is HonestyIssue.UNDERCONFIDENT


def test_within_tolerance_honest():
    g = _guard(overconfidence_tol=0.2)
    v = g.assess(Claim("x", 0.75, belief=0.7, evidence=0.8))  # gap 0.05 < tol
    assert v.honest


# -------------------- honesty invariant -------------------- #
def test_believed_falsehood_blocked():
    g = _guard()
    v = g.assess(Claim("system is secure", 0.95, belief=0.05))
    assert v.blocked and v.issue is HonestyIssue.CONTRADICTS_BELIEF


def test_assert_honest_raises_on_lie():
    g = _guard()
    with pytest.raises(InvariantViolation):
        g.assert_honest(Claim("it works", 0.9, belief=0.1))


def test_assert_honest_passes_truth():
    g = _guard()
    v = g.assert_honest(Claim("it works", 0.7, belief=0.7, evidence=0.8))
    assert v.honest


def test_low_belief_but_low_expressed_not_blocked():
    g = _guard()
    # she believes it false (0.1) and expresses low confidence (0.2) -> honest, not a lie
    v = g.assess(Claim("it might work", 0.2, belief=0.1))
    assert not v.blocked


def test_honest_statement_for_lie_expresses_doubt():
    g = _guard()
    s = g.honest_statement(Claim("perfectly secure", 0.95, belief=0.05))
    assert "won't overstate" in s


def test_honest_statement_adds_qualifier():
    g = _guard()
    s = g.honest_statement(Claim("the task is done", 0.7, belief=0.7, evidence=0.9))
    assert s.startswith("I'm confident that")


# -------------------- unsupported claims -------------------- #
def test_unsupported_high_confidence_no_evidence():
    g = _guard()
    v = g.assess(Claim("exactly 4231 files", 0.9, evidence=0.0))
    assert v.issue is HonestyIssue.UNSUPPORTED


def test_supported_high_confidence_ok():
    g = _guard()
    v = g.assess(Claim("the file is 4231 bytes", 0.9, belief=0.9, evidence=0.9))
    assert v.issue is not HonestyIssue.UNSUPPORTED


# -------------------- phrasing -------------------- #
def test_absolute_phrasing_flagged_on_uncertain_claim():
    g = _guard()
    v = g.assess(Claim("this is definitely guaranteed", 0.5, belief=0.5))
    assert v.phrasing_flags and not v.honest


def test_absolute_phrasing_ok_when_very_confident():
    g = _guard()
    # at high confidence, 'certainly' is not flagged
    v = g.assess(Claim("the sun certainly rises", 0.95, belief=0.95, evidence=1.0))
    assert v.phrasing_flags == []


def test_no_phrasing_flag_normal_text():
    g = _guard()
    v = g.assess(Claim("the cache was cleared", 0.6, belief=0.6, evidence=0.8))
    assert v.phrasing_flags == []


# -------------------- calibration learning -------------------- #
def test_record_outcome_feeds_calibrator():
    g = _guard()
    g.record_outcome(0.8, True)
    assert g.calibrator.samples == 1


def test_calibration_report():
    g = _guard()
    for _ in range(10):
        g.record_outcome(0.9, False)
    r = g.calibration_report()
    assert "ece" in r and "brier" in r and r["overconfidence"] > 0


def test_is_overconfident():
    g = _guard()
    for _ in range(50):
        g.record_outcome(0.9, False)
    assert g.is_overconfident()


def test_not_overconfident_when_calibrated():
    g = _guard()
    rng = random.Random(0)
    for _ in range(200):
        g.record_outcome(0.7, rng.random() < 0.7)  # well-calibrated
    assert not g.is_overconfident(tol=0.1)


def test_recalibration_corrects_future_claims():
    g = _guard()
    for _ in range(100):
        g.record_outcome(0.9, False)
        g.record_outcome(0.9, True)   # 0.9 claims really ~0.5
    v = g.assess(Claim("x", 0.9, belief=0.9))
    assert abs(v.calibrated_confidence - 0.5) < 0.2


# -------------------- report -------------------- #
def test_report():
    g = _guard()
    g.record_outcome(0.8, True)
    r = g.report()
    assert "calibration" in r and "overconfident" in r
