"""Tests for nyxara.mind.uncertainty."""

from __future__ import annotations

import random

import pytest

from nyxara.mind.uncertainty import (
    AbstentionPolicy,
    BetaBelief,
    Calibrator,
    DirichletBelief,
    UncertaintyModel,
    Verdict,
)


# -------------------- BetaBelief -------------------- #
def test_beta_prior_uniform():
    b = BetaBelief()
    assert b.mean == 0.5
    assert b.epistemic == 1.0  # no observations
    assert b.observations == 0.0


def test_beta_rejects_nonpositive():
    with pytest.raises(ValueError):
        BetaBelief(alpha=0, beta=1)


def test_beta_observe_shifts_mean():
    b = BetaBelief()
    for _ in range(9):
        b.observe(True)
    b.observe(False)
    assert b.mean > 0.7
    assert b.observations == 10.0


def test_beta_epistemic_shrinks_with_data():
    b = BetaBelief()
    e0 = b.epistemic
    for _ in range(100):
        b.observe(True)
    assert b.epistemic < e0
    assert b.epistemic < 0.15


def test_beta_aleatoric_max_at_half():
    fair = BetaBelief(alpha=50, beta=50)
    skewed = BetaBelief(alpha=99, beta=1)
    assert fair.aleatoric > skewed.aleatoric
    assert fair.aleatoric == pytest.approx(1.0, abs=0.02)


def test_beta_confidence_is_max_side():
    b = BetaBelief(alpha=9, beta=1)
    assert b.confidence == pytest.approx(0.9)
    b2 = BetaBelief(alpha=1, beta=9)
    assert b2.confidence == pytest.approx(0.9)  # confident it's FALSE


def test_beta_mode():
    assert BetaBelief(alpha=3, beta=2).mode == pytest.approx(2 / 3)  # (a-1)/(a+b-2)
    assert BetaBelief(alpha=1, beta=1).mode is None  # interior mode undefined


def test_beta_credible_interval_contains_mean():
    b = BetaBelief(alpha=8, beta=2)
    lo, hi = b.credible_interval()
    assert lo <= b.mean <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_beta_update():
    b = BetaBelief()
    b.update(successes=5, failures=3)
    assert b.alpha == 6 and b.beta == 4


# -------------------- DirichletBelief -------------------- #
def test_dirichlet_uniform_prior():
    d = DirichletBelief(("a", "b", "c"))
    probs = d.probs
    assert all(abs(p - 1 / 3) < 1e-9 for p in probs.values())


def test_dirichlet_observe_and_most_likely():
    d = DirichletBelief(("benign", "hostile"))
    d.observe("hostile")
    d.observe("hostile")
    assert d.most_likely()[0] == "hostile"


def test_dirichlet_entropy():
    uniform = DirichletBelief(("a", "b"))
    assert uniform.entropy == pytest.approx(1.0)
    skewed = DirichletBelief(("a", "b"), alphas=[99, 1])
    assert skewed.entropy < 0.2


def test_dirichlet_update_counts():
    d = DirichletBelief(("x", "y"))
    d.update({"x": 5, "y": 1})
    assert d.most_likely()[0] == "x"


def test_dirichlet_length_mismatch():
    with pytest.raises(ValueError):
        DirichletBelief(("a", "b"), alphas=[1.0])


def test_dirichlet_epistemic_shrinks():
    d = DirichletBelief(("a", "b"))
    e0 = d.epistemic
    for _ in range(100):
        d.observe("a")
    assert d.epistemic < e0


# -------------------- Calibrator -------------------- #
def test_calibrator_perfect():
    cal = Calibrator()
    for _ in range(100):
        cal.add(1.0, True)
        cal.add(0.0, False)
    assert cal.ece() < 0.01
    assert abs(cal.overconfidence()) < 0.01


def test_calibrator_detects_overconfidence():
    cal = Calibrator()
    rng = random.Random(1)
    for _ in range(1000):
        cal.add(0.9, rng.random() < 0.6)
    assert cal.overconfidence() > 0.2
    assert cal.ece() > 0.2


def test_calibrator_recalibrate():
    cal = Calibrator()
    rng = random.Random(2)
    for _ in range(1000):
        cal.add(0.9, rng.random() < 0.6)
    assert abs(cal.recalibrate(0.9) - 0.6) < 0.1


def test_calibrator_recalibrate_empty_bin_passthrough():
    cal = Calibrator()
    assert cal.recalibrate(0.42) == 0.42


def test_calibrator_brier():
    cal = Calibrator()
    cal.add(1.0, True)   # perfect -> 0
    cal.add(0.0, True)   # worst -> 1
    assert cal.brier_score() == pytest.approx(0.5)


def test_calibrator_reliability_curve():
    cal = Calibrator()
    cal.add(0.95, True)
    cal.add(0.15, False)
    curve = cal.reliability_curve()
    assert len(curve) == 2


# -------------------- AbstentionPolicy -------------------- #
def test_policy_abstains_low_confidence():
    pol = AbstentionPolicy(min_confidence=0.7)
    d = pol.decide(0.55, epistemic=0.1)
    assert d.verdict is Verdict.ABSTAIN
    assert "confidence" in d.reason


def test_policy_abstains_high_epistemic():
    pol = AbstentionPolicy(min_confidence=0.6, max_epistemic=0.4)
    d = pol.decide(0.9, epistemic=0.7)
    assert d.verdict is Verdict.ABSTAIN
    assert "epistemic" in d.reason


def test_policy_answers_when_confident_and_informed():
    pol = AbstentionPolicy(min_confidence=0.6, max_epistemic=0.5)
    d = pol.decide(0.85, epistemic=0.2)
    assert d.verdict is Verdict.ANSWER
    assert d.should_answer


def test_policy_uses_calibrated_confidence():
    pol = AbstentionPolicy(min_confidence=0.7)
    # raw 0.9 but calibrated to 0.5 -> abstain
    d = pol.decide(0.9, epistemic=0.1, calibrated=0.5)
    assert d.should_abstain


# -------------------- UncertaintyModel facade -------------------- #
def test_model_observe_and_prob():
    m = UncertaintyModel()
    for _ in range(8):
        m.observe("k", True)
    m.observe("k", False)
    assert m.prob("k") > 0.7


def test_model_abstains_with_little_evidence():
    m = UncertaintyModel(policy=AbstentionPolicy(min_confidence=0.6, max_epistemic=0.4))
    m.observe("rare", True)
    d = m.assess("rare")
    assert d.should_abstain


def test_model_answers_with_strong_evidence():
    m = UncertaintyModel(policy=AbstentionPolicy(min_confidence=0.6, max_epistemic=0.4))
    for _ in range(50):
        m.observe("certain", True)
    assert m.assess("certain").should_answer


def test_model_abstains_on_genuine_5050():
    m = UncertaintyModel(policy=AbstentionPolicy(min_confidence=0.7, max_epistemic=0.9))
    for _ in range(50):
        m.observe("coin", True)
        m.observe("coin", False)
    d = m.assess("coin")
    assert d.should_abstain
    assert "confidence" in d.reason


def test_model_knows_it_doesnt_know():
    m = UncertaintyModel(policy=AbstentionPolicy(max_epistemic=0.4))
    m.observe("mystery", True)
    assert m.knows_it_doesnt_know("mystery")
    for _ in range(50):
        m.observe("mystery", True)
    assert not m.knows_it_doesnt_know("mystery")


def test_model_categorical():
    m = UncertaintyModel()
    d = m.categorical("threat", ["low", "high"])
    d.observe("high")
    assert d.most_likely()[0] == "high"


def test_model_report_structure():
    m = UncertaintyModel()
    for _ in range(10):
        m.observe("x", True)
    rep = m.report("x")
    assert "belief" in rep and "decision" in rep and "calibration" in rep


def test_model_calibrated_assessment():
    m = UncertaintyModel(policy=AbstentionPolicy(min_confidence=0.7))
    # teach the calibrator that 0.9-confidence claims are only 50% right
    rng = random.Random(3)
    for _ in range(500):
        m.record_outcome(0.9, rng.random() < 0.5)
    d = m.assess_confidence(0.9, epistemic=0.1)
    # calibrated down to ~0.5 -> abstain despite raw 0.9
    assert d.should_abstain
