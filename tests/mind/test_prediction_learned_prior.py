"""Prediction confidence is learned, not hardcoded (capabilities 7 & 8).

`PredictionEngine._calibrate` used a flat 0.5 plus hardcoded keyword priors (`"2+2"→0.97`). It
now anchors on a `LearnedPrior` — a Beta-Bernoulli base rate over query features that NYXARA
updates from real outcomes and that survives a restart. These tests prove it starts honest,
learns from experience, and round-trips through serialisation.
"""

from __future__ import annotations

from nyxara.mind.prediction_engine import LearnedPrior, PredictionEngine


def test_cold_start_is_honest_half():
    pe = PredictionEngine()
    r = pe.predict("will 7 + 5 = 12 hold?")
    assert r.probability == 0.5
    assert any("honest 0.5" in step for step in r.reasoning_chain)


def test_base_rate_learns_up_from_positive_outcomes():
    prior = LearnedPrior()
    base0, why0 = prior.base_rate("does 3 + 4 = 7 hold?")
    assert base0 == 0.5 and why0 is None
    for _ in range(30):
        prior.observe("does 3 + 4 = 7 hold?", correct=True)
    base1, why1 = prior.base_rate("will 9 + 2 = 11 hold?")
    assert base1 > base0            # arithmetic-style predictions now trusted more
    assert why1 and "arithmetic" in why1


def test_base_rate_learns_down_from_negative_outcomes():
    prior = LearnedPrior()
    for _ in range(30):
        prior.observe("this is impossible and never works", correct=False)
    base, _ = prior.base_rate("that plan is impossible, it never works")
    assert base < 0.5


def test_no_hardcoded_tautology_prior():
    # the old string-matched "2+2 → 0.97" must be gone: with no learning it is an honest 0.5.
    pe = PredictionEngine()
    assert pe.predict("2+2").probability == 0.5


def test_prior_round_trips_through_serialisation():
    prior = LearnedPrior()
    for _ in range(10):
        prior.observe("delete the production database", correct=False)
    restored = LearnedPrior()
    restored.load_dict(prior.to_dict())
    assert restored.base_rate("delete everything now") == prior.base_rate("delete everything now")
    assert "risky" in restored._arms
