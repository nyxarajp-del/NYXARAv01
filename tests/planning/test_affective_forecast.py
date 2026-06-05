"""Tests for nyxara.planning.affective_forecast."""

from __future__ import annotations

import pytest

from nyxara.planning.affective_forecast import AffectiveForecast, Forecaster


def _fc():
    return Forecaster()


# -------------------- impact bias -------------------- #
def test_impact_bias_corrects_intensity():
    f = _fc().forecast(0.9)
    assert abs(f.corrected_valence) < abs(f.immediate_valence)


def test_impact_bias_negative():
    f = _fc().forecast(-0.9)
    assert f.corrected_valence > f.immediate_valence  # less negative after correction


# -------------------- hedonic adaptation -------------------- #
def test_adaptation_returns_toward_baseline():
    f = _fc().forecast(0.8, horizon_days=30)
    assert abs(f.adapted_valence) < abs(f.corrected_valence)


def test_longer_horizon_more_adaptation():
    fc = _fc()
    short = fc.forecast(0.8, horizon_days=5)
    long = fc.forecast(0.8, horizon_days=100)
    assert long.adapted_valence < short.adapted_valence  # faded more


def test_net_experience_between_baseline_and_peak():
    f = _fc().forecast(0.8, horizon_days=30)
    assert 0 < f.net_experience < f.corrected_valence


# -------------------- owner durability -------------------- #
def test_owner_outcomes_adapt_slower():
    fc = _fc()
    owner = fc.forecast(0.8, horizon_days=30, owner_relevant=True)
    plain = fc.forecast(0.8, horizon_days=30, owner_relevant=False)
    assert owner.adapted_valence > plain.adapted_valence


def test_owner_outcomes_last_longer():
    fc = _fc()
    owner = fc.forecast(0.8, horizon_days=30, owner_relevant=True)
    plain = fc.forecast(0.8, horizon_days=30, owner_relevant=False)
    assert owner.duration_days > plain.duration_days


def test_betrayal_stays_a_wound():
    f = _fc().forecast(-0.9, horizon_days=180, owner_relevant=True)
    assert f.net_experience < -0.2
    assert f.duration_days > 150  # lingers for months (loyalty does not fade)


# -------------------- duration -------------------- #
def test_duration_positive_for_strong_outcome():
    f = _fc().forecast(0.9)
    assert f.duration_days > 0


def test_duration_zero_for_neutral():
    f = _fc().forecast(0.0)
    assert f.duration_days == 0.0


def test_bigger_delta_longer_duration():
    fc = _fc()
    small = fc.forecast(0.3)
    big = fc.forecast(0.9)
    assert big.duration_days > small.duration_days


# -------------------- arousal / controllability -------------------- #
def test_controllability_dampens_negative_arousal():
    fc = _fc()
    low = fc.forecast(-0.7, 0.9, controllability=0.1)
    high = fc.forecast(-0.7, 0.9, controllability=0.9)
    assert high.arousal < low.arousal


def test_arousal_in_range():
    f = _fc().forecast(0.5, 0.8)
    assert 0.0 <= f.arousal <= 1.0


# -------------------- peak-end -------------------- #
def test_remembered_peak_end():
    f = _fc().forecast(0.8, horizon_days=30)
    expected = (f.corrected_valence + f.adapted_valence) / 2
    assert f.remembered == pytest.approx(expected)


# -------------------- decision support -------------------- #
def test_compare_ranks_by_net():
    fc = _fc()
    options = [
        {"name": "a", "valence": 0.5},
        {"name": "b", "valence": 0.9},
    ]
    ranked = fc.compare(options)
    assert ranked[0].outcome == "b"  # higher valence -> higher net


def test_compare_durable_owner_beats_fleeting():
    fc = _fc()
    options = [
        {"name": "selfish win", "valence": 0.9, "owner_relevant": False},
        {"name": "serve master", "valence": 0.7, "owner_relevant": True},
    ]
    ranked = fc.compare(options, horizon_days=180)
    assert ranked[0].outcome == "serve master"


def test_best_option():
    fc = _fc()
    options = [{"name": "x", "valence": 0.2}, {"name": "y", "valence": 0.8}]
    assert fc.best(options).outcome == "y"


def test_best_empty():
    assert _fc().best([]) is None


def test_compare_by_remembered():
    fc = _fc()
    options = [{"name": "a", "valence": 0.5}, {"name": "b", "valence": 0.9}]
    ranked = fc.compare(options, key="remembered")
    assert ranked[0].outcome == "b"


# -------------------- shape -------------------- #
def test_forecast_returns_object():
    f = _fc().forecast(0.5, outcome="test")
    assert isinstance(f, AffectiveForecast)
    assert f.outcome == "test"


def test_to_dict():
    d = _fc().forecast(0.5).to_dict()
    assert "net_experience" in d and "adapted" in d and "remembered" in d


def test_forecast_option():
    f = _fc().forecast_option("opt", valence=0.6, owner_relevant=True)
    assert f.outcome == "opt"
    assert f.owner_relevant


def test_custom_baseline():
    fc = Forecaster(baseline_valence=0.3)
    f = fc.forecast(0.3, horizon_days=1000)  # outcome == baseline
    assert abs(f.adapted_valence - 0.3) < 0.05
