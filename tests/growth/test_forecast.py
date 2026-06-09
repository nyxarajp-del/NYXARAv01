"""Tests for nyxara.growth.forecast."""

from __future__ import annotations

import math

from nyxara.growth.forecast import CapabilityForecast, CurveFit


def _record_curve(fc, cap, ceiling=0.9, rate=0.3, n=9, noise=0.0):
    import random
    rng = random.Random(0)
    for t in range(n):
        true = ceiling * (1 - math.exp(-rate * t))
        fc.record(cap, t, min(1.0, max(0.0, true + (rng.gauss(0, noise) if noise else 0))))


# -------------------- CurveFit math -------------------- #
def test_curve_predict():
    f = CurveFit("x", ceiling=1.0, rate=0.5, p0=0.0, current_t=0, current_p=0.5, r2=1.0, n=5)
    # approaches the ceiling
    assert f.predict(0) == 0.5
    assert f.predict(100) > 0.99


def test_curve_time_to_already_there():
    f = CurveFit("x", 1.0, 0.5, 0.0, 0, 0.8, 1.0, 5)
    assert f.time_to(0.7) == 0.0


def test_curve_time_to_reachable():
    f = CurveFit("x", 1.0, 0.5, 0.0, 0, 0.5, 1.0, 5)
    t = f.time_to(0.8)
    assert t is not None and t > 0


def test_curve_time_to_unreachable():
    f = CurveFit("x", ceiling=0.6, rate=0.5, p0=0.0, current_t=0, current_p=0.5, r2=1.0, n=5)
    assert f.time_to(0.8) is None  # ceiling below target


def test_curve_growth_rate_collapses_near_ceiling():
    near = CurveFit("x", 1.0, 0.5, 0.0, 0, 0.99, 1.0, 5)
    far = CurveFit("x", 1.0, 0.5, 0.0, 0, 0.2, 1.0, 5)
    assert near.growth_rate < far.growth_rate


def test_curve_plateaued():
    assert CurveFit("x", 1.0, 0.5, 0.0, 0, 0.999, 1.0, 5).plateaued
    assert not CurveFit("x", 1.0, 0.5, 0.0, 0, 0.2, 1.0, 5).plateaued


def test_curve_to_dict():
    d = CurveFit("x", 0.9, 0.3, 0.0, 0, 0.5, 0.99, 5).to_dict()
    assert d["capability"] == "x" and "growth_rate" in d


# -------------------- recording -------------------- #
def test_record_sorts():
    fc = CapabilityForecast()
    fc.record("s", 5, 0.5)
    fc.record("s", 1, 0.1)
    assert fc.observations("s")[0][0] == 1


def test_observations_empty():
    assert CapabilityForecast().observations("nope") == []


# -------------------- fitting -------------------- #
def test_fit_insufficient_data():
    fc = CapabilityForecast(min_points=3)
    fc.record("s", 0, 0.1)
    fc.record("s", 1, 0.2)
    assert fc.fit("s") is None


def test_fit_recovers_ceiling_and_rate():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.9, rate=0.3)
    fit = fc.fit("s")
    assert fit is not None
    assert abs(fit.ceiling - 0.9) < 0.12
    assert abs(fit.rate - 0.3) < 0.2
    assert fit.r2 > 0.9


def test_fit_high_r2_clean_data():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.8, rate=0.4)
    assert fc.fit("s").r2 > 0.98


def test_fit_rejects_non_learning():
    fc = CapabilityForecast()
    # decreasing data is not a learning curve
    for t in range(6):
        fc.record("s", t, 0.9 - 0.1 * t)
    assert fc.fit("s") is None


def test_fit_flat_data_returns_none():
    fc = CapabilityForecast()
    for t in range(6):
        fc.record("s", t, 0.5)
    # perfectly flat -> no decay of (C-p), no valid fit
    assert fc.fit("s") is None


# -------------------- forecast / time-to-mastery -------------------- #
def test_forecast_projects_forward():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.9, rate=0.3)
    cur = fc.fit("s").current_p
    assert fc.forecast("s", 5) > cur


def test_forecast_no_data():
    assert CapabilityForecast().forecast("nope", 5) is None


def test_time_to_mastery():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.95, rate=0.2, n=5)
    eta = fc.time_to_mastery("s", 0.8)
    assert eta is not None and eta > 0


def test_time_to_mastery_unreachable():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.5, rate=0.4)
    assert fc.time_to_mastery("s", 0.8) is None


def test_ceiling_estimate():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.7, rate=0.3)
    assert abs(fc.ceiling("s") - 0.7) < 0.15


def test_ceiling_no_data():
    assert CapabilityForecast().ceiling("nope") is None


def test_growth_rate():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.9, rate=0.3, n=4)
    assert fc.growth_rate("s") > 0


def test_is_plateaued_false_for_growing():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.9, rate=0.3, n=4)
    assert not fc.is_plateaued("s")


def test_is_plateaued_true_for_maxed():
    fc = CapabilityForecast()
    for t in range(13):
        fc.record("s", t, 0.96 * (1 - math.exp(-0.7 * t)))
    assert fc.is_plateaued("s")


# -------------------- bottlenecks -------------------- #
def test_bottleneck_ceiling_below_target():
    fc = CapabilityForecast()
    _record_curve(fc, "weak", ceiling=0.5, rate=0.4)
    bn = fc.bottlenecks({"weak": 0.8})
    assert bn and bn[0]["severity"] == 1.0
    assert "ceiling" in bn[0]["reason"]


def test_bottleneck_insufficient_data():
    fc = CapabilityForecast()
    fc.record("new", 0, 0.1)
    bn = fc.bottlenecks({"new": 0.8})
    assert bn and "insufficient" in bn[0]["reason"]


def test_bottleneck_already_met_excluded():
    fc = CapabilityForecast()
    _record_curve(fc, "good", ceiling=0.95, rate=0.5)
    bn = fc.bottlenecks({"good": 0.5})  # already well past 0.5
    assert all(b["capability"] != "good" for b in bn)


def test_bottleneck_ranking():
    fc = CapabilityForecast()
    _record_curve(fc, "weak", ceiling=0.4, rate=0.4)   # unreachable -> sev 1.0
    _record_curve(fc, "slow", ceiling=0.95, rate=0.05, n=6)  # slow but reachable
    bn = fc.bottlenecks({"weak": 0.8, "slow": 0.8})
    assert bn[0]["capability"] == "weak"  # worst first


# -------------------- summary -------------------- #
def test_summary_fitted():
    fc = CapabilityForecast()
    _record_curve(fc, "s", ceiling=0.9, rate=0.3)
    s = fc.summary("s")
    assert s["capability"] == "s" and "ceiling" in s


def test_summary_insufficient():
    fc = CapabilityForecast()
    fc.record("s", 0, 0.1)
    s = fc.summary("s")
    assert s["status"] == "insufficient data"
