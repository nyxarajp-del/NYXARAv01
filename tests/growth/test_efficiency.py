"""Tests for nyxara.growth.efficiency — compute efficiency edge (Pillar F · Edge 3)."""

from __future__ import annotations

from nyxara.growth.efficiency import (
    ComputeLedger,
    EfficiencyFrontier,
    EfficiencyPoint,
    estimate_cost,
)


# --------------------------------------------------------------------------- #
# cost + point
# --------------------------------------------------------------------------- #
def test_cost_grows_with_size_and_latency():
    assert estimate_cost(1000) == 1000.0
    assert estimate_cost(1000, latency_s=1.0) == 2000.0      # 1s -> ×2
    assert estimate_cost(0) == 1.0                            # floored at 1

def test_point_efficiency_rewards_small_capable_models():
    small = EfficiencyPoint("s", capability=0.7, params=1_000_000)
    big = EfficiencyPoint("b", capability=0.72, params=100_000_000)
    assert small.efficiency > big.efficiency                  # tiny gain, huge cost -> less efficient

def test_dominance():
    a = EfficiencyPoint("a", capability=0.8, params=1_000)
    b = EfficiencyPoint("b", capability=0.7, params=2_000)    # worse and costlier
    assert a.dominates(b) and not b.dominates(a)


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #
def _ledger() -> ComputeLedger:
    led = ComputeLedger()
    led.record("nano", capability=0.62, params=500_000_000, latency_s=0.05)
    led.record("small", capability=0.78, params=3_000_000_000, latency_s=0.30)
    led.record("big", capability=0.80, params=7_000_000_000, latency_s=0.90)
    return led

def test_best_capability_is_the_biggest():
    assert _ledger().best_capability().label == "big"

def test_recommend_compresses_to_cheaper_within_epsilon():
    # within 0.05 of best (0.80), the cheapest qualifying model is 'small' (0.78)
    rec = _ledger().recommend(epsilon=0.05)
    assert rec["choice"]["label"] == "small"
    assert rec["cost_saved_frac"] > 0.0
    assert rec["capability_sacrificed"] >= 0.0

def test_recommend_tight_epsilon_keeps_best():
    rec = _ledger().recommend(epsilon=0.0)
    assert rec["choice"]["label"] == "big"                   # nothing else is within 0 of best

def test_recommend_empty_ledger_is_none():
    assert ComputeLedger().recommend() is None

def test_pareto_excludes_dominated():
    led = ComputeLedger()
    led.record("good", capability=0.8, params=1_000)
    led.record("dominated", capability=0.7, params=2_000)    # worse + costlier than 'good'
    labels = [p.label for p in led.pareto()]
    assert "good" in labels and "dominated" not in labels


# --------------------------------------------------------------------------- #
# from foundry-style versions (duck-typed)
# --------------------------------------------------------------------------- #
def test_from_versions_reads_metrics_and_params():
    class _V:
        def __init__(self, v, cap, params):
            self.version, self.metrics, self.param_count = v, {"capability": cap}, params
    led = ComputeLedger.from_versions([_V(1, 0.5, 1000), _V(2, 0.6, 2000)])
    assert len(led) == 2
    assert led.best_capability().label == "v2"

def test_from_versions_accepts_dicts():
    led = ComputeLedger.from_versions([
        {"version": 1, "metrics": {"capability": 0.4}, "param_count": 500}])
    assert len(led) == 1 and led.points()[0].capability == 0.4


# --------------------------------------------------------------------------- #
# promotion rule
# --------------------------------------------------------------------------- #
def test_prefer_cheaper_promotes_smaller_at_equal_capability():
    active = EfficiencyPoint("active", capability=0.80, params=7_000_000_000)
    cand = EfficiencyPoint("cand", capability=0.79, params=3_000_000_000)
    res = EfficiencyFrontier.prefer_cheaper(active, cand, epsilon=0.02)
    assert res["promote"] is True and "cheaper" in res["reason"]

def test_prefer_cheaper_promotes_more_capable():
    active = EfficiencyPoint("active", capability=0.70, params=1_000)
    cand = EfficiencyPoint("cand", capability=0.85, params=2_000)     # costlier but much better
    assert EfficiencyFrontier.prefer_cheaper(active, cand)["promote"] is True

def test_prefer_cheaper_keeps_active_when_worse_and_costlier():
    active = EfficiencyPoint("active", capability=0.80, params=1_000)
    cand = EfficiencyPoint("cand", capability=0.60, params=5_000)
    assert EfficiencyFrontier.prefer_cheaper(active, cand)["promote"] is False


# --------------------------------------------------------------------------- #
# frontier driver
# --------------------------------------------------------------------------- #
def test_frontier_report_includes_compute_and_recommendation():
    fr = EfficiencyFrontier(ledger=_ledger())
    rep = fr.report()
    assert "compute" in rep and "recommendation" in rep
    assert rep["compute"]["recommended_device"] in ("cuda", "mps", "cpu")
