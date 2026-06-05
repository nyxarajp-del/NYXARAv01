"""Tests for nyxara.sim.montecarlo."""

from __future__ import annotations

import math
import random

from nyxara.planning.planner import Action
from nyxara.sim.montecarlo import (
    Bernoulli,
    Categorical,
    Constant,
    Estimate,
    MonteCarlo,
    MonteCarloPlanner,
    Normal,
    PlanEstimate,
    Rollout,
    StochasticAction,
    Triangular,
    Uniform,
)


# -------------------- distributions -------------------- #
def test_constant_sample_and_mean():
    rng = random.Random(0)
    d = Constant(3.5)
    assert d.sample(rng) == 3.5 and d.mean() == 3.5


def test_bernoulli_frequency():
    rng = random.Random(1)
    d = Bernoulli(0.3)
    n = 5000
    hits = sum(d.sample(rng) for _ in range(n)) / n
    assert abs(hits - 0.3) < 0.03
    assert d.mean() == 0.3


def test_uniform_bounds_and_mean():
    rng = random.Random(2)
    d = Uniform(2.0, 4.0)
    assert all(2.0 <= d.sample(rng) <= 4.0 for _ in range(200))
    assert d.mean() == 3.0


def test_normal_floor_ceil():
    rng = random.Random(3)
    d = Normal(0.0, 5.0, floor=0.0, ceil=1.0)
    assert all(0.0 <= d.sample(rng) <= 1.0 for _ in range(500))
    assert d.mean() == 0.0


def test_triangular_mean():
    d = Triangular(0.0, 1.0, 2.0)
    assert d.mean() == 1.0


def test_categorical_mean_weighted():
    d = Categorical(outcomes=(0.0, 10.0), weights=(3.0, 1.0))
    assert d.mean() == 2.5


def test_distribution_callable():
    rng = random.Random(4)
    d = Constant(9.0)
    assert d(rng) == 9.0


# -------------------- Estimate -------------------- #
def test_estimate_basic_stats():
    e = Estimate([1.0, 2.0, 3.0, 4.0, 5.0])
    assert e.n == 5
    assert e.mean == 3.0
    assert e.median == 3.0
    assert e.minimum == 1.0 and e.maximum == 5.0
    assert e.variance == 2.5  # sample variance


def test_estimate_empty_is_safe():
    e = Estimate([])
    assert e.mean == 0.0 and e.std == 0.0 and e.percentile(50) == 0.0
    assert e.prob(lambda x: True) == 0.0


def test_estimate_percentiles():
    e = Estimate([float(i) for i in range(101)])  # 0..100
    assert abs(e.percentile(0) - 0.0) < 1e-9
    assert abs(e.percentile(100) - 100.0) < 1e-9
    assert abs(e.percentile(50) - 50.0) < 1e-9


def test_estimate_ci95_contains_mean():
    e = Estimate([float(i) for i in range(100)])
    lo, hi = e.ci95()
    assert lo < e.mean < hi


def test_estimate_sem_and_rel_sem():
    e = Estimate([10.0] * 100)
    assert e.sem == 0.0
    assert e.rel_sem == 0.0  # zero variance


def test_value_at_risk_and_cvar():
    # mostly small, rare big losses
    samples = [1.0] * 95 + [100.0] * 5
    e = Estimate(samples)
    assert e.value_at_risk(0.95) >= 1.0
    assert e.cvar(0.95) > e.mean  # tail worse than average
    assert e.cvar(0.95) <= 100.0


def test_estimate_prob_predicate():
    e = Estimate([1.0, 2.0, 3.0, 4.0])
    assert e.prob(lambda x: x > 2.0) == 0.5


def test_estimate_histogram_buckets():
    e = Estimate([float(i) for i in range(100)])
    h = e.histogram(bins=10)
    assert len(h) == 10
    assert sum(c for _, _, c in h) == 100


def test_estimate_to_dict():
    e = Estimate([1.0, 2.0, 3.0])
    d = e.to_dict()
    assert d["n"] == 3 and "ci95" in d and "p95" in d


# -------------------- MonteCarlo engine -------------------- #
def test_run_is_reproducible():
    mc = MonteCarlo(seed=11)
    a = mc.run(lambda r: r.random(), trials=1000)
    b = mc.run(lambda r: r.random(), trials=1000)
    assert a.samples == b.samples  # same seed -> identical


def test_run_estimates_pi():
    mc = MonteCarlo(seed=7)
    est = mc.run(lambda r: 4.0 if r.random() ** 2 + r.random() ** 2 <= 1.0 else 0.0,
                 trials=20000)
    assert abs(est.mean - math.pi) < 0.1


def test_run_zero_trials_safe():
    mc = MonteCarlo(seed=0)
    est = mc.run(lambda r: 1.0, trials=0)
    assert est.n == 0


def test_run_until_converges():
    mc = MonteCarlo(seed=5)
    est = mc.run_until(lambda r: r.gauss(10.0, 2.0), rel_tol=0.01, max_trials=50000)
    assert abs(est.mean - 10.0) < 0.3
    assert est.rel_sem <= 0.01 or est.n >= 50000


def test_run_until_respects_max_trials():
    mc = MonteCarlo(seed=5)
    # impossible tolerance -> must stop at max_trials
    est = mc.run_until(lambda r: r.gauss(0.0, 1.0), rel_tol=1e-9,
                       batch=500, max_trials=2000)
    assert est.n == 2000


def test_run_until_zero_variance_converges_fast():
    mc = MonteCarlo(seed=1)
    est = mc.run_until(lambda r: 5.0, rel_tol=0.01, min_trials=500, max_trials=10000)
    assert est.mean == 5.0 and est.n == 500


# -------------------- StochasticAction -------------------- #
def test_action_applicable():
    a = StochasticAction("x", preconditions=frozenset({"ready"}))
    assert a.applicable({"ready"})
    assert not a.applicable(set())


def test_action_success_applies_effects():
    rng = random.Random(0)
    a = StochasticAction("x", add_effects=frozenset({"done"}),
                         del_effects=frozenset({"todo"}), success_prob=1.0)
    ns, cost, ok = a.simulate({"todo"}, rng)
    assert ok and "done" in ns and "todo" not in ns


def test_action_failure_applies_no_effects():
    rng = random.Random(0)
    a = StochasticAction("x", add_effects=frozenset({"done"}), success_prob=0.0,
                         cost=Constant(2.0), failure_cost_fraction=0.5)
    ns, cost, ok = a.simulate(set(), rng)
    assert not ok and "done" not in ns
    assert cost == 1.0  # half of 2.0


def test_action_effect_probs():
    rng = random.Random(123)
    a = StochasticAction("x", add_effects=frozenset({"maybe"}), success_prob=1.0,
                         effect_probs={"maybe": 0.0})
    ns, _, ok = a.simulate(set(), rng)
    assert ok and "maybe" not in ns  # effect never realises


def test_from_action_wraps_planner_action():
    det = Action.make("open_door", pre=["have_key"], add=["door_open"],
                      delete=["door_closed"], cost=2.0)
    sto = StochasticAction.from_action(det, success_prob=0.8)
    assert sto.name == "open_door"
    assert sto.preconditions == frozenset({"have_key"})
    assert sto.add_effects == frozenset({"door_open"})
    assert sto.del_effects == frozenset({"door_closed"})
    assert sto.success_prob == 0.8
    assert sto.cost.mean() == 2.0


# -------------------- MonteCarloPlanner -------------------- #
def _flaky():
    return [StochasticAction("go", add_effects=frozenset({"done"}),
                             success_prob=0.6, cost=Constant(1.0))]


def _reliable():
    return [
        StochasticAction("prep", add_effects=frozenset({"ready"}),
                         success_prob=0.99, cost=Constant(1.0)),
        StochasticAction("exec", preconditions=frozenset({"ready"}),
                         add_effects=frozenset({"done"}),
                         success_prob=0.97, cost=Normal(2.0, 0.5, floor=0.0)),
    ]


def test_evaluate_success_prob_matches_config():
    planner = MonteCarloPlanner(seed=42)
    est = planner.evaluate(_flaky(), [], ["done"], trials=5000)
    assert isinstance(est, PlanEstimate)
    assert 0.55 < est.success_prob < 0.65  # ~0.6


def test_evaluate_reliable_plan_high_success():
    planner = MonteCarloPlanner(seed=42)
    est = planner.evaluate(_reliable(), [], ["done"], trials=4000)
    assert est.success_prob > 0.9
    assert est.expected_cost > 2.0  # prep + exec


def test_evaluate_records_abort_blocker():
    planner = MonteCarloPlanner(seed=1)
    # action whose precondition is never satisfied -> always blocks immediately
    plan = [StochasticAction("needs", preconditions=frozenset({"never"}),
                             add_effects=frozenset({"done"}))]
    est = planner.evaluate(plan, [], ["done"], trials=100)
    assert est.success_prob == 0.0
    assert est.worst_blocker == "needs"


def test_robustness_prefers_reliable():
    planner = MonteCarloPlanner(seed=42)
    a = planner.evaluate(_flaky(), [], ["done"], trials=4000, label="flaky")
    b = planner.evaluate(_reliable(), [], ["done"], trials=4000, label="reliable")
    assert b.robustness > a.robustness


def test_compare_ranks_robust_plan_first():
    planner = MonteCarloPlanner(seed=42)
    ranked = planner.compare({"flaky": _flaky(), "reliable": _reliable()},
                             [], ["done"], trials=4000, risk_aversion=0.6)
    assert [e.label for e in ranked][0] == "reliable"


def test_plan_estimate_to_dict():
    planner = MonteCarloPlanner(seed=42)
    est = planner.evaluate(_reliable(), [], ["done"], trials=1000)
    d = est.to_dict()
    assert "success_prob" in d and "tail_cost_cvar95" in d and "risk_score" in d


def test_tail_cost_at_least_median():
    planner = MonteCarloPlanner(seed=42)
    est = planner.evaluate(_reliable(), [], ["done"], trials=2000)
    assert est.tail_cost >= est.cost.median


def test_evaluate_reproducible():
    p = MonteCarloPlanner(seed=99)
    a = p.evaluate(_reliable(), [], ["done"], trials=1500)
    b = p.evaluate(_reliable(), [], ["done"], trials=1500)
    assert a.success_prob == b.success_prob
    assert a.cost.samples == b.cost.samples
