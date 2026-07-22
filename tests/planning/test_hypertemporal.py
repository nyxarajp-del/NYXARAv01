"""Tests for F16 — hyper-temporal counterfactual planning (planning/hypertemporal.py).

She must choose the present action whose branching futures are best (looking past a greedy trap), stay
within a node budget (falling back to sampling honestly), and report the rollout cap — no infinite-
timeline claim. No LLM anywhere.
"""

from __future__ import annotations

import random

from nyxara.planning.hypertemporal import BranchingPlanner


# delayed-reward MDP: 'greedy' pays +1 now then dead-ends; 'patient' pays 0 now then +10
def _model(s, a):
    return {("start", "greedy"): [(1.0, "g", 1.0)], ("start", "patient"): [(1.0, "p", 0.0)],
            ("g", "go"): [(1.0, "t", 0.0)], ("p", "go"): [(1.0, "big", 10.0)]}[(s, a)]


def _actions(s):
    return {"start": ["greedy", "patient"], "g": ["go"], "p": ["go"], "t": [], "big": []}[s]


def test_backward_induction_looks_past_the_greedy_trap():
    r = BranchingPlanner(gamma=0.95).plan("start", _actions, _model, horizon=3)
    assert r.best_action == "patient"                 # delayed +10 beats immediate +1
    assert r.method == "backward-induction"
    assert r.action_values["patient"] > r.action_values["greedy"]


def test_parallel_rollouts_agree_and_report_the_cap():
    mc = BranchingPlanner(gamma=0.95).parallel_rollouts(
        "start", _actions, _model, horizon=3, rollouts=300, rng=random.Random(0))
    assert mc.best_action == "patient"
    assert mc.method == "parallel-rollouts"
    assert mc.rollout_cap == 300                       # the honest bound is reported
    assert mc.rollouts == 600                          # 300 per root action × 2 actions


def test_terminal_state_yields_no_action():
    r = BranchingPlanner().plan("big", _actions, _model, horizon=3)
    assert r.best_action is None


def test_node_budget_falls_back_to_sampling_honestly():
    # a tiny node budget on a branching problem → she does NOT pretend to have searched it all
    r = BranchingPlanner(max_nodes=1).plan("start", _actions, _model, horizon=4)
    assert r.method == "parallel-rollouts"             # honest degradation
    assert r.best_action in ("greedy", "patient")


def test_stochastic_model_expected_value():
    # a coin flip: action 'bet' → +10 half the time, -2 the other half → EV ~ +4
    def model(s, a):
        return {("s", "bet"): [(0.5, "end", 10.0), (0.5, "end", -2.0)],
                ("s", "safe"): [(1.0, "end", 3.0)]}[(s, a)]

    def actions(s):
        return {"s": ["bet", "safe"], "end": []}[s]

    r = BranchingPlanner(gamma=1.0).plan("s", actions, model, horizon=1)
    assert abs(r.action_values["bet"] - 4.0) < 1e-9    # exact expectation over the branch
    assert r.best_action == "bet"
