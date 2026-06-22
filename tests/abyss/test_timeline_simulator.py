"""Tests for nyxara.abyss.timeline_simulator."""

from __future__ import annotations

import random

import pytest

from nyxara.abyss.timeline_simulator import (
    ActionOutcome,
    TimelineReport,
    TimelineSimulator,
)
from nyxara.mind.world_model import WorldModel


def _true_step(x, a):
    return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]


def _trained(seed=0, n=400):
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        nx = _true_step(x, a)
        wm.observe((x,), a, (nx,), reward=-abs(nx))
    return wm


def _home_reward(s, a, ns):
    return -abs(ns[0])


# -------------------- core ranking -------------------- #
def test_best_action_moves_toward_goal():
    """From x=+5, 'left' (toward home at 0) must outrank 'right'."""
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    report = sim.simulate((5.0,), ["left", "right", "stay"],
                          branches=900, horizon=6, reward_fn=_home_reward, noise_scale=0.05)
    assert isinstance(report, TimelineReport)
    assert report.best_action == "left"
    ranks = {o.action: o for o in report.action_rankings}
    assert ranks["left"].score > ranks["right"].score


def test_branches_split_across_actions():
    sim = TimelineSimulator(world_model=_trained(), seed=1)
    report = sim.simulate((3.0,), ["left", "right"], branches=600, horizon=4,
                          reward_fn=_home_reward)
    assert report.total_branches == 600
    for o in report.action_rankings:
        assert isinstance(o, ActionOutcome)
        assert o.branches == 300


def test_goal_probability_reported():
    sim = TimelineSimulator(world_model=_trained(), seed=2)
    goal = lambda st: abs(st[0]) <= 1.0  # noqa: E731
    report = sim.simulate((4.0,), ["left", "right", "stay"], branches=900, horizon=10,
                          reward_fn=_home_reward, goal_fn=goal, noise_scale=0.02)
    assert report.best_action == "left"
    assert report.goal_probability is not None
    # honest, relative claim: 'left' (toward home) reaches the goal more often than 'right'
    gp = {o.action: o.goal_probability for o in report.action_rankings}
    assert gp["left"] > gp["right"] and gp["left"] > 0.0


def test_goal_probability_none_without_goal_fn():
    sim = TimelineSimulator(world_model=_trained(), seed=3)
    report = sim.simulate((2.0,), ["left", "right"], branches=400, horizon=4,
                          reward_fn=_home_reward)
    assert report.goal_probability is None


# -------------------- honest uncertainty -------------------- #
def test_untrained_model_low_confidence():
    sim = TimelineSimulator(world_model=WorldModel(), seed=0)
    report = sim.simulate((0.0,), ["left", "right"], branches=200, horizon=4)
    assert report.confidence < 0.3


def test_trained_model_higher_confidence_than_untrained():
    trained = TimelineSimulator(world_model=_trained(), seed=0)
    blank = TimelineSimulator(world_model=WorldModel(), seed=0)
    r_t = trained.simulate((5.0,), ["left", "right"], branches=400, horizon=5,
                           reward_fn=_home_reward)
    r_b = blank.simulate((5.0,), ["left", "right"], branches=400, horizon=5,
                         reward_fn=_home_reward)
    assert r_t.confidence > r_b.confidence


# -------------------- edges & reproducibility -------------------- #
def test_no_candidate_actions():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    report = sim.simulate((1.0,), [], branches=100, horizon=4)
    assert report.best_action is None
    assert report.action_rankings == []
    assert report.confidence == 0.0


def test_recommend_matches_best_action():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    actions = ["left", "right", "stay"]
    best = sim.recommend((6.0,), actions, branches=600, horizon=6, reward_fn=_home_reward)
    report = sim.simulate((6.0,), actions, branches=600, horizon=6, reward_fn=_home_reward)
    assert best == report.best_action == "left"


def test_reproducible_for_fixed_seed():
    a = TimelineSimulator(world_model=_trained(), seed=7)
    b = TimelineSimulator(world_model=_trained(), seed=7)
    ra = a.simulate((5.0,), ["left", "right", "stay"], branches=600, horizon=6,
                    reward_fn=_home_reward, noise_scale=0.1)
    rb = b.simulate((5.0,), ["left", "right", "stay"], branches=600, horizon=6,
                    reward_fn=_home_reward, noise_scale=0.1)
    assert ra.to_dict() == rb.to_dict()


def test_to_dict_roundtrip_shapes():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    d = sim.simulate((5.0,), ["left", "right"], branches=400, horizon=5,
                     reward_fn=_home_reward).to_dict()
    assert set(d) >= {"start_state", "total_branches", "horizon", "best_action",
                      "confidence", "expected_outcome", "action_rankings"}
    assert isinstance(d["action_rankings"], list) and d["action_rankings"]
    assert "ci95" in d["action_rankings"][0]


def test_default_world_model_built_when_none():
    sim = TimelineSimulator(seed=0)
    assert sim.world_model is not None
    # a fresh model has no experience → honest low confidence
    report = sim.simulate((0.0,), ["left", "right"], branches=100, horizon=3)
    assert report.confidence < 0.3
