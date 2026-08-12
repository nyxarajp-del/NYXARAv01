"""Tests for nyxara.abyss.timeline_simulator."""

from __future__ import annotations

import random
import time


from nyxara.abyss.timeline_simulator import (
    DEFAULT_NOISE_SCALE,
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


# -------------------- divergence is on by default -------------------- #
def test_divergence_is_injected_by_default():
    """A pass over one committed action must still produce a *distribution* of futures.

    With ``noise_scale=0`` the learned model is treated as exact, so every branch replays the
    same trajectory: p05 == p95, a zero-width interval, and a "tail risk" that is just the mean.
    """
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    off = sim.simulate((5.0,), ["left"], branches=200, horizon=5,
                       reward_fn=_home_reward, noise_scale=0.0).action_rankings[0]
    assert off.p95 == off.p05                      # one future, counted 200 times
    assert off.ci95[0] == off.ci95[1]

    on = sim.simulate((5.0,), ["left"], branches=200, horizon=5,
                      reward_fn=_home_reward).action_rankings[0]
    assert on.p95 > on.p05                         # a real spread, by default
    assert on.ci95[1] > on.ci95[0]


def test_default_noise_scale_is_on():
    assert DEFAULT_NOISE_SCALE > 0.0


# -------------------- the wall-clock budget -------------------- #
def test_budget_clips_the_branch_count():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    started = time.monotonic()
    report = sim.simulate((5.0,), ["left", "right", "stay"], branches=200_000, horizon=8,
                          reward_fn=_home_reward, budget_ms=250)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    assert report.total_branches < 200_000         # the clock, not the ask, decided
    assert elapsed_ms < 250 * 4                    # generous: it stopped rather than ran on
    assert report.total_branches == sum(o.branches for o in report.action_rankings)


def test_budget_is_split_evenly_so_the_ranking_stays_comparable():
    """A clipped pass must not give the first action all the futures and the last one three."""
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    report = sim.simulate((5.0,), ["left", "right", "stay"], branches=200_000, horizon=8,
                          reward_fn=_home_reward, budget_ms=300)
    drawn = [o.branches for o in report.action_rankings]
    assert min(drawn) >= 1
    assert max(drawn) <= 2 * min(drawn)             # comparable evidence per action


def test_a_budget_too_small_still_imagines_something():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    report = sim.simulate((5.0,), ["left", "right"], branches=500, horizon=4,
                          reward_fn=_home_reward, budget_ms=0.001)
    assert report.best_action in ("left", "right")
    assert all(o.branches >= 1 for o in report.action_rankings)


def test_no_budget_means_every_branch_asked_for():
    sim = TimelineSimulator(world_model=_trained(), seed=0)
    report = sim.simulate((5.0,), ["left", "right"], branches=400, horizon=4,
                          reward_fn=_home_reward)
    assert report.total_branches == 400
