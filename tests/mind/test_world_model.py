"""Tests for nyxara.mind.world_model."""

from __future__ import annotations

import random

import pytest

from nyxara.mind.world_model import (
    CounterfactualResult,
    Prediction,
    Trajectory,
    Transition,
    WorldModel,
    _dist,
)


def _true_step(x, a):
    return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]


def _trained(seed=0, n=400, k=3, scale=2.0):
    wm = WorldModel(k=k, distance_scale=scale)
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        nx = _true_step(x, a)
        wm.observe((x,), a, (nx,), reward=-abs(nx))
    return wm


# -------------------- distance -------------------- #
def test_dist_numeric():
    assert _dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_dist_symbolic():
    assert _dist(("a", "b"), ("a", "c")) == pytest.approx(1.0)
    assert _dist(("a",), ("a",)) == 0.0


def test_dist_mixed():
    d = _dist((1.0, "x"), (1.0, "y"))
    assert d == pytest.approx(1.0)


# -------------------- observe / stats -------------------- #
def test_observe_and_len():
    wm = WorldModel()
    wm.observe((0,), "right", (1,), reward=1.0)
    assert len(wm) == 1
    assert "right" in wm.actions()


def test_observe_many():
    wm = WorldModel()
    wm.observe_many([Transition((0,), "a", (1,), 0.0), Transition((1,), "a", (2,), 0.0)])
    assert len(wm) == 2


def test_max_transitions_eviction():
    wm = WorldModel(max_transitions=3)
    for i in range(5):
        wm.observe((i,), "a", (i + 1,))
    assert len(wm) == 3


def test_stats():
    wm = _trained(n=30)
    s = wm.stats()
    assert s["transitions"] == 30
    assert "left" in s["actions"]


# -------------------- prediction -------------------- #
def test_predict_learns_dynamics():
    wm = _trained()
    p = wm.predict((4.0,), "left")
    assert isinstance(p, Prediction)
    assert abs(p.next_state[0] - 3.0) < 0.3


def test_predict_right_and_stay():
    wm = _trained()
    assert abs(wm.predict((2.0,), "right").next_state[0] - 3.0) < 0.3
    assert abs(wm.predict((2.0,), "stay").next_state[0] - 2.0) < 0.3


def test_predict_unknown_action_zero_confidence():
    wm = _trained()
    p = wm.predict((0.0,), "fly")
    assert p.confidence == 0.0
    assert p.next_state == (0.0,)
    assert p.neighbors == 0


def test_confidence_drops_far_from_data():
    wm = _trained()
    near = wm.predict((3.0,), "left").confidence
    far = wm.predict((1000.0,), "left").confidence
    assert far < near
    assert near > 0.5


def test_predict_reward():
    wm = _trained()
    p = wm.predict((5.0,), "stay")
    assert p.reward < 0  # reward = -|x|, so negative


def test_coverage():
    wm = _trained()
    assert wm.coverage((3.0,), "left") > wm.coverage((1000.0,), "left")


def test_symbolic_states():
    wm = WorldModel()
    wm.observe(("room_a",), "go", ("room_b",), reward=1.0)
    wm.observe(("room_b",), "go", ("room_c",), reward=1.0)
    p = wm.predict(("room_a",), "go")
    assert p.next_state == ("room_b",)
    assert p.confidence == 1.0  # exact match


# -------------------- rollouts -------------------- #
def test_rollout_with_action_list():
    wm = _trained()
    traj = wm.rollout((0.0,), ["right", "right", "right"], steps=3)
    assert isinstance(traj, Trajectory)
    assert traj.length == 3
    assert abs(traj.final_state[0] - 3.0) < 0.5


def test_rollout_with_policy_callable():
    wm = _trained()

    def go_home(state):
        x = state[0]
        return "left" if x > 0.5 else ("right" if x < -0.5 else "stay")

    traj = wm.rollout((8.0,), go_home, steps=15)
    assert abs(traj.final_state[0]) < 1.5


def test_rollout_terminal_fn():
    wm = _trained()
    traj = wm.rollout((5.0,), ["left"] * 10, steps=10,
                      terminal_fn=lambda s: s[0] <= 2.0)
    assert traj.final_state[0] <= 2.5
    assert traj.length < 10  # stopped early


def test_rollout_custom_reward_fn():
    wm = _trained()
    traj = wm.rollout((0.0,), ["right", "right"], steps=2,
                      reward_fn=lambda s, a, ns: 100.0)
    assert traj.total_reward == 200.0


def test_rollout_policy_exhausted():
    wm = _trained()
    traj = wm.rollout((0.0,), ["right"], steps=5)  # only one action provided
    assert traj.length == 1


def test_imagine_repeated_action():
    wm = _trained()
    traj = wm.imagine((0.0,), "right", steps=4)
    assert traj.length == 4
    assert abs(traj.final_state[0] - 4.0) < 0.6


def test_trajectory_properties():
    wm = _trained()
    traj = wm.rollout((0.0,), ["right"], steps=1)
    assert traj.mean_confidence > 0
    assert traj.total_reward == sum(traj.rewards)
    assert "total_reward" in traj.to_dict()


# -------------------- counterfactuals -------------------- #
def test_counterfactual_left_better_than_right():
    wm = _trained()
    cf = wm.counterfactual((5.0,), ["left"] * 6, ["right"] * 6, steps=6)
    assert isinstance(cf, CounterfactualResult)
    assert cf.better() == "a"
    assert cf.reward_difference > 0


def test_counterfactual_divergence_step():
    wm = _trained()
    cf = wm.counterfactual((5.0,), ["left"] * 4, ["right"] * 4, steps=4)
    assert cf.divergence_step == 1  # different first action -> diverge at step 1


def test_counterfactual_identical_policies():
    wm = _trained()
    cf = wm.counterfactual((5.0,), ["left"] * 3, ["left"] * 3, steps=3)
    assert cf.divergence_step is None
    assert abs(cf.reward_difference) < 1e-6
    assert cf.better() == "tie"


# -------------------- interventions -------------------- #
def test_intervene_forces_action():
    wm = _trained()
    base = wm.rollout((4.0,), ["left"] * 5, steps=5)
    iv = wm.intervene((4.0,), ["left"] * 5, at_step=2, action="right", steps=5)
    # forcing a 'right' detour leaves it further from home than the pure-left plan
    assert iv.final_state[0] > base.final_state[0]


def test_intervene_at_step_zero():
    wm = _trained()
    iv = wm.intervene((0.0,), ["stay"] * 3, at_step=0, action="right", steps=3)
    # first move is forced right, then stay -> ends near 1
    assert abs(iv.final_state[0] - 1.0) < 0.6


# --------------------------------------------------------------------------- #
# Neural forward model (Pillar B6): generalises dynamics, honest confidence
# --------------------------------------------------------------------------- #
from nyxara.mind.world_model import NeuralWorldModel


def _train_1d(wm, n=500, seed=0):
    def step(x, a):
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (step(x, a),), reward=-abs(step(x, a)))
    return wm


def test_neural_world_model_is_a_drop_in():
    wm = NeuralWorldModel()
    assert isinstance(wm, WorldModel)          # inherits rollout/counterfactual/intervene


def test_neural_model_learns_dynamics():
    wm = _train_1d(NeuralWorldModel(hidden=12, lr=0.05, epochs=5))
    p = wm.predict((4.0,), "left")
    assert abs(p.next_state[0] - 3.0) < 0.4    # learned the Δ for 'left'
    assert p.confidence > 0.5


def test_neural_confidence_is_honest_out_of_distribution():
    wm = _train_1d(NeuralWorldModel())
    near = wm.predict((4.0,), "left").confidence
    far = wm.predict((1000.0,), "left").confidence
    assert far < near and far < 0.1            # no hallucinated certainty far from data


def test_neural_rollout_plans_a_path_home():
    wm = _train_1d(NeuralWorldModel(hidden=12, lr=0.05, epochs=5))

    def go_home(state):
        x = state[0]
        return "left" if x > 0.5 else ("right" if x < -0.5 else "stay")

    traj = wm.rollout((8.0,), go_home, steps=14)
    assert abs(traj.final_state[0]) < 1.5      # imagined its way home via the learned model


def test_neural_unknown_action_is_a_zero_confidence_noop():
    wm = _train_1d(NeuralWorldModel())
    p = wm.predict((0.0,), "teleport")
    assert p.confidence == 0.0 and p.next_state == (0.0,)


def test_neural_ignores_symbolic_states():
    wm = NeuralWorldModel()
    wm.observe(("home",), "go", ("away",), reward=1.0)   # non-numeric -> not learned
    assert len(wm) == 0
