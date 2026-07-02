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


def test_learning_progress_bounds():
    wm = _trained()
    for state, action in [((3.0,), "left"), ((1000.0,), "left"), ((0.0,), "never_tried")]:
        lp = wm.learning_progress(state, action)
        assert 0.0 <= lp <= 1.0


def test_learning_progress_unseen_action_is_maximal():
    wm = _trained()
    # a never-observed action is entirely un-modelled ⇒ maximal information gain
    assert wm.learning_progress((3.0,), "never_tried") == pytest.approx(1.0)


def test_learning_progress_falls_with_competence():
    wm = _trained()
    # well-modelled (state, action) is less worth exploring than an out-of-distribution one
    known = wm.learning_progress((3.0,), "left")
    ood = wm.learning_progress((1000.0,), "left")
    assert known < ood
    assert known < 1.0


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


# --------------------------------------------------------------------------- #
# Deep-ensemble model: real learned dynamics + epistemic uncertainty
# --------------------------------------------------------------------------- #
np = pytest.importorskip("numpy")  # the ensemble model requires numpy

from nyxara.mind.world_model import (  # noqa: E402
    EnsembleWorldModel,
    TransferWorldModel,
    build_world_model,
)


def _train_1d_ens(wm, n=600, seed=0):
    def step(x, a):
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (step(x, a),), reward=-abs(step(x, a)))
    return wm


def test_ensemble_is_a_drop_in():
    wm = EnsembleWorldModel()
    assert isinstance(wm, WorldModel)                    # inherits rollout/counterfactual/intervene


def test_ensemble_learns_dynamics():
    wm = _train_1d_ens(EnsembleWorldModel(seed=0))
    p = wm.predict((4.0,), "left")
    assert abs(p.next_state[0] - 3.0) < 0.4
    assert p.confidence > 0.5
    # states stay native python floats (no numpy types leaking into trajectories)
    assert type(p.next_state[0]) is float


def test_ensemble_confidence_and_epistemic_are_honest_ood():
    wm = _train_1d_ens(EnsembleWorldModel(seed=0))
    near = wm.predict((4.0,), "left")
    far = wm.predict((1000.0,), "left")
    assert far.confidence < near.confidence and far.confidence < 0.1
    # epistemic uncertainty (ensemble disagreement) is never negative and is reported
    assert near.epistemic >= 0.0 and far.epistemic >= 0.0


def test_ensemble_rollout_plans_a_path_home():
    wm = _train_1d_ens(EnsembleWorldModel(seed=0))

    def go_home(state):
        x = state[0]
        return "left" if x > 0.5 else ("right" if x < -0.5 else "stay")

    traj = wm.rollout((8.0,), go_home, steps=14)
    assert abs(traj.final_state[0]) < 1.5


def test_ensemble_unknown_action_is_zero_confidence_noop():
    wm = _train_1d_ens(EnsembleWorldModel())
    p = wm.predict((0.0,), "teleport")
    assert p.confidence == 0.0 and p.next_state == (0.0,)


def test_ensemble_handles_symbolic_states_via_exact_memory():
    # the ensemble is a strict superset of the kNN: symbolic states fall back to exact memory
    wm = EnsembleWorldModel()
    wm.observe(("room_a",), "go", ("room_b",), reward=1.0)
    wm.observe(("room_b",), "go", ("room_c",), reward=1.0)
    assert len(wm) == 2
    p = wm.predict(("room_a",), "go")
    assert p.next_state == ("room_b",) and p.confidence == 1.0   # exact recall
    assert "go" in wm.actions()


def test_ensemble_mixes_numeric_and_symbolic():
    wm = EnsembleWorldModel(seed=0)
    _train_1d_ens(wm)                                    # numeric dynamics in the ensemble
    wm.observe(("here",), "warp", ("there",), reward=2.0)  # symbolic in the kNN fallback
    assert abs(wm.predict((4.0,), "left").next_state[0] - 3.0) < 0.4
    assert wm.predict(("here",), "warp").next_state == ("there",)


def test_ensemble_generalises_to_unseen_states():
    # train only on |x| < 5; a kNN would clamp/interpolate, the ensemble extrapolates the linear Δ
    wm = EnsembleWorldModel(seed=0)
    rng = random.Random(3)
    for _ in range(600):
        x = rng.uniform(-5, 5)
        wm.observe((x,), "right", (x + 1.0,), reward=-abs(x + 1.0))
    # a modestly-unseen state still predicts the learned +1 delta
    p = wm.predict((6.5,), "right")
    assert abs(p.next_state[0] - 7.5) < 0.8


def test_build_world_model_factory_backends_with_numpy():
    # "auto" now prefers the cross-domain TransferWorldModel (still a WorldModel drop-in)
    auto = build_world_model("auto")
    assert isinstance(auto, TransferWorldModel) and isinstance(auto, WorldModel)
    # the per-action deep ensemble is still available explicitly
    assert isinstance(build_world_model("ensemble"), EnsembleWorldModel)
    assert isinstance(build_world_model("knn"), WorldModel)
    assert isinstance(build_world_model("neural"), NeuralWorldModel)
    # unknown kwargs are filtered per backend (callers may pass a superset safely)
    assert isinstance(build_world_model("knn", ensemble=9, k=2), WorldModel)


def test_ensemble_to_dict_includes_epistemic():
    wm = _train_1d_ens(EnsembleWorldModel(seed=0))
    d = wm.predict((2.0,), "right").to_dict()
    assert "epistemic" in d and "confidence" in d
