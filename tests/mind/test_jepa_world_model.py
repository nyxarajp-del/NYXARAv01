"""Tests for the hierarchical JEPA latent world model (mind/jepa_world_model.py)."""

from __future__ import annotations

import math
import random

import pytest

from nyxara.mind.world_model import (
    GroundedWorldModel,
    build_world_model,
    load_world_model,
    save_world_model,
)

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - depends on install
    np = None
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

if _HAS_NUMPY:
    from nyxara.mind.jepa_world_model import JEPAWorldModel, PlanResult

MOVES = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def _true_step(p, a):
    dx, dy = MOVES[a]
    return (p[0] + dx, p[1] + dy)


def _small(**kw):
    """A fast-training JEPA for tests (full-size defaults are exercised in the demo)."""
    params = dict(latent_dim=16, coarse_dim=8, enc_hidden=(64,), pred_hidden=(64, 64),
                  n_predictors=3, horizons=(1, 2, 4), coarse_from_horizon=4,
                  batch=32, iters=2, seed=0)
    params.update(kw)
    return JEPAWorldModel(**params)


def _feed_trajectories(model, episodes: int = 40, length: int = 12, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(episodes):
        pos = (float(rng.randint(-5, 5)), float(rng.randint(-5, 5)))
        for t in range(length):
            a = rng.choice(list(MOVES))
            nxt = _true_step(pos, a)
            model.observe(pos, a, nxt, reward=-abs(nxt[0]) - abs(nxt[1]),
                          done=t == length - 1)
            pos = nxt


# --------------------------------------------------------------------------- #
# drop-in contract
# --------------------------------------------------------------------------- #
@needs_numpy
def test_factory_builds_jepa_drop_in():
    m = build_world_model("jepa")
    assert isinstance(m, JEPAWorldModel)
    _feed_trajectories(m, episodes=6, length=6)
    p = m.predict((1.0, 1.0), "left")
    assert 0.0 <= p.confidence <= 1.0
    traj = m.rollout((2.0, 2.0), ["left", "down"], steps=2)
    assert traj.length <= 2
    cf = m.counterfactual((2.0, 0.0), ["left"] * 3, ["right"] * 3, steps=3)
    assert cf.better() in ("a", "b", "tie")
    ivn = m.intervene((2.0, 0.0), ["left"] * 3, at_step=1, action="up", steps=3)
    assert ivn.length <= 3


def test_factory_degrades_without_numpy(monkeypatch):
    import nyxara.mind.world_model as wm
    monkeypatch.setattr(wm, "_HAS_NUMPY", False)
    m = wm.build_world_model("jepa")
    # never raises; lands on a stdlib learner that still serves the surface
    m.observe((1.0,), "a", (2.0,))
    assert m.predict((1.0,), "a") is not None


# --------------------------------------------------------------------------- #
# learning: dynamics, multi-scale, hierarchy
# --------------------------------------------------------------------------- #
@needs_numpy
def test_learns_one_step_dynamics():
    m = _small()
    _feed_trajectories(m)
    m.consolidate(iters=40)
    p = m.predict((4.0, 0.0), "left")
    assert abs(p.next_state[0] - 3.0) < 0.8
    assert abs(p.next_state[1]) < 0.8
    assert p.confidence > 0.1


@needs_numpy
def test_multiscale_horizons():
    m = _small()
    _feed_trajectories(m, episodes=60)
    m.consolidate(iters=60)
    p2 = m.predict_horizon((4.0, 0.0), "left", 2)
    # two lefts move x by about -2; direction must be right even if magnitude is soft
    assert p2.next_state[0] < 3.5
    multi = m.imagine_multiscale((4.0, 0.0), "left")
    assert set(multi) == {1, 2, 4}
    assert all(hasattr(p, "confidence") for p in multi.values())


@needs_numpy
def test_coarse_hierarchy_exists_and_serves_long_horizons():
    m = _small()
    _feed_trajectories(m, episodes=60)
    m.consolidate(iters=60)
    assert 4 in m._c_preds                       # coarse bank owns horizon 4
    p4 = m.predict_horizon((5.0, 0.0), "left", 4)
    assert p4.next_state[0] < 5.0                # moves left, through the coarse route
    s = m.stats()
    assert s["hierarchy"]["coarse_dim"] == 8
    assert s["backend"] == "jepa-hierarchical"


# --------------------------------------------------------------------------- #
# energy, plausibility, surprise
# --------------------------------------------------------------------------- #
@needs_numpy
def test_energy_ranks_transitions():
    m = _small()
    _feed_trajectories(m)
    m.consolidate(iters=40)
    e_true = m.energy((4.0, 0.0), "left", (3.0, 0.0))
    e_false = m.energy((4.0, 0.0), "left", (6.0, 3.0))
    assert e_true < e_false
    p_true = m.plausibility((4.0, 0.0), "left", (3.0, 0.0))
    p_false = m.plausibility((4.0, 0.0), "left", (6.0, 3.0))
    assert 0.0 <= p_false < p_true <= 1.0


@needs_numpy
def test_energy_honest_on_unknown():
    m = _small()
    _feed_trajectories(m, episodes=6, length=6)
    assert m.energy((1.0, 1.0), "teleport", (0.0, 0.0)) == float("inf")
    assert m.plausibility((1.0, 1.0), "teleport", (0.0, 0.0)) == 0.0
    assert m.surprise((1.0, 1.0), "teleport", (0.0, 0.0)) == 1.0


@needs_numpy
def test_surprise_decreases_with_training():
    m = _small()
    rng = random.Random(1)
    early, late = [], []
    for i in range(300):
        pos = (float(rng.randint(-5, 5)), float(rng.randint(-5, 5)))
        a = rng.choice(list(MOVES))
        m.observe(pos, a, _true_step(pos, a))
        (early if i < 50 else late).append(m.last_surprise)
    assert sum(late[-50:]) / 50 < sum(early) / len(early)
    lp = m.learning_progress_by_action()
    assert lp and all(0.0 <= v <= 1.0 for v in lp.values())


# --------------------------------------------------------------------------- #
# JEPA internals: EMA target, anti-collapse, honest uncertainty
# --------------------------------------------------------------------------- #
@needs_numpy
def test_ema_target_lags_and_tracks():
    m = _small()
    _feed_trajectories(m, episodes=10, length=6)
    # target must differ from context (it lags) …
    gap = sum(float(np.abs(w1 - w2).sum())
              for w1, w2 in zip(m._enc.W, m._tgt.W))
    assert gap > 0.0
    # … but track it: far closer to the live encoder than a fresh random net is
    fresh = JEPAWorldModel(latent_dim=16, coarse_dim=8, enc_hidden=(64,),
                           pred_hidden=(64, 64), n_predictors=3, horizons=(1, 2, 4),
                           coarse_from_horizon=4, seed=99)
    fresh._build_nets(2)
    fresh_gap = sum(float(np.abs(w1 - w2).sum())
                    for w1, w2 in zip(m._enc.W, fresh._enc.W))
    assert gap < fresh_gap


@needs_numpy
def test_anti_collapse():
    m = _small()
    _feed_trajectories(m)
    m.consolidate(iters=40)
    assert m.stats()["collapse_std"] > 0.1
    # distant states must get distant latents
    za, _ = m._encode((5.0, 5.0))
    zb, _ = m._encode((-5.0, -5.0))
    assert float(np.linalg.norm(za - zb)) > 0.5


@needs_numpy
def test_ood_and_unknown_action_honest():
    m = _small()
    _feed_trajectories(m)
    m.consolidate(iters=20)
    near = m.predict((2.0, 2.0), "left").confidence
    far = m.predict((500.0, -500.0), "left").confidence
    assert far < near
    unknown = m.predict((2.0, 2.0), "warp")
    assert unknown.confidence == 0.0
    assert m.predict((2.0, 2.0), "left").epistemic > 0.0


@needs_numpy
def test_symbolic_falls_back_to_knn():
    m = _small()
    m.observe(("sunny",), "rain_dance", ("wet",))
    p = m.predict(("sunny",), "rain_dance")
    assert p.next_state == ("wet",)
    assert m.stats()["symbolic_transitions"] == 1


@needs_numpy
def test_calibration_contract_for_kernel():
    m = _small()
    _feed_trajectories(m)
    assert m.prediction_error_ema() >= 0.0
    assert m.mean_epistemic() >= 0.0
    assert 0.0 <= m.learning_progress((2.0, 2.0), "left") <= 1.0


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
@needs_numpy
def test_planner_moves_toward_goal():
    m = _small()
    _feed_trajectories(m, episodes=60)
    m.consolidate(iters=60)
    plan = m.plan((5.0, 5.0), goal_state=(0.0, 0.0), horizon=6, n_samples=48, iters=3)
    assert isinstance(plan, PlanResult)
    assert len(plan.actions) == 6
    toward = sum(1 for a in plan.actions if a in ("left", "down"))
    assert toward >= 3
    assert len(plan.predictions) == 6
    assert 0.0 <= plan.confidence <= 1.0


@needs_numpy
def test_planner_honest_when_untrained():
    m = _small()
    plan = m.plan((1.0, 1.0), horizon=4)
    assert plan.actions == [] and plan.confidence == 0.0


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
@needs_numpy
def test_persistence_roundtrip(tmp_path):
    m = _small()
    _feed_trajectories(m)
    m.consolidate(iters=20)
    path = str(tmp_path / "jepa.json")
    assert save_world_model(m, path)
    clone = _small()
    assert load_world_model(clone, path)
    a = m.predict((3.0, 1.0), "down")
    b = clone.predict((3.0, 1.0), "down")
    assert all(abs(x - y) < 1e-6 for x, y in zip(a.next_state, b.next_state))
    ea = m.energy((3.0, 1.0), "down", (3.0, 0.0))
    eb = clone.energy((3.0, 1.0), "down", (3.0, 0.0))
    assert math.isclose(ea, eb, rel_tol=1e-6)


@needs_numpy
def test_persistence_rejects_mismatched_architecture(tmp_path):
    m = _small()
    _feed_trajectories(m, episodes=6, length=6)
    path = str(tmp_path / "jepa.json")
    assert save_world_model(m, path)
    other = _small(latent_dim=32)
    assert not load_world_model(other, path)     # honest refusal, fresh start


# --------------------------------------------------------------------------- #
# grounded integration — text transitions through the wrapper
# --------------------------------------------------------------------------- #
@needs_numpy
def test_grounded_wrapper_serves_jepa_surface():
    g = GroundedWorldModel(_small(), embedder=None, state_latent_dim=8)
    for i in range(80):
        g.observe(f"master asked question {i}", "act:respond",
                  f"nyxara replied {i}", reward=0.5)
    p = g.predict("master asked question 3", "act:respond")
    assert hasattr(p, "confidence")
    e = g.energy("master asked question 3", "act:respond", "nyxara replied 3")
    assert isinstance(e, float)
    pl = g.plausibility("master asked question 3", "act:respond", "nyxara replied 3")
    assert 0.0 <= pl <= 1.0
    s = g.surprise("master asked question 3", "act:respond", "nyxara replied 3")
    assert 0.0 <= s <= 1.0
    assert g.consolidate(4) >= 0
    ph = g.predict_horizon("master asked question 3", "act:respond", 2)
    assert hasattr(ph, "confidence")
    plan = g.plan("master asked question 3", horizon=2)
    assert plan is None or hasattr(plan, "actions")
    assert g.stats()["backend"] == "jepa-hierarchical"


@needs_numpy
def test_grounded_wrapper_safe_on_non_jepa_backends():
    g = GroundedWorldModel(build_world_model("knn"), embedder=None)
    g.observe("hello", "act", "world")
    assert g.energy("hello", "act", "world") == float("inf")
    assert g.plausibility("hello", "act", "world") == 0.0
    assert g.surprise("hello", "act", "world") == 1.0
    assert g.consolidate() == 0
    assert g.plan("hello") is None
    assert hasattr(g.predict_horizon("hello", "act", 4), "confidence")
