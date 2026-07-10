"""Tests for the world-model upgrades: done head, calibration, persistence, grounding."""

from __future__ import annotations

import random

import pytest

from nyxara.mind.world_model import (
    GroundedWorldModel,
    NeuralWorldModel,
    WorldModel,
    build_world_model,
    load_world_model,
    save_world_model,
)

try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - depends on install
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


def _true_step(x: float, a: str) -> float:
    return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]


def _feed(model, n: int = 300, seed: int = 0, with_done: bool = True) -> None:
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        nx = _true_step(x, a)
        done = with_done and abs(nx) < 0.2
        model.observe((x,), a, (nx,), reward=-abs(nx), done=done)


def test_knn_done_head_and_calibration():
    wm = WorldModel(k=3, distance_scale=2.0)
    _feed(wm)
    near_goal = wm.predict((0.9,), "left")
    far = wm.predict((8.0,), "left")
    assert near_goal.done_prob > far.done_prob
    assert wm.prediction_error_ema() >= 0.0
    assert wm.mean_epistemic() == 0.0            # a kNN has no ensemble disagreement


def test_every_backend_has_mean_epistemic():
    for backend in ("knn", "neural", "ensemble", "auto"):
        model = build_world_model(backend)
        assert callable(getattr(model, "mean_epistemic", None))
        assert model.mean_epistemic() >= 0.0
        assert callable(getattr(model, "prediction_error_ema", None))


def test_neural_persistence_round_trip():
    nm = NeuralWorldModel(seed=3)
    _feed(nm, n=150, with_done=False)
    clone = NeuralWorldModel(seed=99)
    assert clone.load_dict(nm.to_dict()) is True
    p1 = nm.predict((4.0,), "left")
    p2 = clone.predict((4.0,), "left")
    assert p1.next_state == p2.next_state
    assert p1.confidence == pytest.approx(p2.confidence)


@needs_numpy
def test_transfer_persistence_round_trip(tmp_path):
    tm = build_world_model("auto", seed=0)
    _feed(tm, n=250)
    path = str(tmp_path / "wm.json")
    assert save_world_model(tm, path) == path
    clone = build_world_model("auto", seed=0)
    assert load_world_model(clone, path) is True
    p1 = tm.predict((4.0,), "left")
    p2 = clone.predict((4.0,), "left")
    assert p1.next_state[0] == pytest.approx(p2.next_state[0])
    assert p1.confidence == pytest.approx(p2.confidence)
    assert p1.done_prob == pytest.approx(p2.done_prob)


@needs_numpy
def test_ensemble_done_head_and_realized_error():
    em = build_world_model("ensemble", seed=0)
    _feed(em, n=400)
    near_goal = em.predict((0.9,), "left")
    far = em.predict((8.0,), "left")
    assert near_goal.done_prob > far.done_prob
    assert em.prediction_error_ema() > 0.0       # realized errors were tracked
    assert em.mean_epistemic() >= 0.0


def test_rollout_stops_on_predicted_done():
    wm = WorldModel(k=3, distance_scale=2.0)
    # every 'end' action deterministically terminates
    for i in range(30):
        wm.observe((float(i % 5),), "end", (float(i % 5),), reward=0.0, done=True)
    traj = wm.rollout((2.0,), ["end"] * 10, steps=10)
    assert traj.length == 1                      # imagination stopped at the learned end


def test_grounded_model_encodes_text_states():
    g = GroundedWorldModel(WorldModel(k=2, distance_scale=1.0), embedder=None,
                           state_latent_dim=8)
    for _ in range(10):
        g.observe("disk nearly full", "clean_tmp", "disk healthy", reward=1.0)
    pred = g.predict("disk nearly full", "clean_tmp")
    assert len(pred.next_state) == 8
    assert pred.confidence > 0.5                 # an exactly-seen situation is known
    assert g.stats()["grounded"] is True
    # same text always encodes to the same latent (deterministic grounding)
    assert g.encode_state("disk nearly full") == g.encode_state("disk nearly full")
    # numeric states pass through untouched
    assert g.encode_state((1.0, 2.0)) == (1.0, 2.0)


def test_grounded_model_uses_shared_embedder():
    from nyxara.memory.store import LexicalSemanticEmbedder
    g = GroundedWorldModel(WorldModel(k=2, distance_scale=1.0),
                           embedder=LexicalSemanticEmbedder(dim=64),
                           state_latent_dim=8)
    a = g.encode_state("the disk is nearly full")
    b = g.encode_state("bird song at dawn")
    assert a != b
    assert len(a) == len(b) == 8


class _StubCausal:
    """A causal model that knows one action ('known') and its effect."""

    def __init__(self):
        self._links = {("act:known", "outcome"): object()}

    def effects_of(self, cause):
        return ["outcome"] if cause == "act:known" else []


def test_grounded_causal_prior_damps_unknown_actions():
    inner = WorldModel(k=1, distance_scale=0.001)   # low confidence everywhere
    g = GroundedWorldModel(inner, state_latent_dim=4, causal_model=_StubCausal())
    g.observe((0.0, 0.0, 0.0, 0.0), "mystery", (4.0, 4.0, 4.0, 4.0), reward=2.0)
    raw = inner.predict((2.0, 2.0, 2.0, 2.0), "mystery")
    damped = g.predict((2.0, 2.0, 2.0, 2.0), "mystery")
    assert raw.confidence < 0.35                    # prior only applies when unsure
    assert abs(damped.next_state[0]) < abs(raw.next_state[0])
    assert abs(damped.reward) < abs(raw.reward)
