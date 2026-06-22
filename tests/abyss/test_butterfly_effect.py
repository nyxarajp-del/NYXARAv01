"""Tests for nyxara.abyss.butterfly_effect."""

from __future__ import annotations

import random

import pytest

from nyxara.abyss.butterfly_effect import ButterflyEffect, CascadeReport, Perturbation
from nyxara.mind.world_model import WorldModel


def _grow_decay_model(seed=0, n=1200):
    """1-D world: grow (x->1.5x, chaotic) and decay (x->0.5x, stable)."""
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(seed)
    for _ in range(n):
        x = rng.uniform(-5, 5)
        wm.observe((x,), "grow", (1.5 * x,))
        wm.observe((x,), "decay", (0.5 * x,))
    return wm


def _two_dim_model(seed=1, n=1500):
    """2-D world: only the first dimension amplifies."""
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.uniform(-5, 5)
        b = rng.uniform(-5, 5)
        wm.observe((a, b), "step", (1.5 * a, b))
    return wm


# -------------------- cascade / amplification -------------------- #
def test_chaotic_amplifies():
    bfe = ButterflyEffect(world_model=_grow_decay_model())
    rep = bfe.propagate((2.0,), ["grow"] * 6, Perturbation(0, 0.05, "seed"), horizon=6)
    assert isinstance(rep, CascadeReport)
    assert rep.is_chaotic
    assert rep.amplification > 2.0
    assert rep.final_divergence > rep.initial_divergence


def test_stable_damps():
    bfe = ButterflyEffect(world_model=_grow_decay_model())
    rep = bfe.propagate((2.0,), ["decay"] * 6, Perturbation(0, 0.05, "seed"), horizon=6)
    assert not rep.is_chaotic
    assert rep.amplification < 1.0
    assert rep.final_divergence < rep.initial_divergence


def test_divergence_curve_starts_at_nudge():
    bfe = ButterflyEffect(world_model=_grow_decay_model())
    rep = bfe.propagate((1.0,), ["grow"] * 5, Perturbation(0, 0.05), horizon=5)
    assert rep.divergence_curve[0] == pytest.approx(0.05, abs=1e-9)
    assert rep.initial_divergence == pytest.approx(0.05)


# -------------------- sensitivity ranking -------------------- #
def test_sensitivity_ranks_amplifying_dimension_first():
    bfe = ButterflyEffect(world_model=_two_dim_model())
    ranking = bfe.sensitivity((2.0, 2.0), ["step"] * 6, horizon=6, delta=0.05,
                              labels=["latency", "typing_speed"])
    assert len(ranking) == 2
    assert ranking[0].perturbation.dimension == 0
    assert ranking[0].amplification > ranking[1].amplification


def test_most_sensitive_returns_top():
    bfe = ButterflyEffect(world_model=_two_dim_model())
    top = bfe.most_sensitive((2.0, 2.0), ["step"] * 6, horizon=6, delta=0.05,
                             labels=["latency", "typing_speed"])
    assert top is not None
    assert top.perturbation.label == "latency"


def test_sensitivity_respects_explicit_dims():
    bfe = ButterflyEffect(world_model=_two_dim_model())
    ranking = bfe.sensitivity((2.0, 2.0), ["step"] * 4, horizon=4, delta=0.05, dims=[1])
    assert len(ranking) == 1
    assert ranking[0].perturbation.dimension == 1


# -------------------- honest uncertainty & edges -------------------- #
def test_untrained_model_low_confidence():
    bfe = ButterflyEffect(world_model=WorldModel())
    rep = bfe.propagate((1.0,), ["grow"] * 4, Perturbation(0, 0.01), horizon=4)
    assert rep.confidence < 0.3


def test_out_of_range_dimension_raises():
    bfe = ButterflyEffect(world_model=_grow_decay_model())
    with pytest.raises(IndexError):
        bfe.propagate((1.0,), ["grow"] * 3, Perturbation(5, 0.01), horizon=3)


def test_to_dict_shapes():
    bfe = ButterflyEffect(world_model=_grow_decay_model())
    d = bfe.propagate((2.0,), ["grow"] * 5, Perturbation(0, 0.05, "seed"), horizon=5).to_dict()
    assert set(d) >= {"perturbation", "dimension", "delta", "divergence_curve",
                      "amplification", "is_chaotic", "reward_shift", "confidence"}
    assert isinstance(d["divergence_curve"], list)


def test_default_world_model_built_when_none():
    bfe = ButterflyEffect()
    assert bfe.world_model is not None


def test_reproducible():
    a = ButterflyEffect(world_model=_grow_decay_model())
    b = ButterflyEffect(world_model=_grow_decay_model())
    ra = a.propagate((2.0,), ["grow"] * 6, Perturbation(0, 0.05), horizon=6)
    rb = b.propagate((2.0,), ["grow"] * 6, Perturbation(0, 0.05), horizon=6)
    assert ra.to_dict() == rb.to_dict()
