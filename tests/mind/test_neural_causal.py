"""NYXARA · tests/mind/test_neural_causal.py — the causal core that LEARNS BY GRADIENT DESCENT.

These prove the two learners in :mod:`nyxara.mind.neural_causal` are *real* gradient-descent learners,
not counters: a nonlinear mechanism trained by back-prop that genuinely beats the closed-form linear
ridge, and a NOTEARS structure learner that recovers a known causal DAG by optimising a differentiable
acyclicity objective — plus that both are wired into :class:`CausalWorldModel`, persist byte-for-byte,
and degrade to the symbolic floor. No external LLM anywhere.
"""

from __future__ import annotations

import json

import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - depends on install
    np = None
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

from nyxara.mind.causal_world_model import CausalWorldModel
from nyxara.mind.neural_causal import DifferentiableCausalStructure, NeuralCausalMechanism


# --------------------------------------------------------------------------- #
# 1. NeuralCausalMechanism — real back-prop, and strictly better on a nonlinear law
# --------------------------------------------------------------------------- #
@needs_numpy
def test_neural_mechanism_beats_ridge_on_nonlinear_law():
    rng = np.random.default_rng(0)
    m = NeuralCausalMechanism(seed=1, warmup=8, steps=20)
    xs = rng.uniform(-2.0, 2.0, 400)
    for x in xs:
        m.add(float(x), float(x * x))                 # y = x² — a line cannot fit this
    tx = np.linspace(-2.0, 2.0, 60)
    neural_mse = float(np.mean([(m.predict(float(x)) - x * x) ** 2 for x in tx]))
    ridge_mse = float(np.mean([(m._ridge.predict(float(x)) - x * x) ** 2 for x in tx]))
    assert m._trained
    assert neural_mse < 0.25 * ridge_mse              # the learned net is far better than the line
    assert m.r2 > 0.9


@needs_numpy
def test_neural_mechanism_gradient_actually_moves_the_fit():
    rng = np.random.default_rng(2)
    xs = [float(v) for v in rng.uniform(-2.0, 2.0, 24)]
    m = NeuralCausalMechanism(seed=2, warmup=8, steps=1)   # only ONE gradient step per add (slow)
    for x in xs:
        m.add(x, 3.0 * x)                             # y = 3x — non-degenerate target
    tx = np.linspace(-2.0, 2.0, 40)
    before = float(np.mean([(m.predict(float(x)) - 3.0 * x) ** 2 for x in tx]))
    for _ in range(40):                               # many more gradient steps
        for x in xs:
            m.add(x, 3.0 * x)
    after = float(np.mean([(m.predict(float(x)) - 3.0 * x) ** 2 for x in tx]))
    assert after < before                             # the weights measurably learned the target


@needs_numpy
def test_neural_mechanism_interventional_upweighting_shifts_the_fit():
    # Same cue value 1.0 seen with two different effects; the intervention (weight>1) should pull
    # the learned prediction toward the interventional target.
    m = NeuralCausalMechanism(seed=3, warmup=6, steps=6)
    for _ in range(40):
        m.add(1.0, 0.0, weight=1.0)                   # plentiful observational evidence: y≈0
    for _ in range(40):
        m.add(1.0, 10.0, weight=5.0)                  # up-weighted do-evidence: y≈10
    assert m.predict(1.0) > 3.0                       # the do-samples visibly move the estimate


@needs_numpy
def test_neural_mechanism_roundtrips_through_json():
    rng = np.random.default_rng(4)
    m = NeuralCausalMechanism(seed=5, warmup=8, steps=15)
    for x in rng.uniform(-1.5, 1.5, 200):
        m.add(float(x), float(0.5 * x * x - x))
    d = json.loads(json.dumps(m.to_dict()))           # must be JSON-safe
    m2 = NeuralCausalMechanism.from_dict(d)
    for x in (-1.0, 0.0, 0.7, 1.3):
        assert abs(m2.predict(x) - m.predict(x)) < 1e-9


def test_neural_mechanism_delegates_to_ridge_before_warmup():
    # Below the warmup threshold it IS the closed-form ridge — never worse than today's floor.
    m = NeuralCausalMechanism(warmup=100)
    for x in (0.0, 1.0, 2.0, 3.0):
        m.add(float(x), float(2.0 * x + 1.0))
    assert not m._trained
    assert abs(m.predict(2.0) - m._ridge.predict(2.0)) < 1e-9
    assert abs(m.slope - m._ridge.slope) < 1e-9


# --------------------------------------------------------------------------- #
# 2. DifferentiableCausalStructure — the DAG learned by gradient descent (NOTEARS)
# --------------------------------------------------------------------------- #
@needs_numpy
def test_notears_recovers_a_known_chain():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(500)
    B = 2.0 * A + 0.2 * rng.standard_normal(500)
    C = 1.5 * B + 0.2 * rng.standard_normal(500)
    X = np.stack([A, B, C], axis=1)
    s = DifferentiableCausalStructure(["A", "B", "C"]).fit(X)
    assert s.fitted
    # forward edges learned strong, reverse edges suppressed to ~0
    assert s.weight("A", "B") > 1.0 and s.weight("B", "C") > 0.8
    assert abs(s.weight("B", "A")) < 0.2 and abs(s.weight("C", "B")) < 0.2
    assert s.agrees_with("A", "B") and not s.agrees_with("B", "A")
    assert abs(s.h_final) < 1e-3                       # the learned graph is (near-)acyclic


@needs_numpy
def test_notears_roundtrips_through_json():
    rng = np.random.default_rng(1)
    A = rng.standard_normal(300)
    B = 2.0 * A + 0.1 * rng.standard_normal(300)
    X = np.stack([A, B], axis=1)
    s = DifferentiableCausalStructure(["A", "B"]).fit(X)
    d = json.loads(json.dumps(s.to_dict()))
    s2 = DifferentiableCausalStructure.from_dict(d)
    assert abs(s2.weight("A", "B") - s.weight("A", "B")) < 1e-6


# --------------------------------------------------------------------------- #
# 3. Wired into CausalWorldModel — neural mechanisms + learned structure, persisted
# --------------------------------------------------------------------------- #
def _valued_chain(seed: int = 7) -> CausalWorldModel:
    import random
    rng = random.Random(seed)
    cwm = CausalWorldModel(window=10.0, min_pairs_fit=8, structure_min_samples=8)
    for base in range(0, 30000, 100):
        if rng.random() < 0.7:
            r = rng.uniform(1.0, 5.0)
            cwm.observe("rain", at=base, value=r)
            cwm.observe("wet", at=base + 1, value=2.0 * r + rng.uniform(-0.2, 0.2))
            cwm.observe("slippery", at=base + 2, value=3.0 * r + rng.uniform(-0.2, 0.2))
    cwm.discover()
    return cwm


@needs_numpy
def test_causal_world_model_fits_neural_mechanisms():
    cwm = _valued_chain()
    m = cwm.mechanism("rain", "wet")
    assert m is not None and getattr(m, "KIND", "") == "neural"
    assert 1.5 < m.slope < 2.5                         # learned dwet/drain ≈ 2
    assert abs(m.effect_of_change(1.0, 3.0) - 4.0) < 1.0
    assert cwm.stats()["neural_mechanisms"] >= 1


@needs_numpy
def test_causal_world_model_learns_structure_by_gradient_descent():
    cwm = _valued_chain()
    s = cwm.learned_structure()
    assert s is not None and s.fitted
    assert s.weight("rain", "wet") > s.weight("wet", "rain")
    assert "learned_structure" in cwm.stats()


@needs_numpy
def test_causal_world_model_persists_neural_state():
    cwm = _valued_chain()
    blob = json.loads(json.dumps(cwm.to_dict()))       # JSON-safe end to end
    restored = CausalWorldModel(window=10.0)
    restored.load_dict(blob)
    m0, m1 = cwm.mechanism("rain", "wet"), restored.mechanism("rain", "wet")
    assert m1 is not None and getattr(m1, "KIND", "") == "neural"
    assert abs(m1.predict(3.0) - m0.predict(3.0)) < 1e-6
    assert restored.learned_structure() is not None
