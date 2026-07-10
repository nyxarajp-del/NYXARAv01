"""Tests for the causal-model upgrades: valued events + learned functional mechanisms."""

from __future__ import annotations

import json
import random

import pytest

from nyxara.mind.causal_world_model import (
    CausalWorldModel,
    FunctionalCausalMechanism,
)


def _valued_chain(seed: int = 3, n: int = 150, slope: float = 2.0) -> CausalWorldModel:
    """A causal stream where b = slope·a + small noise, a → b within the window."""
    rng = random.Random(seed)
    cwm = CausalWorldModel(window=10.0, min_pairs_fit=8)
    for k in range(n):
        base = k * 100.0
        a = rng.uniform(0, 5)
        cwm.observe("a", at=base, value=a, intervention=True)
        cwm.observe("b", at=base + 1, value=slope * a + rng.gauss(0, 0.1))
    cwm.discover()
    return cwm


def test_mechanism_regression_recovers_slope():
    mech = FunctionalCausalMechanism()
    rng = random.Random(0)
    for _ in range(50):
        x = rng.uniform(-3, 3)
        mech.add(x, 2.0 * x + 1.0 + rng.gauss(0, 0.05))
    assert mech.slope == pytest.approx(2.0, abs=0.1)
    assert mech.intercept == pytest.approx(1.0, abs=0.2)
    assert mech.r2 > 0.95
    assert mech.confidence() > 0.9


def test_discover_fits_mechanism_on_valued_causal_edge():
    cwm = _valued_chain()
    assert cwm.is_causal("a", "b").verdict == "causal"
    mech = cwm.mechanism("a", "b")
    assert mech is not None
    assert mech.slope == pytest.approx(2.0, abs=0.15)
    assert mech.r2 > 0.9
    assert cwm.stats()["mechanisms"] == 1


def test_counterfactual_uses_learned_effect_size():
    cwm = _valued_chain()
    cf = cwm.counterfactual("a", "b", factual=3.0, counter=1.0)
    # b = 2a, so moving a from 3 → 1 shifts b by ≈ −4 — a REAL effect size,
    # not a probability lift (which would be bounded in [−1, 1])
    assert cf.delta == pytest.approx(-4.0, abs=0.4)


def test_mechanism_survives_persistence():
    cwm = _valued_chain()
    restored = CausalWorldModel()
    restored.load_dict(json.loads(json.dumps(cwm.to_dict())))
    mech = restored.mechanism("a", "b")
    assert mech is not None
    assert mech.slope == pytest.approx(cwm.mechanism("a", "b").slope)
    cf = restored.counterfactual("a", "b", factual=3.0, counter=1.0)
    assert cf.delta == pytest.approx(-4.0, abs=0.4)


def test_unvalued_edges_get_no_mechanism():
    rng = random.Random(1)
    cwm = CausalWorldModel(window=10.0)
    for k in range(100):
        base = k * 100.0
        if rng.random() < 0.6:
            cwm.observe("rain", at=base)
            cwm.observe("wet", at=base + 1)
    cwm.discover()
    assert cwm.is_causal("rain", "wet").verdict == "causal"
    assert cwm.mechanism("rain", "wet") is None      # no values, no function — honest


def test_valued_events_do_not_change_verdicts():
    """Values are additive: the discovery verdicts stay exactly as before."""
    rng = random.Random(7)
    plain = CausalWorldModel(window=10.0)
    valued = CausalWorldModel(window=10.0)
    for k in range(120):
        base = k * 100.0
        if rng.random() < 0.5:
            for m in (plain, valued):
                m.observe("summer", at=base)
            if rng.random() < 0.85:
                plain.observe("ice_cream", at=base + 1)
                valued.observe("ice_cream", at=base + 1, value=rng.uniform(10, 30))
            if rng.random() < 0.8:
                plain.observe("drowning", at=base + 2)
                valued.observe("drowning", at=base + 2, value=rng.uniform(0, 3))
    plain.discover()
    valued.discover()
    assert (plain.is_causal("ice_cream", "drowning").verdict
            == valued.is_causal("ice_cream", "drowning").verdict == "confounded")
