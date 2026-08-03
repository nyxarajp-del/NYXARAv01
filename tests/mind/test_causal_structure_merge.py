"""NOTEARS may corroborate the symbolic graph. It may not overrule it, and it may not add to it."""

from __future__ import annotations

import random

from nyxara.growth.dataset_causal import learn_causal_graph
from nyxara.mind.causal_world_model import CausalWorldModel


class _FakeStructure:
    """A fitted gradient structure with hand-set weights, so the merge can be tested exactly."""

    fitted = True

    def __init__(self, weights):
        self._w = dict(weights)

    def weight(self, cause, effect):
        return self._w.get((cause, effect), 0.0)


def _chain_model(n: int = 60, seed: int = 0) -> CausalWorldModel:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        rows.append({"x": x, "y": 2.0 * x + rng.gauss(0, 0.2)})
    model = CausalWorldModel()
    learn_causal_graph(["x", "y"], rows, model=model, order=["x", "y"])
    return model


def test_without_a_fitted_structure_nothing_changes():
    model = _chain_model()
    assert model.as_causal_graph(corroborate=True) == model.as_causal_graph(corroborate=False)


def test_agreement_leaves_the_weight_alone():
    model = _chain_model()
    model._structure = _FakeStructure({("x", "y"): 1.9, ("y", "x"): 0.0})
    on = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=True))
    off = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=False))
    assert on == off


def test_a_clear_reversal_damps_the_edge_rather_than_flipping_it():
    """NOTEARS inverts arrows on nonlinear laws — measured against sim/ ground truth. Its
    disagreement is a reason for less confidence, not a reason to trust the fit that just failed
    a check against a law true by construction."""
    model = _chain_model()
    before = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=False))
    model._structure = _FakeStructure({("x", "y"): 0.01, ("y", "x"): 1.5})
    after = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=True))

    assert set(after) == set(before)                      # the arrow is NOT flipped
    assert abs(after[("x", "y")]) < abs(before[("x", "y")])   # it is damped
    assert after[("x", "y")] == before[("x", "y")] * model.structure_disagreement_damping


def test_ordinary_gradient_noise_is_not_treated_as_disagreement():
    model = _chain_model()
    before = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=False))
    model._structure = _FakeStructure({("x", "y"): 1.0, ("y", "x"): 0.04})
    after = dict(((c, e), w) for c, e, w in model.as_causal_graph(corroborate=True))
    assert after == before


def test_the_gradient_never_creates_an_edge_the_symbolic_layer_refused():
    """The load-bearing restriction: an edge only NOTEARS believes in is not added at all. On
    v = sqrt(T/mu) it invents one with full confidence, so this must stay a hard rule."""
    model = _chain_model()
    model._structure = _FakeStructure({("y", "x"): 2.0, ("q", "z"): 5.0})
    edges = {(c, e) for c, e, _w in model.as_causal_graph(corroborate=True)}
    assert ("q", "z") not in edges
    assert ("y", "x") not in edges


def test_a_broken_structure_object_is_survivable():
    class _Exploding:
        fitted = True

        def weight(self, *_a):
            raise RuntimeError("boom")

    model = _chain_model()
    model._structure = _Exploding()
    assert model.as_causal_graph(corroborate=True)        # uncorroborated, never a crash


def test_do_and_counterfactual_now_see_the_merge():
    """The whole point of merging: as_causal_graph feeds _structural_model, which is what do()
    and counterfactual() propagate over. Before this, NOTEARS surfaced only in stats()."""
    model = _chain_model()
    plain = model.do({"x": 1.0}, target="y")
    model._structure = _FakeStructure({("x", "y"): 0.01, ("y", "x"): 1.5})
    damped = model.do({"x": 1.0}, target="y")
    assert abs(damped) < abs(plain)
