"""Tests for nyxara.quantum.superposition_states."""

from __future__ import annotations

import math

import pytest

from nyxara.quantum.superposition_states import (
    CollapseResult,
    Hypothesis,
    Superposition,
)


def _three():
    sp = Superposition(collapse_threshold=0.6)
    sp.add_many({"sabotage": 1.0, "accident": 1.0, "coincidence": 1.0})
    return sp


# -------------------- superposition & Born rule -------------------- #
def test_equal_superposition_is_uniform():
    sp = _three()
    probs = sp.probabilities()
    assert all(p == pytest.approx(1 / 3) for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0)


def test_entropy_maximal_when_uniform():
    sp = _three()
    assert sp.entropy() == pytest.approx(math.log2(3))


def test_probabilities_sum_to_one_after_add():
    sp = Superposition().add("a", 2.0).add("b", 1.0).add("c", 0.5)
    assert sum(sp.probabilities().values()) == pytest.approx(1.0)


# -------------------- holding contradictions without error -------------------- #
def test_holds_contradictions_without_error():
    sp = _three()
    sp.mark_contradictory("sabotage", "accident")
    sp.mark_contradictory("sabotage", "coincidence")
    live = sp.live_contradictions()
    assert len(live) == 2
    # collapsing while undecided does not raise and does not commit
    res = sp.collapse()
    assert isinstance(res, CollapseResult)
    assert not res.decided


def test_not_decided_when_superposed():
    assert not _three().is_decided()


# -------------------- evidence sharpens the posterior -------------------- #
def test_observe_sharpens_toward_evidence():
    sp = _three()
    before = sp.probabilities()["sabotage"]
    sp.observe({"sabotage": 0.8, "accident": 0.2, "coincidence": 0.1})
    after = sp.probabilities()["sabotage"]
    assert after > before
    assert sum(sp.probabilities().values()) == pytest.approx(1.0)


def test_entropy_drops_with_evidence():
    sp = _three()
    h0 = sp.entropy()
    sp.observe({"sabotage": 0.9, "accident": 0.1, "coincidence": 0.05})
    assert sp.entropy() < h0


def test_observe_is_bayesian_posterior():
    # uniform prior; posterior should equal normalised likelihoods
    sp = Superposition().add_many({"a": 1.0, "b": 1.0})
    sp.observe({"a": 0.75, "b": 0.25})
    probs = sp.probabilities()
    assert probs["a"] == pytest.approx(0.75, abs=1e-9)
    assert probs["b"] == pytest.approx(0.25, abs=1e-9)


def test_observe_unknown_label_ignored():
    sp = Superposition().add_many({"a": 1.0, "b": 1.0})
    sp.observe({"a": 0.6, "b": 0.4, "ghost": 0.9})
    assert "ghost" not in sp
    assert sum(sp.probabilities().values()) == pytest.approx(1.0)


# -------------------- collapse / decision -------------------- #
def test_collapse_chooses_max_probability():
    sp = _three()
    sp.observe({"sabotage": 0.8, "accident": 0.2, "coincidence": 0.1})
    sp.observe({"sabotage": 0.7, "accident": 0.25, "coincidence": 0.1})
    res = sp.collapse()
    assert res.label == "sabotage"
    assert res.decided and res.probability >= 0.6
    assert res.margin > 0.0
    assert res.runner_up in ("accident", "coincidence")


def test_ambiguous_stays_undecided_but_forceable():
    sp = Superposition(collapse_threshold=0.6).add_many({"A": 1.0, "B": 1.0})
    sp.observe({"A": 0.55, "B": 0.45})
    amb = sp.collapse()
    assert not amb.decided
    forced = sp.collapse(force=True)
    assert forced.decided and forced.label == "A"


def test_collapse_carries_payload():
    sp = Superposition(collapse_threshold=0.5)
    sp.add("yes", 3.0, payload={"action": "go"})
    sp.add("no", 1.0)
    res = sp.collapse()
    assert res.label == "yes"
    assert res.payload == {"action": "go"}


def test_empty_superposition_collapse():
    res = Superposition().collapse()
    assert res.label is None and not res.decided


# -------------------- bridge & serialization -------------------- #
def test_as_dirichlet_bridge():
    sp = _three()
    sp.observe({"sabotage": 0.9, "accident": 0.1, "coincidence": 0.05})
    d = sp.as_dirichlet()
    assert d is not None
    assert d.most_likely()[0] == "sabotage"


def test_to_dict_shapes():
    sp = _three()
    sp.mark_contradictory("sabotage", "accident")
    d = sp.to_dict()
    assert {"hypotheses", "entropy", "decided", "live_contradictions"} <= set(d)
    assert len(d["hypotheses"]) == 3


def test_hypothesis_dataclass():
    h = Hypothesis(label="x", amplitude=0.5, payload=1)
    assert h.to_dict()["label"] == "x"
