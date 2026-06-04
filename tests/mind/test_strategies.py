"""Tests for nyxara.mind.strategies."""

from __future__ import annotations

import pytest

from nyxara.mind.reasoner import Reasoner, ReasoningMode, ReasoningQuery
from nyxara.mind.strategies import (
    AbductiveStrategy,
    AnalogicalStrategy,
    BayesianStrategy,
    CausalModel,
    CausalStrategy,
    NeuroSymbolicStrategy,
    default_strategies,
)


# -------------------- Bayesian -------------------- #
def test_bayesian_base_rate():
    s = BayesianStrategy()
    c = s.reason(ReasoningQuery(payload={
        "hypotheses": {"sick": 0.01, "healthy": 0.99},
        "evidence": [{"sick": 0.99, "healthy": 0.05}]}))
    assert c.answer == "healthy"
    assert 0.8 < c.confidence < 0.85


def test_bayesian_strong_evidence_flips():
    s = BayesianStrategy()
    c = s.reason(ReasoningQuery(payload={
        "hypotheses": {"sick": 0.01, "healthy": 0.99},
        "evidence": [{"sick": 0.99, "healthy": 0.05}] * 5}))  # five positive tests
    assert c.answer == "sick"


def test_bayesian_posterior_normalised():
    s = BayesianStrategy()
    c = s.reason(ReasoningQuery(payload={
        "hypotheses": {"a": 0.5, "b": 0.5}, "evidence": [{"a": 0.8, "b": 0.2}]}))
    post = c.support["posterior"]
    assert abs(sum(post.values()) - 1.0) < 1e-6
    assert c.answer == "a"


def test_bayesian_applicability():
    s = BayesianStrategy()
    assert s.applicability(ReasoningQuery(payload={"hypotheses": {}, "evidence": []})) > 0
    assert s.applicability(ReasoningQuery(payload={})) == 0.0


# -------------------- Causal -------------------- #
def test_causal_model_effect():
    m = CausalModel([("rain", "wet", 0.9), ("wet", "slippery", 0.8)])
    assert m.effect_of("rain", 1.0, "slippery") == pytest.approx(0.72)


def test_causal_model_multiple_paths():
    m = CausalModel([("a", "c", 0.5), ("a", "b", 0.5), ("b", "c", 0.5)])
    # paths a->c (0.5) and a->b->c (0.25) sum to 0.75
    assert m.effect_of("a", 1.0, "c") == pytest.approx(0.75)


def test_causal_model_causes_of():
    m = CausalModel([("rain", "wet", 0.9), ("sprinkler", "wet", 0.7)])
    causes = [c for c, _ in m.causes_of("wet")]
    assert set(causes) == {"rain", "sprinkler"}
    assert causes[0] == "rain"  # stronger first


def test_causal_model_counterfactual():
    m = CausalModel([("x", "y", 2.0)])
    # going from x=1 to x=3 changes y by 2*(3-1)=4
    assert m.counterfactual("x", "y", factual=1.0, counter=3.0) == pytest.approx(4.0)


def test_causal_strategy_effect():
    s = CausalStrategy()
    c = s.reason(ReasoningQuery(payload={
        "graph": [("a", "b", 0.5)], "effect_of": {"do": {"a": 2.0}, "on": "b"}}))
    assert c.answer == pytest.approx(1.0)


def test_causal_strategy_causes():
    s = CausalStrategy()
    c = s.reason(ReasoningQuery(payload={
        "graph": [("rain", "wet", 0.9)], "causes_of": "wet"}))
    assert c.answer == ["rain"]


def test_causal_applicability():
    s = CausalStrategy()
    assert s.applicability(ReasoningQuery(payload={"graph": [], "causes_of": "x"})) > 0
    assert s.applicability(ReasoningQuery(payload={"graph": []})) == 0.0


# -------------------- Abductive -------------------- #
def test_abductive_best_explanation():
    s = AbductiveStrategy()
    c = s.reason(ReasoningQuery(payload={
        "observations": ["fever", "cough", "fatigue"],
        "hypotheses": {
            "flu": {"explains": ["fever", "cough", "fatigue"], "prior": 0.3, "complexity": 1},
            "cold": {"explains": ["cough"], "prior": 0.5, "complexity": 1},
        }}))
    assert c.answer == "flu"


def test_abductive_occam_penalises_complexity():
    s = AbductiveStrategy()
    c = s.reason(ReasoningQuery(payload={
        "observations": ["x", "y"],
        "hypotheses": {
            "simple": {"explains": ["x", "y"], "prior": 0.5, "complexity": 1},
            "complex": {"explains": ["x", "y"], "prior": 0.5, "complexity": 5},
        }}))
    assert c.answer == "simple"


def test_abductive_no_explanation():
    s = AbductiveStrategy()
    c = s.reason(ReasoningQuery(payload={
        "observations": ["z"],
        "hypotheses": {"h": {"explains": ["q"], "prior": 0.5, "complexity": 1}}}))
    assert c.answer is None
    assert c.confidence == 0.0


def test_abductive_empty_observations():
    s = AbductiveStrategy()
    c = s.reason(ReasoningQuery(payload={"observations": [], "hypotheses": {}}))
    assert c.answer is None


# -------------------- Analogical -------------------- #
def test_analogical_solar_to_atom():
    s = AnalogicalStrategy()
    c = s.reason(ReasoningQuery(payload={
        "source_relations": [("sun", "attracts", "planet"), ("planet", "orbits", "sun")],
        "mapping": {"sun": "nucleus", "planet": "electron"}}))
    assert ("nucleus", "attracts", "electron") in c.answer
    assert ("electron", "orbits", "nucleus") in c.answer


def test_analogical_specific_transfer():
    s = AnalogicalStrategy()
    c = s.reason(ReasoningQuery(payload={
        "source_relations": [("a", "rel", "b"), ("b", "other", "a")],
        "mapping": {"a": "x", "b": "y"},
        "transfer": ("a", "rel", "b")}))
    assert c.answer == [("x", "rel", "y")]


def test_analogical_partial_mapping_lower_confidence():
    s = AnalogicalStrategy()
    full = s.reason(ReasoningQuery(payload={
        "source_relations": [("a", "r", "b")], "mapping": {"a": "x", "b": "y"}}))
    partial = s.reason(ReasoningQuery(payload={
        "source_relations": [("a", "r", "b"), ("c", "r", "d")],
        "mapping": {"a": "x", "b": "y"}}))  # c,d unmapped
    assert partial.confidence < full.confidence


def test_analogical_applicability():
    s = AnalogicalStrategy()
    assert s.applicability(ReasoningQuery(payload={
        "source_relations": [], "mapping": {}})) > 0
    assert s.applicability(ReasoningQuery(payload={})) == 0.0


# -------------------- Neuro-symbolic -------------------- #
def test_neuro_symbolic_gate_then_rank():
    s = NeuroSymbolicStrategy()
    c = s.reason(ReasoningQuery(payload={
        "candidates": [3, 4, 6, 7, 8, 9],
        "constraint": lambda x: x % 2 == 0,
        "score": lambda x: x / 10.0}))
    assert c.answer == 8  # largest even


def test_neuro_symbolic_no_survivor():
    s = NeuroSymbolicStrategy()
    c = s.reason(ReasoningQuery(payload={
        "candidates": [1, 3, 5],
        "constraint": lambda x: x % 2 == 0,
        "score": lambda x: x}))
    assert c.answer is None
    assert c.confidence == 0.0


def test_neuro_symbolic_verifiable_overrides_high_score():
    s = NeuroSymbolicStrategy()
    # 9 has the highest score but fails the constraint -> excluded
    c = s.reason(ReasoningQuery(payload={
        "candidates": [8, 9],
        "constraint": lambda x: x % 2 == 0,
        "score": lambda x: x}))
    assert c.answer == 8


def test_neuro_symbolic_applicability():
    s = NeuroSymbolicStrategy()
    assert s.applicability(ReasoningQuery(payload={
        "candidates": [], "constraint": lambda x: True, "score": lambda x: 0})) > 0
    assert s.applicability(ReasoningQuery(payload={})) == 0.0


# -------------------- integration -------------------- #
def test_default_strategies_count():
    assert len(default_strategies()) == 5


def test_strategies_only_fire_on_their_input():
    reasoner = Reasoner(default_strategies())
    # bayesian-shaped payload -> only bayesian applies
    res = reasoner.reason(ReasoningQuery(payload={
        "hypotheses": {"a": 0.5, "b": 0.5}, "evidence": [{"a": 0.9, "b": 0.1}]}),
        mode=ReasoningMode.CONSENSUS)
    assert res.conclusion.answer == "a"
    assert len(res.consensus.conclusions) == 1  # only one strategy was applicable


def test_strategies_selector_routes_to_causal():
    reasoner = Reasoner(default_strategies())
    chosen = reasoner.selector.select(ReasoningQuery(payload={
        "graph": [("a", "b", 1.0)], "causes_of": "b"}))
    assert chosen.name == "causal"
