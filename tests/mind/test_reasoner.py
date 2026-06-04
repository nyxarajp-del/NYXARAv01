"""Tests for nyxara.mind.reasoner."""

from __future__ import annotations

import random

import pytest

from nyxara.mind.faculties import CallableFaculty, FacultyRegistry, FacultySelector, Task, TaskType
from nyxara.mind.proposal import Proposal, ProposalKind
from nyxara.mind.reasoner import (
    CallableStrategy,
    Conclusion,
    ConsensusEngine,
    FacultyReasoner,
    ModelBasedReasoner,
    Reasoner,
    ReasoningMode,
    ReasoningQuery,
    StrategySelector,
    _consensus_key,
)
from nyxara.mind.world_model import WorldModel


def _world():
    def step(x, a):
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (step(x, a),), reward=-abs(step(x, a)))
    return wm


# -------------------- consensus key -------------------- #
def test_consensus_key_float_rounds():
    assert _consensus_key(0.1 + 0.2) == _consensus_key(0.3)


def test_consensus_key_list():
    assert _consensus_key(["a", 1.0]) == ("a", 1.0)


# -------------------- CallableStrategy -------------------- #
def test_callable_strategy():
    s = CallableStrategy("s", lambda q: ("ans", 0.9, "because"))
    c = s.reason(ReasoningQuery())
    assert c.answer == "ans" and c.confidence == 0.9
    assert c.strategy == "s"


def test_callable_strategy_applicability():
    s = CallableStrategy("s", lambda q: ("x", 1.0, ""),
                         applicability_fn=lambda q: 0.3)
    assert s.applicability(ReasoningQuery()) == 0.3


# -------------------- ModelBasedReasoner -------------------- #
def test_model_based_picks_best_policy():
    wm = _world()
    mbr = ModelBasedReasoner(wm)
    q = ReasoningQuery(start=(5.0,), preference=(0.0,), horizon=6,
                       candidates=[["left"] * 6, ["right"] * 6, ["stay"] * 6])
    c = mbr.reason(q)
    assert c.answer == ["left"] * 6


def test_model_based_applicability():
    mbr = ModelBasedReasoner(_world())
    assert mbr.applicability(ReasoningQuery(start=(0.0,), candidates=[["left"]])) > 0
    assert mbr.applicability(ReasoningQuery()) == 0.0


def test_model_based_no_candidates():
    mbr = ModelBasedReasoner(_world())
    c = mbr.reason(ReasoningQuery(start=(0.0,), candidates=[]))
    assert c.answer is None
    assert c.confidence == 0.0


def test_model_based_evaluate_ranks():
    wm = _world()
    mbr = ModelBasedReasoner(wm)
    q = ReasoningQuery(start=(5.0,), preference=(0.0,), horizon=6,
                       candidates=[["right"] * 6, ["left"] * 6])
    scores = mbr.evaluate(q)
    assert scores[0].policy == ["left"] * 6  # best first
    assert scores[0].score > scores[-1].score


def test_model_based_custom_reward():
    wm = _world()
    mbr = ModelBasedReasoner(wm)
    q = ReasoningQuery(start=(0.0,), horizon=3,
                       candidates=[["right"] * 3, ["left"] * 3],
                       reward_fn=lambda s, a, ns: 1.0 if a == "right" else 0.0)
    c = mbr.reason(q)
    assert c.answer == ["right"] * 3


# -------------------- FacultyReasoner -------------------- #
def test_faculty_reasoner():
    math_fac = CallableFaculty("math", [TaskType.ARITHMETIC],
                               lambda t: (sum(t.payload), 1.0), verifiable=True)
    reg = FacultyRegistry()
    reg.register(math_fac)
    fr = FacultyReasoner(FacultySelector(reg))
    q = ReasoningQuery(task=Task(TaskType.ARITHMETIC, payload=[2, 3, 4]))
    assert fr.applicability(q) > 0
    c = fr.reason(q)
    assert c.answer == 9


def test_faculty_reasoner_no_faculty():
    reg = FacultyRegistry()
    fr = FacultyReasoner(FacultySelector(reg))
    c = fr.reason(ReasoningQuery(task=Task(TaskType.ARITHMETIC, payload=[1])))
    assert c.answer is None


# -------------------- StrategySelector -------------------- #
def test_strategy_selector_picks_most_applicable():
    a = CallableStrategy("a", lambda q: ("x", 1.0, ""), applicability_fn=lambda q: 0.3)
    b = CallableStrategy("b", lambda q: ("y", 1.0, ""), applicability_fn=lambda q: 0.9)
    sel = StrategySelector([a, b])
    assert sel.select(ReasoningQuery()).name == "b"


def test_strategy_selector_filters_inapplicable():
    a = CallableStrategy("a", lambda q: ("x", 1.0, ""), applicability_fn=lambda q: 0.0)
    sel = StrategySelector([a])
    assert sel.select(ReasoningQuery()) is None
    assert sel.applicable(ReasoningQuery()) == []


# -------------------- ConsensusEngine -------------------- #
def test_consensus_weighted_majority():
    eng = ConsensusEngine()
    res = eng.aggregate([
        Conclusion("yes", 0.8, strategy="a"),
        Conclusion("yes", 0.7, strategy="b"),
        Conclusion("no", 0.9, strategy="c"),
    ])
    assert res.answer == "yes"  # 1.5 > 0.9
    assert len(res.dissent) == 1
    assert res.dissent[0].answer == "no"


def test_consensus_agreement_and_divided():
    eng = ConsensusEngine()
    res = eng.aggregate([
        Conclusion("A", 0.6, strategy="x"),
        Conclusion("B", 0.6, strategy="y"),
    ])
    assert res.agreement == pytest.approx(0.5)
    assert res.divided


def test_consensus_unanimous_high_confidence():
    eng = ConsensusEngine()
    res = eng.aggregate([
        Conclusion("yes", 0.9, strategy="a"),
        Conclusion("yes", 0.8, strategy="b"),
    ])
    assert res.agreement == 1.0
    assert not res.divided
    assert res.confidence > 0.8


def test_consensus_ignores_null_and_zero():
    eng = ConsensusEngine()
    res = eng.aggregate([
        Conclusion(None, 0.0, strategy="a"),
        Conclusion("x", 0.0, strategy="b"),
        Conclusion("y", 0.7, strategy="c"),
    ])
    assert res.answer == "y"


def test_consensus_all_invalid():
    eng = ConsensusEngine()
    res = eng.aggregate([Conclusion(None, 0.0, strategy="a")])
    assert res.answer is None
    assert res.confidence == 0.0


# -------------------- Reasoner facade -------------------- #
def test_reasoner_best_mode():
    wm = _world()
    r = Reasoner([ModelBasedReasoner(wm)])
    q = ReasoningQuery(start=(5.0,), preference=(0.0,), horizon=6,
                       candidates=[["left"] * 6, ["right"] * 6])
    result = r.reason(q, mode=ReasoningMode.BEST)
    assert result.mode is ReasoningMode.BEST
    assert result.conclusion.answer == ["left"] * 6
    assert isinstance(result.proposal, Proposal)


def test_reasoner_no_applicable_strategy():
    r = Reasoner([ModelBasedReasoner(_world())])
    result = r.reason(ReasoningQuery(description="unmappable"))
    assert result.conclusion.answer is None


def test_reasoner_consensus_mode():
    panel = Reasoner([
        CallableStrategy("a", lambda q: ("yes", 0.8, ""), applicability_fn=lambda q: 1.0),
        CallableStrategy("b", lambda q: ("yes", 0.7, ""), applicability_fn=lambda q: 1.0),
        CallableStrategy("c", lambda q: ("no", 0.9, ""), applicability_fn=lambda q: 1.0),
    ])
    result = panel.reason(ReasoningQuery(), mode=ReasoningMode.CONSENSUS)
    assert result.mode is ReasoningMode.CONSENSUS
    assert result.conclusion.answer == "yes"
    assert result.consensus is not None


def test_reasoner_register():
    r = Reasoner()
    r.register(CallableStrategy("s", lambda q: ("x", 1.0, ""), applicability_fn=lambda q: 1.0))
    assert r.selector.select(ReasoningQuery()) is not None


def test_reasoner_output_is_proposal():
    r = Reasoner([CallableStrategy("s", lambda q: ("ans", 0.9, "r"),
                                   applicability_fn=lambda q: 1.0)])
    result = r.reason(ReasoningQuery(kind=ProposalKind.ANSWER))
    assert isinstance(result.proposal, Proposal)
    assert result.proposal.source_faculty == "reasoner"
    assert result.proposal.metadata["strategy"] == "s"


def test_result_to_dict():
    r = Reasoner([CallableStrategy("s", lambda q: ("ans", 0.9, "r"),
                                   applicability_fn=lambda q: 1.0)])
    d = r.reason(ReasoningQuery()).to_dict()
    assert "conclusion" in d and "proposal" in d and d["mode"] == "best"
