"""Tests for nyxara.mind.dual_process."""

from __future__ import annotations

import pytest

from nyxara.mind.dual_process import (
    Arbitrator,
    DualProcess,
    FastResponse,
    Outcome,
    ProcessType,
    System1,
    System2,
)
from nyxara.mind.faculties import (
    CallableFaculty,
    FacultyRegistry,
    FacultySelector,
    Task,
    TaskType,
)
from nyxara.mind.proposal import Proposal, ProposalKind


def _selector():
    math_fac = CallableFaculty("math", [TaskType.ARITHMETIC],
                               lambda t: (sum(t.payload), 1.0),
                               verifiable=True, reliability=1.0, cost=0.2)
    generalist = CallableFaculty("gen", [TaskType.GENERATION, TaskType.CLASSIFICATION],
                                 lambda t: ("answer", 0.7), verifiable=False)
    reg = FacultyRegistry()
    reg.register(math_fac)
    reg.register(generalist)
    return FacultySelector(reg)


def _dp(gut=None):
    return DualProcess(System1(gut), System2(_selector()))


# -------------------- System1 -------------------- #
def test_system1_default_low_confidence():
    s1 = System1()
    r = s1.respond(Task(TaskType.GENERATION, "x", payload="hello"))
    assert r.confidence == 0.3
    assert r.answer == "hello"


def test_system1_custom_intuition():
    s1 = System1(lambda t: ("fast", 0.95))
    r = s1.respond(Task(TaskType.GENERATION, "x"))
    assert r.answer == "fast" and r.confidence == 0.95


# -------------------- System2 -------------------- #
def test_system2_requires_engine():
    with pytest.raises(ValueError):
        System2()


def test_system2_dispatches_via_selector():
    s2 = System2(_selector())
    prop = s2.respond(Task(TaskType.ARITHMETIC, payload=[2, 3]))
    assert prop.content == 5


def test_system2_deliberate_callable():
    s2 = System2(deliberate=lambda t: Proposal(ProposalKind.ANSWER, "deliberated", confidence=0.9))
    prop = s2.respond(Task(TaskType.GENERATION, "x"))
    assert prop.content == "deliberated"


def test_system2_prefer_verifiable_falls_back():
    # generation has no verifiable faculty; prefer_verifiable should not crash
    s2 = System2(_selector())
    prop = s2.respond(Task(TaskType.GENERATION, "x"), prefer_verifiable=True)
    assert prop.source_faculty == "gen"


# -------------------- Arbitrator -------------------- #
def test_arbitrator_high_stakes_deliberates():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.CLASSIFICATION, features={"stakes": 0.9}),
                   FastResponse("x", 0.95))
    assert d.process is ProcessType.SYSTEM_2
    assert d.prefer_verifiable


def test_arbitrator_confident_low_stakes_intuition():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.GENERATION, features={"stakes": 0.2}),
                   FastResponse("x", 0.95))
    assert d.process is ProcessType.SYSTEM_1


def test_arbitrator_low_confidence_reflects():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.GENERATION, features={"stakes": 0.3}),
                   FastResponse("x", 0.3))
    assert d.process is ProcessType.SYSTEM_2
    assert "confidence" in d.reason


def test_arbitrator_emergency_uses_instinct():
    arb = Arbitrator()
    # high stakes but extreme time pressure -> System 1
    d = arb.decide(Task(TaskType.CLASSIFICATION, features={"stakes": 0.9}),
                   FastResponse("x", 0.5), time_budget=0.05)
    assert d.process is ProcessType.SYSTEM_1
    assert "emergency" in d.reason


def test_arbitrator_low_energy_conserves():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.GENERATION, features={"stakes": 0.3}),
                   FastResponse("x", 0.5), energy=0.1)
    assert d.process is ProcessType.SYSTEM_1
    assert "energy" in d.reason


def test_arbitrator_requires_verifiable_raises_stakes():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.ARITHMETIC, requires_verifiable=True),
                   FastResponse("x", 0.95))
    assert d.process is ProcessType.SYSTEM_2
    assert d.prefer_verifiable


def test_arbitrator_explicit_stakes_override():
    arb = Arbitrator()
    d = arb.decide(Task(TaskType.GENERATION), FastResponse("x", 0.95), stakes=0.95)
    assert d.process is ProcessType.SYSTEM_2


def test_arbitrator_pressure_threshold():
    arb = Arbitrator(deliberate_threshold=0.4)
    # difficult + novel pushes pressure up
    d = arb.decide(Task(TaskType.REASONING, features={"difficulty": 0.9, "novelty": 0.9,
                                                       "stakes": 0.3}),
                   FastResponse("x", 0.7))
    assert d.process is ProcessType.SYSTEM_2


# -------------------- DualProcess facade -------------------- #
def test_think_system1_path():
    dp = _dp(gut=lambda t: ("intuitive", 0.95))
    o = dp.think(Task(TaskType.GENERATION, "hi", features={"stakes": 0.2}))
    assert o.process is ProcessType.SYSTEM_1
    assert o.proposal.content == "intuitive"
    assert o.proposal.source_faculty == "system_1"
    assert not o.reflected


def test_think_system2_path_reflects():
    dp = _dp(gut=lambda t: ("rough guess", 0.3))
    o = dp.think(Task(TaskType.ARITHMETIC, payload=[2, 3, 4], features={"stakes": 0.3}))
    assert o.process is ProcessType.SYSTEM_2
    assert o.proposal.content == 9  # exact via math faculty
    assert o.reflected
    assert o.proposal.metadata.get("deliberated")


def test_think_high_stakes_deliberates():
    dp = _dp(gut=lambda t: ("confident", 0.99))
    o = dp.think(Task(TaskType.CLASSIFICATION, "fraud?", features={"stakes": 0.9}))
    assert o.process is ProcessType.SYSTEM_2


def test_think_emergency_instinct():
    dp = _dp(gut=lambda t: ("snap", 0.5))
    o = dp.think(Task(TaskType.CLASSIFICATION, features={"stakes": 0.9}), time_budget=0.05)
    assert o.process is ProcessType.SYSTEM_1
    assert o.proposal.content == "snap"


def test_outcome_always_proposal():
    dp = _dp(gut=lambda t: ("x", 0.95))
    o = dp.think(Task(TaskType.GENERATION, features={"stakes": 0.1}))
    assert isinstance(o.proposal, Proposal)


def test_outcome_to_dict():
    dp = _dp(gut=lambda t: ("x", 0.95))
    o = dp.think(Task(TaskType.GENERATION, features={"stakes": 0.1}))
    d = o.to_dict()
    assert "process" in d and "decision" in d and "proposal" in d
