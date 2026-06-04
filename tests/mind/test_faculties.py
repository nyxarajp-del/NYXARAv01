"""Tests for nyxara.mind.faculties."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.faculties import (
    CallableFaculty,
    Faculty,
    FacultyRegistry,
    FacultySelector,
    LLMFaculty,
    NoFacultyError,
    Task,
    TaskType,
)
from nyxara.mind.llm import LLM
from nyxara.mind.proposal import Proposal, ProposalKind


def _math():
    return CallableFaculty("math", [TaskType.ARITHMETIC], lambda t: (sum(t.payload), 1.0),
                           verifiable=True, reliability=1.0, cost=0.2)


def _llm_fac():
    return LLMFaculty(LLM(settings=NyxaraSettings.for_profile(Profile.TEST)))


def _registry(*faculties):
    reg = FacultyRegistry()
    for f in faculties:
        reg.register(f)
    return reg


# -------------------- Task -------------------- #
def test_task_to_dict():
    t = Task(TaskType.ARITHMETIC, "add", requires_verifiable=True)
    d = t.to_dict()
    assert d["type"] == "arithmetic"
    assert d["requires_verifiable"] is True


# -------------------- Faculty base -------------------- #
def test_default_suitability():
    class F(Faculty):
        name = "f"
        handles = frozenset({TaskType.LOGIC})

    f = F()
    assert f.suitability(Task(TaskType.LOGIC)) == 1.0
    assert f.suitability(Task(TaskType.GENERATION)) == 0.0


def test_callable_faculty_handle_returns_proposal():
    f = _math()
    prop = f.handle(Task(TaskType.ARITHMETIC, payload=[2, 3, 5]))
    assert isinstance(prop, Proposal)
    assert prop.content == 10
    assert prop.source_faculty == "math"
    assert prop.confidence == 1.0


def test_callable_faculty_verifiable_flag():
    assert _math().verifiable is True


def test_callable_faculty_custom_suitability():
    f = CallableFaculty("x", [TaskType.LOGIC], lambda t: ("v", 0.5),
                        suitability_fn=lambda t: 0.42)
    assert f.suitability(Task(TaskType.GENERATION)) == 0.42


def test_callable_faculty_availability():
    state = {"up": False}
    f = CallableFaculty("x", [TaskType.LOGIC], lambda t: ("v", 0.5),
                        available_fn=lambda: state["up"])
    assert not f.available()
    state["up"] = True
    assert f.available()


# -------------------- LLM faculty -------------------- #
def test_llm_faculty_handles_generation():
    f = _llm_fac()
    prop = f.handle(Task(TaskType.GENERATION, "say hi"))
    assert isinstance(prop, Proposal)
    assert prop.source_faculty == "llm"
    assert prop.provenance.source.value == "llm_inference"


def test_llm_faculty_is_generalist():
    f = _llm_fac()
    # weakly applicable even to tasks not in `handles`
    assert f.suitability(Task(TaskType.ARITHMETIC)) > 0
    assert f.suitability(Task(TaskType.GENERATION)) == 0.9


def test_llm_faculty_not_verifiable():
    assert _llm_fac().verifiable is False


# -------------------- Registry -------------------- #
def test_registry_register_get():
    reg = _registry(_math())
    assert reg.get("math") is not None
    assert len(reg) == 1


def test_registry_unregister():
    reg = _registry(_math())
    assert reg.unregister("math") is True
    assert reg.unregister("nope") is False


def test_registry_by_task():
    reg = _registry(_math(), _llm_fac())
    arith = reg.by_task(TaskType.ARITHMETIC)
    assert any(f.name == "math" for f in arith)


# -------------------- Selector: verifiable > probabilistic -------------------- #
def test_selector_prefers_verifiable_for_arithmetic():
    reg = _registry(_math(), _llm_fac())
    sel = FacultySelector(reg)
    t = Task(TaskType.ARITHMETIC, payload=[1, 2])
    assert sel.select(t).name == "math"


def test_selector_picks_llm_for_generation():
    reg = _registry(_math(), _llm_fac())
    sel = FacultySelector(reg)
    assert sel.select(Task(TaskType.GENERATION, "poem")).name == "llm"


def test_selector_requires_verifiable_filters_guessers():
    reg = _registry(_math(), _llm_fac())
    sel = FacultySelector(reg)
    t = Task(TaskType.ARITHMETIC, payload=[1], requires_verifiable=True)
    ranked = sel.rank(t)
    assert all(s.verifiable for s in ranked)
    assert "llm" not in [s.faculty.name for s in ranked]


def test_selector_dispatch_produces_proposal():
    reg = _registry(_math(), _llm_fac())
    sel = FacultySelector(reg)
    prop = sel.dispatch(Task(TaskType.ARITHMETIC, payload=[4, 6]))
    assert isinstance(prop, Proposal)
    assert prop.content == 10


def test_selector_no_faculty_raises():
    reg = _registry(_math())  # only arithmetic
    sel = FacultySelector(reg)
    with pytest.raises(NoFacultyError):
        sel.dispatch(Task(TaskType.PREDICTION, "forecast"))


def test_selector_skips_unavailable():
    down = CallableFaculty("down", [TaskType.LOGIC], lambda t: ("x", 1.0),
                           verifiable=True, available_fn=lambda: False)
    up = CallableFaculty("up", [TaskType.LOGIC], lambda t: ("y", 0.9),
                         verifiable=False)
    reg = _registry(down, up)
    sel = FacultySelector(reg)
    prop = sel.dispatch(Task(TaskType.LOGIC))
    assert prop.source_faculty == "up"


def test_selector_explain_ranking():
    reg = _registry(_math(), _llm_fac())
    sel = FacultySelector(reg)
    exp = sel.explain(Task(TaskType.ARITHMETIC, payload=[1]))
    assert exp["ranking"][0]["faculty"] == "math"
    assert "score" in exp["ranking"][0]


def test_selector_empty_registry_returns_none():
    sel = FacultySelector(FacultyRegistry())
    assert sel.select(Task(TaskType.GENERATION)) is None


def test_two_verifiable_tie_break_prefers_cheaper():
    cheap = CallableFaculty("cheap", [TaskType.LOGIC], lambda t: ("c", 1.0),
                            verifiable=True, reliability=1.0, cost=0.5)
    pricey = CallableFaculty("pricey", [TaskType.LOGIC], lambda t: ("p", 1.0),
                             verifiable=True, reliability=1.0, cost=2.0)
    sel = FacultySelector(_registry(pricey, cheap))
    assert sel.select(Task(TaskType.LOGIC)).name == "cheap"
