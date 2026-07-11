"""Fast, model-free test for the *honest* deepen_reasoning self-tuning (Problem #1).

The self-improvement engine's ``deepen_reasoning`` directive used to only bump an integer that
controlled text-polish passes. It must now deepen genuine *search*: switch the always-max deep
reasoning on and widen its self-consistency width (more lines of reasoning searched), not merely
run more critique passes over one answer.
"""

from __future__ import annotations

from nyxara.growth.recursive_improvement import RecursiveSelfImprovement
from nyxara.kernel.config import NyxaraSettings, Profile


def _engine():
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.self_improvement.allow_tuning = True
    s.llm.deep_reasoning.enabled = False
    s.llm.deep_reasoning.samples = 3
    eng = RecursiveSelfImprovement(settings=s)
    eng._effort = {}          # short-circuit _plan_effort (no index/model work)
    eng._bench = {"accuracy": 0.6}
    return eng, s


def test_deepen_reasoning_widens_real_search_not_just_polish():
    eng, s = _engine()
    eng._directive = {"action": "deepen_reasoning"}
    res = eng._maybe_tune(None, enact=True)
    assert res is not None
    # it deepened genuine search: switched on the ladder and widened self-consistency
    assert s.llm.deep_reasoning.enabled is True
    assert s.llm.deep_reasoning.samples > 3
    assert "deep_reasoning_samples" in res
    # and still bumped the refinement-rung budget (rung 4 of the ladder)
    assert "recursive_improvement_iterations" in res


def test_escalate_on_plateau_also_deepens_search():
    eng, s = _engine()
    eng._bench = {"accuracy": 0.95}  # capability fine, but the index plateaued
    eng._directive = {"action": "hold", "escalate": True}
    res = eng._maybe_tune(None, enact=True)
    assert res is not None
    assert s.llm.deep_reasoning.enabled is True
    assert s.llm.deep_reasoning.samples > 3


def test_no_directive_no_change():
    eng, s = _engine()
    eng._bench = {"accuracy": 0.95}   # strong capability, no escalate, no deepen
    eng._directive = {"action": "hold"}
    assert eng._maybe_tune(None, enact=True) is None
    assert s.llm.deep_reasoning.enabled is False  # untouched
