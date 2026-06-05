"""Tests for nyxara.planning.scenario."""

from __future__ import annotations

import pytest

from nyxara.planning.scenario import (
    FailureMode,
    PreMortem,
    Scenario,
    ScenarioAnalysis,
    ScenarioPlanner,
    ScenarioSet,
    ScenarioType,
)


# -------------------- scenario generation -------------------- #
def test_generate_four_scenarios():
    ss = ScenarioPlanner().generate("plan")
    assert set(ss.scenarios) == {ScenarioType.BEST, ScenarioType.LIKELY,
                                 ScenarioType.WORST, ScenarioType.BLACK_SWAN}


def test_scenario_ordering():
    ss = ScenarioPlanner().generate("plan", base_value=0.6, upside=0.3, downside=0.7)
    s = ss.scenarios
    assert s[ScenarioType.BEST].value > s[ScenarioType.LIKELY].value
    assert s[ScenarioType.LIKELY].value > s[ScenarioType.WORST].value
    assert s[ScenarioType.BLACK_SWAN].value < s[ScenarioType.WORST].value


def test_probabilities_sum_near_one():
    ss = ScenarioPlanner().generate("plan")
    total = sum(s.probability for s in ss.scenarios.values())
    assert total == pytest.approx(1.0, abs=0.01)


def test_expected_value_between_worst_and_best():
    ss = ScenarioPlanner().generate("plan", base_value=0.5)
    assert ss.worst_case <= ss.expected_value <= ss.scenarios[ScenarioType.BEST].value


def test_weighted_value():
    s = Scenario(ScenarioType.LIKELY, value=0.5, probability=0.6)
    assert s.weighted_value == pytest.approx(0.3)


def test_robust_plan():
    ss = ScenarioPlanner().generate("safe", base_value=0.6, downside=0.4)
    assert ss.robust  # worst ~0.2 above floor -0.6


def test_fragile_plan_not_robust():
    ss = ScenarioPlanner().generate("risky", base_value=0.5, downside=1.3)
    assert not ss.robust


def test_catastrophic_risk():
    ss = ScenarioPlanner().generate("plan", black_swan_value=-1.0)
    assert ss.catastrophic_risk > 0


def test_scenario_set_to_dict():
    d = ScenarioPlanner().generate("plan").to_dict()
    assert "expected_value" in d and "scenarios" in d and "robust" in d


def test_custom_probabilities():
    probs = {ScenarioType.LIKELY: 0.5, ScenarioType.BEST: 0.3,
             ScenarioType.WORST: 0.15, ScenarioType.BLACK_SWAN: 0.05}
    ss = ScenarioPlanner().generate("plan", probabilities=probs)
    assert ss.scenarios[ScenarioType.BEST].probability == 0.3


# -------------------- pre-mortem -------------------- #
def test_premortem_generic_modes():
    modes = PreMortem().analyze("a plan")
    assert len(modes) >= 6  # all generic categories
    assert all(isinstance(m, FailureMode) for m in modes)


def test_premortem_sorted_by_risk():
    modes = PreMortem().analyze("a plan")
    risks = [m.risk for m in modes]
    assert risks == sorted(risks, reverse=True)


def test_premortem_considers_owner_misread():
    modes = PreMortem().analyze("a plan")
    assert any("Master" in m.cause for m in modes)


def test_premortem_considers_adversary():
    modes = PreMortem().analyze("a plan")
    assert any("adversary" in m.cause for m in modes)


def test_premortem_custom_factor():
    modes = PreMortem().analyze("deploy", factors=[
        {"cause": "config typo", "likelihood": 0.5, "severity": 0.9,
         "mitigation": "dry-run"}])
    assert any("config typo" in m.cause for m in modes)
    typo = next(m for m in modes if "config typo" in m.cause)
    assert typo.mitigation == "dry-run"


def test_premortem_factor_defaults():
    modes = PreMortem().analyze("x", factors=[{"cause": "bare risk"}], include_generic=False)
    assert len(modes) == 1
    assert modes[0].likelihood == 0.3  # default
    assert modes[0].mitigation  # has a default mitigation


def test_premortem_no_generic():
    modes = PreMortem().analyze("x", factors=[{"cause": "only this"}], include_generic=False)
    assert len(modes) == 1


def test_top_risks():
    top = PreMortem().top_risks("plan", n=2)
    assert len(top) == 2
    assert top[0].risk >= top[1].risk


def test_failure_mode_risk():
    fm = FailureMode("x", likelihood=0.5, severity=0.8, mitigation="y")
    assert fm.risk == pytest.approx(0.4)


def test_failure_mode_to_dict():
    d = FailureMode("x", 0.5, 0.8, "fix").to_dict()
    assert d["risk"] == 0.4 and d["mitigation"] == "fix"


# -------------------- facade -------------------- #
def test_analysis_returns_scenarios_and_premortem():
    a = ScenarioAnalysis().analyze("plan", base_value=0.6)
    assert "scenarios" in a and "premortem" in a and "recommendation" in a


def test_analysis_recommends_proceed_when_favourable():
    a = ScenarioAnalysis().analyze("good plan", base_value=0.7, upside=0.3, downside=0.3)
    assert "proceed" in a["recommendation"].lower() or "defens" in a["recommendation"].lower()


def test_analysis_warns_when_fragile():
    a = ScenarioAnalysis().analyze("fragile", base_value=0.4, downside=1.3)
    assert not a["robust"]
    assert "mitigate" in a["recommendation"].lower() or "not proceed" in a["recommendation"].lower()


def test_analysis_factors_flow_to_premortem():
    a = ScenarioAnalysis().analyze("x", factors=[
        {"cause": "special risk", "likelihood": 0.6, "severity": 0.8, "mitigation": "m"}])
    causes = [m["cause"] for m in a["premortem"]]
    assert any("special risk" in c for c in causes)
