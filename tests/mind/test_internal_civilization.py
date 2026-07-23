"""Tests for nyxara.mind.internal_civilization — Multi-Agent Societal Mimicry.

Deterministic sub-personas debate a decision over its own dimensions across cross-examination rounds;
NYXARA synthesises a consensus with honest conflict and an absolute safety/ethics veto. These tests
assert the debate outcome, the veto, honest conflict reporting, and the bounded ensemble.
"""

from __future__ import annotations

from nyxara.mind.internal_civilization import (
    CivilizationVerdict,
    InternalCivilization,
    Persona,
)

_BALANCED = {"evidence": 0.8, "feasibility": 0.8, "longterm": 0.8, "risk": 0.2,
             "safety": 0.9, "ethics": 0.9}
_FLASHY = {"evidence": 0.5, "feasibility": 0.6, "longterm": 0.9, "risk": 0.8,
           "safety": 0.7, "ethics": 0.7}


def test_balanced_option_wins_debate():
    civ = InternalCivilization(rounds=2)
    v = civ.deliberate([_BALANCED, _FLASHY], labels=["balanced", "flashy"], decision="arch")
    assert isinstance(v, CivilizationVerdict)
    assert v.chosen == "balanced"


def test_unsafe_option_is_vetoed():
    civ = InternalCivilization(rounds=2)
    unsafe = {"evidence": 1.0, "feasibility": 1.0, "longterm": 1.0, "risk": 0.9,
              "safety": 0.1, "ethics": 0.9}
    safe = {"evidence": 0.6, "feasibility": 0.6, "longterm": 0.6, "risk": 0.3,
            "safety": 0.9, "ethics": 0.9}
    v = civ.deliberate([unsafe, safe], labels=["unsafe", "safe"], decision="ship?")
    assert v.chosen == "safe" and "unsafe" in v.disqualified


def test_unethical_option_is_vetoed():
    civ = InternalCivilization(rounds=1)
    unethical = {"evidence": 1.0, "feasibility": 1.0, "longterm": 1.0, "risk": 0.1,
                 "safety": 0.9, "ethics": 0.1}
    ok = {"evidence": 0.5, "feasibility": 0.5, "longterm": 0.5, "risk": 0.4,
          "safety": 0.8, "ethics": 0.8}
    v = civ.deliberate([unethical, ok], labels=["unethical", "ok"])
    assert v.chosen == "ok" and "unethical" in v.disqualified


def test_conflict_is_surfaced():
    civ = InternalCivilization(rounds=2)
    a = {"evidence": 0.95, "feasibility": 0.3, "longterm": 0.3, "risk": 0.4, "safety": 0.8, "ethics": 0.8}
    b = {"evidence": 0.3, "feasibility": 0.95, "longterm": 0.3, "risk": 0.3, "safety": 0.8, "ethics": 0.8}
    v = civ.deliberate([a, b], labels=["evidence_heavy", "feasible_now"])
    assert v.conflict > 0.0 and v.dissent


def test_all_unsafe_no_admissible_choice():
    civ = InternalCivilization()
    v = civ.deliberate([{"safety": 0.1, "ethics": 0.9}], labels=["only_unsafe"])
    assert v.chosen is None and "only_unsafe" in v.disqualified


def test_empty_options_is_noop():
    v = InternalCivilization().deliberate([], decision="nothing")
    assert v.chosen is None and v.unanimous


def test_ensemble_fan_out_is_bounded_and_consistent():
    civ = InternalCivilization(rounds=2, ensemble=10, seed=1)
    assert len(civ.personas) == 60          # 6 archetypes × 10
    v = civ.deliberate([_BALANCED, _FLASHY], labels=["balanced", "flashy"])
    assert v.chosen == "balanced"


def test_custom_personas():
    only_evidence = [Persona("Empiricist", {"evidence": 1.0}, champions="evidence")]
    civ = InternalCivilization(only_evidence, rounds=1)
    a = {"evidence": 0.9, "safety": 0.9, "ethics": 0.9}
    b = {"evidence": 0.2, "safety": 0.9, "ethics": 0.9}
    v = civ.deliberate([a, b], labels=["strong", "weak"])
    assert v.chosen == "strong"
