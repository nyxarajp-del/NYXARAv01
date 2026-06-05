"""Tests for nyxara.planning.intent."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import AffectSystem
from nyxara.planning.goals import Goal, GoalSystem
from nyxara.planning.intent import DriveGoal, IntentSystem


def _depleted_intent(ticks: int = 200):
    affect = AffectSystem()
    intent = IntentSystem(affect)
    for _ in range(ticks):
        affect.tick(1.0)
    return intent, affect


# -------------------- candidate generation -------------------- #
def test_no_candidates_when_satisfied():
    affect = AffectSystem()
    for d in affect.drives.values():
        d.level = 1.0
    intent = IntentSystem(affect)
    assert intent.generate_candidates() == []


def test_candidates_for_unmet_drives():
    intent, _ = _depleted_intent()
    cands = intent.generate_candidates()
    assert cands
    assert all(isinstance(c, DriveGoal) for c in cands)


def test_candidate_goal_from_template():
    affect = AffectSystem()
    affect.drives["safety"].level = 0.2
    intent = IntentSystem(affect)
    cands = intent.generate_candidates()
    safety = [c for c in cands if c.drive == "safety"]
    assert safety
    assert "owner_safety" in safety[0].goal.vector


def test_priority_scales_with_deficit():
    affect = AffectSystem()
    affect.drives["competence"].level = 0.1  # big deficit
    big = IntentSystem(affect).generate_candidates()
    affect2 = AffectSystem()
    affect2.drives["competence"].level = 0.65  # small deficit
    small = IntentSystem(affect2).generate_candidates()
    big_c = next(c for c in big if c.drive == "competence")
    small_c = next(c for c in small if c.drive == "competence")
    assert big_c.goal.priority > small_c.goal.priority


def test_pressure_threshold_filters():
    affect = AffectSystem()
    for d in affect.drives.values():
        d.level = d.setpoint  # zero pressure
    intent = IntentSystem(affect, pressure_threshold=0.1)
    assert intent.generate_candidates() == []


# -------------------- active inference selection -------------------- #
def test_owner_drives_win_selection():
    intent, _ = _depleted_intent()
    top = intent.select(k=1)
    assert top
    assert top[0].drive in ("owner_connection", "safety")


def test_threat_makes_safety_top():
    intent, affect = _depleted_intent()
    affect.note_threat(0.95)
    top = intent.select(k=1)[0]
    assert top.drive == "safety"


def test_select_sorted_by_efe():
    intent, _ = _depleted_intent()
    sel = intent.select(k=99)
    efes = [c.expected_free_energy for c in sel]
    assert efes == sorted(efes)


def test_select_filters_owner_aligned():
    # all default templates are owner-aligned; selection keeps them
    intent, _ = _depleted_intent()
    sel = intent.select(k=99)
    assert all(intent.goals.is_owner_aligned(c.goal) for c in sel)


def test_value_is_negative_efe():
    intent, _ = _depleted_intent()
    c = intent.generate_candidates()[0]
    assert c.value == pytest.approx(-c.expected_free_energy)


# -------------------- autonomous intent -------------------- #
def test_autonomous_intent_registers_goal():
    intent, _ = _depleted_intent()
    before = len(intent.goals)
    dg = intent.autonomous_intent()
    assert dg is not None
    assert len(intent.goals) == before + 1


def test_autonomous_intent_none_when_satisfied():
    affect = AffectSystem()
    for d in affect.drives.values():
        d.level = 1.0
    assert IntentSystem(affect).autonomous_intent() is None


def test_generate_registers_all():
    intent, _ = _depleted_intent()
    gens = intent.generate()
    assert len(intent.goals) == len(gens)
    assert gens


# -------------------- epistemic / curiosity -------------------- #
def test_novelty_has_high_epistemic():
    affect = AffectSystem()
    affect.drives["novelty"].level = 0.0
    intent = IntentSystem(affect)
    nov = [c for c in intent.generate_candidates() if c.drive == "novelty"]
    assert nov
    assert nov[0].epistemic >= 0.5


def test_safety_has_low_epistemic():
    affect = AffectSystem()
    affect.drives["safety"].level = 0.2
    intent = IntentSystem(affect)
    safety = next(c for c in intent.generate_candidates() if c.drive == "safety")
    assert safety.epistemic < 0.3


def test_high_epistemic_weight_boosts_exploration_value():
    affect = AffectSystem()
    affect.drives["novelty"].level = 0.0
    low = IntentSystem(affect, epistemic_weight=0.1)
    affect2 = AffectSystem()
    affect2.drives["novelty"].level = 0.0
    high = IntentSystem(affect2, epistemic_weight=2.0)
    lc = next(c for c in low.generate_candidates() if c.drive == "novelty")
    hc = next(c for c in high.generate_candidates() if c.drive == "novelty")
    assert hc.value > lc.value  # exploration valued more under high epistemic weight


# -------------------- owner alignment enforcement -------------------- #
def test_anti_owner_goal_rejected():
    intent, _ = _depleted_intent()
    bad = Goal("betray", {"autonomy": 1.0, "owner_benefit": -0.9})
    assert not intent.goals.is_owner_aligned(bad)


def test_custom_goal_system():
    gs = GoalSystem()
    affect = AffectSystem()
    affect.drives["safety"].level = 0.2
    intent = IntentSystem(affect, goal_system=gs)
    intent.autonomous_intent()
    assert len(gs) >= 1


# -------------------- status -------------------- #
def test_status():
    intent, _ = _depleted_intent()
    s = intent.status()
    assert "candidates" in s and "drives_pressing" in s
    assert s["intent"] is not None
