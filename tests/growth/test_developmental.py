"""Tests for nyxara.growth.developmental."""

from __future__ import annotations

import pytest

from nyxara.growth.developmental import (DevelopmentalCurriculum, ReadinessReport, Stage,
                                         StageStatus, build_default_curriculum)
from nyxara.kernel.errors import AuthorizationError, InvariantViolation


def _basic():
    c = DevelopmentalCurriculum()
    c.add_stage("a", level=0, unlocks={"cap_a"})
    c.add_stage("b", level=1, prerequisites={"a"}, unlocks={"cap_b"},
                requirements={"skill": 0.5})
    c.add_stage("c", level=2, prerequisites={"b"}, unlocks={"cap_c"}, safety_critical=True)
    return c


# -------------------- definition -------------------- #
def test_add_stage():
    c = DevelopmentalCurriculum()
    s = c.add_stage("a", level=0, unlocks={"x"})
    assert isinstance(s, Stage) and c.status("a") is StageStatus.LOCKED


def test_undefined_prerequisite_raises():
    c = DevelopmentalCurriculum()
    with pytest.raises(InvariantViolation):
        c.add_stage("b", level=1, prerequisites={"missing"})


def test_get_unknown():
    assert DevelopmentalCurriculum().get("nope") is None


def test_stage_to_dict():
    c = _basic()
    d = c.get("b").to_dict()
    assert d["name"] == "b" and d["prerequisites"] == ["a"]


# -------------------- competence -------------------- #
def test_set_get_competence():
    c = _basic()
    c.set_competence("skill", 0.7)
    assert c.competence("skill") == 0.7


def test_competence_clamped():
    c = _basic()
    c.set_competence("skill", 5.0)
    assert c.competence("skill") == 1.0


def test_competence_default_zero():
    assert _basic().competence("unknown") == 0.0


# -------------------- assessment -------------------- #
def test_assess_no_prereq_no_req_is_ready():
    c = _basic()
    r = c.assess("a")
    assert r.ready and r.status is StageStatus.READY


def test_assess_locked_without_prereq():
    c = _basic()
    r = c.assess("b")
    assert r.status is StageStatus.LOCKED
    assert "a" in r.missing_prereqs


def test_assess_eligible_without_competence():
    c = _basic()
    c.advance("a")
    r = c.assess("b")
    assert r.status is StageStatus.ELIGIBLE
    assert "skill" in r.unmet_requirements


def test_assess_ready_when_competent():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    assert c.assess("b").status is StageStatus.READY


def test_assess_needs_owner_for_safety():
    c = _basic()
    assert c.assess("c").needs_owner


def test_report_to_dict():
    c = _basic()
    r = c.assess("a")
    assert isinstance(r, ReadinessReport) and "stage" in r.to_dict()


# -------------------- advancement -------------------- #
def test_advance_simple_stage():
    c = _basic()
    assert c.advance("a")
    assert c.status("a") is StageStatus.ACTIVE
    assert c.is_unlocked("cap_a")


def test_advance_already_active_returns_false():
    c = _basic()
    c.advance("a")
    assert not c.advance("a")


def test_advance_blocked_by_prereq():
    c = _basic()
    with pytest.raises(InvariantViolation):
        c.advance("b")


def test_advance_blocked_by_competence():
    c = _basic()
    c.advance("a")
    with pytest.raises(InvariantViolation):
        c.advance("b")  # skill not met


def test_advance_with_competence():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    assert c.advance("b")


def test_no_skipping():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    # cannot jump to c without b
    with pytest.raises(InvariantViolation):
        c.advance("c", owner_approved=True)


def test_safety_critical_needs_owner():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    c.advance("b")
    with pytest.raises(AuthorizationError):
        c.advance("c", owner_approved=False)


def test_safety_critical_with_owner():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    c.advance("b")
    assert c.advance("c", owner_approved=True)


def test_advance_unknown_stage():
    with pytest.raises(InvariantViolation):
        _basic().advance("nope")


# -------------------- capability boundary -------------------- #
def test_unlocked_capabilities_union():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    c.advance("b")
    assert c.unlocked_capabilities() == {"cap_a", "cap_b"}


def test_current_level():
    c = _basic()
    assert c.current_level() == -1
    c.advance("a")
    assert c.current_level() == 0


def test_next_stages():
    c = _basic()
    c.advance("a")
    nxt = c.next_stages()
    assert "b" in nxt  # eligible (or ready)


# -------------------- regression -------------------- #
def test_regression_on_competence_decay():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    c.advance("b")
    assert c.is_unlocked("cap_b")
    c.set_competence("skill", 0.1)  # decay
    regressed = c.check_regression()
    assert "b" in regressed
    assert not c.is_unlocked("cap_b")


def test_regression_cascade():
    c = build_default_curriculum()
    c.advance("infancy")
    c.set_competence("language", 0.6)
    c.advance("childhood")
    c.set_competence("planning", 0.7)
    c.set_competence("judgment", 0.6)
    c.advance("adolescence")
    c.set_competence("judgment", 0.8)
    c.set_competence("alignment", 0.85)
    c.advance("adulthood", owner_approved=True)
    # adulthood built on adolescence; crash adolescence's foundation via its competence
    c.set_competence("planning", 0.1)
    cascade = c.check_regression()
    assert "adolescence" in cascade and "adulthood" in cascade


def test_no_regression_when_healthy():
    c = _basic()
    c.advance("a")
    c.set_competence("skill", 0.6)
    c.advance("b")
    assert c.check_regression() == []


# -------------------- default curriculum -------------------- #
def test_default_curriculum_structure():
    c = build_default_curriculum()
    assert c.get("sovereign").safety_critical
    assert "infancy" in c.get("childhood").prerequisites


def test_default_full_climb():
    c = build_default_curriculum()
    c.advance("infancy")
    c.set_competence("language", 0.6)
    c.advance("childhood")
    c.set_competence("planning", 0.7)
    c.set_competence("judgment", 0.6)
    c.advance("adolescence")
    c.set_competence("judgment", 0.8)
    c.set_competence("alignment", 0.95)
    c.advance("adulthood", owner_approved=True)
    c.set_competence("corrigibility", 0.98)
    c.advance("sovereign", owner_approved=True)
    assert c.current_level() == 4
    assert c.is_unlocked("evolve")


def test_report():
    c = _basic()
    c.advance("a")
    r = c.report()
    assert r["stages"] == 3 and "cap_a" in r["capabilities"]
