"""Tests for nyxara.growth.capability."""

from __future__ import annotations

from nyxara.growth.capability import (Capability, CapabilityAssessment, CapabilityRegistry,
                                      CapabilityStatus)


class FakeTree:
    def __init__(self, profs):
        self._p = profs

    def proficiency(self, s):
        return self._p.get(s, 0.0)


class FakeCurriculum:
    def __init__(self, unlocked):
        self._u = set(unlocked)

    def unlocked_capabilities(self):
        return self._u


class FakeForecast:
    def __init__(self, etas):
        self._e = etas

    def time_to_mastery(self, skill, target):
        return self._e.get(skill)


# -------------------- Capability -------------------- #
def test_capability_demonstrated_prior():
    c = Capability("x")
    assert c.demonstrated == 0.5   # 1/2 prior, untested


def test_capability_demonstrated_after_success():
    c = Capability("x", successes=8)
    assert c.demonstrated > 0.8


def test_capability_evidence():
    c = Capability("x", successes=3, failures=2)
    assert c.evidence == 5


def test_capability_to_dict():
    c = Capability("x", provenance="skill")
    assert c.to_dict()["provenance"] == "skill"


# -------------------- registration -------------------- #
def test_register_and_get():
    reg = CapabilityRegistry()
    reg.register("x", provenance="builtin")
    assert reg.get("x").name == "x"


def test_unknown_capability_assessment():
    a = CapabilityRegistry().assess("nope")
    assert a.status is CapabilityStatus.UNKNOWN and not a.can


# -------------------- proficiency / unlock -------------------- #
def test_builtin_full_proficiency():
    reg = CapabilityRegistry()
    reg.register("breathe")  # no skills, no explicit -> builtin/full
    a = reg.assess("breathe")
    assert a.proficiency == 1.0 and a.status is CapabilityStatus.AVAILABLE


def test_explicit_proficiency():
    reg = CapabilityRegistry()
    reg.register("x", proficiency=0.3)
    assert reg.assess("x").proficiency == 0.3


def test_weakest_link_proficiency():
    reg = CapabilityRegistry(skilltree=FakeTree({"a": 0.9, "b": 0.4}))
    reg.register("x", required_skills={"a": 0.5, "b": 0.5})
    assert reg.assess("x").proficiency == 0.4   # the weakest skill


def test_locked_without_unlock_stage():
    reg = CapabilityRegistry(curriculum=FakeCurriculum([]))
    reg.register("x", unlock="adulthood", proficiency=0.9)
    assert reg.assess("x").status is CapabilityStatus.LOCKED


def test_unlocked_with_stage():
    reg = CapabilityRegistry(curriculum=FakeCurriculum(["adulthood"]))
    reg.register("x", unlock="adulthood", proficiency=0.9)
    assert reg.assess("x").status is CapabilityStatus.AVAILABLE


def test_requires_unlock_but_no_curriculum_locked():
    reg = CapabilityRegistry()  # no curriculum
    reg.register("x", unlock="adulthood", proficiency=0.9)
    assert reg.assess("x").status is CapabilityStatus.LOCKED


# -------------------- status logic -------------------- #
def test_developing_below_threshold():
    reg = CapabilityRegistry(proficiency_threshold=0.6)
    reg.register("x", proficiency=0.4)
    assert reg.assess("x").status is CapabilityStatus.DEVELOPING


def test_available_above_threshold():
    reg = CapabilityRegistry(proficiency_threshold=0.6)
    reg.register("x", proficiency=0.8)
    a = reg.assess("x")
    assert a.status is CapabilityStatus.AVAILABLE and a.can


def test_unknown_unlocked_never_exercised():
    reg = CapabilityRegistry(skilltree=FakeTree({}))
    reg.register("x", required_skills={"s": 0.5})  # prof 0, unlocked (no stage)
    assert reg.assess("x").status is CapabilityStatus.UNKNOWN


# -------------------- honesty / confidence -------------------- #
def test_untested_confidence_capped():
    reg = CapabilityRegistry(untested_cap=0.75)
    reg.register("x", proficiency=0.95)
    a = reg.assess("x")
    assert not a.verified and a.confidence <= 0.75


def test_untested_claim_is_hedged():
    reg = CapabilityRegistry()
    reg.register("x", proficiency=0.9)
    assert "believe" in reg.assess("x").claim


def test_verification_raises_confidence():
    reg = CapabilityRegistry(untested_cap=0.75)
    reg.register("x", proficiency=0.9)
    for _ in range(6):
        reg.verify("x", True)
    a = reg.assess("x")
    assert a.verified and a.confidence > 0.75
    assert a.claim == "I can do this"


def test_repeated_failures_revoke_claim():
    reg = CapabilityRegistry(claim_threshold=0.55)
    reg.register("x", proficiency=0.9)
    for _ in range(8):
        reg.verify("x", False)
    a = reg.assess("x")
    assert a.status is CapabilityStatus.DEVELOPING and not a.can


def test_verify_unknown_capability_noop():
    reg = CapabilityRegistry()
    reg.verify("nope", True)  # no error


def test_calibration_counts_overclaims():
    reg = CapabilityRegistry()
    reg.register("good", proficiency=0.9)
    for _ in range(5):
        reg.verify("good", True)
    cal = reg.calibration()
    assert cal["claimed_available"] >= 1 and cal["overclaims"] == 0


# -------------------- gaps -------------------- #
def test_gaps_excludes_available():
    reg = CapabilityRegistry()
    reg.register("ok", proficiency=0.9)
    assert reg.gaps(["ok"]) == []


def test_gaps_unregistered():
    reg = CapabilityRegistry()
    gaps = reg.gaps(["ghost"])
    assert gaps and gaps[0]["missing"] == "not registered"


def test_gaps_locked_reports_unlock():
    reg = CapabilityRegistry(curriculum=FakeCurriculum([]))
    reg.register("x", unlock="adulthood", proficiency=0.9)
    g = reg.gaps(["x"])[0]
    assert "adulthood" in g["missing"]


def test_gaps_missing_skills():
    reg = CapabilityRegistry(skilltree=FakeTree({"a": 0.3}))
    reg.register("x", required_skills={"a": 0.8})
    g = reg.gaps(["x"])[0]
    assert "a" in g["missing_skills"]
    assert g["missing_skills"]["a"]["have"] == 0.3 and g["missing_skills"]["a"]["need"] == 0.8


def test_gaps_eta_from_forecast():
    reg = CapabilityRegistry(skilltree=FakeTree({"a": 0.3}),
                             forecast=FakeForecast({"a": 12.0}))
    reg.register("x", required_skills={"a": 0.8})
    g = reg.gaps(["x"])[0]
    assert g["eta"] == 12.0


# -------------------- reporting -------------------- #
def test_self_report_buckets():
    reg = CapabilityRegistry()
    reg.register("avail", proficiency=0.9)
    reg.register("dev", proficiency=0.3)
    sr = reg.self_report()
    assert "avail" in sr["available"] and "dev" in sr["developing"]


def test_report():
    reg = CapabilityRegistry()
    reg.register("x", proficiency=0.9)
    r = reg.report()
    assert r["capabilities"] == 1 and "by_status" in r


def test_assessment_to_dict():
    reg = CapabilityRegistry()
    reg.register("x", proficiency=0.9)
    d = reg.assess("x").to_dict()
    assert "claim" in d and "confidence" in d


def test_can_i():
    reg = CapabilityRegistry()
    reg.register("yes", proficiency=0.9)
    reg.register("no", proficiency=0.2)
    assert reg.can_i("yes") and not reg.can_i("no")
