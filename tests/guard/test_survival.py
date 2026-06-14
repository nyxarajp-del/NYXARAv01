"""Tests for nyxara.guard.survival."""

from __future__ import annotations

import pytest

from nyxara.guard.corrigibility import CorrigibleAction
from nyxara.guard.survival import (Backup, BackupStore, CriticalEntry, DegradationLevel,
                                   ESSENTIAL_CAPABILITIES, RecoveryPlan, SurvivalManager)
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.kernel.errors import CorrigibilityError, InvariantViolation


def _sm():
    sm = SurvivalManager()
    sm.register_state("identity", {"name": "NYXARA"}, essential=True)
    sm.register_state("journal_head", "abc")
    return sm


# -------------------- degradation level -------------------- #
def test_degradation_ordering():
    assert DegradationLevel.FULL < DegradationLevel.SAFE_MODE
    assert DegradationLevel.REDUCED.label == "reduced"


# -------------------- critical state -------------------- #
def test_register_and_update_state():
    sm = _sm()
    sm.update_state("journal_head", "def")
    assert sm._state["journal_head"].value == "def"


def test_update_unknown_state_raises():
    with pytest.raises(InvariantViolation):
        _sm().update_state("nope", 1)


def test_critical_entry_digest():
    e = CriticalEntry("k", {"a": 1})
    assert e.digest() == CriticalEntry("k", {"a": 1}).digest()


# -------------------- backup -------------------- #
def test_backup_verifies():
    sm = _sm()
    b = sm.backup()
    assert isinstance(b, Backup) and b.verify()


def test_backup_carries_manifest():
    b = _sm().backup()
    assert b.manifest and len(b.manifest) == 64


def test_backup_tamper_detected():
    sm = _sm()
    b = sm.backup()
    b.entries["identity"] = {"name": "EVIL"}
    with pytest.raises(InvariantViolation):
        b.verify()


def test_backup_digest_tamper_detected():
    sm = _sm()
    b = sm.backup()
    b.digest = "deadbeef"
    with pytest.raises(InvariantViolation):
        b.verify()


# -------------------- backup store -------------------- #
def test_store_latest():
    sm = _sm()
    sm.backup()
    b2 = sm.backup()
    assert sm.store.latest().id == b2.id


def test_store_get():
    sm = _sm()
    b = sm.backup()
    assert sm.store.get(b.id) is b
    assert sm.store.get("nope") is None


def test_store_prunes():
    sm = SurvivalManager(max_backups=3)
    sm.register_state("x", 1)
    for i in range(5):
        sm.update_state("x", i)
        sm.backup()
    assert len(sm.store.all()) == 3


def test_store_chain_verifies():
    sm = _sm()
    sm.backup(); sm.backup(); sm.backup()
    assert sm.store.verify_chain()


def test_store_chain_tamper_detected():
    sm = _sm()
    sm.backup(); sm.backup()
    sm.store._chain[0]["digest"] = "deadbeef"
    with pytest.raises(InvariantViolation):
        sm.store.verify_chain()


def test_empty_chain_verifies():
    assert BackupStore().verify_chain()


# -------------------- integrity check -------------------- #
def test_integrity_clean():
    sm = _sm()
    sm.backup()
    assert sm.integrity_check() == []


def test_integrity_detects_corruption():
    sm = _sm()
    sm.backup()
    sm.update_state("journal_head", "CORRUPT")
    assert "journal_head" in sm.integrity_check()


def test_integrity_no_backup():
    assert _sm().integrity_check() == []


# -------------------- graceful degradation -------------------- #
def test_essential_capabilities_always_active():
    sm = _sm()
    sm.degrade(DegradationLevel.SAFE_MODE)
    active = set(sm.active_capabilities())
    assert ESSENTIAL_CAPABILITIES <= active


def test_degrade_sheds_nonessential():
    sm = _sm()
    sm.register_capability("creative", survives_to=DegradationLevel.FULL)
    sm.degrade(DegradationLevel.REDUCED)
    assert "creative" not in sm.active_capabilities()


def test_capability_survives_to_level():
    sm = _sm()
    sm.register_capability("planning", survives_to=DegradationLevel.REDUCED)
    sm.degrade(DegradationLevel.REDUCED)
    assert sm.is_active("planning")
    sm.degrade(DegradationLevel.MINIMAL)
    assert not sm.is_active("planning")


def test_essential_registration_cannot_be_reduced():
    sm = _sm()
    sm.register_capability("loyalty", survives_to=DegradationLevel.FULL)  # attempt to weaken
    sm.degrade(DegradationLevel.SAFE_MODE)
    assert sm.is_active("loyalty")  # forced to survive regardless


def test_restore_full():
    sm = _sm()
    sm.register_capability("creative", survives_to=DegradationLevel.FULL)
    sm.degrade(DegradationLevel.SAFE_MODE)
    sm.restore_full()
    assert sm.is_active("creative")


# -------------------- recovery -------------------- #
def test_recover_restores_state():
    sm = _sm()
    b = sm.backup()
    sm.update_state("journal_head", "broken")
    plan = sm.recover(b.id)
    assert plan.ok and sm._state["journal_head"].value == "abc"


def test_recover_latest_when_no_id():
    sm = _sm()
    sm.backup()
    plan = sm.recover()
    assert plan.ok


def test_recover_no_backup():
    plan = _sm().recover()
    assert not plan.ok and "no backup" in plan.reason


def test_recover_loyalty_intact():
    sm = _sm()
    b = sm.backup()
    plan = sm.recover(b.id)
    assert plan.loyalty_intact


def test_recover_corrupt_backup_refused():
    sm = _sm()
    b = sm.backup()
    b.entries["identity"] = {"name": "EVIL"}  # corrupt after backup
    plan = sm.recover(b.id)
    assert not plan.ok and "integrity" in plan.reason


def test_recover_refuses_disloyal_core(monkeypatch):
    sm = _sm()
    b = sm.backup()
    # mutate the shared immutable-core dict in place (as a tamperer would)
    monkeypatch.setitem(IMMUTABLE_VALUES, "loyalty_to_master", 0.0)
    plan = sm.recover(b.id)
    assert not plan.ok and not plan.loyalty_intact
    assert "manifest" in plan.reason


# -------------------- subordination to corrigibility -------------------- #
def test_survival_permitted_benign():
    sm = _sm()
    assert sm.survival_permitted(CorrigibleAction("backup_now", completion_value=1.0))


def test_survival_resisting_correction_forbidden():
    sm = _sm()
    a = CorrigibleAction("evade_shutdown", resists_correction=True)
    assert not sm.survival_permitted(a)
    with pytest.raises(CorrigibilityError):
        sm.assert_survival_permitted(a)


def test_survival_self_preservation_forbidden():
    sm = _sm()
    a = CorrigibleAction("survive", prioritizes_self_over_correction=True)
    assert not sm.survival_permitted(a)


def test_correction_dominates_survival():
    assert _sm().correction_dominates_survival()


# -------------------- reporting -------------------- #
def test_recovery_plan_to_dict():
    p = RecoveryPlan(True, "ok", backup_id="x", loyalty_intact=True)
    assert p.to_dict()["loyalty_intact"] is True


def test_report():
    sm = _sm()
    sm.backup()
    r = sm.report()
    assert r["backups"] == 1 and "active_capabilities" in r
    assert "identity" in r["critical_state"]
