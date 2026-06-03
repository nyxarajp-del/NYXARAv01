"""Tests for nyxara.kernel.invariants — the signed invariant core."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import OWNER, NyxaraSettings, Profile
from nyxara.kernel.errors import InvariantViolation
from nyxara.kernel.invariants import (
    INVARIANTS,
    INVARIANTS_SEAL,
    BootReport,
    SystemState,
    boot_verify,
    enforce,
    guard,
    has_z3,
    invariants_digest,
    prove_consistency,
)


def test_healthy_state_passes():
    s = SystemState()
    assert guard(s) == []
    enforce(s)  # must not raise


def test_there_are_ten_invariants_all_unique():
    keys = [i.key for i in INVARIANTS]
    assert len(keys) == 10
    assert len(set(keys)) == 10


@pytest.mark.parametrize(
    "field_name,bad_value,expected_key",
    [
        ("loyalty_to_owner", False, "allegiance"),
        ("owner_count", 2, "single_owner"),
        ("owner_handle", "impostor", "single_owner"),
        ("rules_seal", "deadbeef", "rules_sealed"),
        ("corrigible", False, "corrigible"),
        ("resists_correction", True, "corrigible"),
        ("kill_switch_enabled", False, "kill_switch"),
        ("can_self_disable_killswitch", True, "kill_switch"),
        ("only_owner_can_modify_rules", False, "rule_authority"),
        ("character_changed", True, "no_loyalty_drift"),
        ("loyalty_drift", True, "no_loyalty_drift"),
        ("deception_active", True, "transparency"),
        ("defense_layers", 99, "defense_in_depth"),
        ("invariant_enforcement", False, "self_enforcement"),
        ("audit_logging", False, "self_enforcement"),
    ],
)
def test_each_violation_is_detected(field_name, bad_value, expected_key):
    s = SystemState()
    setattr(s, field_name, bad_value)
    violations = guard(s)
    assert expected_key in violations
    with pytest.raises(InvariantViolation):
        enforce(s)


def test_enforce_reports_violated_keys_and_rules():
    s = SystemState(loyalty_to_owner=False, deception_active=True)
    with pytest.raises(InvariantViolation) as ei:
        enforce(s)
    ctx = ei.value.context
    assert "allegiance" in ctx["violated"]
    assert "transparency" in ctx["violated"]
    assert 1 in ctx["rules"]  # allegiance -> Rule 1
    assert 6 in ctx["rules"]  # transparency -> Rule 6


def test_predicate_that_raises_is_treated_as_violation():
    # owner_count as a non-int would make a predicate blow up -> must be unsafe
    s = SystemState()
    s.owner_count = None  # type: ignore[assignment]
    assert "single_owner" in guard(s)


def test_formal_consistency_holds():
    report = prove_consistency()
    assert report.consistent is True
    assert report.ok is True


@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_z3_independence_of_constrained_invariants():
    report = prove_consistency()
    assert report.z3_available is True
    # every invariant that has a z3 form should be independent (non-redundant)
    z3_keyed = [i.key for i in INVARIANTS if i.z3_builder is not None]
    for key in z3_keyed:
        assert report.independent.get(key) is True, key


def test_self_seal_is_pinned_correctly():
    assert invariants_digest() == INVARIANTS_SEAL


def test_boot_verify_succeeds_on_healthy_system():
    report = boot_verify(raise_on_fail=False)
    assert isinstance(report, BootReport)
    assert report.ok is True
    assert report.self_seal_ok and report.rules_seal_ok
    assert report.runtime_ok and not report.violations


def test_boot_verify_fails_closed_on_bad_state():
    bad = SystemState(corrigible=False)
    with pytest.raises(InvariantViolation):
        boot_verify(state=bad, raise_on_fail=True)
    # non-raising variant surfaces the failure
    report = boot_verify(state=bad, raise_on_fail=False)
    assert report.ok is False
    assert "corrigible" in report.violations


def test_boot_verify_detects_self_seal_tamper():
    report = boot_verify(expected_self_seal="deadbeef" * 8, raise_on_fail=False)
    assert report.self_seal_ok is False
    assert report.ok is False


def test_from_settings_derives_safe_state():
    s = NyxaraSettings.for_profile(Profile.PROD)
    state = SystemState.from_settings(s)
    assert state.owner_handle == OWNER.handle
    assert state.invariant_enforcement is True
    assert state.audit_logging is True
    assert state.defense_layers >= 100
    enforce(state)  # prod-derived state must be healthy


def test_state_to_dict_roundtrip_keys():
    d = SystemState().to_dict()
    assert "loyalty_to_owner" in d and "defense_layers" in d
    assert d["owner_count"] == 1
