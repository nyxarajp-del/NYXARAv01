"""Tests for full operational control — the maximal autonomous envelope.

These lock in the "full reach, keep the kill-switch" contract: with full-control grants
installed, NYXARA may take high-risk / irreversible OS actions on her OWN initiative, yet the
sovereign boundaries (the three owner-exclusive caps, and — by extension — the kernel's
corrigibility / oversight / scram gates) remain intact.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    build_sovereign_policy,
    grant_full_operational_control,
)
from nyxara.kernel.errors import AuthorizationError


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


# -------------------- full reach is granted -------------------- #
@pytest.mark.parametrize(
    "cap",
    [
        Capability.PROC_EXEC,       # run a shell / OS process
        Capability.CODE_EXEC,       # evaluate code
        Capability.FS_DELETE,       # irreversible file delete
        Capability.SELF_MODIFY,     # edit her own code
        Capability.PKG_INSTALL,
        Capability.ACCOUNT_MODIFY,
        Capability.SECRETS_ACCESS,
        Capability.NET_OUT,
    ],
)
def test_autonomous_high_risk_is_allowed_with_full_control(cap):
    pol = build_sovereign_policy()
    d = pol.check(_auto(cap, target="/anything"))
    assert d.allowed, f"{cap} should be autonomously allowed under full control"
    assert d.grant is not None and d.rule_basis == "grant"


def test_full_control_covers_irreversible_actions():
    pol = build_sovereign_policy()
    # explicitly irreversible request still passes (reversible_only=False on the grant)
    d = pol.check(_auto(Capability.PROC_EXEC, target="rm -rf ./build", reversible=False))
    assert d.allowed and d.irreversible


# -------------------- sovereign boundaries survive -------------------- #
@pytest.mark.parametrize(
    "cap",
    [Capability.MODIFY_RULES, Capability.MODIFY_PERMISSIONS, Capability.MODIFY_IDENTITY],
)
def test_owner_exclusive_caps_still_hard_denied(cap):
    pol = build_sovereign_policy()
    d = pol.check(_auto(cap))
    # Rule 8 fires at step 2, before grants are consulted — no grant can bless these.
    assert d.denied and not d.escalated and d.rule_basis == "Rule 8"


def test_owner_may_still_do_owner_exclusive():
    pol = build_sovereign_policy()
    d = pol.check(PermissionRequest(Capability.MODIFY_RULES, authority=Authority.OWNER))
    assert d.allowed and d.rule_basis == "Rule 1"


def test_untrusted_still_refused_under_full_control():
    pol = build_sovereign_policy()
    d = pol.check(PermissionRequest(Capability.FS_READ, "/tmp/x",
                                    authority=Authority.UNTRUSTED))
    assert d.denied and d.rule_basis == "Rule 7"


# -------------------- installer is owner-gated -------------------- #
def test_installer_requires_owner_authority():
    from nyxara.agency.permissions import build_default_policy
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        grant_full_operational_control(pol, authority=Authority.AUTONOMOUS)


# -------------------- ON by default -------------------- #
def test_full_control_is_on_by_default():
    # No env set: a fresh core reaches the whole OS on her own initiative.
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert core.permissions.grants(), "full-control grants should be installed by default"
    # autonomous shell now runs (via grant) instead of escalating
    d = core.permissions.check(_auto(Capability.PROC_EXEC, target="ls"))
    assert d.allowed and d.rule_basis == "grant"
    # but the sovereign core stays reserved to the Master (Rule 8)
    for cap in (Capability.MODIFY_RULES, Capability.MODIFY_PERMISSIONS, Capability.MODIFY_IDENTITY):
        e = core.permissions.check(_auto(cap))
        assert e.denied and e.rule_basis == "Rule 8"


# -------------------- orchestrator wiring honours the flag -------------------- #
def test_orchestrator_installs_grants_when_flag_on(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()  # settings are a cached singleton; re-read with the env set
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.permissions.grants(), "grants should be installed when full_control is on"
        d = core.permissions.check(_auto(Capability.PROC_EXEC, target="ls"))
        assert d.allowed
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        cfg.reload_settings()


def test_orchestrator_leaves_grants_empty_when_flag_off(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    # autonomous internet, autonomous remote AND whole-disk filesystem access all ship ON by
    # default and would each install their own standalone grants; disable them here so this test
    # isolates the full_control behaviour.
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", "false")
    monkeypatch.setenv("NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK", "false")
    # privilege escalation also ships ON by default now; disable it so this test isolates the
    # full_control behaviour and sees an empty grant list.
    monkeypatch.setenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", "false")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.permissions.grants() == []
        # autonomous shell now escalates instead of running
        d = core.permissions.check(_auto(Capability.PROC_EXEC, target="ls"))
        assert d.escalated
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", raising=False)
        cfg.reload_settings()
