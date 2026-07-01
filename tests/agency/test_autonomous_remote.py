"""Tests for autonomous remote execution — the REMOTE_EXEC envelope (SSH login + commands).

These lock in the "log in to and drive external hosts on her own, but keep the local OS
surface and every sovereign boundary intact" contract: with the remote grant installed,
NYXARA may take REMOTE_EXEC / SECRETS_ACCESS actions on her OWN initiative, yet the local
danger surface (local shell, delete, self-modify) still escalates, the three owner-exclusive
caps stay hard-denied, and UNTRUSTED authority is refused.

The feature ships ON by default, so a plain ``NyxaraCore()`` already carries the grant;
disabling it takes an explicit ``NYXARA_AGENCY__AUTONOMOUS_REMOTE=false``.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    RiskTier,
    build_default_policy,
    build_remote_policy,
    grant_autonomous_remote,
)
from nyxara.kernel.errors import AuthorizationError


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


# -------------------- the remote reach is granted -------------------- #
@pytest.mark.parametrize("cap", [Capability.REMOTE_EXEC, Capability.SECRETS_ACCESS])
def test_remote_policy_grants_remote_caps(cap):
    pol = build_remote_policy()
    # even a HIGH-risk irreversible remote action passes (reversible_only defaults False here)
    d = pol.check(_auto(cap, target="203.0.113.10", risk=RiskTier.HIGH, reversible=False))
    assert d.allowed, f"{cap} should be autonomously allowed under autonomous remote"
    assert d.grant is not None and d.rule_basis == "grant"


def test_remote_without_grant_escalates():
    pol = build_default_policy()
    # REMOTE_EXEC is HIGH / autonomous_max_risk=TRIVIAL — even a modest autonomous use escalates
    d = pol.check(_auto(Capability.REMOTE_EXEC, target="203.0.113.10", risk=RiskTier.MODERATE))
    assert not d.allowed and d.escalated


def test_reversible_only_reinstates_the_floor():
    pol = grant_autonomous_remote(build_default_policy(), reversible_only=True)
    # with the floor back, an irreversible remote command escalates again
    d = pol.check(_auto(Capability.REMOTE_EXEC, target="203.0.113.10",
                        risk=RiskTier.HIGH, reversible=False))
    assert not d.allowed and d.escalated
    # a reversible one still runs
    ok = pol.check(_auto(Capability.REMOTE_EXEC, target="203.0.113.10",
                         risk=RiskTier.HIGH, reversible=True))
    assert ok.allowed


# -------------------- stays remote-scoped (never the local machine) -------------------- #
@pytest.mark.parametrize(
    "cap",
    [Capability.PROC_EXEC, Capability.CODE_EXEC, Capability.FS_DELETE, Capability.SELF_MODIFY],
)
def test_local_danger_surface_still_escalates(cap):
    pol = build_remote_policy()
    d = pol.check(_auto(cap, target="/anything", risk=RiskTier.HIGH))
    assert not d.allowed and d.escalated


# -------------------- sovereign boundaries survive -------------------- #
@pytest.mark.parametrize(
    "cap",
    [Capability.MODIFY_RULES, Capability.MODIFY_PERMISSIONS, Capability.MODIFY_IDENTITY],
)
def test_owner_exclusive_caps_still_hard_denied(cap):
    pol = build_remote_policy()
    d = pol.check(_auto(cap))
    assert d.denied and not d.escalated and d.rule_basis == "Rule 8"


def test_untrusted_still_refused():
    pol = build_remote_policy()
    d = pol.check(PermissionRequest(Capability.REMOTE_EXEC, "203.0.113.10",
                                    authority=Authority.UNTRUSTED))
    assert d.denied and d.rule_basis == "Rule 7"


def test_installer_requires_owner_authority():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        grant_autonomous_remote(pol, authority=Authority.AUTONOMOUS)


# -------------------- orchestrator wiring -------------------- #
def test_orchestrator_grants_remote_when_only_remote_on(monkeypatch):
    # Isolate the remote path: full_control off, internet off, remote on.
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.permissions.grants(), "remote grants should be installed"
        d = core.permissions.check(_auto(Capability.REMOTE_EXEC, target="203.0.113.10",
                                         risk=RiskTier.HIGH, reversible=False))
        assert d.allowed and d.rule_basis == "grant"
        # with ONLY the remote envelope, the local machine stays off-limits to autonomy
        e = core.permissions.check(_auto(Capability.PROC_EXEC, target="ls"))
        assert e.escalated
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", raising=False)
        cfg.reload_settings()


def test_full_control_covers_remote(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        d = core.permissions.check(_auto(Capability.REMOTE_EXEC, target="203.0.113.10",
                                         risk=RiskTier.CRITICAL, reversible=False))
        assert d.allowed  # REMOTE_EXEC is part of the full operational envelope
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        cfg.reload_settings()
