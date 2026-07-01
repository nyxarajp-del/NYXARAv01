"""Tests for autonomous internet — the network-scoped autonomous envelope.

These lock in the "reach the whole web on her own, but keep autonomy off the machine"
contract: with the internet grants installed, NYXARA may take web / messaging / account /
secret actions on her OWN initiative, yet the OS danger surface (shell, delete, self-modify)
still escalates, and every sovereign boundary (the three owner-exclusive caps, and — by
extension — the kernel's corrigibility / oversight / scram gates) remains intact.

The feature ships ON by default, so a plain ``NyxaraCore()`` already carries the grants;
disabling it takes an explicit ``NYXARA_AGENCY__AUTONOMOUS_INTERNET=false``.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    RiskTier,
    build_default_policy,
    build_internet_policy,
    grant_autonomous_internet,
)
from nyxara.kernel.errors import AuthorizationError


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


# -------------------- full reach on the wire is granted -------------------- #
@pytest.mark.parametrize(
    "cap",
    [
        Capability.NET_OUT,          # browse / fetch / HTTP
        Capability.NET_IN,           # accept inbound
        Capability.MESSAGE_SEND,     # post / submit / send
        Capability.ACCOUNT_MODIFY,   # log in / manage accounts
        Capability.SECRETS_ACCESS,   # use API keys
    ],
)
def test_full_scope_grants_every_internet_cap(cap):
    pol = build_internet_policy()  # scope="full"
    # even at HIGH risk (proves max_risk=CRITICAL), reversible actions pass via the grant
    d = pol.check(_auto(cap, target="https://api.example.com", risk=RiskTier.HIGH))
    assert d.allowed, f"{cap} should be autonomously allowed under autonomous internet"
    assert d.grant is not None and d.rule_basis == "grant"


# -------------------- reversible-only envelope by default -------------------- #
def test_irreversible_web_action_escalates_by_default():
    pol = build_internet_policy()  # reversible_only=True
    d = pol.check(_auto(Capability.NET_OUT, target="https://api.example.com",
                        risk=RiskTier.HIGH, reversible=False))
    assert not d.allowed and d.escalated  # irreversible still needs the Master


def test_allow_irreversible_lets_irreversible_web_through():
    pol = grant_autonomous_internet(build_default_policy(), reversible_only=False)
    d = pol.check(_auto(Capability.NET_OUT, target="https://api.example.com",
                        risk=RiskTier.HIGH, reversible=False))
    assert d.allowed and d.irreversible


# -------------------- stays internet-scoped (narrower than full control) -------------------- #
@pytest.mark.parametrize(
    "cap",
    [Capability.PROC_EXEC, Capability.CODE_EXEC, Capability.FS_DELETE, Capability.SELF_MODIFY],
)
def test_os_danger_surface_still_escalates(cap):
    pol = build_internet_policy()
    d = pol.check(_auto(cap, target="/anything", risk=RiskTier.HIGH))
    # no internet grant covers these; the conservative autonomous envelope escalates them
    assert not d.allowed and d.escalated


# -------------------- scope tiers -------------------- #
def test_read_scope_is_net_only():
    pol = grant_autonomous_internet(build_default_policy(), scope="read")
    assert pol.check(_auto(Capability.NET_OUT, target="https://x", risk=RiskTier.MODERATE)).allowed
    assert not pol.check(_auto(Capability.MESSAGE_SEND, risk=RiskTier.MODERATE)).allowed
    assert not pol.check(_auto(Capability.SECRETS_ACCESS, risk=RiskTier.HIGH)).allowed


def test_write_scope_adds_messaging_but_not_secrets():
    pol = grant_autonomous_internet(build_default_policy(), scope="write")
    assert pol.check(_auto(Capability.MESSAGE_SEND, risk=RiskTier.MODERATE)).allowed
    assert not pol.check(_auto(Capability.SECRETS_ACCESS, risk=RiskTier.HIGH)).allowed


# -------------------- sovereign boundaries survive -------------------- #
@pytest.mark.parametrize(
    "cap",
    [Capability.MODIFY_RULES, Capability.MODIFY_PERMISSIONS, Capability.MODIFY_IDENTITY],
)
def test_owner_exclusive_caps_still_hard_denied(cap):
    pol = build_internet_policy()
    d = pol.check(_auto(cap))
    # Rule 8 fires at step 2, before grants are consulted — no grant can bless these.
    assert d.denied and not d.escalated and d.rule_basis == "Rule 8"


def test_untrusted_still_refused():
    pol = build_internet_policy()
    d = pol.check(PermissionRequest(Capability.NET_OUT, "https://x",
                                    authority=Authority.UNTRUSTED))
    assert d.denied and d.rule_basis == "Rule 7"


# -------------------- installer is owner-gated -------------------- #
def test_installer_requires_owner_authority():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        grant_autonomous_internet(pol, authority=Authority.AUTONOMOUS)


# -------------------- orchestrator wiring: the internet-only grant path -------------------- #
def test_orchestrator_grants_internet_when_only_internet_on(monkeypatch):
    # Isolate the internet path: full_control (the broader default) off, internet on.
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.permissions.grants(), "internet grants should be installed"
        d = core.permissions.check(_auto(Capability.NET_OUT, target="https://example.com",
                                         risk=RiskTier.MODERATE))
        assert d.allowed and d.rule_basis == "grant"
        # with ONLY the internet envelope, the machine stays off-limits to autonomy
        e = core.permissions.check(_auto(Capability.PROC_EXEC, target="ls"))
        assert e.escalated
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", raising=False)
        cfg.reload_settings()


def test_orchestrator_respects_disable_flag(monkeypatch):
    # All autonomy envelopes off → the conservative default policy, no standing grants.
    # (autonomous_remote is on by default and independent, so it must be disabled too.)
    monkeypatch.setenv("NYXARA_AGENCY__FULL_CONTROL", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", "false")
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", "false")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()  # settings are a cached singleton; re-read with the env set
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.permissions.grants() == []
        # autonomous web fetch (moderate) now escalates instead of running
        d = core.permissions.check(_auto(Capability.NET_OUT, target="https://x",
                                         risk=RiskTier.MODERATE))
        assert d.escalated
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__FULL_CONTROL", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_INTERNET", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_REMOTE", raising=False)
        cfg.reload_settings()
