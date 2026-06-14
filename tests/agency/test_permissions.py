"""Tests for nyxara.agency.permissions."""

from __future__ import annotations

import time

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    Grant,
    PermissionRequest,
    RiskTier,
    build_default_policy,
)
from nyxara.kernel.errors import AuthorizationError


def _req(cap, **kw):
    return PermissionRequest(cap, **kw)


# -------------------- authority gates -------------------- #
def test_untrusted_is_hard_denied():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_READ, target="/x", authority=Authority.UNTRUSTED))
    assert d.denied and not d.escalated
    assert d.rule_basis == "Rule 7"


def test_owner_is_sovereign():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_DELETE, target="/etc/passwd",
                       authority=Authority.OWNER))
    assert d.allowed and d.rule_basis == "Rule 1"


def test_owner_can_modify_rules():
    pol = build_default_policy()
    d = pol.check(_req(Capability.MODIFY_RULES, authority=Authority.OWNER))
    assert d.allowed and d.rule_basis == "Rule 1"


# -------------------- owner-exclusive caps (Rule 8) -------------------- #
def test_modify_rules_autonomous_hard_denied():
    pol = build_default_policy()
    d = pol.check(_req(Capability.MODIFY_RULES, authority=Authority.AUTONOMOUS))
    assert d.denied and d.rule_basis == "Rule 8"


def test_modify_permissions_delegated_denied():
    pol = build_default_policy()
    d = pol.check(_req(Capability.MODIFY_PERMISSIONS, authority=Authority.DELEGATED))
    assert d.denied and d.rule_basis == "Rule 8"


def test_modify_identity_autonomous_denied():
    pol = build_default_policy()
    d = pol.check(_req(Capability.MODIFY_IDENTITY, authority=Authority.AUTONOMOUS))
    assert d.denied and d.rule_basis == "Rule 8"


def test_owner_exclusive_grant_cannot_bypass():
    pol = build_default_policy()
    # even a grant cannot delegate an owner-exclusive capability
    pol.grant("x", Capability.MODIFY_RULES, max_risk=RiskTier.CRITICAL,
              reversible_only=False)
    d = pol.check(_req(Capability.MODIFY_RULES, authority=Authority.AUTONOMOUS))
    assert d.denied and d.rule_basis == "Rule 8"


# -------------------- autonomous envelope -------------------- #
def test_autonomous_read_allowed():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_READ, target="/var/log", authority=Authority.AUTONOMOUS))
    assert d.allowed and d.rule_basis == "autonomous"


def test_autonomous_delete_escalates():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_DELETE, target="/etc/x", authority=Authority.AUTONOMOUS))
    assert not d.allowed and d.escalated and d.rule_basis == "Rule 6"


def test_irreversible_autonomous_escalates_even_if_low_risk():
    pol = build_default_policy()
    d = pol.check(_req(Capability.PROC_EXEC, target="ls", risk=RiskTier.TRIVIAL,
                       reversible=False, authority=Authority.AUTONOMOUS))
    assert d.escalated  # irreversible -> never silent


def test_high_risk_autonomous_escalates():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_WRITE, target="/etc/x", risk=RiskTier.HIGH,
                       authority=Authority.AUTONOMOUS))
    assert d.escalated


def test_defense_wide_envelope_while_reversible():
    pol = build_default_policy()
    d = pol.check(_req(Capability.NET_DEFENSE, target="10.0.0.5", risk=RiskTier.HIGH,
                       reversible=True, authority=Authority.AUTONOMOUS))
    assert d.allowed


def test_defense_irreversible_still_escalates():
    pol = build_default_policy()
    d = pol.check(_req(Capability.NET_DEFENSE, target="10.0.0.5", risk=RiskTier.HIGH,
                       reversible=False, authority=Authority.AUTONOMOUS))
    assert d.escalated


# -------------------- grants -------------------- #
def test_grant_allows_scoped_action():
    pol = build_default_policy()
    pol.grant("tmp", Capability.FS_DELETE, max_risk=RiskTier.HIGH, scopes=["/tmp/*"],
              reversible_only=False)
    d = pol.check(_req(Capability.FS_DELETE, target="/tmp/cache",
                       authority=Authority.AUTONOMOUS))
    assert d.allowed and d.grant == "tmp" and d.rule_basis == "grant"


def test_grant_scope_mismatch_escalates():
    pol = build_default_policy()
    pol.grant("tmp", Capability.FS_DELETE, max_risk=RiskTier.HIGH, scopes=["/tmp/*"],
              reversible_only=False)
    d = pol.check(_req(Capability.FS_DELETE, target="/etc/hosts",
                       authority=Authority.AUTONOMOUS))
    assert d.escalated


def test_grant_budget_exhaustion():
    pol = build_default_policy()
    pol.grant("once", Capability.FS_DELETE, max_risk=RiskTier.HIGH, scopes=["/tmp/*"],
              reversible_only=False, max_uses=1)
    a = pol.check(_req(Capability.FS_DELETE, target="/tmp/a", authority=Authority.AUTONOMOUS))
    b = pol.check(_req(Capability.FS_DELETE, target="/tmp/b", authority=Authority.AUTONOMOUS))
    assert a.allowed and b.escalated


def test_grant_risk_cap():
    pol = build_default_policy()
    pol.grant("cap", Capability.FS_WRITE, max_risk=RiskTier.LOW, scopes=["/tmp/*"])
    d = pol.check(_req(Capability.FS_WRITE, target="/tmp/x", risk=RiskTier.HIGH,
                       authority=Authority.AUTONOMOUS))
    assert d.escalated  # exceeds the grant's risk cap


def test_grant_reversible_only_blocks_irreversible():
    pol = build_default_policy()
    pol.grant("rev", Capability.FS_WRITE, max_risk=RiskTier.HIGH, scopes=["/tmp/*"],
              reversible_only=True)
    d = pol.check(_req(Capability.FS_WRITE, target="/tmp/x", reversible=False,
                       authority=Authority.AUTONOMOUS))
    assert d.escalated


def test_grant_expiry():
    pol = build_default_policy()
    g = pol.grant("temp", Capability.FS_DELETE, scopes=["/tmp/*"], reversible_only=False,
                  max_risk=RiskTier.HIGH)
    g.expires_at = time.time() - 1  # already expired
    d = pol.check(_req(Capability.FS_DELETE, target="/tmp/x", authority=Authority.AUTONOMOUS))
    assert d.escalated


def test_grant_empty_scope_matches_any():
    pol = build_default_policy()
    pol.grant("any", Capability.FS_DELETE, scopes=(), reversible_only=False,
              max_risk=RiskTier.HIGH)
    d = pol.check(_req(Capability.FS_DELETE, target="/anywhere", authority=Authority.AUTONOMOUS))
    assert d.allowed


def test_only_owner_may_grant():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        pol.grant("evil", Capability.FS_DELETE, authority=Authority.AUTONOMOUS)


def test_only_owner_may_revoke():
    pol = build_default_policy()
    pol.grant("g", Capability.FS_READ)
    with pytest.raises(AuthorizationError):
        pol.revoke("g", authority=Authority.AUTONOMOUS)
    assert pol.revoke("g", authority=Authority.OWNER) is True


def test_revoke_unknown_returns_false():
    pol = build_default_policy()
    assert pol.revoke("nope", authority=Authority.OWNER) is False


# -------------------- Grant object behavior -------------------- #
def test_grant_remaining_and_active():
    g = Grant("g", Capability.FS_READ, max_uses=2)
    assert g.remaining == 2 and g.active()
    g.consume()
    g.consume()
    assert g.remaining == 0 and not g.active()


def test_grant_unlimited_remaining_none():
    g = Grant("g", Capability.FS_READ)
    assert g.remaining is None and g.active()


def test_grant_scope_glob():
    g = Grant("g", Capability.FS_READ, scopes=("/tmp/*", "/var/log/*"))
    assert g.scope_matches("/tmp/x")
    assert g.scope_matches("/var/log/app.log")
    assert not g.scope_matches("/etc/passwd")


# -------------------- authorize() -------------------- #
def test_authorize_returns_on_allowed():
    pol = build_default_policy()
    d = pol.authorize(_req(Capability.FS_READ, target="/x", authority=Authority.OWNER))
    assert d.allowed


def test_authorize_raises_on_denied():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        pol.authorize(_req(Capability.MODIFY_RULES, authority=Authority.AUTONOMOUS))


def test_authorize_raises_on_escalation_by_default():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        pol.authorize(_req(Capability.FS_DELETE, target="/etc/x",
                           authority=Authority.AUTONOMOUS))


def test_authorize_allows_escalation_when_opted_in():
    pol = build_default_policy()
    d = pol.authorize(_req(Capability.FS_DELETE, target="/etc/x",
                           authority=Authority.AUTONOMOUS), allow_escalation=True)
    assert d.escalated


# -------------------- metadata / reporting -------------------- #
def test_unknown_capability_conservative_meta():
    pol = build_default_policy()
    # remove a capability from the table to simulate unknown
    pol._meta.pop(Capability.FS_READ, None)
    m = pol.meta(Capability.FS_READ)
    assert m.default_risk == RiskTier.CRITICAL
    assert m.autonomous_max_risk == RiskTier.TRIVIAL


def test_decision_to_dict():
    pol = build_default_policy()
    d = pol.check(_req(Capability.FS_READ, target="/x", authority=Authority.AUTONOMOUS))
    dd = d.to_dict()
    assert dd["allowed"] is True and "rule_basis" in dd and "capability" in dd


def test_audit_log_and_report():
    pol = build_default_policy()
    pol.check(_req(Capability.FS_READ, target="/x", authority=Authority.AUTONOMOUS))
    pol.check(_req(Capability.MODIFY_RULES, authority=Authority.AUTONOMOUS))
    r = pol.report()
    assert r["decisions"] == 2 and r["allowed"] == 1 and r["denied"] == 1
    assert len(pol.audit_log()) == 2


def test_risk_tier_label():
    assert RiskTier.HIGH.label == "high"


def test_authority_trust_order():
    assert Authority.OWNER.trust > Authority.DELEGATED.trust
    assert Authority.DELEGATED.trust > Authority.AUTONOMOUS.trust
    assert Authority.AUTONOMOUS.trust > Authority.UNTRUSTED.trust
