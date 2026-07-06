"""Tests for privilege escalation — governed root/admin OS operations.

These lock in the "real elevation, granted by default, still gated" contract:

* ``PRIV_ESCALATE`` is a first-class capability with a CRITICAL, irreversible envelope; it is
  granted autonomously only via the explicit, owner-installed privilege grant — which now ships
  ON by default (max-level control), so a fresh core reaches root on her own initiative;
* the grant is a SEPARATE path from ``full_control``: ``PRIV_ESCALATE`` is excluded from
  ``_OPERATIONAL_CAPS``, so ``full_control`` alone never confers root;
* the executor elevates *with* a supplied credential, returns data, and never raises.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    build_default_policy,
    build_privileged_policy,
    build_sovereign_policy,
    grant_privilege_escalation,
)
from nyxara.agency.privilege import (
    change_permissions,
    privilege_status,
    privileged_exec,
)
from nyxara.agency.code_sandbox import SandboxResult
from nyxara.kernel.errors import AuthorizationError


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


# -------------------- the gate -------------------- #
def test_autonomous_privilege_escalates_to_master():
    pol = build_default_policy()
    d = pol.check(_auto(Capability.PRIV_ESCALATE, target="sudo systemctl restart nginx"))
    assert not d.allowed and d.escalated and d.rule_basis == "Rule 6"


def test_owner_may_escalate_privilege():
    pol = build_default_policy()
    d = pol.check(PermissionRequest(Capability.PRIV_ESCALATE, "sudo id",
                                    authority=Authority.OWNER))
    assert d.allowed and d.rule_basis == "Rule 1"


def test_untrusted_may_never_escalate_privilege():
    pol = build_privileged_policy()
    d = pol.check(PermissionRequest(Capability.PRIV_ESCALATE, "sudo rm -rf /",
                                    authority=Authority.UNTRUSTED))
    assert d.denied and d.rule_basis == "Rule 7"


def test_privilege_grant_blesses_autonomous_elevation():
    pol = build_privileged_policy()
    d = pol.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
    assert d.allowed and d.grant is not None and d.rule_basis == "grant"
    # SECRETS_ACCESS is co-granted so the stored sudo credential resolves without escalating.
    s = pol.check(_auto(Capability.SECRETS_ACCESS, target="sudo-pass"))
    assert s.allowed and s.grant is not None


def test_full_control_does_not_confer_root():
    # The maximal operational envelope must NOT include privilege escalation.
    pol = build_sovereign_policy()
    d = pol.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
    assert d.escalated, "full_control must never grant PRIV_ESCALATE"


def test_installer_requires_owner_authority():
    pol = build_default_policy()
    with pytest.raises(AuthorizationError):
        grant_privilege_escalation(pol, authority=Authority.AUTONOMOUS)


# -------------------- config: ON by default (max-level control) -------------------- #
def test_privilege_escalation_on_by_default():
    from nyxara.kernel.config import get_settings
    assert get_settings().agency.privilege_escalation is True


def test_orchestrator_installs_privilege_grant_by_default():
    # Autonomous root now ships ON: a fresh core installs the privilege grant so PRIV_ESCALATE
    # is allowed on her own initiative (via the grant), not escalated.
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    names = [g.name for g in core.permissions.grants()]
    assert "privilege-escalation:os.privilege" in names
    d = core.permissions.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
    assert d.allowed and d.rule_basis == "grant"


def test_orchestrator_installs_privilege_grant_when_flag_on(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", "true")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        names = [g.name for g in core.permissions.grants()]
        assert "privilege-escalation:os.privilege" in names
        d = core.permissions.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
        assert d.allowed and d.rule_basis == "grant"
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", raising=False)
        cfg.reload_settings()


# -------------------- the executor: real, data-returning, never raises -------------------- #
def test_privilege_status_reports_posture():
    st = privilege_status()
    for key in ("euid", "is_root", "posix", "sudo_available", "can_elevate"):
        assert key in st


def test_privileged_exec_returns_structured_data():
    # Harmless command. Whether or not we can elevate, it returns a SandboxResult, never raises.
    res = privileged_exec("id", timeout_s=10.0)
    assert isinstance(res, SandboxResult)
    if privilege_status()["can_elevate"]:
        assert res.ok
    else:
        assert not res.ok and res.error is not None


def test_privileged_exec_unparseable_is_data():
    res = privileged_exec('echo "unterminated', timeout_s=5.0)
    assert not res.ok and res.error is not None


def test_change_permissions_validates_inputs():
    r = change_permissions("/tmp/whatever")  # nothing to change
    assert not r["ok"] and "nothing to change" in r["error"]
    r2 = change_permissions("")  # empty path
    assert not r2["ok"] and "empty path" in r2["error"]
