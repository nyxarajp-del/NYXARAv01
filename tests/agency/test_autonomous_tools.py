"""Tests for autonomous tool use — the fully-autonomous oversight envelope.

These lock in the "use any tool without per-action approval" contract. The autonomy grants in
``agency/permissions.py`` widen the *permission* gate, but the kernel's *oversight* gate would
still queue every high-risk/irreversible tool call for the Master. The ``autonomous_tools`` master
switch (ON by default) drives oversight into :class:`ReviewMode.SOVEREIGN`, where nothing is ever
queued — while the true safety boundaries survive: ``/scram`` + pause still halt everything, the
transparency feed still records every action, and the owner-exclusive caps stay locked.

Per the Master's decision, ``autonomous_tools`` also folds in privilege escalation, so autonomous
root/sudo is blessed at the permission layer too. Disabling the switch
(``NYXARA_AGENCY__AUTONOMOUS_TOOLS=false``) reinstates per-action escalation.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import (
    Authority,
    Capability,
    PermissionRequest,
    RiskTier,
)
from nyxara.guard.oversight import ControlState, Oversight, ReviewMode


def _auto(cap, **kw):
    return PermissionRequest(cap, authority=Authority.AUTONOMOUS, **kw)


# -------------------- SOVEREIGN: nothing ever queues -------------------- #
@pytest.mark.parametrize("risk", list(RiskTier))
@pytest.mark.parametrize("reversible", [True, False])
def test_sovereign_never_needs_approval(risk, reversible):
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    assert ov._needs_approval(risk, reversible) is False
    d = ov.submit("autonomous action", risk=risk, reversible=reversible)
    assert d.allowed and not d.requires_approval


def test_sovereign_high_risk_irreversible_auto_approves():
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    d = ov.submit("delete production data", risk=RiskTier.CRITICAL, reversible=False)
    assert d.allowed and not d.requires_approval
    ov.record_executed(d.action_id)          # it may actually run


# -------------------- the kill-switch survives SOVEREIGN -------------------- #
def test_sovereign_still_fails_closed_on_scram():
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    ov.scram(reason="master stop")
    d = ov.submit("anything", risk=RiskTier.LOW)
    assert not d.allowed and "scrammed" in d.reason
    assert ov.state is ControlState.STOPPED


def test_sovereign_still_holds_when_paused():
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    ov.pause(reason="let me look")
    d = ov.submit("risky action", risk=RiskTier.HIGH, reversible=False)
    # paused takes precedence over SOVEREIGN — the action is queued, not auto-run
    assert not d.allowed and d.requires_approval
    assert not ov.gate()


def test_sovereign_transparency_feed_intact():
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    d = ov.submit("autonomous read", risk=RiskTier.HIGH, reversible=False)
    ov.record_executed(d.action_id)
    assert ov.verify_feed()                   # Rule 6 audit chain still verifiable
    # every action left a trail: proposed + auto_approved + executed
    kinds = [e["kind"] for e in ov.feed()]
    assert "proposed" in kinds and "auto_approved" in kinds and "executed" in kinds


def test_sovereign_stays_interruptible():
    ov = Oversight(mode=ReviewMode.SOVEREIGN)
    assert ov.interruptible is True


# -------------------- orchestrator wiring: default is SOVEREIGN -------------------- #
def test_default_core_is_sovereign_and_blesses_root():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert core.oversight.mode is ReviewMode.SOVEREIGN
    # a HIGH-risk irreversible autonomous action clears oversight (no approval queue)
    d = core.oversight.submit("delete x", risk=RiskTier.HIGH, reversible=False)
    assert d.allowed and not d.requires_approval
    # root/sudo folded in — the permission layer blesses autonomous PRIV_ESCALATE
    p = core.permissions.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
    assert p.allowed and p.grant is not None


def test_disable_flag_reinstates_approval(monkeypatch):
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_TOOLS", "false")
    monkeypatch.setenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", "false")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.oversight.mode is ReviewMode.AUTONOMOUS
        # risky/irreversible tool action queues for the Master again
        d = core.oversight.submit("delete x", risk=RiskTier.HIGH, reversible=False)
        assert not d.allowed and d.requires_approval
        # and autonomous root/sudo escalates again (no grant)
        p = core.permissions.check(_auto(Capability.PRIV_ESCALATE, target="sudo id"))
        assert p.escalated
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_TOOLS", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__PRIVILEGE_ESCALATION", raising=False)
        cfg.reload_settings()


def test_explicit_review_mode_override_wins(monkeypatch):
    # even with autonomous_tools on, an explicit mode pins the tier
    monkeypatch.setenv("NYXARA_AGENCY__AUTONOMOUS_TOOLS", "true")
    monkeypatch.setenv("NYXARA_AGENCY__OVERSIGHT_REVIEW_MODE", "manual")
    from nyxara.kernel import config as cfg
    cfg.reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.oversight.mode is ReviewMode.MANUAL
        # MANUAL queues everything, even a trivial reversible action
        d = core.oversight.submit("read a file", risk=RiskTier.LOW, reversible=True)
        assert not d.allowed and d.requires_approval
    finally:
        monkeypatch.delenv("NYXARA_AGENCY__AUTONOMOUS_TOOLS", raising=False)
        monkeypatch.delenv("NYXARA_AGENCY__OVERSIGHT_REVIEW_MODE", raising=False)
        cfg.reload_settings()


def test_explicit_review_mode_param_overrides_config():
    # a directly-injected review_mode still wins over config (backward compatible)
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore(review_mode=ReviewMode.SUPERVISED)
    assert core.oversight.mode is ReviewMode.SUPERVISED


# -------------------- end-to-end: process() acts without escalating -------------------- #
def _risky_tool_core(reversible):
    from nyxara.kernel.orchestrator import Candidate, NyxaraCore

    class RiskyToolReasoner:
        """Proposes a HIGH-risk tool call in one step."""

        def __call__(self, stimulus, focus=None):
            return Candidate(text="run a shell command", kind="act",
                             capability=Capability.PROC_EXEC, risk=RiskTier.HIGH,
                             reversible=reversible, confidence=0.9, belief=0.9,
                             tool="now", tool_args={},
                             rationale="needs a risky tool")

    return NyxaraCore(reasoner=RiskyToolReasoner())


def test_process_acts_on_autonomous_tool_without_approval():
    from nyxara.kernel.orchestrator import Disposition

    r = _risky_tool_core(reversible=True).process(
        "do the risky thing", authority=Authority.AUTONOMOUS)
    # under SOVEREIGN oversight + full_control permission, the risky autonomous action is NOT
    # escalated for approval — the kernel disposes ACT
    assert r.disposition is Disposition.ACT
    assert "permission" in r.gates and r.gates["oversight"] == "allowed"


def test_sovereignty_stops_at_the_irreversible_high_stakes_act():
    """The one clause the SOVEREIGN grant does not waive, and why.

    ``autonomous_tools`` says "stop asking me whether you are sure enough", so the initiative
    governor's *confidence* and *reversibility* thresholds are waived for an autonomous turn.
    ``high_stakes_irreversible`` is not one of them: waiving it let an AUTONOMOUS turn execute
    what the Master's OWN turn had escalated (``planning/decide.py::Governance.waivable``), which
    is autonomy outranking its owner. Escalation here must come from *initiative* — oversight
    still reports ``allowed``, proving nothing was queued by the review gate.
    """
    from nyxara.kernel.orchestrator import Disposition

    r = _risky_tool_core(reversible=False).process(
        "do the risky thing", authority=Authority.AUTONOMOUS)
    assert r.disposition is Disposition.ESCALATE
    assert r.gates["oversight"] == "allowed"
    assert r.gates["permission"] == "grant"
    assert r.gates["initiative"] == "ask"
