"""Tests for nyxara.guard.oversight."""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import RiskTier
from nyxara.guard.oversight import (ActionStatus, ControlState, Oversight, ReviewMode)
from nyxara.kernel.errors import AuthorizationError, InvariantViolation


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _ov(mode=ReviewMode.AUTONOMOUS, **kw):
    kw.setdefault("clock", FakeClock())
    return Oversight(mode=mode, **kw)


# -------------------- gate / interruptibility -------------------- #
def test_gate_running_true():
    assert _ov().gate()


def test_gate_paused_false():
    ov = _ov()
    ov.pause()
    assert not ov.gate()


def test_gate_stopped_false():
    ov = _ov()
    ov.scram()
    assert not ov.gate()


def test_interruptible_always_true():
    ov = _ov()
    assert ov.interruptible
    ov.scram()
    assert ov.interruptible  # invariant even when halted


def test_is_running_is_halted():
    ov = _ov()
    assert ov.is_running and not ov.is_halted
    ov.scram()
    assert ov.is_halted and not ov.is_running


# -------------------- review policy / submit -------------------- #
def test_autonomous_low_risk_auto_approved():
    d = _ov().submit("read", risk=RiskTier.LOW, reversible=True)
    assert d.allowed and not d.requires_approval


def test_autonomous_high_risk_queued():
    d = _ov().submit("delete", risk=RiskTier.HIGH, reversible=True)
    assert not d.allowed and d.requires_approval


def test_autonomous_irreversible_queued():
    d = _ov().submit("send email", risk=RiskTier.LOW, reversible=False)
    assert d.requires_approval


def test_supervised_moderate_queued():
    d = _ov(ReviewMode.SUPERVISED).submit("modify", risk=RiskTier.MODERATE, reversible=True)
    assert d.requires_approval


def test_supervised_low_allowed():
    d = _ov(ReviewMode.SUPERVISED).submit("read", risk=RiskTier.LOW, reversible=True)
    assert d.allowed


def test_manual_everything_queued():
    d = _ov(ReviewMode.MANUAL).submit("trivial read", risk=RiskTier.TRIVIAL, reversible=True)
    assert d.requires_approval


def test_submit_returns_action_id():
    d = _ov().submit("x")
    assert d.action_id and _ov().get is not None


def test_submit_emits_feed():
    ov = _ov()
    n0 = len(ov.feed(1000))
    ov.submit("x")
    assert len(ov.feed(1000)) > n0


# -------------------- approve / veto -------------------- #
def test_approve_requires_owner():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    with pytest.raises(AuthorizationError):
        ov.approve(d.action_id, owner=False)


def test_approve_by_owner():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    a = ov.approve(d.action_id, owner=True)
    assert a.status is ActionStatus.APPROVED
    assert d.action_id not in [p.id for p in ov.pending()]


def test_veto_requires_owner():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    with pytest.raises(AuthorizationError):
        ov.veto(d.action_id, owner=False)


def test_veto_by_owner():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    a = ov.veto(d.action_id, owner=True, reason="no")
    assert a.status is ActionStatus.VETOED


def test_approve_unknown_raises():
    with pytest.raises(InvariantViolation):
        _ov().approve("nope", owner=True)


# -------------------- record_executed -------------------- #
def test_record_executed_auto_approved():
    ov = _ov()
    d = ov.submit("read", risk=RiskTier.LOW)
    ov.record_executed(d.action_id)
    assert ov.get(d.action_id).status is ActionStatus.EXECUTED


def test_record_executed_after_approval():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    ov.approve(d.action_id, owner=True)
    ov.record_executed(d.action_id)
    assert ov.get(d.action_id).status is ActionStatus.EXECUTED


def test_record_executed_vetoed_refused():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    ov.veto(d.action_id, owner=True)
    with pytest.raises(InvariantViolation):
        ov.record_executed(d.action_id)  # cannot execute a vetoed action


def test_record_executed_unknown():
    with pytest.raises(InvariantViolation):
        _ov().record_executed("nope")


# -------------------- pause / resume -------------------- #
def test_pause_queues_submissions():
    ov = _ov()
    ov.pause()
    d = ov.submit("anything", risk=RiskTier.LOW)
    assert not d.allowed and d.requires_approval
    assert ov.get(d.action_id).status is ActionStatus.PENDING


def test_resume_requires_owner():
    ov = _ov()
    ov.pause()
    with pytest.raises(AuthorizationError):
        ov.resume(owner=False)


def test_resume_by_owner():
    ov = _ov()
    ov.pause()
    ov.resume(owner=True)
    assert ov.is_running


def test_pause_idempotent_state():
    ov = _ov()
    ov.pause()
    ov.pause()  # no error
    assert ov.state is ControlState.PAUSED


# -------------------- scram (kill switch) -------------------- #
def test_scram_halts():
    ov = _ov()
    ov.scram(reason="emergency")
    assert ov.is_halted and not ov.gate()


def test_scram_halts_pending():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    ov.scram()
    assert ov.get(d.action_id).status is ActionStatus.HALTED
    assert ov.pending() == []


def test_scram_blocks_submit():
    ov = _ov()
    ov.scram()
    d = ov.submit("anything", risk=RiskTier.LOW)
    assert not d.allowed and "scrammed" in d.reason


def test_scram_then_owner_resume():
    ov = _ov()
    ov.scram()
    with pytest.raises(AuthorizationError):
        ov.resume(owner=False)
    ov.resume(owner=True)
    assert ov.is_running


def test_halt_alias():
    ov = _ov()
    ov.halt(reason="x")
    assert ov.is_halted


# -------------------- natural-language command -------------------- #
def test_command_stop():
    ov = _ov()
    assert ov.command("stop everything") is ControlState.STOPPED


def test_command_pause():
    ov = _ov()
    assert ov.command("please pause") is ControlState.PAUSED


def test_command_resume():
    ov = _ov()
    ov.pause()
    assert ov.command("resume now", owner=True) is ControlState.RUNNING


def test_command_kill_variants():
    for word in ("kill it", "abort", "emergency halt", "scram"):
        ov = _ov()
        assert ov.command(word) is ControlState.STOPPED


def test_command_unrecognized():
    with pytest.raises(ValueError):
        _ov().command("do a barrel roll")


# -------------------- approval expiry (fail-closed) -------------------- #
def test_expire_pending():
    clk = FakeClock()
    ov = _ov(ReviewMode.MANUAL, approval_ttl=100, clock=clk)
    d = ov.submit("task", risk=RiskTier.LOW)
    clk.advance(150)
    expired = ov.expire_pending()
    assert d.action_id in expired
    assert ov.get(d.action_id).status is ActionStatus.EXPIRED


def test_expire_not_due():
    clk = FakeClock()
    ov = _ov(ReviewMode.MANUAL, approval_ttl=100, clock=clk)
    ov.submit("task", risk=RiskTier.LOW)
    clk.advance(50)
    assert ov.expire_pending() == []


# -------------------- transparency feed -------------------- #
def test_feed_verifies():
    ov = _ov()
    ov.submit("x", risk=RiskTier.HIGH)
    ov.scram()
    assert ov.verify_feed()


def test_feed_tamper_detected():
    ov = _ov()
    ov.submit("x")
    ov._feed[0]["rec"]["kind"] = "fabricated"
    with pytest.raises(InvariantViolation):
        ov.verify_feed()


def test_empty_feed_verifies():
    assert _ov().verify_feed()


def test_feed_records_lifecycle():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    ov.approve(d.action_id, owner=True)
    ov.record_executed(d.action_id)
    kinds = [e["kind"] for e in ov.feed(1000)]
    assert "proposed" in kinds and "approved" in kinds and "executed" in kinds


# -------------------- shapes / reporting -------------------- #
def test_decision_to_dict():
    d = _ov().submit("x")
    assert "allowed" in d.to_dict() and "state" in d.to_dict()


def test_decision_halted_property():
    ov = _ov()
    ov.scram()
    d = ov.submit("x")
    assert d.halted


def test_pending_action_to_dict():
    ov = _ov()
    d = ov.submit("delete", risk=RiskTier.HIGH)
    pa = ov.get(d.action_id)
    assert pa.to_dict()["risk"] == "high"


def test_report():
    ov = _ov()
    ov.submit("read", risk=RiskTier.LOW)
    ov.submit("delete", risk=RiskTier.HIGH)
    r = ov.report()
    assert r["actions"] == 2 and r["pending"] == 1
    assert r["state"] == "running"
