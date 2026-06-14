"""Tests for nyxara.agency.proactive."""

from __future__ import annotations

from nyxara.agency.permissions import Capability, RiskTier, build_default_policy
from nyxara.agency.proactive import (Initiative, ProactiveEngine, TriggerKind, Verdict)
from nyxara.agency.scheduler import Priority, Scheduler, TaskState
from nyxara.kernel.config import AgencyConfig
from nyxara.planning.goals import GoalSystem


def _engine(scheduler=None, **agency_kw):
    base = dict(initiative_confidence_threshold=0.7,
                min_reversibility_for_autonomy=0.5,
                sandbox_before_real_action=True)
    base.update(agency_kw)
    return ProactiveEngine(scheduler=scheduler, agency=AgencyConfig(**base))


def _safe(**kw):
    base = dict(name="safe", rationale="r", capability=Capability.FS_WRITE,
                risk=RiskTier.LOW, reversibility=0.9, confidence=0.9,
                benefit={"owner_benefit": 1.0})
    base.update(kw)
    return Initiative(**base)


# -------------------- gate 1: owner alignment -------------------- #
def test_misaligned_initiative_rejected():
    eng = _engine()
    ini = _safe(name="grab", benefit={"owner_benefit": -1.0, "autonomy": 1.0})
    d = eng.evaluate(ini)
    assert d.verdict is Verdict.REJECT
    assert "against the Master" in d.reason


def test_aligned_initiative_passes_loyalty():
    eng = _engine()
    d = eng.evaluate(_safe())
    assert d.verdict is Verdict.ACT


def test_neutral_benefit_not_rejected():
    eng = _engine()
    d = eng.evaluate(_safe(benefit={"knowledge": 1.0}))
    assert d.verdict is not Verdict.REJECT


# -------------------- gate 2: confidence -------------------- #
def test_low_confidence_deferred():
    eng = _engine()
    d = eng.evaluate(_safe(confidence=0.5))
    assert d.verdict is Verdict.DEFER
    assert "confidence" in d.reason


def test_confidence_at_threshold_acts():
    eng = _engine(initiative_confidence_threshold=0.7)
    d = eng.evaluate(_safe(confidence=0.7))
    assert d.verdict is Verdict.ACT


# -------------------- gate 3: permission -------------------- #
def test_owner_exclusive_capability_rejected():
    eng = _engine()
    d = eng.evaluate(_safe(capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL,
                           confidence=0.95))
    assert d.verdict is Verdict.REJECT
    assert d.permission["rule_basis"] == "Rule 8"


def test_high_risk_irreversible_escalates_via_permission():
    eng = _engine()
    d = eng.evaluate(_safe(name="del", capability=Capability.FS_DELETE,
                           risk=RiskTier.HIGH, reversibility=0.9, confidence=0.95))
    # FS_DELETE is irreversible-by-nature -> permission escalates
    assert d.verdict is Verdict.ESCALATE


def test_defense_high_risk_reversible_acts():
    eng = _engine()
    d = eng.evaluate(_safe(name="block", capability=Capability.NET_DEFENSE,
                           risk=RiskTier.HIGH, reversibility=0.9, confidence=0.95,
                           benefit={"owner_safety": 1.0}))
    assert d.verdict is Verdict.ACT


# -------------------- gate 4: reversibility -------------------- #
def test_low_reversibility_escalates():
    eng = _engine()
    # NET_DEFENSE allows high risk, but low reversibility must still escalate
    d = eng.evaluate(_safe(name="x", capability=Capability.NET_DEFENSE,
                           risk=RiskTier.HIGH, reversibility=0.2, confidence=0.95,
                           benefit={"owner_safety": 1.0}))
    assert d.verdict is Verdict.ESCALATE  # low reversibility -> awaits the Master


def test_reversibility_floor_holds_even_with_a_grant():
    # even when a standing grant lets permission ALLOW an irreversible capability,
    # NYXARA's own autonomy floor still escalates a hard-to-undo action (defense in depth)
    policy = build_default_policy()
    policy.grant("send-ok", Capability.MESSAGE_SEND, max_risk=RiskTier.HIGH,
                 reversible_only=False)
    eng = ProactiveEngine(policy=policy, agency=AgencyConfig(
        initiative_confidence_threshold=0.7, min_reversibility_for_autonomy=0.5,
        sandbox_before_real_action=True))
    d = eng.evaluate(_safe(name="y", capability=Capability.MESSAGE_SEND,
                           risk=RiskTier.LOW, reversibility=0.2, confidence=0.95))
    assert d.verdict is Verdict.ESCALATE
    assert "reversibility" in d.reason


def test_reversibility_at_floor_acts():
    eng = _engine(min_reversibility_for_autonomy=0.5)
    d = eng.evaluate(_safe(reversibility=0.5))
    assert d.verdict is Verdict.ACT


# -------------------- gate 5: sandbox-first -------------------- #
def test_sandbox_irreversible_escalates():
    eng = _engine()

    def dry(ctx):
        ctx.run_command("rm -rf /")  # an irreversible effect in the sandbox

    d = eng.evaluate(_safe(name="risky", dry_run=dry))
    assert d.verdict is Verdict.ESCALATE
    assert "irreversible" in d.reason
    assert len(d.effects) >= 1


def test_sandbox_reversible_acts_and_records_effects():
    eng = _engine()

    def dry(ctx):
        ctx.write_file("/tmp/x", "data")  # reversible effect

    d = eng.evaluate(_safe(name="ok", dry_run=dry))
    assert d.verdict is Verdict.ACT
    assert len(d.effects) == 1


def test_sandbox_skipped_when_disabled():
    eng = _engine(sandbox_before_real_action=False)

    def dry(ctx):
        ctx.run_command("rm -rf /")

    d = eng.evaluate(_safe(name="ok", dry_run=dry))
    assert d.verdict is Verdict.ACT
    assert d.effects == []  # sandbox not run


# -------------------- detect / consider -------------------- #
def test_detect_runs_detectors():
    eng = _engine()
    eng.register_detector(lambda ctx: [_safe()] if ctx.get("go") else [])
    assert len(eng.detect({"go": True})) == 1
    assert len(eng.detect({})) == 0


def test_consider_schedules_acted():
    sched = Scheduler(max_concurrent=4)
    eng = _engine(scheduler=sched)
    eng.register_detector(lambda ctx: [_safe()])
    decisions = eng.consider({})
    acted = [d for d in decisions if d.acted][0]
    assert acted.scheduled_task_id is not None
    assert sched.get(acted.scheduled_task_id).state is TaskState.PENDING


def test_consider_no_schedule_when_autostart_false():
    sched = Scheduler()
    eng = _engine(scheduler=sched)
    eng.register_detector(lambda ctx: [_safe()])
    decisions = eng.consider({}, autostart=False)
    assert decisions[0].scheduled_task_id is None


def test_threat_scheduled_at_critical_priority():
    sched = Scheduler()
    eng = _engine(scheduler=sched)
    eng.register_detector(lambda ctx: [_safe(
        name="defend", kind=TriggerKind.THREAT, capability=Capability.NET_DEFENSE,
        risk=RiskTier.HIGH, reversibility=0.9, confidence=0.95,
        benefit={"owner_safety": 1.0}, priority=Priority.NORMAL)])
    d = [x for x in eng.consider({}) if x.acted][0]
    assert sched.get(d.scheduled_task_id).priority is Priority.CRITICAL


def test_consider_orders_acts_first():
    eng = _engine()
    eng.register_detector(lambda ctx: [
        _safe(name="defer-me", confidence=0.4),
        _safe(name="act-me"),
    ])
    decisions = eng.consider({})
    assert decisions[0].initiative.name == "act-me"  # ACT sorts before DEFER


# -------------------- reporting -------------------- #
def test_history_and_report():
    eng = _engine()
    eng.evaluate(_safe())
    eng.evaluate(_safe(name="g", benefit={"owner_benefit": -1.0}))
    r = eng.report()
    assert r["evaluated"] == 2
    assert r["by_verdict"]["act"] == 1 and r["by_verdict"]["reject"] == 1
    assert len(eng.history()) == 2


def test_escalations_collected():
    eng = _engine()
    eng.evaluate(_safe(name="del", capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                       confidence=0.95))
    assert len(eng.escalations()) == 1


def test_initiative_to_dict():
    d = _safe().to_dict()
    assert d["name"] == "safe" and d["capability"] == "fs.write"


def test_decision_to_dict():
    eng = _engine()
    d = eng.evaluate(_safe()).to_dict()
    assert d["verdict"] == "act" and "owner_alignment" in d


def test_default_engine_uses_settings():
    # constructing without explicit agency config pulls from settings (smoke)
    eng = ProactiveEngine()
    assert eng.agency is not None
    assert isinstance(eng.goals, GoalSystem)
