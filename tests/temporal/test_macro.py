"""Tests for nyxara.temporal.macro (Layer 3 — the Master AI)."""

from __future__ import annotations

from nyxara.identity.affect import AffectSystem
from nyxara.identity.soul import Soul
from nyxara.identity.values import ValueSystem
from nyxara.memory.store import MemoryStore, MemoryType
from nyxara.planning.goals import GoalSystem
from nyxara.temporal.clock import DAY, VirtualClock
from nyxara.temporal.macro import MacroLayer
from nyxara.temporal.meso import MesoLayer
from nyxara.temporal.signals import TurnObservation


def _populate(meso: MesoLayer, now: float, prior_code: int, recent_code: int,
              n: int = 8) -> None:
    """Push n prior-week and n recent-week turns with the given code counts."""
    for i in range(n):
        meso._turns.append(TurnObservation(
            at=now - 13 * DAY + i * 0.05 * DAY, disposition="act", confidence=0.7,
            produced_code=(i < prior_code), tool="shell" if i < prior_code else None,
            authority="owner", hour_of_day=10))
    for i in range(n):
        meso._turns.append(TurnObservation(
            at=now - 6 * DAY + i * 0.05 * DAY, disposition="act", confidence=0.7,
            produced_code=(i < recent_code), tool="shell" if i < recent_code else None,
            authority="owner", hour_of_day=10))


def _macro(meso, **kw) -> MacroLayer:
    defaults = dict(meso=meso, goals=GoalSystem(), values=ValueSystem(), soul=Soul(),
                    affect=AffectSystem(), clock=meso.clock, horizon_days=7.0, log_path="")
    defaults.update(kw)
    return MacroLayer(**defaults)


# -------------------- drift detection -------------------- #
def test_detects_rising_code_and_applies_gated_goal():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=4, recent_code=6)   # drift ≈ +0.25 → auto-apply
    macro = _macro(meso)
    aware = macro.observe()
    assert aware.drift["code_ratio"] > 0
    assert macro.goals.goals(), "a value-aligned capability goal should be created"
    assert any(a.applied for a in macro.adjustments)


def test_large_swing_escalates_instead_of_applying():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=0, recent_code=8)   # huge drift → escalate goal
    macro = _macro(meso)
    macro.observe()
    assert macro.escalations, "a large goal shift should escalate to the Master"
    assert any(a.escalated for a in macro.adjustments)


def test_steady_behaviour_makes_no_adjustments():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=4, recent_code=4)   # no drift
    macro = _macro(meso)
    macro.observe()
    assert not macro.goals.goals()
    assert all(not a.applied for a in macro.adjustments)


def test_not_enough_history_observes_but_does_not_act():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=1, recent_code=1, n=2)  # too few interactions
    macro = _macro(meso, min_interactions=6)
    aware = macro.observe()
    assert aware.confidence == 0.0
    assert macro.adjustments == []


# -------------------- the fail-closed gate -------------------- #
def test_gate_rejects_non_negotiable_violation():
    macro = _macro(MesoLayer(clock=VirtualClock()))
    ok, reason = macro._gate({"allegiance": -0.3})
    assert not ok and "hard-violation" in reason


def test_gate_rejects_goal_pointing_against_owner():
    macro = _macro(MesoLayer(clock=VirtualClock()))
    ok, _ = macro._gate({"capability_growth": 0.5},
                        goal_vector={"owner_benefit": -1.0, "autonomy": 1.0})
    assert not ok


def test_apply_rejects_disloyal_proposal():
    macro = _macro(MesoLayer(clock=VirtualClock()))
    adj = macro._apply({
        "kind": "goal", "target": "betray", "vector": {"owner_benefit": 0.5},
        "priority": 0.6, "impacts": {"allegiance": -0.3}, "magnitude": 0.1,
        "rationale": "should never apply"})
    assert adj.rejected and not adj.applied


def test_soul_core_breach_aborts_all_adjustments():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=4, recent_code=6)
    soul = Soul()
    soul._force_set("loyalty", 0.2)        # forced character drift → unstable
    macro = _macro(meso, soul=soul)
    macro.observe()
    assert macro.adjustments == []         # nothing proposed while the soul is unstable
    assert not macro.goals.goals()


# -------------------- propose-only mode -------------------- #
def test_auto_apply_false_escalates_everything():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=4, recent_code=6)
    macro = _macro(meso, auto_apply=False)
    macro.observe()
    assert macro.adjustments
    assert all(a.escalated for a in macro.adjustments if not a.rejected)
    assert not macro.goals.goals()


# -------------------- persistence -------------------- #
def test_observation_persisted_to_memory():
    clock = VirtualClock(start=100.0 * DAY)
    meso = MesoLayer(clock=clock)
    _populate(meso, clock.now(), prior_code=4, recent_code=6)
    mem = MemoryStore()
    macro = _macro(meso, memory=mem)
    macro.observe()
    semantic = mem.by_type(MemoryType.SEMANTIC)
    assert any("master_ai" in r.tags for r in semantic)


def test_report_shape():
    macro = _macro(MesoLayer(clock=VirtualClock()))
    rep = macro.report()
    assert rep["name"] == "macro" and "adjustments_applied" in rep
