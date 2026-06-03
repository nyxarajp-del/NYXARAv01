"""Tests for nyxara.kernel.presence."""

from __future__ import annotations

import asyncio

from nyxara.kernel.presence import Presence, PresenceState, StateProfile, Transition


def test_state_arousal_ordering():
    assert PresenceState.ASLEEP.arousal < PresenceState.IDLE.arousal
    assert PresenceState.IDLE.arousal < PresenceState.ALERT.arousal
    assert PresenceState.ALERT.arousal < PresenceState.ENGAGED.arousal


def test_profiles_exist_for_all_states():
    for s in PresenceState:
        p = Presence(initial=s, now=0.0)
        assert isinstance(p.profile, StateProfile)
        assert p.profile.state is s


def test_owner_input_forces_engaged():
    p = Presence(initial=PresenceState.ASLEEP, now=0.0)
    changed = p.on_owner_input(now=1.0)
    assert changed
    assert p.state is PresenceState.ENGAGED


def test_owner_present_keeps_engaged_despite_timeout():
    p = Presence(initial=PresenceState.IDLE, now=0.0, engaged_timeout_s=5.0)
    p.on_owner_input(now=0.0)
    # tick far past the engaged timeout; owner still present -> stay engaged
    for t in range(1, 100, 5):
        p.tick(now=float(t))
    assert p.state is PresenceState.ENGAGED


def test_threat_escalates_to_alert():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    p.on_threat(salience=0.9, now=1.0)
    assert p.state is PresenceState.ALERT
    assert p._threat_active is True


def test_loud_stimulus_wakes_from_sleep():
    p = Presence(initial=PresenceState.ASLEEP, now=0.0)
    assert p.on_stimulus(0.9, now=1.0) is True
    assert p.state is PresenceState.ALERT


def test_quiet_stimulus_does_not_wake():
    p = Presence(initial=PresenceState.ASLEEP, now=0.0)
    assert p.on_stimulus(0.3, now=1.0) is False
    assert p.state is PresenceState.ASLEEP


def test_engaged_timeout_deescalates_to_idle():
    p = Presence(initial=PresenceState.ENGAGED, now=0.0, engaged_timeout_s=10.0)
    # no activity; tick beyond timeout
    tr = None
    for t in range(1, 30):
        tr = p.tick(now=float(t))
        if p.state is PresenceState.IDLE:
            break
    assert p.state is PresenceState.IDLE


def test_idle_sleeps_after_timeout():
    p = Presence(initial=PresenceState.IDLE, now=0.0, idle_sleep_timeout_s=20.0,
                 target_awake_s=10_000)
    for t in range(1, 60):
        p.tick(now=float(t))
        if p.state is PresenceState.ASLEEP:
            break
    assert p.state is PresenceState.ASLEEP
    assert p.metrics.sleeps == 1


def test_energy_recovers_while_asleep():
    p = Presence(initial=PresenceState.ASLEEP, energy=0.2, now=0.0)
    p.tick(now=10.0)
    assert p.energy > 0.2


def test_energy_drains_while_engaged():
    p = Presence(initial=PresenceState.ENGAGED, energy=0.9, now=0.0)
    p.tick(now=10.0)
    assert p.energy < 0.9


def test_fatigue_drops_engaged_without_latch():
    p = Presence(initial=PresenceState.ENGAGED, energy=0.11, now=0.0, critical_energy=0.1)
    p.tick(now=1.0)
    state = p.state
    for t in range(2, 10):
        p.tick(now=float(t))
        if p.state is not PresenceState.ENGAGED:
            break
    assert p.state is not PresenceState.ENGAGED


def test_owner_overrides_fatigue():
    p = Presence(initial=PresenceState.IDLE, energy=0.0, now=0.0)
    p.on_owner_input(now=0.0)
    # even with zero energy, owner presence keeps us engaged
    p.tick(now=1.0)
    assert p.state is PresenceState.ENGAGED


def test_resource_budget_scales_with_energy():
    p_full = Presence(initial=PresenceState.ENGAGED, energy=1.0, now=0.0)
    p_low = Presence(initial=PresenceState.ENGAGED, energy=0.0, now=0.0)
    assert p_full.resource_budget() > p_low.resource_budget()


def test_attention_threshold_scales_by_state():
    asleep = Presence(initial=PresenceState.ASLEEP, now=0.0)
    alert = Presence(initial=PresenceState.ALERT, now=0.0)
    # asleep demands far more salience to grab attention than alert
    assert asleep.attention_threshold(1.0) > alert.attention_threshold(1.0)


def test_sleep_pressure_rises_with_wakefulness():
    p = Presence(initial=PresenceState.IDLE, now=0.0, target_awake_s=100.0)
    early = p.sleep_pressure(now=10.0)
    late = p.sleep_pressure(now=90.0)
    assert late > early


def test_task_lifecycle():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    p.task_started(now=1.0)
    assert p.state is PresenceState.ENGAGED
    p.task_finished(now=2.0)
    assert p.state is PresenceState.IDLE


def test_task_finished_with_active_threat_goes_alert():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    p.on_threat(0.9, now=0.0)
    p.task_started(now=1.0)
    p.task_finished(now=2.0)
    assert p.state is PresenceState.ALERT


def test_invalid_transition_blocked_without_force():
    p = Presence(initial=PresenceState.ENGAGED, now=0.0)
    # ENGAGED cannot jump straight to ASLEEP except via force
    assert p._transition(PresenceState.ASLEEP, "x", 1.0, force=False) is False
    assert p.state is PresenceState.ENGAGED
    assert p._transition(PresenceState.ASLEEP, "x", 1.0, force=True) is True
    assert p.state is PresenceState.ASLEEP


def test_listener_and_history_recorded():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    seen = []
    p.add_listener(lambda tr: seen.append(tr))
    p.on_owner_input(now=1.0)
    assert len(seen) == 1
    assert isinstance(seen[0], Transition)
    assert p.history[-1].to_state is PresenceState.ENGAGED


def test_listener_exception_isolated():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    ok = []
    p.add_listener(lambda tr: (_ for _ in ()).throw(RuntimeError("x")))
    p.add_listener(lambda tr: ok.append(1))
    p.on_owner_input(now=1.0)
    assert ok == [1]


def test_status_keys():
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    st = p.status(now=0.0)
    for k in ("state", "arousal", "energy", "sleep_pressure", "resource_budget"):
        assert k in st


def test_bus_emits_transition():
    from nyxara.kernel.bus import EventBus

    async def go():
        bus = EventBus(workers=1)
        got = []
        bus.subscribe("presence.transition", lambda ev: got.append(ev.payload))
        p = Presence(initial=PresenceState.IDLE, bus=bus, now=0.0)
        async with bus:
            p.on_owner_input(now=1.0)
            await bus.drain()
        return got

    got = asyncio.run(go())
    assert len(got) == 1
    assert isinstance(got[0], Transition)


def test_is_awake():
    assert Presence(initial=PresenceState.IDLE, now=0.0).is_awake()
    assert not Presence(initial=PresenceState.ASLEEP, now=0.0).is_awake()
