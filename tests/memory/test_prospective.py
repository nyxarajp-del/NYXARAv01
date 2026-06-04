"""Tests for nyxara.memory.prospective."""

from __future__ import annotations

import asyncio

import pytest

from nyxara.memory.prospective import (
    Intention,
    IntentionStatus,
    ProspectiveMemory,
    Trigger,
    TriggerType,
)


# -------------------- triggers -------------------- #
def test_trigger_at_time_ready():
    t = Trigger.at_time(100.0)
    assert not t.ready_time(50.0)
    assert t.ready_time(100.0)
    assert t.ready_time(150.0)


def test_trigger_after():
    t = Trigger.after(10.0, now=100.0)
    assert t.at == 110.0


def test_trigger_every_recurring():
    t = Trigger.every(10.0, now=0.0, max_count=2)
    assert t.ttype is TriggerType.RECURRING
    assert t.at == 10.0
    assert t.advance(10.0) is True   # count=1, reschedule to 20
    assert t.at == 20.0
    assert t.advance(20.0) is False  # count=2 == max -> inactive


def test_trigger_advance_catches_up_missed():
    t = Trigger.every(10.0, now=0.0)
    t.advance(35.0)  # jumped ahead; next should be > 35
    assert t.at > 35.0


def test_trigger_event_kind_match():
    t = Trigger.on_event("security.intrusion")

    class Ev:
        topic = "security.intrusion"

    assert t.ready_event(Ev())

    class Other:
        topic = "weather.update"

    assert not t.ready_event(Other())


def test_trigger_event_dict_match():
    t = Trigger.on_event("alarm")
    assert t.ready_event({"kind": "alarm"})
    assert not t.ready_event({"kind": "calm"})


def test_trigger_event_predicate():
    t = Trigger.on_event(match=lambda e: getattr(e, "level", 0) > 5)

    class E:
        level = 9

    assert t.ready_event(E())

    class E2:
        level = 1

    assert not t.ready_event(E2())


def test_trigger_context_match():
    t = Trigger.in_context(lambda ctx: ctx.get("mood") == "anxious")
    assert t.ready_context({"mood": "anxious"})
    assert not t.ready_context({"mood": "calm"})


def test_trigger_predicate_exception_safe():
    t = Trigger.on_event(match=lambda e: e.missing_attr)
    assert t.ready_event(object()) is False


# -------------------- intention firing -------------------- #
def test_time_intention_fires():
    pm = ProspectiveMemory()
    out = []
    pm.add("ping", Trigger.at_time(100.0), action=lambda i, now: out.append("pinged"))
    assert pm.tick(now=50.0) == []
    fired = pm.tick(now=100.0)
    assert len(fired) == 1
    assert out == ["pinged"]


def test_intention_completes_after_time_fire():
    pm = ProspectiveMemory()
    i = pm.add("once", Trigger.at_time(10.0))
    pm.tick(now=10.0)
    assert i.status is IntentionStatus.COMPLETED
    # does not fire again
    assert pm.tick(now=20.0) == []


def test_recurring_fires_capped():
    pm = ProspectiveMemory()
    out = []
    pm.every(10.0, "beat", now=0.0, max_count=3, action=lambda i, now: out.append(now))
    for t in (10.0, 20.0, 30.0, 40.0, 50.0):
        pm.tick(now=t)
    assert len(out) == 3


def test_recurring_reschedules_and_stays_pending():
    pm = ProspectiveMemory()
    i = pm.every(10.0, "beat", now=0.0)  # unbounded
    pm.tick(now=10.0)
    assert i.status is IntentionStatus.PENDING
    assert i.fired_count == 1


def test_event_intention_fires():
    pm = ProspectiveMemory()
    out = []
    pm.remind_when("intrusion", action=lambda i, now: out.append("shields"))

    class Ev:
        topic = "intrusion"

    fired = pm.on_event(Ev())
    assert len(fired) == 1
    assert out == ["shields"]


def test_context_intention_fires():
    pm = ProspectiveMemory()
    out = []
    pm.remind_in_context(lambda ctx: ctx.get("threat"), action=lambda i, now: out.append("scan"))
    assert pm.tick(now=1.0, context={"threat": False}) == []
    fired = pm.tick(now=2.0, context={"threat": True})
    assert len(fired) == 1
    assert out == ["scan"]


def test_action_exception_does_not_crash():
    pm = ProspectiveMemory()
    pm.add("boom", Trigger.at_time(10.0),
           action=lambda i, now: (_ for _ in ()).throw(RuntimeError("x")))
    fired = pm.tick(now=10.0)
    assert len(fired) == 1
    assert "error" in str(fired[0].result)


# -------------------- deadlines & expiry -------------------- #
def test_deadline_expiry():
    pm = ProspectiveMemory()
    i = pm.add("late task", Trigger.at_time(1000.0), deadline=500.0)
    pm.tick(now=600.0)  # past deadline, before trigger
    assert i.status is IntentionStatus.EXPIRED


def test_deadline_not_expired_if_fires_first():
    pm = ProspectiveMemory()
    out = []
    i = pm.add("ok", Trigger.at_time(100.0), deadline=200.0,
              action=lambda i, now: out.append("done"))
    pm.tick(now=150.0)  # trigger ready and within deadline
    assert i.status is IntentionStatus.COMPLETED
    assert out == ["done"]


def test_overdue_list():
    pm = ProspectiveMemory()
    pm.add("x", Trigger.manual(), deadline=100.0)
    assert pm.overdue(now=50.0) == []
    assert len(pm.overdue(now=200.0)) == 1


# -------------------- salience / superiority -------------------- #
def test_urgency_rises_toward_deadline():
    i = Intention("x", Trigger.at_time(1000.0), deadline=1000.0, horizon_s=1000.0)
    assert i.urgency(0.0) < i.urgency(900.0)
    assert i.urgency(1000.0) == 1.0
    assert i.urgency(1100.0) == 1.0  # overdue


def test_salience_blends_importance_and_urgency():
    i = Intention("x", Trigger.at_time(100.0), deadline=100.0, horizon_s=100.0,
                  importance=0.5)
    assert i.salience(0.0) < i.salience(95.0)


def test_most_salient_ordering():
    pm = ProspectiveMemory()
    far = pm.add("far", Trigger.at_time(10000.0), deadline=10000.0, horizon_s=10000.0,
                 importance=0.5)
    near = pm.add("near", Trigger.at_time(1000.0), deadline=1000.0, horizon_s=10000.0,
                  importance=0.5)
    top = pm.most_salient(now=950.0, k=2)
    assert top[0] is near


def test_event_intention_steady_urgency():
    i = Intention("evt", Trigger.on_event("x"), importance=0.8)
    assert i.urgency(123.0) == 0.25  # no deadline -> quiet hum


# -------------------- lifecycle -------------------- #
def test_cancel():
    pm = ProspectiveMemory()
    i = pm.add("x", Trigger.at_time(100.0))
    assert pm.cancel(i.intention_id) is True
    assert i.status is IntentionStatus.CANCELLED
    assert pm.tick(now=100.0) == []  # cancelled does not fire


def test_complete():
    pm = ProspectiveMemory()
    i = pm.add("x", Trigger.manual())
    assert pm.complete(i.intention_id) is True
    assert i.status is IntentionStatus.COMPLETED


def test_pending_and_due():
    pm = ProspectiveMemory()
    pm.add("a", Trigger.at_time(100.0))
    pm.add("b", Trigger.at_time(500.0))
    assert len(pm.pending()) == 2
    assert len(pm.due(now=200.0)) == 1


def test_stats():
    pm = ProspectiveMemory()
    pm.add("a", Trigger.at_time(10.0))
    pm.tick(now=10.0)
    s = pm.stats()
    assert s["total"] == 1
    assert s["by_status"]["completed"] == 1


def test_on_fire_callback():
    fired_log = []
    pm = ProspectiveMemory(on_fire=lambda f: fired_log.append(f))
    pm.add("x", Trigger.at_time(10.0))
    pm.tick(now=10.0)
    assert len(fired_log) == 1


# -------------------- bus integration -------------------- #
def test_bus_emits_on_fire():
    from nyxara.kernel.bus import EventBus

    async def go():
        bus = EventBus(workers=1)
        got = []
        bus.subscribe("prospective.fired", lambda ev: got.append(ev.payload))
        pm = ProspectiveMemory(bus=bus)
        pm.add("x", Trigger.at_time(10.0))
        async with bus:
            pm.tick(now=10.0)
            await bus.drain()
        return got

    got = asyncio.run(go())
    assert len(got) == 1
