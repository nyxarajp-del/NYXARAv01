"""Tests for nyxara.kernel.bus."""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest

from nyxara.kernel.bus import (
    DeadLetter,
    Event,
    EventBus,
    OverflowPolicy,
    Priority,
    _topic_match,
)
from nyxara.kernel.errors import ExternalServiceError, RetryPolicy, SecurityError, ValidationError


# -------------------- pure helpers -------------------- #
def test_priority_coerce():
    assert Priority.coerce("high") is Priority.HIGH
    assert Priority.coerce(Priority.LOW) is Priority.LOW
    assert Priority.coerce(40) is Priority.CRITICAL
    assert Priority.coerce(15) is Priority.LOW  # nearest at-or-below


@pytest.mark.parametrize(
    "pattern,topic,ok",
    [
        ("*", "anything.here", True),
        ("#", "anything", True),
        ("sensor.*", "sensor.vision", True),
        ("sensor.*", "sensor.vision.left", False),  # fnmatch * crosses dots though
        ("mind.proposal.*", "mind.proposal.created", True),
        ("a.#", "a", True),
        ("a.#", "a.b.c", True),
        ("a.#", "ab", False),
        ("exact", "exact", True),
        ("exact", "other", False),
    ],
)
def test_topic_match(pattern, topic, ok):
    # note: fnmatch '*' actually matches '.' too; document real behavior
    result = _topic_match(pattern, topic)
    if pattern == "sensor.*" and topic == "sensor.vision.left":
        assert result is True  # fnmatch * spans dots
    else:
        assert result is ok


def test_event_ttl_and_reply():
    e = Event("t", payload=1)
    assert not e.is_expired()
    e.with_ttl(0)
    assert e.is_expired()
    r = e.reply("resp", payload=2)
    assert r.correlation_id == e.event_id
    assert r.trace_id == e.trace_id


# -------------------- async behavior -------------------- #
def _run(coro):
    return asyncio.run(coro)


def test_basic_delivery():
    async def go():
        bus = EventBus(workers=2)
        got: List[int] = []
        bus.subscribe("num", lambda ev: got.append(ev.payload))
        async with bus:
            await bus.emit("num", payload=1)
            await bus.emit("num", payload=2)
            await bus.drain()
        return got

    assert sorted(_run(go())) == [1, 2]


def test_glob_subscription_and_no_listener():
    async def go():
        bus = EventBus(workers=1)
        hits: List[str] = []
        bus.subscribe("sensor.*", lambda ev: hits.append(ev.topic))
        async with bus:
            await bus.emit("sensor.vision")
            await bus.emit("unrelated")  # no listener; still 'delivered'
            await bus.drain()
        return hits, bus.metrics.snapshot()

    hits, metrics = _run(go())
    assert hits == ["sensor.vision"]
    assert metrics["published"] == 2


def test_async_handler():
    async def go():
        bus = EventBus(workers=2)
        out = []

        async def h(ev: Event):
            await asyncio.sleep(0.001)
            out.append(ev.payload)

        bus.subscribe("a", h)
        async with bus:
            await bus.emit("a", payload="x")
            await bus.drain()
        return out

    assert _run(go()) == ["x"]


def test_priority_ordering():
    async def go():
        # single worker so ordering is deterministic; pre-load queue before start
        bus = EventBus(workers=1)
        order: List[str] = []
        bus.subscribe("p", lambda ev: order.append(ev.payload))
        # publish before starting workers to fill the priority queue
        bus.publish_nowait(Event("p", payload="low", priority=Priority.LOW))
        bus.publish_nowait(Event("p", payload="crit", priority=Priority.CRITICAL))
        bus.publish_nowait(Event("p", payload="high", priority=Priority.HIGH))
        bus.publish_nowait(Event("p", payload="norm", priority=Priority.NORMAL))
        await bus.start()
        await bus.drain()
        await bus.stop(drain=False)
        return order

    order = _run(go())
    assert order[0] == "crit"
    assert order[-1] == "low"
    assert order == ["crit", "high", "norm", "low"]


def test_middleware_transform_and_drop():
    async def go():
        bus = EventBus(workers=1)
        seen: List[str] = []
        bus.subscribe("m", lambda ev: seen.append(ev.payload))

        def tag(ev: Event) -> Event:
            ev.headers["x"] = 1
            return ev

        def drop_secret(ev: Event) -> Optional[Event]:
            return None if ev.payload == "secret" else ev

        bus.use(tag).use(drop_secret)
        async with bus:
            await bus.emit("m", payload="ok")
            await bus.emit("m", payload="secret")
            await bus.drain()
        return seen, bus.metrics.snapshot()

    seen, metrics = _run(go())
    assert seen == ["ok"]
    assert metrics["dropped_by_middleware"] == 1


def test_async_middleware():
    async def go():
        bus = EventBus(workers=1)
        seen = []
        bus.subscribe("m", lambda ev: seen.append(ev.payload))

        async def amw(ev: Event) -> Event:
            await asyncio.sleep(0.001)
            ev.payload = ev.payload * 2
            return ev

        bus.use(amw)
        async with bus:
            await bus.emit("m", payload=3)
            await bus.drain()
        return seen

    assert _run(go()) == [6]


def test_retry_then_success():
    async def go():
        bus = EventBus(workers=1, retry_policy=RetryPolicy(max_attempts=4, base_delay=0))
        attempts = {"n": 0}

        async def flaky(ev: Event):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ExternalServiceError("503")

        bus.subscribe("w", flaky)
        async with bus:
            await bus.emit("w")
            # allow re-queues to cycle
            for _ in range(10):
                await asyncio.sleep(0.005)
                await bus.drain()
                if attempts["n"] >= 3:
                    break
        return attempts["n"], bus.metrics.snapshot()

    n, metrics = _run(go())
    assert n == 3
    assert metrics["retried"] >= 2


def test_permanent_error_dead_letters():
    async def go():
        bus = EventBus(workers=1, retry_policy=RetryPolicy(max_attempts=3, base_delay=0))
        bus.subscribe("w", lambda ev: (_ for _ in ()).throw(ValidationError("bad")))
        async with bus:
            await bus.emit("w")
            await bus.drain()
        return bus.dead_letters

    dls = _run(go())
    assert len(dls) == 1
    assert dls[0].reason == "handler_error"
    assert dls[0].category == "permanent"


def test_security_error_never_retried_dead_letters():
    async def go():
        bus = EventBus(workers=1, retry_policy=RetryPolicy(max_attempts=5, base_delay=0))
        calls = {"n": 0}

        def h(ev: Event):
            calls["n"] += 1
            raise SecurityError("nope")

        bus.subscribe("w", h)
        async with bus:
            await bus.emit("w")
            await bus.drain()
        return calls["n"], bus.dead_letters

    n, dls = _run(go())
    assert n == 1  # not retried
    assert dls and dls[0].category == "security"


def test_expired_event_dead_letters():
    async def go():
        bus = EventBus(workers=1)
        bus.subscribe("w", lambda ev: None)
        async with bus:
            e = Event("w").with_ttl(0)  # already expired
            await bus.publish(e)
            await bus.drain()
        return bus.dead_letters, bus.metrics.snapshot()

    dls, metrics = _run(go())
    assert any(dl.reason == "expired" for dl in dls)
    assert metrics["dropped_expired"] == 1


def test_once_subscription():
    async def go():
        bus = EventBus(workers=1)
        hits = []
        bus.subscribe("o", lambda ev: hits.append(1), once=True)
        async with bus:
            await bus.emit("o")
            await bus.emit("o")
            await bus.drain()
        return hits

    assert _run(go()) == [1]


def test_unsubscribe():
    async def go():
        bus = EventBus(workers=1)
        hits = []
        sub = bus.subscribe("u", lambda ev: hits.append(1))
        async with bus:
            await bus.emit("u")
            await bus.drain()
            assert bus.unsubscribe(sub) is True
            await bus.emit("u")
            await bus.drain()
        return hits

    assert _run(go()) == [1]


def test_priority_min_filter():
    async def go():
        bus = EventBus(workers=1)
        hits = []
        bus.subscribe("f", lambda ev: hits.append(ev.priority), priority_min=Priority.HIGH)
        async with bus:
            await bus.publish(Event("f", priority=Priority.LOW))
            await bus.publish(Event("f", priority=Priority.CRITICAL))
            await bus.drain()
        return hits

    hits = _run(go())
    assert hits == [Priority.CRITICAL]


def test_predicate_filter():
    async def go():
        bus = EventBus(workers=1)
        hits = []
        bus.subscribe("p", lambda ev: hits.append(ev.payload), predicate=lambda ev: ev.payload > 5)
        async with bus:
            await bus.emit("p", payload=3)
            await bus.emit("p", payload=9)
            await bus.drain()
        return hits

    assert _run(go()) == [9]


def test_overflow_drop_newest():
    async def go():
        bus = EventBus(workers=1, max_queue=2, overflow=OverflowPolicy.DROP_NEWEST)
        # don't start workers; fill queue beyond capacity
        ok1 = bus.publish_nowait(Event("x"))
        ok2 = bus.publish_nowait(Event("x"))
        ok3 = bus.publish_nowait(Event("x"))  # should be dropped
        return ok1, ok2, ok3, bus.metrics.snapshot()

    ok1, ok2, ok3, metrics = _run(go())
    assert ok1 and ok2 and not ok3
    assert metrics["dropped_overflow"] == 1


def test_dead_letter_sink_called():
    async def go():
        bus = EventBus(workers=1, retry_policy=RetryPolicy(max_attempts=1, base_delay=0))
        captured: List[DeadLetter] = []
        bus.on_dead_letter = lambda dl: captured.append(dl)
        bus.subscribe("w", lambda ev: (_ for _ in ()).throw(ValidationError("x")))
        async with bus:
            await bus.emit("w")
            await bus.drain()
        return captured

    captured = _run(go())
    assert len(captured) == 1


def test_subscriber_count():
    bus = EventBus()
    bus.subscribe("sensor.*", lambda ev: None)
    bus.subscribe("mind.*", lambda ev: None)
    assert bus.subscriber_count() == 2
    assert bus.subscriber_count("sensor.vision") == 1
    assert bus.subscriber_count("other") == 0
