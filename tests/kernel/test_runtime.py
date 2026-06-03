"""Tests for nyxara.kernel.runtime."""

from __future__ import annotations

import asyncio

import pytest

from nyxara.kernel.errors import ExternalServiceError, SecurityError, ValidationError
from nyxara.kernel.runtime import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitRegistry,
    CircuitState,
    LifecycleState,
    Runtime,
    Saga,
    SagaError,
)


# ---------------- circuit breaker ---------------- #
def _cb(**kw):
    clk = {"t": 0.0}
    cfg = CircuitBreakerConfig(**kw)
    cb = CircuitBreaker("x", cfg, clock=lambda: clk["t"])
    return cb, clk


def test_breaker_starts_closed():
    cb, _ = _cb()
    assert cb.state is CircuitState.CLOSED


def test_breaker_trips_after_threshold():
    cb, _ = _cb(failure_threshold=3)

    def boom():
        raise ExternalServiceError("503")

    for _ in range(3):
        with pytest.raises(ExternalServiceError):
            cb.call(boom)
    assert cb.state is CircuitState.OPEN
    assert cb.metrics.trips == 1


def test_open_breaker_fails_fast():
    cb, _ = _cb(failure_threshold=1)
    with pytest.raises(ExternalServiceError):
        cb.call(lambda: (_ for _ in ()).throw(ExternalServiceError("x")))
    assert cb.state is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "nope")
    assert cb.metrics.rejected == 1


def test_breaker_half_open_then_close_on_success():
    cb, clk = _cb(failure_threshold=1, recovery_timeout=10, success_threshold=1)
    with pytest.raises(ExternalServiceError):
        cb.call(lambda: (_ for _ in ()).throw(ExternalServiceError("x")))
    assert cb.state is CircuitState.OPEN
    # before timeout -> still rejected
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "x")
    # after timeout -> probe admitted, success closes
    clk["t"] = 20.0
    assert cb.call(lambda: "ok") == "ok"
    assert cb.state is CircuitState.CLOSED


def test_breaker_half_open_failure_reopens():
    cb, clk = _cb(failure_threshold=1, recovery_timeout=10)
    with pytest.raises(ExternalServiceError):
        cb.call(lambda: (_ for _ in ()).throw(ExternalServiceError("x")))
    clk["t"] = 20.0
    with pytest.raises(ExternalServiceError):
        cb.call(lambda: (_ for _ in ()).throw(ExternalServiceError("again")))
    assert cb.state is CircuitState.OPEN


def test_security_errors_do_not_trip():
    cb, _ = _cb(failure_threshold=2)
    for _ in range(5):
        with pytest.raises(SecurityError):
            cb.call(lambda: (_ for _ in ()).throw(SecurityError("nope")))
    assert cb.state is CircuitState.CLOSED


def test_breaker_success_resets_consecutive():
    cb, _ = _cb(failure_threshold=3)

    def boom():
        raise ExternalServiceError("x")

    for _ in range(2):
        with pytest.raises(ExternalServiceError):
            cb.call(boom)
    cb.call(lambda: "ok")  # resets
    for _ in range(2):
        with pytest.raises(ExternalServiceError):
            cb.call(boom)
    assert cb.state is CircuitState.CLOSED  # didn't reach 3 consecutive


def test_breaker_async():
    cb, _ = _cb(failure_threshold=1)

    async def boom():
        raise ExternalServiceError("x")

    async def go():
        try:
            await cb.acall(boom)
        except ExternalServiceError:
            pass
        assert cb.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await cb.acall(boom)

    asyncio.run(go())


def test_breaker_listener_fires():
    cb, _ = _cb(failure_threshold=1)
    seen = []
    cb.add_listener(lambda n, o, s: seen.append((o.value, s.value)))
    with pytest.raises(ExternalServiceError):
        cb.call(lambda: (_ for _ in ()).throw(ExternalServiceError("x")))
    assert ("closed", "open") in seen


def test_circuit_registry():
    reg = CircuitRegistry()
    a = reg.get("llm")
    b = reg.get("llm")
    assert a is b
    reg.get("tools")
    assert set(reg.health()) == {"llm", "tools"}


# ---------------- saga ---------------- #
def test_saga_all_succeed():
    async def go():
        saga = Saga("ok")
        saga.add_step("a", lambda ctx: 1)
        saga.add_step("b", lambda ctx: ctx["a"] + 1)
        result = await saga.run()
        return result

    res = asyncio.run(go())
    assert res == {"a": 1, "b": 2}


def test_saga_compensates_on_failure():
    async def go():
        applied = []
        saga = Saga("s")
        saga.add_step("reserve", lambda ctx: applied.append("reserve"),
                      lambda ctx: applied.remove("reserve"))
        saga.add_step("charge", lambda ctx: applied.append("charge"),
                      lambda ctx: applied.remove("charge"))
        saga.add_step("ship", lambda ctx: (_ for _ in ()).throw(ExternalServiceError("x")))
        try:
            await saga.run()
        except SagaError as e:
            return applied, e.context
        return None

    applied, ctx = asyncio.run(go())
    assert applied == []  # both compensated
    assert ctx["failed_step"] == "ship"
    assert set(ctx["compensated"]) == {"reserve", "charge"}


def test_saga_async_steps_and_compensation():
    async def go():
        log = []

        async def act(ctx):
            log.append("act")
            return "v"

        async def comp(ctx):
            log.append("comp")

        async def fail(ctx):
            raise ValidationError("boom")

        saga = Saga("a")
        saga.add_step("one", act, comp)
        saga.add_step("two", fail)
        with pytest.raises(SagaError):
            await saga.run()
        return log

    log = asyncio.run(go())
    assert log == ["act", "comp"]


def test_saga_decorator_form():
    async def go():
        saga = Saga("deco")

        @saga.step("a")
        def a(ctx):
            return 10

        result = await saga.run()
        return result

    assert asyncio.run(go()) == {"a": 10}


def test_saga_compensation_failure_keeps_unwinding():
    async def go():
        log = []
        saga = Saga("s")
        saga.add_step("a", lambda ctx: log.append("a"),
                      lambda ctx: (_ for _ in ()).throw(RuntimeError("comp-fail")))
        saga.add_step("b", lambda ctx: log.append("b"),
                      lambda ctx: log.append("comp-b"))
        saga.add_step("c", lambda ctx: (_ for _ in ()).throw(ExternalServiceError("x")))
        with pytest.raises(SagaError):
            await saga.run()
        return log

    log = asyncio.run(go())
    # b's compensation ran even though a's compensation raised
    assert "comp-b" in log


# ---------------- runtime lifecycle ---------------- #
def test_runtime_service_start_stop_order():
    async def go():
        rt = Runtime()
        order = []

        def mk(n):
            async def start():
                order.append(f"start-{n}")
            async def stop():
                order.append(f"stop-{n}")
            return start, stop

        s1, st1 = mk(1)
        s2, st2 = mk(2)
        rt.register_service("a", s1, st1)
        rt.register_service("b", s2, st2)
        await rt.start()
        assert rt.state is LifecycleState.RUNNING
        await rt.stop()
        assert rt.state is LifecycleState.STOPPED
        return order

    order = asyncio.run(go())
    assert order == ["start-1", "start-2", "stop-2", "stop-1"]  # stop is reverse


def test_runtime_start_failure_unwinds():
    async def go():
        rt = Runtime()
        started = []

        async def s1():
            started.append("s1")
        async def st1():
            started.append("stop-s1")
        async def s2():
            raise ExternalServiceError("cannot start")
        async def st2():
            started.append("stop-s2")

        rt.register_service("a", s1, st1)
        rt.register_service("b", s2, st2)
        with pytest.raises(ExternalServiceError):
            await rt.start()
        return started, rt.state

    started, state = asyncio.run(go())
    assert "stop-s1" in started  # partially-started service was rolled back
    assert state is LifecycleState.STOPPED


def test_runtime_spawn_task_runs():
    async def go():
        rt = Runtime()
        done = []
        async with rt:
            rt.spawn(_set_after(done, 1), name="t")
            await asyncio.sleep(0.02)
        return done

    assert asyncio.run(go()) == [1]


async def _set_after(box, v):
    await asyncio.sleep(0.001)
    box.append(v)


def test_runtime_supervise_restarts_until_success():
    async def go():
        rt = Runtime()
        n = {"c": 0}

        async def daemon():
            n["c"] += 1
            if n["c"] < 3:
                raise ExternalServiceError("hiccup")
            await asyncio.sleep(0.005)

        async with rt:
            rt.supervise(daemon, name="d", max_restarts=5, backoff=0.001)
            await asyncio.sleep(0.1)
        return n["c"]

    assert asyncio.run(go()) >= 3


def test_runtime_health_view():
    async def go():
        rt = Runtime()

        async def noop():
            pass

        rt.register_service("svc", noop, noop)
        async with rt:
            rt.breakers.get("llm")
            h = rt.health()
        return h

    h = asyncio.run(go())
    assert h["state"] == "running"  # captured inside the context
    assert "svc" in h["services"]
    assert "llm" in h["breakers"]


def test_runtime_context_manager_stops():
    async def go():
        rt = Runtime()
        flags = []

        async def start():
            flags.append("up")
        async def stop():
            flags.append("down")

        rt.register_service("s", start, stop)
        async with rt:
            assert rt.state is LifecycleState.RUNNING
        assert rt.state is LifecycleState.STOPPED
        return flags

    assert asyncio.run(go()) == ["up", "down"]
