"""NYXARA · kernel/runtime.py — async execution engine.

The substrate the orchestrator runs on. Three cooperating mechanisms:

1. **CircuitBreaker** — wraps any flaky dependency (an LLM provider, a tool, the
   network) in the classic CLOSED → OPEN → HALF_OPEN state machine. After enough
   failures it *trips open* and fails fast (no hammering a dead service); after a
   cooldown it admits a probe; a success closes it again. Fail-closed errors
   (security/invariant) pass straight through and never count as "faculty health".

2. **Saga** — multi-step actions with **compensation**. If step *k* fails, the
   already-completed steps are rolled back in reverse via their compensations, so
   the system never strands itself in a half-applied state. (Used by agency/ for
   real-world actions and by growth/ for staged self-edits.)

3. **Runtime** — lifecycle + supervision: ordered start/stop of services, spawning
   of tracked one-shot tasks and **auto-restarting** supervised daemons (with
   backoff), graceful shutdown, and a health view. Every uncaught task error is
   recorded to the learn-once LEDGER.

Depends on :mod:`config` and :mod:`errors`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from nyxara.kernel.errors import (
    LEDGER,
    ErrorCategory,
    NyxaraError,
    ResourceExhausted,
    classify,
)

__all__ = [
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitRegistry",
    "SagaStep",
    "SagaError",
    "Saga",
    "LifecycleState",
    "ServiceRef",
    "TaskHandle",
    "Runtime",
]

T = TypeVar("T")


# =========================================================================== #
# Circuit breaker
# =========================================================================== #
class CircuitState(str, Enum):
    CLOSED = "closed"        # healthy — calls flow
    OPEN = "open"            # tripped — calls fail fast
    HALF_OPEN = "half_open"  # probing — limited calls allowed


class CircuitOpenError(NyxaraError):
    """Raised when a call is rejected because its circuit is OPEN."""

    default_category = ErrorCategory.EXTERNAL


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5       # consecutive failures to trip OPEN
    recovery_timeout: float = 30.0   # seconds OPEN before a probe is admitted
    success_threshold: int = 1       # successes in HALF_OPEN to close
    half_open_max: int = 1           # concurrent probes allowed in HALF_OPEN
    # categories that count toward faculty ill-health (fail-closed never counts)
    count_categories: Tuple[ErrorCategory, ...] = (
        ErrorCategory.TRANSIENT, ErrorCategory.EXTERNAL,
        ErrorCategory.RESOURCE, ErrorCategory.UNKNOWN, ErrorCategory.PERMANENT,
    )


@dataclass
class _CBMetrics:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    rejected: int = 0
    trips: int = 0

    def snapshot(self) -> Dict[str, int]:
        return dict(self.__dict__)


StateListener = Callable[[str, CircuitState, CircuitState], None]


class CircuitBreaker:
    """Per-dependency circuit breaker (async + sync)."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._half_open_inflight = 0
        self._opened_at = 0.0
        self.metrics = _CBMetrics()
        self._listeners: List[StateListener] = []

    @property
    def state(self) -> CircuitState:
        return self._state

    def add_listener(self, fn: StateListener) -> None:
        self._listeners.append(fn)

    def _set_state(self, new: CircuitState) -> None:
        if new is self._state:
            return
        old = self._state
        self._state = new
        if new is CircuitState.OPEN:
            self._opened_at = self._clock()
            self.metrics.trips += 1
        if new is CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            self._half_open_inflight = 0
        if new is CircuitState.CLOSED:
            self._consecutive_failures = 0
        for fn in list(self._listeners):
            try:
                fn(self.name, old, new)
            except Exception:  # noqa: BLE001
                pass

    def _admit(self) -> bool:
        """Decide whether a call may proceed *now* (and do lazy OPEN->HALF_OPEN)."""
        if self._state is CircuitState.CLOSED:
            return True
        if self._state is CircuitState.OPEN:
            if self._clock() - self._opened_at >= self.config.recovery_timeout:
                self._set_state(CircuitState.HALF_OPEN)
            else:
                return False
        # HALF_OPEN
        if self._half_open_inflight < self.config.half_open_max:
            self._half_open_inflight += 1
            return True
        return False

    def _should_count(self, exc: BaseException) -> bool:
        cat = exc.category if isinstance(exc, NyxaraError) else classify(exc)
        if cat.fail_closed:
            return False
        if isinstance(exc, CircuitOpenError):
            return False
        return cat in self.config.count_categories

    def _on_success(self) -> None:
        self.metrics.successes += 1
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.success_threshold:
                self._set_state(CircuitState.CLOSED)
        else:
            self._consecutive_failures = 0

    def _on_failure(self, exc: BaseException) -> None:
        if not self._should_count(exc):
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
            return
        self.metrics.failures += 1
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._set_state(CircuitState.OPEN)
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.failure_threshold:
            self._set_state(CircuitState.OPEN)

    def _reject(self) -> "CircuitOpenError":
        self.metrics.rejected += 1
        return CircuitOpenError(
            f"circuit '{self.name}' is OPEN — failing fast",
            context={"breaker": self.name, "state": self._state.value},
            retryable=True,
        )

    async def acall(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        self.metrics.calls += 1
        if not self._admit():
            raise self._reject()
        try:
            result = await fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            self._on_failure(exc)
            raise
        self._on_success()
        return result

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self.metrics.calls += 1
        if not self._admit():
            raise self._reject()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            self._on_failure(exc)
            raise
        self._on_success()
        return result

    def reset(self) -> None:
        self._set_state(CircuitState.CLOSED)
        self._consecutive_failures = 0


class CircuitRegistry:
    """Named collection of breakers, one per protected dependency."""

    def __init__(self, default: Optional[CircuitBreakerConfig] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._default = default or CircuitBreakerConfig()
        self._clock = clock
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        cb = self._breakers.get(name)
        if cb is None:
            cb = CircuitBreaker(name, config or self._default, clock=self._clock)
            self._breakers[name] = cb
        return cb

    def all(self) -> Dict[str, CircuitBreaker]:
        return dict(self._breakers)

    def health(self) -> Dict[str, str]:
        return {n: cb.state.value for n, cb in self._breakers.items()}


# =========================================================================== #
# Saga (compensating transactions)
# =========================================================================== #
ActionFn = Callable[[Dict[str, Any]], Any]          # may be sync or async
CompensationFn = Callable[[Dict[str, Any]], Any]    # may be sync or async


@dataclass
class SagaStep:
    name: str
    action: ActionFn
    compensation: Optional[CompensationFn] = None


class SagaError(NyxaraError):
    """A saga failed; completed steps were compensated (rolled back)."""

    default_category = ErrorCategory.PERMANENT


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


@dataclass
class _StepResult:
    name: str
    ok: bool
    result: Any = None
    error: Optional[str] = None


class Saga:
    """An ordered set of steps with reverse-order compensation on failure.

    Results accumulate in a shared ``context`` dict (keyed by step name) so later
    steps and compensations can see earlier outputs.
    """

    def __init__(self, name: str = "saga") -> None:
        self.name = name
        self._steps: List[SagaStep] = []
        self.context: Dict[str, Any] = {}
        self.completed: List[str] = []
        self.compensated: List[str] = []

    def add_step(self, name: str, action: ActionFn,
                 compensation: Optional[CompensationFn] = None) -> "Saga":
        self._steps.append(SagaStep(name, action, compensation))
        return self

    def step(self, name: str, compensation: Optional[CompensationFn] = None):
        """Decorator form: register a step's action."""
        def deco(action: ActionFn) -> ActionFn:
            self.add_step(name, action, compensation)
            return action
        return deco

    async def run(self) -> Dict[str, Any]:
        executed: List[SagaStep] = []
        for st in self._steps:
            try:
                res = await _maybe_await(st.action(self.context))
                self.context[st.name] = res
                self.completed.append(st.name)
                executed.append(st)
            except BaseException as exc:  # noqa: BLE001
                await self._compensate(executed)
                raise SagaError(
                    f"saga '{self.name}' failed at step '{st.name}' — rolled back "
                    f"{len(self.compensated)} step(s)",
                    cause=exc if isinstance(exc, Exception) else None,
                    context={
                        "failed_step": st.name,
                        "completed": list(self.completed),
                        "compensated": list(self.compensated),
                    },
                ) from exc
        return dict(self.context)

    async def _compensate(self, executed: Sequence[SagaStep]) -> None:
        for st in reversed(executed):
            if st.compensation is None:
                continue
            try:
                await _maybe_await(st.compensation(self.context))
                self.compensated.append(st.name)
            except Exception as exc:  # noqa: BLE001 - record but keep unwinding
                LEDGER.record(exc, note=f"saga compensation failed: {st.name}")


# =========================================================================== #
# Runtime / lifecycle / supervision
# =========================================================================== #
class LifecycleState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceRef:
    name: str
    start: Callable[[], Awaitable[None]]
    stop: Callable[[], Awaitable[None]]


@dataclass
class TaskHandle:
    name: str
    task: "asyncio.Task"
    supervised: bool = False
    restarts: int = 0
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def done(self) -> bool:
        return self.task.done()

    def cancel(self) -> None:
        self.task.cancel()


class Runtime:
    """Async lifecycle manager: services, tasks, supervision, graceful shutdown."""

    def __init__(self, *, name: str = "nyxara-runtime") -> None:
        self.name = name
        self.state = LifecycleState.CREATED
        self.breakers = CircuitRegistry()
        self._services: List[ServiceRef] = []
        self._tasks: Dict[str, TaskHandle] = {}
        self._shutdown = asyncio.Event()

    # ---- services ---- #
    def register_service(
        self,
        name: str,
        start: Callable[[], Awaitable[None]],
        stop: Callable[[], Awaitable[None]],
    ) -> None:
        self._services.append(ServiceRef(name=name, start=start, stop=stop))

    def register_object(self, name: str, obj: Any) -> None:
        """Register anything exposing async ``start()`` / ``stop()``."""
        self.register_service(name, obj.start, obj.stop)

    # ---- tasks ---- #
    def spawn(self, coro: Awaitable[Any], *, name: Optional[str] = None) -> TaskHandle:
        """Run a tracked one-shot coroutine; exceptions are captured to the LEDGER."""
        name = name or f"task-{len(self._tasks)}"
        task = asyncio.ensure_future(self._wrap_oneshot(name, coro))
        handle = TaskHandle(name=name, task=task, supervised=False)
        self._tasks[handle.task_id] = handle
        return handle

    def supervise(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        name: str,
        max_restarts: int = 5,
        backoff: float = 0.5,
        max_backoff: float = 30.0,
    ) -> TaskHandle:
        """Run a daemon that auto-restarts on crash (capped, with backoff)."""
        task = asyncio.ensure_future(
            self._supervise_loop(name, factory, max_restarts, backoff, max_backoff)
        )
        handle = TaskHandle(name=name, task=task, supervised=True)
        self._tasks[handle.task_id] = handle
        return handle

    async def _wrap_oneshot(self, name: str, coro: Awaitable[Any]) -> Any:
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LEDGER.record(exc, note=f"task '{name}' crashed")
            raise

    async def _supervise_loop(
        self, name: str, factory: Callable[[], Awaitable[Any]],
        max_restarts: int, backoff: float, max_backoff: float,
    ) -> None:
        restarts = 0
        while not self._shutdown.is_set():
            try:
                await factory()
                return  # clean completion -> daemon is done
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LEDGER.record(exc, note=f"supervised '{name}' crashed (restart {restarts})")
                restarts += 1
                # reflect restart count on the handle
                for h in self._tasks.values():
                    if h.name == name and h.supervised:
                        h.restarts = restarts
                if restarts > max_restarts:
                    raise ResourceExhausted(
                        f"supervised task '{name}' exceeded {max_restarts} restarts",
                        cause=exc,
                    )
                delay = min(max_backoff, backoff * (2 ** (restarts - 1)))
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                    return  # shutdown requested during backoff
                except asyncio.TimeoutError:
                    continue

    # ---- lifecycle ---- #
    async def start(self) -> None:
        if self.state in (LifecycleState.RUNNING, LifecycleState.STARTING):
            return
        self.state = LifecycleState.STARTING
        started: List[ServiceRef] = []
        try:
            for svc in self._services:
                await svc.start()
                started.append(svc)
        except Exception:
            # unwind any partially-started services in reverse, then re-raise
            for svc in reversed(started):
                try:
                    await svc.stop()
                except Exception:  # noqa: BLE001
                    pass
            self.state = LifecycleState.STOPPED
            raise
        self._shutdown.clear()
        self.state = LifecycleState.RUNNING

    async def stop(self, *, timeout: float = 10.0) -> None:
        if self.state in (LifecycleState.STOPPED, LifecycleState.STOPPING):
            return
        self.state = LifecycleState.STOPPING
        self._shutdown.set()

        # cancel tracked tasks
        for h in list(self._tasks.values()):
            if not h.task.done():
                h.task.cancel()
        if self._tasks:
            await asyncio.gather(*(h.task for h in self._tasks.values()),
                                 return_exceptions=True)
        self._tasks.clear()

        # stop services in reverse registration order
        for svc in reversed(self._services):
            try:
                await asyncio.wait_for(svc.stop(), timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                LEDGER.record(exc, note=f"service '{svc.name}' stop failed")
        self.state = LifecycleState.STOPPED

    async def run_forever(self) -> None:
        await self.start()
        try:
            await self._shutdown.wait()
        finally:
            await self.stop()

    async def __aenter__(self) -> "Runtime":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # ---- introspection ---- #
    def health(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "services": [s.name for s in self._services],
            "tasks": {
                h.name: {"done": h.done(), "supervised": h.supervised, "restarts": h.restarts}
                for h in self._tasks.values()
            },
            "breakers": self.breakers.health(),
        }


# =========================================================================== #
# Self-test / demo
# =========================================================================== #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.errors import ExternalServiceError, SecurityError

    print("=" * 70)
    print("NYXARA runtime self-test")
    print("=" * 70)

    # --- circuit breaker (deterministic clock) --- #
    clk = {"t": 0.0}
    cb = CircuitBreaker("llm", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=10),
                        clock=lambda: clk["t"])
    states: List[str] = []
    cb.add_listener(lambda n, o, s: states.append(s.value))

    def boom():
        raise ExternalServiceError("503")

    for _ in range(3):
        try:
            cb.call(boom)
        except ExternalServiceError:
            pass
    assert cb.state is CircuitState.OPEN
    print(f"breaker tripped     : {cb.state.value} after 3 failures")

    # calls now fail fast
    try:
        cb.call(lambda: "should not run")
        raise SystemExit("ERROR: open circuit admitted a call")
    except CircuitOpenError:
        print("open circuit fails fast: OK")

    # after recovery timeout, a probe is admitted and success closes it
    clk["t"] = 20.0
    assert cb.call(lambda: "ok") == "ok"
    assert cb.state is CircuitState.CLOSED
    print(f"breaker recovered   : {cb.state.value}")

    # security errors don't count toward tripping
    cb2 = CircuitBreaker("auth", CircuitBreakerConfig(failure_threshold=2))
    for _ in range(5):
        try:
            cb2.call(lambda: (_ for _ in ()).throw(SecurityError("nope")))
        except SecurityError:
            pass
    assert cb2.state is CircuitState.CLOSED
    print("security errors don't trip breaker: OK")

    # --- saga with compensation --- #
    async def saga_demo():
        applied: List[str] = []
        saga = Saga("provision")
        saga.add_step("reserve",
                      lambda ctx: applied.append("reserve") or "r1",
                      lambda ctx: applied.remove("reserve"))
        saga.add_step("charge",
                      lambda ctx: applied.append("charge") or "c1",
                      lambda ctx: applied.remove("charge"))

        async def failing(ctx):
            raise ExternalServiceError("ship failed")

        saga.add_step("ship", failing, lambda ctx: applied.append("ship-comp"))
        try:
            await saga.run()
            raise SystemExit("ERROR: saga should have failed")
        except SagaError as e:
            print(f"saga rolled back    : completed={e.context['completed']} "
                  f"compensated={e.context['compensated']}")
            assert applied == []  # reserve & charge were compensated
        return True

    asyncio.run(saga_demo())

    # --- runtime lifecycle + supervision --- #
    async def runtime_demo():
        rt = Runtime()
        log: List[str] = []

        class Svc:
            async def start(self_inner):
                log.append("start")
            async def stop(self_inner):
                log.append("stop")

        rt.register_object("svc", Svc())

        crash_count = {"n": 0}

        async def flaky_daemon():
            crash_count["n"] += 1
            if crash_count["n"] < 3:
                raise ExternalServiceError("daemon hiccup")
            # third time: live a bit then finish
            await asyncio.sleep(0.01)

        async with rt:
            assert rt.state is LifecycleState.RUNNING
            rt.supervise(flaky_daemon, name="daemon", max_restarts=5, backoff=0.001)
            await asyncio.sleep(0.1)
        assert log == ["start", "stop"]
        print(f"runtime lifecycle   : {log}, daemon restarts until success={crash_count['n']}")
        assert crash_count["n"] >= 3

    asyncio.run(runtime_demo())
    print("\nALL SELF-TESTS PASSED ✓")
