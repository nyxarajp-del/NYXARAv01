"""NYXARA · kernel/bus.py — asynchronous event bus.

The nervous system's wiring: every subsystem publishes :class:`Event` objects and
subscribes to topic patterns. The bus delivers them, in **priority order**, through
a **middleware pipeline**, to matching handlers — with **backpressure**, bounded
**retry**, and a **dead-letter queue** for anything that can't be delivered.

Highlights
----------
* **Priority scheduling** — CRITICAL events overtake NORMAL/LOW (stable FIFO within
  a priority via a monotonic sequence).
* **Glob topics** — subscribe to ``"sensor.*"``, ``"mind.proposal.*"``, ``"*"`` …
  (fnmatch semantics) plus an optional payload predicate.
* **Sync & async handlers** — a handler may be a plain function or a coroutine.
* **Middleware** — ordered transforms that can rewrite, tag, or drop events before
  dispatch (e.g. shield.py will sit here to sanitise untrusted input).
* **Resilience** — handler failures are classified (kernel/errors): retryable
  errors are re-queued with capped attempts; everything else (and exhausted
  retries) goes to the dead-letter queue. The learn-once LEDGER records failures.
* **Backpressure** — a bounded queue; ``publish`` can block (await) or drop-oldest.
* **Observability** — live :class:`BusMetrics`.

Async-first, but usable from synchronous code via :meth:`EventBus.run` /
:meth:`EventBus.emit_sync`.

Depends on :mod:`config` (queue sizing) and :mod:`errors` (classification/retry).
"""

from __future__ import annotations

import asyncio
import fnmatch
import itertools
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import (
    LEDGER,
    ErrorCategory,
    NyxaraError,
    RetryPolicy,
    classify,
)

__all__ = [
    "Priority",
    "Event",
    "Subscription",
    "Middleware",
    "BusMetrics",
    "DeadLetter",
    "EventBus",
    "OverflowPolicy",
]


# --------------------------------------------------------------------------- #
# Priority
# --------------------------------------------------------------------------- #
class Priority(IntEnum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40

    @classmethod
    def coerce(cls, v: Union["Priority", int, str]) -> "Priority":
        if isinstance(v, Priority):
            return v
        if isinstance(v, str):
            return cls[v.upper()]
        # nearest defined level at or below the integer
        best = cls.LOW
        for p in cls:
            if int(p) <= int(v):
                best = p
        return best


class OverflowPolicy(str):
    BLOCK = "block"            # await space (backpressure)
    DROP_NEWEST = "drop_newest"
    DROP_OLDEST = "drop_oldest"


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """A message on the bus.

    ``topic`` is a dotted name matched by subscriptions (glob). ``priority`` drives
    scheduling. ``trace_id``/``correlation_id`` thread causality across subsystems.
    """

    topic: str
    payload: Any = None
    priority: Priority = Priority.NORMAL
    source: Optional[str] = None
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # absolute epoch; None == never
    attempts: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at is not None and (now or time.time()) >= self.expires_at

    def with_ttl(self, ttl_s: float) -> "Event":
        self.expires_at = time.time() + ttl_s
        return self

    def reply(self, topic: str, payload: Any = None, **kw: Any) -> "Event":
        """Construct a correlated response event."""
        return Event(
            topic=topic,
            payload=payload,
            correlation_id=self.correlation_id or self.event_id,
            trace_id=self.trace_id,
            **kw,
        )


# --------------------------------------------------------------------------- #
# Subscription & middleware
# --------------------------------------------------------------------------- #
Handler = Callable[[Event], Union[None, Awaitable[None]]]
Middleware = Callable[[Event], Union[Optional[Event], Awaitable[Optional[Event]]]]
Predicate = Callable[[Event], bool]


@dataclass
class Subscription:
    pattern: str
    handler: Handler
    sub_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority_min: Priority = Priority.LOW
    predicate: Optional[Predicate] = None
    once: bool = False
    _dead: bool = False

    def matches(self, event: Event) -> bool:
        if self._dead:
            return False
        if event.priority < self.priority_min:
            return False
        if not _topic_match(self.pattern, event.topic):
            return False
        if self.predicate is not None:
            try:
                return bool(self.predicate(event))
            except Exception:
                return False
        return True


def _topic_match(pattern: str, topic: str) -> bool:
    """Glob match with a couple of routing conveniences.

    ``*`` matches within a segment; ``#`` matches one-or-more trailing segments;
    a bare ``*`` or ``#`` matches everything.
    """
    if pattern in ("*", "#"):
        return True
    if pattern.endswith(".#"):
        prefix = pattern[:-2]
        return topic == prefix or topic.startswith(prefix + ".")
    return fnmatch.fnmatchcase(topic, pattern)


# --------------------------------------------------------------------------- #
# Metrics & dead-letter
# --------------------------------------------------------------------------- #
@dataclass
class BusMetrics:
    published: int = 0
    delivered: int = 0
    handler_invocations: int = 0
    failed: int = 0
    retried: int = 0
    dead_lettered: int = 0
    dropped_overflow: int = 0
    dropped_expired: int = 0
    dropped_by_middleware: int = 0

    def snapshot(self) -> Dict[str, int]:
        return dict(self.__dict__)


@dataclass
class DeadLetter:
    event: Event
    reason: str
    error: Optional[str]
    category: Optional[str]
    at: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# The bus
# --------------------------------------------------------------------------- #
# Internal heap item: (-priority, sequence, event). Lower tuple == popped first.
_QItem = Tuple[int, int, Event]


class EventBus:
    """Priority, middleware-aware, fault-tolerant async event bus."""

    def __init__(
        self,
        *,
        workers: int = 4,
        max_queue: Optional[int] = None,
        overflow: str = OverflowPolicy.BLOCK,
        retry_policy: Optional[RetryPolicy] = None,
        dead_letter_capacity: int = 1024,
        settings: Optional[NyxaraSettings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._max_queue = max_queue or self._settings.resources.max_event_queue
        self._workers_n = max(1, workers)
        self._overflow = overflow
        self._retry = retry_policy or RetryPolicy(max_attempts=4, base_delay=0.05, max_delay=2.0)
        self._dl_capacity = dead_letter_capacity

        self._queue: "asyncio.PriorityQueue[_QItem]" = asyncio.PriorityQueue(maxsize=self._max_queue)
        self._seq = itertools.count()
        self._subs: List[Subscription] = []
        self._middleware: List[Middleware] = []
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()
        self.metrics = BusMetrics()
        self.dead_letters: List[DeadLetter] = []
        # optional sink invoked for every dead letter (e.g. audit/oversight)
        self.on_dead_letter: Optional[Callable[[DeadLetter], None]] = None

    # ---- subscription management ---- #
    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        priority_min: Union[Priority, int, str] = Priority.LOW,
        predicate: Optional[Predicate] = None,
        once: bool = False,
    ) -> Subscription:
        sub = Subscription(
            pattern=pattern,
            handler=handler,
            priority_min=Priority.coerce(priority_min),
            predicate=predicate,
            once=once,
        )
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Union[Subscription, str]) -> bool:
        sub_id = sub.sub_id if isinstance(sub, Subscription) else sub
        for s in self._subs:
            if s.sub_id == sub_id:
                s._dead = True
        before = len(self._subs)
        self._subs = [s for s in self._subs if s.sub_id != sub_id]
        return len(self._subs) != before

    def use(self, middleware: Middleware) -> "EventBus":
        """Append a middleware to the pre-dispatch pipeline (order preserved)."""
        self._middleware.append(middleware)
        return self

    def subscriber_count(self, topic: Optional[str] = None) -> int:
        if topic is None:
            return len(self._subs)
        ev = Event(topic=topic, priority=Priority.CRITICAL)
        return sum(1 for s in self._subs if s.matches(ev))

    # ---- publishing ---- #
    async def publish(self, event: Event) -> bool:
        """Enqueue ``event``. Returns False if dropped by the overflow policy."""
        item: _QItem = (-int(event.priority), next(self._seq), event)
        self.metrics.published += 1
        if self._overflow == OverflowPolicy.BLOCK:
            await self._queue.put(item)
            return True
        # non-blocking variants
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            if self._overflow == OverflowPolicy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self._queue.put_nowait(item)
                    self.metrics.dropped_overflow += 1
                    return True
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            self.metrics.dropped_overflow += 1
            return False

    def publish_nowait(self, event: Event) -> bool:
        item: _QItem = (-int(event.priority), next(self._seq), event)
        try:
            self._queue.put_nowait(item)
            self.metrics.published += 1
            return True
        except asyncio.QueueFull:
            self.metrics.dropped_overflow += 1
            return False

    async def emit(self, topic: str, payload: Any = None, **kw: Any) -> bool:
        return await self.publish(Event(topic=topic, payload=payload, **kw))

    # ---- lifecycle ---- #
    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return
            self._running = True
            self._tasks = [
                asyncio.create_task(self._worker(i), name=f"bus-worker-{i}")
                for i in range(self._workers_n)
            ]

    async def stop(self, *, drain: bool = True) -> None:
        if drain:
            await self.drain()
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def drain(self) -> None:
        """Block until every queued event has been fully processed."""
        await self._queue.join()

    async def __aenter__(self) -> "EventBus":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop(drain=True)

    # ---- worker / dispatch ---- #
    async def _worker(self, idx: int) -> None:
        while self._running:
            try:
                neg_prio, seq, event = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._dispatch(event)
            except asyncio.CancelledError:
                self._queue.task_done()
                break
            except Exception as exc:  # noqa: BLE001 - dispatch must never kill a worker
                self._dead_letter(event, "dispatch_crash", exc)
            finally:
                self._queue.task_done()

    async def _run_middleware(self, event: Event) -> Optional[Event]:
        cur: Optional[Event] = event
        for mw in self._middleware:
            if cur is None:
                return None
            res = mw(cur)
            if asyncio.iscoroutine(res):
                res = await res
            cur = res
        return cur

    async def _dispatch(self, event: Event) -> None:
        now = time.time()
        if event.is_expired(now):
            self.metrics.dropped_expired += 1
            self._dead_letter(event, "expired", None)
            return

        processed = await self._run_middleware(event)
        if processed is None:
            self.metrics.dropped_by_middleware += 1
            return
        event = processed

        matched = [s for s in self._subs if s.matches(event)]
        if not matched:
            self.metrics.delivered += 1  # accepted into the bus, simply no listeners
            return

        any_failed = False
        for sub in matched:
            self.metrics.handler_invocations += 1
            try:
                await self._invoke(sub.handler, event)
                if sub.once:
                    self.unsubscribe(sub)
            except Exception as exc:  # noqa: BLE001
                any_failed = True
                self.metrics.failed += 1
                LEDGER.record(exc, note=f"bus handler {sub.pattern}")
                if self._should_retry(exc, event):
                    await self._requeue(event)
                else:
                    self._dead_letter(event, "handler_error", exc)
        if not any_failed:
            self.metrics.delivered += 1

    async def _invoke(self, handler: Handler, event: Event) -> None:
        res = handler(event)
        if asyncio.iscoroutine(res):
            await res

    def _should_retry(self, exc: Exception, event: Event) -> bool:
        cat = exc.category if isinstance(exc, NyxaraError) else classify(exc)
        if cat.fail_closed:
            return False
        return self._retry.should_retry(exc, event.attempts + 1)

    async def _requeue(self, event: Event) -> None:
        event.attempts += 1
        self.metrics.retried += 1
        delay = self._retry.delay_for(event.attempts)
        # schedule re-publish after backoff without blocking the worker
        async def _later() -> None:
            await asyncio.sleep(delay)
            await self.publish(event)

        asyncio.create_task(_later())

    def _dead_letter(self, event: Event, reason: str, exc: Optional[BaseException]) -> None:
        cat = None
        if exc is not None:
            c = exc.category if isinstance(exc, NyxaraError) else classify(exc)
            cat = c.value
        dl = DeadLetter(
            event=event,
            reason=reason,
            error=repr(exc) if exc else None,
            category=cat,
        )
        self.dead_letters.append(dl)
        if len(self.dead_letters) > self._dl_capacity:
            self.dead_letters.pop(0)
        self.metrics.dead_lettered += 1
        if self.on_dead_letter is not None:
            try:
                self.on_dead_letter(dl)
            except Exception:  # noqa: BLE001
                pass

    # ---- synchronous convenience ---- #
    def run(self, coro: Awaitable[Any]) -> Any:
        """Run a coroutine against a fresh loop (for sync call-sites/tests)."""
        return asyncio.run(coro)

    async def emit_and_drain(self, *events: Event) -> None:
        """Start (if needed), publish events, drain, used for one-shot processing."""
        started = not self._running
        if started:
            await self.start()
        for e in events:
            await self.publish(e)
        await self.drain()
        if started:
            await self.stop(drain=False)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import collections

    print("=" * 70)
    print("NYXARA bus self-test")
    print("=" * 70)

    async def demo() -> None:
        bus = EventBus(workers=3)
        seen = collections.Counter()
        order: List[str] = []

        async def slow(ev: Event) -> None:
            await asyncio.sleep(0.001)
            seen[ev.topic] += 1

        def record_priority(ev: Event) -> None:
            order.append(ev.payload)

        bus.subscribe("sensor.*", slow)
        bus.subscribe("prio", record_priority)

        # middleware that tags then a filter that drops "secret"
        def tagger(ev: Event) -> Event:
            ev.headers["seen_by_mw"] = True
            return ev

        def drop_secret(ev: Event) -> Optional[Event]:
            return None if ev.payload == "secret" else ev

        bus.use(tagger).use(drop_secret)

        async with bus:
            await bus.emit("sensor.vision", payload={"frame": 1})
            await bus.emit("sensor.audio", payload={"frame": 2})
            await bus.emit("sensor.audio", payload="secret")  # dropped by mw
            # priority ordering: publish LOW then CRITICAL then NORMAL
            await bus.publish(Event("prio", payload="low", priority=Priority.LOW))
            await bus.publish(Event("prio", payload="crit", priority=Priority.CRITICAL))
            await bus.publish(Event("prio", payload="norm", priority=Priority.NORMAL))
            await bus.drain()

        assert seen["sensor.vision"] == 1
        assert seen["sensor.audio"] == 1  # the 'secret' one was dropped pre-dispatch
        print(f"delivered topics    : {dict(seen)}")
        print(f"priority dispatch   : {order}")
        print(f"metrics             : {bus.metrics.snapshot()}")

        # retry + dead-letter
        bus2 = EventBus(workers=1, retry_policy=RetryPolicy(max_attempts=3, base_delay=0))
        attempts = {"n": 0}

        async def flaky(ev: Event) -> None:
            attempts["n"] += 1
            from nyxara.kernel.errors import ExternalServiceError, ValidationError
            if ev.payload == "transient" and attempts["n"] < 3:
                raise ExternalServiceError("503")
            if ev.payload == "permanent":
                raise ValidationError("bad")

        bus2.subscribe("work", flaky)
        async with bus2:
            await bus2.emit("work", payload="transient")
            await asyncio.sleep(0.05)
            await bus2.drain()
            await bus2.emit("work", payload="permanent")
            await bus2.drain()

        print(f"retry attempts      : {attempts['n']} (expected >=3)")
        print(f"dead letters        : {len(bus2.dead_letters)} "
              f"({[dl.reason for dl in bus2.dead_letters]})")
        assert attempts["n"] >= 3
        assert any(dl.reason == "handler_error" for dl in bus2.dead_letters)

    asyncio.run(demo())
    print("\nALL SELF-TESTS PASSED ✓")
