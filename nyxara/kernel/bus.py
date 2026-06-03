"""Event bus — async pub/sub with a priority queue, middleware, and a dead-letter sink.

The bus is the nervous system. Producers publish :class:`Event`s; the bus orders them by
:class:`Priority`, runs each through the middleware chain, and dispatches to subscribers
matched by topic (exact or ``prefix.*`` glob). Handler exceptions never kill the bus —
the offending event is routed to the dead-letter queue for inspection / replay.
"""
from __future__ import annotations

import asyncio
import fnmatch
import itertools
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from nyxara.contracts import Event, Priority

Handler = Callable[[Event], Awaitable[None]]
Middleware = Callable[[Event], Awaitable[Event | None]]  # return None to drop


@dataclass(order=True)
class _Queued:
    priority: int
    seq: int
    event: Event = field(compare=False)


@dataclass
class _Subscription:
    pattern: str
    handler: Handler

    def matches(self, topic: str) -> bool:
        return self.pattern == topic or fnmatch.fnmatchcase(topic, self.pattern)


class EventBus:
    """A single-process async event bus."""

    def __init__(self, *, maxsize: int = 10_000) -> None:
        self._q: asyncio.PriorityQueue[_Queued] = asyncio.PriorityQueue(maxsize)
        self._subs: list[_Subscription] = []
        self._middleware: list[Middleware] = []
        self._dead_letter: list[tuple[Event, BaseException]] = []
        self._seq = itertools.count()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._published = 0
        self._dispatched = 0

    # ── wiring ────────────────────────────────────────────────────────────────
    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        """Subscribe a coroutine handler to a topic pattern. Returns an unsubscribe fn."""
        sub = _Subscription(pattern, handler)
        self._subs.append(sub)
        return lambda: self._subs.remove(sub) if sub in self._subs else None

    def use(self, middleware: Middleware) -> None:
        """Append a middleware. Each may transform, observe, or drop (return None)."""
        self._middleware.append(middleware)

    # ── publishing ────────────────────────────────────────────────────────────
    async def publish(self, event: Event) -> None:
        self._published += 1
        await self._q.put(_Queued(int(event.priority), next(self._seq), event))

    def publish_nowait(self, event: Event) -> None:
        self._published += 1
        self._q.put_nowait(_Queued(int(event.priority), next(self._seq), event))

    async def emit(self, topic: str, *, priority: Priority = Priority.NORMAL,
                   source: str = "bus", **payload: object) -> None:
        await self.publish(Event(topic=topic, payload=payload, priority=priority,
                                 source=source))

    # ── run loop ──────────────────────────────────────────────────────────────
    async def _dispatch(self, event: Event) -> None:
        cur: Event | None = event
        for mw in self._middleware:
            cur = await mw(cur)  # type: ignore[arg-type]
            if cur is None:
                return  # dropped by middleware
        targets = [s for s in self._subs if s.matches(cur.topic)]
        for sub in targets:
            try:
                await sub.handler(cur)
                self._dispatched += 1
            except Exception as exc:  # noqa: BLE001 — isolate handler failures
                self._dead_letter.append((cur, exc))

    async def run(self) -> None:
        """Drain the queue forever. Cancel the task to stop."""
        self._running = True
        try:
            while self._running:
                queued = await self._q.get()
                try:
                    await self._dispatch(queued.event)
                finally:
                    self._q.task_done()
        except asyncio.CancelledError:
            self._running = False
            raise

    def start(self) -> asyncio.Task[None]:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="event-bus")
        return self._task

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def drain(self) -> None:
        """Wait until every queued event has been dispatched."""
        await self._q.join()

    # ── introspection (read-only; used by observe/) ────────────────────────────
    @property
    def stats(self) -> dict[str, int]:
        return {
            "published": self._published,
            "dispatched": self._dispatched,
            "subscriptions": len(self._subs),
            "middleware": len(self._middleware),
            "dead_letter": len(self._dead_letter),
            "pending": self._q.qsize(),
        }

    @property
    def dead_letter(self) -> list[tuple[Event, BaseException]]:
        return list(self._dead_letter)


__all__ = ["EventBus", "Handler", "Middleware"]
