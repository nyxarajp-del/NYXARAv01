"""Background tasks the kernel schedules must be strongly referenced until they run.

``asyncio`` keeps only a WEAK reference to a task. A bare ``create_task(...)`` whose result is
discarded can therefore be garbage-collected before the coroutine ever runs — the work simply
never happens, with no error raised anywhere. Two kernel paths did exactly that:

* ``EventBus._requeue`` — the backed-off re-publish of a failed event, i.e. the retry the bus
  had just counted in ``metrics.retried``.
* ``GlobalWorkspace._publish`` — the fallback publish taken when the bus queue is full, i.e.
  precisely the broadcast most at risk of being dropped.

These pin that both hold a reference for the task's lifetime and release it once it completes.
"""

from __future__ import annotations

import asyncio
import gc

from nyxara.kernel.bus import Event, EventBus
from nyxara.kernel.workspace import Content, GlobalWorkspace


def _run(coro):
    return asyncio.run(coro)


def test_bus_retry_task_is_retained_until_it_fires():
    async def go():
        bus = EventBus()
        await bus._requeue(Event(topic="t.retry", payload={"n": 1}, source="test"))
        assert bus._retry_tasks, "the backoff task must be held by the bus, not only the loop"

        gc.collect()          # a full GC pass while pending must not collect it
        assert bus._retry_tasks

        await asyncio.gather(*list(bus._retry_tasks), return_exceptions=True)
        await asyncio.sleep(0)
        assert not bus._retry_tasks, "completed retry tasks must be discarded"
        await bus.stop(drain=False)

    _run(go())


def test_bus_stop_cancels_pending_retries():
    async def go():
        bus = EventBus()
        await bus._requeue(Event(topic="t.retry", payload={"n": 2}, source="test"))
        pending = list(bus._retry_tasks)
        assert pending

        await bus.stop(drain=False)
        assert not bus._retry_tasks
        assert all(t.done() for t in pending), "a pending retry must not outlive the bus"

    _run(go())


def test_workspace_fallback_publish_is_retained_and_delivers():
    """When publish_nowait declines, the scheduled publish must survive a GC pass."""
    class _FullBus:
        def __init__(self) -> None:
            self.published: list = []

        def publish_nowait(self, ev) -> bool:
            return False                       # queue full -> force the fallback path

        async def publish(self, ev) -> None:
            await asyncio.sleep(0)
            self.published.append(ev)

    async def go():
        bus = _FullBus()
        ws = GlobalWorkspace(bus=bus)
        assert ws._pending_publishes == set()

        ws.submit(Content(payload="alpha", source="test", activation=1.0, urgency=1.0))
        broadcasts = ws.cycle()
        assert broadcasts, "a lone high-salience candidate should reach consciousness"

        gc.collect()                           # the exact moment the old code lost the task
        assert ws._pending_publishes, "the fallback publish must be held, not only by the loop"

        await asyncio.gather(*list(ws._pending_publishes), return_exceptions=True)
        await asyncio.sleep(0)
        assert not ws._pending_publishes, "completed publishes must be discarded"
        assert bus.published, "the retained task must actually have published"

    _run(go())
