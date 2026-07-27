"""NYXARA · nyx5/event_queue.py — the event-driven scheduler (⚡, pillar 1).

The heart of "no backpropagation, no clock-stepped tensors": NYX-5 does not sweep a dense array every
millisecond. It keeps a **priority queue of spike deliveries** keyed on logical delivery time, and
advances by popping the *next* event — so silent regions of the network cost nothing and computation
follows activity, the way an asynchronous neuromorphic fabric would.

A hard ``max_events`` budget bounds every drain, so one turn can never run away (a pathological
positive-feedback loop halts instead of hanging). Pure ``heapq``, deterministic: ties broken by a
monotonic sequence number, so equal-time events keep insertion order.

Depends on nyx5/neuron.py (SpikeEvent).
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable, List, Optional, Tuple

from nyxara.nyx5.neuron import SpikeEvent

__all__ = ["EventDrivenScheduler"]


class EventDrivenScheduler:
    """A logical-time priority queue of spike deliveries.

    ``schedule`` pushes a :class:`SpikeEvent`; ``run_until`` drains events in time order up to a time
    horizon or an event budget, invoking a handler for each. The handler may itself schedule further
    events (a spike that arrives can cause downstream spikes) — the budget still bounds the total.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, SpikeEvent]] = []
        self._seq = itertools.count()
        self.now: float = 0.0
        self.delivered: int = 0

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return not self._heap

    def schedule(self, event: SpikeEvent) -> None:
        """Enqueue ``event`` for delivery at ``event.time`` (must be >= current logical time)."""
        t = max(event.time, self.now)
        event.time = t
        heapq.heappush(self._heap, (t, next(self._seq), event))

    def run_until(self, t_max: float, *, max_events: int,
                  handler: Callable[[SpikeEvent], None]) -> int:
        """Drain events with ``time <= t_max``, newest logical time last, up to ``max_events``.

        Returns the number of events delivered this call. Halts early — without error — when the
        budget is exhausted, guaranteeing bounded work per turn even under runaway feedback.
        """
        count = 0
        while self._heap and self._heap[0][0] <= t_max:
            if count >= max_events:
                break
            t, _, event = heapq.heappop(self._heap)
            self.now = t
            handler(event)
            self.delivered += 1
            count += 1
        return count

    def drain(self, *, max_events: int, handler: Callable[[SpikeEvent], None]) -> int:
        """Drain the whole queue (no time horizon), still bounded by ``max_events``."""
        return self.run_until(float("inf"), max_events=max_events, handler=handler)

    def clear(self) -> None:
        self._heap.clear()
        self.now = 0.0

    def peek_time(self) -> Optional[float]:
        return self._heap[0][0] if self._heap else None
