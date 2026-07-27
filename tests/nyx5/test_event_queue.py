"""Event-driven scheduler: drains in time order, honours the budget, empty is a no-op."""

from __future__ import annotations

from nyxara.nyx5.event_queue import EventDrivenScheduler
from nyxara.nyx5.neuron import SpikeEvent


def test_drains_in_time_order():
    s = EventDrivenScheduler()
    s.schedule(SpikeEvent(time=3.0, neuron_id=1))
    s.schedule(SpikeEvent(time=1.0, neuron_id=2))
    s.schedule(SpikeEvent(time=2.0, neuron_id=3))
    seen = []
    s.run_until(10.0, max_events=100, handler=lambda e: seen.append(e.neuron_id))
    assert seen == [2, 3, 1]


def test_ties_keep_insertion_order():
    s = EventDrivenScheduler()
    for nid in (5, 6, 7):
        s.schedule(SpikeEvent(time=1.0, neuron_id=nid))
    seen = []
    s.drain(max_events=100, handler=lambda e: seen.append(e.neuron_id))
    assert seen == [5, 6, 7]


def test_event_budget_halts_runaway():
    s = EventDrivenScheduler()

    # a handler that schedules a fresh event each time -> would loop forever without a budget
    def handler(e: SpikeEvent) -> None:
        s.schedule(SpikeEvent(time=e.time + 1.0, neuron_id=e.neuron_id))

    s.schedule(SpikeEvent(time=0.0, neuron_id=0))
    delivered = s.run_until(float("inf"), max_events=50, handler=handler)
    assert delivered == 50            # halted at the budget, not hung


def test_empty_queue_is_noop():
    s = EventDrivenScheduler()
    assert s.is_empty()
    assert s.run_until(10.0, max_events=10, handler=lambda e: None) == 0


def test_time_horizon_respected():
    s = EventDrivenScheduler()
    s.schedule(SpikeEvent(time=1.0, neuron_id=1))
    s.schedule(SpikeEvent(time=5.0, neuron_id=2))
    seen = []
    s.run_until(3.0, max_events=100, handler=lambda e: seen.append(e.neuron_id))
    assert seen == [1]                # the t=5 event stays queued
    assert len(s) == 1
