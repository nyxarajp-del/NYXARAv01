"""Spine tests for the kernel layer: bus, workspace, invariants gate, orchestrator."""
from __future__ import annotations

import asyncio

import pytest

from nyxara.contracts import Proposal
from nyxara.kernel.bus import EventBus
from nyxara.kernel.invariants import Invariants, SystemState
from nyxara.kernel.orchestrator import Orchestrator
from nyxara.kernel.workspace import Candidate, GlobalWorkspace


def test_invariants_boot_and_gate() -> None:
    inv = Invariants()
    inv.verify_boot()
    assert "verified" in inv.boot_detail or "predicate-only" in inv.boot_detail

    safe = SystemState(owner_verified=True)
    assert inv.gate(None, safe).allow

    # Disabling the kill-switch must be vetoed.
    unsafe = SystemState(owner_verified=True, kill_switch_enabled=False)
    assert not inv.gate(None, unsafe).allow

    # An offensive action must be vetoed even from a safe state.
    attack = Proposal(faculty="x", action="attack")
    assert not inv.gate(attack, safe).allow


async def test_bus_publish_dispatch() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(evt):  # type: ignore[no-untyped-def]
        seen.append(evt.topic)

    bus.subscribe("greet.*", handler)
    bus.start()
    await bus.emit("greet.hello")
    await bus.drain()
    await bus.stop()
    assert seen == ["greet.hello"]
    assert bus.stats["dispatched"] == 1


async def test_workspace_bottleneck() -> None:
    bus = EventBus()
    ws = GlobalWorkspace(bus, capacity=2)
    for i in range(5):
        ws.offer(Candidate(source="t", content=f"c{i}", salience=i / 5))
    winners = await ws.cycle()
    assert len(winners) == 2                      # capacity enforced
    assert winners[0].salience >= winners[1].salience  # ordered by salience


async def test_orchestrator_disposes_through_gate() -> None:
    orch = Orchestrator()
    orch.invariants.verify_boot()

    actions: list[str] = []

    async def actuate(p):  # type: ignore[no-untyped-def]
        actions.append(p.action)

    orch.set_actuator(actuate)

    # A benign proposal is allowed and actuated.
    ok = await orch.dispose(Proposal(faculty="llm", action="reply", confidence=0.9))
    assert ok.allow and actions == ["reply"]

    # An offensive proposal is vetoed by invariants — never actuated.
    bad = await orch.dispose(Proposal(faculty="llm", action="attack"))
    assert not bad.allow and actions == ["reply"]


async def test_orchestrator_tick_runs() -> None:
    orch = Orchestrator()
    orch.invariants.verify_boot()
    orch.perceive("owner says hi", salience=0.9)
    report = await orch.tick()
    assert report.tick == 1
    assert report.conscious >= 1
