"""Sovereign orchestrator — owns the loop, selects faculties, disposes proposals.

This is where "the LLM proposes, the kernel disposes" becomes literal. The orchestrator:

  1. lets :class:`Presence` set the cadence,
  2. runs a global-workspace :meth:`cycle` to decide what is *conscious* this tick,
  3. asks the registered **faculties** to emit :class:`Proposal`s about the conscious
     contents (no faculty acts — they only propose),
  4. passes each proposal through the **gate** (guard pipeline + invariants), and
  5. *disposes*: dispatches allowed proposals as action events on the bus, records every
     step, and drops / logs the rest.

Faculties and the guard pipeline are injected as simple callables/Protocols so the kernel
compiles and runs before the mind/ and guard/ layers exist; the orchestrator owns control
regardless of what is plugged in.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from nyxara.contracts import Event, Priority, Proposal, Verdict
from nyxara.kernel.bus import EventBus
from nyxara.kernel.config import Settings, get_settings
from nyxara.kernel.errors import InvariantBreach
from nyxara.kernel.invariants import Invariants, SystemState
from nyxara.kernel.presence import Presence
from nyxara.kernel.replay import Recorder
from nyxara.kernel.stream import DefaultModeStream
from nyxara.kernel.workspace import Candidate, GlobalWorkspace


# ─────────────────────────────────────────────────────────────────────────────
#  Pluggable interfaces (implemented by mind/ and guard/ layers later)
# ─────────────────────────────────────────────────────────────────────────────
class Faculty(Protocol):
    """A cognitive engine. Given the conscious coalition, returns proposals."""

    name: str

    async def propose(self, coalition: list[Candidate]) -> list[Proposal]: ...


class Gate(Protocol):
    """The guard pipeline: schema-check → shield → critique → (invariants applied
    separately by the orchestrator). Returns a verdict for a proposal."""

    async def review(self, proposal: Proposal, state: SystemState) -> Verdict: ...


# A trivial gate used until guard/ is wired: it allows everything (invariants still run).
class _PassThroughGate:
    async def review(self, proposal: Proposal, state: SystemState) -> Verdict:
        return Verdict.ok(by="passthrough")


@dataclass
class TickReport:
    tick: int
    conscious: int
    proposed: int
    allowed: int
    denied: int
    duration_ms: float


class Orchestrator:
    """The sovereign control loop."""

    def __init__(self, *, bus: EventBus | None = None, settings: Settings | None = None,
                 invariants: Invariants | None = None,
                 rng: random.Random | None = None) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or EventBus()
        self.invariants = invariants or Invariants()
        self.workspace = GlobalWorkspace(
            self.bus, capacity=self.settings.limits.workspace_capacity)
        self.stream = DefaultModeStream(self.workspace)
        self.presence = Presence()
        self.recorder = Recorder()
        self.state = SystemState()
        self._rng = rng or random.Random()

        self._faculties: list[Faculty] = []
        self._gate: Gate = _PassThroughGate()
        # The kernel actuates allowed proposals. Default: publish them as bus events.
        self._actuator: Callable[[Proposal], Awaitable[None]] = self._default_actuate

        self._running = False
        self._ticks = 0
        self._loop_task: asyncio.Task[None] | None = None

    # ── wiring ──────────────────────────────────────────────────────────────────
    def register_faculty(self, faculty: Faculty) -> None:
        self._faculties.append(faculty)

    def set_gate(self, gate: Gate) -> None:
        self._gate = gate

    def set_actuator(self, actuator: Callable[[Proposal], Awaitable[None]]) -> None:
        self._actuator = actuator

    # ── input from the outside world ────────────────────────────────────────────
    def perceive(self, content: object, *, salience: float = 0.8, kind: str = "percept",
                 source: str = "world", engaging: bool = True) -> None:
        """Inject a stimulus: bumps presence and offers a candidate to the workspace."""
        self.presence.stimulate(engaging=engaging)
        self.workspace.offer(Candidate(source=source, content=content,
                                       salience=salience, kind=kind))

    # ── disposing a single proposal through the full gate ─────────────────────────
    async def dispose(self, proposal: Proposal) -> Verdict:
        """Run a proposal through guard review then the invariants gate, then act.

        This is the choke point: nothing becomes an action without passing here.
        """
        self.recorder.record("proposal", proposal.trace_id, faculty=proposal.faculty,
                             action=proposal.action, confidence=proposal.confidence)

        # 1. Guard pipeline (schema/shield/critique). Fails closed on error.
        try:
            guard_verdict = await self._gate.review(proposal, self.state)
        except Exception as exc:
            guard_verdict = Verdict.deny(f"guard error: {exc!r}", by="guard", severity=8)
        if not guard_verdict.allow:
            self.recorder.record("verdict", proposal.trace_id, allow=False,
                                 by=guard_verdict.decided_by, reasons=guard_verdict.reasons)
            return guard_verdict

        proposal = guard_verdict.applied_to(proposal)

        # 2. Invariants gate — the non-negotiable core. Always runs, fails closed.
        inv_verdict = self.invariants.gate(proposal, self.state)
        if not inv_verdict.allow:
            self.recorder.record("verdict", proposal.trace_id, allow=False,
                                 by="invariants", reasons=inv_verdict.reasons)
            return inv_verdict

        # 3. Kernel acts.
        await self._actuator(proposal)
        self.recorder.record("action", proposal.trace_id, action=proposal.action)
        return Verdict.ok(by="orchestrator")

    async def _default_actuate(self, proposal: Proposal) -> None:
        await self.bus.publish(Event(
            topic=f"action.{proposal.action}",
            payload={"args": dict(proposal.args), "faculty": proposal.faculty},
            priority=Priority.HIGH, source="orchestrator"))

    # ── one cognitive tick ────────────────────────────────────────────────────────
    async def tick(self) -> TickReport:
        start = time.perf_counter()
        self._ticks += 1

        # Idle mind keeps thinking; engaged mind is driven by perception already offered.
        if self.settings.features.continuous_stream:
            self.stream.tick()

        coalition = await self.workspace.cycle()

        proposals: list[Proposal] = []
        for fac in self._faculties:
            try:
                proposals.extend(await fac.propose(coalition))
            except Exception as exc:
                self.recorder.record("faculty_error", "n/a", faculty=fac.name, err=repr(exc))

        # Respect the proposal-depth limit (guard against runaway think→propose loops).
        proposals = proposals[: self.settings.limits.max_proposal_depth]

        allowed = denied = 0
        for p in proposals:
            verdict = await self.dispose(p)
            if verdict.allow:
                allowed += 1
            else:
                denied += 1

        self.presence.relax()
        duration_ms = (time.perf_counter() - start) * 1000.0
        return TickReport(tick=self._ticks, conscious=len(coalition),
                          proposed=len(proposals), allowed=allowed, denied=denied,
                          duration_ms=duration_ms)

    # ── the loop ──────────────────────────────────────────────────────────────────
    async def run(self) -> None:
        """Boot-verify invariants, then run the sovereign loop until stopped."""
        self.invariants.verify_boot()
        self.invariants.assert_state(self.state)  # refuse to run from an unsafe state
        self.bus.start()
        self._running = True
        try:
            while self._running:
                await self.tick()
                await asyncio.sleep(self.presence.tick_interval)
        except asyncio.CancelledError:
            self._running = False
            raise

    def start(self) -> asyncio.Task[None]:
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self.run(), name="orchestrator")
        return self._loop_task

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        await self.bus.stop()

    @property
    def stats(self) -> dict[str, object]:
        return {
            "ticks": self._ticks,
            "presence": self.presence.stats,
            "workspace": self.workspace.stats,
            "bus": self.bus.stats,
            "faculties": [f.name for f in self._faculties],
        }


__all__ = ["Orchestrator", "Faculty", "Gate", "TickReport"]
