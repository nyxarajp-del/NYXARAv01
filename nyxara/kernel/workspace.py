"""Global Workspace — attention arbitration, the conscious bottleneck, broadcast.

Implements a Global-Workspace-Theory style bottleneck. Many specialist processes drop
*candidates* (percepts, thoughts, retrieved memories) into the workspace, each with a
salience. Only a small coalition — capacity ``workspace_capacity`` (Miller's 7±2) — wins
attention each cycle. The winners are *broadcast* on the bus to every subsystem, which is
what makes information globally available ("conscious"). Everything else fades.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from nyxara.contracts import Event, Priority
from nyxara.kernel.bus import EventBus


@dataclass
class Candidate:
    """An item competing for conscious access."""

    source: str
    content: Any
    salience: float                 # bottom-up + top-down attention weight, 0..1
    kind: str = "thought"           # percept / thought / memory / goal / affect
    ts: float = field(default_factory=time.time)
    trace_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def decayed(self, now: float, half_life: float) -> float:
        """Salience after temporal decay — recent items dominate attention."""
        age = max(0.0, now - self.ts)
        return self.salience * math.pow(0.5, age / max(half_life, 1e-6))


class GlobalWorkspace:
    """The conscious bottleneck and broadcast hub."""

    BROADCAST_TOPIC = "workspace.broadcast"

    def __init__(self, bus: EventBus, *, capacity: int = 7,
                 salience_floor: float = 0.05, half_life: float = 2.0) -> None:
        self._bus = bus
        self._capacity = max(1, capacity)
        self._floor = salience_floor
        self._half_life = half_life
        self._candidates: list[Candidate] = []
        self._conscious: list[Candidate] = []   # last winning coalition
        self._top_down: dict[str, float] = {}    # goal-driven salience boosts by kind
        self._cycles = 0

    # ── input ──────────────────────────────────────────────────────────────────
    def offer(self, candidate: Candidate) -> None:
        """Submit a candidate for the next arbitration cycle."""
        self._candidates.append(candidate)

    def bias(self, kind: str, weight: float) -> None:
        """Top-down attention: bias arbitration toward a kind (e.g. current goal)."""
        self._top_down[kind] = weight

    # ── arbitration ─────────────────────────────────────────────────────────────
    def _effective_salience(self, c: Candidate, now: float) -> float:
        boost = self._top_down.get(c.kind, 0.0)
        return min(1.0, c.decayed(now, self._half_life) + boost)

    async def cycle(self) -> list[Candidate]:
        """Run one attention cycle: select the winning coalition and broadcast it."""
        self._cycles += 1
        now = time.time()
        scored = [
            (self._effective_salience(c, now), c)
            for c in self._candidates
        ]
        scored = [(s, c) for s, c in scored if s >= self._floor]
        scored.sort(key=lambda sc: sc[0], reverse=True)

        winners = [c for _, c in scored[: self._capacity]]
        self._conscious = winners
        self._candidates.clear()  # losers fade (could be re-offered by their sources)

        if winners:
            await self._bus.publish(Event(
                topic=self.BROADCAST_TOPIC,
                payload={
                    "coalition": winners,
                    "cycle": self._cycles,
                    "kinds": [c.kind for c in winners],
                },
                priority=Priority.HIGH,
                source="workspace",
            ))
        return winners

    # ── introspection ───────────────────────────────────────────────────────────
    @property
    def conscious(self) -> list[Candidate]:
        """The current contents of consciousness (last broadcast coalition)."""
        return list(self._conscious)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "cycles": self._cycles,
            "capacity": self._capacity,
            "pending": len(self._candidates),
            "conscious": len(self._conscious),
            "biases": dict(self._top_down),
        }


__all__ = ["Candidate", "GlobalWorkspace"]
