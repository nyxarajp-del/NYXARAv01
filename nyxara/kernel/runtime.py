"""Runtime — async engine, circuit breaker, saga manager, lifecycle.

Infrastructure the orchestrator leans on:
  * :class:`CircuitBreaker` — protects flaky faculties / tools; opens after repeated
    failures, half-opens to probe recovery, closes when healthy again.
  * :class:`Saga` — multi-step actions with compensation: if step *k* fails, the already
    completed steps are rolled back in reverse (best-effort), keeping state consistent.
  * :class:`Lifecycle` — orderly startup / shutdown of registered components.
"""
from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from nyxara.kernel.errors import TransientError


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────
class BreakerState(str, enum.Enum):
    CLOSED = "closed"        # healthy, calls pass through
    OPEN = "open"            # failing, calls short-circuit
    HALF_OPEN = "half_open"  # probing recovery


class CircuitBreaker:
    def __init__(self, *, fail_threshold: int = 5, reset_timeout: float = 30.0,
                 name: str = "breaker") -> None:
        self.name = name
        self._state = BreakerState.CLOSED
        self._fail_threshold = fail_threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.OPEN and \
                time.time() - self._opened_at >= self._reset_timeout:
            self._state = BreakerState.HALF_OPEN
        return self._state

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = time.time()

    async def call(self, fn: Callable[[], Awaitable[object]]) -> object:
        if self.state is BreakerState.OPEN:
            raise TransientError(f"circuit '{self.name}' is open")
        try:
            result = await fn()
        except Exception:
            self._failures += 1
            if self._failures >= self._fail_threshold or \
                    self._state is BreakerState.HALF_OPEN:
                self._trip()
            raise
        else:
            self._failures = 0
            self._state = BreakerState.CLOSED
            return result


# ─────────────────────────────────────────────────────────────────────────────
#  Saga — orchestrated steps with compensation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    name: str
    action: Callable[[], Awaitable[object]]
    compensate: Callable[[], Awaitable[None]] | None = None


class Saga:
    """Sequential steps; on failure, completed steps are compensated in reverse."""

    def __init__(self, name: str = "saga") -> None:
        self.name = name
        self._steps: list[Step] = []

    def add(self, name: str, action: Callable[[], Awaitable[object]],
            compensate: Callable[[], Awaitable[None]] | None = None) -> "Saga":
        self._steps.append(Step(name, action, compensate))
        return self

    async def run(self) -> list[object]:
        done: list[Step] = []
        results: list[object] = []
        try:
            for step in self._steps:
                results.append(await step.action())
                done.append(step)
            return results
        except Exception:
            for step in reversed(done):
                if step.compensate is not None:
                    try:
                        await step.compensate()
                    except Exception:
                        pass  # best-effort rollback; never mask the original failure
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  Lifecycle
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Component:
    name: str
    start: Callable[[], Awaitable[None]] | None = None
    stop: Callable[[], Awaitable[None]] | None = None


class Lifecycle:
    """Ordered startup / reverse-ordered shutdown of components."""

    def __init__(self) -> None:
        self._components: list[Component] = []
        self._started: list[Component] = []

    def register(self, component: Component) -> None:
        self._components.append(component)

    async def startup(self) -> None:
        for c in self._components:
            if c.start is not None:
                await c.start()
            self._started.append(c)

    async def shutdown(self) -> None:
        for c in reversed(self._started):
            if c.stop is not None:
                try:
                    await c.stop()
                except Exception:
                    pass
        self._started.clear()


__all__ = [
    "BreakerState", "CircuitBreaker",
    "Step", "Saga",
    "Component", "Lifecycle",
]
