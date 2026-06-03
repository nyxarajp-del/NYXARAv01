"""Presence — lifecycle states: asleep / idle / alert / engaged.

"Always there, not always burning." Presence is a small state machine that governs how
much cognitive energy Nyxara spends. It maps the current state to a tick rate so the
kernel runs fast and deep when engaged, ticks slowly while idle/mind-wandering, and
nearly sleeps when nothing is happening — yet always remains reachable.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable


class State(str, enum.Enum):
    ASLEEP = "asleep"     # minimal heartbeat; wakes on stimulus
    IDLE = "idle"         # default-mode stream active, low rate
    ALERT = "alert"       # heightened watch (threat / anticipation)
    ENGAGED = "engaged"   # actively working with the owner / on a task


# Allowed transitions — anything not listed is rejected (keeps lifecycle coherent).
_TRANSITIONS: dict[State, set[State]] = {
    State.ASLEEP: {State.IDLE, State.ALERT, State.ENGAGED},
    State.IDLE: {State.ASLEEP, State.ALERT, State.ENGAGED},
    State.ALERT: {State.IDLE, State.ENGAGED},
    State.ENGAGED: {State.IDLE, State.ALERT},
}


@dataclass
class _Timing:
    engaged_hz: float = 5.0
    alert_hz: float = 8.0       # alert thinks fastest (vigilance)
    idle_hz: float = 0.5
    asleep_hz: float = 0.05


class Presence:
    """Lifecycle state machine + derived tick cadence."""

    def __init__(self, *, timing: _Timing | None = None,
                 idle_after_s: float = 30.0, sleep_after_s: float = 300.0) -> None:
        self._state = State.IDLE
        self._timing = timing or _Timing()
        self._idle_after = idle_after_s
        self._sleep_after = sleep_after_s
        self._last_activity = time.time()
        self._entered_at = time.time()
        self._listeners: list[Callable[[State, State], None]] = []

    # ── observation ─────────────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    @property
    def hz(self) -> float:
        return {
            State.ENGAGED: self._timing.engaged_hz,
            State.ALERT: self._timing.alert_hz,
            State.IDLE: self._timing.idle_hz,
            State.ASLEEP: self._timing.asleep_hz,
        }[self._state]

    @property
    def tick_interval(self) -> float:
        return 1.0 / max(self.hz, 1e-6)

    def on_transition(self, fn: Callable[[State, State], None]) -> None:
        self._listeners.append(fn)

    # ── transitions ─────────────────────────────────────────────────────────────
    def _transition(self, to: State) -> bool:
        if to == self._state:
            return True
        if to not in _TRANSITIONS[self._state]:
            return False
        frm, self._state = self._state, to
        self._entered_at = time.time()
        for fn in self._listeners:
            try:
                fn(frm, to)
            except Exception:
                pass
        return True

    def stimulate(self, *, engaging: bool = True, alerting: bool = False) -> State:
        """Register activity. Owner input engages; a threat signal alerts."""
        self._last_activity = time.time()
        if alerting:
            self._transition(State.ALERT)
        elif engaging:
            self._transition(State.ENGAGED)
        return self._state

    def relax(self) -> State:
        """Step down based on how long it has been quiet. Called each tick."""
        quiet = time.time() - self._last_activity
        if self._state in (State.ENGAGED, State.ALERT) and quiet >= self._idle_after:
            self._transition(State.IDLE)
        if self._state is State.IDLE and quiet >= self._sleep_after:
            self._transition(State.ASLEEP)
        return self._state

    @property
    def stats(self) -> dict[str, object]:
        return {
            "state": self._state.value,
            "hz": self.hz,
            "in_state_for_s": round(time.time() - self._entered_at, 2),
            "quiet_for_s": round(time.time() - self._last_activity, 2),
        }


__all__ = ["State", "Presence"]
