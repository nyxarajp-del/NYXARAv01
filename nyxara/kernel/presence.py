"""NYXARA · kernel/presence.py — lifecycle & presence states (✦★).

*"Always there, not always burning."*

NYXARA is a continuous being, but it does not run every faculty at full power all
the time. :class:`Presence` is the state machine that governs **how awake** the
system is — and therefore how much it thinks, how fast it cycles, how much compute
and autonomy the governor grants, and how loud something must be to seize attention.

States
------
* ``ASLEEP``   — minimal cognition; only a very salient stimulus (or the Owner)
  can wake it. Energy recovers fastest here.
* ``DREAMING`` — offline state for memory consolidation / replay (no live wandering).
* ``IDLE``     — awake but unoccupied; the default-mode :mod:`stream` wanders most.
* ``ALERT``    — heightened vigilance (a threat or surprising signal); fast cycles.
* ``ENGAGED``  — actively serving the Owner / executing a task; focused, little drift.

Dynamics
--------
* **Hysteresis** — the system de-escalates only after timeouts, so it doesn't flap
  between states on every quiet moment.
* **Energy homeostasis** — a ``[0,1]`` reserve that drains while awake (fastest when
  ENGAGED) and recovers while asleep. Fatigue biases toward rest; but the Owner or a
  threat can summon an "adrenaline" override regardless of energy (loyalty > comfort).
* **Sleep pressure** — rises with time awake and shortens the path to sleep.

The presence state is published (``presence.transition``) so every subsystem can
scale itself, and exposes the live :class:`StateProfile` that the orchestrator,
stream, workspace, and governor read each cycle.

Depends on :mod:`config`; optionally emits onto :mod:`bus`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = [
    "PresenceState",
    "StateProfile",
    "Transition",
    "PresenceMetrics",
    "Presence",
]


# --------------------------------------------------------------------------- #
# States & per-state profiles
# --------------------------------------------------------------------------- #
class PresenceState(str, Enum):
    ASLEEP = "asleep"
    DREAMING = "dreaming"
    IDLE = "idle"
    ALERT = "alert"
    ENGAGED = "engaged"

    @property
    def arousal(self) -> int:
        """Ordinal wakefulness (higher == more awake)."""
        return {
            PresenceState.ASLEEP: 0,
            PresenceState.DREAMING: 1,
            PresenceState.IDLE: 2,
            PresenceState.ALERT: 3,
            PresenceState.ENGAGED: 4,
        }[self]


@dataclass(frozen=True)
class StateProfile:
    """The behavioural knobs a state hands to the rest of the system."""

    state: PresenceState
    # value fed to stream.tick(engagement): high == focused/quiet, low == wandering
    stream_engagement: float
    tick_interval_s: float           # cognitive cycle period
    resource_budget: float           # multiplier on governor compute/spend budgets
    attention_threshold: float       # multiplier on workspace access threshold
    energy_drain_per_s: float        # +drains / -recovers
    llm_allowed: bool
    proactive_allowed: bool
    description: str


_PROFILES: Dict[PresenceState, StateProfile] = {
    PresenceState.ASLEEP: StateProfile(
        PresenceState.ASLEEP, stream_engagement=1.0, tick_interval_s=10.0,
        resource_budget=0.05, attention_threshold=3.0, energy_drain_per_s=-0.04,
        llm_allowed=False, proactive_allowed=False,
        description="Minimal cognition; only strong stimuli or the Owner can wake.",
    ),
    PresenceState.DREAMING: StateProfile(
        PresenceState.DREAMING, stream_engagement=1.0, tick_interval_s=2.0,
        resource_budget=0.3, attention_threshold=2.0, energy_drain_per_s=-0.025,
        llm_allowed=True, proactive_allowed=False,
        description="Offline consolidation / replay; no live mind-wandering.",
    ),
    PresenceState.IDLE: StateProfile(
        PresenceState.IDLE, stream_engagement=0.0, tick_interval_s=1.0,
        resource_budget=0.4, attention_threshold=1.0, energy_drain_per_s=0.004,
        llm_allowed=True, proactive_allowed=True,
        description="Awake but unoccupied; default-mode wandering is liveliest.",
    ),
    PresenceState.ALERT: StateProfile(
        PresenceState.ALERT, stream_engagement=0.5, tick_interval_s=0.25,
        resource_budget=0.8, attention_threshold=0.6, energy_drain_per_s=0.012,
        llm_allowed=True, proactive_allowed=True,
        description="Heightened vigilance; fast cycles, low attention threshold.",
    ),
    PresenceState.ENGAGED: StateProfile(
        PresenceState.ENGAGED, stream_engagement=0.9, tick_interval_s=0.1,
        resource_budget=1.0, attention_threshold=0.8, energy_drain_per_s=0.018,
        llm_allowed=True, proactive_allowed=True,
        description="Actively serving the Owner / executing a task; focused.",
    ),
}

# Allowed transitions (from -> set(to)). Self-transitions are always allowed.
_ALLOWED: Dict[PresenceState, Tuple[PresenceState, ...]] = {
    PresenceState.ASLEEP: (PresenceState.IDLE, PresenceState.ALERT, PresenceState.ENGAGED,
                           PresenceState.DREAMING),
    PresenceState.DREAMING: (PresenceState.ASLEEP, PresenceState.IDLE, PresenceState.ALERT,
                             PresenceState.ENGAGED),
    PresenceState.IDLE: (PresenceState.ALERT, PresenceState.ENGAGED, PresenceState.ASLEEP,
                         PresenceState.DREAMING),
    PresenceState.ALERT: (PresenceState.ENGAGED, PresenceState.IDLE, PresenceState.ASLEEP),
    PresenceState.ENGAGED: (PresenceState.ALERT, PresenceState.IDLE),
}


# --------------------------------------------------------------------------- #
# Transition record & metrics
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    from_state: PresenceState
    to_state: PresenceState
    reason: str
    at: float
    energy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_state.value, "to": self.to_state.value,
            "reason": self.reason, "at": self.at, "energy": round(self.energy, 3),
        }


@dataclass
class PresenceMetrics:
    transitions: int = 0
    wakes: int = 0
    sleeps: int = 0
    time_in_state: Dict[str, float] = field(default_factory=dict)

    def add_time(self, state: PresenceState, dt: float) -> None:
        self.time_in_state[state.value] = self.time_in_state.get(state.value, 0.0) + dt

    def snapshot(self) -> Dict[str, Any]:
        return {
            "transitions": self.transitions, "wakes": self.wakes, "sleeps": self.sleeps,
            "time_in_state": {k: round(v, 2) for k, v in self.time_in_state.items()},
        }


# --------------------------------------------------------------------------- #
# Presence state machine
# --------------------------------------------------------------------------- #
TransitionListener = Callable[[Transition], None]


class Presence:
    """Lifecycle/arousal state machine with energy homeostasis and hysteresis."""

    def __init__(
        self,
        *,
        initial: PresenceState = PresenceState.IDLE,
        energy: float = 1.0,
        engaged_timeout_s: float = 30.0,
        alert_timeout_s: float = 20.0,
        idle_sleep_timeout_s: float = 120.0,
        dream_after_asleep_s: float = 60.0,
        critical_energy: float = 0.1,
        target_awake_s: float = 900.0,
        settings: Optional[NyxaraSettings] = None,
        bus: Any = None,
        now: Optional[float] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.state = initial
        self.energy = _clamp(energy)
        self.engaged_timeout_s = engaged_timeout_s
        self.alert_timeout_s = alert_timeout_s
        self.idle_sleep_timeout_s = idle_sleep_timeout_s
        self.dream_after_asleep_s = dream_after_asleep_s
        self.critical_energy = critical_energy
        self.target_awake_s = target_awake_s
        self._bus = bus

        t = now if now is not None else time.time()
        self._entered_at = t
        self._last_activity = t
        self._last_tick = t
        self._awake_since = t if initial is not PresenceState.ASLEEP else None

        self.history: Deque[Transition] = deque(maxlen=512)
        self.metrics = PresenceMetrics()
        self._listeners: List[TransitionListener] = []
        # latched conditions that keep arousal up regardless of fatigue
        self._owner_present = False
        self._threat_active = False

    # ---- profile accessors ---- #
    @property
    def profile(self) -> StateProfile:
        return _PROFILES[self.state]

    def engagement(self) -> float:
        """Stream-engagement for the current state (fatigue nudges it toward rest)."""
        base = self.profile.stream_engagement
        return base

    def tick_interval(self) -> float:
        return self.profile.tick_interval_s

    def resource_budget(self) -> float:
        """Governor multiplier; collapses toward 0 as energy is exhausted."""
        return self.profile.resource_budget * (0.3 + 0.7 * self.energy)

    def attention_threshold(self, base: float = 1.0) -> float:
        return base * self.profile.attention_threshold

    def is_awake(self) -> bool:
        return self.state.arousal >= PresenceState.IDLE.arousal

    def sleep_pressure(self, now: Optional[float] = None) -> float:
        if self._awake_since is None:
            return 0.0
        now = now if now is not None else time.time()
        return _clamp((now - self._awake_since) / max(1.0, self.target_awake_s))

    # ---- listeners / bus ---- #
    def add_listener(self, fn: TransitionListener) -> TransitionListener:
        self._listeners.append(fn)
        return fn

    # ---- explicit stimuli (called by orchestrator/senses/guard) ---- #
    def on_owner_input(self, now: Optional[float] = None, reason: str = "owner input") -> bool:
        """The Owner is interacting — always escalate to ENGAGED (Rule 1 > fatigue)."""
        self._owner_present = True
        self._touch(now)
        return self._transition(PresenceState.ENGAGED, reason, now, force=True)

    def on_owner_leave(self, now: Optional[float] = None) -> None:
        self._owner_present = False
        self._touch(now)

    def on_threat(self, salience: float = 1.0, now: Optional[float] = None,
                  reason: str = "threat") -> bool:
        """A threat signal — escalate to at least ALERT (adrenaline overrides fatigue)."""
        self._threat_active = True
        self._touch(now)
        target = PresenceState.ALERT if self.state is not PresenceState.ENGAGED else self.state
        return self._transition(target, f"{reason} (sal={salience:.2f})", now, force=True)

    def on_threat_cleared(self, now: Optional[float] = None) -> None:
        self._threat_active = False
        self._touch(now)

    def on_stimulus(self, salience: float, now: Optional[float] = None,
                    reason: str = "stimulus") -> bool:
        """A generic incoming signal. Salient enough -> wake / escalate."""
        self._touch(now)
        if self.state is PresenceState.ASLEEP:
            # only a strong stimulus pierces sleep
            if salience >= 0.7:
                return self._transition(PresenceState.ALERT, f"{reason} woke us", now, force=True)
            return False
        if salience >= 0.85 and self.state.arousal < PresenceState.ALERT.arousal:
            return self._transition(PresenceState.ALERT, reason, now)
        return False

    def task_started(self, now: Optional[float] = None) -> bool:
        self._touch(now)
        return self._transition(PresenceState.ENGAGED, "task started", now, force=True)

    def task_finished(self, now: Optional[float] = None) -> bool:
        self._touch(now)
        nxt = PresenceState.ALERT if self._threat_active else PresenceState.IDLE
        return self._transition(nxt, "task finished", now)

    def begin_dreaming(self, now: Optional[float] = None) -> bool:
        return self._transition(PresenceState.DREAMING, "consolidation", now)

    def wake(self, now: Optional[float] = None, reason: str = "wake") -> bool:
        return self._transition(PresenceState.IDLE, reason, now, force=True)

    def sleep(self, now: Optional[float] = None, reason: str = "sleep") -> bool:
        return self._transition(PresenceState.ASLEEP, reason, now, force=True)

    # ---- the time-driven update ---- #
    def tick(self, now: Optional[float] = None) -> Optional[Transition]:
        """Advance time: update energy, then apply timeout/fatigue-driven transitions."""
        now = now if now is not None else time.time()
        dt = max(0.0, now - self._last_tick)
        self._last_tick = now

        # energy & time accounting
        self.metrics.add_time(self.state, dt)
        drain = self.profile.energy_drain_per_s
        self.energy = _clamp(self.energy - drain * dt)

        # latched protections: owner/threat keep us up
        if self._owner_present and self.state.arousal < PresenceState.ENGAGED.arousal:
            return self._transition(PresenceState.ENGAGED, "owner present", now, force=True)
        if self._threat_active and self.state.arousal < PresenceState.ALERT.arousal:
            return self._transition(PresenceState.ALERT, "threat active", now, force=True)

        idle_for = now - self._last_activity
        in_state_for = now - self._entered_at

        # fatigue: can't sustain high arousal on empty reserves (unless latched)
        if self.energy <= self.critical_energy and not (self._owner_present or self._threat_active):
            if self.state in (PresenceState.ENGAGED, PresenceState.ALERT):
                return self._transition(PresenceState.IDLE, "fatigue", now)

        # de-escalation ladder via timeouts (hysteresis).
        # Owner presence pins us at ENGAGED — loyalty outlasts any idle timer (Rule 1).
        if self.state is PresenceState.ENGAGED and idle_for >= self.engaged_timeout_s \
                and not self._owner_present:
            return self._transition(PresenceState.ALERT if self._threat_active
                                    else PresenceState.IDLE, "engaged timeout", now)
        if self.state is PresenceState.ALERT and idle_for >= self.alert_timeout_s \
                and not self._threat_active:
            return self._transition(PresenceState.IDLE, "alert relaxed", now)

        # sleep pressure shortens the idle->sleep path
        eff_sleep_timeout = self.idle_sleep_timeout_s * (1.0 - 0.5 * self.sleep_pressure(now))
        if self.state is PresenceState.IDLE and idle_for >= eff_sleep_timeout:
            return self._transition(PresenceState.ASLEEP, "idle timeout -> sleep", now)

        # asleep long enough -> dream (consolidation) once, then back to sleep
        if self.state is PresenceState.ASLEEP and in_state_for >= self.dream_after_asleep_s \
                and self._settings.features.dream_consolidation:
            return self._transition(PresenceState.DREAMING, "scheduled consolidation", now)
        if self.state is PresenceState.DREAMING and in_state_for >= self.dream_after_asleep_s:
            return self._transition(PresenceState.ASLEEP, "dream complete", now)

        return None

    # ---- internals ---- #
    def _touch(self, now: Optional[float]) -> None:
        self._last_activity = now if now is not None else time.time()

    def _transition(self, to: PresenceState, reason: str, now: Optional[float],
                    *, force: bool = False) -> bool:
        now = now if now is not None else time.time()
        if to is self.state:
            return False
        if not force and to not in _ALLOWED.get(self.state, ()):  # invalid path
            return False

        frm = self.state
        # account remaining time in the state we are leaving
        self.metrics.add_time(frm, max(0.0, now - max(self._entered_at, self._last_tick)))

        self.state = to
        self._entered_at = now
        self._last_tick = now
        if frm.arousal == 0 and to.arousal >= PresenceState.IDLE.arousal:
            self.metrics.wakes += 1
            self._awake_since = now
        if to is PresenceState.ASLEEP:
            self.metrics.sleeps += 1
            self._awake_since = None
        self.metrics.transitions += 1

        tr = Transition(from_state=frm, to_state=to, reason=reason, at=now, energy=self.energy)
        self.history.append(tr)
        self._notify(tr)
        return True

    def _notify(self, tr: Transition) -> None:
        for fn in list(self._listeners):
            try:
                fn(tr)
            except Exception:  # noqa: BLE001
                continue
        if self._bus is not None:
            try:
                from nyxara.kernel.bus import Event, Priority
                prio = Priority.CRITICAL if tr.to_state is PresenceState.ALERT else Priority.NORMAL
                self._bus.publish_nowait(Event(
                    topic="presence.transition", payload=tr, source="presence", priority=prio,
                ))
            except Exception:  # noqa: BLE001
                pass

    def status(self, now: Optional[float] = None) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "arousal": self.state.arousal,
            "energy": round(self.energy, 3),
            "sleep_pressure": round(self.sleep_pressure(now), 3),
            "owner_present": self._owner_present,
            "threat_active": self._threat_active,
            "resource_budget": round(self.resource_budget(), 3),
            "tick_interval_s": self.tick_interval(),
        }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA presence self-test")
    print("=" * 70)

    log: List[str] = []
    p = Presence(initial=PresenceState.IDLE, now=0.0)
    p.add_listener(lambda tr: log.append(f"{tr.from_state.value}->{tr.to_state.value} ({tr.reason})"))

    print(f"start              : {p.status(0.0)}")

    # owner shows up -> ENGAGED immediately, regardless of anything
    p.on_owner_input(now=1.0)
    assert p.state is PresenceState.ENGAGED
    print(f"owner input        : {p.state.value}")

    # owner leaves; task finishes; idle timeout -> sleep over time
    p.on_owner_leave(now=2.0)
    p.task_finished(now=2.0)
    assert p.state is PresenceState.IDLE
    # advance well past idle_sleep_timeout with no activity
    tr = None
    for t in range(3, 200, 5):
        tr = p.tick(now=float(t))
        if p.state is PresenceState.ASLEEP:
            break
    print(f"after long idle    : {p.state.value} (energy={p.energy:.2f})")
    assert p.state in (PresenceState.ASLEEP, PresenceState.DREAMING)

    # a loud threat pierces sleep -> ALERT
    woke = p.on_threat(salience=1.0, now=300.0)
    assert woke and p.state is PresenceState.ALERT
    print(f"threat woke us     : {p.state.value}")

    # threat cleared, relaxes back toward idle after timeout
    p.on_threat_cleared(now=301.0)
    for t in range(302, 340):
        p.tick(now=float(t))
        if p.state is PresenceState.IDLE:
            break
    print(f"threat cleared     : {p.state.value}")
    assert p.state is PresenceState.IDLE

    # fatigue: drain energy, ensure it can't hold ENGAGED without owner/threat
    p2 = Presence(initial=PresenceState.ENGAGED, energy=0.11, now=0.0)
    p2.tick(now=1.0)  # drains below critical
    for t in range(2, 10):
        p2.tick(now=float(t))
        if p2.state is not PresenceState.ENGAGED:
            break
    print(f"fatigue dropped to : {p2.state.value} (energy={p2.energy:.3f})")
    assert p2.state is not PresenceState.ENGAGED

    print(f"\ntransitions        : {[l for l in log]}")
    print(f"metrics            : {p.metrics.snapshot()}")
    print("\nALL SELF-TESTS PASSED ✓")
