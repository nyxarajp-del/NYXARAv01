"""NYXARA · agency/health.py — self-health monitoring & self-healing (♥, watchdog).

A sovereign that cannot feel its own body fails silently. This module is NYXARA's
**interoception for her machinery**: every subsystem reports a heartbeat and its error
rate, pull-based checks probe resources, and the monitor fuses these into a live verdict
— ``HEALTHY``, ``DEGRADED``, ``UNHEALTHY`` or ``DOWN`` — both per-subsystem and overall.

When a subsystem degrades, the monitor does not merely note it: registered **recovery
handlers** are fired to try to bring it back (a watchdog). Recovery is **bounded** — a
subsystem that cannot be healed after a few attempts is *quarantined* and flagged for the
Master rather than looped on forever. Serious classes of failure (security / invariant /
corrigibility, surfaced through the error ledger) are never quietly "healed": they raise
the overall state and escalate, fail-closed.

The monitor binds optionally to the kernel's :class:`~nyxara.kernel.errors.ErrorLedger`
(to see what is actually failing) and the :class:`~nyxara.agency.governor.Governor` (to
feel resource pressure — an exhausted budget or saturated concurrency degrades health).
Everything is clock-injectable, so liveness/staleness logic is exactly testable with no
real time.

Depends on :mod:`kernel.errors` (ledger) and optionally :mod:`agency.governor`. Pure
standard library otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from nyxara.kernel.errors import ErrorLedger

__all__ = [
    "HealthState",
    "Subsystem",
    "HealthReport",
    "HealthMonitor",
]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class HealthState(IntEnum):
    """Ordered by severity — a higher value is worse, so ``max`` gives the worst."""
    HEALTHY = 0
    UNKNOWN = 1
    DEGRADED = 2
    UNHEALTHY = 3
    DOWN = 4

    @property
    def label(self) -> str:
        return self.name.lower()


def _worst(*states: HealthState) -> HealthState:
    return HealthState(max(int(s) for s in states)) if states else HealthState.UNKNOWN


# a pull check returns a state, or a (state, detail) pair
CheckResult = Union[HealthState, Tuple[HealthState, str]]
Check = Callable[[], CheckResult]
RecoveryHandler = Callable[["Subsystem"], bool]


# --------------------------------------------------------------------------- #
# Subsystem
# --------------------------------------------------------------------------- #
@dataclass
class Subsystem:
    name: str
    heartbeat_timeout: float = 30.0       # seconds before a beat is "stale"
    down_grace: float = 3.0               # multiple of timeout before "down"
    degraded_error_rate: float = 0.1
    unhealthy_error_rate: float = 0.3
    ewma_alpha: float = 0.3
    # live metrics:
    last_beat: Optional[float] = None
    beats: int = 0
    successes: int = 0
    errors: int = 0
    error_rate: float = 0.0               # EWMA of failures
    latency_ewma: float = 0.0
    state: HealthState = HealthState.UNKNOWN
    recovery_attempts: int = 0
    quarantined: bool = False
    message: str = ""

    @property
    def has_data(self) -> bool:
        return self.beats > 0 or self.successes > 0 or self.errors > 0

    def beat(self, now: float, latency: Optional[float] = None) -> None:
        self.last_beat = now
        self.beats += 1
        if latency is not None:
            self.latency_ewma = (self.ewma_alpha * latency
                                 + (1 - self.ewma_alpha) * self.latency_ewma)

    def record(self, ok: bool) -> None:
        x = 0.0 if ok else 1.0
        self.error_rate = self.ewma_alpha * x + (1 - self.ewma_alpha) * self.error_rate
        if ok:
            self.successes += 1
        else:
            self.errors += 1

    def _error_state(self) -> HealthState:
        if self.error_rate >= self.unhealthy_error_rate:
            return HealthState.UNHEALTHY
        if self.error_rate >= self.degraded_error_rate:
            return HealthState.DEGRADED
        return HealthState.HEALTHY

    def evaluate(self, now: float) -> HealthState:
        if self.quarantined:
            self.state = HealthState.DOWN
            return self.state
        if not self.has_data:
            self.state = HealthState.UNKNOWN
            return self.state
        state = self._error_state()
        if self.last_beat is not None:
            age = now - self.last_beat
            if age > self.heartbeat_timeout * self.down_grace:
                state = _worst(state, HealthState.DOWN)
                self.message = f"no heartbeat for {age:.0f}s"
            elif age > self.heartbeat_timeout:
                state = _worst(state, HealthState.UNHEALTHY)
                self.message = f"heartbeat stale ({age:.0f}s)"
        self.state = state
        return state

    def metrics(self) -> Dict[str, Any]:
        return {"name": self.name, "state": self.state.label,
                "error_rate": round(self.error_rate, 4),
                "latency_ewma": round(self.latency_ewma, 4), "beats": self.beats,
                "errors": self.errors, "successes": self.successes,
                "recovery_attempts": self.recovery_attempts,
                "quarantined": self.quarantined, "message": self.message}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class HealthReport:
    at: float
    overall: HealthState
    subsystems: List[Dict[str, Any]]
    recovered: List[str] = field(default_factory=list)
    escalate: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.overall <= HealthState.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "overall": self.overall.label, "healthy": self.healthy,
                "subsystems": self.subsystems, "recovered": self.recovered,
                "escalate": self.escalate}


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
class HealthMonitor:
    """Fuses heartbeats, error rates and pull checks; fires bounded recovery."""

    def __init__(self, *, clock=None, max_recovery_attempts: int = 3) -> None:
        self._clock = clock or time.monotonic
        self.max_recovery_attempts = max_recovery_attempts
        self._subsystems: Dict[str, Subsystem] = {}
        self._checks: Dict[str, Check] = {}
        self._recovery: Dict[str, List[RecoveryHandler]] = {}
        self._events: List[Dict[str, Any]] = []

    # ---- registration ---- #
    def register(self, name: str, **kw: Any) -> Subsystem:
        s = Subsystem(name=name, **kw)
        self._subsystems[name] = s
        return s

    def _ensure(self, name: str) -> Subsystem:
        s = self._subsystems.get(name)
        if s is None:
            s = self.register(name)
        return s

    def register_check(self, name: str, fn: Check) -> None:
        self._ensure(name)
        self._checks[name] = fn

    def on_recover(self, name: str, handler: RecoveryHandler) -> None:
        self._ensure(name)
        self._recovery.setdefault(name, []).append(handler)

    # ---- signals ---- #
    def beat(self, name: str, *, latency: Optional[float] = None) -> None:
        self._ensure(name).beat(self._clock(), latency)

    def record_success(self, name: str) -> None:
        self._ensure(name).record(True)

    def record_error(self, name: str) -> None:
        self._ensure(name).record(False)

    # ---- optional integrations ---- #
    def bind_governor(self, governor: Any) -> None:
        """Feel resource pressure: an exhausted budget or saturated concurrency degrades."""
        def _check() -> CheckResult:
            st = governor.status()
            spend, conc = st["spend"], st["concurrency"]
            state = HealthState.HEALTHY
            msgs: List[str] = []
            if spend["limit"] > 0 and spend["remaining"] <= 0:
                state = _worst(state, HealthState.DEGRADED)
                msgs.append("spend budget exhausted")
            if conc["max"] > 0 and conc["active"] >= conc["max"]:
                state = _worst(state, HealthState.DEGRADED)
                msgs.append("concurrency saturated")
            return state, "; ".join(msgs)
        self.register_check("resources", _check)

    def bind_error_ledger(self, ledger: ErrorLedger) -> None:
        """Surface what is actually failing; never auto-heal security/invariant classes."""
        def _check() -> CheckResult:
            summ = ledger.summary()
            by = summ.get("by_category", {})
            serious = by.get("invariant", 0) + by.get("security", 0) + by.get("corrigibility", 0)
            if serious > 0:
                return HealthState.UNHEALTHY, f"{serious} serious failures (fail-closed)"
            if summ.get("total_occurrences", 0) > 0:
                return HealthState.DEGRADED, f"{summ['total_occurrences']} errors logged"
            return HealthState.HEALTHY, ""
        self.register_check("errors", _check)

    # ---- evaluation + recovery ---- #
    def _run_check(self, name: str) -> Tuple[HealthState, str]:
        res = self._checks[name]()
        if isinstance(res, tuple):
            return res[0], res[1]
        return res, ""

    def evaluate(self) -> HealthReport:
        now = self._clock()
        recovered: List[str] = []
        escalate: List[str] = []

        for name, s in self._subsystems.items():
            prev = s.state
            state = s.evaluate(now)
            if name in self._checks:
                cstate, cmsg = self._run_check(name)
                # for a check-only subsystem (no heartbeat data) the check is
                # authoritative; otherwise take the worse of the two
                state = cstate if not s.has_data else _worst(state, cstate)
                if cmsg:
                    s.message = cmsg
                s.state = state

            # bounded self-healing on serious degradation
            if state >= HealthState.UNHEALTHY and not s.quarantined:
                if self._attempt_recovery(s):
                    recovered.append(name)
            # a quarantined subsystem always needs the Master (fail-closed)
            if s.quarantined:
                escalate.append(name)
            if prev != s.state:
                self._events.append({"at": now, "subsystem": name,
                                    "from": prev.label, "to": s.state.label})

        subs = [s.metrics() for s in self._subsystems.values()]
        overall = _worst(*[s.state for s in self._subsystems.values()]) \
            if self._subsystems else HealthState.UNKNOWN
        return HealthReport(at=now, overall=overall, subsystems=subs,
                            recovered=recovered, escalate=escalate)

    def _attempt_recovery(self, s: Subsystem) -> bool:
        handlers = self._recovery.get(s.name, [])
        if not handlers:
            return False
        s.recovery_attempts += 1
        for h in handlers:
            try:
                if h(s):
                    s.state = s.evaluate(self._clock())
                    self._events.append({"at": self._clock(), "subsystem": s.name,
                                        "recovered_by": getattr(h, "__name__", "handler")})
                    return True
            except Exception:  # noqa: BLE001 — a failing healer must not crash the monitor
                continue
        if s.recovery_attempts >= self.max_recovery_attempts:
            s.quarantined = True
            s.message = f"quarantined after {s.recovery_attempts} failed recoveries"
        return False

    # ---- queries ---- #
    def get(self, name: str) -> Optional[Subsystem]:
        return self._subsystems.get(name)

    def liveness(self) -> bool:
        """Quick aggregate: are we at least not DOWN/UNHEALTHY anywhere?"""
        return self.evaluate().overall < HealthState.UNHEALTHY

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def report(self) -> Dict[str, Any]:
        return self.evaluate().to_dict()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.governor import Governor
    from nyxara.kernel.config import ResourceLimits

    print("=" * 70)
    print("NYXARA self-health self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    mon = HealthMonitor(clock=lambda: now["t"], max_recovery_attempts=2)

    mon.register("memory", heartbeat_timeout=10.0)
    mon.register("planner", heartbeat_timeout=10.0,
                 degraded_error_rate=0.1, unhealthy_error_rate=0.3)

    # healthy: fresh heartbeats, no errors
    mon.beat("memory", latency=0.005)
    mon.beat("planner", latency=0.02)
    for _ in range(10):
        mon.record_success("planner")
    rep = mon.evaluate()
    print(f"\ninitial overall     : {rep.overall.label} (healthy={rep.healthy})")
    assert rep.healthy

    # planner starts failing -> error rate climbs into UNHEALTHY
    for _ in range(8):
        mon.record_error("planner")
    mon.beat("planner")
    rep = mon.evaluate()
    print(f"after failures      : planner={mon.get('planner').state.label} "
          f"(error_rate={mon.get('planner').error_rate:.2f})")
    assert mon.get("planner").state >= HealthState.UNHEALTHY

    # a recovery handler that "restarts" the planner (clears its error rate)
    def restart_planner(s: Subsystem) -> bool:
        s.error_rate = 0.0
        s.beat(now["t"])
        return True
    mon.on_recover("planner", restart_planner)
    rep = mon.evaluate()
    print(f"after recovery      : planner={mon.get('planner').state.label} "
          f"recovered={rep.recovered}")
    assert "planner" in rep.recovered
    assert mon.get("planner").state <= HealthState.DEGRADED

    # memory stops beating -> goes stale, then DOWN as the clock advances
    now["t"] += 35.0   # > 3x timeout (10s)
    rep = mon.evaluate()
    print(f"stale memory        : memory={mon.get('memory').state.label}")
    assert mon.get("memory").state is HealthState.DOWN

    # a subsystem that cannot be healed gets quarantined and escalated (bounded)
    mon.register("flaky", heartbeat_timeout=10.0)
    mon.beat("flaky")
    for _ in range(8):
        mon.record_error("flaky")
    mon.on_recover("flaky", lambda s: False)   # healer always fails
    for _ in range(3):
        rep = mon.evaluate()
    print(f"quarantine          : flaky quarantined={mon.get('flaky').quarantined} "
          f"escalate={rep.escalate}")
    assert mon.get("flaky").quarantined
    assert "flaky" in rep.escalate

    # bind the governor: an exhausted budget shows up as resource pressure
    gov = Governor(ResourceLimits(daily_spend_budget=1.0))
    gov.charge(1.0)   # spend it all
    mon.bind_governor(gov)
    rep = mon.evaluate()
    print(f"resource pressure   : resources={mon.get('resources').state.label} "
          f"({mon.get('resources').message})")
    assert mon.get("resources").state >= HealthState.DEGRADED

    print(f"\noverall             : {rep.overall.label}")
    print(f"events logged       : {len(mon.events())}")
    print("\nALL SELF-TESTS PASSED ✓")
