"""NYXARA · agency/scheduler.py — the autonomous task scheduler (⏲).

NYXARA does not only react — she keeps her own queue of work: things to do now, things
to do later, things to do again on a cadence, and things that must wait for other work
to finish first. This module is that queue, made deterministic and safe.

The scheduler is a **priority queue with dependencies, delays and recurrence**, bounded
by the governor's concurrency cap and ordered so that **the Master's work always comes
first** (owner-priority preemption — a task the Master asked for is never made to wait
behind NYXARA's own initiative, and is never throttled by the autonomous concurrency
cap). Tasks may:

* carry a :class:`Priority` (``OWNER`` outranks everything);
* be **delayed** (``run_at``) and **recurring** (``every`` seconds, optionally capped);
* declare **dependencies** — a task becomes ready only once its prerequisites have
  completed, and is **blocked** (never silently run) if a prerequisite failed.

Execution is synchronous and **clock-injectable**, so the whole scheduler is exactly
testable with no real time and no threads: :meth:`Scheduler.tick` promotes due tasks,
selects the highest-priority ready set (all owner tasks plus up to the concurrency cap
of the rest), runs them, captures results/errors as data, reschedules recurring tasks,
and blocks the dependents of anything that failed.

Depends on :mod:`agency.governor` (optional concurrency cap) and :mod:`kernel.errors`.
Pure standard library otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.agency.governor import Governor

__all__ = [
    "Priority",
    "TaskState",
    "Task",
    "Scheduler",
]

Clock = Callable[[], float]


# --------------------------------------------------------------------------- #
# Priority & state
# --------------------------------------------------------------------------- #
class Priority(IntEnum):
    """Lower value = scheduled sooner. OWNER outranks all (Rule 1)."""
    OWNER = 0
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    IDLE = 5

    @property
    def label(self) -> str:
        return self.name.lower()


class TaskState(str, Enum):
    PENDING = "pending"      # waiting to run (possibly delayed or on dependencies)
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"      # a prerequisite failed/was cancelled — will never run


_TERMINAL = {TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED, TaskState.BLOCKED}


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    name: str
    fn: Callable[[], Any]
    priority: Priority = Priority.NORMAL
    run_at: float = 0.0
    deps: frozenset = frozenset()
    every: Optional[float] = None         # recurrence interval (seconds)
    max_runs: Optional[int] = None        # cap on recurrences (None = unbounded)
    seq: int = 0                          # submission order (deterministic tie-break)
    state: TaskState = TaskState.PENDING
    runs: int = 0
    result: Any = None
    error: Optional[str] = None

    @property
    def is_owner(self) -> bool:
        return self.priority is Priority.OWNER

    @property
    def recurring(self) -> bool:
        return self.every is not None

    def due(self, now: float) -> bool:
        return self.run_at <= now

    def sort_key(self):
        return (int(self.priority), self.run_at, self.seq)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "priority": self.priority.label,
                "state": self.state.value, "run_at": self.run_at,
                "deps": sorted(self.deps), "every": self.every, "runs": self.runs,
                "result": self.result, "error": self.error}


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class Scheduler:
    """A deterministic, owner-first priority scheduler with deps, delays & recurrence."""

    def __init__(self, *, clock: Optional[Clock] = None,
                 governor: Optional[Governor] = None,
                 max_concurrent: Optional[int] = None) -> None:
        self._clock = clock or time.monotonic
        self.governor = governor
        if max_concurrent is not None:
            self.max_concurrent = max_concurrent
        elif governor is not None:
            self.max_concurrent = governor.limits.max_concurrent_tasks
        else:
            self.max_concurrent = 4
        self._tasks: Dict[str, Task] = {}
        self._seq = 0

    # ---- submission ---- #
    def submit(self, name: str, fn: Callable[[], Any], *,
               priority: Priority = Priority.NORMAL, delay: float = 0.0,
               at: Optional[float] = None, deps: Sequence[str] = (),
               every: Optional[float] = None, max_runs: Optional[int] = None) -> str:
        for d in deps:
            if d not in self._tasks:
                raise ValueError(f"dependency {d!r} is not a known task")
        if every is not None and every <= 0:
            raise ValueError("recurrence interval must be positive")
        tid = f"t{self._seq}"
        run_at = at if at is not None else self._clock() + max(0.0, delay)
        self._tasks[tid] = Task(id=tid, name=name, fn=fn, priority=priority,
                                run_at=run_at, deps=frozenset(deps), every=every,
                                max_runs=max_runs, seq=self._seq)
        self._seq += 1
        return tid

    def every(self, interval: float, name: str, fn: Callable[[], Any], *,
              priority: Priority = Priority.NORMAL, max_runs: Optional[int] = None,
              delay: float = 0.0) -> str:
        """Convenience: submit a recurring task."""
        return self.submit(name, fn, priority=priority, delay=delay,
                           every=interval, max_runs=max_runs)

    def cancel(self, tid: str) -> bool:
        t = self._tasks.get(tid)
        if t is None or t.state in _TERMINAL or t.state is TaskState.RUNNING:
            return False
        t.state = TaskState.CANCELLED
        return True

    def get(self, tid: str) -> Optional[Task]:
        return self._tasks.get(tid)

    # ---- readiness ---- #
    def _deps_ok(self, t: Task) -> Optional[bool]:
        """True if all deps done; False if still pending; None if a dep blocks this task."""
        all_done = True
        for d in t.deps:
            dep = self._tasks.get(d)
            if dep is None or dep.state in {TaskState.FAILED, TaskState.CANCELLED,
                                            TaskState.BLOCKED}:
                return None  # a prerequisite can never satisfy -> block
            if dep.state is not TaskState.DONE:
                all_done = False
        return all_done

    def ready(self) -> List[Task]:
        """Tasks eligible to run now, highest priority first."""
        now = self._clock()
        out: List[Task] = []
        for t in self._tasks.values():
            if t.state is not TaskState.PENDING or not t.due(now):
                continue
            ok = self._deps_ok(t)
            if ok is None:
                t.state = TaskState.BLOCKED
            elif ok:
                out.append(t)
        out.sort(key=lambda t: t.sort_key())
        return out

    # ---- execution ---- #
    def _execute(self, t: Task) -> None:
        t.state = TaskState.RUNNING
        try:
            if self.governor is not None and not t.is_owner:
                with self.governor.slot():
                    t.result = t.fn()
            else:
                t.result = t.fn()
            t.error = None
            t.runs += 1
            self._after_success(t)
        except Exception as exc:  # noqa: BLE001 — failures are captured as data
            t.error = f"{type(exc).__name__}: {exc}"
            t.runs += 1
            t.state = TaskState.FAILED

    def _after_success(self, t: Task) -> None:
        if t.recurring and (t.max_runs is None or t.runs < t.max_runs):
            t.run_at = self._clock() + t.every  # type: ignore[operator]
            t.state = TaskState.PENDING          # re-arm for the next cadence
        else:
            t.state = TaskState.DONE

    def tick(self) -> List[str]:
        """Run one scheduling step. Returns the ids of tasks that ran this tick."""
        ready = self.ready()
        if not ready:
            return []
        owners = [t for t in ready if t.is_owner]
        rest = [t for t in ready if not t.is_owner]
        # owner tasks always run (never throttled); fill the rest up to the cap
        selected = owners + rest[:max(0, self.max_concurrent)]
        ran: List[str] = []
        for t in selected:
            self._execute(t)
            ran.append(t.id)
        return ran

    def run_until_idle(self, max_ticks: int = 10_000) -> int:
        """Tick until nothing is ready (or ``max_ticks``). Returns ticks performed."""
        ticks = 0
        while ticks < max_ticks and self.ready():
            self.tick()
            ticks += 1
        return ticks

    def purge_terminal(self) -> int:
        """Drop finished (DONE/FAILED/CANCELLED) tasks that nothing still depends on.

        Keeps the task table bounded for a long-running daemon that submits a fresh
        one-shot task on every tick. Returns how many were removed."""
        terminal = {TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED}
        depended: set = set()
        for t in self._tasks.values():
            depended |= set(t.deps)
        drop = [tid for tid, t in self._tasks.items()
                if t.state in terminal and tid not in depended]
        for tid in drop:
            del self._tasks[tid]
        return len(drop)

    # ---- reporting ---- #
    def pending(self) -> List[Task]:
        return [t for t in self._tasks.values() if t.state is TaskState.PENDING]

    def by_state(self, state: TaskState) -> List[Task]:
        return [t for t in self._tasks.values() if t.state is state]

    def report(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for t in self._tasks.values():
            counts[t.state.value] = counts.get(t.state.value, 0) + 1
        return {"tasks": len(self._tasks), "by_state": counts,
                "ready": len(self.ready()), "max_concurrent": self.max_concurrent}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA scheduler self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    clk = lambda: now["t"]
    log: List[str] = []

    sch = Scheduler(clock=clk, max_concurrent=2)

    # priority ordering: owner work jumps ahead of normal work
    sch.submit("normal-A", lambda: log.append("A"), priority=Priority.NORMAL)
    sch.submit("normal-B", lambda: log.append("B"), priority=Priority.NORMAL)
    sch.submit("owner", lambda: log.append("OWNER"), priority=Priority.OWNER)
    ran = sch.tick()
    print(f"\nfirst tick ran      : {ran}  log={log}")
    assert log[0] == "OWNER"   # the Master's task ran first, despite the cap of 2

    sch.run_until_idle()
    print(f"after idle          : log={log}")
    assert set(log) == {"OWNER", "A", "B"}

    # dependencies: 'build' must finish before 'deploy'
    log.clear()
    b = sch.submit("build", lambda: log.append("build"))
    sch.submit("deploy", lambda: log.append("deploy"), deps=[b])
    sch.run_until_idle()
    print(f"\ndependency order    : log={log}")
    assert log == ["build", "deploy"]

    # a failed prerequisite blocks its dependents (never silently run)
    log.clear()
    def _boom():
        raise RuntimeError("build broke")
    fb = sch.submit("bad-build", _boom)
    dep = sch.submit("ship", lambda: log.append("ship"), deps=[fb])
    sch.run_until_idle()
    print(f"blocked dependent   : ship state={sch.get(dep).state.value} log={log}")
    assert sch.get(fb).state is TaskState.FAILED
    assert sch.get(dep).state is TaskState.BLOCKED
    assert log == []   # ship never ran

    # delayed task: not due until the clock advances
    log.clear()
    sch.submit("later", lambda: log.append("later"), delay=50.0)
    assert sch.run_until_idle() == 0 and log == []
    now["t"] += 60.0
    sch.run_until_idle()
    print(f"\ndelayed task        : log={log} (ran after the clock advanced)")
    assert log == ["later"]

    # recurring task: runs every 10s, capped at 3 times
    log.clear()
    sch.every(10.0, "heartbeat", lambda: log.append("beat"), max_runs=3)
    for _ in range(5):
        sch.tick()
        now["t"] += 10.0
    print(f"recurring task      : {len(log)} beats (capped at 3)")
    assert len(log) == 3

    # cancellation
    c = sch.submit("doomed", lambda: log.append("doomed"), delay=5.0)
    assert sch.cancel(c)
    now["t"] += 10.0
    sch.run_until_idle()
    assert "doomed" not in log
    print("cancellation        : cancelled task never ran ✓")

    print(f"\nreport              : {sch.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
