"""Tests for nyxara.agency.scheduler."""

from __future__ import annotations

import pytest

from nyxara.agency.governor import Governor
from nyxara.agency.scheduler import Priority, Scheduler, Task, TaskState
from nyxara.kernel.config import ResourceLimits


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _sched(**kw):
    kw.setdefault("clock", FakeClock())
    return Scheduler(**kw)


# -------------------- submission -------------------- #
def test_submit_returns_id_and_pending():
    s = _sched()
    tid = s.submit("x", lambda: 1)
    assert s.get(tid).state is TaskState.PENDING
    assert s.get(tid).name == "x"


def test_submit_unknown_dep_rejected():
    s = _sched()
    with pytest.raises(ValueError):
        s.submit("x", lambda: 1, deps=["ghost"])


def test_submit_bad_interval_rejected():
    s = _sched()
    with pytest.raises(ValueError):
        s.submit("x", lambda: 1, every=0)


def test_every_helper_recurring():
    s = _sched()
    tid = s.every(10.0, "beat", lambda: 1)
    assert s.get(tid).recurring


# -------------------- basic execution -------------------- #
def test_tick_runs_ready_task():
    s = _sched()
    out = []
    s.submit("x", lambda: out.append("ran"))
    ran = s.tick()
    assert len(ran) == 1
    assert out == ["ran"]


def test_task_result_captured():
    s = _sched()
    tid = s.submit("x", lambda: 42)
    s.tick()
    t = s.get(tid)
    assert t.state is TaskState.DONE and t.result == 42


def test_task_failure_captured_as_data():
    s = _sched()

    def boom():
        raise RuntimeError("nope")

    tid = s.submit("x", boom)
    s.tick()
    t = s.get(tid)
    assert t.state is TaskState.FAILED and "RuntimeError" in t.error


def test_run_until_idle_drains():
    s = _sched()
    out = []
    for i in range(5):
        s.submit(f"t{i}", lambda i=i: out.append(i))
    s.run_until_idle()
    assert sorted(out) == [0, 1, 2, 3, 4]
    assert s.ready() == []


# -------------------- priority ordering -------------------- #
def test_owner_runs_first():
    s = _sched(max_concurrent=1)
    log = []
    s.submit("normal", lambda: log.append("normal"), priority=Priority.NORMAL)
    s.submit("owner", lambda: log.append("owner"), priority=Priority.OWNER)
    s.tick()
    assert log[0] == "owner"


def test_owner_bypasses_concurrency_cap():
    s = _sched(max_concurrent=1)
    log = []
    s.submit("o1", lambda: log.append("o1"), priority=Priority.OWNER)
    s.submit("o2", lambda: log.append("o2"), priority=Priority.OWNER)
    s.submit("n1", lambda: log.append("n1"), priority=Priority.NORMAL)
    ran = s.tick()
    # both owner tasks run + 1 normal (cap), so 3 total despite cap of 1
    assert len(ran) == 3
    assert log[0] in ("o1", "o2") and log[1] in ("o1", "o2")


def test_concurrency_cap_limits_non_owner():
    s = _sched(max_concurrent=2)
    for i in range(5):
        s.submit(f"t{i}", lambda: 1, priority=Priority.NORMAL)
    ran = s.tick()
    assert len(ran) == 2  # only 2 non-owner tasks per tick


def test_priority_ordering_within_tick():
    s = _sched(max_concurrent=10)
    log = []
    s.submit("low", lambda: log.append("low"), priority=Priority.LOW)
    s.submit("high", lambda: log.append("high"), priority=Priority.HIGH)
    s.submit("crit", lambda: log.append("crit"), priority=Priority.CRITICAL)
    s.tick()
    assert log == ["crit", "high", "low"]


# -------------------- dependencies -------------------- #
def test_dependency_runs_after_prereq():
    s = _sched()
    log = []
    a = s.submit("a", lambda: log.append("a"))
    s.submit("b", lambda: log.append("b"), deps=[a])
    s.run_until_idle()
    assert log == ["a", "b"]


def test_dependent_not_ready_until_prereq_done():
    s = _sched()
    a = s.submit("a", lambda: 1)
    b = s.submit("b", lambda: 2, deps=[a])
    # only 'a' is ready initially
    assert [t.id for t in s.ready()] == [a]
    s.tick()
    assert [t.id for t in s.ready()] == [b]


def test_failed_prereq_blocks_dependent():
    s = _sched()

    def boom():
        raise RuntimeError("x")

    a = s.submit("a", boom)
    b = s.submit("b", lambda: 1, deps=[a])
    s.run_until_idle()
    assert s.get(a).state is TaskState.FAILED
    assert s.get(b).state is TaskState.BLOCKED


def test_cancelled_prereq_blocks_dependent():
    s = _sched()
    a = s.submit("a", lambda: 1, delay=100.0)
    b = s.submit("b", lambda: 1, deps=[a])
    s.cancel(a)
    s.run_until_idle()
    assert s.get(b).state is TaskState.BLOCKED


def test_chain_of_dependencies():
    s = _sched()
    log = []
    a = s.submit("a", lambda: log.append("a"))
    b = s.submit("b", lambda: log.append("b"), deps=[a])
    s.submit("c", lambda: log.append("c"), deps=[b])
    s.run_until_idle()
    assert log == ["a", "b", "c"]


# -------------------- delays & recurrence -------------------- #
def test_delayed_task_not_due_early():
    clk = FakeClock()
    s = _sched(clock=clk)
    log = []
    s.submit("later", lambda: log.append("later"), delay=50.0)
    s.run_until_idle()
    assert log == []
    clk.advance(60.0)
    s.run_until_idle()
    assert log == ["later"]


def test_at_absolute_time():
    clk = FakeClock(t=1000.0)
    s = _sched(clock=clk)
    log = []
    s.submit("x", lambda: log.append("x"), at=1100.0)
    s.run_until_idle()
    assert log == []
    clk.advance(150.0)
    s.run_until_idle()
    assert log == ["x"]


def test_recurring_capped():
    clk = FakeClock()
    s = _sched(clock=clk)
    log = []
    s.every(10.0, "beat", lambda: log.append("beat"), max_runs=3)
    for _ in range(6):
        s.tick()
        clk.advance(10.0)
    assert len(log) == 3


def test_recurring_reschedules():
    clk = FakeClock()
    s = _sched(clock=clk)
    log = []
    tid = s.every(10.0, "beat", lambda: log.append("beat"), max_runs=2)
    s.tick()
    assert len(log) == 1
    assert s.get(tid).state is TaskState.PENDING  # re-armed
    clk.advance(10.0)
    s.tick()
    assert len(log) == 2
    assert s.get(tid).state is TaskState.DONE


def test_recurring_not_due_before_interval():
    clk = FakeClock()
    s = _sched(clock=clk)
    log = []
    s.every(10.0, "beat", lambda: log.append("beat"))
    s.tick()
    s.tick()  # same time, not due again
    assert len(log) == 1


# -------------------- cancellation -------------------- #
def test_cancel_pending():
    s = _sched()
    tid = s.submit("x", lambda: 1, delay=10.0)
    assert s.cancel(tid)
    assert s.get(tid).state is TaskState.CANCELLED


def test_cancel_done_returns_false():
    s = _sched()
    tid = s.submit("x", lambda: 1)
    s.tick()
    assert not s.cancel(tid)


def test_cancel_unknown_returns_false():
    s = _sched()
    assert not s.cancel("nope")


def test_cancelled_task_never_runs():
    clk = FakeClock()
    s = _sched(clock=clk)
    log = []
    tid = s.submit("x", lambda: log.append("x"), delay=5.0)
    s.cancel(tid)
    clk.advance(10.0)
    s.run_until_idle()
    assert log == []


# -------------------- governor integration -------------------- #
def test_governor_supplies_concurrency_cap():
    gov = Governor(ResourceLimits(max_concurrent_tasks=3))
    s = Scheduler(clock=FakeClock(), governor=gov)
    assert s.max_concurrent == 3


def test_governor_slot_used_for_non_owner():
    gov = Governor(ResourceLimits(max_concurrent_tasks=2))
    s = Scheduler(clock=FakeClock(), governor=gov, max_concurrent=5)
    out = []
    for i in range(3):
        s.submit(f"t{i}", lambda i=i: out.append(i), priority=Priority.NORMAL)
    # governor slot is acquired/released synchronously per task; all still run
    s.run_until_idle()
    assert sorted(out) == [0, 1, 2]


# -------------------- reporting -------------------- #
def test_report_counts_states():
    s = _sched()
    s.submit("ok", lambda: 1)
    s.submit("bad", lambda: (_ for _ in ()).throw(ValueError("x")))
    s.run_until_idle()
    r = s.report()
    assert r["by_state"]["done"] == 1 and r["by_state"]["failed"] == 1


def test_by_state_filter():
    s = _sched()
    s.submit("x", lambda: 1)
    s.run_until_idle()
    assert len(s.by_state(TaskState.DONE)) == 1


def test_pending_filter():
    s = _sched()
    s.submit("x", lambda: 1, delay=100.0)
    assert len(s.pending()) == 1


def test_task_to_dict():
    s = _sched()
    tid = s.submit("x", lambda: 1, priority=Priority.HIGH)
    d = s.get(tid).to_dict()
    assert d["name"] == "x" and d["priority"] == "high" and d["state"] == "pending"


def test_priority_label():
    assert Priority.OWNER.label == "owner"
