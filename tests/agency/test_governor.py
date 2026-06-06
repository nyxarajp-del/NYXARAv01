"""Tests for nyxara.agency.governor."""

from __future__ import annotations

import pytest

from nyxara.agency.governor import (
    ConcurrencyLimiter,
    Deadline,
    Governor,
    GovernorDecision,
    TokenBucket,
    WindowBudget,
)
from nyxara.kernel.config import ResourceLimits
from nyxara.kernel.errors import ResourceExhausted, TimeoutErrorNX


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# -------------------- TokenBucket -------------------- #
def test_bucket_starts_full_and_drains():
    clk = FakeClock()
    tb = TokenBucket(10, 1.0, clock=clk)
    got = sum(1 for _ in range(10) if tb.try_acquire())
    assert got == 10
    assert not tb.try_acquire()


def test_bucket_refills_over_time():
    clk = FakeClock()
    tb = TokenBucket(10, 2.0, clock=clk)  # 2 tokens/sec
    for _ in range(10):
        tb.try_acquire()
    clk.advance(3.0)  # +6 tokens
    assert tb.try_acquire(6)
    assert not tb.try_acquire(1)


def test_bucket_cap_does_not_exceed_capacity():
    clk = FakeClock()
    tb = TokenBucket(5, 1.0, clock=clk)
    clk.advance(1000.0)
    assert tb.available == 5  # capped


def test_bucket_retry_after():
    clk = FakeClock()
    tb = TokenBucket(1, 1.0, clock=clk)  # 1/sec
    assert tb.try_acquire()
    assert abs(tb.retry_after() - 1.0) < 1e-9
    clk.advance(0.5)
    assert abs(tb.retry_after() - 0.5) < 1e-9


def test_bucket_retry_after_zero_when_available():
    tb = TokenBucket(5, 1.0, clock=FakeClock())
    assert tb.retry_after() == 0.0


def test_bucket_per_minute_factory():
    clk = FakeClock()
    tb = TokenBucket.per_minute(60, clock=clk)
    assert tb.capacity == 60
    assert abs(tb.refill_per_sec - 1.0) < 1e-9


def test_bucket_per_minute_custom_burst():
    tb = TokenBucket.per_minute(120, burst=10, clock=FakeClock())
    assert tb.capacity == 10
    assert abs(tb.refill_per_sec - 2.0) < 1e-9


def test_bucket_reset():
    clk = FakeClock()
    tb = TokenBucket(5, 1.0, clock=clk)
    for _ in range(5):
        tb.try_acquire()
    tb.reset()
    assert tb.available == 5


def test_bucket_rejects_bad_params():
    with pytest.raises(ValueError):
        TokenBucket(0, 1.0)
    with pytest.raises(ValueError):
        TokenBucket(1, 0)


# -------------------- WindowBudget -------------------- #
def test_budget_charges_under_ceiling():
    wb = WindowBudget(10.0, 100.0, clock=FakeClock())
    assert wb.charge(7.0)
    assert wb.remaining == 3.0
    assert wb.spent == 7.0


def test_budget_refuses_overspend():
    wb = WindowBudget(10.0, 100.0, clock=FakeClock())
    wb.charge(7.0)
    assert not wb.charge(5.0)  # would exceed
    assert wb.remaining == 3.0  # unchanged


def test_budget_can_afford():
    wb = WindowBudget(10.0, 100.0, clock=FakeClock())
    wb.charge(8.0)
    assert wb.can_afford(2.0)
    assert not wb.can_afford(3.0)


def test_budget_rolls_over_window():
    clk = FakeClock()
    wb = WindowBudget(10.0, 100.0, clock=clk)
    wb.charge(10.0)
    assert wb.remaining == 0.0
    clk.advance(100.0)
    assert wb.remaining == 10.0


def test_budget_no_rollover_before_window():
    clk = FakeClock()
    wb = WindowBudget(10.0, 100.0, clock=clk)
    wb.charge(10.0)
    clk.advance(99.0)
    assert wb.remaining == 0.0


def test_budget_negative_amount_rejected():
    wb = WindowBudget(10.0, clock=FakeClock())
    with pytest.raises(ValueError):
        wb.charge(-1.0)


def test_budget_reset():
    wb = WindowBudget(10.0, clock=FakeClock())
    wb.charge(5.0)
    wb.reset()
    assert wb.spent == 0.0


# -------------------- ConcurrencyLimiter -------------------- #
def test_concurrency_acquire_release():
    cl = ConcurrencyLimiter(2)
    assert cl.try_acquire() and cl.try_acquire()
    assert not cl.try_acquire()
    assert cl.active == 2
    cl.release()
    assert cl.available == 1


def test_concurrency_slot_context():
    cl = ConcurrencyLimiter(1)
    with cl.slot():
        assert cl.active == 1
    assert cl.active == 0


def test_concurrency_slot_raises_when_full():
    cl = ConcurrencyLimiter(1)
    with cl.slot():
        with pytest.raises(ResourceExhausted):
            with cl.slot():
                pass


def test_concurrency_slot_releases_on_exception():
    cl = ConcurrencyLimiter(1)
    try:
        with cl.slot():
            raise ValueError("boom")
    except ValueError:
        pass
    assert cl.active == 0


def test_concurrency_release_floor():
    cl = ConcurrencyLimiter(2)
    cl.release()  # nothing acquired
    assert cl.active == 0


def test_concurrency_rejects_bad_param():
    with pytest.raises(ValueError):
        ConcurrencyLimiter(0)


# -------------------- Deadline -------------------- #
def test_deadline_not_expired_initially():
    dl = Deadline(10.0, clock=FakeClock())
    assert not dl.expired
    assert dl.remaining == 10.0
    dl.check()  # no raise


def test_deadline_expires():
    clk = FakeClock()
    dl = Deadline(5.0, clock=clk)
    clk.advance(6.0)
    assert dl.expired
    assert dl.remaining == 0.0
    with pytest.raises(TimeoutErrorNX):
        dl.check()


def test_deadline_elapsed():
    clk = FakeClock()
    dl = Deadline(10.0, clock=clk)
    clk.advance(3.0)
    assert dl.elapsed == 3.0
    assert dl.remaining == 7.0


# -------------------- Governor -------------------- #
def _gov(clk, **kw):
    base = dict(max_llm_calls_per_min=2, max_tool_calls_per_min=4,
                max_web_fetches_per_min=2, daily_spend_budget=5.0,
                max_concurrent_tasks=1, max_spawned_agents=1, step_timeout_s=10.0)
    base.update(kw)
    return Governor(ResourceLimits(**base), clock=clk)


def test_governor_rate_limits_llm():
    clk = FakeClock()
    gov = _gov(clk)
    a = gov.allow("llm")
    b = gov.allow("llm")
    c = gov.allow("llm")
    assert a.allowed and b.allowed and not c.allowed
    assert c.retry_after > 0


def test_governor_unmetered_resource_allowed():
    gov = _gov(FakeClock())
    d = gov.allow("unknown-resource")
    assert d.allowed and "unmetered" in d.reason


def test_governor_allow_or_raise():
    clk = FakeClock()
    gov = _gov(clk, max_llm_calls_per_min=1)
    gov.allow_or_raise("llm")
    with pytest.raises(ResourceExhausted):
        gov.allow_or_raise("llm")


def test_governor_charge_budget():
    gov = _gov(FakeClock())
    assert gov.charge(4.0).allowed
    assert not gov.charge(2.0).allowed
    assert gov.budget_remaining == 1.0


def test_governor_charge_or_raise():
    gov = _gov(FakeClock(), daily_spend_budget=1.0)
    gov.charge_or_raise(1.0)
    with pytest.raises(ResourceExhausted):
        gov.charge_or_raise(0.5)


def test_governor_slot():
    gov = _gov(FakeClock())
    with gov.slot():
        assert gov.active_tasks == 1
    assert gov.active_tasks == 0


def test_governor_slot_limit():
    gov = _gov(FakeClock())
    with gov.slot():
        with pytest.raises(ResourceExhausted):
            with gov.slot():
                pass


def test_governor_agent_slot():
    gov = _gov(FakeClock(), max_spawned_agents=1)
    with gov.agent_slot():
        pass  # no raise


def test_governor_agent_disabled():
    gov = _gov(FakeClock(), max_spawned_agents=0)
    with pytest.raises(ResourceExhausted):
        with gov.agent_slot():
            pass


def test_governor_deadline_uses_step_timeout():
    clk = FakeClock()
    gov = _gov(clk, step_timeout_s=5.0)
    dl = gov.deadline()
    assert dl.limit_s == 5.0
    clk.advance(6.0)
    assert dl.expired


def test_governor_add_bucket():
    clk = FakeClock()
    gov = _gov(clk)
    gov.add_bucket("vision", 1)
    assert gov.allow("vision").allowed
    assert not gov.allow("vision").allowed


def test_governor_status_and_reset():
    clk = FakeClock()
    gov = _gov(clk)
    gov.allow("llm")
    gov.allow("llm")
    gov.allow("llm")  # denied
    gov.charge(1.0)
    s = gov.status()
    assert s["denied"] == 1
    assert s["spend"]["spent"] == 1.0
    gov.reset()
    assert gov.status()["denied"] == 0
    assert gov.spent == 0.0


def test_governor_decision_to_dict():
    gov = _gov(FakeClock())
    d = gov.allow("llm")
    assert isinstance(d, GovernorDecision)
    assert "resource" in d.to_dict()
