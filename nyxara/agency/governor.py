"""NYXARA · agency/governor.py — the resource & rate governor (⏱, fail-closed).

Where :mod:`agency.permissions` decides *whether* an action is allowed, the **governor**
decides *how much* and *how fast* NYXARA may act. It is the throttle and the budget on
autonomous activity: rate limits on external calls (LLM / tools / web), a ceiling on
in-flight concurrency, a daily spend budget, and per-step wall-clock deadlines.

The primitives are deterministic and clock-injectable (so they are exactly testable):

* :class:`TokenBucket` — smooth rate limiting with burst capacity and a precise
  ``retry_after``.
* :class:`WindowBudget` — a charge ledger with a ceiling that rolls over each window
  (e.g. a daily autonomous-spend cap). Fail-closed: a charge that would exceed the cap
  is refused, never silently overspent.
* :class:`ConcurrencyLimiter` — a counting slot guard with a context manager.
* :class:`Deadline` — a per-step wall-clock guard that reports remaining time and
  raises :class:`~nyxara.kernel.errors.TimeoutErrorNX` when exceeded.

:class:`Governor` composes these from :class:`~nyxara.kernel.config.ResourceLimits`,
exposing ``allow(resource)`` (rate), ``charge(amount)`` (spend), ``slot()`` (concurrency)
and ``deadline()`` (time), each returning a :class:`GovernorDecision` or raising on the
``*_or_raise`` variants. A governor that is throttling does not endanger the Master — it
protects him from a runaway autonomous loop.

Depends on :mod:`kernel.config` and :mod:`kernel.errors`. Pure standard library, no
real time required for tests (inject a clock).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

from nyxara.kernel.config import ResourceLimits, get_settings
from nyxara.kernel.errors import ResourceExhausted, TimeoutErrorNX

__all__ = [
    "TokenBucket",
    "WindowBudget",
    "ConcurrencyLimiter",
    "Deadline",
    "GovernorDecision",
    "Governor",
]


Clock = Callable[[], float]


# --------------------------------------------------------------------------- #
# Token bucket — smooth rate limiting
# --------------------------------------------------------------------------- #
class TokenBucket:
    """Classic token bucket: ``capacity`` tokens, refilled at ``refill_per_sec``."""

    def __init__(self, capacity: float, refill_per_sec: float, *,
                 clock: Optional[Clock] = None, start_full: bool = True) -> None:
        if capacity <= 0 or refill_per_sec <= 0:
            raise ValueError("capacity and refill_per_sec must be positive")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._clock = clock or time.monotonic
        self._tokens = float(capacity) if start_full else 0.0
        self._last = self._clock()
        self._lock = threading.Lock()

    @classmethod
    def per_minute(cls, per_min: float, *, burst: Optional[float] = None,
                   clock: Optional[Clock] = None) -> "TokenBucket":
        """A bucket sized to ``per_min`` events/minute (burst defaults to one minute)."""
        cap = float(burst if burst is not None else per_min)
        return cls(cap, per_min / 60.0, clock=clock)

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        self._last = now

    def try_acquire(self, n: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def retry_after(self, n: float = 1.0) -> float:
        """Seconds until ``n`` tokens would be available (0 if available now)."""
        with self._lock:
            self._refill()
            if self._tokens >= n:
                return 0.0
            return (n - self._tokens) / self.refill_per_sec

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def reset(self) -> None:
        with self._lock:
            self._tokens = self.capacity
            self._last = self._clock()


# --------------------------------------------------------------------------- #
# Window budget — rolling spend ceiling
# --------------------------------------------------------------------------- #
class WindowBudget:
    """A spend ledger with a ceiling that resets each ``window_s`` window."""

    def __init__(self, limit: float, window_s: float = 86_400.0, *,
                 clock: Optional[Clock] = None) -> None:
        self.limit = float(limit)
        self.window_s = float(window_s)
        self._clock = clock or time.monotonic
        self._spent = 0.0
        self._window_start = self._clock()
        self._lock = threading.Lock()

    def _roll(self) -> None:
        now = self._clock()
        if now - self._window_start >= self.window_s:
            self._spent = 0.0
            self._window_start = now

    def can_afford(self, amount: float) -> bool:
        with self._lock:
            self._roll()
            return self._spent + amount <= self.limit

    def charge(self, amount: float) -> bool:
        """Charge ``amount`` if it fits under the ceiling; else refuse (fail-closed)."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        with self._lock:
            self._roll()
            if self._spent + amount <= self.limit:
                self._spent += amount
                return True
            return False

    @property
    def spent(self) -> float:
        with self._lock:
            self._roll()
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            self._roll()
            return max(0.0, self.limit - self._spent)

    def reset(self) -> None:
        with self._lock:
            self._spent = 0.0
            self._window_start = self._clock()


# --------------------------------------------------------------------------- #
# Concurrency limiter
# --------------------------------------------------------------------------- #
class ConcurrencyLimiter:
    """A counting slot guard: at most ``max_slots`` activities in flight."""

    def __init__(self, max_slots: int) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be >= 1")
        self.max_slots = int(max_slots)
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active < self.max_slots:
                self._active += 1
                return True
            return False

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def available(self) -> int:
        with self._lock:
            return self.max_slots - self._active

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self.try_acquire():
            raise ResourceExhausted("concurrency limit reached",
                                    context={"max_slots": self.max_slots})
        try:
            yield
        finally:
            self.release()


# --------------------------------------------------------------------------- #
# Deadline — per-step wall-clock guard
# --------------------------------------------------------------------------- #
class Deadline:
    """A wall-clock budget for a single step; reports remaining time and can raise."""

    def __init__(self, limit_s: float, *, clock: Optional[Clock] = None,
                 label: str = "step") -> None:
        self.limit_s = float(limit_s)
        self.label = label
        self._clock = clock or time.monotonic
        self._start = self._clock()

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_s - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.elapsed >= self.limit_s

    def check(self) -> None:
        """Raise :class:`TimeoutErrorNX` if the deadline has passed (call at checkpoints)."""
        if self.expired:
            raise TimeoutErrorNX(
                f"{self.label} exceeded its {self.limit_s:g}s deadline",
                context={"elapsed": round(self.elapsed, 3), "limit_s": self.limit_s})


# --------------------------------------------------------------------------- #
# Governor decision
# --------------------------------------------------------------------------- #
@dataclass
class GovernorDecision:
    allowed: bool
    resource: str
    retry_after: float = 0.0
    remaining: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "resource": self.resource,
                "retry_after": round(self.retry_after, 4),
                "remaining": self.remaining, "reason": self.reason}


# --------------------------------------------------------------------------- #
# Governor — composes the primitives from ResourceLimits
# --------------------------------------------------------------------------- #
class Governor:
    """Throttle and budget on autonomous activity, built from ResourceLimits."""

    def __init__(self, limits: Optional[ResourceLimits] = None, *,
                 clock: Optional[Clock] = None) -> None:
        self.limits = limits or get_settings().resources
        self._clock = clock or time.monotonic
        L = self.limits
        self._buckets: Dict[str, TokenBucket] = {
            "llm": TokenBucket.per_minute(L.max_llm_calls_per_min, clock=clock),
            "tool": TokenBucket.per_minute(L.max_tool_calls_per_min, clock=clock),
            "web": TokenBucket.per_minute(L.max_web_fetches_per_min, clock=clock),
        }
        self._budget = WindowBudget(L.daily_spend_budget, 86_400.0, clock=clock)
        self._concurrency = ConcurrencyLimiter(L.max_concurrent_tasks)
        self._agents = ConcurrencyLimiter(max(1, L.max_spawned_agents)) \
            if L.max_spawned_agents > 0 else None
        self._denied = 0

    # ---- rate ---- #
    def add_bucket(self, resource: str, per_minute: float, *,
                   burst: Optional[float] = None) -> TokenBucket:
        b = TokenBucket.per_minute(per_minute, burst=burst, clock=self._clock)
        self._buckets[resource] = b
        return b

    def allow(self, resource: str, n: float = 1.0) -> GovernorDecision:
        """Try to spend ``n`` from a rate-limited resource bucket."""
        b = self._buckets.get(resource)
        if b is None:
            return GovernorDecision(True, resource, reason="unmetered resource")
        if b.try_acquire(n):
            return GovernorDecision(True, resource, remaining=b.available)
        self._denied += 1
        return GovernorDecision(False, resource, retry_after=b.retry_after(n),
                                remaining=b.available, reason="rate limit reached")

    def allow_or_raise(self, resource: str, n: float = 1.0) -> GovernorDecision:
        d = self.allow(resource, n)
        if not d.allowed:
            raise ResourceExhausted(f"rate limit on {resource!r}",
                                    context=d.to_dict(), retryable=True)
        return d

    # ---- spend ---- #
    def charge(self, amount: float) -> GovernorDecision:
        if self._budget.charge(amount):
            return GovernorDecision(True, "spend", remaining=self._budget.remaining)
        self._denied += 1
        return GovernorDecision(False, "spend", remaining=self._budget.remaining,
                                reason="daily spend budget exhausted")

    def charge_or_raise(self, amount: float) -> GovernorDecision:
        d = self.charge(amount)
        if not d.allowed:
            raise ResourceExhausted("daily spend budget exhausted", context=d.to_dict())
        return d

    @property
    def spent(self) -> float:
        return self._budget.spent

    @property
    def budget_remaining(self) -> float:
        return self._budget.remaining

    # ---- concurrency ---- #
    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._concurrency.slot():
            yield

    @contextmanager
    def agent_slot(self) -> Iterator[None]:
        if self._agents is None:
            raise ResourceExhausted("agent spawning is disabled (max_spawned_agents=0)")
        with self._agents.slot():
            yield

    @property
    def active_tasks(self) -> int:
        return self._concurrency.active

    # ---- time ---- #
    def deadline(self, limit_s: Optional[float] = None, *, label: str = "step") -> Deadline:
        return Deadline(limit_s if limit_s is not None else self.limits.step_timeout_s,
                        clock=self._clock, label=label)

    # ---- reporting ---- #
    def status(self) -> Dict[str, Any]:
        return {
            "buckets": {r: round(b.available, 2) for r, b in self._buckets.items()},
            "spend": {"spent": round(self._budget.spent, 3),
                      "remaining": round(self._budget.remaining, 3),
                      "limit": self._budget.limit},
            "concurrency": {"active": self._concurrency.active,
                            "max": self._concurrency.max_slots},
            "denied": self._denied,
        }

    def reset(self) -> None:
        for b in self._buckets.values():
            b.reset()
        self._budget.reset()
        self._denied = 0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA resource-governor self-test")
    print("=" * 70)

    # a controllable fake clock for deterministic timing
    now = {"t": 1000.0}
    clk = lambda: now["t"]

    # ---- token bucket ---- #
    tb = TokenBucket.per_minute(60, clock=clk)   # 60/min => 1/sec, burst 60
    got = sum(1 for _ in range(60) if tb.try_acquire())
    print(f"\nbucket burst        : drained {got}/60 tokens")
    assert got == 60
    assert not tb.try_acquire()                  # empty now
    ra = tb.retry_after()
    print(f"retry_after empty   : {ra:.2f}s")
    assert abs(ra - 1.0) < 1e-6                   # 1 token/sec
    now["t"] += 5.0                               # 5 seconds pass -> 5 tokens
    assert tb.try_acquire(5)
    print("refill after 5s     : 5 tokens acquired ✓")

    # ---- window budget ---- #
    wb = WindowBudget(10.0, window_s=100.0, clock=clk)
    assert wb.charge(7.0) and wb.remaining == 3.0
    assert not wb.charge(5.0)                      # would exceed -> refused
    assert wb.charge(3.0) and wb.remaining == 0.0
    print(f"\nbudget              : spent={wb.spent} (overspend refused) ✓")
    now["t"] += 100.0                             # window rolls over
    assert wb.remaining == 10.0
    print("budget rollover     : reset after window ✓")

    # ---- concurrency ---- #
    cl = ConcurrencyLimiter(2)
    with cl.slot():
        with cl.slot():
            assert cl.active == 2
            try:
                with cl.slot():
                    raise SystemExit("ERROR: should have blocked")
            except ResourceExhausted:
                print("\nconcurrency         : 3rd slot refused at limit 2 ✓")
    assert cl.active == 0

    # ---- deadline ---- #
    dl = Deadline(10.0, clock=clk, label="think")
    dl.check()                                    # fine
    now["t"] += 11.0
    assert dl.expired
    try:
        dl.check()
        raise SystemExit("ERROR: deadline should have fired")
    except TimeoutErrorNX:
        print("deadline            : fires past limit ✓")

    # ---- governor composition ---- #
    gov = Governor(ResourceLimits(max_llm_calls_per_min=2, daily_spend_budget=5.0,
                                  max_concurrent_tasks=1), clock=clk)
    a = gov.allow("llm"); b = gov.allow("llm"); c = gov.allow("llm")
    print(f"\ngovernor llm        : {a.allowed}, {b.allowed}, then {c.allowed} "
          f"(retry_after={c.retry_after:.1f}s)")
    assert a.allowed and b.allowed and not c.allowed
    assert gov.charge(4.0).allowed and not gov.charge(2.0).allowed
    with gov.slot():
        assert gov.active_tasks == 1
    print(f"governor status     : {gov.status()}")

    print("\nALL SELF-TESTS PASSED ✓")
