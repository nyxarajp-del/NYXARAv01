"""Tests for nyxara.temporal.clock."""

from __future__ import annotations

from nyxara.temporal.clock import (
    DAY,
    HOUR,
    MILLISECOND,
    MINUTE,
    SECOND,
    WEEK,
    RealClock,
    TemporalClock,
    VirtualClock,
)


# -------------------- scale constants -------------------- #
def test_scales_are_ordered():
    assert MILLISECOND < SECOND < MINUTE < HOUR < DAY < WEEK
    assert MINUTE == 60 * SECOND
    assert HOUR == 60 * MINUTE
    assert DAY == 24 * HOUR


# -------------------- virtual clock -------------------- #
def test_virtual_clock_advances():
    c = VirtualClock(start=100.0)
    assert c.now() == 100.0
    assert c.monotonic() == 100.0
    assert c.advance(5.0) == 105.0
    assert c.now() == 105.0 and c.monotonic() == 105.0


def test_virtual_clock_never_goes_backward():
    c = VirtualClock()
    c.advance(-10.0)  # negative is clamped to no-op
    assert c.now() == 0.0


def test_virtual_clock_satisfies_protocol():
    assert isinstance(VirtualClock(), TemporalClock)
    assert isinstance(RealClock(), TemporalClock)


# -------------------- real clock -------------------- #
def test_real_clock_monotonic_non_decreasing():
    c = RealClock()
    a = c.monotonic()
    b = c.monotonic()
    assert b >= a
    # advance is a no-op on a real clock (time moves itself), still returns a wall time
    assert c.advance(123.0) > 0
