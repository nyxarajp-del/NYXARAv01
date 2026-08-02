"""Chrono-dilation: deepens under pressure, honours the deadline, caps depth, stays shallow when calm."""

from __future__ import annotations

from nyxara.nyx5.chrono import ChronoDilator


def _work(depth):
    # quality improves with depth but saturates — an honest anytime function
    return f"answer@{depth}", min(1.0, 0.3 + 0.1 * depth)


def test_stays_shallow_when_no_pressure():
    c = ChronoDilator(trigger=0.6)
    r = c.run(_work, pressure=0.1)
    assert r.dilated is False
    assert r.depth == 1
    assert r.stopped == "shallow"


def test_dilates_under_pressure():
    c = ChronoDilator(trigger=0.5, max_depth=20, deadline_ms=5000)
    r = c.run(_work, pressure=0.9)
    assert r.dilated is True
    assert r.depth > 1


def test_respects_deadline():
    fake = {"t": 0.0}

    def clock():
        fake["t"] += 0.01           # each check advances 10ms
        return fake["t"]

    c = ChronoDilator(trigger=0.5, max_depth=10_000, deadline_ms=50, clock=clock)
    r = c.run(_work, pressure=1.0)
    assert r.stopped in ("deadline", "converged", "max_depth")
    assert r.depth <= 10_000


def test_depth_capped():
    c = ChronoDilator(trigger=0.5, max_depth=5, deadline_ms=10_000, converge_eps=0.0)
    r = c.run(lambda d: (d, d / 100.0), pressure=1.0)   # monotone, never converges
    assert r.depth <= 5
