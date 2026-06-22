"""NYXARA · temporal/clock.py — an injectable clock for the fractal time layers.

The fractal hierarchy lives at three time scales at once (milliseconds, seconds, days).
For that to be *real* in production and *deterministic* in tests, every layer reads time
through a small clock abstraction instead of calling :func:`time.time` directly:

* :class:`RealClock` — wall-clock time (``time.time``) and a monotonic reading for
  measuring durations safely across NTP steps. This is what runs in production.
* :class:`VirtualClock` — a manually-advanced clock. Tests (and the deterministic
  ``run_for`` driver) step it forward by an exact number of seconds so a "day" can pass
  in a microsecond and nothing ever sleeps.

Both satisfy the same :class:`TemporalClock` protocol (``now()`` / ``monotonic()`` /
``advance()``), so a layer takes a clock and never knows or cares which one it holds.

Pure standard library.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

__all__ = [
    "MILLISECOND",
    "SECOND",
    "MINUTE",
    "HOUR",
    "DAY",
    "WEEK",
    "MONTH",
    "TemporalClock",
    "RealClock",
    "VirtualClock",
]

# --- the scales the fractal mind thinks at (seconds) --- #
MILLISECOND: float = 0.001
SECOND: float = 1.0
MINUTE: float = 60.0
HOUR: float = 3600.0
DAY: float = 86400.0
WEEK: float = 7.0 * DAY
MONTH: float = 30.0 * DAY


@runtime_checkable
class TemporalClock(Protocol):
    """The minimal time source every temporal layer reads from."""

    def now(self) -> float:
        """Wall-clock seconds since the epoch (for timestamps that match memory)."""
        ...

    def monotonic(self) -> float:
        """A monotonic reading (seconds) for measuring durations safely."""
        ...

    def advance(self, seconds: float) -> float:
        """Move a virtual clock forward (no-op on a real clock). Returns the new ``now()``."""
        ...


class RealClock:
    """Wall-clock time — production. ``advance`` is a no-op (real time moves itself)."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def advance(self, seconds: float) -> float:  # noqa: ARG002 — real time advances itself
        return time.time()


class VirtualClock:
    """A manually-advanced clock for deterministic runs and tests.

    ``now()`` and ``monotonic()`` both read the same virtual instant, so a whole day of
    simulated time passes the moment :meth:`advance` is called — no sleeping, fully
    reproducible.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def monotonic(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += max(0.0, float(seconds))
        return self._t
