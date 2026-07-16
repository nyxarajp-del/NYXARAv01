"""NYXARA · sim/optics_world.py — a vibrating string whose wave speed *emerges* (🌊, stdlib-only).

Another honest discovery sandbox: nothing stores ``v = √(T/μ)``. A string is modelled as a chain of
masses and springs and the **wave equation is integrated directly** (a finite-difference stencil of
``∂²y/∂t² = (T/μ)·∂²y/∂x²``). NYXARA plucks the string, watches a pulse travel, and *measures* its
speed from how far the crest moves per unit time. Sweeping tension and linear density, she discovers
that ``v² = T/μ`` — the wave law — from motion she simulated, no formula and no LLM.

Deterministic, pure stdlib, in-memory. The speed she reads out is a property of the propagation
dynamics, not of the initial pluck, so it genuinely reflects the physics of the medium.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

__all__ = ["WaveString"]


class WaveString:
    """A 1-D string integrating the wave equation ``ÿ = (T/μ)·y''`` with fixed ends."""

    def __init__(self, *, length: float = 1.0, n_points: int = 401) -> None:
        self.length = float(length)
        self.n = max(9, int(n_points))
        self.dx = self.length / (self.n - 1)

    def measure_speed(self, tension: float, mu: float, *,
                      cfl: float = 0.4) -> Optional[float]:
        """Pluck a stationary crest at the centre and measure how fast the travelling pulse moves.

        The initial state is a still Gaussian bump (zero velocity), which splits into two equal
        pulses running in opposite directions. We track the crest of the *right-going* pulse and fit
        its position against time — the slope is the emergent wave speed. ``None`` if it cannot be
        measured cleanly (e.g. the pulse reaches the boundary too fast)."""
        import math
        c_expect = math.sqrt(max(1e-12, tension / max(1e-12, mu)))
        dt = cfl * self.dx / c_expect                 # CFL-stable step (uses c only for stability)
        coef = (tension / mu) * (dt * dt) / (self.dx * self.dx)
        n = self.n
        # still Gaussian crest at the centre
        cx = self.length * 0.5
        width = self.length * 0.04
        y_prev = [math.exp(-((i * self.dx - cx) / width) ** 2) for i in range(n)]
        y = list(y_prev)                              # zero initial velocity ⇒ y == y_prev
        y[0] = y[-1] = 0.0
        y_prev[0] = y_prev[-1] = 0.0
        centre = n // 2
        samples: List[Tuple[float, float]] = []
        t = 0.0
        max_steps = int(0.42 * self.length / (c_expect * dt))   # stop before the crest hits the end
        for step in range(max(4, max_steps)):
            y_next = [0.0] * n
            for i in range(1, n - 1):
                y_next[i] = 2.0 * y[i] - y_prev[i] + coef * (y[i + 1] - 2.0 * y[i] + y[i - 1])
            y_next[0] = y_next[-1] = 0.0
            y_prev, y = y, y_next
            t += dt
            # locate the crest of the right-going pulse (search the right half of the string)
            best_i, best_v = centre, -1.0
            for i in range(centre + 1, n - 1):
                if y[i] > best_v:
                    best_v, best_i = y[i], i
            if best_v > 0.15:                          # only once the pulse has clearly separated
                samples.append((t, best_i * self.dx))
        if len(samples) < 6:
            return None
        # linear fit x = v·t + b over the clean middle portion → slope v
        mid = samples[len(samples) // 4: -len(samples) // 4] or samples
        m = len(mid)
        st = sum(p[0] for p in mid); sx = sum(p[1] for p in mid)
        stt = sum(p[0] * p[0] for p in mid); stx = sum(p[0] * p[1] for p in mid)
        denom = m * stt - st * st
        if abs(denom) < 1e-12:
            return None
        return (m * stx - st * sx) / denom
