"""NYXARA · sim/circuit_world.py — RC / RL circuits whose laws *emerge* (🔌, stdlib-only).

The electrical counterpart of the other discovery sandboxes. Nothing stores ``τ = R·C`` or
``V = I·R``: the circuit ODEs are integrated numerically and NYXARA *measures* the outcome. She
watches a charged capacitor discharge through a resistor and reads the time it takes to fall to
``1/e`` of its voltage — discovering the decay law ``τ = R·C``; and she drives an inductive branch
to steady state and reads the current — discovering Ohm's law ``V = I·R``. From simulated dynamics,
no formula, no LLM.

Deterministic, pure stdlib, in-memory.
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["RCCircuit"]


class RCCircuit:
    """A resistor–capacitor / resistor–inductor circuit integrated from first principles."""

    def measure_time_constant(self, resistance: float, capacitance: float, *,
                              v0: float = 1.0, resolution: int = 4000) -> Optional[float]:
        """Discharge a capacitor through a resistor (``dV/dt = −V/(R·C)``) and return the measured
        time for the voltage to fall to ``1/e`` of its start — the emergent RC time constant.

        Integrated with classical **4th-order Runge–Kutta** and the ``1/e`` crossing time is
        **linearly interpolated** within the step that straddles it, so the measured τ is limited by
        physics, not by the integrator — a genuine improvement over first-order Euler."""
        R = float(resistance); C = float(capacitance)
        if R <= 0.0 or C <= 0.0:
            return None
        rc = R * C
        dt = rc / max(50, resolution)
        target = v0 / math.e

        def deriv(v: float) -> float:
            return -v / rc                              # dV/dt = −V/(R·C)

        v = float(v0)
        t = 0.0
        for _ in range(resolution * 8):
            k1 = deriv(v)
            k2 = deriv(v + 0.5 * dt * k1)
            k3 = deriv(v + 0.5 * dt * k2)
            k4 = deriv(v + dt * k3)
            v_next = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)   # RK4 step
            t_next = t + dt
            if v_next <= target <= v:                   # the 1/e crossing lies in [t, t_next]
                span = v - v_next
                frac = (v - target) / span if span > 1e-18 else 0.0
                return t + frac * dt                    # sub-step-accurate crossing time ⇒ τ
            v, t = v_next, t_next
        return None

    def measure_current(self, voltage: float, resistance: float, *,
                        inductance: float = 1.0, steps: int = 6000) -> Optional[float]:
        """Drive an R–L branch (``L·dI/dt = V − I·R``) to steady state and return the current — which
        settles to ``V/R``, letting NYXARA discover Ohm's law ``V = I·R`` from the dynamics.

        Integrated with 4th-order Runge–Kutta; the run stops early once the current has genuinely
        settled (successive steps agree to a tight tolerance), so the reported value is the true
        steady state rather than a fixed-horizon approximation."""
        R = float(resistance); V = float(voltage); L = float(inductance)
        if R <= 0.0 or L <= 0.0:
            return None
        tau = L / R
        dt = tau / 200.0

        def deriv(i: float) -> float:
            return (V - i * R) / L                       # L·dI/dt = V − I·R

        i = 0.0
        for _ in range(max(100, steps)):
            k1 = deriv(i)
            k2 = deriv(i + 0.5 * dt * k1)
            k3 = deriv(i + 0.5 * dt * k2)
            k4 = deriv(i + dt * k3)
            i_next = i + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)   # RK4 step
            if abs(i_next - i) < 1e-12:                   # settled to steady state (I → V/R)
                return i_next
            i = i_next
        return i
