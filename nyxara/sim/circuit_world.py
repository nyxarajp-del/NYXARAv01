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
        time for the voltage to fall to ``1/e`` of its start — the emergent RC time constant."""
        R = float(resistance); C = float(capacitance)
        if R <= 0.0 or C <= 0.0:
            return None
        rc = R * C
        dt = rc / max(50, resolution)
        v = float(v0)
        target = v0 / math.e
        t = 0.0
        for _ in range(resolution * 8):
            v += (-v / rc) * dt                        # explicit Euler on the discharge ODE
            t += dt
            if v <= target:
                return t                               # first crossing of 1/e ⇒ measured τ
        return None

    def measure_current(self, voltage: float, resistance: float, *,
                        inductance: float = 1.0, steps: int = 6000) -> Optional[float]:
        """Drive an R–L branch (``L·dI/dt = V − I·R``) to steady state and return the current — which
        settles to ``V/R``, letting NYXARA discover Ohm's law ``V = I·R`` from the dynamics."""
        R = float(resistance); V = float(voltage); L = float(inductance)
        if R <= 0.0 or L <= 0.0:
            return None
        tau = L / R
        dt = tau / 200.0
        i = 0.0
        for _ in range(max(100, steps)):
            i += ((V - i * R) / L) * dt                # explicit Euler on the R–L ODE
        return i
