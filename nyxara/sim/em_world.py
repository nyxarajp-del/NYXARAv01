"""NYXARA · sim/em_world.py — a real electrostatics micro-world (⚡, stdlib-only).

A sandbox NYXARA runs her *own* experiments in to discover the law of electric force — the same
way :mod:`nyxara.sim.physics_world` lets her discover mechanics. A fixed source charge sits at the
origin; a test charge of known mass is released from rest nearby and the world integrates its
motion under the **Coulomb force alone**. She never sees the formula: she measures how far the test
charge moves in a short time and infers the force from kinematics (``F = m·a``, ``d = ½·a·t²``),
then discovers ``F = k·q₁·q₂ / r²`` from the table ``(q₁, q₂, r) → F`` she collected.

Honest, like the rest of the sandbox: the world *implements* Coulomb's law in its integrator (as
the real universe does), and NYXARA rediscovers it from observed motion — no equation is handed to
her, no LLM anywhere near the loop. Pure stdlib, in-memory, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["Charge", "ElectrostaticWorld"]


@dataclass
class Charge:
    """A point charge: value ``q``, position, velocity, and (for a mobile test charge) mass."""
    q: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0


class ElectrostaticWorld:
    """A deterministic point-charge world governed by Coulomb's law ``F = k·q₁·q₂ / r²``.

    ``k`` is the (arbitrary, in-sandbox) Coulomb constant; only its presence matters, not its SI
    value. The world integrates the test charge under the field of the fixed source charges."""

    def __init__(self, *, k: float = 1.0, softening: float = 0.0) -> None:
        self.k = float(k)
        self.softening = float(softening)   # optional r-softening to avoid singularities (0 = pure)
        self.sources: List[Charge] = []

    def add_source(self, q: float, x: float, y: float) -> None:
        self.sources.append(Charge(q=q, x=x, y=y))

    def force_on(self, test: Charge) -> Tuple[float, float]:
        """The net Coulomb force vector on a test charge from all fixed sources."""
        fx = fy = 0.0
        for s in self.sources:
            dx, dy = test.x - s.x, test.y - s.y
            r2 = dx * dx + dy * dy + self.softening
            if r2 < 1e-18:
                continue
            r = math.sqrt(r2)
            f = self.k * s.q * test.q / r2          # Coulomb magnitude (signed by the charges)
            fx += f * (dx / r)
            fy += f * (dy / r)
        return fx, fy

    def measure_force(self, q_source: float, q_test: float, r: float, *,
                      mass: float = 1.0, dt: float = 1e-4, steps: int = 5) -> float:
        """Release a test charge from rest at distance ``r`` and infer the force it feels from the
        motion it undergoes — a genuine kinematic *measurement*, not a formula lookup.

        The displacement over a short time gives the acceleration (``d = ½·a·t²``); ``F = m·a``.
        The interval is kept tiny so ``r`` barely changes during the measurement. The motion is
        integrated with **velocity Verlet** (2nd-order, force re-evaluated at the new position each
        step) rather than first-order Euler, so the measured displacement — and the force inferred
        from it — is more faithful to the true Coulomb dynamics."""
        self.sources = []
        self.add_source(q_source, 0.0, 0.0)
        test = Charge(q=q_test, x=r, y=0.0, mass=mass)
        t = 0.0
        fx, fy = self.force_on(test)
        ax, ay = fx / mass, fy / mass
        for _ in range(max(1, steps)):
            # velocity Verlet: advance position with current accel, then correct velocity with the
            # average of old and new accelerations.
            test.x += test.vx * dt + 0.5 * ax * dt * dt
            test.y += test.vy * dt + 0.5 * ay * dt * dt
            fx, fy = self.force_on(test)
            ax_new, ay_new = fx / mass, fy / mass
            test.vx += 0.5 * (ax + ax_new) * dt
            test.vy += 0.5 * (ay + ay_new) * dt
            ax, ay = ax_new, ay_new
            t += dt
        displacement = math.hypot(test.x - r, test.y - 0.0)
        if t <= 0.0:
            return 0.0
        accel = 2.0 * displacement / (t * t)        # from d = ½·a·t²
        return mass * accel                          # F = m·a
