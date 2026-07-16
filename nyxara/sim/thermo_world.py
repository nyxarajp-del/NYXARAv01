"""NYXARA · sim/thermo_world.py — an ideal gas whose pressure genuinely *emerges* (🌡️, stdlib-only).

This is the honest kind of thermodynamics sandbox: nothing implements ``P·V = N·k·T``. Instead a
box holds ``N`` non-interacting point particles that fly in straight lines and bounce **elastically
off the walls**; the pressure is *measured* as the momentum they hand to the walls per unit time —
exactly how pressure arises in the real kinetic theory of gases. NYXARA sweeps particle count and
temperature, measures the emergent wall force, and discovers that it is proportional to ``N·T`` —
rediscovering the ideal-gas law from mechanics she simulated, with no formula and no LLM.

Deterministic (seeded), pure stdlib, in-memory. The container size is held fixed across a sweep, so
its geometric factor is absorbed into the single fitted constant — leaving the clean law ``P ∝ N·T``.
"""

from __future__ import annotations

import math
import random
from typing import List

__all__ = ["GasBox"]


class GasBox:
    """A 2-D box of ``N`` ideal (non-interacting) particles bouncing elastically off the walls.

    Temperature sets the speed scale (mean kinetic energy ∝ T); the measured wall force — total
    momentum delivered to the walls per unit time — is the emergent pressure signal."""

    def __init__(self, *, size: float = 1.0, dt: float = 0.005, seed: int = 0) -> None:
        self.size = float(size)
        self.dt = float(dt)
        self.seed = int(seed)

    def measure_wall_force(self, n_particles: int, temperature: float, *,
                           steps: int = 4000, warmup: int = 200) -> float:
        """Simulate the gas and return the emergent wall force (momentum transferred per unit time).

        Speeds are drawn so that mean ½·v² ≈ temperature (unit mass, unit k_B); every wall bounce
        reverses the perpendicular velocity and deposits ``2·|v_perp|`` of momentum. Averaged over
        the run (after a short warm-up) this force is proportional to ``N·T`` at fixed volume."""
        rng = random.Random((self.seed * 1_000_003) ^ (n_particles * 7919) ^ hash(round(temperature, 6)))
        L = self.size
        speed = math.sqrt(2.0 * max(1e-9, temperature))     # ½·v² = T  ⇒  v = √(2T)  (unit mass)
        xs: List[float] = []; ys: List[float] = []
        vxs: List[float] = []; vys: List[float] = []
        for _ in range(max(1, n_particles)):
            xs.append(rng.uniform(0.0, L)); ys.append(rng.uniform(0.0, L))
            ang = rng.uniform(0.0, 2.0 * math.pi)
            vxs.append(speed * math.cos(ang)); vys.append(speed * math.sin(ang))
        dt = self.dt
        momentum = 0.0
        measured_time = 0.0
        for s in range(max(1, steps)):
            counting = s >= warmup
            for i in range(len(xs)):
                xs[i] += vxs[i] * dt
                ys[i] += vys[i] * dt
                # elastic wall bounces; each deposits 2·|v_perp| of momentum onto the wall
                if xs[i] < 0.0:
                    xs[i] = -xs[i]
                    if counting:
                        momentum += 2.0 * abs(vxs[i])
                    vxs[i] = -vxs[i]
                elif xs[i] > L:
                    xs[i] = 2.0 * L - xs[i]
                    if counting:
                        momentum += 2.0 * abs(vxs[i])
                    vxs[i] = -vxs[i]
                if ys[i] < 0.0:
                    ys[i] = -ys[i]
                    if counting:
                        momentum += 2.0 * abs(vys[i])
                    vys[i] = -vys[i]
                elif ys[i] > L:
                    ys[i] = 2.0 * L - ys[i]
                    if counting:
                        momentum += 2.0 * abs(vys[i])
                    vys[i] = -vys[i]
            if counting:
                measured_time += dt
        if measured_time <= 0.0:
            return 0.0
        return momentum / measured_time          # force = momentum / time  (∝ pressure at fixed V)
