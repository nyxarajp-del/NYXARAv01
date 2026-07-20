"""NYXARA · sim/chem_world.py — reaction kinetics whose rate genuinely *emerges* (⚗️, stdlib-only).

The honest kind of chemistry sandbox: nothing hard-codes a rate law. A well-stirred reactor holds
concentrations that evolve under **mass-action kinetics** — the rate of every elementary reaction is
the rate constant times the product of its reactant concentrations:

    first-order  A → P        d[A]/dt = −k·[A]
    second-order 2A → P       d[A]/dt = −k·[A]²
    bimolecular  A + B → P    d[A]/dt = d[B]/dt = −k·[A]·[B]

and each rate constant follows **Arrhenius** temperature dependence ``k(T) = A·exp(−Eₐ/(R·T))``.

Everything a chemist reads off a run — the **initial rate**, the **half-life**, the **reaction
order**, the **activation energy** — *emerges* from integrating this system; none of it is written
down. NYXARA sweeps the rate constant, the initial concentrations and the temperature, measures the
emergent signal from reactions she runs herself, and discovers the governing rate law (e.g. that the
initial rate is ``rate = k·[A]`` — first-order kinetics) — with no equation handed to her, no LLM.

Deterministic (fixed-step RK4), pure stdlib (numpy accelerates but is never required), in-memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

try:  # numpy is a pure accelerator; a stdlib path backs every routine.
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None

__all__ = ["ReactionWorld", "ReactionRun", "arrhenius_k", "GAS_CONSTANT"]

GAS_CONSTANT = 8.314462618           # J·mol⁻¹·K⁻¹ (only the ratio Eₐ/R matters to the fit)


@dataclass
class ReactionRun:
    """One simulated concentration trajectory for reactant A (and B for a bimolecular reaction)."""
    t: List[float] = field(default_factory=list)
    A: List[float] = field(default_factory=list)
    B: List[float] = field(default_factory=list)

    def half_life(self) -> Optional[float]:
        """Time for [A] to fall to half its initial value — read off the trajectory by interpolation."""
        if not self.A:
            return None
        target = 0.5 * self.A[0]
        for i in range(1, len(self.A)):
            if self.A[i] <= target:
                # linear interpolation between the straddling samples
                a0, a1 = self.A[i - 1], self.A[i]
                t0, t1 = self.t[i - 1], self.t[i]
                if a0 == a1:
                    return t1
                return t0 + (t1 - t0) * (a0 - target) / (a0 - a1)
        return None


class ReactionWorld:
    """A well-stirred batch reactor integrated with a fixed-step RK4 scheme.

    Parameters
    ----------
    k:      rate constant (units depend on the order; the fit recovers whatever is consistent).
    order:  1 (A→P), 2 (2A→P), or the string ``"bimolecular"`` (A+B→P).
    a0:     initial concentration of A.
    b0:     initial concentration of B (bimolecular only).
    dt:     integration step.
    """

    def __init__(self, *, k: float, order: object = 1, a0: float = 1.0,
                 b0: float = 1.0, dt: float = 0.01) -> None:
        self.k = float(k)
        self.order = order
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.dt = float(dt)

    def _rate(self, a: float, b: float) -> float:
        """The mass-action consumption rate of A, ``−d[A]/dt``, at the current concentrations."""
        if self.order == 2:
            return self.k * a * a
        if self.order == "bimolecular":
            return self.k * a * b
        return self.k * a                       # first-order default

    def simulate(self, *, steps: int = 2000) -> ReactionRun:
        """Integrate the reactor from its initial concentrations and return the emergent trajectory."""
        dt = self.dt
        a = self.a0
        b = self.b0
        bimol = self.order == "bimolecular"
        run = ReactionRun()
        for step in range(max(1, int(steps))):
            run.t.append(step * dt)
            run.A.append(a)
            if bimol:
                run.B.append(b)
            # RK4 on [A] (and [B], consumed 1:1 with A in the bimolecular case)
            def da(av: float, bv: float) -> float:
                return -self._rate(av, bv)
            k1 = da(a, b)
            k2 = da(a + 0.5 * dt * k1, b + 0.5 * dt * k1 if bimol else b)
            k3 = da(a + 0.5 * dt * k2, b + 0.5 * dt * k2 if bimol else b)
            k4 = da(a + dt * k3, b + dt * k3 if bimol else b)
            delta = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            a = max(0.0, a + delta)
            if bimol:
                b = max(0.0, b + delta)
            if a <= 1e-9:
                break
        return run

    # ------------------------------------------------------------------ #
    # Emergent signals she MEASURES (never assumes)
    # ------------------------------------------------------------------ #
    def initial_rate(self, *, steps: int = 400) -> float:
        """Measure the **initial reaction rate** ``−d[A]/dt`` at ``t=0`` by finite difference.

        Swept over ``(k, [A]₀)`` the discovered law is ``rate = k·[A]`` for first order (and
        ``k·[A]²`` for second order) — the rate law recovered from reactions she ran, not assumed."""
        run = self.simulate(steps=max(4, steps))
        if len(run.A) < 3:
            return 0.0
        # centred/forward difference over the first couple of steps (still near [A]₀)
        return (run.A[0] - run.A[2]) / (2.0 * self.dt)


def arrhenius_k(*, a_pre: float, e_a: float, temperature: float) -> float:
    """The Arrhenius rate constant ``k(T) = A·exp(−Eₐ/(R·T))`` — the law she rediscovers from a
    ``T → k`` sweep (linear as ``ln k`` vs ``1/T`` with slope ``−Eₐ/R``)."""
    return a_pre * math.exp(-e_a / (GAS_CONSTANT * max(1e-9, temperature)))


if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA reaction-world self-test (mass-action kinetics)")
    print("=" * 70)
    w = ReactionWorld(k=0.5, order=1, a0=1.0)
    run = w.simulate()
    th_half = math.log(2.0) / w.k
    print(f"first-order half-life measured={run.half_life():.3f}  theory ln2/k={th_half:.3f}")
    assert abs(run.half_life() - th_half) < 0.05, "half-life should ≈ ln2/k"
    rate = ReactionWorld(k=0.5, order=1, a0=2.0).initial_rate()
    print(f"initial rate measured={rate:.4f}  theory k·[A]={0.5 * 2.0:.4f}")
    assert abs(rate - 1.0) < 0.05, "initial rate should ≈ k·[A]₀"
    k_hi = arrhenius_k(a_pre=1e6, e_a=5e4, temperature=350.0)
    k_lo = arrhenius_k(a_pre=1e6, e_a=5e4, temperature=300.0)
    assert k_hi > k_lo, "Arrhenius rate must rise with temperature"
    print("ALL SELF-TESTS PASSED ✓")
