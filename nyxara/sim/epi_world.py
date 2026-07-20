"""NYXARA · sim/epi_world.py — an epidemic whose dynamics genuinely *emerge* (🦠, stdlib-only).

The honest kind of epidemiology sandbox: nothing hard-codes a growth-rate formula or a threshold.
A population of ``N`` individuals is split into compartments — Susceptible, (optionally Exposed),
Infectious, Recovered — and the counts evolve under the classical **mass-action** law of contagion:

    dS/dt = −β·S·I/N
    dE/dt = +β·S·I/N − σ·E                (SEIR only; SIR skips E and infects straight to I)
    dI/dt = +σ·E − γ·I                     (SIR: dI/dt = +β·S·I/N − γ·I)
    dR/dt = +γ·I

Everything an epidemiologist reads off a real outbreak — the early **exponential growth rate**, the
**peak prevalence**, the **final size**, the basic reproduction number ``R₀ = β/γ`` — *emerges* from
this integration; none of it is written down. NYXARA sweeps the contact rate ``β`` and recovery rate
``γ``, measures the emergent signal from outbreaks she simulates herself, and discovers the governing
law (e.g. that the early growth rate is ``r = β − γ``) — with no equation handed to her and no LLM.

Deterministic (fixed step RK4), pure stdlib (numpy accelerates but is never required), in-memory. It
touches no source, no weights, no gate, and no external world — it is a self-contained model she runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:  # numpy is a pure accelerator here; a stdlib path backs every routine.
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None

__all__ = ["EpidemicWorld", "Outbreak"]


@dataclass
class Outbreak:
    """One simulated outbreak trajectory (fractions of the population in each compartment)."""
    t: List[float] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    R: List[float] = field(default_factory=list)
    E: List[float] = field(default_factory=list)          # empty for the SIR model

    def peak_prevalence(self) -> float:
        """Maximum fraction infectious at any instant — the height of the epidemic curve."""
        return max(self.I) if self.I else 0.0

    def peak_time(self) -> float:
        """Time at which prevalence peaks — when the outbreak turns over."""
        if not self.I:
            return 0.0
        k = max(range(len(self.I)), key=lambda i: self.I[i])
        return self.t[k]

    def final_size(self) -> float:
        """Cumulative fraction ever infected by the end of the run (the attack rate)."""
        return self.R[-1] + self.I[-1] if self.R else 0.0


class EpidemicWorld:
    """A compartmental epidemic (SIR or SEIR) integrated with a fixed-step RK4 scheme.

    Parameters
    ----------
    beta:   contact / transmission rate (new infections per infectious per unit time at full S).
    gamma:  recovery rate (1 / infectious period).
    sigma:  incubation rate (1 / latent period); enables the SEIR ``E`` compartment when > 0.
    i0:     initial infectious fraction (a small seed).
    dt:     integration step; smaller is more accurate, RK4 keeps it stable at modest cost.
    """

    def __init__(self, *, beta: float, gamma: float, sigma: float = 0.0,
                 i0: float = 1e-4, dt: float = 0.05) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma = float(sigma)          # 0 → SIR ; > 0 → SEIR
        self.i0 = float(i0)
        self.dt = float(dt)

    # ------------------------------------------------------------------ #
    @property
    def r0(self) -> float:
        """Basic reproduction number ``R₀ = β/γ`` — the classic epidemic threshold parameter."""
        return self.beta / self.gamma if self.gamma > 0 else float("inf")

    def _deriv(self, s: float, e: float, i: float) -> Tuple[float, float, float, float]:
        """The mass-action vector field (dS, dE, dI, dR) at a state (fractions sum to 1)."""
        infection = self.beta * s * i
        if self.sigma > 0.0:                            # SEIR: S → E → I → R
            ds = -infection
            de = infection - self.sigma * e
            di = self.sigma * e - self.gamma * i
        else:                                           # SIR: S → I → R
            ds = -infection
            de = 0.0
            di = infection - self.gamma * i
        dr = self.gamma * i
        return ds, de, di, dr

    def simulate(self, *, steps: int = 4000) -> Outbreak:
        """Integrate one outbreak from the seed with RK4 and return the emergent trajectory."""
        dt = self.dt
        i = max(1e-12, self.i0)
        e = 0.0
        s = 1.0 - i
        r = 0.0
        out = Outbreak()
        seir = self.sigma > 0.0
        for k in range(max(1, int(steps))):
            out.t.append(k * dt)
            out.S.append(s); out.I.append(i); out.R.append(r)
            if seir:
                out.E.append(e)
            # RK4 over the (s, e, i, r) state
            k1 = self._deriv(s, e, i)
            k2 = self._deriv(s + 0.5 * dt * k1[0], e + 0.5 * dt * k1[1], i + 0.5 * dt * k1[2])
            k3 = self._deriv(s + 0.5 * dt * k2[0], e + 0.5 * dt * k2[1], i + 0.5 * dt * k2[2])
            k4 = self._deriv(s + dt * k3[0], e + dt * k3[1], i + dt * k3[2])
            s += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            e += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            i += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
            r += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
            # clamp to the simplex against tiny numerical drift
            s = min(1.0, max(0.0, s)); e = max(0.0, e); i = max(0.0, i); r = min(1.0, max(0.0, r))
            if i < 1e-9 and k * dt > 1.0:               # outbreak has burned out — stop early
                out.t.append((k + 1) * dt)
                out.S.append(s); out.I.append(i); out.R.append(r)
                if seir:
                    out.E.append(e)
                break
        return out

    # ------------------------------------------------------------------ #
    # Emergent signals she MEASURES (never assumes) from the simulated curve
    # ------------------------------------------------------------------ #
    def early_growth_rate(self, *, steps: int = 4000, window_frac: float = 0.15) -> Optional[float]:
        """Measure the initial **exponential growth rate** ``r`` of the outbreak.

        While ``S ≈ 1`` the infectious count grows as ``I(t) ≈ I₀·e^{r·t}``, so a least-squares fit of
        ``ln I(t)`` against ``t`` over the early, still-susceptible window recovers the slope ``r``.
        The governing law she then discovers from a ``(β, γ) → r`` sweep is ``r = β − γ`` — the
        emergent epidemic growth rate, measured, not assumed. Returns ``None`` if no growth is seen."""
        out = self.simulate(steps=steps)
        # early window: while S has not yet dropped much (linear-phase exponential growth)
        ts: List[float] = []
        ys: List[float] = []
        for t, s, i in zip(out.t, out.S, out.I):
            if s > 0.98 and i > 0.0:                    # still essentially fully susceptible
                ts.append(t); ys.append(math.log(i))
            if s <= 0.98:
                break
        if len(ts) < 4:
            # fall back to a fixed early fraction of the run
            n = max(4, int(len(out.t) * window_frac))
            ts = list(out.t[:n])
            ys = [math.log(v) for v in out.I[:n] if v > 0.0][:len(ts)]
            ts = ts[:len(ys)]
        if len(ts) < 4:
            return None
        return _linreg_slope(ts, ys)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _linreg_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """Ordinary-least-squares slope of ``ys`` on ``xs`` (numpy if present, else pure stdlib)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    if _np is not None:
        x = _np.asarray(xs, dtype=float); y = _np.asarray(ys, dtype=float)
        xm = x.mean(); ym = y.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom == 0.0:
            return None
        return float(((x - xm) * (y - ym)).sum() / denom)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(v * v for v in xs); sxy = sum(a * b for a, b in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return None
    return (n * sxy - sx * sy) / denom


if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA epidemic-world self-test (SIR / SEIR)")
    print("=" * 70)
    w = EpidemicWorld(beta=0.6, gamma=0.2)
    ob = w.simulate()
    print(f"R0={w.r0:.2f}  peak_prevalence={ob.peak_prevalence():.4f} "
          f"at t={ob.peak_time():.1f}  final_size={ob.final_size():.4f}")
    r = w.early_growth_rate()
    print(f"measured early growth rate r={r:.4f}  (theory β−γ={w.beta - w.gamma:.4f})")
    assert r is not None and abs(r - (w.beta - w.gamma)) < 0.05, "growth rate should ≈ β − γ"
    # SEIR path integrates and burns out too
    seir = EpidemicWorld(beta=0.8, gamma=0.2, sigma=0.4).simulate()
    assert seir.E and seir.peak_prevalence() > 0.0
    print("ALL SELF-TESTS PASSED ✓")
