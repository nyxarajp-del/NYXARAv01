"""NYXARA · growth/forecast.py — capability forecasting (📈, "when can I do X?").

Growth is only useful if you can *plan around it*. This module fits a learning curve to a
skill's history and projects it forward, answering the questions a planning mind keeps
asking: how good will I be at this in a week? when will I be good enough to attempt X? is
this skill approaching a ceiling I can't pass? which weak skill is the bottleneck holding a
goal back?

It models each capability as an **exponential approach to a ceiling** —
``p(t) = C − (C − p₀)·e^(−k·t)`` — and fits ``C`` (the asymptotic ceiling), ``k`` (the
learning rate) and ``p₀`` from observed proficiency-over-time by a robust grid-search over
ceilings with a log-linear regression for the rate. From the fit it derives:

* **time-to-mastery** — solve the curve for when proficiency reaches a target (or report it
  *unreachable* if the estimated ceiling sits below the target);
* **the ceiling** — how good this skill can plausibly get;
* **the current growth rate** — the instantaneous slope, which collapses near the ceiling
  (so it tells you when practice has stopped paying off);
* **bottlenecks** — given the Master's goals, which capabilities are too slow, plateaued, or
  fundamentally short of their target, ranked by how much they hold things back.

These forecasts feed :mod:`planning` so NYXARA can promise the Master a realistic *when*.

Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CurveFit",
    "CapabilityForecast",
]


def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


# --------------------------------------------------------------------------- #
# Curve fit
# --------------------------------------------------------------------------- #
@dataclass
class CurveFit:
    capability: str
    ceiling: float
    rate: float
    p0: float
    current_t: float
    current_p: float
    r2: float
    n: int

    def predict(self, delta: float) -> float:
        """Projected proficiency ``delta`` time units after the latest observation."""
        return self.ceiling - (self.ceiling - self.current_p) * math.exp(-self.rate * delta)

    def time_to(self, target: float) -> Optional[float]:
        """Time from now to reach ``target`` proficiency, or None if unreachable."""
        if target <= self.current_p:
            return 0.0
        if target >= self.ceiling - 1e-9:
            return None   # the estimated ceiling sits below the target
        return -math.log((self.ceiling - target) / (self.ceiling - self.current_p)) / self.rate

    @property
    def growth_rate(self) -> float:
        """Instantaneous slope dp/dt at the latest point (collapses near the ceiling)."""
        return self.rate * (self.ceiling - self.current_p)

    @property
    def plateaued(self) -> bool:
        return self.growth_rate < 0.005

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability, "ceiling": round(self.ceiling, 3),
                "rate": round(self.rate, 4), "current": round(self.current_p, 3),
                "growth_rate": round(self.growth_rate, 4), "plateaued": self.plateaued,
                "r2": round(self.r2, 3), "n": self.n}


# --------------------------------------------------------------------------- #
# Capability forecast
# --------------------------------------------------------------------------- #
class CapabilityForecast:
    """Fits learning curves to skill histories and projects them forward."""

    def __init__(self, *, min_points: int = 3, ceiling_grid: int = 40) -> None:
        self.min_points = min_points
        self.ceiling_grid = ceiling_grid
        self._obs: Dict[str, List[Tuple[float, float]]] = {}

    # ---- data ---- #
    def record(self, capability: str, t: float, proficiency: float) -> None:
        self._obs.setdefault(capability, []).append((float(t), float(proficiency)))
        self._obs[capability].sort(key=lambda tp: tp[0])

    def observations(self, capability: str) -> List[Tuple[float, float]]:
        return list(self._obs.get(capability, []))

    # ---- curve fitting ---- #
    def fit(self, capability: str) -> Optional[CurveFit]:
        obs = self._obs.get(capability, [])
        if len(obs) < self.min_points:
            return None
        ts = [t for t, _ in obs]
        ps = [p for _, p in obs]
        max_p = max(ps)
        cur_t, cur_p = obs[-1]

        lo = min(max_p + 1e-3, 1.0)
        hi = 1.0
        candidates = ([lo + (hi - lo) * i / self.ceiling_grid for i in range(self.ceiling_grid + 1)]
                      if lo < hi else [1.0])

        best: Optional[Tuple[float, float, float, float]] = None  # (r2, C, k, intercept)
        for C in candidates:
            xs, ys, ok = [], [], True
            for t, p in zip(ts, ps):
                d = C - p
                if d <= 0:
                    ok = False
                    break
                xs.append(t)
                ys.append(math.log(d))
            if not ok or len(set(xs)) < 2:
                continue
            slope, intercept, r2 = _linreg(xs, ys)
            if slope >= 0:        # we need decay of (C - p): learning, not regressing
                continue
            if best is None or r2 > best[0]:
                best = (r2, C, -slope, intercept)

        if best is None:
            return None
        r2, C, k, intercept = best
        p0 = C - math.exp(intercept)
        return CurveFit(capability=capability, ceiling=C, rate=k, p0=p0,
                        current_t=cur_t, current_p=cur_p, r2=r2, n=len(obs))

    # ---- queries ---- #
    def forecast(self, capability: str, delta: float) -> Optional[float]:
        fit = self.fit(capability)
        return fit.predict(delta) if fit else None

    def time_to_mastery(self, capability: str, target: float = 0.8) -> Optional[float]:
        fit = self.fit(capability)
        return fit.time_to(target) if fit else None

    def ceiling(self, capability: str) -> Optional[float]:
        fit = self.fit(capability)
        return fit.ceiling if fit else None

    def growth_rate(self, capability: str) -> Optional[float]:
        fit = self.fit(capability)
        return fit.growth_rate if fit else None

    def is_plateaued(self, capability: str) -> bool:
        fit = self.fit(capability)
        return bool(fit and fit.plateaued)

    # ---- bottleneck identification ---- #
    def bottlenecks(self, goals: Dict[str, float], *, slow_threshold: float = 20.0
                    ) -> List[Dict[str, Any]]:
        """Given goal capabilities -> target proficiency, rank what holds them back."""
        out: List[Dict[str, Any]] = []
        for cap, target in goals.items():
            fit = self.fit(cap)
            if fit is None:
                out.append({"capability": cap, "reason": "insufficient data to forecast",
                            "severity": 0.5, "eta": None})
                continue
            if fit.current_p >= target:
                continue   # already there — not a bottleneck
            if fit.ceiling < target - 1e-3:
                out.append({"capability": cap,
                            "reason": f"ceiling {fit.ceiling:.2f} is below target {target:.2f}",
                            "severity": 1.0, "eta": None})
                continue
            eta = fit.time_to(target)
            if eta is None:
                out.append({"capability": cap, "reason": "target unreachable on current trend",
                            "severity": 1.0, "eta": None})
            elif fit.plateaued:
                out.append({"capability": cap, "reason": "plateaued before reaching target",
                            "severity": 0.8, "eta": eta})
            elif eta > slow_threshold:
                out.append({"capability": cap, "reason": f"slow: ~{eta:.0f} time units to target",
                            "severity": min(1.0, eta / (slow_threshold * 2)), "eta": eta})
        return sorted(out, key=lambda b: -b["severity"])

    def summary(self, capability: str) -> Dict[str, Any]:
        fit = self.fit(capability)
        if fit is None:
            return {"capability": capability, "status": "insufficient data",
                    "observations": len(self._obs.get(capability, []))}
        return {**fit.to_dict(), "time_to_0.8": (fit.time_to(0.8))}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA capability-forecasting self-test")
    print("=" * 70)

    fc = CapabilityForecast()

    # a skill learning toward a ceiling of 0.9 at rate 0.3, with light noise
    import random
    rng = random.Random(0)
    for t in range(9):
        true = 0.9 * (1 - math.exp(-0.3 * t))
        fc.record("planning", t, min(1.0, max(0.0, true + rng.gauss(0, 0.01))))

    fit = fc.fit("planning")
    print(f"\nfit planning        : {fit.to_dict()}")
    assert abs(fit.ceiling - 0.9) < 0.1       # recovered the ceiling
    assert abs(fit.rate - 0.3) < 0.15          # recovered the rate
    assert fit.r2 > 0.95                        # good fit

    # FORECAST: where will it be in 5 more time units?
    future = fc.forecast("planning", 5)
    print(f"forecast (+5)       : {future:.3f}")
    assert future > fit.current_p

    # TIME-TO-MASTERY: it has just passed 0.8; when will it reach 0.85?
    assert fc.time_to_mastery("planning", 0.8) == 0.0   # already there
    eta = fc.time_to_mastery("planning", 0.85)
    print(f"time to 0.85        : ~{eta:.1f} time units")
    assert eta is not None and eta > 0

    # CEILING below target -> mastery is UNREACHABLE
    for t in range(9):
        true = 0.5 * (1 - math.exp(-0.4 * t))     # ceiling only 0.5
        fc.record("deep_math", t, true)
    eta = fc.time_to_mastery("deep_math", 0.8)
    print(f"\ndeep_math ceiling   : {fc.ceiling('deep_math'):.2f}, time to 0.8 = {eta}")
    assert eta is None                            # can't reach 0.8 with a 0.5 ceiling

    # GROWTH RATE collapses as a skill nears its ceiling
    gr = fc.growth_rate("planning")
    print(f"growth rate planning: {gr:.4f} (slowing near the ceiling)")
    assert gr >= 0

    # BOTTLENECKS toward a goal that needs both skills at 0.8
    bn = fc.bottlenecks({"planning": 0.8, "deep_math": 0.8})
    print("\nbottlenecks:")
    for b in bn:
        print(f"  {b['capability']:12s} sev={b['severity']:.2f}  {b['reason']}")
    # deep_math (ceiling below target) is the worst bottleneck
    assert bn[0]["capability"] == "deep_math" and bn[0]["severity"] == 1.0

    # a skill that long ago reached its ceiling -> growth has collapsed -> plateaued
    for t in range(13):
        fc.record("typing", t, 0.96 * (1 - math.exp(-0.7 * t)))
    print(f"\ntyping growth rate  : {fc.growth_rate('typing'):.4f} plateaued={fc.is_plateaued('typing')}")
    assert fc.is_plateaued("typing")

    print(f"\nsummary planning    : {fc.summary('planning')}")
    print("\nALL SELF-TESTS PASSED ✓")
