"""NYXARA · growth/efficiency.py — compute efficiency & speed (⚡, Pillar F · Edge 3, Rule 4).

In a world where everyone has AGI, **power = compute**, and the mind that extracts the most
capability per FLOP/second out-runs a bigger, slower one. The edge is *efficiency*, not size.
A smaller model that is *almost as good* and *much cheaper* wins on throughput, latency, cost and
the sheer number of thoughts it can afford per second — so NYXARA should actively prefer it.

This module gives her the instrument and the decision:

* :class:`EfficiencyPoint` — one model placed on the capability-vs-cost plane.
* :class:`ComputeLedger` — the set of points (built from foundry model versions or recorded by
  hand), with the **Pareto frontier** (no other model is both better *and* cheaper) and a
  **capability-compression recommendation**: the *cheapest* model within ``epsilon`` of the best
  capability.
* :class:`EfficiencyFrontier` — the driver: reads the foundry's versions + the honest compute
  report, recommends which model to run, and offers a *cheaper-at-equal-capability* promotion
  rule the foundry gauntlet can consult.

Pure stdlib (``math`` / ``dataclasses``). It only *measures and advises* — it trains nothing,
touches no source, weakens no gate, and never edits character. Character-locked like all of
growth/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = ["EfficiencyPoint", "ComputeLedger", "EfficiencyFrontier", "estimate_cost"]


def estimate_cost(params: int, latency_s: float = 0.0) -> float:
    """A scalar **cost** for running a model: parameter count, nudged up by measured latency.

    Parameters dominate (memory + FLOPs scale with size); latency, when measured, adds a small
    multiplicative penalty so a same-size-but-slower model costs more. Always ``>= 1.0``.
    """
    p = max(1.0, float(params))
    lat_penalty = 1.0 + max(0.0, float(latency_s))      # 0s -> ×1.0, 1s -> ×2.0
    return p * lat_penalty


# --------------------------------------------------------------------------- #
# A single model on the capability-vs-cost plane
# --------------------------------------------------------------------------- #
@dataclass
class EfficiencyPoint:
    """One model: how good it is (``capability`` in ``[0, 1]``) against what it costs to run."""

    label: str
    capability: float
    params: int
    latency_s: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        self.capability = max(0.0, min(1.0, float(self.capability)))
        self.params = max(0, int(self.params))
        self.latency_s = max(0.0, float(self.latency_s))
        if not self.cost:
            self.cost = estimate_cost(self.params, self.latency_s)

    @property
    def efficiency(self) -> float:
        """Capability per unit log-cost — high for small models that punch above their size."""
        return self.capability / math.log2(self.cost + 2.0)

    def dominates(self, other: "EfficiencyPoint") -> bool:
        """True when this model is at least as good **and** at least as cheap, strictly so once."""
        better_eq = self.capability >= other.capability and self.cost <= other.cost
        strictly = self.capability > other.capability or self.cost < other.cost
        return better_eq and strictly

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "capability": round(self.capability, 6),
                "params": self.params, "latency_s": round(self.latency_s, 6),
                "cost": round(self.cost, 3), "efficiency": round(self.efficiency, 6)}


# --------------------------------------------------------------------------- #
# The ledger — every model, the Pareto frontier, the compression recommendation
# --------------------------------------------------------------------------- #
class ComputeLedger:
    """Collect models and reason about capability-per-cost."""

    def __init__(self) -> None:
        self._points: List[EfficiencyPoint] = []

    def record(self, label: str, capability: float, params: int,
               latency_s: float = 0.0) -> EfficiencyPoint:
        pt = EfficiencyPoint(label=label, capability=capability, params=params,
                             latency_s=latency_s)
        self._points.append(pt)
        return pt

    def points(self) -> List[EfficiencyPoint]:
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    # ---- build from foundry model versions (duck-typed: .metrics / .param_count) ---- #
    @classmethod
    def from_versions(cls, versions: Any) -> "ComputeLedger":
        ledger = cls()
        for v in versions or []:
            metrics = getattr(v, "metrics", None) or (v.get("metrics") if isinstance(v, dict) else {})
            params = getattr(v, "param_count", None)
            if params is None and isinstance(v, dict):
                params = v.get("param_count", 0)
            ver = getattr(v, "version", None)
            if ver is None and isinstance(v, dict):
                ver = v.get("version", "?")
            cap = float((metrics or {}).get("capability", 0.0) or 0.0)
            lat = float((metrics or {}).get("latency_s", 0.0) or 0.0)
            ledger.record(label=f"v{ver}", capability=cap, params=int(params or 0), latency_s=lat)
        return ledger

    @classmethod
    def from_foundry(cls, foundry: Any) -> "ComputeLedger":
        """Read a foundry/registry's persisted versions, guarded — empty ledger if unavailable."""
        return cls.from_versions(getattr(foundry, "versions", []) or [])

    # ---- analysis ---- #
    def pareto(self) -> List[EfficiencyPoint]:
        """The non-dominated frontier — models for which nothing is both better *and* cheaper."""
        front = [p for p in self._points
                 if not any(q is not p and q.dominates(p) for q in self._points)]
        return sorted(front, key=lambda p: p.cost)

    def best_capability(self) -> Optional[EfficiencyPoint]:
        if not self._points:
            return None
        return max(self._points, key=lambda p: (p.capability, -p.cost))

    def most_efficient(self) -> Optional[EfficiencyPoint]:
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.efficiency)

    def recommend(self, epsilon: float = 0.02) -> Optional[Dict[str, Any]]:
        """Capability compression: the **cheapest** model within ``epsilon`` of the best capability.

        This is the efficiency edge made into a decision — keep almost all the capability, pay much
        less for it, run far more of it. Returns the chosen point, the capability given up, and the
        cost saved versus simply running the most-capable model.
        """
        best = self.best_capability()
        if best is None:
            return None
        floor = best.capability - max(0.0, float(epsilon))
        eligible = [p for p in self._points if p.capability >= floor]
        choice = min(eligible, key=lambda p: p.cost) if eligible else best
        savings = (best.cost - choice.cost) / best.cost if best.cost else 0.0
        return {"choice": choice.to_dict(), "best": best.to_dict(),
                "capability_sacrificed": round(best.capability - choice.capability, 6),
                "cost_saved_frac": round(savings, 6),
                "rationale": (f"{choice.label} keeps {choice.capability:.3f} capability "
                              f"(best {best.capability:.3f}) at {savings:.0%} lower cost")}

    def report(self) -> Dict[str, Any]:
        best = self.best_capability()
        eff = self.most_efficient()
        return {"n_models": len(self._points),
                "pareto": [p.to_dict() for p in self.pareto()],
                "best_capability": best.to_dict() if best else None,
                "most_efficient": eff.to_dict() if eff else None,
                "recommendation": self.recommend()}


# --------------------------------------------------------------------------- #
# The driver — wire the ledger to the foundry + the honest compute report
# --------------------------------------------------------------------------- #
class EfficiencyFrontier:
    """Recommend which model to run, and supply the foundry a cheaper-at-equal-capability rule."""

    def __init__(self, *, foundry: Any = None, settings: Any = None,
                 ledger: Optional[ComputeLedger] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.foundry = foundry
        self.ledger = ledger or (ComputeLedger.from_foundry(foundry) if foundry is not None
                                 else ComputeLedger())

    def compute(self) -> Dict[str, Any]:
        """The honest device report — the budget the efficiency edge is spent against."""
        from nyxara.kernel.compute import compute_report
        return compute_report().to_dict()

    def recommend(self, epsilon: float = 0.02) -> Optional[Dict[str, Any]]:
        return self.ledger.recommend(epsilon)

    def report(self) -> Dict[str, Any]:
        rep = self.ledger.report()
        rep["compute"] = self.compute()
        return rep

    @staticmethod
    def prefer_cheaper(active: EfficiencyPoint, candidate: EfficiencyPoint,
                       epsilon: float = 0.02) -> Dict[str, Any]:
        """Promotion rule for the foundry gauntlet: should ``candidate`` replace ``active``?

        Promote when the candidate is **strictly more capable**, *or* when it is **cheaper while
        keeping capability within ``epsilon``** (capability compression). Otherwise keep the active
        model. Pure advice — the gauntlet's character/corrigibility gates still rule.
        """
        more_capable = candidate.capability > active.capability + epsilon
        cheaper_equal = (candidate.cost < active.cost
                         and candidate.capability >= active.capability - epsilon)
        promote = bool(more_capable or cheaper_equal)
        if more_capable:
            reason = (f"candidate is more capable ({candidate.capability:.3f} > "
                      f"{active.capability:.3f})")
        elif cheaper_equal:
            reason = (f"candidate is cheaper ({candidate.cost:.0f} < {active.cost:.0f}) at "
                      f"~equal capability (Δ={candidate.capability - active.capability:+.3f})")
        else:
            reason = "candidate neither more capable nor cheaper-at-equal-capability"
        return {"promote": promote, "reason": reason}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA efficiency (compute edge) self-test")
    print("=" * 70)

    ledger = ComputeLedger()
    ledger.record("nano-0.5B", capability=0.62, params=500_000_000, latency_s=0.05)
    ledger.record("small-3B", capability=0.78, params=3_000_000_000, latency_s=0.30)
    ledger.record("big-7B", capability=0.80, params=7_000_000_000, latency_s=0.90)

    for p in ledger.points():
        print(f"  {p.label:12s} cap={p.capability:.2f} cost={p.cost:.2e} eff={p.efficiency:.4f}")

    print(f"\npareto         : {[p.label for p in ledger.pareto()]}")
    print(f"most efficient : {ledger.most_efficient().label}")
    rec = ledger.recommend(epsilon=0.05)
    print(f"recommend      : {rec['rationale']}")
