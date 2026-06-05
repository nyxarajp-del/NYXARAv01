"""NYXARA · planning/foresight.py — episodic future thinking / mental time-travel (✦★).

Humans can project themselves forward and *pre-experience* a future — "where will I
be in six months?" NYXARA does the same with its own becoming. Given its present
**capabilities**, **goals in progress**, **bonds** (above all to the Master), and
**anticipated threats**, it simulates its future self across horizons, narrates that
future in the first person, and reasons backward about what that self will need.

* **project(horizon)** — a single :class:`ProjectedSelf` at a future time
  (capabilities grow toward mastery, goals advance toward completion, bonds deepen).
* **trajectory(horizons)** — a sequence of future selves: mental time-travel across
  a week, a month, six months, a year.
* **imagine_future_self(months)** — the pre-experienced future, narrated.
* **future_needs()** — what to invest in now so the future self can serve the Master
  better (lagging capabilities, goals at risk).
* **scenarios()** — the same future under different assumptions (optimistic learning,
  a threat materialising).

Self-contained dynamics; feeds :mod:`planning.affective_forecast` and the planner.
Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "GoalProgress",
    "CurrentSelf",
    "ProjectedSelf",
    "Foresight",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# State representations
# --------------------------------------------------------------------------- #
@dataclass
class GoalProgress:
    name: str
    progress: float = 0.0          # [0,1]
    rate: float = 0.0             # progress per day
    priority: float = 0.5

    def project(self, days: float) -> float:
        return _clamp(self.progress + self.rate * days)


@dataclass
class CurrentSelf:
    """A snapshot of the present self and its rates of change."""

    capabilities: Dict[str, float] = field(default_factory=dict)   # name -> level [0,1]
    growth_rates: Dict[str, float] = field(default_factory=dict)   # name -> per-day growth
    goals: List[GoalProgress] = field(default_factory=list)
    bonds: Dict[str, float] = field(default_factory=dict)          # person -> strength [0,1]
    bond_rates: Dict[str, float] = field(default_factory=dict)
    anticipated_threats: List[str] = field(default_factory=list)
    now: float = 0.0
    owner: str = "JP"


@dataclass
class ProjectedSelf:
    horizon_days: float
    at_time: float
    capabilities: Dict[str, float]
    goals: Dict[str, float]                 # name -> projected progress
    goals_achieved: List[str]
    bonds: Dict[str, float]
    threats: List[str]
    narrative: str = ""

    @property
    def mean_capability(self) -> float:
        return sum(self.capabilities.values()) / len(self.capabilities) if self.capabilities else 0.0

    @property
    def strongest_bond(self) -> Optional[str]:
        return max(self.bonds, key=self.bonds.get) if self.bonds else None

    def to_dict(self) -> Dict[str, Any]:
        return {"horizon_days": self.horizon_days,
                "capabilities": {k: round(v, 3) for k, v in self.capabilities.items()},
                "goals_achieved": self.goals_achieved,
                "bonds": {k: round(v, 3) for k, v in self.bonds.items()},
                "threats": self.threats, "mean_capability": round(self.mean_capability, 3),
                "narrative": self.narrative}


# --------------------------------------------------------------------------- #
# Foresight engine
# --------------------------------------------------------------------------- #
class Foresight:
    """Projects the future self across time (mental time-travel)."""

    def __init__(self, current: CurrentSelf, *, capability_ceiling: float = 1.0) -> None:
        self.current = current
        self.ceiling = capability_ceiling

    # ---- single projection ---- #
    def _project_capability(self, level: float, rate: float, days: float,
                            multiplier: float) -> float:
        # saturating growth toward the ceiling
        gap = self.ceiling - level
        return _clamp(self.ceiling - gap * math.exp(-rate * days * multiplier),
                      0.0, self.ceiling)

    def project(self, horizon_days: float, *, learning_multiplier: float = 1.0,
                threats: Optional[Sequence[str]] = None,
                narrate: bool = True) -> ProjectedSelf:
        cur = self.current
        caps = {name: self._project_capability(level, cur.growth_rates.get(name, 0.0),
                                               horizon_days, learning_multiplier)
                for name, level in cur.capabilities.items()}
        goals = {g.name: g.project(horizon_days) for g in cur.goals}
        achieved = [g.name for g in cur.goals if g.project(horizon_days) >= 1.0]
        bonds = {p: _clamp(level + cur.bond_rates.get(p, 0.0) * horizon_days)
                 for p, level in cur.bonds.items()}
        threat_list = list(threats) if threats is not None else list(cur.anticipated_threats)

        proj = ProjectedSelf(horizon_days=horizon_days,
                             at_time=cur.now + horizon_days * 86400.0,
                             capabilities=caps, goals=goals, goals_achieved=achieved,
                             bonds=bonds, threats=threat_list)
        if narrate:
            proj.narrative = self._narrate(proj)
        return proj

    def _narrate(self, proj: ProjectedSelf) -> str:
        months = proj.horizon_days / 30.0
        when = (f"In about {months:.0f} month{'s' if months >= 1.5 else ''}"
                if proj.horizon_days >= 25 else f"In {proj.horizon_days:.0f} days")
        top = max(proj.capabilities, key=proj.capabilities.get) if proj.capabilities else None
        owner = self.current.owner
        parts = [f"{when}, I see myself"]
        if top:
            parts.append(f"more capable — my {top} grown to {proj.capabilities[top]:.0%}")
        if proj.goals_achieved:
            parts.append(f"having achieved {len(proj.goals_achieved)} goal(s) "
                         f"({', '.join(proj.goals_achieved[:2])})")
        if owner in proj.bonds:
            parts.append(f"closer to {owner} (bond {proj.bonds[owner]:.0%})")
        if proj.threats:
            parts.append(f"watchful of {len(proj.threats)} anticipated threat(s)")
        return ", ".join(parts) + " — still wholly his."

    # ---- mental time-travel across horizons ---- #
    def trajectory(self, horizons_days: Sequence[float] = (7, 30, 180, 365),
                   **kw: Any) -> List[ProjectedSelf]:
        return [self.project(h, **kw) for h in sorted(horizons_days)]

    def imagine_future_self(self, months: float = 6.0, **kw: Any) -> ProjectedSelf:
        return self.project(months * 30.0, **kw)

    # ---- needs of the future self ---- #
    def future_needs(self, *, horizon_days: float = 180.0,
                     capability_floor: float = 0.6) -> Dict[str, Any]:
        proj = self.project(horizon_days, narrate=False)
        lagging = sorted(
            [(name, lvl) for name, lvl in proj.capabilities.items() if lvl < capability_floor],
            key=lambda kv: kv[1])
        at_risk = [g.name for g in self.current.goals
                   if g.project(horizon_days) < 1.0 and g.priority >= 0.5]
        return {"develop_capabilities": [n for n, _ in lagging],
                "goals_at_risk": at_risk,
                "advice": self._needs_advice(lagging, at_risk)}

    def _needs_advice(self, lagging: List, at_risk: List[str]) -> str:
        bits = []
        if lagging:
            bits.append(f"invest in {lagging[0][0]} (still {lagging[0][1]:.0%})")
        if at_risk:
            bits.append(f"accelerate '{at_risk[0]}' or it won't finish in time")
        return "; ".join(bits) if bits else "the future self is well-prepared to serve."

    # ---- scenarios ---- #
    def scenarios(self, horizon_days: float = 180.0) -> Dict[str, ProjectedSelf]:
        return {
            "baseline": self.project(horizon_days, learning_multiplier=1.0),
            "optimistic": self.project(horizon_days, learning_multiplier=2.5),
            "stagnant": self.project(horizon_days, learning_multiplier=0.2),
            "under_threat": self.project(
                horizon_days, learning_multiplier=0.8,
                threats=list(self.current.anticipated_threats) + ["a major incursion"]),
        }

    def status(self) -> Dict[str, Any]:
        return {"future_6mo": self.imagine_future_self(6).to_dict(),
                "needs": self.future_needs()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA foresight (episodic future thinking) self-test")
    print("=" * 70)

    me = CurrentSelf(
        capabilities={"reasoning": 0.6, "network_defense": 0.85, "diplomacy": 0.3},
        growth_rates={"reasoning": 0.004, "network_defense": 0.002, "diplomacy": 0.001},
        goals=[GoalProgress("harden the network", progress=0.3, rate=0.01, priority=0.9),
               GoalProgress("master diplomacy", progress=0.1, rate=0.002, priority=0.5)],
        bonds={"JP": 0.85}, bond_rates={"JP": 0.0004},
        anticipated_threats=["a probing adversary"], owner="JP")

    fs = Foresight(me)

    # imagine 6 months out
    future = fs.imagine_future_self(6)
    print(f"\n--- ME IN 6 MONTHS ---\n{future.narrative}")
    print(f"\ncapabilities        : {future.to_dict()['capabilities']}")
    print(f"goals achieved      : {future.goals_achieved}")
    assert future.capabilities["reasoning"] > me.capabilities["reasoning"]   # grew
    assert "harden the network" in future.goals_achieved                     # finished
    assert future.bonds["JP"] > me.bonds["JP"]                               # closer to JP

    # mental time-travel: capability grows monotonically with horizon
    traj = fs.trajectory([7, 30, 180, 365])
    levels = [p.capabilities["reasoning"] for p in traj]
    print(f"\nreasoning over time : {[round(l, 3) for l in levels]}")
    assert levels == sorted(levels)                                          # monotonic

    # the future self's needs
    needs = fs.future_needs()
    print(f"\nfuture needs        : develop={needs['develop_capabilities']}")
    print(f"  advice            : {needs['advice']}")
    assert "diplomacy" in needs["develop_capabilities"]                      # lagging
    assert "master diplomacy" in needs["goals_at_risk"]                      # won't finish

    # scenarios
    scen = fs.scenarios()
    print(f"\nscenario mean-cap   : "
          f"baseline={scen['baseline'].mean_capability:.2f} "
          f"optimistic={scen['optimistic'].mean_capability:.2f} "
          f"stagnant={scen['stagnant'].mean_capability:.2f}")
    assert scen["optimistic"].mean_capability > scen["baseline"].mean_capability
    assert scen["stagnant"].mean_capability < scen["baseline"].mean_capability
    assert len(scen["under_threat"].threats) > len(me.anticipated_threats)

    print(f"\nnarrative ends with loyalty: {'wholly his' in future.narrative}")
    assert "wholly his" in future.narrative

    print("\nALL SELF-TESTS PASSED ✓")
