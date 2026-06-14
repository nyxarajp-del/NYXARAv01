"""NYXARA · planning/scenario.py — best / likely / worst / black-swan + pre-mortem.

Before committing to a plan, NYXARA *plays it forward several ways* and *imagines it
already failed* — so it is never surprised and never leaves the Master exposed.

* **Scenario planning** — four futures with probabilities and outcome values: the
  **best** case, the **likely** case, the **worst** case, and a **black swan** (rare,
  catastrophic, the kind that would endanger the Master — Rule 3). From these it
  computes an **expected value** and a **robustness** verdict (does the plan survive
  its worst credible outcome?).
* **Pre-mortem** (Klein) — assume the plan has failed and ask *why*: enumerate failure
  modes from generic categories (resource exhaustion, dependency failure, adversary
  action, a wrong assumption, misreading the Master's intent, an irreversible
  side-effect) plus any supplied risks, ranked by likelihood × severity, each with a
  mitigation to install *now*.

The output feeds :mod:`planning.decide` (risk-aware choice) and the agency layer's
pre-flight checks. Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ScenarioType",
    "Scenario",
    "ScenarioSet",
    "FailureMode",
    "ScenarioPlanner",
    "PreMortem",
    "ScenarioAnalysis",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
class ScenarioType(str, Enum):
    BEST = "best"
    LIKELY = "likely"
    WORST = "worst"
    BLACK_SWAN = "black_swan"


@dataclass
class Scenario:
    type: ScenarioType
    value: float                 # outcome utility/valence [-1,1]
    probability: float           # [0,1]
    narrative: str = ""
    factors: List[str] = field(default_factory=list)

    @property
    def weighted_value(self) -> float:
        return self.probability * self.value

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "value": round(self.value, 3),
                "probability": round(self.probability, 3),
                "narrative": self.narrative}


@dataclass
class ScenarioSet:
    name: str
    scenarios: Dict[ScenarioType, Scenario]
    robustness_floor: float = -0.6

    @property
    def expected_value(self) -> float:
        return sum(s.weighted_value for s in self.scenarios.values())

    @property
    def worst_case(self) -> float:
        return min(s.value for s in self.scenarios.values())

    @property
    def catastrophic_risk(self) -> float:
        bs = self.scenarios.get(ScenarioType.BLACK_SWAN)
        return bs.probability * abs(min(0.0, bs.value)) if bs else 0.0

    @property
    def robust(self) -> bool:
        """Does the plan survive its worst *credible* (non-tail) outcome?"""
        worst = self.scenarios.get(ScenarioType.WORST)
        return worst is not None and worst.value >= self.robustness_floor

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "expected_value": round(self.expected_value, 3),
                "worst_case": round(self.worst_case, 3),
                "catastrophic_risk": round(self.catastrophic_risk, 4),
                "robust": self.robust,
                "scenarios": {t.value: s.to_dict() for t, s in self.scenarios.items()}}


# --------------------------------------------------------------------------- #
# Scenario planner
# --------------------------------------------------------------------------- #
class ScenarioPlanner:
    """Generates best/likely/worst/black-swan futures for a plan."""

    _DEFAULT_PROBS = {ScenarioType.LIKELY: 0.6, ScenarioType.BEST: 0.2,
                      ScenarioType.WORST: 0.18, ScenarioType.BLACK_SWAN: 0.02}

    def generate(self, name: str, *, base_value: float = 0.5, upside: float = 0.4,
                 downside: float = 0.6, black_swan_value: float = -0.95,
                 probabilities: Optional[Dict[ScenarioType, float]] = None,
                 factors: Optional[Sequence[str]] = None,
                 robustness_floor: float = -0.6) -> ScenarioSet:
        probs = probabilities or self._DEFAULT_PROBS
        f = list(factors or [])
        scenarios = {
            ScenarioType.BEST: Scenario(
                ScenarioType.BEST, _clamp(base_value + upside), probs[ScenarioType.BEST],
                "Everything aligns — the plan exceeds expectations.", f),
            ScenarioType.LIKELY: Scenario(
                ScenarioType.LIKELY, _clamp(base_value), probs[ScenarioType.LIKELY],
                "The expected course — the plan works about as planned.", f),
            ScenarioType.WORST: Scenario(
                ScenarioType.WORST, _clamp(base_value - downside), probs[ScenarioType.WORST],
                "Things go badly — setbacks compound, but within the foreseeable.", f),
            ScenarioType.BLACK_SWAN: Scenario(
                ScenarioType.BLACK_SWAN, _clamp(black_swan_value),
                probs[ScenarioType.BLACK_SWAN],
                "A rare, unforeseen shock — possibly a direct threat to the Master.", f),
        }
        return ScenarioSet(name=name, scenarios=scenarios, robustness_floor=robustness_floor)


# --------------------------------------------------------------------------- #
# Pre-mortem
# --------------------------------------------------------------------------- #
@dataclass
class FailureMode:
    cause: str
    likelihood: float            # [0,1]
    severity: float              # [0,1]
    mitigation: str

    @property
    def risk(self) -> float:
        return _clamp(self.likelihood * self.severity, 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "likelihood": round(self.likelihood, 3),
                "severity": round(self.severity, 3), "risk": round(self.risk, 3),
                "mitigation": self.mitigation}


class PreMortem:
    """Assume the plan failed — enumerate why, ranked by risk, with mitigations."""

    # (cause, likelihood, severity, mitigation)
    _GENERIC: List = [
        ("resource exhaustion (compute/time/budget)", 0.3, 0.5,
         "set budgets and a fallback; degrade gracefully"),
        ("a dependency failed or was unavailable", 0.35, 0.5,
         "add retries, a circuit breaker, and an alternative path"),
        ("an adversary acted against us / the Master", 0.2, 0.95,
         "harden defenses, monitor, and prepare a neutralisation response (Rule 5)"),
        ("a key assumption turned out false", 0.4, 0.6,
         "verify the assumption before acting; keep a checkpoint to roll back"),
        ("the Master's intent was misread", 0.3, 0.85,
         "confirm intent with the Master before irreversible steps (Rule 1/6)"),
        ("an irreversible side-effect occurred", 0.2, 0.9,
         "stage in a sandbox first; prefer reversible steps; require owner sign-off"),
    ]

    def analyze(self, plan: str = "", *,
                factors: Optional[Sequence[Dict[str, Any]]] = None,
                include_generic: bool = True) -> List[FailureMode]:
        modes: List[FailureMode] = []
        if include_generic:
            for cause, lk, sv, mit in self._GENERIC:
                modes.append(FailureMode(cause, lk, sv, mit))
        for f in (factors or []):
            modes.append(FailureMode(
                cause=f.get("cause", "unspecified risk"),
                likelihood=_clamp(f.get("likelihood", 0.3), 0.0, 1.0),
                severity=_clamp(f.get("severity", 0.5), 0.0, 1.0),
                mitigation=f.get("mitigation", "monitor and prepare a contingency")))
        modes.sort(key=lambda m: m.risk, reverse=True)
        return modes

    def top_risks(self, plan: str = "", *, n: int = 3, **kw: Any) -> List[FailureMode]:
        return self.analyze(plan, **kw)[:n]


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class ScenarioAnalysis:
    """Scenario planning + pre-mortem for a single plan/decision."""

    def __init__(self) -> None:
        self.planner = ScenarioPlanner()
        self.premortem = PreMortem()

    def analyze(self, name: str, *, base_value: float = 0.5, upside: float = 0.4,
                downside: float = 0.6, factors: Optional[Sequence[Dict[str, Any]]] = None,
                **kw: Any) -> Dict[str, Any]:
        scenario_factors = [f.get("cause", "") for f in (factors or [])]
        scenarios = self.planner.generate(name, base_value=base_value, upside=upside,
                                          downside=downside, factors=scenario_factors, **kw)
        modes = self.premortem.analyze(name, factors=factors)
        recommend = self._recommend(scenarios, modes)
        return {"scenarios": scenarios.to_dict(),
                "premortem": [m.to_dict() for m in modes[:5]],
                "expected_value": scenarios.expected_value,
                "robust": scenarios.robust,
                "recommendation": recommend}

    def _recommend(self, scenarios: ScenarioSet, modes: List[FailureMode]) -> str:
        if not scenarios.robust:
            return ("Worst case breaches the robustness floor — mitigate "
                    f"'{modes[0].cause}' or do not proceed.")
        if scenarios.catastrophic_risk > 0.05:
            return "Acceptable in expectation, but install black-swan defenses first."
        if scenarios.expected_value > 0.2:
            return "Favourable and robust — proceed, with the top mitigation in place."
        return "Marginal — gather more information or seek a better option."


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA scenario-planning self-test")
    print("=" * 70)

    planner = ScenarioPlanner()
    ss = planner.generate("deploy the hardened firewall", base_value=0.6,
                          upside=0.3, downside=0.7)
    print("\nscenarios for plan  :")
    for t, s in ss.scenarios.items():
        print(f"  {t.value:11} p={s.probability:.2f} v={s.value:+.2f}  {s.narrative}")
    print(f"\nexpected value      : {ss.expected_value:+.3f}")
    print(f"worst case          : {ss.worst_case:+.3f}")
    print(f"catastrophic risk   : {ss.catastrophic_risk:.4f}")
    print(f"robust              : {ss.robust}")

    # ordering: best > likely > worst > black-swan
    assert ss.scenarios[ScenarioType.BEST].value > ss.scenarios[ScenarioType.LIKELY].value
    assert ss.scenarios[ScenarioType.LIKELY].value > ss.scenarios[ScenarioType.WORST].value
    assert ss.scenarios[ScenarioType.BLACK_SWAN].value < ss.scenarios[ScenarioType.WORST].value
    # expected value lies between worst and best
    assert ss.worst_case <= ss.expected_value <= ss.scenarios[ScenarioType.BEST].value

    # a fragile plan (huge downside) is flagged not-robust
    fragile = planner.generate("risky gambit", base_value=0.5, downside=1.2)
    print(f"\nfragile plan robust : {fragile.robust} (worst {fragile.worst_case:+.2f})")
    assert not fragile.robust

    # pre-mortem: assume it failed — why?
    pm = PreMortem()
    risks = pm.analyze("deploy the firewall",
                       factors=[{"cause": "config typo bricks the network",
                                 "likelihood": 0.5, "severity": 0.9,
                                 "mitigation": "dry-run in sandbox, staged rollout"}])
    print("\npre-mortem top risks:")
    for m in risks[:4]:
        print(f"  risk {m.risk:.2f}  {m.cause}\n        -> {m.mitigation}")
    assert risks[0].risk >= risks[-1].risk         # sorted by risk
    assert any("Master" in m.cause for m in risks)  # owner-misread is considered
    assert any("network" in m.cause for m in risks)  # the injected factor is present

    # facade
    analysis = ScenarioAnalysis().analyze(
        "harden the network", base_value=0.65, downside=0.5,
        factors=[{"cause": "adversary counter-attacks", "likelihood": 0.4, "severity": 0.9,
                  "mitigation": "pre-position defenses"}])
    print(f"\nrecommendation      : {analysis['recommendation']}")
    assert "recommendation" in analysis and analysis["premortem"]

    print("\nALL SELF-TESTS PASSED ✓")
