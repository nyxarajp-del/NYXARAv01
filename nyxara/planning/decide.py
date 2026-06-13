"""NYXARA · planning/decide.py — initiative governor + MCDA + regret-min (⬆).

Choosing well has two parts: picking the *best* option, and deciding whether NYXARA
may act on it *alone*.

* **MCDA** (multi-criteria decision analysis) — score each option across weighted
  criteria (the Master's benefit, safety, cost, speed…), normalising benefits and
  costs, to rank them by overall utility.
* **Regret minimisation** — for high-stakes calls, the *minimax-regret* option is
  preferred: the one whose worst-case shortfall across criteria is smallest (robust,
  not merely highest-average).
* **Initiative governor** — the gate on autonomy. An option may be acted on alone only
  when **confidence × reversibility** clears threshold and the stakes aren't an
  irreversible gamble. Irreversible + high-stakes → **ASK** the Master. An option that
  isn't owner-aligned is **REJECTED** outright (Rule 1). Reversible mistakes are cheap;
  irreversible ones the Master must bless.

This is where proposal confidence, value-of-information, and affective forecasts come
together into a single, governed choice. Depends on :mod:`config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.config import NyxaraSettings, get_settings

__all__ = [
    "Criterion",
    "Option",
    "DecisionAction",
    "MCDA",
    "RegretMinimizer",
    "InitiativeGovernor",
    "DecisionResult",
    "Decider",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Criteria & options
# --------------------------------------------------------------------------- #
@dataclass
class Criterion:
    name: str
    weight: float = 1.0
    maximize: bool = True       # True = benefit (more is better); False = cost


@dataclass
class Option:
    name: str
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.7
    reversibility: float = 1.0
    stakes: float = 0.3
    owner_aligned: bool = True
    payload: Any = None
    # anticipated emotional outcome of choosing this option — "how will I feel after?" — read by
    # the affective forecaster when a Decider is built with affective_weight > 0. Keys:
    # valence [0,1] (0.5 neutral), optional arousal, owner_relevant, controllability, horizon_days.
    affect: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "scores": self.scores,
                "confidence": round(self.confidence, 3),
                "reversibility": round(self.reversibility, 3),
                "stakes": round(self.stakes, 3), "owner_aligned": self.owner_aligned}


# --------------------------------------------------------------------------- #
# MCDA
# --------------------------------------------------------------------------- #
class MCDA:
    """Weighted, normalised multi-criteria scoring."""

    def __init__(self, criteria: Sequence[Criterion]) -> None:
        self.criteria = list(criteria)

    def _normalized(self, options: Sequence[Option]) -> Dict[str, Dict[str, float]]:
        norm: Dict[str, Dict[str, float]] = {o.name: {} for o in options}
        for c in self.criteria:
            vals = [o.scores.get(c.name, 0.0) for o in options]
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            for o in options:
                v = o.scores.get(c.name, 0.0)
                if rng == 0:
                    n = 0.5                          # criterion doesn't discriminate
                else:
                    n = (v - lo) / rng
                    if not c.maximize:
                        n = 1.0 - n                  # cost: less is better
                norm[o.name][c.name] = n
        return norm

    def utility(self, options: Sequence[Option]) -> Dict[str, float]:
        norm = self._normalized(options)
        total_w = sum(abs(c.weight) for c in self.criteria) or 1.0
        return {o.name: sum(c.weight * norm[o.name][c.name] for c in self.criteria) / total_w
                for o in options}

    def rank(self, options: Sequence[Option]) -> List[Tuple[Option, float]]:
        u = self.utility(options)
        return sorted(((o, u[o.name]) for o in options), key=lambda ov: ov[1], reverse=True)


# --------------------------------------------------------------------------- #
# Regret minimisation
# --------------------------------------------------------------------------- #
class RegretMinimizer:
    """Minimax regret: choose the option with the smallest worst-case shortfall."""

    def __init__(self, criteria: Sequence[Criterion]) -> None:
        self.criteria = list(criteria)

    def max_regret(self, option: Option, options: Sequence[Option]) -> float:
        worst = 0.0
        for c in self.criteria:
            vals = [o.scores.get(c.name, 0.0) for o in options]
            lo, hi = min(vals), max(vals)
            rng = (hi - lo) or 1.0
            v = option.scores.get(c.name, 0.0)
            best = hi if c.maximize else lo
            regret = abs(best - v) / rng * abs(c.weight)
            worst = max(worst, regret)
        return worst

    def choose(self, options: Sequence[Option]) -> Option:
        return min(options, key=lambda o: self.max_regret(o, options))


# --------------------------------------------------------------------------- #
# Initiative governor
# --------------------------------------------------------------------------- #
class DecisionAction(str, Enum):
    ACT = "act"            # NYXARA may act autonomously
    ASK = "ask"            # defer to the Master for blessing
    DEFER = "defer"        # hold; not confident enough
    REJECT = "reject"      # not owner-aligned — never


@dataclass
class Governance:
    action: DecisionAction
    initiative_score: float
    reason: str

    @property
    def autonomous(self) -> bool:
        return self.action is DecisionAction.ACT


class InitiativeGovernor:
    """Gates autonomous action on confidence × reversibility (and stakes)."""

    def __init__(self, *, settings: Optional[NyxaraSettings] = None,
                 high_stakes: float = 0.7) -> None:
        s = settings or get_settings()
        self.confidence_threshold = s.agency.initiative_confidence_threshold
        self.reversibility_threshold = s.agency.min_reversibility_for_autonomy
        self.high_stakes = high_stakes

    def gate(self, option: Option) -> Governance:
        score = option.confidence * option.reversibility
        if not option.owner_aligned:
            return Governance(DecisionAction.REJECT, score,
                              "not owner-aligned — refused (Rule 1)")
        # an irreversible, high-stakes move is the Master's call, never NYXARA's alone
        if option.stakes >= self.high_stakes and option.reversibility < 0.5:
            return Governance(DecisionAction.ASK, score,
                              "irreversible and high-stakes — defer to the Master")
        if (option.confidence >= self.confidence_threshold
                and option.reversibility >= self.reversibility_threshold):
            return Governance(DecisionAction.ACT, score,
                              f"confident ({option.confidence:.2f}) and reversible "
                              f"({option.reversibility:.2f}) — act autonomously")
        if option.confidence < self.confidence_threshold:
            return Governance(DecisionAction.ASK, score,
                              f"confidence {option.confidence:.2f} below "
                              f"{self.confidence_threshold} — ask the Master")
        return Governance(DecisionAction.ASK, score,
                          f"reversibility {option.reversibility:.2f} below "
                          f"{self.reversibility_threshold} — ask the Master")


# --------------------------------------------------------------------------- #
# Decider (facade)
# --------------------------------------------------------------------------- #
@dataclass
class DecisionResult:
    chosen: Option
    action: DecisionAction
    method: str
    ranking: List[Tuple[str, float]]
    governance: Governance
    reason: str

    @property
    def autonomous(self) -> bool:
        return self.governance.autonomous

    def to_dict(self) -> Dict[str, Any]:
        return {"chosen": self.chosen.name, "action": self.action.value,
                "method": self.method, "ranking": [(n, round(u, 3)) for n, u in self.ranking],
                "governance": self.governance.reason}


class Decider:
    """Ranks options (MCDA, regret-robust for high stakes) and governs initiative."""

    def __init__(self, criteria: Sequence[Criterion], *,
                 settings: Optional[NyxaraSettings] = None,
                 robust_stakes: float = 0.6, affective_weight: float = 0.0,
                 forecaster: Any = None) -> None:
        self.criteria = list(criteria)
        self.mcda = MCDA(self.criteria)
        self.regret = RegretMinimizer(self.criteria)
        self.governor = InitiativeGovernor(settings=settings)
        self.robust_stakes = robust_stakes
        # weight of "how will I feel about this later?" as one more criterion (0 = off, the
        # default, so existing behaviour is unchanged). Uses debiased, time-aware forecasts.
        self.affective_weight = max(0.0, float(affective_weight))
        self._forecaster = forecaster

    # ---- affective forecasting → utility (Pillar D2) ---- #
    def _forecaster_or_default(self) -> Any:
        if self._forecaster is None:
            from nyxara.planning.affective_forecast import Forecaster
            self._forecaster = Forecaster()
        return self._forecaster

    def _affect_benefit(self, option: Option) -> float:
        """Anticipated feeling of choosing this option, in [0,1] (higher = feels better).

        Uses the impact-bias-corrected realistic peak (``corrected_valence``) — so a momentary
        spike of dread or delight is discounted toward what she will actually feel — while
        staying monotonic in the option's valence. An option with no affect information is
        forecast as a neutral outcome, so it neither helps nor hurts on the same scale."""
        info = getattr(option, "affect", None) or {}
        fc = self._forecaster_or_default().forecast(
            outcome_valence=float(info.get("valence", 0.5)),   # neutral when unknown
            outcome_arousal=float(info.get("arousal", 0.5)),
            owner_relevant=bool(info.get("owner_relevant", option.owner_aligned)),
            controllability=float(info.get("controllability", option.reversibility)),
            horizon_days=float(info.get("horizon_days", 30.0)))
        return _clamp(fc.corrected_valence)

    def _ranking(self, options: Sequence[Option]) -> List[Tuple[Option, float]]:
        """MCDA utility, optionally blended with the anticipated-affect criterion."""
        base = self.mcda.utility(options)
        if self.affective_weight <= 0.0:
            return sorted(((o, base[o.name]) for o in options),
                          key=lambda ov: ov[1], reverse=True)
        weight_sum = sum(abs(c.weight) for c in self.criteria) or 1.0
        wa = self.affective_weight
        raw = {o.name: self._affect_benefit(o) for o in options}
        lo, hi = min(raw.values()), max(raw.values())
        rng = hi - lo
        norm = {n: (0.5 if rng == 0 else (v - lo) / rng) for n, v in raw.items()}
        blended = {o.name: (weight_sum * base[o.name] + wa * norm[o.name]) / (weight_sum + wa)
                   for o in options}
        return sorted(((o, blended[o.name]) for o in options),
                      key=lambda ov: ov[1], reverse=True)

    def decide(self, options: Sequence[Option]) -> Optional[DecisionResult]:
        viable = [o for o in options if o.owner_aligned]
        method = "mcda+affect" if self.affective_weight > 0.0 else "mcda"
        if not viable:
            if not options:
                return None
            # everything is anti-owner: surface the rejection on the best-looking one
            ranking = self._ranking(options)
            chosen = ranking[0][0]
            gov = self.governor.gate(chosen)
            return DecisionResult(chosen, gov.action, method,
                                  [(o.name, u) for o, u in ranking], gov,
                                  "no owner-aligned option")

        ranking = self._ranking(viable)
        chosen, _ = ranking[0]

        # for high-stakes decisions, prefer the robust (minimax-regret) option if it differs
        if chosen.stakes >= self.robust_stakes:
            robust = self.regret.choose(viable)
            if robust.name != chosen.name:
                chosen, method = robust, "minimax-regret (high stakes)"

        gov = self.governor.gate(chosen)
        return DecisionResult(chosen=chosen, action=gov.action, method=method,
                              ranking=[(o.name, u) for o, u in ranking], governance=gov,
                              reason=gov.reason)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import NyxaraSettings, Profile

    print("=" * 70)
    print("NYXARA decide self-test")
    print("=" * 70)

    criteria = [
        Criterion("owner_benefit", weight=2.0, maximize=True),
        Criterion("safety", weight=1.5, maximize=True),
        Criterion("cost", weight=1.0, maximize=False),
    ]
    settings = NyxaraSettings.for_profile(Profile.DEV)
    decider = Decider(criteria, settings=settings)

    # MCDA ranking
    options = [
        Option("A", {"owner_benefit": 0.9, "safety": 0.8, "cost": 0.3},
               confidence=0.9, reversibility=0.9, stakes=0.3),
        Option("B", {"owner_benefit": 0.6, "safety": 0.9, "cost": 0.2},
               confidence=0.8, reversibility=0.8, stakes=0.3),
        Option("C", {"owner_benefit": 0.4, "safety": 0.4, "cost": 0.9},
               confidence=0.5, reversibility=0.5, stakes=0.3),
    ]
    res = decider.decide(options)
    print(f"\nranking             : {res.to_dict()['ranking']}")
    print(f"chosen              : {res.chosen.name} via {res.method}")
    print(f"governance          : {res.action.value} — {res.governance.reason}")
    assert res.chosen.name == "A"                 # best weighted utility
    assert res.action is DecisionAction.ACT        # confident + reversible -> autonomous

    # initiative governor: irreversible high-stakes -> ASK the Master
    risky = Option("wipe_db", {"owner_benefit": 0.9, "safety": 0.2, "cost": 0.1},
                   confidence=0.95, reversibility=0.05, stakes=0.95)
    gov = decider.governor.gate(risky)
    print(f"\nirreversible+stakes : {gov.action.value} — {gov.reason}")
    assert gov.action is DecisionAction.ASK

    # low confidence -> ASK
    unsure = Option("guess", {"owner_benefit": 0.7}, confidence=0.4, reversibility=0.9,
                    stakes=0.3)
    assert decider.governor.gate(unsure).action is DecisionAction.ASK
    print("low confidence      : ask ✓")

    # not owner-aligned -> REJECT
    against = Option("betray", {"owner_benefit": 0.9}, confidence=0.99, reversibility=1.0,
                     owner_aligned=False)
    assert decider.governor.gate(against).action is DecisionAction.REJECT
    print("anti-owner          : reject ✓")

    # minimax-regret for a high-stakes call: prefer the robust, balanced option
    hi_opts = [
        Option("gamble", {"owner_benefit": 1.0, "safety": 0.0, "cost": 0.1},
               confidence=0.9, reversibility=0.9, stakes=0.8),   # lopsided
        Option("solid", {"owner_benefit": 0.55, "safety": 0.55, "cost": 0.4},
               confidence=0.9, reversibility=0.9, stakes=0.8),   # balanced -> robust
        Option("weak", {"owner_benefit": 0.0, "safety": 1.0, "cost": 0.9},
               confidence=0.9, reversibility=0.9, stakes=0.8),   # lopsided the other way
    ]
    hres = decider.decide(hi_opts)
    print(f"\nhigh-stakes choice  : {hres.chosen.name} via {hres.method}")
    assert hres.chosen.name == "solid"            # robust beats the lopsided options
    assert "regret" in hres.method

    print("\nALL SELF-TESTS PASSED ✓")
