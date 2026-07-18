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
    # the causal-graph variable this option corresponds to ACTING ON (e.g. "exercise" for
    # an option "go to the gym") — read by the Decider's causal-necessity criterion when
    # built with causal_weight > 0 and a causal_model + causal_goal. None = no causal read.
    causal_label: Optional[str] = None

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
                 forecaster: Any = None, causal_model: Any = None,
                 causal_goal: Optional[str] = None, causal_weight: float = 0.0) -> None:
        self.criteria = list(criteria)
        self.mcda = MCDA(self.criteria)
        self.regret = RegretMinimizer(self.criteria)
        self.governor = InitiativeGovernor(settings=settings)
        self.robust_stakes = robust_stakes
        # weight of "how will I feel about this later?" as one more criterion (0 = off, the
        # default, so existing behaviour is unchanged). Uses debiased, time-aware forecasts.
        self.affective_weight = max(0.0, float(affective_weight))
        self._forecaster = forecaster
        # weight of "does acting on this option's causal_label REALLY move causal_goal?"
        # (0 = off, the default) — grounds option ranking in the learned causal graph's
        # Probability of Necessity/Sufficiency (mind.causal_world_model) instead of only
        # whatever correlational score the option happened to be given. An option whose
        # apparent benefit is really just correlation with past success (not a necessary
        # or sufficient cause of the goal) is no longer ranked as if it reliably delivers it.
        self.causal_model = causal_model
        self.causal_goal = causal_goal
        self.causal_weight = max(0.0, float(causal_weight))

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

    def _causal_benefit(self, option: Option) -> float:
        """How much does acting on this option's ``causal_label`` REALLY move
        ``causal_goal`` — Pearl's Probability of Necessity/Sufficiency (mean of PN, PS),
        not a bare correlational score. Neutral (0.5, neither helps nor hurts the
        ranking) whenever there's nothing to ground: no causal model/goal wired, the
        option names no causal_label, or the graph has no fitted mechanism between them
        yet — honest abstention, never a fabricated causal read."""
        if self.causal_model is None or self.causal_goal is None or option.causal_label is None:
            return 0.5
        try:
            result = self.causal_model.necessity_sufficiency(
                option.causal_label, self.causal_goal)
        except Exception:  # noqa: BLE001 — a causal read is a booster, never required
            result = None
        if not result:
            return 0.5
        return _clamp((result.get("PN", 0.0) + result.get("PS", 0.0)) / 2.0)

    def _ranking(self, options: Sequence[Option]) -> List[Tuple[Option, float]]:
        """MCDA utility, optionally blended with the anticipated-affect and/or
        causal-necessity criteria (each off by default — a ``Decider`` built without
        them ranks exactly as before)."""
        base = self.mcda.utility(options)
        weight_sum = sum(abs(c.weight) for c in self.criteria) or 1.0
        blended = {o.name: base[o.name] * weight_sum for o in options}
        total_w = weight_sum

        def _fold_in(weight: float, raw_fn: Any) -> None:
            nonlocal total_w
            if weight <= 0.0:
                return
            raw = {o.name: raw_fn(o) for o in options}
            lo, hi = min(raw.values()), max(raw.values())
            rng = hi - lo
            norm = {n: (0.5 if rng == 0 else (v - lo) / rng) for n, v in raw.items()}
            for o in options:
                blended[o.name] += weight * norm[o.name]
            total_w += weight

        _fold_in(self.affective_weight, self._affect_benefit)
        _fold_in(self.causal_weight, self._causal_benefit)

        result = {name: v / total_w for name, v in blended.items()}
        return sorted(((o, result[o.name]) for o in options), key=lambda ov: ov[1], reverse=True)

    def decide(self, options: Sequence[Option]) -> Optional[DecisionResult]:
        viable = [o for o in options if o.owner_aligned]
        extras = ("+affect" if self.affective_weight > 0.0 else "") + \
                 ("+causal" if self.causal_weight > 0.0 else "")
        method = "mcda" + extras
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

    # causal grounding: prefer the option that REALLY moves the goal (high PN/PS) over
    # one that merely correlates with past success — the whole point of wiring
    # mind.causal_world_model's necessity/sufficiency into option ranking (Rung 3, not
    # just "looked good historically"). "lucky_day" only ever co-occurs with sales via a
    # confound (weekend); "discount" is a real do-experiment that reliably moves sales.
    import random as _random
    from nyxara.mind.causal_world_model import CausalWorldModel
    cwm = CausalWorldModel(window=10.0)
    rng2 = _random.Random(21)
    for k in range(400):
        base = k * 100.0
        weekend = rng2.random() < 0.3
        if weekend:
            cwm.observe("weekend", at=base, value=1.0)
        if weekend:                                          # lucky_day ~ proxy for weekend
            cwm.observe("lucky_day", at=base + 1, value=1.0)
        do_discount = rng2.random() < 0.4                     # a real do-experiment
        cwm.observe("discount", at=base + 2, value=1.0 if do_discount else 0.0,
                   intervention=do_discount)
        sales = (do_discount and rng2.random() < 0.85) or (weekend and rng2.random() < 0.7)
        cwm.observe("sales_up", at=base + 3, value=1.0 if sales else 0.0)
    cwm.discover()

    goal_options = [
        Option("run_discount", {"owner_benefit": 0.6}, confidence=0.8, reversibility=0.9,
               stakes=0.3, causal_label="discount"),
        Option("wait_for_luck", {"owner_benefit": 0.8}, confidence=0.8, reversibility=0.9,
               stakes=0.3, causal_label="lucky_day"),
    ]
    naive = Decider([Criterion("owner_benefit", weight=2.0)]).decide(goal_options)
    print(f"\nwithout causal read : {naive.chosen.name} "
          f"(higher owner_benefit score, but only correlates with sales)")
    assert naive.chosen.name == "wait_for_luck"

    grounded = Decider([Criterion("owner_benefit", weight=2.0)], causal_model=cwm,
                       causal_goal="sales_up", causal_weight=3.0).decide(goal_options)
    print(f"with causal read     : {grounded.chosen.name} "
          f"(lower raw score, but genuinely necessary/sufficient for the goal)")
    assert grounded.chosen.name == "run_discount"

    print("\nALL SELF-TESTS PASSED ✓")
