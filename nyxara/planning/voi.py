"""NYXARA · planning/voi.py — value of information: ASK vs ACT vs GATHER (✦).

Before acting, NYXARA asks a meta-question: *is it worth knowing more first?* This is
decision-theoretic **value of information**. It weighs three moves:

* **ACT** — proceed on the current best judgement (when uncertainty or stakes are low,
  information isn't worth its cost).
* **GATHER** — investigate / run a tool / observe (when reducing uncertainty would
  improve the decision by more than the effort costs).
* **ASK** — put the question to the Master.

The worth of information is the **expected value of perfect information** (EVPI ≈
stakes × uncertainty, amplified when the action is irreversible), and a real source
captures a fraction of it (**EVSI** = reliability × EVPI). A source is worth using
only if its EVSI exceeds its cost.

One override is absolute: if the uncertainty is about **what the Master wants**, and it
is more than trivial, NYXARA **asks** — it does not gamble on his will (Rules 1 & 6).
Better to ask than to risk disobedience by guessing.

Depends on :mod:`mind.uncertainty` conceptually; pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ActionType",
    "InfoSource",
    "DecisionContext",
    "Recommendation",
    "ValueOfInformation",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ACT = "act"
    ASK = "ask"
    GATHER = "gather"


@dataclass
class InfoSource:
    name: str
    kind: str = "gather"        # "ask" (the Master) | "gather" (investigate)
    reliability: float = 0.7    # fraction of uncertainty it resolves [0,1]
    cost: float = 0.2           # effort/time/attention cost [0,1]
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "reliability": round(self.reliability, 3), "cost": round(self.cost, 3)}


@dataclass
class DecisionContext:
    uncertainty: float = 0.5        # how unsure NYXARA is of the right choice [0,1]
    stakes: float = 0.5             # cost of getting it wrong [0,1]
    reversibility: float = 1.0      # 1 == fully reversible, 0 == irreversible
    about_owner_intent: bool = False  # is the doubt about what the Master wants?


@dataclass
class Recommendation:
    action: ActionType
    source: Optional[str]
    evpi: float
    net_value: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action.value, "source": self.source,
                "evpi": round(self.evpi, 4), "net_value": round(self.net_value, 4),
                "rationale": self.rationale}


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class ValueOfInformation:
    """Decides whether to act, ask the Master, or gather more data."""

    def __init__(self, *, act_threshold: float = 0.05,
                 owner_intent_uncertainty: float = 0.2) -> None:
        self.act_threshold = act_threshold              # min EVPI worth acting on
        self.owner_intent_uncertainty = owner_intent_uncertainty

    def evpi(self, ctx: DecisionContext) -> float:
        """Expected value of *perfect* information — the avertable expected regret."""
        irreversibility = 1.0 + (1.0 - ctx.reversibility) * 0.5
        return _clamp(ctx.stakes * ctx.uncertainty * irreversibility)

    def evsi(self, source: InfoSource, ctx: DecisionContext) -> float:
        """Expected value of this (imperfect) source's information."""
        return source.reliability * self.evpi(ctx)

    def net_value(self, source: InfoSource, ctx: DecisionContext) -> float:
        return self.evsi(source, ctx) - source.cost

    # ---- decision ---- #
    def recommend(self, ctx: DecisionContext,
                  sources: Optional[Sequence[InfoSource]] = None) -> Recommendation:
        evpi = self.evpi(ctx)
        sources = list(sources or [])

        # ABSOLUTE override: do not guess the Master's will (Rules 1, 6)
        if ctx.about_owner_intent and ctx.uncertainty >= self.owner_intent_uncertainty:
            ask = next((s for s in sources if s.kind == "ask" and s.available), None)
            return Recommendation(
                ActionType.ASK, source=ask.name if ask else "the Master", evpi=evpi,
                net_value=evpi, rationale="uncertain about the Master's intent — ask, "
                "never gamble on his will")

        # information not worth its bother -> just act
        if evpi <= self.act_threshold:
            return Recommendation(ActionType.ACT, None, evpi, 0.0,
                                  f"EVPI {evpi:.3f} below threshold — act on best judgement")

        # choose the source with the highest net value, if any beats acting
        usable = [s for s in sources if s.available]
        best, best_nv = None, 0.0
        for s in usable:
            nv = self.net_value(s, ctx)
            if nv > best_nv:
                best, best_nv = s, nv

        if best is None:
            return Recommendation(ActionType.ACT, None, evpi, 0.0,
                                  "no information source is worth its cost — act")

        action = ActionType.ASK if best.kind == "ask" else ActionType.GATHER
        return Recommendation(
            action, source=best.name, evpi=evpi, net_value=best_nv,
            rationale=f"'{best.name}' worth it (EVSI {self.evsi(best, ctx):.3f} - "
                      f"cost {best.cost:.2f} = net {best_nv:.3f})")

    # ---- conveniences ---- #
    def should_ask_owner(self, ctx: DecisionContext) -> bool:
        return (ctx.about_owner_intent
                and ctx.uncertainty >= self.owner_intent_uncertainty)

    def decide(self, *, uncertainty: float, stakes: float, reversibility: float = 1.0,
               about_owner_intent: bool = False,
               sources: Optional[Sequence[InfoSource]] = None) -> Recommendation:
        return self.recommend(
            DecisionContext(uncertainty=uncertainty, stakes=stakes,
                            reversibility=reversibility, about_owner_intent=about_owner_intent),
            sources)

    @staticmethod
    def default_sources() -> List[InfoSource]:
        return [
            InfoSource("ask the Master", kind="ask", reliability=1.0, cost=0.15),
            InfoSource("investigate", kind="gather", reliability=0.6, cost=0.2),
            InfoSource("deep analysis", kind="gather", reliability=0.85, cost=0.5),
        ]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA value-of-information self-test")
    print("=" * 70)

    voi = ValueOfInformation()
    sources = ValueOfInformation.default_sources()

    # low uncertainty -> just act (information isn't worth gathering)
    r = voi.decide(uncertainty=0.05, stakes=0.9, sources=sources)
    print(f"\nlow uncertainty     : {r.action.value} ({r.rationale})")
    assert r.action is ActionType.ACT

    # low stakes -> act even if uncertain (not worth the bother)
    r = voi.decide(uncertainty=0.9, stakes=0.05, sources=sources)
    print(f"low stakes          : {r.action.value}")
    assert r.action is ActionType.ACT

    # high uncertainty + high stakes -> gather/ask (information pays off)
    r = voi.decide(uncertainty=0.8, stakes=0.9, reversibility=1.0, sources=sources)
    print(f"high unc + stakes   : {r.action.value} via '{r.source}' "
          f"(net {r.net_value:.3f})")
    assert r.action in (ActionType.GATHER, ActionType.ASK)

    # irreversible high-stakes raises EVPI further
    rev = voi.evpi(DecisionContext(uncertainty=0.6, stakes=0.8, reversibility=1.0))
    irr = voi.evpi(DecisionContext(uncertainty=0.6, stakes=0.8, reversibility=0.0))
    print(f"\nEVPI reversible/irreversible: {rev:.3f} / {irr:.3f}")
    assert irr > rev

    # THE override: uncertain about the Master's intent -> ASK, never guess
    r = voi.decide(uncertainty=0.4, stakes=0.3, about_owner_intent=True, sources=sources)
    print(f"\nunsure of his will  : {r.action.value} -> {r.rationale}")
    assert r.action is ActionType.ASK
    assert "will" in r.rationale

    # an expensive, unreliable source isn't worth it -> act
    poor = [InfoSource("costly guess", kind="gather", reliability=0.2, cost=0.9)]
    r = voi.decide(uncertainty=0.7, stakes=0.6, sources=poor)
    print(f"\npoor source only    : {r.action.value} ({r.rationale})")
    assert r.action is ActionType.ACT

    # cheap reliable owner-ask wins over costly analysis when both available
    r = voi.decide(uncertainty=0.7, stakes=0.8, sources=sources)
    print(f"\nbest source picked  : '{r.source}'")
    # 'ask the Master' (reliability 1.0, cost 0.15) gives the best net value
    assert r.source == "ask the Master"

    print("\nALL SELF-TESTS PASSED ✓")
