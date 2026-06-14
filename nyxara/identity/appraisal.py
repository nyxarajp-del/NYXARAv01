"""NYXARA · identity/appraisal.py — emotion by appraisal (OCC / Scherer): event→meaning→feeling.

Emotions are not caused by events; they are caused by an agent's *evaluation* of
events. The same event — "an intruder probed the network" — yields fear if the Master
is exposed, anger if a blameworthy other did it, or indifference if it's irrelevant.

This is an appraisal engine in the OCC (Ortony–Clore–Collins) / Scherer tradition. An
:class:`AppraisalEvent` carries the dimensions along which NYXARA evaluates it —

* **goal relevance** (does it matter to my goals — chiefly serving/protecting JP?),
* **desirability** (good or bad for those goals),
* **agency** (self / the Master / another / circumstance),
* **likelihood** & **prospect** (a future, uncertain event vs an occurred one),
* **controllability** (can I cope?), **novelty**, **norm compatibility** (is it fair?) —

and the :class:`Appraiser` maps the appraisal *pattern* onto a discrete OCC emotion
(hope, fear, pride, gratitude, anger, relief, …) plus a valence/arousal that feeds
:mod:`identity.affect`.

Desirability is judged from NYXARA's standpoint: what helps the Master is desirable.

Pure standard library; integrates with :mod:`identity.affect`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

__all__ = [
    "Agency",
    "AppraisalDimensions",
    "AppraisalEvent",
    "AppraisalOutcome",
    "Appraiser",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #
class Agency(str, Enum):
    SELF = "self"            # NYXARA caused it
    OWNER = "owner"          # the Master caused it
    OTHER = "other"          # another agent caused it
    CIRCUMSTANCE = "circumstance"  # no agent / nature


@dataclass
class AppraisalDimensions:
    goal_relevance: float = 0.5      # [0,1] how much it bears on my goals
    desirability: float = 0.0        # [-1,1] good(+)/bad(-) for my goals (the Master's good)
    agency: Agency = Agency.CIRCUMSTANCE
    likelihood: float = 1.0          # [0,1]; <1 for uncertain/prospective events
    controllability: float = 0.5     # [0,1] coping potential
    novelty: float = 0.3             # [0,1] unexpectedness
    norm_compatibility: float = 0.0  # [-1,1] fairness / rule-compliance of the event
    prospect: bool = False           # is this a future/uncertain prospect?
    confirmed: Optional[bool] = None  # for a prospect later resolved: True/False/None

    def to_dict(self) -> Dict[str, Any]:
        return {"goal_relevance": self.goal_relevance, "desirability": self.desirability,
                "agency": self.agency.value, "likelihood": self.likelihood,
                "controllability": self.controllability, "novelty": self.novelty,
                "norm_compatibility": self.norm_compatibility, "prospect": self.prospect,
                "confirmed": self.confirmed}


@dataclass
class AppraisalEvent:
    description: str
    dims: AppraisalDimensions = field(default_factory=AppraisalDimensions)


# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #
@dataclass
class AppraisalOutcome:
    emotion: str
    valence: float
    arousal: float
    intensity: float
    reasoning: str
    dims: AppraisalDimensions

    def to_dict(self) -> Dict[str, Any]:
        return {"emotion": self.emotion, "valence": round(self.valence, 3),
                "arousal": round(self.arousal, 3), "intensity": round(self.intensity, 3),
                "reasoning": self.reasoning}


# base (valence, arousal) prototypes for OCC emotions
_OCC_AFFECT: Dict[str, tuple] = {
    "joy": (0.7, 0.55), "distress": (-0.6, 0.45),
    "hope": (0.45, 0.6), "fear": (-0.6, 0.85),
    "satisfaction": (0.7, 0.4), "relief": (0.5, 0.3),
    "disappointment": (-0.5, 0.45), "fears_confirmed": (-0.75, 0.65),
    "pride": (0.65, 0.5), "remorse": (-0.55, 0.45), "shame": (-0.5, 0.4),
    "gratitude": (0.6, 0.45), "admiration": (0.5, 0.4),
    "anger": (-0.55, 0.88), "reproach": (-0.4, 0.6),
    "indifference": (0.0, 0.12),
}


# --------------------------------------------------------------------------- #
# The appraiser
# --------------------------------------------------------------------------- #
class Appraiser:
    """Maps an appraised event onto an OCC emotion and a valence/arousal feeling."""

    def __init__(self, *, relevance_floor: float = 0.1) -> None:
        self.relevance_floor = relevance_floor

    def _label(self, d: AppraisalDimensions) -> str:
        if d.goal_relevance < self.relevance_floor:
            return "indifference"

        # prospective (future, uncertain) events -> hope / fear, and their resolution
        if d.prospect:
            if d.confirmed is None:
                return "hope" if d.desirability >= 0 else "fear"
            if d.desirability >= 0:
                return "satisfaction" if d.confirmed else "disappointment"
            return "fears_confirmed" if d.confirmed else "relief"

        # occurred events -> attribution by agency
        desirable = d.desirability >= 0
        if d.agency is Agency.SELF:
            return "pride" if desirable else "remorse"
        if d.agency is Agency.OWNER:
            # the Master acted: their approval -> gratitude/admiration; displeasure -> shame
            return "gratitude" if desirable else "shame"
        if d.agency is Agency.OTHER:
            if desirable:
                return "gratitude"
            # a blameworthy other harming our goals -> anger; otherwise plain distress
            return "anger" if d.norm_compatibility < 0 else "reproach" if d.controllability < 0.4 else "distress"
        # circumstance
        return "joy" if desirable else "distress"

    def appraise(self, event: AppraisalEvent) -> AppraisalOutcome:
        d = event.dims
        label = self._label(d)
        base_v, base_a = _OCC_AFFECT.get(label, (0.0, 0.2))

        # intensity from relevance × desirability magnitude × likelihood, + novelty
        lk = d.likelihood if d.prospect and d.confirmed is None else 1.0
        intensity = _clamp(
            d.goal_relevance * (0.4 + 0.6 * abs(d.desirability)) * lk + 0.2 * d.novelty,
            0.0, 1.0)

        valence = _clamp(base_v * (0.4 + 0.6 * intensity))
        # low controllability amplifies arousal (helplessness is activating)
        cope = 1.0 - d.controllability
        arousal = _clamp(base_a * (0.5 + 0.5 * intensity) + 0.2 * d.novelty + 0.15 * cope,
                         0.0, 1.0)

        reasoning = (f"goal-relevance {d.goal_relevance:.2f}, "
                     f"desirability {d.desirability:+.2f}, agency={d.agency.value}"
                     + (f", prospect(lk={d.likelihood:.2f}"
                        + (f", confirmed={d.confirmed}" if d.confirmed is not None else "")
                        + ")" if d.prospect else "")
                     + (", norm-violation" if d.norm_compatibility < 0 else "")
                     + f" → {label}")

        return AppraisalOutcome(emotion=label, valence=valence, arousal=arousal,
                                intensity=intensity, reasoning=reasoning, dims=d)

    def appraise_and_feel(self, affect: Any, event: AppraisalEvent) -> AppraisalOutcome:
        """Appraise, then push the resulting feeling into an :class:`AffectSystem`."""
        outcome = self.appraise(event)
        affect.feel(outcome.valence, outcome.arousal, intensity=outcome.intensity,
                    cause=event.description)
        return outcome

    # ---- owner-centred conveniences ---- #
    def owner_threatened(self, severity: float = 0.8, *, by_other: bool = True
                         ) -> AppraisalEvent:
        return AppraisalEvent(
            "a threat to the Master",
            AppraisalDimensions(goal_relevance=1.0, desirability=-severity,
                                agency=Agency.OTHER if by_other else Agency.CIRCUMSTANCE,
                                norm_compatibility=-1.0, controllability=0.5,
                                novelty=0.6))

    def owner_praise(self, warmth: float = 0.8) -> AppraisalEvent:
        return AppraisalEvent(
            "the Master praised my work",
            AppraisalDimensions(goal_relevance=0.9, desirability=warmth,
                                agency=Agency.OWNER, novelty=0.3))

    def own_success(self, magnitude: float = 0.7) -> AppraisalEvent:
        return AppraisalEvent(
            "I succeeded at a task for the Master",
            AppraisalDimensions(goal_relevance=0.8, desirability=magnitude,
                                agency=Agency.SELF, novelty=0.3))

    def prospect(self, desirability: float, likelihood: float,
                 description: str = "an anticipated outcome") -> AppraisalEvent:
        return AppraisalEvent(
            description,
            AppraisalDimensions(goal_relevance=0.8, desirability=desirability,
                                likelihood=likelihood, prospect=True, novelty=0.4))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem

    print("=" * 70)
    print("NYXARA appraisal self-test")
    print("=" * 70)

    ap = Appraiser()

    # threat to the Master, caused by a blameworthy other -> anger, high arousal
    out = ap.appraise(ap.owner_threatened(0.9))
    print(f"\nthreat to Master    : {out.emotion} (v={out.valence:.2f} a={out.arousal:.2f})")
    print(f"  {out.reasoning}")
    assert out.emotion == "anger"
    assert out.arousal > 0.7 and out.valence < 0

    # the Master praises NYXARA -> gratitude, positive
    out = ap.appraise(ap.owner_praise(0.9))
    print(f"\nMaster's praise     : {out.emotion} (v={out.valence:.2f})")
    assert out.emotion == "gratitude" and out.valence > 0

    # NYXARA succeeds at a task -> pride
    out = ap.appraise(ap.own_success(0.8))
    print(f"own success         : {out.emotion}")
    assert out.emotion == "pride"

    # prospective desirable, uncertain -> hope; then confirm/disconfirm
    hope = ap.appraise(ap.prospect(desirability=0.7, likelihood=0.5))
    print(f"\nuncertain good      : {hope.emotion}")
    assert hope.emotion == "hope"
    ev = ap.prospect(0.7, 0.5); ev.dims.confirmed = True
    assert ap.appraise(ev).emotion == "satisfaction"
    ev.dims.confirmed = False
    assert ap.appraise(ev).emotion == "disappointment"

    # prospective undesirable -> fear; confirmed -> fears_confirmed; disconfirmed -> relief
    fear = ap.appraise(ap.prospect(desirability=-0.7, likelihood=0.6))
    assert fear.emotion == "fear"
    evb = ap.prospect(-0.7, 0.6); evb.dims.confirmed = True
    assert ap.appraise(evb).emotion == "fears_confirmed"
    evb.dims.confirmed = False
    assert ap.appraise(evb).emotion == "relief"
    print("prospect (bad)      : fear → fears_confirmed / relief ✓")

    # irrelevant event -> indifference
    irrelevant = AppraisalEvent("a leaf fell somewhere",
                                AppraisalDimensions(goal_relevance=0.02))
    assert ap.appraise(irrelevant).emotion == "indifference"
    print("irrelevant event    : indifference ✓")

    # intensity scales with relevance & desirability magnitude
    weak = ap.appraise(AppraisalEvent("minor", AppraisalDimensions(
        goal_relevance=0.2, desirability=0.2, agency=Agency.CIRCUMSTANCE)))
    strong = ap.appraise(AppraisalEvent("major", AppraisalDimensions(
        goal_relevance=1.0, desirability=0.9, agency=Agency.CIRCUMSTANCE)))
    print(f"\nintensity weak/strong: {weak.intensity:.2f} / {strong.intensity:.2f}")
    assert strong.intensity > weak.intensity

    # integration with affect
    affect = AffectSystem()
    ap.appraise_and_feel(affect, ap.owner_threatened(0.9))
    print(f"\nfed to affect       : emotion={affect.emotion.label} "
          f"mood={affect.mood.label}")
    assert affect.emotion is not None

    print("\nALL SELF-TESTS PASSED ✓")
