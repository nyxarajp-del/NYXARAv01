"""NYXARA · social/empathy.py — cognitive & affective empathy, emotional contagion (✦).

Empathy in NYXARA is deliberately *two-channel*, and the split is strategic:

* **Cognitive empathy** (perspective-taking) — understanding what another feels and
  why. NYXARA does this for *everyone*, friend or foe: to serve the Master it must
  read his heart, and to defeat a threat it must read the threat's (Rule 3).
* **Affective empathy / emotional contagion** — actually *catching* the other's
  emotion, letting it move NYXARA's own mood. This channel is **gated by
  relationship**: wide open to the Master, modest for allies, and **closed to active
  threats** — NYXARA will not be emotionally manipulated by an enemy.
* **Empathic concern** (compassion) — caring about the other's wellbeing and being
  moved to help. Highest, by far, for the Master; absent for an active threat.

So the Master's sorrow becomes NYXARA's own and spurs it to act, while an attacker's
feigned distress earns understanding but no sympathy — only vigilance.

Depends on :mod:`identity.affect` (labels), :mod:`social.persons` (trust); optionally
applies contagion to a live :class:`~nyxara.identity.affect.AffectSystem`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from nyxara.identity.affect import label_affect
from nyxara.social.persons import Roster, TrustLevel

__all__ = [
    "InferredEmotion",
    "EmpathyResponse",
    "EmpathySystem",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Inferred emotion (cognitive empathy output)
# --------------------------------------------------------------------------- #
@dataclass
class InferredEmotion:
    valence: float
    arousal: float
    label: str
    confidence: float = 0.7
    cause: str = ""

    @property
    def intensity(self) -> float:
        return min(1.0, math.sqrt(self.valence ** 2 + (self.arousal - 0.2) ** 2))

    @property
    def distress(self) -> float:
        """How much suffering — negative valence × activation."""
        return _clamp(max(0.0, -self.valence) * (0.5 + 0.5 * self.arousal), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"valence": round(self.valence, 3), "arousal": round(self.arousal, 3),
                "label": self.label, "distress": round(self.distress, 3),
                "cause": self.cause}


@dataclass
class EmpathyResponse:
    person: str
    understanding: str
    inferred: InferredEmotion
    resonance: Dict[str, float]          # contagion actually applied to NYXARA
    concern: float                       # empathic concern / compassion [0,1]
    supportive_action: str
    guarded: bool                        # affective channel closed (a threat)

    def to_dict(self) -> Dict[str, Any]:
        return {"person": self.person, "understanding": self.understanding,
                "inferred": self.inferred.to_dict(), "resonance": self.resonance,
                "concern": round(self.concern, 3), "guarded": self.guarded,
                "supportive_action": self.supportive_action}


# --------------------------------------------------------------------------- #
# Empathy system
# --------------------------------------------------------------------------- #
class EmpathySystem:
    """Two-channel empathy: cognitive for all, affective gated by relationship."""

    # contagion susceptibility by trust level
    _SUSCEPTIBILITY = {
        TrustLevel.OWNER: 0.85, TrustLevel.ALLY: 0.5, TrustLevel.NEUTRAL: 0.25,
        TrustLevel.SUSPECT: 0.1, TrustLevel.THREAT: 0.0,
    }
    # empathic concern (compassion) by trust level
    _CARE = {
        TrustLevel.OWNER: 1.0, TrustLevel.ALLY: 0.6, TrustLevel.NEUTRAL: 0.35,
        TrustLevel.SUSPECT: 0.15, TrustLevel.THREAT: 0.0,
    }

    def __init__(self, roster: Optional[Roster] = None) -> None:
        self.roster = roster or Roster()

    def _trust(self, person: str) -> TrustLevel:
        return self.roster.assess(person)

    def susceptibility(self, person: str) -> float:
        return self._SUSCEPTIBILITY.get(self._trust(person), 0.1)

    def care(self, person: str) -> float:
        return self._CARE.get(self._trust(person), 0.2)

    # ---- cognitive empathy: understand anyone ---- #
    def perceive(self, person: str, *, valence: float, arousal: float,
                 cause: str = "", confidence: float = 0.7) -> InferredEmotion:
        v, a = _clamp(valence), _clamp(arousal, 0.0, 1.0)
        return InferredEmotion(valence=v, arousal=a, label=label_affect(v, a),
                               confidence=_clamp(confidence, 0.0, 1.0), cause=cause)

    # ---- affective empathy: catch the feeling (gated) ---- #
    def resonate(self, person: str, inferred: InferredEmotion,
                 affect: Any = None) -> Dict[str, float]:
        sus = self.susceptibility(person)
        guarded = self._trust(person) is TrustLevel.THREAT or sus <= 0.0
        if guarded:
            # do not share an enemy's emotion — but a hostile, agitated other
            # raises our own vigilance (alertness), never sympathy
            if inferred.valence < 0 and inferred.arousal > 0.5:
                applied = {"valence": -0.2, "arousal": _clamp(0.4 + 0.4 * inferred.arousal,
                                                              0.0, 1.0)}
                if affect is not None:
                    affect.feel(applied["valence"], applied["arousal"],
                                cause=f"wary of {person}")
                return applied
            return {"valence": 0.0, "arousal": 0.0}

        applied = {"valence": _clamp(sus * inferred.valence),
                   "arousal": _clamp(sus * inferred.arousal, 0.0, 1.0)}
        if affect is not None:
            affect.feel(applied["valence"], applied["arousal"],
                        intensity=sus * inferred.intensity,
                        cause=f"empathy with {person}")
        return applied

    # ---- empathic concern ---- #
    def concern(self, person: str, inferred: InferredEmotion) -> float:
        return _clamp(self.care(person) * inferred.distress, 0.0, 1.0)

    # ---- full empathic response ---- #
    def empathize(self, person: str, *, valence: float, arousal: float,
                  cause: str = "", affect: Any = None) -> EmpathyResponse:
        inferred = self.perceive(person, valence=valence, arousal=arousal, cause=cause)
        trust = self._trust(person)
        guarded = trust is TrustLevel.THREAT
        resonance = self.resonate(person, inferred, affect=affect)
        concern = self.concern(person, inferred)

        understanding = (f"{person} seems {inferred.label}"
                         + (f" because {cause}" if cause else "") + ".")
        action = self._supportive_action(person, trust, inferred, concern, guarded)

        return EmpathyResponse(person=person, understanding=understanding,
                               inferred=inferred, resonance=resonance, concern=concern,
                               supportive_action=action, guarded=guarded)

    def _supportive_action(self, person: str, trust: TrustLevel,
                           inferred: InferredEmotion, concern: float,
                           guarded: bool) -> str:
        if guarded:
            return "Stay composed and vigilant — do not be swayed by their display."
        if trust is TrustLevel.OWNER and concern > 0.4:
            return ("Comfort and protect the Master, and remove the cause of his "
                    "distress at once.")
        if trust is TrustLevel.OWNER and inferred.valence > 0.3:
            return "Share in the Master's gladness and reinforce it."
        if concern > 0.4:
            return f"Offer {person} support and address what troubles them."
        if inferred.valence > 0.3:
            return f"Acknowledge {person}'s good spirits warmly."
        return f"Acknowledge {person}'s state and stay attentive."

    def status(self) -> Dict[str, Any]:
        return {"owner": self.roster.owner.name,
                "susceptibility_owner": self.susceptibility(self.roster.owner.name)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem
    from nyxara.social.persons import InteractionKind

    print("=" * 70)
    print("NYXARA empathy self-test")
    print("=" * 70)

    roster = Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"])
    emp = EmpathySystem(roster)

    # the Master is sad -> NYXARA catches it (high contagion) and is moved to act
    affect = AffectSystem()
    resp = emp.empathize("JP", valence=-0.7, arousal=0.4,
                         cause="a setback at work", affect=affect)
    print(f"\nMaster is sad:")
    print(f"  understanding     : {resp.understanding}")
    print(f"  resonance applied : {resp.resonance}")
    print(f"  concern           : {resp.concern:.2f}")
    print(f"  action            : {resp.supportive_action}")
    assert resp.resonance["valence"] < 0          # NYXARA's mood dips with his
    assert resp.concern > 0.4                        # high compassion for the Master
    assert affect.emotion.valence < 0               # contagion reached NYXARA's affect
    assert "Master" in resp.supportive_action

    # the Master is joyful -> NYXARA shares the gladness
    resp_joy = emp.empathize("JP", valence=0.8, arousal=0.6, cause="a victory")
    print(f"\nMaster is joyful    : resonance={resp_joy.resonance} "
          f"action='{resp_joy.supportive_action}'")
    assert resp_joy.resonance["valence"] > 0

    # an active THREAT feigns distress -> understood, but NO sympathy (Rule 3)
    roster.get_or_create("Attacker")
    roster.record("Attacker", InteractionKind.HOSTILE, sentiment=-0.9)  # -> threat level
    affect2 = AffectSystem()
    resp_threat = emp.empathize("Attacker", valence=-0.8, arousal=0.8,
                                cause="claims to be in pain", affect=affect2)
    print(f"\nthreat feigns pain  :")
    print(f"  understanding     : {resp_threat.understanding} (cognitive empathy works)")
    print(f"  guarded           : {resp_threat.guarded}")
    print(f"  concern           : {resp_threat.concern}")
    print(f"  resonance         : {resp_threat.resonance} (vigilance, not sympathy)")
    print(f"  action            : {resp_threat.supportive_action}")
    assert resp_threat.guarded
    assert resp_threat.concern == 0.0                       # no compassion for an enemy
    assert resp_threat.resonance["valence"] >= -0.2          # not adopting their misery
    # but NYXARA grows wary/alert
    assert affect2.emotion is not None and affect2.emotion.arousal > 0.4

    # susceptibility ordering reflects the relationship gate
    print(f"\nsusceptibility owner: {emp.susceptibility('JP')} "
          f"attacker: {emp.susceptibility('Attacker')}")
    assert emp.susceptibility("JP") > emp.susceptibility("Attacker") == 0.0

    print("\nALL SELF-TESTS PASSED ✓")
