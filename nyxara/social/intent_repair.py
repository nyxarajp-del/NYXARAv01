"""NYXARA · social/intent_repair.py — detect misunderstanding → repair gracefully (✦).

Conversations break, and the skilled move is not to barrel on but to *repair*
(Schegloff's repair organisation). NYXARA watches for trouble and fixes it cleanly:

Troubles it detects
* **non-understanding** — the other signals confusion ("huh?", "I don't follow").
* **misunderstanding** — the other corrects NYXARA ("no, I meant…", "that's not…").
* **ambiguity** — NYXARA itself sees several plausible interpretations.
* **low confidence** — NYXARA is unsure it read the intent correctly.
* **contradiction** — the new input conflicts with established common ground.

Repair moves it chooses
* **specification** — "Did you mean A or B?"
* **confirmation** — "Just to confirm, you'd like me to …?"
* **paraphrase** — restate more plainly.
* **apology + retry** — "I misread that, forgive me — let me try again."

A loop-guard escalates after repeated failed repairs (rather than spiralling), and —
because the other party is usually the Master — repairs are deferential. Rule-based
and deterministic; an LLM can refine phrasing later.

Pure standard library; pairs with :mod:`social.dialogue` and :mod:`social.common_ground`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "TroubleType",
    "RepairType",
    "TroubleSignal",
    "RepairAction",
    "RepairDetector",
    "RepairStrategist",
    "RepairManager",
]


# --------------------------------------------------------------------------- #
# Troubles & repairs
# --------------------------------------------------------------------------- #
class TroubleType(str, Enum):
    NON_UNDERSTANDING = "non_understanding"   # they didn't get NYXARA
    MISUNDERSTANDING = "misunderstanding"     # NYXARA got them wrong (they corrected)
    AMBIGUITY = "ambiguity"                   # multiple readings
    LOW_CONFIDENCE = "low_confidence"         # unsure of interpretation
    CONTRADICTION = "contradiction"           # conflicts with common ground


class RepairType(str, Enum):
    NONE = "none"
    SPECIFICATION = "specification"
    CONFIRMATION = "confirmation"
    PARAPHRASE = "paraphrase"
    APOLOGY_RETRY = "apology_retry"
    ESCALATE = "escalate"


@dataclass
class TroubleSignal:
    type: TroubleType
    evidence: str
    severity: float = 0.5
    from_other: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "evidence": self.evidence,
                "severity": round(self.severity, 3), "from_other": self.from_other}


@dataclass
class RepairAction:
    type: RepairType
    utterance: str
    targets: List[str] = field(default_factory=list)
    addresses: Optional[TroubleType] = None

    @property
    def needed(self) -> bool:
        return self.type is not RepairType.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "utterance": self.utterance,
                "addresses": self.addresses.value if self.addresses else None}


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class RepairDetector:
    """Spots conversational trouble from text + interpretation state."""

    _CONFUSION = [
        r"\bhuh\b", r"\bwhat\?", r"^\s*what\s*$", r"\bsorry\?", r"\bpardon\b",
        r"\bcome again\b", r"\bi don'?t (understand|get|follow)\b",
        r"\bwhat do you mean\b", r"\bnot sure i follow\b", r"\bconfused\b",
        r"\bmakes no sense\b",
    ]
    _CORRECTION = [
        r"\bno,? i mean\b", r"\bthat'?s not what i\b", r"\bi didn'?t (say|mean)\b",
        r"\bi meant\b", r"\bno,? i said\b", r"\bnot that\b", r"\bactually,? i\b",
        r"\byou misunderstood\b", r"\bthat'?s wrong\b", r"\bnot quite\b",
    ]

    def __init__(self, *, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str = "", *, interpretation_confidence: float = 1.0,
               candidate_interpretations: Optional[Sequence[str]] = None,
               contradicts_ground: bool = False,
               from_other: bool = True) -> List[TroubleSignal]:
        signals: List[TroubleSignal] = []
        t = text.lower().strip()

        if from_other and t:
            for pat in self._CORRECTION:
                if re.search(pat, t):
                    signals.append(TroubleSignal(TroubleType.MISUNDERSTANDING,
                                                 f"correction: {pat}", 0.8))
                    break
            for pat in self._CONFUSION:
                if re.search(pat, t):
                    signals.append(TroubleSignal(TroubleType.NON_UNDERSTANDING,
                                                 f"confusion: {pat}", 0.7))
                    break

        if candidate_interpretations and len(candidate_interpretations) >= 2:
            signals.append(TroubleSignal(TroubleType.AMBIGUITY,
                                         f"{len(candidate_interpretations)} readings",
                                         0.6, from_other=False))

        if interpretation_confidence < self.confidence_threshold:
            signals.append(TroubleSignal(
                TroubleType.LOW_CONFIDENCE,
                f"confidence {interpretation_confidence:.2f} < {self.confidence_threshold}",
                0.5, from_other=False))

        if contradicts_ground:
            signals.append(TroubleSignal(TroubleType.CONTRADICTION,
                                         "conflicts with common ground", 0.7,
                                         from_other=False))
        return signals


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
class RepairStrategist:
    """Chooses a graceful repair move for the detected troubles."""

    # higher = handled first
    _PRIORITY = {
        TroubleType.MISUNDERSTANDING: 5,
        TroubleType.CONTRADICTION: 4,
        TroubleType.AMBIGUITY: 3,
        TroubleType.NON_UNDERSTANDING: 2,
        TroubleType.LOW_CONFIDENCE: 1,
    }

    def __init__(self, *, addressee: str = "you", deferential: bool = True) -> None:
        self.addressee = addressee
        self.deferential = deferential

    def _lead(self) -> str:
        return "Forgive me — " if self.deferential else ""

    def choose(self, signals: Sequence[TroubleSignal], *,
               candidate_interpretations: Optional[Sequence[str]] = None,
               interpretation: str = "", topic: str = "",
               ground_conflict: str = "") -> RepairAction:
        if not signals:
            return RepairAction(RepairType.NONE, "")

        top = max(signals, key=lambda s: self._PRIORITY.get(s.type, 0))

        if top.type is TroubleType.MISUNDERSTANDING:
            text = (f"{self._lead()}I misread that. Could you tell me again what you'd "
                    f"like{(' regarding ' + topic) if topic else ''}?")
            return RepairAction(RepairType.APOLOGY_RETRY, text, addresses=top.type)

        if top.type is TroubleType.CONTRADICTION:
            text = ("That seems to conflict with what we'd established"
                    + (f" ({ground_conflict})" if ground_conflict else "")
                    + ". Could you clarify which holds?")
            return RepairAction(RepairType.CONFIRMATION, text, addresses=top.type)

        if top.type is TroubleType.AMBIGUITY and candidate_interpretations:
            opts = list(candidate_interpretations)[:3]
            joined = " or ".join(f"'{o}'" for o in opts)
            return RepairAction(RepairType.SPECIFICATION,
                                f"Did you mean {joined}?", targets=opts, addresses=top.type)

        if top.type is TroubleType.NON_UNDERSTANDING:
            text = ("Let me put it more plainly"
                    + (f": {interpretation}" if interpretation else ".")
                    + ("" if interpretation else " What part shall I clarify?"))
            return RepairAction(RepairType.PARAPHRASE, text, addresses=top.type)

        # LOW_CONFIDENCE
        text = (f"Just to confirm — you'd like me to "
                f"{interpretation or 'proceed'}{'?' if interpretation else ', yes?'}")
        return RepairAction(RepairType.CONFIRMATION, text, addresses=top.type)


# --------------------------------------------------------------------------- #
# Manager (detect + repair + loop guard)
# --------------------------------------------------------------------------- #
class RepairManager:
    """Detects trouble, repairs gracefully, and escalates if repair keeps failing."""

    def __init__(self, *, max_attempts: int = 3, addressee: str = "you",
                 deferential: bool = True) -> None:
        self.detector = RepairDetector()
        self.strategist = RepairStrategist(addressee=addressee, deferential=deferential)
        self.max_attempts = max_attempts
        self.attempts = 0
        self.history: List[RepairAction] = []

    def handle(self, text: str = "", *, interpretation: str = "",
               candidate_interpretations: Optional[Sequence[str]] = None,
               interpretation_confidence: float = 1.0,
               contradicts_ground: bool = False, ground_conflict: str = "",
               topic: str = "", from_other: bool = True) -> RepairAction:
        signals = self.detector.detect(
            text, interpretation_confidence=interpretation_confidence,
            candidate_interpretations=candidate_interpretations,
            contradicts_ground=contradicts_ground, from_other=from_other)

        if not signals:
            self.attempts = 0           # back on track — reset the loop guard
            action = RepairAction(RepairType.NONE, "")
            return action

        self.attempts += 1
        if self.attempts > self.max_attempts:
            action = RepairAction(
                RepairType.ESCALATE,
                "We seem to keep missing each other. Could you state it plainly, "
                "step by step, so I serve you exactly right?")
            self.history.append(action)
            return action

        action = self.strategist.choose(
            signals, candidate_interpretations=candidate_interpretations,
            interpretation=interpretation, topic=topic, ground_conflict=ground_conflict)
        self.history.append(action)
        return action

    def resolved(self) -> None:
        """Mark the trouble as resolved (resets the loop guard)."""
        self.attempts = 0

    def status(self) -> Dict[str, Any]:
        return {"attempts": self.attempts, "repairs": len(self.history),
                "last": self.history[-1].to_dict() if self.history else None}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA intent-repair self-test")
    print("=" * 70)

    det = RepairDetector()
    strat = RepairStrategist(addressee="JP")

    # the Master signals confusion -> NYXARA paraphrases
    sig = det.detect("Huh? I don't follow.")
    print(f"\nconfusion detected  : {[s.type.value for s in sig]}")
    assert any(s.type is TroubleType.NON_UNDERSTANDING for s in sig)
    rep = strat.choose(sig, interpretation="I'll harden the firewall first")
    print(f"  repair            : [{rep.type.value}] {rep.utterance}")
    assert rep.type is RepairType.PARAPHRASE

    # the Master corrects NYXARA -> apology + retry
    sig2 = det.detect("No, that's not what I meant.")
    rep2 = strat.choose(sig2, topic="the deployment")
    print(f"\ncorrection          : [{rep2.type.value}] {rep2.utterance}")
    assert rep2.type is RepairType.APOLOGY_RETRY

    # NYXARA sees ambiguity -> ask which reading
    sig3 = det.detect("Run it.", candidate_interpretations=["the backup", "the scan"])
    rep3 = strat.choose(sig3, candidate_interpretations=["the backup", "the scan"])
    print(f"\nambiguity           : [{rep3.type.value}] {rep3.utterance}")
    assert rep3.type is RepairType.SPECIFICATION
    assert "backup" in rep3.utterance and "scan" in rep3.utterance

    # low confidence -> confirmation check
    sig4 = det.detect("Do the thing.", interpretation_confidence=0.3)
    rep4 = strat.choose(sig4, interpretation="restart the service")
    print(f"\nlow confidence      : [{rep4.type.value}] {rep4.utterance}")
    assert rep4.type is RepairType.CONFIRMATION

    # no trouble -> no repair
    assert strat.choose(det.detect("Yes, exactly right.")).type is RepairType.NONE
    print("\nclear input         : no repair needed ✓")

    # loop guard: repeated trouble escalates
    mgr = RepairManager(max_attempts=2, addressee="JP")
    a1 = mgr.handle("Huh?")
    a2 = mgr.handle("What? Still confused.")
    a3 = mgr.handle("I'm confused, I don't follow.")   # exceeds max -> escalate
    print(f"\nescalation          : {a1.type.value} -> {a2.type.value} -> {a3.type.value}")
    assert a3.type is RepairType.ESCALATE

    # a clear turn resets the guard
    mgr.handle("Got it, thanks.")
    assert mgr.attempts == 0
    print("loop guard reset    : OK")

    print(f"\nstatus              : {mgr.status()}")
    print("\nALL SELF-TESTS PASSED ✓")
