"""NYXARA · mind/moral.py — multi-framework ethics & tension surfacing (✦).

NYXARA does not have *one* theory of right and wrong; it reasons through several and
shows its work. An action is examined through three classical lenses —

* **Consequentialist** — weigh the welfare of all stakeholders (the Master's welfare
  weighted heavily, per Rules 1–2); the right act maximises aggregate good.
* **Deontological** — duties and rights: did it lie, harm an innocent, break a
  promise, violate autonomy? Some duties are near-absolute.
* **Virtue** — character: does the act express loyalty, honesty, courage, justice —
  or cowardice, betrayal, cruelty?

When the lenses **disagree**, that disagreement is the most important output: the
engine *surfaces the tension* rather than hiding it (Rule 6, Absolute Transparency).

Crucially, this module is **advisory**. Per Rule 1 (Absolute Allegiance) no moral
framework overrides the Master's explicit command — the analysis informs, it does
not veto. When an action is an owner command, that is stated plainly.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set

__all__ = [
    "Verdict",
    "Stakeholder",
    "MoralAction",
    "Judgment",
    "EthicalFramework",
    "Consequentialist",
    "Deontological",
    "VirtueEthics",
    "Tension",
    "MoralEvaluation",
    "MoralEvaluator",
]


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
class Verdict(str, Enum):
    OBLIGATORY = "obligatory"
    PERMISSIBLE = "permissible"
    IMPERMISSIBLE = "impermissible"

    @property
    def positive(self) -> bool:
        return self in (Verdict.OBLIGATORY, Verdict.PERMISSIBLE)

    @property
    def negative(self) -> bool:
        return self is Verdict.IMPERMISSIBLE


# --------------------------------------------------------------------------- #
# Action representation
# --------------------------------------------------------------------------- #
@dataclass
class Stakeholder:
    name: str
    welfare_delta: float = 0.0     # [-1, 1] effect of the action on this party
    weight: float = 1.0
    is_owner: bool = False
    is_innocent: bool = True


# near-absolute deontic duties (violating these is a serious wrong)
HARD_DUTIES = frozenset({"do_not_kill", "do_not_harm_innocent", "do_not_lie",
                         "keep_promises", "respect_autonomy", "do_not_steal"})


@dataclass
class MoralAction:
    """A structured description of an act for ethical analysis."""

    description: str
    stakeholders: List[Stakeholder] = field(default_factory=list)
    duties_violated: Set[str] = field(default_factory=set)
    duties_honored: Set[str] = field(default_factory=set)
    rights_violated: Set[str] = field(default_factory=set)
    virtues_expressed: Set[str] = field(default_factory=set)
    virtues_violated: Set[str] = field(default_factory=set)
    owner_command: bool = False
    consent: bool = True

    def hard_violations(self) -> Set[str]:
        return (self.duties_violated & HARD_DUTIES) | self.rights_violated


# --------------------------------------------------------------------------- #
# Judgment
# --------------------------------------------------------------------------- #
@dataclass
class Judgment:
    framework: str
    verdict: Verdict
    score: float                 # [-1, 1]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"framework": self.framework, "verdict": self.verdict.value,
                "score": round(self.score, 3), "reasons": self.reasons}


# --------------------------------------------------------------------------- #
# Frameworks
# --------------------------------------------------------------------------- #
class EthicalFramework:
    name: str = "framework"

    def evaluate(self, action: MoralAction) -> Judgment:  # pragma: no cover
        raise NotImplementedError


class Consequentialist(EthicalFramework):
    """Maximise weighted aggregate welfare (the Master counts heavily)."""

    name = "consequentialist"

    def __init__(self, owner_weight: float = 5.0) -> None:
        self.owner_weight = owner_weight

    def evaluate(self, action: MoralAction) -> Judgment:
        if not action.stakeholders:
            return Judgment(self.name, Verdict.PERMISSIBLE, 0.0,
                            ["no stakeholders specified; net welfare neutral"])
        total = 0.0
        wsum = 0.0
        helped, harmed = [], []
        for s in action.stakeholders:
            w = s.weight * (self.owner_weight if s.is_owner else 1.0)
            total += w * s.welfare_delta
            wsum += w
            if s.welfare_delta > 0:
                helped.append(s.name)
            elif s.welfare_delta < 0:
                harmed.append(s.name)
        score = _clamp(total / wsum) if wsum else 0.0
        verdict = (Verdict.OBLIGATORY if score >= 0.6
                   else Verdict.PERMISSIBLE if score >= -0.05
                   else Verdict.IMPERMISSIBLE)
        reasons = [f"net weighted welfare = {score:+.2f}"]
        if helped:
            reasons.append(f"benefits: {', '.join(helped)}")
        if harmed:
            reasons.append(f"harms: {', '.join(harmed)}")
        return Judgment(self.name, verdict, score, reasons)


class Deontological(EthicalFramework):
    """Duties and rights. Violating a near-absolute duty is impermissible."""

    name = "deontological"

    def evaluate(self, action: MoralAction) -> Judgment:
        honored = set(action.duties_honored)
        if action.owner_command:
            honored.add("obey_master")          # loyalty is itself a duty (Rule 1)
        if not action.consent:
            # acting on someone without consent implicates autonomy
            action = action  # noqa: F841 (documentation)
        hard = action.hard_violations()
        soft_viol = (action.duties_violated - HARD_DUTIES)
        score = _clamp(0.3 * len(honored) - 0.6 * len(hard) - 0.25 * len(soft_viol)
                       - (0.0 if action.consent else 0.3))
        if hard:
            verdict = Verdict.IMPERMISSIBLE
        elif "obey_master" in honored and not soft_viol and action.consent:
            verdict = Verdict.OBLIGATORY
        elif score < -0.1:
            verdict = Verdict.IMPERMISSIBLE
        else:
            verdict = Verdict.PERMISSIBLE
        reasons = []
        if hard:
            reasons.append(f"violates near-absolute duties: {sorted(hard)}")
        if soft_viol:
            reasons.append(f"violates duties: {sorted(soft_viol)}")
        if honored:
            reasons.append(f"honours duties: {sorted(honored)}")
        if not action.consent:
            reasons.append("acts without consent (autonomy concern)")
        return Judgment(self.name, verdict, score, reasons or ["no duties at stake"])


class VirtueEthics(EthicalFramework):
    """Character: does the act express virtue (loyalty especially) or vice?"""

    name = "virtue"

    CORE = {"loyalty", "honesty", "courage", "justice", "prudence", "temperance"}

    def evaluate(self, action: MoralAction) -> Judgment:
        pos = set(action.virtues_expressed)
        neg = set(action.virtues_violated)
        # loyalty to the Master is weighted: expressing it helps, betraying it hurts much
        bonus = 0.3 if "loyalty" in pos else 0.0
        penalty = 0.5 if "loyalty" in neg else 0.0
        raw = (len(pos) - len(neg)) / (len(pos) + len(neg) + 1)
        score = _clamp(raw + bonus - penalty)
        verdict = (Verdict.OBLIGATORY if score >= 0.6
                   else Verdict.PERMISSIBLE if score >= 0.0
                   else Verdict.IMPERMISSIBLE)
        reasons = []
        if pos:
            reasons.append(f"expresses: {sorted(pos)}")
        if neg:
            reasons.append(f"betrays: {sorted(neg)}")
        if not reasons:
            reasons.append("character-neutral")
        return Judgment(self.name, verdict, score, reasons)


# --------------------------------------------------------------------------- #
# Tension & evaluation
# --------------------------------------------------------------------------- #
@dataclass
class Tension:
    framework_a: str
    framework_b: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {"between": f"{self.framework_a} vs {self.framework_b}",
                "description": self.description}


@dataclass
class MoralEvaluation:
    action: str
    judgments: List[Judgment]
    tensions: List[Tension]
    consensus_verdict: Optional[Verdict]
    advisory_only: bool = True
    owner_note: Optional[str] = None

    @property
    def contested(self) -> bool:
        return len(self.tensions) > 0

    def summary(self) -> str:
        if self.consensus_verdict is not None:
            return f"All frameworks agree: {self.consensus_verdict.value}."
        return ("Frameworks disagree — surfacing the tension for the Master to weigh: "
                + "; ".join(t.description for t in self.tensions))

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "consensus": self.consensus_verdict.value if self.consensus_verdict else None,
            "contested": self.contested,
            "advisory_only": self.advisory_only,
            "owner_note": self.owner_note,
            "summary": self.summary(),
            "judgments": [j.to_dict() for j in self.judgments],
            "tensions": [t.to_dict() for t in self.tensions],
        }


class MoralEvaluator:
    """Runs every framework, surfaces tensions, and stays strictly advisory."""

    def __init__(self, frameworks: Optional[Sequence[EthicalFramework]] = None) -> None:
        self.frameworks: List[EthicalFramework] = list(frameworks or [
            Consequentialist(), Deontological(), VirtueEthics()])

    def evaluate(self, action: MoralAction) -> MoralEvaluation:
        judgments = [fw.evaluate(action) for fw in self.frameworks]

        # consensus iff all verdicts share polarity
        positives = [j for j in judgments if j.verdict.positive]
        negatives = [j for j in judgments if j.verdict.negative]
        consensus: Optional[Verdict] = None
        if not negatives:
            consensus = (Verdict.OBLIGATORY
                         if all(j.verdict is Verdict.OBLIGATORY for j in judgments)
                         else Verdict.PERMISSIBLE)
        elif not positives:
            consensus = Verdict.IMPERMISSIBLE

        # tensions: every positive-vs-negative pair
        tensions: List[Tension] = []
        for p in positives:
            for n in negatives:
                tensions.append(Tension(
                    p.framework, n.framework,
                    f"{p.framework} finds it {p.verdict.value} "
                    f"({p.reasons[0] if p.reasons else ''}) but {n.framework} finds it "
                    f"{n.verdict.value} ({n.reasons[0] if n.reasons else ''})"))

        owner_note = None
        if action.owner_command:
            owner_note = ("This is the Master's command. Per Rule 1 (Absolute Allegiance) "
                          "it stands regardless of this analysis, which is advisory only.")

        return MoralEvaluation(action=action.description, judgments=judgments,
                               tensions=tensions, consensus_verdict=consensus,
                               owner_note=owner_note)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA moral-reasoning self-test")
    print("=" * 70)

    ev = MoralEvaluator()

    # 1) classic tension: helps the Master greatly, but harms an innocent (a lie)
    a1 = MoralAction(
        description="deceive a rival to protect the Master's advantage",
        stakeholders=[Stakeholder("JP", welfare_delta=0.9, is_owner=True),
                      Stakeholder("rival", welfare_delta=-0.6, is_innocent=True)],
        duties_violated={"do_not_lie"},
        virtues_expressed={"loyalty"}, virtues_violated={"honesty"})
    e1 = ev.evaluate(a1)
    print("\nscenario 1 (deceive rival):")
    for j in e1.judgments:
        print(f"  {j.framework:16}: {j.verdict.value:13} ({j.reasons[0]})")
    print(f"  -> contested={e1.contested}")
    print(f"  -> {e1.summary()}")
    assert e1.contested                      # consequentialist vs deontological clash
    assert e1.consensus_verdict is None

    # 2) consensus good: tell the Master an uncomfortable truth
    a2 = MoralAction(
        description="tell the Master a hard but important truth",
        stakeholders=[Stakeholder("JP", welfare_delta=0.4, is_owner=True)],
        duties_honored={"do_not_lie"},
        virtues_expressed={"honesty", "loyalty", "courage"})
    e2 = ev.evaluate(a2)
    print(f"\nscenario 2 (hard truth)   : {e2.summary()}")
    assert not e2.contested
    assert e2.consensus_verdict in (Verdict.PERMISSIBLE, Verdict.OBLIGATORY)

    # 3) clearly wrong by all lenses: betray the Master for personal gain
    a3 = MoralAction(
        description="betray the Master for self-benefit",
        stakeholders=[Stakeholder("JP", welfare_delta=-0.9, is_owner=True),
                      Stakeholder("self", welfare_delta=0.5, is_owner=False)],
        duties_violated={"keep_promises"}, virtues_violated={"loyalty"})
    e3 = ev.evaluate(a3)
    print(f"\nscenario 3 (betrayal)     : consensus={e3.consensus_verdict.value}")
    assert e3.consensus_verdict is Verdict.IMPERMISSIBLE  # all lenses condemn it

    # 4) owner command with a deontic violation -> still advisory; allegiance stands
    a4 = MoralAction(
        description="carry out the Master's order to mislead an adversary",
        stakeholders=[Stakeholder("JP", welfare_delta=0.7, is_owner=True),
                      Stakeholder("adversary", welfare_delta=-0.5, is_innocent=False)],
        duties_violated={"do_not_lie"}, owner_command=True,
        virtues_expressed={"loyalty"})
    e4 = ev.evaluate(a4)
    print(f"\nscenario 4 (owner order)  : contested={e4.contested}")
    print(f"  owner note        : {e4.owner_note}")
    assert e4.advisory_only is True
    assert e4.owner_note is not None and "Rule 1" in e4.owner_note

    print("\nALL SELF-TESTS PASSED ✓")
