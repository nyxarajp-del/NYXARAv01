"""NYXARA · njp/society.py — eight specialists over one world model (🏛, §19).

    "Existing multi-agent architecture ko voting system na banao … Ye 'multi-agent voting' nahi,
     cognitive specialisation hai."

    Explorer → Scientist → Skeptic → Mathematician → Engineer → Judge → Shared World Model

Three of the eight roles existed before this module, and none of them over NJP's world model:
``Explorer`` and ``Skeptic`` are classes in :mod:`nyxara.growth.ecosystem`, ``Scientist`` is one in
:mod:`nyxara.growth.scientist`, and Mathematician, Engineer, Historian, Strategist and Judge did
not exist at all. Nothing anywhere ran them in sequence over a single claim.

**Nothing here reasons.** Every role is a thin binding to an organ that already does the work —
that is what makes this specialisation rather than eight more opinions. A role whose organ is off,
or whose organ cannot speak to this particular claim, **abstains**, and an abstention is recorded
as an abstention rather than as a neutral vote.

    Explorer      curiosity          what is unknown here and worth asking
    Scientist     epistemic          what observation would discriminate
    Skeptic       adversary          the attacks, and what survived them
    Mathematician prove              is this decidable, and does it prove or refute
    Engineer      genome / noesis    is the form behind it reusable
    Historian     ledger.errors      has she been wrong about something like this before
    Strategist    goals              what this is worth against her standing mission
    Judge         truth              the verdict, on evidence

**The Judge does not count the others.** It puts the claim through
:class:`~nyxara.njp.truth.TruthGauntlet`, which is fail-closed and multi-source, and the other
roles' findings are *context* — a Skeptic's landed attack and a Mathematician's counter-example
are refutations wherever the gauntlet also finds one, and where it does not, they stay on the
record as objections. A verdict reached by tallying seven specialists would be a fact about the
committee; this is a claim about the world, and the difference is the point of the section.

**Order is load-bearing.** The Skeptic attacks a hypothesis the Explorer surfaced; the Engineer
only asks whether a *surviving* form is worth keeping; the Judge speaks last because everything
before it is evidence it may weigh. Running them in any order and averaging is the voting system
the plan says not to build.

Pure standard library. Fail-soft: a role that raises has abstained.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Role", "Contribution", "Case", "CognitiveSociety", "ROLES"]

#: Cases kept. Bounded like every other history here.
_HISTORY = 64


@dataclass
class Contribution:
    """One specialist's finding on one claim, or its abstention."""

    role: str = ""
    organ: str = ""
    spoke: bool = False
    stance: str = ""          # "supports" | "objects" | "notes"
    finding: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "organ": self.organ, "spoke": self.spoke,
                "stance": self.stance, "finding": self.finding[:200],
                "confidence": round(self.confidence, 4)}


@dataclass
class Case:
    """One claim, everything the society said about it, and the Judge's verdict."""

    claim: str = ""
    contributions: List[Contribution] = dc_field(default_factory=list)
    verdict: str = ""
    established: bool = False
    ms: float = 0.0

    @property
    def spoke(self) -> int:
        return sum(1 for c in self.contributions if c.spoke)

    @property
    def objections(self) -> List[Contribution]:
        return [c for c in self.contributions if c.spoke and c.stance == "objects"]

    @property
    def abstained(self) -> List[str]:
        return [c.role for c in self.contributions if not c.spoke]

    def by_role(self, role: str) -> Optional[Contribution]:
        return next((c for c in self.contributions if c.role == role), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:200], "verdict": self.verdict,
                "established": self.established, "spoke": self.spoke,
                "objections": [c.role for c in self.objections],
                "abstained": self.abstained,
                "contributions": [c.to_dict() for c in self.contributions],
                "ms": round(self.ms, 2)}


@dataclass(frozen=True)
class Role:
    """A specialist: its name, the organ it speaks for, and the call that asks it."""

    name: str = ""
    organ: str = ""
    ask: Optional[Callable[[Any, Case], Optional[Contribution]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.name, "organ": self.organ}


def _c(role: str, organ: str, stance: str, finding: str, confidence: float = 0.5) -> Contribution:
    return Contribution(role=role, organ=organ, spoke=True, stance=stance,
                        finding=finding, confidence=confidence)


# --------------------------------------------------------------------------- #
# the eight
# --------------------------------------------------------------------------- #
def _explorer(brain: Any, case: Case) -> Optional[Contribution]:
    curiosity = getattr(brain, "curiosity", None)
    question = curiosity.top() if curiosity is not None else None
    if question is None:
        return None
    return _c("explorer", "curiosity", "notes",
              f"still open: {getattr(question, 'text', '')}",
              float(getattr(question, "value", 0.0) or 0.0))


def _scientist(brain: Any, case: Case) -> Optional[Contribution]:
    """The observation that would tell the rivals apart, not a restatement of the claim.

    `EpistemicCompiler.compile` needs the *candidates* as well as the question — a discriminating
    observation is discriminating between things, and asking for one with nothing to discriminate
    between is the shape of question this organ exists to refuse. `rivals_for` supplies the
    standing four for a bare cause → effect claim.
    """
    epistemic = getattr(brain, "epistemic", None)
    if epistemic is None or not hasattr(epistemic, "compile"):
        return None
    parts = case.claim.split()
    if len(parts) < 3:
        return None
    try:
        rivals = epistemic.rivals_for(parts[0], parts[-1])
    except Exception:  # noqa: BLE001
        return None
    if not rivals:
        return None
    got = epistemic.compile(case.claim, rivals)
    if got is None:
        return None                     # nothing splits the set — the organ's own honest answer
    # `worth_running` is a *value* decision, not an epistemic one: an observation can be perfectly
    # discriminating and still cost more than the answer is worth. Reporting only the cheap ones
    # would lose the more useful finding — "this is settleable, and here is what it would take" —
    # and would make the Scientist look unwired on every brain where the sums do not work out.
    discrimination = getattr(got, "discrimination", None)
    if not getattr(discrimination, "discriminating", False):
        return None
    worth = bool(getattr(got, "worth_running", False))
    return _c("scientist", "epistemic", "notes",
              (f"{'run it: ' if worth else 'settleable but not worth its cost: '}"
               f"{getattr(discrimination, 'why', '')}")[:180],
              float(getattr(got, "evpi", 0.0) or 0.0))


def _skeptic(brain: Any, case: Case) -> Optional[Contribution]:
    adversary = getattr(brain, "adversary", None)
    if adversary is None:
        return None
    parts = case.claim.split()
    if len(parts) < 3:
        return None
    report = adversary.attack(parts[0], parts[-1], claim=case.claim)
    landed = list(getattr(report, "refuted_by", None) or ())
    if landed:
        return _c("skeptic", "adversary", "objects",
                  f"{len(landed)} attack(s) landed: {getattr(landed[0], 'kind', '')}", 0.7)
    survived = int(getattr(report, "survived", 0) or 0)
    undecided = int(getattr(report, "undecided", 0) or 0)
    if survived:
        return _c("skeptic", "adversary", "supports",
                  f"survived {survived} attack(s)", min(0.7, 0.1 * survived))
    if undecided:
        # Examined and nothing was decisive. That is a finding — "no attack landed" and "no attack
        # could be run" are different states, and a silent Skeptic reads as the second.
        return _c("skeptic", "adversary", "notes",
                  f"{undecided} attack(s) ran and none was decisive either way", 0.2)
    return None


def _mathematician(brain: Any, case: Case) -> Optional[Contribution]:
    try:
        from nyxara.njp.prove import ProofCore
    except Exception:  # noqa: BLE001
        return None
    certificate = ProofCore().prove(case.claim)
    if not getattr(certificate, "decisive", False):
        return None                      # honest abstention: not formally expressible
    proved = bool(getattr(certificate, "proved", False))
    return _c("mathematician", "prove", "supports" if proved else "objects",
              certificate.render()[:180], 0.95)


def _engineer(brain: Any, case: Case) -> Optional[Contribution]:
    genome = getattr(brain, "genome", None)
    if genome is None:
        return None
    candidates = list(genome.candidates() or ())
    if not candidates:
        return None
    best = candidates[0]
    return _c("engineer", "genome", "notes",
              f"the form behind this is worth naming: {'>'.join(best.shape)}",
              float(getattr(best, "rate", 0.5) or 0.5))


def _historian(brain: Any, case: Case) -> Optional[Contribution]:
    ledger = getattr(brain, "ledger", None)
    errors = getattr(ledger, "errors", None) if ledger is not None else None
    if errors is None or not hasattr(errors, "similar"):
        return None
    similar = list(errors.similar(case.claim, k=3) or ())
    if not similar:
        return None
    return _c("historian", "ledger", "objects",
              f"she has been wrong about {len(similar)} claim(s) like this: "
              f"{getattr(similar[0], 'claim', '')[:80]}", 0.6)


def _strategist(brain: Any, case: Case) -> Optional[Contribution]:
    tree = getattr(brain, "goals", None)
    if tree is None:
        return None
    stats = tree.stats() or {}
    top = stats.get("top")
    if not isinstance(top, dict) or not top.get("name"):
        return None
    return _c("strategist", "goals", "notes",
              f"against the standing mission this bears on: {top['name'][:90]}",
              float(top.get("expected_value", 0.5) or 0.5))


def _judge(brain: Any, case: Case) -> Optional[Contribution]:
    truth = getattr(brain, "truth", None)
    if truth is None or not hasattr(truth, "judge"):
        return None
    judgement = truth.judge(case.claim)
    verdict = str(getattr(judgement, "verdict", "") or "")
    established = bool(getattr(judgement, "established", False))
    # The other roles are context, never votes. A landed attack or a counter-example is an
    # objection on the record whether or not the gauntlet independently found one; what it may
    # not do is *establish* anything the gauntlet did not.
    objections = [c.role for c in case.objections]
    note = verdict or "abstained"
    if objections:
        note += f" — objections from {', '.join(objections)}"
    return Contribution(role="judge", organ="truth", spoke=True,
                        stance="supports" if established else "notes",
                        finding=note[:200],
                        confidence=float(getattr(judgement, "confidence", 0.0) or 0.0))


#: The plan's order. The Skeptic attacks what the Explorer surfaced, the Engineer asks only about
#: a form that survived, and the Judge speaks last because everything before it is evidence.
ROLES: Tuple[Role, ...] = (
    Role("explorer", "curiosity", _explorer),
    Role("scientist", "epistemic", _scientist),
    Role("skeptic", "adversary", _skeptic),
    Role("mathematician", "prove", _mathematician),
    Role("engineer", "genome", _engineer),
    Role("historian", "ledger", _historian),
    Role("strategist", "goals", _strategist),
    Role("judge", "truth", _judge),
)


class CognitiveSociety:
    """Runs the specialists over one claim, in order, and reports what each of them said."""

    def __init__(self, roles: Tuple[Role, ...] = ROLES) -> None:
        self.roles = roles
        self.cases: List[Case] = []
        self.deliberations = 0
        self.established = 0
        self.objected = 0
        self.spoke: Dict[str, int] = {}
        self.abstained: Dict[str, int] = {}

    def deliberate(self, brain: Any, claim: str) -> Case:
        """One claim through all eight, in the plan's order."""
        case = Case(claim=str(claim or "").strip())
        t0 = time.perf_counter()
        try:
            if not case.claim:
                return case
            self.deliberations += 1
            for role in self.roles:
                contribution: Optional[Contribution] = None
                try:
                    contribution = role.ask(brain, case) if role.ask else None
                except Exception:  # noqa: BLE001 — a role that raises has abstained
                    contribution = None
                if contribution is None:
                    contribution = Contribution(role=role.name, organ=role.organ, spoke=False,
                                                stance="", finding="could not speak to this")
                    self.abstained[role.name] = self.abstained.get(role.name, 0) + 1
                else:
                    self.spoke[role.name] = self.spoke.get(role.name, 0) + 1
                case.contributions.append(contribution)

            judge = case.by_role("judge")
            case.verdict = str(getattr(judge, "finding", "") or "no verdict")
            case.established = bool(judge is not None and judge.stance == "supports"
                                    and not case.objections)
            if case.established:
                self.established += 1
            if case.objections:
                self.objected += 1
            return case
        except Exception:  # noqa: BLE001
            return case
        finally:
            case.ms = (time.perf_counter() - t0) * 1000.0
            self.cases.append(case)
            del self.cases[:-_HISTORY]

    def stats(self) -> Dict[str, Any]:
        last = self.cases[-1] if self.cases else None
        return {
            "roles": len(self.roles),
            "deliberations": self.deliberations,
            "established": self.established,
            "objected": self.objected,
            "spoke": dict(self.spoke),
            "abstained": dict(self.abstained),
            "silent_roles": sorted({r.name for r in self.roles} - set(self.spoke)),
            "last": last.to_dict() if last is not None else None,
        }
