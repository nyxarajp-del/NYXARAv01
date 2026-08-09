"""NYXARA · nyx/dialectic.py — an answer that has been attacked before you see it (⚔, NYX V.02).

:mod:`nyxara.nyx.workspace` already runs a competition — but it is a competition for *salience*.
The winner is the loudest coalition, not the one that survived being torn apart. Nothing in NYX
has ever attacked a conclusion **after** it won and **before** it shipped.

:class:`AdversarialSelfDialogue` is that step. The winning answer is defended by a Proponent and
attacked by a Skeptic, and only what survives reaches the Master:

* **The Skeptic's weapons are real.** :class:`~nyxara.mind.inner_critic.InnerCritic` for
  vagueness, contradiction, unsupported claim and cliché; :mod:`nyxara.nyx.theorem_prover` for
  the sharpest one there is — *"prove it"* — which can **refute** an answer outright; and a
  check for a claim with no mechanism behind it.
* **Three outcomes, all real.** The answer survives; or it survives with its **confidence cut**;
  or she **abstains**, because an answer she cannot defend is worse than saying nothing.
* **Every objection is visible.** They ride on the result and into ``/nyx why``, so the Master
  can see what was attacked and what was left standing.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **This is not "bulletproof".** It is a *structured critique* with a **finite** objection set:
  vagueness, contradiction, unsupported assertion, missing mechanism, and formal refutation.
  An error outside that set walks straight through, and calling it bulletproof would be the
  usual lie.
* **It is budgeted, and the budget is enforced rather than declared.** The prover is the only
  expensive weapon — z3, with its own timeout an order of magnitude past a turn's budget — so
  it is handed *the time that is actually left* instead of its own worst case. A proof that
  fits still happens; one that does not comes back as "could not prove", which this module
  already treats as no objection. When nothing is left it does without the prover entirely and
  counts that, so the difference is never invisible.
* Confidence going *down* after a debate is a **success**, not a failure. The point is not to
  win the argument; it is to know how much of the answer was load-bearing.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Objection", "Debate", "AdversarialSelfDialogue"]

#: Words that assert without carrying anything — the Skeptic's cheapest and most useful catch.
_HEDGELESS = re.compile(r"\b(always|never|every|all|none|obviously|clearly|certainly|proves?)\b",
                        re.I)
#: A claim with a *because* in it has offered a mechanism; one without has offered an assertion.
_MECHANISM = re.compile(r"\b(because|since|due\s+to|by|via|through|so\s+that|therefore|"
                        r"kyunki|wajah|क्योंकि)\b", re.I)


@dataclass
class Objection:
    """One attack the Skeptic actually landed."""

    kind: str = ""           # vagueness | contradiction | unsupported | no-mechanism | refuted
    detail: str = ""
    severity: float = 0.0    # [0,1]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "severity": round(self.severity, 3)}


@dataclass
class Debate:
    """One round of Proponent versus Skeptic, and what was left standing."""

    claim: str = ""
    debated: bool = False
    objections: List[Objection] = field(default_factory=list)
    defence: str = ""
    survived: bool = True
    abstained: bool = False
    truncated: bool = False   # the budget cut the attack short — fewer weapons were used
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    answer: str = ""
    note: str = ""
    elapsed_ms: float = 0.0

    @property
    def severity(self) -> float:
        return sum(o.severity for o in self.objections)

    @property
    def strongest(self) -> Optional[Objection]:
        return max(self.objections, key=lambda o: o.severity, default=None)

    def render(self) -> str:
        if not self.debated:
            return f"(not debated — {self.note})"
        lines = [f"claim      : {self.claim[:110]}"]
        for obj in self.objections:
            lines.append(f"  ✗ {obj.kind:14} {obj.detail[:80]}  (severity {obj.severity:.2f})")
        if self.defence:
            lines.append(f"  defence    : {self.defence[:100]}")
        lines.append(f"  confidence : {self.confidence_before:.2f} → {self.confidence_after:.2f}")
        if self.truncated:
            lines.append("  budget     : the attack was cut short — fewer weapons than usual")
        lines.append(f"  outcome    : "
                     f"{'abstained' if self.abstained else 'survived' if self.survived else 'weakened'}"
                     f" — {self.note}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim, "debated": self.debated,
                "objections": [o.to_dict() for o in self.objections],
                "defence": self.defence, "survived": self.survived,
                "abstained": self.abstained, "truncated": self.truncated,
                "severity": round(self.severity, 3),
                "confidence_before": round(self.confidence_before, 4),
                "confidence_after": round(self.confidence_after, 4),
                "answer": self.answer, "note": self.note,
                "elapsed_ms": round(self.elapsed_ms, 2)}


class AdversarialSelfDialogue:
    """Attack the winner before it ships. What does not survive does not reach the Master."""

    def __init__(self, brain: Any = None, *, abstain_above: float = 1.2,
                 weaken_above: float = 0.4, budget_ms: float = 250.0,
                 use_prover: bool = True) -> None:
        self.brain = brain
        # Total severity above which she says nothing rather than something she cannot defend.
        self.abstain_above = max(0.0, float(abstain_above))
        self.weaken_above = max(0.0, float(weaken_above))
        self.budget_ms = max(10.0, float(budget_ms))
        self.use_prover = bool(use_prover)

        self._critic: Any = None
        self.budget_bites = 0        # times the budget made her do WITHOUT the prover
        self.prover_bounded = 0      # times it ran, but on the time left rather than its own
        self.debates = 0
        self.weakened = 0
        self.abstentions = 0
        self.refutations = 0
        self.skipped = 0

    def critic(self) -> Any:
        if self._critic is None:
            try:
                from nyxara.mind.inner_critic import InnerCritic
                self._critic = InnerCritic()
            except Exception:  # noqa: BLE001 — the Skeptic loses a weapon, not its job
                self._critic = None
        return self._critic

    # ---- the one call ---------------------------------------------------------- #
    def debate(self, claim: str, *, confidence: float = 0.5,
               verified: bool = False, evidence: Optional[List[str]] = None) -> Debate:
        """Proponent, Skeptic, synthesis. Three real outcomes, and one of them is silence."""
        out = Debate(claim=str(claim or "").strip(), answer=str(claim or "").strip(),
                     confidence_before=float(confidence), confidence_after=float(confidence))
        t0 = time.perf_counter()
        try:
            if not out.claim:
                out.note = "there was nothing to defend"
                self.skipped += 1
                return out

            deadline = t0 + self.budget_ms / 1000.0
            out.objections, out.truncated = self._attack(
                out.claim, evidence or [], verified=verified, deadline=deadline)
            out.defence = self._defend(out.claim, verified=verified, evidence=evidence or [])
            out.debated = True
            self.debates += 1

            severity = out.severity
            refuted = any(o.kind == "refuted" for o in out.objections)
            if refuted or severity >= self.abstain_above:
                # An answer she cannot defend is worse than saying nothing.
                out.abstained, out.survived = True, False
                out.answer = ""
                out.confidence_after = 0.0
                self.abstentions += 1
                out.note = ("the Skeptic broke it and the Proponent could not put it back "
                            "together — she says nothing rather than something she cannot "
                            "defend")
            elif severity >= self.weaken_above:
                out.survived = True
                out.confidence_after = max(0.0, out.confidence_before * (1.0 - min(0.8, severity)))
                self.weakened += 1
                out.note = ("it survived, but not intact — the confidence you see is what was "
                            "left after the attack, which is the point")
            elif out.truncated:
                out.note = ("the budget ran out mid-attack, so it stands against FEWER weapons "
                            "than usual — surviving a shortened attack is not the same claim")
            else:
                out.note = "attacked and left standing"
            return out
        except Exception as exc:  # noqa: BLE001 — a failed debate never eats the answer
            out.note = f"the debate failed ({type(exc).__name__}) — the answer stands unattacked"
            return out
        finally:
            out.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # ---- the Skeptic ------------------------------------------------------------- #
    def _attack(self, claim: str, evidence: List[str], *, verified: bool,
                deadline: float = 0.0) -> Tuple[List[Objection], bool]:
        """Returns the objections **and** whether the budget cut the attack short.

        Surviving a truncated attack is not the same as surviving a full one, and a caller that
        cannot tell them apart will read the first as the second.
        """
        out: List[Objection] = []

        # The sharpest weapon there is: ask for a proof, and take a refutation seriously.
        # It is also the only expensive one — z3, with its own timeout, serialised behind the
        # process-wide lock — which is what the budget exists for. The budget was declared and
        # then never consulted, so "a turn that cannot afford a debate is answered without it"
        # was a promise with no mechanism. It is bounded here rather than hoped for: the prover
        # is given the time that is actually left, never its own worst case.
        if self.use_prover:
            prover = self._affordable_prover(deadline)
            if prover is not None:
                try:
                    got = prover.prove(claim)
                    if getattr(got, "verdict", "") == "refuted":
                        self.refutations += 1
                        out.append(Objection(
                            kind="refuted", severity=1.0,
                            detail=f"formally refuted: {got.note[:100]}"))
                except Exception:  # noqa: BLE001 — a prover failure is not an objection
                    pass

        if deadline > 0.0 and time.perf_counter() >= deadline:
            # The budget is spent. She stops attacking rather than running over, and the count
            # says she did — a shorter attack that admits it beats a longer one that hides it.
            self.budget_bites += 1
            return out, True

        critic = self.critic()
        if critic is not None:
            try:
                report = critic.attack(claim)
                for objection in (getattr(report, "objections", []) or [])[:4]:
                    out.append(Objection(kind=str(getattr(objection, "kind", "critique")),
                                         detail=str(getattr(objection, "detail", "")),
                                         severity=float(getattr(objection, "severity", 0.0))))
            except Exception:  # noqa: BLE001
                pass

        if deadline > 0.0 and time.perf_counter() >= deadline:
            self.budget_bites += 1
            return out, True

        # A universal with no evidence behind it is the commonest overreach there is.
        if _HEDGELESS.search(claim) and not evidence and not verified:
            out.append(Objection(
                kind="unsupported", severity=0.4,
                detail="it makes a universal claim with nothing behind it"))

        # An assertion is not an explanation.
        if not _MECHANISM.search(claim) and len(claim.split()) > 6 and not verified:
            out.append(Objection(
                kind="no-mechanism", severity=0.3,
                detail="it asserts what happens and never says by what means"))
        return out, False

    def _affordable_prover(self, deadline: float) -> Any:
        """The prover, bounded by what is left of the budget — or ``None`` when nothing is.

        Her own prover carries a 3000 ms timeout and the debate is budgeted at 250 ms, so
        letting it run unbounded would overrun the budget by an order of magnitude on exactly
        the claim that is hardest to decide. Rather than drop the sharpest weapon, it is given
        the remaining time: a proof that fits still happens, one that does not comes back as
        "could not prove", which is a verdict this module already treats as no objection.
        """
        prover = getattr(self.brain, "prover", None)
        if prover is None:
            return None
        if deadline <= 0.0:
            return prover
        remaining_ms = (deadline - time.perf_counter()) * 1000.0
        if remaining_ms <= 1.0:
            self.budget_bites += 1
            return None
        own = float(getattr(prover, "timeout_ms", 0.0) or 0.0)
        if own <= remaining_ms:
            return prover
        try:
            from nyxara.nyx.theorem_prover import ProofCore
            # Bounding is the ordinary path, not an exception — her prover's own worst case is
            # an order of magnitude past a turn's budget — so it is counted separately from
            # actually going without, which is the thing worth noticing.
            self.prover_bounded += 1
            return ProofCore(timeout_ms=int(max(1.0, remaining_ms)),
                             max_vars=int(getattr(prover, "max_vars", 8)))
        except Exception:  # noqa: BLE001 — without a bounded one she does without, not over
            self.budget_bites += 1
            return None

    # ---- the Proponent ------------------------------------------------------------- #
    @staticmethod
    def _defend(claim: str, *, verified: bool, evidence: List[str]) -> str:
        """What can honestly be said for it. A defence that invents support is not a defence."""
        if verified:
            return "it was derived and checked, which is the strongest thing I can say for it"
        if evidence:
            return f"it stands on {len(evidence)} piece(s) of retrieved evidence"
        if _MECHANISM.search(claim):
            return "it offers a mechanism rather than only an outcome"
        return "I have nothing to offer in its defence beyond having said it"

    # ---- the seam think() uses ------------------------------------------------------- #
    def review(self, thought: Any) -> Debate:
        """Attack the winning proposal of a thought. Called before anything is spoken."""
        try:
            winner = getattr(thought, "winner", None)
            if winner is None:
                out = Debate(note="nothing reached awareness, so there was nothing to attack")
                self.skipped += 1
                return out
            return self.debate(str(getattr(winner, "content", "") or ""),
                               confidence=float(getattr(winner, "confidence", 0.0)),
                               verified=bool(getattr(winner, "verifiable", False)),
                               evidence=list(getattr(winner, "evidence", []) or []))
        except Exception:  # noqa: BLE001
            return Debate(note="the review failed; the answer stands unattacked")

    # ---- introspection ----------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "debates": self.debates, "weakened": self.weakened,
                "abstentions": self.abstentions, "refutations": self.refutations,
                "skipped": self.skipped, "budget_bites": self.budget_bites,
                "prover_bounded": self.prover_bounded, "budget_ms": self.budget_ms,
                "abstain_above": self.abstain_above, "weaken_above": self.weaken_above,
                "objection_kinds": ["vagueness", "contradiction", "unsupported",
                                    "no-mechanism", "refuted"],
                "note": ("this is a STRUCTURED CRITIQUE with a finite objection set, not a "
                         "bulletproof review: an error outside that set walks straight "
                         "through. It is budgeted at one debate per turn, and a turn that was "
                         "not debated records debated=False so the difference is never "
                         "invisible. Confidence going DOWN after a debate is a success — the "
                         "point is not to win the argument, it is to know how much of the "
                         "answer was load-bearing"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"debates": self.debates, "weakened": self.weakened,
                "abstentions": self.abstentions, "refutations": self.refutations,
                "skipped": self.skipped, "budget_bites": self.budget_bites,
                "prover_bounded": self.prover_bounded}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            for key in ("debates", "weakened", "abstentions", "refutations", "skipped",
                        "budget_bites", "prover_bounded"):
                setattr(self, key, int(data.get(key, 0)))
            return True
        except Exception:  # noqa: BLE001
            return False
