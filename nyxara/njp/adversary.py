"""NYXARA · njp/adversary.py — the subsystem that tries to break her own conclusions (☠, P3).

Everything else in :mod:`nyxara.njp` is built to reach a conclusion and to be honest about how
well supported it is. Nothing was built to *go after* one.

The gap was measurable rather than philosophical. :class:`~nyxara.njp.beliefs.Belief` has carried
a ``falsifier`` field from the start, and the two places that fill it write a template::

    falsifier = f"an observation in which {subject} does not {predicate} {obj}"

That is the claim negated, not an attack on it. It names no rival explanation, proposes no test,
and — decisively — nothing ever goes looking for it. Meanwhile the machinery an attack needs was
all present and pointed elsewhere: :meth:`nyxara.njp.world.WorldView.counterfactual` answers
"would the effect have happened anyway", :class:`~nyxara.njp.world.CausalLink` carries the lift
that separates a cause from a coincidence, and :meth:`nyxara.njp.universe.InternalUniverse.intervene`
implements the real do-operator. Two finished weapons and no trigger.

**The four attacks**, each settled by evidence that already exists::

    BELIEF   A causes B
      ↓
    RIVAL          is there another cause of B, and is it better supported?
    COINCIDENCE    does A actually raise B's odds, or does B happen anyway?
    COUNTEREXAMPLE has B ever arrived without A?
    INTERVENTION   under do(A), does B move at all?

**Undecided is the default and is not a pass.** An attack she cannot settle from the record
returns :attr:`Stance.UNDECIDED`, and undecided attacks never raise confidence. The failure this
avoids is the one that makes adversarial self-checking worthless everywhere it is done badly:
running four challenges, finding no evidence for any of them, and reporting "survived 4 attacks"
as though absence of a refutation were corroboration.

**Refutation is fail-closed**, matching :mod:`nyxara.njp.truth`: one attack that genuinely lands
ends the claim. It does not average against the ones that did not.

**Surviving may strengthen, but only through the one gate.** A survived attack that consulted
*independent* evidence is exactly what :func:`nyxara.njp.relevance.revise_confidence` was written
to price, so this calls it rather than doing its own arithmetic. An attack that merely failed to
find anything contributes zero independent evidence and therefore cannot move the number up.

This module only ever *reports*. Retraction goes through
:meth:`nyxara.njp.beliefs.BeliefLedger.retract`, which is where revision history lives; nothing
here edits the fact store, and the kernel's gate is untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Stance", "AttackKind", "Attack", "AttackReport", "SelfAttacker"]


class Stance:
    """What one attack settled. Three states, and the third is not a synonym for the first."""

    SURVIVED = "survived"      # checked against evidence, and the claim came through
    REFUTED = "refuted"        # checked against evidence, and the claim did not
    UNDECIDED = "undecided"    # the record cannot settle it — not a pass


class AttackKind:
    RIVAL = "rival_cause"
    COINCIDENCE = "coincidence"
    COUNTEREXAMPLE = "counterexample"
    INTERVENTION = "intervention"


@dataclass
class Attack:
    """One challenge, what it asked, and what the evidence actually said."""

    kind: str = ""
    question: str = ""          # the challenge, in words, so it can be shown to the Master
    stance: str = Stance.UNDECIDED
    finding: str = ""           # what was found, or why nothing could be
    rival: str = ""             # the competing explanation, where the attack found one
    #: Whether settling this consulted evidence independent of the claim itself. Only these may
    #: raise confidence; an attack answered from the claim's own support proves nothing new.
    independent: bool = False
    #: *Which* body of evidence settled it. Three attacks reading one event log are three views
    #: of one dataset, not three independent confirmations, and counting them as three is how a
    #: single thin record talks itself into certainty. See :meth:`SelfAttacker._revise`.
    source: str = ""

    @property
    def landed(self) -> bool:
        return self.stance == Stance.REFUTED

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "question": self.question, "stance": self.stance,
                "finding": self.finding[:200], "rival": self.rival,
                "independent": self.independent, "source": self.source}


@dataclass
class AttackReport:
    """Every attack run against one claim, and what it did to her confidence in it."""

    cause: str = ""
    effect: str = ""
    attacks: List[Attack] = field(default_factory=list)
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    retracted: bool = False
    ms: float = 0.0

    @property
    def refuted_by(self) -> List[Attack]:
        return [a for a in self.attacks if a.landed]

    @property
    def survived(self) -> int:
        return sum(1 for a in self.attacks if a.stance == Stance.SURVIVED)

    @property
    def undecided(self) -> int:
        return sum(1 for a in self.attacks if a.stance == Stance.UNDECIDED)

    @property
    def verdict(self) -> str:
        if self.refuted_by:
            return "refuted"
        if self.survived and not self.undecided:
            return "survived every attack that could be settled"
        if self.survived:
            return f"survived {self.survived}, {self.undecided} could not be settled"
        return "nothing could be settled from the record"

    def to_dict(self) -> Dict[str, Any]:
        return {"cause": self.cause, "effect": self.effect,
                "attacks": [a.to_dict() for a in self.attacks],
                "survived": self.survived, "undecided": self.undecided,
                "refuted": bool(self.refuted_by), "retracted": self.retracted,
                "confidence": [round(self.confidence_before, 4),
                               round(self.confidence_after, 4)],
                "verdict": self.verdict, "ms": round(self.ms, 3)}


class SelfAttacker:
    """Takes a causal claim she holds and tries, with evidence, to break it.

    Every collaborator is optional. With none attached the attacks all return
    :attr:`Stance.UNDECIDED`, which is the honest reading of "there is no record to check against"
    and is deliberately not the same as reporting that the claim survived.
    """

    def __init__(self, *, world: Any = None, universe: Any = None, beliefs: Any = None,
                 min_support: int = 3, coincidence_lift: float = 0.1,
                 rival_margin: float = 0.0) -> None:
        self.world = world
        self.universe = universe
        self.beliefs = beliefs
        self.min_support = max(1, int(min_support))
        #: Below this, the cause does not raise the effect's odds enough to be telling us anything
        #: the base rate was not already telling us.
        self.coincidence_lift = float(coincidence_lift)
        #: How much better a rival has to be before it counts against this claim. Zero means an
        #: equally-supported rival is already a problem, which is the right default: two
        #: explanations that fit equally well means neither is established.
        self.rival_margin = float(rival_margin)

        self.attacked = 0
        self.refuted = 0
        self.retracted = 0
        self.by_kind: Dict[str, int] = {}

    # ---- the one call ------------------------------------------------------ #
    def attack(self, cause: str, effect: str, *,
               confidence: Optional[float] = None,
               claim: str = "") -> AttackReport:
        """Run every attack against ``cause → effect`` and report what survived.

        ``confidence`` is the prior she holds; omitted, it is read from the belief ledger, and
        failing that from the world's own strength for the link. A report is produced either way —
        an unheld claim can still be attacked, which is what makes this usable as a check *before*
        committing rather than only as an audit afterwards.
        """
        report = AttackReport(cause=str(cause or ""), effect=str(effect or ""))
        t0 = time.perf_counter()
        try:
            if not report.cause or not report.effect:
                return report
            self.attacked += 1
            report.confidence_before = self._prior(report.cause, report.effect, confidence)

            for attack in (self._attack_rival(report.cause, report.effect),
                           self._attack_coincidence(report.cause, report.effect),
                           self._attack_counterexample(report.cause, report.effect),
                           self._attack_intervention(report.cause, report.effect)):
                report.attacks.append(attack)
                self.by_kind[attack.kind] = self.by_kind.get(attack.kind, 0) + 1

            report.confidence_after = self._revise(report)
            if report.refuted_by:
                self.refuted += 1
                report.retracted = self._retract(report, claim=claim)
            return report
        except Exception:  # noqa: BLE001 — a failed attack leaves the belief exactly as it was
            report.confidence_after = report.confidence_before
            return report
        finally:
            report.ms = (time.perf_counter() - t0) * 1000.0

    def attack_strongest(self, *, limit: int = 1) -> List[AttackReport]:
        """Go after the causal claims she is currently most committed to.

        The strongest rather than the weakest on purpose. A weak claim being wrong costs little
        and she already doubts it; a claim she is about to rely on is the one worth trying to
        break, and it is also the one nothing else in the package will ever question.
        """
        out: List[AttackReport] = []
        try:
            if self.world is None:
                return out
            for link in self.world.links(causal_only=True)[:max(1, int(limit))]:
                out.append(self.attack(link.cause, link.effect))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- the four attacks -------------------------------------------------- #
    def _attack_rival(self, cause: str, effect: str) -> Attack:
        """*Could something else account for this effect?*"""
        out = Attack(kind=AttackKind.RIVAL, source="record",
                     question=f"could something other than {cause} account for {effect}?")
        try:
            if self.world is None:
                out.finding = "no world record to look for rivals in"
                return out
            mine = self.world.link(cause, effect)
            rivals = [l for l in self.world.links(causal_only=True)
                      # A thing is not a rival explanation of itself, and a self-link is a real
                      # entry in a co-occurrence table: an event that recurs precedes itself.
                      if l.effect == effect and l.cause != cause and l.cause != effect]
            if not rivals:
                # Knowing of no rival is not evidence that none exists — it is usually evidence
                # that the record is thin, and treating it as a pass is exactly the failure this
                # module was written against. It counts only where the effect has been seen often
                # enough that a competing cause would plausibly have shown up by now, and even
                # then it earns no independent evidence: nothing new was learned about the world.
                if mine.effect_total >= self.min_support:
                    out.stance = Stance.SURVIVED
                    out.finding = (f"{effect} seen {mine.effect_total} time(s) and no other "
                                   f"cause has ever preceded it")
                else:
                    out.finding = (f"{effect} seen only {mine.effect_total} time(s) — too thin "
                                   f"to conclude anything from the absence of a rival")
                return out
            best = max(rivals, key=lambda l: l.lift)
            out.rival = best.cause
            if best.lift >= mine.lift + self.rival_margin:
                out.stance = Stance.REFUTED
                out.finding = (f"{best.cause} explains {effect} at least as well "
                               f"(lift {best.lift:.3f} vs {mine.lift:.3f})")
            else:
                out.stance = Stance.SURVIVED
                out.independent = True
                out.finding = (f"the best rival {best.cause} explains less "
                               f"(lift {best.lift:.3f} vs {mine.lift:.3f})")
            return out
        except Exception:  # noqa: BLE001
            out.finding = "the rival check failed"
            return out

    def _attack_coincidence(self, cause: str, effect: str) -> Attack:
        """*Is this a cause, or something that merely turns up nearby?*

        The discriminator is lift — how much the cause raises the effect's odds over the rate at
        which the effect happens anyway. A high co-occurrence with no lift is exactly what a
        coincidence looks like, and it is what a count-based causal record will happily report as
        a strong link.
        """
        out = Attack(kind=AttackKind.COINCIDENCE, source="record",
                     question=f"does {cause} raise the odds of {effect}, or does {effect} "
                              f"happen anyway?")
        try:
            if self.world is None:
                out.finding = "no world record to compute a base rate from"
                return out
            link = self.world.link(cause, effect)
            if link.together < self.min_support:
                # Testimony is not coincidence, and it is not observation either. A law she was
                # told and has never seen cannot be attacked *this* way — that is what the
                # intervention attack is for — so this one declines rather than guessing.
                out.finding = (f"seen together only {link.together} time(s) — too few to tell "
                               f"a cause from a coincidence")
                return out
            if link.lift <= self.coincidence_lift:
                out.stance = Stance.REFUTED
                out.finding = (f"{effect} follows {cause} {link.conditional:.2f} of the time and "
                               f"happens {link.base_rate:.2f} of the time anyway "
                               f"(lift {link.lift:.3f})")
            else:
                out.stance = Stance.SURVIVED
                out.independent = True
                out.finding = (f"{cause} raises {effect} from {link.base_rate:.2f} to "
                               f"{link.conditional:.2f} (lift {link.lift:.3f})")
            return out
        except Exception:  # noqa: BLE001
            out.finding = "the coincidence check failed"
            return out

    def _attack_counterexample(self, cause: str, effect: str) -> Attack:
        """*Has the effect ever arrived without this cause?*

        This is the falsifier the belief ledger has always named and never gone looking for, and
        the record can answer it: :meth:`~nyxara.njp.world.WorldView.counterfactual` counts the
        times the effect happened with the cause absent.
        """
        out = Attack(kind=AttackKind.COUNTEREXAMPLE, source="record",
                     question=f"has {effect} ever happened without {cause}?")
        try:
            if self.world is None:
                out.finding = "no world record to search for a counterexample in"
                return out
            got = self.world.counterfactual(cause, effect)
            if got.still_happens is None:
                out.finding = got.reason or "the record cannot say"
                return out
            out.independent = True
            if got.still_happens and got.confidence > 0.5:
                out.stance = Stance.REFUTED
                out.finding = got.reason or f"{effect} arrives by other routes"
            elif not got.still_happens:
                out.stance = Stance.SURVIVED
                out.finding = got.reason or f"{effect} has not been seen without {cause}"
            else:
                out.independent = False
                out.finding = (got.reason or "") + " — too close to call"
            return out
        except Exception:  # noqa: BLE001
            out.finding = "the counterexample check failed"
            return out

    def _attack_intervention(self, cause: str, effect: str) -> Attack:
        """*Does the relation hold when the cause is set rather than merely observed?*

        The one attack that is not answerable from co-occurrence at all, and the reason the
        do-operator exists. Observational data cannot distinguish "A drives B" from "both follow
        C"; severing A's incoming arrows can.
        """
        out = Attack(kind=AttackKind.INTERVENTION, source="simulation",
                     question=f"under do({cause}), does {effect} move?")
        try:
            if self.universe is None:
                out.finding = "no simulator to intervene in"
                return out
            result = self.universe.without(cause)
            if not result.answerable:
                out.finding = result.reason or "the simulator has no arrow to propagate"
                return out
            touched = {d.variable for d in result.deltas if d.variable != cause}
            out.independent = True
            if effect in touched:
                out.stance = Stance.SURVIVED
                out.finding = f"removing {cause} moves {effect} in the simulation"
            else:
                out.stance = Stance.REFUTED
                out.finding = (f"removing {cause} leaves {effect} unchanged in the simulation; "
                               f"it moved {sorted(touched)[:4]} instead")
            return out
        except Exception:  # noqa: BLE001
            out.finding = "the intervention check failed"
            return out

    # ---- what the attacks do to the belief --------------------------------- #
    def _prior(self, cause: str, effect: str, given: Optional[float]) -> float:
        if given is not None:
            return max(0.0, min(1.0, float(given)))
        try:
            if self.world is not None:
                return max(0.0, min(1.0, float(self.world.link(cause, effect).strength)))
        except Exception:  # noqa: BLE001
            pass
        return 0.5

    def _revise(self, report: AttackReport) -> float:
        """Fail-closed on a refutation; otherwise priced by :func:`revise_confidence`.

        A survived attack counts as independent evidence only where it actually consulted
        something independent of the claim. Attacks that came back undecided contribute nothing at
        all, which is the difference between "four challenges survived" meaning something and
        meaning "I could not check four times".
        """
        try:
            if report.refuted_by:
                return 0.0
            from nyxara.njp.relevance import revise_confidence
            # Counted by distinct *source*, not by attack. The rival, coincidence and
            # counterexample checks all read one event log — they are three questions put to one
            # dataset, and scoring them as three independent confirmations let a thin record
            # argue itself from 0.33 to 0.83 on six observations. Only the simulator is a
            # genuinely separate line of evidence, because it answers by severing arrows rather
            # than by counting co-occurrences.
            earned = len({a.source for a in report.attacks
                          if a.stance == Stance.SURVIVED and a.independent and a.source})
            return revise_confidence(report.confidence_before,
                                     independent_evidence=earned,
                                     relevance=1.0, consistent=True)
        except Exception:  # noqa: BLE001
            return report.confidence_before

    def _retract(self, report: AttackReport, *, claim: str = "") -> bool:
        """Send the refutation to the ledger, which is where revision history belongs."""
        try:
            if self.beliefs is None:
                return False
            text = claim or f"{report.cause} causes {report.effect}"
            landed = report.refuted_by[0]
            got = self.beliefs.retract(text, why=f"{landed.kind}: {landed.finding}"[:200])
            if got is not None:
                self.retracted += 1
                return True
            return False
        except Exception:  # noqa: BLE001
            return False

    # ---- reporting --------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        return {
            "attacked": self.attacked,
            "refuted": self.refuted,
            "retracted": self.retracted,
            "refutation_rate": (round(self.refuted / self.attacked, 4)
                                if self.attacked else None),
            "by_kind": dict(self.by_kind),
            "world_attached": self.world is not None,
            "universe_attached": self.universe is not None,
            "beliefs_attached": self.beliefs is not None,
        }
