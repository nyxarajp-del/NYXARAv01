"""NYXARA · njp/assume.py — the assumptions she never tested (🕳, NJP V.11, Phase 3).

**The third knowledge state.** She could already say what she knows and what she does not:
:mod:`nyxara.njp.beliefs` answers the first, :mod:`nyxara.njp.curiosity` the second. Neither can
answer the one that actually gets a model killed —

    *"which assumptions is my model resting on that I have never tested?"*

An arrow she believes is not one claim. ``aag → garmi`` asserts four separate things at once, and
she was holding all four on the evidence for one of them:

======================  =====================================================================
``SOLE_CAUSE``          nothing *else* produces the effect — "does C also cause B?"
``UNCONDITIONAL``       it holds outright, not only under some circumstance nobody named —
                        "does A cause B only under condition X?"
``DIRECTED``            it runs this way and not the other — "could B cause A?"
``UNCONFOUNDED``        nothing produces *both* ends — "is there a hidden variable D?"
======================  =====================================================================

Those are the four classical threats to a causal claim, and the Master's own plan names all four.
An untested one is not a gap in what she knows; it is a gap she does not know she has. The count
of them is the honest size of her unknown-unknown surface, and it is the number this module
exists to produce.

**Every one is tested against her own record, never guessed.** That is what makes this an organ
rather than a checklist:

* *sole cause* — ask :meth:`~nyxara.njp.world.WorldView.links` what else has lift into the same
  effect. Another cause with comparable lift refutes it.
* *unconditional* — ``P(effect | cause)`` is what the record already computes. A cause that is
  present and is often *not* followed by its effect is one whose arrow depends on something
  unmodelled, and the conditional probability says so without needing to know what.
* *directed* — the reverse pair is in the same table. If the effect also precedes the cause with
  real support, the direction is not established, whatever the arrow says.
* *unconfounded* — scan for a third event that precedes both ends. A shared upstream cause is the
  one thing that makes a genuine correlation a false arrow.

**A refuted assumption is a discovery.** This is the point at which the Phase 3 chain closes:

    unknown  →  hypothesis  →  experiment  →  result  →  discovery

The unknown is an untested assumption. The hypothesis is its negation — "something else causes
this too". The experiment is the record query above. The result is the verdict. And a refutation
is a *discovery*, because she has found structure her model did not have and can say what it is:
the alternative cause, the reversal, the confounder, by name.

**Refuting is the useful outcome, and holding is the weaker one.** An assumption that survives is
recorded as ``HOLDS`` *on the evidence available*, never as proven — absence of a counterexample
in a short record is not a demonstration, and the module says so rather than promoting a survival
into a certainty. That asymmetry is the same one :mod:`nyxara.njp.truth` keeps.

Pure standard library. No LLM. Fail-soft: every entry point degrades to an empty result rather
than breaking a turn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

__all__ = ["AssumptionKind", "AssumptionStatus", "Assumption", "AssumptionMiner"]


class AssumptionKind:
    """The four things believing an arrow commits you to. Each is separately falsifiable."""

    SOLE_CAUSE = "sole_cause"
    UNCONDITIONAL = "unconditional"
    DIRECTED = "directed"
    UNCONFOUNDED = "unconfounded"

    ALL: Tuple[str, ...] = (SOLE_CAUSE, UNCONDITIONAL, DIRECTED, UNCONFOUNDED)

    #: What each one asks, in the form the Master's plan states it.
    QUESTION: Dict[str, str] = {
        SOLE_CAUSE: "does anything else cause {effect}?",
        UNCONDITIONAL: "does {cause} cause {effect} only under some condition?",
        DIRECTED: "could {effect} cause {cause} instead?",
        UNCONFOUNDED: "is there a hidden variable behind both {cause} and {effect}?",
    }


class AssumptionStatus:
    #: Never examined. **This is the unknown-unknown surface** — the honest count of things the
    #: model rests on that nothing has ever looked at.
    UNTESTED = "untested"
    #: Examined and no counterexample found *in the record available*. Never "proven".
    HOLDS = "holds"
    #: Examined and contradicted. A discovery: structure the model did not have.
    REFUTED = "refuted"
    #: Examined and the record cannot decide — too few occurrences to say anything either way.
    #: Kept apart from ``UNTESTED`` because "I looked and could not tell" and "nobody looked" call
    #: for different next actions: one wants more data, the other wants attention.
    UNDECIDABLE = "undecidable"


#: How much lift a rival cause needs, relative to the arrow's own, before it refutes sole-cause.
#: A rival that barely registers is noise; one that explains the effect comparably well is a real
#: alternative and the arrow was never the only story.
_RIVAL_SHARE = 0.6

#: ``P(effect | cause)`` below which the arrow plainly depends on something unmodelled. Well under
#: 1.0 on purpose: a cause that produces its effect four times in five is not unconditional, and
#: demanding near-certainty before saying so would make this test unable to fire on real data.
_CONDITIONAL_FLOOR = 0.85

#: Occurrences of the cause below which nothing here can be decided. Under it every verdict is an
#: artefact of the sample rather than a fact about the world, and the honest answer is UNDECIDABLE.
_MIN_OCCURRENCES = 3


@dataclass
class Assumption:
    """One thing an arrow commits her to, and what the record made of it."""

    cause: str = ""
    effect: str = ""
    kind: str = AssumptionKind.SOLE_CAUSE
    status: str = AssumptionStatus.UNTESTED
    #: What the record said. Empty while untested — an assumption with a reason attached has been
    #: looked at, and one without has not, which is exactly the distinction being counted.
    why: str = ""
    #: The thing found, when something was: the rival cause, the reversal, the confounder. This is
    #: what makes a refutation a *discovery* rather than a complaint — she can name what she found.
    found: str = ""
    tested_at: float = 0.0

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.cause, self.effect, self.kind)

    @property
    def question(self) -> str:
        return AssumptionKind.QUESTION.get(self.kind, "").format(
            cause=self.cause, effect=self.effect)

    @property
    def untested(self) -> bool:
        return self.status == AssumptionStatus.UNTESTED

    @property
    def discovery(self) -> bool:
        """A refuted assumption is structure the model did not have."""
        return self.status == AssumptionStatus.REFUTED

    def to_dict(self) -> Dict[str, Any]:
        out = {"cause": self.cause, "effect": self.effect, "kind": self.kind,
               "status": self.status, "question": self.question}
        if self.why:
            out["why"] = self.why
        if self.found:
            out["found"] = self.found
        return out


class AssumptionMiner:
    """Enumerates what every arrow assumes, tests each against the record, and reports the rest.

    Stateful only in what it has already examined, so an assumption tested once is not re-counted
    as a fresh unknown on every pass.
    """

    def __init__(self, *, capacity: int = 512) -> None:
        self.capacity = max(16, int(capacity))
        self.assumptions: Dict[Tuple[str, str, str], Assumption] = {}
        self.mined = 0
        self.tested = 0
        self.refuted = 0
        self.undecidable = 0

    # ---- mine --------------------------------------------------------------- #
    def mine(self, world: Any, *, limit: int = 8) -> List[Assumption]:
        """Every assumption the arrows in ``world`` are resting on, newly surfaced ones first.

        Four per arrow, and they are *created untested*. That is deliberate: the moment an arrow
        is believed, four claims come into existence that nothing has examined, and the honest
        record of that is four untested assumptions rather than silence.
        """
        out: List[Assumption] = []
        try:
            for link in self._arrows(world)[:limit]:
                for kind in AssumptionKind.ALL:
                    key = (link.cause, link.effect, kind)
                    if key in self.assumptions:
                        continue
                    if len(self.assumptions) >= self.capacity:
                        break
                    assumption = Assumption(cause=link.cause, effect=link.effect, kind=kind)
                    self.assumptions[key] = assumption
                    self.mined += 1
                    out.append(assumption)
        except Exception:  # noqa: BLE001
            return out
        return out

    @staticmethod
    def _arrows(world: Any) -> List[Any]:
        try:
            return [l for l in (world.links(causal_only=True) or [])
                    if getattr(l, "cause", "") and getattr(l, "effect", "")
                    # A self-loop is not a causal claim. It arises whenever one kind of event
                    # repeats inside the window, and every test here is degenerate on it: its
                    # reverse is itself, so the direction test always reports the arrow as
                    # equally strong backwards and "refutes" a claim nobody made.
                    and getattr(l, "cause", "") != getattr(l, "effect", "")
                    and not getattr(l, "refuted", 0)]
        except Exception:  # noqa: BLE001
            return []

    # ---- test --------------------------------------------------------------- #
    def test(self, world: Any, *, limit: int = 4) -> List[Assumption]:
        """Examine the untested assumptions against the record. Returns what was decided."""
        decided: List[Assumption] = []
        try:
            pending = [a for a in self.assumptions.values() if a.untested][:limit]
            for assumption in pending:
                self._test_one(world, assumption)
                if assumption.status != AssumptionStatus.UNTESTED:
                    self.tested += 1
                    assumption.tested_at = time.time()
                    if assumption.status == AssumptionStatus.REFUTED:
                        self.refuted += 1
                    elif assumption.status == AssumptionStatus.UNDECIDABLE:
                        self.undecidable += 1
                    decided.append(assumption)
        except Exception:  # noqa: BLE001
            return decided
        return decided

    def _test_one(self, world: Any, assumption: Assumption) -> None:
        link = world.link(assumption.cause, assumption.effect)
        # Nothing below can mean anything on a handful of occurrences, and saying so is a
        # different answer from having not looked.
        if int(getattr(link, "cause_total", 0) or 0) < _MIN_OCCURRENCES:
            assumption.status = AssumptionStatus.UNDECIDABLE
            assumption.why = (f"{assumption.cause} has occurred "
                              f"{getattr(link, 'cause_total', 0)} time(s) — too few to examine "
                              f"what the arrow assumes")
            return
        {
            AssumptionKind.SOLE_CAUSE: self._test_sole_cause,
            AssumptionKind.UNCONDITIONAL: self._test_unconditional,
            AssumptionKind.DIRECTED: self._test_directed,
            AssumptionKind.UNCONFOUNDED: self._test_unconfounded,
        }[assumption.kind](world, assumption, link)

    # ---- the four tests ----------------------------------------------------- #
    def _test_sole_cause(self, world: Any, assumption: Assumption, link: Any) -> None:
        """"Does anything else cause this effect?" — another cause with comparable lift."""
        own = float(getattr(link, "lift", 0.0) or 0.0)
        rivals = [l for l in self._arrows(world)
                  if l.effect == assumption.effect and l.cause != assumption.cause]
        best = max(rivals, key=lambda l: float(getattr(l, "lift", 0.0) or 0.0), default=None)
        if best is None:
            assumption.status = AssumptionStatus.HOLDS
            assumption.why = "nothing else in the record has lift into this effect"
            return
        share = float(getattr(best, "lift", 0.0) or 0.0)
        if own <= 0.0 or share >= own * _RIVAL_SHARE:
            assumption.status = AssumptionStatus.REFUTED
            assumption.found = best.cause
            assumption.why = (f"{best.cause} also has lift into {assumption.effect} "
                              f"({share:.2f} against {own:.2f}) — this was never the only cause")
            return
        assumption.status = AssumptionStatus.HOLDS
        assumption.why = f"the nearest rival ({best.cause}) explains far less ({share:.2f})"

    def _test_unconditional(self, _world: Any, assumption: Assumption, link: Any) -> None:
        """"Only under condition X?" — the record answers *whether*, never *which*.

        And that is the whole value of the test. ``P(effect | cause)`` well below 1 says the arrow
        depends on something the model does not have, without needing to know what it is — which
        is precisely the shape of an unknown-unknown. Naming the condition is a further question,
        and one she can now ask because this one was answered.
        """
        conditional = float(getattr(link, "conditional", 0.0) or 0.0)
        if conditional >= _CONDITIONAL_FLOOR:
            assumption.status = AssumptionStatus.HOLDS
            assumption.why = (f"{assumption.cause} was followed by {assumption.effect} "
                              f"{conditional:.0%} of the time")
            return
        assumption.status = AssumptionStatus.REFUTED
        assumption.found = f"an unnamed condition on {assumption.cause}"
        assumption.why = (f"{assumption.cause} was followed by {assumption.effect} only "
                          f"{conditional:.0%} of the time — something unmodelled decides when")

    def _test_directed(self, world: Any, assumption: Assumption, link: Any) -> None:
        """"Could it run the other way?" — the reverse pair is in the same table."""
        reverse = world.link(assumption.effect, assumption.cause)
        forward = float(getattr(link, "lift", 0.0) or 0.0)
        backward = float(getattr(reverse, "lift", 0.0) or 0.0)
        if backward <= 0.0:
            assumption.status = AssumptionStatus.HOLDS
            assumption.why = f"{assumption.effect} never precedes {assumption.cause}"
            return
        if backward >= forward * _RIVAL_SHARE:
            assumption.status = AssumptionStatus.REFUTED
            assumption.found = f"{assumption.effect} → {assumption.cause}"
            assumption.why = (f"{assumption.effect} precedes {assumption.cause} about as often "
                              f"({backward:.2f} against {forward:.2f}) — the direction is not "
                              f"established by this record")
            return
        assumption.status = AssumptionStatus.HOLDS
        assumption.why = f"the reverse is much weaker ({backward:.2f} against {forward:.2f})"

    def _test_unconfounded(self, world: Any, assumption: Assumption, _link: Any) -> None:
        """"A hidden variable behind both?" — a third event that precedes each end.

        The one threat that makes a genuine correlation a false arrow, and the only one of the
        four that cannot be seen by looking at the pair alone.
        """
        arrows = self._arrows(world)
        into_cause = {l.cause for l in arrows if l.effect == assumption.cause}
        into_effect = {l.cause for l in arrows if l.effect == assumption.effect}
        shared = sorted(into_cause & into_effect - {assumption.cause, assumption.effect})
        if not shared:
            assumption.status = AssumptionStatus.HOLDS
            assumption.why = "nothing in the record precedes both ends"
            return
        assumption.status = AssumptionStatus.REFUTED
        assumption.found = shared[0]
        assumption.why = (f"{shared[0]} precedes both {assumption.cause} and "
                          f"{assumption.effect} — it may be producing both")

    # ---- what it found ------------------------------------------------------ #
    def unknown_unknowns(self) -> List[Assumption]:
        """Assumptions nothing has examined. The honest size of what she cannot see."""
        return [a for a in self.assumptions.values() if a.untested]

    def discoveries(self) -> List[Assumption]:
        """Refuted assumptions — structure the model did not have, each naming what was found."""
        return [a for a in self.assumptions.values() if a.discovery]

    def stats(self) -> Dict[str, Any]:
        held = sum(1 for a in self.assumptions.values()
                   if a.status == AssumptionStatus.HOLDS)
        return {
            "assumptions": len(self.assumptions),
            "mined": self.mined,
            "tested": self.tested,
            # The three knowledge states the plan asks for, over her own causal model.
            "unknown_unknowns": len(self.unknown_unknowns()),
            "holds": held,
            "refuted": self.refuted,
            "undecidable": self.undecidable,
            "by_kind": {kind: sum(1 for a in self.assumptions.values()
                                  if a.kind == kind and a.discovery)
                        for kind in AssumptionKind.ALL},
            "discoveries": [a.to_dict() for a in self.discoveries()[:4]],
        }
