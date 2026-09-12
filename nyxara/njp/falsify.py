"""NYXARA · njp/falsify.py — what would end this belief, and has it already happened (🔪, NJP V.02).

:attr:`~nyxara.njp.beliefs.Belief.falsifier` has been written on every belief she holds since the
ledger existed, and :attr:`~nyxara.njp.beliefs.Belief.falsifiable` is ``bool(falsifier)``. Both
were true of everything. Measured over a thirty-six turn session: **36 beliefs, 36 falsifiable,
0 with hard evidence** — a property that is true of every member of a set discriminates nothing,
and this one was true by construction because one template filled it in for all of them.

The template's output was not even well formed. It read
``f"an observation in which {subject} does not {predicate} {obj}"``, and a predicate is a relation
name rather than a verb, so the record actually contained:

    an observation in which fire does not causes heat
    an observation in which plant does not water 2
    an observation in which sparrow does not is_a bird

*"What evidence would make me abandon this?"* is the question the ledger was built to hold, and
the answer it held was ungrammatical, unspecific, and — this is the part that matters — **never
looked for**. Nothing in the package ever went and checked whether the observation a falsifier
names had already occurred. A falsifier nobody searches for is a sentence, not a test, and a
system whose beliefs can only gain support is not doing epistemology.

**Two properties, deliberately separate.** :attr:`~nyxara.njp.beliefs.Belief.falsifiable` keeps
its meaning — *she stated what would kill this* — and :attr:`Falsifier.checkable` is the stronger
one this module adds: *and there is a record I can go and look in*. Collapsing them would either
make every belief checkable, which is the defect above, or make a stated-but-unsearchable
falsifier count as no falsifier at all, which throws away a real distinction.

**Three killers, each read off a record that already exists.** None of them is a new store and
none is a heuristic:

===================  ========================================================================
kind                 the observation that would end the belief, and where it is looked for
===================  ========================================================================
``different_value``  a **functional** predicate holding a different object for the same
                     subject — ``grounder._lookup``. "Ravi works at A" dies on "Ravi works at
                     B", and only for predicates that can have one answer: two properties are
                     not a contradiction.
``absent_effect``    the cause occurring without the effect, often enough that it is not
                     noise — ``world.link``. This is the one that kills a causal claim.
``reversed``         the arrow established in the other direction, by testimony or by
                     intervention — ``universe.relations``. Her own inference losing to
                     evidence from outside it.
===================  ========================================================================

**Not finding it is a result.** :attr:`Verdict.checked` and :attr:`Verdict.found` are counted
apart for the same reason ``asked`` and ``scored`` are everywhere else in this package: a belief
that survived a search is in a different state from one nobody searched, and a single number that
merges them can only report the wrong one. ``survived`` is a claim this module is allowed to make
precisely because ``checked`` is what it is measured against.

Pure standard library. Duck-typed on every organ it reads, and fail-soft: a record it cannot read
is a search that found nothing, which is what it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Killer", "Falsifier", "Verdict", "TheoryKiller"]

#: Confidence at or above which a belief is worth trying to kill. The strongest rather than the
#: weakest, on :meth:`~nyxara.njp.adversary.SelfAttacker.attack_strongest`'s reasoning: a weak
#: claim being wrong costs little and she already doubts it.
_THRESHOLD = 0.6

#: How many beliefs one hunt examines. Bounded like everything else here.
_LIMIT = 4

#: Times the cause must have occurred before its effect's absence means anything. Below this,
#: "the effect did not follow" is a small sample rather than a refutation.
_MIN_OCCURRENCES = 4

#: Share of the cause's occurrences the effect must fail to follow before the causal claim is
#: counted as refuted. Not zero: a genuine cause with an occasional miss is still a cause, and
#: demanding perfection would retract every real arrow she has.
_MISS_RATE = 0.75

#: Verdicts kept for the record.
_HISTORY = 128


class Killer:
    """What kind of observation would end a belief."""

    DIFFERENT_VALUE = "different_value"
    ABSENT_EFFECT = "absent_effect"
    REVERSED = "reversed"
    #: Stated, but naming nothing any record could be searched for. Honest and common: most of
    #: what she is told is a property, and a property's negation is not an observation she can
    #: go and look up.
    NONE = ""


#: Predicates that can hold exactly one object for a subject, so a second one is a contradiction
#: rather than a second fact. Imported from the grounder where it is already defined and already
#: enforced — a second copy here would be a second answer to "what is functional", and the two
#: would drift.
def _functional() -> frozenset:
    try:
        from nyxara.njp.grounding import _FUNCTIONAL
        return frozenset(_FUNCTIONAL)
    except Exception:  # noqa: BLE001
        return frozenset()


#: Predicates that assert an arrow, so the reverse being established is what ends them.
_CAUSAL = frozenset({"causes", "cause", "leads_to", "results_in", "produces"})


@dataclass(frozen=True)
class Falsifier:
    """The observation that would end a belief, in a form something can go and look for."""

    kind: str = Killer.NONE
    subject: str = ""
    predicate: str = ""
    object: str = ""

    @property
    def checkable(self) -> bool:
        """Is there a record that could contain it? The property the prose string could not have."""
        return bool(self.kind) and bool(self.subject)

    def stated(self) -> str:
        """The falsifier as a sentence — well formed, and specific about what would end it.

        Replaces a template that produced *"an observation in which fire does not causes heat"*.
        One source for the sentence, so the record and the search cannot describe different things.
        """
        subject, predicate, obj = self.subject, self.predicate, self.object
        if self.kind == Killer.DIFFERENT_VALUE:
            return (f"an observation giving {subject} a different {predicate.replace('_', ' ')} "
                    f"than {obj}").strip()
        if self.kind == Killer.ABSENT_EFFECT:
            return f"{subject} occurring repeatedly without {obj} following".strip()
        if self.kind == Killer.REVERSED:
            return f"evidence that {obj} drives {subject} rather than the other way".strip()
        return (f"an observation contradicting that {subject} "
                f"{predicate.replace('_', ' ')} {obj}").strip()

    @classmethod
    def of(cls, claim: str) -> "Falsifier":
        """Derive the killer from the claim's own shape. ``subject predicate object``.

        The claim strings this reads are built by ``field._record_beliefs`` as exactly that, so
        the structure is recoverable without a parser. A claim that is not of that shape gets
        :attr:`Killer.NONE` — stated, unsearchable, and honestly marked so rather than given a
        killer that would never fire.
        """
        try:
            parts = str(claim or "").strip().split()
            if len(parts) < 2:
                return cls()
            subject, predicate = parts[0], parts[1]
            obj = " ".join(parts[2:])
            if predicate in _CAUSAL and obj:
                return cls(kind=Killer.ABSENT_EFFECT, subject=subject,
                           predicate=predicate, object=obj)
            if predicate in _functional() and obj:
                return cls(kind=Killer.DIFFERENT_VALUE, subject=subject,
                           predicate=predicate, object=obj)
            return cls(subject=subject, predicate=predicate, object=obj)
        except Exception:  # noqa: BLE001
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "predicate": self.predicate,
                "object": self.object, "checkable": self.checkable, "stated": self.stated()}


@dataclass
class Verdict:
    """One belief, hunted. ``checked`` and ``found`` are counted apart on purpose."""

    claim: str = ""
    confidence: float = 0.0
    falsifier: Optional[Falsifier] = None
    checked: bool = False              # a record was actually searched
    found: bool = False                # the killing observation was in it
    evidence: str = ""
    retracted: bool = False
    why: str = ""

    @property
    def survived(self) -> bool:
        """Searched for and not found. Only meaningful because ``checked`` is separate."""
        return self.checked and not self.found

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:140], "confidence": round(self.confidence, 4),
                "falsifier": self.falsifier.to_dict() if self.falsifier else None,
                "checked": self.checked, "found": self.found, "survived": self.survived,
                "retracted": self.retracted, "evidence": self.evidence[:160],
                "why": self.why[:160]}


class TheoryKiller:
    """Goes looking for the observation that would end what she is most committed to."""

    def __init__(self, brain: Any = None, *, threshold: float = _THRESHOLD,
                 limit: int = _LIMIT, miss_rate: float = _MISS_RATE) -> None:
        self.brain = brain
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.limit = max(1, int(limit))
        self.miss_rate = max(0.0, min(1.0, float(miss_rate)))

        self.hunts = 0
        self.examined = 0
        self.checkable = 0          # of those, how many named a record to search
        self.checked = 0            # of those, how many were actually searched
        self.found = 0              # of those, how many turned up their own killer
        self.retracted = 0
        self.history: List[Verdict] = []
        #: Claims already hunted, so one belief is one search rather than a new one every cycle.
        self._seen: Dict[str, float] = {}

    # ---- what to go after ---------------------------------------------------- #
    def targets(self, limit: Optional[int] = None) -> List[Any]:
        """The beliefs she is most committed to, strongest first, least recently hunted.

        The strongest rather than the weakest, and for the reason
        :meth:`~nyxara.njp.adversary.SelfAttacker.attack_strongest` gives: a claim she is about to
        rely on is the one worth trying to break, and it is also the one nothing else will ever
        question.
        """
        try:
            ledger = getattr(self.brain, "beliefs", None)
            if ledger is None:
                return []
            held = [b for b in ledger.beliefs.values()
                    if float(getattr(b, "confidence", 0.0)) >= self.threshold
                    and getattr(b, "outcome", None) is not False]
            held.sort(key=lambda b: (self._seen.get(getattr(b, "claim", ""), 0.0),
                                     -float(getattr(b, "confidence", 0.0))))
            return held[: (self.limit if limit is None else max(1, int(limit)))]
        except Exception:  # noqa: BLE001
            return []

    # ---- the hunt -------------------------------------------------------------- #
    def hunt(self, limit: Optional[int] = None) -> List[Verdict]:
        """Search her own record for what would end each of her strongest beliefs."""
        out: List[Verdict] = []
        try:
            self.hunts += 1
            ledger = getattr(self.brain, "beliefs", None)
            for belief in self.targets(limit):
                claim = str(getattr(belief, "claim", "") or "")
                if not claim:
                    continue
                self._seen[claim] = time.time()
                self.examined += 1
                verdict = Verdict(claim=claim,
                                  confidence=float(getattr(belief, "confidence", 0.0)))
                verdict.falsifier = Falsifier.of(claim)
                if not verdict.falsifier.checkable:
                    verdict.why = "stated, but naming no record that could be searched"
                    out.append(verdict)
                    self._remember(verdict)
                    continue
                self.checkable += 1
                found, evidence, searched = self._look(verdict.falsifier)
                verdict.checked = searched
                verdict.found = found
                verdict.evidence = evidence
                if searched:
                    self.checked += 1
                if found:
                    self.found += 1
                    verdict.why = f"her own record contains it: {evidence}"
                    # The objective stops being "strengthen the belief". This is that sentence
                    # made operational: the killing observation was already there, so the belief
                    # goes, and what she used to think stays on the record as data about her.
                    if ledger is not None and ledger.retract(
                            claim, why=f"falsified: {evidence}"[:160]) is not None:
                        verdict.retracted = True
                        self.retracted += 1
                elif searched:
                    verdict.why = "searched for and not found"
                else:
                    verdict.why = "no record attached to search"
                out.append(verdict)
                self._remember(verdict)
            return out
        except Exception:  # noqa: BLE001 — a failed hunt kills nothing and breaks no turn
            return out

    def _remember(self, verdict: Verdict) -> None:
        self.history.append(verdict)
        del self.history[:-_HISTORY]

    # ---- the three searches ------------------------------------------------------ #
    def _look(self, falsifier: Falsifier) -> Tuple[bool, str, bool]:
        """``(found, evidence, searched)`` — and ``searched`` is why the other two mean anything."""
        try:
            if falsifier.kind == Killer.DIFFERENT_VALUE:
                return self._look_different(falsifier)
            if falsifier.kind == Killer.ABSENT_EFFECT:
                return self._look_absent(falsifier)
            if falsifier.kind == Killer.REVERSED:
                return self._look_reversed(falsifier)
            return (False, "", False)
        except Exception:  # noqa: BLE001
            return (False, "", False)

    def _look_different(self, falsifier: Falsifier) -> Tuple[bool, str, bool]:
        """A functional predicate already holding a different object for the same subject."""
        grounder = getattr(self.brain, "grounder", None)
        if grounder is None or not hasattr(grounder, "_lookup"):
            return (False, "", False)
        want = falsifier.object.strip().lower()
        for triple in grounder._lookup(falsifier.subject, falsifier.predicate) or ():
            if getattr(triple, "superseded", False):
                continue                # already revised away; it cannot kill anything twice
            got = str(getattr(triple, "object", "") or "").strip().lower()
            if got and got != want:
                return (True, f"{falsifier.subject} {falsifier.predicate} {got}", True)
        return (False, "", True)

    def _look_absent(self, falsifier: Falsifier) -> Tuple[bool, str, bool]:
        """The cause occurring repeatedly without the effect following.

        A rate rather than a single miss, and the rate is deliberately not 1.0. A genuine cause
        with an occasional miss is still a cause; requiring the effect every single time would
        retract every real arrow she has the first time anything else interfered.
        """
        world = getattr(self.brain, "world", None)
        if world is None or not hasattr(world, "link"):
            return (False, "", False)
        link = world.link(falsifier.subject, falsifier.object)
        total = int(getattr(link, "cause_total", 0) or 0)
        if total < _MIN_OCCURRENCES:
            # Too thin to mean anything, and saying so is a different answer from having looked
            # and found nothing — so this reports "not searched", not "survived".
            return (False, "", False)
        together = int(getattr(link, "together", 0) or 0)
        missed = total - together
        if missed / total >= self.miss_rate:
            return (True, f"{falsifier.subject} occurred {total}× and {falsifier.object} "
                          f"followed {together}×", True)
        return (False, "", True)

    def _look_reversed(self, falsifier: Falsifier) -> Tuple[bool, str, bool]:
        """The arrow established the other way by something stronger than her own inference."""
        universe = getattr(self.brain, "universe", None)
        if universe is None or not hasattr(universe, "relations"):
            return (False, "", False)
        try:
            from nyxara.njp.universe import Orientation
        except Exception:  # noqa: BLE001
            return (False, "", False)
        forward = universe.relations.get((falsifier.subject, falsifier.object))
        back = universe.relations.get((falsifier.object, falsifier.subject))
        if back is None:
            return (False, "", True)
        stronger = {Orientation.ASSERTED, Orientation.VERIFIED}
        if (getattr(back, "orientation", "") in stronger
                and getattr(forward, "orientation", "") not in stronger):
            return (True, f"{falsifier.object} → {falsifier.subject} is "
                          f"{getattr(back, 'orientation', '')}", True)
        return (False, "", True)

    # ---- reporting ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        return {
            "hunts": self.hunts,
            "examined": self.examined,
            # Three denominators, never merged. `examined` counts beliefs looked at, `checkable`
            # those naming a record, `checked` those a record was actually read for — and only
            # against the last of those does `found` mean anything at all.
            "checkable": self.checkable,
            "checked": self.checked,
            "found": self.found,
            "retracted": self.retracted,
            "survived": max(0, self.checked - self.found),
            "kill_rate": (round(self.found / self.checked, 4) if self.checked else None),
            "last": last.to_dict() if last is not None else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"counters": {"hunts": self.hunts, "examined": self.examined,
                             "checkable": self.checkable, "checked": self.checked,
                             "found": self.found, "retracted": self.retracted}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            c = (d or {}).get("counters") or {}
            self.hunts = int(c.get("hunts", 0))
            self.examined = int(c.get("examined", 0))
            self.checkable = int(c.get("checkable", 0))
            self.checked = int(c.get("checked", 0))
            self.found = int(c.get("found", 0))
            self.retracted = int(c.get("retracted", 0))
        except Exception:  # noqa: BLE001
            pass
