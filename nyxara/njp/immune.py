"""NYXARA · njp/immune.py — a single new fact may not corrupt the graph (🛡, NJP V.44).

This organ exists because of a number, and the number is in `docs/CAPABILITIES.md` under V.41.
Loading ConceptNet's 198,455 English assertions beside the curated corpus took subjects from 4,382
to 130,274 — thirty times the breadth — and took ``recall`` on the curated corpus from **1.00 to
0.70**. Questions that had one clean answer stopped having one.

Nothing was broken. `Grounder.answer` behaved exactly as designed: two equally supported readings
and it declines to pick, which has been the right call since V.13. What went wrong is upstream of
that. **A crowd-sourced ``is_a`` was allowed to stand beside a curated one as though the two had
the same standing**, and once they were both in the store no downstream organ could tell them
apart — the provenance the ingest recorded says *how* a claim was got, never *who said it*.

So the broad corpus shipped switched off, which is honest and is not a fix.

The immune metaphor, taken literally
-------------------------------------

A body does not reject everything unfamiliar; it would starve. It does not accept everything
either. It **isolates what is both unfamiliar and potentially damaging**, tests it, and integrates
or clears it::

    new fact
       ↓
    is it even in contact with anything? ──no──→ ADMIT     (new subject: nothing to damage)
       ↓ yes
    does it compete, or merely add? ──adds──────→ ADMIT     (a fifth `causes` is a fifth cause)
       ↓ competes
    whose standing is higher? ──incumbent───────→ QUARANTINE
       ↓ challenger
    ADMIT, and quarantine the incumbent instead

Three things follow from taking that seriously.

**Most of a large corpus is harmless and is admitted.** A fact about a subject nothing else
mentions cannot degrade an answer, because there was no answer to degrade. This is why the organ
buys breadth *and* keeps precision rather than trading one for the other: the 126,000 subjects
ConceptNet adds are almost all of this kind.

**Only competing relations are guarded.** A second ``has_kind`` is a second kind and a second
``causes`` is a second cause — those are richness. A second ``is_a`` or ``means`` on the same
subject is a **rival answer to the same question**, and that is the whole of the regression.
:data:`COMPETING` is that list and it is short on purpose.

**Standing is earned, not declared.** A source begins at zero: its competing claims are isolated,
not because it is wrong but because nothing yet says it is right. It earns standing when
quarantined claims are confirmed and loses it when they are refuted, and
:meth:`Immune.standing` is a ratio over counts rather than a constant somebody chose. A curated
corpus that has been examined for twenty versions outranks a crowd on the evidence, which is a
different sentence from outranking it by decree.

Quarantine is not rejection
---------------------------

A quarantined fact is **kept, retrievable and marked**. It never reaches the fact store, so it
cannot degrade an answer, and it is not lost, so it can be released the moment something
corroborates it. :meth:`Immune.release` and :meth:`Immune.reject` are the two ways out and both
take a reason. Deleting on arrival would make the organ a filter; keeping it makes it an immune
system, and the difference shows up the first time the crowd turns out to be right.

What it may not do
------------------

**It may not quarantine what it has no incumbent for.** Isolation requires something to protect;
without one the fact is admitted, whatever its source.

**It may not decide truth.** Nothing here says the challenger is false. It says the incumbent
answers the question today and the challenger has not yet earned the right to make that question
unanswerable.

**It may not silently drop.** Every verdict carries a reason and every quarantined claim is
listed.

Pure standard library, deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence, Tuple

__all__ = ["Verdict", "Antigen", "Standing", "Immune", "COMPETING", "MANY_VALUED"]

#: Relations where a second value is a **rival answer to the same question** rather than more
#: knowledge. Short on purpose: everything not here is treated as additive, because guarding a
#: relation that is naturally many-valued would reject richness and buy nothing.
COMPETING: Tuple[str, ...] = (
    "is_a", "means", "located_in", "capital", "currency", "symbol", "formula", "unit",
    "birthplace", "inventor", "discoverer", "author",
)

#: Named for the docstring's sake and for a test: these are what the guard must **not** touch.
MANY_VALUED: Tuple[str, ...] = (
    "causes", "has_kind", "has_part", "has_property", "requires", "purpose", "consists_of",
    "involves", "capable_of", "occurs_when", "has_step",
)


class Verdict(str, Enum):
    ADMITTED = "admitted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


@dataclass
class Antigen:
    """An incoming claim that was isolated, and everything needed to judge it later."""

    subject: str
    predicate: str
    object: str
    source: str = ""
    confidence: float = 0.0
    #: What it would have displaced or crowded.
    incumbent: Tuple[str, ...] = ()
    reason: str = ""
    corroborations: int = 0

    def key(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def render(self) -> str:
        return (f"{self.subject} —{self.predicate}→ {self.object}"
                f"  (from {self.source or 'nowhere'}; incumbent: "
                f"{', '.join(self.incumbent) or 'none'})")

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object,
                "source": self.source, "incumbent": list(self.incumbent),
                "reason": self.reason, "corroborations": self.corroborations}


@dataclass
class Standing:
    """What a source has earned. Counts, not a constant somebody chose."""

    source: str
    admitted: int = 0
    quarantined: int = 0
    confirmed: int = 0
    refuted: int = 0

    @property
    def score(self) -> float:
        """Confirmed against refuted, with an unproven source at zero rather than at a half.

        Zero rather than a half is the decision. A source nobody has checked has not earned the
        right to overwrite one that has been examined for twenty versions, and starting it at
        parity would have let the whole of ConceptNet outrank the curated corpus on arrival by
        sheer volume.
        """
        tested = self.confirmed + self.refuted
        if not tested:
            return 0.0
        return round((self.confirmed - self.refuted) / tested, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "admitted": self.admitted,
                "quarantined": self.quarantined, "confirmed": self.confirmed,
                "refuted": self.refuted, "score": self.score}


class Immune:
    """Isolates what is both unfamiliar and able to do damage. Admits the rest."""

    def __init__(self, *, competing: Sequence[str] = COMPETING,
                 trusted: Sequence[str] = ()) -> None:
        self.competing = frozenset(competing)
        #: Sources whose standing is taken as established — the corpus already examined. A
        #: declaration, and it is the only one in the file, which is why it is empty by default
        #: and has to be passed in by whoever is doing the declaring.
        self.trusted = frozenset(trusted)
        self.held: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
        self.quarantine: List[Antigen] = []
        self.sources: Dict[str, Standing] = {}
        self.verdicts: Dict[str, int] = defaultdict(int)

    # ---- what she already holds ------------------------------------------ #
    def learn_incumbents(self, grounder: Any) -> int:
        """Read the existing store so the organ knows what there is to protect."""
        facts = getattr(grounder, "facts", None)
        if not isinstance(facts, dict):
            return 0
        count = 0
        for key, triples in facts.items():
            try:
                subject, predicate = key
            except Exception:  # noqa: BLE001
                continue
            for triple in triples:
                if getattr(triple, "superseded", False):
                    continue
                obj = str(getattr(triple, "object", "") or "").strip()
                if not obj:
                    continue
                source = str(getattr(triple, "source", "") or "")
                self.held[(str(subject).lower(), predicate)].append((obj, source))
                count += 1
        return count

    def note(self, subject: str, predicate: str, obj: str, *, source: str = "") -> None:
        """Record an admitted claim as an incumbent, so the next one is judged against it."""
        self.held[(str(subject).lower(), predicate)].append((obj, source))

    # ---- standing --------------------------------------------------------- #
    def standing(self, source: str) -> float:
        if source in self.trusted:
            return 1.0
        got = self.sources.get(source)
        return got.score if got else 0.0

    def _record(self, source: str, verdict: Verdict) -> Standing:
        got = self.sources.setdefault(source, Standing(source=source))
        if verdict is Verdict.ADMITTED:
            got.admitted += 1
        elif verdict is Verdict.QUARANTINED:
            got.quarantined += 1
        return got

    # ---- the decision ------------------------------------------------------ #
    def consider(self, subject: str, predicate: str, obj: str, *,
                 source: str = "", confidence: float = 0.0) -> Tuple[Verdict, str]:
        """One incoming claim. Returns the verdict and the reason for it."""
        key = (str(subject).lower(), predicate)
        incumbents = self.held.get(key, [])
        existing = [value for value, _src in incumbents]

        if not existing:
            reason = "nothing here to damage"                     # a new subject cannot degrade
        elif predicate not in self.competing:
            reason = f"{predicate} takes many values; this adds rather than competes"
        elif any(value.strip().lower() == obj.strip().lower() for value in existing):
            reason = "already held; corroboration"
        else:
            mine = self.standing(source)
            theirs = max((self.standing(src) for _v, src in incumbents), default=0.0)
            if mine > theirs:
                # The challenger outranks the incumbent. It is admitted and the **incumbent** goes
                # into quarantine — the immune response runs in both directions, which is the half
                # of the Master's design that stops the organ from being mere conservatism.
                for value, src in incumbents:
                    self.quarantine.append(Antigen(
                        subject=subject, predicate=predicate, object=value, source=src,
                        incumbent=(obj,),
                        reason=f"displaced by a better-standing source ({source})"))
                self.held[key] = []
                self.verdicts["displaced"] += len(incumbents)
                reason = f"outranks the incumbent ({mine:.2f} > {theirs:.2f})"
            else:
                got = Antigen(subject=subject, predicate=predicate, object=obj, source=source,
                              confidence=confidence, incumbent=tuple(existing),
                              reason=f"would make '{subject} {predicate} ?' unanswerable; "
                                     f"{source or 'this source'} has not earned that")
                self.quarantine.append(got)
                self._record(source, Verdict.QUARANTINED)
                self.verdicts[Verdict.QUARANTINED.value] += 1
                return Verdict.QUARANTINED, got.reason

        self.note(subject, predicate, obj, source=source)
        self._record(source, Verdict.ADMITTED)
        self.verdicts[Verdict.ADMITTED.value] += 1
        return Verdict.ADMITTED, reason

    # ---- the two ways out --------------------------------------------------- #
    def release(self, antigen: Antigen, *, why: str = "") -> Antigen:
        """Let a quarantined claim into the store, and credit its source for it."""
        if antigen in self.quarantine:
            self.quarantine.remove(antigen)
        self.note(antigen.subject, antigen.predicate, antigen.object, source=antigen.source)
        got = self.sources.setdefault(antigen.source, Standing(source=antigen.source))
        got.confirmed += 1
        antigen.reason = why or "released"
        return antigen

    def reject(self, antigen: Antigen, *, why: str = "") -> Antigen:
        """Clear a quarantined claim, and debit its source."""
        if antigen in self.quarantine:
            self.quarantine.remove(antigen)
        got = self.sources.setdefault(antigen.source, Standing(source=antigen.source))
        got.refuted += 1
        antigen.reason = why or "rejected"
        self.verdicts[Verdict.REJECTED.value] += 1
        return antigen

    def held_against(self, subject: str, predicate: str) -> List[Antigen]:
        """What was isolated about this pair. Kept, retrievable, and marked."""
        key = (str(subject).lower(), predicate)
        return [a for a in self.quarantine
                if (a.subject.lower(), a.predicate) == key]

    # ---- ingesting a whole corpus through it -------------------------------- #
    def filter_triples(self, rows: Iterable[Dict[str, Any]], *,
                       source: str = "") -> Iterable[Dict[str, Any]]:
        """Yield only what is admitted. The quarantine keeps the rest."""
        for row in rows:
            subject = str(row.get("subject") or "")
            predicate = str(row.get("predicate") or "")
            obj = str(row.get("object") or "")
            if not (subject and predicate and obj):
                continue
            verdict, _why = self.consider(subject, predicate, obj,
                                          source=str(row.get("source") or source),
                                          confidence=float(row.get("confidence") or 0.0))
            if verdict is Verdict.ADMITTED:
                yield row

    def report(self) -> Dict[str, Any]:
        return {"incumbents": sum(len(v) for v in self.held.values()),
                "quarantined": len(self.quarantine),
                "verdicts": dict(self.verdicts),
                "sources": {k: v.to_dict() for k, v in self.sources.items()}}
