"""NYXARA · njp/conceptgenome.py — what a concept is made of, before it is called anything (🧬, NJP V.47).

Named ``conceptgenome`` and not ``genome``, and the reason is a mistake rather than a preference:
this file was first written *as* ``njp/genome.py`` and **destroyed the module already there** —
:mod:`nyxara.njp.genome`, the Reasoning Genome of V.06, which compresses what she knows about her
own reasoning rather than about concepts. Fifteen tests went red and the suite caught it. Both
organs are about compressing a structure and neither is the other; the one that arrived second
takes the longer name.

:mod:`nyxara.njp.fusion` matches subgraphs across domains and finds that a control system and a
body's temperature regulation are the same shape. It matches on *edges*, which is enough to find
the resemblance and not enough to say what the two things **are**. The Master's next form of it:

    Concept
    ├── roles                 what fills the positions
    ├── relations             what connects them
    ├── constraints           what must hold
    ├── causal behavior       what it does
    ├── temporal behavior     when it does it
    ├── transformations       what it turns into
    ├── exceptions            where it does not hold
    └── invariants            what never changes

    Then compare concepts by **structural fingerprint**, not by words.
    Structure first. Label second.

That last line is the whole design and it is the opposite of how a fact store usually works. A
store keyed on spelling compares *heart* to *heart*; a genome compares what a heart **is made of**
to what a pump is made of, and finds them nearly the same object with one slot different.

Eight slots, and why they are not one bag of edges
---------------------------------------------------

Fusion's ``Pattern`` is a set of edges. That representation cannot distinguish *"this thing has a
part that does X"* from *"this thing turns into something that does X"*, and those are different
concepts with the same edge count. Splitting the record into named slots means the fingerprint can
say **where** two concepts differ, which is what makes a near-match informative rather than a
number.

So :meth:`Genome.compare` returns a :class:`Kinship` with a per-slot breakdown. Two concepts that
agree on roles, relations and causal behaviour and differ only in ``temporal`` are related in a way
worth knowing about; the single similarity score that would summarise them is exactly the thing
that hides it.

The fingerprint is vocabulary-free
-----------------------------------

:attr:`Genome.fingerprint` contains **no names** — only shapes: how many roles, which relation
kinds and how many of each, the degree profile, the count of constraints and exceptions. Two
genomes with the same fingerprint are *candidates*, and candidacy is cheap; kinship is then
established by an alignment, exactly as :mod:`nyxara.njp.fusion` does, because a fingerprint match
is a filter and never a finding.

**A concept is not named here.** :meth:`Kinship.gloss` reports the shape and what fills each role
on both sides and stops. Calling the common structure *"feedback regulation"* would be the module
supplying the insight it claims to have found — the same refusal ``fusion.Abstraction`` makes, for
the same reason.

What it may not do
------------------

**It may not invent a slot.** Every entry comes from a stated relation; an empty slot is empty.

**It may not compare on names.** :meth:`Genome.compare` never reads a node's spelling except to
report which node filled which role.

**It may not call a partial match a match.** :attr:`Kinship.aligned` is False unless a bijection
carries every relation, and the per-slot scores are reported beside it rather than instead of it.

Pure standard library, deterministic.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Genome", "Kinship", "SLOTS", "read_genome", "SLOT_RELATIONS"]

#: The eight slots, in the Master's order. Named as a tuple so a report cannot silently gain one.
SLOTS: Tuple[str, ...] = (
    "roles", "relations", "constraints", "causal", "temporal",
    "transformations", "exceptions", "invariants",
)

#: Which stored relations fill which slot. A relation in no row goes nowhere — an unclassified
#: edge is not quietly swept into ``relations``, because that slot would then absorb everything
#: and the fingerprint would stop discriminating.
SLOT_RELATIONS: Dict[str, Tuple[str, ...]] = {
    "roles": ("has_part", "consists_of", "has_role"),
    "relations": ("part_of", "located_in", "involves"),
    "constraints": ("requires", "excludes", "needs"),
    "causal": ("causes", "produces", "increases", "decreases", "purpose"),
    "temporal": ("occurs_when", "precedes", "follows", "has_step", "has_stage"),
    "transformations": ("becomes", "turns_into", "converts_to"),
    "exceptions": ("except_when", "unless", "fails_when"),
    "invariants": ("always", "conserved", "has_property"),
}


@dataclass
class Genome:
    """One concept, as eight slots of stated relations. Holds no name it did not read."""

    subject: str
    slots: Dict[str, Tuple[Tuple[str, str], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for slot in SLOTS:
            self.slots.setdefault(slot, ())

    @property
    def nodes(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for slot in SLOTS:
            for _relation, obj in self.slots[slot]:
                if obj not in seen:
                    seen.append(obj)
        return tuple(seen)

    @property
    def size(self) -> int:
        return sum(len(self.slots[slot]) for slot in SLOTS)

    @property
    def filled(self) -> Tuple[str, ...]:
        return tuple(slot for slot in SLOTS if self.slots[slot])

    @property
    def fingerprint(self) -> Tuple[Any, ...]:
        """No names. Only shapes — which is what makes two vocabularies comparable at all."""
        out: List[Any] = []
        for slot in SLOTS:
            counts: Dict[str, int] = defaultdict(int)
            for relation, _obj in self.slots[slot]:
                counts[relation] += 1
            out.append((slot, tuple(sorted(counts.items()))))
        return tuple(out)

    def edges(self) -> FrozenSet[Tuple[str, str, str]]:
        """The genome as triples over its own nodes, for alignment."""
        return frozenset((self.subject, relation, obj)
                         for slot in SLOTS for relation, obj in self.slots[slot])

    def render(self) -> str:
        lines = [f"CONCEPT {self.subject}"]
        for slot in SLOTS:
            entries = self.slots[slot]
            if entries:
                body = ", ".join(f"{relation}={obj}" for relation, obj in entries)
                lines.append(f"  {slot:16} {body}")
        return "\n".join(lines)

    def compare(self, other: "Genome") -> "Kinship":
        """Structural kinship, slot by slot, with the alignment that establishes it."""
        return _compare(self, other)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "size": self.size, "filled": list(self.filled),
                "slots": {k: [list(p) for p in v] for k, v in self.slots.items()}}


@dataclass
class Kinship:
    """How two concepts are related, and **where** they differ."""

    left: str
    right: str
    aligned: bool = False
    mapping: Dict[str, str] = field(default_factory=dict)
    per_slot: Dict[str, float] = field(default_factory=dict)
    differs: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.per_slot:
            return 0.0
        return round(sum(self.per_slot.values()) / len(self.per_slot), 4)

    def gloss(self) -> str:
        """The shape and what fills it on both sides. **No name for the common structure.**"""
        lines = [f"{self.left} ~ {self.right}  "
                 f"{'aligned' if self.aligned else 'not aligned'}  ({self.score:.2f})"]
        for slot in SLOTS:
            got = self.per_slot.get(slot)
            if got is None:
                continue
            mark = "=" if got == 1.0 else ("~" if got > 0 else "≠")
            lines.append(f"  {slot:16} {mark} {got:.2f}")
        if self.mapping:
            lines.append("  roles")
            for a, b in sorted(self.mapping.items()):
                lines.append(f"    {a} ↔ {b}")
        if self.differs:
            lines.append(f"  differs in: {', '.join(self.differs)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left, "right": self.right, "aligned": self.aligned,
                "score": self.score, "per_slot": dict(self.per_slot),
                "differs": list(self.differs), "mapping": dict(self.mapping)}


# --------------------------------------------------------------------------- #
# Reading one out of a store
# --------------------------------------------------------------------------- #
def read_genome(explainer: Any, subject: str) -> Genome:
    """Build a concept's genome from what the store actually says about it.

    A relation that belongs to no slot is **dropped**, not swept into ``relations``. That slot
    would otherwise absorb every unclassified edge and the fingerprint would stop discriminating —
    which is the same failure mode ``related_to`` has in ConceptNet and the reason
    ``prepare_conceptnet.py`` refuses it.
    """
    got = Genome(subject=subject)
    slots: Dict[str, List[Tuple[str, str]]] = {slot: [] for slot in SLOTS}
    for slot, relations in SLOT_RELATIONS.items():
        for relation in relations:
            try:
                values = explainer._out(subject, relation)
            except Exception:  # noqa: BLE001
                values = []
            for obj, _confidence in values:
                slots[slot].append((relation, str(obj)))
    got.slots = {slot: tuple(sorted(set(entries))) for slot, entries in slots.items()}
    return got


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _compare(left: Genome, right: Genome) -> Kinship:
    out = Kinship(left=left.subject, right=right.subject)
    for slot in SLOTS:
        here = [relation for relation, _obj in left.slots[slot]]
        there = [relation for relation, _obj in right.slots[slot]]
        if not here and not there:
            continue
        shared = len(set(here) & set(there))
        total = len(set(here) | set(there))
        out.per_slot[slot] = round(shared / total, 4) if total else 0.0
        if out.per_slot[slot] < 1.0:
            out.differs.append(slot)

    # Alignment: a bijection over the two genomes' object nodes carrying every relation. Names are
    # never compared — only which relation connects which position — which is the whole point.
    ours, theirs = list(left.nodes), list(right.nodes)
    if len(ours) != len(theirs) or not ours:
        return out
    wanted = {(relation, obj) for slot in SLOTS for relation, obj in left.slots[slot]}
    target = {(relation, obj) for slot in SLOTS for relation, obj in right.slots[slot]}
    if len(wanted) != len(target):
        return out
    for candidate in itertools.permutations(theirs, len(ours)):
        mapping = dict(zip(ours, candidate))
        carried = {(relation, mapping[obj]) for relation, obj in wanted}
        if carried == target:
            out.aligned = True
            out.mapping = mapping
            break
    return out
