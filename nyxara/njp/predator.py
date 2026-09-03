"""NYXARA · njp/predator.py — the process that goes after her own explanations (☠🧭, NJP V.39).

:mod:`nyxara.njp.adversary` attacks a **belief** — one claim, with its evidence and its track
record. Nothing attacked an **explanation**, and the difference is not academic: an explanation is
a composition, every fact in it can be true, and the composition can still be false. V.36 shipped
one — ``heart → atrium → bringing daylight into the middle of a deep plan`` — where all three
edges were correct and the sentence was not.

:mod:`nyxara.njp.explaingauntlet` measured what that costs on two papers where the walk scored
**0.000 and could not have scored anything else**:

``contradiction``
    Two causes of one effect, and the store also says the two exclude each other. The walk
    returned both, as a list, cheerfully. Listing two claims is not noticing that they cannot both
    hold, and a reader given that list is worse off than one given nothing, because the list looks
    like knowledge.

``legs``
    Two chains reaching one target, where the target ``requires`` a node on each. They are a
    **conjunction** and the walk reported them as alternatives. *"Because A, or because B"* and
    *"because A and B together"* are different claims built from the same two names.

Both are one gap: the walk **builds** and nothing **examines**. So this module is the second half
of the loop the Master drew::

    facts → chains → EXPLANATION
                          ↓
                       PREDATOR
                          ↓
              "how could this be wrong?"
                          ↓
        exclusion · conjunction · unsupported assumption · counterexample
                          ↓
                  survives / repaired / withdrawn

Four attacks, and what each one is looking for
----------------------------------------------

**Exclusion.** Two surviving chains whose roots the store says cannot both hold — ``A excludes B``,
or a stated denial of one by the other. The verdict is not *pick the better one*. It is
:attr:`Survival.conflict`, and the answer she gives says so: *these two cannot both hold and
nothing here settles which.* This is the same refusal ``Grounder.answer`` has made since V.13 when
two triples are equally supported, moved up a level from a fact to a composition.

**Conjunction.** Two chains reaching the target where the target ``requires`` something on each.
Then neither is *an* explanation; together they are *the* explanation.
:attr:`Survival.joint` says so, and the rendering changes from *"because A"* / *"because B"* to
*"because A and B together"* — because the walk had the facts to know that and said the wrong
thing with them.

**Unsupported assumption.** A hop whose confidence is at the floor, or a link stated only as a
defeasible generalisation, carrying a chain that is then stated flatly. The chain is not withdrawn
— it is **marked**, and the mark travels with it, so a reader can see which link the whole thing
rests on.

**Counterexample.** A case in the store where the chain's cause holds and its effect does not.
Cheap to look for and rare in a curated corpus, and it is here because the day it fires it is the
most valuable of the four: it is the store contradicting a derivation out of its own contents.

What the predator may not do
----------------------------

**It may not invent.** Every attack is grounded in a stated fact — an ``excludes`` edge, a
``requires`` edge, a confidence, a counterexample. An attack that needed a fact nobody stated
would be the predator confabulating in order to accuse, which is worse than the confabulation it
is hunting.

**It may not resolve a conflict it found.** Reporting *"A and B cannot both hold"* and then
answering with A is worse than never having noticed. :meth:`Predator.attack` returns the finding;
choosing is somebody else's job and in this package it is usually nobody's, which is correct.

**It runs after the walk and never inside it.** A predator wired into chain-building would
suppress the evidence that corrects it — the exact self-confirming failure V.26's figurative guard
was built out of and then had to be rescued from. It sees the finished explanation, the same one a
reader would.

Pure standard library, deterministic, and it holds no knowledge of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Attack", "Survival", "Predator", "EXCLUDES", "KINDS"]

#: Relations that state *these two cannot both hold*. ``excludes`` is the direct form; the others
#: are how the same claim is written when the author was describing one of the pair.
EXCLUDES: Tuple[str, ...] = ("excludes", "incompatible_with", "rules_out", "contradicts")

#: What an attack can find. Kept as names rather than an enum so a finding can be added by a
#: caller without editing this module.
KINDS: Tuple[str, ...] = ("exclusion", "conjunction", "assumption", "counterexample")


@dataclass
class Attack:
    """One finding against one explanation, with the stated fact it rests on."""

    kind: str
    finding: str
    #: The nodes the finding is about — the two roots of an exclusion, the two legs of a
    #: conjunction, the weak hop of an assumption.
    about: Tuple[str, ...] = ()
    #: The triple in the store that licenses this attack. An attack with no evidence is not made.
    evidence: Tuple[str, str, str] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "finding": self.finding, "about": list(self.about),
                "evidence": list(self.evidence)}


@dataclass
class Survival:
    """What became of an explanation once something went after it."""

    topic: str
    attacks: List[Attack] = field(default_factory=list)

    @property
    def survived(self) -> bool:
        return not self.attacks

    @property
    def conflict(self) -> bool:
        return any(a.kind == "exclusion" for a in self.attacks)

    @property
    def joint(self) -> bool:
        return any(a.kind == "conjunction" for a in self.attacks)

    def of_kind(self, kind: str) -> List[Attack]:
        return [a for a in self.attacks if a.kind == kind]

    def note(self) -> str:
        """The line a reader is owed, or "" when the explanation survived intact."""
        return "\n".join(a.finding for a in self.attacks)

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "survived": self.survived, "conflict": self.conflict,
                "joint": self.joint, "attacks": [a.to_dict() for a in self.attacks]}


class Predator:
    """Goes after a finished explanation using nothing but what the store already says."""

    def __init__(self, explainer: Any) -> None:
        self.explainer = explainer

    # ---- helpers over the store ----------------------------------------- #
    def _out(self, subject: str, relation: str) -> List[str]:
        try:
            return [obj for obj, _c in self.explainer._out(subject, relation)]
        except Exception:  # noqa: BLE001
            return []

    def _key(self, text: str) -> str:
        try:
            return self.explainer._key(text)
        except Exception:  # noqa: BLE001
            return " ".join(str(text or "").split()).lower()

    def _excludes(self, first: str, second: str) -> Optional[Tuple[str, str, str]]:
        """Does the store say these two cannot both hold? Returns the triple that says it."""
        want = self._key(second)
        for relation in EXCLUDES:
            for obj in self._out(first, relation):
                if self._key(obj) == want:
                    return (first, relation, obj)
        return None

    # ---- the attacks ---------------------------------------------------- #
    def attack(self, explanation: Any) -> Survival:
        """Every attack, on one explanation. The order is stable so a report is stable."""
        out = Survival(topic=str(getattr(explanation, "topic", "") or ""))
        chains = list(getattr(explanation, "chains", ()) or ())
        if not chains:
            return out
        out.attacks.extend(self._exclusion(chains))
        out.attacks.extend(self._conjunction(out.topic, chains))
        out.attacks.extend(self._assumption(chains))
        out.attacks.extend(self._counterexample(chains))
        return out

    def _exclusion(self, chains: Sequence[Any]) -> List[Attack]:
        """Two chains whose roots the store says cannot both hold.

        The **roots**, not every node: an exclusion between two middles is a claim about those
        middles and may be irrelevant to the question. The root is what the explanation actually
        offers as the answer, and two answers that exclude each other is the case worth reporting.
        """
        found: List[Attack] = []
        seen: Set[Tuple[str, str]] = set()
        for i, first in enumerate(chains):
            for second in chains[i + 1:]:
                a, b = str(first.head), str(second.head)
                if self._key(a) == self._key(b):
                    continue
                pair = tuple(sorted((self._key(a), self._key(b))))
                if pair in seen:
                    continue
                evidence = self._excludes(a, b) or self._excludes(b, a)
                if evidence is None:
                    continue
                seen.add(pair)
                found.append(Attack(
                    kind="exclusion", about=(a, b), evidence=evidence,
                    finding=f"{a} and {b} cannot both hold — the store says "
                            f"{evidence[0]} {evidence[1]} {evidence[2]} — and nothing here "
                            f"settles which"))
        return found

    def _conjunction(self, topic: str, chains: Sequence[Any]) -> List[Attack]:
        """Chains that are conjuncts rather than alternatives.

        The evidence is the topic's own ``requires`` edges: when the target requires something on
        two different chains, neither chain is sufficient and reporting them side by side as
        *"because A"* and *"because B"* is a false reading of facts she already had.
        """
        needs = [self._key(n) for n in self._out(topic, "requires")]
        if len(needs) < 2:
            return []
        by_chain: List[Tuple[Any, Set[str]]] = []
        for chain in chains:
            hit = {self._key(n) for n in chain.nodes} & set(needs)
            if hit:
                by_chain.append((chain, hit))
        if len(by_chain) < 2:
            return []
        covered: Set[str] = set()
        for _chain, hit in by_chain:
            covered |= hit
        if len(covered) < 2:
            return []
        roots = tuple(str(chain.head) for chain, _hit in by_chain)
        return [Attack(
            kind="conjunction", about=roots,
            evidence=(topic, "requires", sorted(covered)[0]),
            finding=f"{topic} requires " + " and ".join(sorted(covered)) +
                    f" — so {' and '.join(roots)} are jointly required, not alternatives")]

    def _assumption(self, chains: Sequence[Any]) -> List[Attack]:
        """A chain resting on a link the store itself is unsure of.

        Not a withdrawal. The chain stands and carries a mark, because a defeasible link is how
        most true generalisations are stated and refusing them would leave her able to explain
        almost nothing.
        """
        found: List[Attack] = []
        for chain in chains:
            weakest = min(chain.steps, key=lambda s: float(s.confidence), default=None)
            if weakest is None or float(weakest.confidence) >= 0.7:
                continue
            found.append(Attack(
                kind="assumption", about=(weakest.subject, weakest.object),
                evidence=weakest.as_tuple(),
                finding=f"this rests on {weakest.subject} {weakest.relation} "
                        f"{weakest.object}, held at {float(weakest.confidence):.2f}"))
        return found

    def _counterexample(self, chains: Sequence[Any]) -> List[Attack]:
        """A stated case where the chain's cause holds and its effect does not.

        Looked for through ``despite`` and ``without`` edges, which is how the corpus states an
        exception when it states one at all. Rare, and worth the two lines the day it fires: it is
        the store contradicting a derivation out of its own contents.
        """
        found: List[Attack] = []
        for chain in chains:
            for step in chain.steps:
                # The triple's own direction, not the walk's. A `causes` triple has its cause as
                # subject and its effect as object however the chain traversed it, and reading
                # `Step.source`/`Step.target` here — which *do* flip with the walk — looked for
                # `cause despite effect` and found nothing, on every backward chain, which is all
                # of them on a why-question.
                cause, effect = step.subject, step.object
                for relation in ("despite", "without"):
                    for obj in self._out(effect, relation):
                        if self._key(obj) == self._key(cause):
                            found.append(Attack(
                                kind="counterexample", about=(cause, effect),
                                evidence=(effect, relation, obj),
                                finding=f"{effect} is stated to happen {relation} {obj}, "
                                        f"which this chain says causes it"))
        return found
