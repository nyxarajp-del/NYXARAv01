"""NYXARA · njp/regularity.py — generating a law nobody wrote down (⚖, NJP V.90).

V.89 could say what an operation is by trying to break nine **supplied** laws. The laws were the
vocabulary, and a vocabulary is the thing this package keeps catching itself being handed. Giving
it nine laws and asking it to rediscover them would be the same promotion one level out.

So nothing here is given a law. What is given is a handful of **moves** — ways of pushing a row of
numbers around, each one content-free — and a schema:

    ⟨ one side ⟩   reduced somehow   ⟨ related somehow ⟩   ⟨ the other side ⟩

where a *side* is a move applied either to the input or to the operation's output. Neither half is
a law. The pairing is, and the pairing is what gets searched: **1,568 candidates**, enumerated
exhaustively, of which almost all are nonsense and a handful are not.

**What falls out has no name.** ``every: f(x) ≥ x`` is dominance and arrives called ``law 641``.
``first: f(x) ≥ reverse(f(x))`` is *the output is sorted downward* and arrives called ``law 88``.
Nobody wrote either down; both are a side, a reduction and a relation, composed.

**A law that everything obeys explains nothing.** The search keeps only rules that **split** the
operations in front of it, and two rules that induce the same split are one discovery wearing two
notations — the V.88 duplicate gate, at the level of laws. A law's whole claim to exist is the
distinction it creates.

**And the milestone is not "a law was found".** It is that a discovered law separates two
operations that the nine supplied ones put in the same family. V.89's battery merges *sort upward*
with *sort downward*, *a smallest-of-three window* with *a largest-of-three*, and *hold the first*
with *hold the last* — three pairs it cannot tell apart at any number of tries, because its nine
laws do not contain the distinction. A law that splits one of those has earned its existence; one
that only re-sorts what was already sorted has not.

**Four numbers, not three.** V.89 counted ``invented``, ``flattered`` and ``buried``. This adds the
one V.89's own ``op 14`` implied:

======================  ====================================================================
invented                a real distinction found
flattered               two that behave identically, separated — false novelty
buried                  two that differ, merged, **and a searched rule would have split them**
resolution-limited      two that differ, merged, and **no rule in the whole space** splits them
======================  ====================================================================

The last two look identical from outside and are completely different inside. One is a search that
failed; the other is a vocabulary that is too small — which is a finding, and the only honest
version of *we could not express the difference*. Telling them apart is possible **only because
the search is exhaustive**, and if it stopped being exhaustive that distinction would go with it.

Pure standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Move", "Rule", "Split", "MOVES", "REDUCTIONS", "RELATIONS", "rules", "holds",
           "discover", "differ", "TRIES", "WIDTH", "TOLERANCE", "SEED"]

#: Attempts made to break each rule on each operation. As in V.89, *not refuted* means nothing
#: without this number, and it is carried rather than dropped.
TRIES = 60

WIDTH = 7
TOLERANCE = 1e-9
SEED = 90

Row = List[float]


@dataclass(frozen=True)
class Move:
    """A way of pushing a row of numbers around. Content-free, and **not** a law.

    None of these says anything about an operation. ``reverse`` is not *the operation is symmetric*
    and ``negate`` is not *the operation is odd* — they are things one can do to seven numbers, and
    a claim only appears when two of them are put either side of a relation.
    """

    name: str = ""
    do: Optional[Callable[[Row], Row]] = None

    def __call__(self, row: Row) -> Row:
        return list(self.do(row)) if self.do else list(row)


MOVES: Tuple[Move, ...] = (
    Move("as it is", lambda r: list(r)),
    Move("negated", lambda r: [-v for v in r]),
    Move("reversed", lambda r: list(reversed(r))),
    Move("slid by one", lambda r: r[1:] + r[:1]),
    Move("slid by two", lambda r: r[2:] + r[:2]),
    Move("doubled", lambda r: [2.0 * v for v in r]),
    Move("lifted", lambda r: [v + 1.0 for v in r]),
)

#: How a row is reduced before the two sides are compared. Without this the only available relation
#: is *pointwise, everywhere*, and a descending row beats its own reverse in the front half and
#: loses in the back — so the law that says *the output is sorted downward* cannot be stated at all.
#: Found by trying to state it and failing.
REDUCTIONS: Tuple[Tuple[str, Callable[[Row], Row]], ...] = (
    ("every", lambda r: r),
    ("first", lambda r: r[:1]),
    ("last", lambda r: r[-1:]),
    ("total", lambda r: [float(sum(r))]),
)

#: Only two. ``≤`` is ``≥`` with the sides swapped, and including both would double the space with
#: nothing in it — every ``≤`` rule would be found twice and the duplicate gate would spend its
#: time on an artefact of the notation rather than on real re-descriptions.
RELATIONS: Tuple[Tuple[str, Callable[[float, float], bool]], ...] = (
    ("==", lambda a, b: abs(a - b) <= TOLERANCE),
    (">=", lambda a, b: a >= b - TOLERANCE),
)


@dataclass(frozen=True)
class Rule:
    """One candidate law: a side, a reduction, a relation, and the other side.

    A *side* is ``(which, move)`` where ``which`` is ``x`` for the input or ``f`` for the output.
    The rule claims the relation holds between the two reduced sides, for every input, always — and
    a single counterexample ends it.
    """

    left: Tuple[str, str] = ("f", "as it is")
    right: Tuple[str, str] = ("x", "as it is")
    how: str = "every"
    relation: str = "=="
    number: int = 0
    called: str = ""

    @property
    def name(self) -> str:
        return self.called or f"law {self.number}"

    @property
    def statement(self) -> str:
        def _side(side: Tuple[str, str]) -> str:
            what = "f(x)" if side[0] == "f" else "x"
            return what if side[1] == "as it is" else f"{side[1]}({what})"
        return f"{self.how}: {_side(self.left)} {self.relation} {_side(self.right)}"

    def christen(self, called: str, *, splits: int = 0) -> "Rule":
        """Name it, afterwards, and only once it has drawn a distinction.

        A law that separates nothing is not a law about operations, it is a sentence that happens
        to be true, and naming it would give it standing it has not got.
        """
        if splits < 1:
            raise ValueError(f"{self.name} separates nothing, so it has not earned a name")
        return Rule(left=self.left, right=self.right, how=self.how, relation=self.relation,
                    number=self.number, called=called)


def _moves() -> Dict[str, Move]:
    return {m.name: m for m in MOVES}


def rules() -> List[Rule]:
    """Every candidate the schema admits. Exhaustive, and small enough to stay exhaustive.

    Two kinds are dropped. Reflexive rules — the same side against itself — because ``f(x) ==
    f(x)`` is true of everything and says so about nothing.

    And **rules that never mention the operation**. ``first: lifted(x) >= negated(x)`` is a claim
    about seven random numbers, not about ``f``, and whether it survives depends only on which rows
    were drawn. The first version kept them, and the exam immediately produced a "distinction"
    between an operation and *itself* — a rule about ``x`` that happened to survive on one draw and
    not on another. A candidate that cannot be about the operation is not a candidate.
    """
    out: List[Rule] = []
    sides = [(which, move.name) for which in ("x", "f") for move in MOVES]
    n = 0
    for left in sides:
        for right in sides:
            if left == right:
                continue
            if left[0] == "x" and right[0] == "x":
                continue
            for how, _ in REDUCTIONS:
                for relation, _ in RELATIONS:
                    n += 1
                    out.append(Rule(left=left, right=right, how=how, relation=relation, number=n))
    return out


def holds(rule: Rule, operation: Callable[[Sequence[float]], Sequence[float]], *,
          tries: int = TRIES, width: int = WIDTH,
          rng: Optional[random.Random] = None) -> Tuple[bool, int]:
    """Try to break one rule on one operation. Returns ``(unbroken, attempts made)``.

    Never *the rule is true*. As in V.89 the only direction a probe establishes anything in is
    refutation, and the attempt count is the whole content of the other answer.
    """
    rng = rng or random.Random(SEED)
    moves = _moves()
    reduce = dict(REDUCTIONS)[rule.how]
    relate = dict(RELATIONS)[rule.relation]
    for attempt in range(1, max(1, tries) + 1):
        row = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        try:
            out = list(operation(row))
            if len(out) != len(row):
                return False, attempt        # a rule over two lengths is not a rule
            left = reduce(moves[rule.left[1]](out if rule.left[0] == "f" else row))
            right = reduce(moves[rule.right[1]](out if rule.right[0] == "f" else row))
        except Exception:  # noqa: BLE001
            return False, attempt
        if len(left) != len(right) or not all(relate(a, b) for a, b in zip(left, right)):
            return False, attempt
    return True, max(1, tries)


def differ(one: Callable[[Sequence[float]], Sequence[float]],
           two: Callable[[Sequence[float]], Sequence[float]], *,
           tries: int = 200, width: int = WIDTH,
           rng: Optional[random.Random] = None) -> bool:
    """Do these two operations ever disagree? Sampled, so a ``False`` is *not seen to differ*.

    This is what separates a merge that is **right** from one that is a failure, and it is the only
    reason the four-way accounting below can be computed at all.
    """
    rng = rng or random.Random(SEED)
    for _ in range(max(1, tries)):
        row = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        try:
            a, b = list(one(row)), list(two(row))
        except Exception:  # noqa: BLE001
            return True
        if len(a) != len(b) or any(abs(p - q) > TOLERANCE for p, q in zip(a, b)):
            return True
    return False


@dataclass
class Split:
    """A law, and the distinction it draws over the operations it was searched against."""

    rule: Optional[Rule] = None
    obeyed_by: Tuple[str, ...] = ()
    broken_by: Tuple[str, ...] = ()
    #: Rules that draw exactly this distinction and are therefore the same discovery.
    same_as: List[str] = field(default_factory=list)

    @property
    def useful(self) -> bool:
        return bool(self.obeyed_by) and bool(self.broken_by)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.rule.name if self.rule else "", "says": self.rule.statement
                if self.rule else "", "obeyed_by": list(self.obeyed_by),
                "broken_by": list(self.broken_by), "same_as": self.same_as}

    def render(self) -> str:
        also = f"  (and {len(self.same_as)} others say the same)" if self.same_as else ""
        return (f"  ! {self.rule.name if self.rule else '?':<8} "
                f"{self.rule.statement if self.rule else '':<40}{also}\n"
                f"      obeyed by {', '.join(self.obeyed_by)}\n"
                f"      broken by {', '.join(self.broken_by)}")


def discover(operations: Dict[str, Callable[[Sequence[float]], Sequence[float]]], *,
             tries: int = TRIES, rng: Optional[random.Random] = None
             ) -> Tuple[List[Split], Dict[str, Any]]:
    """Search the whole schema and keep the laws that draw a distinction.

    Returns the distinct splits, strongest-evidence first, and a report of what was searched. A law
    every operation obeys, and a law none of them obeys, are both dropped — a law's whole claim to
    exist is the distinction it creates, and by that standard most of a schema is furniture.
    """
    rng = rng or random.Random(SEED)
    names = list(operations)
    every = rules()
    seen: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Split] = {}
    vacuous = 0
    for rule in every:
        # One seed per **rule**, shared by every operation it is tried on. The first version drew a
        # fresh seed per operation, so two operations were compared on two different sets of rows —
        # the defect V.82 found in `ascent`, where a before and an after came from different
        # samples, arriving again at the level of laws. It manufactured a distinction between an
        # operation and an identical copy of itself.
        seed = rng.randrange(1 << 30)
        obeyed, broken = [], []
        for name in names:
            unbroken, _ = holds(rule, operations[name], tries=tries, rng=random.Random(seed))
            (obeyed if unbroken else broken).append(name)
        if not obeyed or not broken:
            vacuous += 1
            continue
        key = (tuple(obeyed), tuple(broken))
        if key in seen:
            seen[key].same_as.append(rule.name)
            continue
        seen[key] = Split(rule=rule, obeyed_by=tuple(obeyed), broken_by=tuple(broken))
    found = sorted(seen.values(), key=lambda s: (-min(len(s.obeyed_by), len(s.broken_by)),
                                                 s.rule.number if s.rule else 0))
    return found, {"searched": len(every), "vacuous": vacuous, "distinct": len(found),
                   "operations": names}
