"""NYXARA · njp/worth.py — what a vocabulary costs and what it buys (💰, NJP V.92).

V.91 ended owing the substrate itself: the moves were generated from four numbers, but the *shape*
of those four numbers — an index affine, a value affine, three scales, two lifts — was chosen by
hand. The obvious next version invents a substrate. It cannot honestly be written yet, because
**"a better substrate" was not a claim anything could check.**

There is also a specific danger, and it is why this comes first: a search told to invent a
representation will invent an enormous one. Everything becomes expressible, coverage looks perfect,
and nothing has been learned. Guarding against that needs a number, and the number is not coverage.

    **the whole thing, in bits** — what the vocabulary costs to write down, *plus* what is
    left over: every pair it cannot explain has to be described some other way, and that
    costs too.

``cost`` is ``log₂(distinct moves) + 1`` bits: one to say a move happens here, and the rest to say
which. ``worth`` is how many **held-out** pairs of near-identical operations it can tell apart.

The first version stopped at ``worth ÷ cost``, and that measure **rewards uselessness**. A
vocabulary of one move separated 41 of 105 pairs, cost one bit, and won — the exact mirror of the
failure the measure was built to prevent, arriving from the other side. Charging for the leftovers
fixes it without a thumb on the scale: a vocabulary that explains nothing pays for all 105 pairs
by hand, and a vocabulary that explains everything pays only for itself. That is description
length, which is somebody else's idea and older than this repository.

**And the first thing it measured was that V.91's substrate is too big.** Seven plain slides —
offsets zero through six, no strides, no scales, no lifts — separate *every* held-out pair at about
three times the value per bit of the 252-move substrate V.91 shipped. The stride and scale
dimensions cost five extra bits and buy nothing here.

That refines V.91 rather than contradicting it. V.91 concluded ``scale = −1`` was load-bearing, and
it was — **given the supplied seven**, whose offsets are ``{0, 1, 2, 6}`` and which spend three of
their slots on moves all sitting at offset zero. The cheap repair was never a new dimension. It was
spending those slots on offsets three, four and five.

**Random pairs cannot measure this.** Every vocabulary here separates every randomly drawn pair of
rearrangements, so the measure saturates and says nothing. It bites only where the operations are
nearly identical — V.91's swap fixture generalised to every transposition — which is worth writing
down on its own: *a vocabulary's quality is invisible on easy cases*.

**What this is not.** No substrate is invented here. What is built is the scale that a claim to
have invented a better one would have to be weighed on, and the first thing it weighed was mine.

Pure standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Move", "Vocabulary", "Weighed", "VOCABULARIES", "hard_pairs", "contested",
           "separates", "weigh", "compare", "by_hand", "WIDTH", "SEED"]

WIDTH = 7
SEED = 92

#: A move, as V.91 left it: where each output reads from, and what is done to the value. A plain
#: tuple, because this module counts vocabularies rather than running them.
Move = Tuple[int, int, float]


@dataclass(frozen=True)
class Vocabulary:
    """A set of moves, and the two numbers that decide whether it is any good."""

    name: str = ""
    moves: Tuple[Move, ...] = ()

    @property
    def distinct(self) -> Tuple[Move, ...]:
        return tuple(sorted(set(self.moves)))

    @property
    def offsets(self) -> List[int]:
        return sorted({o % WIDTH for _, o, _ in self.distinct})

    @property
    def signs(self) -> List[float]:
        return sorted({s for _, _, s in self.distinct})

    @property
    def cost(self) -> float:
        """Bits to name one of its moves, plus one to say a move happens at all.

        The ``+1`` is not a fudge. Without it a one-move vocabulary costs nothing and wins every
        ratio by dividing by zero — and a vocabulary you never have to choose within still has to
        be invoked.
        """
        return math.log2(max(1, len(self.distinct))) + 1.0

    @property
    def reach(self) -> int:
        """Position pairs a ``first`` law can compare. V.91's measure, carried forward."""
        return len(self.offsets) ** 2


def by_hand(pairs: int) -> float:
    """Bits to describe one unexplained pair on its own, with no vocabulary to lean on.

    Naming one pair out of all of them. It is what the leftovers cost, and charging it is the whole
    difference between a measure that prefers small vocabularies and one that prefers *good* ones.
    """
    return math.log2(max(2, pairs))


def _swap(i: int, j: int) -> Tuple[int, ...]:
    out = list(range(WIDTH))
    out[i], out[j] = out[j], out[i]
    return tuple(out)


#: The vocabularies weighed against each other. Every one is a proposal somebody could make,
#: including the two this repository actually shipped.
VOCABULARIES: Tuple[Vocabulary, ...] = (
    Vocabulary("the supplied seven", ((1, 0, 1.0), (1, 0, -1.0), (-1, WIDTH - 1, 1.0),
                                      (1, 1, 1.0), (1, 2, 1.0), (1, 0, 2.0))),
    Vocabulary("slides only", tuple((1, t, 1.0) for t in range(WIDTH))),
    Vocabulary("slides and signs", tuple((1, t, s) for t in range(WIDTH) for s in (1.0, -1.0))),
    Vocabulary("V.91's substrate", tuple(
        (s, t, sc) for s in range(-WIDTH, WIDTH + 1)
        if s and math.gcd(abs(s), WIDTH) == 1
        for t in range(WIDTH) for sc in (1.0, -1.0, 2.0))),
    Vocabulary("every regrouping", tuple(
        (s, t, 1.0) for s in list(range(1, WIDTH)) + [-s for s in range(1, WIDTH)]
        for t in range(WIDTH))),
    Vocabulary("one move", ((1, 0, 1.0),)),
)


def hard_pairs(rng: Optional[random.Random] = None) -> Tuple[List[Any], List[Any]]:
    """Near-identical operation pairs, split into a half that is looked at and a half that is not.

    Random pairs are useless: every vocabulary separates every one of them, the measure saturates
    and says nothing. A vocabulary's quality is invisible on easy cases, which is why this exists
    rather than a call to ``random.sample``.
    """
    rng = rng or random.Random(SEED)
    swaps = [_swap(i, j) for i in range(WIDTH) for j in range(i + 1, WIDTH)]
    every = [(a, b) for i, a in enumerate(swaps) for b in swaps[i + 1:]]
    rng.shuffle(every)
    at = len(every) // 2
    return every[:at], every[at:]


def contested(pairs: Sequence[Any], vocabularies: Sequence["Vocabulary"] = ()) -> List[Any]:
    """The pairs the vocabularies **disagree** about — some explain them and some do not.

    Every other pair is wasted measurement. One nothing explains distinguishes nothing; one
    everything explains distinguishes nothing either, and V.92 found five of six vocabularies
    explaining every randomly drawn pair. What is left is the boundary, and a vocabulary's real
    discriminating power only shows there.

    Selecting on the vocabularies means these pairs are **seen**, so a held-out measurement has to
    contest a *different* half — which is what :func:`hard_pairs` splits for.
    """
    pool = list(vocabularies) or list(VOCABULARIES)
    out = []
    for one, two in pairs:
        verdicts = {separates(one, two, v) for v in pool}
        if len(verdicts) > 1:
            out.append((one, two))
    return out


def separates(one: Sequence[int], two: Sequence[int], vocabulary: Vocabulary) -> bool:
    """Can any law this vocabulary builds hold for one operation and not the other?

    For an operation that rearranges positions, ``f(x)[t₁]`` is ``x[one[t₁]]`` — so a law
    ``first: move(f(x)) == move(x)`` holds exactly when ``one[t₁] == t₂``. That collapses the whole
    question to set arithmetic, which is what makes weighing six vocabularies affordable at all.
    """
    offsets = vocabulary.offsets
    for left in offsets:
        for right in offsets:
            if (one[left] == right) != (two[left] == right):
                return True
    if len(vocabulary.signs) > 1:
        # `>=` carries an orientation `==` does not, and a sign is what reaches it — V.91's
        # finding, kept because it is true and not because it turned out to be the cheap repair.
        for left in offsets:
            for right in offsets:
                if (one[left] < right) != (two[left] < right):
                    return True
    return False


@dataclass
class Weighed:
    """One vocabulary, what it cost, and what it bought on pairs it never saw."""

    vocabulary: Optional[Vocabulary] = None
    worth: int = 0
    of: int = 0
    seen_worth: int = 0

    @property
    def cost(self) -> float:
        return self.vocabulary.cost if self.vocabulary else 1.0

    @property
    def leftover(self) -> int:
        return max(0, self.of - self.worth)

    @property
    def total(self) -> float:
        """The whole description, in bits: the vocabulary, plus everything it failed to explain.

        This is what vocabularies are ranked on. Lower is better, and there is no way to win it by
        being small and useless or by being large and complete.
        """
        return round(self.cost + self.leftover * by_hand(self.of), 3)

    @property
    def value(self) -> float:
        """Worth per bit — kept as a **diagnostic only**, because on its own it prefers a
        vocabulary that explains almost nothing and costs almost nothing."""
        return round(self.worth / self.cost, 3)

    @property
    def complete(self) -> bool:
        return self.of > 0 and self.worth == self.of

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.vocabulary.name if self.vocabulary else "",
                "moves": len(self.vocabulary.distinct) if self.vocabulary else 0,
                "cost": round(self.cost, 3), "worth": self.worth, "of": self.of,
                "leftover": self.leftover, "total": self.total, "value": self.value,
                "complete": self.complete,
                "reach": self.vocabulary.reach if self.vocabulary else 0}

    def render(self) -> str:
        name = self.vocabulary.name if self.vocabulary else "?"
        moves = len(self.vocabulary.distinct) if self.vocabulary else 0
        mark = "ok " if self.complete else " ~ "
        return (f"  {mark} {name:<22}{moves:>6}{self.cost:>8.2f}"
                f"{self.worth:>6} /{self.of:<5}{self.total:>11.1f}{self.value:>9.1f}")


def weigh(vocabulary: Vocabulary, held: Sequence[Any], seen: Sequence[Any] = ()) -> Weighed:
    """What it buys on pairs it never saw, over what it costs to write down."""
    return Weighed(vocabulary=vocabulary,
                   worth=sum(1 for a, b in held if separates(a, b, vocabulary)),
                   of=len(held),
                   seen_worth=sum(1 for a, b in seen if separates(a, b, vocabulary)))


def compare(vocabularies: Sequence[Vocabulary] = VOCABULARIES,
            rng: Optional[random.Random] = None, *, adversarial: bool = False) -> List[Weighed]:
    """Weigh every vocabulary on the same held-out pairs, shortest whole description first.

    ``adversarial`` narrows the held-out set to the pairs the vocabularies **disagree** about,
    chosen on the half that is looked at and then applied to the half that is not. That is the
    honest version of "make the benchmark harder": the boundary is found on one sample and the
    measurement is taken on another, so the narrowing cannot be tuned into the answer.
    """
    seen, held = hard_pairs(rng)
    if adversarial:
        wanted = {frozenset((a, b)) for a, b in contested(seen, vocabularies)}
        sharper = [(a, b) for a, b in held if frozenset((a, b)) in wanted] or contested(
            held, vocabularies)
        held = sharper or held
    return sorted((weigh(v, held, seen) for v in vocabularies),
                  key=lambda w: (w.total, w.cost))
