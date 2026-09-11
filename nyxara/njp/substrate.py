"""NYXARA · njp/substrate.py — earning the moves a law is made of (🧱, NJP V.91).

V.90 generated laws from seven supplied **moves** — reverse, negate, slide by one, slide by two,
double, lift, leave alone. Smaller than nine laws and still a human-designed vocabulary. So this
asks the question one level down: *can the moves themselves be earned rather than handed over?*

**The substrate is four numbers.** A move is

    out[i] = scale · row[(stride·i + offset) mod n] + lift

and nothing else. All seven of V.90's moves are special cases — ``reversed`` is
``stride −1, offset n−1``; ``negated`` is ``scale −1``; ``lifted`` is ``lift 1`` — which is checked
by construction rather than asserted. At width seven the substrate yields **252 behaviourally
distinct moves**, of which seven were supplied and 245 were not.

**Why four numbers and not a programming language.** Given an arbitrary program, a search finds
``O(x) = lookup_table[x]`` and manufactures perfect separation out of nothing. The guard here is
structural rather than statistical: a move is four numbers, so a lookup table **cannot be written
down in it at all**. The bound is on what is expressible, not on what is preferred.

**A move earns its place by the law it makes possible.** Not by looking interesting. The test is:
is there a pair of operations that **no** law built from the supplied moves separates, and that
some law using this move does? Anything else is a move that was already there under another name —
the V.88 duplicate gate, arriving at the level of vocabulary.

**And the thing that makes this measurable.** A ``first`` law reads ``f(x)[t₁]`` against ``x[t₂]``,
where ``t₁`` and ``t₂`` are the two moves' offsets — moves apply to **each side independently**, so
what a law can see is a *pair* of positions, and the reachable set is a product. The supplied seven
carry offsets ``{0, 1, 2, 6}``, so they reach **16 of the 49 pairs**. That number is what a
generated move has to buy, and it is computed rather than guessed: :func:`reach`.

Two findings follow, and the second is the one that keeps the first honest:

* An operation pair living in the other 33 pairs — a swap of positions 3 and 4 against a swap of 3
  and 5 — is separated by **no** law the supplied moves can build, and by many that a generated one
  can. A generated move is load-bearing.
* And for the family of **stride permutations**, generation buys nothing at all: for any two
  distinct strides some supplied pair always separates them, because blocking all four supplied
  offsets at once admits only a single stride. Where generation is worth something is narrow and
  sayable, which is more use than a claim that it is always worth something.

Pure standard library.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Shape", "SUPPLIED", "every_move", "reach", "separations", "earns", "WIDTH",
           "SCALES", "LIFTS", "TRIES", "SEED"]

#: How long the rows are. Prime, so every non-zero stride is a permutation and the substrate does
#: not quietly produce many-to-one maps that are not moves at all.
WIDTH = 7

#: The scales and lifts the substrate admits. Few and fixed: this is a **bound on what can be
#: expressed**, and widening it to "any real number" is what turns a substrate into a fitter.
SCALES: Tuple[float, ...] = (1.0, -1.0, 2.0)
LIFTS: Tuple[float, ...] = (0.0, 1.0)

TRIES = 40
TOLERANCE = 1e-9
SEED = 91


@dataclass(frozen=True)
class Shape:
    """A move as four numbers: where each output reads from, and what is done to the value.

    There is no name in here and no room for one. ``Shape(stride=-1, offset=6)`` is what somebody
    would call *reversed*, and the class does not know that.
    """

    stride: int = 1
    offset: int = 0
    scale: float = 1.0
    lift: float = 0.0
    called: str = ""

    @property
    def name(self) -> str:
        return self.called or f"({self.stride}, {self.offset}, {self.scale:g}, {self.lift:g})"

    @property
    def size(self) -> int:
        """Its description length, in numbers. Four, always — which is the whole guard."""
        return 4

    def of(self, row: Sequence[float], width: int = WIDTH) -> List[float]:
        return [self.scale * row[(self.stride * i + self.offset) % width] + self.lift
                for i in range(width)]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "stride": self.stride, "offset": self.offset,
                "scale": self.scale, "lift": self.lift}


#: V.90's seven, written in the substrate. Every one is a special case, and
#: :func:`nyxara.njp.substrateschool.the_supplied_are_special_cases` checks it against V.90's own
#: implementations rather than trusting the arithmetic here.
SUPPLIED: Tuple[Shape, ...] = (
    Shape(1, 0, 1.0, 0.0, "as it is"),
    Shape(1, 0, -1.0, 0.0, "negated"),
    Shape(-1, WIDTH - 1, 1.0, 0.0, "reversed"),
    Shape(1, 1, 1.0, 0.0, "slid by one"),
    Shape(1, 2, 1.0, 0.0, "slid by two"),
    Shape(1, 0, 2.0, 0.0, "doubled"),
    Shape(1, 0, 1.0, 1.0, "lifted"),
)

#: Rows used to tell two moves apart. Two shapes that agree on both are the same move however
#: differently their four numbers read — the duplicate gate, at the level of vocabulary.
_WITNESS: Tuple[Tuple[float, ...], ...] = (
    (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
    (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0),
)


def every_move(width: int = WIDTH) -> List[Shape]:
    """Every behaviourally distinct move the substrate admits. Enumerated, then deduplicated.

    Strides sharing a factor with the width are dropped: they fold several inputs onto one output,
    which is not a rearrangement of the row but a loss of it, and a law built on one is comparing
    against something that threw information away.
    """
    out: Dict[Tuple[float, ...], Shape] = {}
    strides = [s for s in range(-width, width + 1) if s != 0 and math.gcd(abs(s), width) == 1]
    for stride in strides:
        for offset in range(width):
            for scale in SCALES:
                for lift in LIFTS:
                    shape = Shape(stride, offset, scale, lift)
                    key = tuple(round(v, 9) for row in _WITNESS for v in shape.of(row, width))
                    out.setdefault(key, shape)
    return list(out.values())


def reach(moves: Sequence[Shape], width: int = WIDTH) -> Set[Tuple[int, int]]:
    """Which ``(position of f(x), position of x)`` pairs a ``first`` law can compare.

    This is the quantity that says what a vocabulary can and cannot see, and it is a *product*
    because the two sides take their moves independently. Computing it is what turned "the supplied
    moves feel expressive" into "they reach sixteen of forty-nine".
    """
    offsets = sorted({m.offset % width for m in moves})
    return {(a, b) for a in offsets for b in offsets}


def _reads(shape: Shape, row: Sequence[float], how: str, width: int) -> List[float]:
    got = shape.of(row, width)
    return {"every": got, "first": got[:1], "last": got[-1:],
            "total": [float(sum(got))]}[how]


def separations(one: Callable[[Sequence[float]], Sequence[float]],
                two: Callable[[Sequence[float]], Sequence[float]],
                moves: Sequence[Shape], *, width: int = WIDTH, tries: int = TRIES,
                rng: Optional[random.Random] = None) -> List[Tuple[str, str, str, str]]:
    """Laws over these moves that hold for one operation and not the other.

    Both operations are tried on **the same rows** for each candidate law, which is V.90's
    correction carried forward: two operations judged on two different draws are two different
    questions, and the difference between them is then about the draw.
    """
    rng = rng or random.Random(SEED)
    sides = [(which, m) for which in ("x", "f") for m in moves]
    found: List[Tuple[str, str, str, str]] = []
    for left in sides:
        for right in sides:
            if left[0] == "x" and right[0] == "x":
                continue
            if left[0] == right[0] and left[1].name == right[1].name:
                continue
            for how in ("every", "first", "last", "total"):
                for relation in ("==", ">="):
                    seed = rng.randrange(1 << 30)
                    a = _survives(one, left, right, how, relation, width, tries,
                                  random.Random(seed))
                    b = _survives(two, left, right, how, relation, width, tries,
                                  random.Random(seed))
                    if a != b:
                        found.append((f"{left[0]}:{left[1].name}", f"{right[0]}:{right[1].name}",
                                      how, relation))
    return found


def _survives(operation, left, right, how, relation, width, tries, rng) -> bool:
    for _ in range(max(1, tries)):
        row = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        try:
            out = list(operation(row))
            if len(out) != width:
                return False
            a = _reads(left[1], out if left[0] == "f" else row, how, width)
            b = _reads(right[1], out if right[0] == "f" else row, how, width)
        except Exception:  # noqa: BLE001
            return False
        if relation == "==" and any(abs(p - q) > TOLERANCE for p, q in zip(a, b)):
            return False
        if relation == ">=" and any(p < q - TOLERANCE for p, q in zip(a, b)):
            return False
    return True


@dataclass
class Earned:
    """What a move bought, which is the only reason for it to exist."""

    move: Optional[Shape] = None
    #: Pairs it separates that the supplied vocabulary cannot.
    bought: List[Tuple[str, str]] = field(default_factory=list)
    supplied_already: bool = False

    @property
    def earns(self) -> bool:
        return bool(self.bought) and not self.supplied_already

    def to_dict(self) -> Dict[str, Any]:
        return {"move": self.move.to_dict() if self.move else None, "bought": self.bought,
                "supplied_already": self.supplied_already, "earns": self.earns}


def earns(move: Shape, pairs: Sequence[Tuple[str, Any, str, Any]], *,
          supplied: Sequence[Shape] = SUPPLIED, width: int = WIDTH,
          tries: int = TRIES, rng: Optional[random.Random] = None) -> Earned:
    """Does this move buy a separation the supplied vocabulary cannot make?

    The only question that matters about a generated move. A move that separates something the
    supplied seven already separate is the supplied seven wearing four different numbers, and
    crediting it is how a vocabulary search comes to report its own inputs back.
    """
    rng = rng or random.Random(SEED)
    got = Earned(move=move)
    got.supplied_already = any(
        all(abs(p - q) <= TOLERANCE
            for row in _WITNESS for p, q in zip(move.of(row, width), other.of(row, width)))
        for other in supplied)
    for name_a, one, name_b, two in pairs:
        with_supplied = separations(one, two, list(supplied), width=width, tries=tries,
                                    rng=random.Random(SEED))
        if with_supplied:
            continue
        with_it = separations(one, two, list(supplied) + [move], width=width, tries=tries,
                              rng=random.Random(SEED))
        if with_it:
            got.bought.append((name_a, name_b))
    return got
