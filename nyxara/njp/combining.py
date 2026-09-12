"""NYXARA · njp/combining.py — the haystack is charged to the needle (🧩, NJP V.95).

V.94 found the few transformations everything else is made of, from behaviour alone. But it was
still told **how** things are made of each other: *apply one, then the other*. Composition was
supplied, and it is the last thing in the chain that was.

So here that is searched too. A way of combining two observed transformations is written as a short
**word** — ``ab`` means *look up through a, then through b*; ``ba`` is the other order; ``a`` alone
is a way of combining that ignores its second argument entirely. Nothing says which of them is the
right one, and composition is simply one member of the list.

**And then the search is charged for its own size.** This is the whole of the version. Told to
invent a way of combining things, a search will consider an enormous number of them, find something
that fits, and report a triumph — and the fit is a property of the haystack rather than of the
needle. So the bits are counted honestly:

    total = log₂(ways considered) + the description V.94 would have given

Picking one of two ways costs one bit. Picking one of a million costs twenty, and a discovery has
to be twenty bits better than longhand before it has said anything at all. **A bigger search cannot
buy a cheaper answer**, and :func:`charged` is where that is enforced rather than hoped for.

**The degenerate ways are left in on purpose.** ``a`` and ``b`` ignore an argument, so everything
built from them collapses and they reach almost nothing — which is what they should do. A search
whose candidate list contains only sensible candidates has had its answer chosen for it, and
leaving the useless ones in is how the list stops being a hint.

Pure standard library, and it leans on :mod:`nyxara.njp.generators` for everything V.94 already
settled rather than restating it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.generators import (DEPTH, LOOKED_AT, WIDTH, Observed, Recipe, longhand)

__all__ = ["Way", "Found", "ways", "charged", "search", "WAYS", "LETTERS", "LONGEST", "SEED"]

#: The two things a way of combining may look through. Nothing else is available to it, which is
#: what keeps the list of ways small enough to enumerate and to pay for.
LETTERS: Tuple[str, ...] = ("a", "b")

#: How long a word may be. Three letters is enough for both orders of composition, for ignoring an
#: argument, and for conjugation — and short enough that the whole list is enumerated rather than
#: sampled, so the charge below is a real count and not an estimate.
LONGEST = 3

SEED = 95


@dataclass(frozen=True)
class Way:
    """A way of combining two transformations, as a word read right to left.

    ``ab`` sends position ``i`` to ``a[b[i]]``. That happens to be composition; the class does not
    know that and nothing here treats it differently from ``bba``.
    """

    word: str = "ab"

    @property
    def name(self) -> str:
        return self.word

    def of(self, one: Observed, two: Observed) -> Optional[Observed]:
        """Combine two transformations, or ``None`` when the result is not a rearrangement."""
        table = {"a": one, "b": two}
        width = len(one)
        if len(two) != width:
            return None
        out = []
        for i in range(width):
            at = i
            for letter in reversed(self.word):
                look = table.get(letter)
                if look is None or at >= len(look):
                    return None
                at = look[at]
            out.append(at)
        got = tuple(out)
        return got if len(set(got)) == width else None


def ways(longest: int = LONGEST) -> List[Way]:
    """Every word up to ``longest`` letters. Enumerated, so the charge is a count.

    The degenerate ones — ``a``, ``b``, ``aa`` — are here on purpose. A candidate list containing
    only sensible candidates has had its answer chosen for it.
    """
    out: List[Way] = []
    for size in range(1, max(1, longest) + 1):
        for letters in product(LETTERS, repeat=size):
            out.append(Way("".join(letters)))
    return out


#: The list as it stands, so a caller can see what was paid for.
WAYS: Tuple[Way, ...] = tuple(ways())


def charged(considered: int) -> float:
    """Bits owed simply for having chosen one way out of ``considered``.

    The needle pays for the haystack. Without this a search can widen its own candidate list until
    something fits and call the fit a discovery — which is the failure that makes *invent a
    representation* an unfalsifiable instruction rather than an experiment.
    """
    return math.log2(max(1, considered))


def _reach(seeds: Sequence[Observed], way: Way, want: Sequence[Observed],
           depth: int = DEPTH, looked_at: int = LOOKED_AT) -> Dict[Observed, Recipe]:
    """Everything ``seeds`` build under ``way``, shortest recipe first."""
    width = len(want[0]) if want else WIDTH
    still = tuple(range(width))
    wanted = set(want)
    found: Dict[Observed, Recipe] = {still: Recipe(())}
    edge: List[Tuple[Observed, Recipe]] = [(still, Recipe(()))]
    for _ in range(max(1, depth)):
        nxt: List[Tuple[Observed, Recipe]] = []
        for where, recipe in edge:
            for i, seed in enumerate(seeds):
                step = way.of(where, seed)
                if step is None or step in found:
                    continue
                made = Recipe(recipe.steps + (i,))
                found[step] = made
                nxt.append((step, made))
        edge = nxt
        if not edge or wanted <= set(found) or len(found) > looked_at:
            break
    return {k: v for k, v in found.items() if k in wanted}


@dataclass
class Found:
    """A way of combining, a set of seeds, and what the two of them cost together."""

    way: Optional[Way] = None
    seeds: List[Observed] = field(default_factory=list)
    recipes: Dict[Observed, Recipe] = field(default_factory=dict)
    covered: int = 0
    of: int = 0
    width: int = WIDTH
    considered: int = 1

    @property
    def complete(self) -> bool:
        return self.of > 0 and self.covered == self.of

    @property
    def bare(self) -> float:
        """What V.94 would have charged: the seeds written out, plus a recipe each."""
        if not self.seeds:
            return longhand(self.of, self.width)
        naming = math.log2(len(self.seeds)) + 1.0
        spelling = len(self.seeds) * self.width * math.log2(max(2, self.width))
        following = sum(max(1.0, r.length) * naming for r in self.recipes.values())
        missing = (self.of - self.covered) * self.width * math.log2(max(2, self.width))
        return spelling + following + missing

    @property
    def cost(self) -> float:
        """Everything: the way it was chosen, the seeds, and the recipes."""
        return round(charged(self.considered) + self.bare, 2)

    @property
    def worth_it(self) -> bool:
        return (self.way is not None and bool(self.seeds) and self.complete
                and self.cost < longhand(self.of, self.width) - longhand(1, self.width))

    def to_dict(self) -> Dict[str, Any]:
        return {"way": self.way.name if self.way else "", "seeds": len(self.seeds),
                "covered": self.covered, "of": self.of, "cost": self.cost,
                "charged": round(charged(self.considered), 2), "bare": round(self.bare, 2),
                "longhand": longhand(self.of, self.width), "worth_it": self.worth_it,
                "considered": self.considered}

    def render(self) -> str:
        if not self.worth_it:
            return (f"  no way of combining pays for itself; writing all {self.of} out longhand "
                    f"costs {longhand(self.of, self.width):.0f} bits")
        return (f"  `{self.way.name}` with {len(self.seeds)} seeds covers {self.covered}/{self.of}"
                f" at {self.cost:.0f} bits — {charged(self.considered):.1f} of them just for "
                f"choosing it out of {self.considered} — against "
                f"{longhand(self.of, self.width):.0f} longhand")


def search(observed: Sequence[Observed], *, longest: int = LONGEST, depth: int = DEPTH,
           most: int = 2, width: int = WIDTH) -> Found:
    """Search ways of combining **and** seeds together, charging for the size of the search.

    Both halves are unknown here, which is the difference from V.94: it knew how things combine and
    looked only for what to combine. The cost of not knowing is a real number and it is added to
    every answer, so a way of combining has to earn back the bits spent finding it.
    """
    unique = sorted(set(observed))
    every = ways(longest)
    nothing = Found(way=None, seeds=[], of=len(unique), width=width, considered=len(every))
    best: Optional[Found] = None
    for way in every:
        for size in range(1, max(1, most) + 1):
            for picked in combinations(unique, size):
                reached = _reach(picked, way, unique, depth=depth)
                here = Found(way=way, seeds=list(picked), recipes=reached,
                             covered=len(reached), of=len(unique), width=width,
                             considered=len(every))
                if here.worth_it and (best is None or here.cost < best.cost):
                    best = here
    return best if best is not None else nothing
