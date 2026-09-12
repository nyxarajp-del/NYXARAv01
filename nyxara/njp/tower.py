"""NYXARA · njp/tower.py — where the tower stops, in bits (🗼, NJP V.96).

V.95 charged a search for its own size: naming one way of combining out of fourteen costs 3.8 bits,
and a wider search pays more for the same answer. But it charged only for **picking from the list**,
never for the list existing. *Two letters and words up to three* is a language, and somebody chose
it — so the honest bill has another line on it, and adding that line opens a hole underneath:

    a language is chosen from a family of languages, which is chosen from …

That is a real regress and it cannot be closed by asserting it stops. It closes by **arithmetic**.
Each storey costs a fixed toll — the bits to say which of its options was taken — and buys whatever
the storey below could not explain. The toll does not shrink as you climb. The savings do, because
there is only so much structure in any set of observations. So there is a height above which
climbing costs more than it returns, and **that height is a number this module computes** rather
than a judgement anybody makes.

Three storeys are built and priced:

======================  ====================================================================
storey                  what it costs
======================  ====================================================================
0 · longhand            every transformation written out on its own
1 · a way               pick one way from a fixed language, then V.94's description
2 · a language          pick one language from a family, *then* pick a way inside it
======================  ====================================================================

**And the answer here is that the second storey does not pay.** Choosing the language costs more
than choosing well inside it saves, so the tower stops at one — which is a measured result and not
a preference. To show that is a measurement and not a foregone conclusion, the exam also contains a
world where storey two **does** pay: when the default language is enormous, naming a small one back
is worth more than it costs.

Pure standard library, and every price below is V.95's or V.94's rather than a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.combining import Found, charged, search
from nyxara.njp.generators import Observed, WIDTH, longhand

__all__ = ["Language", "Storey", "Climb", "LANGUAGES", "price", "climb", "SEED"]

SEED = 96


@dataclass(frozen=True)
class Language:
    """A way of generating ways. **One** number, and somebody had to choose it.

    ``longest`` is how long a word may be; the alphabet is the two arguments a way of combining
    has, and it is fixed because a way of combining has two arguments.

    The first version carried a ``letters`` field as well, so that a language read as two numbers —
    and :func:`~nyxara.njp.combining.ways` never varied it. The size was computed as ``k + k² + …``
    over an alphabet the search did not use, which prices a language for expressiveness it does not
    have. A parameter nothing reads is not a parameter, and charging for it is charging for
    nothing.
    """

    longest: int = 3

    @property
    def name(self) -> str:
        return f"words to {self.longest}"

    @property
    def size(self) -> int:
        """How many ways it offers: ``2 + 4 + … + 2^L``, over the two arguments there are."""
        return sum(2 ** n for n in range(1, max(1, self.longest) + 1))

    @property
    def toll(self) -> float:
        """Bits to name one way inside it — V.95's charge, unchanged."""
        return charged(self.size)


#: The family of languages a storey-two search picks from. Small on purpose: a family nobody could
#: enumerate would have to be charged for *its* size, which is the regress this module is about.
LANGUAGES: Tuple[Language, ...] = tuple(Language(longest) for longest in (1, 2, 3, 4, 5))


@dataclass
class Storey:
    """One height of the tower: what it cost and what it managed to explain."""

    height: int = 0
    name: str = ""
    #: Bits paid simply to be at this height — naming a language, naming a way.
    toll: float = 0.0
    #: Bits for the description this height made possible.
    description: float = 0.0
    explained: bool = False
    says: str = ""

    @property
    def total(self) -> float:
        return round(self.toll + self.description, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"height": self.height, "name": self.name, "toll": round(self.toll, 2),
                "description": round(self.description, 2), "total": self.total,
                "explained": self.explained, "says": self.says}

    def render(self) -> str:
        mark = "ok " if self.explained else " ~ "
        return (f"  {mark} {self.height}· {self.name:<26}{self.toll:>7.1f} + "
                f"{self.description:>8.1f} = {self.total:>8.1f}")


@dataclass
class Climb:
    """Every storey priced, and the height the arithmetic stops at."""

    storeys: List[Storey] = field(default_factory=list)

    @property
    def stops_at(self) -> int:
        """The cheapest height. Not the tallest that works — the one that costs least."""
        if not self.storeys:
            return 0
        best = min(self.storeys, key=lambda s: s.total)
        return best.height

    @property
    def worth_climbing(self) -> List[int]:
        """Heights that beat every height below them."""
        out: List[int] = []
        for storey in sorted(self.storeys, key=lambda s: s.height):
            below = [s.total for s in self.storeys if s.height < storey.height]
            if not below or storey.total < min(below):
                out.append(storey.height)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"storeys": [s.to_dict() for s in self.storeys], "stops_at": self.stops_at,
                "worth_climbing": self.worth_climbing}

    def render(self) -> str:
        lines = [f"  {'':<4}{'storey':<27}{'toll':>8}   {'describing':>8}   {'total':>8}"]
        lines += [s.render() for s in sorted(self.storeys, key=lambda s: s.height)]
        lines.append(f"  → cheapest at storey {self.stops_at}; climbing pays at "
                     f"{self.worth_climbing}")
        return "\n".join(lines)


def price(observed: Sequence[Observed], language: Language, *,
          width: int = WIDTH) -> Tuple[float, bool, str]:
    """What storey one costs in a given language: its toll, plus V.94's description.

    The toll is separated from the description on purpose. They are different kinds of debt — one
    is owed for having chosen, the other for what was said — and a tower that adds them without
    naming them cannot show where it stops.
    """
    got: Found = search(observed, longest=language.longest, width=width)
    if not got.worth_it:
        return longhand(len(set(observed)), width), False, "nothing in this language paid"
    return got.bare, True, (f"`{got.way.name}` with {len(got.seeds)} seeds"
                            if got.way else "")


def climb(observed: Sequence[Observed], *, languages: Sequence[Language] = LANGUAGES,
          default: Optional[Language] = None, width: int = WIDTH) -> Climb:
    """Price every storey and let the arithmetic say where to stop.

    Storey zero writes everything out. Storey one uses ``default`` and pays that language's toll.
    Storey two picks the best of ``languages`` and pays **both** tolls — naming the language, then
    naming the way inside it. Nothing here prefers a height; the totals do.
    """
    default = default or Language(3)
    unique = sorted(set(observed))
    out = Climb()

    flat = longhand(len(unique), width)
    out.storeys.append(Storey(height=0, name="everything written out", toll=0.0,
                              description=flat, explained=True,
                              says=f"{len(unique)} transformations, one at a time"))

    bare, worked, says = price(unique, default, width=width)
    out.storeys.append(Storey(height=1, name=f"a way in {default.name}", toll=default.toll,
                              description=bare, explained=worked, says=says))

    best: Optional[Tuple[float, Language, float, bool, str]] = None
    picking = charged(len(languages))
    for candidate in languages:
        inner, ok, note = price(unique, candidate, width=width)
        here = picking + candidate.toll + inner
        if best is None or here < best[0]:
            best = (here, candidate, inner, ok, note)
    if best is not None:
        _, chosen, inner, ok, note = best
        out.storeys.append(Storey(
            height=2, name="a language, then a way", toll=picking + chosen.toll,
            description=inner, explained=ok,
            says=f"{chosen.name}; {note}"))
    return out
