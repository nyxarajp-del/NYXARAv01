"""NYXARA · njp/supply.py — the bill charges for choosing, not for knowing (🎁, NJP V.99).

Eight versions of accounting, and one line undoes all of them:

    charged(1) == 0.0

The whole apparatus from V.92 to V.98 prices a **choice**: how many options were on the table, and
how many bits it takes to say which was taken. Supplied structure involves no choosing. So the
cheapest description is always the one that considered nothing — **which is the description with
the most handed to it**, and that is the exact opposite of what all of it was built to measure.

Measured, on the eight observations V.97 and V.98 were built around:

======================================  ========
route                                   bits
======================================  ========
V.98's baseline, naming L out of three  57.17
**someone writes L = 2 into the source**  **55.58**
======================================  ========

Hard-coding wins. Not by a trick — by the rule the accounting is made of.

**And the closure is that one world was the wrong unit.** A hard-coded ``L`` is correct for the
world it was written for and wrong for the next, so across worlds it has to be re-supplied — and a
constant drawn from ``L_max`` possibilities costs ``log₂(L_max)`` bits **of source**, which is
exactly what deriving it costs at run time. The bits move from one column to the other and a
two-part code sees them either way. Measured across three worlds the two routes come out **equal to
the digit**, which is the theorem rather than a coincidence:

    **Nothing is free. A tower that pays nothing at run time is carrying the same bits in
    its source, once per world it is right about.**

**What this means for everything above it.** The V.92–V.98 numbers are valid **between searches
compared on the same worlds** and license nothing absolute. A single world cannot tell a search from
a lookup, because on a single world a lookup is cheaper and the bill has no column for having been
told.

So the reason *Level 7 stays unclaimed* is deeper than *some vocabulary is still supplied*. **The
instrument could not have detected the difference on one world.** It can now, and only by being run
on several.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from nyxara.njp.combining import charged

__all__ = ["Route", "Across", "carried", "free_lunch", "across", "SEED"]

SEED = 99


def free_lunch() -> float:
    """What a choice among one option costs. Zero, and that is the hole.

    Kept as a function rather than a comment because it is the whole of the version, and a claim
    this load-bearing should be something a test can call.
    """
    return charged(1)


def carried(bound: int) -> float:
    """Bits of **source** a hard-coded answer carries, for one world it is right about.

    Writing a constant drawn from ``bound`` possibilities into a program costs what naming it costs
    anywhere else. The accounting above priced only run-time choices and therefore saw this as
    free — which is why a hard-code beat every search it was compared against.
    """
    return charged(max(1, bound))


@dataclass
class Route:
    """One way of arriving at an answer, priced at run time **and** in its source."""

    name: str = ""
    #: Bits spent choosing while running, summed over the worlds.
    chosen: float = 0.0
    #: Bits the program carries because it was told, summed over the worlds.
    told: float = 0.0
    #: Everything the answer still has to say once the route has been settled.
    describing: float = 0.0

    @property
    def total(self) -> float:
        return round(self.chosen + self.told + self.describing, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "chosen": round(self.chosen, 2),
                "told": round(self.told, 2), "describing": round(self.describing, 2),
                "total": self.total}

    def render(self) -> str:
        return (f"  {self.name:<28}{self.chosen:>9.2f}{self.told:>9.2f}"
                f"{self.describing:>12.2f}{self.total:>10.2f}")


@dataclass
class Across:
    """Two routes, priced over several worlds, where the difference is supposed to show."""

    worlds: int = 0
    searching: Optional[Route] = None
    told: Optional[Route] = None

    @property
    def same(self) -> bool:
        """Do they cost the same? They should, and a gap either way is a defect somewhere.

        ``worlds > 0`` is not bookkeeping. Over no worlds both routes total zero and the first
        version duly reported agreement — a comparison of nothing with nothing, passing. That is
        the same shape as a check that could not run being written down as one that passed, which
        this package has now caught at five levels and once more here, in the module that exists to
        catch it.
        """
        if self.worlds < 1 or self.searching is None or self.told is None:
            return False
        return abs(self.searching.total - self.told.total) < 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {"worlds": self.worlds, "same": self.same,
                "searching": self.searching.to_dict() if self.searching else None,
                "told": self.told.to_dict() if self.told else None}

    def render(self) -> str:
        lines = [f"  {'route':<28}{'chosen':>9}{'told':>9}{'describing':>12}{'total':>10}"]
        for route in (self.searching, self.told):
            if route is not None:
                lines.append(route.render())
        if self.worlds < 1:
            lines.append("  → nothing was compared, which is not agreement")
        else:
            lines.append(f"  → over {self.worlds} world(s) they cost "
                         + ("the same, to the digit" if self.same else "differently"))
        return "\n".join(lines)


def across(bounds: Sequence[int], describing: Sequence[float]) -> Across:
    """Price a search and a hard-code over several worlds, counting source as well as run time.

    ``bounds`` is how many answers were possible in each world and ``describing`` what each world
    still costs once the answer is settled. The search names its answer at run time; the hard-code
    names nothing then and carries the same constant in its program, **once per world it is right
    about**. Over one world that looks like a saving. Over several it is not.
    """
    n = min(len(bounds), len(describing))
    searching = Route(name="derives it, every world",
                      chosen=sum(charged(b) for b in bounds[:n]),
                      told=0.0, describing=sum(describing[:n]))
    handed = Route(name="is told it, every world",
                   chosen=0.0,
                   told=sum(carried(b) for b in bounds[:n]),
                   describing=sum(describing[:n]))
    return Across(worlds=n, searching=searching, told=handed)
