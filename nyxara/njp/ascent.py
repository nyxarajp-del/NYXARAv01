"""NYXARA · njp/ascent.py — the whole loop, and whether diagnosing is worth anything (🧗, V.82).

Six organs now exist that each carry their own floor:

===================  ====================================================================
:mod:`~nyxara.njp.measurement`   is the number measuring what it appears to?
:mod:`~nyxara.njp.attribution`   why did it fail — tested, not guessed?
:mod:`~nyxara.njp.reach`         how far does it reach, and where exactly does it stop?
:mod:`~nyxara.njp.universe`      which experiment is worth running?
:mod:`~nyxara.njp.fusion`        the same shape twice, without inventing one
:mod:`~nyxara.njp.evolve`        and what may be kept
===================  ====================================================================

Separately they are six measured mechanisms. Together they are supposed to be a loop that acquires
a capability it did not have:

    find the edge → take the rung past it → ask why it fails → repair *that* → judge on hidden
    problems → keep it only if the edge moved

This module runs that loop. And it exists to answer one question, because without an answer to it
everything above is elaborate bookkeeping:

    **Does diagnosing the cause help you choose a better repair than picking one at random?**

That is not rhetorical and the answer is not obvious. A loop that diagnoses carefully and then
repairs no better than chance has learned nothing about itself; it has only spent longer. So every
run here is scored against a `blind` twin that skips attribution entirely and picks a repair at
random from the same set, and against a `greedy` twin that always reaches for the same favourite.
The comparison is the result. The loop is the apparatus.

**What is deliberately not claimed.** Nothing here writes source code. The repairs are drawn from a
supplied set, which is what makes the comparison clean — both twins choose from the same shelf, so
what differs is *how they choose*, not what is on it. Whether a system can also **invent** the
repair is a different question and a harder one, and answering this one first is what makes that
one askable: if choosing well among known repairs were worth nothing, inventing new ones would be
worth less.

Pure standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.attribution import Failure, attribute
from nyxara.njp.measurement import Benchmark
from nyxara.njp.reach import Ladder, climb

__all__ = ["Repair", "Attempt", "Ascent", "ascend", "STRATEGIES", "HIDDEN"]

#: How the repair is chosen. `diagnosed` is the loop; the other two are what it has to beat.
STRATEGIES: Tuple[str, ...] = ("diagnosed", "blind", "greedy")

#: How many problems the kept-or-killed decision is made on — drawn fresh, never seen by the
#: repair or by the diagnosis. The edge itself is measured on other problems again, so a repair
#: cannot be kept for fitting the very rung that motivated it.
HIDDEN = 300


@dataclass
class Repair:
    """One thing that could be changed, and what changing it does.

    ``addresses`` is what this repair would fix **if** that were the cause — the label
    :mod:`nyxara.njp.attribution` uses. It is not a promise that the repair works: a repair
    addressing the wrong cause changes nothing, which is exactly what makes choosing among them
    worth measuring.
    """

    name: str = ""
    addresses: str = ""
    #: Rebuild the capability with this repair applied, ready to be measured at any setting.
    apply: Optional[Callable[[Any], Benchmark]] = None
    cost: float = 1.0


@dataclass
class Attempt:
    """One turn of the loop: what was wrong, what was tried, and whether it was kept."""

    strategy: str = ""
    edge_before: Any = None
    at: Any = None
    cause: str = ""
    rivals: List[str] = field(default_factory=list)
    chose: str = ""
    addressed: str = ""
    kept: bool = False
    edge_after: Any = None
    hidden_before: float = 0.0
    hidden_after: float = 0.0

    @property
    def gained(self) -> float:
        return round(self.hidden_after - self.hidden_before, 4)

    @property
    def right_repair(self) -> bool:
        """Did it reach for the repair that addresses the cause the failure actually had?"""
        return bool(self.cause) and self.addressed == self.cause

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "edge_before": self.edge_before, "at": self.at,
                "cause": self.cause, "rivals": self.rivals, "chose": self.chose,
                "addressed": self.addressed, "right_repair": self.right_repair,
                "kept": self.kept, "edge_after": self.edge_after, "gained": self.gained}

    def render(self) -> str:
        mark = "kept " if self.kept else "killed"
        return (f"  {self.strategy:<10} edge {str(self.edge_before):<4} → "
                f"{str(self.edge_after):<4}  at {str(self.at):<4} because "
                f"{self.cause or '(unestablished)':<13} tried {self.chose:<14} "
                f"{mark} {self.gained:+.4f}")


@dataclass
class Ascent:
    """What a strategy did across many capabilities."""

    strategy: str = ""
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def moved(self) -> float:
        """Share of capabilities whose edge actually moved outward. The only thing that counts."""
        out = [a for a in self.attempts if a.edge_before is not None]
        if not out:
            return 0.0
        return round(sum(1 for a in out
                         if a.edge_after is not None and a.edge_after > a.edge_before)
                     / len(out), 4)

    @property
    def kept(self) -> float:
        return round(sum(1 for a in self.attempts if a.kept) / len(self.attempts), 4) \
            if self.attempts else 0.0

    @property
    def right_repair(self) -> float:
        """How often it reached for the repair addressing the established cause."""
        known = [a for a in self.attempts if a.cause]
        return round(sum(1 for a in known if a.right_repair) / len(known), 4) if known else 0.0

    @property
    def gained(self) -> float:
        out = [a.gained for a in self.attempts]
        return round(sum(out) / len(out), 4) if out else 0.0

    @property
    def kept_nothing(self) -> float:
        """Kept a repair that did not move the edge. The cost of a gate that is too generous."""
        kept = [a for a in self.attempts if a.kept]
        if not kept:
            return 0.0
        return round(sum(1 for a in kept
                         if a.edge_after is None or a.edge_after <= a.edge_before)
                     / len(kept), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "moved": self.moved, "kept": self.kept,
                "right_repair": self.right_repair, "gained": self.gained,
                "kept_nothing": self.kept_nothing, "attempts": len(self.attempts)}

    def render(self) -> str:
        return (f"  {self.strategy:<10} moved the edge {self.moved:.3f}   "
                f"chose the repair that fits {self.right_repair:.3f}   "
                f"kept {self.kept:.3f}   kept-but-no-gain {self.kept_nothing:.3f}")


# --------------------------------------------------------------------------------------------- #
#  one turn of the loop
# --------------------------------------------------------------------------------------------- #
def _choose(strategy: str, repairs: Sequence[Repair], cause: str,
            rng: random.Random, favourite: str) -> Optional[Repair]:
    """Which repair to try. The three strategies differ **here and nowhere else**."""
    if not repairs:
        return None
    if strategy == "blind":
        return rng.choice(list(repairs))
    if strategy == "greedy":
        fits = [r for r in repairs if r.addresses == favourite]
        return fits[0] if fits else rng.choice(list(repairs))
    fits = [r for r in repairs if r.addresses == cause]
    # No established cause, or no repair for it: the diagnosis has nothing to offer and falls back
    # to chance rather than pretending. Counted, so the advantage is never borrowed from a case
    # where the diagnosis said nothing.
    return fits[0] if fits else rng.choice(list(repairs))


def ascend(make: Callable[[Any], Benchmark], dial: Sequence[Any], repairs: Sequence[Repair],
           experiments: Callable[[Any, Any], Failure], hidden: Callable[[Any, Any], float],
           *, strategy: str = "diagnosed", rng: Optional[random.Random] = None,
           favourite: str = "data") -> Attempt:
    """One turn: find the edge, diagnose the rung past it, repair, and keep only if it moved.

    ``hidden`` scores a (repair, setting) pair on problems neither the diagnosis nor the repair has
    seen. It is what decides *kept*, and it is drawn separately from the problems the edge is
    measured on — a repair that is kept for fitting the rung that motivated it is the defect
    :mod:`nyxara.njp.evolve` refuses and this loop must not reintroduce one level up.
    """
    rng = rng or random.Random(0)
    before: Ladder = climb(make, dial, called="difficulty")
    out = Attempt(strategy=strategy, edge_before=before.edge)
    at = before.next_to_build
    if at is None:
        return out                      # no edge, so nothing to reach past — see `Ladder.edge`
    out.at = at

    got = attribute(experiments(at, None))
    out.cause, out.rivals = got.root, got.rivals

    repair = _choose(strategy, repairs, out.cause, rng, favourite)
    if repair is None or repair.apply is None:
        return out
    out.chose, out.addressed = repair.name, repair.addresses

    # Both readings on the **same** hidden problems, so the only thing that differs between them
    # is the repair. Drawing them separately was the first version, and it made noise alone raise
    # the second figure about half the time — a wrong repair was then "kept" at 0.683 while only
    # 0.233 of them were the right one. That is exactly the defect `measurement.stability` exists
    # to catch, reintroduced one level up by the module that imports it.
    out.hidden_before = hidden(None, at)
    out.hidden_after = hidden(repair, at)
    out.kept = out.hidden_after > out.hidden_before
    if not out.kept:
        out.edge_after = before.edge
        return out
    after: Ladder = climb(lambda s: repair.apply(s), dial, called="difficulty")
    out.edge_after = _reach(after, before)
    return out


def _reach(after: Ladder, before: Ladder) -> Any:
    """How far it reaches now — which is not the same question as where its edge is.

    A repair good enough that **every** setting holds leaves the new ladder `exhausted`, and
    :attr:`Ladder.edge` is then `None` by design: there is no boundary within what was tried. That
    is the right answer to *where does it stop* and the wrong one to *how far does it reach*, and
    the first version of this loop conflated them — reporting a total success as no movement at
    all, because `None` fell back to the old edge.

    So an exhausted ladder reaches at least its last rung, and a barren or patchy one has not been
    shown to reach anywhere, which leaves the old edge standing.
    """
    if after.edge is not None:
        return after.edge
    if after.exhausted and after.rungs:
        return after.rungs[-1].at
    return before.edge
