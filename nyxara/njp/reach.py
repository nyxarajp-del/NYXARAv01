"""NYXARA · njp/reach.py — how far does it reach, and where exactly does it stop (📏➡, NJP V.78).

A capability reported as one number is a capability nobody can act on. *Causal reasoning: 0.73* does
not say whether the third of cases it misses are the hard ones or scattered at random, and those
need different work. What is wanted instead is where it **stops**:

    one hop     0.99
    two hops    0.96
    three hops  0.89
    four hops   0.64
    five hops   0.31      <- it stops here

That edge is the next thing to build, and it is a far more useful output than a percentage. But an
edge is easy to invent, so most of this module is about the four ways a ladder can fail to have one
and the refusal to report one anyway:

* **exhausted** — every rung held. There is no boundary *within what was tried*, which is not the
  same as no boundary, and saying the second when you measured the first is how a capability gets
  claimed that nobody tested.
* **barren** — no rung held. There is no capability here to put a boundary on, and an "edge at
  zero" would dress that up as a finding.
* **patchy** — it holds at one and three and not at two. That is not a boundary, whatever the
  highest passing rung says; a boundary needs competence *beneath* it, and reporting the top rung
  regardless is how noise becomes an architecture roadmap.
* **graceful** — it declines steadily with no single step falling away. Real, common, and a
  different thing from a cliff: there is no one place to attack, so naming one would send work
  somewhere arbitrary.

Every rung is measured the way :mod:`nyxara.njp.measurement` insists — against **its own floor**,
because difficulty usually changes the floor too, and a rung whose majority answer is right 90% of
the time is not evidence of competence at 0.88. And a rung whose score will not hold still across
repeats does not count as held, for the same reason the forge's timing gate did not.

What the edge is *for* is :mod:`nyxara.njp.attribution`: the first rung past it is a failure with a
known neighbour that works, which is the most informative failure there is.

Pure standard library.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.measurement import Benchmark, Critique, critique

__all__ = ["Rung", "Ladder", "climb", "the_gap", "CLEAR", "WOBBLE", "CLIFF"]

#: How far above its own floor a rung has to score before it counts as held. Not a claim about
#: what is good — a claim about what is **distinguishable from the floor on this many items**.
CLEAR = 0.08

#: How far a rung's score may move across repeats and still count. A rung that will not hold still
#: is not a rung that held, for the reason the forge's timing gate was not a gate.
WOBBLE = 0.05

#: How far the score must fall in a single step for it to be a cliff rather than a decline. Below
#: this the capability is fading, which is real and is not a place to attack.
CLIFF = 0.15


@dataclass(frozen=True)
class Rung:
    """One setting of the difficulty dial, and what was found there."""

    at: Any = None
    score: float = 0.0
    floor: float = 0.0
    spread: float = 0.0
    leaks: bool = False
    note: str = ""

    @property
    def above(self) -> float:
        """What is left once the rung's own trivial strategy is subtracted."""
        return round(self.score - self.floor, 4)

    @property
    def holds(self) -> bool:
        """Clear of its own floor, steady across repeats, and not measured on what it was taught."""
        return self.above >= CLEAR and self.spread <= WOBBLE and not self.leaks

    def to_dict(self) -> Dict[str, Any]:
        return {"at": self.at, "score": self.score, "floor": self.floor,
                "above": self.above, "spread": self.spread, "holds": self.holds,
                "leaks": self.leaks, "note": self.note}

    def render(self) -> str:
        mark = "ok" if self.holds else "  "
        why = ""
        if not self.holds:
            why = ("leaked" if self.leaks else
                   "unsteady" if self.spread > WOBBLE else "at its floor")
        return (f"  [{mark}] {str(self.at):<10} {self.score:.3f}  floor {self.floor:.3f}  "
                f"above {self.above:+.3f}  {why}")


@dataclass
class Ladder:
    """A capability measured across a dial, and what can honestly be said about where it stops."""

    name: str = ""
    dial: str = ""
    rungs: List[Rung] = field(default_factory=list)

    @property
    def holding(self) -> List[Rung]:
        return [r for r in self.rungs if r.holds]

    @property
    def barren(self) -> bool:
        """Nothing held anywhere. There is no capability here to put a boundary on."""
        return bool(self.rungs) and not self.holding

    @property
    def exhausted(self) -> bool:
        """Every rung held. No boundary **within what was tried**, which is a different claim."""
        return bool(self.rungs) and len(self.holding) == len(self.rungs)

    @property
    def contiguous(self) -> List[Rung]:
        """The unbroken run of held rungs from the easiest end."""
        out: List[Rung] = []
        for rung in self.rungs:
            if not rung.holds:
                break
            out.append(rung)
        return out

    @property
    def patchy(self) -> bool:
        """It holds above a rung it does not hold at. Whatever that is, it is not a boundary."""
        return len(self.contiguous) < len(self.holding)

    @property
    def edge(self) -> Optional[Any]:
        """The hardest setting it reliably reaches — or nothing, when there is no edge to report.

        Empty for every one of the four ways a ladder has no edge: nothing held, everything held,
        competence that skips a rung, and a ladder with no rungs. Each of those is reported by its
        own flag instead, because *"edge at 4"* and *"held at 4 but not at 3"* are different facts
        and collapsing them loses the one that matters.
        """
        if self.barren or self.exhausted or self.patchy:
            return None
        run = self.contiguous
        return run[-1].at if run else None

    @property
    def cliff(self) -> Optional[Tuple[Any, float]]:
        """Where it falls away, and by how much — when it falls rather than fades."""
        drop, where = 0.0, None
        for before, after in zip(self.rungs, self.rungs[1:]):
            fell = before.above - after.above
            if fell > drop:
                drop, where = fell, after.at
        return (where, round(drop, 4)) if drop >= CLIFF else None

    @property
    def graceful(self) -> bool:
        """It declines across the dial and no single step falls away.

        Not an alternative to having an edge — a ladder that fades past its floor has a last rung
        that held like any other. What it lacks is a **place to attack**: with a cliff there is one
        setting where something breaks, and with a fade the work is spread along the whole dial.
        Reporting this *instead of* the edge was the first version's error, and it threw away the
        more useful of the two facts.
        """
        if len(self.rungs) < 3 or self.barren:
            return False
        return self.cliff is None and self.rungs[0].above - self.rungs[-1].above >= CLEAR

    @property
    def verdict(self) -> str:
        if not self.rungs:
            return "nothing measured"
        if self.barren:
            return "no capability at any setting tried"
        if self.exhausted:
            return "no boundary within what was tried — the dial was not turned far enough"
        if self.patchy:
            return ("it holds above a setting it does not hold at, so this is not a boundary; "
                    "the measurement or the dial is wrong before anything else can be said")
        edge = self.edge
        cliff = self.cliff
        if cliff:
            return f"it reaches {edge} and falls away at {cliff[0]} by {cliff[1]:.3f}"
        if self.graceful:
            return (f"it reaches {edge}, but fades rather than stops — no single setting breaks "
                    f"it, so there is no one place to attack")
        return f"it reaches {edge}"

    @property
    def next_to_build(self) -> Optional[Any]:
        """The first setting past the edge. The most informative failure available, because the
        setting next to it is known to work."""
        edge = self.edge
        if edge is None:
            return None
        run = len(self.contiguous)
        return self.rungs[run].at if run < len(self.rungs) else None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "dial": self.dial, "edge": self.edge,
                "cliff": list(self.cliff) if self.cliff else None,
                "barren": self.barren, "exhausted": self.exhausted, "patchy": self.patchy,
                "graceful": self.graceful, "verdict": self.verdict,
                "next_to_build": self.next_to_build,
                "rungs": [r.to_dict() for r in self.rungs]}

    def render(self) -> str:
        head = f"{self.name or 'capability'} across {self.dial or 'a dial'}:"
        body = "\n".join(r.render() for r in self.rungs)
        return f"{head}\n{body}\n  → {self.verdict}"


def _rung(at: Any, bench: Benchmark, repeats: int) -> Rung:
    """Measure one setting, against its own floor and its own repeats."""
    got: Critique = critique(bench)
    floors = [f.got for f in got.ran if f.check in ("majority", "chance", "shuffled")]
    scores = [bench.score() for _ in range(max(1, repeats))]
    leak = next((f for f in got.findings if f.check == "leakage"), None)
    return Rung(at=at, score=round(statistics.mean(scores), 4),
                floor=round(max(floors), 4) if floors else 0.0,
                spread=round(max(scores) - min(scores), 4),
                leaks=bool(leak is not None and leak.verdict == "broken"),
                note=got.render())


def the_gap(ladder: "Ladder", make: Callable[[Any], Any]) -> Optional[Any]:
    """The first setting past the edge, described as a failure ready to be attributed.

    This is the whole point of finding an edge, and why this module ends here rather than at a
    number. A failure picked at random is hard to diagnose because everything about it is a
    candidate; a failure **one rung past a setting that demonstrably works** has almost everything
    held constant by construction, and what differs is the dial. That is the most informative
    failure available, and :mod:`nyxara.njp.attribution` is what to hand it to.

    ``make`` builds whatever the caller wants to hand the setting to — in practice a
    :class:`nyxara.njp.attribution.Failure`, which is deliberately **not** imported here. This
    module measures where a capability stops; what to do about that is somebody else's question,
    and an organ that reaches into the one downstream of it is an organ that cannot be used without
    it.

    ``None`` where there is no edge, and for the same reason as everywhere else here: without one,
    there is no *next* setting, only a list of settings that did not work.
    """
    at = ladder.next_to_build
    return None if at is None else make(at)


def climb(make: Callable[[Any], Benchmark], dial: Sequence[Any], *,
          name: str = "", called: str = "", repeats: int = 3) -> Ladder:
    """Measure a capability at each setting of a dial, easiest first.

    ``dial`` must run from easiest to hardest — the contiguity that :attr:`Ladder.edge` rests on is
    meaningless otherwise, and a caller that hands them in the wrong order gets ``patchy`` rather
    than a wrong edge, which is the right way round to fail.
    """
    out = Ladder(name=name, dial=called)
    for at in dial:
        try:
            out.rungs.append(_rung(at, make(at), repeats))
        except Exception as error:  # noqa: BLE001 — a rung that cannot be built says so
            out.rungs.append(Rung(at=at, note=f"could not be built: {type(error).__name__}"))
    return out
