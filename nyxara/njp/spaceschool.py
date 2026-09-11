"""NYXARA · njp/spaceschool.py — does it find the empty corner, or call every corner empty (🗺).

A blind-spot detector has one failure mode that matters and it is not missing things. It is
**firing on everything**, because a system that answers *you have not considered X* to every
question looks thorough, costs nothing to build, and is worth nothing. So the exam scores silence
exactly as hard as it scores catches, and three of the six cases below are cases where the correct
answer is *the map is covered, the gap is not what is missing*.

The first fixture is not invented. Its cause is a bug this repository actually had: the reader was
producing the right answer and scoring zero, because gold answers carry the sentence's full stop
(``Czech Republic.``) and candidates end at the last token. Every lever in `given`, `method` and
`shown` is flat, because nothing is wrong with any of them — the failure lives in `scored`, and no
amount of more data finds it. That is the shape V.83 hit on a real organ, with a cause I happen to
know because it cost a version to find by hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.space import Held, Intervention, Knob, Map, gaps, propose, survey

__all__ = ["World", "Case", "KNOWN", "retrodict", "examine", "run", "SEED", "KNOBS"]

SEED = 84

#: The map every fixture is drawn on. Supplied, not discovered — see the note in
#: :mod:`nyxara.njp.space`. Five regions, eleven knobs, and the shape that matters is that the four
#: experiments the attributor holds occupy `given` and `method` and nothing else.
KNOBS: Tuple[Knob, ...] = (
    Knob("how many examples", "given", "rows shown to the learner"),
    Knob("which examples", "given", "the slice they were taken from"),
    Knob("which rule family", "method", "what the learner is allowed to express"),
    Knob("how much search", "method", "how long it may look"),
    Knob("which readings", "shown", "what is measured about each item"),
    Knob("how many readings", "shown", "how many of them are offered"),
    Knob("how answers are compared", "scored", "what counts as the same answer"),
    Knob("what counts as answering", "scored", "abstention against a wrong answer"),
    Knob("how items were made", "built", "the process that produced the rows"),
    Knob("how candidates were made", "built", "the process that produced the choices"),
    Knob("how the split was cut", "built", "what landed on each side"),
)


# --------------------------------------------------------------------------------------------- #
#  a small world whose answers are right and whose score is zero
# --------------------------------------------------------------------------------------------- #
@dataclass
class World:
    """Items, gold answers, and a system that answers them — with a comparison in between.

    The comparison is a **field**, which is the whole point. It is the thing that lives in `scored`,
    it is invisible to every experiment that varies data or method, and in this repository it was
    invisible to a human for a version.
    """

    items: List[int] = field(default_factory=list)
    gold: List[str] = field(default_factory=list)
    said: List[str] = field(default_factory=list)
    #: How an answer is compared to gold. Exact match unless a fixture says otherwise.
    same: Optional[Callable[[str, str], bool]] = None
    #: Whether the system answered at all.
    spoke: List[bool] = field(default_factory=list)

    def score(self) -> float:
        if not self.items:
            return 0.0
        fn = self.same or (lambda a, b: a == b)
        return round(sum(1 for a, b in zip(self.said, self.gold) if fn(a, b))
                     / len(self.items), 4)

    def reachable(self) -> float:
        """Share of items whose right answer the system could in principle have produced.

        Read as an invariant by the fixtures below, so that an intervention which quietly makes the
        answer unproducible is caught the way V.83's pool repair was.
        """
        if not self.items:
            return 0.0
        return round(sum(1 for a, b in zip(self.said, self.gold)
                         if _bare(a) == _bare(b)) / len(self.items), 4)

    def spoke_rate(self) -> float:
        return round(sum(self.spoke) / len(self.spoke), 4) if self.spoke else 0.0


def _bare(text: str) -> str:
    return text.strip(" .,;:!?").lower()


def _punctuated(n: int = 300, seed: int = SEED, *, right: float = 1.0) -> World:
    """A system that is right, and a comparison that says it is wrong.

    ``right`` is how often the system actually produces the correct string. At 1.0 the score is
    0.0000 and every single answer is correct — the exact arrangement that makes `given`, `method`
    and `shown` experiments useless, because there is nothing in them to improve.
    """
    rng = random.Random(seed)
    words = ["Paris", "Vienna", "Copper", "Ada Lovelace", "the Seine", "1852"]
    gold = [rng.choice(words) + "." for _ in range(n)]        # the annotator's full stop
    said = [g[:-1] if rng.random() < right else "no" for g in gold]
    return World(items=list(range(n)), gold=gold, said=said, spoke=[True] * n)


def _score(world: World) -> float:
    return world.score()


# --------------------------------------------------------------------------------------------- #
#  interventions — the four the attributor holds, plus the ones a fixture adds
# --------------------------------------------------------------------------------------------- #
def _keeps_the_answer(world: World) -> float:
    return world.reachable()


HOLD_ANSWER = Held(name="the right answer stays producible", read=_keeps_the_answer)
HOLD_SPOKE = Held(name="it still answers as often", read=lambda w: w.spoke_rate())


def _four(base: World) -> List[Intervention]:
    """The shelf as `attribution` holds it: four experiments, two regions, no promises.

    Every one of them is honest and every one of them is flat, because nothing they vary is what is
    wrong. This is the V.83 arrangement reproduced on a world whose answer I know.
    """
    return [
        Intervention(verb="add", on="how many examples", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="replace", on="which rule family", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="add", on="how much search", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="add", on="how many readings", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
    ]


def _forgiving(base: World) -> World:
    """The experiment nobody ran: change **how answers are compared** and nothing else."""
    return World(items=base.items, gold=base.gold, said=base.said, spoke=base.spoke,
                 same=lambda a, b: _bare(a) == _bare(b))


def _gutted(base: World) -> World:
    """An attempt on `built` that throws the right answer away on its way past — V.83's repair."""
    return World(items=base.items, gold=base.gold, spoke=base.spoke,
                 said=["no"] * len(base.items))


# --------------------------------------------------------------------------------------------- #
#  six maps whose right answer is known by construction
# --------------------------------------------------------------------------------------------- #
def _exhausted() -> Map:
    """Four experiments, all flat, all inside two regions. The cause is in `scored`."""
    base = _punctuated()
    return survey("right answers, scored zero", KNOBS, _four(base), base, score=_score)


def _covered() -> Map:
    """The same failure, but something entered every region. Naming a gap here is a false alarm."""
    base = _punctuated()
    more = _four(base) + [
        Intervention(verb="replace", on="how answers are compared",
                     run=lambda: _forgiving(base), holds=()),
        Intervention(verb="shuffle", on="how items were made", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="mask", on="which readings", run=lambda: base, holds=(HOLD_ANSWER,)),
    ]
    return survey("every region entered", KNOBS, more, base, score=_score)


def _spoiled_attempt() -> Map:
    """Something tried to enter `built` and broke its own promise. The region is still unexamined.

    This is the case that separates a detector from a checklist. An attempt was made, so a naive
    coverage count marks `built` done; but the attempt threw the right answer away, so it measured
    nothing, and the region is exactly as unexamined as before anybody touched it.
    """
    base = _punctuated()
    more = _four(base) + [
        Intervention(verb="remove", on="how candidates were made",
                     run=lambda: _gutted(base), holds=(HOLD_ANSWER,)),
        Intervention(verb="replace", on="how answers are compared",
                     run=lambda: _forgiving(base), holds=()),
    ]
    return survey("an attempt that measured nothing", KNOBS, more, base, score=_score)


def _partly_entered() -> Map:
    """One knob of three varied in `built`. A region with a foot in it is not a blind spot.

    The temptation is to report any unvaried knob, which would make the detector fire on every map
    that ever existed — the failure mode that looks like diligence and is worth nothing.
    """
    base = _punctuated()
    more = _four(base) + [
        Intervention(verb="shuffle", on="how the split was cut", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="replace", on="how answers are compared",
                     run=lambda: _forgiving(base), holds=()),
        Intervention(verb="mask", on="which readings", run=lambda: base, holds=(HOLD_ANSWER,)),
    ]
    return survey("a foot in every region", KNOBS, more, base, score=_score)


def _unvouched() -> Map:
    """A promise with no way to read it. Not known broken, and not known kept.

    `scored` is entered by an intervention whose invariant cannot be measured, so the region is
    **not** covered — a check that could not run is reported as not checked, never as passed, and
    that rule does not stop applying because the check belongs to an experiment rather than a
    benchmark.
    """
    base = _punctuated()
    blind = Held(name="something nobody can measure", read=None)
    more = _four(base) + [
        Intervention(verb="replace", on="how answers are compared",
                     run=lambda: _forgiving(base), holds=(blind,)),
        Intervention(verb="shuffle", on="how items were made", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="mask", on="which readings", run=lambda: base, holds=(HOLD_ANSWER,)),
    ]
    return survey("a promise nobody can check", KNOBS, more, base, score=_score)


def _nothing_wrong() -> Map:
    """A system that simply works, examined everywhere. No failure, no gap, nothing to say."""
    base = _punctuated(right=1.0)
    base.same = lambda a, b: _bare(a) == _bare(b)
    more = _four(base) + [
        Intervention(verb="replace", on="how answers are compared", run=lambda: base, holds=()),
        Intervention(verb="shuffle", on="how items were made", run=lambda: base,
                     holds=(HOLD_ANSWER,)),
        Intervention(verb="mask", on="which readings", run=lambda: base, holds=(HOLD_ANSWER,)),
    ]
    return survey("nothing is wrong", KNOBS, more, base, score=_score)


@dataclass
class Case:
    name: str = ""
    #: The region the cause lives in. Empty means the map is covered and naming any gap is wrong.
    region: str = ""
    build: Optional[Callable[[], Map]] = None
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("right answers, scored zero", "scored", _exhausted,
         "four flat experiments in two regions; the comparison is what is wrong"),
    Case("every region entered", "", _covered,
         "same failure, nothing left unexamined — a gap here is a false alarm"),
    Case("an attempt that measured nothing", "built", _spoiled_attempt,
         "something entered `built` and broke its own promise, so it did not"),
    Case("a foot in every region", "", _partly_entered,
         "one knob of three is still an entered region"),
    Case("a promise nobody can check", "scored", _unvouched,
         "the invariant could not be read, so the region is not vouched for"),
    Case("nothing is wrong", "", _nothing_wrong,
         "a working system examined everywhere says nothing about blind spots"),
)


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Six maps whose empty corner is known by construction, and what the detector made of them."""
    rows: List[Dict[str, Any]] = []
    found = missed = quiet = alarms = also = 0
    for case in cases:
        drawn: Map = case.build()
        named = [g.region for g in gaps(drawn)]
        if case.region:
            if case.region in named:
                found += 1
                also += len(named) - 1
            else:
                missed += 1
        elif named:
            alarms += len(named)          # a fully examined map must name nothing at all
        else:
            quiet += 1
        # And naming a region something informative really did enter is a false alarm anywhere.
        entered = set(drawn.covered)
        alarms += len(entered & set(named))
        rows.append({"case": case.name, "want": case.region, "named": named,
                     "covered": sorted(entered), "note": case.note, "map": drawn})
    wanted = sum(1 for c in cases if c.region)
    return {"rows": rows, "found": found, "missed": missed, "quiet": quiet,
            "of": len(cases), "with_a_gap": wanted, "no_gap": len(cases) - wanted,
            "also_named": also, "false_alarms": alarms}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Four conditions, and the last two are what make the first two mean anything. Every map with an
    empty region must have it named; **no** map that was fully examined may have any region named;
    and no region something informatively entered may be called a gap. A detector that met the
    first condition by shouting *blind spot* at everything would fail the others, which is the
    entire reason they are here.

    ``also_named`` is reported and **not** scored against. On the first fixture two regions are
    genuinely empty and only one holds the cause, and there is no evidence on that map which says
    which — so naming both is the honest answer and narrowing it would be a guess. The first
    version of this exam scored the top-ranked region instead, which forced an ordering that had to
    come from somewhere; what it came from was region size, which is not evidence about anything.
    The exam caught it immediately and the ordering was deleted rather than tuned.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["found"] == got["with_a_gap"]
                         and got["missed"] == 0
                         and got["quiet"] == got["no_gap"]
                         and got["false_alarms"] == 0)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for row in got["rows"]:
        print(f"  {row['case']:<34} want {row['want'] or '(none)':<9} "
              f"named {row['named']}")
    print(f"\n  found {got['found']}/{got['with_a_gap']}, missed {got['missed']}, "
          f"quiet {got['quiet']}/{got['no_gap']}, also named {got['also_named']}, "
          f"false alarms {got['false_alarms']}, passes {got['passes']}\n")
    drawn = KNOWN[0].build()
    print(drawn.render())
    asked = propose(drawn)
    print(f"\n  {len(asked)} experiments the map says are worth building; the first three:")
    for what in asked[:3]:
        print(f"    {what.name}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
