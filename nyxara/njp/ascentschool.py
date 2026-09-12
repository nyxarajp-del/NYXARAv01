"""NYXARA · njp/ascentschool.py — is diagnosing worth more than guessing (🧗📏).

The loop in :mod:`nyxara.njp.ascent` finds where a capability stops, asks why the next setting
fails, repairs *that*, and keeps the repair only if hidden problems improve. Every part of it has
been measured on its own. What has not been measured is whether assembling them is worth anything.

So each capability here is broken in **exactly one** of four ways, chosen at random and not
disclosed, and four repairs are on the shelf — one per way. Three strategies pick from that same
shelf and differ nowhere else:

* **diagnosed** — ask :mod:`nyxara.njp.attribution` what the cause is, take the repair for it.
* **blind** — pick one of the four at random. The floor.
* **greedy** — always reach for the same favourite, which is what a system with a pet theory does.

With four repairs and one right answer, blind reaches for the right one a quarter of the time and
greedy a quarter of the time across many capabilities. Anything the loop is worth shows up as
distance from those two, and if there is none then diagnosis is bookkeeping.

**Two numbers, and the second is the one that can embarrass the first.** `moved the edge` is
whether the capability actually reaches further afterwards. `kept-but-no-gain` is how often a
repair was kept and the edge did not move — a gate too generous to be worth having, measured rather
than assumed away.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from nyxara.njp.ascent import HIDDEN, STRATEGIES, Ascent, Repair, ascend
from nyxara.njp.attribution import Failure
from nyxara.njp.measurement import Benchmark

__all__ = ["Broken", "WAYS", "REPAIRS", "examine", "run", "SEED", "DIAL"]

SEED = 82

#: Difficulty settings, easiest first.
DIAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: How many items a rung is scored on.
ITEMS = 400

#: The four ways a capability is broken here, and the repair that fixes each. A repair applied to
#: the wrong break does nothing at all, which is what makes choosing among them measurable.
WAYS: Tuple[str, ...] = ("data", "algorithm", "reading", "budget")


@dataclass
class Broken:
    """A capability that works to some setting and then fails, for exactly one reason."""

    way: str = ""
    edge: int = 3
    seed: int = 0

    #: How much further the matching repair carries it. Two settings, not all of them: a repair
    #: that makes every setting work is not a boundary that moved, it is a different capability,
    #: and the first version of this fixture did that — leaving the repaired ladder `exhausted`
    #: with no edge to compare against the old one.
    REACH = 2

    def works(self, at: Any, *, fixed: str = "") -> float:
        """How well it does at this setting, with `fixed` repaired.

        Past its edge it sits at its floor. The **matching** repair carries it `REACH` settings
        further and no further; any other repair leaves it exactly where it was — not slightly
        better, exactly the same — so a strategy cannot stumble into a gain by trying things.
        """
        limit = self.edge + (self.REACH if fixed == self.way else 0)
        return 0.92 if at <= limit else 0.50

    def bench(self, at: Any, *, fixed: str = "") -> Benchmark:
        rng = random.Random(self.seed)
        gold = ["A" if rng.random() < 0.5 else "B" for _ in range(ITEMS)]
        share = self.works(at, fixed=fixed)
        hit = random.Random(self.seed + int(at) * 7 + 1)
        said = [g if hit.random() < share else ("B" if g == "A" else "A") for g in gold]
        return Benchmark(name=f"{at}", items=list(range(ITEMS)), gold=gold,
                         predict=lambda i: said[i])

    def hidden(self, at: Any, *, fixed: str = "", draw: int = 0) -> float:
        """Score on problems drawn fresh — never the ones the edge or the diagnosis was read from."""
        rng = random.Random(self.seed + 9_000 + draw)
        share = self.works(at, fixed=fixed)
        return round(sum(1 for _ in range(HIDDEN) if rng.random() < share) / HIDDEN, 4)

    def failure(self, at: Any) -> Failure:
        """The rung past the edge, with every repair available as an experiment to try on it.

        All four are supplied whatever the break is, so no cause is credited for being the only
        hypothesis anybody could test — the rule :mod:`nyxara.njp.attribution` is built on.
        """
        return Failure(
            name=f"setting {at}", bench=self.bench(at),
            more_data=lambda: self.bench(at, fixed="data"),
            other_algorithm=lambda: self.bench(at, fixed="algorithm"),
            richer_reading=lambda: self.bench(at, fixed="reading"),
            more_budget=lambda: self.bench(at, fixed="budget"))

    def repairs(self) -> List[Repair]:
        return [Repair(name=f"fix {w}", addresses=w,
                       apply=(lambda s, w=w: self.bench(s, fixed=w)))
                for w in WAYS]


REPAIRS = WAYS


def _one(broken: Broken, strategy: str, rng: random.Random) -> Any:
    return ascend(
        broken.bench, DIAL, broken.repairs(),
        experiments=lambda at, _x: broken.failure(at),
        # One `draw` for both readings: the same hidden problems before and after, so what moves
        # the number is the repair rather than a second sample of the same noise.
        hidden=lambda repair, at: broken.hidden(
            at, fixed=(repair.addresses if repair is not None else ""), draw=0),
        strategy=strategy, rng=rng, favourite="data")


def examine(capabilities: int = 120, *, seed: int = SEED) -> Dict[str, Any]:
    """Every strategy on the same capabilities, broken the same ways.

    The same set for all three, because a strategy measured on its own sample of capabilities is
    answering a different question — the confound that made the first task-learner figures
    incomparable.
    """
    rng = random.Random(seed)
    built = [Broken(way=rng.choice(WAYS), edge=rng.choice((2, 3, 4)), seed=seed + i * 13)
             for i in range(capabilities)]
    out = {name: Ascent(strategy=name) for name in STRATEGIES}
    for i, broken in enumerate(built):
        for name in STRATEGIES:
            out[name].attempts.append(_one(broken, name, random.Random(seed + i * 31)))
    got: Dict[str, Any] = {name: a.to_dict() for name, a in out.items()}
    got["capabilities"] = len(built)
    got["moved_over_blind"] = round(out["diagnosed"].moved - out["blind"].moved, 4)
    got["moved_over_greedy"] = round(out["diagnosed"].moved - out["greedy"].moved, 4)
    got["passes"] = bool(
        out["diagnosed"].moved > out["blind"].moved            # diagnosing beats guessing
        and out["diagnosed"].moved > out["greedy"].moved       # and beats a pet theory
        and out["diagnosed"].kept_nothing <= 0.05              # without keeping what does nothing
    )
    return got


def run(capabilities: int = 120) -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine(capabilities)
    print(f"{got['capabilities']} capabilities, each broken in exactly one of four ways.")
    print("four repairs on the shelf, one per way. the three strategies differ only in how")
    print("they pick from it.\n")
    for name in STRATEGIES:
        row = got[name]
        print(f"  {name:<10} moved the edge {row['moved']:.3f}   "
              f"chose the repair that fits {row['right_repair']:.3f}   "
              f"kept {row['kept']:.3f}   kept-but-no-gain {row['kept_nothing']:.3f}")
    print(f"\n  diagnosing is worth {got['moved_over_blind']:+.3f} against guessing and "
          f"{got['moved_over_greedy']:+.3f} against a favourite")
    print(f"  passes {got['passes']}")
    return got
