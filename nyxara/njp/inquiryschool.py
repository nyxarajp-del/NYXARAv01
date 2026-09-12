"""NYXARA · njp/inquiryschool.py — is designing an experiment better than picking one (🔬📏).

:class:`nyxara.njp.universe.ExperimentDesigner` has existed since V.04 and is careful work. It
computes expected information gain honestly, it refuses an experiment every live hypothesis
predicts identically, and it kills a hypothesis that made a commitment and was contradicted rather
than softening it into a decrement.

**It has never been compared to anything.** One test hands it three named experiments and checks it
picks the one a person would pick. That is the shape of defect this package has found four times in
a week — the entailer's single rule, the span stage's 0.0259, the forge's timing gate, the task
learner's fifth — and in every case the mechanism turned out to be fine, or not, and nobody could
say which, because there was no floor under the number.

So the question here is the only one that matters about an experiment designer:

    Does choosing the most informative experiment find the truth in fewer experiments than
    choosing one at random?

If it does not, the entropy arithmetic is decoration. Four strategies over the same worlds:

* **designed** — the organ.
* **random** — a uniform pick among the experiments not yet run. The floor.
* **in order** — the fixed list, top to bottom. What a person does without a designer.
* **worst** — deliberately the *lowest* positive gain. Not a competitor: a check that the ranking
  has a direction at all, since a designer whose best and worst pick alike would be sorting noise.

And the worlds are built so that the answer is not guaranteed: some of them are **unidentifiable**,
where no sequence of the available experiments separates the hypotheses. Those are counted apart,
because a strategy that never finishes on an impossible world is not failing.

**Where it works, and where it stops.** The answer is not one number, and reporting only the
setting where the organ looks best would be the same defect one level up. Swept across how many
experiments are available per hypothesis (300 worlds each):

===========  ====  =========  ========  ==========  ======  ========
hypotheses   exps   designed    random    in order    worst     saved
===========  ====  =========  ========  ==========  ======  ========
          4    12      1.164     1.808       1.833    2.204    +0.644
          5     8      1.800     2.496       2.387    2.668    +0.696
          8     5      2.185     2.815       2.733    2.831    +0.630
          6     3      2.375     2.562       2.473    2.305    +0.187
         10     4      3.154     3.454       3.323    3.092    +0.300
         12     3      2.945     2.782       2.746    2.600    -0.163
===========  ====  =========  ========  ==========  ======  ========

It saves a third of the experiments while they are plentiful, and the advantage fades to nothing —
then reverses — as they become scarce. Worse, in the last three rows **the least informative choice
finishes sooner than the most informative one**, which means the ranking has inverted rather than
merely flattened.

The reason is a mismatch nobody had noticed because nobody had measured it: the organ maximises
**expected bits per experiment**, and what is wanted is **experiments until settled**. Those agree
while there is room to halve the hypothesis set repeatedly. They come apart when there is not: an
experiment with a lopsided outcome distribution has low *expected* gain and may, on the outcome
that actually occurs, rule out almost everything at once. Averaging over outcomes is the right thing
for bits and the wrong thing for steps.

That is a real boundary on a real capability, and it is written here rather than in a future
version, because the setting this organ is used at in :mod:`nyxara.njp.field` is the plentiful one
and the claim is true there.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.universe import ExperimentDesigner

__all__ = ["World", "Run", "Report", "make_world", "inquire", "examine", "run", "SEED",
           "STRATEGIES", "SETTLED"]

SEED = 79

#: How sure she has to be of one hypothesis before the question counts as settled. Not certainty —
#: the update rules out a contradicted hypothesis outright, so what is left is the chance that
#: several survivors remain consistent with everything run so far.
SETTLED = 0.90

#: The strategies compared. `designed` is the organ; the rest are what it has to beat, and `worst`
#: is there to show the ranking points somewhere rather than to compete.
STRATEGIES: Tuple[str, ...] = ("designed", "random", "in order", "worst")


@dataclass
class World:
    """A set of rival hypotheses, the experiments available, and which hypothesis is true.

    ``truth`` is never shown to any strategy. It decides only what outcome comes back when an
    experiment is run, which is exactly what reality does.
    """

    hypotheses: Tuple[str, ...] = ()
    experiments: Tuple[str, ...] = ()
    #: hypothesis -> experiment -> the outcome that hypothesis predicts.
    predicts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    truth: str = ""

    def outcome(self, experiment: str) -> str:
        """What actually happens: whatever the true hypothesis said would."""
        return self.predicts.get(self.truth, {}).get(experiment, "")

    @property
    def identifiable(self) -> bool:
        """Could *any* sequence of these experiments single the truth out?

        No, when some other hypothesis predicts the same outcome for every available experiment —
        then the two are observationally equivalent given what can be done, and no strategy can
        separate them. A world like that is not a hard world; it is a world with no answer in it,
        and scoring a strategy on one measures nothing.
        """
        mine = self.predicts.get(self.truth, {})
        for name, said in self.predicts.items():
            if name == self.truth:
                continue
            if all(said.get(e) == mine.get(e) for e in self.experiments):
                return False
        return True


def make_world(rng: random.Random, *, hypotheses: int = 5, experiments: int = 8,
               outcomes: int = 3) -> World:
    """One world: each hypothesis commits to an outcome for each experiment.

    Committing to *every* experiment is what makes the hypotheses falsifiable, and the
    commitments are drawn at random, so some worlds come out identifiable and some do not without
    anybody arranging it.
    """
    # Lower case, because the designer normalises every name it is handed and returns the
    # normalised form. Handing it `E0` and looking for `E0` in the list of what is left finds
    # nothing, and the loop falls over — which it did.
    names = tuple(f"h{i}" for i in range(hypotheses))
    doing = tuple(f"e{i}" for i in range(experiments))
    predicts = {h: {e: f"o{rng.randrange(outcomes)}" for e in doing} for h in names}
    return World(hypotheses=names, experiments=doing, predicts=predicts,
                 truth=rng.choice(names))


@dataclass
class Run:
    """What one strategy did on one world."""

    strategy: str = ""
    asked: int = 0
    settled: bool = False
    right: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "asked": self.asked,
                "settled": self.settled, "right": self.right}


def _teach(world: World) -> ExperimentDesigner:
    """A fresh designer that knows the hypotheses and their commitments, and not the truth."""
    designer = ExperimentDesigner()
    for name in world.hypotheses:
        designer.propose(name, probability=1.0 / len(world.hypotheses),
                         predictions=dict(world.predicts[name]))
    return designer


def _pick(strategy: str, designer: ExperimentDesigner, left: Sequence[str],
          rng: random.Random) -> Optional[str]:
    """Which experiment this strategy runs next, from the ones not yet run."""
    if not left:
        return None
    if strategy == "random":
        return rng.choice(list(left))
    if strategy == "in order":
        return left[0]
    scored = [designer.evaluate(name) for name in left]
    worth = [e for e in scored if e.gain > 1e-9]
    if not worth:
        return None                       # nothing left tells her anything: stop rather than flail
    if strategy == "worst":
        return min(worth, key=lambda e: (e.gain, e.name)).name
    chosen = designer.design(list(left))
    return chosen.name if chosen is not None else None


def inquire(world: World, strategy: str, rng: random.Random, *, budget: int = 0) -> Run:
    """Run one strategy on one world until it is sure, or out of experiments."""
    designer = _teach(world)
    left = list(world.experiments)
    out = Run(strategy=strategy)
    for _ in range(budget or len(world.experiments)):
        name = _pick(strategy, designer, left, rng)
        if name is None:
            break
        left.remove(name)
        out.asked += 1
        designer.observe_result(name, world.outcome(name))
        best = max(designer.hypotheses.values(), key=lambda h: h.probability)
        if best.probability >= SETTLED:
            out.settled = True
            out.right = best.name == world.truth
            break
    return out


@dataclass
class Report:
    """How a strategy did across many worlds."""

    strategy: str = ""
    runs: List[Run] = field(default_factory=list)

    @property
    def settled(self) -> List[Run]:
        return [r for r in self.runs if r.settled]

    @property
    def asked(self) -> float:
        """Experiments to reach an answer, over the worlds where it reached one.

        Averaged over *settled* runs only, and that needs saying: a strategy that gives up early on
        hard worlds would otherwise look fast. :attr:`solved` is printed beside it so the two
        cannot be read apart.
        """
        return round(statistics.mean([r.asked for r in self.settled]), 3) if self.settled else 0.0

    @property
    def solved(self) -> float:
        return round(len(self.settled) / len(self.runs), 4) if self.runs else 0.0

    @property
    def accuracy(self) -> float:
        """Of the questions it settled, how many it settled correctly. A fast wrong answer is worse
        than a slow right one, so this is reported beside the speed and never folded into it."""
        return round(sum(1 for r in self.settled if r.right) / len(self.settled), 4) \
            if self.settled else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, "asked": self.asked, "solved": self.solved,
                "accuracy": self.accuracy, "worlds": len(self.runs)}

    def render(self) -> str:
        return (f"  {self.strategy:<9} {self.asked:>6.3f} experiments   "
                f"settled {self.solved:.3f}   right when settled {self.accuracy:.3f}")


def examine(worlds: int = 400, *, seed: int = SEED, hypotheses: int = 5,
            experiments: int = 8, outcomes: int = 3) -> Dict[str, Any]:
    """Every strategy on the same worlds, and the pass condition written down.

    The same worlds for all four — a strategy measured on its own sample of worlds is measured
    against a different question, which is the confound that made the first 90-row task figures
    incomparable.
    """
    rng = random.Random(seed)
    built = [make_world(rng, hypotheses=hypotheses, experiments=experiments, outcomes=outcomes)
             for _ in range(worlds)]
    live = [w for w in built if w.identifiable]
    out = {name: Report(strategy=name) for name in STRATEGIES}
    for i, world in enumerate(live):
        for name in STRATEGIES:
            # One seed per (world, strategy) so the random strategy is random and the whole
            # examination still repeats exactly.
            out[name].runs.append(inquire(world, name, random.Random(seed + i * 17)))
    got: Dict[str, Any] = {name: rep.to_dict() for name, rep in out.items()}
    got["worlds"] = len(built)
    got["identifiable"] = len(live)
    got["unidentifiable"] = len(built) - len(live)
    designed, chance = out["designed"], out["random"]
    got["saved_vs_random"] = round(chance.asked - designed.asked, 3)
    got["saved_vs_in_order"] = round(out["in order"].asked - designed.asked, 3)
    got["worst_costs"] = round(out["worst"].asked - designed.asked, 3)
    got["passes"] = bool(
        designed.asked < chance.asked                 # it beats picking at random
        and designed.solved >= chance.solved - 0.01   # and not by giving up on hard worlds
        and designed.accuracy >= 0.99                 # and a fast wrong answer is not a win
        and out["worst"].asked > designed.asked       # and the ranking points somewhere
    )
    return got


#: The regimes the sweep walks, from most experiments per hypothesis to fewest.
REGIMES: Tuple[Tuple[int, int, int], ...] = (
    (4, 12, 4), (5, 8, 3), (8, 5, 3), (6, 3, 2), (10, 4, 2), (12, 3, 2),
)


def sweep(worlds: int = 300, regimes: Sequence[Tuple[int, int, int]] = REGIMES
          ) -> List[Dict[str, Any]]:
    """How the advantage changes as experiments become scarce relative to hypotheses.

    Ordered by experiments **per hypothesis**, most to fewest — the dial is the ratio, not the raw
    count, and listing them by raw count put a 0.4 regime before a 0.5 one.

    Reported as a ladder rather than a single figure, for the reason :mod:`nyxara.njp.reach` gives:
    one number about a capability cannot say whether what it misses is the hard end or scattered,
    and here it is emphatically the hard end.
    """
    out: List[Dict[str, Any]] = []
    for hyps, exps, outs in regimes:
        got = examine(worlds, hypotheses=hyps, experiments=exps, outcomes=outs)
        out.append({"hypotheses": hyps, "experiments": exps, "outcomes": outs,
                    "designed": got["designed"]["asked"], "random": got["random"]["asked"],
                    "in order": got["in order"]["asked"], "worst": got["worst"]["asked"],
                    "saved": got["saved_vs_random"],
                    "ranking_holds": got["worst"]["asked"] > got["designed"]["asked"],
                    "unidentifiable": got["unidentifiable"], "passes": got["passes"]})
    return out


def run(worlds: int = 400) -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine(worlds)
    print(f"{got['worlds']} worlds, {got['identifiable']} of them identifiable "
          f"({got['unidentifiable']} have no answer in them and are set aside).\n")
    for name in STRATEGIES:
        row = got[name]
        print(f"  {name:<9} {row['asked']:>6.3f} experiments   settled {row['solved']:.3f}   "
              f"right when settled {row['accuracy']:.3f}")
    print(f"\n  designing saves {got['saved_vs_random']:.3f} experiments against picking at "
          f"random, {got['saved_vs_in_order']:.3f} against working down the list")
    print(f"  choosing the least informative costs {got['worst_costs']:.3f} more than choosing "
          f"the most, so the ranking has a direction")
    print(f"  passes {got['passes']}")
    print("\nand where that stops being true, as experiments get scarce:\n")
    print("  hyps  exps |  designed   random   worst |    saved  ranking holds")
    rungs = sweep(min(300, worlds))
    for row in rungs:
        print(f"  {row['hypotheses']:>4} {row['experiments']:>5} | {row['designed']:>9.3f} "
              f"{row['random']:>8.3f} {row['worst']:>7.3f} | {row['saved']:>+8.3f}  "
              f"{'yes' if row['ranking_holds'] else 'NO'}")
    held = [r for r in rungs if r["passes"]]
    print(f"\n  it holds down to {held[-1]['experiments']} experiments per "
          f"{held[-1]['hypotheses']} hypotheses, and not below" if held else
          "\n  it does not hold anywhere swept")
    got["sweep"] = rungs
    return got
