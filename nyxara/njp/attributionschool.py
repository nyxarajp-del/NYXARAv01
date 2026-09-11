"""NYXARA · njp/attributionschool.py — does it find the cause, or just name one (🩺).

An attributor has three failure modes and only the first is obvious.

* It can name the wrong cause. Obvious, and the least dangerous, because the repair then visibly
  fails.
* It can name **one cause for everything**. This reads as decisiveness and is worth nothing — the
  test for it is that the roots it returns across a spread of failures are not all the same word.
* It can name a cause it never tested. This is the one that matters here, because it is how a
  favourite explanation survives: the hypothesis that was never put at risk comes out looking as
  good as the one that was.

So eight failures are built whose cause is known by construction — one for each hypothesis the
organ holds, plus one that nothing tested can explain. Every fixture supplies **every** experiment,
so a cause is not credited for being the only one anybody looked at, and the discriminating repair
is the only thing that separates them.

Scored three ways. **Correct** is naming the cause the fixture was built around. **Wrong** is naming
a different one, which is worse than naming none. **Unsettled** is declining to name one, which is
the right answer for the eighth fixture and a miss for the other seven.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.attribution import Attribution, Failure, attribute, chain
from nyxara.njp.measurement import Benchmark

__all__ = ["Case", "KNOWN", "examine", "retrodict", "walk_the_chain", "run",
           "SEED", "FEATURES"]

SEED = 77

#: How many readings the synthetic world offers. The first few carry the answer; the rest are
#: noise, which is what makes `reading` and `algorithm` separable at all.
FEATURES = 40

#: How many values a reading takes. Twelve rather than four, so that a *disjunction* of six
#: readings can still be near even. The first version used four, and "A if any of six readings is
#: zero" then made A the answer 82% of the time — so always saying A already scored 0.815, no
#: learner could beat it, and three fixtures built around `data` and `algorithm` collapsed into
#: `floor`. A task whose majority answer is nearly always right tests nothing.
VALUES = 12


# --------------------------------------------------------------------------------------------- #
#  one small world, and learners that differ in exactly one way each
# --------------------------------------------------------------------------------------------- #
def _trigger(ways: int) -> int:
    """How many values of one reading say `A`, chosen so the two answers come out near even.

    P(A) is ``1 - (1 - t/VALUES) ** ways``, so the trigger has to shrink as the disjunction widens
    or the task skews until the majority answer wins by itself. Solved for a half and rounded up,
    which keeps every fixture's floor near 0.5 whatever its shape.
    """
    share = 1.0 - 0.5 ** (1.0 / max(1, ways))
    return max(1, round(share * VALUES))


def _world(n: int, seed: int = SEED, ways: int = 1) -> Tuple[List[List[int]], List[str]]:
    """Items and answers, where the answer is carried by the first ``ways`` readings.

    With ``ways`` above one the answer is a *disjunction* — any of several readings can carry it —
    which is the shape a single-rule learner cannot express however much data it is given. That is
    what makes `algorithm` distinguishable from `data` rather than a matter of taste, and the
    trigger is sized by :func:`_trigger` so that widening it does not quietly skew the answers.
    """
    rng = random.Random(seed)
    hot = _trigger(ways)
    rows = [[rng.randrange(VALUES) for _ in range(FEATURES)] for _ in range(n)]
    gold = ["A" if any(r[w] < hot for w in range(ways)) else "B" for r in rows]
    return rows, gold


def _learner(rows: Sequence[Sequence[int]], readings: Sequence[int], rules: int = 6):
    """Keep up to ``rules`` single-reading rules that are pure enough on what it was shown.

    Three knobs and nothing else: which readings it may look at, how many rules it may keep, and
    (through the caller) how many rows it sees. Each fixture below moves exactly one of them.
    """
    def _learn(train: Sequence[int], gold: Sequence[str]) -> Callable[[int], str]:
        commonest = max(set(gold), key=list(gold).count) if len(gold) else "B"
        found: List[Tuple[int, int, str]] = []
        for f in readings:
            for value in range(VALUES):
                hits = [g for i, g in zip(train, gold) if rows[i][f] == value]
                if len(hits) < 4:
                    continue
                said = max(set(hits), key=hits.count)
                if hits.count(said) / len(hits) >= 0.9:
                    found.append((f, value, said))
        kept = found[:max(1, rules)]
        # The fallback is the commonest answer **among the rows no kept rule covers**, not the
        # commonest overall. With the latter, a learner that had found every rule for one answer
        # and none for the other fell back to that same answer and predicted it everywhere — so
        # `richer readings` scored exactly what `thin readings` did and the fixture built around
        # readings collapsed into `floor`.
        left = [g for i, g in zip(train, gold)
                if not any(rows[i][f] == v for f, v, _s in kept)]
        spare = max(set(left), key=left.count) if left else commonest

        def _say(i: int) -> str:
            for f, value, said in kept:
                if rows[i][f] == value:
                    return said
            return spare
        return _say
    return _learn


#: How many rows every fixture holds back. **The same rows, for every experiment**, and that is
#: not a detail. The first version cut the held-out set at the end of training, so an experiment
#: that added rows also moved the examination — and the before and after were then two different
#: questions. `data` and `algorithm` both came out flat and the attributor fell through to `floor`
#: on cases built around them. The organ's own rule is that an experiment changes exactly one
#: thing; the fixtures broke it before the organ did.
HELD = 200


def _bench(name: str, rows, gold, readings, *, n_train: int, rules: int = 6,
           reachable: Optional[Callable[[int, str], bool]] = None,
           train_overlaps: bool = False, unstable: bool = False) -> Benchmark:
    learn = _learner(rows, readings, rules)
    held = list(range(len(rows) - HELD, len(rows)))
    pool = list(range(len(rows) - HELD))
    train = held[:n_train] if train_overlaps else pool[:n_train]
    said = learn(train, [gold[i] for i in train])
    rng = random.Random(SEED)
    predict = (lambda i: said(i) if rng.random() < 0.5 else "B") if unstable else said
    return Benchmark(name=name, items=held, gold=[gold[i] for i in held], predict=predict,
                     learn=learn, train=train, key=lambda i: i, reachable=reachable)


# --------------------------------------------------------------------------------------------- #
#  eight failures whose cause is known by construction
# --------------------------------------------------------------------------------------------- #
def _all_experiments(rows, gold, *, n_train: int, readings, rules: int,
                     data_rows: Optional[int] = None, better_readings=None,
                     better_rules: Optional[int] = None,
                     better_budget: Optional[int] = None) -> Dict[str, Any]:
    """Every experiment, so no hypothesis is credited merely for being the only one tried."""
    return {
        "more_data": lambda: _bench("more data", rows, gold, readings,
                                    n_train=data_rows or n_train, rules=rules),
        "other_algorithm": lambda: _bench("other algorithm", rows, gold, readings,
                                          n_train=n_train, rules=better_rules or rules),
        "richer_reading": lambda: _bench("richer readings", rows, gold,
                                         better_readings if better_readings is not None
                                         else readings, n_train=n_train, rules=rules),
        "more_budget": lambda: _bench("more budget", rows, gold, readings,
                                      n_train=n_train, rules=better_budget or rules),
    }


def _starved() -> Failure:
    """Enough readings, enough rules, too few rows. Only more data helps.

    The answer is a disjunction of six readings, so twenty-four rows cannot contain enough of each
    disjunct to find them — and *that* is what makes this separable from
    :func:`_wrong_algorithm`, where the rows are plentiful and the rules are not. The first version
    of this fixture used a single-reading answer and the learner scored **1.000** on twenty-four
    rows, so there was no failure to attribute at all and the exam was testing nothing.
    """
    rows, gold = _world(900, SEED + 1, ways=6)
    readings = list(range(FEATURES))
    return Failure(name="too few examples",
                   bench=_bench("as found", rows, gold, readings, n_train=18, rules=12),
                   **_all_experiments(rows, gold, n_train=18, readings=readings, rules=12,
                                      data_rows=700, better_rules=24))


def _wrong_algorithm() -> Failure:
    """The answer is a disjunction of six readings, and the chooser may keep one rule."""
    rows, gold = _world(700, SEED + 2, ways=6)
    readings = list(range(FEATURES))
    return Failure(name="one rule where six are needed",
                   bench=_bench("as found", rows, gold, readings, n_train=500, rules=1),
                   **_all_experiments(rows, gold, n_train=500, readings=readings, rules=1,
                                      data_rows=600, better_rules=12))


def _thin_readings() -> Failure:
    """Plenty of rows and rules, but the reading that carries the answer is not offered."""
    rows, gold = _world(700, SEED + 3)
    thin = list(range(1, 12))
    return Failure(name="the answer is not in the readings",
                   bench=_bench("as found", rows, gold, thin, n_train=500, rules=12),
                   **_all_experiments(rows, gold, n_train=500, readings=thin, rules=12,
                                      data_rows=600, better_readings=list(range(FEATURES))))


def _low_ceiling() -> Failure:
    """The right answer is not producible for most items, so nothing downstream can reach it."""
    rows, gold = _world(700, SEED + 4)
    readings = list(range(FEATURES))
    reach = {i: (i % 10 < 3) for i in range(700)}
    return Failure(name="the answer is rarely producible",
                   bench=_bench("as found", rows, gold, readings, n_train=500, rules=12,
                                reachable=lambda i, _w: reach[i]),
                   **_all_experiments(rows, gold, n_train=500, readings=readings, rules=12,
                                      data_rows=600))


def _leaky() -> Failure:
    """A good score, taken on the rows it was taught."""
    rows, gold = _world(700, SEED + 5)
    readings = list(range(FEATURES))
    return Failure(name="examined on what it was taught",
                   bench=_bench("as found", rows, gold, readings, n_train=500, rules=12,
                                train_overlaps=True),
                   **_all_experiments(rows, gold, n_train=500, readings=readings, rules=12,
                                      data_rows=600))


def _no_floor() -> Failure:
    """It scores well and so does always saying the commonest answer."""
    rng = random.Random(SEED + 6)
    rows = [[rng.randrange(VALUES) for _ in range(FEATURES)] for _ in range(700)]
    gold = ["A" if rng.random() < 0.88 else "B" for _ in range(700)]
    readings = list(range(FEATURES))
    return Failure(name="a score its own floor already reaches",
                   bench=_bench("as found", rows, gold, readings, n_train=500, rules=12),
                   **_all_experiments(rows, gold, n_train=500, readings=readings, rules=12,
                                      data_rows=600))


def _unstable() -> Failure:
    """The reading moves between identical runs, so there is nothing yet to attribute."""
    rows, gold = _world(700, SEED + 7)
    readings = list(range(FEATURES))
    return Failure(name="a score that will not hold still",
                   bench=_bench("as found", rows, gold, readings, n_train=500, rules=12, unstable=True),
                   **_all_experiments(rows, gold, n_train=500, readings=readings, rules=12,
                                      data_rows=600))


def _irreducible() -> Failure:
    """Already at the ceiling the task allows: well clear of its floor, and no lever reaches higher.

    The answer is carried by one reading and then a third of the labels are flipped, so the best
    any of these levers can reach is about 0.70 against a floor of 0.50. Nothing is broken and
    nothing is repairable, and **naming no cause is the correct answer** — which is why this case
    exists. An attributor that always produces a word would score a hit on the seven above and be
    wrong here, and that is the difference between diagnosing and labelling.
    """
    rng = random.Random(SEED + 8)
    hot = _trigger(1)
    rows = [[rng.randrange(VALUES) for _ in range(FEATURES)] for _ in range(900)]
    # Eight percent, not a third. The learner keeps a rule only at purity 0.9 or better, so a
    # third of the labels flipped puts every real rule under its bar, it finds nothing, and the
    # fixture stops being "already at the ceiling" and becomes "at the floor" — a different case,
    # already covered by `_no_floor`.
    gold = [("A" if r[0] < hot else "B") if rng.random() > 0.08 else
            ("B" if r[0] < hot else "A") for r in rows]
    readings = list(range(FEATURES))
    return Failure(name="already at what the task allows",
                   bench=_bench("as found", rows, gold, readings, n_train=600, rules=12),
                   **_all_experiments(rows, gold, n_train=600, readings=readings, rules=12,
                                      data_rows=800, better_rules=20))


@dataclass
class Case:
    name: str = ""
    #: The cause the fixture was built around. Empty means no available experiment explains it,
    #: and declining to name one is then the right answer rather than a miss.
    cause: str = ""
    build: Optional[Callable[[], Failure]] = None
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("too few examples", "data", _starved, "enough readings and rules, 24 rows"),
    Case("one rule where six are needed", "algorithm", _wrong_algorithm,
         "a disjunction of six, and one rule allowed"),
    Case("the answer is not in the readings", "reading", _thin_readings,
         "the reading that carries it is not offered"),
    Case("the answer is rarely producible", "reachability", _low_ceiling,
         "producible for three items in ten"),
    Case("examined on what it was taught", "leakage", _leaky, "the split overlaps"),
    Case("a score its own floor already reaches", "floor", _no_floor,
         "the majority answer scores the same and no repair moves it"),
    Case("a score that will not hold still", "measurement", _unstable,
         "identical runs disagree, so nothing downstream can be concluded"),
    Case("already at what the task allows", "", _irreducible,
         "clear of its floor, and no lever reaches higher; naming no cause is correct"),
)


def walk_the_chain() -> List[Attribution]:
    """The failure behind the failure behind the failure, as it actually ran here.

    The span stage scored 0.0259 and two versions went into the ranker. The chain underneath it:

        the reader picks the wrong span
          └── the generator never proposed the right one       (reachability)
                └── the readings it generates from are wrong   (reading)
                      └── which is what the measurement was not able to say

    Only the last link is worth repairing, and only the first was visible. That is the whole case
    for walking it: an attribution that stops at the immediate cause sends the work to the ranker.
    """
    deepest = _thin_readings()
    middle = _low_ceiling()
    middle.upstream = deepest
    top = _starved()
    top.upstream = middle
    return chain(top)


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Eight failures with known causes, and what the attributor made of them."""
    rows: List[Dict[str, Any]] = []
    correct = wrong = unsettled = 0
    roots: List[str] = []
    for case in cases:
        got: Attribution = attribute(case.build())
        root = got.root
        roots.append(root)
        if root == case.cause:
            correct += 1
        elif root:
            wrong += 1
        else:
            unsettled += 1
        rows.append({"case": case.name, "want": case.cause, "got": root,
                     "rivals": got.rivals, "refuted": got.refuted,
                     "untested": got.untested, "note": case.note, "attribution": got})
    named = {r for r in roots if r}
    return {"rows": rows, "correct": correct, "wrong": wrong, "unsettled": unsettled,
            "of": len(cases), "accuracy": round(correct / max(1, len(cases)), 4),
            "distinct_roots": len(named), "roots": sorted(named)}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Three conditions. Most of them right; **no** confident wrong answers, since a wrong cause sends
    the repair somewhere real and costs more than an honest shrug; and more than one distinct cause
    named across the eight, because an attributor that says `data` to everything would otherwise
    score well on any set of fixtures that happened to be about data.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["accuracy"] >= 0.75 and got["wrong"] == 0
                         and got["distinct_roots"] >= 4)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print("eight failures whose cause is known by construction. every experiment is supplied for")
    print("each, so no hypothesis is credited for being the only one anybody tried.\n")
    for row in got["rows"]:
        mark = "ok" if row["got"] == row["want"] else ("!!" if row["got"] else " ~")
        want = row["want"] or "(none)"
        print(f" [{mark}] {row['case']:<38} want {want:<13} got {row['got'] or '(none)':<13}"
              f"{'  rivals: ' + '/'.join(row['rivals']) if row['rivals'] else ''}")
    print(f"\n  correct {got['correct']} of {got['of']}   wrong {got['wrong']}   "
          f"unsettled {got['unsettled']}   distinct causes named {got['distinct_roots']}")
    print(f"  passes  {got['passes']}")
    print("\nand the failure behind the failure:\n")
    from nyxara.njp.attribution import render_chain
    print(render_chain(walk_the_chain()))
    print("\nin full:\n")
    for row in got["rows"]:
        print(row["attribution"].render())
        print()
    return {k: v for k, v in got.items() if k != "rows"}
