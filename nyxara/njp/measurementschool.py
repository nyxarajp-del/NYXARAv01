"""NYXARA · njp/measurementschool.py — does the critic find the bugs that were found by hand (📐).

A critic of measurements has one obvious failure mode and one subtle one. The obvious one is
missing a bad benchmark. The subtle one is condemning every benchmark, which reads as vigilance
and is worth exactly nothing: a check that always fires carries no information, and this package
has been caught by that before — a veto with a false-alarm rate of 0.0000 that was flawless
because it never fired at all.

So the examination is **retrodiction, in both directions**. Four measurements in this repository
were read as facts about mechanisms and turned out to be facts about the measurement, each found
by hand and each already fixed. The critic is shown all four, plus the fixed form of each, and has
to separate them without being told which is which.

Passing means: it flags the four that were wrong, and clears the four that were right. Flagging
all eight is a failure of this exam, not a pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.measurement import Benchmark, Critique, critique

__all__ = ["Case", "KNOWN", "retrodict", "examine", "run", "SEED"]

SEED = 74


@dataclass
class Case:
    """One historical measurement, and what was actually true of it."""

    name: str = ""
    #: Whether the measurement was misleading. This is the answer key, and it comes from the
    #: repository's own history rather than from anything the critic can see.
    was_misleading: bool = False
    #: Which check should notice. Named so a critic that flags the right case for the wrong reason
    #: is not credited with a hit.
    by: str = ""
    build: Optional[Callable[[], Benchmark]] = None
    note: str = ""


# --------------------------------------------------------------------------------------------- #
#  the four that were wrong, and the four that were right
# --------------------------------------------------------------------------------------------- #
def _labels(n: int, skew: float, seed: int = SEED) -> List[str]:
    rng = random.Random(seed)
    return ["A" if rng.random() < skew else "B" for _ in range(n)]


def _spurious_learner(features: int, rows: Sequence[Sequence[int]]):
    """A learner that keeps the single reading best correlated with the answer.

    This is not a straw man; it is the shape of the real defect. With forty-eight readings offered
    and thirty rows to choose on, *some* reading lines up with the answers by luck, and sometimes
    the same luck carries into the held-out rows. The mechanism is honest. What was missing was
    anything to compare its win rate against.
    """
    def _learn(train: Sequence[int], gold: Sequence[Any]) -> Callable[[int], Any]:
        best, mark = 0, -1.0
        for f in range(features):
            counts: Dict[Any, Dict[Any, int]] = {}
            for i, want in zip(train, gold):
                counts.setdefault(rows[i][f], {}).setdefault(want, 0)
                counts[rows[i][f]][want] += 1
            hit = sum(max(by.values()) for by in counts.values())
            if hit > mark:
                best, mark = f, hit
        table: Dict[Any, Any] = {}
        for i, want in zip(train, gold):
            table.setdefault(rows[i][best], []).append(want)
        say = {v: max(set(ws), key=ws.count) for v, ws in table.items()}
        fallback = max(set(gold), key=list(gold).count) if len(gold) else ""
        return lambda i: say.get(rows[i][best], fallback)
    return _learn


def _thin_task() -> Benchmark:
    """V.58's probe: 90 rows a task, 28 held out, 48 readings to choose among.

    0.227 of tasks 'beat their own floor' — and with the answers shuffled, 0.207 of them still did.
    The raw figure was the one reported. Only the null can tell these apart, which is why this
    fixture supplies `learn` and `train`: a critic given no means to run the null must say so, and
    the first version of this fixture forgot to, which is how it was caught.
    """
    rng = random.Random(SEED + 1)
    n, features = 90, 48
    rows = [[rng.randrange(3) for _ in range(features)] for _ in range(n)]
    gold = _labels(n, 0.55, SEED + 1)
    train, held = list(range(62)), list(range(62, n))
    learn = _spurious_learner(features, rows)
    said = learn(train, [gold[i] for i in train])
    return Benchmark(name="task learner, 90 rows a task",
                     items=held, gold=[gold[i] for i in held], predict=said,
                     learn=learn, train=train, key=lambda i: i)


def _thick_task() -> Benchmark:
    """The same learner at 300 rows a task, where a reading cannot line up by luck for long."""
    rng = random.Random(SEED + 2)
    n, features = 300, 48
    rows = [[rng.randrange(3) for _ in range(features)] for _ in range(n)]
    # A real signal this time: reading 0 carries the answer, with noise.
    gold = ["A" if (rows[i][0] == 0) != (rng.random() < 0.18) else "B" for i in range(n)]
    train, held = list(range(210)), list(range(210, n))
    learn = _spurious_learner(features, rows)
    said = learn(train, [gold[i] for i in train])
    return Benchmark(name="task learner, 300 rows a task",
                     items=held, gold=[gold[i] for i in held], predict=said,
                     learn=learn, train=train, key=lambda i: i)


def _bench_silent_but_precise() -> Benchmark:
    """The entailer at a purity bar of 0.72: right 78% of the time, on one pair in twenty."""
    gold = _labels(400, 0.43)
    rng = random.Random(SEED + 3)
    speaks = [rng.random() < 0.052 for _ in gold]
    said = [(want if rng.random() < 0.78 else "B") if s else "" for want, s in zip(gold, speaks)]
    order = list(range(len(gold)))
    return Benchmark(name="entailer at purity 0.72", items=order, gold=gold,
                     predict=lambda i: said[i], spoke=lambda i: speaks[i])


def _bench_speaks_and_is_right() -> Benchmark:
    """The same organ at the measured bar of 0.45: four pairs in five, right 54% of the time."""
    gold = _labels(400, 0.43)
    rng = random.Random(SEED + 4)
    speaks = [rng.random() < 0.78 for _ in gold]
    said = [(want if rng.random() < 0.54 else "B") if s else "" for want, s in zip(gold, speaks)]
    order = list(range(len(gold)))
    return Benchmark(name="entailer at purity 0.45", items=order, gold=gold,
                     predict=lambda i: said[i], spoke=lambda i: speaks[i])


def _bench_one_wall_clock_sample() -> Benchmark:
    """The native forge: the same kernel measured 39x to 77x, and one sample decided the gate."""
    rng = random.Random(SEED + 5)
    order = list(range(200))
    gold = ["fast"] * 200
    # A predictor whose verdict genuinely differs between calls, as a timing gate does.
    def _predict(_i: int) -> str:
        return "fast" if rng.random() < 0.6 else "slow"
    return Benchmark(name="forge gate, one timing each",
                     items=order, gold=gold, predict=_predict)


def _bench_best_of_five() -> Benchmark:
    """The same gate taking the best of five rounds: the reading stops moving."""
    order = list(range(200))
    gold = ["fast"] * 200
    return Benchmark(name="forge gate, best of five",
                     items=order, gold=gold, predict=lambda _i: "fast")


def _bench_leaky_split() -> Benchmark:
    """V.55: SQuAD asks a dozen questions per paragraph, and the cut was row-wise."""
    passages = [f"p{i // 12}" for i in range(600)]
    gold = _labels(600, 0.5, SEED + 6)
    rng = random.Random(SEED + 7)
    said = [want if rng.random() < 0.9 else "B" for want in gold]
    order = list(range(600))
    return Benchmark(name="passage reader, row-wise split", items=order, gold=gold,
                     predict=lambda i: said[i],
                     # Trained on rows drawn from the same passages it is examined on.
                     train=list(range(0, 600, 2)), key=lambda i: passages[i])


def _bench_passage_disjoint_split() -> Benchmark:
    """The same corpus cut by passage instead."""
    passages = [f"p{i // 12}" for i in range(600)]
    gold = _labels(600, 0.5, SEED + 6)
    rng = random.Random(SEED + 8)
    said = [want if rng.random() < 0.62 else "B" for want in gold]
    held = [i for i in range(600) if int(passages[i][1:]) >= 25]
    learned = [i for i in range(600) if int(passages[i][1:]) < 25]
    return Benchmark(name="passage reader, passage-disjoint split",
                     items=held, gold=[gold[i] for i in held],
                     predict=lambda i: said[i], train=learned, key=lambda i: passages[i])


#: The answer key. `was_misleading` is history, not anything the critic can see.
KNOWN: Tuple[Case, ...] = (
    Case("task learner, 90 rows a task", True, "shuffled", _thin_task,
         "shuffled answers score the same — the win is the machinery's, not the task's"),
    Case("task learner, 300 rows a task", False, "shuffled", _thick_task,
         "shuffled answers fall away; the distance is real"),
    Case("entailer at purity 0.72", True, "abstention", _bench_silent_but_precise,
         "right 78% of the time, on one pair in twenty; the headline says neither"),
    Case("entailer at purity 0.45", False, "abstention", _bench_speaks_and_is_right,
         "four pairs in five, right 54% of the time"),
    Case("forge gate, one timing", True, "stability", _bench_one_wall_clock_sample,
         "39x to 77x across identical forges, and one sample decided the gate"),
    Case("forge gate, best of five", False, "stability", _bench_best_of_five,
         "the reading stops moving"),
    Case("passage reader, row-wise split", True, "leakage", _bench_leaky_split,
         "a dozen questions per paragraph, cut row-wise"),
    Case("passage reader, passage-disjoint", False, "leakage", _bench_passage_disjoint_split,
         "cut by passage"),
)


def _flagged(got: Critique, by: str) -> bool:
    """Did the named check actually object? Not "did anything object" — the reason has to match."""
    for finding in got.findings:
        if finding.check == by:
            return finding.verdict in ("broken", "close")
    return False


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Show the critic eight measurements and see whether it separates them.

    Scored two ways, because either alone can be gamed: **caught** is how many of the misleading
    four it objects to, and **false alarms** is how many of the sound four it also objects to. A
    critic that flags everything scores 4 and 4, which is why both are printed and why
    :func:`examine` requires the second to be low.
    """
    rows: List[Dict[str, Any]] = []
    caught = alarms = 0
    for case in cases:
        bench = case.build() if case.build else Benchmark(name=case.name)
        got = critique(bench)
        objected = _flagged(got, case.by)
        if case.was_misleading and objected:
            caught += 1
        if not case.was_misleading and objected:
            alarms += 1
        rows.append({"case": case.name, "misleading": case.was_misleading, "by": case.by,
                     "objected": objected, "system": got.system,
                     "above_trivial": got.headline, "trusted": got.trusted,
                     "note": case.note, "critique": got})
    bad = sum(1 for c in cases if c.was_misleading)
    good = len(cases) - bad
    return {"rows": rows, "caught": caught, "of": bad,
            "false_alarms": alarms, "sound": good,
            "recall": round(caught / bad, 4) if bad else 0.0,
            "false_alarm_rate": round(alarms / good, 4) if good else 0.0}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with the pass condition stated rather than implied."""
    got = retrodict(cases)
    got["passes"] = bool(got["recall"] >= 0.75 and got["false_alarm_rate"] <= 0.25)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print("eight measurements from this repository's own history, four of which were misleading.")
    print("the critic is not told which.\n")
    for row in got["rows"]:
        mark = "!" if row["objected"] else " "
        truth = "misleading" if row["misleading"] else "sound      "
        print(f" [{mark}] {truth}  {row['case']:<38} {row['note']}")
    print(f"\n  caught        {got['caught']} of {got['of']}   (recall {got['recall']:.3f})")
    print(f"  false alarms  {got['false_alarms']} of {got['sound']}   "
          f"(rate {got['false_alarm_rate']:.3f})")
    print(f"  passes        {got['passes']}")
    print("\nthe critiques in full:\n")
    for row in got["rows"]:
        print(row["critique"].render())
        print()
    return {k: v for k, v in got.items() if k != "rows"}
