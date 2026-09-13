"""NYXARA · njp/askableschool.py — the exam for the wire (🔌, NJP V.100).

The protocol is `docs/V100_PROTOCOL.md`, committed before this file existed. Nothing here may
change it; if a number is disappointing the number is what gets reported.

Three buckets, per row: **correct**, **confabulation** (an answer that is wrong — including any
answer at all on an absent row), **miss** (UNKNOWN where something was there). Per family, never
one aggregate.

Both nulls run on every split, through the same scorer, from the same worlds.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.askable import Answer, World, always_unknown, ask, overlap_span, read_world

__all__ = ["Row", "Score", "Marks", "load_rows", "splits", "grade", "examine", "sit",
           "LOAD_BEARING", "intact"]

#: Symbols that must still exist for a number produced here to mean anything. Not a file list:
#: V.100 destroyed `njp.grounding` by *overwriting* it, and a fingerprint over paths would have
#: seen one tracked file change — which is what a normal edit looks like. What actually vanished
#: was a name. So the precondition asks for names.
LOAD_BEARING: Tuple[Tuple[str, str], ...] = (
    ("nyxara.njp.grounding", "Grounder"),
    ("nyxara.njp.passage", "PassageReader"),
    ("nyxara.njp.supply", "across"),
    ("nyxara.njp.integrity", "claim"),
    ("nyxara.njp.askable", "read_world"),
)


def intact() -> Tuple[str, ...]:
    """What is missing from :data:`LOAD_BEARING`. Empty when the laboratory is whole.

    V.93 made repository integrity a precondition and gave the package `integrity.claim`. V.100
    then overwrote a 3,770-line module anyway, because `claim` only helps a caller who remembers
    to call it, and the caller did not. A guard you must remember is a guard that has already
    failed. This one is not optional: :func:`examine` runs it before it will produce a number.
    """
    missing = []
    for module, symbol in LOAD_BEARING:
        try:
            mod = __import__(module, fromlist=[symbol])
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{module} does not import ({exc.__class__.__name__})")
            continue
        if not hasattr(mod, symbol):
            missing.append(f"{module}.{symbol} is gone")
    return tuple(missing)

CORPUS = os.path.join(os.path.dirname(__file__), "data", "flan_reading.jsonl.gz")
SEED = 100
DEVELOP, HELD, TRANSFER, ABSENT = 300, 500, 500, 500


@dataclass(frozen=True)
class Row:
    passage: str
    question: str
    answer: str
    family: str
    #: True when the passage genuinely does not contain the answer, so UNKNOWN is correct.
    absent: bool = False


def _family(task: str) -> str:
    return str(task or "?").split("/")[0].split(":")[0]


def load_rows(path: str = CORPUS) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def splits(path: str = CORPUS) -> Dict[str, List[Row]]:
    """The four splits of the protocol, built the same way every time."""
    raw = load_rows(path)
    squad = [r for r in raw if _family(r["task"]) == "squad"]
    other = [r for r in raw if _family(r["task"]) != "squad"]
    rng = random.Random(SEED)
    rng.shuffle(squad)
    rng.shuffle(other)

    def rows(source: Sequence[Dict[str, Any]]) -> List[Row]:
        return [Row(passage=r["passage"], question=r["question"], answer=str(r["answer"]),
                    family=_family(r["task"])) for r in source]

    develop = rows(squad[:DEVELOP])
    held = rows(squad[DEVELOP:DEVELOP + HELD])
    transfer = rows(other[:TRANSFER])

    # absent: a real question against a passage from a different row. The answer is genuinely not
    # there — this is a real absence, not one manufactured to look like the organ's blind spot.
    pool = squad[DEVELOP + HELD:DEVELOP + HELD + ABSENT * 3]
    absent: List[Row] = []
    for i in range(min(ABSENT, len(pool) // 2)):
        q, p = pool[2 * i], pool[2 * i + 1]
        if _norm(str(q["answer"])) and _norm(str(q["answer"])) in _norm(p["passage"]):
            continue  # the answer happens to be in the other passage: not an absence
        absent.append(Row(passage=p["passage"], question=q["question"],
                          answer="", family="absent", absent=True))
    return {"develop": develop, "held": held, "transfer": transfer, "absent": absent}


# --------------------------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------------------------- #
_ARTICLE = re.compile(r"\b(a|an|the)\b")


def _norm(text: str) -> str:
    low = str(text or "").lower()
    low = re.sub(r"[^\w\s]", " ", low)
    low = _ARTICLE.sub(" ", low)
    return " ".join(low.split())


def _f1(got: str, gold: str) -> float:
    g, k = _norm(got).split(), _norm(gold).split()
    if not g or not k:
        return float(bool(g) == bool(k))
    common = 0
    pool = list(k)
    for word in g:
        if word in pool:
            pool.remove(word)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(g), common / len(k)
    return 2 * precision * recall / (precision + recall)


@dataclass
class Score:
    """One method on one split."""

    name: str = ""
    split: str = ""
    n: int = 0
    correct: int = 0        # answered and right
    confabulated: int = 0   # answered and wrong (on an absent row, any answer at all)
    missed: int = 0         # UNKNOWN where the answer was there
    exact: int = 0          # subset of correct: exact match after normalisation

    @property
    def answered(self) -> int:
        return self.correct + self.confabulated

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.n, 3) if self.n else 0.0

    @property
    def confabulation(self) -> float:
        """Of everything it said, how much was wrong. The metric that stops guessing from winning."""
        return round(self.confabulated / self.answered, 3) if self.answered else 0.0

    @property
    def exact_rate(self) -> float:
        return round(self.exact / self.n, 3) if self.n else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "split": self.split, "n": self.n,
                "accuracy": self.accuracy, "exact": self.exact_rate,
                "confabulation": self.confabulation, "correct": self.correct,
                "confabulated": self.confabulated, "missed": self.missed}

    def render(self) -> str:
        return (f"  {self.name:<16}{self.split:<10}{self.n:>6}{self.accuracy:>10.3f}"
                f"{self.exact_rate:>9.3f}{self.confabulation:>15.3f}")


def grade(name: str, split: str, rows: Sequence[Row],
          method: Callable[[World, str], Answer], *, threshold: float = 0.5) -> Score:
    """Run one method over one split. Worlds are read once per row and shared by every method."""
    score = Score(name=name, split=split, n=len(rows))
    for row in rows:
        world = read_world(row.passage)
        got = method(world, row.question)
        if got.unknown:
            if row.absent:
                score.correct += 1
                score.exact += 1
            else:
                score.missed += 1
            continue
        if row.absent:
            score.confabulated += 1
            continue
        f1 = _f1(got.text, row.answer)
        if f1 >= threshold:
            score.correct += 1
            if _norm(got.text) == _norm(row.answer):
                score.exact += 1
        else:
            score.confabulated += 1
    return score


@dataclass
class Marks:
    scores: Tuple[Score, ...] = ()
    supplied_bits: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"supplied_bits": self.supplied_bits,
                "scores": [s.to_dict() for s in self.scores]}

    def beat_the_null(self, split: str) -> Optional[bool]:
        """Did the world model beat word-overlap on this split? ``None`` when it did not run.

        A comparison that could not be made is not a comparison the organ won — V.99, and the five
        levels before it.
        """
        mine = next((s for s in self.scores if s.name == "world model" and s.split == split), None)
        null = next((s for s in self.scores if s.name == "word overlap" and s.split == split), None)
        if mine is None or null is None or mine.n == 0 or null.n == 0:
            return None
        return mine.accuracy > null.accuracy

    def render(self) -> str:
        lines = [f"  {'method':<16}{'split':<10}{'n':>6}{'accuracy':>10}{'exact':>9}"
                 f"{'confabulation':>15}"]
        lines += [s.render() for s in self.scores]
        lines.append(f"  supplied grammar: {self.supplied_bits} bits, charged per V.99")
        return "\n".join(lines)


METHODS: Tuple[Tuple[str, Callable[[World, str], Answer]], ...] = (
    ("world model", ask),
    ("word overlap", overlap_span),
    ("always unknown", always_unknown),
)


def examine(which: Sequence[str] = ("held", "transfer", "absent"),
            *, path: str = CORPUS, limit: int = 0) -> Marks:
    """Sit every method on every named split. Nothing here decides what counts as a good result."""
    from nyxara.njp.askable import supplied_price

    # The precondition, before the corpus is even opened. An exam run against a package with a
    # hole in it produces a number about nothing, and V.100 spent an hour proving that.
    missing = intact()
    if missing:
        raise RuntimeError(
            "the laboratory is not intact, so nothing measured here would mean anything: "
            + "; ".join(missing))

    made = splits(path)
    scores: List[Score] = []
    for split in which:
        rows = made.get(split, [])
        if limit:
            rows = rows[:limit]
        for name, method in METHODS:
            scores.append(grade(name, split, rows, method))
    return Marks(scores=tuple(scores), supplied_bits=supplied_price())


def sit(**kwargs: Any) -> Marks:
    """Alias kept for the brain, which calls its organs by a verb."""
    return examine(**kwargs)
