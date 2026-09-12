"""NYXARA · njp/askedschool.py — is the expected kind the kind that came (📏).

129,954 real questions with real answers, shuffled once and cut once. Two tables, and they set two
different numbers because predicting and refusing are not the same bet.

:func:`sweep` sets ``purity``, the bar for **predicting**, and reads:

* **accuracy when it fires**, over held-out questions it has a rule for;
* **how often it fires**, printed beside it — a predictor that speaks for a fifth of the questions
  at 0.89 and one that speaks for all of them at 0.60 are different things and one number cannot
  tell them apart;
* **the base rate**, always guessing the commonest kind, which here is ``span`` at 0.607. Anything
  that does not beat it has done nothing.

:func:`veto_bar` sets ``veto_purity``, the bar for **refusing**, and it is the table that decides
whether :meth:`~nyxara.njp.asked.Asked.contradicts` may be wired into anything. Every held-out
question is handed *its own correct answer* and the veto is asked whether that answer is the wrong
kind of thing. Every yes is wrong and would have suppressed a right answer. A veto trades one
failure for another — a wrong answer becomes an abstention, and so does a wrongly-refused right
one — and only the first trade is worth making.

**A zero in that table is not automatically good news.** Measured against the corpus before its
multiple-choice labels and category-name answers were removed, *no rule reached a bar of 0.90* —
the purest was 0.855 — so the veto never fired and the cost column came back a flawless 0.0000.
For about an hour this module reported that as its result. A mechanism that does nothing is never
wrong. What caught it was printing how often the veto fired beside how often it erred, which is
why :func:`veto_bar` returns both and why a test refuses a row that has one without the other.

The cost that remains at the shipped bar is small and it is not random. Nearly all of it is one
construction: an **alternative question**, which opens exactly like a polar one and is not one —
*"Does Ridge Pond have more nitrogen or oxygen?"*, *"Does Kasetsart University or Bilkent
University focus upon agriculture?"*. ``opens is does`` predicts ``polar`` at 0.84 and these are
the sixteen percent. Naming that is better than hiding it in a rate.

And the ablation. ``no_rules`` is the same object with induction switched off: it fires on nothing
and vetoes nothing, which is the honest floor for something shown questions and told to conclude.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.asked import Asked, KINDS, Question, read_questions

__all__ = ["Result", "SEED", "TRAIN", "LEARN_FROM", "HELD_OUT",
           "shuffled", "split", "examine", "sweep", "veto_bar", "false_vetoes", "run"]

#: One shuffle, forever. A held-out set that moves between runs is not held out.
SEED = 2026

#: The share learned from, and the bound on how much of it is actually read. The induction is
#: superlinear and 20,000 questions is 20 seconds; the whole 107,000 is not twenty times that.
#: The slice comes off the shuffle, so it is the same mixture as the whole — what changes is how
#: much is read, not what.
TRAIN = 0.7
LEARN_FROM = 20_000
HELD_OUT = 5_000


@dataclass
class Result:
    name: str = ""
    right: int = 0
    fired: int = 0
    asked: int = 0
    rules: int = 0
    vetoed_correct: int = 0
    by_kind: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def when_fires(self) -> float:
        return round(self.right / self.fired, 4) if self.fired else 0.0

    @property
    def coverage(self) -> float:
        return round(self.fired / self.asked, 4) if self.asked else 0.0

    @property
    def false_veto(self) -> float:
        """Share of held-out questions whose own correct answer the veto would have rejected."""
        return round(self.vetoed_correct / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "when_fires": self.when_fires, "coverage": self.coverage,
                "false_veto": self.false_veto, "rules": self.rules, "asked": self.asked,
                "by_kind": {k: {"right": v[0], "asked": v[1]} for k, v in self.by_kind.items()}}

    def render(self) -> str:
        return (f"{self.name:<14} {self.when_fires:.3f} when it fires, and it fires on "
                f"{self.coverage:.3f}   false veto {self.false_veto:.4f}   "
                f"({self.rules} rules)")


def shuffled(questions: Optional[Sequence[Question]] = None) -> List[Question]:
    """The corpus in one fixed pseudo-random order, and with the unreadable rows dropped.

    Never a prefix of the file. The fold lands on disk in the order the nine shards were read, so
    the first N rows are one part of one submix — which is exactly what made the first entailment
    learning curve measure composition and report it as size.
    """
    # Shuffled *then* filtered, and the order of those two steps is not free: the same seed over
    # a differently-ordered list is a different permutation, so swapping them silently rebuilds
    # the training slice and every number measured against it. Three rows in 129,954 have no
    # readable answer shape; they are dropped after the shuffle so the cut stays the cut that the
    # tables in `njp.asked` were measured under.
    rows = list(questions if questions is not None else read_questions())
    random.Random(SEED).shuffle(rows)
    return [q for q in rows if q.kind]


def split(questions: Optional[Sequence[Question]] = None
          ) -> Tuple[List[Question], List[Question]]:
    rows = shuffled(questions)
    cut = int(len(rows) * TRAIN)
    return rows[:cut][:LEARN_FROM], rows[cut:][:HELD_OUT]


def _mark(engine: Asked, held: Sequence[Question], name: str) -> Result:
    out = Result(name=name, rules=len(engine.rules))
    for question in held:
        out.asked += 1
        kind, _why = engine.expects(question.question)
        if kind:
            out.fired += 1
            hit = int(kind == question.kind)
            out.right += hit
            right, asked = out.by_kind.get(kind, (0, 0))
            out.by_kind[kind] = (right + hit, asked + 1)
        # The question handed its own correct answer. A veto here is a veto of the truth.
        rejected, _reason = engine.contradicts(question.question, question.answer)
        out.vetoed_correct += int(rejected)
    return out


def examine(questions: Optional[Sequence[Question]] = None,
            **kwargs: Any) -> Dict[str, Result]:
    learn, held = split(questions)
    out: Dict[str, Result] = {}
    if not learn or not held:
        return out

    taught = Asked(**kwargs)
    taught.learn_from(learn)
    out["taught"] = _mark(taught, held, "taught")

    blind = Asked(learning=False, **kwargs)
    blind.learn_from(learn)
    out["no_rules"] = _mark(blind, held, "no rules")

    commonest = Counter(q.kind for q in learn).most_common(1)[0][0]
    base = Result(name=f"base rate ({commonest})", asked=len(held), fired=len(held))
    base.right = sum(1 for q in held if q.kind == commonest)
    out["base_rate"] = base
    return out


def sweep(values: Sequence[float] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
          questions: Optional[Sequence[Question]] = None) -> List[Tuple[float, Result]]:
    """The examination at each purity. What sets the threshold is this table, not a preference."""
    learn, held = split(questions)
    out: List[Tuple[float, Result]] = []
    for value in values:
        engine = Asked(purity=value)
        engine.learn_from(learn)
        out.append((value, _mark(engine, held, f"purity {value}")))
    return out


def veto_bar(bars: Sequence[float] = (0.70, 0.75, 0.80, 0.85, 0.90),
             questions: Optional[Sequence[Question]] = None) -> List[Tuple[float, float, float]]:
    """How far the veto reaches at each bar, and what it costs. The table that decides shipping.

    Returns ``(bar, fires_on, rejects_a_correct_answer)`` per row. One induction, several bars,
    because the bar is a property of *using* the rules and not of finding them.

    Read the two columns together, always. A bar high enough that no rule clears it makes the veto
    inert, and an inert veto has a flawless cost column — which is what happened here, against the
    corpus before its multiple-choice labels were removed: nothing reached 0.90, the purest rule
    was 0.855, and this function would have reported ``(0.90, 0.0, 0.0)`` as a result. It reports
    reach first for that reason.
    """
    learn, held = split(questions)
    engine = Asked()
    engine.learn_from(learn)
    out: List[Tuple[float, float, float]] = []
    for bar in bars:
        engine.veto_purity = bar
        fires = rejects = 0
        for question in held:
            label, _why, purity = engine._best(question.question)
            if label and purity >= bar:
                fires += 1
            rejects += int(engine.contradicts(question.question, question.answer)[0])
        out.append((bar, round(fires / max(1, len(held)), 4),
                    round(rejects / max(1, len(held)), 4)))
    return out


def false_vetoes(engine: Optional[Asked] = None,
                 questions: Optional[Sequence[Question]] = None,
                 limit: int = 12) -> List[Tuple[str, str, str]]:
    """The held-out questions whose own correct answer the veto rejects. The damage, itemised."""
    learn, held = split(questions)
    if engine is None:
        engine = Asked()
        engine.learn_from(learn)
    out: List[Tuple[str, str, str]] = []
    for question in held:
        rejected, reason = engine.contradicts(question.question, question.answer)
        if rejected:
            out.append((question.question, question.answer, reason))
        if len(out) >= limit:
            break
    return out


def run() -> Dict[str, Any]:  # pragma: no cover — a report, not a test
    rows = read_questions()
    if not rows:
        print("no corpus; run scripts/stream_flan.py and scripts/merge_flan_shards.py")
        return {}
    print(f"{len(rows):,} questions with answers")
    print("kinds: " + "  ".join(f"{k} {v:,}" for k, v in
                                Counter(q.kind for q in rows).most_common()))
    print()
    print("purity   when it fires   fires on   false veto   rules")
    for value, result in sweep():
        print(f"  {value:.2f}       {result.when_fires:.3f}       {result.coverage:.3f}"
              f"       {result.false_veto:.4f}      {result.rules}")
    print()
    print("veto bar   fires on   rejects a correct answer")
    for bar, fires, rejects in veto_bar():
        print(f"  {bar:.2f}       {fires:.3f}          {rejects:.4f}")
    print()
    got = examine()
    for key in ("base_rate", "no_rules", "taught"):
        print("  " + got[key].render())
    print("\nby kind, taught:")
    for kind in KINDS:
        if kind in got["taught"].by_kind:
            right, asked = got["taught"].by_kind[kind]
            print(f"    {kind:<8} {right}/{asked}  {right / max(1, asked):.3f}")
    print("\nwhat she worked out:")
    learn, _held = split()
    engine = Asked()
    engine.learn_from(learn)
    for kind, rules in engine.learned()["by_kind"].items():
        for rule in rules:
            print(f"    {kind:<8} {rule}")
    bad = false_vetoes(engine)
    print(f"\ncorrect answers the veto would have rejected ({len(bad)} shown):")
    for question, answer, reason in bad[:8]:
        print(f"    {question[:60]!r} -> {answer[:30]!r}")
        print(f"        {reason}")
    return {k: v.to_dict() for k, v in got.items()}


if __name__ == "__main__":  # pragma: no cover
    run()
