"""NYXARA · njp/findingschool.py — did she find the span, or pick one (📏).

Every row carries its own answer key, and the key is checkable: the corpus builder kept only rows
whose answer appears once, character for character, inside the passage. So the exam is not a
judgement call. Two numbers per reader:

* **exact** — the span she returned, normalised, is the span the annotator marked.
* **overlap** — token F1 against the marked span, which credits *"Felipa Moñiz"* for
  *"amb Felipa Moñiz"* rather than scoring it zero. Both are printed, always: a reader that never
  quite gets the boundary right and one that points at the wrong sentence are different failures.

And the share she **answered at all**, beside them. Abstention is allowed here and is not the same
as being wrong.

Three things to beat, and the first two are the ones that matter:

* **cold** — a reader constructed and never taught. It returns nothing. The floor.
* **first span** — return the passage's first candidate. The stupidest thing that is not nothing.
* **most overlap** — return the longest candidate from the sentence sharing the most content
  words with the question. This is the heuristic a person writes in ten minutes without any
  learning at all, and a learned reader that does not beat it has not earned its induction.

Then the ablation that this whole module was arranged to permit. ``no_shape`` is the same reader
with :mod:`nyxara.njp.asked` unplugged — the ``shape_fits`` measurement removed and nothing else
changed. The distance between it and ``taught`` is what V.54 was worth downstream, and if it is
zero then V.54 bought nothing here and that is the finding.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.asked import Asked
from nyxara.njp.askedschool import split as split_questions
from nyxara.njp.finding import (
    Finder, Reading, Setting, candidates, gold_key, read_passages,
)

__all__ = ["Result", "SEED", "TRAIN", "LEARN_FROM", "HELD_OUT",
           "split", "taught_finder", "examine", "decompose", "sentence_baselines",
           "gold_sentence", "run"]

SEED = 55
TRAIN = 0.7
LEARN_FROM = 1_500
HELD_OUT = 600


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split()).strip(" .,;:'\"")


def _toks(text: str) -> List[str]:
    return _norm(text).split()


@dataclass
class Result:
    name: str = ""
    exact_right: int = 0
    overlap: float = 0.0
    answered: int = 0
    asked: int = 0
    rules: int = 0
    wrong: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def exact(self) -> float:
        return round(self.exact_right / self.asked, 4) if self.asked else 0.0

    @property
    def f1(self) -> float:
        return round(self.overlap / self.asked, 4) if self.asked else 0.0

    @property
    def coverage(self) -> float:
        return round(self.answered / self.asked, 4) if self.asked else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "exact": self.exact, "overlap": self.f1,
                "answered": self.coverage, "asked": self.asked, "rules": self.rules}

    def render(self) -> str:
        return (f"{self.name:<14} exact {self.exact:.3f}   overlap {self.f1:.3f}   "
                f"answered {self.coverage:.3f}   ({self.rules} rules)")


def _score(said: str, wanted: str) -> Tuple[int, float]:
    """Exact match, and token F1. Both against the same normalisation."""
    if not said:
        return 0, 0.0
    a, b = _toks(said), _toks(wanted)
    if not a or not b:
        return 0, 0.0
    exact = int(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    if not shared:
        return exact, 0.0
    precision, recall = shared / len(a), shared / len(b)
    return exact, 2 * precision * recall / (precision + recall)


def split(readings: Optional[Sequence[Reading]] = None
          ) -> Tuple[List[Reading], List[Reading]]:
    """One deterministic cut over a deterministic shuffle, and only rows that keep their promise.

    A row whose answer is not actually a span of its passage cannot be found and cannot be scored;
    it is dropped here rather than counted as a failure of the reader.
    """
    rows = [r for r in (readings if readings is not None else read_passages())
            if r.holds_its_answer()]
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * TRAIN)
    return rows[:cut][:LEARN_FROM], rows[cut:][:HELD_OUT]


def taught_finder(learn: Sequence[Reading], *, use_shape: bool = True,
                  asked: Optional[Asked] = None, **kwargs: Any) -> Finder:
    """A finder with the passages in it, and the answer-shape organ plugged in or not."""
    engine = Finder(use_shape=use_shape, **kwargs)
    if use_shape:
        engine.asked = asked if asked is not None else _taught_asked()
    engine.learn_from(learn)
    return engine


def _taught_asked() -> Optional[Asked]:
    """The V.54 organ, taught on its own corpus. Never on anything this module holds out."""
    try:
        questions, _held = split_questions()
        engine = Asked()
        engine.learn_from(questions)
        return engine
    except Exception:  # noqa: BLE001 — without it the shape feature is simply absent
        return None


# --------------------------------------------------------------------------------------------- #
#  what there is to beat
# --------------------------------------------------------------------------------------------- #
def _first_span(reading: Reading) -> str:
    spans = candidates(reading.passage)
    return spans[0].text if spans else ""


def _most_overlap(reading: Reading) -> str:
    """The longest candidate in the sentence that shares most content with the question.

    Written to be a fair opponent rather than a straw one: it uses the same candidate generator,
    the same notion of content word, and the one signal anybody would reach for first.
    """
    from nyxara.njp.finding import _SENTENCE, _content
    spans = candidates(reading.passage)
    if not spans:
        return ""
    sentences = _SENTENCE.split(reading.passage)
    asked = _content(reading.question)
    weight = [len(_content(s) & asked) for s in sentences]
    if not weight:
        return spans[0].text
    best = max(range(len(weight)), key=lambda i: weight[i])
    here = [s for s in spans if s.sentence == best] or spans
    return max(here, key=lambda s: len(s.text)).text


def _mark(name: str, held: Sequence[Reading], answer: Any, rules: int = 0) -> Result:
    out = Result(name=name, rules=rules)
    for reading in held:
        out.asked += 1
        said = answer(reading)
        if said:
            out.answered += 1
        exact, overlap = _score(said, reading.answer)
        out.exact_right += exact
        out.overlap += overlap
        if not exact and len(out.wrong) < 12:
            out.wrong.append((reading.question, reading.answer, said))
    return out


def examine(readings: Optional[Sequence[Reading]] = None) -> Dict[str, Result]:
    learn, held = split(readings)
    out: Dict[str, Result] = {}
    if not learn or not held:
        return out

    out["cold"] = _mark("cold", held, lambda r: Finder().find(r.passage, r.question)[0])
    out["first_span"] = _mark("first span", held, _first_span)
    out["most_overlap"] = _mark("most overlap", held, _most_overlap)

    asked = _taught_asked()
    taught = taught_finder(learn, use_shape=True, asked=asked)
    out["taught"] = _mark("taught", held,
                          lambda r: taught.find(r.passage, r.question)[0], len(taught.rules))

    blind = taught_finder(learn, use_shape=False)
    out["no_shape"] = _mark("no answer-shape", held,
                            lambda r: blind.find(r.passage, r.question)[0], len(blind.rules))
    return out


def gold_sentence(reading: Reading) -> int:
    """Which sentence of the passage actually holds the answer."""
    fixed = Setting.of(reading)
    at = reading.passage.lower().find(gold_key(reading.answer))
    return max((i for i, (_t, start) in enumerate(fixed.spans) if start <= at), default=0)


def decompose(engine: Finder, held: Sequence[Reading]) -> Dict[str, float]:
    """Where the accuracy goes: the wrong sentence, or the right one and the wrong words.

    One number is not enough to act on. A reader that is looking in the wrong place and one that
    is looking in the right place and picking the wrong phrase out of it need different repairs,
    and the overall figure cannot tell them apart. The first row is the ceiling on both: whether
    the marked answer is a candidate at all, since a span the generator never proposes cannot be
    chosen however good the ranking is.
    """
    n = max(1, len(held))
    reachable = right_sentence = answered = right_both = 0
    for reading in held:
        gold = gold_sentence(reading)
        fixed = Setting.of(reading)
        said, _start = fixed.spans[gold] if gold < len(fixed.spans) else ("", 0)
        wanted = gold_key(reading.answer)
        reachable += int(any(s.key == wanted for s in candidates(said)))
        if engine.sentence(reading.passage, reading.question) != gold:
            continue
        right_sentence += 1
        got, _why = engine.find(reading.passage, reading.question)
        answered += int(bool(got))
        right_both += int(bool(got) and _norm(got) == _norm(reading.answer))
    return {
        "gold_is_a_candidate": round(reachable / n, 4),
        "sentence_right": round(right_sentence / n, 4),
        "answered_when_sentence_right": round(answered / max(1, right_sentence), 4),
        "exact_when_sentence_right": round(right_both / max(1, right_sentence), 4),
        "exact_overall": round(right_both / n, 4),
    }


def sentence_baselines(held: Sequence[Reading]) -> Dict[str, float]:
    """What the first stage has to beat, and the number that turned out to beat it."""
    from nyxara.njp.finding import _content
    n = max(1, len(held))
    overlap = first = 0
    for reading in held:
        fixed = Setting.of(reading)
        asked = _content(reading.question)
        weight = [len(fixed.carried[i] & asked) for i in range(len(fixed.spans))]
        pick = max(range(len(weight)), key=lambda i: weight[i]) if weight else 0
        gold = gold_sentence(reading)
        overlap += int(pick == gold)
        first += int(gold == 0)
    return {"argmax_overlap": round(overlap / n, 4), "always_first": round(first / n, 4)}


def run() -> Dict[str, Any]:  # pragma: no cover — a report, not a test
    rows = read_passages()
    if not rows:
        print("no corpus; run scripts/stream_flan.py and scripts/merge_flan_shards.py")
        return {}
    kept = [r for r in rows if r.holds_its_answer()]
    print(f"{len(rows):,} passages, {len(kept):,} of which hold their answer exactly once")
    learn, held = split(rows)
    print(f"{len(learn):,} learned from, {len(held):,} held out\n")
    got = examine(rows)
    for key in ("cold", "first_span", "most_overlap", "no_shape", "taught"):
        if key in got:
            print("  " + got[key].render())
    print("\nwhat she worked out:")
    asked = _taught_asked()
    engine = taught_finder(learn, use_shape=True, asked=asked)
    for rule, purity in zip(engine.learned()["rules"], engine.learned()["purities"]):
        print(f"    {purity:.3f}  {rule}")
    print("\nwhere it went wrong:")
    for question, wanted, said in got["taught"].wrong[:8]:
        print(f"    {question[:62]!r}\n        wanted {wanted[:44]!r}  got {said[:44]!r}")
    return {k: v.to_dict() for k, v in got.items()}


if __name__ == "__main__":  # pragma: no cover
    run()
