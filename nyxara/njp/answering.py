"""NYXARA · njp/answering.py — she learns to *do* the dataset's tasks, not to hold them (🎓).

Everything before this stored. :mod:`nyxara.njp.shapes` recovers what a task's prompt looks like
and what its answers are drawn from — genuinely induced, nothing typed — and it is still a
description. A shape says *"this task asks a question about a passage and answers `yes` or `no`"*.
It cannot answer one.

This is the other thing. For each task, from examples of it:

    which readings of what was **poured into the slots** predict which answer?

That is a different claim from anything else here, and it is checkable in the plainest way there
is: hold examples back, ask her, count. The floor is not zero — it is **always saying whichever
answer was commonest**, which on a two-way task is around 0.5 and on some tasks is 0.9. A learner
that does not beat its own task's majority has learned nothing about that task, whatever its
accuracy looks like.

**Why the slots matter.** The prompt is mostly instruction, and the instruction is identical in
every row of a task — so every word in it is equally present in every answer's rows and carries no
signal at all. What varies is what was poured in. :mod:`nyxara.njp.shapes` already found where
that is, so the reading is taken there and the instruction is skipped rather than drowned in.

**What the readings are.** Whether the slot content holds each of the words that task uses most —
and nothing else about them. No sentiment lexicon, no polarity list, no notion of what any word
means. If *terrible* predicts *negative*, that is something :mod:`nyxara.njp.induce` found by
counting, in a task whose name it never read.

**What this cannot do, said before the numbers.** A task whose answer is free text — translate this
sentence, summarise this article, write a question — cannot be learned this way and is not
attempted. The machinery picks among answers it has seen; it does not compose new ones. Roughly a
third of FLAN's tasks name a small answer space and are in scope. The rest are counted and
reported as out of scope rather than quietly dropped.

Pure standard library.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from nyxara.njp.induce import SEEDS, Rule, cover
from nyxara.njp.shapes import Group, Shape, _lines, induce

__all__ = ["Example", "Learned", "TaskLearner", "read_examples", "probe"]

#: How many of a task's own commonest words are offered as readings. The induction narrows this
#: further on its own — :func:`~nyxara.njp.induce.attend` keeps the twelve that tell the answers
#: apart — so this is a budget rather than a choice about which words matter.
VOCABULARY = 120

#: An answer space wider than this is not a space, and a task with one is out of scope here.
MAX_ANSWERS = 8

#: The fewest examples a task needs before anything is claimed about it. Below this the majority
#: baseline is itself noise and beating it means nothing.
LEAST_EXAMPLES = 30

_WORD = re.compile(r"[^\W\d_][\w'’\-]*|\d+", re.UNICODE)


def _closed() -> Dict[str, str]:
    try:
        from nyxara.njp.semantics import _CLOSED  # noqa: WPS433
        return _CLOSED
    except Exception:  # noqa: BLE001
        return {}


@dataclass(frozen=True)
class Example:
    """One instance of a task: what was asked, and what the answer was."""

    prompt: str = ""
    answer: str = ""
    task: str = ""
    source: str = ""


def _content(text: str) -> List[str]:
    closed = _closed()
    return [w for w in (m.group(0).lower() for m in _WORD.finditer(str(text or "")))
            if w not in closed and len(w) > 1]


def probe(text: str, vocabulary: Sequence[str]) -> Dict[str, Any]:
    """Generic readings of what was poured into a task's slots.

    Presence of each word the task uses often, and four measurements of the text itself. Not one
    of them says what any word means: ``terrible`` is a reading here in exactly the way ``the`` is,
    and if it predicts an answer that is something the induction counted rather than something
    anybody knew.
    """
    words = _content(text)
    seen = set(words)
    out: Dict[str, Any] = {f"has:{w}": (w in seen) for w in vocabulary}
    out["length"] = "short" if len(words) <= 12 else "medium" if len(words) <= 40 else "long"
    out["has_number"] = any(ch.isdigit() for ch in str(text or ""))
    out["opens"] = words[0] if words else ""
    out["ends"] = words[-1] if words else ""
    return out


@dataclass
class Learned:
    """What was worked out about one task, and how well it did on examples it never saw."""

    task: str = ""
    answers: Tuple[str, ...] = ()
    rules: List[Rule] = field(default_factory=list)
    commonest: str = ""
    shape: Optional[Shape] = None
    #: The readings this was induced over. Kept because it has to be: a rule says ``has:terrible``
    #: and :func:`probe` only emits that key for a word in the vocabulary it is handed, so asking
    #: the same rules with a vocabulary rebuilt from one prompt reads every term as missing and no
    #: rule can ever fire. What comes back then is the majority answer, every time, silently.
    vocabulary: Tuple[str, ...] = ()
    learned_from: int = 0
    asked: int = 0
    right: int = 0
    answered: int = 0
    majority_right: int = 0

    @property
    def accuracy(self) -> float:
        return round(self.right / self.asked, 4) if self.asked else 0.0

    @property
    def majority(self) -> float:
        """Always saying the commonest answer. The floor this has to clear to mean anything."""
        return round(self.majority_right / self.asked, 4) if self.asked else 0.0

    @property
    def coverage(self) -> float:
        return round(self.answered / self.asked, 4) if self.asked else 0.0

    @property
    def learned_something(self) -> bool:
        """Beat its own task's majority. Not "scored well" — those are different claims."""
        return self.asked > 0 and self.accuracy > self.majority

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task, "answers": list(self.answers),
                "accuracy": self.accuracy, "majority": self.majority,
                "coverage": self.coverage, "rules": [r.render() for r in self.rules],
                "learned_from": self.learned_from, "asked": self.asked}

    def render(self) -> str:
        mark = "+" if self.learned_something else " "
        return (f" {mark} {self.accuracy:.3f} vs {self.majority:.3f}  "
                f"{len(self.answers)}-way  {self.task[:46]}")


class TaskLearner:
    """Learns one task from examples of it, and can then be asked."""

    def __init__(self, purity: float = 0.75, min_support: int = 4, max_rules: int = 6,
                 max_terms: int = 2, seed: int = 57) -> None:
        self.purity = purity
        self.min_support = min_support
        self.max_rules = max_rules
        self.max_terms = max_terms
        self.seed = seed

    # -- the part that reads ---------------------------------------------------------------- #
    def fields(self, shape: Optional[Shape], prompt: str) -> str:
        """The parts of the prompt that vary, or the whole thing when no shape is known.

        Falling back to the whole prompt is deliberate and is measured against: a task whose shape
        was not recovered can still be learned from, just with the instruction's words in the way.
        :func:`nyxara.njp.answeringschool.examine` runs both and prints what the shape was worth.
        """
        if shape is None:
            return str(prompt or "")
        got = shape.read(str(prompt or ""))
        if not got:
            return str(prompt or "")
        return " \n ".join(got)

    def vocabulary(self, texts: Sequence[str]) -> List[str]:
        counts = Counter(w for text in texts for w in set(_content(text)))
        return [w for w, _n in counts.most_common(VOCABULARY)]

    # -- the part that learns --------------------------------------------------------------- #
    def learn(self, examples: Sequence[Example],
              shape: Optional[Shape] = None) -> Optional[Learned]:
        """Split the examples, induce on one half, and answer the other."""
        rows = [e for e in examples if e.answer.strip()]
        if len(rows) < LEAST_EXAMPLES:
            return None
        answers = sorted({e.answer.strip() for e in rows})
        if not 1 < len(answers) <= MAX_ANSWERS:
            return None

        order = list(range(len(rows)))
        random.Random(self.seed).shuffle(order)
        cut = int(len(order) * 0.7)
        learn = [rows[i] for i in order[:cut]]
        held = [rows[i] for i in order[cut:]]
        if not held:
            return None

        texts = [self.fields(shape, e.prompt) for e in learn]
        words = self.vocabulary(texts)
        readings = [probe(t, words) for t in texts]
        labels = [e.answer.strip() for e in learn]

        out = Learned(task=rows[0].task, answers=tuple(answers), shape=shape,
                      learned_from=len(learn), vocabulary=tuple(words))
        out.commonest = Counter(labels).most_common(1)[0][0]
        for answer in answers:
            positives = [r for r, a in zip(readings, labels) if a == answer]
            negatives = [r for r, a in zip(readings, labels) if a != answer]
            if len(positives) < self.min_support:
                continue
            # `seeds` is what makes a task decided by *any one of several words* learnable at
            # all: with a single seed the cover settles for one impure rule that swallows the
            # positives and the clean rules underneath are never looked for. See
            # :data:`nyxara.njp.induce.SEEDS` for the measurement and for why it is asked for here
            # rather than turned on for every organ.
            rules, _near = cover(positives, negatives, label=answer,
                                 min_support=self.min_support, min_share=0.05,
                                 max_rules=self.max_rules, max_terms=self.max_terms,
                                 purity=self.purity, seeds=SEEDS)
            out.rules.extend(rules)

        for example in held:
            out.asked += 1
            said = self.answer(out, example.prompt, words)
            if said:
                out.answered += 1
            out.right += int(said == example.answer.strip())
            out.majority_right += int(out.commonest == example.answer.strip())
        return out

    def answer(self, learned: Learned, prompt: str,
               vocabulary: Optional[Sequence[str]] = None) -> str:
        """What she says, or the commonest answer when no rule fires.

        Falling back to the majority is what makes ``accuracy`` comparable to ``majority``: both
        columns then answer every question, and the difference between them is what the rules are
        worth rather than a difference in how often each spoke.
        """
        words = learned.vocabulary if vocabulary is None else vocabulary
        marks = probe(self.fields(learned.shape, prompt), words)
        best: Optional[Rule] = None
        for rule in learned.rules:
            if not rule.holds(marks):
                continue
            if best is None or (rule.purity, rule.support) > (best.purity, best.support):
                best = rule
        return best.label if best is not None else learned.commonest


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def read_examples(path: Path) -> Iterator[Tuple[str, List[Example]]]:
    """The collector's by-task output, as one task's examples at a time."""
    if not Path(path).exists():
        return
    for line in _lines(Path(path)):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        task = str(row.get("task") or "")
        out = [Example(prompt=str(r.get("inputs") or ""),
                       answer=str(r.get("targets") or ""),
                       task=task, source=str(r.get("source") or ""))
               for r in (row.get("rows") or [])]
        if out:
            yield task, out


def shape_for(group: Optional[Group]) -> Optional[Shape]:
    """The induced shape of a task, when its rows were collected for aligning."""
    return induce(group) if group is not None else None
