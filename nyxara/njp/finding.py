"""NYXARA · njp/finding.py — the answer is in the passage; find it (🔎).

V.54 ended on a number that has not moved all session: asked a thousand real questions, her fact
store answers seventy and gets **none** right. That is not a defect in the reader. It is that she
does not know when the Battle of the Coral Sea was fought, and no amount of better reading of an
empty store will produce a date.

So this asks the question she can actually be held to: **here is the passage — now find it.**

    "El 1479, o el 1480, va contraure matrimoni amb Felipa Moñiz, probablement a Lisboa, filla
     del colonitzador de les illes Madeira..."
        Com es deia la seva dona?   ->   amb Felipa Moñiz

That is a different capability from recall and it is the honest one for a system with no world
model. It is also **verifiable end to end**, which almost nothing else here is: the corpus keeps
only rows whose answer appears, character for character and exactly once, inside the passage it
came with. A row that passes cannot be a classification label or an option index, because neither
is a span of the text.

**How a span is chosen.** Every span of the passage up to :data:`MAX_TOKENS` long that begins and
ends on an open-class word is a candidate — a few thousand of them for a paragraph. Each is
measured, and *not one measurement names an answer*: how much of the question's content the
candidate's own sentence carries, how far the candidate sits from the nearest question word,
whether it repeats the question's words, how long it is, whether it is capitalised mid-sentence,
whether it holds a number. Which combination of those marks the answer is induced by
:mod:`nyxara.njp.induce`, from passages she has been shown and against passages she has not.

**And one measurement comes from another organ.** :mod:`nyxara.njp.asked` learned what *kind* of
thing a question wants — a count, a year, something short — and that expectation is handed in here
as a feature, so a candidate can be measured on whether it is the right shape for what was asked.
This is the first place in the package where one induced organ feeds another, and the point of
doing it this way rather than hard-wiring it is that :func:`nyxara.njp.findingschool.examine` can
take the feature away again and print what it was worth. If it is worth nothing, V.54 bought
nothing downstream and that is a result too.

Nothing here writes to the fact store. Finding a span in a passage somebody supplied is not
learning a fact about the world, and a module that filed one would be asserting that the passage
is true.

Pure standard library.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nyxara.njp.asked import Asked, satisfies, shape_of
from nyxara.njp.induce import Rule, cover

__all__ = ["Reading", "Span", "Setting", "Finder", "CORPUS", "read_passages",
           "candidates", "probe"]

CORPUS = Path(__file__).with_name("data") / "flan_reading.jsonl.gz"

#: The longest span that may be an answer, in tokens. Not a taste: the corpus builder already
#: refused any answer over 120 characters, and of what survives the ninety-ninth percentile is
#: under this. A longer candidate is a sentence, and a sentence is not a span.
MAX_TOKENS = 10

#: How many non-answer candidates each training passage contributes. Every candidate is scored at
#: examination time; only the induction is sampled, because a paragraph offers a few thousand
#: negatives and a hundred passages would otherwise be half a million readings.
NEGATIVES = 24

_WORD = re.compile(r"[^\W\d_][\w'’\-]*|\d+(?:[.,]\d+)*", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _closed() -> Dict[str, str]:
    try:
        from nyxara.njp.semantics import _CLOSED  # noqa: WPS433
        return _CLOSED
    except Exception:  # noqa: BLE001 — with no closed class every word counts as content
        return {}


def _content(text: str) -> Set[str]:
    closed = _closed()
    return {w for w in (m.group(0).lower() for m in _WORD.finditer(str(text or "")))
            if w not in closed and len(w) > 1}


@dataclass(frozen=True)
class Reading:
    """A passage, a question about it, and the span of the passage that answers it."""

    passage: str = ""
    question: str = ""
    answer: str = ""
    task: str = ""
    source: str = ""
    licence: str = ""

    def holds_its_answer(self) -> bool:
        """The guarantee the corpus is built on, checkable on any row at any time."""
        pattern = r"(?<!\w)" + re.escape(self.answer.lower()) + r"(?!\w)"
        return len(re.findall(pattern, self.passage.lower())) == 1


@dataclass(frozen=True)
class Span:
    """One candidate answer: its text, where it sits, and which sentence it belongs to."""

    text: str = ""
    start: int = 0
    end: int = 0
    sentence: int = 0

    @property
    def key(self) -> str:
        return " ".join(self.text.lower().split())


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(str(text or ""))]


def candidates(passage: str) -> List[Span]:
    """Every span that could be an answer: open-class at both ends, inside one sentence.

    The two constraints are what keep this to thousands rather than tens of thousands, and both
    are statements about phrases rather than about answers. A span opening on ``of`` or closing on
    ``the`` is a fragment; a span running across a full stop is two things.
    """
    closed = _closed()
    raw = str(passage or "")
    bounds: List[int] = []
    at = 0
    for piece in _SENTENCE.split(raw):
        at = raw.find(piece, at)
        bounds.append(at)
        at += len(piece)
    toks = _tokens(raw)
    out: List[Span] = []
    for i, (word, start, _end) in enumerate(toks):
        if word.lower() in closed:
            continue
        for j in range(i, min(len(toks), i + MAX_TOKENS)):
            last, _s, end = toks[j]
            if last.lower() in closed:
                continue
            text = raw[start:end]
            if "\n" in text:
                break
            sentence = sum(1 for b in bounds if b <= start) - 1
            if any(b > start and b < end for b in bounds):
                break                       # it has run past the end of its own sentence
            out.append(Span(text=text, start=start, end=end, sentence=max(0, sentence)))
    return out


def _bucket(n: int, edges: Sequence[int], names: Sequence[str]) -> str:
    for edge, name in zip(edges, names):
        if n <= edge:
            return name
    return names[-1]


@dataclass(frozen=True)
class Setting:
    """Everything about a passage-and-question that does not change from candidate to candidate.

    Computed once per reading rather than once per span, and that is not a micro-optimisation. The
    first version worked out where the question's words fall by scanning the whole passage for each
    of them *inside* :func:`probe`, which runs a thousand times per passage — so a six-hundred-row
    examination did some tens of millions of regular-expression scans and took long enough that it
    looked like a hang rather than a cost.
    """

    asked: frozenset = frozenset()
    sentences: Tuple[str, ...] = ()
    carried: Tuple[frozenset, ...] = ()
    marks: Tuple[int, ...] = ()
    length: int = 0

    @classmethod
    def of(cls, reading: "Reading") -> "Setting":
        asked = _content(reading.question)
        sentences = tuple(_SENTENCE.split(reading.passage))
        low = reading.passage.lower()
        marks: List[int] = []
        for word in asked:
            marks.extend(m.start() for m in
                         re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", low))
        return cls(asked=frozenset(asked), sentences=sentences,
                   carried=tuple(frozenset(_content(s)) for s in sentences),
                   marks=tuple(sorted(marks)), length=len(reading.passage))

    def nearest(self, at: int) -> int:
        if not self.marks:
            return self.length
        import bisect
        i = bisect.bisect_left(self.marks, at)
        best = self.length
        if i < len(self.marks):
            best = min(best, abs(self.marks[i] - at))
        if i:
            best = min(best, abs(self.marks[i - 1] - at))
        return best


def probe(reading: Reading, span: Span, wanted: str = "",
          *, use_shape: bool = True, setting: Optional[Setting] = None) -> Dict[str, Any]:
    """Generic measurements of one candidate. Not one of them names an answer.

    ``wanted`` is what :mod:`nyxara.njp.asked` expects the question to be answered by, and it
    enters through ``shape_fits`` and nowhere else, so ``use_shape=False`` removes that organ's
    contribution cleanly and the ablation means something.
    """
    fixed = setting if setting is not None else Setting.of(reading)
    asked_words = fixed.asked
    span_words = _content(span.text)
    here = fixed.carried[span.sentence] if span.sentence < len(fixed.carried) else frozenset()
    nearest = fixed.nearest(span.start)

    reading_shape = shape_of(span.text)
    out: Dict[str, Any] = {
        "sentence_carries": _bucket(len(here & asked_words), (0, 1, 3), ("none", "one", "few",
                                                                        "many")),
        "repeats_question": _bucket(len(span_words & asked_words), (0, 1), ("none", "one",
                                                                           "several")),
        "distance": _bucket(nearest, (20, 80, 200), ("touching", "near", "far", "elsewhere")),
        "length": _bucket(len(_tokens(span.text)), (1, 3, 6), ("one", "short", "medium", "long")),
        "capitalised": bool(span.text[:1].isupper()),
        "has_number": any(ch.isdigit() for ch in span.text),
        "shape": reading_shape,
        "opens_passage": span.start < 40,
    }
    if use_shape:
        # The only place the other organ enters. `unknown` when it declined to say, which it does
        # for most questions and which is a different thing from saying the shape does not fit.
        out["shape_fits"] = ("unknown" if not wanted
                             else "yes" if satisfies(reading_shape, wanted) else "no")
    return out


def read_passages(path: Optional[Path] = None, limit: int = 0) -> List[Reading]:
    source = Path(path) if path is not None else CORPUS
    out: List[Reading] = []
    if not source.exists():
        return out
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out.append(Reading(passage=str(row.get("passage") or ""),
                               question=str(row.get("question") or ""),
                               answer=str(row.get("answer") or ""),
                               task=str(row.get("task") or ""),
                               source=str(row.get("source") or ""),
                               licence=str(row.get("licence") or "")))
            if limit and len(out) >= limit:
                break
    return out


@dataclass
class Finder:
    """Picks the span of a passage that answers a question, from rules it was shown.

    Constructed and never taught it returns nothing at all, which is the honest floor: a thing
    that has been shown no passages has no idea which part of one is an answer.
    """

    purity: float = 0.80
    min_support: int = 12
    min_share: float = 0.02
    max_rules: int = 6
    max_terms: int = 3
    learning: bool = True
    use_shape: bool = True
    seed: int = 7
    rules: List[Rule] = field(default_factory=list)
    near_misses: List[Rule] = field(default_factory=list)
    shown: int = 0
    #: The answer-shape organ, taught separately and consulted as one feature. ``None`` means the
    #: feature is simply absent, which is what the ablation runs on.
    asked: Optional[Asked] = None

    def wants(self, question: str) -> str:
        if self.asked is None or not self.use_shape:
            return ""
        try:
            kind, _why = self.asked.expects(question)
            return kind
        except Exception:  # noqa: BLE001
            return ""

    def learn_from(self, readings: Sequence[Reading]) -> List[Rule]:
        self.rules, self.near_misses = [], []
        self.shown = len(readings)
        if not self.learning or not readings:
            return self.rules
        rng = random.Random(self.seed)
        positives: List[Dict[str, Any]] = []
        negatives: List[Dict[str, Any]] = []
        for reading in readings:
            spans = candidates(reading.passage)
            if not spans:
                continue
            wanted = self.wants(reading.question)
            gold = reading.answer.lower().strip()
            right = [s for s in spans if s.key == gold]
            wrong = [s for s in spans if s.key != gold]
            if not right:
                continue                    # the corpus promised the span is there; this row lies
            fixed = Setting.of(reading)
            positives.append(probe(reading, right[0], wanted, use_shape=self.use_shape,
                                   setting=fixed))
            for span in rng.sample(wrong, min(NEGATIVES, len(wrong))):
                negatives.append(probe(reading, span, wanted, use_shape=self.use_shape,
                                       setting=fixed))
        if not positives:
            return self.rules
        rules, near = cover(positives, negatives, label="answer",
                            min_support=self.min_support, min_share=self.min_share,
                            max_rules=self.max_rules, max_terms=self.max_terms,
                            purity=self.purity)
        self.rules, self.near_misses = rules, near
        return self.rules

    # -- using it ------------------------------------------------------------------------- #
    def find(self, passage: str, question: str) -> Tuple[str, str]:
        """The span this passage offers in answer, and the rule that chose it.

        ``("", "")`` when no induced rule fires on any candidate — which is an abstention and is
        meant to be. A reader with no reason to prefer one span over another has not found the
        answer; it has picked one.
        """
        reading = Reading(passage=str(passage or ""), question=str(question or ""))
        spans = candidates(reading.passage)
        if not spans or not self.rules:
            return "", ""
        wanted = self.wants(reading.question)
        fixed = Setting.of(reading)
        best: Optional[Tuple[float, int, Span, Rule]] = None
        for span in spans:
            marks = probe(reading, span, wanted, use_shape=self.use_shape, setting=fixed)
            for rule in self.rules:
                if not rule.holds(marks):
                    continue
                # Purest rule wins; among spans the same rule fires on, the earliest, because a
                # tie broken by anything cleverer would be a preference nobody demonstrated.
                score = (rule.purity, rule.support)
                if best is None or score > (best[3].purity, best[3].support) or (
                        score == (best[3].purity, best[3].support) and span.start < best[1]):
                    best = (rule.purity, span.start, span, rule)
        if best is None:
            return "", ""
        return best[2].text, best[3].render()

    def learned(self) -> Dict[str, Any]:
        return {"shown": self.shown, "rules": [r.render() for r in self.rules],
                "purities": [round(r.purity, 4) for r in self.rules],
                "near_misses": len(self.near_misses),
                "consults_asked": bool(self.use_shape and self.asked is not None)}
