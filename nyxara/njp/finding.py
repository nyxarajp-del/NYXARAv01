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
           "candidates", "gold_key", "probe"]

CORPUS = Path(__file__).with_name("data") / "flan_reading.jsonl.gz"

#: The longest span that may be an answer, in tokens. Not a taste: the corpus builder already
#: refused any answer over 120 characters, and of what survives the ninety-ninth percentile is
#: under this. A longer candidate is a sentence, and a sentence is not a span.
MAX_TOKENS = 10

#: How many non-answer candidates each training passage contributes. Every candidate is scored at
#: examination time; only the induction is sampled, because a paragraph offers a few thousand
#: negatives and a hundred passages would otherwise be half a million readings.
NEGATIVES = 24

#: Punctuation that may sit at either edge of a span without being part of it.
_EDGE = " .,;:!?\"'()[]“”‘’"

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
        """The form a candidate is compared to a gold answer in.

        Trailing and leading punctuation is stripped from both sides, and that is not tidiness:
        an annotator's span very often carries the sentence's own full stop — ``Karabakh police.``,
        ``Czech Republic.`` — while a candidate ends at the last token. Comparing them raw made a
        tenth of the corpus unreachable by construction, and the decomposition reported it as the
        generator's failure to propose the answer rather than as a mismatch in how two strings
        were spelled.
        """
        return " ".join(self.text.lower().split()).strip(_EDGE)


def gold_key(answer: str) -> str:
    """A marked answer in the same form :attr:`Span.key` puts a candidate in."""
    return " ".join(str(answer or "").lower().split()).strip(_EDGE)


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(str(text or ""))]


def candidates(passage: str) -> List[Span]:
    """Every span that could be an answer: open-class at both ends, inside one sentence.

    The two constraints are what keep this to thousands rather than tens of thousands, and both
    are statements about phrases rather than about answers. A span opening on ``of`` or closing on
    ``the`` is a fragment; a span running across a full stop is two things.
    """
    closed = _closed()
    try:
        from nyxara.njp.semantics import Tag  # noqa: WPS433
        determiner = Tag.DET
    except Exception:  # noqa: BLE001
        determiner = "DET"
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
        # A determiner may open a span even though it is closed class, because that is how a span
        # is marked: `the Henry Cole Wing`, `the mid-19th century`. Nothing else closed-class may,
        # so `of the wing` is still refused. Measured: 3.7% of the corpus's answers open on `the`
        # and were unreachable without this.
        if word.lower() in closed and closed.get(word.lower()) != determiner:
            continue
        if closed.get(word.lower()) == determiner and i + 1 >= len(toks):
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


def _neighbours(passage: str, span: "Span") -> Tuple[str, str]:
    """What sits either side of a span: another content word, a closed-class word, or an edge."""
    closed = _closed()
    before = passage[:span.start]
    after = passage[span.end:]
    left_words = _WORD.findall(before)
    right_words = _WORD.findall(after)
    def name(words: Sequence[str], edge: str) -> str:
        if not words:
            return "edge"
        word = (words[-1] if edge == "left" else words[0]).lower()
        return "closed" if word in closed else "open"
    # Punctuation directly against the span is as good a boundary as a closed-class word, and
    # better: an annotator's span very often ends at a comma or a full stop.
    touching_left = before[-1:] in (",", ".", ";", ":", "(", "[", '"', "'")
    touching_right = after[:1] in (",", ".", ";", ":", ")", "]", '"', "'", "?", "!")
    return ("stop" if touching_left else name(left_words, "left"),
            "stop" if touching_right else name(right_words, "right"))


def _bucket(n: int, edges: Sequence[int], names: Sequence[str]) -> str:
    for edge, name in zip(edges, names):
        if n <= edge:
            return name
    return names[-1]


def sentences_of(passage: str) -> List[Tuple[str, int]]:
    """The passage's sentences with the character each starts at."""
    raw = str(passage or "")
    out: List[Tuple[str, int]] = []
    at = 0
    for piece in _SENTENCE.split(raw):
        at = raw.find(piece, at)
        out.append((piece, at))
        at += len(piece)
    return out


def probe_sentence(reading: "Reading", index: int, wanted: str = "",
                   *, use_shape: bool = True,
                   setting: Optional["Setting"] = None) -> Dict[str, Any]:
    """Generic measurements of one **sentence** of the passage.

    The reader works in two stages, and this is the first, because a one-stage reader cannot be
    made to work with this machinery: a paragraph offers a thousand spans, one of which is the
    answer, and :func:`~nyxara.njp.induce.cover` maximises how many positives a rule covers rather
    than how few negatives it admits. Asked to separate 1 from 999 it returns ``shape is span`` at
    a purity of 0.075 — true of the answer and of six hundred other things — and ranking by it
    picks whichever short span comes first. Measured, that scored 0.026 overlap against 0.143 for
    a ten-minute heuristic.

    Choosing the sentence first is not a trick to make the number better; it is the step that has
    a base rate an induction can work with. A passage has a handful of sentences, so the answer's
    sentence is one in five or six rather than one in a thousand.
    """
    fixed = setting if setting is not None else Setting.of(reading)
    said, start = fixed.spans[index] if index < len(fixed.spans) else ("", 0)
    here = fixed.carried[index] if index < len(fixed.carried) else frozenset()
    asked = fixed.asked
    shared = len(here & asked)
    out: Dict[str, Any] = {
        "carries": _bucket(shared, (0, 1, 2, 4), ("none", "one", "two", "few", "many")),
        "share_of_question": _share(shared, len(asked)),
        "share_of_sentence": _share(shared, len(here)),
        "position": _bucket(index, (0, 1, 3), ("first", "second", "early", "later")),
        "length": _bucket(len(_tokens(said)), (8, 20, 40), ("short", "medium", "long",
                                                            "very long")),
        "has_number": any(ch.isdigit() for ch in said),
        "is_only": len(fixed.spans) <= 1,
    }
    if use_shape:
        # The other organ again, and at this level it says whether the sentence contains anything
        # of the right shape at all — a `when` question wants a sentence with a date in it.
        out["holds_wanted"] = ("unknown" if not wanted
                               else "yes" if _holds_kind(said, wanted) else "no")
    return out


def _holds_kind(sentence: str, wanted: str) -> bool:
    """Does this sentence contain any span of the kind the question wants?"""
    for span in candidates(sentence):
        if satisfies(shape_of(span.text), wanted):
            return True
    return False


def _share(part: int, whole: int) -> str:
    if not whole:
        return "none"
    value = part / whole
    if value < 0.2:
        return "little"
    if value < 0.5:
        return "some"
    return "most"


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
    #: Each sentence with the character it starts at, so a span can be placed in one.
    spans: Tuple[Tuple[str, int], ...] = ()
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
                   spans=tuple(sentences_of(reading.passage)),
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
    # Whether the span is a whole phrase or a piece of one. Within a single sentence this is the
    # signal that matters, and it was missing: `Felipa`, `Felipa Moñiz` and `amb Felipa Moñiz` are
    # all candidates, they share every other measurement, and only one of them is what the
    # annotator marked. A span whose neighbours are closed class or punctuation is bounded; one
    # that could be extended by another content word is a fragment of something longer.
    left, right = _neighbours(reading.passage, span)
    out: Dict[str, Any] = {
        "bounded_left": left,
        "bounded_right": right,
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

    purity: float = 0.30
    #: The first stage separates one sentence from a handful, so it can be held to a higher bar
    #: than the second, which separates one span from dozens. Both are set by the sweep in
    #: :mod:`nyxara.njp.findingschool` and neither is a taste.
    sentence_purity: float = 0.45
    min_support: int = 12
    min_share: float = 0.02
    max_rules: int = 6
    max_terms: int = 3
    learning: bool = True
    use_shape: bool = True
    #: How the first stage picks its sentence: ``"overlap"`` (argmax over shared content words) or
    #: ``"rules"`` (the induced conjunctions). Measured, not preferred — see :meth:`sentence`.
    sentence_by: str = "overlap"
    seed: int = 7
    rules: List[Rule] = field(default_factory=list)
    sentence_rules: List[Rule] = field(default_factory=list)
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
        """Two inductions: which sentence holds the answer, and which span inside it is one."""
        self.rules, self.sentence_rules, self.near_misses = [], [], []
        self.shown = len(readings)
        if not self.learning or not readings:
            return self.rules
        rng = random.Random(self.seed)
        say_yes: List[Dict[str, Any]] = []
        say_no: List[Dict[str, Any]] = []
        span_yes: List[Dict[str, Any]] = []
        span_no: List[Dict[str, Any]] = []
        for reading in readings:
            fixed = Setting.of(reading)
            wanted = self.wants(reading.question)
            gold = gold_key(reading.answer)
            at = reading.passage.lower().find(gold)
            if at < 0:
                continue
            holds = max((i for i, (_t, start) in enumerate(fixed.spans) if start <= at),
                        default=0)
            for index in range(len(fixed.spans)):
                marks = probe_sentence(reading, index, wanted, use_shape=self.use_shape,
                                       setting=fixed)
                (say_yes if index == holds else say_no).append(marks)

            # The span stage learns **inside the right sentence only**, because that is the
            # situation it will be used in. Training it against the whole passage would teach it
            # to solve a problem the first stage has already solved.
            said, start = fixed.spans[holds] if holds < len(fixed.spans) else ("", 0)
            here = [Span(text=sp.text, start=sp.start + start, end=sp.end + start,
                         sentence=holds) for sp in candidates(said)]
            right = [sp for sp in here if sp.key == gold]
            wrong = [sp for sp in here if sp.key != gold]
            if not right:
                continue
            span_yes.append(probe(reading, right[0], wanted, use_shape=self.use_shape,
                                  setting=fixed))
            for span in rng.sample(wrong, min(NEGATIVES, len(wrong))):
                span_no.append(probe(reading, span, wanted, use_shape=self.use_shape,
                                     setting=fixed))

        if say_yes:
            rules, near = cover(say_yes, say_no, label="holds",
                                min_support=self.min_support, min_share=self.min_share,
                                max_rules=self.max_rules, max_terms=self.max_terms,
                                purity=self.sentence_purity)
            self.sentence_rules, self.near_misses = rules, near
        if span_yes:
            self.rules, near = cover(span_yes, span_no, label="answer",
                                     min_support=self.min_support, min_share=self.min_share,
                                     max_rules=self.max_rules, max_terms=self.max_terms,
                                     purity=self.purity)
            self.near_misses.extend(near)
        return self.rules

    # -- using it ------------------------------------------------------------------------- #
    def _score(self, marks: Dict[str, Any], rules: Sequence[Rule]) -> Tuple[int, float]:
        """How much induced evidence this reading carries: how many rules fire, then how pure.

        Counting first rather than taking the single purest is what makes a set of rules rank
        rather than merely classify. It is an aggregation of the induction and not a weighting on
        top of it: no rule is worth more than another until the evidence says one of them is
        purer, which is exactly the tie-break.
        """
        fired = [r for r in rules if r.holds(marks)]
        return len(fired), sum(r.purity for r in fired)

    def sentence(self, passage: str, question: str) -> int:
        """Which sentence holds the answer. ``-1`` when nothing has been learned.

        Which mechanism decides is :attr:`sentence_by`, and the honest answer is not the induced
        one. Measured on 600 held-out passages, ranking sentences by how many induced rules fire
        levels off at 0.570 however many rules are allowed, while taking the sentence with the most
        content words in common with the question — one line, no learning — gets 0.655. The
        induction *rediscovers* that signal (``carries is many``, ``carries is few``, ``carries is
        two``, correctly ordered by purity) and cannot use it as well, because
        :func:`~nyxara.njp.induce.cover` builds equality tests over buckets and every sentence
        sharing five or more words falls in the same bucket and ties.

        So the default is ``"overlap"`` and the heuristic is load-bearing. It is kept as a switch
        rather than hard-wired so the cost of the induced version stays runnable and visible, and
        so that the span stage — which the induction *does* earn its place in — is measured on top
        of the better first stage rather than a worse one.
        """
        reading = Reading(passage=str(passage or ""), question=str(question or ""))
        fixed = Setting.of(reading)
        if not fixed.spans:
            return -1
        if self.sentence_by == "overlap":
            weight = [len(fixed.carried[i] & fixed.asked) for i in range(len(fixed.spans))]
            return max(range(len(weight)), key=lambda i: weight[i]) if weight else -1
        if not self.sentence_rules:
            return -1
        wanted = self.wants(reading.question)
        best, chosen = (0, 0.0), -1
        for index in range(len(fixed.spans)):
            marks = probe_sentence(reading, index, wanted, use_shape=self.use_shape,
                                   setting=fixed)
            score = self._score(marks, self.sentence_rules)
            if score > best:
                best, chosen = score, index
        return chosen

    def find(self, passage: str, question: str) -> Tuple[str, str]:
        """The span this passage offers in answer, and the rule that chose it.

        ``("", "")`` when nothing has been learned, or when no induced rule fires on any candidate
        in the chosen sentence. That is an abstention and is meant to be: a reader with no reason
        to prefer one span over another has not found the answer, it has picked one.
        """
        reading = Reading(passage=str(passage or ""), question=str(question or ""))
        fixed = Setting.of(reading)
        if not self.rules or not fixed.spans:
            return "", ""  # nothing learned about spans; the sentence alone is not an answer
        index = self.sentence(reading.passage, reading.question)
        if index < 0:
            return "", ""
        said, start = fixed.spans[index]
        wanted = self.wants(reading.question)
        best, chosen, why = (0, 0.0), None, ""
        for span in candidates(said):
            placed = Span(text=span.text, start=span.start + start, end=span.end + start,
                          sentence=index)
            marks = probe(reading, placed, wanted, use_shape=self.use_shape, setting=fixed)
            score = self._score(marks, self.rules)
            if score > best:
                best, chosen = score, placed
                why = "; ".join(r.render() for r in self.rules if r.holds(marks))
        if chosen is None:
            return "", ""
        return chosen.text, why

    def learned(self) -> Dict[str, Any]:
        return {"shown": self.shown, "rules": [r.render() for r in self.rules],
                "purities": [round(r.purity, 4) for r in self.rules],
                "sentence_rules": [r.render() for r in self.sentence_rules],
                "sentence_purities": [round(r.purity, 4) for r in self.sentence_rules],
                "near_misses": len(self.near_misses),
                "consults_asked": bool(self.use_shape and self.asked is not None)}
