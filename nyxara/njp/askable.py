"""NYXARA · njp/askable.py — the wire from reading to answering (🔌, NJP V.100).

V.99 ended on a live failure: a three-sentence passage was read, ``12,262 metres`` was in it, and
the question *"how deep did it reach?"* returned ``''``. The reader produced a
:class:`~nyxara.njp.passage.KnowledgeObject` with ``relations=()`` and nothing downstream could
query it. **Information entered the system in a form nothing could ask a question of.**

This module is that wire, and it is deliberately *only* that wire. What it adds:

    passage → typed facts with provenance → a question turned into a query → an answer or UNKNOWN

What it does **not** add, and must not be read as claiming: discovered representation. The
extraction grammar below is written by hand. V.99 proved supplied structure is free under an
accounting that only prices choices, so :func:`supplied_price` charges it explicitly and the school
reports the number whether or not it flatters the organ.

**The design decision that matters.** A fixed schema — measurement, time, cause, location — was
tried against the develop split and covers a minority of real questions. *"What type of animal
crosses between Europe and Africa during the Autumn?"* has no slot. So facts here carry an **open
relation**: whatever phrase the sentence puts between subject and value. The question's wh-word
then constrains the **type of the answer**, not the name of the relation. That is a weaker claim
than understanding and a stronger one than string matching, and the school exists to find out
whether the difference is worth anything.

**Why this file is not called ``grounding.py``.** It was, for about an hour. ``nyxara/njp/
grounding.py`` already existed — 3,770 lines, the :class:`Grounder`, the organ every question
travels through — and it was overwritten in place by the first draft of this module. The brain
then built no grounder, ``percept.grounding`` stayed ``None`` for every input, ``_deliberate``
returned at its first line, and the calculator, the causal engine and the whole strategy table
became unreachable. Four separate findings were measured and reported off that wreck before the
cause was found, and every one of them was false.

The check that would have caught it is :mod:`nyxara.njp.integrity`, written in V.93 after the
second time this happened. It takes one second. It was not run. That is now the first thing
:mod:`nyxara.njp.askableschool` does.

Pure standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Fact", "Span", "World", "Answer",
    "read_world", "ask", "overlap_span", "always_unknown",
    "answer_type", "supplied_price", "UNKNOWN", "TYPES", "RULES",
]

#: What the organ says when the passage does not answer the question. Never an empty string: an
#: empty answer and a refusal look identical to a caller, and V.99 spent a version on exactly that
#: confusion.
UNKNOWN = "UNKNOWN"

#: The answer types a wh-word can constrain. `thing` is the open case and carries no constraint,
#: which is honest: most of the corpus is `what`/`which` and most of it is not typed.
TYPES = ("date", "person", "place", "number", "measure", "cause", "manner", "thing")

_MONTHS = ("january february march april may june july august september october november december "
           "jan feb mar apr jun jul aug sep sept oct nov dec").split()

_UNITS = ("metres meters metre meter m km kilometres kilometers miles mile feet foot ft inches inch "
          "cm mm degrees degree celsius fahrenheit kg kilograms pounds lb tonnes tons percent % "
          "years year months month days day hours hour minutes minute seconds second "
          "dollars usd euros pounds people births deaths").split()

_STOP = set("""a an the of in on at to for from by with as and or but is are was were be been being
it its this that these those which who whom whose what when where why how did do does done had has
have will would can could may might must not no nor so than then there their them they he she his
her him i you your we our us if into over under between during about after before above below only
such other some any each both more most much many few own same too very s t don now here also""".split())

# --------------------------------------------------------------------------------------------- #
#  what the passage is read into
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Span:
    """A stretch of the passage, with the type it was recognised as and where it came from."""

    text: str
    kind: str = "thing"
    sentence: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "sentence": self.sentence}


@dataclass(frozen=True)
class Fact:
    """``subject — relation — value``, with the relation left open.

    The relation is not drawn from a schema. It is whatever words the sentence puts between the
    subject and the value, lower-cased. A schema was tried first and could not hold the corpus.
    """

    subject: str
    relation: str
    value: Span
    sentence: int = -1
    #: The sentence verbatim. A fact that cannot be traced back to its sentence is a fact nobody
    #: can check, and this package does not keep those.
    support: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "relation": self.relation,
                "value": self.value.to_dict(), "sentence": self.sentence}


@dataclass
class Answer:
    """What came back, and — when nothing did — which kind of nothing it was.

    ``why`` is the beginning of the user's layer-12 distinction: *missing fact* is not the same
    unknown as *representation insufficient*. Only three reasons are separated here, because only
    three can be told apart with what this module knows.
    """

    text: str = UNKNOWN
    kind: str = "thing"
    support: str = ""
    score: float = 0.0
    why: str = ""

    @property
    def unknown(self) -> bool:
        return self.text == UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "score": round(self.score, 3),
                "why": self.why, "support": self.support[:120]}


@dataclass
class World:
    """Everything one passage was read into, with the passage kept."""

    text: str = ""
    sentences: Tuple[str, ...] = ()
    facts: Tuple[Fact, ...] = ()
    spans: Tuple[Span, ...] = ()
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"sentences": len(self.sentences), "facts": len(self.facts),
                "spans": len(self.spans), "source": self.source}


# --------------------------------------------------------------------------------------------- #
#  reading
# --------------------------------------------------------------------------------------------- #
def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])|\n+", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9'%$.,-]+", str(text or ""))


def _stem(word: str) -> str:
    """Crudest possible stemming, and it earns its place.

    Without it *"what does dialysis require?"* and *"Dialysis **requires** a semipermeable
    membrane"* share no word beyond the subject, so that sentence tied with the one before it and
    the tie went to whichever came first. An existing test in this repository caught it, which is
    the only reason it is fixed: the exam could not have, because the exam reports an aggregate.
    """
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _content(text: str) -> List[str]:
    out = []
    for w in _words(text):
        low = w.strip(".,'").lower()
        if low and low not in _STOP and not low.isdigit() or (low.isdigit() and len(low) == 4):
            if low:
                out.append(low if low.isdigit() else _stem(low))
    return out


def _is_year(token: str) -> bool:
    bare = token.strip(".,()")
    return bare.isdigit() and len(bare) == 4 and 1000 <= int(bare) <= 2999


def _dates(sentence: str, index: int) -> List[Span]:
    """Years, and month-day-year runs. A date is the one type this can recognise almost perfectly."""
    found: List[Span] = []
    words = _words(sentence)
    for i, w in enumerate(words):
        if _is_year(w):
            start = i
            if i and words[i - 1].strip(".,").isdigit() and len(words[i - 1].strip(".,")) <= 2:
                start = i - 1
            if start and words[start - 1].strip(".,").lower() in _MONTHS:
                start -= 1
            found.append(Span(" ".join(words[start:i + 1]).strip(".,"), "date", index))
        elif w.lower().strip(".,") in _MONTHS:
            run = [w]
            if i + 1 < len(words) and words[i + 1].strip(".,").isdigit():
                run.append(words[i + 1])
            if len(run) > 1:
                found.append(Span(" ".join(run).strip(".,"), "date", index))
    return found


def _numbers(sentence: str, index: int) -> List[Span]:
    """Bare counts and number-plus-unit measures, kept apart: *how many* and *how deep* differ."""
    found: List[Span] = []
    words = _words(sentence)
    for i, w in enumerate(words):
        bare = w.strip(".,()").replace(",", "")
        if not bare or _is_year(w):
            continue
        if re.fullmatch(r"\$?\d+(\.\d+)?%?", bare):
            unit = ""
            if i + 1 < len(words) and words[i + 1].strip(".,").lower() in _UNITS:
                unit = words[i + 1].strip(".,")
            if unit:
                tail = words[i + 1:i + 5]
                run = [w]
                for t in tail:
                    run.append(t)
                    if t.strip(".,").lower() in _UNITS and len(run) > 1:
                        break
                found.append(Span(" ".join(run).strip(".,"), "measure", index))
            found.append(Span(w.strip(".,"), "number", index))
    return found


def _proper_runs(sentence: str, index: int, *, first_word_ok: bool = False) -> List[Span]:
    """Capitalised runs — the only handle on people and places without a lexicon."""
    found: List[Span] = []
    words = _words(sentence)
    run: List[str] = []
    start = -1
    for i, w in enumerate(words):
        bare = w.strip(".,()'")
        cap = bool(bare) and bare[0].isupper() and not bare.isupper()
        if cap and (i > 0 or first_word_ok):
            if not run:
                start = i
            run.append(bare)
        else:
            if run and (len(run) > 1 or start > 0):
                found.append(Span(" ".join(run), "proper", index))
            run = []
    if run:
        found.append(Span(" ".join(run), "proper", index))
    return found


_CAUSE = re.compile(r"\b(because(?: of)?|due to|owing to|as a result of|thanks to|caused by|"
                    r"in order to|so that|since)\b", re.I)


def _causes(sentence: str, index: int) -> List[Span]:
    found: List[Span] = []
    for m in _CAUSE.finditer(sentence):
        tail = sentence[m.end():].strip()
        tail = re.split(r"[;,.]", tail)[0].strip()
        if len(tail.split()) >= 2:
            found.append(Span(tail, "cause", index))
    return found


_PLACE = re.compile(r"\b(?:in|at|near|from|on)\s+((?:the\s+)?(?:[A-Z][\w'-]+)(?:\s+(?:of|de|the)?\s*"
                    r"[A-Z][\w'-]+){0,3})")


def _places(sentence: str, index: int) -> List[Span]:
    return [Span(m.group(1).strip(), "place", index) for m in _PLACE.finditer(sentence)]


#: The hand-written grammar, named so it can be counted. Each entry is one supplied rule, and
#: :func:`supplied_price` charges for the whole list.
RULES = ("date", "number", "measure", "proper", "place", "cause", "copula", "verb-object")


def _triples(sentence: str, index: int) -> List[Fact]:
    """Open relations: subject, whatever lies between, value.

    Two shapes carry most of the corpus — a copula (*X is Y*) and a verb with an object. The
    relation is never normalised into a schema; ``was condemned by`` stays ``was condemned by``.
    """
    facts: List[Fact] = []
    words = _words(sentence)
    if len(words) < 3:
        return facts
    lowered = [w.lower().strip(".,") for w in words]

    # subject: the leading run up to the first verb-ish token
    verbs = {"is", "are", "was", "were", "has", "have", "had", "became", "become", "includes",
             "include", "reached", "began", "started", "stopped", "ended", "founded", "built",
             "contains", "produces", "made", "used", "called", "known", "consists", "remains"}
    cut = next((i for i, w in enumerate(lowered) if w in verbs), -1)
    if cut <= 0:
        # no copula: take the first capitalised run as subject and the main verb after it
        cut = next((i for i, w in enumerate(lowered)
                    if i and w.endswith("ed") and len(w) > 4), -1)
    if cut <= 0 or cut >= len(words) - 1:
        return facts

    subject = " ".join(words[:cut]).strip(".,")
    rest = words[cut:]
    # relation runs until the first content-bearing value token
    rel_len = 1
    for i in range(1, min(len(rest), 5)):
        low = rest[i].lower().strip(".,")
        if low in _STOP or low in verbs:
            rel_len = i + 1
        else:
            break
    relation = " ".join(rest[:rel_len]).lower().strip(".,")
    value_text = " ".join(rest[rel_len:]).strip(".,")
    if not value_text:
        return facts

    kind = "thing"
    for span in (_dates(value_text, index) + _numbers(value_text, index)):
        kind = span.kind
        break
    facts.append(Fact(subject=subject, relation=relation,
                      value=Span(value_text, kind, index), sentence=index, support=sentence))
    return facts


def read_world(passage: str, *, source: str = "") -> World:
    """Read a passage into typed spans and open triples. **This is the half that was missing.**"""
    text = str(passage or "")
    sentences = _sentences(text)
    spans: List[Span] = []
    facts: List[Fact] = []
    for i, sentence in enumerate(sentences):
        spans.extend(_dates(sentence, i))
        spans.extend(_numbers(sentence, i))
        spans.extend(_proper_runs(sentence, i))
        spans.extend(_places(sentence, i))
        spans.extend(_causes(sentence, i))
        facts.extend(_triples(sentence, i))
    return World(text=text, sentences=tuple(sentences), facts=tuple(facts),
                 spans=tuple(spans), source=source)


# --------------------------------------------------------------------------------------------- #
#  asking
# --------------------------------------------------------------------------------------------- #
def answer_type(question: str) -> str:
    """What kind of thing the wh-word demands. ``thing`` when it demands nothing, which is common."""
    q = " " + str(question or "").lower().strip() + " "
    if re.search(r"\b(when|what year|which year|what date)\b", q):
        return "date"
    if re.search(r"\bwho\b|\bwhose\b|\bwhom\b", q):
        return "person"
    if re.search(r"\bwhere\b", q):
        return "place"
    if re.search(r"\bwhy\b|\bfor what reason\b", q):
        return "cause"
    if re.search(r"\bhow (many|much)\b", q):
        return "number"
    if re.search(r"\bhow (deep|long|far|tall|high|old|fast|wide|big|large)\b", q):
        return "measure"
    if re.search(r"\bhow\b", q):
        return "manner"
    return "thing"


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa)


#: How far ahead the best sentence must be before answering from it. A tie is not evidence.
MARGIN = 0.001


#: Which recognised span kinds satisfy which demanded answer type. One table, read by the sentence
#: chooser and by both answerers, because three copies of it drifting apart is its own defect.
SATISFIES: Dict[str, Tuple[str, ...]] = {
    "date": ("date",), "number": ("number", "measure"), "measure": ("measure", "number"),
    "place": ("place", "proper"), "person": ("proper",), "cause": ("cause",),
}


def _best_sentence(world: World, question: str) -> Tuple[int, float]:
    """The sentence the question is about, or nothing when two are equally good.

    Ties used to go to whichever came first, which is a coin-flip wearing the clothes of a
    decision. On *"what does dialysis require?"* it picked the definition over the requirement and
    intercepted a path that had the right answer. An ambiguous best sentence is grounds to abstain,
    not grounds to guess.
    """
    qw = _content(question)
    asked = set(qw)
    wanted = SATISFIES.get(answer_type(question), ())
    ranked = []
    for i, sentence in enumerate(world.sentences):
        words = set(_content(sentence))
        # Two keys, and the second is not a tie-break dressed as one. `_overlap` divides by the
        # question's length, so every sentence sharing the same *fraction* ties — on the develop
        # split, 51 of 300 rows, thrown away by abstaining. Raw shared count is more evidence, not
        # arbitrary order, so it decides between equal fractions.
        # Third key: does this sentence even contain the kind of thing being asked for? "How far
        # from each other were the motors?" wants a measure, and a sentence with no measure in it
        # cannot be the one holding the answer however many words it shares. On develop, 48 of the
        # 51 tie-abstentions had real overlap and were genuinely answerable — this is what tells
        # them apart, and it is evidence rather than an ordering rule.
        holds = int(any(sp.sentence == i and sp.kind in wanted for sp in world.spans)) if wanted else 0
        ranked.append((len(asked & words) / len(asked) if asked else 0.0,
                       len(asked & words), holds, -i))
    ranked.sort(reverse=True)
    if not ranked:
        return -1, 0.0
    score, shared, holds, negative = ranked[0]
    if len(ranked) > 1:
        rival = ranked[1]
        if score - rival[0] < MARGIN and shared == rival[1] and holds == rival[2]:
            return -1, score
    return -negative, score


#: Below this the organ says UNKNOWN rather than guessing. Set on the develop split only.
FLOOR = 0.20


def ask(world: World, question: str, *, floor: float = FLOOR) -> Answer:
    """Turn a question into a query over the world, or say which kind of nothing was found."""
    want = answer_type(question)
    if not world.sentences:
        return Answer(why="nothing was read", kind=want)

    index, support_score = _best_sentence(world, question)
    if index < 0 or support_score < floor:
        return Answer(kind=want, score=support_score,
                      why="no sentence is about this" if index >= 0 else "missing fact")

    sentence = world.sentences[index]
    qw = set(_content(question))

    # typed lookup: a span of the demanded type, from the sentence the question is about,
    # preferring one whose words are not already in the question.
    wanted = {"date": ("date",), "number": ("number", "measure"), "measure": ("measure", "number"),
              "place": ("place", "proper"), "person": ("proper",), "cause": ("cause",)}.get(want, ())
    if wanted:
        # `wanted` is a preference order, and honouring it means sorting by it. Reading the span
        # list in its own order instead answered "where is the borehole located?" with
        # "Kola Superdeep Borehole" — the question's own subject, offered as a place. A wrong
        # answer, not an abstention, which is the one failure this exam punishes hardest.
        def rank(span: Span) -> Tuple[int, int, int]:
            shared = len(set(_content(span.text)) & qw)
            return (wanted.index(span.kind), 0 if span.sentence == index else 1, shared)

        pool = sorted((s for s in world.spans if s.kind in wanted), key=rank)
        pool = [s for s in pool if s.sentence == index] + [s for s in pool if s.sentence != index]
        fresh = [s for s in pool if not set(_content(s.text)) <= qw]
        pick = (fresh or pool)
        if pick:
            top = min(pick, key=rank)
            return Answer(text=top.text, kind=top.kind, support=sentence,
                          score=support_score, why="typed span")
        return Answer(kind=want, score=support_score,
                      why="representation insufficient: nothing of that type was recognised")

    # open case: the value of the fact from this sentence whose subject and relation the question
    # overlaps most. This is where a schema would have had nothing to say at all.
    remainder = _by_extent(sentence, question)
    if remainder:
        return Answer(text=remainder, kind="thing", support=sentence,
                      score=support_score, why="what the question did not already say")

    here = [f for f in world.facts if f.sentence == index]
    best, best_score = None, 0.0
    for fact in here:
        got = _overlap(_content(question), _content(fact.subject + " " + fact.relation))
        if got > best_score:
            best, best_score = fact, got
    if best is not None and best_score > 0.0:
        return Answer(text=best.value.text, kind=best.value.kind, support=sentence,
                      score=support_score, why="open relation")
    return Answer(kind=want, score=support_score, why="causal model ambiguous: no relation matched")


# --------------------------------------------------------------------------------------------- #
#  extent, discovered rather than supplied
# --------------------------------------------------------------------------------------------- #
def _runs_absent_from(sentence: str, question: str) -> List[Tuple[int, int, List[str]]]:
    """Maximal stretches of the sentence the question does **not** already say.

    V.100 measured its own transfer failure precisely: gold answers are 2 words on SQuAD and 8 on
    the transfer corpora, and the organ emitted 3-word spans either way. It knew the answer's type
    and not its **extent**, and extent was a constant somebody supplied.

    The hypothesis here makes extent a function of the question instead of a constant: **what is
    being asked for is the part of the sentence the asker did not already say.** A question that
    quotes most of its sentence leaves a short remainder; one that shares only a topic word leaves
    a long one. That is the SQuAD/QuAC difference, arrived at without being told about either.
    """
    words = _words(sentence)
    if not words:
        return []
    asked = set(_content(question))
    marked = [(_stem(w.strip(".,'").lower()) in asked) for w in words]
    runs: List[Tuple[int, int, List[str]]] = []
    start = -1
    for i, seen in enumerate(marked + [True]):
        if seen:
            if start >= 0:
                runs.append((start, i, words[start:i]))
                start = -1
        elif start < 0:
            start = i
    out: List[Tuple[int, int, List[str]]] = []
    for a, b, run in runs:
        while run and run[0].strip(".,'").lower() in _STOP:
            run, a = run[1:], a + 1
        while run and run[-1].strip(".,'").lower() in _STOP:
            run, b = run[:-1], b - 1
        if run:
            out.append((a, b, run))
    return out


def _by_extent(sentence: str, question: str, *, near: Optional[Tuple[int, int]] = None) -> str:
    """The best remainder, preferring one that sits beside something the question did mention.

    Adjacency is not a tuning knob, it is the same idea again: the answer to *"what year did X
    begin"* stands next to *begin*, not at the far end of the sentence.
    """
    runs = _runs_absent_from(sentence, question)
    if not runs:
        return ""
    words = _words(sentence)
    asked = set(_content(question))

    def touches(a: int, b: int) -> int:
        before = a - 1 >= 0 and _stem(words[a - 1].strip(".,'").lower()) in asked
        after = b < len(words) and _stem(words[b].strip(".,'").lower()) in asked
        return int(before) + int(after)

    def rank(item: Tuple[int, int, List[str]]) -> Tuple[int, int, int]:
        a, b, run = item
        inside = 0
        if near is not None:
            inside = int(a <= near[0] and b >= near[1])
        return (inside, touches(a, b), len(run))

    return " ".join(max(runs, key=rank)[2]).strip(".,")


# --------------------------------------------------------------------------------------------- #
#  the two nulls the protocol requires
# --------------------------------------------------------------------------------------------- #
def always_unknown(world: World, question: str) -> Answer:
    """Null 1. Perfect on absence, useless on everything else — and it must be beaten on accuracy."""
    return Answer(why="null: never answers")


def overlap_span(world: World, question: str, *, floor: float = FLOOR) -> Answer:
    """Null 2, the one that matters: best sentence, then the first span of the demanded type.

    No facts, no relations, no subject. If the world model cannot beat this, the world model is
    unpaid structure — V.92, one level up.
    """
    want = answer_type(question)
    index, score = _best_sentence(world, question)
    if index < 0 or score < floor:
        return Answer(kind=want, score=score, why="null: nothing overlapped")
    sentence = world.sentences[index]
    wanted = {"date": ("date",), "number": ("number", "measure"), "measure": ("measure", "number"),
              "place": ("place", "proper"), "person": ("proper",), "cause": ("cause",)}.get(want, ())
    if wanted:
        # The same preference-order fix as `ask`. Giving the organ a correction and withholding it
        # from the null would make the null weaker on purpose, which is a target-shaped control.
        pool = sorted((s for s in world.spans if s.sentence == index and s.kind in wanted),
                      key=lambda s: wanted.index(s.kind))
        if pool:
            return Answer(text=pool[0].text, kind=pool[0].kind, support=sentence,
                          score=score, why="null: typed span")
        return Answer(kind=want, score=score, why="null: no span of that type")
    return Answer(text=sentence, kind="thing", support=sentence, score=score,
                  why="null: whole sentence")


# --------------------------------------------------------------------------------------------- #
#  what the grammar costs, per V.99
# --------------------------------------------------------------------------------------------- #
def supplied_price(rules: int = len(RULES), types: int = len(TYPES)) -> float:
    """Bits of source this module carries because it was **told**, not because it searched.

    V.99: an accounting that prices only run-time choices reports supplied structure as free, and
    then a hard-code beats every search. So the grammar is charged here — one choice per rule and
    one per answer type — and the school reports it beside whatever the grammar bought.
    """
    from nyxara.njp.supply import carried

    # No floor under the bound. A first draft wrote `max(2, rules)`, which keeps the price above
    # zero for a one-rule grammar — and a one-rule grammar *is* free to choose among, which is the
    # entire V.99 finding. Putting a floor here would have hidden that phenomenon inside the
    # function written to charge for it.
    return round(rules * carried(rules) + types * carried(types), 2)
