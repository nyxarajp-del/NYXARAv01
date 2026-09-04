"""NYXARA · njp/asking.py — the question form, induced rather than listed (🗝, NJP V.38).

:mod:`nyxara.njp.explainread` is a table of regular expressions, and
:mod:`nyxara.njp.explaingauntlet` measured what that is worth: **4 of 40**, on ten phrasings of
one question. Nine of the ten were unreachable. Not misread — unreachable, because a grammar of
surface forms is a list and a list has an edge.

    why does X happen?                    ✓   the one the table was built around
    what brings about X?                  ✗
    on account of what does X occur?      ✗
    through what process does X arise?    ✗
    what is responsible for X?            ✗
    how come X happens?                   ✗
    what gives rise to X?                 ✗
    X happens because of what?            ✗
    what accounts for X?                  ✗
    what lies behind X?                   ✗

Adding nine more rows would move that number and would teach her nothing. The tenth phrasing would
miss, and the eleventh, and the author of each new row is doing the work the mind is supposed to do.

So this module holds no phrasings. It holds the **mechanism that acquires one**, and everything it
knows arrives through :meth:`Asking.show` — a question, the topic it was about, and which walk it
called for.

What a demonstration leaves behind
----------------------------------

A **cue**, not a template. Given *"what brings about thunder?"* and the fact that the topic was
``thunder``, what is left is ``what brings about ?``, and of those words ``what`` and ``about`` are
already in :data:`nyxara.njp.semantics._CLOSED` — she has had the closed class since V.31 and
induced it from exposure rather than being handed it. The word that carries the question's identity
is the one that is **not** closed: ``brings``. That is the cue, and it is what generalises.

This is the same move :mod:`nyxara.njp.discourse`'s ``ActLearner`` makes and the same one
``Induction`` makes for a construction's function: the closed class is the skeleton, the open class
is the content, and what a construction *means* is carried by which open-class words sit in which
skeleton. Here the skeleton says *this is a question* and the cue says *this is a question about
causes*.

Three rules, and each is a rule this repository has paid for
-----------------------------------------------------------

**A cue is kept only once two demonstrations that differ in their topic agree on it.** One
demonstration cannot separate the cue from the topic: shown *"what brings about thunder?"* alone,
``thunder`` is as good a candidate for the causal word as ``brings``. Two demonstrations with
different topics and the same frame leave exactly the cue standing. The rule is
``discourse.ActLearner``'s and it is not weakened here.

**Where two demonstrations disagree about the walk, the cue is contested and never read.** ``what``
appears in a why-question, a how-question and *"what is a mammal"* — so ``what`` alone can never
be a cue, and it is not excluded by a list of stop words, it is excluded by having been
demonstrated with three different answers. :attr:`Cue.contested` is the record, and a contested cue
is dropped rather than resolved by majority: a cue that means two things is a cue that means
nothing.

**Negative demonstrations are demonstrations — and on this lesson they are not load-bearing, which
is measured rather than assumed.** *"What is a mammal?"* shown with walk ``""`` is how a form is
taught to carry no causal claim. The honest result of running the lesson with and without them:
**settled 24 either way, false positives 0 either way.** The reason is that ``what`` and ``is`` are
already in the closed class, so they never become cues in the first place and there is nothing for
a negative to contest. Exactly one cue is contested by the lesson as written — ``job``, shared
between *"what job does {x} do?"* and *"whose job is {x}?"* — and switching the contest rule off
changes no reading, because the specific levels still carry that form.

They are kept, and labelled as what they are: a guard that fires when a lesson teaches an ambiguous
cue, not a mechanism this lesson needs. A guard that has never fired is worth keeping and is not
worth claiming credit for, and the difference between those two sentences is the whole reason this
paragraph reports a number.

What it cannot do, stated rather than hidden
--------------------------------------------

**A cue nobody demonstrated is not read.** She cannot know that *"what precipitates X"* asks for a
cause if no demonstration ever carried a word in that position — and she can learn it from two.
That is the honest shape of an induced grammar, and it is why the gauntlet reports the taught and
the untaught halves of ``wording`` **separately**: one measures whether teaching works, the other
measures generalisation past what was taught, and a single number over both would let teaching
more phrasings look like reasoning better.

**Word order beyond "where the topic sits" is not modelled.** The frame keeps the tokens before
the slot and the tokens after it, and matches on both. *"X happens because of what?"* — slot first
— is a different frame from *"what accounts for X?"* and is learned separately, which is correct
and is not generalisation.

Pure standard library, deterministic, and empty until it is shown something.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Cue", "Frame", "Demonstration", "Asking", "SLOT", "LEVELS", "teach", "LESSON"]

#: The token that stands where the topic was.
SLOT = "▫"

#: The levels a demonstration is generalised to, most specific first. Reading tries them in this
#: order and stops at the first that is neither empty nor contested — the ``ActLearner`` ladder,
#: which is there so that a specific frame is preferred and a general one is a fallback rather
#: than a first guess.
LEVELS: Tuple[str, ...] = ("frame", "cued", "cue")

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(str(text or ""))]


def _closed() -> Dict[str, str]:
    try:
        from nyxara.njp.semantics import _CLOSED, tag_tokens  # noqa: F401
        return _CLOSED
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# What a demonstration is, and what it leaves
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Demonstration:
    question: str
    topic: str
    walk: str = ""      # "" is a negative: this question is not about causes at all

    @property
    def negative(self) -> bool:
        return not self.walk


@dataclass(frozen=True)
class Frame:
    """A question with its topic removed: the tokens before the slot and the tokens after it.

    Both halves, and matched on both. A frame that kept only the prefix would read *"what causes
    X"* and *"what causes X to fail"* as one question, and they are two.
    """

    before: Tuple[str, ...]
    after: Tuple[str, ...]

    def render(self) -> str:
        return " ".join(list(self.before) + [SLOT] + list(self.after))


@dataclass
class Cue:
    """One induced form, at one level, with the evidence for it.

    ``walks`` is a count per walk rather than a single value, because that is what makes
    :attr:`contested` a measurement instead of a decision. A cue demonstrated once as ``because``
    and once as ``mechanism`` has not been narrowly outvoted; it has been shown to carry neither.
    """

    level: str
    key: Any
    walks: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    topics: Set[str] = field(default_factory=set)
    #: How many **distinct surfaces** this cue has been seen in. At the general level a cue is
    #: only read once this is at least two: see :attr:`generalises`.
    frames: Set[Frame] = field(default_factory=set)

    @property
    def support(self) -> int:
        return sum(self.walks.values())

    @property
    def positive(self) -> int:
        return sum(n for walk, n in self.walks.items() if walk)

    @property
    def contested(self) -> bool:
        """Demonstrated with more than one answer. Never read, never resolved by majority."""
        return len([w for w, n in self.walks.items() if n]) > 1 or (
            self.positive > 0 and self.walks.get("", 0) > 0)

    @property
    def generalises(self) -> bool:
        """Has this cue been seen carrying its meaning in more than one surface?

        Only checked at the general level, and it is the two-demonstrations rule applied to the
        axis it was being skipped on. The specific levels are corroborated by two *topics* — that
        is what separates the cue from the topic. The general level exists to reach a **surface**
        nobody demonstrated, and a cue seen in exactly one surface has produced no evidence at all
        that it is the word carrying the meaning rather than an accident of that one phrasing.

        Measured, this is not a hypothetical. ``what is {x} there for?`` has ``what``, ``is`` and
        ``for`` all in the closed class, so the cue it leaves is ``there`` alone — and that cue
        read *"how many bones are there?"* as a purpose question. I only believed ``brings`` and
        disbelieved ``there`` because I am a speaker of English, which is exactly the knowledge
        this module is forbidden to import. One surface is one surface either way.
        """
        return len(self.frames) >= 2

    @property
    def settled(self) -> Optional[str]:
        """The walk this cue calls for, or ``None`` while the evidence does not support one.

        Two topics, not two demonstrations. The same question shown twice teaches nothing about
        which of its words is the topic — that is what a second *filler* is for, and it is the rule
        `ActLearner` was built on.
        """
        if self.contested or len(self.topics) < 2 or not self.positive:
            return None
        if self.level == "cue" and not self.generalises:
            return None
        return max((w for w in self.walks if w), key=lambda w: self.walks[w])

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level,
                "key": self.key.render() if isinstance(self.key, Frame) else self.key,
                "walks": dict(self.walks), "topics": len(self.topics),
                "frames": len(self.frames), "generalises": self.generalises,
                "contested": self.contested, "settled": self.settled}


# --------------------------------------------------------------------------- #
# The learner
# --------------------------------------------------------------------------- #
class Asking:
    """A reader that starts knowing nothing about how a question is phrased.

    Constructed empty and stays empty: :meth:`read` returns ``None`` for everything until
    :meth:`show` has been called at least twice with the same frame and different topics. That is
    asserted by a test, and it is the whole claim — nothing in this file is a phrasing.
    """

    def __init__(self) -> None:
        self.cues: Dict[Tuple[str, Any], Cue] = {}
        self.shown: List[Demonstration] = []

    # ---- being shown ----------------------------------------------------- #
    def show(self, question: str, *, topic: str, walk: str = "") -> bool:
        """One demonstration. Returns whether it left anything new standing.

        ``walk=""`` is a **negative**: this question carries no causal claim. It contests any cue
        it shares with a positive. On the shipped lesson that guard fires once and changes no
        reading — see the module docstring, which reports the measurement rather than the intent.
        """
        got = Demonstration(question=str(question or ""), topic=str(topic or ""), walk=str(walk))
        frame = self._frame(got.question, got.topic)
        if frame is None:
            return False
        self.shown.append(got)
        before = self._settled_count()
        for level, key in self._keys(frame):
            cue = self.cues.get((level, key))
            if cue is None:
                cue = self.cues[(level, key)] = Cue(level=level, key=key)
            cue.walks[got.walk] += 1
            cue.frames.add(frame)
            if got.topic:
                cue.topics.add(got.topic.lower())
        return self._settled_count() != before

    def _settled_count(self) -> int:
        return sum(1 for cue in self.cues.values() if cue.settled)

    def _frame(self, question: str, topic: str) -> Optional[Frame]:
        """The question with the topic taken out, or ``None`` when the topic is not in it."""
        words = _tokens(question)
        wanted = _tokens(topic)
        if not words or not wanted:
            return None
        for start in range(len(words) - len(wanted) + 1):
            if words[start:start + len(wanted)] == wanted:
                return Frame(before=tuple(words[:start]),
                             after=tuple(words[start + len(wanted):]))
        return None

    def _keys(self, frame: Frame) -> List[Tuple[str, Any]]:
        """The three levels one frame generalises to.

        ``frame`` is the exact surface minus the topic. ``cued`` keeps the open-class words and
        **the shape** — how many tokens sat before the slot and how many after — throwing away
        *which* function words those were, so *"what brings about X"* and *"how does X get brought
        about"* generalise over the auxiliaries without generalising over the length. ``cue`` is
        the open-class words alone, and it is the level that can reach a surface nobody
        demonstrated.

        The shape used to be two booleans — *were there words before, were there words after* —
        and that was the defect. ``what is {x} there for?`` has the shape ``2 ▫ 2``; *"how many
        bones are there?"* has ``1 ▫ 1`` for the same cue word, and with only the booleans it
        matched and came back as a purpose question. Keeping the counts costs nothing that
        generalisation needed and closes it exactly.
        """
        closed = _closed()
        words = list(frame.before) + list(frame.after)
        open_class = tuple(w for w in words if w not in closed)
        return [
            ("frame", frame),
            ("cued", (open_class, len(frame.before), len(frame.after))),
            ("cue", open_class),
        ]

    # ---- reading --------------------------------------------------------- #
    def read(self, question: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
        """``(walk, candidate topics)`` or ``None``. Most specific level first.

        The topic is what is left when the matched form's own words are removed, which is the
        inverse of how the form was learned and is why no separate slot-finder is needed. Its
        spellings come from :func:`nyxara.njp.explainread.candidates_for`, so the store's own
        folding still applies.
        """
        words = _tokens(question)
        if not words:
            return None
        for level in LEVELS:
            got = self._read_at(level, words)
            if got is not None:
                return got
        return None

    def _read_at(self, level: str, words: Sequence[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
        from nyxara.njp.explainread import candidates_for

        best: Optional[Tuple[int, str, List[str]]] = None
        for (kind, key), cue in self.cues.items():
            if kind != level:
                continue
            walk = cue.settled
            if not walk:
                continue
            span = self._match(level, key, words)
            if span is None:
                continue
            topic = " ".join(span)
            if not topic:
                continue
            # Longest matched form wins: a more specific form that fits is a better reading than
            # a general one that also fits, and at one level "more specific" is "used more words".
            weight = len(words) - len(span)
            if best is None or weight > best[0]:
                best = (weight, walk, span)
        if best is None:
            return None
        _weight, walk, span = best
        got = candidates_for(" ".join(span), verbal=(walk == "procedure"))
        return (walk, tuple(got)) if got else None

    def _match(self, level: str, key: Any, words: Sequence[str]) -> Optional[List[str]]:
        """What the topic would be if this form matched, or ``None``.

        Returns the *topic tokens*, so a caller gets the reading and the slot in one pass.
        """
        words = list(words)
        if level == "frame":
            frame: Frame = key
            n, m = len(frame.before), len(frame.after)
            if len(words) <= n + m:
                return None
            if tuple(words[:n]) != frame.before:
                return None
            if m and tuple(words[len(words) - m:]) != frame.after:
                return None
            return words[n:len(words) - m] if m else words[n:]
        if level == "cued":
            open_class, before_n, after_n = key
            if len(words) <= before_n + after_n:
                return None
            span = words[before_n:len(words) - after_n] if after_n else words[before_n:]
            if not span:
                return None
            closed = _closed()
            rest = words[:before_n] + (words[len(words) - after_n:] if after_n else [])
            if tuple(w for w in rest if w not in closed) != tuple(open_class):
                return None
            # And the topic itself must not be a run of function words: "how many bones are there"
            # would otherwise offer `many bones` as a topic on a shape that happened to line up.
            if all(w in closed for w in span):
                return None
            return span
        return self._carve(words, key)

    def _carve(self, words: Sequence[str], open_class: Sequence[str]) -> Optional[List[str]]:
        """Remove the form's own words; whatever contiguous run is left is the topic.

        Contiguous is the check that matters. Removing the cue words from *"what brings about X"*
        leaves ``X`` in one piece; removing them from a sentence that merely happens to contain
        them leaves two fragments, and two fragments is not a topic — it is a coincidence, and
        returning the longer of them is how a reader starts answering questions it did not
        understand.
        """
        closed = _closed()
        wanted = list(open_class)
        if not wanted:
            return None
        low = [w.lower() for w in words]
        positions: List[int] = []
        cursor = 0
        for word in wanted:
            try:
                at = low.index(word, cursor)
            except ValueError:
                return None
            positions.append(at)
            cursor = at + 1
        taken = set(positions)
        runs: List[List[str]] = []
        current: List[str] = []
        for i, word in enumerate(words):
            if i in taken or word in closed:
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(word)
        if current:
            runs.append(current)
        if len(runs) != 1:
            return None
        return runs[0]

    # ---- what it knows --------------------------------------------------- #
    def settled(self) -> List[Cue]:
        return sorted((c for c in self.cues.values() if c.settled),
                      key=lambda c: (LEVELS.index(c.level), -c.support))

    def contested(self) -> List[Cue]:
        return [c for c in self.cues.values() if c.contested]

    def to_dict(self) -> Dict[str, Any]:
        return {"shown": len(self.shown), "cues": len(self.cues),
                "settled": len(self.settled()), "contested": len(self.contested())}


# --------------------------------------------------------------------------- #
# A lesson, which is a set of demonstrations and not a set of answers
# --------------------------------------------------------------------------- #
#: What a teacher demonstrates. Each entry is a phrasing and the walk it calls for, and it is used
#: **twice with different minted topics** so the two-fillers rule can separate the cue from the
#: topic.
#:
#: This is a lesson rather than a table, and the difference is testable: the phrasings here are
#: deliberately **not** the phrasings :data:`nyxara.njp.explaingauntlet.WHY_FORMS` examines on.
#: The gauntlet reports the taught and the untaught halves separately — so teaching more phrasings
#: cannot be mistaken for reasoning better, and the gap between the two halves is the honest size
#: of what induction bought.
LESSON: Tuple[Tuple[str, str], ...] = (
    # causes
    ("what brings about {x}?", "because"),
    ("what gives rise to {x}?", "because"),
    ("what leads to {x}?", "because"),
    ("what is responsible for {x}?", "because"),
    ("how come {x} happens?", "because"),
    ("what sets off {x}?", "because"),
    # purposes
    ("what is {x} there for?", "for"),
    ("what job does {x} do?", "for"),
    # mechanism
    ("by what means does {x} operate?", "mechanism"),
    ("what makes {x} function?", "mechanism"),
    # procedure
    ("what is the recipe for {x}?", "procedure"),
    ("walk me through {x}", "procedure"),
    # negatives: these carry no causal claim, and without them the general level swallows every
    # wh-question. `what` and `is` become contested by construction, which is correct.
    ("what is {x}?", ""),
    ("who discovered {x}?", ""),
    # These two share an open-class word with a positive above — `job` with *"what job does {x}
    # do?"* and `makes` with *"what makes {x} function?"* — so the shared cue is demonstrated with
    # two different answers and `Cue.contested` drops it. That is what a negative is *for*, and
    # the lesson would not exercise it without a pair like this: a negative that shares nothing
    # contests nothing, which is measured in `tests/njp/test_asking.py` rather than assumed.
    ("whose job is {x}?", ""),
    ("who makes {x}?", ""),
    ("where is {x}?", ""),
    ("when did {x} happen?", ""),
    ("is {x} a metal?", ""),
    ("how many {x} are there?", ""),
)


def teach(asking: Optional[Asking] = None, *, seed: int = 20260905,
          lesson: Sequence[Tuple[str, str]] = LESSON) -> Asking:
    """Show every entry of *lesson* twice, with minted topics, and return the learner.

    Minted topics rather than real ones for the reason every school in this package mints: a topic
    she might recognise lets a demonstration teach the answer instead of the form. ``vokith`` is
    not in any corpus, so the only thing a demonstration can leave behind is where the topic sat
    and which words were not it.
    """
    import random

    got = asking if asking is not None else Asking()
    rng = random.Random(seed)

    def word() -> str:
        return "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou")
                       for _ in range(rng.randint(2, 3)))

    for form, walk in lesson:
        for _ in range(2):
            topic = word()
            got.show(form.format(x=topic), topic=topic, walk=walk)
    return got


def install(asking: Optional[Asking] = None, **kwargs: Any) -> Asking:
    """Teach a learner and hand it to :mod:`nyxara.njp.explainread` as its fallback reader.

    One call, because the alternative — every caller wiring it up — is how an organ ends up being
    used by one code path and not the other. :meth:`nyxara.njp.brain.NJPBrain.learn_question_forms`
    is that one call.
    """
    from nyxara.njp import explainread

    got = teach(asking, **kwargs)
    explainread.LEARNED = got
    return got


def uninstall() -> None:
    """Put the reader back to the table alone. Used by tests that measure the difference."""
    from nyxara.njp import explainread

    explainread.LEARNED = None
