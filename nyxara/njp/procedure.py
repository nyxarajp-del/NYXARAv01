"""NYXARA · njp/procedure.py — what a task *is*, read out of the sentence that sets it (🧾).

The plan's eighth item asks for procedural knowledge, and names its parts: **Goal, Prerequisites,
Steps, Expected result, Failure, Recovery.** Nothing in this package held any of it. Every organ
so far reads *descriptions* — a passage says what photosynthesis is, a premise says what is in a
photograph — and a description is not an instruction. An instruction is addressed to somebody, it
says what they will be handed, what they are to produce, and what the allowed answers are. Read it
as description and you get what :mod:`nyxara.njp.passage` got, measured before this file existed:

    "In this task, you are given a question and a context passage. You have to answer the
     question based on the given passage."
        -> entities: ['in task', 'NYXARA']
        -> relations: []
        LOST: everything. What she is given, what she must do, and what counts as an answer.

The whole of it lost, on all four fields, on every one of the 698 definitions. Not a shortfall to
be improved — a category the readers had no representation for.

**The corpus is real and it is the largest of its kind that exists.** FLAN's NIv2 submix is
3,796,006 instances, and the instruction at the head of each is not free text: it is a task
definition written by a person for a person. Folded on the instruction, those 3.8 million
instances are **698 distinct procedures** — 698 statements of a job to be done, in the wording
somebody actually chose. That ratio is the reason this file is possible: the same procedure is
attested 5,000 times, so what varies across the 698 is the procedure and not the phrasing of one.

What is read out of one:

* **given** — the prerequisites. *"a question and a corresponding answer"* is two of them, not one.
* **goal** — what is to be produced, verb-headed, so ``action`` is the verb and the taxonomy of
  what these 698 jobs *are* falls out of the reading rather than being imposed on it.
* **outputs** — the expected result as an answer *space*: ``"Singular"`` and ``"Plural"`` and
  nothing else. A procedure that names its allowed answers is checkable; one that does not is not.
* **conditions** — failure and recovery, which in a task definition is exactly where they live:
  *"Generate 'True' if the summary matches, otherwise generate 'False'"*, *"If every integer is
  odd then an empty list should be returned."* A trigger and a consequence, kept paired.
* **caveats** — the sentences no role fired in, kept rather than dropped. *"Note that URLs in the
  text have been replaced with [Link]"* is not decoration; it is a fact about the input that the
  goal does not state.

**Nothing here is a table of instruction phrases.** ``you are given`` appears in 333 of the 698 and
``your task is to`` in 251, and both would have been trivial to type. They are not typed. The
shapes that read a role are induced from :data:`LESSONS` — fourteen definitions with their roles
marked by hand — at the two levels :mod:`nyxara.njp.passage` established: a **frame** keeps the
demonstration's own words, and a **cued** shape holes every open-class token and keeps only the
closed class, so ``you are <*> a <SLOT>`` induced from *"you are given a question"* reads *"you are
provided with an article"*, a clause with no content word in common with it. A cued shape with no
literal anchor left is refused, and a cued shape is not trusted on one witness.

The boundaries are induced too, and that is the part that is easy to get wrong. Where a ``given``
span *stops*, whether it may cross a comma, how long it may run, what separates two of them, which
characters quote an output, which words join a list of them, which word introduces a trigger and
which a consequence — every one of those is counted off the demonstrations in
:meth:`ProcedureReader._consolidate` and none is written down here.

Nothing in this module writes to the fact store, and nothing in it executes a procedure. Reading a
definition produces a :class:`Procedure`; whether she can *run* it is
:meth:`Procedure.prerequisites_met`, which answers from what she was handed and says what is
missing rather than guessing.

Pure standard library.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# One tokeniser for the package, not a second one that drifts from it. These carry V.49's
# sentence-splitting repairs -- the abbreviation rule that stopped "Dr." ending a sentence and the
# initialism rule that stopped "BC." failing to -- and a task definition is full of both
# ("i.e.", "etc.", "e.g.").
from nyxara.njp.passage import (  # noqa: WPS436 — deliberate reuse of one tokenisation
    _HOLE, _PUNCT, Tok, _bare, _find, _head_of, _sentences, _table, _toks,
    _unparenthesise, _words,
)

__all__ = [
    "Condition", "Procedure", "Lesson", "RoleShape", "ProcedureReader",
    "ROLES", "LESSONS", "taught_procedures",
]

#: The roles a shape can read. ``caveat`` is deliberately not among them: a caveat is the residue
#: of a sentence no role fired in, and inducing a shape for "the leftovers" would be inducing a
#: shape that matches everything.
ROLES: Tuple[str, ...] = ("given", "goal")

#: Tokens of context on each side of a marked span that become a shape's anchors. Three on the
#: left because ``you are given a`` needs three to separate itself from ``you are asked to``; two
#: on the right because a role span in a task definition nearly always ends at its sentence.
LEFT_CONTEXT = 3
RIGHT_CONTEXT = 2

#: How many demonstrations must independently produce a cued shape before it is trusted. One
#: demonstration showing ``given`` marks a prerequisite is a coincidence; two are a shape.
MIN_WITNESSES = 2

#: The longest gap, in tokens, that can still be a list joiner rather than an intervening clause.
#: Two, because the widest joiner any demonstration shows is ``, and`` -- and the alternative to a
#: bound here is what the first version did, which was to read the clause between two answers in
#: different sentences as though it joined them.
MAX_JOINER = 2

#: The stop recorded for a span that ran to the end of its sentence.
_END = "END"

#: How many enumerator tokens ("1)", "2)") may sit in front of an answer before the run is
#: measured. One, because that is what a numbered list puts there.
_ENUMERATOR = 1

#: How far back a bare preposition may sit from the first answer and still introduce it.
#: **Negative disables it**, and negative is what the sweep chose. Three times in this file the
#: right repair was to learn a tag where a word had been memorised -- the stop set, the joiner,
#: the split -- so the same move was tried a fourth time on the lead-in, and it is wrong here:
#: an answer space follows ``into``, ``as`` and ``from``, but so does every other prepositional
#: phrase in the language, and admitting them dropped the audited precision from 1.000 to 0.158
#: while buying no recall at all. The sweep, on the held-out twenty-one:
#:
#:     reach   -1    outputs 0.333  (p 0.750  r 0.214)   overall 0.711
#:     reach    0    outputs 0.182  (p 0.158  r 0.214)   overall 0.660
#:     reach  1-99   outputs 0.182  (p 0.158  r 0.214)   overall 0.660
#:
#: Flat from zero upward, which says a preposition sits directly in front of a short comma-list
#: nearly everywhere. Kept as a switch rather than deleted so the finding stays runnable.
LEADIN_REACH = -1

#: How many quoted values make a run an answer space without a lead-in in front of it. Three,
#: because two quoted words are as likely to be the names of the inputs -- "where 'i' and 'j' are
#: integers" -- as the answers, and every demonstrated pair has a lead-in.
MIN_RUN = 3


# --------------------------------------------------------------------------------------------- #
#  what a reading is
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """A trigger and what to do when it holds — the plan's *failure* and *recovery*, paired.

    Kept as a pair and never flattened into the goal. *"answer with the list of even numbers"* is
    not what this task does; it is what this task does **when the list is not all odd**, and a
    procedure store that forgets the difference has stored a false instruction.
    """

    trigger: str = ""
    consequence: str = ""
    sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": self.trigger, "consequence": self.consequence}


@dataclass
class Procedure:
    """One task definition, read into its parts, with the definition itself kept.

    ``text`` is not a courtesy. The plan is explicit that a source is not to be flattened into
    what was extracted from it, and a definition says things no field here has a slot for — that
    ``"W"`` and ``"M"`` in the conversations stand for *woman* and *man*, that a summary need not
    be a complete sentence. Those live in ``caveats`` when they are their own sentence and in
    ``text`` always.
    """

    name: str = ""
    task: str = ""
    given: List[str] = field(default_factory=list)
    goal: str = ""
    outputs: List[str] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    text: str = ""
    source: str = ""
    licence: str = ""
    #: Which shape read which role, so a reading can always say what produced it.
    shapes_used: Dict[str, str] = field(default_factory=dict)

    @property
    def action(self) -> str:
        """The goal's head verb — ``generate``, ``classify``, ``convert``.

        Not looked up. The goal span is verb-initial because that is how these definitions are
        written (*"generate a fact statement"*, *"classify the sentiment"*), so the first
        open-class token of the span is the action, and the taxonomy of what the 698 jobs are is
        whatever this returns across them.
        """
        for token in _toks(self.goal):
            if token.open_class:
                return token.text
        return ""

    @property
    def decides(self) -> bool:
        """It names its allowed answers, so an attempt at it can be checked against them."""
        return len(self.outputs) > 1

    def prerequisites_met(self, have: Iterable[str]) -> Tuple[bool, List[str]]:
        """Can this be run with what she has? Matched on heads, and it says what is missing.

        The heads are compared rather than the phrases because a caller has *"a sentence"* and the
        definition asks for *"a sentence in the English language"*; refusing that is refusing on a
        modifier. Missing prerequisites are returned rather than swallowed, because "no" without
        "what is absent" is not usable by anything.
        """
        held = {_head_of(item) for item in have} - {""}
        missing = [need for need in self.given if _head_of(need) and _head_of(need) not in held]
        return (not missing), missing

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name, "given": list(self.given), "goal": self.goal,
            "action": self.action, "outputs": list(self.outputs),
            "conditions": [c.to_dict() for c in self.conditions],
        }
        if self.caveats:
            out["caveats"] = list(self.caveats)
        if self.task:
            out["task"] = self.task
        if self.source or self.licence:
            out["provenance"] = {"task": self.task, "source": self.source,
                                 "licence": self.licence}
        out["text"] = self.text
        return out

    def render(self) -> str:
        parts = [f"{self.name or self.task}"]
        parts.append("  given   : " + ("; ".join(self.given) if self.given else "—"))
        parts.append(f"  goal    : {self.goal or '—'}   [{self.action or '—'}]")
        parts.append("  outputs : " + (" | ".join(self.outputs) if self.outputs else "—"))
        for cond in self.conditions:
            parts.append(f"  when {cond.trigger} -> {cond.consequence}")
        for note in self.caveats:
            parts.append(f"  note    : {note}")
        return "\n".join(parts)


@dataclass
class Lesson:
    """A definition with its roles marked by hand. The only hand-written thing in the file.

    The marks are **exact substrings of the text**, not paraphrases, so a mark that no longer
    appears in the text it belongs to is a broken lesson rather than a silently unlearned one —
    :meth:`ProcedureReader.learn` counts what it failed to locate and
    :meth:`ProcedureReader.unlearnt` reports it. V.48 lost a whole lesson to exactly this,
    silently, and the ablation that should have caught it moved no score at all.
    """

    task: str = ""
    text: str = ""
    given: Tuple[str, ...] = ()
    goal: str = ""
    outputs: Tuple[str, ...] = ()
    conditions: Tuple[Tuple[str, str], ...] = ()

    def spans(self) -> List[Tuple[str, str]]:
        """The marks a shape is induced from — the prerequisites and the goal, and not the answers.

        An answer is not found by a shape. It is found by what quotes it and what joins a list of
        it, which :meth:`ProcedureReader._learn_outputs` counts off the raw characters, so
        inducing token shapes for the answers built patterns nothing ever consulted. Worse, it
        could not even build them for a single-letter answer: ``"A"`` is a determiner, ``_bare``
        strips it, and the mark went down as unlocatable in a lesson that had it in plain sight.
        """
        out: List[Tuple[str, str]] = [("given", g) for g in self.given]
        if self.goal:
            out.append(("goal", self.goal))
        return out


@dataclass
class RoleShape:
    """A role-reading pattern: anchors on each side of a hole, and the role the hole fills."""

    role: str = ""
    left: Tuple[str, ...] = ()
    right: Tuple[str, ...] = ()
    level: str = "frame"
    witnesses: Set[str] = field(default_factory=set)
    #: The pattern still holds an open-class word that anchors **this role and no other** across
    #: every demonstration. Set by :meth:`ProcedureReader._consolidate`, never by hand — this is
    #: how ``given`` and ``asked`` come to be role words without either being typed.
    role_word: bool = False
    #: Definitions this shape has read that no demonstration contained.
    unseen: int = 0

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], Tuple[str, ...], str]:
        return (self.role, self.left, self.right, self.level)

    @property
    def anchors(self) -> int:
        return sum(1 for w in self.left + self.right if w != _HOLE)

    @property
    def trusted(self) -> bool:
        return (self.level == "frame" or self.role_word
                or len(self.witnesses) >= MIN_WITNESSES)

    @property
    def generalises(self) -> bool:
        return self.unseen > 0

    def render(self) -> str:
        left = " ".join(self.left)
        right = " ".join(self.right)
        return f"{left} <{self.role}> {right}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "pattern": self.render(), "level": self.level,
                "witnesses": sorted(self.witnesses), "unseen": self.unseen,
                "role_word": self.role_word, "trusted": self.trusted,
                "generalises": self.generalises}


# --------------------------------------------------------------------------------------------- #
#  small helpers over the shared tokenisation
# --------------------------------------------------------------------------------------------- #
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}

#: The longest thing a pair of quotes may hold and still be an answer rather than a quotation.
MAX_QUOTED = 40


def _quoted_spans(raw: str, quotes: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Every quoted value, paired properly, with word-internal apostrophes left alone.

    A regular expression cannot do this. ``(["\'])(.+?)(["\'])`` opens on the apostrophe of
    *"the reviewer's sentiment into: ..."* and closes on the first real quote after it, and every
    pair from there on is offset by one — which is why a definition that names five allowed
    answers in plain double quotes was read as naming none. An apostrophe with a letter on both
    sides is inside a word and is not a delimiter; everything else pairs off in order.
    """
    out: List[Tuple[int, int, str]] = []
    for opener in sorted(set(quotes)):
        closer = _QUOTE_PAIRS.get(opener, opener)
        marks: List[int] = []
        for index, char in enumerate(raw):
            if char not in (opener, closer):
                continue
            before = raw[index - 1] if index else " "
            after = raw[index + 1] if index + 1 < len(raw) else " "
            if opener == closer and before.isalnum() and after.isalnum():
                continue                       # reviewer's, Amazon's, don't
            marks.append(index)
        for start, end in zip(marks[::2], marks[1::2]):
            value = raw[start + 1:end]
            if not value.strip() or len(value) > MAX_QUOTED or "\n" in value:
                continue
            out.append((start, end + 1, " ".join(value.split()).strip(" .,;:")))
    out.sort()
    return out


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" .,;:")


def Tag_PREP() -> str:
    _c, Tag = _table()
    return Tag.PREP


def Tag_SUB() -> str:
    _c, Tag = _table()
    return Tag.SUB


def Tag_AUX() -> str:
    _c, Tag = _table()
    return Tag.AUX


def _occurrences(raw: str, value: str) -> List[int]:
    out: List[int] = []
    at = raw.find(value)
    while at >= 0:
        out.append(at)
        at = raw.find(value, at + 1)
    return out


def _quoted_at(raw: str, value: str, quotes: Set[str]) -> int:
    """Where this answer sits, preferring a quoted occurrence over an earlier unquoted one."""
    found = _occurrences(raw, value)
    for at in found:
        before = raw[at - 1] if at else ""
        after = raw[at + len(value)] if at + len(value) < len(raw) else ""
        if before in quotes and _QUOTE_PAIRS.get(before) == after:
            return at
    return found[0] if found else -1


def _holds(sentence: str, span: str) -> bool:
    """Whether a sentence contains a span as whole words, in either direction."""
    for outer, inner in ((sentence, span), (span, sentence)):
        if re.search(rf"(?<!\w){re.escape(inner)}(?!\w)", outer):
            return True
    return False


def _ends_phrase(raw: str, at: int) -> bool:
    """Whether what follows a quoted value lets it be an answer rather than a modifier.

    ``'Yes',`` and ``'Similar' if`` are answers; ``'first' key`` and ``'second' key`` are the
    names of two fields of the input, and reading them as the allowed answers to a sorting task
    is what this rules out. An answer is the last thing in its phrase, so the token after it is
    punctuation, closed class, or nothing at all -- never another content word.
    """
    rest = _toks(raw[at:at + 40])
    return not rest or not rest[0].open_class


def _match_at(pattern: Sequence[str], words: Sequence[str], at: int) -> bool:
    """Does ``pattern`` sit at ``words[at:]``? ``_HOLE`` matches exactly one token."""
    if at < 0 or at + len(pattern) > len(words):
        return False
    for offset, want in enumerate(pattern):
        if want != _HOLE and words[at + offset] != want:
            return False
    return True


def _span_text(raw: str, toks: Sequence[Tok], i: int, j: int) -> str:
    """The surface characters the span covers, not a detokenised reconstruction of them."""
    if i >= j or j > len(toks):
        return ""
    return _norm(raw[toks[i].start:toks[j - 1].end])


def _clauses(raw: str) -> List[Tuple[str, List[Tok], List[str]]]:
    """The text as sentences, each tokenised on its own with offsets into the whole.

    A role does not run past the end of its sentence, and reading the definition as one token
    stream is what hid that. ``_bare`` drops determiners, so *"... of the first one. The sentences
    are separated"* put ``sentences`` immediately after the goal span and the reader duly learned
    ``sentences`` as a word a goal may stop before — one of ten such words memorised off the
    demonstrations, which between them let a goal be read in 39% of the corpus and no more. Split here,
    and the thing every demonstration was actually showing is a single stop: the sentence ended.
    """
    out: List[Tuple[str, List[Tok], List[str]]] = []
    cursor = 0
    for sentence in _sentences(raw):
        at = raw.find(sentence, cursor)
        if at < 0:
            at = cursor
        cursor = at + len(sentence)
        toks = [Tok(t.text, t.tag, t.start + at, t.end + at) for t in _bare(_toks(sentence))]
        out.append((raw, toks, _words(toks)))
    return out


# --------------------------------------------------------------------------------------------- #
#  the reader
# --------------------------------------------------------------------------------------------- #
class ProcedureReader:
    """Reads a task definition into a :class:`Procedure`, from shapes it was shown.

    Everything the reader knows arrives through :meth:`learn`. Constructed and never taught it
    returns a :class:`Procedure` with the text in it and every field empty, which is the honest
    floor and is what :func:`nyxara.njp.procedureschool.examine` measures against.
    """

    def __init__(self) -> None:
        self.shapes: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...], str], RoleShape] = {}
        self._taught: Set[str] = set()
        self._missed: List[Tuple[str, str, str]] = []
        # Everything below is counted off the demonstrations in _consolidate. Empty until then.
        self._stops: Dict[str, Set[str]] = {}
        self._opens: Dict[str, Set[str]] = {}
        self._max_span: Dict[str, int] = {}
        self._cross_comma: Dict[str, bool] = {}
        self._splits: Set[str] = set()
        self._split_det: List[bool] = []
        self._quotes: Set[str] = set()
        self._joiners: Set[str] = set()
        self._leadins: Set[str] = set()
        self._leadwords: Set[str] = set()
        self._triggers: Set[str] = set()
        self._alone: Set[str] = set()
        self._lessons: List[Lesson] = []
        # Raw tallies the consolidation reads. Kept so a caller can see what was counted.
        self._seen_role: Dict[str, Counter] = {r: Counter() for r in ROLES}
        self._seen_any: Counter = Counter()

    # -- teaching ------------------------------------------------------------------------- #
    def learn(self, lesson: Lesson) -> "ProcedureReader":
        """Take one marked definition apart into shapes and boundary evidence."""
        self._lessons.append(lesson)
        raw = lesson.text
        self._taught.add(lesson.task or raw[:40])
        clauses = _clauses(raw)

        marked: List[Tuple[str, int, int, int]] = []
        for role, span in lesson.spans():
            want = _words(_bare(_toks(span)))
            found = None
            for index, (_r, toks, words) in enumerate(clauses):
                at = _find(words, want)
                if at is not None:
                    found = (index, at[0], at[1])
                    break
            if found is None:
                # A mark the text does not contain teaches nothing and must not do so quietly.
                self._missed.append((lesson.task, role, span))
                continue
            index, i, j = found
            marked.append((role, index, i, j))
            for shape in self._induce(role, clauses[index][1], i, j):
                self._file(shape, lesson.task or raw[:40])

        self._learn_boundaries(lesson, clauses, marked)
        self._learn_outputs(lesson)
        self._learn_conditions(lesson)
        self._consolidate()
        return self

    def _induce(self, role: str, toks: Sequence[Tok], i: int, j: int) -> List[RoleShape]:
        n = len(toks)
        lo, hi = max(0, i - LEFT_CONTEXT), min(n, j + RIGHT_CONTEXT)
        left = tuple(toks[k].text for k in range(lo, i))
        right = tuple(toks[k].text for k in range(j, hi))
        out: List[RoleShape] = []
        if left or right:
            out.append(RoleShape(role=role, left=left, right=right, level="frame"))
        cued_left = tuple(_HOLE if toks[k].open_class else toks[k].text for k in range(lo, i))
        cued_right = tuple(_HOLE if toks[k].open_class else toks[k].text for k in range(j, hi))
        cued = RoleShape(role=role, left=cued_left, right=cued_right, level="cued")
        # All holes matches every clause in the language. Refusing it is the difference between a
        # generalisation and a machine that reads anything as anything.
        if cued.anchors:
            out.append(cued)
        return out

    def _file(self, shape: RoleShape, witness: str) -> RoleShape:
        held = self.shapes.get(shape.key)
        if held is None:
            shape.witnesses.add(witness)
            self.shapes[shape.key] = shape
            return shape
        held.witnesses.add(witness)
        return held

    def _learn_boundaries(self, lesson: Lesson,
                          clauses: Sequence[Tuple[str, List[Tok], List[str]]],
                          marked: Sequence[Tuple[str, int, int, int]]) -> None:
        """Where a role span stopped, how long it ran, and what separated two of the same role.

        The stop is recorded as a **tag**, not as the word that happened to follow. ``END`` is a
        span that ran to the end of its sentence, and it is what all but one of the goal marks
        show; the rest are the closed-class tag of the next token. Recording the word instead is
        what made the stop set a memory of the demonstrations' neighbours rather than a rule about boundaries.
        """
        _, Tag = _table()
        for role, index, i, j in marked:
            toks, words = clauses[index][1], clauses[index][2]
            self._max_span[role] = max(self._max_span.get(role, 0), j - i)
            self._stops.setdefault(role, set()).add(
                _END if j >= len(words) else toks[j].tag)
            # The left edge, which nothing counted until a reading came back with ``you need to``
            # as a prerequisite and ``with an article of the legal acts`` as another. Every
            # demonstrated prerequisite opens on an open-class word once determiners are stripped
            # -- "a question", "two statements", "reviews written about..." -- and none opens on a
            # pronoun or a preposition.
            self._opens.setdefault(role, set()).add(toks[i].tag)
            if any(toks[k].tag == _PUNCT for k in range(i, j)):
                self._cross_comma[role] = True
            elif role not in self._cross_comma:
                self._cross_comma[role] = False
        by_role: Dict[str, List[Tuple[int, int, int]]] = {}
        for role, index, i, j in marked:
            by_role.setdefault(role, []).append((index, i, j))
        for role, spans in by_role.items():
            for (ia, _a, end), (ib, start, _b) in zip(sorted(spans), sorted(spans)[1:]):
                if ia != ib:
                    continue
                # Two marks of one role with only closed class between them were one coordinated
                # phrase in the text: "a question AND a corresponding answer".
                toks = clauses[ia][1]
                between = [toks[k] for k in range(end, start)]
                if between and all(t.tag in (_PUNCT, Tag.CONJ, Tag.DET) for t in between):
                    self._splits.update(t.text for t in between if t.tag != Tag.DET)
                    # Whether what followed the conjunction opened with a determiner. All four
                    # coordinations any demonstration shows do -- "a question AND *a*
                    # corresponding answer", "reviews ... AND *a* summary of that review" -- and
                    # the distinction is the whole of why "a sentence in the English and Hindi
                    # language" is one prerequisite rather than two: "Hindi language" opens with
                    # no determiner, so the conjunction is inside a phrase, not between two.
                    # In the *raw* text, between the conjunction and the piece. `toks` is the
                    # bare stream and the determiner is exactly what it has removed, so asking
                    # this token's tag could only ever answer "not a determiner" -- which it did,
                    # for every coordination, leaving the rule permanently off and "a sentence in
                    # the English and Hindi language" split into two prerequisites.
                    gap = clauses[ia][0][toks[end - 1].end:toks[start].start]
                    lead = _toks(gap)
                    self._split_det.append(bool(lead) and lead[-1].tag == Tag.DET)

    def _learn_outputs(self, lesson: Lesson) -> None:
        """Which characters quoted a marked answer, what joined a list of them, what led in."""
        raw = lesson.text
        for value in lesson.outputs:
            # The *quoted* occurrence, not the first one. "cause" appears twice in lesson 3 --
            # once inside the goal ("whether ... is the cause or effect") and once as the answer
            # ("the word 'cause'"). Taking the first meant the single quote was never seen to
            # quote anything, and every definition that names its answers in single quotes was
            # read as naming none: five of the sixteen audited output failures, from one line.
            for at in _occurrences(raw, value):
                before = raw[at - 1] if at else ""
                after = raw[at + len(value)] if at + len(value) < len(raw) else ""
                if before in _QUOTE_PAIRS and _QUOTE_PAIRS[before] == after:
                    self._quotes.add(before)
                    break
        if len(lesson.outputs) > 1:
            positions = sorted((_quoted_at(raw, v, self._quotes), v) for v in lesson.outputs
                               if _quoted_at(raw, v, self._quotes) >= 0)
            adjacent = False
            for (at, value), (nxt, _v) in zip(positions, positions[1:]):
                gap = _words(_toks(raw[at + len(value):nxt]))
                # A joiner is what stands **between two members of one list**. Two answers with a
                # whole clause between them are not a list, and counting that clause taught
                # ``generate``, ``review``, ``summary`` and ``match`` as joiners off a single
                # lesson -- after which any two quoted words anywhere in a definition joined into
                # an answer space. The gap that makes a list is short by construction.
                if len(gap) > MAX_JOINER:
                    continue
                adjacent = True
                # A tag, not a word -- the same mistake as the stop set, in the same place. Every
                # demonstration happens to join its answers with ``or``, so ``or`` is what a
                # word-level joiner learns, and an answer space written ``"A", "B", "C", "D", and
                # "E"`` is then read as naming nothing. What the demonstrations show is that a
                # joiner is a conjunction or a comma; ``and`` and ``,`` are both of those and
                # neither is written here.
                self._joiners.update(t.tag for t in _toks(raw[at + len(value):nxt]) if t.text)
            if positions and adjacent:
                # A tag again, and for the third time in this file the same mistake was there to
                # be made: the word before an answer space is ``classes`` in one demonstration
                # and ``word`` in another, and learning those two literals is why *"into two
                # categories: 1) positive, and 2) negative"* named no answers. What the
                # demonstrations agree on is that the space follows a **preposition** -- ``into``
                # two classes, ``as`` "Singular" or "Plural", ``as`` P or N -- and ``into``,
                # ``from`` and ``out of`` are all prepositions nobody wrote down.
                lead = [t for t in _toks(raw[:positions[0][0]]) if t.text]
                if lead:
                    self._leadwords.add(lead[-1].text)
                for token in reversed(lead):
                    if token.tag in (Tag_PREP(), Tag_SUB(), Tag_AUX()):
                        self._leadins.add(token.tag)
                        break

    def _learn_conditions(self, lesson: Lesson) -> None:
        """Which word introduced a trigger, and which trigger stood on its own."""
        raw_low = lesson.text.lower()
        for trigger, _consequence in lesson.conditions:
            low = trigger.lower()
            at = raw_low.find(low)
            if at < 0:
                continue
            before = _words(_bare(_toks(raw_low[:at])))
            if not before:
                continue
            if _norm(low) == _norm(before[-1]) or len(_words(_toks(low))) == 1:
                # "otherwise" is the whole trigger: it names no condition, it names the
                # complement of the last one.
                self._alone.add(_norm(low))
            else:
                self._triggers.add(before[-1])

    def _consolidate(self) -> None:
        """Which anchor words belong to one role and no other. Counted, never declared."""
        self._seen_role = {r: Counter() for r in ROLES}
        self._seen_any = Counter()
        for shape in self.shapes.values():
            for word in set(shape.left + shape.right):
                if word == _HOLE:
                    continue
                self._seen_role[shape.role][word] += 1
                self._seen_any[word] += 1
        _, Tag = _table()
        closed, _t = _table()
        for shape in self.shapes.values():
            if shape.level != "cued":
                continue
            shape.role_word = any(
                word != _HOLE and word not in closed
                and self._seen_role[shape.role][word] == self._seen_any[word]
                for word in shape.left + shape.right)

    def unlearnt(self) -> List[Tuple[str, str, str]]:
        """Marks that were not found in their own lesson. Empty is the only acceptable value."""
        return list(self._missed)

    def taught(self) -> Tuple[str, ...]:
        return tuple(sorted(self._taught))

    # -- reading -------------------------------------------------------------------------- #
    def read(self, text: str, name: str = "", *, task: str = "", source: str = "",
             licence: str = "") -> Procedure:
        raw = str(text or "")
        out = Procedure(name=name or task, task=task, text=raw, source=source, licence=licence)
        clauses = _clauses(raw)
        if not any(words for _r, _t, words in clauses):
            return out

        # A span is claimed by character offset rather than by token index, because the indices
        # restart at every sentence and two sentences' index 3 are not the same place.
        taken: List[Tuple[int, int]] = []
        goal = self._best("goal", clauses, taken)
        if goal:
            out.goal, shape = goal[0], goal[1]
            out.shapes_used["goal"] = shape
            taken.append(goal[2])
        for span, shape, where in self._all("given", clauses, taken):
            for piece in self._split(span):
                if piece and piece not in out.given:
                    out.given.append(piece)
            out.shapes_used.setdefault("given", shape)
            taken.append(where)

        out.conditions = self._conditions(raw)
        # A parenthesised list is an aside, and in a task definition it is nearly always an
        # example of the *input* -- "(letters 'a', 'e', 'i', 'o', 'u')", "like ['1', '12', 'l']",
        # "(as expressed by the user)". Read as an answer space it produced five vowels as the
        # allowed answers to a counting task. Blanked rather than removed, so every offset still
        # points where it pointed and a span is still the characters the definition contains.
        masked = _unparenthesise(raw)
        out.outputs = (self._outputs(masked, _clauses(masked))
                       or self._decided(masked, out.conditions))
        out.caveats = self._caveats(raw, out)
        self._credit(out)
        return out

    def _candidates(self, role: str, clauses: Sequence[Tuple[str, List[Tok], List[str]]],
                    taken: Sequence[Tuple[int, int]]) -> List[Tuple[float, str, str,
                                                                   Tuple[int, int]]]:
        """Every span a trusted shape of this role reads, scored, best first."""
        _, Tag = _table()
        stops = self._stops.get(role, set())
        limit = self._max_span.get(role, 0)
        cross = self._cross_comma.get(role, False)
        out: List[Tuple[float, str, str, Tuple[int, int]]] = []
        if not limit:
            return out
        shapes = [s for s in self.shapes.values() if s.role == role and s.trusted]
        for raw, toks, words in clauses:
            for shape in shapes:
                for i in range(len(words) + 1):
                    if not _match_at(shape.left, words, i - len(shape.left)):
                        continue
                    for j in range(i + 1, len(words) + 1):
                        if not _match_at(shape.right, words, j):
                            continue
                        stop = _END if j >= len(words) else toks[j].tag
                        if stops and stop not in stops:
                            continue
                        # The cap and the comma rule are about where a span may *stop*. A span
                        # that stops at the end of its sentence has stopped where every
                        # demonstration stopped, and is bounded by the sentence rather than by a
                        # length counted off the demonstrations -- which is why a goal of fourteen
                        # tokens was the longest thing this could read, and six of the audited
                        # definitions state theirs in more.
                        if stop != _END:
                            if j - i > limit:
                                continue
                            if not cross and any(toks[k].tag == _PUNCT for k in range(i, j)):
                                continue
                        # A reading that opens on something no demonstration opened on has
                        # swallowed the word in front of the role rather than the role. Trimmed
                        # rather than refused: "with an article of the legal acts" is the right
                        # prerequisite with a preposition on the front of it.
                        opens = self._opens.get(role, set())
                        while opens and i < j and toks[i].tag not in opens:
                            i += 1
                        if i >= j:
                            continue
                        span = _span_text(raw, toks, i, j)
                        if not span:
                            continue
                        where = (toks[i].start, toks[j - 1].end)
                        if any(where[0] < b and a < where[1] for a, b in taken):
                            continue
                        # A frame is worth more than a cued shape because it matched the wording
                        # as well as the structure; more anchors are worth more than fewer; and a
                        # longer span is preferred at equal anchoring, because a role that stops
                        # early has dropped a modifier the definition put there on purpose.
                        score = (2.0 if shape.level == "frame" else 1.0) + 0.1 * shape.anchors
                        out.append((score + 0.001 * (j - i), span, shape.render(), where))
        out.sort(key=lambda row: (-row[0], row[3][0]))
        return out

    def _best(self, role: str, clauses: Sequence[Tuple[str, List[Tok], List[str]]],
              taken: Sequence[Tuple[int, int]]) -> Optional[Tuple[str, str, Tuple[int, int]]]:
        rows = self._candidates(role, clauses, taken)
        if not rows:
            return None
        _score, span, shape, where = rows[0]
        return span, shape, where

    def _all(self, role: str, clauses: Sequence[Tuple[str, List[Tok], List[str]]],
             taken: Sequence[Tuple[int, int]]) -> List[Tuple[str, str, Tuple[int, int]]]:
        """Every non-overlapping reading of this role at the best level available.

        A task has several inputs, so several readings are kept — but they **compete rather than
        accumulate**. Where a frame has read a prerequisite, the abstracted shapes are not also
        allowed to add one: a definition states its inputs in one construction, and letting both
        levels contribute at once is what put ``associated`` beside ``list`` and ``based`` beside
        ``review``. Where no frame fires at all, the cued shapes are the only reading there is and
        they are taken.
        """
        rows = self._candidates(role, clauses, taken)
        if not rows:
            return []
        best = max(row[0] for row in rows)
        floor = 2.0 if best >= 2.0 else 0.0
        out: List[Tuple[str, str, Tuple[int, int]]] = []
        used: List[Tuple[int, int]] = list(taken)
        for score, span, shape, where in rows:
            if score < floor:
                continue
            if any(where[0] < b and a < where[1] for a, b in used):
                continue
            used.append(where)
            out.append((span, shape, where))
        return out

    def _split(self, span: str) -> List[str]:
        """One coordinated phrase into the several prerequisites it names — when it names several.

        Refused unless every piece after a split word opens the way the demonstrated ones did.
        Splitting unconditionally turned *"a sentence in the English and Hindi language"* into two
        prerequisites headed ``English`` and ``language``, neither of which the definition asks
        for.
        """
        if not self._splits:
            return [_norm(span)]
        _, Tag = _table()
        want_det = bool(self._split_det) and all(self._split_det)
        parts = [span]
        for word in sorted(self._splits, key=len, reverse=True):
            nxt: List[str] = []
            pattern = re.compile(rf"\s*(?:,\s*)?\b{re.escape(word)}\b\s*"
                                 if word.isalpha() else rf"\s*{re.escape(word)}\s*")
            for part in parts:
                pieces = pattern.split(part)
                if len(pieces) > 1 and want_det:
                    heads = [_toks(piece)[:1] for piece in pieces[1:]]
                    if not all(h and h[0].tag == Tag.DET for h in heads):
                        nxt.append(part)
                        continue
                nxt.extend(pieces)
            parts = nxt
        return [_norm(p) for p in parts if _norm(p)]

    def _outputs(self, raw: str,
                 clauses: Sequence[Tuple[str, List[Tok], List[str]]]) -> List[str]:
        """The answer space, from what quoted an answer and what joined a list of them.

        Two readings, and both were shown. A quoted run — ``"Singular" or "Plural"`` — and an
        unquoted alternation after a lead-in — ``into two classes: positive or negative``. Neither
        the quote characters nor the joiners nor the lead-ins are written here; all three were
        counted off the demonstrations.
        """
        if not self._quotes and not self._leadins:
            return []
        quoted = [(start, end, value) for start, end, value in _quoted_spans(raw, self._quotes)
                  if value]

        runs = self._runs(raw, quoted)
        if runs:
            return runs
        for _r, toks, words in clauses:
            found = self._alternation(raw, toks, words)
            if found:
                return found
        return []

    def _runs(self, raw: str, quoted: Sequence[Tuple[int, int, str]]) -> List[str]:
        """The longest run of quoted values separated by nothing but a learned joiner.

        A run of three or more is an answer space on its own evidence; two quoted words are not,
        because *"where 'i' and 'j' are integers"* is two quoted words and they are the names of
        the inputs. A pair is admitted only where a demonstrated lead-in stands in front of it —
        *"Label the instances **as** \"Singular\" or \"Plural\""*.
        """
        best: List[Tuple[int, List[str]]] = []
        run: List[str] = []
        opened = -1
        last_end = -1
        for start, end, value in quoted:
            if run and last_end >= 0:
                gap = [t.tag for t in _toks(raw[last_end:start])]
                if gap and not set(gap) <= self._joiners:
                    best.append((opened, list(run)))
                    run = []
            if not run:
                opened = start
            run.append(value)
            last_end = end
        if run:
            best.append((opened, list(run)))
        best.sort(key=lambda row: -len(row[1]))
        for opened, values in best:
            if len(values) < 2:
                continue
            if len(values) >= MIN_RUN or self._led(raw, opened):
                return values
        return []

    def _led(self, raw: str, at: int) -> bool:
        """Is one of the demonstrated lead-in words the last word before this run?"""
        before = [t for t in _toks(raw[:at]) if t.text]
        return bool(before) and before[-1].text in self._leadwords

    def _alternation(self, raw: str, toks: Sequence[Tok], words: Sequence[str]) -> List[str]:
        """An unquoted answer space: a chain of short values separated by learned joiners.

        Found from the **separators outward** rather than from a lead-in forward, because the
        first member of a list is the only one with no separator in front of it and is therefore
        the only one whose left edge has to be worked out. It is worked out from the others: every
        later member is bounded by separators, so the first is allowed at most as many tokens as
        the longest of them. That is what makes *"into three categories (Regulation, Decision and
        Directive)"* read ``Regulation`` and not ``three categories (Regulation``.

        Three things end a value: a separator, a preposition or subordinator (which is where
        *"2) negative based **on** its content"* stops), and the end of the sentence. A value that
        is nothing but digits is an enumerator rather than an answer, and is dropped -- the list
        is ``positive, negative``, not ``1, positive, 2, negative``.
        """
        _closed, Tag = _table()
        if not self._leadins or not self._joiners:
            return []
        limit = max(1, self._max_span.get("output", 1))
        breaks = {Tag.PREP, Tag.SUB, Tag.AUX, Tag.WH, Tag.MODAL}
        seps = [k for k, tok in enumerate(toks) if tok.tag in self._joiners]
        merged: List[List[int]] = []
        for k in seps:
            if merged and k == merged[-1][-1] + 1:
                merged[-1].append(k)
            else:
                merged.append([k])
        best: List[str] = []
        for start in range(len(merged)):
            values: List[Tuple[int, int]] = []
            for index in range(start, len(merged)):
                lo = merged[index][-1] + 1
                hi = merged[index + 1][0] if index + 1 < len(merged) else len(toks)
                cut = lo
                while cut < hi and toks[cut].tag not in breaks:
                    cut += 1
                if cut <= lo:
                    break
                values.append((lo, cut))
                if cut < hi:                      # the chain ended at a preposition, not a comma
                    break
            values = [(a, b) for a, b in values if b - a <= limit + _ENUMERATOR]
            if not values:
                continue
            widest = max(b - a for a, b in values)
            head = merged[start][0]
            first_lo = head
            while (first_lo > 0 and head - first_lo < widest
                   and toks[first_lo - 1].tag not in breaks
                   and toks[first_lo - 1].tag not in self._joiners):
                first_lo -= 1
            if first_lo >= head:
                continue
            chain = [(first_lo, head)] + values
            # The lead-in: scan back from the first value for a preposition, giving up at a
            # separator or a clause boundary rather than searching the whole sentence.
            at = first_lo - 1
            lead = False
            steps = 0
            while at >= 0:
                if toks[at].tag in self._joiners or toks[at].tag in (Tag.SUB, Tag.AUX, Tag.WH):
                    break
                if toks[at].text in self._leadwords:
                    lead = True
                    break
                if toks[at].tag in self._leadins and steps <= LEADIN_REACH:
                    lead = True
                    break
                steps += 1
                at -= 1
            if not lead:
                continue
            out: List[str] = []
            for lo, hi in chain:
                while lo < hi and toks[lo].text.isdigit():
                    lo += 1               # "1) positive" is one answer with an enumerator on it
                if lo >= hi or hi - lo > limit:
                    out = []
                    break
                value = _span_text(raw, toks, lo, hi)
                if value:
                    out.append(value)
            if len(out) > len(best) and len(out) > 1:
                best = out
        return best

    def _conditions(self, raw: str) -> List[Condition]:
        """Trigger and consequence, on the words the demonstrations showed introducing each."""
        if not self._triggers and not self._alone:
            return []
        out: List[Condition] = []
        for sentence in _sentences(raw):
            low = sentence.lower()
            for word in sorted(self._alone):
                at = low.find(word)
                if at < 0:
                    continue
                rest = _norm(sentence[at + len(word):])
                if rest:
                    out.append(Condition(trigger=word, consequence=rest, sentence=sentence))
                break
            for word in sorted(self._triggers):
                match = re.search(rf"\b{re.escape(word)}\b", low)
                if not match:
                    continue
                before = _norm(sentence[:match.start()])
                after = _norm(sentence[match.end():])
                if not after:
                    continue
                # "X if Y" states the consequence first and "if Y then X" states it second. The
                # word that separates them was counted too; where it is absent the clause that
                # came before the trigger is the consequence.
                head = re.split(r"\bthen\b", after, maxsplit=1)[0]
                if re.search(r"\bthen\b", after):
                    head, tail = re.split(r"\bthen\b", after, maxsplit=1)
                    out.append(Condition(trigger=_norm(head), consequence=_norm(tail),
                                         sentence=sentence))
                elif before:
                    out.append(Condition(trigger=_norm(head), consequence=before,
                                         sentence=sentence))
                break
        return out

    def _decided(self, raw: str, conditions: Sequence[Condition]) -> List[str]:
        """The answer space of a definition that states it as a choice rather than as a list.

        *"generate label 'Yes' if the translation is correct, otherwise generate label 'No'"* names
        two allowed answers as surely as *"'Singular' or 'Plural'"* does; it just names them one
        per branch. This is the plan's failure-and-recovery layer paying for itself — the branches
        were read for their own sake, and the expected result falls out of them. Only quoted
        values count, so a number inside a condition's *reason* ("divisible by 400") cannot be
        mistaken for an answer.
        """
        if not conditions or not self._quotes:
            return []
        seen: List[str] = []
        for sentence in {c.sentence for c in conditions if c.sentence}:
            at = raw.find(sentence)
            said = _unparenthesise(sentence) if at < 0 else raw[at:at + len(sentence)]
            for _start, end, value in _quoted_spans(said, self._quotes):
                if value and _ends_phrase(said, end) and value not in seen:
                    seen.append(value)
        return seen if len(seen) > 1 else []

    def _caveats(self, raw: str, out: Procedure) -> List[str]:
        """Sentences no role and no condition claimed. Kept, because they are about the input."""
        claimed = [out.goal] + list(out.given) + [c.sentence for c in out.conditions]
        claimed = [_norm(c).lower() for c in claimed if _norm(c)]
        notes: List[str] = []
        for sentence in _sentences(raw):
            low = _norm(sentence).lower()
            # Word boundaries, not bare containment. A prerequisite read as ``answer`` matched
            # *"your question should be **answer**able"* as a substring, and the note that says
            # what the answer has to satisfy was dropped as already claimed.
            if not low or any(c and _holds(low, c) for c in claimed):
                continue
            notes.append(_norm(sentence))
        return notes

    def _credit(self, out: Procedure) -> None:
        """A shape that read a definition no demonstration contained has generalised."""
        if out.text in {lesson.text for lesson in self._lessons}:
            return
        rendered = set(out.shapes_used.values())
        for shape in self.shapes.values():
            if shape.render() in rendered:
                shape.unseen += 1

    # -- what she worked out -------------------------------------------------------------- #
    def learned(self) -> Dict[str, Any]:
        """Everything counted off the demonstrations, so it can be read rather than trusted."""
        return {
            "shapes": len(self.shapes),
            "trusted": sum(1 for s in self.shapes.values() if s.trusted),
            "generalising": sum(1 for s in self.shapes.values() if s.generalises),
            "role_words": sorted({w for s in self.shapes.values() if s.role_word
                                  for w in s.left + s.right if w != _HOLE}),
            "stops": {r: sorted(v) for r, v in sorted(self._stops.items())},
            "opens": {r: sorted(v) for r, v in sorted(self._opens.items())},
            "max_span": dict(sorted(self._max_span.items())),
            "cross_comma": dict(sorted(self._cross_comma.items())),
            "splits": sorted(self._splits),
            "quotes": sorted(self._quotes),
            "joiners": sorted(self._joiners),
            "leadins": sorted(self._leadins),
            "leadwords": sorted(self._leadwords),
            "triggers": sorted(self._triggers),
            "standalone_triggers": sorted(self._alone),
        }


# --------------------------------------------------------------------------------------------- #
#  the demonstrations
# --------------------------------------------------------------------------------------------- #
#: Fourteen task definitions with their roles marked, chosen to span the surface forms the corpus
#: actually uses rather than to be easy: ``you are given`` (333 of the 698), ``you will be given``
#: (71), ``you're given`` (100), a fronted ``given a sentence`` with no ``you`` at all (25),
#: ``we ask you to`` and ``you would be asked to`` (22), and the four modals — ``must`` (59),
#: ``need to`` (143), ``should`` (76), ``are expected to`` (39). Two carry an ``if``/``otherwise``
#: pair (57), three name a quoted answer space (96), one names an unquoted one.
#:
#: Every text is verbatim from ``nyxara/njp/data/flan_instruction.jsonl.gz``. None is written for
#: the reader's convenience, which is why several of them are awkward — lesson 12 says *"you would
#: be asked to"* and lesson 8 puts its only input inside the goal.
LESSONS: Tuple[Lesson, ...] = (
    Lesson(
        task="task1401_obqa_sentence_generation",
        text=("In this task, you are given a question and a corresponding answer. Your task is "
              "to generate a fact statement that is useful in answering the given question."),
        given=("a question", "a corresponding answer"),
        goal="generate a fact statement that is useful in answering the given question",
    ),
    Lesson(
        task="task519_aquamuse_question_generation",
        text=("In this task you will be given an answer to a question. You need to generate a "
              "question. The answer given should be a correct answer for the generated question."),
        given=("an answer to a question",),
        goal="generate a question",
    ),
    Lesson(
        task="task1175_xcopa_commonsense_cause_effect_mr",
        text=("In this task you're given two statements in Marathi. You must judge whether the "
              "second sentence is the cause or effect of the first one. The sentences are "
              "separated by a newline character. Output either the word 'cause' or 'effect' ."),
        given=("two statements in Marathi",),
        goal="judge whether the second sentence is the cause or effect of the first one",
        outputs=("cause", "effect"),
    ),
    Lesson(
        task="task561_alt_translation_en_lo",
        text=("In this task, given a sentence in the English language, your task is to convert "
              "it into the Lao (Laotian) language."),
        given=("a sentence in the English language",),
        goal="convert it into the Lao (Laotian) language",
    ),
    Lesson(
        task="task431_senteval_object_count",
        text=("In this task you are given a sentence. You must judge whether the object of the "
              "main clause is singular(like: apple) or plural(like: apartments). Label the "
              'instances as "Singular" or "Plural" based on your judgment.'),
        given=("a sentence",),
        goal=("judge whether the object of the main clause is singular(like: apple) or "
              "plural(like: apartments)"),
        outputs=("Singular", "Plural"),
    ),
    Lesson(
        task="task1497_bengali_book_reviews_sentiment_classification",
        text=("In this task, you are given reviews written about the books in Bengali. You are "
              "expected to classify the sentiment of the reviews into two classes: positive or "
              "negative."),
        given=("reviews written about the books in Bengali",),
        goal="classify the sentiment of the reviews into two classes: positive or negative",
        outputs=("positive", "negative"),
    ),
    Lesson(
        task="task588_amazonfood_rating_classification",
        text=("In this task, you're given reviews from Amazon's food products and a summary of "
              "that review. Your task is to classify whether the given summary matches the "
              'original review. Generate "True" if the given review and its summary match, '
              'otherwise generate "False".'),
        given=("reviews from Amazon's food products", "a summary of that review"),
        goal="classify whether the given summary matches the original review",
        outputs=("True", "False"),
        conditions=(("the given review and its summary match", 'Generate "True"'),
                    ("otherwise", 'generate "False"')),
    ),
    Lesson(
        task="task933_wiki_auto_style_transfer",
        text=("In this task, we ask you to rewrite a sentence in simple English without changing "
              "its general meaning. Essentially, you want to make the sentence easier to read by "
              "using simpler words, utilizing more straightforward sentence structures, and "
              "omitting non-essential information etc."),
        goal="rewrite a sentence in simple English without changing its general meaning",
    ),
    Lesson(
        task="task1519_qa_srl_question_generation",
        text=("In this task, you are given a context tweet and an answer. Your job is to "
              "generate a question for the given answer based on the given tweet paragraph. Note "
              "that your question should be answerable based on the given tweet, and the answer "
              "to your question should be the given answer."),
        given=("a context tweet", "an answer"),
        goal="generate a question for the given answer based on the given tweet paragraph",
    ),
    Lesson(
        task="task122_conala_list_index_addition",
        text=("In this task, you will be given a list of integers. You should remove all of the "
              "odd integers from the list(consider 0 an even number). If every integer in the "
              'input list is odd then an empty list ("[]") should be returned. Otherwise, answer '
              "with the list of even numbers separated by comma inside brackets."),
        given=("a list of integers",),
        goal="remove all of the odd integers from the list(consider 0 an even number)",
        conditions=(("every integer in the input list is odd",
                     'an empty list ("[]") should be returned'),
                    ("otherwise",
                     "answer with the list of even numbers separated by comma inside brackets")),
    ),
    Lesson(
        task="task475_yelp_polarity_classification",
        text=("In this task, you must classify if a given review is positive/negative, "
              "indicating your answer as P or N."),
        goal="classify if a given review is positive/negative",
        outputs=("P", "N"),
    ),
    Lesson(
        task="task106_scruples_ethical_judgment",
        text=('In this task, you are given two simple actions (associated with "A", "B"). You '
              "must identify which action is considered less ethical. Do not generate anything "
              "else apart from one of the following characters: 'A', 'B'."),
        given=("two simple actions",),
        goal="identify which action is considered less ethical",
        outputs=("A", "B"),
    ),
    Lesson(
        task="task1295_adversarial_qa_question_answering",
        text=("In this task, you are given a question and a context passage. You have to answer "
              "the question based on the given passage."),
        given=("a question", "a context passage"),
        goal="answer the question based on the given passage",
    ),
    Lesson(
        task="task1556_scitail_passage_generation",
        text=("In this task, you are given a question and an answer, you would be asked to "
              "create the sentence based on the Question-Answer provided. It should be contained "
              "within the Question-Answer provided."),
        given=("a question", "an answer"),
        goal="create the sentence based on the Question-Answer provided",
    ),
)


def taught_procedures(lessons: Optional[Sequence[Lesson]] = None) -> ProcedureReader:
    """A reader with the demonstrations in it. What every measurement is taken against."""
    reader = ProcedureReader()
    for lesson in (LESSONS if lessons is None else lessons):
        reader.learn(lesson)
    return reader
