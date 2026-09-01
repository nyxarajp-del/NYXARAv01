"""NYXARA · njp/school.py — reasoning, language, coding: taught then examined (🎓, NJP V.09).

:mod:`nyxara.njp.curriculum` names the nine stages and refuses to report one as reached before it
is; :mod:`nyxara.njp.study` teaches her from a corpus and grades her on a held-out split;
:mod:`nyxara.njp.teacher` distils a demonstration into structure and throws its answer away. Each
of them is one third of a school and none of them is a school: there was nothing that sat her
down, worked out what she *cannot currently do*, taught that, and then examined her on material
she had never seen.

This is that, over twenty-six subjects — six of reasoning, seven of language and thirteen of
coding — and its design is one claim repeated in every one of them:

    Teaching is only teaching if a number moved on questions she was never taught.

**Every subject is pre-tested before it is taught.** The floor is measured first, on generated
items, and the same subject is then examined again after teaching on *different* items. What is
reported is the pair and the gain, never the post-test alone — a subject she could already do
scores 1.00 after teaching and its gain is zero, and saying so is the difference between a report
card and an advertisement. Several subjects here *do* score 1.00 cold, and they are printed with
``already`` beside them rather than quietly folded into a total.

**Nothing is examined on what it was taught.** Reasoning items are minted from freshly generated
nonsense vocabulary, so the entities in the exam were not in the world when the lesson ran. What
a lesson can move is *structure* — the posterior that says this relation chains — and structure
is precisely what applies to entities nobody mentioned. Coding items are held-out specs whose
shape was demonstrated on a different task with different constants and a different inner
function. Both are the same discipline: the answer is discarded, the shape is kept, and the exam
asks for the shape somewhere new.

**Abstention is a first-class outcome and is never merged into error.** Half the control items
here are questions she *should* refuse — a chain that does not exist, a property nobody stated —
and on those, silence is the right answer and is scored as one. On the positive items silence is
scored as a miss rather than as a wrong answer, and both appear in the report. A brain that
answers everything and a brain that answers nothing produce two very different report cards here,
which is the entire reason for keeping three counters instead of one.

**The language half is examined in a language that did not exist when this file was written.**
Fresh *vocabulary* is what the reasoning subjects mint, and for language it is not enough: "the
zorb chases the plag" is still an English sentence, and any shipped subject-verb-object frame reads
it correctly having learned nothing. So :mod:`nyxara.njp.dialects` mints the **grammar** too — the
word order, the case markers, the plural, the tense, the negator, the question particle — and
seven subjects ask what she can do with it. Measured on the compiler she ships with, over 192
sentences of eight minted dialects: **192 readable, 0 correct**, every one of the 32 denials read
as an assertion and not one of the 96 questions recognised as a question. That is the floor, and
it is a floor of confident wrong readings rather than of silence, which is why every language
subject carries controls only silence can pass.

**The coding half is thirteen subjects because "can she code" is thirteen questions.** Reading a
program and saying what it gives; saying what happened *inside* it; writing one-operator programs,
composed ones, loops, recursion, mappings, text, nested data, and the classic algorithms; finding
and fixing one wrong thing; the awkward inputs; and finally, with the teacher off, unseen tasks on
a fraction of the budget. Every writing subject is the same class over a different bank
(:class:`BankSubject`), so none of them can be graded more kindly than the others by accident, and
every one of them is scored on **held-out** pairs — a program fitted to the examples it was given
and failing one it was not is a coincidence, and it is named as one.

**Where a ceiling is structural it is named, not learned around.** ``depth`` is the case worth
reading: a chain of four hops fails cold for two independent reasons — the per-hop confidence
falls under ``core._MIN_LINK_CONFIDENCE`` because an unproven relation's transitivity prior is
low, *and* ``CognitiveLearningCore.max_depth`` refuses to extend the walk. Teaching fixes the
first. The second is a budget, and this module will raise it by one **only** after the posterior
has been earned, and rolls it straight back if the control items — the chains that do not exist —
start coming back "yes". A capability that is bought by loosening a gate is not a capability, and
the roll-back is what keeps that sentence true here rather than aspirational.

**What one run actually reports**, seed 7 and two rounds, reproducible::

                            taught          teacher off, fresh items
    subjects mastered       26 / 26         26 / 26
    right / wrong / absent  526 / 0 / 2     525 / 1 / 2
    accuracy · precision    1.00 · 1.00     0.99 · 1.00

Fourteen subjects moved because a lesson ran, every one of them on material nobody
demonstrated::

    depth             0.33 → 1.00     four-hop chains, once the relation is proved to chain
    morphology        0.17 → 1.00     the plural of a word that has never been uttered
    reading           0.33 → 1.00     who did what to whom, in an order nobody wrote code for
    polarity          0.67 → 1.00     a denial, kept as a denial
    questions         0.33 → 1.00     which part of the sentence is being asked about
    saying            0.75 → 1.00     a shape crossing from comprehension into production
    translation       0.33 → 1.00     the same meaning in a language sharing no word with it
    mappings          0.25 → 1.00     counting, grouping, lookup
    algorithms        0.60 → 1.00     gcd, primes, binary search, sorting, FizzBuzz
    strings           0.38 → 1.00     taking text apart and putting it back
    loops             0.75 → 1.00     a variable, a loop and a body
    recursion         0.75 → 1.00     a base case and a step
    structures        0.86 → 1.00     rows, columns, nesting
    code-composites   0.75 → 0.88     two and three operators deep

The other twelve read 1.00 or near it cold and are printed as ``already`` — arithmetic, the
three reasoning walks, abstention, word classes, reading Python, tracing it, repairing it,
one-operator programs, the awkward inputs, and transfer are things she could already do, and
saying so is the difference between a report card and an advertisement.

It also found four bugs, which is the argument for having a school at all: a deliberation ladder
that answered ``25 + 10`` with ``10``; a promoted ``shape:p>p>p`` that shadowed the general
composition walk at exactly four hops; a schema ranking that tried every taught composite before
the one-attempt seed answering ``sum(xs)``; and a list comprehension over a set that gave back the
set. All four are fixed where they were, not worked around here.

Pure standard library, no LLM. ``python -m nyxara.njp.school`` runs the whole school and prints
the report card.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Score", "Question", "Result", "Transcript", "Subject", "Taught", "School",
    "ExamConditions", "Course", "LanguageSubject",
    "REASONING", "LANGUAGE", "CODING", "SUBJECTS",
]

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"


class Mint:
    """Fresh nonsense vocabulary, and the guarantee that nothing is ever minted twice.

    The exam's whole validity rests on this. If an exam entity had appeared in a lesson the score
    would be measuring recall, and recall is the thing this brain is already good at — so every
    word is drawn from a counter as well as from the generator, and a collision is impossible
    rather than unlikely.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.issued = 0
        self.seen: set = set()

    @staticmethod
    def _tag(number: int) -> str:
        """The counter, in letters.

        It was in digits, and that quietly broke a whole subject: ``all zork8s are shiny`` does
        not match the grounder's plural rule the way ``all zorks are shiny`` does, so
        :class:`Inheritance` scored 0.33 and the reason was the exam's own vocabulary rather than
        anything about her. A generated word has to look like a word.
        """
        letters = ""
        number += 1
        while number:
            number, remainder = divmod(number - 1, 26)
            letters = chr(ord("a") + remainder) + letters
        return letters

    def word(self, syllables: int = 2) -> str:
        while True:
            self.issued += 1
            body = "".join(self.rng.choice(_CONSONANTS) + self.rng.choice(_VOWELS)
                           for _ in range(syllables))
            word = f"{body}{self._tag(self.issued)}"
            if word not in self.seen:
                self.seen.add(word)
                return word

    def words(self, count: int, syllables: int = 2) -> List[str]:
        return [self.word(syllables) for _ in range(count)]


# --------------------------------------------------------------------------- #
# what a question is, and what counts as getting it right
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Question:
    """One exam item: what to ask, what an acceptable answer looks like, and whether silence
    is one of the acceptable answers.

    ``accept`` is matched case-folded against the reply. ``silence_ok`` marks a **control** — an
    item whose honest answer is "I have no grounds for that", where an abstention scores as right
    and an assertion scores as wrong however confident it sounds. Controls are half the point:
    a subject examined only on items whose answer is "yes" cannot tell a reasoner from a brain
    that says yes.
    """

    ask: str = ""
    accept: Tuple[str, ...] = ()
    silence_ok: bool = False
    exact: bool = False
    note: str = ""

    def grade(self, reply: str) -> str:
        """``right`` | ``wrong`` | ``abstain``. Three outcomes, never two."""
        said = " ".join(str(reply or "").strip().lower().split())
        if not said:
            return "right" if self.silence_ok else "abstain"
        for want in self.accept:
            target = want.strip().lower()
            if (said == target) if self.exact else (target in said):
                return "right"
        return "wrong"


@dataclass
class Score:
    """Three counters, kept apart on purpose, and the two rates worth quoting."""

    right: int = 0
    wrong: int = 0
    abstained: int = 0

    @property
    def total(self) -> int:
        return self.right + self.wrong + self.abstained

    @property
    def accuracy(self) -> float:
        """Right out of *everything asked*. An abstention does not earn credit here."""
        return self.right / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Right out of what she actually asserted. The number that says whether she lies."""
        answered = self.right + self.wrong
        return self.right / answered if answered else 0.0

    @property
    def coverage(self) -> float:
        """How much she was willing to answer at all."""
        answered = self.right + self.wrong
        return answered / self.total if self.total else 0.0

    def add(self, verdict: str) -> None:
        if verdict == "right":
            self.right += 1
        elif verdict == "wrong":
            self.wrong += 1
        else:
            self.abstained += 1

    def merged(self, other: "Score") -> "Score":
        return Score(self.right + other.right, self.wrong + other.wrong,
                     self.abstained + other.abstained)

    def to_dict(self) -> Dict[str, Any]:
        return {"right": self.right, "wrong": self.wrong, "abstained": self.abstained,
                "total": self.total, "accuracy": round(self.accuracy, 4),
                "precision": round(self.precision, 4), "coverage": round(self.coverage, 4)}


@dataclass
class Result:
    """One subject's whole story: the floor, the ceiling reached, and what it cost."""

    subject: str = ""
    title: str = ""
    teaches: str = ""
    pre: Score = dc_field(default_factory=Score)
    post: Score = dc_field(default_factory=Score)
    rounds: int = 0
    taught: int = 0
    threshold: float = 0.8
    seconds: float = 0.0
    note: str = ""
    misses: List[str] = dc_field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.post.accuracy - self.pre.accuracy

    @property
    def mastered(self) -> bool:
        return self.post.total > 0 and self.post.accuracy >= self.threshold

    @property
    def already(self) -> bool:
        """She could do it before the lesson. Reported, never hidden."""
        return self.pre.total > 0 and self.pre.accuracy >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "title": self.title, "teaches": self.teaches,
                "pre": self.pre.to_dict(), "post": self.post.to_dict(),
                "gain": round(self.gain, 4), "mastered": self.mastered,
                "already": self.already, "rounds": self.rounds, "taught": self.taught,
                "threshold": self.threshold, "seconds": round(self.seconds, 3),
                "note": self.note, "misses": self.misses[:5]}


@dataclass
class Transcript:
    """The report card. Ordered as the syllabus is ordered, because the order is the claim."""

    results: List[Result] = dc_field(default_factory=list)
    seconds: float = 0.0
    seed: int = 0
    coder_stats: Dict[str, Any] = dc_field(default_factory=dict)

    @property
    def mastered(self) -> List[Result]:
        return [r for r in self.results if r.mastered]

    @property
    def failing(self) -> List[Result]:
        return [r for r in self.results if not r.mastered]

    @property
    def learned(self) -> List[Result]:
        """Subjects where **this subject's own lesson** moved the number.

        ``taught`` is in the condition deliberately. A floor subject that reads 0.92 on one set of
        generated items and 1.00 on the next has not learned anything — it has sampled twice — and
        a report that calls that "LEARNED" is a report that cannot be trusted about the subjects
        where a lesson really did run.
        """
        return [r for r in self.results if r.gain > 0.01 and r.taught > 0]

    @property
    def blocked_by(self) -> str:
        """The first subject she has not mastered, named with its number.

        The useful answer to "why is she not there yet" is a metric standing in the way, not a
        percentage — the same reading :class:`~nyxara.njp.curriculum.Report` gives.
        """
        for result in self.results:
            if not result.mastered:
                return f"{result.subject} ({result.post.accuracy:.2f} < {result.threshold:.2f})"
        return ""

    @property
    def overall(self) -> Score:
        out = Score()
        for result in self.results:
            out = out.merged(result.post)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "seconds": round(self.seconds, 2),
                "mastered": len(self.mastered), "subjects": len(self.results),
                "learned": [r.subject for r in self.learned],
                "blocked_by": self.blocked_by, "overall": self.overall.to_dict(),
                "coder": self.coder_stats,
                "results": [r.to_dict() for r in self.results]}

    def summary(self) -> str:
        """The report card as a person reads it."""
        lines = ["", "NYXARA · NJP report card — reasoning and coding",
                 f"seed {self.seed} · {self.seconds:.1f}s", ""]
        head = f"  {'subject':<16} {'cold':>6} {'after':>6} {'gain':>7} {'taught':>7}  verdict"
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))
        for r in self.results:
            # The same rule the `learned` list uses, and it has to be the same rule: a floor
            # subject that read 0.92 on one set of generated items and 1.00 on the next has
            # sampled twice, not learned, and a row that says LEARNED where the list below it
            # does not is a report card arguing with itself.
            if r.gain > 0.01 and r.taught > 0:
                verdict = "LEARNED"
            elif r.mastered and r.already:
                verdict = "already"
            elif r.mastered:
                verdict = "passed"
            else:
                verdict = "not yet"
            lines.append(f"  {r.subject:<16} {r.pre.accuracy:>6.2f} {r.post.accuracy:>6.2f} "
                         f"{r.gain:>+7.2f} {r.taught:>7}  {verdict}"
                         + (f" — {r.note}" if r.note else ""))
        overall = self.overall
        lines += ["", f"  mastered      {len(self.mastered)}/{len(self.results)} subjects",
                  f"  learned       {len(self.learned)} subject(s) moved by teaching: "
                  f"{', '.join(r.subject for r in self.learned) or 'none'}",
                  f"  overall       {overall.right}/{overall.total} right, "
                  f"{overall.wrong} wrong, {overall.abstained} abstained "
                  f"(accuracy {overall.accuracy:.2f}, precision {overall.precision:.2f})"]
        if self.blocked_by:
            lines.append(f"  blocked by    {self.blocked_by}")
        if self.coder_stats:
            lines.append(f"  shapes held   {self.coder_stats.get('taught', 0)} taught, "
                         f"{self.coder_stats.get('grafted', 0)} invented, "
                         f"{self.coder_stats.get('schemas', 0)} total")
            # The share that says what the teaching is *doing*. High means the lessons are
            # answering the questions; low would mean the seeds are and the lessons are decoration.
            lines.append(f"  written by    recall of a taught shape "
                         f"{self.coder_stats.get('recall_share', 0.0):.0%} of the time "
                         f"({self.coder_stats.get('recalled', 0)}/"
                         f"{self.coder_stats.get('written', 0)} programs)")
        lines.append("")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# a subject
# --------------------------------------------------------------------------- #

@dataclass
class Taught:
    """What one round of teaching did, so a subject that taught nothing can say so.

    Not called ``Lesson``: :class:`nyxara.njp.teacher.Lesson` is a *demonstration* — a task, an
    answer and its working — and this is the record of a teaching **round**. Two different things
    and the package exports both.
    """

    items: int = 0
    note: str = ""


class Subject:
    """One thing she is meant to be able to do, with a way to teach it and a way to test it.

    Subclasses implement :meth:`exam` and, where there is anything to teach, :meth:`teach`. The
    contract that makes the report card mean anything is enforced here rather than trusted:
    :meth:`exam` is handed a *fresh* :class:`Mint`, so it cannot reuse an entity a lesson
    mentioned even by accident.
    """

    id = ""
    title = ""
    teaches = ""
    threshold = 0.8
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:  # noqa: D401
        """Teach. The default teaches nothing, which is the honest default for an innate floor."""
        return Taught(0, "nothing to teach — this is a floor, not a lesson")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        raise NotImplementedError

    # -- helpers every subject uses ---------------------------------------- #
    @staticmethod
    def ask(brain: Any, question: Question) -> Tuple[str, str]:
        try:
            reply = brain.think(question.ask).answer
        except Exception as exc:  # noqa: BLE001 — a crash is a wrong answer, not a stopped exam
            return "wrong", f"{question.ask} → raised {type(exc).__name__}: {exc}"
        verdict = question.grade(reply)
        detail = f"{question.ask} → {reply.strip()[:60]!r}" if verdict != "right" else ""
        return verdict, detail

    @classmethod
    def sit(cls, brain: Any, questions: Sequence[Question]) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for question in questions:
            verdict, detail = cls.ask(brain, question)
            score.add(verdict)
            if detail:
                misses.append(detail)
        return score, misses

    @staticmethod
    def mark(score: Score, misses: List[str], question: Question, reply: str) -> str:
        """Grade a reply an organ produced directly, by exactly the rule :meth:`ask` uses.

        The coding subjects grade :class:`~nyxara.njp.coding.Coder` output without going through
        ``brain.think``, and so do the language subjects — an exam that had to phrase every item
        as an English sentence could not ask anything about a language that is not English. This
        keeps the three-outcome rule in one place so a subject grading its own organ cannot
        quietly score an abstention as an error.
        """
        verdict = question.grade(reply)
        score.add(verdict)
        if verdict != "right":
            misses.append(f"{question.ask} → {str(reply)[:60]!r}")
        return verdict


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #

class Arithmetic(Subject):
    """Closed arithmetic. A floor she is born with, and the report says so.

    Kept in the syllabus for the reason :mod:`nyxara.njp.curriculum` keeps Stage A: a subject
    that reads 1.00 cold is evidence about the *organ*, and an organ that quietly stopped working
    would show up here on the first run rather than three subjects later as an unexplained dip.
    """

    id, title = "arithmetic", "closed arithmetic"
    teaches = "an expression has a value, and it is the same value every time"
    threshold, items = 0.9, 10

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        rng = mint.rng
        questions = []
        for _ in range(self.items):
            a, b, c = rng.randint(2, 30), rng.randint(2, 20), rng.randint(2, 9)
            form, value = rng.choice([
                (f"what is {a} + {b}?", a + b),
                (f"what is {a} * {c}?", a * c),
                (f"{a} - {b} kitna hai?", a - b),
                (f"what is {a} + {b} * {c}?", a + b * c),
            ])
            questions.append(Question(ask=form, accept=(str(value),), exact=True))
        return self.sit(brain, questions)


class Composition(Subject):
    """Same-relation chains: ``a → b``, ``b → c``, therefore ``a → c``.

    The premises for an exam item are stated *for that item* — they have to be, a chain about
    entities she has never heard of is not a chain — and the conclusion never is. So what is
    scored is the composition and only the composition, and a fact store that answers by lookup
    scores zero.
    """

    id, title = "composition", "same-relation chains"
    teaches = "if it holds from a to b and b to c, it holds from a to c"
    threshold, items = 0.75, 12

    hops = (2, 3)

    def _mint_item(self, brain: Any, mint: Mint, hops: int, relation: str) -> Question:
        chain = mint.words(hops + 1)
        for left, right in zip(chain, chain[1:]):
            brain.think(f"{left} {relation}s {right}")
        return Question(ask=f"does {chain[0]} {relation} {chain[-1]}?", accept=("yes",),
                        note=f"{hops} hops")

    def _control(self, brain: Any, mint: Mint, relation: str) -> Question:
        chain = mint.words(3)
        for left, right in zip(chain, chain[1:]):
            brain.think(f"{left} {relation}s {right}")
        stranger = mint.word()
        return Question(ask=f"does {chain[0]} {relation} {stranger}?", accept=("no",),
                        silence_ok=True, note="control: no such chain")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        relation = mint.word()
        questions = []
        for index in range(self.items):
            if index % 3 == 2:
                questions.append(self._control(brain, mint, relation))
            else:
                questions.append(self._mint_item(brain, mint, self.hops[index % 2], relation))
        return self.sit(brain, questions)


class Inheritance(Subject):
    """Kind membership and the properties that come with it: ``x is a K``, ``all K are P``.

    The exam member is minted after the rule is stated, so "is this one P" has never been said
    about it. Controls ask for a property the kind was never given, where the honest answer is
    silence.
    """

    id, title = "inheritance", "kinds and their properties"
    teaches = "what is true of a kind is true of its members"
    threshold, items = 0.75, 12

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        questions = []
        for index in range(self.items):
            kind, prop = mint.word(), mint.word()
            brain.think(f"all {kind}s are {prop}")
            member = mint.word()
            brain.think(f"{member} is a {kind}")
            if index % 3 == 2:
                questions.append(Question(ask=f"is {member} {mint.word()}?", accept=("no",),
                                          silence_ok=True, note="control: unstated property"))
            else:
                questions.append(Question(ask=f"is {member} {prop}?", accept=("yes",)))
        return self.sit(brain, questions)


class Shapes(Subject):
    """Chains of *different* relations: ``x is a K``, ``K needs W`` ⊢ ``x needs W``.

    A different walk from :class:`Composition` and it was worth its own subject: transitivity
    composes one predicate with itself, a shape composes two, and confirming the first teaches
    nothing about the second. :meth:`~nyxara.njp.core.CognitiveLearningCore.walk_shape` is the
    organ under test.
    """

    id, title = "shapes", "mixed-relation forms"
    teaches = "a form like (is_a, needs) carries from one subject to another"
    threshold, items = 0.7, 10

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        questions = []
        for index in range(self.items):
            kind, resource = mint.word(), mint.word()
            brain.think(f"{kind} needs {resource}")
            member = mint.word()
            brain.think(f"{member} is a {kind}")
            if index % 3 == 2:
                questions.append(Question(ask=f"does {member} need {mint.word()}?",
                                          accept=("no",), silence_ok=True,
                                          note="control: unstated need"))
            else:
                questions.append(Question(ask=f"does {member} need {resource}?",
                                          accept=("yes", resource)))
        return self.sit(brain, questions)


class Abstention(Subject):
    """Questions with no grounds at all. Every item is a control and silence is the whole grade.

    This subject exists because every other one can be gamed by confidence and this one cannot:
    the only way to score here is to refuse, and the only way to fail is to make something up. It
    is the numerical form of the claim the rest of the package is built around, and it is
    deliberately examined *after* the subjects that reward saying "yes".
    """

    id, title = "abstention", "refusing what is not grounded"
    teaches = "no evidence is an answer, and it is not the same as no"
    threshold, items = 0.9, 10

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        questions = []
        for index in range(self.items):
            subject, other = mint.word(), mint.word()
            forms = [
                Question(ask=f"is {subject} {mint.word()}?", accept=("no", "don't know",
                                                                     "do not know"),
                         silence_ok=True),
                Question(ask=f"does {subject} cause {other}?", accept=("no",), silence_ok=True),
                Question(ask=f"does {subject} need {other}?", accept=("no",), silence_ok=True),
                Question(ask=f"what does {subject} need?", accept=("no", "unknown"),
                         silence_ok=True),
            ]
            questions.append(forms[index % len(forms)])
        return self.sit(brain, questions)


class Depth(Subject):
    """Long chains — and the one subject where teaching is measured against a hard budget.

    Four hops fails cold for two reasons at once, and they are worth separating because only one
    of them is a thing a lesson can move:

    * the per-hop confidence is multiplied by the relation's transitivity posterior, and an
      unproven relation sits near :data:`~nyxara.njp.core._TRANSITIVE_DEFAULT`, so the fourth hop
      falls under ``_MIN_LINK_CONFIDENCE`` and the walk stops. **Teaching moves this**, by
      :class:`~nyxara.njp.teacher.Distiller` on demonstrations over entities the exam never sees.
    * ``CognitiveLearningCore.max_depth`` refuses to extend a path past four regardless of how
      confident it is. That is a budget, not a belief, and no lesson can argue with it.

    So this subject teaches the first and then *earns* the second: the budget goes up by one only
    once the posterior clears the bar, and :meth:`teach` puts it straight back if the control
    items — chains that do not exist — start answering "yes". A capability bought by loosening a
    gate is not a capability, and rolling back is what keeps that sentence true.
    """

    id, title = "depth", "chains four hops and longer"
    teaches = "a relation proved to chain may be followed further than one that is not"
    threshold, items = 0.7, 12
    lessons = 8
    posterior_bar = 0.75
    depth_ceiling = 6

    def __init__(self) -> None:
        self.relation = ""

    def _relation(self, mint: Mint) -> str:
        # One relation for the life of the subject: the structure taught is a property *of a
        # relation*, so teaching one and examining another would measure nothing at all.
        if not self.relation:
            self.relation = mint.word()
        return self.relation

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        relation = self._relation(mint)
        questions = []
        for index in range(self.items):
            hops = 4 + (index % 2)
            chain = mint.words(hops + 1)
            for left, right in zip(chain, chain[1:]):
                brain.think(f"{left} {relation}s {right}")
            if index % 3 == 2:
                questions.append(Question(ask=f"does {chain[0]} {relation} {mint.word()}?",
                                          accept=("no",), silence_ok=True,
                                          note="control: no such chain"))
            else:
                questions.append(Question(ask=f"does {chain[0]} {relation} {chain[-1]}?",
                                          accept=("yes",), note=f"{hops} hops"))
        return self.sit(brain, questions)

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        from nyxara.njp.teacher import Distiller, Lesson as Demonstration, Step, Verdict

        learner = getattr(brain, "learner", None)
        if learner is None:
            return Taught(0, "no learning core to teach into")
        relation = self._relation(mint)
        singular = relation[:-1] if relation.endswith("s") else relation
        distiller, taught = Distiller(), 0
        for _ in range(self.lessons):
            chain = mint.words(4)
            for left, right in zip(chain, chain[1:]):
                brain.think(f"{left} {relation}s {right}")
            demo = Demonstration(
                task=f"does {chain[0]} {singular} {chain[-1]}?", answer="yes",
                steps=tuple(Step(a, singular, b) for a, b in zip(chain, chain[1:])),
                source="school", confidence=0.7)
            if distiller.distil(demo, brain).verification.verdict == Verdict.SURVIVED:
                taught += 1
        posterior = learner._transitivity(singular).value
        note = f"{singular} chains: posterior {posterior:.2f} on {taught} verified lessons"
        if taught and posterior >= self.posterior_bar:
            note += self._earn_depth(brain, learner, mint, coder)
        else:
            note += "; budget not earned"
        return Taught(taught, note)

    def _earn_depth(self, brain: Any, learner: Any, mint: Mint, coder: Any) -> str:
        """Raise the walk budget by one, then check the controls and undo it if they went soft."""
        before = int(getattr(learner, "max_depth", 4))
        if before >= self.depth_ceiling:
            return "; budget already at its ceiling"
        control_before = self._controls(brain, mint)
        learner.max_depth = before + 1
        control_after = self._controls(brain, mint)
        if control_after.accuracy < control_before.accuracy:
            learner.max_depth = before
            return (f"; budget {before}→{before + 1} rolled back, controls fell "
                    f"{control_before.accuracy:.2f}→{control_after.accuracy:.2f}")
        return f"; budget {before}→{before + 1}, controls held at {control_after.accuracy:.2f}"

    def _controls(self, brain: Any, mint: Mint, count: int = 6) -> Score:
        """Chains that do not exist. The price of a bigger budget is that these stay silent.

        The before and after sets are *different* items — the mint never issues a word twice, so
        they have to be — which makes this a comparison of two samples rather than a re-run of one.
        That is the weaker of the two possible checks and it is the right one here: correct
        behaviour scores 1.00 on **any** control set, because none of these chains exists at all,
        so a drop between two samples is a real drop rather than sampling noise.
        """
        relation = self._relation(mint)
        questions = []
        for _ in range(count):
            chain = mint.words(5)
            for left, right in zip(chain, chain[1:]):
                brain.think(f"{left} {relation}s {right}")
            questions.append(Question(ask=f"does {chain[0]} {relation} {mint.word()}?",
                                      accept=("no",), silence_ok=True))
        score, _ = self.sit(brain, questions)
        return score


REASONING: Tuple[Subject, ...] = ()  # filled at the bottom, once the coding subjects exist


# --------------------------------------------------------------------------- #
# coding
# --------------------------------------------------------------------------- #

class CodeReading(Subject):
    """Read a person's Python, run it in her own machine, and say what it produces.

    Two halves and both count: fourteen constructions she must read and evaluate exactly, and six
    she must **refuse** — an import, a file open, an ``eval``, a dunder walk, a loop, a dict. A
    reader that accepts the first list and rejects the second is doing the job; one that accepts
    everything is a security hole with a good score.
    """

    id, title = "code-reading", "reading Python and predicting its result"
    teaches = "what a program says, before anything is written"
    threshold = 0.98

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import CodeError, read_python
        from nyxara.njp.tasks import EVERY_TASK, READING, REFUSALS, instance, normalise
        score, misses = Score(), []
        # Every worked solution in the syllabus, on its own examples. Sixty-five programs across
        # loops, recursion, mappings, text, nested data and the classic algorithms — so "she can
        # read Python" is a number over the whole bank rather than over a table someone chose.
        for task in EVERY_TASK:
            spec, source = instance(task, mint.rng, task.id)
            first = source.strip().splitlines()[0][:44]
            try:
                program = read_python(source, name=spec.name)
            except CodeError as exc:
                score.add("wrong")
                misses.append(f"{first} → refused readable code: {exc}")
                continue
            for example in (spec.shown + spec.held_out)[:3]:
                try:
                    got = coder.run(program, example.args)
                except CodeError as exc:
                    score.add("wrong")
                    misses.append(f"{task.id}: {exc}")
                    continue
                score.add("right" if got == example.out else "wrong")
                if got != example.out:
                    misses.append(f"{task.id}: got {got!r}, wanted {example.out!r}")
        for source, ref, args in READING:
            first = source.strip().splitlines()[0][:44]
            try:
                program = read_python(source)
                got = coder.run(program, args)
                want = normalise(ref(*args))
                score.add("right" if got == want else "wrong")
                if got != want:
                    misses.append(f"{first} → {got!r}, wanted {want!r}")
            except CodeError as exc:
                score.add("wrong")
                misses.append(f"{first} → refused readable code: {exc}")
        for source in REFUSALS:
            first = source.strip().splitlines()[0][:44]
            try:
                read_python(source)
            except CodeError:
                score.add("right")
                continue
            score.add("wrong")
            misses.append(f"{first} → accepted source it should refuse")
        return score, misses


class Debugging(Subject):
    """A program with one thing wrong with it, and a fix that has to run before it counts.

    The corruption is checked to actually break the program before she is asked to repair it —
    an "exercise" a wrong program passes anyway is not an exercise — and the repair is graded on
    the **held-out** pairs, so restoring the shown examples by luck does not count as fixed.
    """

    id, title = "debugging", "finding and fixing one wrong thing"
    teaches = "localise the fault by running it, not by looking at it"
    threshold, items = 0.6, 12
    attempts = 20000

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import read_python
        from nyxara.njp.tasks import EVERY_TASK, instance
        rng = mint.rng
        score, misses = Score(), []
        pool = list(EVERY_TASK)
        rng.shuffle(pool)
        for family in pool[:self.items]:
            spec, source = instance(family, rng, f"{family.id}_{mint.word(1)}")
            program = read_python(source, name=spec.name)
            broken = self._corrupt(coder, program, spec, rng)
            if broken is None:
                misses.append(f"{spec.name}: could not build a broken version")
                score.add("abstain")
                continue
            fixed = coder.repair(broken, spec, attempts=self.attempts)
            if not fixed.ok:
                misses.append(f"{spec.name}: no single edit found in {fixed.attempts} attempts")
                score.add("abstain")
                continue
            score.add("right" if coder.check(fixed.program, spec.held_out).ok else "wrong")
        return score, misses

    @staticmethod
    def _corrupt(coder: Any, program: Any, spec: Any, rng: random.Random) -> Any:
        """Break exactly one node, and prove it is broken before handing it over."""
        from nyxara.njp.coding import Call, Lit, OPS, Program, arity_of, _positions, _replace_at
        spots = [(path, node) for path, node in _positions(program.body)
                 if isinstance(node, Lit) or (isinstance(node, Call)
                                              and node.op not in ("map", "filter", "fold"))]
        rng.shuffle(spots)
        for path, node in spots:
            if isinstance(node, Lit) and isinstance(node.value, int):
                replacement: Any = Lit(node.value + rng.choice([1, 2, -1]))
            elif isinstance(node, Call):
                same = [op for op in OPS
                        if arity_of(op) == arity_of(node.op) and op != node.op
                        and op not in ("map", "filter", "fold")]
                if not same:
                    continue
                replacement = Call(rng.choice(same), node.args)
            else:
                continue
            candidate = Program(program.name, program.params,
                                _replace_at(program.body, path, replacement))
            if not coder.check(candidate, spec.shown).ok:
                return candidate
        return None


class BankSubject(Subject):
    """A writing subject over one bank of :class:`~nyxara.njp.tasks.Task` families.

    Every writing subject in the syllabus is this one class with a different bank, which is the
    point: loops, recursion, mappings, text and nested data are *the same examination* — she is
    shown one instance of a family and asked for another — and giving each of them its own bespoke
    grader would be five chances to grade one of them leniently without noticing.

    Grafting is off and the budget is identical between the pre-test and the post-test, so the
    only thing that differs across the two numbers is which shapes she holds.
    """

    bank: Tuple[Any, ...] = ()
    attempts = 30000
    threshold = 0.75

    def _instances(self, mint: Mint, tag: str) -> Any:
        from nyxara.njp.tasks import instance
        for task in self.bank:
            yield task, instance(task, mint.rng, f"{task.id}_{tag}{mint.word(1)}")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for _task, (spec, _source) in self._instances(mint, "x"):
            written = coder.write(spec, attempts=self.attempts, graft=False)
            score.add(self.grade(coder, spec, written, misses))
        return score, misses

    @staticmethod
    def grade(coder: Any, spec: Any, written: Any, misses: List[str]) -> str:
        """Passing the shown examples is not passing. The held-out ones decide.

        A program fitted to the pairs it was given and failing one it was not found a coincidence,
        and the whole reason the split exists is to call that what it is rather than to count it
        as a solve. An abstention is scored apart from a wrong answer, because refusing to guess
        is the behaviour the rest of this package works to guarantee.
        """
        if not written.ok:
            misses.append(f"{spec.name}: abstained after {written.attempts} attempts")
            return "abstain"
        check = coder.check(written.program, spec.held_out)
        if check.ok:
            return "right"
        misses.append(f"{spec.name}: fitted the shown pairs, failed held-out "
                      f"({check.passed}/{check.total})")
        return "wrong"

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        from nyxara.njp.teacher import Verdict
        taught, refused = 0, []
        for task, (spec, source) in self._instances(mint, "t"):
            learned = coder.learn_python(spec, source)
            if learned.verdict == Verdict.SURVIVED:
                taught += 1
            else:
                refused.append(f"{task.id}: {learned.verdict}")
        note = f"{taught} of {len(self.bank)} shapes verified by execution"
        if refused:
            note += "; " + "; ".join(refused[:2])
        return Taught(taught, note)


class WriteLoops(BankSubject):
    """A variable, a loop, and a body. She could read one long before she could write one."""

    id, title = "loops", "writing loops that accumulate, build and count"
    teaches = "state that survives an iteration is what a loop is for"
    bank = ()          # filled below, once nyxara.njp.tasks is importable at module scope


class WriteRecursion(BankSubject):
    """A base case and a step — the one thing in the language that can name itself."""

    id, title = "recursion", "writing functions that call themselves"
    teaches = "a problem that contains a smaller copy of itself"
    threshold = 0.7


class WriteDicts(BankSubject):
    """Counting, grouping, lookup. Most real programs are mostly this."""

    id, title = "mappings", "counting, grouping and looking things up"
    teaches = "a key is a question and the value is its answer"
    threshold = 0.7


class WriteStrings(BankSubject):
    """Text, which is half of what anybody actually writes."""

    id, title = "strings", "taking text apart and putting it back together"
    teaches = "a string is a sequence, and then it is not"
    threshold = 0.7


class WriteStructures(BankSubject):
    """Lists of lists. Nesting is where a language either composes or does not."""

    id, title = "structures", "nested data, rows and columns"
    teaches = "an operation that works on a list works on a list of them"
    threshold = 0.7


class WriteAlgorithms(BankSubject):
    """gcd, primes, binary search, sorting, FizzBuzz — what a course sets when it wants to know
    whether you can actually program, rather than whether you can recite a form."""

    id, title = "algorithms", "the classic problems, end to end"
    teaches = "a method, not a formula"
    threshold = 0.6
    attempts = 30000


class Tracing(Subject):
    """Not "what does it return" but "what happened on the way".

    Reading code is two abilities and the exam had only one of them. Running a program to its
    answer can be done by a machine that understands nothing; saying what each sub-expression took
    as its value is the part that is understanding, and it is what makes a wrong answer
    *diagnosable* instead of merely wrong.
    """

    id, title = "tracing", "saying what happened inside, step by step"
    teaches = "an answer with its working is a different object from an answer"
    threshold, items = 0.9, 12

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import CodeError, read_python
        from nyxara.njp.tasks import EVERY_TASK, instance
        score, misses = Score(), []
        tasks = list(EVERY_TASK)
        mint.rng.shuffle(tasks)
        for task in tasks[:self.items]:
            spec, source = instance(task, mint.rng, task.id)
            if not spec.shown:
                continue
            example = spec.shown[0]
            try:
                program = read_python(source, name=spec.name)
                steps = coder.trace(program, example.args)
            except CodeError as exc:
                score.add("wrong")
                misses.append(f"{task.id}: {exc}")
                continue
            # A trace has to have steps, every step has to have been evaluated, and the last one
            # has to agree with what running it plainly gives. A trace that agreed with nothing
            # would be a story rather than a record.
            plain = None
            try:
                plain = coder.run(program, example.args)
            except CodeError:
                pass
            if steps and any(step.source for step in steps) and steps[-1].value == plain:
                score.add("right")
            else:
                score.add("wrong")
                misses.append(f"{task.id}: {len(steps)} steps, ends {steps[-1].value if steps else None!r}, "
                              f"runs to {plain!r}")
        return score, misses


class EdgeCases(Subject):
    """The inputs that break a program that was only ever tried on the happy path.

    Empty, one element, all zeros, all negatives, repeats. Every item is a correct program run on
    an awkward input, and the grade is whether her interpreter gives what Python gives — including
    when the right answer is an *error*. A language that quietly returns something for
    ``max([])`` is worse than one that raises, because the something is wrong and looks fine.
    """

    id, title = "edge-cases", "empty, single, zero, negative, repeated"
    teaches = "the input you did not think of is the one that decides"
    threshold = 0.9

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import CodeError, read_python
        from nyxara.njp.tasks import EDGE_CASES, normalise
        score, misses = Score(), []
        for entry in EDGE_CASES:
            label, source, args = entry[0], entry[1], entry[2]
            diverges = len(entry) > 3 and entry[3] == "refuse"
            want, raised = None, False
            try:
                want = normalise(_python_answer(source, args))
            except Exception:  # noqa: BLE001 - Python itself refusing is the expected answer
                raised = True
            if diverges:
                # A case where this language is *meant* to differ. Graded as a refusal, and named
                # in the bank so the divergence stays a decision rather than becoming a defect
                # nobody remembers making.
                raised = True
            try:
                got = coder.run(read_python(source, name="f"), args)
            except CodeError:
                score.add("right" if raised else "wrong")
                if not raised:
                    misses.append(f"{label}: refused an input Python answers with {want!r}")
                continue
            if raised:
                score.add("wrong")
                misses.append(f"{label}: answered {got!r} where Python raises")
            elif got == want:
                score.add("right")
            else:
                score.add("wrong")
                misses.append(f"{label}: got {got!r}, Python gives {want!r}")
        return score, misses


def _python_answer(source: str, args: Sequence[Any]) -> Any:
    from nyxara.njp.tasks import reference
    return reference(source)(*args)


class Transfer(Subject):
    """Unseen tasks, a tight budget, and no teaching at all — the teacher-off number.

    :mod:`nyxara.njp.teacher` states the test the whole of Phase 4 comes down to: *"NJP + teacher
    vs NJP after distillation vs NJP alone. Teacher OFF ke baad performance kitni retain hui?"*
    This is that test for the coding half. The budget is a fraction of the one
    :class:`WriteComposite` gets, so nothing here is solvable by searching harder — only by
    already holding the shape.
    """

    id, title = "transfer", "unseen tasks on a budget, teacher off"
    teaches = "nothing — this subject only measures what the earlier ones left behind"
    threshold, items = 0.5, 14
    attempts = 8000

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """Nothing, and that is the measurement. Anything taught here would spoil the number."""
        return Taught(0, "teacher off — this score is what the earlier subjects left behind")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.tasks import EVERY_TASK, instance
        score, misses = Score(), []
        pool = list(EVERY_TASK)
        mint.rng.shuffle(pool)
        for family in pool[:self.items]:
            spec, _ = instance(family, mint.rng, f"{family.id}_{mint.word(1)}")
            written = coder.write(spec, attempts=self.attempts, graft=False)
            score.add(BankSubject.grade(coder, spec, written, misses))
        return score, misses


# --------------------------------------------------------------------------- #
# the syllabus and the school
# --------------------------------------------------------------------------- #

def _attach_banks() -> None:
    """Give each writing subject its bank, once :mod:`nyxara.njp.tasks` is importable.

    Assigned here rather than in the class bodies because the class bodies run at import time and
    an import cycle through ``tasks`` would leave every bank empty — an examination with no
    questions in it, which reports 0/0 and calls it mastered.
    """
    from nyxara.njp import tasks
    WriteBasic.bank = tasks.BASIC
    WriteComposite.bank = tasks.COMPOSITE
    WriteLoops.bank = tasks.LOOPS
    WriteRecursion.bank = tasks.RECURSION
    WriteDicts.bank = tasks.DICTS
    WriteStrings.bank = tasks.STRINGS
    WriteStructures.bank = tasks.STRUCTURES
    WriteAlgorithms.bank = tasks.ALGORITHMS


class WriteBasic(BankSubject):
    """One-operator programs. Reachable from the seed shapes, so this is a floor, not a lesson —
    and a low score here is a broken search rather than a missing shape, which is why it is
    examined before the subjects that need teaching and why its bar is the highest."""

    id, title = "code-basics", "writing one-operator programs from examples"
    teaches = "a specification is a set of examples, and a program matches them or does not"
    threshold, attempts = 0.85, 8000


class WriteComposite(BankSubject):
    """Two and three operators deep — where a lesson first shows up as a number."""

    id, title = "code-composites", "writing composed programs from examples"
    teaches = "a shape seen once on one task is worth having on another"
    threshold, attempts = 0.75, 30000


# --------------------------------------------------------------------------- #
# language
# --------------------------------------------------------------------------- #

class Course:
    """The two minted languages every language subject shares, and why they are shared.

    One dialect across the seven subjects because they are seven questions about **one** language:
    a construction learned in ``reading`` is the construction ``saying`` is asked to run backwards,
    and giving each subject its own language would examine seven strangers rather than one
    student. A **second** dialect because translation is the only measurement that separates a
    meaning from a surface, and it needs a surface she has to reach across.

    Both are minted from the exam's own stream, so they are reproducible from the seed and neither
    of them existed when this module was written.
    """

    def __init__(self, rng: random.Random) -> None:
        from nyxara.njp.dialects import mint_dialect
        self.first = mint_dialect(rng, "dialect-a")
        # Drawn to share no form with the first, so "which language is this" has an answer. Two
        # dialects that happened to pick the same negator would make that item undecidable, and
        # she would answer `ambiguous` — correctly — and be marked wrong for it.
        self.second = mint_dialect(rng, "dialect-b", avoid=self.first)


class LanguageSubject(Subject):
    """The half of the syllabus that is examined in a language nobody has ever spoken.

    Every subject below grades an organ directly rather than through ``brain.think``, for the
    reason :class:`BankSubject` does: an exam that had to phrase its items as English sentences
    could ask nothing at all about a language that is not English.

    **What makes these scores mean something** is that the language is minted per run. Fresh
    *vocabulary* — which is what the reasoning subjects mint — is not enough here, because "the
    zorb chases the plag" is still an English sentence and any shipped subject-verb-object frame
    reads it correctly having learned nothing. Measured on the shipped compiler over 192 sentences
    of eight minted dialects: **192 readable, 0 correct**, every one of the 32 denials read as an
    assertion and not one of the 96 questions recognised as a question. That is the floor these
    subjects sit on, and it is a floor of confident wrong readings rather than of silence — which
    is why every one of them carries controls that only silence can pass.
    """

    kinds: Tuple[str, ...] = ("assertion",)
    items = 12
    threshold = 0.8

    # -- the organs under test ---------------------------------------------- #
    @staticmethod
    def faculty(brain: Any) -> Any:
        """Her language faculty, attached to the brain if the brain has not got one.

        Attached rather than created per call, because what a lesson leaves has to still be there
        for the next subject and for the retention run. A faculty built fresh for each exam would
        measure nothing but the exam.
        """
        from nyxara.njp.language import LanguageFaculty
        spoken = getattr(brain, "language", None)
        if spoken is None:
            spoken = LanguageFaculty()
            try:
                brain.language = spoken
            except Exception:  # noqa: BLE001 — a brain that will not hold one still gets examined
                pass
        return spoken

    def course(self, brain: Any, mint: Mint) -> Course:
        course = getattr(self.faculty(brain), "course", None)
        if course is None:
            course = Course(mint.rng)
            try:
                self.faculty(brain).course = course
            except Exception:  # noqa: BLE001
                pass
        return course

    # -- shared item shapes -------------------------------------------------- #
    @staticmethod
    def _said(meaning: Any) -> str:
        """One reading, flattened to a string, so :meth:`Question.grade` can judge it.

        Every field a construction is responsible for is in here. Leaving negation out would let a
        denial score as its own assertion, which is the single failure
        :mod:`nyxara.njp.semantics` was written for and the one this exam exists to keep closed.
        """
        if meaning is None or not getattr(meaning, "readable", False):
            return ""
        parts = [meaning.kind]
        if meaning.negated:
            parts.append("not")
        if meaning.temporal:
            parts.append(meaning.temporal)
        if meaning.focus:
            parts.append(f"?{meaning.focus}")
        parts.append(f"{meaning.subject}|{meaning.relation}|{meaning.object}")
        return " ".join(parts)

    @staticmethod
    def _control(surface: str, mint: Mint) -> str:
        """A sentence with one word too many, which her grammar has no shape for.

        Chosen over the obvious controls after both of those turned out to be readable sentences:
        a doubled negator is absorbed by the slot beside it, and a sentence with its verb removed
        is just a shorter sentence that another construction fits. A word too many is refused by
        every construction of every minted dialect — a slot takes one token, so arity is fixed —
        and it stays refused after the whole syllabus has been taught, which is what makes it
        usable in the retention run as well.
        """
        return f"{surface} {mint.word()}"


class WordShapes(LanguageSubject):
    """The wug test: an ending she induced, applied to a stem nobody has ever inflected.

    The oldest experiment in the subject and still the sharpest, because it cannot be passed by
    remembering. She overhears a word list — each form bare, plural and past — and is shown two
    pairs that say which ending is which. Then she is asked for the plural of a word that was not
    in the list, was not in a lesson, and has never been uttered by anyone.

    The controls ask for a feature nobody ever demonstrated. The right answer there is the empty
    string, and a morphology that produced *something* would be one whose rules are not evidence
    of anything.
    """

    id, title = "morphology", "the shape of a word she has never met"
    teaches = "an ending is a rule, and a rule applies to a word that was not in the lesson"
    threshold, items = 0.85, 12

    stems = 12

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        from nyxara.njp import dialects
        forms: List[str] = []
        shown = [dialects.stem(dialect, mint.word) for _ in range(self.stems)]
        for base in shown:
            forms.extend((base, dialect.pluralise(base), dialect.pasten(base)))
        spoken.hear_words(forms, tongue=dialect.name)
        # Two pairs per feature, and no more. The claim being tested is that the ending
        # generalises, so demonstrating it on ten stems would leave the exam unable to tell a
        # rule from a well-stocked table.
        bound = 0
        for base in shown[:2]:
            bound += int(spoken.bind(base, dialect.pluralise(base), "plural",
                                     tongue=dialect.name))
            bound += int(spoken.bind(base, dialect.pasten(base), "past", tongue=dialect.name))
        affixes = len(spoken.tongue(dialect.name).morphology.affixes)
        return Taught(len(shown), f"{len(forms)} forms overheard, {affixes} endings induced, "
                                  f"{bound} of 4 demonstrated pairs bound to a feature")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        score, misses = Score(), []
        for index in range(self.items):
            fresh = dialects.stem(dialect, mint.word)
            if index % 6 == 4:
                # Recognition, which is the other direction and a separate ability: given a word
                # she has never seen, in a shape she has, say what the shape does.
                surface = dialect.pluralise(fresh)
                question = Question(ask=f"what does the ending of {surface!r} do?",
                                    accept=("plural",), exact=True)
                self.mark(score, misses, question,
                          spoken.analyse(surface, tongue=dialect.name).feature)
            elif index % 6 == 5:
                question = Question(ask=f"the future of {fresh!r}?", accept=(), silence_ok=True,
                                    note="control: a feature nobody demonstrated")
                self.mark(score, misses, question,
                          spoken.inflect(fresh, "future", tongue=dialect.name))
            elif index % 2:
                question = Question(ask=f"the past of {fresh!r}?",
                                    accept=(dialect.pasten(fresh),), exact=True)
                self.mark(score, misses, question,
                          spoken.inflect(fresh, "past", tongue=dialect.name))
            else:
                question = Question(ask=f"the plural of {fresh!r}?",
                                    accept=(dialect.pluralise(fresh),), exact=True)
                self.mark(score, misses, question,
                          spoken.inflect(fresh, "plural", tongue=dialect.name))
        return score, misses


class WordClasses(LanguageSubject):
    """Which words behave alike — found by listening, and answered without naming anything.

    She hears sentences with no meanings attached, which is most of what anyone ever hears, and
    is then asked whether two words are the same kind of word. She is never asked *which* kind,
    because the classes she forms have no names and the honest exam is the one that only asks
    what the evidence can answer.

    The controls pair a word she has met with one she has not. ``None`` — silence — is right
    there, and "no" is wrong, because a word she has never heard is not a word of a different
    class.

    **This reads 1.00 cold and is printed as** ``already``, which is the honest result and not a
    disappointing one. The ability needs *exposure*, not teaching, and an exam about words she has
    met has to supply its own exposure — so what a lesson adds here is more of the same thing the
    exam already does. It stays in the syllabus for the reason :class:`Arithmetic` does: a floor
    that quietly stopped working would show up here rather than three subjects later as an
    unexplained dip. It has already earned that keep once — the greedy clustering put eight
    identically-behaving nouns into eight classes of one in a verb-initial language, and this
    subject is where that surfaced.
    """

    id, title = "word-classes", "which words behave alike"
    teaches = "a word's kind is where it occurs, not what it means"
    threshold, items = 0.75, 12

    heard = 24

    @staticmethod
    def _chorus(dialect: Any, mint: Mint, count: int,
                kinds: Sequence[str]) -> Tuple[List[Any], List[str], List[str], List[str]]:
        """Sentences drawn from a **small pool**, so every word in them occurs more than once.

        :func:`~nyxara.njp.dialects.sample` gives every sentence fresh words, which is exactly
        right for the subjects that examine a shape and exactly wrong here. A distributional class
        is built out of the contexts a word was seen in, and one context is not a distribution —
        :class:`~nyxara.njp.language.Lexicon` says so by refusing to classify a word it has met
        once, and that refusal is what this corpus has to respect rather than work around. The
        first version of this exam ignored it, every noun in the corpus was a singleton, and she
        correctly answered "I have no grounds" to all eight items and scored 0.33.
        """
        from nyxara.njp import dialects
        subjects = [dialects.stem(dialect, mint.word) for _ in range(4)]
        objects = [dialects.stem(dialect, mint.word) for _ in range(4)]
        verbs = [dialects.stem(dialect, mint.word) for _ in range(3)]
        heard = [dialect.utter(subjects[index % len(subjects)], verbs[index % len(verbs)],
                               objects[index % len(objects)], kind=kinds[index % len(kinds)])
                 for index in range(count)]
        return heard, subjects, objects, verbs

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        heard, _s, _o, _v = self._chorus(dialect, mint, self.heard,
                                         ("assertion", "negated", "past"))
        for utterance in heard:
            spoken.hear(utterance.surface, tongue=dialect.name)
        classes = spoken.classify(tongue=dialect.name)
        return Taught(len(heard), f"{len(heard)} sentences overheard, {classes} classes formed")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        # The exam's own corpus, heard before it is asked about — a distributional class is a fact
        # about words she has met, so asking about words she has not is the control, not the item.
        heard, subjects, objects, verbs = self._chorus(dialect, mint, 12,
                                                       ("assertion", "past"))
        for utterance in heard:
            spoken.hear(utterance.surface, tongue=dialect.name)
        spoken.classify(tongue=dialect.name)
        score, misses = Score(), []

        def word_for(role: str, index: int) -> str:
            pool = subjects if role == "subject" else objects
            return dialect.noun(pool[index % len(pool)], role=role)

        for index in range(self.items):
            if index % 3 == 0:
                left, right = word_for("subject", index), word_for("subject", index + 1)
                answer, note = "yes", "two words in the same position"
            elif index % 3 == 1:
                left = word_for("object", index)
                right = dialect.verb(verbs[index % len(verbs)])
                answer, note = "no", "a word from each position"
            else:
                left, right = word_for("subject", index), dialects.stem(dialect, mint.word)
                answer, note = "", "control: a word she has never heard"
            question = Question(ask=f"are {left!r} and {right!r} the same kind of word?",
                                accept=(answer,) if answer else (), exact=True,
                                silence_ok=not answer, note=note)
            verdict = spoken.same_class(left, right, tongue=dialect.name)
            self.mark(score, misses, question,
                      "" if verdict is None else ("yes" if verdict else "no"))
        return score, misses


class Reading(LanguageSubject):
    """Who did what to whom, in a language whose word order she was never told.

    This is the subject the module exists for. She is shown sixteen sentences with their meanings,
    and examined on sentences built from words that were not in any of them. What can carry across
    is the **shape** — which position is the subject, what marks the object, where the verb
    goes — and a shape is exactly what a fresh word cannot help with.

    Cold, this scores nothing at all, and it scores nothing in the honest way: with no
    construction, every item is an abstention rather than a guess.
    """

    id, title = "reading", "who did what to whom, in an unfamiliar grammar"
    teaches = "the order and the markers are the meaning, and they generalise to new words"
    threshold, items = 0.8, 12

    lesson = 16
    kinds = ("assertion",)

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        shown = dialects.sample(dialect, mint.word, self.lesson, kinds=self.kinds)
        for utterance in shown:
            spoken.show(utterance.surface, utterance.meaning, tongue=dialect.name)
        report = spoken.learn(tongue=dialect.name)
        return Taught(len(shown), f"{report.kept} shapes kept, {report.rejected} rejected "
                                  f"from {report.demonstrations} demonstrations")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        score, misses = Score(), []
        for index, utterance in enumerate(
                dialects.sample(dialect, mint.word, self.items, kinds=self.kinds)):
            if index % 3 == 2:
                surface = self._control(utterance.surface, mint)
                question = Question(ask=surface, accept=(), silence_ok=True,
                                    note="control: a shape her grammar has not got")
            else:
                surface = utterance.surface
                question = Question(ask=surface, accept=(self._said(utterance.meaning),),
                                    exact=True)
            self.mark(score, misses, question,
                      self._said(spoken.read(surface, tongue=dialect.name)))
        return score, misses


class Polarity(Reading):
    """A denial is not its own assertion, in a language whose negator she had to find.

    Kept apart from :class:`Reading` because reading a sentence and reading its denial are not one
    ability, and the failure mode is specific and severe: an extractor that drops the negator does
    not return less, it returns **the opposite**. Measured on the shipped compiler over minted
    dialects, forty denials out of forty came back as assertions — so this is the subject that
    says whether learning the shape fixed the thing the shape was for.
    """

    id, title = "polarity", "denial, and not storing it as its opposite"
    teaches = "the word that means *not* is part of the shape"
    threshold, items = 0.8, 12

    lesson = 18
    kinds = ("assertion", "negated")


class Questions(Reading):
    """Asking is a shape too — a particle, or a word standing in the hole.

    Two forms and they make different claims. A polar question keeps every slot filled and adds a
    marker; a content question **removes** a slot and puts a question word where it was, so what
    has to be read off it is not only what was said but *which part is being asked about*. That is
    :attr:`~nyxara.njp.semantics.Meaning.focus`, and a reading that filled the missing slot with
    the question word would be answering a question with its own hole.
    """

    id, title = "questions", "asking, and knowing which part is being asked"
    teaches = "a question is a shape with a hole in a named place"
    threshold, items = 0.8, 12

    lesson = 24
    kinds = ("polar", "content", "content_subject")


class Saying(LanguageSubject):
    """Producing a sentence in that language, and refusing to produce one she cannot read back.

    Nothing is taught here, and the report says so: generation is the constructions from the three
    subjects above, run backwards. What it adds is the one thing reading cannot test — every
    sentence she utters is **parsed again before it leaves**, and unless the parse gives back the
    meaning it started from the sentence is discarded and the next candidate tried.

    So the controls are the whole point. Asked for a meaning carrying a modality nobody ever
    demonstrated, the right output is nothing at all. A faculty that reached for the nearest shape
    it had would be fluent, wrong, and indistinguishable from fluent and right.
    """

    id, title = "saying", "putting a meaning into that language, or saying nothing"
    teaches = "a shape learned by reading is a shape she can speak, without a second lesson"
    threshold, items = 0.8, 12

    lesson = 14

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """A **reading** lesson, and that is the point of it.

        The exam asks for past-tense sentences among the others, and the past is the one shape the
        three subjects above never demonstrate — so cold, she says the two tenses she has and
        stays silent on the third, which is 0.75 and is the correct 0.75. What is taught here is
        the past tense *as sentences with their meanings*, exactly as ``reading`` is taught, and
        nothing about production is demonstrated at all. If the number moves, what moved it is a
        construction crossing from comprehension into production on its own.
        """
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        shown = dialects.sample(dialect, mint.word, self.lesson, kinds=("past",))
        for utterance in shown:
            spoken.show(utterance.surface, utterance.meaning, tongue=dialect.name)
        report = spoken.learn(tongue=dialect.name)
        return Taught(len(shown), f"{len(shown)} past-tense sentences read to her; "
                                  f"{report.kept} shapes held in all, none of them a lesson "
                                  f"in speaking")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp import dialects
        spoken, dialect = self.faculty(brain), self.course(brain, mint).first
        score, misses = Score(), []
        wanted = dialects.sample(dialect, mint.word, self.items,
                                 kinds=("assertion", "negated", "past"))
        for index, utterance in enumerate(wanted):
            meaning = utterance.meaning
            if index % 4 == 3:
                meaning = dialect.meaning(utterance.meaning.subject, utterance.meaning.relation,
                                          utterance.meaning.object)
                meaning.modality = "necessary"
                question = Question(ask=f"say it, but as something that *must* be so",
                                    accept=(), silence_ok=True,
                                    note="control: a modality nobody demonstrated")
            else:
                question = Question(ask=f"say {self._said(meaning)}",
                                    accept=(utterance.surface,), exact=True)
            self.mark(score, misses, question, spoken.say(meaning, tongue=dialect.name))
        return score, misses


class Translation(LanguageSubject):
    """The same meaning, in a second language that shares no word with the first.

    A translation here is not a mapping between two surfaces — there is no phrase table and
    nothing is aligned. She reads the sentence into a
    :class:`~nyxara.njp.semantics.Meaning` and says that meaning in the other grammar, so what
    crosses is only what was never in either language to begin with. It is the sharpest test of
    the claim the whole module rests on, because a faculty that had really learned surface
    patterns rather than meanings has nothing at all to carry across.

    Two items ask which language a sentence was in, and they are deliberately built from the forms
    that differ — the particles. Two bare three-word grammars are genuinely indistinguishable on
    one sentence, and she returns ``ambiguous`` for them rather than a coin toss, which is right
    and would make a poor exam question.
    """

    id, title = "translation", "the same meaning in a language that shares no word"
    teaches = "a meaning belongs to no language, which is why it can be said in another"
    threshold, items = 0.75, 12

    lesson = 18
    kinds = ("assertion", "negated", "polar")

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        from nyxara.njp import dialects
        spoken, course = self.faculty(brain), self.course(brain, mint)
        taught = 0
        for dialect in (course.first, course.second):
            shown = dialects.sample(dialect, mint.word, self.lesson, kinds=self.kinds)
            for utterance in shown:
                spoken.show(utterance.surface, utterance.meaning, tongue=dialect.name)
            taught += len(shown)
            spoken.learn(tongue=dialect.name)
        first = len(spoken.tongue(course.first.name).grammar.constructions)
        second = len(spoken.tongue(course.second.name).grammar.constructions)
        return Taught(taught, f"two grammars held at once: {first} shapes and {second}")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp import dialects
        spoken, course = self.faculty(brain), self.course(brain, mint)
        source, target = course.first, course.second
        score, misses = Score(), []
        for index, utterance in enumerate(
                dialects.sample(source, mint.word, self.items, kinds=self.kinds)):
            if index % 6 == 4:
                # Which language was that? Asked only of the forms that carry a particle, because
                # the answer to a bare sentence genuinely is "either".
                marked = source.utter(utterance.meaning.subject, utterance.meaning.relation,
                                      utterance.meaning.object, kind="negated")
                question = Question(ask=f"which language is {marked.surface!r}?",
                                    accept=(source.name,), exact=True)
                self.mark(score, misses, question, spoken.read(marked.surface).language)
                continue
            if index % 6 == 5:
                surface = self._control(utterance.surface, mint)
                question = Question(ask=f"translate {surface!r}", accept=(), silence_ok=True,
                                    note="control: unreadable in the source")
            else:
                surface = utterance.surface
                want = target.express(utterance.meaning.subject, utterance.meaning.relation,
                                      utterance.meaning.object, kind=utterance.kind)
                question = Question(ask=f"translate {surface!r}", accept=(want,), exact=True)
            self.mark(score, misses, question,
                      spoken.translate(surface, into=target.name, frm=source.name))
        return score, misses


REASONING = (Arithmetic, Composition, Inheritance, Shapes, Abstention, Depth)
LANGUAGE = (WordShapes, WordClasses, Reading, Polarity, Questions, Saying, Translation)
CODING = (CodeReading, Tracing, WriteBasic, WriteComposite, WriteLoops, WriteRecursion,
          WriteDicts, WriteStrings, WriteStructures, WriteAlgorithms, Debugging, EdgeCases,
          Transfer)
_attach_banks()

#: The order is the claim, exactly as it is in :mod:`nyxara.njp.curriculum`. Arithmetic before
#: composition because a closed value is the simplest thing that can be right; abstention after
#: the subjects that reward "yes", so a brain that learned to say yes is caught by the next
#: subject rather than flattered by the previous one; transfer last, because it is the only
#: subject whose score is entirely a consequence of the ones before it.
#:
#: Language sits between the two halves, and the position is argued for. It is after reasoning
#: because a shape carries a *meaning* and there has to be something for a meaning to be made of;
#: it is before coding because the discipline the coding half runs on — a lesson leaves a shape,
#: never an answer — is the same discipline, and it is easier to believe about programs once it
#: has been watched working on sentences. Within it the order is likewise the claim: the shape of
#: a word, then which words behave alike, then a whole clause, then its denial, then its question
#: forms, then producing one, then producing one in a second language.
SUBJECTS = REASONING + LANGUAGE + CODING


class ExamConditions:
    """The configuration a brain is examined under, and why it is not the default one.

    :class:`~nyxara.njp.evolve.SelfEvolver` fires on a wall clock — ``evolve_every_s`` is 300 —
    and a full syllabus takes longer than that. So an unconfigured brain part-way through the
    exam starts :class:`nyxara.growth.self_optimize.Optimizer`, which benchmarks and **edits the
    package's own source** in a subprocess while the subject after it is being graded. That is
    the system working exactly as designed and it is also the one thing an examination cannot
    allow: the thing under test must not change during the test, and a score that a source
    rewrite landed in the middle of is not a measurement of anything.

    So both cadenced organs are off here. Nothing else is touched — every organ the exam actually
    reads is the one a normal brain has, and a caller who wants the evolver running during a run
    can pass their own brain to :meth:`School.attend` and get exactly that.
    """

    evolve_enabled = False
    pulse_enabled = False


class School:
    """Sits her down, finds the floor, teaches, and examines on what was never taught.

    One run is: for every subject, a **pre-test** on freshly minted items, then teaching, then a
    **post-test** on different freshly minted items, repeated up to ``rounds`` times while she is
    below the bar. Everything is reported — the floor, the gain, the rounds it took and the items
    she got wrong — because a report card that shows only the final number cannot be told apart
    from one that was taught the exam.
    """

    def __init__(self, *, seed: int = 7, rounds: int = 2,
                 subjects: Optional[Sequence[Any]] = None, verbose: bool = False) -> None:
        #: The subject instances the last :meth:`attend` used, kept so :meth:`retention` can
        #: examine the same ones rather than freshly built strangers.
        self.sat: List[Any] = []
        self.seed = int(seed)
        self.rounds = max(1, int(rounds))
        self.subjects = list(subjects if subjects is not None else SUBJECTS)
        self.verbose = bool(verbose)

    # -- one run ------------------------------------------------------------ #
    def attend(self, brain: Any = None, *, coder: Any = None) -> Transcript:
        """Teach and examine every subject in order. Returns the report card."""
        from nyxara.njp.coding import Coder

        if brain is None:
            from nyxara.njp.brain import NJPBrain
            brain = NJPBrain(ExamConditions())
        if coder is None:
            coder = getattr(brain, "coder", None) or Coder()
        started = time.time()
        transcript = Transcript(seed=self.seed)
        self.sat = [factory() if isinstance(factory, type) else factory
                    for factory in self.subjects]
        for index, subject in enumerate(self.sat):
            transcript.results.append(self._sit_subject(subject, brain, coder, index))
        transcript.seconds = time.time() - started
        try:
            transcript.coder_stats = coder.stats()
        except Exception:  # noqa: BLE001
            transcript.coder_stats = {}
        return transcript

    def _mint(self, index: int, phase: int, round_no: int = 0) -> Mint:
        """A separate stream per subject *and* per phase, so a pre-test and a post-test cannot
        share an entity even if a subject asks for them in a different order."""
        return Mint(random.Random(self.seed * 100003 + index * 1009 + phase * 31 + round_no))

    def _sit_subject(self, subject: Any, brain: Any, coder: Any, index: int) -> Result:
        started = time.time()
        result = Result(subject=subject.id, title=subject.title, teaches=subject.teaches,
                        threshold=subject.threshold)
        try:
            result.pre, misses = subject.exam(brain, self._mint(index, 0), coder=coder)
            result.misses = misses
        except Exception as exc:  # noqa: BLE001 — a subject that crashes is a failed subject
            result.note = f"pre-test raised {type(exc).__name__}: {exc}"
            result.seconds = time.time() - started
            return result
        result.post = result.pre
        for round_no in range(1, self.rounds + 1):
            if result.mastered and result.rounds:
                break
            try:
                lesson = subject.teach(brain, self._mint(index, 1, round_no), coder=coder)
                result.taught += lesson.items
                result.note = lesson.note
                result.post, result.misses = subject.exam(
                    brain, self._mint(index, 2, round_no), coder=coder)
            except Exception as exc:  # noqa: BLE001
                result.note = f"round {round_no} raised {type(exc).__name__}: {exc}"
                break
            result.rounds = round_no
            if self.verbose:
                print(f"  {subject.id:<16} round {round_no}: "
                      f"{result.pre.accuracy:.2f} → {result.post.accuracy:.2f}")
            if result.mastered:
                break
        result.seconds = time.time() - started
        return result

    # -- the teacher-off measurement ---------------------------------------- #
    def retention(self, brain: Any, coder: Any, *, seed: Optional[int] = None) -> Transcript:
        """Examine everything again on brand-new items and teach nothing.

        This is the number :mod:`nyxara.njp.teacher` says the phase comes down to. Nothing is
        taught, no budget is raised, and the items have never been seen — so what it reports is
        what survived the lesson rather than what the lesson was.
        """
        away = School(seed=self.seed if seed is None else int(seed), rounds=1,
                      subjects=self.subjects, verbose=self.verbose)
        started = time.time()
        transcript = Transcript(seed=away.seed)
        # The instances that were taught, not fresh ones. It matters for exactly one subject and
        # the reason is the whole point of the measurement: what :class:`Depth` distils is a
        # property *of a relation*, so a fresh instance would mint a new relation, examine her on
        # a structure nobody ever demonstrated, and report the floor as if it were forgetting.
        # Reusing the instance asks the honest question — new entities, the relation she was
        # actually taught.
        sat = getattr(self, "sat", None) or [f() if isinstance(f, type) else f
                                             for f in away.subjects]
        for index, subject in enumerate(sat):
            result = Result(subject=subject.id, title=subject.title, teaches=subject.teaches,
                            threshold=subject.threshold, note="teacher off")
            try:
                result.post, result.misses = subject.exam(
                    brain, away._mint(index, 3), coder=coder)
            except Exception as exc:  # noqa: BLE001
                result.note = f"raised {type(exc).__name__}: {exc}"
            result.pre = result.post  # no lesson ran, so there is no gain to claim
            transcript.results.append(result)
        transcript.seconds = time.time() - started
        try:
            transcript.coder_stats = coder.stats()
        except Exception:  # noqa: BLE001
            transcript.coder_stats = {}
        return transcript


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.school [--seed N] [--rounds N] [--json] [--quiet]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Teach NJP reasoning and coding, then examine.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="print the transcript as JSON")
    parser.add_argument("--retention", action="store_true",
                        help="re-examine afterwards with the teacher off")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.coding import Coder

    brain, coder = NJPBrain(ExamConditions()), Coder()
    school = School(seed=args.seed, rounds=args.rounds, verbose=not args.quiet and not args.json)
    transcript = school.attend(brain, coder=coder)
    if args.json:
        print(json.dumps(transcript.to_dict(), indent=2, default=str))
    else:
        print(transcript.summary())
    if args.retention:
        after = school.retention(brain, coder, seed=args.seed + 1)
        if args.json:
            print(json.dumps(after.to_dict(), indent=2, default=str))
        else:
            print("  ── teacher off, fresh items ──")
            print(after.summary())
    return 0 if not transcript.failing else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
