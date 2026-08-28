"""NYXARA · njp/school.py — reasoning and coding, taught and then examined (🎓, NJP V.09).

:mod:`nyxara.njp.curriculum` names the nine stages and refuses to report one as reached before it
is; :mod:`nyxara.njp.study` teaches her from a corpus and grades her on a held-out split;
:mod:`nyxara.njp.teacher` distils a demonstration into structure and throws its answer away. Each
of them is one third of a school and none of them is a school: there was nothing that sat her
down, worked out what she *cannot currently do*, taught that, and then examined her on material
she had never seen.

This is that, over the two subjects the Master asked for — **reasoning** and **coding** — and its
design is one claim repeated in eleven places:

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
    subjects mastered       11 / 11         11 / 11
    right / wrong / absent  107 / 0 / 3     109 / 0 / 1
    accuracy · precision    0.97 · 1.00     0.99 · 1.00

Two subjects moved because a lesson ran — ``depth`` 0.33 → 1.00 and ``code-composites``
0.17 → 1.00 — and both are gains on material that was never taught. Nine of the eleven read 1.00
cold and are printed as ``already``, which is the honest shape of this result: most of what the
syllabus asks for, she could already do, and the report says which two she could not.

It also found three bugs in the brain, which is the argument for having a school at all: a
deliberation ladder that answered ``25 + 10`` with ``10`` and out-ranked the calculator; a
promoted ``shape:p>p>p`` that shadowed the general composition walk at exactly four hops, so a
lesson that had worked looked like one that had not; and a schema ranking that tried every taught
composite before the one-attempt seed answering ``sum(xs)``, which made learning a hard thing
break an easy one. All three are fixed where they were, not worked around here.

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
    "ExamConditions",
    "REASONING", "CODING", "SUBJECTS",
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
            if r.mastered and r.already and r.gain <= 0.01:
                verdict = "already"
            elif r.mastered and r.gain > 0.01:
                verdict = "LEARNED"
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

@dataclass(frozen=True)
class Family:
    """A *kind* of programming task, and a worked solution for any instance of it.

    Teaching and examining draw from the same families and never from the same instance. That is
    the whole transfer claim in one sentence: the shape is demonstrated on ``sum of triple the
    odd ones`` and examined on ``sum of double the even ones``, over different data, and the only
    thing carried between them is the skeleton :func:`~nyxara.njp.coding.abstract` kept.
    """

    id: str = ""
    intent: str = ""
    build: Any = None          # (rng, name) -> (ref, source)
    composite: bool = True


def _int_data(rng: random.Random, rows: int = 10, size: int = 6,
              lo: int = 0, hi: int = 20) -> List[Tuple[int, ...]]:
    return [tuple(rng.randrange(lo, hi) for _ in range(size)) for _ in range(rows)]


def _spec_from(name: str, intent: str, ref: Any, rows: Sequence[Tuple[int, ...]]) -> Any:
    from nyxara.njp.coding import Spec
    keep = [row for row in rows if _safe(ref, row) is not _FAILED]
    # Six shown, four held out. Four shown was not enough: `sum(x for x in xs if x > 3)` fitted
    # four pairs of a task whose threshold was 9 and failed the fifth, which is the coincidence
    # the split exists to catch — but a split that catches it *often* is a split whose training
    # half is too small, and catching it is more expensive than not producing it.
    return Spec.of(name, intent, ["xs"],
                   [([row], ref(row)) for row in keep[:6]],
                   [([row], ref(row)) for row in keep[6:]])


_FAILED = object()


def _safe(ref: Any, row: Tuple[int, ...]) -> Any:
    """A reference implementation that raises on an unlucky row (``max`` of an empty filter) is
    not a broken family — it is a row that cannot be an example, and it is dropped."""
    try:
        return ref(row)
    except Exception:  # noqa: BLE001
        return _FAILED


def _f_sum_mapped_filtered(rng: random.Random, name: str) -> Tuple[Any, str]:
    k, remainder = rng.choice([2, 3, 4]), rng.choice([0, 1])
    ref = lambda xs: sum(x * k for x in xs if x % 2 == remainder)  # noqa: E731
    return ref, (f"def {name}(xs):\n"
                 f"    return sum([x * {k} for x in xs if x % 2 == {remainder}])")


def _f_count_over(rng: random.Random, name: str) -> Tuple[Any, str]:
    threshold = rng.choice([5, 8, 10, 12])
    ref = lambda xs: len([x for x in xs if x > threshold])  # noqa: E731
    return ref, (f"def {name}(xs):\n"
                 f"    return len([x for x in xs if x > {threshold}])")


def _f_sum_over(rng: random.Random, name: str) -> Tuple[Any, str]:
    threshold = rng.choice([4, 7, 9, 11])
    ref = lambda xs: sum(x for x in xs if x > threshold)  # noqa: E731
    return ref, (f"def {name}(xs):\n"
                 f"    return sum([x for x in xs if x > {threshold}])")


def _f_max_mapped(rng: random.Random, name: str) -> Tuple[Any, str]:
    k = rng.choice([2, 3, 5])
    ref = lambda xs: max(x * k for x in xs)  # noqa: E731
    return ref, f"def {name}(xs):\n    return max([x * {k} for x in xs])"


def _f_sum_mapped(rng: random.Random, name: str) -> Tuple[Any, str]:
    k = rng.choice([2, 3, 4])
    ref = lambda xs: sum(x + k for x in xs)  # noqa: E731
    return ref, f"def {name}(xs):\n    return sum([x + {k} for x in xs])"


def _f_nth_smallest(rng: random.Random, name: str) -> Tuple[Any, str]:
    index = rng.choice([0, 1, 2])
    ref = lambda xs: sorted(xs)[index]  # noqa: E731
    return ref, f"def {name}(xs):\n    return sorted(xs)[{index}]"


def _f_descending(rng: random.Random, name: str) -> Tuple[Any, str]:
    ref = lambda xs: tuple(sorted(xs)[::-1])  # noqa: E731
    return ref, f"def {name}(xs):\n    return sorted(xs)[::-1]"


def _f_distinct_count(rng: random.Random, name: str) -> Tuple[Any, str]:
    ref = lambda xs: len(tuple(dict.fromkeys(xs)))  # noqa: E731
    return ref, f"def {name}(xs):\n    return len(list(dict.fromkeys(xs)))"


#: Shapes she is not seeded with. Every one of them is two or three operators deep, which is
#: exactly the class :data:`~nyxara.njp.coding.SEED_SHAPES` cannot reach without grafting.
COMPOSITE_FAMILIES: Tuple[Family, ...] = (
    Family("sum_mapped_filtered", "add up a multiple of the ones that pass a test",
           _f_sum_mapped_filtered),
    Family("count_over", "how many are over a threshold", _f_count_over),
    Family("sum_over", "add up only the ones over a threshold", _f_sum_over),
    Family("max_mapped", "the largest after a transform", _f_max_mapped),
    Family("sum_mapped", "add up a shifted copy of each", _f_sum_mapped),
    Family("nth_smallest", "the n-th smallest", _f_nth_smallest),
)

#: One operator deep. She can reach these from :data:`~nyxara.njp.coding.SEED_SHAPES` alone, and
#: the subject that uses them is a floor rather than a lesson.
BASIC_FAMILIES: Tuple[Family, ...] = (
    Family("total", "add them up", lambda rng, n: (sum, f"def {n}(xs):\n    return sum(xs)"),
           composite=False),
    Family("count", "how many", lambda rng, n: (len, f"def {n}(xs):\n    return len(xs)"),
           composite=False),
    Family("largest", "the biggest one",
           lambda rng, n: (max, f"def {n}(xs):\n    return max(xs)"), composite=False),
    Family("ordered", "put them in order",
           lambda rng, n: ((lambda xs: tuple(sorted(xs))),
                           f"def {n}(xs):\n    return sorted(xs)"), composite=False),
    Family("backwards", "reverse them",
           lambda rng, n: ((lambda xs: tuple(xs[::-1])),
                           f"def {n}(xs):\n    return xs[::-1]"), composite=False),
    Family("evens", "keep the even ones",
           lambda rng, n: ((lambda xs: tuple(x for x in xs if x % 2 == 0)),
                           f"def {n}(xs):\n    return [x for x in xs if x % 2 == 0]"),
           composite=False),
)


def _instance(family: Family, rng: random.Random, name: str) -> Tuple[Any, str]:
    """One task from a family: a spec with a shown/held-out split, and its worked solution."""
    ref, source = family.build(rng, name)
    spec = _spec_from(name, family.intent, ref, _int_data(rng))
    return spec, source


#: Human Python, and what it should come back as. The reader is under test here, not the brain —
#: each entry is a construction a person actually writes, and the expected value is computed by
#: the reference lambda rather than typed in, because a hand-computed table is a table with a
#: mistake in it.
_READING: Tuple[Tuple[str, Any, Tuple[Any, ...]], ...] = (
    ("def f(xs):\n    return sum(xs) + len(xs)",
     lambda xs: sum(xs) + len(xs), ((3, 1, 4, 1, 5),)),
    ("def f(xs):\n    return [x * x for x in xs]",
     lambda xs: tuple(x * x for x in xs), ((2, 3, 4),)),
    ("def f(xs):\n    return [x for x in xs if x % 3 == 0]",
     lambda xs: tuple(x for x in xs if x % 3 == 0), ((3, 4, 9, 10),)),
    ("def f(xs):\n    return sorted(xs)[::-1]",
     lambda xs: tuple(sorted(xs)[::-1]), ((5, 2, 9),)),
    ("def f(xs):\n    return xs[1:]", lambda xs: tuple(xs[1:]), ((7, 8, 9),)),
    ("def f(xs):\n    return xs[:2]", lambda xs: tuple(xs[:2]), ((7, 8, 9),)),
    ("def f(n):\n    return n if n > 0 else -n", lambda n: n if n > 0 else -n, (-6,)),
    ("def f(a, b):\n    return max(a, b) - min(a, b)",
     lambda a, b: max(a, b) - min(a, b), (4, 11)),
    ("def f(xs):\n    return len([x for x in xs if x > 2]) * 10",
     lambda xs: len([x for x in xs if x > 2]) * 10, ((1, 2, 3, 4),)),
    ("def f(s):\n    return len(s.split())", lambda s: len(s.split()), ("do the thing now",)),
    ("def f(s):\n    return s.upper()", lambda s: s.upper(), ("nyxara",)),
    ("def f(xs):\n    return 3 in xs", lambda xs: 3 in xs, ((1, 2, 4),)),
    ("lambda xs: sum([x % 2 for x in xs])",
     lambda xs: sum(x % 2 for x in xs), ((1, 2, 3, 4, 5),)),
    ("def f(xs):\n    return sum(x * 2 for x in xs if x > 1)",
     lambda xs: sum(x * 2 for x in xs if x > 1), ((1, 2, 3),)),
)

#: Source a reader must **refuse**. Half of reading code is knowing what you are not going to run.
_REFUSE: Tuple[str, ...] = (
    "def f(x):\n    return __import__('os').system('ls')",
    "def f(x):\n    return open('/etc/passwd').read()",
    "def f(x):\n    return eval('2+2')",
    "def f(x):\n    return x.__class__.__bases__",
    "def f(x):\n    for i in range(x):\n        x += i\n    return x",
    "def f(x):\n    return {'a': 1}",
)


class CodeReading(Subject):
    """Read a person's Python, run it in her own machine, and say what it produces.

    Two halves and both count: fourteen constructions she must read and evaluate exactly, and six
    she must **refuse** — an import, a file open, an ``eval``, a dunder walk, a loop, a dict. A
    reader that accepts the first list and rejects the second is doing the job; one that accepts
    everything is a security hole with a good score.
    """

    id, title = "code-reading", "reading Python and predicting its result"
    teaches = "what a program says, before anything is written"
    threshold = 0.9

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import CodeError, read_python
        score, misses = Score(), []
        for source, ref, args in _READING:
            first = source.strip().splitlines()[0][:44]
            try:
                program = read_python(source)
                got = coder.run(program, args)
                want = ref(*args)
                if isinstance(want, list):
                    want = tuple(want)
                score.add("right" if got == want else "wrong")
                if got != want:
                    misses.append(f"{first} → {got!r}, wanted {want!r}")
            except CodeError as exc:
                score.add("wrong")
                misses.append(f"{first} → refused readable code: {exc}")
        for source in _REFUSE:
            first = source.strip().splitlines()[0][:44]
            try:
                read_python(source)
            except CodeError:
                score.add("right")
                continue
            score.add("wrong")
            misses.append(f"{first} → accepted source it should refuse")
        return score, misses


class WriteBasic(Subject):
    """One-operator programs from examples. Her floor, and it is meant to read high.

    :data:`~nyxara.njp.coding.SEED_SHAPES` covers these by construction, so a low score here is a
    broken search rather than a missing lesson — which is precisely why it is examined before the
    subject that needs teaching, and why its threshold is the highest in the syllabus.
    """

    id, title = "code-basics", "writing one-operator programs from examples"
    teaches = "a specification is a set of examples, and a program either matches them or does not"
    threshold, items = 0.85, 6
    attempts = 4000

    families = BASIC_FAMILIES

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for index, family in enumerate(self.families[:self.items]):
            spec, _ = _instance(family, mint.rng, f"{family.id}_{mint.word(1)}")
            written = coder.write(spec, attempts=self.attempts, graft=False)
            score.add(self._grade(coder, spec, written, misses))
        return score, misses

    @staticmethod
    def _grade(coder: Any, spec: Any, written: Any, misses: List[str]) -> str:
        """Passing the shown examples is not passing. The held-out ones decide.

        A program fitted to four pairs that fails a fifth found a coincidence, and the whole
        reason the split exists is to call that what it is rather than to count it as a solve.
        """
        if not written.ok:
            misses.append(f"{spec.name}: abstained after {written.attempts} attempts")
            return "abstain"
        check = coder.check(written.program, spec.held_out)
        if check.ok:
            return "right"
        misses.append(f"{spec.name}: fitted the shown pairs, failed held-out "
                      f"({check.passed}/{check.total}) — {written.program.source().splitlines()[-1].strip()}")
        return "wrong"


class WriteComposite(Subject):
    """Two- and three-operator programs — the subject the whole module is built around.

    Grafting is switched **off** for both the pre-test and the post-test, and the attempt budget
    is identical across them. So exactly one thing differs between the two numbers: which shapes
    she holds. That is what makes the gain attributable to the lesson rather than to the budget,
    and it is why the cold column here is usually zero.
    """

    id, title = "code-composites", "writing composed programs from examples"
    teaches = "a shape seen once on one task is worth having on another"
    threshold, items = 0.6, 6
    attempts = 30000

    families = COMPOSITE_FAMILIES

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for family in self.families[:self.items]:
            spec, _ = _instance(family, mint.rng, f"{family.id}_{mint.word(1)}")
            written = coder.write(spec, attempts=self.attempts, graft=False)
            score.add(WriteBasic._grade(coder, spec, written, misses))
        return score, misses

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        from nyxara.njp.teacher import Verdict
        taught, notes = 0, []
        for family in self.families[:self.items]:
            spec, source = _instance(family, mint.rng, f"{family.id}_{mint.word(1)}")
            learned = coder.learn_python(spec, source)
            if learned.verdict == Verdict.SURVIVED:
                taught += 1
            else:
                notes.append(f"{family.id}: {learned.verdict} — {learned.why}")
        note = f"{taught} shapes verified by execution"
        if notes:
            note += "; " + "; ".join(notes[:2])
        return Taught(taught, note)


class Debugging(Subject):
    """A program with one thing wrong with it, and a fix that has to run before it counts.

    The corruption is checked to actually break the program before she is asked to repair it —
    an "exercise" a wrong program passes anyway is not an exercise — and the repair is graded on
    the **held-out** pairs, so restoring the shown examples by luck does not count as fixed.
    """

    id, title = "debugging", "finding and fixing one wrong thing"
    teaches = "localise the fault by running it, not by looking at it"
    threshold, items = 0.6, 6
    attempts = 20000

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        from nyxara.njp.coding import read_python
        rng = mint.rng
        score, misses = Score(), []
        for family in COMPOSITE_FAMILIES[:self.items]:
            spec, source = _instance(family, rng, f"{family.id}_{mint.word(1)}")
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
    threshold, items = 0.5, 6
    attempts = 8000

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """Nothing, and that is the measurement. Anything taught here would spoil the number."""
        return Taught(0, "teacher off — this score is what the earlier subjects left behind")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for family in COMPOSITE_FAMILIES[:self.items]:
            spec, _ = _instance(family, mint.rng, f"{family.id}_{mint.word(1)}")
            written = coder.write(spec, attempts=self.attempts, graft=False)
            score.add(WriteBasic._grade(coder, spec, written, misses))
        return score, misses


# --------------------------------------------------------------------------- #
# the syllabus and the school
# --------------------------------------------------------------------------- #

REASONING = (Arithmetic, Composition, Inheritance, Shapes, Abstention, Depth)
CODING = (CodeReading, WriteBasic, WriteComposite, Debugging, Transfer)

#: The order is the claim, exactly as it is in :mod:`nyxara.njp.curriculum`. Arithmetic before
#: composition because a closed value is the simplest thing that can be right; abstention after
#: the subjects that reward "yes", so a brain that learned to say yes is caught by the next
#: subject rather than flattered by the previous one; transfer last, because it is the only
#: subject whose score is entirely a consequence of the ones before it.
SUBJECTS = REASONING + CODING


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
