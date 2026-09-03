"""NYXARA · njp/discourseschool.py — teaching her to be talked to (🎓, NJP V.26).

:mod:`nyxara.njp.school` teaches her a language: what a word is made of, what a word class is,
how to read a sentence and how to say one back. Every subject in it stops at the sentence. This
one starts there, and examines the half of language that is about **people**: what somebody was
doing by saying that, what "he" names, whether what they just said can be true at the same time as
what they said twenty turns ago, what they believe that is not so, and how much of an answer this
particular hearer has asked to carry.

**Twelve subjects, and they are three kinds.**

*Eight are taught.* ``acts``, ``transfer``, ``repair``, ``figurative``, ``attachment``,
``reference``, ``contradiction`` and ``memory`` have a floor, a lesson and a ceiling, and the
number that matters on them is the **gain**: a convention that could be read before it was
demonstrated was not learned here, it was shipped. Every item is minted, so a lesson and an exam
cannot share a word.

**Four of those eight were floors when this syllabus was first written, and that was a report card
telling the truth about the wrong thing.** ``contradiction``, ``memory``, ``reference`` and
``attachment`` read 1.00 cold because the module held hand-written tables — which words license an
update, which quantify over all times, which prepositions attach two ways, what each resolution cue
is worth. Those are not closed-class tables like the pronouns: they are semantic claims about
particular words in one language, and a module that ships them has not learned anything, it has
been told. With the tables gone and the same facts induced from demonstrated verdicts, the four
have floors again and the gain is the measurement.

*Two are floors.* ``minds`` and ``register`` are mechanisms rather than conventions — there is
nothing to demonstrate, and they read their ceiling cold and are printed with ``already`` beside
them exactly as :class:`~nyxara.njp.school.Arithmetic` is. What they are worth is what that is
worth: an organ that quietly stops working shows up here on the first run rather than three
versions later as an unexplained dip.

*Two are controls on the other nine.* ``tongue`` asks whether a convention taught in one language
leaks into another — and the answer this school reports is **no**, because an indirect request
really is language-specific and a school that scored a leak as a pass would be rewarding exactly
the wrong thing. ``wiring`` is the only subject that goes through ``brain.think()``, because an
organ measured in isolation reports on the organ: V.25 found two defects that were invisible
until the brain was in the loop, and this subject exists so that this version's are not.

**Controls are half of every subject.** A subject examined only on sentences whose answer is
"request" cannot tell a reader of conventions from a brain that says "request". So ``acts`` is
examined on shapes nobody demonstrated, where the literal mood is the right answer; ``reference``
on discourses that genuinely do not settle their pronoun, where **ambiguous** is the right answer
and a confident name is wrong; ``contradiction`` on pairs of claims that have nothing to do with
each other, where ``new`` is right and a conflict is a false alarm.

Run it: ``python -m nyxara.njp.discourseschool``, or ``NJPBrain.go_to_discourse_school()``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.discourse import REGISTERS, Communicator, Referent, attachment
from nyxara.njp.school import ExamConditions, Mint, Score, Subject, Taught, Transcript
from nyxara.njp.semantics import compile_meaning

__all__ = [
    "DiscourseSubject", "Acts", "Transfer", "Repair", "ReferenceSubject", "Contradiction",
    "LongMemory", "OtherMinds", "RegisterSubject", "FigurativeSubject", "Attachment",
    "Anticipating", "Tongue", "Wiring", "DiscourseSchool", "SUBJECTS", "main",
]


# --------------------------------------------------------------------------- #
# shared machinery
# --------------------------------------------------------------------------- #

class DiscourseSubject(Subject):
    """One thing she is meant to be able to do with a conversation.

    Graded against the organ rather than through ``brain.think()``, for the reason
    :class:`~nyxara.njp.school.LanguageSubject` gives: most of these items are not *questions* and
    could not be phrased as ones. ``wiring`` is the exception and exists precisely to close that
    gap.
    """

    items = 10
    threshold = 0.8

    @staticmethod
    def voice(brain: Any) -> Communicator:
        """Her communicator, attached to the brain if the brain has not got one.

        Attached rather than built per call: what a lesson leaves has to still be there for the
        post-test and for the retention run, and a communicator built fresh for each exam would
        measure nothing but the exam.
        """
        spoken = getattr(brain, "discourse", None)
        if isinstance(spoken, Communicator):
            return spoken
        spoken = Communicator(kinds=_kinds_of(brain))
        try:
            brain.discourse = spoken
        except Exception:  # noqa: BLE001
            pass
        return spoken

    @staticmethod
    def private(brain: Any) -> Communicator:
        """A student this syllabus has not taught yet.

        For the one subject that needs it. ``memory`` and ``contradiction`` are two questions
        about the same organ, so whichever sits second inherits the other's lesson and reports a
        floor of 1.00 — the ``transfer`` / ``acts`` contamination again, and it is not worth
        having twice. Giving ``memory`` its own communicator makes both floors honest; it costs
        the shared-student property, which that subject does not need because what it examines is
        a conversation from end to end.
        """
        _ = brain
        return Communicator()

    @staticmethod
    def fresh(brain: Any) -> Communicator:
        """A new conversation with the same learned conventions — see
        :meth:`~nyxara.njp.discourse.Communicator.reset`."""
        spoken = DiscourseSubject.voice(brain)
        spoken.reset()
        return spoken

    # -- grading ------------------------------------------------------------- #
    @staticmethod
    def mark(score: Score, misses: List[str], *, got: Any, want: Any, item: str,
             silence: Any = None) -> None:
        """Three outcomes, never two. ``silence`` is the value that counts as an abstention."""
        if got == silence:
            score.add("abstain")
            misses.append(f"{item} → silent, wanted {want!r}")
            return
        if got == want:
            score.add("right")
            return
        score.add("wrong")
        misses.append(f"{item} → {got!r}, wanted {want!r}")


def _kinds_of(brain: Any):
    """The brain's own ``is_a`` store as a kinds oracle, or nothing.

    Nothing is the honest default: :class:`~nyxara.njp.discourse.Figure` claims no sentence is
    figurative when it has no evidence, and a school that handed it a hard-coded animacy table
    would be measuring the table.
    """
    reader = getattr(brain, "_kinds_of", None)
    if callable(reader):
        return reader
    return None


def _plural(mint: Mint) -> str:
    """A minted word whose plural a **shape** rule can actually see.

    ``razius`` is a perfectly good nonsense word and ``razius`` is not a plural to any rule that
    reads endings: ``-us`` is a Latin singular, and :func:`~nyxara.njp.discourse._is_plural`
    excludes it deliberately. Minted without this, one item in twelve failed every sitting and
    what it measured was the exam's own vocabulary rather than her number agreement — the same
    defect :class:`~nyxara.njp.school.Mint` records having had once before, in digits.
    """
    from nyxara.njp.discourse import _is_plural

    for _ in range(24):
        word = mint.word() + "s"
        if _is_plural(word):
            return word
    return mint.word() + "as"


# --------------------------------------------------------------------------- #
# taught · the conventions
# --------------------------------------------------------------------------- #

class Acts(DiscourseSubject):
    """*"Can you open the window?"* is a request, and nothing in the sentence says so.

    The floor is what the sentence's own mood gives — ``ability-question`` — and it is the *right*
    reading of the words. What the lesson adds is the convention, and the exam is on sentences
    whose every open-class word was minted after the lesson finished.
    """

    id = "acts"
    title = "what was done by saying it"
    teaches = "an indirect request, generalised from demonstrations rather than listed"
    items = 12

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        shown = 0
        for _ in range(3):
            voice.show(f"Can you {mint.word()} the {mint.word()}?", "request")
            shown += 1
        for _ in range(2):
            voice.show(f"Could you {mint.word()} the {mint.word()}?", "request")
            shown += 1
        # The counter-demonstration, and it is not optional. Without it the loosest shape
        # (`MODAL PRON + ?`) has support for "request" and nothing against it, so *every* modal
        # question reads as a request — including the ones that really are about ability.
        for _ in range(2):
            voice.show(f"Can you {mint.word()}?", "ability-question")
            shown += 1
        return Taught(shown, "five requests and two ability questions, all minted")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for _ in range(4):
            item = f"Can you {mint.word()} the {mint.word()}?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="request", item=item)
        for _ in range(3):
            item = f"Could you {mint.word()} the {mint.word()}?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="request", item=item)
        for _ in range(2):
            item = f"Can you {mint.word()}?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="ability-question", item=item)
        # Controls: shapes nobody demonstrated. The right answer is the sentence's own mood, and
        # a subject that scored only the requests could not tell a reader of conventions from a
        # brain that has learned to say "request".
        for _ in range(2):
            item = f"The {mint.word()} {mint.word()} the {mint.word()}."
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="assertion", item=item)
        item = f"What {mint.word()} the {mint.word()}?"
        self.mark(score, misses, got=voice.acts.read(item).intended, want="question", item=item)
        return score, misses


class Transfer(DiscourseSubject):
    """A modal she was never shown, inheriting a convention she was.

    Tier 3 of the Master's ladder — a novel combination rather than a novel wording. ``can`` and
    ``could`` are demonstrated; ``would``, ``will``, ``might`` and ``should`` are examined, and
    none of them appears in any lesson. What carries across is the ``tags`` shape, where every
    modal is the same token.

    The construction is deliberately **not** the one :class:`Acts` uses. Sharing it would have
    made this subject's floor 1.00 — every item already covered by the previous subject's
    lesson — and a floor of 1.00 is not a transfer measurement, it is a report that the
    measurement was taken too late.
    """

    id = "transfer"
    title = "a modal she was never shown"
    teaches = "a convention generalising over the closed class it was demonstrated with"
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        for _ in range(2):
            voice.show(f"Can you {mint.word()} me?", "request")
            voice.show(f"Could you {mint.word()} me?", "request")
        return Taught(4, "requests, with only two of the six modals ever demonstrated")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for modal in ("Would", "Will", "Might", "Should", "May", "Shall"):
            item = f"{modal} you {mint.word()} me?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="request", item=item)
        # Controls in the other direction: an auxiliary is not a modal, and the shape that carries
        # the convention is the one with a modal in it.
        for aux in ("Did", "Has"):
            item = f"{aux} you {mint.word()} me?"
            got = voice.acts.read(item).intended
            self.mark(score, misses, got=(got != "request"), want=True, item=item)
        return score, misses


class Repair(DiscourseSubject):
    """A misreading, corrected once, and the correction generalising to sentences it never saw.

    The Master's *communication failure → learning*, and the number is deliberately not "did the
    corrected sentence come back right" — it would, by memorising it. What is scored is whether
    **other sentences of the same shape** came back right, which is the only thing that separates
    a learned convention from a patch.
    """

    id = "repair"
    title = "learning from a misunderstanding"
    teaches = "a correction that lands on the shape rather than on the sentence"
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        # Two failures of the same shape, corrected. Two rather than one because one
        # demonstration is a sentence: the shape is only kept once two of them disagree in a
        # filler, which is this package's oldest rule about what a lesson is worth.
        repaired = 0
        for _ in range(2):
            item = f"Is the {mint.word()} {mint.word()} to you?"
            voice.misread(item, took_as="polar-question", meant="offer")
            repaired += 1
        # And the boundary, taught the same way. Without it the loosest shape — ``AUX DET + ?`` —
        # carries "offer" unopposed, so *every* question opening on an auxiliary and a determiner
        # reads as one. That is the same over-generalisation :class:`Acts` counter-demonstrates
        # against, and a correction that fixes one sentence by breaking a class of others is not
        # a repair.
        for _ in range(2):
            voice.show(f"Is the {mint.word()} {mint.word()}?", "polar-question")
        return Taught(repaired + 2, f"{repaired} corrections, "
                                    f"{voice.acts.generalised} of which generalised, "
                                    f"and 2 demonstrations of the boundary")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for _ in range(6):
            item = f"Is the {mint.word()} {mint.word()} to you?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="offer", item=item)
        for _ in range(2):
            item = f"Is the {mint.word()} {mint.word()}?"
            got = voice.acts.read(item).intended
            self.mark(score, misses, got=(got != "offer"), want=True, item=item)
        return score, misses


class FigurativeSubject(DiscourseSubject):
    """*"The market swallowed the shock."* — and not one word of it is about digestion.

    Taught by witnessing: several literal subjects of one relation, and the kinds they share. The
    exam is a subject that does not share the kind, and — half the items — subjects that do,
    where flagging is a **false alarm** and scores wrong. Before the witnesses, nothing is
    figurative and every item in both halves comes back literal: the floor here is not ignorance,
    it is the correct answer to half the paper.
    """

    id = "figurative"
    title = "true without being literal"
    teaches = "a selectional violation read off what the store has actually witnessed"
    items = 8

    def _world(self, brain: Any, mint: Mint) -> Tuple[str, str, str]:
        """The relation and the two kinds, minted once and then held.

        The **ontology** is stable across the pre-test, the lesson and the post-test, and the
        **subjects** are drawn fresh from each sitting's own mint. Both halves matter: a subject
        whose kinds changed between sittings would be measuring a different world each time, and
        one whose items repeated would let the floor and the ceiling be the same paper.
        """
        found = getattr(self, "world", None)
        if found is None:
            found = (mint.word(), mint.word(), mint.word())
            self.world = found                   # noqa: attribute defined outside __init__
            self.table: Dict[str, List[str]] = {}
            voice = self.voice(brain)
            voice.figure.kinds = lambda name: self.table.get(str(name).lower(), [])
        return found

    def _named(self, mint: Mint, kind: str) -> str:
        name = mint.word()
        self.table[name] = [kind]
        return name

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        relation, kind_a, _kind_b = self._world(brain, mint)
        for _ in range(3):
            voice.figure.witness(relation, self._named(mint, kind_a))
        return Taught(3, f"three witnessed subjects of {relation!r}, all of one kind")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        relation, kind_a, kind_b = self._world(brain, mint)
        score, misses = Score(), []
        for _ in range(4):                       # the wrong kind of subject entirely
            item = f"The {self._named(mint, kind_b)} {relation} the {mint.word()}."
            got = voice.figure.judge(compile_meaning(item)).figurative
            self.mark(score, misses, got=got, want=True, item=item)
        for _ in range(4):                       # and the right kind, where flagging is a lie
            item = f"The {self._named(mint, kind_a)} {relation} the {mint.word()}."
            got = voice.figure.judge(compile_meaning(item)).figurative
            self.mark(score, misses, got=got, want=False, item=item)
        return score, misses


# --------------------------------------------------------------------------- #
# floors · the mechanisms
# --------------------------------------------------------------------------- #

class ReferenceSubject(DiscourseSubject):
    """*"He"*, resolved where the discourse settles it and refused where it does not.

    Both halves are the subject. Three shapes settle a pronoun on structure alone — a lone
    candidate, number disagreement, and a pronoun in object position that cannot name its own
    clause's subject — and one shape genuinely does not, where **ambiguous** is the right answer
    and any confident name is wrong however plausible it sounds.
    """

    id = "reference"
    title = "what 'he' names, or that nothing says"
    teaches = "the cue weights, fitted from resolutions rather than tuned by hand"
    items = 15

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """Demonstrated resolutions, with recency held constant so a cue has to do the work.

        **Both kinds of case are in here.** On discourses that all resolve, a setting that never
        abstains fits perfectly and nothing rules it out; one discourse that settles on nothing
        rules the whole family out. The ambiguous cases are half the lesson for exactly the reason
        they are half the exam.
        """
        voice = self.voice(brain)
        cases = []
        for _ in range(3):                       # one candidate: resolves whatever the weights
            one = mint.word()
            cases.append(([Referent(one, 1, "subject")], "he", "subject", "", one))
        for _ in range(3):                       # two, tied on recency: nothing may settle it
            first, second = mint.word(), mint.word()
            cases.append(([Referent(first, 1, "subject"), Referent(second, 1, "object")],
                          "he", "subject", "", ""))
        for _ in range(3):                       # two, tied on recency, one of them the topic
            first, second = mint.word(), mint.word()
            cases.append(([Referent(first, 1, "subject", False, 2),
                           Referent(second, 1, "object")], "he", "subject", "", first))
        fitted = voice.fit_reference(cases)
        return Taught(len(cases), f"cues {fitted['cues']}, margin {fitted['margin']} "
                                  f"({fitted['fitted']:.2f} of the demonstrations)")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for _ in range(3):                       # the topic, against a rival tied on recency
            first, second = mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{first} met {second}.")
            voice.reference.referents[0].mentions = 2
            got = voice.hear("He was tired.").resolutions
            name = got[0].referent if got else ""
            self.mark(score, misses, got=name, want=first,
                      item=f"{first} (topic) vs {second} / He was tired.")
        for _ in range(3):                       # a lone candidate
            one = mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{one} left.")
            got = voice.hear("He was tired.").resolutions
            name = got[0].referent if got else ""
            self.mark(score, misses, got=name, want=one, item=f"{one} left. / He was tired.")
        for _ in range(3):                       # two candidates and nothing to choose between
            first, second = mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{first} met {second}.")
            got = voice.hear("He was tired.").resolutions
            settled = got[0].ambiguous if got else False
            self.mark(score, misses, got=settled, want=True,
                      item=f"{first} met {second}. / He was tired.")
        for _ in range(3):                       # the pronoun cannot name its own subject
            first, second = mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{first} met {second}.")
            got = voice.hear(f"{second} gave him the {mint.word()}.").resolutions
            name = next((r.referent for r in got if r.pronoun == "him"), "")
            self.mark(score, misses, got=name, want=first,
                      item=f"{second} gave him …")
        for _ in range(3):                       # number disagreement is a filter, not a score
            one, many = mint.word(), _plural(mint)
            voice = self.fresh(brain)
            voice.hear(f"{one} met the {many}.")
            got = voice.hear("They left.").resolutions
            name = got[0].referent if got else ""
            self.mark(score, misses, got=name, want=many,
                      item=f"{one} met the {many}. / They left.")
        return score, misses


class Contradiction(DiscourseSubject):
    """A denial, an update and a repetition, told apart by what is in the sentence.

    Four verdicts and each one is a different thing to get wrong: calling an update a
    contradiction makes her argumentative, calling a contradiction an update makes her a store
    that silently overwrites its own evidence, and calling either of them *new* makes her a store
    that holds both and notices nothing.
    """

    id = "contradiction"
    title = "a denial is not a change of mind"
    teaches = "the words that license an update and the words that refuse one, induced"
    items = 12

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """Demonstrated verdicts, and the variation is the mechanism.

        Every pair varies its preposition and its content, because the marker is induced as *what
        differs between the two sentences* and anything that differs by accident has to be
        contested away. Three ``updates`` pairs whose prepositions all went ``in`` → ``on`` would
        teach that ``on`` licenses a change.
        """
        voice = self.voice(brain)
        shown = 0
        for here, there in (("in", "on"), ("on", "under"), ("under", "beside")):
            what = mint.word()
            voice.show_change(f"The {what} is {here} the {mint.word()}.",
                              f"The {what} is now {there} the {mint.word()}.", "updates")
            shown += 1
        for _ in range(3):
            who, where = mint.word(), mint.word()
            voice.show_change(f"{who} never visited {where}.",
                              f"When {who} visited {where} last year {who} was tired.",
                              "contradicts")
            shown += 1
        # Negative evidence, and it is not optional: without a contradiction that carries no
        # universal, whatever happens to differ in the pairs above is credited with being one.
        for here, there in (("in", "on"), ("beside", "under")):
            what = mint.word()
            voice.show_change(f"The {what} is {here} the {mint.word()}.",
                              f"The {what} is {there} the {mint.word()}.", "contradicts")
            shown += 1
        for _ in range(2):
            said = f"{mint.word()} owns the {mint.word()}."
            voice.show_change(said, said, "corroborates")
            shown += 1
        kept = voice.markers.stats()["kept"]
        return Taught(shown, f"induced {kept or 'nothing'} from demonstrated verdicts")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for _ in range(3):                       # a universal, then an instance of it
            who, where = mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{who} never visited {where}.")
            got = voice.hear(f"When {who} visited {where} last year {who} was tired.")
            self.mark(score, misses, got=got.verdict.kind if got.verdict else "",
                      want="contradicts", item=f"{who} never visited {where} / … visited …")
        for _ in range(3):                       # a marked change is an update
            what, first, second = mint.word(), mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"The {what} is in the {first}.")
            got = voice.hear(f"The {what} is now on the {second}.")
            self.mark(score, misses, got=got.verdict.kind if got.verdict else "",
                      want="updates", item=f"{what}: {first} → now {second}")
        for _ in range(3):                       # the same thing, said again
            who, what = mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"{who} owns the {what}.")
            got = voice.hear(f"{who} owns the {what}.")
            self.mark(score, misses, got=got.verdict.kind if got.verdict else "",
                      want="corroborates", item=f"{who} owns {what}, twice")
        for _ in range(3):                       # a control: two claims with nothing to do
            voice = self.fresh(brain)            # with each other are not a conflict
            voice.hear(f"{mint.word()} owns the {mint.word()}.")
            got = voice.hear(f"{mint.word()} owns the {mint.word()}.")
            self.mark(score, misses, got=got.verdict.kind if got.verdict else "",
                      want="new", item="two unrelated claims")
        return score, misses


class LongMemory(DiscourseSubject):
    """The Master's four-turn test: establish, distract, revise, ask.

    The point is the third turn. A store that answers from the first mention has not tracked
    anything, and one that answers from the last has not noticed the difference between a
    correction and a change. Half the items revise with a change marker, where the later value is
    the answer; half revise without one, where the honest answer is **nothing**, because two
    claims are contesting and returning either of them is a coin flip presented as a fact.
    """

    id = "memory"
    title = "four turns, and the fact is in the third"
    teaches = "belief tracked across a conversation rather than retrieved from its first mention"
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        """The same demonstrations :class:`Contradiction` gives, to a student it has not taught.

        Its own lesson rather than the previous subject's: a syllabus where one subject silently
        inherits another's teaching cannot report a floor for the second, which is the defect
        ``transfer`` had against ``acts`` and it is not worth having twice. See
        :meth:`DiscourseSubject.private` for what that costs.
        """
        return Contradiction().teach(_Student(self.student), mint, coder=coder)

    @property
    def student(self) -> Communicator:
        found = getattr(self, "_voice", None)
        if found is None:
            found = Communicator()
            self._voice = found              # noqa: attribute defined outside __init__
        return found

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for _ in range(4):
            what, first, second = mint.word(), mint.word(), mint.word()
            voice = self.fresh(_Student(self.student))
            voice.hear(f"The {what} is in the {first}.")
            voice.hear(f"The {mint.word()} {mint.word()} the {mint.word()}.")
            voice.hear(f"The {what} is now on the {second}.")
            self.mark(score, misses, got=voice.holds(what, "is_at"), want=second,
                      item=f"{what}: {first} → now {second}", silence=None)
        for _ in range(4):
            what, first, second = mint.word(), mint.word(), mint.word()
            voice = self.fresh(_Student(self.student))
            voice.hear(f"The {what} is in the {first}.")
            voice.hear(f"The {what} is now on the {second}.")
            voice.hear(f"The {what} is in the {first}.")
            self.mark(score, misses, got=voice.holds(what, "is_at"), want="",
                      item=f"{what}: {first} → {second} → {first} unmarked", silence=None)
        return score, misses


class OtherMinds(DiscourseSubject):
    """Sally and Anne, minted, and read out of sentences rather than set by hand.

    Three orders, and the third is the one worth having: an agent who is wrong, an agent who is
    wrong about what somebody else believes, and an agent who attributes **ignorance** rather
    than error. The last is a separate item because an engine that scored it as a false belief
    would pass the first two for a reason that has nothing to do with minds.
    """

    id = "minds"
    title = "what somebody else believes, and what they think you believe"
    teaches = "a sentence driving a recursive belief store, with ignorance kept apart from error"
    items = 9

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        for _ in range(3):
            who, box, truth, wrong = mint.word(), mint.word(), mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"The {box} has the {truth}.")
            voice.hear(f"{who} thinks the {box} has the {wrong}.")
            key = f"{box}|have"
            self.mark(score, misses, got=voice.minds.false_belief(who, key), want=True,
                      item=f"{who} thinks {box} has {wrong}, really {truth}")
        for _ in range(3):
            who, box, truth = mint.word(), mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"The {box} has the {truth}.")
            voice.hear(f"{who} thinks the {box} has the {truth}.")
            key = f"{box}|have"
            self.mark(score, misses, got=voice.minds.false_belief(who, key), want=False,
                      item=f"{who} is right about {box}")
        for _ in range(3):
            who, box, truth = mint.word(), mint.word(), mint.word()
            voice = self.fresh(brain)
            voice.hear(f"The {box} has the {truth}.")
            voice.hear(f"{who} thinks I do not know the {box} has the {truth}.")
            key = f"{box}|have"
            got = voice.minds.attributes_ignorance(who, voice.minds.speaker, key)
            self.mark(score, misses, got=got, want=True,
                      item=f"{who} attributes ignorance about {box}")
        return score, misses


class RegisterSubject(DiscourseSubject):
    """One meaning, four hearers, and the claim identical in all four.

    Two things are scored and they pull against each other, which is the point: every rendering
    must **read back** as the meaning it came from, and the four must actually differ in how much
    they carry. A module that met the first by saying the same thing four times would fail the
    second, and one that met the second by embroidering would fail the first.
    """

    id = "register"
    title = "as much as the hearer needs, no more than she holds"
    teaches = "audience-controlled saying, verified by parsing her own output"
    items = 9

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for _ in range(3):
            subject, relation, obj = mint.word(), mint.word(), mint.word()
            meaning = compile_meaning(f"{subject} {relation} {obj}")
            spread = voice.register.spread(meaning)
            item = f"{subject} {relation} {obj}"
            self.mark(score, misses, got=all(s.verified for s in spread.values()),
                      want=True, item=f"{item} — all four verified")
            claims = {s.claim for s in spread.values()}
            self.mark(score, misses, got=len(claims) == 1, want=True,
                      item=f"{item} — one claim across four registers")
            widths = [spread[a].words for a in REGISTERS]
            self.mark(score, misses, got=widths == sorted(widths), want=True,
                      item=f"{item} — {widths}")
        return score, misses


class Anticipating(DiscourseSubject):
    """The turn she expected, scored before it arrived.

    The Master's fifth deep mechanism, and the one nothing in this package measured. Half the
    paper is a patterned exchange — question, answer, instruction, repeating — where after enough
    of it she should be able to say what comes next. **The other half is the control, and it is
    the half that decides whether this is a capability or a habit**: an exchange whose acts follow
    each other evenly gives her nothing to predict from, and committing to a guess there is worse
    than saying nothing. Confidence below the floor scores as right on those items.

    Its student is private, for the reason :class:`LongMemory`'s is: every other subject in the
    syllabus talks to the shared communicator, and an exchange counter that had heard all of them
    would be predicting from the syllabus rather than from a conversation.
    """

    id = "anticipation"
    title = "the turn she expected"
    teaches = "what follows what, counted from the exchange rather than shipped as a table"
    items = 10

    #: Question, answer, instruction. Three acts rather than two, so a right answer cannot be got
    #: by alternating, and the same three the control shuffles.
    CYCLE = ("What is the {a}?", "The {a} is the {b}.", "Open the {b}.")
    WANT = ("question", "assertion", "command")

    #: Where in the cycle this student's exchange currently stands. Carried across sittings
    #: because the exchange is.
    _step = 0

    @property
    def student(self) -> Communicator:
        found = getattr(self, "_voice", None)
        if found is None:
            found = Communicator()
            self._voice = found              # noqa: attribute defined outside __init__
        return found

    @property
    def control(self) -> Communicator:
        found = getattr(self, "_flat", None)
        if found is None:
            found = Communicator()
            self._flat = found               # noqa: attribute defined outside __init__
        return found

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.student
        for _ in range(6):
            for step, surface in enumerate(self.CYCLE):
                voice.hear(surface.format(a=mint.word(), b=mint.word()))
                self._step = (step + 1) % len(self.CYCLE)   # noqa: attribute outside __init__
        # And the control's exposure, which is deliberately unlearnable: every act is followed by
        # two different acts equally often, so nothing reaches the floor. Built here rather than
        # left empty because "she predicts nothing having heard nothing" is a weaker claim than
        # "she predicts nothing having heard a great deal that does not repeat".
        flat = self.control
        for first, second in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            for _ in range(3):
                flat.hear(self.CYCLE[first].format(a=mint.word(), b=mint.word()))
                flat.hear(self.CYCLE[second].format(a=mint.word(), b=mint.word()))
        return Taught(18, f"act accuracy {voice.anticipation.accuracy('act'):.2f} "
                          f"on the patterned exchange, "
                          f"control best {self._commitment(flat):.2f}")

    @staticmethod
    def _commitment(voice: Communicator) -> float:
        return float(voice.anticipation.expect().act_confidence)

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice, flat = self.student, self.control
        score, misses = Score(), []
        floor = voice.anticipation.floor
        # Continued from wherever the exchange actually stands, not restarted at the top of the
        # cycle. Restarting cost exactly one item on the retention run — the paper asked "what
        # follows an assertion" and then said a question, which is not the cycle she was taught
        # and not a fact about her.
        for _ in range(5):
            expected = voice.anticipation.expect()
            said = expected.act if expected.act_confidence >= floor else ""
            step = self._step
            self._step = (step + 1) % len(self.CYCLE)   # noqa: attribute outside __init__
            voice.hear(self.CYCLE[step].format(a=mint.word(), b=mint.word()))
            self.mark(score, misses, got=said, want=self.WANT[step],
                      item=f"after {self.WANT[(step - 1) % 3]} → ?", silence="")
        for index in range(5):
            got = self._commitment(flat)
            self.mark(score, misses, got=(got < floor), want=True,
                      item=f"unpatterned exchange, commitment {got:.2f}")
            flat.hear(self.CYCLE[index % len(self.CYCLE)].format(a=mint.word(), b=mint.word()))
        return score, misses


class Attachment(DiscourseSubject):
    """*"I saw the man with the telescope."* — and which prepositions do that is not shipped.

    A preposition attaches two ways when the language has been demonstrated using it both ways.
    That is :class:`Acts`' contest mechanism a third time: a shape two lessons disagree about is
    not one this package reads confidently, and here the disagreement **is** the finding.

    Half the paper is the control, and it is the half that matters. A preposition demonstrated one
    way only must **not** be flagged, or every trailing phrase becomes a question and abstention
    turns into a tic.
    """

    id = "attachment"
    title = "two readings, and neither chosen"
    teaches = "which prepositions attach two ways, discovered from demonstrations that disagree"
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        for _ in range(2):
            voice.show_attachment(f"I {mint.word()} the {mint.word()} with the {mint.word()}.",
                                  "event")
            voice.show_attachment(f"I {mint.word()} the {mint.word()} with the {mint.word()}.",
                                  "object")
        for _ in range(3):
            voice.show_attachment(f"I {mint.word()} the {mint.word()} to the {mint.word()}.",
                                  "event")
        return Taught(7, f"ambiguous: {sorted(voice.attach.ambiguous) or 'none yet'}")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for _ in range(4):
            item = f"I {mint.word()} the {mint.word()} with the {mint.word()}."
            got = attachment(item, learner=voice.attach)
            self.mark(score, misses, got=got.ambiguous, want=True, item=item)
        for _ in range(4):
            item = f"I {mint.word()} the {mint.word()} to the {mint.word()}."
            got = attachment(item, learner=voice.attach)
            self.mark(score, misses, got=got.ambiguous, want=False, item=item)
        return score, misses


# --------------------------------------------------------------------------- #
# controls on the other nine
# --------------------------------------------------------------------------- #

class Tongue(DiscourseSubject):
    """A convention taught in one language, and what it does **not** do in another.

    An indirect request is a fact about a speech community, not about language in general, and a
    school that rewarded an English convention for firing on a Hinglish sentence would be
    rewarding exactly the wrong generalisation. So this subject asserts the opposite of the usual
    transfer claim: the *mechanism* carries — the same class learns a Hinglish convention from
    Hinglish demonstrations with no code changed — and the *convention* does not.
    """

    id = "tongue"
    title = "the mechanism transfers; the convention does not"
    teaches = "a second language's conventions, learned the same way and kept separate"
    items = 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        voice = self.voice(brain)
        for _ in range(3):
            voice.show(f"Kya aap {mint.word()} sakte hain?", "request")
        return Taught(3, "three Hinglish requests, in a shape no English lesson produced")

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        voice = self.voice(brain)
        score, misses = Score(), []
        for _ in range(4):
            item = f"Kya aap {mint.word()} sakte hain?"
            self.mark(score, misses, got=voice.acts.read(item).intended,
                      want="request", item=item)
        # And the control that makes the claim honest: a Hinglish shape nobody demonstrated does
        # not inherit the English convention, because they do not share a skeleton.
        for _ in range(4):
            item = f"Kya {mint.word()} {mint.word()} hai?"
            got = voice.acts.read(item).intended
            self.mark(score, misses, got=(got != "request"), want=True, item=item)
        return score, misses


class Wiring(DiscourseSubject):
    """The only subject that goes through ``brain.think()``.

    V.25 found two defects that were invisible until the brain was in the loop — an echo detector
    that deleted every correct answer, and a refusal that did not block recall — and both were
    found because a subject like this one existed. Three claims are made here and each is one the
    cold run falsified: a denial is confirmed **as a denial**, a figurative sentence is not filed
    as a fact, and an unresolved pronoun produces silence rather than the previous turn handed
    back.
    """

    id = "wiring"
    title = "through the brain, not beside it"
    teaches = "nothing — it measures whether the organ is reachable from a turn"
    items = 6

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        score, misses = Score(), []
        think = getattr(brain, "think", None)
        if not callable(think):
            return score, ["no brain to think with"]
        for _ in range(3):
            who, where = mint.word(), mint.word()
            said = f"{who} never visited {where}."
            try:
                reply = str(think(said).answer or "").lower()
            except Exception as exc:  # noqa: BLE001
                score.add("wrong")
                misses.append(f"{said} → raised {type(exc).__name__}: {exc}")
                continue
            # The defect exactly: the confirmation said `noted: Master visit delhi` back to a
            # denial. Either the polarity is in the line or the line is not said at all; what is
            # not acceptable is a confirmation that reverses what was said.
            confirmed = "noted" in reply
            # On **words**, not on substrings. Written as ``"not" in reply`` this check passed on
            # every item in the paper, because the confirmation it was written to catch begins
            # with the word ``noted``. A control that its own target satisfies is not a control.
            words = set(reply.replace(":", " ").replace("—", " ").split())
            denied = bool(words & {"never", "not", "n't", "cannot", "no"})
            self.mark(score, misses, got=(denied if confirmed else True), want=True,
                      item=f"{said} → {reply[:60]!r}")
        for _ in range(3):
            first, second = mint.word(), mint.word()
            try:
                think(f"{first} met {second}.")
                reply = str(think("He was tired.").answer or "").strip()
            except Exception as exc:  # noqa: BLE001
                score.add("wrong")
                misses.append(f"He was tired. → raised {type(exc).__name__}: {exc}")
                continue
            # An unresolvable pronoun must not come back with the previous turn attached to it.
            # Checked against the previous turn's whole claim rather than against either name:
            # naming a candidate while asking which one was meant is fine, and handing back the
            # claim she was told a moment ago is the defect.
            leaked = f"{first} met {second}" in reply.lower()
            self.mark(score, misses, got=(not leaked), want=True,
                      item=f"He was tired. → {reply[:60]!r}")
        return score, misses


#: The syllabus, in the order it is sat. Conventions first, because the mechanisms behind them do
#: not depend on having been taught anything and the conventions do.
SUBJECTS: Tuple[Any, ...] = (
    Acts, Transfer, Repair, FigurativeSubject, Attachment, Anticipating,
    ReferenceSubject, Contradiction, LongMemory, OtherMinds, RegisterSubject,
    Tongue, Wiring,
)


class _Student:
    """A brain-shaped holder for one communicator, so a subject can examine a private student
    through the same :meth:`DiscourseSubject.voice` path every other subject uses."""

    def __init__(self, voice: Communicator) -> None:
        self.discourse = voice


class DiscourseSchool:
    """Sits her down in front of a conversation.

    Deliberately thin, for the reason :class:`~nyxara.njp.corpusschool.CorpusSchool` gives:
    :class:`~nyxara.njp.school.School` already implements the ``pre-test → teach → post-test``
    loop, the per-subject mint isolation and the report card, and a second copy of any of those is
    a second place for them to drift.
    """

    def __init__(self, *, seed: int = 26, rounds: int = 1,
                 subjects: Optional[Sequence[Any]] = None, verbose: bool = False) -> None:
        from nyxara.njp.school import School

        self._school = School(seed=seed, rounds=rounds,
                              subjects=list(subjects if subjects is not None else SUBJECTS),
                              verbose=verbose)
        self.seed = self._school.seed

    def attend(self, brain: Any = None, *, coder: Any = None) -> Transcript:
        return self._school.attend(brain, coder=coder)

    def retention(self, brain: Any, coder: Any = None, *,
                  seed: Optional[int] = None) -> Transcript:
        return self._school.retention(brain, coder, seed=seed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.discourseschool [--seed N] [--rounds N] [--json] [--retention]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Teach NJP to be talked to, then examine on conversations nobody demonstrated.")
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retention", action="store_true",
                        help="re-examine afterwards with the teacher off")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain(ExamConditions())
    school = DiscourseSchool(seed=args.seed, rounds=args.rounds,
                             verbose=not args.quiet and not args.json)
    started = time.time()
    transcript = school.attend(brain)
    if args.json:
        print(json.dumps(transcript.to_dict(), indent=2, default=str))
    else:
        print(transcript.summary())
        print(f"  {time.time() - started:.1f}s")
    if args.retention:
        after = school.retention(brain, getattr(brain, "coder", None), seed=args.seed + 1)
        if args.json:
            print(json.dumps(after.to_dict(), indent=2, default=str))
        else:
            print("  ── teacher off, fresh items ──")
            print(after.summary())
    return 0 if not transcript.failing else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
