"""The language half of the syllabus: whether the exam can be gamed, and whether a lesson moved it.

These are mostly *about the exam*, for the reason ``test_school.py`` gives: a report card is only
worth reading if the examination it came from cannot be passed by something other than the ability
it names. So what is asserted here is that the language is minted rather than English, that the
compiler she ships with fails it, that the controls are real controls, that the exam vocabulary
never appeared in a lesson, and that what a lesson leaves survives the teacher being switched off.

One of them is a regression test for a flaw the faculty found in the *exam*: the first word-class
subject built its corpus with :func:`~nyxara.njp.dialects.sample`, which gives every sentence
fresh words, so every noun in it was a singleton, and :class:`~nyxara.njp.language.Lexicon`
correctly refused to classify any of them. She scored 0.33 by answering "I have no grounds" to
eight items where that was the truthful answer.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.dialects import KINDS, mint_dialect, sample
from nyxara.njp.language import LanguageFaculty
from nyxara.njp.school import (
    LANGUAGE,
    Course,
    ExamConditions,
    Mint,
    Polarity,
    Questions,
    Reading,
    Saying,
    School,
    Translation,
    WordClasses,
    WordShapes,
)
from nyxara.njp.semantics import compile_meaning


@pytest.fixture(scope="module")
def sat():
    """One whole language syllabus, taught and then re-examined with the teacher off."""
    brain = NJPBrain(ExamConditions())
    school = School(seed=7, rounds=2, subjects=LANGUAGE)
    transcript = school.attend(brain)
    after = school.retention(brain, getattr(brain, "coder", None), seed=8)
    return brain, transcript, after


# --------------------------------------------------------------------------- #
# the syllabus is in the syllabus
# --------------------------------------------------------------------------- #

def test_the_seven_language_subjects_are_in_the_order_that_is_the_claim():
    from nyxara.njp.school import SUBJECTS
    assert LANGUAGE == (WordShapes, WordClasses, Reading, Polarity, Questions, Saying,
                        Translation)
    ids = [subject.id for subject in SUBJECTS]
    assert ids.index("morphology") > ids.index("arithmetic")
    assert ids.index("translation") < ids.index("code-basics")


# --------------------------------------------------------------------------- #
# the exam cannot be passed by something that is not the ability
# --------------------------------------------------------------------------- #

def test_the_exam_language_is_not_english():
    """Fresh vocabulary is not enough: "the zorb chases the plag" is still an English sentence.

    So the grammar is minted too, and the compiler she ships with is measured on it. It does not
    abstain — it reads every sentence and gets every one wrong, which is the floor these subjects
    sit on and the reason each of them carries controls only silence can pass.
    """
    readable = correct = denials = denials_kept = questions = questions_ok = total = 0
    for seed in range(1, 9):
        rng = random.Random(seed)
        dialect = mint_dialect(rng, f"d{seed}")
        counter = [0]

        def word() -> str:
            counter[0] += 1
            number, tag = counter[0], ""
            while number:
                number, remainder = divmod(number - 1, 26)
                tag = chr(ord("a") + remainder) + tag
            return "".join(rng.choice("bdgklmnprstvz") + rng.choice("aeiou")
                           for _ in range(2)) + tag

        for utterance in sample(dialect, word, 24, kinds=KINDS):
            total += 1
            got = compile_meaning(utterance.surface)
            readable += bool(got.readable)
            correct += (got.subject == utterance.meaning.subject
                        and got.object == utterance.meaning.object
                        and got.relation == utterance.meaning.relation)
            if utterance.meaning.negated:
                denials += 1
                denials_kept += bool(got.negated)
            if utterance.meaning.kind != "assertion":
                questions += 1
                questions_ok += (got.kind == utterance.meaning.kind)
    assert total == 192
    assert readable == total          # it never abstains …
    assert correct == 0               # … and it is never right
    assert denials == 32 and denials_kept == 0
    assert questions == 96 and questions_ok == 0


def test_a_control_is_refused_by_every_construction_after_the_whole_syllabus(sat):
    """The control has to stay a control once everything has been taught, or the retention run
    is scoring items that stopped being items."""
    brain, _transcript, _after = sat
    faculty, dialect = brain.language, brain.language.course.first
    mint = Mint(random.Random(4242))
    for utterance in sample(dialect, mint.word, 8, kinds=KINDS):
        assert faculty.read(utterance.surface, tongue=dialect.name).readable
        control = f"{utterance.surface} {mint.word()}"
        assert not faculty.read(control, tongue=dialect.name).readable


def test_the_word_class_corpus_repeats_its_words_on_purpose(sat):
    """A distributional class needs a distribution, and one context is not one.

    This is the flaw the faculty found in the exam: built from
    :func:`~nyxara.njp.dialects.sample`, every noun was a singleton, ``Lexicon`` refused to
    classify any of them, and the honest refusal read as a failing subject.
    """
    brain, _transcript, _after = sat
    dialect = brain.language.course.first
    mint = Mint(random.Random(11))
    heard, subjects, objects, verbs = WordClasses._chorus(
        dialect, mint, 12, ("assertion", "past"))
    assert len(heard) == 12
    assert len(set(subjects)) == 4 and len(set(objects)) == 4 and len(set(verbs)) == 3
    seen = {}
    for utterance in heard:
        for token in utterance.surface.split():
            seen[token] = seen.get(token, 0) + 1
    assert min(seen.values()) >= 2


# --------------------------------------------------------------------------- #
# what the run actually reports
# --------------------------------------------------------------------------- #

def test_every_language_subject_is_mastered_and_none_of_it_is_a_wrong_answer(sat):
    _brain, transcript, _after = sat
    assert len(transcript.mastered) == len(LANGUAGE), \
        [(r.subject, round(r.post.accuracy, 2)) for r in transcript.failing]
    assert sum(result.post.wrong for result in transcript.results) == 0
    assert all(result.post.precision == 1.0 for result in transcript.results
               if result.post.right + result.post.wrong)


def test_six_of_the_seven_moved_because_a_lesson_ran(sat):
    """``word-classes`` is the seventh and reads 1.00 cold, which is reported as ``already``.

    That is the honest outcome rather than a disappointing one: the ability needs exposure, not
    teaching, and an exam about words she has met has to supply its own exposure.
    """
    _brain, transcript, _after = sat
    moved = {result.subject for result in transcript.learned}
    assert moved == {"morphology", "reading", "polarity", "questions", "saying", "translation"}
    classes = next(r for r in transcript.results if r.subject == "word-classes")
    assert classes.already and classes.mastered


def test_the_wug_subject_starts_at_the_floor_and_the_floor_is_abstention(sat):
    """Cold, she has no ending bound to anything, so every positive item is silence."""
    _brain, transcript, _after = sat
    shapes = next(r for r in transcript.results if r.subject == "morphology")
    assert shapes.pre.wrong == 0
    assert shapes.pre.abstained >= 8
    assert shapes.gain > 0.5


def test_saying_is_moved_by_a_reading_lesson_and_nothing_about_speaking(sat):
    """The shape crosses from comprehension into production without a production lesson."""
    _brain, transcript, _after = sat
    saying = next(r for r in transcript.results if r.subject == "saying")
    assert saying.pre.accuracy == pytest.approx(0.75, abs=0.01)
    assert saying.post.accuracy == 1.0
    assert "none of them a lesson in speaking" in saying.note


def test_it_survives_the_teacher_being_switched_off(sat):
    """Nothing taught, no budget raised, and items that have never been seen."""
    _brain, _transcript, after = sat
    assert len(after.mastered) == len(LANGUAGE), \
        [(r.subject, round(r.post.accuracy, 2)) for r in after.failing]
    assert sum(result.post.wrong for result in after.results) == 0


@pytest.mark.parametrize("seed", [3, 19, 42, 101, 2024, 55, 88, 7777])
def test_it_is_not_one_lucky_language(seed):
    """A different seed is a different word order, different markers and different particles.

    Eight here rather than the forty the docs quote, because the whole point of the sweep is that
    it is cheap — the language half of the syllabus is thirty milliseconds — and a test suite is
    not the place to spend the other thirty-two seeds proving the same thing twice.
    """
    brain = NJPBrain(ExamConditions())
    school = School(seed=seed, rounds=2, subjects=LANGUAGE)
    transcript = school.attend(brain)
    after = school.retention(brain, getattr(brain, "coder", None), seed=seed + 1000)
    for report in (transcript, after):
        assert sum(result.post.wrong for result in report.results) == 0
        assert len(report.mastered) == len(LANGUAGE), \
            [(r.subject, round(r.post.accuracy, 2)) for r in report.failing]


# --------------------------------------------------------------------------- #
# the machinery the subjects share
# --------------------------------------------------------------------------- #

def test_the_two_minted_languages_are_shared_and_stable(sat):
    """Seven subjects examine one student, so the dialect is minted once and kept on her."""
    brain, _transcript, _after = sat
    course = brain.language.course
    assert isinstance(course, Course)
    assert course.first.name == "dialect-a" and course.second.name == "dialect-b"
    subject = Reading()
    assert subject.course(brain, Mint(random.Random(1))).first is course.first


def test_a_brain_with_no_faculty_still_gets_examined():
    """A subject that could not run without an organ would report an absence as a failure."""

    class Bare:
        pass

    bare = Bare()
    subject = WordShapes()
    faculty = subject.faculty(bare)
    assert isinstance(faculty, LanguageFaculty)
    assert getattr(bare, "language", None) is faculty


def test_the_grader_keeps_three_outcomes_apart():
    from nyxara.njp.school import Question, Score
    score, misses = Score(), []
    positive = Question(ask="?", accept=("wugik",), exact=True)
    control = Question(ask="?", accept=(), silence_ok=True)
    assert WordShapes.mark(score, misses, positive, "wugik") == "right"
    assert WordShapes.mark(score, misses, positive, "") == "abstain"
    assert WordShapes.mark(score, misses, positive, "wugoz") == "wrong"
    assert WordShapes.mark(score, misses, control, "") == "right"
    assert WordShapes.mark(score, misses, control, "wugik") == "wrong"
    assert (score.right, score.wrong, score.abstained) == (2, 2, 1)
