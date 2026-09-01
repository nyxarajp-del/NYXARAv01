"""English and Hindi: whether the curriculum is honest, and whether she learned a grammar from it.

Everything the faculty had learned before this was a language nobody speaks, minted per seed to
prove the mechanism is a mechanism. That is the right thing to measure and it leaves one question
unasked, so what is asserted here is the answer to it: taught two real languages by hand, does she
read sentences no lesson contained?

Most of these are about the **curriculum**, because a curriculum is an exam and can be dishonest
in all the same ways. So: no held-out sentence appears in a lesson, the paradigms are real English
rather than a tidied version of it, and the exam does not ask for a form the lesson never taught.

Two are about the ceiling rather than the score, and they matter more than the score: she refuses
a real English sentence outside what she was taught, and she refuses to translate a sentence with
one word she has no gloss for.
"""

from __future__ import annotations

import json

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.language import LanguageFaculty
from nyxara.njp.lessons import (
    COURSES,
    ENGLISH,
    GLOSSARY,
    HINDI,
    HINGLISH,
    enrol,
    examine,
    glosses,
    teach,
)
from nyxara.njp.school import ExamConditions
from nyxara.njp.semantics import Meaning


@pytest.fixture(scope="module")
def taught():
    faculty, reports = enrol()
    return faculty, {report.language: report for report in reports}


# --------------------------------------------------------------------------- #
# the curriculum, as an exam
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("course", COURSES, ids=lambda c: c.name)
def test_no_exam_sentence_was_ever_a_lesson(course):
    shown = {surface for surface, _meaning in course.lessons} | set(course.heard)
    for surface, _meaning in course.exam:
        assert surface not in shown, (course.name, surface)


@pytest.mark.parametrize("course", COURSES, ids=lambda c: c.name)
def test_no_wug_stem_was_ever_inflected_in_front_of_her(course):
    demonstrated = {base for base, _inflected, _f in course.inflections}
    for stem, _feature, want in course.wug:
        if stem in demonstrated:
            continue                     # the irregulars, where recall *is* the ability
        assert want not in course.forms, (course.name, want)


def test_the_english_paradigms_are_real_english():
    """The first draft regularised them — ``pushs``, ``carrys``, ``mouses`` — and the exam then
    asked for the real forms, which measures the curriculum rather than the learner."""
    forms = set(ENGLISH.forms)
    for wrong in ("pushs", "carrys", "watchs", "mouses", "childs", "leafs", "eated"):
        assert wrong not in forms
    for right in ("pushes", "watches", "children", "mice", "leaves", "went"):
        assert right in forms


def test_the_exam_never_asks_for_a_word_the_lesson_never_taught():
    """Every content word in every held-out sentence is one she has met."""
    from nyxara.njp.language import tokenize_surface
    for course in COURSES:
        known = set(course.forms)
        for text in course.heard:
            known.update(tokenize_surface(text))
        for surface, _meaning in course.lessons:
            known.update(tokenize_surface(surface))
        for surface, _meaning in course.exam:
            for token in tokenize_surface(surface):
                assert token in known, (course.name, surface, token)


def test_hindi_is_the_same_grammar_in_two_scripts():
    """Not a translation of the course — the same sentences, transliterated."""
    assert len(HINDI.lessons) == len(HINGLISH.lessons)
    assert len(HINDI.exam) == len(HINGLISH.exam)
    for (_deva, one), (_latn, two) in zip(HINDI.exam, HINGLISH.exam):
        assert one.kind == two.kind and one.negated == two.negated
        assert one.focus == two.focus and one.temporal == two.temporal


# --------------------------------------------------------------------------- #
# what she scores
# --------------------------------------------------------------------------- #

def test_she_reads_and_says_held_out_sentences_in_both_languages(taught):
    _faculty, reports = taught
    for name in ("en", "hi", "hi-latn"):
        report = reports[name]
        assert report.read_right == report.read_total > 0, report.to_dict()
        assert report.said_right == report.said_total > 0, report.to_dict()
        assert report.wug_right == report.wug_total > 0, report.to_dict()


def test_a_faculty_that_was_taught_nothing_reads_none_of_it():
    """Every one of those sentences is unreadable on day one, which is what makes the run mean
    something. She ships a tokeniser and 242 closed-class words, and no vocabulary at all."""
    empty = LanguageFaculty()
    for course in COURSES:
        for surface, _meaning in course.exam:
            assert not empty.read(surface, tongue=course.name).readable
        for stem, feature, _want in course.wug:
            assert empty.inflect(stem, feature, tongue=course.name) == ""


def test_the_wug_test_in_real_english(taught):
    """The word the test is named after, and a rule that has to reach it."""
    faculty, _reports = taught
    assert faculty.inflect("wug", "plural", tongue="en") == "wugs"
    assert faculty.inflect("blicket", "past", tongue="en") == "blicketed"
    assert faculty.inflect("gostak", "progressive", tongue="en") == "gostaking"
    # An irregular is recalled and does not leak onto anything else.
    assert faculty.inflect("go", "past", tongue="en") == "went"
    assert faculty.inflect("wug", "past", tongue="en") == "wuged"


def test_an_ending_crosses_from_one_construction_to_another(taught):
    """``pushes`` was only ever shown inside the transitive family; the possessive family knows
    ``-s`` alone. The morphology holds the whole paradigm, so both read it and both say it."""
    faculty, _reports = taught
    got = faculty.read("the king 's cow pushes the tree", tongue="en")
    assert (got.subject, got.relation, got.object) == ("cow", "push", "tree")
    said = faculty.say(Meaning(kind="assertion", subject="cow", relation="push", object="tree",
                               roles={"owner": "king"}), tongue="en")
    assert "pushes" in said


# --------------------------------------------------------------------------- #
# translation, and the lexicon it needs
# --------------------------------------------------------------------------- #

def test_she_translates_in_every_direction(taught):
    faculty, _reports = taught
    assert faculty.translate("the farmer chases the cat", into="hi", frm="en") \
        == "लड़का बिल्ली देखता है"
    assert faculty.translate("लड़का बिल्ली देखता है", into="en", frm="hi") \
        == "the farmer chases the cat"
    assert faculty.translate("लड़का बिल्ली देखता है", into="hi-latn", frm="hi") \
        == "ladka billi dekhta hai"
    # Polarity and speech act cross with it, because they are part of the meaning.
    assert faculty.translate("the teacher does not open the book", into="hi", frm="en") \
        == "शिक्षक किताब नहीं पढ़ता है"


def test_a_sentence_with_one_unglossed_word_translates_to_nothing(taught):
    """Rather than to a sentence with a hole in it."""
    faculty, _reports = taught
    assert faculty.read("the goat chases the cat", tongue="en").readable
    assert not any(row[0] == "goat" for row in GLOSSARY)
    assert faculty.translate("the goat chases the cat", into="hi", frm="en") == ""


def test_the_glossary_is_taught_rather_than_derived():
    """There is no amount of monolingual evidence from which ``dog`` and ``कुत्ता`` follow."""
    faculty = LanguageFaculty()
    assert faculty.translate("the dog chases the cat", into="hi", frm="en") == ""
    assert glosses(faculty) == len(GLOSSARY) * 3
    assert faculty.gloss[("en", "hi")]["dog"] == "कुत्ता"
    assert faculty.gloss[("hi", "en")]["कुत्ता"] == "dog"


# --------------------------------------------------------------------------- #
# the ceiling, stated as a test rather than as a caveat
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,tongue", [
    ("the quick brown fox jumps over the lazy dog", "en"),
    ("yesterday I would have preferred the other one", "en"),
    ("the goat does not walk", "en"),
    ("मैंने कल तुम्हें बताया था कि यह मुश्किल है", "hi"),
])
def test_she_refuses_a_real_sentence_outside_what_she_was_taught(taught, text, tongue):
    """Including one that is only *nearly* in it: negation was demonstrated on transitive clauses
    and never on an intransitive one, so ``the goat does not walk`` is a shape she has not got."""
    faculty, _reports = taught
    assert not faculty.read(text, tongue=tongue).readable


def test_the_vocabulary_is_small_and_that_is_stated(taught):
    faculty, _reports = taught
    english = faculty.tongue("en").morphology.vocabulary
    assert len(english) < 400          # a vocabulary you can read in a minute
    for absent in ("computer", "gravity", "yesterday", "beautiful", "because"):
        assert absent not in english


# --------------------------------------------------------------------------- #
# through the brain, and across a restart
# --------------------------------------------------------------------------- #

def test_the_brain_learns_them_and_keeps_them_across_a_restart():
    brain = NJPBrain(ExamConditions())
    assert not brain.read_language("the dog chases the cat", tongue="en").readable
    reports = brain.learn_languages()
    assert reports and len(reports) == 3
    assert brain.language.known() == ["en", "hi", "hi-latn"]

    woken = NJPBrain(ExamConditions())
    woken.load_dict(json.loads(json.dumps(brain.to_dict())))
    got = woken.read_language("the farmer paints the window", tongue="en")
    assert (got.subject, got.relation, got.object) == ("farmer", "paint", "window")
    assert woken.translate("the teacher opens the book", into="hi", frm="en") \
        == "शिक्षक किताब पढ़ता है"


def test_teaching_one_course_leaves_the_others_alone():
    """Three grammars held at once, none of them contaminating the others."""
    faculty = LanguageFaculty()
    teach(ENGLISH, faculty)
    assert not faculty.read("लड़का आम खाता है", tongue="hi").readable
    teach(HINDI, faculty)
    report = examine(ENGLISH, faculty)
    assert report.read_right == report.read_total
