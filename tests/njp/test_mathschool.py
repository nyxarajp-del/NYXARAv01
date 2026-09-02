"""The mathematics school: what a lesson can move here, and what it cannot.

Sixteen of the eighteen subjects are decision procedures and read 1.00 cold — that is evidence
about the organ, not about a lesson, and the report says `already` beside every one of them. One
subject can be taught, because stating mathematics and doing it are different capabilities. One
subject is entirely controls, because the failure this whole change set exists for was **not**
silence: it was five sentences of arithmetic filed into the knowledge store as facts.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.mathschool import (
    MATHS_SUBJECTS,
    MathExam,
    MathSchool,
    Restraint,
    Vocabulary,
    _VOCABULARY,
)
from nyxara.njp.school import ExamConditions, Mint


@pytest.fixture(scope="module")
def examined() -> tuple:
    brain = NJPBrain(ExamConditions())
    return brain, MathExam(brain, limit=12).sit()


def test_every_paper_is_sat_and_none_of_them_crashes(examined):
    _, transcript = examined
    assert len(transcript.results) == len(MathExam.PAPERS)
    for result in transcript.results:
        assert "raised" not in result.note, f"{result.subject}: {result.note}"
        assert result.post.total > 0


def test_she_passes_every_paper(examined):
    _, transcript = examined
    failing = [(r.subject, round(r.post.accuracy, 3), r.misses[:2])
               for r in transcript.results if not r.mastered]
    assert failing == []


def test_the_exam_writes_nothing(examined):
    """It may be run twice around something that mutates her, or the two numbers mean nothing."""
    brain, _ = examined
    before = len(brain.grounder.facts)
    MathExam(brain, limit=4).sit()
    assert len(brain.grounder.facts) == before


def test_an_examination_cannot_answer_its_own_later_questions():
    """Measured: `restraint` scored 8/10 inside a full exam and 10/10 asked on its own.

    "what is 7 divided by 0?" came back **0.7** — sixteen papers of arithmetic had gone into
    episodic memory ahead of it and recall found the nearest thing in the store. `MathSubject.ask`
    passes `remember=False` so the ordering of the papers cannot change the score.
    """
    brain = NJPBrain(ExamConditions())
    for _ in range(3):
        brain.think("what is 24 + 18?", remember=False)
    score, _ = Restraint().exam(brain, Mint(__import__("random").Random(3)))
    assert score.wrong == 0


def test_the_taught_half_is_fixed_before_the_pre_test_not_by_the_lesson():
    """Chosen inside `teach`, the pre-test has no split to grade against.

    Every term is then a control, she is silent on all twenty, and the floor reads **1.00 for
    knowing nothing at all** — with the post-test at 0.85 reported as a lesson that lost points.
    """
    subject = Vocabulary()
    brain = NJPBrain(ExamConditions())
    pre, _ = subject.exam(brain, Mint(__import__("random").Random(11)))
    assert subject.planned, "the split was not fixed by the pre-test"
    assert 0.4 <= pre.accuracy <= 0.6
    assert subject.stated == ()


def test_a_stated_definition_can_be_asked_back_and_a_withheld_one_stays_silent():
    """The only number in this module a lesson moves, and both halves of it are the claim."""
    school = MathSchool(seed=11, rounds=1, subjects=(Vocabulary,))
    transcript = school.attend()
    result = transcript.results[0]
    assert result.pre.accuracy < result.post.accuracy
    assert result.post.accuracy == 1.0
    assert result.taught == len(_VOCABULARY) // 2


@pytest.mark.parametrize("term,sentence,expected", _VOCABULARY)
def test_every_definition_round_trips_through_the_brain(term, sentence, expected):
    """A lesson that does not land reports a gain it did not produce.

    Two phrasings had to be found by measurement. "The median is the middle value **when** the
    values are in order" is filed as a *condition*, not a definition, and comes back unaskable;
    and the definite article matters — "the mean is …" does not read back where "a mean is …"
    does. Neither is a fact about mathematics; both are facts about the reader.
    """
    brain = NJPBrain(ExamConditions())
    assert brain.think(f"what is {term}?", remember=False).answer.strip() == "", \
        f"{term} was answerable before the lesson — it is not a valid control"
    brain.think(sentence)
    answer = brain.think(f"what is {term}?", remember=False).answer.lower()
    assert expected.lower() in answer, f"{term}: taught {sentence!r}, got {answer!r}"


def test_the_school_teaches_nothing_it_cannot_teach():
    """Sixteen subjects are procedures. A lesson for one would change nothing and say it had."""
    procedures = [s for s in MATHS_SUBJECTS if s not in (Vocabulary,)]
    for factory in procedures:
        subject = factory()
        lesson = subject.teach(None, Mint(__import__("random").Random(1)))
        assert lesson.items == 0
        assert "decision procedure" in lesson.note


def test_the_full_syllabus_runs_and_masters_every_subject():
    transcript = MathSchool(seed=11, rounds=1).attend()
    assert len(transcript.results) == len(MATHS_SUBJECTS)
    failing = [(r.subject, round(r.post.accuracy, 3), r.misses[:2])
               for r in transcript.results if not r.mastered]
    assert failing == []


def test_the_brain_entry_points_reach_both():
    brain = NJPBrain(ExamConditions())
    assert brain.sit_maths_exam(limit=3, papers=(Restraint,)) is not None
    assert brain.go_to_maths_school(rounds=1, subjects=(Restraint,)) is not None
