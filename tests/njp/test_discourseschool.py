"""The conversation syllabus, and the properties that make its report card mean anything.

A school's number is only worth what its construction is worth, so what is asserted here is the
construction:

* every item is **minted**, so a lesson and an exam cannot share a word — checked by minting two
  sittings and asserting they overlap in nothing;
* the four taught subjects have a real **floor**, which is the claim a shipped convention would
  falsify: a brain that could already read an indirect request would score 1.00 cold;
* the five floor subjects are floors — they teach nothing and are asserted to say so;
* ``tongue`` asserts the opposite of the usual transfer claim, and the control is the load-bearing
  half: an English convention must **not** fire on a Hinglish sentence;
* ``wiring`` goes through ``brain.think`` and is asserted to be capable of failing — its two
  checks are re-run against the defects they were written for.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.discourse import Communicator
from nyxara.njp.discourseschool import (SUBJECTS, Acts, Attachment, Contradiction,
                                        DiscourseSchool, DiscourseSubject, LongMemory,
                                        ReferenceSubject, Tongue, Transfer, Wiring, _plural)
from nyxara.njp.school import ExamConditions, Mint

SEED = 26


@pytest.fixture(scope="module")
def taught():
    """One sitting, shared by the tests that read its transcript."""
    brain = NJPBrain(ExamConditions())
    school = DiscourseSchool(seed=SEED, rounds=2)
    return brain, school.attend(brain)


def test_every_subject_is_mastered_and_the_taught_ones_moved(taught):
    _brain, transcript = taught
    assert not transcript.failing, [r.subject for r in transcript.failing]
    moved = {r.subject for r in transcript.learned}
    assert {"acts", "transfer", "repair", "figurative", "attachment", "anticipation",
            "exchange", "alternations", "anchor", "vocabulary", "reference", "contradiction",
            "memory", "tongue"} <= moved


def test_the_taught_subjects_have_a_real_floor(taught):
    """A convention that could be read before it was demonstrated was shipped, not learned.

    Four of these eight were floors of 1.00 when this syllabus was first written, and that was the
    module holding hand-written tables rather than her knowing anything. They are here now.
    """
    _brain, transcript = taught
    floors = {r.subject: r.pre.accuracy for r in transcript.results}
    for subject in ("acts", "transfer", "repair", "figurative", "attachment", "anticipation",
                    "exchange", "alternations", "anchor", "vocabulary", "reference",
                    "contradiction", "memory", "tongue"):
        assert floors[subject] < 0.9, f"{subject} floor {floors[subject]}"


def test_the_floor_subjects_teach_nothing_and_say_so(taught):
    """Two of them, now. `minds` and `register` are mechanisms; `wiring` measures reachability."""
    _brain, transcript = taught
    for result in transcript.results:
        if result.subject in ("minds", "register", "ground", "retell", "unseen", "method",
                              "standing", "inferred", "wiring"):
            assert result.taught == 0
            assert "floor" in result.note or "measures" in result.teaches


def test_two_sittings_share_no_minted_word():
    """The exam's validity rests on this and nothing else."""
    def words(seed):
        mint = Mint(random.Random(seed))
        return {mint.word() for _ in range(64)}

    assert not (words(1) & words(2))


def test_an_untaught_brain_cannot_read_an_indirect_request():
    subject, brain = Acts(), NJPBrain(ExamConditions())
    score, _misses = subject.exam(brain, Mint(random.Random(11)))
    assert score.accuracy < 0.9


def test_transfer_does_not_share_a_construction_with_acts():
    """Sharing one made this subject's floor 1.00, which is a report that it measured too late."""
    voice = Communicator()
    brain = _Holder(voice)
    Acts().teach(brain, Mint(random.Random(3)))
    score, _misses = Transfer().exam(brain, Mint(random.Random(4)))
    assert score.accuracy < 0.9


def test_an_english_convention_does_not_fire_on_a_hinglish_sentence():
    voice = Communicator()
    brain = _Holder(voice)
    Acts().teach(brain, Mint(random.Random(5)))
    mint = Mint(random.Random(6))
    assert voice.acts.read(f"Kya aap {mint.word()} sakte hain?").intended != "request"
    # And the mechanism does carry: the same class learns the Hinglish convention from Hinglish
    # demonstrations with nothing changed.
    Tongue().teach(brain, Mint(random.Random(7)))
    assert voice.acts.read(f"Kya aap {mint.word()} sakte hain?").intended == "request"


def test_the_wiring_subject_can_fail():
    """A control its own target satisfies is not a control — this one did, on ``"not" in reply``."""
    subject = Wiring()
    score, misses = subject.exam(_Deaf(), Mint(random.Random(8)))
    assert score.wrong == 6 and not score.right, misses


def test_the_wiring_subject_passes_against_the_real_brain():
    score, misses = Wiring().exam(NJPBrain(ExamConditions()), Mint(random.Random(9)))
    assert score.accuracy == 1.0, misses


def test_a_minted_plural_is_one_a_shape_rule_can_see():
    from nyxara.njp.discourse import _is_plural

    mint = Mint(random.Random(10))
    for _ in range(16):
        assert _is_plural(_plural(mint))


def test_the_long_memory_subject_scores_silence_as_right_where_nothing_holds():
    """Half its items are contested pairs, where returning either claim is a coin flip.

    And it needs its lesson first: without the induced markers nothing has shown her that a
    speaker can signal a change, so the reversion is more information rather than a conflict.
    """
    subject, brain = LongMemory(), _Holder(Communicator())
    assert subject.exam(brain, Mint(random.Random(12)))[0].accuracy < 1.0
    subject.teach(brain, Mint(random.Random(120)))
    score, misses = subject.exam(brain, Mint(random.Random(12)))
    assert score.accuracy == 1.0, misses


def test_reference_is_examined_on_both_resolution_and_refusal():
    subject, brain = ReferenceSubject(), _Holder(Communicator())
    assert subject.exam(brain, Mint(random.Random(13)))[0].accuracy < 1.0
    subject.teach(brain, Mint(random.Random(130)))
    score, misses = subject.exam(brain, Mint(random.Random(13)))
    assert score.accuracy == 1.0, misses


def test_the_markers_are_induced_by_the_contradiction_lesson():
    """`now` and `never` are in no table any more; six demonstrated verdicts put them there."""
    voice = Communicator()
    brain = _Holder(voice)
    assert voice.markers.kept == {}
    Contradiction().teach(brain, Mint(random.Random(14)))
    assert voice.markers.kept["now"].role == "licence"
    assert voice.markers.kept["never"].role == "universal"
    # The prepositions are varied across the demonstrations precisely so they cannot survive.
    assert set(voice.markers.kept) == {"now", "never"}


def test_the_attachment_lesson_finds_one_preposition_and_not_the_other():
    voice = Communicator()
    brain = _Holder(voice)
    assert voice.attach.ambiguous == set()
    Attachment().teach(brain, Mint(random.Random(15)))
    assert voice.attach.ambiguous == {"with"}


def test_the_syllabus_is_thirty_subjects_with_unique_ids():
    made = [factory() for factory in SUBJECTS]
    assert len(made) == 30
    assert len({subject.id for subject in made}) == 30
    assert all(isinstance(subject, DiscourseSubject) for subject in made)


class _Holder:
    """A stand-in for a brain that only has to hold a communicator."""

    def __init__(self, voice: Communicator) -> None:
        self.discourse = voice


class _Deaf:
    """A brain whose every reply is the defect the ``wiring`` subject was written against."""

    def think(self, text: str):
        words = str(text).replace(".", "").split()
        if text.startswith("He was tired"):
            return _Reply(f"noted: {self.first} met {self.second}")
        self.first, self.second = words[0], words[-1]
        return _Reply(f"noted: {words[0]} visit {words[-1]}")


class _Reply:
    def __init__(self, answer: str) -> None:
        self.answer = answer


def test_the_anticipation_control_is_built_to_be_unlearnable():
    """Its floor is not "she has heard nothing" — it is "she has heard a great deal that does
    not repeat", which is the stronger claim and the one worth making."""
    from nyxara.njp.discourseschool import Anticipating

    subject = Anticipating()
    subject.teach(_Holder(Communicator()), Mint(random.Random(21)))
    assert subject.student.anticipation.accuracy("act") > 0.9
    assert subject.control.anticipation.turns >= 30
    assert subject._commitment(subject.control) < subject.control.anticipation.floor


def test_the_lesson_is_handed_the_previous_sitting_s_failures():
    """The stage the training loop was missing: teaching against what was actually missed."""
    import random as _random

    from nyxara.njp.discourseschool import Retraining
    from nyxara.njp.school import School

    transcript = School(seed=26, rounds=2, subjects=[Retraining]).attend(
        _Holder(Communicator()))
    result = transcript.results[0]
    assert result.pre.accuracy == 0.0          # two of six demonstrated, and the exam is all six
    assert result.post.accuracy == 1.0
    assert "missed last time" in result.note

    # And the control: with no failures to work from it falls back to an arbitrary two.
    blind = Retraining()
    blind.teach(_Holder(Communicator()), Mint(_random.Random(4)))
    score, _misses = blind.exam(_Holder(blind.student), Mint(_random.Random(5)))
    assert score.accuracy < 1.0
