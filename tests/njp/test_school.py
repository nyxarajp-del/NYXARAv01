"""The school: whether teaching moved a number on questions that were never taught.

The tests here are mostly *about the exam* rather than about her, and that is deliberate. A report
card is only worth reading if the exam it came from cannot be gamed, so what is asserted is: the
vocabulary is fresh, the controls are real, an abstention is scored apart from an error, and a
subject that could already do the thing is reported as already rather than as learned.

Two of these are regression tests for bugs the school found in the brain itself, which is the
argument for having a school at all:

* asked ``"what is 25 + 10?"`` after a few similar turns she answered **10** — the deliberation
  ladder settled on a token the question contained and out-ranked the calculator, because the
  meta-reasoner learns its ordering from outcomes rather than from priors;
* teaching a relation to chain also minted ``shape:p>p>p`` in the genome, which was promoted as a
  rival arm, pinned at three hops, and shadowed the general composition walk on every four-hop
  question — so the lesson worked and the report said it had not.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.coding import Coder
from nyxara.njp.school import (
    Abstention,
    Arithmetic,
    Composition,
    Depth,
    ExamConditions,
    Inheritance,
    Mint,
    Question,
    School,
    Score,
    Transcript,
    WriteComposite,
)


@pytest.fixture(scope="module")
def brain():
    return NJPBrain(ExamConditions())


def _mint(seed=1):
    return Mint(random.Random(seed))


# --------------------------------------------------------------------------- #
# the exam has to be an exam
# --------------------------------------------------------------------------- #

def test_generated_words_look_like_words():
    """They ended in digits, and ``all zork8s are shiny`` does not match the grounder's plural
    rule the way ``all zorks are shiny`` does — so `inheritance` scored 0.33 and the reason was
    the exam's own vocabulary."""
    words = _mint().words(50)
    assert all(word.isalpha() for word in words)
    assert len(set(words)) == 50


def test_a_control_grades_silence_as_right_and_an_assertion_as_wrong():
    control = Question(ask="does x cause y?", accept=("no",), silence_ok=True)
    assert control.grade("") == "right"
    assert control.grade("yes") == "wrong"
    assert control.grade("no") == "right"


def test_a_positive_item_grades_silence_as_an_abstention_not_an_error():
    """Three counters, never two. A brain that answers everything and one that answers nothing
    produce very different report cards, and one number cannot tell them apart."""
    item = Question(ask="does x cause y?", accept=("yes",))
    assert item.grade("") == "abstain"
    assert item.grade("yes") == "right"
    assert item.grade("no") == "wrong"


def test_the_score_keeps_accuracy_and_precision_apart():
    score = Score(right=3, wrong=1, abstained=6)
    assert score.accuracy == pytest.approx(0.3)
    assert score.precision == pytest.approx(0.75)
    assert score.coverage == pytest.approx(0.4)


def test_a_subject_that_could_already_do_it_is_reported_as_already_not_learned():
    """``taught`` is in the condition for a reason: a floor subject that reads 0.92 on one set of
    generated items and 1.00 on the next has sampled twice, not learned."""
    from nyxara.njp.school import Result
    transcript = Transcript()
    floor = Result(subject="floor", pre=Score(right=9, abstained=1),
                   post=Score(right=10), threshold=0.8, taught=0)
    real = Result(subject="real", pre=Score(right=2, abstained=8),
                  post=Score(right=9, abstained=1), threshold=0.8, taught=5)
    transcript.results = [floor, real]
    assert floor.already and floor.mastered
    assert [r.subject for r in transcript.learned] == ["real"]


# --------------------------------------------------------------------------- #
# what she can already do, measured rather than assumed
# --------------------------------------------------------------------------- #

def test_arithmetic_survives_a_memory_full_of_arithmetic(brain):
    """Regression: the ladder answered ``25 + 10`` with ``10``. A question with a decision
    procedure does not get a guess offered against it at all."""
    subject, seen = Arithmetic(), Score()
    for seed in (11, 12, 13):
        score, misses = subject.exam(brain, _mint(seed), coder=brain.coder)
        seen = seen.merged(score)
        assert not misses, misses
    assert seen.accuracy == 1.0


def test_she_composes_a_chain_she_was_never_told(brain):
    score, misses = Composition().exam(brain, _mint(21), coder=brain.coder)
    assert score.accuracy >= 0.75, misses
    assert score.wrong == 0, misses


def test_she_refuses_what_nothing_grounds(brain):
    """The one subject that cannot be gamed by confidence: the only way to score is to refuse."""
    score, misses = Abstention().exam(brain, _mint(31), coder=brain.coder)
    assert score.accuracy >= 0.9, misses


def test_a_member_inherits_what_its_kind_was_given(brain):
    score, misses = Inheritance().exam(brain, _mint(41), coder=brain.coder)
    assert score.accuracy >= 0.75, misses


# --------------------------------------------------------------------------- #
# the part that has to be taught
# --------------------------------------------------------------------------- #

def test_teaching_a_relation_to_chain_reaches_four_hops():
    """The reasoning half's acquisition claim, with its control kept in the same exam.

    Cold, a four-hop chain fails twice over: the per-hop confidence falls under the link floor
    because an unproven relation's transitivity prior is low, and the walk budget refuses to
    extend that far. Teaching moves the first and earns the second — and the controls, chains
    that do not exist, have to stay silent throughout or the capability was bought rather than
    learned.
    """
    brain = NJPBrain(ExamConditions())
    subject = Depth()
    before, _ = subject.exam(brain, _mint(51), coder=brain.coder)
    lesson = subject.teach(brain, _mint(52), coder=brain.coder)
    after, misses = subject.exam(brain, _mint(53), coder=brain.coder)

    assert lesson.items >= 4, lesson.note
    assert before.accuracy < 0.5, "a four-hop chain should not be reachable cold"
    assert after.accuracy > before.accuracy + 0.3, misses
    assert after.wrong == 0, f"the controls must hold: {misses}"


def test_a_promoted_same_predicate_shape_no_longer_shadows_composition():
    """Regression, and it looked exactly like the teaching had failed: after distillation the core
    derived the four-hop chain and `_strategy_compose` returned "yes" when called directly, while
    `think` answered nothing and named ``shape:p>p>p`` as the arm it used."""
    brain = NJPBrain(ExamConditions())
    names = [f"qa{letter}" for letter in "abcdefg"]
    for left, right in zip(names, names[1:]):
        brain.think(f"{left} zibars {right}")
    assert brain.genome is not None, "no genome means this test proves nothing"
    brain._promote_shapes()
    assert not any(name.startswith("shape:") and len(set(name[6:].split(">"))) == 1
                   for name in brain.metareason.strategies)


def test_a_coding_shape_taught_on_one_task_carries_to_another():
    """The coding half's acquisition claim. Grafting is off and the budget is identical on both
    sides, so the only thing that differs between the two numbers is which shapes she holds."""
    coder, subject = Coder(), WriteComposite()
    before, _ = subject.exam(None, _mint(61), coder=coder)
    lesson = subject.teach(None, _mint(62), coder=coder)
    after, misses = subject.exam(None, _mint(63), coder=coder)

    assert lesson.items >= 5, lesson.note
    assert after.accuracy > before.accuracy + 0.3, misses
    # Not asserted: that no program ever fits the shown pairs and fails a held-out one. That does
    # happen — search over examples finds coincidences — and the split exists to *catch* it, which
    # it did: the item is scored wrong and named in `misses`. Demanding it never occur would be
    # demanding a property of the data rather than of the method, and the way to pass such a test
    # is to stop looking.
    assert after.accuracy >= WriteComposite.threshold, misses


def test_teaching_composites_does_not_starve_the_search_for_basics():
    """Regression: taught shapes were ranked ahead of seeds whatever they cost, so every two-hole
    composite was enumerated in full before the zero-hole seed that answers ``sum(xs)`` in one
    attempt — and ``code-basics`` fell from 1.00 to 0.50 on tasks it had just aced."""
    from nyxara.njp.school import WriteBasic
    coder = Coder()
    WriteComposite().teach(None, _mint(71), coder=coder)
    score, misses = WriteBasic().exam(None, _mint(72), coder=coder)
    assert score.accuracy >= 0.85, misses


# --------------------------------------------------------------------------- #
# the whole run
# --------------------------------------------------------------------------- #

def test_a_short_run_reports_a_floor_a_gain_and_a_transcript():
    school = School(seed=3, rounds=1, subjects=(Arithmetic, WriteComposite))
    transcript = school.attend()
    assert len(transcript.results) == 2
    assert transcript.overall.total > 0
    assert transcript.overall.wrong == 0
    composites = transcript.results[1]
    assert composites.taught > 0 and composites.gain > 0
    assert "report card" in transcript.summary()
    assert transcript.to_dict()["results"][1]["subject"] == "code-composites"


def test_retention_examines_what_was_taught_rather_than_a_fresh_stranger():
    """Teacher off, fresh items, and the *same* subject instances — because what `Depth` distils
    is a property of a relation, and a fresh instance would mint a new one and report the floor
    as if it were forgetting."""
    school = School(seed=4, rounds=1, subjects=(Depth,))
    brain = NJPBrain(ExamConditions())
    coder = brain.coder
    taught = school.attend(brain, coder=coder)
    assert taught.results[0].gain > 0.3, taught.results[0].note

    kept = school.retention(brain, coder, seed=99)
    assert kept.results[0].post.accuracy >= 0.7, kept.results[0].misses
    assert kept.results[0].post.wrong == 0
