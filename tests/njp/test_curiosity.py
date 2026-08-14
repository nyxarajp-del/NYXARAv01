"""Her own questions, and which one is worth asking.

Questions come from measured gaps, never a list. Which one to ask is a decision, not an appetite.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.curiosity import Curiosity, Gap, Question


@pytest.fixture()
def peopled() -> NJPBrain:
    """Three people; two have an employer and one does not. Nothing says they should."""
    b = NJPBrain()
    for line in ("Ravi lives in Delhi", "Ravi works at Infosys",
                 "Sara lives in Chennai", "Sara works at Wipro",
                 "Amit lives in Pune"):
        b.think(line)
    return b


# ---- gaps are discovered ------------------------------------------------------- #
def test_a_missing_relation_becomes_a_question(peopled: NJPBrain):
    """Nothing declares that a person ought to have an employer. It comes from the asymmetry."""
    peopled.curiosity.wonder()
    questions = [q for q in peopled.curiosity.open_questions()
                 if q.gap == Gap.MISSING_RELATION]
    assert questions
    assert any(q.subject == "amit" and q.predicate == "works_at" for q in questions)


def test_no_gap_is_invented_when_the_record_is_complete():
    b = NJPBrain()
    for line in ("Ravi lives in Delhi", "Sara lives in Chennai", "Amit lives in Pune"):
        b.think(line)
    b.curiosity.wonder()
    assert not [q for q in b.curiosity.open_questions() if q.gap == Gap.MISSING_RELATION]


def test_too_few_entities_means_peers_mean_nothing():
    b = NJPBrain()
    b.think("Ravi lives in Delhi")
    b.curiosity.wonder()
    assert not [q for q in b.curiosity.open_questions() if q.gap == Gap.MISSING_RELATION]


def test_a_repeated_failure_becomes_a_question():
    """Wrong once is noise. Wrong the same way five times is a gap."""
    from nyxara.njp.predict import Diagnosis, ErrorKind, Outcome

    b = NJPBrain()
    for _ in range(5):
        b.predictor.outcomes.append(
            Outcome(error=1.0, diagnosis=Diagnosis(kind=ErrorKind.MEMORY)))
    b.curiosity.wonder()
    assert any(q.gap == Gap.REPEATED_FAILURE for q in b.curiosity.open_questions())


def test_one_failure_is_not_a_gap():
    from nyxara.njp.predict import Diagnosis, ErrorKind, Outcome

    b = NJPBrain()
    b.predictor.outcomes.append(
        Outcome(error=1.0, diagnosis=Diagnosis(kind=ErrorKind.MEMORY)))
    b.curiosity.wonder()
    assert not [q for q in b.curiosity.open_questions() if q.gap == Gap.REPEATED_FAILURE]


# ---- deciding what is worth asking ---------------------------------------------- #
def test_every_open_question_is_priced(peopled: NJPBrain):
    peopled.curiosity.wonder()
    for question in peopled.curiosity.open_questions():
        assert question.value >= 0.0
        assert question.action in ("act", "ask", "gather", "")


def test_the_most_valuable_question_is_the_one_she_asks():
    c = Curiosity()
    c.questions[("g", "a", "p")] = Question(text="cheap", value=0.1)
    c.questions[("g", "b", "p")] = Question(text="valuable", value=0.9)
    assert c.top().text == "valuable"


def test_a_worthless_question_is_not_asked():
    """A mind that asks about everything is not curious, it is exhausting."""
    c = Curiosity(min_value=0.5)
    c.questions[("g", "a", "p")] = Question(text="trivial", value=0.01)
    assert c.top() is None


def test_asking_the_master_costs_more_than_investigating():
    """A mind that asks about everything has moved its work onto him."""
    from nyxara.njp.curiosity import _COST_ASK, _COST_GATHER

    assert _COST_ASK > _COST_GATHER


# ---- curiosity that resolves ------------------------------------------------------ #
def test_an_answered_question_does_not_come_back(peopled: NJPBrain):
    """Curiosity that never resolves is just anxiety."""
    peopled.curiosity.wonder()
    question = peopled.curiosity.open_questions()[0]
    assert peopled.curiosity.resolve(question, "answered")

    peopled.curiosity.wonder()
    assert question.key() not in {q.key() for q in peopled.curiosity.open_questions()}


def test_a_question_asked_three_times_goes_stale_rather_than_being_reasked():
    c = Curiosity()
    question = Question(text="q", value=0.9, asked=3)
    c.questions[question.key()] = question
    assert question.stale
    assert c.top() is None
    assert question in c.stale_questions()


def test_asking_increments_the_count():
    c = Curiosity()
    c.questions[("g", "s", "p")] = Question(text="q", value=0.9)
    assert c.ask().asked == 1


def test_resolving_an_unknown_question_is_a_no_op():
    assert Curiosity().resolve(Question(text="never raised")) is False


# ---- persistence -------------------------------------------------------------------- #
def test_questions_survive_a_reload(peopled: NJPBrain):
    peopled.curiosity.wonder()
    fresh = Curiosity()
    fresh.load_dict(peopled.curiosity.to_dict())
    assert {q.text for q in fresh.open_questions()} == {
        q.text for q in peopled.curiosity.open_questions()}


# ---- wired ---------------------------------------------------------------------------- #
def test_the_pulse_wonders_on_its_own_cadence(peopled: NJPBrain):
    report = peopled.pulse.beat(force=True)
    assert report.questions_raised >= 1


def test_an_organ_that_is_off_is_absent():
    class _Off:
        curiosity_enabled = False

    b = NJPBrain(_Off())
    assert b.curiosity is None
    assert "curiosity" not in b.stats()
