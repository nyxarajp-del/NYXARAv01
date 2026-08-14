"""Something has to lose.

Every organ wants to speak on every turn. Handed to a caller as a pile, that is not integration —
it is six opinions and no cognition.
"""

from __future__ import annotations

import pytest

from nyxara.njp.attention import Attention, Candidate
from nyxara.njp.brain import NJPBrain


@pytest.fixture()
def brain() -> NJPBrain:
    b = NJPBrain()
    b.think("my name is Jay")
    b.think("I live in Mumbai")
    return b


# ---- salience is measured ------------------------------------------------------ #
def test_salience_uses_all_five_terms():
    base = Candidate(source="s")
    assert base.salience == 0.0
    for term in ("novelty", "relevance", "uncertainty", "goal_relevance", "prediction_error"):
        one = Candidate(source="s", **{term: 1.0})
        assert one.salience > 0.0, f"{term} does not contribute to salience"


def test_being_wrong_weighs_most():
    """Prediction error is the one signal that reliably means "attend to this now"."""
    wrong = Candidate(source="a", prediction_error=1.0)
    new = Candidate(source="b", novelty=1.0)
    assert wrong.salience > new.salience


def test_novelty_alone_does_not_dominate():
    """Every cold-start turn is novel; letting that win makes a new brain permanently distractible."""
    novel = Candidate(source="a", novelty=1.0)
    relevant = Candidate(source="b", relevance=1.0, goal_relevance=1.0)
    assert relevant.salience > novel.salience


# ---- the bottleneck -------------------------------------------------------------- #
def test_exactly_one_candidate_wins():
    got = Attention().attend(candidates=[
        Candidate(source="a", relevance=0.9),
        Candidate(source="b", relevance=0.5),
        Candidate(source="c", relevance=0.1)])
    assert got.attended
    assert got.considered == 3
    assert len(got.losers) == 2


def test_the_winner_is_not_listed_among_the_losers():
    """Habituation means the top-ranked bid often does not win, so rank is not the answer."""
    got = Attention().attend(candidates=[
        Candidate(source="a", relevance=0.9), Candidate(source="b", relevance=0.5)])
    assert got.source not in got.losers


def test_no_bids_means_nothing_is_attended_to():
    got = Attention().attend(candidates=[])
    assert not got.attended and got.considered == 0


def test_inhibition_of_return_hands_attention_over():
    """What just won is suppressed, so a rival that lost narrowly takes the next cycle.

    IOR is a suppression, not a veto: with nothing else competing, the dampened item can still
    win, and that is correct — there is nothing to move on *to*. What must not happen is one
    source holding her attention while something comparable is waiting.
    """
    attention = Attention()
    field = lambda: [Candidate(source="loud", relevance=0.9, tags=("x",)),  # noqa: E731
                     Candidate(source="rival", relevance=0.85, tags=("y",))]
    first = attention.attend(candidates=field())
    second = attention.attend(candidates=field())
    assert first.source == "loud"
    assert second.source == "rival", "the same source held attention through a live rival"


def test_a_lone_suppressed_candidate_can_still_win():
    """With nothing to move on to, suppression must not mean paralysis."""
    attention = Attention()
    bid = lambda: [Candidate(source="same", relevance=0.9, tags=("x",))]  # noqa: E731
    attention.attend(candidates=bid())
    assert attention.attend(candidates=bid()).attended


def test_a_different_source_can_still_win_after_a_suppression():
    attention = Attention()
    attention.attend(candidates=[Candidate(source="a", relevance=0.9, tags=("x",))])
    got = attention.attend(candidates=[Candidate(source="b", relevance=0.9, tags=("y",))])
    assert got.attended and got.source == "b"


def test_capacity_one_is_the_point():
    assert Attention().capacity == 1


# ---- wired ------------------------------------------------------------------------ #
def test_a_turn_records_what_won_it(brain: NJPBrain):
    thought = brain.think("what is my name")
    assert thought.focus is not None
    assert thought.focus.considered > 0


def test_the_organs_actually_bid(brain: NJPBrain):
    """A silent organ must not dilute the field, but a live one must be in it."""
    thought = brain.think("where do I live")
    sources = {c.source for c in brain.attention.gather(thought)}
    assert sources, "no organ bid at all"
    assert sources <= {"grounding", "memory", "prediction", "surprise", "curiosity", "goal"}


def test_an_organ_that_is_off_is_absent():
    class _Off:
        attention_enabled = False

    b = NJPBrain(_Off())
    assert b.attention is None
    assert "attention" not in b.stats()
