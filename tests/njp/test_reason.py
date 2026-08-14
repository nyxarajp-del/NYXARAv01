"""Compute matched to the question, and a principled "I am not sure".

Before this there was one mode: the fabric settled, something came back, and whether that was a
reflex or a conclusion was indistinguishable from outside.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.reason import Conclusion, Hypothesis, ProblemState, Reasoner, Rung


@pytest.fixture()
def brain() -> NJPBrain:
    b = NJPBrain()
    b.think("my name is Jay")
    b.think("I live in Mumbai")
    return b


# ---- routing ----------------------------------------------------------------- #
def test_an_easy_certain_question_stays_shallow():
    """Running a proof engine over "what is my name" is not careful, it is broken."""
    assert Reasoner().depth_for(uncertainty=0.05, stakes=0.05) == Rung.INTUITION


def test_a_hard_uncertain_question_goes_deep():
    assert Reasoner().depth_for(uncertainty=0.95, stakes=0.95) == Rung.PROOF


def test_stakes_deepen_the_search_even_when_she_feels_sure():
    """The cost of being confidently wrong scales with what depends on it."""
    r = Reasoner()
    shallow = r.depth_for(uncertainty=0.2, stakes=0.05)
    deep = r.depth_for(uncertainty=0.2, stakes=0.95)
    assert Rung.cost(deep) > Rung.cost(shallow)


def test_the_ceiling_is_respected():
    r = Reasoner(max_rung=Rung.ASSOCIATION)
    assert r.depth_for(uncertainty=1.0, stakes=1.0) == Rung.ASSOCIATION


def test_an_easy_question_actually_stops_early(brain: NJPBrain):
    got = Reasoner(brain).reason("what is my name", uncertainty=0.2, stakes=0.2)
    assert got.answer == "Jay"
    assert got.depth <= Rung.cost(Rung.DELIBERATION)


def test_a_hard_question_actually_descends(brain: NJPBrain):
    got = Reasoner(brain).reason("is the universe finite", uncertainty=0.95, stakes=0.9)
    assert got.depth >= Rung.cost(Rung.VERIFICATION)


# ---- deciding, and refusing to ------------------------------------------------ #
def test_a_clear_winner_is_decided():
    r = Reasoner()
    got = r.reason("q", candidates=[Hypothesis(claim="a", support=0.9),
                                    Hypothesis(claim="b", support=0.2)],
                   uncertainty=0.5)
    assert got.decided and got.answer == "a"


def test_two_close_readings_are_reported_as_a_tie():
    """Returning the marginally higher number would invent a decision she has not made."""
    r = Reasoner()
    got = r.reason("q", candidates=[Hypothesis(claim="a", support=0.51),
                                    Hypothesis(claim="b", support=0.49)],
                   uncertainty=0.5)
    assert not got.decided
    assert any("too close to call" in w for w in got.why)


def test_a_zero_scored_candidate_is_not_an_answer():
    """A caller that only reads `answer` must not be handed something nothing supports."""
    r = Reasoner()
    got = r.reason("q", candidates=[Hypothesis(claim="unsupported", support=0.0)],
                   uncertainty=0.5)
    assert got.answer == "" and not got.decided


def test_a_refuted_hypothesis_scores_zero_however_supported():
    h = Hypothesis(claim="a", support=1.0, refuted=True)
    assert h.score == 0.0


def test_a_proved_hypothesis_wins_outright():
    assert Hypothesis(claim="a", support=0.1, proved=True).score == 1.0


# ---- the problem state --------------------------------------------------------- #
def test_a_problem_is_resumed_not_restarted():
    r = Reasoner()
    r.reason("a long problem", uncertainty=0.5)
    r.reason("a long problem", uncertainty=0.5)
    assert r.state_for("a long problem").turns == 2


def test_a_rejected_hypothesis_is_kept_and_not_reproposed():
    """Without this she walks into the same dead end on every revisit."""
    r = Reasoner()
    state = r.state_for("p")
    dead_end = Hypothesis(claim="the wrong idea", support=0.5)
    r.propose(state, dead_end)
    state.reject(dead_end, why="tried it, did not work")

    assert state.was_rejected("the wrong idea")
    assert r.propose(state, Hypothesis(claim="the wrong idea", support=0.9)) is None
    assert dead_end in state.rejected


def test_an_unanswerable_question_is_recorded_as_a_known_unknown(brain: NJPBrain):
    """What turns a stuck problem into a question worth asking."""
    r = Reasoner(brain)
    r.reason("what is my pin code", uncertainty=0.5, stakes=0.5)
    assert "what is my pin code" in r.state_for("what is my pin code").unknown


def test_a_problem_with_nothing_live_reports_stuck():
    r = Reasoner()
    state = r.state_for("p")
    state.turns = 2
    assert state.stuck


def test_duplicate_proposals_merge_rather_than_stack():
    r = Reasoner()
    state = r.state_for("p")
    r.propose(state, Hypothesis(claim="same", support=0.3))
    r.propose(state, Hypothesis(claim="same", support=0.8))
    assert len(state.hypotheses) == 1 and state.hypotheses[0].support == 0.8


# ---- verification -------------------------------------------------------------- #
def test_a_reading_that_contradicts_what_she_believes_is_penalised(brain: NJPBrain):
    r = Reasoner(brain)
    state = r.state_for("where does the Master live")
    contradicting = Hypothesis(claim="Master located_in Delhi", support=0.9)
    r.propose(state, contradicting)
    r._verify(state, Conclusion())
    assert contradicting.contradicted and contradicting.against > 0.0


# ---- reporting ------------------------------------------------------------------ #
def test_the_depth_reached_is_recorded(brain: NJPBrain):
    """"She thought hard about this" has to be a checkable claim."""
    r = Reasoner(brain)
    r.reason("what is my name", uncertainty=0.1, stakes=0.1)
    r.reason("something genuinely hard", uncertainty=0.95, stakes=0.95)
    stats = r.stats()
    assert stats["passes"] == 2
    assert stats["mean_depth"] is not None and sum(stats["by_rung"].values()) == 2


# ---- wired ----------------------------------------------------------------------- #
def test_the_brain_builds_a_reasoner():
    assert NJPBrain().reasoner is not None


def test_an_organ_that_is_off_is_absent():
    class _Off:
        reason_enabled = False

    b = NJPBrain(_Off())
    assert b.reasoner is None
    assert "reason" not in b.stats()
