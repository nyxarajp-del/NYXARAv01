"""A debate needs more than one voice, and the causal record was never in the room.

Measured over 1,320 turns: ``reason.passes`` 237, ``undecided`` **235**, ``mean_depth`` 1.03,
``by_rung`` almost entirely ``association``. Inspected on a live brain, the two "competing
readings" being weighed for *"pyaas ka karan kya hai?"* were::

    'noted: aag causes garmi'      score 0.0
    'noted: garmi causes pasina'   score 0.0

Her own past acknowledgements, recalled at similarity zero. Three defects behind that, each in a
different module and all of the same shape — the observational path works and the testimony path
is invisible.

1. ``world.why`` walked ``_pairs`` and skipped anything with ``together <= 0``. A law she was
   *told* lives in ``_stated`` and has no co-occurrence count, so on a brain taught from text —
   where every causal link is stated and none is observed — it could not name a single cause.
   ``links()`` in the same class already reads both intakes.
2. ``reason._propose_causal`` found its token by stem-matching ``world._counts``, which is a
   register of *occurrences* and is empty for the same reason. So its one entry point was never
   reached, and the causal record never contributed a rival to any debate.
3. ``propose`` accepted hypotheses with zero support, which can never win and do pad the field
   enough to report "weighed 2 competing readings".
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.reason import Conclusion, Hypothesis, ProblemState
from nyxara.njp.world import Event, WorldView


@pytest.fixture()
def told() -> NJPBrain:
    """Two stated causes of one effect, and nothing ever observed."""
    brain = NJPBrain()
    for sentence in ("aag se garmi hoti hai", "dhoop se garmi hoti hai"):
        brain.think(sentence)
    return brain


# --------------------------------------------------------------------------- #
# 1. A law she was told can be asked about
# --------------------------------------------------------------------------- #
def test_a_stated_law_can_explain_an_effect(told):
    explanation = told.world.why("garmi")
    assert explanation.explained
    assert {c.cause for c in explanation.causes} == {"aag", "dhoop"}


def test_an_unconnected_pair_is_still_no_explanation(told):
    assert not told.world.why("keechad").explained


def test_why_and_links_agree_about_what_is_causal(told):
    """They read one record and used to disagree about whether testimony counted."""
    from_links = {(l.cause, l.effect) for l in told.world.links(causal_only=True)}
    from_why = {(c.cause, "garmi") for c in told.world.why("garmi").causes}
    assert from_why <= from_links


# --------------------------------------------------------------------------- #
# 2. The causal record reaches the debate
# --------------------------------------------------------------------------- #
def test_the_question_names_the_effect_without_stem_matching(told):
    """The grounder already parses the question; stem-matching an empty table did not."""
    found = told.reasoner._effects_asked_about(told.world, "garmi ka karan kya hai?")
    assert "garmi" in found


def test_the_causal_record_contributes_rivals(told):
    state = ProblemState(problem="garmi ka karan kya hai?")
    told.reasoner._propose_causal(state)
    claims = {h.claim for h in state.hypotheses}
    assert any("aag" in c for c in claims)
    assert any("dhoop" in c for c in claims)
    assert all(h.source == "world" for h in state.hypotheses)


def test_a_stated_law_says_it_was_stated_rather_than_seen(told):
    """"seen together 0 time(s)" describes the opposite of the case for a law."""
    state = ProblemState(problem="garmi ka karan kya hai?")
    told.reasoner._propose_causal(state)
    assert all("stated as a law" in h.evidence[0] for h in state.hypotheses)


def test_an_observed_link_still_reports_its_count():
    brain = NJPBrain()
    world: WorldView = brain.world
    for _ in range(6):
        world.observe(Event(action="rooster"))
        world.observe(Event(action="sunrise"))
    state = ProblemState(problem="sunrise ka karan kya hai?")
    brain.reasoner._propose_causal(state)
    if state.hypotheses:
        assert any("seen together" in h.evidence[0] for h in state.hypotheses)


# --------------------------------------------------------------------------- #
# 3. A candidate nothing supports is not a candidate
# --------------------------------------------------------------------------- #
def test_a_zero_support_hypothesis_is_refused(told):
    state = ProblemState(problem="anything")
    assert told.reasoner.propose(state, Hypothesis(claim="a guess", support=0.0)) is None
    assert not state.hypotheses


def test_a_proved_hypothesis_is_admitted_whatever_its_support(told):
    state = ProblemState(problem="anything")
    assert told.reasoner.propose(
        state, Hypothesis(claim="a theorem", support=0.0, proved=True)) is not None


# --------------------------------------------------------------------------- #
# What the debate is for: two explanations, and an honest tie
# --------------------------------------------------------------------------- #
def test_two_equally_supported_causes_leave_her_undecided(told):
    state = ProblemState(problem="garmi ka karan kya hai?")
    told.reasoner._propose_causal(state)
    out = Conclusion()
    told.reasoner._settle(state, out)
    assert len(state.live) == 2
    assert not out.decided
    assert out.margin == pytest.approx(0.0)
    assert any("too close to call" in w for w in out.why)


def test_being_torn_is_reported_in_bits(told):
    state = ProblemState(problem="garmi ka karan kya hai?")
    told.reasoner._propose_causal(state)
    out = Conclusion()
    told.reasoner._settle(state, out)
    assert out.uncertainty_bits == pytest.approx(1.0, abs=0.01)


def test_a_clearly_better_cause_wins():
    brain = NJPBrain()
    for _ in range(3):
        brain.think("aag se garmi hoti hai")
    brain.think("dhoop se garmi hoti hai")
    state = ProblemState(problem="garmi ka karan kya hai?")
    brain.reasoner._propose_causal(state)
    out = Conclusion()
    brain.reasoner._settle(state, out)
    if len(state.live) > 1 and out.margin > brain.reasoner.decide_margin:
        assert out.decided
        assert "aag" in out.answer


# --------------------------------------------------------------------------- #
# The ladder still refuses to spend depth it does not need
# --------------------------------------------------------------------------- #
def test_a_question_the_structure_answers_stops_at_association(told):
    """Running a debate over a fact she simply holds is the other failure mode."""
    conclusion = told.reasoner.reason("garmi ka karan kya hai?")
    assert conclusion.decided
    assert conclusion.rung == "association"
