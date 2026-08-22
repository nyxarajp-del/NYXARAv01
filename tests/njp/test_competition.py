"""Ranking rival accounts instead of counting them — njp/compete.py.

``router.py`` reconciles two seats on structural rules, which is right for two and does not extend
to three. This is the ranking, and it is deliberately not a vote: three seats agreeing says more
about their shared inputs than about the world, and a vote makes every seat equal when NJP owns the
fact store, the cortex proposes and the fabric associates.

**The fabric is entered as the weak source it is.** Its proposals come from a head trained on
turn-to-turn concept succession, measured at MRR 0.238, carrying no provenance and no derivation.
It scores ~0 on the two heaviest axes and loses — with the axis that cost it recorded. That is the
competition working, not a reason to keep it out: ranking is what makes entering a weak account
safe where a vote would have made it dangerous.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.compete import (SUPPORT_FLOOR, WEIGHTS, Axis, Competition, Hypothesis,
                                Verdict)
from nyxara.njp.grounding import Epistemic, Provenance


def _triple(subject: str, predicate: str, obj: str):
    return type("_T", (), {"as_tuple": lambda self: (subject, predicate, obj), "object": obj})()


def _record(claim: str = "water", confidence: float = 0.9) -> Hypothesis:
    return Hypothesis(seat="njp", claim=claim, confidence=confidence,
                      provenance=Provenance.OBSERVED, state=Epistemic.KNOWN,
                      support=[_triple("bird", "requires", "water")])


def _proposal(seat: str, claim: str, confidence: float = 0.9) -> Hypothesis:
    return Hypothesis(seat=seat, claim=claim, confidence=confidence,
                      provenance=Provenance.PROPOSED, state=Epistemic.BELIEVED)


# --------------------------------------------------------------------------- #
# It is not a vote
# --------------------------------------------------------------------------- #
def test_two_proposals_do_not_outvote_the_record():
    outcome = Competition().compete(
        [_record(), _proposal("cortex", "sunlight"), _proposal("fabric", "grain")])
    assert outcome.verdict == Verdict.RECORD
    assert outcome.winner == "njp"
    assert outcome.claim == "water"


def test_the_most_confident_account_does_not_win_by_being_confident():
    """Stated confidence is the one axis a seat can move by itself, so it is worth least."""
    outcome = Competition().compete([
        _record("water", confidence=0.60),
        _proposal("cortex", "sunlight", confidence=0.99),
    ])
    assert outcome.claim == "water"


def test_confidence_is_the_lowest_weighted_axis():
    assert WEIGHTS[Axis.CONFIDENCE] == min(WEIGHTS.values())
    assert WEIGHTS[Axis.EVIDENCE] == max(WEIGHTS.values())


# --------------------------------------------------------------------------- #
# Nothing wins by default
# --------------------------------------------------------------------------- #
def test_three_unsupported_accounts_produce_a_question_not_an_answer():
    outcome = Competition().compete([
        _proposal("cortex", "sunlight"), _proposal("fabric", "grain"),
        _proposal("njp", "seeds", confidence=0.4)])
    assert outcome.verdict == Verdict.UNSUPPORTED
    assert not outcome.settled
    assert len(outcome.rivals) == 3, "the rivals are kept whole for an experiment"


def test_an_inseparable_field_declares_no_winner():
    """A margin inside the noise of the weights is a coin flip wearing a score."""
    a = _record("water", confidence=0.9)
    b = _record("food", confidence=0.9)
    b.seat = "cortex"
    outcome = Competition(floor=0.0).compete([a, b])
    assert outcome.verdict == Verdict.TIED
    assert not outcome.settled


def test_an_empty_field_is_silent():
    assert Competition().compete([]).verdict == Verdict.SILENT


def test_a_blank_claim_is_not_an_account():
    assert Competition().compete([_proposal("cortex", "  ")]).verdict == Verdict.SILENT


# --------------------------------------------------------------------------- #
# A loss has a reason
# --------------------------------------------------------------------------- #
def test_every_account_keeps_its_per_axis_breakdown():
    outcome = Competition().compete([_record(), _proposal("fabric", "grain")])
    assert outcome.scores
    for score in outcome.scores:
        assert set(score.axes) == set(Axis.ALL)


def test_the_fabric_loses_on_evidence_and_says_so():
    outcome = Competition().compete([_record(), _proposal("fabric", "grain")])
    fabric = next(s for s in outcome.scores if s.hypothesis.seat == "fabric")
    assert fabric.weakest == Axis.EVIDENCE


def test_repeating_a_support_does_not_inflate_evidence():
    """A claim resting on the same fact three times rests on one fact."""
    once = Hypothesis(seat="njp", claim="water", provenance=Provenance.DERIVED,
                      support=[_triple("bird", "requires", "water")])
    thrice = Hypothesis(seat="njp", claim="water", provenance=Provenance.DERIVED,
                        support=[_triple("bird", "requires", "water")] * 3)
    competition = Competition()
    assert competition.score(once).axes[Axis.EVIDENCE] == \
        competition.score(thrice).axes[Axis.EVIDENCE]


def test_a_shorter_derivation_is_preferred():
    short = Hypothesis(seat="njp", claim="a", support=[_triple("x", "p", "y")])
    long = Hypothesis(seat="njp", claim="a", support=[_triple("x", "p", "y")] * 4)
    competition = Competition()
    assert competition.score(short).axes[Axis.PARSIMONY] > \
        competition.score(long).axes[Axis.PARSIMONY]


# --------------------------------------------------------------------------- #
# Untested is not untrustworthy
# --------------------------------------------------------------------------- #
def test_a_seat_with_no_track_record_is_scored_neutrally():
    """A new seat must not be scored as one with a bad history, or it could never win."""
    assert Competition().score(_proposal("brand_new", "x")).axes[Axis.RECORD] == 0.5


def test_no_world_model_scores_causal_consistency_neutrally():
    """An unanswerable question must not be scored as a failed one."""
    assert Competition(world=None).score(_record()).axes[Axis.CAUSAL] == 0.5


# --------------------------------------------------------------------------- #
# On the ordinary turn path, with no language model attached
# --------------------------------------------------------------------------- #
def test_the_competition_runs_without_any_cortex():
    """An organ that only runs when a language model is attached is the defect this avoids."""
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "birds need water"):
        brain.think(turn)
    brain.think("what does a sparrow need?")
    assert brain.competition.stats()["run"] > 0


def test_a_turn_carries_its_ranked_field():
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "birds need water") * 4:
        brain.think(turn)
    thought = brain.think("what does a sparrow need?")
    assert thought.competition is not None
    assert thought.competition.scores


def test_the_floor_is_what_stops_a_weak_field_from_answering():
    strict = Competition(floor=0.99)
    outcome = strict.compete([_record(), _proposal("fabric", "grain")])
    assert outcome.verdict in (Verdict.RECORD, Verdict.UNSUPPORTED)


def test_stats_report_the_floor_and_weights_rather_than_hiding_them():
    stats = Competition().stats()
    assert stats["floor"] == SUPPORT_FLOOR
    assert stats["weights"] == WEIGHTS
