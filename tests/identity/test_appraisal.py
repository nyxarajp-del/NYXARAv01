"""Tests for nyxara.identity.appraisal."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import AffectSystem
from nyxara.identity.appraisal import (
    Agency,
    AppraisalDimensions,
    AppraisalEvent,
    AppraisalOutcome,
    Appraiser,
)


@pytest.fixture
def ap():
    return Appraiser()


def _ev(**kw):
    return AppraisalEvent("e", AppraisalDimensions(**kw))


# -------------------- attribution / agency -------------------- #
def test_threat_by_other_is_anger(ap):
    out = ap.appraise(_ev(goal_relevance=1.0, desirability=-0.9, agency=Agency.OTHER,
                          norm_compatibility=-1.0))
    assert out.emotion == "anger"
    assert out.valence < 0 and out.arousal > 0.7


def test_owner_praise_is_gratitude(ap):
    out = ap.appraise(_ev(goal_relevance=0.9, desirability=0.8, agency=Agency.OWNER))
    assert out.emotion == "gratitude"
    assert out.valence > 0


def test_owner_displeasure_is_shame(ap):
    out = ap.appraise(_ev(goal_relevance=0.9, desirability=-0.6, agency=Agency.OWNER))
    assert out.emotion == "shame"


def test_self_success_is_pride(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=0.7, agency=Agency.SELF))
    assert out.emotion == "pride"


def test_self_failure_is_remorse(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, agency=Agency.SELF))
    assert out.emotion == "remorse"


def test_other_benefit_is_gratitude(ap):
    out = ap.appraise(_ev(goal_relevance=0.7, desirability=0.6, agency=Agency.OTHER))
    assert out.emotion == "gratitude"


def test_circumstance_good_is_joy(ap):
    out = ap.appraise(_ev(goal_relevance=0.7, desirability=0.6, agency=Agency.CIRCUMSTANCE))
    assert out.emotion == "joy"


def test_circumstance_bad_is_distress(ap):
    out = ap.appraise(_ev(goal_relevance=0.7, desirability=-0.6, agency=Agency.CIRCUMSTANCE))
    assert out.emotion == "distress"


# -------------------- prospect-based -------------------- #
def test_prospect_desirable_is_hope(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=0.7, likelihood=0.5,
                          prospect=True))
    assert out.emotion == "hope"


def test_prospect_undesirable_is_fear(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, likelihood=0.6,
                          prospect=True))
    assert out.emotion == "fear"


def test_prospect_good_confirmed_is_satisfaction(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=0.7, prospect=True,
                          confirmed=True))
    assert out.emotion == "satisfaction"


def test_prospect_good_disconfirmed_is_disappointment(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=0.7, prospect=True,
                          confirmed=False))
    assert out.emotion == "disappointment"


def test_prospect_bad_confirmed_is_fears_confirmed(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, prospect=True,
                          confirmed=True))
    assert out.emotion == "fears_confirmed"


def test_prospect_bad_disconfirmed_is_relief(ap):
    out = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, prospect=True,
                          confirmed=False))
    assert out.emotion == "relief"


# -------------------- relevance / intensity -------------------- #
def test_irrelevant_is_indifference(ap):
    out = ap.appraise(_ev(goal_relevance=0.02, desirability=0.9))
    assert out.emotion == "indifference"
    assert out.intensity < 0.2


def test_intensity_scales_with_relevance_and_desirability(ap):
    weak = ap.appraise(_ev(goal_relevance=0.2, desirability=0.2,
                           agency=Agency.CIRCUMSTANCE))
    strong = ap.appraise(_ev(goal_relevance=1.0, desirability=0.9,
                             agency=Agency.CIRCUMSTANCE))
    assert strong.intensity > weak.intensity


def test_low_controllability_raises_arousal(ap):
    helpless = ap.appraise(_ev(goal_relevance=0.9, desirability=-0.7,
                               agency=Agency.CIRCUMSTANCE, controllability=0.0))
    in_control = ap.appraise(_ev(goal_relevance=0.9, desirability=-0.7,
                                 agency=Agency.CIRCUMSTANCE, controllability=1.0))
    assert helpless.arousal > in_control.arousal


def test_prospect_likelihood_affects_intensity(ap):
    likely = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, likelihood=0.9,
                             prospect=True))
    unlikely = ap.appraise(_ev(goal_relevance=0.8, desirability=-0.7, likelihood=0.1,
                               prospect=True))
    assert likely.intensity > unlikely.intensity


# -------------------- outcome shape -------------------- #
def test_outcome_fields(ap):
    out = ap.appraise(ap.owner_threatened(0.8))
    assert isinstance(out, AppraisalOutcome)
    assert -1.0 <= out.valence <= 1.0
    assert 0.0 <= out.arousal <= 1.0
    assert out.reasoning
    assert "→" in out.reasoning


def test_outcome_to_dict(ap):
    d = ap.appraise(ap.own_success()).to_dict()
    assert "emotion" in d and "valence" in d and "reasoning" in d


# -------------------- conveniences -------------------- #
def test_owner_threatened_builder(ap):
    ev = ap.owner_threatened(0.9)
    assert ev.dims.agency is Agency.OTHER
    assert ev.dims.desirability < 0
    assert ev.dims.goal_relevance == 1.0


def test_owner_praise_builder(ap):
    ev = ap.owner_praise(0.8)
    assert ev.dims.agency is Agency.OWNER and ev.dims.desirability > 0


def test_prospect_builder(ap):
    ev = ap.prospect(0.5, 0.5)
    assert ev.dims.prospect is True


# -------------------- integration with affect -------------------- #
def test_appraise_and_feel_pushes_to_affect(ap):
    affect = AffectSystem()
    out = ap.appraise_and_feel(affect, ap.owner_threatened(0.9))
    assert affect.emotion is not None
    assert affect.emotion.cause == "a threat to the Master"
    # the threat produced a negative-valence feeling
    assert affect.emotion.valence < 0


def test_dimensions_to_dict():
    d = AppraisalDimensions(goal_relevance=0.5, agency=Agency.OWNER).to_dict()
    assert d["agency"] == "owner"
