"""The predictive brain: expecting the world, being wrong, and paying for it in bits.

The tests that matter are the ones a plausible-looking predictor would fail — that learning shows
up as *falling* surprise, that an honest "I don't know" is not punished like a confident error,
that nothing is ever assigned probability zero, and that a thought which cannot decide says so.
"""

from __future__ import annotations

import math

from nyxara.njp.brain import NJPBrain
from nyxara.njp.predictive import (
    PredictiveWorldModel,
    ThoughtState,
    WorldState,
)


def _glass_model(repeats: int = 6) -> PredictiveWorldModel:
    model = PredictiveWorldModel()
    for _ in range(repeats):
        model.observe(WorldState.of(["glass on table"]), "push",
                      next_state=WorldState.of(["glass moved"]))
    return model


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
def test_a_state_is_order_independent():
    """The same situation described in a different order is the same situation."""
    assert WorldState.of(["b", "a"]) == WorldState.of(["a", "b"])
    assert WorldState.of([]).empty


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_it_predicts_what_followed_before():
    model = _glass_model()
    prediction = model.predict(WorldState.of(["glass on table"]), "push")
    assert prediction.answerable
    assert prediction.top == "glass moved"
    assert prediction.confidence > 0.8


def test_the_order_that_answered_is_reported():
    """An answer from order 0 is a claim about the world; from order 3, about this situation."""
    model = _glass_model()
    prediction = model.predict(WorldState.of(["glass on table"]), "push")
    assert prediction.order >= 0
    assert prediction.evidence >= model.min_evidence


def test_backoff_answers_when_the_long_context_is_novel():
    model = _glass_model()
    # A context never seen at high order still gets an answer from a shorter one.
    prediction = model.predict(["never", "seen", "before"], "push")
    assert prediction.answerable
    assert prediction.order < model.max_order


def test_nothing_is_ever_impossible():
    """Zero probability is a claim of impossibility a few dozen observations cannot support."""
    model = _glass_model()
    prediction = model.predict(WorldState.of(["glass on table"]), "push")
    for outcome in prediction.distribution.values():
        assert outcome > 0.0
    assert math.isfinite(prediction.entropy_bits)


def test_an_empty_model_predicts_nothing_rather_than_guessing():
    assert not PredictiveWorldModel().predict().answerable


# --------------------------------------------------------------------------- #
# Surprise
# --------------------------------------------------------------------------- #
def test_the_expected_outcome_is_cheap_and_the_novel_one_is_dear():
    model = _glass_model()
    prediction = model.predict(WorldState.of(["glass on table"]), "push")
    expected = model.score(prediction, WorldState.of(["glass moved"]))
    novel = model.score(prediction, WorldState.of(["glass exploded"]))
    assert expected.correct and not expected.surprising
    assert novel.bits > expected.bits * 5
    assert novel.surprising


def test_surprise_is_finite_even_for_something_never_seen():
    """An unbounded error signal is not a signal — it destroys whatever it is fed into."""
    model = _glass_model()
    prediction = model.predict(WorldState.of(["glass on table"]), "push")
    surprise = model.score(prediction, WorldState.of(["a thing never mentioned"]))
    assert math.isfinite(surprise.bits) and surprise.bits > 0


def test_an_honest_i_dont_know_is_not_punished_like_a_confident_error():
    """The `excess` measure exists so vagueness is not the safe strategy."""
    confident = _glass_model(repeats=20)
    prediction = confident.predict(WorldState.of(["glass on table"]), "push")
    confident_wrong = confident.score(prediction, WorldState.of(["glass stayed"]))

    unsure = PredictiveWorldModel()
    for outcome in ("a", "b", "c", "d", "e", "f"):
        unsure.observe(WorldState.of(["x"]), "push", next_state=WorldState.of([outcome]))
    unsure_prediction = unsure.predict(WorldState.of(["x"]), "push")
    unsure_wrong = unsure.score(unsure_prediction, WorldState.of(["g"]))

    # Both were wrong. Only the one that had claimed certainty carries a real defect.
    assert confident_wrong.excess > unsure_wrong.excess


def test_learning_shows_up_as_falling_surprise():
    """The claim 'it learned' is this number going down, and nothing else."""
    model = PredictiveWorldModel()
    early, late = [], []
    sequence = [WorldState.of([s]) for s in ("a", "b", "c")] * 14
    for i, state in enumerate(sequence):
        prediction, surprise = model.predict_and_observe(state)
        if not prediction.answerable:
            continue
        (early if i < 12 else late).append(surprise.bits)
    assert early and late
    assert sum(late) / len(late) < sum(early) / len(early) / 2, (early, late)
    assert model.accuracy is not None and model.accuracy > 0.7


# --------------------------------------------------------------------------- #
# Thought as an object
# --------------------------------------------------------------------------- #
def _plant_model() -> PredictiveWorldModel:
    model = PredictiveWorldModel()
    for _ in range(7):
        model.observe(WorldState.of(["plant has water"]), "wait",
                      next_state=WorldState.of(["plant grows"]))
    for _ in range(2):
        model.observe(WorldState.of(["plant has water"]), "wait",
                      next_state=WorldState.of(["plant wilts"]))
    model.observe(WorldState.of(["plant has water"]))
    return model


def test_deliberation_walks_every_step_and_records_it():
    thought = ThoughtState(context="plant has water").deliberate(_plant_model(), action="wait")
    assert thought.steps == ["what-do-i-know", "what-am-i-assuming", "what-could-explain",
                             "which-predicts-best", "what-would-prove-me-wrong", "conclude"]


def test_the_assumption_is_named_rather_than_hidden():
    thought = ThoughtState(context="plant has water").deliberate(_plant_model(), action="wait")
    assert thought.assumptions
    assert thought.assumptions[0].checked


def test_every_hypothesis_carries_a_falsifier():
    """A hypothesis without one is not competing; it is decoration."""
    thought = ThoughtState(context="plant has water").deliberate(_plant_model(), action="wait")
    assert thought.hypotheses
    assert all(h.falsifier for h in thought.hypotheses)


def test_a_clear_winner_is_concluded():
    thought = ThoughtState(context="plant has water").deliberate(_plant_model(), action="wait")
    assert thought.decided
    assert "grows" in thought.conclusion


def test_a_tie_is_left_undecided_rather_than_coin_flipped():
    """Two explanations a hair apart are not a conclusion, and reporting one would be a lie.

    Order 1, deliberately. At higher orders an alternating p/q stream is not a tie at all — it is
    a *pattern*, and the model learns it and is right to decide. Constructing the tie means
    denying it the context that would resolve one, which is what ``max_order=1`` does.
    """
    model = PredictiveWorldModel(max_order=1)
    for _ in range(5):
        model.observe(WorldState.of(["x"]), "act", next_state=WorldState.of(["p"]))
        model.observe(WorldState.of(["x"]), "act", next_state=WorldState.of(["q"]))
    prediction = model.predict(WorldState.of(["x"]), "act")
    ranked = sorted(prediction.distribution.values(), reverse=True)
    assert ranked[0] - ranked[1] < 0.1, "the fixture failed to build an actual tie"

    model.observe(WorldState.of(["x"]))
    thought = ThoughtState(context="x").deliberate(model, action="act")
    assert not thought.decided
    assert thought.confidence > 0.0          # it still reports how close it got


def test_uncertainty_is_carried_in_bits():
    thought = ThoughtState(context="plant has water").deliberate(_plant_model(), action="wait")
    assert 0.0 < thought.uncertainty < 4.0


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_builds_and_feeds_the_predictive_model():
    brain = NJPBrain()
    for line in ("aag se garmi milti hai", "garmi se pani ubalta hai",
                 "pani ubalne se bhaap banti hai") * 3:
        brain.think(line)
    stats = brain.stats()["predictive"]
    assert stats["observations"] > 0, "the world model was never fed"
    assert stats["scored"] > 0, "nothing was ever graded against reality"


def test_the_brain_can_deliberate():
    brain = NJPBrain()
    for line in ("aag se garmi milti hai", "garmi se pani ubalta hai") * 4:
        brain.think(line)
    thought = brain.deliberate("aag se garmi milti hai")
    assert thought is not None and thought.steps


def test_state_survives_a_round_trip():
    model = _glass_model()
    revived = PredictiveWorldModel()
    revived.load_dict(model.to_dict())
    assert revived.predict(WorldState.of(["glass on table"]), "push").top == "glass moved"
