"""Tests for nyxara.mind.predictive_core."""

from __future__ import annotations

import math

import pytest

from nyxara.mind.predictive_core import (
    Action,
    EmotionReadout,
    PerceptionResult,
    PredictiveCore,
)


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


# -------------------- perception -------------------- #
def test_belief_converges_to_observation():
    core = PredictiveCore(belief=[0.0, 0.0], sensory_precision=2.0,
                          prior_precision=0.01, learning_rate=0.3)
    for _ in range(40):
        core.perceive([3.0, -1.0])
    assert _dist2(core.mu, [3.0, -1.0]) < 0.2


def test_free_energy_decreases():
    core = PredictiveCore(belief=[0.0], sensory_precision=2.0, learning_rate=0.4)
    F0 = core.perceive([5.0]).free_energy
    for _ in range(20):
        r = core.perceive([5.0])
    assert r.free_energy < F0


def test_perception_result_fields():
    core = PredictiveCore(belief=[0.0])
    r = core.perceive([1.0])
    assert isinstance(r, PerceptionResult)
    assert len(r.belief) == 1
    assert r.error_magnitude >= 0
    assert r.iterations == 8


def test_identity_model_default():
    core = PredictiveCore(belief=[2.0])
    assert core.predict() == [2.0]


def test_prior_pulls_belief():
    # strong prior at 0 resists moving toward observation
    weak = PredictiveCore(belief=[0.0], prior=[0.0], sensory_precision=1.0,
                          prior_precision=0.01, learning_rate=0.2)
    strong = PredictiveCore(belief=[0.0], prior=[0.0], sensory_precision=1.0,
                            prior_precision=5.0, learning_rate=0.2)
    for _ in range(30):
        weak.perceive([10.0])
        strong.perceive([10.0])
    assert abs(weak.mu[0] - 10.0) < abs(strong.mu[0] - 10.0)


# -------------------- nonlinear model -------------------- #
def test_nonlinear_generative_model():
    nl = PredictiveCore(belief=[1.0], generative_model=lambda s: [s[0] ** 2],
                        sensory_precision=1.0, prior_precision=0.01, learning_rate=0.05)
    for _ in range(80):
        nl.perceive([9.0])
    assert abs(nl.mu[0] - 3.0) < 0.3


# -------------------- emotion -------------------- #
def test_emotion_valence_positive_while_improving():
    # online (1-iteration) steps so the belief catches up gradually
    core = PredictiveCore(belief=[0.0], sensory_precision=2.0, learning_rate=0.15)
    core.step([5.0], iterations=1)
    _, em = core.step([5.0], iterations=1)
    assert em.valence >= 0


def test_emotion_surprise_drops_with_learning():
    core = PredictiveCore(belief=[0.0], sensory_precision=2.0, learning_rate=0.15)
    _, em1 = core.step([5.0], iterations=1)
    for _ in range(15):
        _, em = core.step([5.0], iterations=1)
    assert em.surprise < em1.surprise


def test_arousal_spikes_on_surprise():
    core = PredictiveCore(belief=[5.0], sensory_precision=3.0)
    _, calm = core.step([5.0])
    _, shock = core.step([50.0])
    assert shock.arousal > calm.arousal


def test_emotion_ranges():
    core = PredictiveCore(belief=[0.0])
    core.step([1.0])
    em = core.emotion()
    assert -1.0 <= em.valence <= 1.0
    assert 0.0 <= em.arousal <= 1.0
    assert 0.0 <= em.surprise <= 1.0
    assert isinstance(em, EmotionReadout)


def test_emotion_no_history():
    core = PredictiveCore(belief=[0.0])
    em = core.emotion()  # no steps yet
    assert em.valence == 0.0 or em.valence == math.tanh(0.0)


# -------------------- action / active inference -------------------- #
def test_act_chooses_preference_fulfilling():
    agent = PredictiveCore(belief=[0.0], preference=[10.0])
    actions = [
        Action("nothing", [0.0]),
        Action("approach", [9.5]),
        Action("overshoot", [25.0]),
    ]
    choice = agent.act(actions)
    assert choice.action.name == "approach"


def test_act_epistemic_value_breaks_ties():
    agent = PredictiveCore(belief=[0.0], preference=[10.0], epistemic_weight=5.0)
    # both equally pragmatic, but one resolves uncertainty
    a = Action("explore", [10.0], info_gain=1.0)
    b = Action("exploit", [10.0], info_gain=0.0)
    choice = agent.act([b, a])
    assert choice.action.name == "explore"


def test_act_requires_preference():
    agent = PredictiveCore(belief=[0.0])  # no preference
    with pytest.raises(ValueError):
        agent.expected_free_energy(Action("x", [1.0]))


def test_act_empty_raises():
    agent = PredictiveCore(belief=[0.0], preference=[1.0])
    with pytest.raises(ValueError):
        agent.act([])


def test_expected_free_energy_components():
    agent = PredictiveCore(belief=[0.0], preference=[10.0], preference_precision=1.0,
                           epistemic_weight=1.0)
    choice = agent.expected_free_energy(Action("a", [8.0], info_gain=0.5))
    assert choice.pragmatic == pytest.approx(0.5 * 1.0 * (8.0 - 10.0) ** 2)
    assert choice.epistemic == pytest.approx(0.5)
    assert choice.expected_free_energy == pytest.approx(choice.pragmatic - choice.epistemic)


# -------------------- precision / status -------------------- #
def test_update_precision_high_error_lowers():
    core = PredictiveCore(belief=[0.0], sensory_precision=1.0)
    core.perceive([100.0], iterations=1)  # leaves a big error
    p = core.update_precision()
    assert p < 1.0  # noisy channel -> trust it less


def test_step_returns_perception_and_emotion():
    core = PredictiveCore(belief=[0.0])
    perception, feeling = core.step([1.0])
    assert isinstance(perception, PerceptionResult)
    assert isinstance(feeling, EmotionReadout)


def test_status_keys():
    core = PredictiveCore(belief=[0.0])
    core.step([1.0])
    s = core.status()
    for k in ("belief", "prediction", "free_energy", "sensory_precision", "emotion"):
        assert k in s


def test_free_energy_computation():
    # identity model, belief=prior=0, obs=2 -> accuracy=0.5*prec*4, complexity=0
    core = PredictiveCore(belief=[0.0], prior=[0.0], sensory_precision=1.0,
                          prior_precision=1.0)
    F = core.free_energy([2.0])
    assert F == pytest.approx(0.5 * 1.0 * 4.0)
