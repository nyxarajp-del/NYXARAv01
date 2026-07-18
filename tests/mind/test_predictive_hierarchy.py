"""Tests: hierarchical predictive core, per-dimension precision, deep affect."""

from __future__ import annotations

import math
import random

from nyxara.mind.predictive_core import (
    Action,
    HierarchicalPredictiveCore,
    PredictiveCore,
)


def _std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# -------------------- hierarchy -------------------- #
def test_hierarchy_upper_slower_than_lower():
    h = HierarchicalPredictiveCore(belief=[0.0], slow_every=4, slow_lr=0.05,
                                   learning_rate=0.5, sensory_precision=2.0)
    lower_track, upper_track = [], []
    for t in range(60):
        h.perceive([4.0 if t % 2 == 0 else 6.0], iterations=2)   # fast oscillation
        lower_track.append(h.lower.mu[0])
        upper_track.append(h.upper.mu[0])
    assert _std(upper_track[20:]) < _std(lower_track[20:])       # slow level is calmer


def test_upper_prior_guides_lower():
    h = HierarchicalPredictiveCore(belief=[0.0], slow_every=2, slow_lr=0.2,
                                   learning_rate=0.3)
    for _ in range(40):
        h.perceive([5.0], iterations=2)
    # the top-down prediction became the fast level's empirical prior
    assert abs(h.upper.mu[0] - 5.0) < 1.5
    assert abs(h.lower.prior[0]) > 0.5            # prior moved off its initial 0
    assert abs(h.lower.mu[0] - 5.0) < 0.5


def test_hierarchy_is_dropin_for_flat_core():
    h = HierarchicalPredictiveCore(belief=[0.0, 0.0])
    assert len(h.mu) == 2
    r, em = h.step([1.0, -1.0])
    assert r.free_energy >= 0.0
    h.set_preference([2.0, 2.0], precision=1.0)
    choice = h.act([Action("a", [2.0, 2.0]), Action("b", [-9.0, -9.0])])
    assert choice.action.name == "a"
    assert "upper_belief" in h.status()


# -------------------- per-dimension precision + volatility -------------------- #
def test_precision_vector_per_dim():
    core = PredictiveCore(belief=[0.0, 0.0], learning_rate=0.0)   # frozen belief
    rng = random.Random(0)
    for _ in range(60):
        core.perceive([rng.gauss(0.0, 3.0), rng.gauss(0.0, 0.05)], iterations=0)
    vec = core.update_precision_vec()
    assert vec is not None
    assert vec[0] < vec[1]              # the noisy dim is trusted less


def test_volatility_lowers_precision():
    stable = PredictiveCore(belief=[0.0], learning_rate=0.0)
    volatile = PredictiveCore(belief=[0.0], learning_rate=0.0)
    rng = random.Random(1)
    for t in range(80):
        stable.perceive([rng.gauss(0.0, 1.0)], iterations=0)
        # volatile: the error variance itself keeps drifting
        sigma = 0.2 if (t // 10) % 2 == 0 else 4.0
        volatile.perceive([rng.gauss(0.0, sigma)], iterations=0)
    assert volatile.volatility > stable.volatility


def test_precision_vec_default_off_keeps_old_numbers():
    core = PredictiveCore(belief=[0.0], prior=[0.0], sensory_precision=1.0,
                          prior_precision=1.0)
    assert core.precision_vec is None
    assert core.free_energy([2.0]) == 0.5 * 1.0 * 4.0            # pinned value intact


# -------------------- deep affect -------------------- #
def test_mood_tracks_slow_valence():
    good = PredictiveCore(belief=[0.0], sensory_precision=2.0, learning_rate=0.15)
    for _ in range(30):
        good.perceive([5.0], iterations=1)           # error keeps falling — F falls
    assert good.emotion().mood > 0.0

    bad = PredictiveCore(belief=[0.0], learning_rate=0.0)
    for t in range(30):
        bad.perceive([float(t * t)], iterations=0)   # error keeps growing — F rises
    assert bad.emotion().mood < 0.0


def test_anxiety_and_relief_from_efe():
    core = PredictiveCore(belief=[0.0], preference=[0.0], preference_precision=2.0)
    core.act([Action("far", [10.0])])                # only a bad future available
    anxious = core.emotion().anxiety
    assert anxious > 0.5
    core.act([Action("home", [0.0])])                # a good future appears
    em = core.emotion()
    assert em.anxiety < anxious
    assert em.relief > 0.0                           # anxiety falling is felt as relief


def test_policy_posterior_and_sampling():
    core = PredictiveCore(belief=[0.0], preference=[0.0])
    choices = [core.expected_free_energy(Action("good", [0.5])),
               core.expected_free_energy(Action("bad", [8.0]))]
    probs = core.policy_posterior(choices, gamma=4.0)
    assert sum(probs) == 1.0 or abs(sum(probs) - 1.0) < 1e-9
    # default act() is the exact argmin (backward compatible)
    best = core.act([Action("good", [0.5]), Action("bad", [8.0])])
    assert best.action.name == "good"


def test_last_free_energy_exposed():
    core = PredictiveCore(belief=[0.0])
    assert core.last_free_energy() is None
    core.perceive([1.0])
    assert core.last_free_energy() is not None
