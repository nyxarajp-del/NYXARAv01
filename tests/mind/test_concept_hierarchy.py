"""Tests for nyxara.mind.concept_hierarchy."""

from __future__ import annotations

import random

from nyxara.mind.concept_hierarchy import Concept, ConceptHierarchy, EffectSignature


# -------------------- effect signature -------------------- #
def test_effect_signature_running_mean():
    sig = EffectSignature(dim=2)
    sig.update([1.0, 0.0], reward=2.0)
    sig.update([3.0, 0.0], reward=4.0)
    assert sig.n == 2
    assert sig.mean_delta == [2.0, 0.0]
    assert sig.mean_reward == 3.0
    assert sig.magnitude == 2.0


def test_effect_signature_vec_weights_reward():
    sig = EffectSignature(dim=1)
    sig.update([1.0], reward=10.0)
    assert sig.vec(reward_weight=0.5) == [1.0, 5.0]


# -------------------- clustering -------------------- #
def _train(h, moves, n=300, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        a = rng.choice(list(moves))
        x = rng.uniform(-10, 10)
        h.update(a, (x,), (x + moves[a],), reward=-abs(x + moves[a]))
    return h


def test_same_effect_actions_cluster_across_domains():
    # two domains (gridworld + thermostat) with the same underlying concepts
    h = _train(ConceptHierarchy(), {
        "left": -1.0, "right": 1.0, "stay": 0.0,
        "cool": -1.0, "heat": 1.0, "hold": 0.0,
    })
    assert h.concept_of("left") == h.concept_of("cool")     # subtract ≡ subtract
    assert h.concept_of("right") == h.concept_of("heat")    # add ≡ add
    assert h.concept_of("stay") == h.concept_of("hold")     # hold ≡ hold
    # three genuinely distinct concepts
    assert len({h.concept_of(a) for a in ("left", "right", "stay")}) == 3
    assert len(h.concepts()) == 3


def test_dissimilar_actions_do_not_cluster():
    h = _train(ConceptHierarchy(), {"up": 5.0, "down": -5.0})
    assert h.concept_of("up") != h.concept_of("down")


def test_siblings_and_support():
    h = _train(ConceptHierarchy(), {"left": -1.0, "cool": -1.0, "right": 1.0})
    assert set(h.siblings("left")) == {"cool"}
    assert h.own_support("left") > 0
    # concept support pools every member's samples
    assert h.concept_support("left") == h.own_support("left") + h.own_support("cool")


def test_nearest_actions_prefers_concept_mates():
    h = _train(ConceptHierarchy(), {"left": -1.0, "cool": -1.0, "right": 1.0})
    assert h.nearest_actions("left") == ["cool"]


def test_nearest_actions_falls_back_to_delta_similarity():
    # a brand-new action with one sample (no concept-mates yet) finds the nearest by Δstate,
    # ignoring noisy reward
    h = _train(ConceptHierarchy(), {"left": -1.0, "right": 1.0, "stay": 0.0})
    h.update("decrement", (3.0,), (2.0,), reward=-999.0)    # Δ = -1 like 'left'; absurd reward
    assert h.nearest_actions("decrement", k=1) == ["left"]


def test_unknown_action_queries_are_safe():
    h = ConceptHierarchy()
    assert h.concept_of("ghost") is None
    assert h.siblings("ghost") == []
    assert h.nearest_actions("ghost") == []
    assert h.own_support("ghost") == 0
    assert h.concept_support("ghost") == 0
    assert h.transfer_distance("ghost") == 1.0


def test_mismatched_dimension_transition_is_ignored():
    h = ConceptHierarchy()
    h.update("a", (0.0,), (1.0,), reward=0.0)
    h.update("a", (0.0, 0.0), (1.0, 1.0), reward=0.0)       # wrong dim → ignored
    assert h.own_support("a") == 1


# -------------------- reporting -------------------- #
def test_to_dict_and_report():
    h = _train(ConceptHierarchy(), {"left": -1.0, "right": 1.0})
    d = h.to_dict()
    assert d["actions"] == 2
    assert len(d["concepts"]) == 2
    rep = h.report()
    assert "ConceptHierarchy" in rep and "concept#" in rep


def test_concept_dataclass_defaults():
    c = Concept(cid=7)
    assert c.cid == 7 and c.members == [] and c.n == 0
