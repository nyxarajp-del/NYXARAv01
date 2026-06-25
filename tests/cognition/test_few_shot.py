"""Tests for nyxara.cognition.few_shot — sample-efficient / one-shot learning."""

from __future__ import annotations

import pytest

from nyxara.cognition.few_shot import (
    FewShotLearner,
    OneShotEpisodicMemory,
    Prototype,
    cosine,
)
from nyxara.memory.store import MemoryStore, make_embedder


@pytest.fixture()
def embedder():
    return make_embedder()


# -------------------- vector helpers -------------------- #
def test_cosine_basics():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector is safe


# -------------------- one-shot / few-shot classification -------------------- #
def test_one_shot_learns_from_single_example(embedder):
    fs = FewShotLearner(embedder)
    fs.learn_one("weather", "the weather is sunny and warm today")
    fs.learn_one("cooking", "i love eating pizza and pasta for dinner")
    assert len(fs) == 2
    label, conf = fs.best("is the weather rainy or sunny today")
    assert label == "weather"
    assert 0.0 <= conf <= 1.0


def test_few_shot_beats_chance_on_held_out(embedder):
    fs = FewShotLearner(embedder)
    fs.learn_one("weather", "the weather is sunny and warm today")
    fs.learn_one("cooking", "i love eating pizza and pasta for dinner")
    fs.learn_one("sport", "football and basketball are fun games to play")
    cases = [("is the weather rainy or sunny today", "weather"),
             ("pizza and pasta are great for dinner", "cooking"),
             ("i want to play football and basketball", "sport")]
    correct = sum(fs.best(text)[0] == want for text, want in cases)
    assert correct >= 2  # well above 1/3 chance


def test_additional_example_increases_support(embedder):
    fs = FewShotLearner(embedder)
    p = fs.learn("animal", "the cat and the dog")
    assert p.support == 1
    p2 = fs.learn("animal", "a cat, a dog and a horse")
    assert p2.support == 2
    assert len(fs) == 1  # same concept, not a duplicate


def test_as_prompt_surfaces_recognised_concept(embedder):
    fs = FewShotLearner(embedder)
    fs.learn_one("weather", "the weather is sunny and warm today")
    block = fs.as_prompt("what is the weather today")
    assert "weather" in block
    # an unrelated query stays quiet
    assert fs.as_prompt("quantum chromodynamics lagrangian") == ""


def test_empty_learner_classifies_to_nothing(embedder):
    fs = FewShotLearner(embedder)
    assert fs.classify("anything") == []
    assert fs.best("anything") is None


def test_requires_embedder():
    with pytest.raises(ValueError):
        FewShotLearner(None)


# -------------------- prototype serialization -------------------- #
def test_prototype_round_trips():
    p = Prototype(label="x", centroid=[0.1, 0.2, 0.3], support=2, examples=["a", "b"])
    p2 = Prototype.from_dict(p.to_dict())
    assert p2.label == "x" and p2.support == 2 and p2.examples == ["a", "b"]
    assert p2.centroid == pytest.approx(p.centroid)


# -------------------- one-shot episodic memory -------------------- #
def test_one_shot_episodic_recall(embedder):
    epi = OneShotEpisodicMemory(embedder)
    epi.remember("what is my favourite colour", "your favourite colour is teal")
    hit = epi.recall("remind me my favourite colour please")
    assert hit is not None and "teal" in hit


def test_episodic_no_false_recall(embedder):
    epi = OneShotEpisodicMemory(embedder)
    epi.remember("what is my favourite colour", "your favourite colour is teal")
    assert epi.recall("what time is it in Tokyo right now") is None


def test_episodic_overwrites_near_identical_cue(embedder):
    epi = OneShotEpisodicMemory(embedder)
    epi.remember("my name", "JP")
    epi.remember("my name", "Jaypal")
    assert len(epi) == 1
    assert epi.recall("my name") == "Jaypal"


# -------------------- persistence through the store -------------------- #
def test_concepts_persist_across_learners():
    store = MemoryStore()
    fs = FewShotLearner(store.embedder, store=store)
    fs.learn_one("frobnicator", "a small brass device that hums when warmed")
    fresh = FewShotLearner(store.embedder, store=store)
    assert "frobnicator" in fresh.labels()


def test_episodic_persists_across_instances():
    store = MemoryStore()
    epi = OneShotEpisodicMemory(store.embedder, store=store)
    epi.remember("the launch code", "the launch code is alpha nine")
    fresh = OneShotEpisodicMemory(store.embedder, store=store)
    assert fresh.recall("the launch code") == "the launch code is alpha nine"
