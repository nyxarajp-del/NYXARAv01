"""Tests for nyxara.memory.neural_embedder — the self-learned embedding space."""

from __future__ import annotations

import random

import pytest

import nyxara.memory.neural_embedder as ne
from nyxara.memory.neural_embedder import SelfLearnedEmbedder
from nyxara.memory.store import LexicalSemanticEmbedder, MemoryStore, MemoryType


def _experienced(seed: int = 0, n: int = 200) -> SelfLearnedEmbedder:
    """An embedder fed a synthetic life where 'zorble' and 'blint' are used alike."""
    emb = SelfLearnedEmbedder(dim=128)
    contexts = ["controls the plasma flow in sector",
                "regulates the coolant valve of unit",
                "measures the neutron flux near panel",
                "adjusts the containment field around bay"]
    rng = random.Random(seed)
    for _ in range(n):
        word = rng.choice(["zorble", "blint"])
        ctx = rng.choice(contexts)
        emb.learn(f"the {word} {ctx} {rng.randrange(9)}. the {word} is stable today")
    for _ in range(40):
        emb.learn_pair("the zorble controls the plasma flow",
                       "the blint regulates the coolant valve")
    return emb


def test_cold_embed_identical_to_lexical():
    emb = SelfLearnedEmbedder(dim=128)
    lex = LexicalSemanticEmbedder(dim=128)
    for text in ("the reactor core temperature is rising fast",
                 "who is my master?", ""):
        assert emb.embed(text) == lex.embed(text)
    assert emb.is_lexical is True
    assert emb.semantic_scale == 0.0
    assert emb.space_version == 0


def test_training_learns_distributional_similarity():
    emb = _experienced()
    stats = emb.train(budget_s=1.5)
    assert stats["trained"] is True
    assert stats["sgns_steps"] > 0
    assert emb.space_version == 1
    related = emb.latent_similarity("zorble device", "blint device")
    assert related is not None and related > 0.5           # learned from usage alone
    # an unrelated pairing must not ride the same similarity
    emb.learn("bananas are a sweet yellow fruit eaten at breakfast")
    emb.train(budget_s=0.3)
    unrelated = emb.latent_similarity("zorble device", "banana fruit")
    assert unrelated is None or unrelated < related


def test_train_budget_is_respected():
    import time
    emb = _experienced(n=100)
    t0 = time.monotonic()
    emb.train(budget_s=0.3)
    assert time.monotonic() - t0 < 2.0


def test_no_experience_is_a_noop():
    emb = SelfLearnedEmbedder(dim=64)
    stats = emb.train(budget_s=0.2)
    assert stats["trained"] is False
    assert emb.space_version == 0


def test_persistence_round_trip():
    emb = _experienced(n=120)
    emb.train(budget_s=0.8)
    data = emb.to_dict()
    fresh = SelfLearnedEmbedder(dim=128)
    assert fresh.load_dict(data) is True
    text = "the zorble controls the plasma flow"
    # weights are rounded to 5 decimals on disk, so embeddings match to ~1e-4
    assert fresh.embed(text) == pytest.approx(emb.embed(text), abs=1e-3)
    assert fresh.space_version == emb.space_version
    # to_dict() rounds the grade to 6 decimals, so the round-trip error is up to 5e-7. approx's
    # DEFAULT tolerance is relative (rel=1e-6), and the grade here is small — 2e-5 early in
    # training, saturating at 0.3 — so that default asks for ~1e-11 and the rounding alone
    # breaks it. This passed only on a machine fast enough to saturate the grade inside the
    # 0.8s budget; a loaded CI runner trains fewer steps, lands on an arbitrary value, and goes
    # red on serialization rounding rather than on anything real. Pin the tolerance to the
    # precision actually written to disk, as the weight assertion above already does.
    assert fresh.semantic_grade == pytest.approx(emb.semantic_grade, abs=1e-6)


def test_load_rejects_mismatched_geometry():
    emb = _experienced(n=60)
    emb.train(budget_s=0.3)
    other = SelfLearnedEmbedder(dim=64)
    assert other.load_dict(emb.to_dict()) is False


def test_pure_python_path(monkeypatch):
    monkeypatch.setattr(ne, "_HAS_NP", False)
    emb = _experienced(n=60)
    stats = emb.train(budget_s=0.5)
    assert stats["trained"] is True
    sim = emb.latent_similarity("zorble", "blint")
    assert sim is not None
    # persistence works on the stdlib path too
    fresh = SelfLearnedEmbedder(dim=128)
    assert fresh.load_dict(emb.to_dict()) is True


def test_store_stamps_and_reembeds_stale_vectors():
    store = MemoryStore(embedder=SelfLearnedEmbedder(dim=128))
    rec = store.remember("the zorble controls the plasma flow",
                         mem_type=MemoryType.SEMANTIC)
    assert rec.metadata["emb_v"] == 0
    # before any training, nothing is stale
    assert store.reembed_stale() == 0
    emb = store.embedder
    for _ in range(60):
        emb.learn("the zorble controls the plasma flow in sector nine")
        emb.learn_pair("the zorble controls the plasma flow",
                       "plasma flow is under zorble control")
    emb.train(budget_s=0.5)
    assert emb.space_version == 1
    migrated = store.reembed_stale()
    assert migrated == 1
    assert rec.metadata["emb_v"] == 1
    assert store.reembed_stale() == 0            # idempotent once migrated


def test_consolidator_trains_embedder():
    from nyxara.memory.consolidation import Consolidator
    store = MemoryStore(embedder=SelfLearnedEmbedder(dim=128))
    emb = store.embedder
    rng = random.Random(1)
    for i in range(30):
        text = f"episode {i}: the {rng.choice(['zorble', 'blint'])} adjusted the flow"
        store.remember(text, mem_type=MemoryType.EPISODIC)
        emb.learn(text)
    report = Consolidator(store).run_cycle()
    assert report.embedder_trained is True
    assert emb.space_version >= 1
