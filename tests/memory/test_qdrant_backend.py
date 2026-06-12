"""Tests for the Qdrant managed-vector-DB backend and migration-safe loading
(memory/store.py). Qdrant runs in pure in-memory mode here — no server needed."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile, VectorBackend
from nyxara.memory.store import (MemoryStore, MemoryType, QdrantVectorIndex, VectorIndex,
                                 make_vector_index)

_HAVE_QDRANT = QdrantVectorIndex.available()
qdrant_only = pytest.mark.skipif(not _HAVE_QDRANT, reason="qdrant-client not installed")


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def test_available_returns_bool():
    assert isinstance(QdrantVectorIndex.available(), bool)


def test_make_vector_index_selects_qdrant_when_available():
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.memory.vector_backend = VectorBackend.QDRANT
    idx = make_vector_index(64, s)
    if _HAVE_QDRANT:
        assert isinstance(idx, QdrantVectorIndex)
    else:
        assert isinstance(idx, VectorIndex)   # graceful fallback on a bare machine


# --------------------------------------------------------------------------- #
# Qdrant index behaviour
# --------------------------------------------------------------------------- #
@qdrant_only
def test_qdrant_add_search_remove():
    idx = QdrantVectorIndex(3, collection="t_basic")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.add("c", [0.9, 0.1, 0.0])
    assert len(idx) == 3
    hits = idx.search([1.0, 0.0, 0.0], k=2)
    assert hits[0][0] == "a"                  # nearest is the identical vector
    assert {m for m, _ in hits} == {"a", "c"}
    idx.remove("a")
    assert len(idx) == 2
    assert "a" not in {m for m, _ in idx.search([1.0, 0.0, 0.0], k=3)}


@qdrant_only
def test_qdrant_allowed_filter():
    idx = QdrantVectorIndex(3, collection="t_filter")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.9, 0.1, 0.0])
    hits = idx.search([1.0, 0.0, 0.0], k=5, allowed={"b"})
    assert [m for m, _ in hits] == ["b"]


@qdrant_only
def test_qdrant_dim_mismatch_raises():
    idx = QdrantVectorIndex(3, collection="t_dim")
    with pytest.raises(Exception):
        idx.add("x", [1.0, 0.0])              # wrong dimension


@qdrant_only
def test_qdrant_arbitrary_string_ids():
    # non-UUID mem_ids must round-trip via the payload
    idx = QdrantVectorIndex(2, collection="t_ids")
    idx.add("memory:42/weird-id", [1.0, 0.0])
    hits = idx.search([1.0, 0.0], k=1)
    assert hits[0][0] == "memory:42/weird-id"


@qdrant_only
def test_memory_store_over_qdrant_end_to_end():
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.memory.vector_backend = VectorBackend.QDRANT
    s.memory.qdrant_collection = "t_store"
    s.memory.semantic_embeddings = False      # keep the hashing embedder (no torch needed)
    ms = MemoryStore(settings=s)
    assert isinstance(ms._index, QdrantVectorIndex)
    ms.remember("the firewall blocked an unauthorised login attempt",
                mem_type=MemoryType.SEMANTIC)
    ms.remember("the cat sat on a warm windowsill", mem_type=MemoryType.SEMANTIC)
    hits = ms.recall("intrusion blocked by firewall", k=1)
    assert hits and "firewall" in hits[0][0].text()


# --------------------------------------------------------------------------- #
# Migration safety: loading memory saved under a different-dim embedder
# --------------------------------------------------------------------------- #
class _FixedEmbedder:
    """A trivial embedder of a chosen dimension (no heavy deps)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, text: str):
        # deterministic, non-zero vector derived from the text
        v = [0.0] * self.dim
        for i, ch in enumerate(text):
            v[i % self.dim] += (ord(ch) % 13) + 1
        return v


def test_load_reembeds_on_dim_change(tmp_path):
    # save under an 8-dim embedder ...
    src = MemoryStore(embedder=_FixedEmbedder(8))
    src.remember("alpha beta gamma", mem_type=MemoryType.SEMANTIC)
    src.remember("delta epsilon", mem_type=MemoryType.SEMANTIC)
    path = str(tmp_path / "mem.json")
    src.save(path)

    # ... and load under a 16-dim embedder: must NOT crash, and must re-embed + index.
    dst = MemoryStore(embedder=_FixedEmbedder(16))
    n = dst.load(path)
    assert n == 2
    assert len(dst._index) == 2                # both records re-embedded into the new space
    for rec in dst._kv.values():
        assert len(rec.embedding) == 16        # migrated to the current dimension
    # and recall works in the new space
    assert dst.recall("alpha beta gamma", k=1)
