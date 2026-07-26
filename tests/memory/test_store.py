"""Tests for nyxara.memory.store."""

from __future__ import annotations

import os
import tempfile

import pytest

from nyxara.kernel.errors import MemoryError_
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.store import (
    HashingEmbedder,
    MemoryRecord,
    MemoryStore,
    MemoryType,
    VectorIndex,
)


# -------------------- embedder -------------------- #
def test_sentence_transformer_embedder_is_shared_per_model(monkeypatch):
    """One heavy sentence-transformer per (model, device) for the whole process."""
    import nyxara.memory.store as store_mod

    class _Stub:
        def __init__(self, model_name, device):
            self.model_name, self.device = model_name, device

    monkeypatch.setattr(store_mod, "SentenceTransformerEmbedder", _Stub)
    monkeypatch.setattr(store_mod, "_ST_EMBEDDER_CACHE", {})
    a = store_mod._shared_st_embedder("m1", "")
    b = store_mod._shared_st_embedder("m1", "")
    c = store_mod._shared_st_embedder("m2", "")
    assert a is b
    assert c is not a and c.model_name == "m2"


def test_embedder_deterministic_and_normalised():
    e = HashingEmbedder(dim=64)
    v1 = e.embed("the quick brown fox")
    v2 = e.embed("the quick brown fox")
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-9
    assert len(v1) == 64


def test_embedder_similar_texts_closer():
    from nyxara.memory.store import _cosine
    e = HashingEmbedder(dim=256)
    a = e.embed("network security firewall intrusion")
    b = e.embed("firewall intrusion network security")  # same words
    c = e.embed("baking a chocolate cake recipe")
    assert _cosine(a, b) > _cosine(a, c)


def test_embedder_rejects_tiny_dim():
    with pytest.raises(MemoryError_):
        HashingEmbedder(dim=4)


# -------------------- lexical-semantic embedder (dependency-free) -------------------- #
def test_lexical_semantic_deterministic_normalised_and_lexical():
    from nyxara.memory.store import LexicalSemanticEmbedder
    e = LexicalSemanticEmbedder(dim=128)
    v1, v2 = e.embed("guard the Master"), e.embed("guard the Master")
    assert v1 == v2 and len(v1) == 128
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-9
    assert e.is_lexical is True  # keeps the recall-floor calibration


def test_stemmer_collapses_inflections():
    from nyxara.memory.store import _stem
    assert _stem("running") == "run"
    assert _stem("logs") == "log"
    assert _stem("attacks") == "attack"
    assert _stem("quickly") == "quick"
    assert _stem("memories") == "memory"


def test_lexical_semantic_recalls_synonyms_better_than_hashing():
    from nyxara.memory.store import LexicalSemanticEmbedder, _cosine
    old, new = HashingEmbedder(dim=256), LexicalSemanticEmbedder(dim=256)
    q, d = "an intrusion was detected", "unauthorised breach in the system"
    # the curated relation map gives real (bounded) semantic recall the hash embedder lacks
    assert _cosine(new.embed(q), new.embed(d)) > _cosine(old.embed(q), old.embed(d))


def test_lexical_semantic_keeps_unrelated_text_low():
    from nyxara.memory.store import LexicalSemanticEmbedder, _cosine
    e = LexicalSemanticEmbedder(dim=256)
    sim = _cosine(e.embed("please help me"), e.embed("can you assist me"))
    unrel = _cosine(e.embed("the cat sat on the mat"),
                    e.embed("quantum chromodynamics lecture notes"))
    assert sim > 0.2 > unrel  # synonyms recall; unrelated stays low


def test_default_fallback_embedder_is_lexical_semantic():
    from nyxara.memory.store import LexicalSemanticEmbedder, make_embedder
    from nyxara.kernel.config import NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.memory.semantic_embeddings = False  # force the dependency-free fallback
    assert isinstance(make_embedder(s), LexicalSemanticEmbedder)


# -------------------- vector index -------------------- #
def test_vector_index_add_search_remove():
    e = HashingEmbedder(dim=64)
    idx = VectorIndex(64)
    idx.add("a", e.embed("cats and dogs"))
    idx.add("b", e.embed("quantum physics"))
    res = idx.search(e.embed("cats and dogs"), k=1)
    assert res[0][0] == "a"
    idx.remove("a")
    assert len(idx) == 1


def test_vector_index_dim_mismatch():
    idx = VectorIndex(8)
    with pytest.raises(MemoryError_):
        idx.add("x", [0.0] * 4)


def test_vector_index_empty_search():
    assert VectorIndex(8).search([0.0] * 8, k=3) == []


def test_vector_index_allowed_filter():
    e = HashingEmbedder(dim=64)
    idx = VectorIndex(64)
    idx.add("a", e.embed("apples"))
    idx.add("b", e.embed("apples"))
    res = idx.search(e.embed("apples"), k=5, allowed={"b"})
    assert {r[0] for r in res} == {"b"}


# -------------------- record -------------------- #
def test_record_activation_increases_with_access():
    rec = MemoryRecord(content="x", mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.OWNER, confidence=1.0))
    now = rec.created_at
    a0 = rec.activation(now)
    rec.touch(now)
    rec.touch(now)
    a1 = rec.activation(now)
    assert a1 > a0


def test_record_recency_decays():
    rec = MemoryRecord(content="x", mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.OWNER), half_life_days=1)
    now = rec.last_access
    assert rec.recency_factor(now) > rec.recency_factor(now + 5 * 86400)


def test_record_text_extraction():
    assert MemoryRecord("hi", MemoryType.SEMANTIC, Provenance(SourceType.OWNER)).text() == "hi"
    d = MemoryRecord({"a": 1, "b": 2}, MemoryType.SEMANTIC, Provenance(SourceType.OWNER))
    assert "a:1" in d.text()


# -------------------- store: write/read -------------------- #
def test_remember_and_get():
    ms = MemoryStore()
    rec = ms.remember("hello world", mem_type=MemoryType.SEMANTIC)
    assert ms.get(rec.mem_id) is rec
    assert len(ms) == 1
    with pytest.raises(MemoryError_):
        ms.get("missing")


def test_remember_default_provenance_is_self_reflection():
    ms = MemoryStore()
    rec = ms.remember("a thought")
    assert rec.provenance.source is SourceType.SELF_REFLECTION


def test_recall_finds_relevant():
    ms = MemoryStore()
    ms.remember("The owner is Jaypal Khoja JP", mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.OWNER, confidence=1.0),
                tags=["owner"], importance=1.0)
    ms.remember("Recipe for pancakes with syrup", mem_type=MemoryType.SEMANTIC)
    hits = ms.recall("who is the owner", k=1)
    assert hits
    assert "Jaypal" in hits[0][0].text()


def test_recall_min_trust_filter():
    ms = MemoryStore()
    ms.remember("trusted fact about cats", provenance=Provenance(SourceType.OWNER, confidence=1.0),
                tags=["cats"])
    ms.remember("dubious web claim about cats",
                provenance=Provenance(SourceType.WEB, confidence=0.9, half_life_days=0.001),
                tags=["cats"])
    import time
    # in the far future the web claim is stale; high min_trust excludes it
    hits = ms.recall("cats", min_trust=0.5, now=time.time() + 100 * 86400)
    sources = {h[0].provenance.source for h in hits}
    assert SourceType.WEB not in sources
    assert SourceType.OWNER in sources


def test_recall_type_filter():
    ms = MemoryStore()
    ms.remember("episodic event about coffee", mem_type=MemoryType.EPISODIC, tags=["coffee"])
    ms.remember("semantic fact about coffee", mem_type=MemoryType.SEMANTIC, tags=["coffee"])
    hits = ms.recall("coffee", mem_type=MemoryType.SEMANTIC)
    assert all(h[0].mem_type is MemoryType.SEMANTIC for h in hits)


def test_recall_strengthens_memory():
    ms = MemoryStore()
    rec = ms.remember("strengthen me", tags=["x"])
    before = rec.access_count
    ms.recall("strengthen me", k=1)
    assert rec.access_count > before


def test_recall_one():
    ms = MemoryStore()
    ms.remember("unique content alpha", tags=["a"])
    r = ms.recall_one("unique content alpha")
    assert r is not None and "alpha" in r.text()


# -------------------- working memory capacity -------------------- #
def test_working_memory_eviction():
    ms = MemoryStore()
    ms.working_slots = 3
    for i in range(6):
        ms.remember(f"wm {i}", mem_type=MemoryType.WORKING, importance=0.1 + 0.1 * i)
    ws = ms.working_set()
    assert len(ws) == 3
    assert ms.stats()["evicted"] == 3
    # highest-importance survive
    texts = {r.text() for r in ws}
    assert "wm 5" in texts


def test_on_evict_callback():
    evicted = []
    ms = MemoryStore(on_evict=lambda r: evicted.append(r))
    ms.working_slots = 1
    ms.remember("a", mem_type=MemoryType.WORKING, importance=0.9)
    ms.remember("b", mem_type=MemoryType.WORKING, importance=0.1)
    assert len(evicted) == 1


# -------------------- graph / spreading -------------------- #
def test_link_and_neighbors():
    ms = MemoryStore()
    a = ms.remember("a")
    b = ms.remember("b")
    ms.link(a.mem_id, b.mem_id, "related")
    nb = ms.neighbors(a.mem_id)
    assert b.mem_id in {n.mem_id for n in nb}
    # bidirectional
    assert a.mem_id in {n.mem_id for n in ms.neighbors(b.mem_id)}


def test_spreading_activation():
    ms = MemoryStore()
    a = ms.remember("chess", importance=0.9)
    b = ms.remember("strategy", importance=0.9)
    c = ms.remember("war", importance=0.9)
    ms.link(a.mem_id, b.mem_id)
    ms.link(b.mem_id, c.mem_id)
    spread = ms.spread(a.mem_id, hops=2)
    assert b.mem_id in spread
    assert c.mem_id in spread
    assert spread[b.mem_id] > spread[c.mem_id]  # closer = more activation


# -------------------- maintenance -------------------- #
def test_forget_removes_and_scrubs_links():
    ms = MemoryStore()
    a = ms.remember("a")
    b = ms.remember("b")
    ms.link(a.mem_id, b.mem_id)
    assert ms.forget(b.mem_id) is True
    assert b.mem_id not in {n.mem_id for n in ms.neighbors(a.mem_id)}
    assert ms.forget("nonexistent") is False


def test_promote_changes_type():
    ms = MemoryStore()
    rec = ms.remember("episodic thing", mem_type=MemoryType.EPISODIC)
    ms.promote(rec.mem_id, MemoryType.SEMANTIC)
    assert rec.mem_type is MemoryType.SEMANTIC


def test_consolidation_candidates():
    ms = MemoryStore()
    rec = ms.remember("frequently recalled", mem_type=MemoryType.EPISODIC, tags=["x"])
    for _ in range(4):
        ms.recall("frequently recalled")
    cands = ms.consolidation_candidates(min_access=3)
    assert rec in cands


# -------------------- persistence -------------------- #
def test_save_and_load():
    ms = MemoryStore()
    ms.remember("persistent fact", mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.OWNER, confidence=1.0), tags=["p"])
    a = ms.remember("linked-a")
    b = ms.remember("linked-b")
    ms.link(a.mem_id, b.mem_id)

    path = os.path.join(tempfile.gettempdir(), "nyx_mem_test.json")
    try:
        ms.save(path)
        ms2 = MemoryStore()
        n = ms2.load(path)
        assert n == len(ms)
        hit = ms2.recall("persistent fact", k=1)
        assert "persistent" in hit[0][0].text()
        # links preserved
        assert b.mem_id in {x.mem_id for x in ms2.neighbors(a.mem_id)}
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_stats():
    ms = MemoryStore()
    ms.remember("x", mem_type=MemoryType.SEMANTIC)
    ms.remember("y", mem_type=MemoryType.EPISODIC)
    s = ms.stats()
    assert s["total"] == 2
    assert s["by_type"]["semantic"] == 1
    assert s["by_type"]["episodic"] == 1
    assert s["indexed"] == 2
