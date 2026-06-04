"""Tests for nyxara.memory.retrieval."""

from __future__ import annotations

import pytest

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.retrieval import (
    AssociativeRetriever,
    DejaVuSignal,
    EmotionTag,
    FusionWeights,
    RetrievalContext,
)
from nyxara.memory.store import MemoryStore, MemoryType


# -------------------- EmotionTag -------------------- #
def test_emotion_congruence_identical_is_one():
    e = EmotionTag(0.5, 0.5)
    assert e.congruence(e) == pytest.approx(1.0)


def test_emotion_congruence_decreases_with_distance():
    base = EmotionTag(0.8, 0.8)
    near = EmotionTag(0.7, 0.7)
    far = EmotionTag(-0.8, 0.1)
    assert base.congruence(near) > base.congruence(far)


def test_emotion_clamped_on_tag():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    rec = store.remember("x")
    tag = r.tag_emotion(rec.mem_id, valence=5.0, arousal=-3.0)
    assert tag.valence == 1.0
    assert tag.arousal == 0.0


def test_emotion_persisted_in_metadata():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    rec = store.remember("x")
    r.tag_emotion(rec.mem_id, 0.5, 0.5)
    # a fresh retriever reads emotion back from metadata
    r2 = AssociativeRetriever(store)
    e = r2.emotion_of(rec.mem_id)
    assert e.valence == 0.5 and e.arousal == 0.5


def test_emotion_of_unknown_is_neutral():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    rec = store.remember("x")
    assert r.emotion_of(rec.mem_id) == EmotionTag(0.0, 0.0)


# -------------------- mood-congruent recall -------------------- #
def _emotional_store():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    threat = store.remember("intruder attacked at night", mem_type=MemoryType.EPISODIC,
                            tags=["threat"])
    calm = store.remember("peaceful evening", mem_type=MemoryType.EPISODIC, tags=["calm"])
    happy = store.remember("owner praised me", mem_type=MemoryType.EPISODIC, tags=["owner"])
    r.tag_emotion(threat.mem_id, -0.8, 0.9)
    r.tag_emotion(calm.mem_id, 0.6, 0.1)
    r.tag_emotion(happy.mem_id, 0.9, 0.5)
    return store, r, threat, calm, happy


def test_mood_congruent_anxious_surfaces_threat():
    store, r, threat, calm, happy = _emotional_store()
    out = r.mood_congruent(EmotionTag(-0.7, 0.85), k=1)
    assert out[0].record.mem_id == threat.mem_id


def test_mood_congruent_happy_surfaces_praise():
    store, r, threat, calm, happy = _emotional_store()
    out = r.mood_congruent(EmotionTag(0.85, 0.5), k=1)
    assert out[0].record.mem_id == happy.mem_id


# -------------------- fused retrieval -------------------- #
def test_fusion_weights_total():
    w = FusionWeights()
    assert w.total() == pytest.approx(w.semantic + w.context + w.emotion + w.temporal
                                      + w.graph + w.goal)


def test_retrieve_combines_signals():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    a = store.remember("defensive warfare strategy", mem_type=MemoryType.SEMANTIC,
                       tags=["security", "war"], importance=0.6)
    b = store.remember("chocolate cake recipe", mem_type=MemoryType.SEMANTIC,
                       tags=["food"], importance=0.6)
    ctx = RetrievalContext(query="strategy for defense", tags=frozenset({"security"}),
                           goals={"security": 1.0})
    res = r.retrieve(ctx, k=2)
    assert res[0].record.mem_id == a.mem_id
    assert "semantic" in res[0].signals


def test_retrieve_graph_proximity_pulls_linked():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    chess = store.remember("chess theory", mem_type=MemoryType.SEMANTIC, tags=["chess"])
    war = store.remember("warfare tactics", mem_type=MemoryType.SEMANTIC, tags=["war"])
    unrelated = store.remember("gardening tips", mem_type=MemoryType.SEMANTIC, tags=["garden"])
    store.link(chess.mem_id, war.mem_id, "related")
    # query unrelated to war, but we were just attending to chess -> war gets graph pull
    ctx = RetrievalContext(query="zzz", recent_ids=[chess.mem_id])
    res = r.retrieve(ctx, k=3)
    ids = [x.record.mem_id for x in res]
    war_res = next(x for x in res if x.record.mem_id == war.mem_id)
    assert war_res.signals["graph"] > 0.0


def test_retrieve_min_trust_filters():
    import time
    store = MemoryStore()
    r = AssociativeRetriever(store)
    store.remember("solid fact", provenance=Provenance(SourceType.OWNER, confidence=1.0),
                   tags=["x"])
    store.remember("flimsy claim",
                   provenance=Provenance(SourceType.WEB, confidence=0.9, half_life_days=0.001),
                   tags=["x"])
    ctx = RetrievalContext(query="fact claim", tags=frozenset({"x"}),
                           now=time.time() + 100 * 86400)
    res = r.retrieve(ctx, k=5, min_trust=0.5)
    assert all(x.signals["trust"] >= 0.5 for x in res)


def test_retrieve_type_filter():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    store.remember("a semantic note about rivers", mem_type=MemoryType.SEMANTIC, tags=["river"])
    store.remember("an episodic note about rivers", mem_type=MemoryType.EPISODIC, tags=["river"])
    res = r.retrieve(RetrievalContext(query="rivers"), mem_type=MemoryType.SEMANTIC)
    assert all(x.record.mem_type is MemoryType.SEMANTIC for x in res)


# -------------------- familiarity / recollection / déjà-vu -------------------- #
def _scene_store():
    store = MemoryStore()
    r = AssociativeRetriever(store)
    for txt in ["walking through an old european town square at dusk",
                "strolling across a historic plaza in the evening",
                "wandering an ancient cobblestone marketplace at twilight"]:
        store.remember(txt, mem_type=MemoryType.EPISODIC, tags=["travel"])
    return store, r


def test_recollection_high_for_exact_match():
    store, r = _scene_store()
    rec = r.recollection(RetrievalContext(
        query="walking through an old european town square at dusk"))
    assert rec > 0.9


def test_familiarity_present_for_gist_match():
    store, r = _scene_store()
    fam = r.familiarity(RetrievalContext(query="ambling through an antique plaza at nightfall",
                                         tags=frozenset({"travel"})))
    assert fam > 0.0


def test_deja_vu_detected_for_gist_without_specific():
    store, r = _scene_store()
    # lower thresholds so the lexical embedder reliably triggers the regime
    r.familiarity_threshold = 0.15
    r.recollection_threshold = 0.9
    dv = r.deja_vu(RetrievalContext(query="ambling through an antique town plaza come nightfall",
                                    tags=frozenset({"travel"})))
    assert isinstance(dv, DejaVuSignal)
    assert dv.is_deja_vu is True
    assert dv.familiarity > dv.recollection


def test_exact_query_is_recollection_not_deja_vu():
    store, r = _scene_store()
    dv = r.deja_vu(RetrievalContext(
        query="walking through an old european town square at dusk"))
    assert dv.recollection >= r.recollection_threshold
    assert dv.is_deja_vu is False
    assert dv.note == "clear recollection"


def test_deja_vu_novel_query_is_not_deja_vu():
    store, r = _scene_store()
    dv = r.deja_vu(RetrievalContext(query="quantum chromodynamics lattice gauge theory"))
    assert dv.is_deja_vu is False


def test_empty_query_signals_zero():
    store, r = _scene_store()
    ctx = RetrievalContext(query="")
    assert r.familiarity(ctx) == 0.0
    assert r.recollection(ctx) == 0.0
    assert r.deja_vu(ctx).is_deja_vu is False
