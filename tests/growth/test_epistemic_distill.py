"""Tests for Autonomous Epistemic Distillation — growth/epistemic_distill.py (Part B).

A successful teacher turn becomes durable local knowledge: a queryable KnowledgeGraph triple + a
Hyper-Dimensional Vector Rule, honestly provenance-tagged, and only baked when it verifies. All
network-free.
"""
from __future__ import annotations

from nyxara.cognition.hyper_dimensional_vectors import LatentSpaceMap
from nyxara.growth.epistemic_distill import EpistemicDistiller
from nyxara.growth.verified_distill import GroundedVerifier
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.memory.graph import GraphQuery, KnowledgeGraph
from nyxara.memory.provenance import SourceType


def _distiller(**over):
    s = NyxaraSettings.for_profile(Profile.TEST)
    for k, v in over.items():
        setattr(s.epistemic_distill, k, v)
    return EpistemicDistiller(
        knowledge_graph=KnowledgeGraph(),
        latent_map=LatentSpaceMap(dim=2048, seed=1),
        verifier=GroundedVerifier(),
        settings=s,
        graph_path=None,   # no disk write in tests
    )


def test_proven_turn_is_distilled_to_graph_and_hd_rule():
    d = _distiller()
    res = d.distill_turn("what is 6*7?", "42", reasoning="6 groups of 7 = 42", confidence=0.9)
    assert res.stored and res.verdict == "proven" and res.confidence == 1.0
    assert res.triples == 1 and res.hv_key is not None
    assert d.count() == 1

    # queryable in the KnowledgeGraph
    matches = d.knowledge_graph.match(GraphQuery(subject="what is 6*7?"))
    assert any(m.predicate == "has_verified_answer" for m in matches)

    # recallable as an HD-vector rule (associative recall by the condition role)
    hits = d.latent_map.nearest({"condition": "what is 6*7?"}, k=1)
    assert hits and hits[0][0] == res.hv_key


def test_refuted_answer_is_not_baked():
    d = _distiller()
    res = d.distill_turn("what is 6*7?", "41", confidence=0.9)
    assert res.stored is False and res.verdict == "refuted"
    assert d.count() == 0 and len(d.knowledge_graph) == 0 and len(d.latent_map) == 0


def test_provenance_tagged_teacher_derived_not_native():
    d = _distiller()
    d.distill_turn("Who wrote Hamlet?", "William Shakespeare", confidence=0.9)
    triples = list(d.knowledge_graph.match(GraphQuery(predicate="has_verified_answer")))
    assert triples
    prov = triples[0].provenance
    # honest: recorded as a probabilistic-model inference, NEVER as her own observation/owner truth
    assert prov.source is SourceType.LLM_INFERENCE
    assert "teacher" in prov.tags


def test_low_confidence_open_domain_is_skipped():
    d = _distiller(min_confidence=0.6)
    res = d.distill_turn("guess?", "maybe, i'm not sure", confidence=0.9)
    assert res.stored is False and res.verdict == "graded"


def test_duplicate_prompt_skipped():
    d = _distiller()
    assert d.distill_turn("what is 2+2?", "4", confidence=0.9).stored is True
    assert d.distill_turn("what is 2+2?", "4", confidence=0.9).stored is False


def test_disabled_is_noop():
    d = _distiller(enabled=False)
    assert d.distill_turn("what is 6*7?", "42").stored is False
    assert d.count() == 0
