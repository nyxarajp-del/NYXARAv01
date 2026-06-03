"""Tests for the memory layer: store/retrieval, provenance, graph, schema, self-model."""
from __future__ import annotations

from nyxara.memory.consolidation import Consolidator
from nyxara.memory.graph import KnowledgeGraph
from nyxara.memory.prospective import ProspectiveMemory
from nyxara.memory.provenance import ProvenanceLedger
from nyxara.memory.retrieval import AssociativeRetriever
from nyxara.memory.schema import SchemaInducer
from nyxara.memory.self_model import SelfModel
from nyxara.memory.store import MemoryStore, Modality


def test_store_and_search() -> None:
    store = MemoryStore()
    store.remember_episode("owner asked about quantum computing", emotion=0.3)
    store.remember_episode("owner asked about weather forecast")
    store.remember_fact(ProvenanceLedger().attach("Nyxara serves JP", "owner"))
    hits = store.search("quantum", k=1)
    assert hits and "quantum" in hits[0][1].text


def test_associative_recall_and_deja_vu() -> None:
    store = MemoryStore()
    for t in ["the owner loves chai", "chai is brewed with spices", "spices include cardamom"]:
        store.remember_episode(t)
    store.remember_fact(ProvenanceLedger().attach("chai is a spiced tea", "owner"))
    r = AssociativeRetriever(store)
    results = r.recall("chai", k=3)
    assert results and results[0].activation >= results[-1].activation
    sig = r.deja_vu("a spiced tea drink")
    assert 0.0 <= sig.familiarity <= 1.0


def test_provenance_chain() -> None:
    led = ProvenanceLedger()
    p1 = led.tag("owner", "stated")
    p2 = led.tag("web:example.com", "perceived")
    derived = led.derive("inferred", "reasoner", [p1, p2], combine="min")
    assert derived.confidence <= min(p1.confidence, p2.confidence)
    assert len(led.chain(derived.id)) == 3


def test_knowledge_graph_nl_query() -> None:
    g = KnowledgeGraph()
    g.add("JP", "own", "Nyxara")
    g.add("JP", "know", "AI")
    assert any(t.object == "Nyxara" for t in g.query("does JP own Nyxara"))
    assert any(t.subject == "JP" for t in g.query("who owns Nyxara"))


def test_schema_induction() -> None:
    store = MemoryStore()
    for _ in range(4):
        store.remember_episode("owner requested a summary report",
                               context={"task": "summary"})
    inducer = SchemaInducer(store, min_support=3)
    schemas = inducer.induce()
    assert schemas and schemas[0].support >= 3


def test_consolidation_runs() -> None:
    store = MemoryStore()
    for i in range(5):
        store.remember_episode(f"event {i}", emotion=0.5 if i % 2 else -0.2)
    report = Consolidator(store).run()
    assert report.replayed >= 1


def test_prospective_triggers() -> None:
    pm = ProspectiveMemory()
    pm.remind_in("ping owner", -1.0)            # already due
    pm.when_cue("mention chai", cue="chai")
    fired_time = pm.poll()
    assert any(i.description == "ping owner" for i in fired_time)
    fired_cue = pm.poll(cue="I'd love some chai")
    assert any(i.description == "mention chai" for i in fired_cue)


def test_self_model_contradiction() -> None:
    sm = SelfModel()
    sm.believe("sky is blue", 0.9, "perceived")
    sm.believe("not sky is blue", 0.4, "rumor")
    cs = sm.contradictions()
    assert cs
    kept = sm.resolve_contradiction(cs[0])
    assert kept.claim == "sky is blue"
    assert sm.knows("sky is blue")
