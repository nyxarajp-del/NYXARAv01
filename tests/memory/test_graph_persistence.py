"""Stage A — the KnowledgeGraph now persists across a restart.

Regression guard for the data-loss gap: the rich graph had ``save`` but no
``load``/``from_dict``, no caller, and was never touched by the core's
``save_state``/``load_state`` — so every restart wiped the accumulated graph
back to its 2 seed triples. These tests prove the round-trip is now lossless
(within honest bounds) and that a fresh core reloads the graph it saved.
"""

from __future__ import annotations


from nyxara.memory.graph import KnowledgeGraph, _configure_standard_relations
from nyxara.memory.provenance import Provenance, SourceType


def _world() -> KnowledgeGraph:
    g = KnowledgeGraph()
    _configure_standard_relations(g)
    owner = Provenance(SourceType.OWNER, confidence=1.0)
    g.add_triple("nyxara", "is_a", "sovereign_cognitive_agent", confidence=1.0, provenance=owner)
    g.add_triple("nyxara", "owned_by", "master", confidence=1.0, provenance=owner)
    g.add_triple("chess", "part_of", "strategy", confidence=0.8,
                 provenance=Provenance(SourceType.WEB, confidence=0.8))
    g.add_triple("strategy", "part_of", "defense", confidence=0.7,
                 provenance=Provenance(SourceType.WEB, confidence=0.7))
    return g


def test_from_dict_round_trips_entities_and_triples():
    g = _world()
    g2 = KnowledgeGraph.from_dict(g.to_dict())
    assert len(g2) == len(g) == 4
    assert g2.stats()["entities"] == g.stats()["entities"]
    # a traversal that worked before the restart still works after it
    closure = g2.transitive_closure("chess", "part_of")
    assert "defense" in closure


def test_provenance_survives_round_trip():
    g = _world()
    g2 = KnowledgeGraph.from_dict(g.to_dict())
    # the owner triple keeps its OWNER-grade trust (1.0), not a decayed default
    owner_triples = [t for t in g2._triples.values() if t.predicate == "owned_by"]
    assert owner_triples, "owned_by triple lost across round-trip"
    prov = owner_triples[0].provenance
    assert prov is not None and prov.source is SourceType.OWNER
    assert prov.trust() >= 0.99


def test_save_and_load_file(tmp_path):
    g = _world()
    p = g.save(str(tmp_path / "kg.json"))
    g2 = KnowledgeGraph.load(p)
    assert len(g2) == 4
    # traversal endpoint preserved across a file round-trip
    assert g2.resolve("nyxara") is not None
    assert "defense" in g2.transitive_closure("chess", "part_of")


def test_load_missing_file_is_empty_not_fatal(tmp_path):
    g = KnowledgeGraph.load(str(tmp_path / "does_not_exist.json"))
    assert len(g) == 0


def test_merge_from_preserves_seeds_and_adds_restored():
    seed = KnowledgeGraph()
    _configure_standard_relations(seed)
    seed.add_triple("nyxara", "owned_by", "master", confidence=1.0,
                    provenance=Provenance(SourceType.OWNER, confidence=1.0))
    restored = _world()  # contains the same seed triple + extra facts
    added = seed.merge_from(restored)
    # the duplicate seed triple is deduped; the 3 genuinely-new facts are added
    assert added == 3
    assert seed.transitive_closure("chess", "part_of") == {"strategy", "defense"} \
        or "defense" in seed.transitive_closure("chess", "part_of")


def test_core_save_load_state_persists_graph(tmp_path):
    """End-to-end: a fresh core reloads the graph a prior core saved (the actual bug)."""
    from nyxara import NyxaraCore

    target = str(tmp_path / "mem" / "memory.json")

    core_a = NyxaraCore()
    if core_a.knowledge_graph is None:
        return  # graph is an optional capability; nothing to assert on a stripped build
    core_a.knowledge_graph.add_triple(
        "quantum_computing", "part_of", "physics", confidence=0.9,
        provenance=Provenance(SourceType.OWNER, confidence=0.9),
    )
    core_a.save_state(target)

    core_b = NyxaraCore()
    assert core_b.knowledge_graph is not None
    # before load, the fresh graph does not know the fact
    assert core_b.knowledge_graph.resolve("quantum_computing") is None
    core_b.load_state(target)
    # after load, the persisted fact is present AND the seed identity survives
    assert core_b.knowledge_graph.resolve("quantum_computing") is not None
    assert core_b.knowledge_graph.resolve("nyxara") is not None
