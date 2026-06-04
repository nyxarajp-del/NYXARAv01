"""Tests for nyxara.memory.graph."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import MemoryError_
from nyxara.memory.graph import Entity, GraphQuery, KnowledgeGraph, Triple
from nyxara.memory.provenance import Provenance, SourceType


def _world():
    g = KnowledgeGraph()
    g.register_inverse("owns", "owned_by")
    g.register_transitive("part_of")
    g.register_symmetric("related_to")
    g.add_entity("Jaypal Khoja", etype="person", aliases=["JP", "boss"])
    g.add_triple("Jaypal Khoja", "owns", "NYXARA", confidence=1.0,
                 provenance=Provenance(SourceType.OWNER, confidence=1.0))
    g.add_triple("NYXARA", "is_a", "AI", confidence=1.0)
    g.add_triple("NYXARA", "defends", "Jaypal Khoja", confidence=1.0)
    g.add_triple("kernel", "part_of", "NYXARA", confidence=0.95)
    g.add_triple("invariants", "part_of", "kernel", confidence=0.95)
    g.add_triple("chess", "related_to", "strategy", confidence=0.8)
    g.add_triple("strategy", "related_to", "warfare", confidence=0.8)
    g.add_triple("warfare", "related_to", "defense", confidence=0.8)
    return g


# -------------------- entities -------------------- #
def test_entity_normalisation_and_aliases():
    e = Entity(name="Jaypal Khoja", aliases=["JP", "The Owner"])
    assert e.eid == "jaypal_khoja"
    assert "jp" in e.aliases
    assert "the_owner" in e.aliases


def test_resolve_by_name_and_alias():
    g = _world()
    eid = g.resolve("NYXARA")
    assert eid == "nyxara"
    assert g.resolve("JP") == g.resolve("Jaypal Khoja")
    assert g.resolve("boss") == "jaypal_khoja"
    assert g.resolve("unknown thing") is None


def test_get_entity_raises_for_unknown():
    g = _world()
    with pytest.raises(MemoryError_):
        g.get_entity("nonexistent")


def test_add_triple_autocreates_entities():
    g = KnowledgeGraph()
    g.add_triple("cat", "is_a", "animal")
    assert g.resolve("cat") and g.resolve("animal")
    assert len(g) == 1


# -------------------- pattern matching -------------------- #
def test_match_by_subject_predicate():
    g = _world()
    res = g.match(GraphQuery(subject="JP", predicate="owns"))
    assert any(t.object == "nyxara" for t in res)


def test_match_wildcards():
    g = _world()
    all_partof = g.match(GraphQuery(predicate="part_of"))
    assert len(all_partof) == 2


def test_match_inverse_predicate():
    g = _world()
    # who owns NYXARA? query forward predicate with object set
    res = g.match(GraphQuery(predicate="owns", object="NYXARA"))
    assert any(t.subject == "jaypal_khoja" for t in res)


def test_match_min_confidence_filter():
    g = _world()
    g.add_triple("rumor", "is_a", "fact", confidence=0.1)
    res = g.match(GraphQuery(predicate="is_a", min_confidence=0.5))
    assert all(t.confidence >= 0.5 or t.trust() >= 0.5 for t in res)
    assert not any(t.subject == "rumor" for t in res)


def test_triples_of_directions():
    g = _world()
    out = g.triples_of("NYXARA", direction="out")
    assert any(t.predicate == "is_a" for t in out)
    inc = g.triples_of("NYXARA", direction="in")
    assert any(t.predicate == "owns" for t in inc)


def test_neighbors():
    g = _world()
    ns = g.neighbors("NYXARA")
    assert "jaypal_khoja" in ns  # incoming owns
    assert "ai" in ns            # outgoing is_a


# -------------------- reasoning -------------------- #
def test_transitive_closure():
    g = _world()
    parts = g.transitive_closure("invariants", "part_of")
    assert set(parts) == {"kernel", "nyxara"}


def test_transitive_closure_unknown_start():
    g = _world()
    assert g.transitive_closure("nope", "part_of") == []


# -------------------- path finding -------------------- #
def test_shortest_path_found():
    g = _world()
    p = g.shortest_path("chess", "defense")
    assert p is not None
    assert p.nodes[0] == "chess" and p.nodes[-1] == "defense"
    assert len(p.edges) == 3


def test_path_score_is_weakest_link():
    g = _world()
    p = g.shortest_path("chess", "defense")
    assert p.score == pytest.approx(min(e.confidence for e in p.edges), abs=0.2)


def test_no_path_returns_none():
    g = _world()
    g.add_entity("island")
    assert g.shortest_path("chess", "island") is None


def test_paths_respects_max_hops():
    g = _world()
    assert g.paths("chess", "defense", max_hops=1) == []


# -------------------- natural language -------------------- #
def test_nl_who_owns():
    g = _world()
    a = g.ask("Who is the owner of NYXARA?")
    assert "Jaypal" in a.text
    assert a.entities == ["jaypal_khoja"]


def test_nl_what_does_x_own():
    g = _world()
    a = g.ask("What does JP own?")
    assert "NYXARA" in a.text


def test_nl_how_related_path():
    g = _world()
    a = g.ask("How is chess related to defense?")
    assert a.paths
    assert "defense" in a.paths[0].nodes


def test_nl_who_owns_short():
    g = _world()
    a = g.ask("Who owns NYXARA?")
    assert "Jaypal" in a.text


def test_nl_describe_fallback():
    g = _world()
    a = g.ask("Tell me about NYXARA")
    assert a.interpretation == "describe(nyxara)"
    assert a.triples


def test_nl_unparsed():
    g = _world()
    a = g.ask("What is the meaning of life?")
    assert a.interpretation == "unparsed"


def test_nl_confidence_reported():
    g = _world()
    a = g.ask("Who owns NYXARA?")
    assert a.confidence > 0.9  # owner-sourced


# -------------------- persistence / stats -------------------- #
def test_to_dict_and_stats():
    g = _world()
    d = g.to_dict()
    assert len(d["entities"]) >= 6
    assert len(d["triples"]) == 8
    s = g.stats()
    assert s["triples"] == 8
    assert s["predicates"]["part_of"] == 2


def test_save(tmp_path):
    g = _world()
    p = g.save(str(tmp_path / "kg.json"))
    import os
    assert os.path.exists(p)
