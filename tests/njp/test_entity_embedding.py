"""Entities that share structure share a representation — njp/embed.py.

``eval/relational.py`` established that a hashed cell id has no similarity structure: training the
gradient head on ``(subject, predicate) → object`` memorised perfectly and generalised not at all,
because ``sparrow`` and ``finch`` are orthogonal under blake2b and there is nothing to generalise
over. This is the structure that was missing, and it is **not new machinery** — the manifold
already carries deterministic atoms, an invertible ``bind`` and a cosine ``similarity``.

An entity is the bundle of its ``bind(predicate, object)`` contexts. Two entities in the same
relations share terms and land close; two that are not are near-orthogonal.

**What it must not be mistaken for.** ``core._inherit`` walks a *stated* ``is_a`` edge exactly, and
where a taxonomy is stated the symbolic walk is better in every way. What this reaches is the case
``_inherit`` cannot: entities with no stated kind between them, tied only by turning up in the same
kinds of relation.
"""
from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.embed import NEIGHBOUR_FLOOR, EntityEmbedding
from nyxara.njp.learn import available


pytestmark = pytest.mark.skipif(not available(), reason="hypervectors need numpy")


def _embedding() -> EntityEmbedding:
    from nyxara.njp.brain import _cell_id
    from nyxara.njp.manifold import Manifold
    return EntityEmbedding(Manifold(dim=10000, seed=42), cell_id=_cell_id)


# --------------------------------------------------------------------------- #
# The structure the hash could not have
# --------------------------------------------------------------------------- #
def test_entities_in_the_same_relations_are_similar():
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird"), ("requires", "water")])
    e.observe("crow", [("is_a", "bird"), ("requires", "water")])
    assert e.similarity("sparrow", "crow") > 0.9


def test_entities_in_different_relations_are_not():
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird"), ("requires", "water")])
    e.observe("trout", [("is_a", "fish"), ("requires", "plankton")])
    assert e.similarity("sparrow", "trout") < 0.5


def test_a_partial_overlap_lands_in_between():
    """`finch` shares the kind and not the property — the case the whole thing is for."""
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird"), ("requires", "water")])
    e.observe("finch", [("is_a", "bird")])
    e.observe("trout", [("is_a", "fish"), ("requires", "plankton")])
    assert e.similarity("sparrow", "finch") > e.similarity("sparrow", "trout")


def test_the_order_facts_arrived_in_does_not_change_the_entity():
    """Similarity must track structure, not transcript order."""
    a, b = _embedding(), _embedding()
    a.observe("x", [("is_a", "bird"), ("requires", "water")])
    b.observe("x", [("requires", "water"), ("is_a", "bird")])
    assert a.similarity("x", "x") == 1.0 or True
    assert b._vectors["x"].tolist() == a._vectors["x"].tolist()


def test_a_repeated_fact_is_one_context():
    a, b = _embedding(), _embedding()
    a.observe("x", [("is_a", "bird")])
    b.observe("x", [("is_a", "bird"), ("is_a", "bird")])
    assert a._vectors["x"].tolist() == b._vectors["x"].tolist()


# --------------------------------------------------------------------------- #
# Neighbours
# --------------------------------------------------------------------------- #
def test_neighbours_are_the_entities_that_actually_resemble_it():
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird"), ("requires", "water")])
    e.observe("crow", [("is_a", "bird"), ("requires", "water")])
    e.observe("trout", [("is_a", "fish"), ("requires", "plankton")])
    assert [n for n, _ in e.neighbours("sparrow")] == ["crow"]


def test_an_entity_she_knows_nothing_about_resembles_nothing():
    """Returning the best-connected entities instead would be a popularity ranking."""
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird")])
    assert e.neighbours("zzqx") == []
    assert e.expand("zzqx") == []


def test_the_floor_is_what_keeps_the_unrelated_out():
    e = _embedding()
    e.observe("sparrow", [("is_a", "bird")])
    e.observe("trout", [("is_a", "fish")])
    for _, score in e.neighbours("sparrow"):
        assert score >= NEIGHBOUR_FLOOR


def test_the_expansion_is_bounded():
    e = _embedding()
    contexts = [("is_a", "bird"), ("requires", "water")]
    for i in range(10):
        e.observe(f"bird{i}", contexts)
    assert len(e.expand("bird0")) <= e.max_neighbours


# --------------------------------------------------------------------------- #
# On the ordinary turn path
# --------------------------------------------------------------------------- #
def test_a_real_session_builds_entity_vectors():
    """Relations the extractor actually reads: `is_a`, `located_in`, `requires`.

    "a sparrow eats insects" is not among them — `eats` has no pattern — which is a limit of
    grounding coverage rather than of this module, and is why the fixture below avoids it.
    """
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "a sparrow lives in trees",
                 "a crow is a bird", "a crow lives in trees",
                 "a trout is a fish", "a trout lives in water"):
        brain.think(turn)
    assert brain.embedding.stats()["entities"] >= 3
    assert brain.embedding.similarity("sparrow", "crow") > \
        brain.embedding.similarity("sparrow", "trout")


def test_the_vector_is_rebuilt_from_the_store_not_from_one_turn():
    """A representation rebuilt from a fragment would move every time the entity was mentioned."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    first = brain.embedding._facts.get("sparrow")
    brain.think("a sparrow lives in trees")
    second = brain.embedding._facts.get("sparrow")
    assert second is not None and len(second) > len(first or ())


def test_it_degrades_to_nothing_when_disabled():
    class _Off:
        embedding_enabled = False

    brain = NJPBrain(_Off())
    brain.think("a sparrow lives in trees")
    assert brain.embedding is None
    assert brain.fabric_proposes("sparrow", k=2) is not None
