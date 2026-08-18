"""A relation pointing at a thing is not a property of it, and clustering on that says nothing.

``observe_triples`` records the *object* of every relation as an observation carrying one feature,
``is_target_of:<predicate>``, so a superordinate has somewhere to accumulate evidence. That is
right. What was wrong is that everything downstream treated the marker as a property.

Two such observations have identical feature sets, so their Jaccard similarity is 1.0 and they
collapse into one concept. Measured after 1,200 corpus pairs: the largest concept was named
``is_target_of:is_a`` and had **230 members** — "things that something is a kind of", true of
every kind she has ever met and informative about none of them. It dominated the concept set,
which then dragged compression under break-even, which then vetoed every self-improvement trial
in the slow loop.

The marker is also what made the clustering expensive. Profiled over 100 turns, ``_similarity``
was called **12.8 million** times and ``crystallise`` was 58% of the whole turn.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.concepts import ConceptGenesis, _content_features
from nyxara.njp.grounding import GroundedTriple


def _triple(subject: str, predicate: str, obj: str) -> GroundedTriple:
    return GroundedTriple(subject=subject, predicate=predicate, object=obj, confidence=0.9)


@pytest.fixture()
def genesis() -> ConceptGenesis:
    return ConceptGenesis()


# --------------------------------------------------------------------------- #
# The marker is kept, and is not a property
# --------------------------------------------------------------------------- #
def test_the_object_of_a_relation_is_still_recorded(genesis):
    """It has to exist, or a superordinate has nowhere to accumulate evidence."""
    genesis.observe_triples([_triple("dog", "is_a", "animal")])
    assert any(o.subject == "animal" for o in genesis.observations)


def test_a_role_marker_is_not_content():
    assert _content_features({"is_target_of:is_a"}) == set()
    assert _content_features({"is_a:animal", "is_target_of:is_a"}) == {"is_a:animal"}


def test_two_things_pointed_at_are_not_thereby_similar(genesis):
    """`animal` and `mineral` share only that something is a kind of each."""
    genesis.observe_triples([_triple("dog", "is_a", "animal"),
                             _triple("quartz", "is_a", "mineral")])
    animal = next(o for o in genesis.observations if o.subject == "animal")
    mineral = next(o for o in genesis.observations if o.subject == "mineral")
    assert genesis._similarity(animal, mineral) == 0.0


def test_things_with_a_real_shared_property_are_similar(genesis):
    genesis.observe_triples([_triple("dog", "is_a", "animal"),
                             _triple("cat", "is_a", "animal")])
    dog = next(o for o in genesis.observations if o.subject == "dog")
    cat = next(o for o in genesis.observations if o.subject == "cat")
    assert genesis._similarity(dog, cat) > 0.5


# --------------------------------------------------------------------------- #
# No concept may claim a role marker
# --------------------------------------------------------------------------- #
def test_a_concept_never_asserts_a_role_marker(genesis):
    for i in range(12):
        genesis.observe_triples([_triple(f"thing{i}", "known_for", f"trait{i}")])
    genesis.crystallise()
    for concept in genesis.concepts.values():
        assert all(not f.startswith("is_target_of:") for f in concept.invariants), concept.name


def test_the_frequent_itemset_path_does_not_rebuild_the_degenerate_group(genesis):
    """Fixing similarity alone left this: 122 members under `is_target_of:known_for`."""
    for i in range(20):
        genesis.observe_triples([_triple(f"thing{i}", "known_for", f"trait{i}")])
    genesis.crystallise()
    biggest = max(genesis.concepts.values(), key=lambda c: c.support, default=None)
    if biggest is not None:
        assert all(not f.startswith("is_target_of:") for f in biggest.invariants)


def test_a_real_kind_still_forms(genesis):
    for animal in ("dog", "cat", "horse", "wolf"):
        genesis.observe_triples([_triple(animal, "is_a", "animal")])
    genesis.crystallise()
    claims = {frozenset(c.invariants) for c in genesis.concepts.values()}
    assert any("is_a:animal" in c for c in claims)


# --------------------------------------------------------------------------- #
# Compression measures what there is to compress
# --------------------------------------------------------------------------- #
def test_a_node_with_nothing_but_a_role_marker_does_not_move_compression(genesis):
    """It costs one symbol raw and one modelled, so it only ever dilutes the ratio toward 1.0."""
    for animal in ("dog", "cat", "horse", "wolf"):
        genesis.observe_triples([_triple(animal, "is_a", "animal")])
    genesis.crystallise()
    before = genesis.compression()
    for i in range(30):
        genesis.observe(f"pointed_at_{i}", ["is_target_of:mentions"])
    genesis.crystallise()
    assert genesis.compression() == pytest.approx(before, rel=0.35)


# --------------------------------------------------------------------------- #
# The clustering is faster and produces the same grouping
# --------------------------------------------------------------------------- #
def test_the_index_does_not_change_what_clusters_together(genesis):
    """A faster clustering that grouped differently would silently change what a kind is."""
    for i in range(8):
        genesis.observe(f"a{i}", ["shape:round", "colour:red"])
    for i in range(8):
        genesis.observe(f"b{i}", ["shape:flat", "colour:blue"])
    clusters = genesis._cluster(0.5)
    grouped = {frozenset(o.subject for o in c) for c in clusters if len(c) > 1}
    assert {frozenset(f"a{i}" for i in range(8)),
            frozenset(f"b{i}" for i in range(8))} == grouped


def test_disjoint_observations_are_never_compared(genesis):
    """Jaccard over disjoint feature sets is exactly zero — those comparisons were waste."""
    for i in range(40):
        genesis.observe(f"x{i}", [f"unique_feature_{i}"])
    calls = {"n": 0}
    original = genesis._similarity

    def counting(a, b):
        calls["n"] += 1
        return original(a, b)

    genesis._similarity = counting          # type: ignore[assignment]
    genesis._cluster(0.5)
    assert calls["n"] == 0, "nothing shares a feature, so nothing needed scoring"


def test_a_numeric_observation_still_sees_every_centroid(genesis):
    """It can be similar without sharing a symbol, so the index must not hide it."""
    genesis.observe("a", ["kind:metal"], {"mass": 10.0})
    genesis.observe("b", ["kind:wood"], {"mass": 10.0})
    a = next(o for o in genesis.observations if o.subject == "a")
    b = next(o for o in genesis.observations if o.subject == "b")
    assert genesis._similarity(a, b) > 0.0


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_the_largest_concept_says_something_about_its_members():
    brain = NJPBrain()
    for sentence in ("The blue whale is the largest animal on Earth.",
                     "The African elephant is the largest land animal.",
                     "The cheetah is the fastest land animal.",
                     "The ostrich is the heaviest bird in the world."):
        brain.think(sentence)
    largest = brain.stats()["concepts"].get("largest")
    if largest:
        assert not str(largest.get("name", "")).startswith("is_target_of:")
