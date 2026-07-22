"""Tests for the Semantic Concept Graph (P24 · F19, organ 2)."""

import random

from nyxara.mind.concept_graph import ConceptGraph


def test_vectors_are_deterministic_and_name_stable():
    a = ConceptGraph(dim=128)
    b = ConceptGraph(dim=128)
    assert a.vector("quantum") == b.vector("quantum")
    assert a.similarity("quantum", "quantum") == 1.0
    assert abs(a.similarity("quantum", "sonnet")) < 0.6   # unrelated ≈ orthogonal


def test_associate_learns_edges():
    g = ConceptGraph()
    g.associate("quantum entanglement links distant particles")
    assert g.edge_weight("quantum", "entanglement") > 0
    assert "quantum" in [n for n, _ in g.neighbors("entanglement", k=10)]


def test_distant_pair_returns_far_concepts():
    g = ConceptGraph()
    g.associate("quantum entanglement links distant particles across space")
    g.associate("classical sonnet meter shapes the turning line")
    pair = g.distant_pair(rng=random.Random(3))
    assert pair is not None
    a, b = pair
    assert a != b
    assert abs(g.similarity(a, b)) < 0.4


def test_weak_regions_name_thin_concepts():
    g = ConceptGraph()
    g.associate("alpha beta gamma delta")
    g.add_concept("loner")
    weak = g.weak_regions(k=1)
    assert weak == ["loner"]     # zero edges — the thinnest region


def test_growth_is_bounded():
    g = ConceptGraph(dim=64, max_nodes=64)
    for i in range(300):
        g.associate(f"node{i} partner{i % 7} theme{i % 3}")
    assert len(g.concepts()) <= 64


def test_persistence_round_trip():
    g = ConceptGraph()
    g.associate("mirror lattice stores charge", extra_concepts=["idea"])
    clone = ConceptGraph.from_dict(g.to_dict())
    assert set(clone.concepts()) == set(g.concepts())
    assert clone.edge_weight("mirror", "lattice") == g.edge_weight("mirror", "lattice")
    assert clone.associations_seen == g.associations_seen
