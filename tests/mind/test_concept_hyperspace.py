"""Tests for nyxara.mind.concept_hyperspace — a fitted Poincaré space, not a hashed one."""

from __future__ import annotations

import math

import pytest

from nyxara.mind.concept_hyperspace import ConceptHyperspace, relations_from_pairs
from nyxara.mind.latent_geometry import poincare_distance

_TAXONOMY = relations_from_pairs([
    ("mammal", "animal"), ("bird", "animal"),
    ("dog", "mammal"), ("cat", "mammal"), ("wolf", "mammal"),
    ("sparrow", "bird"), ("eagle", "bird"), ("owl", "bird"),
    ("poodle", "dog"), ("husky", "dog"),
])

_SOLAR = relations_from_pairs([("planet", "sun"), ("moon", "planet"),
                               ("comet", "sun"), ("asteroid", "sun")])
_ATOM = relations_from_pairs([("electron", "nucleus"), ("spin", "electron"),
                              ("photon", "nucleus"), ("quark", "nucleus")])


def _fitted(dim: int = 16, seed: int = 0, epochs: int = 300) -> ConceptHyperspace:
    space = ConceptHyperspace(dim=dim, seed=seed, lr=0.1, negatives=6)
    space.fit(_TAXONOMY, epochs=epochs, burn_in=30)
    return space


# -------------------- the objective actually optimises -------------------- #
def test_the_loss_goes_down():
    """The whole point of this module: BallEmbedder is a fixed random projection that fits
    nothing. If the loss does not fall, nothing has been learned and the space is decoration."""
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    report = space.fit(_TAXONOMY, epochs=300, burn_in=30)
    assert report.improved
    assert report.final_loss < report.initial_loss / 2.0


def test_the_true_parent_becomes_the_nearest_concept():
    report = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6).fit(
        _TAXONOMY, epochs=300, burn_in=30)
    assert report.mean_rank <= 1.5      # 1.0 is perfect


def test_specific_concepts_sit_further_from_the_origin():
    """Hierarchy has to be *in the radius* — that is what the fit is for."""
    space = _fitted()
    assert space.concepts["animal"].radius < space.concepts["dog"].radius
    assert space.concepts["dog"].radius < space.concepts["poodle"].radius


def test_every_point_stays_inside_the_ball():
    space = _fitted()
    for concept in space.concepts.values():
        assert concept.radius < 1.0
        assert math.isfinite(poincare_distance(concept.point, [0.0] * space.dim))


def test_a_second_fit_refines_rather_than_destroys_the_first():
    """Fitting a second domain in isolation used to drag the first one's concepts around as
    negatives and quietly wreck an embedding that was already correct."""
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    space.fit(_SOLAR, epochs=300, burn_in=30, domain="solar")
    space.fit(_ATOM, epochs=300, burn_in=30, domain="atom")

    # each domain keeps its own root nearest the origin
    solar = space._domain_members("solar")
    atom = space._domain_members("atom")
    assert min(solar, key=lambda n: space.concepts[n].radius) == "sun"
    assert min(atom, key=lambda n: space.concepts[n].radius) == "nucleus"


def test_no_relations_is_safe():
    report = ConceptHyperspace(dim=8).fit([], epochs=10)
    assert report.relations == 0 and report.notes


# -------------------- queries -------------------- #
def test_nearest_returns_the_semantic_neighbours():
    space = _fitted()
    assert "dog" in [n for n, _d in space.nearest("poodle", 3)]
    assert "bird" in [n for n, _d in space.nearest("eagle", 3)]


def test_distance_is_symmetric_and_unknowns_are_infinite():
    space = _fitted()
    assert space.distance("dog", "cat") == space.distance("cat", "dog")
    assert space.distance("dog", "not-a-concept") == float("inf")


def test_entailment_follows_the_fitted_hierarchy():
    space = _fitted(epochs=400)
    assert space.entails("poodle", "dog")
    assert space.entails("dog", "mammal")
    assert not space.entails("animal", "poodle")     # the root is not a kind of a leaf
    assert not space.entails("eagle", "dog")         # different branch


def test_entailment_requires_the_parent_to_be_more_general():
    space = _fitted()
    assert not space.entails("animal", "animal")
    assert not space.entails("unknown", "dog")


# -------------------- cross-domain correspondence -------------------- #
def test_structurally_identical_domains_are_matched_without_shared_vocabulary():
    """The claim this layer exists to support: surface a shared structure between two fields that
    have no words in common. sun↔nucleus and planet↔electron must fall out of the geometry."""
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    space.fit(_SOLAR, epochs=300, burn_in=30, domain="solar")
    space.fit(_ATOM, epochs=300, burn_in=30, domain="atom")

    found = space.find_correspondences("solar", "atom", min_score=0.5)
    assert found, "no correspondence surfaced between two structurally identical domains"
    mapping = dict(found[0].pairs)
    assert mapping["sun"] == "nucleus"
    assert mapping["planet"] == "electron"
    assert found[0].score >= 0.5


def test_dissimilar_shapes_are_refused_rather_than_asserted():
    """A chain and a star share no structure. Announcing an analogy anyway is the failure mode
    this whole layer has to avoid, so the score must be a shape test, not a scale test."""
    space = ConceptHyperspace(dim=16, seed=1, lr=0.1)
    space.fit(relations_from_pairs([("a", "root1"), ("b", "a"), ("c", "b"), ("d", "c")]),
              epochs=300, burn_in=30, domain="chain")
    space.fit(relations_from_pairs([("p", "root2"), ("q", "root2"), ("r", "root2"),
                                    ("s", "root2")]), epochs=300, burn_in=30, domain="star")
    assert space.find_correspondences("chain", "star", min_score=0.6) == []


def test_a_domain_with_no_variance_correlates_with_nothing():
    space = ConceptHyperspace(dim=8, seed=0)
    space.add("only", domain="tiny")
    assert space.find_correspondences("tiny", "missing") == []


def test_correspondence_is_unverified_without_symbolic_relations():
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    space.fit(_SOLAR, epochs=300, burn_in=30, domain="solar")
    space.fit(_ATOM, epochs=300, burn_in=30, domain="atom")
    found = space.find_correspondences("solar", "atom", min_score=0.5)
    assert found and found[0].verified is False
    assert "geometric only" in found[0].describe()


# -------------------- persistence -------------------- #
def test_round_trip_preserves_the_fitted_space(tmp_path):
    space = _fitted()
    path = tmp_path / "qualia" / "hyperspace.json"
    assert space.save(path)

    restored = ConceptHyperspace(dim=16, seed=0)
    assert restored.load(path)
    assert set(restored.concepts) == set(space.concepts)
    # coordinates round-trip through JSON at 12 decimals, so distances match to well within any
    # tolerance that matters — near the boundary this precision is the reason distances survive
    assert restored.distance("poodle", "dog") == pytest.approx(
        space.distance("poodle", "dog"), abs=1e-9)


def test_a_corrupt_sidecar_never_blocks_boot(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json at all", encoding="utf-8")
    space = ConceptHyperspace(dim=8)
    assert space.load(path) is False
    assert space.concepts == {}


def test_load_dict_survives_garbage():
    space = ConceptHyperspace(dim=8)
    assert space.load_dict({"dim": 8, "concepts": [{"name": "x", "point": [0.1] * 8},
                                                   {"name": "bad", "point": [0.1] * 3},
                                                   "not-a-dict-at-all"]}) is True
    assert set(space.concepts) == {"x"}      # wrong-dimension and non-dict entries are dropped


def test_relations_from_pairs_normalises_and_drops_self_loops():
    assert relations_from_pairs([("Dog", "Mammal"), ("x", "x"), ("", "y")]) == [("dog", "mammal")]
