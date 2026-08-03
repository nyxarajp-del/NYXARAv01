"""Tests for nyxara.qualia — concepts with no word, and the gate that keeps them honest."""

from __future__ import annotations

import random

from nyxara.growth.dataset_causal import learn_causal_graph
from nyxara.mind.causal_world_model import CausalWorldModel
from nyxara.mind.concept_hyperspace import ConceptHyperspace, relations_from_pairs
from nyxara.qualia import BabelCodec, ConceptCertificate, LatentConcept, QualiaWorkspace

_TAXONOMY = relations_from_pairs([
    ("mammal", "animal"), ("bird", "animal"),
    ("dog", "mammal"), ("cat", "mammal"),
    ("sparrow", "bird"), ("eagle", "bird"),
    ("poodle", "dog"), ("husky", "dog"),
])


def _space() -> ConceptHyperspace:
    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    space.fit(_TAXONOMY, epochs=400, burn_in=40)
    return space


def _workspace(**kw) -> QualiaWorkspace:
    workspace = QualiaWorkspace(_space(), min_cluster=2, **kw)
    workspace.synthesise()
    return workspace


# -------------------- discovery -------------------- #
def test_regions_are_found_in_a_fitted_space():
    workspace = _workspace()
    assert workspace.concepts
    for concept in workspace.concepts.values():
        assert len(concept.members) >= 2
        assert concept.centroid


def test_thresholds_come_from_the_space_not_from_a_magic_number():
    """Hyperbolic distance has no natural scale, so any fixed radius is right for exactly one
    space and silently wrong for every other."""
    workspace = _workspace()
    assert workspace.label_threshold > 0.0
    assert any("from this space's own distances" in n
               for n in workspace.synthesise().notes)


def test_an_explicit_radius_overrides_the_quantile():
    workspace = QualiaWorkspace(_space(), min_cluster=2)
    workspace.synthesise(radius=0.001, label_threshold=99.0)
    assert workspace.label_threshold == 99.0
    assert workspace.concepts == {}         # nothing links at that radius


def test_too_few_concepts_is_safe():
    space = ConceptHyperspace(dim=8, seed=0)
    space.add("lonely")
    report = QualiaWorkspace(space, min_cluster=3).synthesise()
    assert report.clusters == 0 and report.notes


def test_a_region_cannot_name_itself():
    """Every cluster would trivially be 'named' by one of its own members, and nothing would ever
    come out alien."""
    workspace = _workspace()
    for concept in workspace.concepts.values():
        assert concept.nearest_anchor not in concept.members


def test_an_unnamed_region_is_reported_as_alien():
    workspace = _workspace()
    alien = workspace.alien_concepts()
    assert alien, "no unnamed region found in a space that has one"
    for concept in alien:
        assert concept.label is None
        assert concept.anchor_distance > workspace.label_threshold


# -------------------- the certification gate -------------------- #
def test_uncertified_concepts_never_become_usable():
    """The whole reason this layer is not a nonsense generator: found is not the same as real."""
    workspace = _workspace()
    assert workspace.concepts
    assert workspace.usable() == []


def test_predictive_certification_requires_beating_the_baseline():
    workspace = _workspace()
    key = next(iter(workspace.concepts))

    assert workspace.certify_predictive(key, scorer=lambda _m: 150.0, baseline=100.0) is None
    assert workspace.usable() == []

    certificate = workspace.certify_predictive(key, scorer=lambda _m: 40.0, baseline=100.0)
    assert certificate is not None and certificate.method == "predictive"
    assert certificate.score == 0.6
    assert len(workspace.usable()) == 1


def test_a_scorer_that_explodes_certifies_nothing():
    workspace = _workspace()
    key = next(iter(workspace.concepts))

    def _boom(_members):
        raise RuntimeError("no")

    assert workspace.certify_predictive(key, scorer=_boom, baseline=100.0) is None
    assert workspace.usable() == []


def test_causal_certification_needs_a_real_fitted_mechanism():
    rng = random.Random(0)
    rows = []
    for _ in range(60):
        dog = rng.gauss(0, 1)
        rows.append({"dog": dog, "mammal": 2.0 * dog + rng.gauss(0, 0.2)})
    model = CausalWorldModel()
    learn_causal_graph(["dog", "mammal"], rows, model=model, order=["dog", "mammal"])

    workspace = _workspace()
    key = next(iter(workspace.concepts))
    certificate = workspace.certify_causal(key, model)
    assert certificate is not None and certificate.method == "causal"
    assert certificate.score > 0.25
    assert len(workspace.usable()) == 1


def test_causal_certification_refuses_an_empty_graph():
    workspace = _workspace()
    key = next(iter(workspace.concepts))
    assert workspace.certify_causal(key, CausalWorldModel()) is None


def test_symbolic_certification_refuses_an_unproven_claim():
    class _Prover:
        def prove(self, _claim):
            class _R:
                proven = False
            return _R()

    workspace = _workspace()
    key = next(iter(workspace.concepts))
    assert workspace.certify_symbolic(key, "2 + 2 = 5", prover=_Prover()) is None


def test_certificate_validity_is_explicit():
    assert not ConceptCertificate().valid
    assert not ConceptCertificate(method="made-up", score=1.0).valid
    assert ConceptCertificate(method="causal", score=0.5).valid


def test_require_certificate_off_returns_everything():
    workspace = _workspace(require_certificate=False)
    assert len(workspace.usable()) == len(workspace.concepts)


# -------------------- the Babel codec -------------------- #
def test_the_codec_never_invents_a_name_for_an_unnamed_region():
    """A fabricated label makes an unnamed concept indistinguishable from a named one in every
    downstream use — which is exactly the distinction this package exists to keep."""
    workspace = _workspace()
    codec = BabelCodec(workspace)
    alien = workspace.alien_concepts()
    assert alien

    description = codec.decode(alien[0].key)
    assert "do not have a word for this" in description
    assert "uncertified" in description


def test_the_codec_says_so_for_an_unknown_key():
    assert BabelCodec(_workspace()).decode("nope") == "(no such concept)"


def test_describe_all_only_speaks_certified_concepts_by_default():
    workspace = _workspace()
    codec = BabelCodec(workspace)
    assert codec.describe_all() == []
    assert codec.describe_all(certified_only=False)


def test_encode_returns_a_fitted_point():
    workspace = _workspace()
    codec = BabelCodec(workspace)
    assert codec.encode("dog")
    assert codec.encode("not-a-concept") is None


# -------------------- persistence -------------------- #
def test_round_trip(tmp_path):
    workspace = _workspace()
    key = next(iter(workspace.concepts))
    workspace.certify_predictive(key, scorer=lambda _m: 10.0, baseline=100.0)
    path = tmp_path / "qualia.json"
    assert workspace.save(path)

    restored = QualiaWorkspace(workspace.space, min_cluster=2)
    assert restored.load(path)
    assert set(restored.concepts) == set(workspace.concepts)
    assert restored.concepts[key].certified


def test_a_corrupt_file_never_blocks_boot(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    workspace = QualiaWorkspace(_space())
    assert workspace.load(path) is False


def test_load_dict_skips_bad_entries():
    workspace = QualiaWorkspace(_space())
    assert workspace.load_dict({"concepts": [{"key": "good", "centroid": [0.1]},
                                             "not-a-dict"]}) is True
    assert set(workspace.concepts) == {"good"}


def test_latent_concept_alien_property():
    assert LatentConcept(key="x").alien is True
    assert LatentConcept(key="x", label="dog").alien is False
