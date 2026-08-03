"""Tests for nyxara.growth.dataset_hyperspace — prose becomes a fitted concept geometry."""

from __future__ import annotations

import json

from nyxara.growth.dataset import ingest_dataset
from nyxara.growth.dataset_hyperspace import _normalize, build_from_dataset_store, extract_is_a
from nyxara.mind.concept_hyperspace import ConceptHyperspace

_TAXONOMY_TEXT = (
    "A poodle is a dog. A husky is a dog. Dogs are a kind of mammal. "
    "A cat is a mammal. A wolf is a mammal. Mammals are a kind of animal. "
    "An eagle is a bird. A sparrow is a bird. Birds are a kind of animal."
)


def _write(tmp_path, text, name="taxonomy.txt"):
    path = tmp_path / "d" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# -------------------- extraction -------------------- #
def test_each_is_a_cue_family_is_read():
    cases = [
        ("A poodle is a dog.", ("poodle", "dog")),
        ("Dogs are a kind of mammal.", ("dog", "mammal")),
        ("Mammals include dogs.", ("dog", "mammal")),
        ("Sparrows are birds.", ("sparrow", "bird")),
    ]
    for sentence, expected in cases:
        relations, _n, _r = extract_is_a(sentence)
        assert relations, f"nothing extracted from {sentence!r}"
        assert relations[0].as_pair() == expected, f"wrong reading of {sentence!r}"


def test_a_list_of_children_becomes_one_relation_each():
    relations, _n, _r = extract_is_a("Mammals include dogs, cats and wolves.")
    assert {r.as_pair() for r in relations} == {("dog", "mammal"), ("cat", "mammal"),
                                               ("wolf", "mammal")}


def test_overloaded_copulas_are_not_read_as_taxonomy():
    """"is a" is heavily overloaded. Inventing a relation corrupts the hierarchy for every
    concept downstream of it, so the patterns must stay fenced."""
    relations, _n, _r = extract_is_a(
        "Today is a Tuesday. The answer is a mystery. That was a surprise. "
        "Speed is a way of measuring motion.")
    assert relations == []


def test_relations_carry_a_traceable_quote():
    relations, _n, _r = extract_is_a("A poodle is a dog.", source="book.txt")
    assert relations[0].quote == "A poodle is a dog."
    assert relations[0].source == "book.txt"


def test_singular_and_plural_land_on_the_same_concept():
    """Consistency is the property that matters — splitting one concept across two points is the
    failure, not an occasional odd-looking label."""
    assert _normalize("dogs") == _normalize("dog") == "dog"
    assert _normalize("wolves") == "wolf"
    assert _normalize("berries") == "berry"
    assert _normalize("boxes") == "box"
    assert _normalize("the heavy boxes") == "heavy box"     # edge stopwords trimmed, not interior
    assert _normalize("glass") == "glass"                   # -ss is not a plural


def test_empty_text_is_safe():
    assert extract_is_a("") == ([], 0, 0)


# -------------------- the fit -------------------- #
def test_a_dataset_of_prose_becomes_a_fitted_hierarchy(tmp_path):
    _write(tmp_path, _TAXONOMY_TEXT)
    store = tmp_path / "dataset.jsonl"
    ingest_dataset(tmp_path / "d", store_path=store)

    space = ConceptHyperspace(dim=16, seed=0, lr=0.1, negatives=6)
    report = build_from_dataset_store(store, space=space, epochs=300)

    assert len(report.relations) >= 8
    assert report.fit is not None and report.fit.improved
    assert report.fit.mean_rank <= 1.5
    # the hierarchy has to end up in the radius, which is the entire point of fitting
    assert space.concepts["animal"].radius < space.concepts["dog"].radius
    assert space.concepts["dog"].radius < space.concepts["poodle"].radius


def test_repeated_claims_are_not_treated_as_more_evidence(tmp_path):
    _write(tmp_path, "A poodle is a dog. A poodle is a dog. A poodle is a dog. "
                     "A husky is a dog. Dogs are a kind of mammal.")
    store = tmp_path / "dataset.jsonl"
    ingest_dataset(tmp_path / "d", store_path=store)

    space = ConceptHyperspace(dim=8, seed=0, lr=0.1)
    build_from_dataset_store(store, space=space, epochs=50)
    assert len(space._relations) == 3          # deduplicated, not counted three times


def test_a_dataset_with_no_taxonomy_fits_nothing(tmp_path):
    _write(tmp_path, "The weather changed. Everyone went home. It rained for hours.")
    store = tmp_path / "dataset.jsonl"
    ingest_dataset(tmp_path / "d", store_path=store)

    report = build_from_dataset_store(store)
    assert report.relations == [] and report.fit is None
    assert any("no taxonomic relations" in n for n in report.notes)


def test_missing_store_is_safe(tmp_path):
    report = build_from_dataset_store(tmp_path / "absent.jsonl")
    assert report.relations == [] and report.fit is None


def test_report_serialises(tmp_path):
    _write(tmp_path, _TAXONOMY_TEXT)
    store = tmp_path / "dataset.jsonl"
    ingest_dataset(tmp_path / "d", store_path=store)
    report = build_from_dataset_store(store, epochs=30)
    blob = json.dumps(report.to_dict())
    assert "relations" in blob and "fit" in blob
