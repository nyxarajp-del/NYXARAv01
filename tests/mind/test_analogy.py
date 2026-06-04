"""Tests for nyxara.mind.analogy — structure-mapping engine."""

from __future__ import annotations

from nyxara.mind.analogy import (
    AnalogyResult,
    StructureMapper,
    analogy,
    entity,
    relation,
)
from nyxara.mind.lot import Const, Predicate

e, r = entity, relation


def _solar_atom():
    base = [
        r("attracts", e("sun"), e("planet")),
        r("revolves_around", e("planet"), e("sun")),
        r("causes", r("attracts", e("sun"), e("planet")),
          r("revolves_around", e("planet"), e("sun"))),
    ]
    target = [
        r("attracts", e("nucleus"), e("electron")),
        r("revolves_around", e("electron"), e("nucleus")),
    ]
    return base, target


# -------------------- representation -------------------- #
def test_entity_and_relation_builders():
    assert isinstance(e("x"), Const)
    rel = r("loves", e("a"), e("b"))
    assert isinstance(rel, Predicate)
    assert rel.name == "loves"


# -------------------- core mapping -------------------- #
def test_solar_system_to_atom_mapping():
    base, target = _solar_atom()
    res = analogy(base, target)
    assert res.entity_mapping == {"sun": "nucleus", "planet": "electron"}


def test_relation_matches_are_relational():
    base, target = _solar_atom()
    res = analogy(base, target)
    matched = {c.base for c in res.relation_matches}
    assert any("attracts" in m for m in matched)
    assert any("revolves_around" in m for m in matched)


def test_candidate_inference_projects_higher_order():
    base, target = _solar_atom()
    res = analogy(base, target)
    inferred = {str(p) for p in res.candidate_inferences}
    assert any("causes" in s and "nucleus" in s and "electron" in s for s in inferred)


def test_returns_analogy_result():
    base, target = _solar_atom()
    assert isinstance(analogy(base, target), AnalogyResult)


# -------------------- constraints -------------------- #
def test_one_to_one_correspondence():
    base = [r("attracts", e("sun"), e("planet"))]
    target = [r("attracts", e("nucleus"), e("electron")),
              r("attracts", e("nucleus"), e("proton"))]
    res = analogy(base, target)
    # sun maps to exactly one target entity
    targets = list(res.entity_mapping.values())
    assert len(targets) == len(set(targets))  # no two base entities share a target


def test_identicality_constraint_requires_same_predicate():
    # different predicate names do not match
    res = analogy([r("loves", e("a"), e("b"))], [r("hates", e("x"), e("y"))])
    assert res.relation_matches == []
    assert res.entity_mapping == {}


def test_arity_mismatch_no_match():
    res = analogy([r("p", e("a"), e("b"))], [r("p", e("x"))])
    assert res.relation_matches == []


def test_parallel_connectivity():
    # arguments must map consistently across relations
    base = [r("above", e("a"), e("b")), r("left_of", e("a"), e("c"))]
    target = [r("above", e("x"), e("y")), r("left_of", e("x"), e("z"))]
    res = analogy(base, target)
    # 'a' is shared in base -> must map to the shared 'x' in target
    assert res.entity_mapping["a"] == "x"
    assert res.entity_mapping["b"] == "y"
    assert res.entity_mapping["c"] == "z"


# -------------------- systematicity & scoring -------------------- #
def test_systematicity_prefers_deep_structure():
    base, target = _solar_atom()
    deep = analogy(base, target)
    flat = analogy([r("hotter", e("sun"), e("planet"))],
                   [r("hotter", e("nucleus"), e("electron"))])
    assert deep.structural_score > flat.structural_score


def test_structural_score_positive_on_match():
    base, target = _solar_atom()
    assert analogy(base, target).structural_score > 0


def test_attributes_downweighted():
    # a 1-place attribute contributes less than a 2-place relation
    attr = analogy([r("yellow", e("sun"))], [r("yellow", e("nucleus"))])
    rel = analogy([r("attracts", e("sun"), e("planet"))],
                  [r("attracts", e("nucleus"), e("electron"))])
    assert attr.structural_score < rel.structural_score


def test_higher_order_match_scores_more():
    # when the higher-order CAUSE itself matches, score reflects the subsystem
    cause = r("causes", r("a", e("p"), e("q")), r("b", e("q"), e("p")))
    base = [cause]
    target = [r("causes", r("a", e("x"), e("y")), r("b", e("y"), e("x")))]
    res = analogy(base, target)
    assert res.structural_score >= 3.0  # cause(1) + a(1) + b(1)
    assert res.systematicity > 0.0


# -------------------- edge cases -------------------- #
def test_no_overlap_empty_result():
    res = analogy([r("foo", e("a"))], [r("bar", e("x"))])
    assert res.relation_matches == []
    assert res.candidate_inferences == []


def test_no_candidate_inference_when_unmapped_entities():
    # base has a relation over an entity that never appears in a matched relation
    base = [r("attracts", e("sun"), e("planet")),
            r("contains", e("galaxy"), e("star"))]  # galaxy/star unmapped
    target = [r("attracts", e("nucleus"), e("electron"))]
    res = analogy(base, target)
    inferred = {str(p) for p in res.candidate_inferences}
    assert not any("contains" in s for s in inferred)  # can't project unmapped entities


def test_maps_helper():
    base, target = _solar_atom()
    res = analogy(base, target)
    assert res.maps("sun") == "nucleus"
    assert res.maps("unknown") is None


def test_to_dict():
    base, target = _solar_atom()
    d = analogy(base, target).to_dict()
    assert "entity_mapping" in d and "candidate_inferences" in d
    assert "systematicity" in d


def test_mapper_custom_attribute_weight():
    sm = StructureMapper(attribute_weight=0.0)
    res = sm.map([r("yellow", e("sun"))], [r("yellow", e("nucleus"))])
    assert res.structural_score == 0.0  # attributes worth nothing
