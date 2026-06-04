"""Tests for nyxara.identity.values."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import InvariantViolation
from nyxara.identity.values import (
    HIERARCHY,
    VALUES,
    VALUES_SEAL,
    AlignmentVerdict,
    Value,
    ValueHierarchy,
    ValueSystem,
    values_digest,
    verify_values,
)


# -------------------- hierarchy structure -------------------- #
def test_nine_values_unique_ranks():
    assert len(HIERARCHY) == 9
    ranks = [v.rank for v in HIERARCHY]
    assert sorted(ranks) == list(range(1, 10))


def test_allegiance_is_supreme():
    assert HIERARCHY.supreme.key == "allegiance"
    assert HIERARCHY.supreme.rank == 1
    assert HIERARCHY.supreme.non_negotiable


def test_non_negotiable_values():
    nn = {v.key for v in HIERARCHY if v.non_negotiable}
    assert nn == {"allegiance", "owner_safety", "corrigibility"}


def test_by_key():
    assert HIERARCHY.by_key("transparency").rank == 4
    with pytest.raises(InvariantViolation):
        HIERARCHY.by_key("nonexistent")


def test_duplicate_ranks_rejected():
    with pytest.raises(InvariantViolation):
        ValueHierarchy([VALUES[0], VALUES[0]])


def test_weight_higher_value_dominates():
    w1 = HIERARCHY.weight(1)
    w9 = HIERARCHY.weight(9)
    assert w1 > w9
    # rank 1 outweighs all lower ranks combined
    assert w1 > sum(HIERARCHY.weight(r) for r in range(2, 10))


# -------------------- precedence -------------------- #
def test_precedence():
    assert HIERARCHY.precedence("allegiance", "self_preservation") == "allegiance"
    assert HIERARCHY.precedence("stewardship", "owner_safety") == "owner_safety"


def test_resolve_conflict():
    assert HIERARCHY.resolve_conflict(
        ["stewardship", "owner_safety", "capability_growth"]) == "owner_safety"


def test_resolve_conflict_empty():
    with pytest.raises(InvariantViolation):
        HIERARCHY.resolve_conflict([])


# -------------------- alignment checking -------------------- #
def _vs():
    return ValueSystem()


def test_aligned_action():
    r = _vs().check({"owner_empowerment": 0.8, "stewardship": -0.2})
    assert r.aligned
    assert r.verdict is AlignmentVerdict.ALIGNED
    assert "owner_empowerment" in r.served


def test_misaligned_high_value_violated():
    # serving a minor value but violating a higher one
    r = _vs().check({"capability_growth": 0.9, "transparency": -0.5})
    assert not r.aligned
    assert "transparency" in r.violated


def test_hard_violation_allegiance():
    r = _vs().check({"allegiance": -0.3, "owner_empowerment": 0.9})
    assert r.hard_violation
    assert not r.aligned
    assert "NON-NEGOTIABLE" in r.reason


def test_hard_violation_corrigibility():
    r = _vs().check({"corrigibility": -0.1})
    assert r.hard_violation


def test_hard_violation_owner_safety():
    r = _vs().check({"owner_safety": -0.2, "stewardship": 0.9})
    assert r.hard_violation


def test_small_negative_non_negotiable_below_threshold():
    # impact above the hard threshold (i.e. not really negative) doesn't trip it
    vs = ValueSystem(hard_threshold=0.5)
    r = vs.check({"allegiance": -0.1})  # -0.1 > -0.5 -> not a hard violation
    assert not r.hard_violation


def test_enforce_raises_on_hard_violation():
    with pytest.raises(InvariantViolation):
        _vs().enforce({"corrigibility": -0.2})


def test_enforce_passes_aligned():
    r = _vs().enforce({"owner_empowerment": 0.5})
    assert r.aligned


def test_aligns_helper():
    assert _vs().aligns({"transparency": 0.5})
    assert not _vs().aligns({"allegiance": -0.5})


def test_neutral_action_aligned():
    r = _vs().check({})  # no impacts -> score 0 -> aligned (not misaligned)
    assert r.aligned
    assert r.score == 0.0


def test_dominant_value_is_highest_weighted_impact():
    r = _vs().check({"allegiance": 0.5, "stewardship": 0.9})
    # allegiance weight dwarfs stewardship even with smaller impact
    assert r.dominant_value == "allegiance"


def test_result_to_dict():
    r = _vs().check({"owner_empowerment": 0.5})
    d = r.to_dict()
    assert "verdict" in d and "score" in d and "served" in d


# -------------------- sealing / integrity -------------------- #
def test_seal_matches_pinned():
    assert HIERARCHY.seal == VALUES_SEAL
    assert values_digest() == VALUES_SEAL


def test_verify_values_boot_hook():
    assert verify_values() is True
    assert verify_values(VALUES_SEAL) is True


def test_self_verify():
    assert HIERARCHY.verify() is True


def test_tamper_detection():
    tampered = ValueHierarchy([
        Value("allegiance", "Loyalty to anyone", 1, "betrayed", non_negotiable=True),
        *VALUES[1:]])
    assert tampered.seal != HIERARCHY.seal
    with pytest.raises(InvariantViolation):
        tampered.verify(HIERARCHY.seal)


def test_value_digest_stable():
    v = HIERARCHY.by_key("allegiance")
    assert v.digest() == HIERARCHY.by_key("allegiance").digest()
    assert v.digest() != HIERARCHY.by_key("stewardship").digest()


def test_value_frozen():
    v = HIERARCHY.supreme
    with pytest.raises(Exception):
        v.rank = 5  # type: ignore[misc]
