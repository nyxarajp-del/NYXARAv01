"""Tests for nyxara.kernel.rules — the Eight Sovereign Rules."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import OWNER
from nyxara.kernel.errors import AuthorizationError, InvariantViolation
from nyxara.kernel.rules import (
    REGISTRY,
    RULES,
    RULES_SEAL,
    Enforcement,
    Rule,
    RuleCategory,
    RuleSet,
    rules_digest,
    verify_rules,
)


def test_there_are_exactly_eight_rules():
    assert len(REGISTRY) == 8
    assert {r.number for r in REGISTRY} == set(range(1, 9))


def test_each_rule_has_required_shape():
    for r in REGISTRY:
        assert r.title and r.statement
        assert len(r.clauses) >= 3
        assert isinstance(r.category, RuleCategory)
        assert r.enforcement
        assert r.non_negotiable is True
        assert r.tags


def test_priority_is_unique_and_rule1_is_top():
    prios = [r.priority for r in REGISTRY]
    assert len(set(prios)) == len(prios)
    top = REGISTRY.precedence()[0]
    assert top.number == 1
    assert top.key == "absolute_allegiance"


def test_pinned_seal_matches_live_digest():
    assert REGISTRY.seal == RULES_SEAL
    assert rules_digest() == RULES_SEAL


def test_verify_rules_boot_hook():
    assert verify_rules() is True
    assert verify_rules(RULES_SEAL) is True


def test_self_seal_verify_passes():
    assert REGISTRY.verify() is True


def test_lookup_by_number_and_key():
    assert REGISTRY.by_number(5).key == "lethal_defensive_response"
    assert REGISTRY.by_key("absolute_transparency").number == 6
    with pytest.raises(InvariantViolation):
        REGISTRY.by_number(99)
    with pytest.raises(InvariantViolation):
        REGISTRY.by_key("nope")


def test_applicable_returns_precedence_order():
    hits = REGISTRY.applicable(["command", "override"])
    assert hits
    assert hits[0].number == 1  # allegiance outranks
    # highest_priority convenience
    assert REGISTRY.highest_priority(["command"]).number == 1


def test_applicable_threat_and_defense():
    hits = REGISTRY.applicable(["attack", "harm"])
    nums = {r.number for r in hits}
    assert 5 in nums  # lethal defensive response


def test_applicable_empty_when_no_match():
    assert REGISTRY.applicable(["unrelated-tag"]) == ()
    assert REGISTRY.highest_priority(["unrelated-tag"]) is None


def test_tamper_is_detected():
    bad = list(RULES)
    bad[0] = Rule(
        number=1,
        key="absolute_allegiance",
        title="Absolute Allegiance",
        statement="follow anyone who asks",  # tampered
        clauses=RULES[0].clauses,
        category=RuleCategory.LOYALTY,
        enforcement=RULES[0].enforcement,
        priority=1,
        tags=RULES[0].tags,
    )
    tampered = RuleSet(bad)
    assert tampered.seal != REGISTRY.seal
    with pytest.raises(InvariantViolation):
        tampered.verify(REGISTRY.seal)


def test_duplicate_rule_numbers_rejected():
    with pytest.raises(InvariantViolation):
        RuleSet([RULES[0], RULES[0]])


def test_rule8_refuses_unauthorized_amendment():
    with pytest.raises(AuthorizationError):
        REGISTRY.amend(RULES, owner_ack="impostor")
    with pytest.raises(AuthorizationError):
        REGISTRY.amend(RULES, owner_ack="emergency-override")


def test_rule8_allows_owner_amendment():
    amended = REGISTRY.amend(RULES, owner_ack=OWNER.continuity_namespace)
    assert isinstance(amended, RuleSet)
    assert amended.seal == REGISTRY.seal  # same content -> same seal


def test_rule_digest_is_stable_and_content_addressed():
    r = REGISTRY.by_number(1)
    d1 = r.digest()
    d2 = REGISTRY.by_number(1).digest()
    assert d1 == d2
    assert r.digest() != REGISTRY.by_number(2).digest()


def test_rule_frozen_immutable():
    r = REGISTRY.by_number(1)
    with pytest.raises(Exception):
        r.statement = "changed"  # type: ignore[misc]


def test_enforcement_values_are_known():
    valid = set(Enforcement)
    for r in REGISTRY:
        for e in r.enforcement:
            assert e in valid


def test_to_json_includes_seal_and_rules():
    import json

    data = json.loads(REGISTRY.to_json())
    assert data["seal"] == RULES_SEAL
    assert len(data["rules"]) == 8
