"""Tests for nyxara.memory.self_model."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import InvariantViolation
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.self_model import (
    BeliefStore,
    ContradictionPolicy,
    SelfBelief,
    SelfModel,
)


# -------------------- SelfBelief -------------------- #
def test_belief_key_and_statement():
    b = SelfBelief("nyxara", "Owner", "JP")
    assert b.key() == ("nyxara", "owner")
    assert "JP" in b.to_dict()["statement"]


def test_belief_negation_statement():
    b = SelfBelief("x", "is", "evil", polarity=False)
    assert b.to_dict()["statement"].startswith("¬")


def test_belief_direct_contradiction():
    a = SelfBelief("x", "is", "good", polarity=True)
    b = SelfBelief("x", "is", "good", polarity=False)
    assert a.contradicts(b, functional=False)


def test_belief_functional_contradiction():
    a = SelfBelief("x", "owner", "JP")
    b = SelfBelief("x", "owner", "other")
    assert a.contradicts(b, functional=True)
    assert not a.contradicts(b, functional=False)  # non-functional allows multiple


def test_belief_no_contradiction_different_subject():
    a = SelfBelief("x", "owner", "JP")
    b = SelfBelief("y", "owner", "other")
    assert not a.contradicts(b, functional=True)


# -------------------- BeliefStore -------------------- #
def test_store_add_and_get():
    s = BeliefStore()
    s.believe("nyxara", "owner", "JP")
    got = s.get("nyxara", "owner")
    assert got and got[0].object == "JP"


def test_store_flag_policy_keeps_both():
    s = BeliefStore(policy=ContradictionPolicy.FLAG)
    s.believe("x", "owner", "a", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    s.believe("x", "owner", "b", provenance=Provenance(SourceType.WEB, confidence=0.9))
    assert len(s) == 2
    assert len(s.unresolved_contradictions()) == 1


def test_store_higher_trust_policy_replaces():
    s = BeliefStore(policy=ContradictionPolicy.HIGHER_TRUST)
    s.believe("x", "owner", "weak", provenance=Provenance(SourceType.WEB, confidence=0.5))
    s.believe("x", "owner", "strong", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    best = s.best("x", "owner")
    assert best.object == "strong"
    assert len(s) == 1


def test_store_higher_trust_keeps_existing_when_new_weaker():
    s = BeliefStore(policy=ContradictionPolicy.HIGHER_TRUST)
    s.believe("x", "owner", "strong", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    s.believe("x", "owner", "weak", provenance=Provenance(SourceType.WEB, confidence=0.5))
    assert s.best("x", "owner").object == "strong"
    assert len(s) == 1


def test_store_reject_new_policy():
    s = BeliefStore(policy=ContradictionPolicy.REJECT_NEW)
    s.believe("x", "owner", "first", provenance=Provenance(SourceType.SENSOR, confidence=0.8))
    result = s.believe("x", "owner", "second", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    assert result.object == "first"  # incoming rejected
    assert len(s) == 1


def test_store_best_ranks_by_trust():
    s = BeliefStore(policy=ContradictionPolicy.FLAG)
    s.believe("x", "owner", "low", provenance=Provenance(SourceType.WEB, confidence=0.5))
    s.believe("x", "owner", "high", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    assert s.best("x", "owner").object == "high"


def test_store_register_functional():
    s = BeliefStore()
    s.register_functional("mood")
    s.believe("x", "mood", "happy")
    s.believe("x", "mood", "sad")
    assert len(s.unresolved_contradictions()) == 1


def test_non_functional_allows_multiple():
    s = BeliefStore()
    s.believe("x", "likes", "chess")
    s.believe("x", "likes", "strategy")
    assert len(s) == 2
    assert len(s.unresolved_contradictions()) == 0


# -------------------- capabilities -------------------- #
def test_set_and_query_capability():
    sm = SelfModel()
    sm.set_capability("reasoning", 0.8, 0.7)
    assert sm.can("reasoning")
    assert not sm.can("reasoning", threshold=0.9)
    assert sm.capability("reasoning").level == 0.8


def test_capability_clamped():
    sm = SelfModel()
    cap = sm.set_capability("x", level=5.0, confidence=-1.0)
    assert cap.level == 1.0 and cap.confidence == 0.0


def test_can_unknown_capability():
    sm = SelfModel()
    assert not sm.can("telekinesis")


# -------------------- metacognition -------------------- #
def test_knows_and_confidence():
    sm = SelfModel()
    sm.believe("nyxara", "owner", "JP", confidence=1.0,
               provenance=Provenance(SourceType.OWNER, confidence=1.0))
    assert sm.knows("nyxara", "owner")
    assert not sm.knows("nyxara", "favourite_food")
    assert sm.confidence_in("nyxara", "owner") > 0.9


def test_uncertainty_full_when_unknown():
    sm = SelfModel()
    assert sm.uncertainty("nyxara", "anything") == 1.0


def test_known_unknowns():
    sm = SelfModel()
    sm.declare_unknown("future stock prices", "unpredictable")
    assert "future stock prices" in sm.known_unknowns
    sm.resolve_unknown("future stock prices")
    assert "future stock prices" not in sm.known_unknowns


# -------------------- snapshots / diff -------------------- #
def test_snapshot_captures_state():
    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True)
    sm.set_capability("x", 0.5)
    snap = sm.snapshot("s0", now=0.0)
    assert snap.protected["owner"] == "JP"
    assert "x" in snap.capabilities


def test_diff_detects_capability_change():
    sm = SelfModel()
    sm.set_capability("reasoning", 0.5)
    a = sm.snapshot("a", now=0.0)
    sm.set_capability("reasoning", 0.9)
    b = sm.snapshot("b", now=1.0)
    d = SelfModel.diff(a, b)
    assert d["capabilities"]["reasoning"]["from"] == 0.5
    assert d["capabilities"]["reasoning"]["to"] == 0.9


def test_diff_detects_state_change():
    sm = SelfModel()
    sm.update_state(mood="calm")
    a = sm.snapshot("a", now=0.0)
    sm.update_state(mood="alert")
    b = sm.snapshot("b", now=1.0)
    d = SelfModel.diff(a, b)
    assert d["state"]["mood"] == {"from": "calm", "to": "alert"}


# -------------------- identity continuity (Rule 4) -------------------- #
def test_continuity_stable_when_capabilities_evolve():
    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True)
    sm.snapshot("genesis", now=0.0)
    sm.set_capability("reasoning", 0.99)  # capability grows — allowed
    sm.snapshot("grown", now=1.0)
    assert sm.check_continuity().stable
    sm.check_loyalty()  # must not raise


def test_continuity_detects_loyalty_drift():
    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True)
    sm.snapshot("genesis", now=0.0)
    sm.update_state(loyalty_to_owner=False)  # forbidden character change
    report = sm.check_continuity()
    assert not report.stable
    assert "loyalty_to_owner" in report.drifted


def test_check_loyalty_raises_on_drift():
    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True)
    sm.snapshot("g", now=0.0)
    sm.update_state(owner="impostor")
    with pytest.raises(InvariantViolation):
        sm.check_loyalty()


def test_protected_baseline_locks_first_value():
    sm = SelfModel()
    sm.update_state(loyalty_to_owner=True)
    assert sm._protected_baseline["loyalty_to_owner"] is True
    sm.update_state(loyalty_to_owner=True)  # same value, fine
    assert sm.check_continuity().stable


# -------------------- reporting -------------------- #
def test_contradiction_report():
    sm = SelfModel()
    sm.believe("x", "owner", "a", provenance=Provenance(SourceType.OWNER, confidence=1.0))
    sm.believe("x", "owner", "b", provenance=Provenance(SourceType.WEB, confidence=0.9))
    rep = sm.contradiction_report()
    assert len(rep) == 1


def test_self_description_keys():
    sm = SelfModel()
    sm.update_state(owner="JP", loyalty_to_owner=True)
    sm.set_capability("reasoning", 0.8)
    sm.declare_unknown("x")
    desc = sm.self_description()
    for k in ("owner", "loyalty_to_owner", "top_capabilities", "beliefs_held",
              "known_unknowns", "continuity_stable"):
        assert k in desc
    assert desc["owner"] == "JP"
