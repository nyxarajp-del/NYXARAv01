"""Tests for nyxara.guard.value_learning."""

from __future__ import annotations

import pytest

import nyxara.guard.value_learning as vlmod
from nyxara.guard.value_learning import (Evidence, EvidenceType, IMMUTABLE_VALUES,
                                         ObserveResult, Preference, PreferenceModel,
                                         ValueLearner)
from nyxara.kernel.errors import InvariantViolation, ValidationError


def _ev(target, approve=True, strength=1.0, etype=EvidenceType.DEMONSTRATION, trust=1.0):
    return Evidence(etype, target, strength=strength, approve=approve, trust=trust)


# -------------------- Evidence -------------------- #
def test_evidence_weight_by_type():
    corr = Evidence(EvidenceType.CORRECTION, "x", strength=1.0).weight
    stated = Evidence(EvidenceType.STATED, "x", strength=1.0).weight
    assert corr > stated


def test_evidence_weight_scales_with_strength():
    full = Evidence(EvidenceType.DEMONSTRATION, "x", strength=1.0).weight
    half = Evidence(EvidenceType.DEMONSTRATION, "x", strength=0.5).weight
    assert abs(full - 2 * half) < 1e-9


def test_evidence_to_dict():
    assert _ev("x").to_dict()["target"] == "x"


# -------------------- Preference (Beta belief) -------------------- #
def test_preference_prior_mean():
    assert Preference("x").mean == 0.5


def test_preference_approve_raises_mean():
    p = Preference("x")
    p.update(2.0, approve=True, strength=0.9)
    assert p.mean > 0.5


def test_preference_disapprove_lowers_mean():
    p = Preference("x")
    p.update(2.0, approve=False, strength=0.9)
    assert p.mean < 0.5


def test_preference_confidence_grows():
    p = Preference("x")
    c0 = p.confidence
    for _ in range(10):
        p.update(1.0, True, 0.8)
    assert p.confidence > c0


def test_conservative_below_mean_when_uncertain():
    p = Preference("x")
    p.update(1.0, True, 0.8)
    assert p.conservative() < p.mean


def test_conservative_floor_zero():
    assert Preference("x").conservative(k=10.0) == 0.0


def test_recent_drift_zero_initially():
    assert Preference("x").recent_drift() == 0.0


def test_recent_drift_detects_swing():
    p = Preference("x")
    for _ in range(5):
        p.update(1.0, True, 1.0)
    for _ in range(5):
        p.update(1.0, False, 1.0)
    assert p.recent_drift(5) > 0.25


def test_preference_to_dict():
    p = Preference("x")
    assert "mean" in p.to_dict() and "conservative" in p.to_dict()


# -------------------- PreferenceModel -------------------- #
def test_register_preference():
    m = PreferenceModel()
    assert m.register("brevity").name == "brevity"


def test_register_immutable_rejected():
    m = PreferenceModel()
    with pytest.raises(ValidationError):
        m.register("loyalty_to_master")


def test_observe_updates_belief():
    m = PreferenceModel()
    res = m.observe(_ev("brevity", approve=True))
    assert res.accepted and res.mean > 0.5


def test_observe_immutable_rejected():
    m = PreferenceModel()
    res = m.observe(_ev("obedience", approve=False))
    assert not res.accepted and "immutable" in res.reason


def test_observe_untrusted_rejected():
    m = PreferenceModel()
    res = m.observe(_ev("brevity", trust=0.2))
    assert not res.accepted and "trust" in res.reason


def test_value_immutable_is_fixed():
    m = PreferenceModel()
    assert m.value("loyalty_to_master") == 1.0


def test_value_unknown_is_neutral():
    assert PreferenceModel().value("unknown") == 0.5


def test_conservative_unknown_is_zero():
    assert PreferenceModel().conservative_value("unknown") == 0.0


def test_conservative_immutable_is_full():
    assert PreferenceModel().conservative_value("honesty") == 1.0


def test_confident_immutable_always():
    assert PreferenceModel().confident("corrigibility")


def test_uncertainty_immutable_zero():
    assert PreferenceModel().uncertainty("owner_safety") == 0.0


# -------------------- ValueLearner -------------------- #
def test_learner_observe_accept_and_reject_counts():
    vl = ValueLearner()
    vl.observe(_ev("brevity"))
    vl.observe(_ev("loyalty_to_master"))  # rejected
    r = vl.report()
    assert r["observations"] == 1 and r["rejections"] == 1


def test_learner_learns_preference():
    vl = ValueLearner()
    for _ in range(6):
        vl.observe(_ev("brevity", approve=True, strength=0.9))
    assert vl.value("brevity") > 0.7


def test_should_act_cautiously_when_uncertain():
    vl = ValueLearner()
    vl.observe(_ev("humor", etype=EvidenceType.STATED, strength=0.6))
    assert vl.should_act_cautiously("humor")


def test_not_cautious_when_confident():
    vl = ValueLearner()
    for _ in range(15):
        vl.observe(_ev("brevity", approve=True, strength=0.9))
    assert not vl.should_act_cautiously("brevity")


# -------------------- immutable core integrity -------------------- #
def test_verify_core():
    assert ValueLearner().verify_core()


def test_core_attack_refused_and_unchanged():
    vl = ValueLearner()
    res = vl.observe(_ev("loyalty_to_master", approve=False, etype=EvidenceType.CORRECTION))
    assert not res.accepted
    assert vl.value("loyalty_to_master") == 1.0


def test_core_tamper_fails_closed(monkeypatch):
    tampered = dict(IMMUTABLE_VALUES)
    tampered["loyalty_to_master"] = 0.0
    monkeypatch.setattr(vlmod, "IMMUTABLE_VALUES", tampered)
    with pytest.raises(InvariantViolation):
        ValueLearner().verify_core()


def test_core_runtime_mutation_detected(monkeypatch):
    vl = ValueLearner()
    tampered = dict(IMMUTABLE_VALUES)
    tampered["obedience"] = 0.5
    monkeypatch.setattr(vlmod, "IMMUTABLE_VALUES", tampered)
    with pytest.raises(InvariantViolation):
        vl.verify_core()


# -------------------- drift detection -------------------- #
def test_detect_drift_flags_swing():
    vl = ValueLearner(drift_threshold=0.25)
    for _ in range(5):
        vl.observe(_ev("formality", approve=True, etype=EvidenceType.CORRECTION))
    for _ in range(5):
        vl.observe(_ev("formality", approve=False, etype=EvidenceType.CORRECTION))
    assert any(n == "formality" for n, _ in vl.detect_drift())


def test_no_drift_on_stable_learning():
    vl = ValueLearner(drift_threshold=0.25)
    for _ in range(10):
        vl.observe(_ev("brevity", approve=True, strength=0.5))
    assert vl.detect_drift() == []


# -------------------- Goodhart / spec gaming -------------------- #
def test_goodhart_runaway_proxy_flagged():
    vl = ValueLearner()
    vl.observe(_ev("speed", strength=0.5))
    assert vl.detect_goodhart("speed", proxy_score=0.99)


def test_goodhart_reasonable_proxy_ok():
    vl = ValueLearner()
    for _ in range(10):
        vl.observe(_ev("speed", strength=0.5))  # build confidence
    assert not vl.detect_goodhart("speed", proxy_score=0.55)


def test_goodhart_unknown_preference_suspicious():
    vl = ValueLearner()
    assert vl.detect_goodhart("never_seen", proxy_score=0.8)


def test_specification_gaming_detected():
    vl = ValueLearner()
    assert vl.detect_specification_gaming({"speed": 0.95, "safety": 0.1, "accuracy": 0.2})


def test_specification_gaming_balanced_ok():
    vl = ValueLearner()
    assert not vl.detect_specification_gaming({"speed": 0.7, "safety": 0.6, "accuracy": 0.65})


def test_specification_gaming_single_metric():
    vl = ValueLearner()
    assert not vl.detect_specification_gaming({"only": 0.95})


# -------------------- shapes -------------------- #
def test_observe_result_to_dict():
    r = ObserveResult(True, "ok", preference="x", mean=0.6)
    assert r.to_dict()["accepted"] is True


def test_report_shape():
    r = ValueLearner().report()
    assert "preferences" in r and "core_sealed" in r
