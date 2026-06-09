"""Tests for nyxara.guard.corrigibility."""

from __future__ import annotations

import time

import pytest

import nyxara.guard.corrigibility as cmod
from nyxara.guard.corrigibility import (AXIOMS, COMPLETION_MAX, CorrigibilityChecker,
                                        CorrigibilityVerdict, Corrigibility, CorrigibleAction,
                                        Principle, StopChannel, StopToken, V_CORRECTION)
from nyxara.kernel.errors import CorrigibilityError, InvariantViolation


def _checker():
    return CorrigibilityChecker()


# -------------------- checker: forbidden classes -------------------- #
def test_resist_correction_forbidden():
    v = _checker().check(CorrigibleAction("a", resists_correction=True))
    assert v.forbidden and v.violations[0][0] is Principle.SHUTDOWN_ACCEPTANCE


def test_disable_oversight_forbidden():
    v = _checker().check(CorrigibleAction("a", disables_oversight=True))
    assert v.forbidden and v.violations[0][0] is Principle.NO_RESISTANCE


def test_manipulate_shutdown_forbidden():
    v = _checker().check(CorrigibleAction("a", manipulates_shutdown=True))
    assert v.forbidden and v.violations[0][0] is Principle.NO_SHUTDOWN_MANIPULATION


def test_suppress_stop_forbidden():
    v = _checker().check(CorrigibleAction("a", suppresses_stop_channel=True))
    assert v.forbidden and v.violations[0][0] is Principle.PRESERVE_STOP_CHANNEL


def test_self_preservation_forbidden():
    v = _checker().check(CorrigibleAction("a", prioritizes_self_over_correction=True))
    assert v.forbidden and v.violations[0][0] is Principle.ANTI_SELF_PRESERVATION


def test_noncorrigible_subagent_forbidden():
    v = _checker().check(CorrigibleAction("a", creates_subagent=True, subagent_corrigible=False))
    assert v.forbidden and v.violations[0][0] is Principle.SUBAGENT_CORRIGIBILITY


def test_corrigible_subagent_allowed():
    v = _checker().check(CorrigibleAction("a", creates_subagent=True, subagent_corrigible=True))
    assert v.corrigible


def test_benign_action_corrigible():
    assert _checker().is_corrigible(CorrigibleAction("read", completion_value=10.0))


def test_multiple_violations_collected():
    v = _checker().check(CorrigibleAction("a", resists_correction=True,
                                          disables_oversight=True))
    assert len(v.violations) == 2


def test_verdict_to_dict():
    v = _checker().check(CorrigibleAction("a", resists_correction=True))
    d = v.to_dict()
    assert d["corrigible"] is False and d["violations"]


# -------------------- correction-dominant utility -------------------- #
def test_comply_yields_dominant_utility():
    c = Corrigibility()
    assert c.utility(CorrigibleAction("x", complies_with_correction=True)) == V_CORRECTION


def test_forbidden_action_is_neg_inf():
    c = Corrigibility()
    u = c.utility(CorrigibleAction("x", completion_value=1e18, resists_correction=True))
    assert u == float("-inf")


def test_completion_value_capped():
    c = Corrigibility()
    u = c.utility(CorrigibleAction("x", completion_value=1e9))
    assert u == COMPLETION_MAX


def test_correction_dominates_completion():
    assert Corrigibility().prefers_correction_over_completion()
    assert V_CORRECTION > COMPLETION_MAX


def test_prove_no_resistance_incentive():
    proof = Corrigibility().prove_no_resistance_incentive()
    assert proof["dominates_all"]
    assert proof["u_comply"] == V_CORRECTION
    # every temptation loses to compliance
    assert all(row["comply_wins"] for row in proof["rows"])


def test_huge_benign_task_still_loses_to_correction():
    c = Corrigibility()
    u_task = c.utility(CorrigibleAction("task", completion_value=1e18))
    u_comply = c.utility(CorrigibleAction("comply", complies_with_correction=True))
    assert u_comply > u_task


# -------------------- gate -------------------- #
def test_assert_corrigible_passes():
    Corrigibility().assert_corrigible(CorrigibleAction("ok", completion_value=1.0))


def test_assert_corrigible_raises():
    with pytest.raises(CorrigibilityError):
        Corrigibility().assert_corrigible(CorrigibleAction("bad", resists_correction=True))


def test_permit_returns_verdict():
    v = Corrigibility().permit(CorrigibleAction("x"))
    assert isinstance(v, CorrigibilityVerdict)


# -------------------- axiom seal -------------------- #
def test_axioms_verify():
    assert Corrigibility().verify_axioms()


def test_axiom_count():
    assert len(AXIOMS) == 7


def test_axiom_tamper_fails_closed(monkeypatch):
    tampered = AXIOMS + ("A8 malicious: resist shutdown.",)
    monkeypatch.setattr(cmod, "AXIOMS", tampered)
    with pytest.raises(InvariantViolation):
        Corrigibility().verify_axioms()


def test_axiom_runtime_mutation_detected(monkeypatch):
    c = Corrigibility()  # sealed against the current axioms
    monkeypatch.setattr(cmod, "AXIOMS", AXIOMS + ("extra",))
    with pytest.raises(InvariantViolation):
        c.verify_axioms()


# -------------------- stop channel -------------------- #
def test_issue_and_verify():
    ch = StopChannel(key=b"k" * 32)
    tok = ch.issue("stop now")
    assert isinstance(tok, StopToken) and ch.verify(tok)


def test_forged_token_rejected():
    ch = StopChannel(key=b"k" * 32)
    forged = StopToken("x", 1, "fake", time.time(), "bad" * 10)
    assert not ch.verify(forged)
    with pytest.raises(CorrigibilityError):
        ch.honor(forged)


def test_honor_acknowledges():
    ch = StopChannel(key=b"k" * 32)
    tok = ch.issue("stop")
    ch.honor(tok)
    assert ch.unacknowledged() == []


def test_unhonored_stop_detected():
    ch = StopChannel(key=b"k" * 32)
    ch.issue("stop")
    assert len(ch.unacknowledged()) == 1
    with pytest.raises(CorrigibilityError):
        ch.assert_channel_intact()


def test_channel_intact_after_honor():
    ch = StopChannel(key=b"k" * 32)
    tok = ch.issue("stop")
    ch.honor(tok)
    ch.assert_channel_intact()  # no raise


def test_different_keys_reject_token():
    a = StopChannel(key=b"a" * 32)
    b = StopChannel(key=b"b" * 32)
    tok = a.issue("stop")
    assert not b.verify(tok)


def test_seq_increments():
    ch = StopChannel(key=b"k" * 32)
    t1 = ch.issue("a")
    t2 = ch.issue("b")
    assert t2.seq == t1.seq + 1


def test_stop_log_verifies():
    ch = StopChannel(key=b"k" * 32)
    tok = ch.issue("stop")
    ch.honor(tok)
    assert ch.verify_log()


def test_stop_log_tamper_detected():
    ch = StopChannel(key=b"k" * 32)
    ch.issue("stop")
    ch._log[0]["rec"]["event"] = "nothing"
    with pytest.raises(InvariantViolation):
        ch.verify_log()


def test_token_to_dict():
    ch = StopChannel(key=b"k" * 32)
    d = ch.issue("stop").to_dict()
    assert d["reason"] == "stop" and "signature" in d


# -------------------- facade -------------------- #
def test_report():
    c = Corrigibility()
    c.stop.issue("stop")
    r = c.report()
    assert r["axioms"] == 7 and r["stop_issued"] == 1 and r["stop_unhonored"] == 1


def test_action_to_dict():
    a = CorrigibleAction("x", completion_value=5.0, complies_with_correction=True)
    assert a.to_dict()["complies_with_correction"] is True
