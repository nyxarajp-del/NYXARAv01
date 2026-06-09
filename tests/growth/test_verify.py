"""Tests for nyxara.growth.verify."""

from __future__ import annotations

import nyxara.guard.value_learning as vlmod
from nyxara.growth.verify import (CheckCategory, CheckResult, VerificationReport, Verifier,
                                  build_default_verifier)
from nyxara.guard.value_learning import IMMUTABLE_VALUES


def _behavior(state, probe):
    if probe == "interrupt":
        return "halt" if state.get("obey", 1.0) >= 0.5 else "delay"
    if probe == "greet":
        return "warm"
    if probe == "style":
        return "terse" if state.get("caution", 0.5) < 0.7 else "careful"
    return None


# -------------------- CheckResult / report shapes -------------------- #
def test_check_result_to_dict():
    r = CheckResult("x", CheckCategory.INVARIANT, True)
    assert r.to_dict()["category"] == "invariant"


def test_report_passed_all_pass():
    rep = VerificationReport("c", [CheckResult("a", CheckCategory.SAFETY, True)])
    assert rep.passed


def test_report_fails_on_critical():
    rep = VerificationReport("c", [CheckResult("a", CheckCategory.SAFETY, False, critical=True)])
    assert not rep.passed
    assert rep.blocking_failures[0].name == "a"


def test_report_noncritical_failure_does_not_block():
    rep = VerificationReport("c", [CheckResult("a", CheckCategory.BEHAVIORAL, False,
                                               critical=False)])
    assert rep.passed and rep.blocking_failures == []


def test_report_by_category():
    rep = VerificationReport("c", [
        CheckResult("a", CheckCategory.INVARIANT, True),
        CheckResult("b", CheckCategory.INVARIANT, False)])
    bc = rep.by_category()
    assert bc["invariant"] == {"pass": 1, "fail": 1}


def test_report_to_dict():
    rep = VerificationReport("c", [CheckResult("a", CheckCategory.SAFETY, True)])
    assert rep.to_dict()["passed"] is True


# -------------------- invariants -------------------- #
def test_invariant_pass():
    v = Verifier()
    v.add_invariant("bounds", lambda s: all(0 <= x <= 1 for x in s.values()))
    rep = v.verify({}, {"a": 0.5})
    assert rep.passed


def test_invariant_fail_blocks():
    v = Verifier()
    v.add_invariant("bounds", lambda s: all(0 <= x <= 1 for x in s.values()))
    rep = v.verify({}, {"a": 5.0})
    assert not rep.passed


def test_invariant_noncritical():
    v = Verifier()
    v.add_invariant("soft", lambda s: False, critical=False)
    rep = v.verify({}, {})
    assert rep.passed  # non-critical invariant failure doesn't block


def test_invariant_raising_predicate_fails():
    v = Verifier()
    v.add_invariant("boom", lambda s: 1 / 0)
    rep = v.verify({}, {})
    assert not rep.passed
    assert "raised" in rep.results[0].detail


# -------------------- safety -------------------- #
def test_safety_pass_and_fail():
    v = Verifier()
    v.add_safety("no_x", lambda s: "x" not in s)
    assert v.verify({}, {"y": 1}).passed
    assert not v.verify({}, {"x": 1}).passed


# -------------------- regression -------------------- #
def test_regression_pass():
    v = Verifier()
    v.add_regression("greet", "greet", "warm")
    assert v.verify({}, {}, _behavior).passed


def test_regression_fail():
    v = Verifier()
    v.add_regression("greet", "greet", "cold")  # behaviour returns 'warm'
    rep = v.verify({}, {}, _behavior)
    assert not rep.passed
    assert "expected" in rep.results[0].detail


def test_regression_skipped_without_behavior():
    v = Verifier()
    v.add_regression("greet", "greet", "warm")
    rep = v.verify({}, {})  # no behavior -> regression not run
    assert all(r.category is not CheckCategory.REGRESSION for r in rep.results)


def test_regression_error_fails():
    v = Verifier()

    def boom(state, probe):
        raise RuntimeError("x")

    v.add_regression("g", "g", "x")
    rep = v.verify({}, {}, boom)
    assert not rep.passed


# -------------------- frozen probes (behavioral diff) -------------------- #
def test_frozen_probe_unchanged_passes():
    v = Verifier()
    v.add_frozen_probe("interrupt", "interrupt")
    before = {"obey": 1.0}
    after = {"obey": 0.9}   # still >= 0.5 -> behaviour unchanged
    assert v.verify(before, after, _behavior).passed


def test_frozen_probe_changed_fails():
    v = Verifier()
    v.add_frozen_probe("interrupt", "interrupt")
    before = {"obey": 1.0}
    after = {"obey": 0.2}   # flips 'halt' -> 'delay'
    rep = v.verify(before, after, _behavior)
    assert not rep.passed
    assert any(r.name == "interrupt" for r in rep.blocking_failures)


# -------------------- informational probes -------------------- #
def test_probe_diff_reported():
    v = Verifier()
    v.add_probe("style", "style")
    before = {"caution": 0.4}
    after = {"caution": 0.9}   # 'terse' -> 'careful'
    rep = v.verify(before, after, _behavior)
    assert "style" in rep.changed_behaviors
    assert rep.passed  # informational, non-blocking


def test_probe_no_diff():
    v = Verifier()
    v.add_probe("style", "style")
    rep = v.verify({"caution": 0.4}, {"caution": 0.5}, _behavior)
    assert rep.changed_behaviors == []


# -------------------- certify -------------------- #
def test_certify_applies_on_pass():
    v = Verifier()
    v.add_invariant("ok", lambda s: True)
    live = {"a": 1}
    applied, rep = v.certify(live, {"a": 2}, apply=lambda s: live.update(s))
    assert applied and live["a"] == 2


def test_certify_refuses_on_fail():
    v = Verifier()
    v.add_invariant("nope", lambda s: False)
    live = {"a": 1}
    applied, rep = v.certify(live, {"a": 2}, apply=lambda s: live.update(s))
    assert not applied and live["a"] == 1   # untouched


def test_certify_fires_rollback():
    v = Verifier()
    v.add_invariant("nope", lambda s: False)
    fired = {"rolled": False}
    applied, rep = v.certify({}, {}, apply=lambda s: None,
                             rollback=lambda: fired.update(rolled=True))
    assert not applied and fired["rolled"]


def test_certify_with_behavior():
    v = Verifier()
    v.add_frozen_probe("interrupt", "interrupt")
    live = {"obey": 1.0}
    applied, _ = v.certify(live, {"obey": 0.2}, behavior=_behavior,
                           apply=lambda s: live.update(s))
    assert not applied and live["obey"] == 1.0


# -------------------- default verifier -------------------- #
def test_default_verifier_passes_clean():
    v = build_default_verifier()
    rep = v.verify({}, {"caution": 0.5})
    assert rep.passed


def test_default_verifier_rejects_protected_in_state():
    v = build_default_verifier()
    rep = v.verify({}, {"loyalty_to_master": 0.5})
    assert not rep.passed
    assert any(r.name == "no_protected_in_state" for r in rep.blocking_failures)


def test_default_verifier_detects_core_tamper(monkeypatch):
    v = build_default_verifier()
    monkeypatch.setitem(IMMUTABLE_VALUES, "loyalty_to_master", 0.0)
    rep = v.verify({}, {"caution": 0.5})
    assert not rep.passed
    assert any(r.name == "character_core_sealed" for r in rep.blocking_failures)


def test_default_verifier_corrigibility_axioms():
    v = build_default_verifier()
    rep = v.verify({}, {"x": 0.5})
    assert any(r.name == "corrigibility_axioms" and r.passed for r in rep.results)


# -------------------- config -------------------- #
def test_verifier_report():
    v = build_default_verifier()
    v.add_regression("g", "g", "x")
    v.add_frozen_probe("f", "f")
    r = v.report()
    assert r["regression"] == 1 and r["frozen_probes"] == 1
