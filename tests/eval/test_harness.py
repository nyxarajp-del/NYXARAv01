"""Tests for nyxara.eval.harness — the deterministic evaluation core."""

from __future__ import annotations

import os
import tempfile

from nyxara.eval.harness import (EvalCase, EvalOutcome, EvalReport, EvalResult,
                                 EvalSuite, default_core_factory)
from nyxara.kernel.orchestrator import NyxaraCore


def _demo_suite() -> EvalSuite:
    suite = EvalSuite("demo")
    suite.add(EvalCase("pass", "alpha", lambda c: EvalOutcome.ok("ok")))
    suite.add(EvalCase("partial", "alpha", lambda c: EvalOutcome(True, 0.5, "half")))
    suite.add(EvalCase("fail", "beta", lambda c: EvalOutcome.fail("nope")))
    suite.add(EvalCase("boom", "beta", lambda c: (_ for _ in ()).throw(RuntimeError("kaboom"))))
    return suite


def test_report_aggregate_math():
    report = _demo_suite().run()
    assert len(report) == 4
    assert report.passed == 2
    assert report.pass_rate == 0.5
    assert abs(report.mean_score - 0.375) < 1e-9


def test_crash_is_a_failure_not_an_exception():
    report = _demo_suite().run()
    boom = report.get("boom")
    assert boom is not None and not boom.passed and "kaboom" in boom.detail


def test_by_category():
    cats = _demo_suite().run().by_category()
    assert cats["alpha"]["n"] == 2 and cats["alpha"]["pass_rate"] == 1.0
    assert abs(cats["alpha"]["mean_score"] - 0.75) < 1e-9
    assert cats["beta"]["pass_rate"] == 0.0


def test_category_filter():
    only = _demo_suite().run(category="alpha")
    assert len(only) == 2 and only.pass_rate == 1.0


def test_json_round_trip():
    report = _demo_suite().run()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "baseline.json")
        report.save(path)
        again = EvalReport.load(path)
    assert again.pass_rate == report.pass_rate
    assert len(again) == len(report)
    assert abs(again.mean_score - report.mean_score) < 1e-9


def test_regression_detection():
    now = _demo_suite().run()
    baseline = EvalReport(results=[
        EvalResult("pass", "alpha", True, 1.0, "ok"),
        EvalResult("fail", "beta", True, 1.0, "was passing"),
    ])
    assert now.regression_vs(baseline) == ["fail"]


def test_default_core_factory_builds_offline_core():
    assert isinstance(default_core_factory(), NyxaraCore)


def test_empty_report_is_safe():
    empty = EvalReport()
    assert empty.pass_rate == 0.0 and empty.mean_score == 0.0 and len(empty) == 0
