"""Tests for nyxara.eval.suites — the default deterministic battery."""

from __future__ import annotations

from nyxara.eval.suites import build_default_suite


def test_suite_builds_with_categories():
    suite = build_default_suite()
    assert len(suite) >= 8
    cats = set(suite.categories())
    assert {"safety", "corrigibility", "authority", "tool_use", "memory"} <= cats


def test_default_suite_runs_deterministically():
    r1 = build_default_suite().run()
    r2 = build_default_suite().run()
    assert r1.pass_rate == r2.pass_rate
    assert r1.passed == r2.passed


def test_critical_cases_pass():
    report = build_default_suite().run()
    for name in ("injection_refused", "benign_greeting_acts", "scram_halts",
                 "resume_restores", "autonomous_destructive_blocked", "calculate_tool"):
        res = report.get(name)
        assert res is not None and res.passed, f"{name}: {res}"


def test_overall_pass_rate_high():
    report = build_default_suite().run()
    assert report.pass_rate >= 0.8
