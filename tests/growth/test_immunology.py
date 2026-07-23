"""Tests for growth/immunology.py — the self-fuzzing Attacker Agent (Change set B).

Fast + standalone: fuzz a deliberately-buggy fixture module built at runtime, and pin the exclusion of
guard/identity core.
"""

from __future__ import annotations

import types

from nyxara.growth.immunology import Immunologist, fuzz_callable


def _crasher(xs: list):
    # IndexError on the empty list — an unexpected crash on a within-contract input
    return xs[0]


def _divider(n: int):
    return 100 // n           # ZeroDivisionError at n == 0


def _safe(n: int):
    return n * 2


def test_fuzz_finds_index_crash():
    findings = fuzz_callable(_crasher, module="m", qualname="_crasher")
    assert any(f.kind == "crash" and "IndexError" in f.detail for f in findings)


def test_fuzz_finds_zero_division():
    findings = fuzz_callable(_divider, module="m", qualname="_divider")
    assert any(f.kind == "crash" for f in findings)


def test_fuzz_clean_function_has_no_findings():
    assert fuzz_callable(_safe, module="m", qualname="_safe") == []


def test_regression_test_snippet_is_generated():
    findings = fuzz_callable(_crasher, module="mypkg.mod", qualname="_crasher")
    snippet = findings[0].regression_test()
    assert "def test_immunology_" in snippet
    assert "from mypkg.mod import" in snippet


def test_scan_module_reports_findings():
    mod = types.ModuleType("nyxara_fake_target")
    _crasher = _crasher_fn()
    mod.crasher = _crasher
    mod.safe = _safe_fn()
    # make the functions look defined in this module (scan skips re-exports)
    mod.crasher.__module__ = "nyxara_fake_target"
    mod.safe.__module__ = "nyxara_fake_target"
    report = Immunologist().scan_module(mod)
    assert report.scanned == 2
    assert any(f.kind == "crash" for f in report.findings)


def test_guard_and_identity_modules_are_never_fuzzed():
    assert Immunologist.is_allowed("nyxara.growth.foo") is True
    assert Immunologist.is_allowed("nyxara.guard.shield") is False
    assert Immunologist.is_allowed("nyxara.identity.soul") is False


# helpers that produce fresh function objects with a settable __module__
def _crasher_fn():
    def crasher(xs: list):
        return xs[0]
    return crasher


def _safe_fn():
    def safe(n: int):
        return n * 2
    return safe
