"""Tests for growth/superoptimize.py — verify-then-benchmark selection (Change set F).

Fast + standalone. Pins: an equivalent-and-faster candidate is adopted; a faster-but-WRONG candidate is
rejected on its counterexample; nothing beating the original keeps the original.
"""

from __future__ import annotations

from nyxara.growth.superoptimize import SuperOptimizer

_CASES = [((n,), {}) for n in range(0, 200)]


def _slow_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total


def _fast_sum(n):
    return n * (n - 1) // 2          # equivalent, O(1)


def _wrong_sum(n):
    return n * n                      # faster but wrong


def test_adopts_equivalent_faster_candidate():
    res = SuperOptimizer(repeat=3).optimize(_slow_sum, {"fast": _fast_sum}, _CASES)
    assert res.chosen == "fast"
    assert res.improved is True
    assert res.speedup >= 1.0
    assert res.rejected == {}


def test_rejects_faster_but_wrong_candidate():
    res = SuperOptimizer(repeat=3).optimize(_slow_sum, {"wrong": _wrong_sum}, _CASES)
    assert res.chosen == "original"
    assert "wrong" in res.rejected


def test_mixed_candidates_pick_only_the_correct_one():
    res = SuperOptimizer(repeat=3).optimize(
        _slow_sum, {"wrong": _wrong_sum, "fast": _fast_sum}, _CASES)
    assert res.chosen == "fast"
    assert "wrong" in res.rejected


def test_equivalence_counterexample_is_reported():
    eq = SuperOptimizer().check_equivalence(_slow_sum, _wrong_sum, _CASES)
    assert eq.equivalent is False
    assert eq.counterexample is not None


def test_no_improvement_keeps_original():
    # candidate is equivalent but not meaningfully faster → keep original (min_speedup bar)
    res = SuperOptimizer(repeat=2, min_speedup=100.0).optimize(
        _slow_sum, {"fast": _fast_sum}, _CASES)
    assert res.chosen == "original"
    assert res.improved is False
