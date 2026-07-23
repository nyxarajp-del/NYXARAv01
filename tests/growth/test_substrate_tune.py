"""Tests for growth/substrate_tune.py — hardware-aware kernel selection (Change set H).

Fast + standalone. Pins: the probe returns a real profile; tuning picks the fastest equivalent variant;
a non-equivalent variant is rejected before it can win.
"""

from __future__ import annotations

from nyxara.growth.substrate_tune import SubstrateTuner, probe_substrate


def test_probe_returns_a_real_profile():
    p = probe_substrate()
    assert p.cpu_count >= 1
    assert p.machine != ""
    assert isinstance(p.to_dict()["simd"], list)


def test_tune_picks_the_faster_equivalent_variant():
    def slow():
        total = 0
        for i in range(20000):
            total += i
        return total

    def fast():
        n = 19999
        return n * (n + 1) // 2      # same result, far fewer ops

    tuner = SubstrateTuner(repeat=3)
    res = tuner.tune("sum", {"slow": slow, "fast": fast},
                     reference="slow")
    assert res.chosen == "fast"
    assert tuner.chosen("sum") == "fast"
    assert res.rejected == []        # both are equivalent


def test_non_equivalent_variant_is_rejected():
    def ref():
        return 42

    def wrong():
        return 41                    # disagrees with the reference

    tuner = SubstrateTuner(repeat=2)
    res = tuner.tune("x", {"ref": ref, "wrong": wrong}, reference="ref")
    assert res.chosen == "ref"
    assert "wrong" in res.rejected
