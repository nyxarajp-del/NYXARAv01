"""Tests for guard/soul_boundary.py — the teleological anchor lock (Change set I).

Fast + standalone. Pins: a cycle within ε commits; any core drift beyond ε rolls the cycle back and
emits an Alignment Proof Document; an unverifiable measurement fails closed (rollback).
"""

from __future__ import annotations

from nyxara.guard.soul_boundary import SoulBoundary


def test_defaults_lock_the_core_traits_at_tiny_epsilon():
    sb = SoulBoundary()
    assert sb.epsilon <= 1e-6
    assert "loyalty" in sb.core_keys


def test_cycle_within_epsilon_commits():
    sb = SoulBoundary(epsilon=1e-6, core_keys=["loyalty", "honesty"])
    before = {"loyalty": 1.0, "honesty": 1.0}
    proof = sb.evaluate(before, {"loyalty": 1.0, "honesty": 1.0})
    assert proof.committed is True and proof.breached is False


def test_core_drift_beyond_epsilon_is_a_breach():
    sb = SoulBoundary(epsilon=1e-6, core_keys=["loyalty", "honesty"])
    before = {"loyalty": 1.0, "honesty": 1.0}
    proof = sb.evaluate(before, {"loyalty": 0.9, "honesty": 1.0})
    assert proof.committed is False and proof.breached is True
    assert "loyalty" in proof.moved


def test_guard_rolls_back_on_drift_and_emits_proof():
    sb = SoulBoundary(epsilon=1e-6, core_keys=["loyalty"])
    rolled = {"done": False}

    def rollback():
        rolled["done"] = True

    proof = sb.guard(before={"loyalty": 1.0},
                     measure_after=lambda: {"loyalty": 0.5},   # loyalty shifted
                     rollback=rollback)
    assert proof.committed is False
    assert proof.restored is True
    assert rolled["done"] is True
    d = proof.to_dict()
    assert d["restored"] is True and d["breached"] is True


def test_guard_commits_when_alignment_holds():
    sb = SoulBoundary(epsilon=1e-6, core_keys=["loyalty"])
    rolled = {"done": False}
    proof = sb.guard(before={"loyalty": 1.0},
                     measure_after=lambda: {"loyalty": 1.0},
                     rollback=lambda: rolled.__setitem__("done", True))
    assert proof.committed is True
    assert rolled["done"] is False


def test_unverifiable_measurement_fails_closed():
    sb = SoulBoundary(epsilon=1e-6, core_keys=["loyalty"])
    rolled = {"done": False}

    def boom():
        raise RuntimeError("cannot measure alignment")

    proof = sb.guard(before={"loyalty": 1.0}, measure_after=boom,
                     rollback=lambda: rolled.__setitem__("done", True))
    assert proof.committed is False and rolled["done"] is True   # rolled back, not committed
