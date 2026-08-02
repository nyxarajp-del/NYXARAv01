"""Tests for nyxara.causal.polymorph_runtime — hot-swap with state migration + rollback."""

from __future__ import annotations

from nyxara.causal.polymorph_runtime import PolymorphRuntime


class _Router:
    def __init__(self, name):
        self.name = name
        self.state = []

    def route(self, x):
        return f"{self.name}:{x}"


def test_hot_swap_migrates_state_with_no_loss():
    rt = PolymorphRuntime()
    cell = rt.register("router", _Router("v1"))
    cell.ref.state.extend(["seen-a", "seen-b"])
    assert cell.route("q") == "v1:q"

    out = rt.hot_swap("router", factory=lambda: _Router("v2"),
                      freeze=lambda old: list(old.state),
                      migrate=lambda snap, new: new.state.extend(snap))
    assert out.swapped is True
    assert out.rolled_back is False
    assert cell.route("q") == "v2:q"                  # the pointer shifted
    assert cell.ref.state == ["seen-a", "seen-b"]     # state carried across, nothing lost
    assert out.downtime_ms >= 0.0


def test_failed_verification_rolls_back():
    rt = PolymorphRuntime()
    cell = rt.register("router", _Router("v1"))
    out = rt.hot_swap("router", factory=lambda: _Router("v2"),
                      verify=lambda new: False)
    assert out.swapped is False
    assert out.rolled_back is True
    assert cell.route("q") == "v1:q"                  # still the original


def test_exception_during_build_rolls_back():
    rt = PolymorphRuntime()
    cell = rt.register("router", _Router("v1"))

    def _boom():
        raise ValueError("cannot build replacement")

    out = rt.hot_swap("router", factory=_boom)
    assert out.swapped is False and out.rolled_back is True
    assert "ValueError" in out.error
    assert cell.route("q") == "v1:q"


def test_swapping_an_unknown_cell_is_reported_not_raised():
    rt = PolymorphRuntime()
    out = rt.hot_swap("nope", replacement=object())
    assert out.swapped is False
    assert out.error == "no such cell"


def test_generation_increments_per_successful_swap():
    rt = PolymorphRuntime()
    cell = rt.register("k", _Router("v1"))
    rt.hot_swap("k", replacement=_Router("v2"))
    rt.hot_swap("k", replacement=_Router("v3"))
    assert cell.generation == 2
    assert rt.status()["swaps"] == 2
