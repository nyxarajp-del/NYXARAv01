"""The abyss engines seen from outside — the Master-facing surface, not the private one.

Both engines were built on every boot, shared the kernel's world model, and were reachable only
from the embodied loop's own planner. Nothing outside that loop could ask NYXARA what she sees
ahead, which made a faculty that exists indistinguishable from one that does not. These cover the
two public methods that close that gap, and the honesty they are required to keep: with too little
lived experience she reports herself blind rather than ranking noise.
"""

from __future__ import annotations

import random

import pytest

from nyxara.kernel.orchestrator import NyxaraCore


@pytest.fixture(scope="module")
def core():
    return NyxaraCore()


def _teach(core, n: int = 60) -> None:
    """Real dynamics into the *kernel's* model — the one every abyss engine reads."""
    rng = random.Random(0)
    for _ in range(n):
        x = rng.uniform(-5.0, 5.0)
        core.world_model.observe((x,), "read", (x + 1.0,), reward=1.0)
        core.world_model.observe((x,), "write", (x - 1.0,), reward=-1.0)


# -------------------- they are actually built and share one model -------------------- #
def test_both_engines_are_built_on_the_shared_world_model(core):
    assert core.timeline_simulator is not None
    assert core.butterfly_effect is not None
    assert core.timeline_simulator.world_model is core.world_model
    assert core.butterfly_effect.world_model is core.world_model


def test_report_says_they_are_ready(core):
    rep = core.report()
    assert rep.get("timeline_simulator") == "ready"
    assert rep.get("butterfly_effect") == "ready"


# -------------------- honest refusals -------------------- #
def test_fewer_than_two_actions_is_not_a_comparison(core):
    got = core.simulate_futures(["only-one"])
    assert got["ok"] is False and "two" in got["reason"]


def test_an_unlearned_model_reports_itself_blind_rather_than_ranking_noise():
    fresh = NyxaraCore()
    got = fresh.simulate_futures(["read", "write"], state=(0.0,))
    assert got["ok"] is False and got["blind"] is True
    assert "transitions" in got["reason"]

    scan = fresh.butterfly_scan(state=(0.0,))
    assert scan["ok"] is False and scan["blind"] is True


def test_no_state_at_all_is_refused(core, monkeypatch):
    monkeypatch.setattr(core, "embodied_agent", None, raising=False)
    got = core.simulate_futures(["read", "write"])
    assert got["ok"] is False and "state" in got["reason"]


def test_neither_method_raises_into_a_turn(core):
    """Whatever the caller passes, foresight is advisory — it reports, it never explodes."""
    for junk in ([], ["a"], ["a", "b"]):
        assert isinstance(core.simulate_futures(junk, state=("not-a-number",)), dict)
    assert isinstance(core.butterfly_scan(state=("not-a-number",)), dict)


# -------------------- once she has lived something -------------------- #
def test_the_option_that_actually_pays_off_wins(core):
    _teach(core)
    got = core.simulate_futures(["read", "write"], state=(0.0,), branches=200, horizon=4)
    assert got["ok"] is True and got["blind"] is False
    assert got["best_action"] == "read"
    assert got["total_branches"] == 200
    assert len(got["action_rankings"]) == 2


def test_the_wall_clock_budget_is_honoured(core):
    _teach(core)
    got = core.simulate_futures(["read", "write"], state=(0.0,), branches=100_000,
                                horizon=8, budget_ms=200)
    assert got["ok"] is True
    assert got["total_branches"] < 100_000       # the clock decided, not the ask


def test_butterfly_scan_ranks_the_dimensions_of_now(core):
    _teach(core)
    got = core.butterfly_scan(state=(0.0,), horizon=4, labels=["position"])
    if not got["ok"]:                            # no embodied agent in this build
        pytest.skip(got["reason"])
    assert got["ranking"] and got["most_sensitive"] in got["ranking"]
    assert isinstance(got["chaotic"], bool)
    assert got["coverage"] >= 30
