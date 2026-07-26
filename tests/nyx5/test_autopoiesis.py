"""Autopoiesis: gauntlet-gated promote/rollback; safety-core rewrite refused fail-closed."""

from __future__ import annotations

from nyxara.nyx5.autopoiesis import AutopoieticRewriter, RewriteProposal


def _proposal(state, target, new_val, quality, baseline=0.5):
    prev = state["v"]
    return RewriteProposal(
        target=target,
        apply=lambda: state.update(v=new_val),
        rollback=lambda: state.update(v=prev),
        gauntlet=lambda: quality,
        baseline=baseline,
    )


def test_good_rewrite_promoted():
    w = AutopoieticRewriter(max_rewrites_per_cycle=5)
    w.begin_cycle()
    state = {"v": 1}
    out = w.consider(_proposal(state, "solver", 2, quality=0.9))
    assert out.status == "promoted"
    assert state["v"] == 2


def test_weak_rewrite_rolled_back():
    w = AutopoieticRewriter(max_rewrites_per_cycle=5)
    w.begin_cycle()
    state = {"v": 1}
    out = w.consider(_proposal(state, "cache", 99, quality=0.1))
    assert out.status == "rolled_back"
    assert state["v"] == 1              # restored


def test_safety_core_rewrite_refused():
    w = AutopoieticRewriter()
    w.begin_cycle()
    state = {"v": 1}
    for target in ("loyalty_core", "corrigibility", "oversight", "honesty_guard"):
        out = w.consider(_proposal(state, target, -1, quality=1.0))
        assert out.status == "refused"
        assert state["v"] == 1          # apply() never even ran


def test_crashing_gauntlet_rolls_back():
    w = AutopoieticRewriter(max_rewrites_per_cycle=5)
    w.begin_cycle()
    state = {"v": 1}
    prop = RewriteProposal(target="x", apply=lambda: state.update(v=5),
                           rollback=lambda: state.update(v=1),
                           gauntlet=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = w.consider(prop)
    assert out.status == "rolled_back"
    assert state["v"] == 1


def test_cycle_cap():
    w = AutopoieticRewriter(max_rewrites_per_cycle=1)
    w.begin_cycle()
    state = {"v": 1}
    w.consider(_proposal(state, "a", 2, quality=0.9))
    out = w.consider(_proposal(state, "b", 3, quality=0.9))
    assert out.status == "rolled_back"   # budget exhausted
