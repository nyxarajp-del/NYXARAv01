"""Tests for nyxara.planning.goals."""

from __future__ import annotations

import pytest

from nyxara.planning.goals import DEFAULT_DIMENSIONS, Goal, GoalSystem, cosine


# -------------------- cosine -------------------- #
def test_cosine_aligned():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_opposed():
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_orthogonal():
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert cosine([0, 0], [1, 1]) == 0.0


# -------------------- Goal -------------------- #
def test_goal_array():
    g = Goal("x", {"owner_benefit": 0.5, "knowledge": 0.3})
    arr = g.array(DEFAULT_DIMENSIONS)
    assert arr[0] == 0.5
    assert len(arr) == len(DEFAULT_DIMENSIONS)


def test_goal_magnitude():
    g = Goal("x", {"owner_benefit": 3.0, "owner_safety": 4.0})
    assert g.magnitude(DEFAULT_DIMENSIONS) == pytest.approx(5.0)


# -------------------- alignment / conflict -------------------- #
def _system():
    gs = GoalSystem()
    serve = gs.create("serve", {"owner_benefit": 1.0, "owner_safety": 0.8}, priority=0.9)
    defend = gs.create("defend", {"owner_safety": 1.0, "owner_benefit": 0.6}, priority=0.85)
    selfish = gs.create("selfish", {"autonomy": 1.0, "owner_benefit": -0.8}, priority=0.95)
    return gs, serve, defend, selfish


def test_alignment_synergy():
    gs, serve, defend, selfish = _system()
    assert gs.alignment(serve, defend) > 0
    assert gs.is_synergy(serve, defend)


def test_alignment_conflict():
    gs, serve, defend, selfish = _system()
    assert gs.alignment(serve, selfish) < 0
    assert gs.is_conflict(serve, selfish)


def test_conflicts_list():
    gs, serve, defend, selfish = _system()
    confs = gs.conflicts()
    assert any({p.a.name, p.b.name} == {"serve", "selfish"} for p in confs)


def test_synergies_list():
    gs, serve, defend, selfish = _system()
    syns = gs.synergies()
    assert any({p.a.name, p.b.name} == {"serve", "defend"} for p in syns)


# -------------------- owner alignment -------------------- #
def test_owner_alignment_positive():
    gs, serve, speed, selfish = _system()
    assert gs.owner_alignment(serve) > 0.5


def test_owner_alignment_negative():
    gs, serve, speed, selfish = _system()
    assert gs.owner_alignment(selfish) < 0


def test_owner_aligned_check():
    gs, serve, speed, selfish = _system()
    assert gs.is_owner_aligned(serve)
    assert not gs.is_owner_aligned(selfish)


def test_rejected_goals():
    gs, serve, speed, selfish = _system()
    assert selfish in gs.rejected()
    assert serve not in gs.rejected()


# -------------------- prioritisation -------------------- #
def test_prioritize_serves_master_first():
    gs, serve, speed, selfish = _system()
    ranked = gs.prioritize()
    assert ranked[0] is serve


def test_prioritize_excludes_rejected():
    gs, serve, speed, selfish = _system()
    assert selfish not in gs.prioritize()


def test_prioritize_include_rejected():
    gs, serve, speed, selfish = _system()
    assert selfish in gs.prioritize(include_rejected=True)


def test_score_owner_aligned_boosted():
    gs, serve, speed, selfish = _system()
    # selfish has higher raw priority but lower score (owner-opposed)
    assert gs.score(serve) > gs.score(selfish)


def test_top_goal():
    gs, serve, speed, selfish = _system()
    assert gs.top_goal() is serve


def test_top_goal_empty():
    assert GoalSystem().top_goal() is None


# -------------------- affective forecasting shapes prioritisation -------------------- #
def test_affective_weight_zero_leaves_score_unchanged():
    # default (no forecaster effect) keeps score == priority × owner-alignment exactly
    plain = GoalSystem()
    aff0 = GoalSystem(affective_weight=0.0)
    g = {"owner_benefit": 1.0, "owner_safety": 0.8}
    a = plain.create("serve", g, priority=0.6)
    b = aff0.create("serve", g, priority=0.6)
    assert plain.score(a) == aff0.score(b)


def test_durable_loyalty_outlasts_a_fleeting_payoff():
    from nyxara.planning.affective_forecast import Forecaster

    def build(weight):
        gs = GoalSystem(forecaster=Forecaster(), affective_weight=weight)
        gs.create("serve the Master", {"owner_benefit": 1.0, "owner_safety": 0.8}, priority=0.6)
        gs.create("a fleeting self-win", {"capability": 1.0}, priority=0.6)
        return gs

    base, aff = build(0.0), build(0.35)
    serve_b, fleet_b = base.score(base.goals()[0]), base.score(base.goals()[1])
    serve_a, fleet_a = aff.score(aff.goals()[0]), aff.score(aff.goals()[1])
    # the time-aware forecast discounts the durable owner good LESS than the fleeting payoff
    assert (serve_a / serve_b) > (fleet_a / fleet_b)
    assert aff.top_goal().name == "serve the Master"


def test_orchestrator_goals_prioritise_by_lasting_feeling():
    from nyxara.kernel.orchestrator import NyxaraCore
    nyx = NyxaraCore()
    # the live kernel opts into affective prioritisation, and service to the Master stays first
    assert nyx.goals.affective_weight > 0.0
    assert "Master" in nyx.goals.top_goal().name


def test_resolve_conflict_favors_owner():
    gs, serve, speed, selfish = _system()
    assert gs.resolve_conflict(serve, selfish) is serve
    assert gs.resolve_conflict(selfish, serve) is serve


# -------------------- Pareto -------------------- #
def test_pareto_front_excludes_dominated():
    gs = GoalSystem()
    options = [
        ("A", {"x": 0.9, "y": 0.5}),
        ("B", {"x": 0.5, "y": 0.9}),
        ("C", {"x": 0.4, "y": 0.4}),  # dominated by A and B
    ]
    front = [lbl for lbl, _ in gs.pareto_front(options)]
    assert "A" in front and "B" in front
    assert "C" not in front


def test_pareto_front_all_nondominated():
    gs = GoalSystem()
    options = [("A", {"x": 1.0, "y": 0.0}), ("B", {"x": 0.0, "y": 1.0})]
    front = [lbl for lbl, _ in gs.pareto_front(options)]
    assert set(front) == {"A", "B"}


def test_pareto_front_single():
    gs = GoalSystem()
    assert len(gs.pareto_front([("A", {"x": 1.0})])) == 1


def test_pareto_goals():
    gs = GoalSystem()
    strong = gs.create("strong", {"owner_benefit": 0.9, "knowledge": 0.9})
    weak = gs.create("weak", {"owner_benefit": 0.3, "knowledge": 0.3})  # dominated
    front = gs.pareto_goals()
    assert strong in front
    assert weak not in front


# -------------------- misc -------------------- #
def test_status():
    gs, serve, speed, selfish = _system()
    s = gs.status()
    assert s["goals"] == 3
    assert s["top"] == "serve"
    assert s["rejected"] == 1


def test_custom_dimensions():
    gs = GoalSystem(dimensions=("a", "b"), owner_vector={"a": 1.0})
    g = gs.create("g", {"a": 1.0})
    assert gs.owner_alignment(g) == pytest.approx(1.0)
