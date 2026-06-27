"""Tests for nyxara.planning.grand_plan — deep, connected, ~1000-step planning."""

from __future__ import annotations

import pytest

from nyxara.planning.grand_plan import DEFAULT_PHASES, GrandPlan, GrandPlanner, PlanLevel


def test_decompose_hits_target_step_budget():
    plan = GrandPlanner().decompose("design and deploy a powered exosuit", target_steps=1000)
    # 8 phases · 5^3 = 1000 leaves with the default depth
    assert plan.leaf_count() == 1000


def test_plan_is_acyclic():
    plan = GrandPlanner().decompose("build a fusion microreactor", target_steps=200)
    assert plan.is_acyclic()
    order = plan.topological_order()
    assert len(order) == plan.leaf_count()


def test_phases_are_connected_cross_phase_dependencies():
    plan = GrandPlanner().decompose("build a robot", target_steps=200)
    by_kind = {}
    for s in plan.steps:
        by_kind.setdefault(s.kind, []).append(s)
    # manufacturing's entry leaf depends on the materials and design phases' exit leaves
    first_manu = by_kind["manufacturing"][0]
    materials_ids = {s.id for s in by_kind["materials"]}
    design_ids = {s.id for s in by_kind["design"]}
    assert any(d in materials_ids for d in first_manu.deps)
    assert any(d in design_ids for d in first_manu.deps)


def test_topological_order_respects_dependencies():
    plan = GrandPlanner().decompose("x", target_steps=64, max_depth=3)
    order = plan.topological_order()
    rank = {nid: i for i, nid in enumerate(order)}
    for s in plan.steps:
        for dep in s.deps:
            assert rank[dep] < rank[s.id]


def test_layers_expose_parallelism():
    plan = GrandPlanner().decompose("x", target_steps=200)
    layers = plan.layers()
    assert len(layers) >= 2
    # every leaf appears in exactly one layer
    flat = [nid for layer in layers for nid in layer]
    assert sorted(flat) == sorted(s.id for s in plan.steps)


def test_to_milestones_preserves_deps_and_order():
    plan = GrandPlanner().decompose("x", target_steps=64, max_depth=3)
    ms = plan.to_milestones()
    assert len(ms) == plan.leaf_count()
    ids = {m.id for m in ms}
    # every dependency referenced by a milestone exists in the set
    for m in ms:
        for d in m.deps:
            assert d in ids
    # milestones are in a dependency-respecting order
    seen = set()
    for m in ms:
        assert all(d in seen for d in m.deps)
        seen.add(m.id)


def test_small_plan_and_depth_control():
    plan = GrandPlanner().decompose("tiny goal", target_steps=8, max_depth=2)
    assert plan.leaf_count() >= 1 and plan.is_acyclic()
    # depth 2 means phases hold leaves directly
    assert plan.root.level is PlanLevel.GOAL
    assert len(plan.root.children) == len(DEFAULT_PHASES)


def test_target_steps_is_bounded():
    # absurd request is clamped, never an unbounded build
    plan = GrandPlanner().decompose("x", target_steps=10_000_000)
    assert plan.leaf_count() <= 20_000 * 8
