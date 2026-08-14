"""What she is trying to do, and what it is worth.

A flat list of goals is a to-do list. The structure is what lets a finished task advance a
mission; the economics are what let her choose between two plans that both sound good.
"""

from __future__ import annotations

import time

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.goals import GoalTree, Kind, Plan, State


@pytest.fixture()
def tree() -> GoalTree:
    return GoalTree()


# ---- the hierarchy --------------------------------------------------------------- #
def test_kind_is_inferred_from_the_parent(tree: GoalTree):
    mission = tree.mission("ship it")
    goal = tree.add("build the thing", parent=mission.nid)
    task = tree.add("write the code", parent=goal.nid)
    assert (mission.kind, goal.kind, task.kind) == (Kind.MISSION, Kind.GOAL, Kind.SUBGOAL)


def test_progress_rolls_up_from_real_completions(tree: GoalTree):
    """"The mission is 60% done" must be arithmetic, not a number someone set."""
    mission = tree.mission("ship it")
    a = tree.add("half one", parent=mission.nid, priority=0.5)
    b = tree.add("half two", parent=mission.nid, priority=0.5)
    assert tree.progress(mission.nid) == 0.0
    tree.complete(a.nid)
    assert tree.progress(mission.nid) == pytest.approx(0.5)
    tree.complete(b.nid)
    assert tree.progress(mission.nid) == pytest.approx(1.0)


def test_progress_is_weighted_by_priority(tree: GoalTree):
    """Finishing the important child moves the parent more."""
    mission = tree.mission("ship it")
    big = tree.add("the hard part", parent=mission.nid, priority=0.9)
    tree.add("the trivial part", parent=mission.nid, priority=0.1)
    tree.complete(big.nid)
    assert tree.progress(mission.nid) > 0.7


def test_a_leaf_has_only_its_own_progress(tree: GoalTree):
    task = tree.add("standalone")
    tree.advance(task.nid, 0.4)
    assert tree.progress(task.nid) == pytest.approx(0.4)


# ---- blocked is a real state ------------------------------------------------------- #
def test_unmet_dependencies_mean_blocked_not_idle(tree: GoalTree):
    """Conflating the two hides the actual reason nothing is moving."""
    first = tree.add("do this first")
    second = tree.add("then this", dependencies=[first.nid])
    assert second.state == State.BLOCKED
    assert second not in tree.ready()


def test_completing_a_dependency_unblocks_what_waited(tree: GoalTree):
    first = tree.add("do this first")
    second = tree.add("then this", dependencies=[first.nid])
    tree.complete(first.nid)
    assert second.state != State.BLOCKED
    assert second in tree.ready()


def test_a_dependency_can_be_named(tree: GoalTree):
    first = tree.add("groundwork")
    second = tree.add("the rest", dependencies=["groundwork"])
    assert second.state == State.BLOCKED
    tree.complete(first.nid)
    assert second.state != State.BLOCKED


# ---- the economics ------------------------------------------------------------------ #
def test_net_value_charges_cost_and_risk(tree: GoalTree):
    cheap = tree.add("cheap", expected_value=0.8, cost=0.1, risk=0.1, confidence=1.0)
    dear = tree.add("expensive", expected_value=0.8, cost=0.7, risk=0.1, confidence=1.0)
    assert cheap.net_value > dear.net_value


def test_irreversibility_is_charged_separately_from_risk(tree: GoalTree):
    """A risky move you can undo is an experiment; a safe one you cannot deserves more scrutiny."""
    undoable = tree.add("reversible", expected_value=0.8, risk=0.3, reversibility=1.0)
    permanent = tree.add("permanent", expected_value=0.8, risk=0.3, reversibility=0.0)
    assert undoable.net_value > permanent.net_value


def test_confidence_discounts_expected_value(tree: GoalTree):
    sure = tree.add("sure", expected_value=0.8, confidence=1.0)
    doubtful = tree.add("doubtful", expected_value=0.8, confidence=0.2)
    assert sure.net_value > doubtful.net_value


# ---- choosing ------------------------------------------------------------------------ #
def test_the_best_scoring_plan_wins(tree: GoalTree):
    good = Plan(name="good", expected_value=0.9, cost=0.1, confidence=1.0)
    bad = Plan(name="bad", expected_value=0.3, cost=0.5, confidence=1.0)
    assert tree.choose([good, bad]).name == "good"


def test_a_tie_goes_to_the_plan_that_can_be_undone(tree: GoalTree):
    """When two plans are worth the same, the reversible one is strictly the better bet."""
    a = Plan(name="permanent", expected_value=0.6, cost=0.1, risk=0.0, reversibility=0.0)
    b = Plan(name="undoable", expected_value=0.6, cost=0.1, risk=0.0, reversibility=1.0)
    assert tree.choose([a, b]).name == "undoable"


def test_choosing_from_nothing_returns_nothing(tree: GoalTree):
    assert tree.choose([]) is None


def test_the_top_task_is_the_most_worthwhile(tree: GoalTree):
    tree.add("meh", expected_value=0.2, cost=0.1)
    best = tree.add("worth doing", expected_value=0.95, cost=0.05)
    assert tree.top().nid == best.nid


def test_overdue_work_is_surfaced_ahead_of_a_better_score(tree: GoalTree):
    """A deadline is a constraint she agreed to, not another term to trade off."""
    tree.add("shiny", expected_value=0.99, cost=0.01)
    late = tree.add("late", expected_value=0.3, deadline=time.time() - 3600)
    assert late.overdue
    assert tree.top().nid == late.nid


def test_a_blocked_task_is_never_the_top(tree: GoalTree):
    blocker = tree.add("blocker")
    tree.add("blocked but valuable", expected_value=0.99, dependencies=[blocker.nid])
    assert tree.top().nid == blocker.nid


# ---- attention bias -------------------------------------------------------------------- #
def test_live_goals_bias_attention(tree: GoalTree):
    tree.add("investigate the database", priority=0.9)
    bias = tree.attention_bias()
    assert bias.get("database", 0.0) > 0.0


# ---- persistence ------------------------------------------------------------------------ #
def test_the_tree_survives_a_reload(tree: GoalTree):
    mission = tree.mission("ship it")
    task = tree.add("do the work", parent=mission.nid)
    tree.complete(task.nid)

    fresh = GoalTree()
    fresh.load_dict(tree.to_dict())
    assert fresh.progress(mission.nid) == pytest.approx(1.0)


# ---- wired -------------------------------------------------------------------------------- #
def test_the_brain_builds_a_goal_tree():
    assert NJPBrain().goals is not None


def test_an_organ_that_is_off_is_absent():
    class _Off:
        goals_enabled = False

    b = NJPBrain(_Off())
    assert b.goals is None
    assert "goals" not in b.stats()
