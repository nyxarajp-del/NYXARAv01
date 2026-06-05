"""Tests for nyxara.planning.planner."""

from __future__ import annotations

from nyxara.planning.planner import (
    Action,
    DependencyGraph,
    Plan,
    Planner,
    ResourceScheduler,
    STRIPSProblem,
)


def _chain():
    actions = [
        Action.make("get_key", add=["have_key"]),
        Action.make("open_door", pre=["have_key"], add=["door_open"]),
        Action.make("enter", pre=["door_open"], add=["inside"], delete=["outside"]),
    ]
    return STRIPSProblem(frozenset({"outside"}), frozenset({"inside"}), actions)


# -------------------- Action -------------------- #
def test_action_applicable():
    a = Action.make("x", pre=["p"])
    assert a.applicable(frozenset({"p", "q"}))
    assert not a.applicable(frozenset({"q"}))


def test_action_apply():
    a = Action.make("x", add=["new"], delete=["old"])
    result = a.apply(frozenset({"old", "keep"}))
    assert "new" in result and "old" not in result and "keep" in result


def test_action_resources():
    a = Action.make("x", resources={"compute": 3})
    assert a.resource_map() == {"compute": 3}


# -------------------- A* planning -------------------- #
def test_plan_finds_chain():
    plan = Planner().plan(_chain())
    assert plan is not None
    assert plan.names() == ["get_key", "open_door", "enter"]


def test_plan_validity():
    problem = _chain()
    plan = Planner().plan(problem)
    state = problem.initial
    for a in plan.actions:
        assert a.applicable(state)
        state = a.apply(state)
    assert problem.goal <= state


def test_plan_already_satisfied():
    problem = STRIPSProblem(frozenset({"done"}), frozenset({"done"}), [])
    plan = Planner().plan(problem)
    assert plan.length == 0
    assert plan.cost == 0.0


def test_no_plan_when_impossible():
    problem = STRIPSProblem(frozenset({"outside"}), frozenset({"moon"}),
                            [Action.make("get_key", add=["have_key"])])
    assert Planner().plan(problem) is None


def test_plan_cost_accumulates():
    actions = [Action.make("a", add=["x"], cost=2.0),
               Action.make("b", pre=["x"], add=["y"], cost=3.0)]
    plan = Planner().plan(STRIPSProblem(frozenset(), frozenset({"y"}), actions))
    assert plan.cost == 5.0


def test_plan_prefers_cheaper():
    actions = [
        Action.make("expensive", add=["goal"], cost=10.0),
        Action.make("step1", add=["mid"], cost=1.0),
        Action.make("step2", pre=["mid"], add=["goal"], cost=1.0),
    ]
    plan = Planner().plan(STRIPSProblem(frozenset(), frozenset({"goal"}), actions))
    assert plan.cost == 2.0  # cheaper two-step path
    assert plan.names() == ["step1", "step2"]


def test_plan_to_dict():
    d = Planner().plan(_chain()).to_dict()
    assert d["length"] == 3 and "cost" in d


# -------------------- dependency graph -------------------- #
def test_dependencies_chain():
    problem = _chain()
    plan = Planner().plan(problem)
    dg = DependencyGraph(plan.actions, problem.initial)
    assert dg.dependencies(1) == {0}
    assert dg.dependencies(2) == {1}


def test_dependencies_initial_facts_no_producer():
    actions = [Action.make("use", pre=["have"], add=["used"])]
    dg = DependencyGraph(actions, initial=frozenset({"have"}))
    assert dg.dependencies(0) == set()  # 'have' is already true


def test_topological_order():
    problem = _chain()
    plan = Planner().plan(problem)
    dg = DependencyGraph(plan.actions, problem.initial)
    order = dg.topological_order()
    assert order.index(0) < order.index(1) < order.index(2)


def test_layers_sequential():
    problem = _chain()
    plan = Planner().plan(problem)
    dg = DependencyGraph(plan.actions, problem.initial)
    assert len(dg.layers()) == 3  # fully sequential


def test_layers_parallel():
    actions = [
        Action.make("a", add=["a_done"]),
        Action.make("b", add=["b_done"]),
        Action.make("c", pre=["a_done", "b_done"], add=["c_done"]),
    ]
    # a and b independent
    dg = DependencyGraph(actions)
    layers = dg.layers()
    assert any(len(l) == 2 for l in layers)


def test_dependency_to_dict():
    problem = _chain()
    plan = Planner().plan(problem)
    d = DependencyGraph(plan.actions, problem.initial).to_dict()
    assert "edges" in d and "layers" in d


# -------------------- resource scheduling -------------------- #
def _parallel_plan():
    actions = [
        Action.make("scan_a", add=["a_done"], resources={"compute": 3}),
        Action.make("scan_b", add=["b_done"], resources={"compute": 3}),
        Action.make("report", pre=["a_done", "b_done"], add=["reported"]),
    ]
    problem = STRIPSProblem(frozenset(), frozenset({"reported"}), actions)
    return Planner().plan(problem), problem


def test_schedule_respects_resource_capacity():
    plan, _ = _parallel_plan()
    tight = ResourceScheduler({"compute": 4}).schedule(plan.actions)
    # two 3-compute scans can't share a 4-compute step
    scan_steps = [s for s in tight if any("scan" in a for a in s.actions)]
    assert all(len([a for a in s.actions if "scan" in a]) == 1 for s in scan_steps)


def test_schedule_parallel_when_budget_allows():
    plan, _ = _parallel_plan()
    loose = ResourceScheduler({"compute": 8}).schedule(plan.actions)
    assert len(loose[0].actions) == 2  # both scans together


def test_schedule_respects_dependencies():
    plan, _ = _parallel_plan()
    steps = ResourceScheduler({"compute": 8}).schedule(plan.actions)
    # report must come after the scans
    report_step = next(i for i, s in enumerate(steps) if "report" in s.actions)
    scan_step = next(i for i, s in enumerate(steps) if "scan_a" in s.actions)
    assert report_step > scan_step


def test_schedule_oversized_action_runs_alone():
    actions = [Action.make("huge", add=["done"], resources={"compute": 100})]
    steps = ResourceScheduler({"compute": 4}).schedule(actions)
    assert len(steps) == 1 and steps[0].actions == ["huge"]


def test_schedule_no_capacity_is_unlimited():
    plan, _ = _parallel_plan()
    steps = ResourceScheduler().schedule(plan.actions)  # no capacities -> unlimited
    assert len(steps[0].actions) == 2  # both scans parallel
