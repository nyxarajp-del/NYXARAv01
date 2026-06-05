"""NYXARA · planning/planner.py — STRIPS + A* + dependency graph + resources.

Classical, *verifiable* planning: when a goal can be reached by composing actions
whose effects are known, NYXARA computes a correct plan rather than guessing one.

* **STRIPS** — the world is a set of facts; each :class:`Action` has preconditions
  (facts that must hold), add-effects, and delete-effects, plus a cost and resource
  demands.
* **A\\*** — search from the initial state to a goal (a set of facts that must hold),
  minimising total cost with an unmet-goal heuristic.
* **Dependency graph** — over a found plan, edges run producer → consumer (one action's
  effect satisfies another's precondition), giving a topological order and the layers
  of mutually-independent actions.
* **Resource scheduling** — pack the plan into time-steps that respect both the
  dependencies and per-step resource budgets, running independent actions in parallel
  where the budget allows.

Pure standard library; the symbolic backbone the orchestrator and agency layer use to
turn goals into ordered, schedulable action sequences.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "Action",
    "STRIPSProblem",
    "Plan",
    "Planner",
    "DependencyGraph",
    "ResourceScheduler",
]

State = FrozenSet[str]


# --------------------------------------------------------------------------- #
# Action & problem
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    name: str
    preconditions: FrozenSet[str] = frozenset()
    add_effects: FrozenSet[str] = frozenset()
    del_effects: FrozenSet[str] = frozenset()
    cost: float = 1.0
    resources: Tuple[Tuple[str, float], ...] = ()   # frozen (resource, amount) pairs

    def applicable(self, state: State) -> bool:
        return self.preconditions <= state

    def apply(self, state: State) -> State:
        return frozenset((state - self.del_effects) | self.add_effects)

    def resource_map(self) -> Dict[str, float]:
        return dict(self.resources)

    @classmethod
    def make(cls, name: str, *, pre: Sequence[str] = (), add: Sequence[str] = (),
             delete: Sequence[str] = (), cost: float = 1.0,
             resources: Optional[Dict[str, float]] = None) -> "Action":
        return cls(name=name, preconditions=frozenset(pre), add_effects=frozenset(add),
                   del_effects=frozenset(delete), cost=cost,
                   resources=tuple(sorted((resources or {}).items())))


@dataclass
class STRIPSProblem:
    initial: State
    goal: FrozenSet[str]
    actions: List[Action]


@dataclass
class Plan:
    actions: List[Action]
    cost: float

    @property
    def length(self) -> int:
        return len(self.actions)

    def names(self) -> List[str]:
        return [a.name for a in self.actions]

    def to_dict(self) -> Dict[str, Any]:
        return {"actions": self.names(), "cost": round(self.cost, 3), "length": self.length}


# --------------------------------------------------------------------------- #
# A* planner
# --------------------------------------------------------------------------- #
class Planner:
    """A* over STRIPS states. Heuristic: number of unmet goal facts (relaxed)."""

    def __init__(self, *, max_expansions: int = 50_000) -> None:
        self.max_expansions = max_expansions

    @staticmethod
    def _heuristic(state: State, goal: FrozenSet[str]) -> int:
        return len(goal - state)

    def plan(self, problem: STRIPSProblem) -> Optional[Plan]:
        start = frozenset(problem.initial)
        goal = problem.goal
        if goal <= start:
            return Plan([], 0.0)

        counter = itertools.count()
        best_g: Dict[State, float] = {start: 0.0}
        heap: List[Tuple[float, float, int, State, Tuple[Action, ...]]] = [
            (self._heuristic(start, goal), 0.0, next(counter), start, ())]
        expansions = 0

        while heap and expansions < self.max_expansions:
            f, g, _, state, path = heapq.heappop(heap)
            if g > best_g.get(state, float("inf")):
                continue
            if goal <= state:
                return Plan(list(path), g)
            expansions += 1
            for action in problem.actions:
                if not action.applicable(state):
                    continue
                ns = action.apply(state)
                ng = g + action.cost
                if ng < best_g.get(ns, float("inf")):
                    best_g[ns] = ng
                    heapq.heappush(heap, (ng + self._heuristic(ns, goal), ng,
                                          next(counter), ns, path + (action,)))
        return None


# --------------------------------------------------------------------------- #
# Dependency graph (producer -> consumer over a plan)
# --------------------------------------------------------------------------- #
class DependencyGraph:
    """Causal dependencies among a plan's actions (producer satisfies consumer)."""

    def __init__(self, plan: Sequence[Action], initial: Optional[State] = None) -> None:
        self.actions = list(plan)
        self.initial = frozenset(initial or frozenset())
        # edges: consumer_index -> set(producer_index)
        self._deps: Dict[int, set] = {i: set() for i in range(len(self.actions))}
        self._build()

    def _build(self) -> None:
        for j, consumer in enumerate(self.actions):
            for fact in consumer.preconditions:
                if fact in self.initial:
                    continue  # already true at the start — no producer needed
                # the latest earlier action that adds this fact is the producer
                for i in range(j - 1, -1, -1):
                    if fact in self.actions[i].add_effects:
                        self._deps[j].add(i)
                        break

    def dependencies(self, index: int) -> set:
        return set(self._deps.get(index, set()))

    def topological_order(self) -> List[int]:
        order, placed = [], set()
        remaining = set(range(len(self.actions)))
        while remaining:
            ready = [i for i in sorted(remaining) if self._deps[i] <= placed]
            if not ready:
                # cycle (shouldn't happen for a valid sequential plan) — break by index
                ready = [min(remaining)]
            for i in ready:
                order.append(i)
                placed.add(i)
                remaining.discard(i)
        return order

    def layers(self) -> List[List[int]]:
        """Group actions into layers of mutually-independent actions (precedence-respecting)."""
        layer_of: Dict[int, int] = {}
        for i in self.topological_order():
            deps = self._deps[i]
            layer_of[i] = 1 + max((layer_of[d] for d in deps), default=-1)
        max_layer = max(layer_of.values(), default=-1)
        return [[i for i in range(len(self.actions)) if layer_of.get(i) == lvl]
                for lvl in range(max_layer + 1)]

    def to_dict(self) -> Dict[str, Any]:
        return {"edges": {self.actions[j].name: [self.actions[i].name for i in deps]
                          for j, deps in self._deps.items() if deps},
                "layers": [[self.actions[i].name for i in layer] for layer in self.layers()]}


# --------------------------------------------------------------------------- #
# Resource scheduler
# --------------------------------------------------------------------------- #
@dataclass
class ScheduleStep:
    actions: List[str]
    resource_use: Dict[str, float]


class ResourceScheduler:
    """Schedules a plan into time-steps honoring dependencies and resource budgets."""

    def __init__(self, capacities: Optional[Dict[str, float]] = None) -> None:
        self.capacities = capacities or {}

    def schedule(self, plan: Sequence[Action],
                 initial: Optional[State] = None) -> List[ScheduleStep]:
        dg = DependencyGraph(plan, initial)
        actions = list(plan)
        scheduled: set = set()
        steps: List[ScheduleStep] = []

        while len(scheduled) < len(actions):
            ready = [i for i in range(len(actions))
                     if i not in scheduled and dg.dependencies(i) <= scheduled]
            ready.sort()
            step_actions: List[int] = []
            use: Dict[str, float] = {}
            for i in ready:
                rmap = actions[i].resource_map()
                if all(use.get(r, 0.0) + amt <= self.capacities.get(r, float("inf"))
                       for r, amt in rmap.items()):
                    step_actions.append(i)
                    for r, amt in rmap.items():
                        use[r] = use.get(r, 0.0) + amt
            if not step_actions:
                # a single action exceeds capacity on its own — run it alone anyway
                step_actions = [ready[0]]
                use = actions[ready[0]].resource_map()
            for i in step_actions:
                scheduled.add(i)
            steps.append(ScheduleStep([actions[i].name for i in step_actions],
                                      {r: round(v, 3) for r, v in use.items()}))
        return steps


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA planner (STRIPS + A* + deps + resources) self-test")
    print("=" * 70)

    # a small task chain: get key -> open door -> enter
    actions = [
        Action.make("get_key", add=["have_key"], cost=1),
        Action.make("open_door", pre=["have_key"], add=["door_open"], cost=1),
        Action.make("enter", pre=["door_open"], add=["inside"], delete=["outside"], cost=1),
    ]
    problem = STRIPSProblem(initial=frozenset({"outside"}), goal=frozenset({"inside"}),
                            actions=actions)
    plan = Planner().plan(problem)
    print(f"\nplan                : {plan.names()} (cost {plan.cost})")
    assert plan is not None
    assert plan.names() == ["get_key", "open_door", "enter"]

    # verify the plan actually reaches the goal
    state = problem.initial
    for a in plan.actions:
        assert a.applicable(state)
        state = a.apply(state)
    assert problem.goal <= state
    print("plan validity       : reaches the goal ✓")

    # unreachable goal -> no plan
    impossible = STRIPSProblem(frozenset({"outside"}), frozenset({"on_the_moon"}), actions)
    assert Planner().plan(impossible) is None
    print("impossible goal     : no plan (correct)")

    # dependency graph: the chain is fully sequential
    dg = DependencyGraph(plan.actions, problem.initial)
    print(f"\ndependency layers   : {[[plan.actions[i].name for i in l] for l in dg.layers()]}")
    assert dg.dependencies(1) == {0}   # open_door needs get_key
    assert dg.dependencies(2) == {1}   # enter needs open_door
    assert len(dg.layers()) == 3       # fully sequential

    # parallelism: independent actions share a layer
    par_actions = [
        Action.make("scan_a", add=["a_done"], cost=1, resources={"compute": 3}),
        Action.make("scan_b", add=["b_done"], cost=1, resources={"compute": 3}),
        Action.make("report", pre=["a_done", "b_done"], add=["reported"], cost=1),
    ]
    par_problem = STRIPSProblem(frozenset(), frozenset({"reported"}), par_actions)
    par_plan = Planner().plan(par_problem)
    par_dg = DependencyGraph(par_plan.actions)
    print(f"\nparallel layers     : {par_dg.to_dict()['layers']}")
    # scan_a and scan_b are independent -> same layer; report follows
    assert any(len(layer) == 2 for layer in par_dg.layers())

    # resource scheduling: compute budget forces the two scans apart
    tight = ResourceScheduler({"compute": 4}).schedule(par_plan.actions)
    loose = ResourceScheduler({"compute": 8}).schedule(par_plan.actions)
    print(f"\nschedule (compute=4): {[s.actions for s in tight]}")
    print(f"schedule (compute=8): {[s.actions for s in loose]}")
    # with only 4 compute, the two 3-cost scans can't share a step
    assert all(len(s.actions) == 1 for s in tight[:2])
    # with 8 compute, both scans run together in step 1
    assert len(loose[0].actions) == 2

    print("\nALL SELF-TESTS PASSED ✓")
