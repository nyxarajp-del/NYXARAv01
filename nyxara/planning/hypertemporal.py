"""NYXARA · planning/hypertemporal.py — hyper-temporal counterfactual planning (🔮, Rule 4, F16).

NYXARA treats the future not as one line but as a **branching probability tree**, and chooses the
present action whose *simulated futures* are best. The value flows **backward** — from the leaves of the
imagined tree to the action at the root (expectimax / Bellman backup) — which is ordinary
planning-backup, **not** literal retrocausality: nothing travels back in time, a *simulation's* value
simply informs a *present* decision.

Two engines, both honestly bounded:

* **Branching backward induction** (:meth:`BranchingPlanner.plan`) — exact expectimax over the tree to a
  horizon, memoised and **capped by a node budget**; if the tree would exceed the cap she falls back to
  sampling rather than pretend to have searched it all.
* **Bounded parallel rollouts** (:meth:`BranchingPlanner.parallel_rollouts`) — the "thousands of
  asynchronous mind-splits across timelines", made honest: a **fixed** number of Monte-Carlo scenario
  rollouts per root action, whose returns are averaged and merged on the best. The report always states
  the cap — there is no infinite-timeline claim.

Reuses the stochastic-model shape of :mod:`nyxara.sim.montecarlo` and the decision spirit of
:mod:`nyxara.planning.voi`. Pure standard library; **no LLM**; it plans, touching no gate or value.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Transition", "Model", "ActionsFn", "PlanResult", "BranchingPlanner"]

# a stochastic model: (state, action) -> [(prob, next_state, reward), ...]
Transition = Tuple[float, Any, float]
Model = Callable[[Any, Any], Sequence[Transition]]
ActionsFn = Callable[[Any], Sequence[Any]]


@dataclass
class PlanResult:
    best_action: Any
    value: float
    method: str                  # "backward-induction" | "parallel-rollouts"
    nodes: int = 0
    rollouts: int = 0
    rollout_cap: int = 0
    action_values: Dict[Any, float] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        return {"best_action": self.best_action, "value": round(self.value, 4),
                "method": self.method, "nodes": self.nodes, "rollouts": self.rollouts,
                "rollout_cap": self.rollout_cap,
                "action_values": {str(k): round(v, 4) for k, v in (self.action_values or {}).items()}}


class BranchingPlanner:
    """Choose the present action whose branching futures are best — bounded, honest."""

    def __init__(self, *, gamma: float = 0.95, max_nodes: int = 50000) -> None:
        self.gamma = gamma
        self.max_nodes = max_nodes

    # ---- exact-ish expectimax with a node budget ---- #
    def plan(self, state: Any, actions_fn: ActionsFn, model: Model, *, horizon: int) -> PlanResult:
        memo: Dict[Tuple[Any, int], float] = {}
        nodes = [0]
        overflow = [False]

        def value(s: Any, d: int) -> float:
            if d <= 0:
                return 0.0
            acts = list(actions_fn(s))
            if not acts:                       # terminal — no further value
                return 0.0
            key = (s, d)
            if key in memo:
                return memo[key]
            nodes[0] += 1
            if nodes[0] > self.max_nodes:
                overflow[0] = True
                return 0.0
            best = -float("inf")
            for a in acts:
                q = sum(p * (r + self.gamma * value(ns, d - 1)) for p, ns, r in model(s, a))
                best = max(best, q)
            memo[key] = best
            return best

        root_actions = list(actions_fn(state))
        if not root_actions:
            return PlanResult(None, 0.0, "backward-induction", nodes[0], action_values={})
        qs: Dict[Any, float] = {}
        for a in root_actions:
            qs[a] = sum(p * (r + self.gamma * value(ns, horizon - 1)) for p, ns, r in model(state, a))
        # if the tree overflowed the budget, be honest and fall back to sampling
        if overflow[0]:
            return self.parallel_rollouts(state, actions_fn, model, horizon=horizon, rollouts=200)
        best = max(root_actions, key=lambda a: qs[a])
        return PlanResult(best, qs[best], "backward-induction", nodes[0], action_values=qs)

    # ---- bounded Monte-Carlo scenario rollouts (the honest "many timelines") ---- #
    def parallel_rollouts(self, state: Any, actions_fn: ActionsFn, model: Model, *,
                          horizon: int, rollouts: int = 200,
                          rng: Optional[random.Random] = None) -> PlanResult:
        rng = rng or random.Random(0)
        root_actions = list(actions_fn(state))
        if not root_actions:
            return PlanResult(None, 0.0, "parallel-rollouts", rollouts=0,
                              rollout_cap=rollouts, action_values={})

        def sample_next(s: Any, a: Any) -> Optional[Tuple[Any, float]]:
            trans = list(model(s, a))
            if not trans:
                return None
            r = rng.random()
            cum = 0.0
            for p, ns, rew in trans:
                cum += p
                if r <= cum:
                    return ns, rew
            return trans[-1][1], trans[-1][2]

        def rollout(first_action: Any) -> float:
            total, disc, s, a = 0.0, 1.0, state, first_action
            for _ in range(horizon):
                nxt = sample_next(s, a)
                if nxt is None:
                    break
                s, rew = nxt
                total += disc * rew
                disc *= self.gamma
                acts = list(actions_fn(s))
                if not acts:
                    break
                a = rng.choice(acts)
            return total

        qs: Dict[Any, float] = {}
        for a in root_actions:
            qs[a] = sum(rollout(a) for _ in range(rollouts)) / rollouts
        best = max(root_actions, key=lambda a: qs[a])
        return PlanResult(best, qs[best], "parallel-rollouts", rollouts=rollouts * len(root_actions),
                          rollout_cap=rollouts, action_values=qs)
