"""NYXARA · njp/agency.py — Stage G: goal → plan → action → outcome (🎯).

:mod:`nyxara.njp.goals` already keeps goals: a tree, progress, completion, abandonment. That is
bookkeeping about intentions, and it is not agency. Agency is the loop that *closes*::

    GOAL → PLAN → ACTION → OUTCOME → prediction error → better model → better plan → ↺

**Planning here is search over a learned model, not over a written one.** The planner asks
:class:`~nyxara.njp.predictive.PredictiveWorldModel` what each action leads to and walks forward,
so the plans she can make are exactly as good as what she has actually learned about the world —
no better, and no worse. That is the property that makes this Stage *G* rather than a scripted
policy: nobody writes the transitions, so nobody writes the plans. A brain that has learned
nothing produces no plan and says so, which is the correct behaviour and the one a hand-written
policy can never exhibit.

**Best-first over expected probability, depth-capped.** Each frontier node carries the product of
the transition probabilities that reached it — the honest likelihood the whole sequence works —
and expansion always takes the most promising node. Depth is capped because a learned model
compounds its own error: three steps out, a plan is a product of three estimates, and past that
the number stops meaning anything. Reporting :attr:`Plan.confidence` alongside the steps is what
keeps a long speculative plan from being read as a short reliable one.

**Acting is optional and outcomes are always scored.** :meth:`Agent.act` takes an *actuator* — a
callable that really does the thing — and falls back to reporting that it cannot act when none is
supplied. Either way the outcome is compared with what the plan predicted, and the difference goes
straight back into the world model. An agent that never checks whether its actions worked is not
an agent; it is a random number generator with ambitions.

**Credit is assigned to the action, not to the plan.** :class:`ActionValue` keeps, per action, how
often it actually produced the transition the model expected. A plan that succeeded because of one
good step and one lucky one should not promote both, and a Beta-style success count over
individual actions is the smallest honest way to tell them apart.

**Safety is not this module's to decide.** Every plan it produces is a *proposal*. Nothing here
executes anything the caller has not handed it an actuator for, and the kernel's gate remains the
only thing that decides whether a proposed action may run. The mind proposes; the kernel disposes.

Pure standard library, no LLM.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from nyxara.njp.predictive import PredictiveWorldModel, WorldState

__all__ = ["Step", "Plan", "Outcome", "ActionValue", "Agent"]

_MAX_DEPTH = 4
_MAX_EXPANSIONS = 200


@dataclass
class Step:
    """One action, what it was expected to lead to, and how likely that was."""

    action: str = ""
    expected: str = ""
    probability: float = 0.0
    order: int = -1              # backoff order the expectation came from

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "expected": self.expected[:80],
                "probability": round(self.probability, 4), "order": self.order}


@dataclass
class Plan:
    """A route from here to the goal, through the world as she has learned it."""

    goal: str = ""
    steps: List[Step] = dc_field(default_factory=list)
    confidence: float = 0.0      # product of the step probabilities
    reason: str = ""
    expansions: int = 0
    ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.steps)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def first(self) -> Optional[Step]:
        return self.steps[0] if self.steps else None

    @property
    def speculative(self) -> bool:
        """A long plan on a learned model compounds its own error. Say so rather than imply it."""
        return self.depth > 2 or self.confidence < 0.25

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal[:80], "found": self.found, "depth": self.depth,
                "confidence": round(self.confidence, 4), "speculative": self.speculative,
                "steps": [s.to_dict() for s in self.steps], "reason": self.reason[:160],
                "expansions": self.expansions, "ms": round(self.ms, 3)}


@dataclass
class Outcome:
    """What actually happened when she acted, against what the plan said would happen."""

    action: str = ""
    expected: str = ""
    actual: str = ""
    surprise_bits: float = 0.0
    excess_bits: float = 0.0
    reached_goal: bool = False
    acted: bool = False
    reason: str = ""

    @property
    def as_expected(self) -> bool:
        return bool(self.actual) and self.actual == self.expected

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "expected": self.expected[:80],
                "actual": self.actual[:80], "as_expected": self.as_expected,
                "reached_goal": self.reached_goal, "acted": self.acted,
                "surprise_bits": round(self.surprise_bits, 4),
                "excess_bits": round(self.excess_bits, 4), "reason": self.reason[:120]}


@dataclass
class ActionValue:
    """How often an action actually did what the model expected of it."""

    action: str = ""
    tried: int = 0
    worked: int = 0

    @property
    def reliability(self) -> float:
        """Laplace-smoothed, so one lucky success is not a perfect action."""
        return (self.worked + 1.0) / (self.tried + 2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "tried": self.tried, "worked": self.worked,
                "reliability": round(self.reliability, 4)}


class Agent:
    """Plans over the learned world model, acts when given hands, and learns from what happened."""

    def __init__(self, model: PredictiveWorldModel, *, max_depth: int = _MAX_DEPTH,
                 max_expansions: int = _MAX_EXPANSIONS, min_probability: float = 0.02) -> None:
        self.model = model
        self.max_depth = max(1, int(max_depth))
        self.max_expansions = max(4, int(max_expansions))
        self.min_probability = float(min_probability)

        self.values: Dict[str, ActionValue] = {}
        self.planned = 0
        self.plans_found = 0
        self.acted = 0
        self.goals_reached = 0
        self.history: List[Outcome] = []

    # ---- what she can even try ---------------------------------------------- #
    def actions(self) -> List[str]:
        """Every action the world model has ever seen. She cannot plan with anything else.

        Deliberately not a configured list. An agent whose action set is written down can plan
        for actions it has never observed the consequences of, and every such plan is fiction.
        """
        seen = {action for (_order, _ctx, action) in self.model._counts if action}
        return sorted(seen)

    # ---- planning ------------------------------------------------------------ #
    def plan(self, goal: Any, *, start: Any = None,
             actions: Optional[Sequence[str]] = None) -> Plan:
        """Best-first search over the learned transitions for a route to ``goal``.

        The frontier is ordered by the accumulated probability of the whole path, so the most
        *likely to actually work* partial plan is always expanded next — not the shortest, and
        not the one that looks most direct. On a learned model those are different things.
        """
        out = Plan(goal=self.model._sig(goal))
        t0 = time.perf_counter()
        try:
            target = out.goal
            if not target:
                out.reason = "no goal given"
                return out
            candidates = list(actions) if actions is not None else self.actions()
            if not candidates:
                out.reason = "no action has ever been observed, so nothing can be planned"
                return out

            history = list(self.model._history) if start is None else \
                self.model._as_history(start)
            # (−log probability, tiebreak, history, steps) — negative log so heapq's min-first
            # ordering pops the *most* likely path, and so long paths do not underflow to zero.
            frontier: List[Tuple[float, int, List[str], List[Step]]] = [(0.0, 0, history, [])]
            tiebreak = 1
            visited: set = set()

            while frontier and out.expansions < self.max_expansions:
                cost, _, path, steps = heapq.heappop(frontier)
                out.expansions += 1
                if len(steps) >= self.max_depth:
                    continue
                signature = ("→".join(path[-self.model.max_order:]), len(steps))
                if signature in visited:
                    continue
                visited.add(signature)

                for action in candidates:
                    # Observed successors only. `predict` smooths, and a planner walking smoothed
                    # mass will route through edges that exist only because nothing is allowed to
                    # be impossible — which is right for surprise and fiction for a plan.
                    successors = self.model.successors(path, action)
                    if not successors:
                        continue
                    order = self.model.predict(path, action).order
                    for outcome, probability in successors.items():
                        if probability < self.min_probability:
                            continue
                        step = Step(action=action, expected=outcome, probability=probability,
                                    order=order)
                        # Reliability shapes the *search*, never the reported probability: an
                        # action that keeps not working should be tried later, but the model's
                        # estimate of what it does is the model's to state, not the planner's.
                        reliability = self.values.get(action, ActionValue(action)).reliability
                        step_cost = cost - math.log(max(probability * reliability, 1e-9))
                        if outcome == target:
                            out.steps = steps + [step]
                            out.confidence = math.exp(-(cost - math.log(max(probability, 1e-9))))
                            out.reason = "reached the goal state"
                            self.plans_found += 1
                            return out
                        tiebreak += 1
                        heapq.heappush(frontier, (step_cost, tiebreak, path + [outcome],
                                                  steps + [step]))
            out.reason = ("no route found within depth "
                          f"{self.max_depth} after {out.expansions} expansions")
            return out
        except Exception:  # noqa: BLE001 — a failed search finds no plan and never raises
            out.reason = "planning failed"
            return out
        finally:
            self.planned += 1
            out.ms = (time.perf_counter() - t0) * 1000.0

    # ---- acting -------------------------------------------------------------- #
    def act(self, plan: Plan, *, actuator: Optional[Callable[[str], Any]] = None) -> Outcome:
        """Take the plan's first step, see what really happened, and learn from the gap.

        One step, not the whole plan. Beyond the first action the plan rests on predictions that
        have not been tested yet, and executing them unchecked is how a model's error compounds
        into the world instead of into a number. Re-plan after each step — that is what makes
        this a loop rather than a script.
        """
        out = Outcome()
        step = plan.first
        if step is None:
            out.reason = "no plan to act on"
            return out
        out.action, out.expected = step.action, step.expected

        if actuator is None:
            out.reason = "no actuator supplied; the plan is a proposal only"
            return out
        try:
            observed = actuator(step.action)
        except Exception as exc:  # noqa: BLE001 — a failed action is an outcome, not a crash
            out.reason = f"action raised: {type(exc).__name__}"
            self._credit(step.action, worked=False)
            return out

        out.acted = True
        self.acted += 1
        out.actual = self.model._sig(observed)

        prediction = self.model.predict(action=step.action)
        if prediction.answerable:
            surprise = self.model.score(prediction, observed)
            out.surprise_bits = surprise.bits
            out.excess_bits = surprise.excess
        self.model.observe(observed, step.action)

        out.reached_goal = out.actual == plan.goal
        if out.reached_goal:
            self.goals_reached += 1
        self._credit(step.action, worked=out.as_expected)
        self.history.append(out)
        del self.history[:-256]
        return out

    def pursue(self, goal: Any, *, actuator: Optional[Callable[[str], Any]] = None,
               steps: int = 4) -> List[Outcome]:
        """The whole loop: plan, act, look, re-plan, until the goal or until out of steps.

        Re-planning every step is the point. The first action's real outcome is new evidence, and
        a plan made before it is a plan made with less information than she now has.
        """
        out: List[Outcome] = []
        for _ in range(max(1, int(steps))):
            plan = self.plan(goal)
            if not plan.found:
                break
            outcome = self.act(plan, actuator=actuator)
            out.append(outcome)
            if not outcome.acted or outcome.reached_goal:
                break
        return out

    def _credit(self, action: str, *, worked: bool) -> None:
        value = self.values.setdefault(action, ActionValue(action=action))
        value.tried += 1
        if worked:
            value.worked += 1

    # ---- reporting ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        ranked = sorted(self.values.values(), key=lambda v: -v.reliability)
        return {
            "known_actions": len(self.actions()),
            "planned": self.planned,
            "plans_found": self.plans_found,
            "plan_rate": round(self.plans_found / self.planned, 4) if self.planned else None,
            "acted": self.acted,
            "goals_reached": self.goals_reached,
            "success_rate": round(self.goals_reached / self.acted, 4) if self.acted else None,
            "as_expected_rate": (round(sum(1 for o in self.history if o.as_expected)
                                       / len(self.history), 4) if self.history else None),
            "best_action": ranked[0].to_dict() if ranked else None,
            "worst_action": ranked[-1].to_dict() if len(ranked) > 1 else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"values": [v.to_dict() | {"tried": v.tried, "worked": v.worked}
                           for v in self.values.values()],
                "planned": self.planned, "plans_found": self.plans_found,
                "acted": self.acted, "goals_reached": self.goals_reached}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.values = {}
            for row in (d.get("values") or []):
                action = str(row.get("action", ""))
                if action:
                    self.values[action] = ActionValue(action=action,
                                                      tried=int(row.get("tried", 0)),
                                                      worked=int(row.get("worked", 0)))
            self.planned = int(d.get("planned", 0))
            self.plans_found = int(d.get("plans_found", 0))
            self.acted = int(d.get("acted", 0))
            self.goals_reached = int(d.get("goals_reached", 0))
        except Exception:  # noqa: BLE001
            pass
