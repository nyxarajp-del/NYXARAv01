"""NYXARA · mind/world_model.py — learned dynamics + counterfactual rollouts (★).

To plan, NYXARA must be able to *imagine*: "if I do this, what happens next?" The
world model learns the environment's dynamics from experience — tuples of
``(state, action, next_state, reward)`` — and then answers questions the agent never
actually tried:

* **predict** ``(state, action) → next_state, reward`` with a **confidence** that
  honestly drops as the query leaves the region it has seen (no hallucinated
  certainty about the unknown);
* **rollout** a policy forward N steps into an imagined trajectory;
* **counterfactual** — compare two policies from the same start and find where, and
  by how much, the futures diverge;
* **intervene** — surgically force a different action at one step and watch the
  consequences ("what if I had turned left?").

The learner is a locally-weighted **k-nearest-neighbour** model over transition
*deltas* (so it generalises smoothly for continuous states and matches exactly for
discrete ones). No training loop, no heavy deps — learning is just remembering.

Pure standard library. Pairs with :mod:`mind.predictive_core` (the per-step
prediction-error loop) and feeds :mod:`planning`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union,
)

__all__ = [
    "State",
    "Transition",
    "Prediction",
    "Trajectory",
    "CounterfactualResult",
    "WorldModel",
]

State = Tuple[Any, ...]
Action = Hashable
Policy = Union[Sequence[Action], Callable[[State], Action]]


def _dist(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Euclidean over numeric dims; 0/1 mismatch over symbolic dims."""
    total = 0.0
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            total += (x - y) ** 2
        else:
            total += 0.0 if x == y else 1.0
    return math.sqrt(total)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    state: State
    action: Action
    next_state: State
    reward: float = 0.0


@dataclass
class Prediction:
    next_state: State
    reward: float
    confidence: float          # [0,1] — how much the model trusts this prediction
    neighbors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"next_state": self.next_state, "reward": round(self.reward, 4),
                "confidence": round(self.confidence, 4), "neighbors": self.neighbors}


@dataclass
class Trajectory:
    states: List[State]
    actions: List[Action]
    rewards: List[float]
    confidences: List[float]

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def final_state(self) -> Optional[State]:
        return self.states[-1] if self.states else None

    @property
    def length(self) -> int:
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {"states": self.states, "actions": self.actions,
                "total_reward": round(self.total_reward, 4),
                "mean_confidence": round(self.mean_confidence, 4),
                "length": self.length}


@dataclass
class CounterfactualResult:
    trajectory_a: Trajectory
    trajectory_b: Trajectory
    divergence_step: Optional[int]   # first step where the futures differ (None if identical)
    reward_difference: float         # A − B

    def better(self) -> str:
        if self.reward_difference > 0:
            return "a"
        if self.reward_difference < 0:
            return "b"
        return "tie"

    def to_dict(self) -> Dict[str, Any]:
        return {"divergence_step": self.divergence_step,
                "reward_difference": round(self.reward_difference, 4),
                "better": self.better(),
                "a_reward": round(self.trajectory_a.total_reward, 4),
                "b_reward": round(self.trajectory_b.total_reward, 4)}


# --------------------------------------------------------------------------- #
# World model
# --------------------------------------------------------------------------- #
class WorldModel:
    """Locally-weighted kNN model of environment dynamics with honest uncertainty."""

    def __init__(self, *, k: int = 3, distance_scale: float = 1.0,
                 max_transitions: int = 50_000) -> None:
        self.k = max(1, k)
        self.distance_scale = distance_scale
        self.max_transitions = max_transitions
        # transitions partitioned by action for fast, action-conditioned lookup
        self._by_action: Dict[Action, List[Transition]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    # ---- learning = remembering ---- #
    def observe(self, state: Sequence[Any], action: Action,
                next_state: Sequence[Any], reward: float = 0.0) -> None:
        t = Transition(tuple(state), action, tuple(next_state), reward)
        bucket = self._by_action.setdefault(action, [])
        bucket.append(t)
        self._count += 1
        if self._count > self.max_transitions:           # forget the oldest
            oldest_action = max(self._by_action, key=lambda a: len(self._by_action[a]))
            self._by_action[oldest_action].pop(0)
            self._count -= 1

    def observe_many(self, transitions: Sequence[Transition]) -> None:
        for t in transitions:
            self.observe(t.state, t.action, t.next_state, t.reward)

    def actions(self) -> List[Action]:
        return [a for a, ts in self._by_action.items() if ts]

    # ---- prediction ---- #
    def predict(self, state: Sequence[Any], action: Action) -> Prediction:
        state = tuple(state)
        bucket = self._by_action.get(action, [])
        if not bucket:
            # never tried this action — assume no-op, zero confidence
            return Prediction(next_state=state, reward=0.0, confidence=0.0, neighbors=0)

        scored = sorted(((_dist(state, t.state), t) for t in bucket), key=lambda x: x[0])
        nn = scored[: self.k]
        eps = 1e-9
        weights = [1.0 / (d + eps) for d, _ in nn]
        wsum = sum(weights) or 1.0

        # weighted average of transition *deltas* (generalises, exact on a hit)
        dim = len(state)
        delta = [0.0] * dim
        reward = 0.0
        all_numeric = all(isinstance(state[i], (int, float)) and not isinstance(state[i], bool)
                          for i in range(dim))
        for w, (d, t) in zip(weights, nn):
            if all_numeric and len(t.state) == dim and len(t.next_state) == dim:
                for i in range(dim):
                    delta[i] += w * (t.next_state[i] - t.state[i])
            reward += w * t.reward
        reward /= wsum

        if all_numeric:
            next_state: State = tuple(state[i] + delta[i] / wsum for i in range(dim))
        else:
            # symbolic: take the nearest neighbour's outcome
            next_state = nn[0][1].next_state

        nearest = nn[0][0]
        confidence = math.exp(-nearest / max(1e-6, self.distance_scale))
        return Prediction(next_state=next_state, reward=reward,
                          confidence=min(1.0, confidence), neighbors=len(nn))

    def coverage(self, state: Sequence[Any], action: Action) -> float:
        """How well the model knows this (state, action) — its prediction confidence."""
        return self.predict(state, action).confidence

    # ---- imagination: rollouts ---- #
    @staticmethod
    def _action_at(policy: Policy, state: State, t: int) -> Optional[Action]:
        if callable(policy):
            return policy(state)
        return policy[t] if t < len(policy) else None

    def rollout(self, start: Sequence[Any], policy: Policy, *, steps: int = 10,
                reward_fn: Optional[Callable[[State, Action, State], float]] = None,
                terminal_fn: Optional[Callable[[State], bool]] = None) -> Trajectory:
        """Imagine forward: simulate ``policy`` from ``start`` for up to ``steps``."""
        cur: State = tuple(start)
        states: List[State] = [cur]
        actions: List[Action] = []
        rewards: List[float] = []
        confidences: List[float] = []
        for t in range(steps):
            action = self._action_at(policy, cur, t)
            if action is None:
                break
            pred = self.predict(cur, action)
            reward = (reward_fn(cur, action, pred.next_state)
                      if reward_fn is not None else pred.reward)
            actions.append(action)
            rewards.append(reward)
            confidences.append(pred.confidence)
            states.append(pred.next_state)
            cur = pred.next_state
            if terminal_fn is not None and terminal_fn(cur):
                break
        return Trajectory(states, actions, rewards, confidences)

    def imagine(self, start: Sequence[Any], action: Action, *, steps: int = 5,
                **kw: Any) -> Trajectory:
        """Convenience: roll out a single repeated action."""
        return self.rollout(start, [action] * steps, steps=steps, **kw)

    # ---- counterfactuals ---- #
    def counterfactual(self, start: Sequence[Any], policy_a: Policy, policy_b: Policy,
                       *, steps: int = 10, eps: float = 1e-6, **kw: Any) -> CounterfactualResult:
        """Compare two policies from the same start; locate where futures diverge."""
        ta = self.rollout(start, policy_a, steps=steps, **kw)
        tb = self.rollout(start, policy_b, steps=steps, **kw)
        divergence: Optional[int] = None
        for i in range(min(len(ta.states), len(tb.states))):
            if _dist(ta.states[i], tb.states[i]) > eps:
                divergence = i
                break
        return CounterfactualResult(ta, tb, divergence,
                                    ta.total_reward - tb.total_reward)

    def intervene(self, start: Sequence[Any], policy: Policy, *, at_step: int,
                  action: Action, steps: int = 10, **kw: Any) -> Trajectory:
        """Roll out ``policy`` but force ``action`` at ``at_step`` (a do-intervention)."""
        counter = {"t": 0}

        def pol(state: State) -> Action:
            t = counter["t"]
            counter["t"] += 1
            return action if t == at_step else self._action_at(policy, state, t)

        return self.rollout(start, pol, steps=steps, **kw)

    # ---- introspection ---- #
    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count,
                "actions": {str(a): len(ts) for a, ts in self._by_action.items()},
                "k": self.k}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA world-model self-test")
    print("=" * 70)

    # A tiny 1-D world: position x; actions move it; reward = -|x| (home is 0).
    def true_step(x: float, a: str) -> float:
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    wm = WorldModel(k=3, distance_scale=2.0)
    # learn dynamics from random experience
    import random
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10, 10)
        a = rng.choice(["left", "right", "stay"])
        nx = true_step(x, a)
        wm.observe((x,), a, (nx,), reward=-abs(nx))

    # PREDICT: learned dynamics match the truth, with high confidence near data
    p = wm.predict((4.0,), "left")
    print(f"\npredict (4,left)    : next={tuple(round(v,2) for v in p.next_state)} "
          f"conf={p.confidence:.2f}")
    assert abs(p.next_state[0] - 3.0) < 0.3

    # CONFIDENCE drops far outside the training range
    far = wm.predict((1000.0,), "left")
    print(f"confidence near/far : {p.confidence:.2f} / {far.confidence:.3f}")
    assert far.confidence < p.confidence

    # ROLLOUT a greedy 'go-home' policy -> position should approach 0
    def go_home(state):
        x = state[0]
        return "left" if x > 0.5 else ("right" if x < -0.5 else "stay")

    traj = wm.rollout((8.0,), go_home, steps=12)
    print(f"\nrollout from x=8    : final x={traj.final_state[0]:.2f} "
          f"total_reward={traj.total_reward:.1f} mean_conf={traj.mean_confidence:.2f}")
    assert abs(traj.final_state[0]) < 1.5   # it imagined its way home

    # COUNTERFACTUAL: from x=5, going right is worse than going left (reward=-|x|)
    cf = wm.counterfactual((5.0,), ["left"] * 6, ["right"] * 6, steps=6)
    print(f"\ncounterfactual      : {cf.to_dict()}")
    assert cf.better() == "a"               # left (toward home) earns more reward
    assert cf.divergence_step == 1          # diverge immediately

    # INTERVENE: force one 'right' at step 2 of a go-home plan and see the detour
    base = wm.rollout((4.0,), ["left"] * 5, steps=5)
    iv = wm.intervene((4.0,), ["left"] * 5, at_step=2, action="right", steps=5)
    print(f"\nintervention        : base final={base.final_state[0]:.1f} "
          f"intervened final={iv.final_state[0]:.1f}")
    assert iv.final_state[0] > base.final_state[0]   # the forced 'right' set it back

    # UNKNOWN action -> zero confidence, no-op
    unk = wm.predict((0.0,), "teleport")
    assert unk.confidence == 0.0 and unk.next_state == (0.0,)
    print(f"\nunknown action      : conf={unk.confidence} (honest about ignorance)")

    print(f"\nstats               : {wm.stats()}")
    print("\nALL SELF-TESTS PASSED ✓")
