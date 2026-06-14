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
import random
from dataclasses import dataclass
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
    "NeuralWorldModel",
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
# Neural forward model (Pillar B6) — a small MLP that generalises dynamics
# --------------------------------------------------------------------------- #
class _ForwardNet:
    """A tiny 1-hidden-layer MLP (tanh) mapping a state to (state_delta, reward).

    Pure-Python online SGD, with running input standardisation for stability. Per-action, so
    its input is just the state (no action encoding), mirroring the kNN's action partitioning.
    It also keeps a running error EMA and the training mean/spread, which let the world model
    report *honest* confidence — high near seen states, decaying out of distribution."""

    def __init__(self, in_dim: int, *, hidden: int = 12, lr: float = 0.05, seed: int = 0) -> None:
        self.in_dim = in_dim
        self.h = max(1, hidden)
        self.out = in_dim + 1
        self.lr = lr
        rng = random.Random(seed)
        s1 = 1.0 / math.sqrt(in_dim + 1)
        s2 = 1.0 / math.sqrt(self.h + 1)
        self.W1 = [[rng.uniform(-s1, s1) for _ in range(in_dim)] for _ in range(self.h)]
        self.b1 = [0.0] * self.h
        self.W2 = [[rng.uniform(-s2, s2) for _ in range(self.h)] for _ in range(self.out)]
        self.b2 = [0.0] * self.out
        self.n = 0
        self.mean = [0.0] * in_dim
        self._M2 = [0.0] * in_dim
        self.err_ema = 1.0
        self.samples = 0

    def _std(self) -> List[float]:
        return [math.sqrt(self._M2[i] / self.n) if self.n > 1 and self._M2[i] > 1e-12 else 1.0
                for i in range(self.in_dim)]

    def _standardize(self, x: Sequence[float]) -> List[float]:
        std = self._std()
        return [(x[i] - self.mean[i]) / std[i] for i in range(self.in_dim)]

    def _forward(self, z: Sequence[float]) -> Tuple[List[float], List[float]]:
        h = [math.tanh(sum(self.W1[j][i] * z[i] for i in range(self.in_dim)) + self.b1[j])
             for j in range(self.h)]
        out = [sum(self.W2[k][j] * h[j] for j in range(self.h)) + self.b2[k]
               for k in range(self.out)]
        return h, out

    def deviation(self, x: Sequence[float]) -> float:
        """Mean absolute standardised distance of ``x`` from the training centre (OOD signal)."""
        z = self._standardize(x)
        return sum(abs(v) for v in z) / max(1, self.in_dim)

    def predict(self, x: Sequence[float]) -> List[float]:
        _, out = self._forward(self._standardize(x))
        return out

    def train(self, x: Sequence[float], target: Sequence[float]) -> float:
        # Welford running mean/variance for input standardisation
        self.n += 1
        for i in range(self.in_dim):
            d = x[i] - self.mean[i]
            self.mean[i] += d / self.n
            self._M2[i] += d * (x[i] - self.mean[i])
        z = self._standardize(x)
        h, out = self._forward(z)
        d_out = [out[k] - target[k] for k in range(self.out)]
        mse = sum(e * e for e in d_out) / self.out
        self.err_ema = 0.97 * self.err_ema + 0.03 * mse
        self.samples += 1
        # hidden deltas use the CURRENT W2 (before the update)
        d_h = [sum(d_out[k] * self.W2[k][j] for k in range(self.out)) * (1.0 - h[j] * h[j])
               for j in range(self.h)]
        for k in range(self.out):
            for j in range(self.h):
                self.W2[k][j] -= self.lr * d_out[k] * h[j]
            self.b2[k] -= self.lr * d_out[k]
        for j in range(self.h):
            for i in range(self.in_dim):
                self.W1[j][i] -= self.lr * d_h[j] * z[i]
            self.b1[j] -= self.lr * d_h[j]
        return mse


class NeuralWorldModel(WorldModel):
    """A neural drop-in for :class:`WorldModel`: a per-action MLP learns (state → Δstate, reward).

    Same surface as the kNN model — ``observe`` then ``predict`` — so it inherits ``rollout`` /
    ``counterfactual`` / ``intervene`` unchanged and plugs straight into planning. Unlike the kNN
    (which only interpolates among stored points), the MLP *generalises* the dynamics, while
    confidence stays honest: it climbs with experience and low error, and decays out of the
    region it has seen. Numeric states only; a symbolic state yields a zero-confidence no-op."""

    def __init__(self, *, hidden: int = 12, lr: float = 0.05, epochs: int = 4,
                 seed: int = 0, ood_tolerance: float = 2.5,
                 max_transitions: int = 50_000) -> None:
        self.hidden = max(1, hidden)
        self.lr = lr
        self.epochs = max(1, epochs)
        self.seed = seed
        self.ood_tolerance = ood_tolerance
        self.max_transitions = max_transitions
        self._nets: Dict[Action, _ForwardNet] = {}
        self._state_dim: Optional[int] = None
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def actions(self) -> List[Action]:
        return list(self._nets.keys())

    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def observe(self, state: Sequence[Any], action: Action,
                next_state: Sequence[Any], reward: float = 0.0) -> None:
        state = tuple(state)
        next_state = tuple(next_state)
        if not self._numeric(state) or not self._numeric(next_state):
            return                               # neural dynamics are for numeric states
        if self._state_dim is None:
            self._state_dim = len(state)
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            return
        net = self._nets.get(action)
        if net is None:
            net = _ForwardNet(self._state_dim, hidden=self.hidden, lr=self.lr,
                              seed=self.seed + len(self._nets))
            self._nets[action] = net
        target = [next_state[i] - state[i] for i in range(self._state_dim)] + [float(reward)]
        for _ in range(self.epochs):
            net.train(list(state), target)
        self._count += 1

    def predict(self, state: Sequence[Any], action: Action) -> Prediction:
        state = tuple(state)
        net = self._nets.get(action)
        if net is None or self._state_dim != len(state) or not self._numeric(state):
            return Prediction(next_state=state, reward=0.0, confidence=0.0, neighbors=0)
        out = net.predict(list(state))
        delta, reward = out[: self._state_dim], out[self._state_dim]
        next_state: State = tuple(state[i] + delta[i] for i in range(self._state_dim))
        # honest confidence: grows with experience + low error, decays out of distribution
        experience = min(1.0, net.samples / 20.0)
        fit = 1.0 / (1.0 + net.err_ema)
        ood = math.exp(-max(0.0, net.deviation(state) - self.ood_tolerance))
        conf = max(0.0, min(1.0, experience * fit * ood))
        return Prediction(next_state=next_state, reward=reward,
                          confidence=conf, neighbors=net.samples)

    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count, "backend": "neural-mlp", "hidden": self.hidden,
                "actions": {str(a): n.samples for a, n in self._nets.items()}}


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
