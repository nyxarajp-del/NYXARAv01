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

Three learners share one surface (``observe`` then ``predict``), so all of them inherit
``rollout`` / ``counterfactual`` / ``intervene`` and plug straight into planning:

* :class:`WorldModel` — a locally-weighted **k-nearest-neighbour** model over transition
  *deltas*. Pure stdlib, zero deps; learning is just remembering. The always-available floor.
* :class:`NeuralWorldModel` — a tiny per-action pure-Python MLP (online SGD) that *generalises*
  the dynamics a little. Still stdlib-only.
* :class:`EnsembleWorldModel` — a real learned model: a **deep ensemble** of numpy MLPs trained
  with **Adam** on **mini-batches sampled from a replay buffer** (bootstrapped per member). It
  genuinely generalises, and — crucially — its **epistemic uncertainty is the ensemble's
  disagreement**: where the members agree it is confident; where they diverge (novel states) it
  honestly is not. Requires numpy; :func:`build_world_model` falls back automatically when it is
  absent, so the system never hard-depends on it.

Pairs with :mod:`mind.predictive_core` (the per-step prediction-error loop) and feeds
:mod:`planning`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple, Union,
)

try:                                       # numpy powers the deep-ensemble model; optional
    import numpy as np
    _HAS_NUMPY = True
except Exception:                          # noqa: BLE001 — degrade to the stdlib learners
    np = None  # type: ignore
    _HAS_NUMPY = False

__all__ = [
    "State",
    "Transition",
    "Prediction",
    "Trajectory",
    "CounterfactualResult",
    "WorldModel",
    "NeuralWorldModel",
    "EnsembleWorldModel",
    "build_world_model",
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
    epistemic: float = 0.0     # model-disagreement (ensemble) — what it does NOT know

    def to_dict(self) -> Dict[str, Any]:
        return {"next_state": self.next_state, "reward": round(self.reward, 4),
                "confidence": round(self.confidence, 4), "neighbors": self.neighbors,
                "epistemic": round(self.epistemic, 4)}


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
# Deep-ensemble forward model — real learned dynamics with epistemic uncertainty
# --------------------------------------------------------------------------- #
class _MLP:
    """A small numpy MLP ``state → (Δstate, reward)`` with tanh hiddens and an Adam optimiser.

    All values are kept normalised by the owning :class:`_ActionModel`; this net just fits the
    standardised mapping. One member of a deep ensemble — different seed + a bootstrapped view
    of the replay buffer give the members the diversity that makes their *disagreement* a real
    epistemic-uncertainty signal."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int],
                 lr: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.dims = [in_dim, *hidden, out_dim]
        self.W: List[Any] = []
        self.b: List[Any] = []
        for i in range(len(self.dims) - 1):
            fan_in = self.dims[i]
            # He-ish init (works well with tanh too at this scale)
            self.W.append(rng.standard_normal((self.dims[i], self.dims[i + 1]))
                          * math.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(self.dims[i + 1]))
        self.lr = lr
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, X: Any) -> Tuple[Any, List[Any]]:
        acts = [X]
        a = X
        for i in range(len(self.W) - 1):
            a = np.tanh(a @ self.W[i] + self.b[i])
            acts.append(a)
        out = a @ self.W[-1] + self.b[-1]              # linear output head
        acts.append(out)
        return out, acts

    def train_batch(self, X: Any, Y: Any) -> float:
        n = X.shape[0]
        out, acts = self.forward(X)
        diff = out - Y
        loss = float(np.mean(diff ** 2))
        gW: List[Any] = [None] * len(self.W)
        gb: List[Any] = [None] * len(self.b)
        delta = (2.0 / n) * diff                       # dL/d(out), linear head
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:                                  # back-prop through tanh activation
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        self._adam(gW, gb)
        return loss

    def _adam(self, gW: List[Any], gb: List[Any],
              b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        bc1 = 1.0 - b1 ** self._t
        bc2 = 1.0 - b2 ** self._t
        for i in range(len(self.W)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * gW[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * (gW[i] ** 2)
            self.W[i] -= self.lr * (self._mW[i] / bc1) / (np.sqrt(self._vW[i] / bc2) + eps)
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * gb[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * (gb[i] ** 2)
            self.b[i] -= self.lr * (self._mb[i] / bc1) / (np.sqrt(self._vb[i] / bc2) + eps)


class _ActionModel:
    """A deep ensemble + replay buffer for ONE action; learns standardised dynamics."""

    def __init__(self, in_dim: int, *, ensemble: int, hidden: Sequence[int], lr: float,
                 batch: int, iters: int, buffer_cap: int, seed: int) -> None:
        self.in_dim = in_dim
        self.out_dim = in_dim + 1
        self.batch = batch
        self.iters = iters
        self.cap = buffer_cap
        self.rng = np.random.default_rng(seed)
        self.members = [_MLP(in_dim, self.out_dim, hidden, lr, seed + 101 * (i + 1))
                        for i in range(max(1, ensemble))]
        self._S: List[List[float]] = []                # states
        self._T: List[List[float]] = []                # targets = [Δstate..., reward]
        self.in_mean = self.in_std = None
        self.out_mean = self.out_std = None
        self.err_ema = 1.0
        self.samples = 0

    def add(self, state: Sequence[float], delta: Sequence[float], reward: float) -> None:
        self._S.append(list(state))
        self._T.append([*delta, float(reward)])
        if len(self._S) > self.cap:
            self._S.pop(0)
            self._T.pop(0)
        self.samples += 1

    def _refresh_stats(self) -> Tuple[Any, Any]:
        S = np.asarray(self._S, dtype=np.float64)
        T = np.asarray(self._T, dtype=np.float64)
        self.in_mean = S.mean(axis=0)
        self.in_std = S.std(axis=0) + 1e-6
        self.out_mean = T.mean(axis=0)
        self.out_std = T.std(axis=0) + 1e-6
        return S, T

    def train(self) -> None:
        if not self._S:
            return
        S, T = self._refresh_stats()
        Xn = (S - self.in_mean) / self.in_std
        Yn = (T - self.out_mean) / self.out_std
        n = Xn.shape[0]
        bs = min(self.batch, n)
        last = self.err_ema
        for m in self.members:
            for _ in range(self.iters):
                idx = self.rng.integers(0, n, size=bs)      # bootstrap mini-batch (replay)
                last = m.train_batch(Xn[idx], Yn[idx])
        self.err_ema = 0.9 * self.err_ema + 0.1 * last

    def predict(self, state: Sequence[float]) -> Optional[Tuple[List[float], float, float, float]]:
        if self.in_mean is None:
            return None
        xn = (np.asarray(state, dtype=np.float64) - self.in_mean) / self.in_std
        outs = np.asarray([m.forward(xn[None, :])[0][0] for m in self.members])  # normalised
        mean_n = outs.mean(axis=0)
        disagree_n = outs.std(axis=0)                                            # epistemic
        mean = mean_n * self.out_std + self.out_mean                             # destandardise
        delta = [float(v) for v in mean[: self.in_dim]]                          # native floats
        reward = float(mean[self.in_dim])
        # epistemic uncertainty: members' disagreement on the state delta (in real units)
        epistemic = float(np.mean(disagree_n[: self.in_dim] * self.out_std[: self.in_dim]))
        epistemic_norm = float(np.mean(disagree_n[: self.in_dim]))               # scale-free
        deviation = float(np.mean(np.abs(xn)))                                   # OOD signal
        return delta, reward, epistemic, _pack_conf(epistemic_norm, deviation)


def _pack_conf(epistemic_norm: float, deviation: float) -> float:
    """Fold ensemble agreement + out-of-distribution distance into one [0,1] reliability."""
    agree = math.exp(-epistemic_norm / 0.5)                  # members agree ⇒ →1
    ood = math.exp(-max(0.0, deviation - 2.0))               # within seen range ⇒ →1
    return max(0.0, min(1.0, agree * ood))


class EnsembleWorldModel(WorldModel):
    """Real learned dynamics: a per-action **deep ensemble** of numpy MLPs, Adam-trained on
    mini-batches drawn from a **replay buffer**, with **epistemic uncertainty = the ensemble's
    disagreement**.

    Drop-in for :class:`WorldModel` (same ``observe``/``predict`` surface ⇒ inherits
    ``rollout``/``counterfactual``/``intervene``). For **numeric** states it learns the dynamics
    with the ensemble — generalising beyond stored points, and *knowing what it does not know*
    (where members diverge, confidence collapses). For **symbolic** states it transparently
    falls back to an internal kNN :class:`WorldModel` (exact memory), so it is a strict
    superset of the kNN's capability, never a regression. Requires numpy."""

    def __init__(self, *, ensemble: int = 5, hidden: Sequence[int] = (64, 64),
                 lr: float = 1e-3, batch: int = 32, iters: int = 4, train_every: int = 1,
                 max_transitions: int = 50_000, experience_full: int = 64,
                 seed: int = 0) -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("EnsembleWorldModel requires numpy")
        self.ensemble = max(1, ensemble)
        self.hidden = tuple(hidden)
        self.lr = lr
        self.batch = batch
        self.iters = iters
        self.train_every = max(1, train_every)
        self.max_transitions = max_transitions
        self.experience_full = max(1, experience_full)
        self.seed = seed
        self._models: Dict[Action, _ActionModel] = {}
        self._state_dim: Optional[int] = None
        self._count = 0
        self._since_train: Dict[Action, int] = {}
        # exact-memory fallback for symbolic / inconsistent-dimension transitions
        self._knn = WorldModel(max_transitions=max_transitions)

    def __len__(self) -> int:
        return self._count

    def actions(self) -> List[Action]:
        return list({*self._models.keys(), *self._knn.actions()})

    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def _fits_ensemble(self, state: State, next_state: Optional[State] = None) -> bool:
        if not self._numeric(state) or (next_state is not None and not self._numeric(next_state)):
            return False
        if self._state_dim is not None and len(state) != self._state_dim:
            return False
        if next_state is not None and self._state_dim is not None \
                and len(next_state) != self._state_dim:
            return False
        return True

    def observe(self, state: Sequence[Any], action: Action,
                next_state: Sequence[Any], reward: float = 0.0) -> None:
        state = tuple(state)
        next_state = tuple(next_state)
        if not (self._numeric(state) and self._numeric(next_state)):
            self._knn.observe(state, action, next_state, reward)   # symbolic → exact memory
            self._count += 1
            return
        if self._state_dim is None:
            self._state_dim = len(state)
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            self._knn.observe(state, action, next_state, reward)   # odd-dim → exact memory
            self._count += 1
            return
        model = self._models.get(action)
        if model is None:
            model = _ActionModel(self._state_dim, ensemble=self.ensemble, hidden=self.hidden,
                                 lr=self.lr, batch=self.batch, iters=self.iters,
                                 buffer_cap=self.max_transitions,
                                 seed=self.seed + 1009 * (len(self._models) + 1))
            self._models[action] = model
            self._since_train[action] = 0
        delta = [next_state[i] - state[i] for i in range(self._state_dim)]
        model.add(state, delta, reward)
        self._count += 1
        self._since_train[action] += 1
        if self._since_train[action] >= self.train_every:
            model.train()
            self._since_train[action] = 0

    def predict(self, state: Sequence[Any], action: Action) -> Prediction:
        state = tuple(state)
        model = self._models.get(action)
        if model is not None and self._fits_ensemble(state):
            out = model.predict(state)
            if out is not None:
                delta, reward, epistemic, reliability = out
                next_state: State = tuple(state[i] + delta[i] for i in range(self._state_dim))
                experience = min(1.0, model.samples / float(self.experience_full))
                fit = 1.0 / (1.0 + model.err_ema)
                conf = max(0.0, min(1.0, experience * fit * reliability))
                return Prediction(next_state=next_state, reward=reward, confidence=conf,
                                  neighbors=model.samples, epistemic=epistemic)
        return self._knn.predict(state, action)          # symbolic / untrained → exact memory

    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count, "backend": "deep-ensemble+knn-fallback",
                "ensemble": self.ensemble, "hidden": list(self.hidden),
                "numeric_actions": {str(a): m.samples for a, m in self._models.items()},
                "symbolic_transitions": len(self._knn)}


# --------------------------------------------------------------------------- #
# Factory — pick the strongest learner the machine can support, never crash
# --------------------------------------------------------------------------- #
def build_world_model(backend: str = "auto", **kwargs: Any) -> WorldModel:
    """Build the best world model available. ``backend``:

    * ``"auto"`` / ``"ensemble"`` → :class:`EnsembleWorldModel` when numpy is present, else the
      pure-Python :class:`NeuralWorldModel` (and the kNN :class:`WorldModel` as the final floor);
    * ``"neural"`` → :class:`NeuralWorldModel`;
    * ``"knn"`` → :class:`WorldModel`.

    Unknown keyword args are filtered per backend so callers can pass a superset safely.
    """
    def _filtered(cls: type) -> Dict[str, Any]:
        import inspect
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
        return {k: v for k, v in kwargs.items() if k in allowed}

    if backend in ("auto", "ensemble"):
        if _HAS_NUMPY:
            try:
                return EnsembleWorldModel(**_filtered(EnsembleWorldModel))
            except Exception:  # noqa: BLE001 — fall through to the stdlib learners
                pass
        try:
            return NeuralWorldModel(**_filtered(NeuralWorldModel))
        except Exception:  # noqa: BLE001
            return WorldModel(**_filtered(WorldModel))
    if backend == "neural":
        return NeuralWorldModel(**_filtered(NeuralWorldModel))
    return WorldModel(**_filtered(WorldModel))


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

    # ---- deep-ensemble model: real learned dynamics + epistemic uncertainty ---- #
    if _HAS_NUMPY:
        em = build_world_model("ensemble", seed=0)
        assert isinstance(em, EnsembleWorldModel)
        rng2 = random.Random(1)
        for _ in range(600):
            x = rng2.uniform(-10, 10)
            a = rng2.choice(["left", "right", "stay"])
            nx = true_step(x, a)
            em.observe((x,), a, (nx,), reward=-abs(nx))
        pe = em.predict((4.0,), "left")
        far = em.predict((1000.0,), "left")
        print(f"\nensemble (4,left)   : next={tuple(round(v,2) for v in pe.next_state)} "
              f"conf={pe.confidence:.2f} epistemic={pe.epistemic:.3f}")
        print(f"ensemble conf n/f   : {pe.confidence:.2f} / {far.confidence:.3f}")
        assert abs(pe.next_state[0] - 3.0) < 0.4         # learned the Δ for 'left'
        assert pe.confidence > 0.5 and far.confidence < 0.1   # honest where it has not been
        traj_e = em.rollout((8.0,), go_home, steps=14)
        assert abs(traj_e.final_state[0]) < 1.5          # imagined its way home
        print(f"ensemble rollout    : final x={traj_e.final_state[0]:.2f}")
        print(f"ensemble stats      : {em.stats()}")
    else:
        print("\nensemble            : numpy absent → factory falls back (stdlib learners)")

    print("\nALL SELF-TESTS PASSED ✓")
