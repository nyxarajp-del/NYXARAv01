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
* :class:`TransferWorldModel` — **cross-domain transfer**. The models above keep a *separate*
  model per action, so nothing learned about one action helps another. This one trains **a single
  shared ensemble** conditioned on a **learned action embedding** (``[state ⊕ e(action)]``), with
  a :class:`~nyxara.mind.concept_hierarchy.ConceptHierarchy` that clusters actions by effect and
  **seeds a new action's embedding near its concept-mates** — so knowledge transfers across actions
  and even across domains, while confidence stays honest. The default (``backend="auto"``) when
  numpy is present.
* :class:`~nyxara.mind.jepa_world_model.JEPAWorldModel` (``backend="jepa"``, its own module) —
  the strongest learner and the kernel's default: a **hierarchical JEPA** that predicts the
  **abstract latent state** instead of raw observations (energy-based, non-generative,
  multi-scale horizons, EMA target encoder, VICReg anti-collapse, latent-space planning and a
  calibrated surprise/curiosity signal).

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
    "TransferWorldModel",
    "GroundedWorldModel",
    "build_world_model",
    "save_world_model",
    "load_world_model",
]

State = Tuple[Any, ...]
Action = Hashable
Policy = Union[Sequence[Action], Callable[[State], Action]]


def _unrepr_action(key: str) -> Action:
    """Recover an action persisted as ``repr(action)`` (str/int/float/tuple survive)."""
    import ast
    try:
        return ast.literal_eval(key)
    except Exception:  # noqa: BLE001 — an exotic action key stays a plain string
        return key


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
    done: bool = False         # did the episode terminate at this transition?


@dataclass
class Prediction:
    next_state: State
    reward: float
    confidence: float          # [0,1] — how much the model trusts this prediction
    neighbors: int = 0
    epistemic: float = 0.0     # model-disagreement (ensemble) — what it does NOT know
    done_prob: float = 0.0     # learned probability the episode ends here

    def to_dict(self) -> Dict[str, Any]:
        return {"next_state": self.next_state, "reward": round(self.reward, 4),
                "confidence": round(self.confidence, 4), "neighbors": self.neighbors,
                "epistemic": round(self.epistemic, 4),
                "done_prob": round(self.done_prob, 4)}


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

    # ---- calibration: every backend tracks how good its predictions REALLY are ---- #
    def _calib(self) -> Dict[str, float]:
        """Lazy shared calibration state (subclasses define their own __init__)."""
        c = self.__dict__.get("_calib_state")
        if c is None:
            c = {"err_ema": -1.0, "epi_ema": -1.0, "checks": 0.0}
            self.__dict__["_calib_state"] = c
        return c

    def _track_realized(self, state: Sequence[Any], action: Action,
                        next_state: Sequence[Any]) -> None:
        """One-step realized-error tracking: before learning a transition, ask the model
        what it *would have predicted* and score it against what actually happened. This
        is the honest signal behind :meth:`prediction_error_ema` and
        :meth:`mean_epistemic` — measured on real experience, never on training fit."""
        c = self._calib()
        c["checks"] += 1.0
        if int(c["checks"]) % 4 != 1:                 # sample 1-in-4: cheap but unbiased
            return
        try:
            pred = self.predict(state, action)
        except Exception:  # noqa: BLE001 — calibration must never block learning
            return
        if pred.confidence <= 0.0:
            return                                     # nothing was claimed, nothing to score
        err = _dist(pred.next_state, tuple(next_state))
        c["err_ema"] = err if c["err_ema"] < 0 else 0.95 * c["err_ema"] + 0.05 * err
        if pred.epistemic > 0.0:
            e = pred.epistemic
            c["epi_ema"] = e if c["epi_ema"] < 0 else 0.95 * c["epi_ema"] + 0.05 * e

    def prediction_error_ema(self) -> float:
        """EMA of realized one-step prediction error (0.0 until anything was scored)."""
        return max(0.0, self._calib()["err_ema"])

    def mean_epistemic(self) -> float:
        """EMA of the model's own epistemic uncertainty on real queries — the honest
        'how much do I not know about the world right now' signal the kernel reads."""
        return max(0.0, self._calib()["epi_ema"])

    # ---- learning = remembering ---- #
    def observe(self, state: Sequence[Any], action: Action,
                next_state: Sequence[Any], reward: float = 0.0,
                done: bool = False) -> None:
        self._track_realized(state, action, next_state)
        t = Transition(tuple(state), action, tuple(next_state), reward, bool(done))
        bucket = self._by_action.setdefault(action, [])
        bucket.append(t)
        self._count += 1
        if self._count > self.max_transitions:           # forget the oldest
            oldest_action = max(self._by_action, key=lambda a: len(self._by_action[a]))
            self._by_action[oldest_action].pop(0)
            self._count -= 1

    def observe_many(self, transitions: Sequence[Transition]) -> None:
        for t in transitions:
            self.observe(t.state, t.action, t.next_state, t.reward, t.done)

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
        done_w = 0.0
        all_numeric = all(isinstance(state[i], (int, float)) and not isinstance(state[i], bool)
                          for i in range(dim))
        for w, (d, t) in zip(weights, nn):
            if all_numeric and len(t.state) == dim and len(t.next_state) == dim:
                for i in range(dim):
                    delta[i] += w * (t.next_state[i] - t.state[i])
            reward += w * t.reward
            done_w += w * (1.0 if t.done else 0.0)
        reward /= wsum
        done_prob = max(0.0, min(1.0, done_w / wsum))

        if all_numeric:
            next_state: State = tuple(state[i] + delta[i] / wsum for i in range(dim))
        else:
            # symbolic: take the nearest neighbour's outcome
            next_state = nn[0][1].next_state

        nearest = nn[0][0]
        confidence = math.exp(-nearest / max(1e-6, self.distance_scale))
        return Prediction(next_state=next_state, reward=reward,
                          confidence=min(1.0, confidence), neighbors=len(nn),
                          done_prob=done_prob)

    def coverage(self, state: Sequence[Any], action: Action) -> float:
        """How well the model knows this (state, action) — its prediction confidence."""
        return self.predict(state, action).confidence

    def learning_progress(self, state: Sequence[Any], action: Action) -> float:
        """Expected information gain of taking ``action`` here — an honest curiosity signal in
        ``[0, 1]``, shared by every embodied loop (filesystem + physics) so exploration is
        driven by one calibrated measure instead of each caller re-deriving ``1 - confidence``.

        A high value means "I do not yet know what this action does here, so doing it would
        teach me a lot" — exactly the drive that makes a child drop things to see what happens.
        Two honest signals feed it, and the *more curious* of the two wins:

        * ``1 - confidence`` — plain novelty: an unseen or out-of-distribution (state, action)
          scores ~1, a well-modelled one scores ~0.
        * ensemble **epistemic disagreement** (when the backend reports it) — the models
          contradicting each other means the region is genuinely under-determined even if any
          single net is locally confident. Squashed into ``[0, 1)`` so its raw magnitude never
          dominates the metric.

        Works on every backend unchanged: ``predict`` always returns ``confidence`` (and
        ``epistemic`` defaults to 0 where a model has no ensemble), so this needs no per-backend
        override. Never raises — a modelling slip yields maximal curiosity (there is clearly
        something here left to learn), never a crash.
        """
        try:
            pred = self.predict(state, action)
        except Exception:  # noqa: BLE001 — an un-predictable state is maximally worth learning
            return 1.0
        conf = max(0.0, min(1.0, pred.confidence))
        info_gain = 1.0 - conf
        if pred.epistemic > 0.0:
            scale = max(1e-6, getattr(self, "distance_scale", 1.0))
            disagreement = 1.0 - math.exp(-pred.epistemic / scale)
            info_gain = max(info_gain, disagreement)
        return max(0.0, min(1.0, info_gain))

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
            # the learned done head: a confidently-predicted episode end stops imagination
            if pred.done_prob > 0.5 and pred.confidence > 0.2:
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

    # ---- persistence: learned dynamics survive restarts (Rule 7) ---- #
    def to_dict(self, *, max_saved: int = 5000) -> Dict[str, Any]:
        per_action = max(64, max_saved // max(1, len(self._by_action) or 1))
        return {
            "kind": "knn",
            "transitions": [
                [list(t.state), t.action, list(t.next_state), t.reward, t.done]
                for ts in self._by_action.values() for t in ts[-per_action:]
            ],
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        loaded = 0
        for row in data.get("transitions", []):
            try:
                state, action, next_state, reward, done = row
                action = tuple(action) if isinstance(action, list) else action
                self.observe(tuple(state), action, tuple(next_state),
                             float(reward), bool(done))
                loaded += 1
            except Exception:  # noqa: BLE001 — one bad row never aborts a restore
                continue
        return loaded > 0


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
        self.out = in_dim + 2                  # [Δstate…, reward, done]
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

    def to_dict(self) -> Dict[str, Any]:
        return {"in_dim": self.in_dim, "h": self.h, "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2, "n": self.n, "mean": self.mean,
                "M2": self._M2, "err_ema": self.err_ema, "samples": self.samples}

    def load_dict(self, d: Dict[str, Any]) -> bool:
        if int(d.get("in_dim", -1)) != self.in_dim or int(d.get("h", -1)) != self.h:
            return False
        self.W1 = [[float(x) for x in row] for row in d["W1"]]
        self.b1 = [float(x) for x in d["b1"]]
        self.W2 = [[float(x) for x in row] for row in d["W2"]]
        self.b2 = [float(x) for x in d["b2"]]
        self.n = int(d.get("n", 0))
        self.mean = [float(x) for x in d.get("mean", self.mean)]
        self._M2 = [float(x) for x in d.get("M2", self._M2)]
        self.err_ema = float(d.get("err_ema", 1.0))
        self.samples = int(d.get("samples", 0))
        return True


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
                next_state: Sequence[Any], reward: float = 0.0,
                done: bool = False) -> None:
        state = tuple(state)
        next_state = tuple(next_state)
        if not self._numeric(state) or not self._numeric(next_state):
            return                               # neural dynamics are for numeric states
        if self._state_dim is None:
            self._state_dim = len(state)
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            return
        self._track_realized(state, action, next_state)
        net = self._nets.get(action)
        if net is None:
            net = _ForwardNet(self._state_dim, hidden=self.hidden, lr=self.lr,
                              seed=self.seed + len(self._nets))
            self._nets[action] = net
        target = ([next_state[i] - state[i] for i in range(self._state_dim)]
                  + [float(reward), 1.0 if done else 0.0])
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
        done_prob = max(0.0, min(1.0, out[self._state_dim + 1]))
        next_state: State = tuple(state[i] + delta[i] for i in range(self._state_dim))
        # honest confidence: grows with experience + low error, decays out of distribution
        experience = min(1.0, net.samples / 20.0)
        fit = 1.0 / (1.0 + net.err_ema)
        ood = math.exp(-max(0.0, net.deviation(state) - self.ood_tolerance))
        conf = max(0.0, min(1.0, experience * fit * ood))
        return Prediction(next_state=next_state, reward=reward,
                          confidence=conf, neighbors=net.samples, done_prob=done_prob)

    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count, "backend": "neural-mlp", "hidden": self.hidden,
                "actions": {str(a): n.samples for a, n in self._nets.items()}}

    def to_dict(self, **_kw: Any) -> Dict[str, Any]:
        return {"kind": "neural", "state_dim": self._state_dim, "count": self._count,
                "hidden": self.hidden,
                "nets": {repr(a): n.to_dict() for a, n in self._nets.items()}}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "neural":
            return False
        if int(data.get("hidden", -1)) != self.hidden:
            return False
        state_dim = data.get("state_dim")
        if state_dim is None:
            return False
        self._state_dim = int(state_dim)
        restored = 0
        for key, nd in (data.get("nets") or {}).items():
            try:
                action = _unrepr_action(key)
                net = _ForwardNet(self._state_dim, hidden=self.hidden, lr=self.lr)
                if net.load_dict(nd):
                    self._nets[action] = net
                    restored += 1
            except Exception:  # noqa: BLE001 — one bad net never aborts a restore
                continue
        self._count = int(data.get("count", 0))
        return restored > 0


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
        # per-sample errors feed prioritized replay in the owning model
        self.last_per_sample = np.mean(diff ** 2, axis=1)
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

    def to_dict(self) -> Dict[str, Any]:
        return {"dims": list(self.dims),
                "W": [w.tolist() for w in self.W],
                "b": [b.tolist() for b in self.b]}

    def load_dict(self, d: Dict[str, Any]) -> bool:
        if list(d.get("dims", [])) != list(self.dims):
            return False
        self.W = [np.asarray(w, dtype=np.float64) for w in d["W"]]
        self.b = [np.asarray(b, dtype=np.float64) for b in d["b"]]
        # Adam moments restart cold — harmless: they re-warm within a few batches
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0
        return True

    def input_grad(self, X: Any, Y: Any) -> Any:
        """Gradient of the loss w.r.t. the *input* ``X`` (weights untouched).

        Lets the owning model learn an **action embedding** packed into ``X``: the
        embedding columns get a real gradient signal, so similar-effect actions drift to
        similar embeddings and the single shared net transfers across them."""
        out, acts = self.forward(X)
        n = X.shape[0]
        delta = (2.0 / n) * (out - Y)                  # dL/d(out)
        for i in reversed(range(len(self.W))):
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
            else:
                return delta @ self.W[0].T             # dL/d(input)
        return delta @ self.W[0].T

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
    """A deep ensemble + prioritized replay buffer for ONE action; standardised dynamics."""

    def __init__(self, in_dim: int, *, ensemble: int, hidden: Sequence[int], lr: float,
                 batch: int, iters: int, buffer_cap: int, seed: int) -> None:
        self.in_dim = in_dim
        self.out_dim = in_dim + 2                      # [Δstate…, reward, done]
        self.hidden = tuple(hidden)
        self.batch = batch
        self.iters = iters
        self.cap = buffer_cap
        self.rng = np.random.default_rng(seed)
        self.members = [_MLP(in_dim, self.out_dim, hidden, lr, seed + 101 * (i + 1))
                        for i in range(max(1, ensemble))]
        self._S: List[List[float]] = []                # states
        self._T: List[List[float]] = []                # targets = [Δstate..., reward, done]
        self._P: List[float] = []                      # replay priorities (last errors)
        self.in_mean = self.in_std = None
        self.out_mean = self.out_std = None
        self.err_ema = 1.0
        self.samples = 0
        # self-tuned confidence calibration: running scale of observed disagreement,
        # so `reliability` adapts to the domain instead of hard-coded constants
        self.epi_scale = 0.5
        self.dev_margin = 2.0

    def add(self, state: Sequence[float], delta: Sequence[float], reward: float,
            done: bool = False) -> None:
        self._S.append(list(state))
        self._T.append([*delta, float(reward), 1.0 if done else 0.0])
        self._P.append(max(self._P) if self._P else 1.0)   # new experience trains soonest
        if len(self._S) > self.cap:
            self._S.pop(0)
            self._T.pop(0)
            self._P.pop(0)
        self.samples += 1

    def _refresh_stats(self) -> Tuple[Any, Any]:
        S = np.asarray(self._S, dtype=np.float64)
        T = np.asarray(self._T, dtype=np.float64)
        self.in_mean = S.mean(axis=0)
        self.in_std = S.std(axis=0) + 1e-6
        self.out_mean = T.mean(axis=0)
        self.out_std = T.std(axis=0) + 1e-6
        return S, T

    def _sample_idx(self, n: int, bs: int) -> Any:
        """Prioritized replay: pick transitions ∝ (last_error + ε)^0.6, so surprising
        experience is rehearsed more — the model spends its budget where it is wrong."""
        p = np.asarray(self._P[:n], dtype=np.float64)
        p = (p + 1e-3) ** 0.6
        p /= p.sum()
        return self.rng.choice(n, size=bs, p=p)

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
                idx = self._sample_idx(n, bs)               # prioritized replay mini-batch
                last = m.train_batch(Xn[idx], Yn[idx])
                per = getattr(m, "last_per_sample", None)
                if per is not None:                          # refresh sampled priorities
                    for r, row in enumerate(idx):
                        self._P[int(row)] = float(per[r])
        self.err_ema = 0.9 * self.err_ema + 0.1 * last

    def predict(self, state: Sequence[float]) -> Optional[
            Tuple[List[float], float, float, float, float]]:
        if self.in_mean is None:
            return None
        xn = (np.asarray(state, dtype=np.float64) - self.in_mean) / self.in_std
        outs = np.asarray([m.forward(xn[None, :])[0][0] for m in self.members])  # normalised
        mean_n = outs.mean(axis=0)
        disagree_n = outs.std(axis=0)                                            # epistemic
        mean = mean_n * self.out_std + self.out_mean                             # destandardise
        delta = [float(v) for v in mean[: self.in_dim]]                          # native floats
        reward = float(mean[self.in_dim])
        done_prob = max(0.0, min(1.0, float(mean[self.in_dim + 1])))
        # epistemic uncertainty: members' disagreement on the state delta (in real units)
        epistemic = float(np.mean(disagree_n[: self.in_dim] * self.out_std[: self.in_dim]))
        epistemic_norm = float(np.mean(disagree_n[: self.in_dim]))               # scale-free
        deviation = float(np.mean(np.abs(xn)))                                   # OOD signal
        # self-calibration: the agreement scale tracks the disagreement actually seen,
        # so "members agree" means agree *for this domain*, not for a magic constant
        self.epi_scale = max(0.05, 0.99 * self.epi_scale + 0.01 * 2.0 * epistemic_norm)
        reliability = _pack_conf(epistemic_norm, deviation,
                                 epi_scale=self.epi_scale, dev_margin=self.dev_margin)
        return delta, reward, done_prob, epistemic, reliability

    def to_dict(self, *, max_buffer: int = 2000) -> Dict[str, Any]:
        keep = min(len(self._S), max_buffer)
        return {"in_dim": self.in_dim, "hidden": list(self.hidden),
                "members": [m.to_dict() for m in self.members],
                "S": self._S[-keep:], "T": self._T[-keep:], "P": self._P[-keep:],
                "err_ema": self.err_ema, "samples": self.samples,
                "epi_scale": self.epi_scale}

    def load_dict(self, d: Dict[str, Any]) -> bool:
        if int(d.get("in_dim", -1)) != self.in_dim or \
                list(d.get("hidden", [])) != list(self.hidden):
            return False
        members = d.get("members") or []
        if len(members) != len(self.members):
            return False
        for m, md in zip(self.members, members):
            if not m.load_dict(md):
                return False
        self._S = [list(map(float, r)) for r in d.get("S", [])]
        self._T = [list(map(float, r)) for r in d.get("T", [])]
        self._P = [float(x) for x in d.get("P", [])] or [1.0] * len(self._S)
        self.err_ema = float(d.get("err_ema", 1.0))
        self.samples = int(d.get("samples", len(self._S)))
        self.epi_scale = float(d.get("epi_scale", 0.5))
        if self._S:
            self._refresh_stats()
        return True


def _pack_conf(epistemic_norm: float, deviation: float, *,
               epi_scale: float = 0.5, dev_margin: float = 2.0) -> float:
    """Fold ensemble agreement + out-of-distribution distance into one [0,1] reliability.

    The scales are *parameters* (self-tuned by the owning model from the disagreement it
    actually observes), not hard-coded truths."""
    agree = math.exp(-epistemic_norm / max(1e-6, epi_scale))  # members agree ⇒ →1
    ood = math.exp(-max(0.0, deviation - dev_margin))         # within seen range ⇒ →1
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
                next_state: Sequence[Any], reward: float = 0.0,
                done: bool = False) -> None:
        state = tuple(state)
        next_state = tuple(next_state)
        if not (self._numeric(state) and self._numeric(next_state)):
            self._knn.observe(state, action, next_state, reward, done)  # symbolic → exact
            self._count += 1
            return
        if self._state_dim is None:
            self._state_dim = len(state)
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            self._knn.observe(state, action, next_state, reward, done)  # odd-dim → exact
            self._count += 1
            return
        self._track_realized(state, action, next_state)
        model = self._models.get(action)
        if model is None:
            model = _ActionModel(self._state_dim, ensemble=self.ensemble, hidden=self.hidden,
                                 lr=self.lr, batch=self.batch, iters=self.iters,
                                 buffer_cap=self.max_transitions,
                                 seed=self.seed + 1009 * (len(self._models) + 1))
            self._models[action] = model
            self._since_train[action] = 0
        delta = [next_state[i] - state[i] for i in range(self._state_dim)]
        model.add(state, delta, reward, done)
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
                delta, reward, done_prob, epistemic, reliability = out
                next_state: State = tuple(state[i] + delta[i] for i in range(self._state_dim))
                experience = min(1.0, model.samples / float(self.experience_full))
                fit = 1.0 / (1.0 + model.err_ema)
                conf = max(0.0, min(1.0, experience * fit * reliability))
                return Prediction(next_state=next_state, reward=reward, confidence=conf,
                                  neighbors=model.samples, epistemic=epistemic,
                                  done_prob=done_prob)
        return self._knn.predict(state, action)          # symbolic / untrained → exact memory

    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count, "backend": "deep-ensemble+knn-fallback",
                "ensemble": self.ensemble, "hidden": list(self.hidden),
                "numeric_actions": {str(a): m.samples for a, m in self._models.items()},
                "symbolic_transitions": len(self._knn)}

    def to_dict(self, **_kw: Any) -> Dict[str, Any]:
        return {"kind": "ensemble", "state_dim": self._state_dim, "count": self._count,
                "ensemble": self.ensemble, "hidden": list(self.hidden),
                "models": {repr(a): m.to_dict() for a, m in self._models.items()},
                "knn": self._knn.to_dict()}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "ensemble":
            return False
        if int(data.get("ensemble", -1)) != self.ensemble or \
                list(data.get("hidden", [])) != list(self.hidden):
            return False
        state_dim = data.get("state_dim")
        restored = 0
        if state_dim is not None:
            self._state_dim = int(state_dim)
            for key, md in (data.get("models") or {}).items():
                try:
                    action = _unrepr_action(key)
                    model = _ActionModel(
                        self._state_dim, ensemble=self.ensemble, hidden=self.hidden,
                        lr=self.lr, batch=self.batch, iters=self.iters,
                        buffer_cap=self.max_transitions,
                        seed=self.seed + 1009 * (len(self._models) + 1))
                    if model.load_dict(md):
                        self._models[action] = model
                        self._since_train[action] = 0
                        restored += 1
                except Exception:  # noqa: BLE001 — one bad model never aborts a restore
                    continue
        if data.get("knn"):
            self._knn.load_dict(data["knn"])
        self._count = int(data.get("count", 0))
        return restored > 0 or len(self._knn) > 0


# --------------------------------------------------------------------------- #
# Transfer model — ONE shared encoder over (state ⊕ learned action embedding),
# with a concept hierarchy so knowledge transfers across actions and domains
# --------------------------------------------------------------------------- #
class TransferWorldModel(WorldModel):
    """Cross-domain transfer: a **single shared deep ensemble** conditioned on a **learned
    action embedding**, instead of one isolated model per action.

    The previous models partitioned everything by action — learning ``left`` taught
    ``move_left`` nothing, and a novel action was a zero-confidence blank. Here, all actions
    train **one** numpy ensemble whose input is ``[normalised_state ⊕ e(action)]``. The
    embedding ``e(action)`` is *learned by gradient* (the embedding columns receive a real
    error signal via :meth:`_MLP.input_grad`), so actions with the same effect drift to nearby
    embeddings and the shared weights generalise across them — genuine transfer.

    A :class:`~nyxara.mind.concept_hierarchy.ConceptHierarchy` sits on top: it clusters actions
    by effect signature, so a brand-new action's embedding is **seeded near its concept-mates**
    (few-shot, even cross-domain transfer) and confidence stays **honest** — discounted while an
    action leans on borrowed (sibling) evidence, rising as it accrues its own. Symbolic /
    odd-dimension transitions fall back to an exact-memory kNN :class:`WorldModel`, so it is a
    strict superset, never a regression. Drop-in for :class:`WorldModel` (same
    ``observe``/``predict`` surface ⇒ inherits ``rollout``/``counterfactual``/``intervene``).
    Requires numpy."""

    def __init__(self, *, ensemble: int = 5, hidden: Sequence[int] = (64, 64),
                 lr: float = 1e-3, batch: int = 32, iters: int = 4, train_every: int = 1,
                 max_transitions: int = 50_000, experience_full: int = 64,
                 embed_dim: int = 8, embed_lr: float = 0.05, transfer_weight: float = 0.5,
                 concept_threshold: float = 0.4, seed: int = 0) -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("TransferWorldModel requires numpy")
        from nyxara.mind.concept_hierarchy import ConceptHierarchy
        self.ensemble = max(1, ensemble)
        self.hidden = tuple(hidden)
        self.lr = lr
        self.batch = batch
        self.iters = iters
        self.train_every = max(1, train_every)
        self.max_transitions = max_transitions
        self.experience_full = max(1, experience_full)
        self.embed_dim = max(1, embed_dim)
        self.embed_lr = embed_lr
        self.transfer_weight = max(0.0, min(1.0, transfer_weight))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.concepts = ConceptHierarchy(threshold=concept_threshold)
        # per-action embeddings + their Adam moments
        self._embed: Dict[Action, Any] = {}
        self._embed_m: Dict[Action, Any] = {}
        self._embed_v: Dict[Action, Any] = {}
        self._embed_t: Dict[Action, int] = {}
        self._own_samples: Dict[Action, int] = {}
        self._members: Optional[List[_MLP]] = None
        self._state_dim: Optional[int] = None
        # ONE shared replay buffer across all actions (states, actions, targets, priorities)
        self._S: List[List[float]] = []
        self._A: List[Action] = []
        self._T: List[List[float]] = []
        self._P: List[float] = []
        self._count = 0
        self._since_train = 0
        # shared normalisation stats
        self._in_mean = self._in_std = None            # over the state part
        self._out_mean = self._out_std = None          # over the target [Δstate, reward, done]
        self.err_ema = 1.0
        # self-tuned confidence calibration (see _pack_conf)
        self._epi_scale = 0.5
        # exact-memory fallback for symbolic / inconsistent-dimension transitions
        self._knn = WorldModel(max_transitions=max_transitions)

    def __len__(self) -> int:
        return self._count

    def actions(self) -> List[Action]:
        return list({*self._embed.keys(), *self._knn.actions()})

    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def _ensure_embed(self, action: Action) -> None:
        """Create an embedding for a new action, seeded near its concept-mates (transfer)."""
        if action in self._embed:
            return
        # concept-mates are homogeneous → average them; otherwise borrow the single nearest
        # action by dynamics (averaging dissimilar actions would dilute the transfer)
        siblings = self.concepts.siblings(action)
        neighbours = siblings if siblings else self.concepts.nearest_actions(action, k=1)
        seeds = [self._embed[a] for a in neighbours if a in self._embed]
        if seeds:                                      # borrow the concept's region of space
            base = np.mean(np.asarray(seeds), axis=0)
            e = base + 0.01 * self._rng.standard_normal(self.embed_dim)
        else:
            e = 0.1 * self._rng.standard_normal(self.embed_dim)
        self._embed[action] = e
        self._embed_m[action] = np.zeros(self.embed_dim)
        self._embed_v[action] = np.zeros(self.embed_dim)
        self._embed_t[action] = 0

    def observe(self, state: Sequence[Any], action: Action,
                next_state: Sequence[Any], reward: float = 0.0,
                done: bool = False) -> None:
        state = tuple(state)
        next_state = tuple(next_state)
        if not (self._numeric(state) and self._numeric(next_state)):
            self._knn.observe(state, action, next_state, reward, done)
            self._count += 1
            return
        if self._state_dim is None:
            self._state_dim = len(state)
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            self._knn.observe(state, action, next_state, reward, done)
            self._count += 1
            return
        self._track_realized(state, action, next_state)
        # update the concept hierarchy *first* so a new action seeds near the right concept
        self.concepts.update(action, state, next_state, reward)
        self._ensure_embed(action)
        delta = [next_state[i] - state[i] for i in range(self._state_dim)]
        self._S.append(list(state))
        self._A.append(action)
        self._T.append([*delta, float(reward), 1.0 if done else 0.0])
        self._P.append(max(self._P) if self._P else 1.0)   # new experience trains soonest
        if len(self._S) > self.max_transitions:
            self._S.pop(0)
            self._A.pop(0)
            self._T.pop(0)
            self._P.pop(0)
        self._own_samples[action] = self._own_samples.get(action, 0) + 1
        self._count += 1
        self._since_train += 1
        if self._members is None:
            in_dim = self._state_dim + self.embed_dim
            self._members = [_MLP(in_dim, self._state_dim + 2, self.hidden, self.lr,
                                  self.seed + 101 * (i + 1)) for i in range(self.ensemble)]
        if self._since_train >= self.train_every:
            self._train()
            self._since_train = 0

    def _embed_matrix(self) -> Any:
        return np.asarray([self._embed[a] for a in self._A], dtype=np.float64)

    def _train(self) -> None:
        if not self._S or self._members is None:
            return
        S = np.asarray(self._S, dtype=np.float64)
        T = np.asarray(self._T, dtype=np.float64)
        self._in_mean = S.mean(axis=0)
        self._in_std = S.std(axis=0) + 1e-6
        self._out_mean = T.mean(axis=0)
        self._out_std = T.std(axis=0) + 1e-6
        Sn = (S - self._in_mean) / self._in_std
        Yn = (T - self._out_mean) / self._out_std
        n = Sn.shape[0]
        bs = min(self.batch, n)
        X = np.concatenate([Sn, self._embed_matrix()], axis=1)   # [state ⊕ embedding]
        prio = (np.asarray(self._P[:n], dtype=np.float64) + 1e-3) ** 0.6
        prio /= prio.sum()
        last = self.err_ema
        for m in self._members:
            for _ in range(self.iters):
                # prioritized replay: rehearse the transitions the model got wrong
                idx = self._rng.choice(n, size=bs, p=prio)
                last = m.train_batch(X[idx], Yn[idx])
                per = getattr(m, "last_per_sample", None)
                if per is not None:
                    for r, row in enumerate(idx):
                        self._P[int(row)] = float(per[r])
        self.err_ema = 0.9 * self.err_ema + 0.1 * last
        self._update_embeddings(Sn, Yn, bs)

    def _update_embeddings(self, Sn: Any, Yn: Any, bs: int) -> None:
        """One Adam step on each action's embedding from the shared net's input gradient."""
        n = Sn.shape[0]
        X = np.concatenate([Sn, self._embed_matrix()], axis=1)
        grad: Dict[Action, Any] = {}
        cnt: Dict[Action, int] = {}
        for m in self._members:
            idx = self._rng.integers(0, n, size=bs)
            demb = m.input_grad(X[idx], Yn[idx])[:, self._state_dim:]   # grad on embed cols
            for r, row in enumerate(idx):
                a = self._A[row]
                if a not in grad:
                    grad[a] = np.zeros(self.embed_dim)
                    cnt[a] = 0
                grad[a] += demb[r]
                cnt[a] += 1
        for a, g in grad.items():
            self._adam_embed(a, g / (max(1, cnt[a]) * len(self._members)))

    def _adam_embed(self, action: Action, g: Any, b1: float = 0.9, b2: float = 0.999,
                    eps: float = 1e-8) -> None:
        self._embed_t[action] += 1
        t = self._embed_t[action]
        self._embed_m[action] = b1 * self._embed_m[action] + (1 - b1) * g
        self._embed_v[action] = b2 * self._embed_v[action] + (1 - b2) * (g * g)
        mhat = self._embed_m[action] / (1 - b1 ** t)
        vhat = self._embed_v[action] / (1 - b2 ** t)
        self._embed[action] -= self.embed_lr * mhat / (np.sqrt(vhat) + eps)

    def predict(self, state: Sequence[Any], action: Action) -> Prediction:
        state = tuple(state)
        if (self._members is not None and action in self._embed and self._in_mean is not None
                and self._state_dim == len(state) and self._numeric(state)):
            sn = (np.asarray(state, dtype=np.float64) - self._in_mean) / self._in_std
            x = np.concatenate([sn, self._embed[action]])
            outs = np.asarray([m.forward(x[None, :])[0][0] for m in self._members])
            mean_n = outs.mean(axis=0)
            disagree_n = outs.std(axis=0)
            mean = mean_n * self._out_std + self._out_mean
            delta = [float(v) for v in mean[: self._state_dim]]
            reward = float(mean[self._state_dim])
            done_prob = max(0.0, min(1.0, float(mean[self._state_dim + 1])))
            next_state: State = tuple(state[i] + delta[i] for i in range(self._state_dim))
            epistemic = float(np.mean(disagree_n[: self._state_dim]
                                      * self._out_std[: self._state_dim]))
            epistemic_norm = float(np.mean(disagree_n[: self._state_dim]))
            deviation = float(np.mean(np.abs(sn)))
            # self-calibration: agreement scale tracks the disagreement actually observed
            self._epi_scale = max(0.05, 0.99 * self._epi_scale + 0.01 * 2.0 * epistemic_norm)
            reliability = _pack_conf(epistemic_norm, deviation, epi_scale=self._epi_scale)
            # honest confidence: own evidence counts fully, borrowed (sibling) evidence is
            # discounted by ``transfer_weight`` — so a transferring action is confident but
            # never *more* certain than one that learned the dynamics first-hand.
            own = self._own_samples.get(action, 0)
            borrowed = max(0, self.concepts.concept_support(action) - own)
            effective = own + self.transfer_weight * borrowed
            experience = min(1.0, effective / float(self.experience_full))
            fit = 1.0 / (1.0 + self.err_ema)
            conf = max(0.0, min(1.0, experience * fit * reliability))
            return Prediction(next_state=next_state, reward=reward, confidence=conf,
                              neighbors=own, epistemic=epistemic, done_prob=done_prob)
        return self._knn.predict(state, action)        # symbolic / unknown → exact memory

    def stats(self) -> Dict[str, Any]:
        return {"transitions": self._count, "backend": "shared-encoder+concept-hierarchy",
                "ensemble": self.ensemble, "hidden": list(self.hidden),
                "embed_dim": self.embed_dim,
                "concepts": len(self.concepts.concepts()),
                "numeric_actions": dict(self._own_samples),
                "symbolic_transitions": len(self._knn)}

    def concept_report(self) -> str:
        """Human-readable view of the learned concept hierarchy over actions."""
        return self.concepts.report()

    # ---- persistence: the learned dynamics + action embeddings survive restarts ---- #
    def to_dict(self, *, max_buffer: int = 4000, **_kw: Any) -> Dict[str, Any]:
        keep = min(len(self._S), max_buffer)
        return {
            "kind": "transfer", "state_dim": self._state_dim, "count": self._count,
            "ensemble": self.ensemble, "hidden": list(self.hidden),
            "embed_dim": self.embed_dim,
            "members": [m.to_dict() for m in self._members] if self._members else [],
            "embed": {repr(a): [float(x) for x in e] for a, e in self._embed.items()},
            "own_samples": {repr(a): int(n) for a, n in self._own_samples.items()},
            "S": self._S[-keep:],
            "A": [repr(a) for a in self._A[-keep:]],
            "T": self._T[-keep:],
            "P": self._P[-keep:],
            "err_ema": self.err_ema, "epi_scale": self._epi_scale,
            "knn": self._knn.to_dict(),
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "transfer":
            return False
        if int(data.get("ensemble", -1)) != self.ensemble or \
                list(data.get("hidden", [])) != list(self.hidden) or \
                int(data.get("embed_dim", -1)) != self.embed_dim:
            return False
        state_dim = data.get("state_dim")
        restored = False
        if state_dim is not None and data.get("members"):
            self._state_dim = int(state_dim)
            in_dim = self._state_dim + self.embed_dim
            members = [_MLP(in_dim, self._state_dim + 2, self.hidden, self.lr,
                            self.seed + 101 * (i + 1)) for i in range(self.ensemble)]
            if len(data["members"]) == len(members) and \
                    all(m.load_dict(md) for m, md in zip(members, data["members"])):
                self._members = members
                restored = True
        if restored:
            for key, e in (data.get("embed") or {}).items():
                a = _unrepr_action(key)
                self._embed[a] = np.asarray(e, dtype=np.float64)
                self._embed_m[a] = np.zeros(self.embed_dim)
                self._embed_v[a] = np.zeros(self.embed_dim)
                self._embed_t[a] = 0
            self._own_samples = {_unrepr_action(k): int(v)
                                 for k, v in (data.get("own_samples") or {}).items()}
            self._S = [list(map(float, r)) for r in data.get("S", [])]
            self._A = [_unrepr_action(k) for k in data.get("A", [])]
            self._T = [list(map(float, r)) for r in data.get("T", [])]
            self._P = [float(x) for x in data.get("P", [])] or [1.0] * len(self._S)
            self.err_ema = float(data.get("err_ema", 1.0))
            self._epi_scale = float(data.get("epi_scale", 0.5))
            self._count = int(data.get("count", len(self._S)))
            if self._S:
                S = np.asarray(self._S, dtype=np.float64)
                T = np.asarray(self._T, dtype=np.float64)
                self._in_mean = S.mean(axis=0)
                self._in_std = S.std(axis=0) + 1e-6
                self._out_mean = T.mean(axis=0)
                self._out_std = T.std(axis=0) + 1e-6
                # rebuild the concept hierarchy by replaying the persisted experience
                for s, a, t in zip(self._S, self._A, self._T):
                    ns = [s[i] + t[i] for i in range(self._state_dim)]
                    self.concepts.update(a, s, ns, t[self._state_dim])
        if data.get("knn"):
            restored = self._knn.load_dict(data["knn"]) or restored
        return restored


# --------------------------------------------------------------------------- #
# Grounded wrapper — text/symbolic experience becomes learnable numeric dynamics
# --------------------------------------------------------------------------- #
class GroundedWorldModel:
    """Grounds ANY experience — text, dicts, symbols — into the learned dynamics model.

    The learners above need numeric state vectors, so NYXARA's real conversational turns
    ("Master asked X, I replied Y, it went well") never used to reach them: the world
    model only ever saw simulators. This wrapper closes that gap. A ``state_encoder``
    maps arbitrary states through the **shared memory embedder** (the same self-learned
    space her recall uses) and a fixed seeded projection down to a compact numeric latent
    (default 16-d) — so a lived turn becomes a real ``(state, action, next_state,
    reward)`` transition the ensemble genuinely trains on.

    An optional ``causal_model`` (mind/causal_world_model.py) acts as a structural
    prior: when the dynamics model predicts a low-confidence effect for an action whose
    causal graph shows NO established effect, the predicted change is damped — causal
    knowledge constrains imagination, imagination feeds causal discovery (the reverse
    link lives in the kernel's idle loop). Same surface as :class:`WorldModel`
    (``observe``/``predict``/``rollout``/``counterfactual``/``intervene``…), stdlib-safe."""

    def __init__(self, inner: WorldModel, *, embedder: Any = None,
                 state_latent_dim: int = 16, causal_model: Any = None,
                 seed: int = 1469598103934665603) -> None:
        self.inner = inner
        self.embedder = embedder
        self.state_latent_dim = max(2, int(state_latent_dim))
        self.causal_model = causal_model
        self._seed = seed
        self._proj_cache: Dict[int, List[List[float]]] = {}

    def __len__(self) -> int:
        return len(self.inner)

    # ---- grounding: any state → a numeric latent the learners can model ---- #
    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def _projection(self, src_dim: int) -> List[List[float]]:
        """A fixed seeded ±1 sparse projection src_dim→latent (deterministic forever)."""
        cached = self._proj_cache.get(src_dim)
        if cached is not None:
            return cached
        rng = random.Random(self._seed ^ src_dim)
        rows = [[0.0] * src_dim for _ in range(self.state_latent_dim)]
        for j in range(src_dim):
            for _ in range(3):  # each source dim feeds 3 latent dims
                rows[rng.randrange(self.state_latent_dim)][j] += rng.choice((-1.0, 1.0))
        self._proj_cache[src_dim] = rows
        return rows

    def encode_state(self, state: Any) -> State:
        """Numeric sequences pass through; anything else embeds → projects to the latent."""
        if isinstance(state, (tuple, list)) and state and self._numeric(state):
            return tuple(state)
        text = state if isinstance(state, str) else str(state)
        if self.embedder is not None:
            try:
                vec = self.embedder.embed(text)
            except Exception:  # noqa: BLE001 — fall back to a hash latent below
                vec = None
        else:
            vec = None
        if not vec:
            # dependency-free fallback: bag-of-hashes straight into the latent
            rng = random.Random(hash(text) & 0xFFFFFFFF)
            return tuple(rng.uniform(-1, 1) for _ in range(self.state_latent_dim))
        proj = self._projection(len(vec))
        out = [sum(row[i] * vec[i] for i in range(len(vec))) for row in proj]
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return tuple(v / norm for v in out)

    # ---- the WorldModel surface, grounded ---- #
    def observe(self, state: Any, action: Action, next_state: Any,
                reward: float = 0.0, done: bool = False) -> None:
        self.inner.observe(self.encode_state(state), action,
                           self.encode_state(next_state), reward, done)

    def observe_many(self, transitions: Sequence[Transition]) -> None:
        for t in transitions:
            self.observe(t.state, t.action, t.next_state,
                         getattr(t, "reward", 0.0), getattr(t, "done", False))

    def predict(self, state: Any, action: Action) -> Prediction:
        pred = self.inner.predict(self.encode_state(state), action)
        return self._apply_causal_prior(pred, action)

    def _apply_causal_prior(self, pred: Prediction, action: Action) -> Prediction:
        """Damp low-confidence predicted change for actions with no known causal effect."""
        cm = self.causal_model
        if cm is None or pred.confidence >= 0.35:
            return pred
        try:
            effects = cm.effects_of(f"act:{action}")
            known = bool(getattr(cm, "_links", None)) or bool(effects)
            if known and not effects:
                # the causal graph is mature AND says this action causes nothing —
                # a speculative predicted change is more likely noise than knowledge
                sq = tuple(v * 0.5 if isinstance(v, (int, float)) else v
                           for v in pred.next_state)
                return Prediction(next_state=sq, reward=pred.reward * 0.5,
                                  confidence=pred.confidence, neighbors=pred.neighbors,
                                  epistemic=pred.epistemic, done_prob=pred.done_prob)
        except Exception:  # noqa: BLE001 — the prior is advisory, never fatal
            pass
        return pred

    def rollout(self, start: Any, policy: Policy, **kw: Any) -> Trajectory:
        return self.inner.rollout(self.encode_state(start), policy, **kw)

    def imagine(self, start: Any, action: Action, **kw: Any) -> Trajectory:
        return self.inner.imagine(self.encode_state(start), action, **kw)

    def counterfactual(self, start: Any, policy_a: Policy, policy_b: Policy,
                       **kw: Any) -> CounterfactualResult:
        return self.inner.counterfactual(self.encode_state(start), policy_a, policy_b, **kw)

    def intervene(self, start: Any, policy: Policy, **kw: Any) -> Trajectory:
        return self.inner.intervene(self.encode_state(start), policy, **kw)

    def learning_progress(self, state: Any, action: Action) -> float:
        return self.inner.learning_progress(self.encode_state(state), action)

    def coverage(self, state: Any, action: Action) -> float:
        return self.inner.coverage(self.encode_state(state), action)

    # ---- JEPA surface, grounded: text hypotheses become scoreable transitions ---- #
    # (each is explicit — the generic __getattr__ delegation below would forward the
    # call but NOT encode the states first, silently scoring garbage latents)
    def energy(self, state: Any, action: Action, next_state: Any, **kw: Any) -> float:
        fn = getattr(self.inner, "energy", None)
        if not callable(fn):
            return float("inf")
        return float(fn(self.encode_state(state), action,
                        self.encode_state(next_state), **kw))

    def plausibility(self, state: Any, action: Action, next_state: Any,
                     **kw: Any) -> float:
        fn = getattr(self.inner, "plausibility", None)
        if not callable(fn):
            return 0.0
        return float(fn(self.encode_state(state), action,
                        self.encode_state(next_state), **kw))

    def surprise(self, state: Any, action: Action, next_state: Any) -> float:
        fn = getattr(self.inner, "surprise", None)
        if not callable(fn):
            return 1.0
        return float(fn(self.encode_state(state), action, self.encode_state(next_state)))

    def predict_horizon(self, state: Any, action: Action, horizon: int = 1) -> Prediction:
        fn = getattr(self.inner, "predict_horizon", None)
        if not callable(fn):
            return self.predict(state, action)
        pred = fn(self.encode_state(state), action, horizon)
        return self._apply_causal_prior(pred, action)

    def plan(self, state: Any, *, goal_state: Any = None, **kw: Any) -> Any:
        fn = getattr(self.inner, "plan", None)
        if not callable(fn):
            return None
        goal = self.encode_state(goal_state) if goal_state is not None else None
        return fn(self.encode_state(state), goal_state=goal, **kw)

    def consolidate(self, iters: int = 12) -> int:
        fn = getattr(self.inner, "consolidate", None)
        return int(fn(iters)) if callable(fn) else 0

    def actions(self) -> List[Action]:
        return self.inner.actions()

    def mean_epistemic(self) -> float:
        fn = getattr(self.inner, "mean_epistemic", None)
        return float(fn()) if callable(fn) else 0.0

    def prediction_error_ema(self) -> float:
        fn = getattr(self.inner, "prediction_error_ema", None)
        return float(fn()) if callable(fn) else 0.0

    def stats(self) -> Dict[str, Any]:
        s = dict(self.inner.stats())
        s["grounded"] = True
        s["state_latent_dim"] = self.state_latent_dim
        s["prediction_error_ema"] = round(self.prediction_error_ema(), 5)
        s["mean_epistemic"] = round(self.mean_epistemic(), 5)
        return s

    def to_dict(self, **kw: Any) -> Dict[str, Any]:
        inner_fn = getattr(self.inner, "to_dict", None)
        return {"kind": "grounded", "state_latent_dim": self.state_latent_dim,
                "inner": inner_fn(**kw) if callable(inner_fn) else {}}

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "grounded":
            return False
        if int(data.get("state_latent_dim", -1)) != self.state_latent_dim:
            return False
        inner_fn = getattr(self.inner, "load_dict", None)
        return bool(inner_fn(data.get("inner") or {})) if callable(inner_fn) else False

    def __getattr__(self, name: str) -> Any:
        """Delegate backend-specific extras (``concept_report``, ``concepts``, …) to the
        wrapped learner, so the grounded wrapper is a drop-in for every backend."""
        inner = self.__dict__.get("inner")
        if inner is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(inner, name)


# --------------------------------------------------------------------------- #
# Module-level persistence — learned dynamics survive restarts (Rule 7)
# --------------------------------------------------------------------------- #
def save_world_model(model: Any, path: str) -> Optional[str]:
    """Atomically persist any world model that exposes ``to_dict`` (best-effort)."""
    import json
    import os
    to_dict = getattr(model, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(to_dict(), f, default=str)
        os.replace(tmp, path)
        return path
    except Exception:  # noqa: BLE001 — persistence is a capability, never fatal
        return None


def load_world_model(model: Any, path: str) -> bool:
    """Restore a world model persisted by :func:`save_world_model` (best-effort)."""
    import json
    load_dict = getattr(model, "load_dict", None)
    if not callable(load_dict):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(load_dict(json.load(f)))
    except Exception:  # noqa: BLE001 — a corrupt snapshot must never block boot
        return False


# --------------------------------------------------------------------------- #
# Factory — pick the strongest learner the machine can support, never crash
# --------------------------------------------------------------------------- #
def build_world_model(backend: str = "auto", **kwargs: Any) -> WorldModel:
    """Build the best world model available. ``backend``:

    * ``"jepa"`` → :class:`~nyxara.mind.jepa_world_model.JEPAWorldModel` (hierarchical
      JEPA: latent-space prediction with an EMA target encoder, energy scoring,
      multi-scale horizons, latent planning) when numpy is present, else the stdlib
      learners — the strongest backend, the kernel's default;
    * ``"auto"`` / ``"transfer"`` → :class:`TransferWorldModel` (one shared encoder + concept
      hierarchy ⇒ cross-domain transfer) when numpy is present, else the pure-Python
      :class:`NeuralWorldModel` (and the kNN :class:`WorldModel` as the final floor);
    * ``"ensemble"`` → :class:`EnsembleWorldModel` (per-action deep ensembles) when numpy is
      present, else the stdlib learners;
    * ``"neural"`` → :class:`NeuralWorldModel`;
    * ``"knn"`` → :class:`WorldModel`.

    Unknown keyword args are filtered per backend so callers can pass a superset safely.
    """
    def _filtered(cls: type) -> Dict[str, Any]:
        import inspect
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
        return {k: v for k, v in kwargs.items() if k in allowed}

    def _stdlib_floor() -> WorldModel:
        try:
            return NeuralWorldModel(**_filtered(NeuralWorldModel))
        except Exception:  # noqa: BLE001
            return WorldModel(**_filtered(WorldModel))

    if backend == "jepa":
        if _HAS_NUMPY:
            try:
                from nyxara.mind.jepa_world_model import JEPAWorldModel
                return JEPAWorldModel(**_filtered(JEPAWorldModel))
            except Exception:  # noqa: BLE001 — fall through to the stdlib learners
                pass
        return _stdlib_floor()
    if backend in ("auto", "transfer"):
        if _HAS_NUMPY:
            try:
                return TransferWorldModel(**_filtered(TransferWorldModel))
            except Exception:  # noqa: BLE001 — fall through to the stdlib learners
                pass
        return _stdlib_floor()
    if backend == "ensemble":
        if _HAS_NUMPY:
            try:
                return EnsembleWorldModel(**_filtered(EnsembleWorldModel))
            except Exception:  # noqa: BLE001 — fall through to the stdlib learners
                pass
        return _stdlib_floor()
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

        # ---- transfer model: ONE shared encoder + concept hierarchy ⇒ cross-domain ---- #
        tm = build_world_model("auto", seed=0)
        assert isinstance(tm, TransferWorldModel)        # the default when numpy is present
        rng3 = random.Random(2)
        for _ in range(600):                             # master three actions thoroughly
            x = rng3.uniform(-10, 10)
            a = rng3.choice(["left", "right", "stay"])
            tm.observe((x,), a, (true_step(x, a),), reward=-abs(true_step(x, a)))
        # a BRAND-NEW action 'descend' that behaves exactly like 'left' — only 8 samples
        for _ in range(8):
            x = rng3.uniform(-10, 10)
            tm.observe((x,), "descend", (x - 1.0,), reward=-abs(x - 1.0))
        pt = tm.predict((4.0,), "descend")
        print(f"\ntransfer (4,descend): next={tuple(round(v,2) for v in pt.next_state)} "
              f"conf={pt.confidence:.2f}  (8 samples, want 3.0)")
        # few-shot transfer: 'descend' inherits 'left''s dynamics via its shared concept
        assert abs(pt.next_state[0] - 3.0) < 0.4 and pt.confidence > 0.5
        assert tm.concepts.concept_of("descend") == tm.concepts.concept_of("left")
        assert tm.predict((0.0,), "teleport").confidence == 0.0   # honest about the unknown
        print(tm.concept_report())
        print(f"transfer stats      : {tm.stats()}")
    else:
        print("\nensemble            : numpy absent → factory falls back (stdlib learners)")

    print("\nALL SELF-TESTS PASSED ✓")
