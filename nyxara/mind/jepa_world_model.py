"""NYXARA · mind/jepa_world_model.py — hierarchical JEPA latent world model (★★).

Next-token prediction wastes capacity on every irrelevant detail. Real anticipation
predicts the **abstract latent state** — "the glass will fall", not every pixel of the
fall. This module is NYXARA's own Joint-Embedding Predictive Architecture, trained by
her own hand-written numpy gradients, with the LLM nowhere in the loop:

* **Latent prediction, not reconstruction** — a context encoder ``f_θ`` maps an
  observation into a learned latent; an **EMA target encoder** ``f_θ̄`` (never touched
  by gradients — a structural stop-gradient) encodes the *next* observation; an
  ensemble of predictors ``g_φᵢ`` is trained to hit the target **in latent space**.
  The loss is a distance between embeddings — an **energy** — never a pixel/token
  reconstruction, so the latent is free to drop what doesn't matter.
* **Energy-based** — :meth:`JEPAWorldModel.energy` scores any hypothetical transition
  ``(state, action, next_state)``; low energy = plausible, high = implausible.
  :meth:`plausibility` squashes it through a self-calibrated scale into ``[0, 1]``.
* **Anti-collapse** — VICReg-style variance + covariance regularisation on both the
  context latents and the predictions (plus the EMA target) keeps the latent space
  from collapsing to a constant, and :meth:`stats` reports the live latent spread so
  the health of the space is honest and observable.
* **Multi-scale** — one predictor bank serves horizons ``(1, 2, 4, 8, 16)`` through a
  learned horizon embedding, and a second, **coarser hierarchy level** (H-JEPA) with
  its own encoder/target/predictors handles the long horizons in an abstract latent
  that drops even more detail — literally "the glass will fall" resolution.
* **Planning by energy minimisation** — :meth:`plan` runs a cross-entropy search over
  action sequences *entirely in latent space*, scoring imagined futures by predicted
  reward, goal distance and the predictors' own disagreement. No generation anywhere.
* **Curiosity** — :meth:`surprise` (realised energy against the self-calibrated scale)
  and a per-action learning-progress EMA give the growth loops an honest signal of
  where the world still surprises her.

Drop-in for :class:`~nyxara.mind.world_model.WorldModel` (same ``observe``/``predict``
surface ⇒ inherits ``rollout``/``counterfactual``/``intervene``), with symbolic or
odd-dimension transitions falling back to an exact-memory kNN. Requires numpy; the
factory in :mod:`mind.world_model` degrades to the stdlib learners without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

try:                                       # numpy is required for the JEPA proper
    import numpy as np
    _HAS_NUMPY = True
except Exception:                          # noqa: BLE001 — the factory degrades gracefully
    np = None  # type: ignore
    _HAS_NUMPY = False

from nyxara.mind.world_model import (
    Prediction, WorldModel, _pack_conf, _unrepr_action,
)

__all__ = ["JEPAWorldModel", "PlanResult"]

Action = Hashable
State = Tuple[Any, ...]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class PlanResult:
    """Outcome of a latent-space plan search (see :meth:`JEPAWorldModel.plan`)."""
    actions: List[Action] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    expected_reward: float = 0.0
    expected_energy: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"actions": [str(a) for a in self.actions],
                "expected_reward": round(self.expected_reward, 4),
                "expected_energy": round(self.expected_energy, 4),
                "confidence": round(self.confidence, 4),
                "steps": len(self.actions)}


# --------------------------------------------------------------------------- #
# Building blocks — numpy nets with externally-injected output gradients
# --------------------------------------------------------------------------- #
class _Net:
    """A numpy MLP (tanh hiddens, linear head) whose backward pass takes the output
    gradient from the caller — JEPA losses are composite (latent distance + VICReg),
    so the net cannot hardcode MSE the way the ensemble's ``_MLP`` does. Also serves
    as an EMA target network via :meth:`ema_from` (never gradient-updated)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int],
                 lr: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.dims = [in_dim, *hidden, out_dim]
        self.W: List[Any] = []
        self.b: List[Any] = []
        for i in range(len(self.dims) - 1):
            fan_in = self.dims[i]
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
        out = a @ self.W[-1] + self.b[-1]
        acts.append(out)
        return out, acts

    def backward(self, acts: List[Any], d_out: Any) -> Tuple[List[Any], List[Any], Any]:
        """Backprop an arbitrary ``dL/d(out)``; returns (gW, gb, dL/d(input))."""
        gW: List[Any] = [None] * len(self.W)
        gb: List[Any] = [None] * len(self.b)
        delta = d_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
            else:
                return gW, gb, delta @ self.W[0].T
        return gW, gb, delta @ self.W[0].T

    def adam(self, gW: List[Any], gb: List[Any],
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

    # ---- EMA target-network updates (the structural stop-gradient) ---- #
    def copy_from(self, src: "_Net") -> None:
        self.W = [w.copy() for w in src.W]
        self.b = [b.copy() for b in src.b]

    def ema_from(self, src: "_Net", tau: float) -> None:
        for i in range(len(self.W)):
            self.W[i] = tau * self.W[i] + (1.0 - tau) * src.W[i]
            self.b[i] = tau * self.b[i] + (1.0 - tau) * src.b[i]

    def to_dict(self) -> Dict[str, Any]:
        return {"dims": list(self.dims),
                "W": [w.tolist() for w in self.W],
                "b": [b.tolist() for b in self.b]}

    def load_dict(self, d: Dict[str, Any]) -> bool:
        if list(d.get("dims", [])) != list(self.dims):
            return False
        self.W = [np.asarray(w, dtype=np.float64) for w in d["W"]]
        self.b = [np.asarray(b, dtype=np.float64) for b in d["b"]]
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0
        return True


class _Linear:
    """A linear read-out with its own Adam. Used for the **detached** decoders
    (latent → observation, coarse → latent): they read the latent out for the
    contract's sake but never push a gradient back into the JEPA objective."""

    def __init__(self, in_dim: int, out_dim: int, lr: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.W = rng.standard_normal((in_dim, out_dim)) * math.sqrt(1.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.lr = lr
        self._mW = np.zeros_like(self.W)
        self._vW = np.zeros_like(self.W)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._t = 0

    def forward(self, X: Any) -> Any:
        return X @ self.W + self.b

    def step(self, X: Any, Y: Any, b1: float = 0.9, b2: float = 0.999,
             eps: float = 1e-8) -> float:
        n = max(1, X.shape[0])
        diff = self.forward(X) - Y
        gW = (2.0 / n) * (X.T @ diff)
        gb = (2.0 / n) * diff.sum(axis=0)
        self._t += 1
        bc1 = 1.0 - b1 ** self._t
        bc2 = 1.0 - b2 ** self._t
        self._mW = b1 * self._mW + (1 - b1) * gW
        self._vW = b2 * self._vW + (1 - b2) * (gW ** 2)
        self.W -= self.lr * (self._mW / bc1) / (np.sqrt(self._vW / bc2) + eps)
        self._mb = b1 * self._mb + (1 - b1) * gb
        self._vb = b2 * self._vb + (1 - b2) * (gb ** 2)
        self.b -= self.lr * (self._mb / bc1) / (np.sqrt(self._vb / bc2) + eps)
        return float(np.mean(diff ** 2))

    def to_dict(self) -> Dict[str, Any]:
        return {"in_dim": self.in_dim, "out_dim": self.out_dim,
                "W": self.W.tolist(), "b": self.b.tolist()}

    def load_dict(self, d: Dict[str, Any]) -> bool:
        if int(d.get("in_dim", -1)) != self.in_dim or \
                int(d.get("out_dim", -1)) != self.out_dim:
            return False
        self.W = np.asarray(d["W"], dtype=np.float64)
        self.b = np.asarray(d["b"], dtype=np.float64)
        return True


# --------------------------------------------------------------------------- #
# The hierarchical JEPA world model
# --------------------------------------------------------------------------- #
class JEPAWorldModel(WorldModel):
    """Hierarchical JEPA: predicts abstract latent state, never tokens or pixels.

    Fine level — encoder ``f_θ`` (obs → latent), EMA target ``f_θ̄``, K predictors
    ``g_φᵢ([z ⊕ e(action) ⊕ h(Δt)]) → ẑ'`` over multi-scale horizons; epistemic
    uncertainty = predictor disagreement. Coarse level (H-JEPA) — a second
    encoder/target/predictor stack over an even smaller latent, trained only on long
    horizons, so far futures are imagined at "the glass will fall" resolution.
    VICReg variance+covariance terms keep both latent spaces alive; detached linear
    read-outs keep the :class:`~nyxara.mind.world_model.WorldModel` contract
    (``predict`` still returns a next observation + reward/done) without ever letting
    a reconstruction gradient corrupt the latent objective. Requires numpy."""

    def __init__(self, *, latent_dim: int = 64, coarse_dim: int = 16,
                 enc_hidden: Sequence[int] = (128, 128),
                 pred_hidden: Sequence[int] = (128,),
                 n_predictors: int = 5,
                 horizons: Sequence[int] = (1, 2, 4, 8, 16),
                 coarse_from_horizon: int = 8,
                 embed_dim: int = 12, embed_lr: float = 0.05,
                 lr: float = 1e-3, batch: int = 64, iters: int = 6,
                 train_every: int = 1, ema_tau: float = 0.996,
                 var_coef: float = 1.0, cov_coef: float = 0.05,
                 var_floor: float = 1.0, uncertainty_penalty: float = 0.5,
                 pred_lr: Optional[float] = None, vic_on_pred: bool = True,
                 max_transitions: int = 100_000, experience_full: int = 64,
                 seed: int = 0) -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("JEPAWorldModel requires numpy")
        self.latent_dim = max(4, int(latent_dim))
        self.coarse_dim = max(2, int(coarse_dim))
        self.enc_hidden = tuple(enc_hidden)
        self.pred_hidden = tuple(pred_hidden)
        self.n_predictors = max(1, int(n_predictors))
        self.horizons = tuple(sorted({max(1, int(h)) for h in horizons})) or (1,)
        self.coarse_from_horizon = max(2, int(coarse_from_horizon))
        self.embed_dim = max(1, int(embed_dim))
        self.embed_lr = embed_lr
        self.lr = lr
        self.batch = max(4, int(batch))
        self.iters = max(1, int(iters))
        self.train_every = max(1, int(train_every))
        self.ema_tau = min(0.9999, max(0.5, float(ema_tau)))
        self.var_coef = max(0.0, float(var_coef))
        self.cov_coef = max(0.0, float(cov_coef))
        self.var_floor = max(1e-3, float(var_floor))
        self.uncertainty_penalty = max(0.0, float(uncertainty_penalty))
        self.pred_lr = float(pred_lr) if pred_lr is not None else lr
        self.vic_on_pred = bool(vic_on_pred)
        self.max_transitions = max(64, int(max_transitions))
        self.experience_full = max(1, int(experience_full))
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        # nets are built lazily once the observation dimension locks
        self._state_dim: Optional[int] = None
        self._enc: Optional[_Net] = None
        self._tgt: Optional[_Net] = None
        # per-horizon predictor banks — separate heads per Δt, so 1-step accuracy
        # never fights 16-step abstraction for the same weights (no interference)
        self._preds: Dict[int, List[_Net]] = {}
        self._head: Optional[_Net] = None          # [z ⊕ e(a)] → (reward, done)
        self._dec: Optional[_Linear] = None        # latent → obs (detached read-out)
        self._c_enc: Optional[_Net] = None         # latent → coarse
        self._c_tgt: Optional[_Net] = None
        self._c_preds: Dict[int, List[_Net]] = {}
        self._c_dec: Optional[_Linear] = None      # coarse → latent (detached read-out)

        # learned per-action embeddings + Adam moments
        self._embed: Dict[Action, Any] = {}
        self._embed_m: Dict[Action, Any] = {}
        self._embed_v: Dict[Action, Any] = {}
        self._embed_t: Dict[Action, int] = {}
        self._own_samples: Dict[Action, int] = {}

        # ordered sequence replay buffer (segments = contiguous episodes)
        self._S: List[List[float]] = []
        self._A: List[Action] = []
        self._NS: List[List[float]] = []
        self._R: List[float] = []
        self._D: List[bool] = []
        self._SEG: List[int] = []
        self._P: List[float] = []                  # horizon-1 replay priorities
        self._seg_id = 0
        self._last_ns: Optional[Tuple[float, ...]] = None
        self._last_done = True

        # running input standardisation (Welford — O(d) per observe, exact enough)
        self._norm_n = 0
        self._norm_mean: Optional[Any] = None
        self._norm_M2: Optional[Any] = None

        # calibration — every scale is learned from what actually happens
        self.err_ema = 1.0
        self.energy_scale = 0.5                    # EMA of realised energies (floor 0.05)
        self._epi_scale = 0.5
        self.dev_margin = 2.0
        self.samples = 0
        self.last_surprise = 0.0
        self._lp: Dict[Action, float] = {}         # per-action learning-progress EMA
        self._last_z_std = 0.0                     # live anti-collapse health readout
        self._steps = 0                            # completed training steps
        self._rr = 0                               # round-robin pointer over horizons > 1
        self._count = 0
        self._since_train = 0
        # exact-memory fallback for symbolic / odd-dimension transitions
        self._knn = WorldModel(max_transitions=max_transitions)

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._count

    def actions(self) -> List[Action]:
        return list({*self._embed.keys(), *self._knn.actions()})

    @staticmethod
    def _numeric(state: Sequence[Any]) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in state)

    def _build_nets(self, dim: int) -> None:
        self._state_dim = dim
        s = self.seed
        self._enc = _Net(dim, self.latent_dim, self.enc_hidden, self.lr, s + 11)
        self._tgt = _Net(dim, self.latent_dim, self.enc_hidden, self.lr, s + 11)
        self._tgt.copy_from(self._enc)             # target starts equal, then EMA-lags
        in_dim = self.latent_dim + self.embed_dim
        self._preds = {
            h: [_Net(in_dim, self.latent_dim, self.pred_hidden, self.pred_lr,
                     s + 101 * (17 * h + i + 1)) for i in range(self.n_predictors)]
            for h in self.horizons
        }
        self._head = _Net(self.latent_dim + self.embed_dim, 2, (32,), self.lr, s + 7)
        self._dec = _Linear(self.latent_dim, dim, 1e-2, s + 5)
        self._c_enc = _Net(self.latent_dim, self.coarse_dim, (32,), self.lr, s + 23)
        self._c_tgt = _Net(self.latent_dim, self.coarse_dim, (32,), self.lr, s + 23)
        self._c_tgt.copy_from(self._c_enc)
        c_in = self.coarse_dim + self.embed_dim
        n_c = min(3, self.n_predictors)
        self._c_preds = {
            h: [_Net(c_in, self.coarse_dim, (64,), self.pred_lr,
                     s + 211 * (13 * h + i + 1)) for i in range(n_c)]
            for h in self.horizons if h >= self.coarse_from_horizon
        }
        self._c_dec = _Linear(self.coarse_dim, self.latent_dim, 1e-2, s + 3)

    def _ensure_embed(self, action: Action) -> None:
        if action in self._embed:
            return
        self._embed[action] = 0.1 * self._rng.standard_normal(self.embed_dim)
        self._embed_m[action] = np.zeros(self.embed_dim)
        self._embed_v[action] = np.zeros(self.embed_dim)
        self._embed_t[action] = 0

    def _adam_embed(self, action: Action, g: Any, b1: float = 0.9, b2: float = 0.999,
                    eps: float = 1e-8) -> None:
        self._embed_t[action] += 1
        t = self._embed_t[action]
        self._embed_m[action] = b1 * self._embed_m[action] + (1 - b1) * g
        self._embed_v[action] = b2 * self._embed_v[action] + (1 - b2) * (g * g)
        mhat = self._embed_m[action] / (1 - b1 ** t)
        vhat = self._embed_v[action] / (1 - b2 ** t)
        self._embed[action] -= self.embed_lr * mhat / (np.sqrt(vhat) + eps)

    # ---- normalisation ---- #
    def _norm_update(self, x: Any) -> None:
        if self._norm_mean is None:
            self._norm_mean = np.zeros(len(x))
            self._norm_M2 = np.zeros(len(x))
        self._norm_n += 1
        d = x - self._norm_mean
        self._norm_mean = self._norm_mean + d / self._norm_n
        self._norm_M2 = self._norm_M2 + d * (x - self._norm_mean)

    def _norm_std(self) -> Any:
        if self._norm_n < 2:
            return np.ones_like(self._norm_mean)
        return np.sqrt(np.maximum(self._norm_M2 / self._norm_n, 1e-12)) + 1e-6

    def _normalize(self, X: Any) -> Any:
        return (X - self._norm_mean) / self._norm_std()

    def _denormalize(self, Xn: Any) -> Any:
        return Xn * self._norm_std() + self._norm_mean

    def _ready(self) -> bool:
        return self._enc is not None and self._steps > 0 and self._norm_n >= 2

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
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
            self._build_nets(len(state))
        if len(state) != self._state_dim or len(next_state) != self._state_dim:
            self._knn.observe(state, action, next_state, reward, done)
            self._count += 1
            return
        self._track_realized(state, action, next_state)
        self._ensure_embed(action)
        # curiosity: how surprising was reality, per the CURRENT model?
        self.last_surprise = self.surprise(state, action, next_state)
        self._lp[action] = 0.9 * self._lp.get(action, self.last_surprise) \
            + 0.1 * self.last_surprise
        # segment bookkeeping: contiguous episodes make multi-step pairs mineable
        if self._last_done or self._last_ns is None or \
                any(abs(a - b) > 1e-9 for a, b in zip(state, self._last_ns)):
            self._seg_id += 1
        self._last_ns = next_state
        self._last_done = bool(done)
        self._S.append(list(state))
        self._A.append(action)
        self._NS.append(list(next_state))
        self._R.append(float(reward))
        self._D.append(bool(done))
        self._SEG.append(self._seg_id)
        self._P.append(max(self._P) if self._P else 1.0)
        if len(self._S) > self.max_transitions:
            for buf in (self._S, self._A, self._NS, self._R, self._D, self._SEG, self._P):
                buf.pop(0)
        self._norm_update(np.asarray(state, dtype=np.float64))
        self._norm_update(np.asarray(next_state, dtype=np.float64))
        self._own_samples[action] = self._own_samples.get(action, 0) + 1
        self.samples += 1
        self._count += 1
        self._since_train += 1
        if self._since_train >= self.train_every:
            self._train()
            self._since_train = 0

    def _train_horizons(self) -> List[int]:
        """Per-observe schedule: always horizon 1, plus ONE longer horizon in
        round-robin — keeps the per-turn cost flat while every scale keeps learning.
        :meth:`consolidate` trains all horizons at once during idle time."""
        longer = [h for h in self.horizons if h > 1]
        if not longer:
            return [1]
        h = longer[self._rr % len(longer)]
        self._rr += 1
        return [1, h]

    def _vicreg(self, Z: Any) -> Tuple[float, Any]:
        """VICReg variance + covariance loss and its closed-form gradient w.r.t. Z."""
        n, d = Z.shape
        mu = Z.mean(axis=0)
        Zc = Z - mu
        var = Zc.var(axis=0)
        std = np.sqrt(var + 1e-4)
        active = (std < self.var_floor).astype(np.float64)
        l_var = float(np.mean(np.maximum(0.0, self.var_floor - std)))
        dZ_var = -(active / d) * Zc / (max(1, n) * std)
        if n > 1:
            C = (Zc.T @ Zc) / (n - 1)
            off = C - np.diag(np.diag(C))
            l_cov = float((off ** 2).sum() / d)
            dZ_cov = (4.0 / (d * (n - 1))) * (Zc @ off)
        else:
            l_cov, dZ_cov = 0.0, np.zeros_like(Z)
        loss = self.var_coef * l_var + self.cov_coef * l_cov
        return loss, self.var_coef * dZ_var + self.cov_coef * dZ_cov

    def _sample_batch(self, horizon: int, prio: Optional[Any]) -> Optional[
            Tuple[Any, Any, List[int], List[List[Action]]]]:
        """Draw a minibatch of ``horizon``-step pairs from the sequence buffer.

        Horizon 1 uses prioritised replay (surprising transitions rehearse more);
        longer horizons sample start rows uniformly and keep only windows that stay
        inside one segment (contiguous ids already exclude interior ``done``)."""
        n = len(self._S)
        if n < 8:
            return None
        bs = min(self.batch, n)
        if horizon == 1:
            if prio is not None:
                idx = [int(i) for i in self._rng.choice(n, size=bs, p=prio)]
            else:
                idx = [int(i) for i in self._rng.integers(0, n, size=bs)]
        else:
            hi = n - horizon + 1
            if hi < 8:
                return None
            cand = self._rng.integers(0, hi, size=bs * 3)
            idx = [int(j) for j in cand
                   if self._SEG[j] == self._SEG[j + horizon - 1]][:bs]
            if len(idx) < 8:
                return None
        X = np.asarray([self._S[j] for j in idx], dtype=np.float64)
        Y = np.asarray([self._NS[j + horizon - 1] for j in idx], dtype=np.float64)
        windows = [[self._A[j + k] for k in range(horizon)] for j in idx]
        return X, Y, idx, windows

    def _train(self, *, horizons: Optional[Sequence[int]] = None,
               iters: Optional[int] = None) -> int:
        if self._enc is None or len(self._S) < 8 or self._norm_n < 2:
            return 0
        prio = None
        p = np.asarray(self._P, dtype=np.float64)
        if p.size:
            p = (p + 1e-3) ** 0.6
            prio = p / p.sum()
        batches = 0
        for h in (horizons or self._train_horizons()):
            if h not in self._preds:
                continue
            for _ in range(iters or self.iters):
                got = self._sample_batch(h, prio)
                if got is None:
                    break
                X, Y, idx, windows = got
                per_sample = self._step_fine(X, Y, windows, h, idx)
                if h == 1 and per_sample is not None:
                    for r, j in enumerate(idx):
                        if j < len(self._P):
                            self._P[j] = float(per_sample[r])
                batches += 1
        return batches

    def _step_fine(self, X: Any, Y: Any, windows: List[List[Action]],
                   h: int, idx: List[int]) -> Optional[Any]:
        """One JEPA gradient step: predictors chase the EMA target in latent space,
        VICReg keeps the space alive, and everything downstream of the latent
        objective (reward head, decoders, coarse level) trains detached."""
        n = X.shape[0]
        d = self.latent_dim
        Xn = self._normalize(X)
        Yn = self._normalize(Y)
        Z, actsE = self._enc.forward(Xn)
        Zt, _ = self._tgt.forward(Yn)              # EMA target — no grad, by construction
        Ea = np.asarray([np.mean([self._embed[a] for a in w], axis=0) for w in windows])
        Inp = np.concatenate([Z, Ea], axis=1)
        bank = self._preds[h]
        K = len(bank)
        # residual prediction: each predictor learns the latent DELTA (ẑ' = z + gᵢ(...)),
        # mirroring how every other NYXARA dynamics model predicts state deltas —
        # the identity is free, so capacity goes entirely into "what changes".
        P_list: List[Any] = []
        acts_list: List[List[Any]] = []
        for pred in bank:
            D_i, acts_i = pred.forward(Inp)
            P_list.append(Z + D_i)
            acts_list.append(acts_i)
        Pbar = np.mean(P_list, axis=0)
        dPbar = self._vicreg(Pbar)[1] if self.vic_on_pred else np.zeros_like(Pbar)
        _, dZ_vic = self._vicreg(Z)
        dInp_sum = np.zeros_like(Inp)
        dZ_res = np.zeros_like(Z)                  # gradient through the skip connection
        loss_pred = 0.0
        for P_i, acts_i, pred in zip(P_list, acts_list, bank):
            diff = P_i - Zt
            loss_pred += float(np.mean(diff ** 2)) / K
            d_out = (2.0 / (K * n * d)) * diff + dPbar / K
            gW, gb, dIn = pred.backward(acts_i, d_out)
            pred.adam(gW, gb)
            dInp_sum += dIn
            dZ_res += d_out
        dZ = dInp_sum[:, :d] + dZ_res + dZ_vic
        gW, gb, _ = self._enc.backward(actsE, dZ)
        self._enc.adam(gW, gb)
        # learned action embeddings: each window member shares the row's gradient
        dEa = dInp_sum[:, d:d + self.embed_dim]
        grad: Dict[Action, Any] = {}
        cnt: Dict[Action, int] = {}
        for r, w in enumerate(windows):
            share = dEa[r] / len(w)
            for a in w:
                if a not in grad:
                    grad[a] = np.zeros(self.embed_dim)
                    cnt[a] = 0
                grad[a] += share
                cnt[a] += 1
        for a, g in grad.items():
            self._adam_embed(a, g / max(1, cnt[a]))
        self._tgt.ema_from(self._enc, self.ema_tau)
        # calibration from what actually happened this batch
        per_sample = np.mean((Pbar - Zt) ** 2, axis=1)
        energies = np.sqrt(np.sum((Pbar - Zt) ** 2, axis=1) / d)
        self.err_ema = 0.9 * self.err_ema + 0.1 * loss_pred
        self.energy_scale = max(0.05, 0.99 * self.energy_scale
                                + 0.01 * 2.0 * float(np.mean(energies)))
        self._last_z_std = float(np.mean(Z.std(axis=0)))
        self._steps += 1
        if h == 1:
            # reward/done head + observation read-out — both detached from the latent loss
            self._step_head_dec(Z, Zt, Yn, windows, idx)
        if h >= self.coarse_from_horizon:
            self._step_coarse(Z, Zt, windows, h)
        return per_sample

    def _step_head_dec(self, Z: Any, Zt: Any, Yn: Any,
                       windows: List[List[Action]], idx: List[int]) -> None:
        """Train the detached read-outs (reward/done head + latent→obs decoder).
        ``Z``/``Zt`` are used as values only — no gradient reaches the encoders."""
        n = Z.shape[0]
        Ea = np.asarray([self._embed[w[0]] for w in windows])
        rd = np.asarray([[self._R[j], 1.0 if self._D[j] else 0.0] for j in idx],
                        dtype=np.float64)
        Inp = np.concatenate([Z, Ea], axis=1)
        out, acts = self._head.forward(Inp)
        d_out = (2.0 / (n * 2)) * (out - rd)
        gW, gb, _ = self._head.backward(acts, d_out)
        self._head.adam(gW, gb)
        self._dec.step(Zt, Yn)                     # sg(target latent) → observation

    def _step_coarse(self, Z: Any, Zt: Any, windows: List[List[Action]], h: int) -> None:
        """One H-JEPA step on the coarse level. Inputs are detached fine latents —
        the hierarchy learns *about* the fine space without deforming it."""
        bank = self._c_preds.get(h)
        if bank is None:
            return
        n, d_c = Z.shape[0], self.coarse_dim
        C, actsC = self._c_enc.forward(Z)
        Ct, _ = self._c_tgt.forward(Zt)
        Ea = np.asarray([np.mean([self._embed[a] for a in w], axis=0) for w in windows])
        Inp = np.concatenate([C, Ea], axis=1)
        K = len(bank)
        dInp_sum = np.zeros_like(Inp)
        dC_res = np.zeros_like(C)
        for pred in bank:
            D_i, acts_i = pred.forward(Inp)        # residual: ĉ' = c + q(...)
            d_out = (2.0 / (K * n * d_c)) * ((C + D_i) - Ct)
            gW, gb, dIn = pred.backward(acts_i, d_out)
            pred.adam(gW, gb)
            dInp_sum += dIn
            dC_res += d_out
        _, dC_vic = self._vicreg(C)
        gW, gb, _ = self._c_enc.backward(actsC, dInp_sum[:, :d_c] + dC_res + dC_vic)
        self._c_enc.adam(gW, gb)
        self._c_tgt.ema_from(self._c_enc, self.ema_tau)
        self._c_dec.step(Ct, Zt)                   # sg(coarse target) → fine latent

    def consolidate(self, iters: int = 12) -> int:
        """Offline rehearsal for idle ticks: train EVERY horizon (fine + coarse)."""
        if iters <= 0:
            return 0
        return self._train(horizons=list(self.horizons), iters=max(1, int(iters)))

    # ------------------------------------------------------------------ #
    # prediction — the WorldModel contract, served from the latent
    # ------------------------------------------------------------------ #
    def _encode(self, state: Sequence[float]) -> Any:
        xn = self._normalize(np.asarray(state, dtype=np.float64))
        return self._enc.forward(xn[None, :])[0][0], xn

    def _fine_predict_latent(self, z: Any, action: Action, h: int) -> Tuple[Any, Any]:
        """All fine predictors' guesses for f(s') given (z, action, Δt). → (K×d, mean).
        Residual: each guess is ``z + gᵢ(...)`` — predictors model the latent delta."""
        inp = np.concatenate([z, self._embed[action]])[None, :]
        outs = z + np.asarray([p.forward(inp)[0][0] for p in self._preds[h]])
        return outs, outs.mean(axis=0)

    def _nearest_horizon(self, horizon: int) -> int:
        below = [h for h in self.horizons if h <= horizon]
        return max(below) if below else self.horizons[0]

    def _can_model(self, state: State, action: Action) -> bool:
        return (self._ready() and action in self._embed
                and self._numeric(state) and len(state) == self._state_dim)

    def predict(self, state: Sequence[Any], action: Action) -> Prediction:
        state = tuple(state)
        if not self._can_model(state, action):
            return self._knn.predict(state, action)
        z, xn = self._encode(state)
        outs, zbar = self._fine_predict_latent(z, action, 1)
        ns_n = self._dec.forward(zbar[None, :])[0]
        next_state: State = tuple(float(v) for v in self._denormalize(ns_n))
        dec_outs = self._dec.forward(outs)                   # decode EVERY member
        epistemic = float(np.mean(dec_outs.std(axis=0) * self._norm_std()))
        epistemic_norm = float(np.mean(outs.std(axis=0)))
        head_in = np.concatenate([z, self._embed[action]])[None, :]
        rd = self._head.forward(head_in)[0][0]
        reward = float(rd[0])
        done_prob = max(0.0, min(1.0, float(rd[1])))
        deviation = float(np.mean(np.abs(xn)))
        self._epi_scale = max(0.05, 0.99 * self._epi_scale + 0.01 * 2.0 * epistemic_norm)
        reliability = _pack_conf(epistemic_norm, deviation,
                                 epi_scale=self._epi_scale, dev_margin=self.dev_margin)
        own = self._own_samples.get(action, 0)
        experience = min(1.0, own / float(self.experience_full))
        fit = 1.0 / (1.0 + self.err_ema)
        conf = max(0.0, min(1.0, experience * fit * reliability))
        return Prediction(next_state=next_state, reward=reward, confidence=conf,
                          neighbors=own, epistemic=epistemic, done_prob=done_prob)

    def predict_horizon(self, state: Sequence[Any], action: Action,
                        horizon: int = 1) -> Prediction:
        """Predict ``horizon`` steps of repeating ``action`` in ONE latent jump.
        Long horizons route through the coarse hierarchy level."""
        state = tuple(state)
        horizon = max(1, int(horizon))
        if horizon == 1 or not self._can_model(state, action):
            return self.predict(state, action)
        h = self._nearest_horizon(horizon)
        z, xn = self._encode(state)
        if h in self._c_preds and self._c_enc is not None:
            c = self._c_enc.forward(z[None, :])[0][0]
            inp = np.concatenate([c, self._embed[action]])[None, :]
            outs_c = c + np.asarray([p.forward(inp)[0][0] for p in self._c_preds[h]])
            z_next = self._c_dec.forward(outs_c.mean(axis=0)[None, :])[0]
            dec_outs = self._dec.forward(self._c_dec.forward(outs_c))
            epistemic_norm = float(np.mean(outs_c.std(axis=0)))
        else:
            outs, zbar = self._fine_predict_latent(z, action, h)
            z_next = zbar
            dec_outs = self._dec.forward(outs)
            epistemic_norm = float(np.mean(outs.std(axis=0)))
        ns_n = self._dec.forward(z_next[None, :])[0]
        next_state: State = tuple(float(v) for v in self._denormalize(ns_n))
        epistemic = float(np.mean(dec_outs.std(axis=0) * self._norm_std()))
        head_in = np.concatenate([z, self._embed[action]])[None, :]
        rd = self._head.forward(head_in)[0][0]
        deviation = float(np.mean(np.abs(xn)))
        reliability = _pack_conf(epistemic_norm, deviation,
                                 epi_scale=self._epi_scale, dev_margin=self.dev_margin)
        own = self._own_samples.get(action, 0)
        conf = max(0.0, min(1.0, min(1.0, own / float(self.experience_full))
                            * (1.0 / (1.0 + self.err_ema)) * reliability))
        return Prediction(next_state=next_state, reward=float(rd[0]) * horizon,
                          confidence=conf, neighbors=own, epistemic=epistemic,
                          done_prob=max(0.0, min(1.0, float(rd[1]))))

    def imagine_multiscale(self, state: Sequence[Any],
                           action: Action) -> Dict[int, Prediction]:
        """One prediction per trained horizon — the multi-scale imagination surface."""
        return {h: self.predict_horizon(state, action, h) for h in self.horizons}

    # ------------------------------------------------------------------ #
    # energy — score ANY hypothetical transition, no generation involved
    # ------------------------------------------------------------------ #
    def energy(self, state: Sequence[Any], action: Action,
               next_state: Sequence[Any], *, horizon: int = 1) -> float:
        """Latent-space energy of ``(state, action, next_state)``: how far the
        predicted latent lands from the target encoder's latent of what (allegedly)
        happened. Low = plausible. ``inf`` when the model honestly cannot say."""
        state = tuple(state)
        next_state = tuple(next_state)
        if not (self._can_model(state, action) and self._numeric(next_state)
                and len(next_state) == self._state_dim):
            return float("inf")
        h = self._nearest_horizon(max(1, int(horizon)))
        z, _ = self._encode(state)
        yn = self._normalize(np.asarray(next_state, dtype=np.float64))
        zt = self._tgt.forward(yn[None, :])[0][0]
        _, zbar = self._fine_predict_latent(z, action, h)
        return float(np.linalg.norm(zbar - zt) / math.sqrt(self.latent_dim))

    def plausibility(self, state: Sequence[Any], action: Action,
                     next_state: Sequence[Any], *, horizon: int = 1) -> float:
        """``exp(-energy / scale)`` with a self-calibrated scale → [0, 1]."""
        e = self.energy(state, action, next_state, horizon=horizon)
        if not math.isfinite(e):
            return 0.0
        return max(0.0, min(1.0, math.exp(-e / max(1e-6, self.energy_scale))))

    def surprise(self, state: Sequence[Any], action: Action,
                 next_state: Sequence[Any]) -> float:
        """How surprising reality was, in [0, 1] — the curiosity signal. An unknown
        (state, action) is maximally surprising: she knows she cannot anticipate it."""
        e = self.energy(state, action, next_state)
        if not math.isfinite(e):
            return 1.0
        return max(0.0, min(1.0, 1.0 - math.exp(-e / max(1e-6, self.energy_scale))))

    def learning_progress_by_action(self) -> Dict[Action, float]:
        """Per-action surprise EMA — where the world still surprises her most."""
        return dict(self._lp)

    # ------------------------------------------------------------------ #
    # planning — cross-entropy search entirely in latent space
    # ------------------------------------------------------------------ #
    def plan(self, state: Sequence[Any], *, goal_state: Optional[Sequence[Any]] = None,
             candidate_actions: Optional[Sequence[Action]] = None, horizon: int = 8,
             n_samples: int = 64, top_frac: float = 0.25, iters: int = 3) -> PlanResult:
        """Plan by energy minimisation: sample action sequences, imagine them forward
        **in latent space only**, score by predicted reward − goal distance −
        predictor disagreement, refit the sampling distribution on the elites."""
        state = tuple(state)
        acts = [a for a in (candidate_actions if candidate_actions is not None
                            else list(self._embed.keys())) if a in self._embed]
        if not acts or not self._ready() or not self._numeric(state) \
                or len(state) != (self._state_dim or -1):
            return PlanResult()
        horizon = max(1, int(horizon))
        n_samples = max(8, int(n_samples))
        n_a = len(acts)
        E = np.asarray([self._embed[a] for a in acts])
        z0, _ = self._encode(state)
        z_goal = None
        if goal_state is not None and self._numeric(tuple(goal_state)) \
                and len(goal_state) == self._state_dim:
            z_goal, _ = self._encode(tuple(goal_state))
        bank1 = self._preds[1]
        probs = np.full((horizon, n_a), 1.0 / n_a)
        n_elite = max(2, int(n_samples * max(0.05, min(1.0, top_frac))))
        best_seq: Optional[List[int]] = None
        best_score = -float("inf")
        best_dis = 0.0
        for _ in range(max(1, int(iters))):
            seqs = np.stack([self._rng.choice(n_a, size=n_samples, p=probs[t])
                             for t in range(horizon)], axis=1)   # (n_samples × horizon)
            Z = np.tile(z0, (n_samples, 1))
            total_r = np.zeros(n_samples)
            total_dis = np.zeros(n_samples)
            for t in range(horizon):
                Ea = E[seqs[:, t]]
                Inp = np.concatenate([Z, Ea], axis=1)
                outs = Z + np.asarray([p.forward(Inp)[0] for p in bank1])
                total_dis += outs.std(axis=0).mean(axis=1)
                rd = self._head.forward(np.concatenate([Z, Ea], axis=1))[0]
                total_r += rd[:, 0]
                Z = outs.mean(axis=0)
            score = total_r - self.uncertainty_penalty * total_dis
            if z_goal is not None:
                score = score - np.linalg.norm(Z - z_goal, axis=1)
            elite = np.argsort(score)[-n_elite:]
            top = int(elite[-1])
            if float(score[top]) > best_score:
                best_score = float(score[top])
                best_seq = [int(i) for i in seqs[top]]
                best_dis = float(total_dis[top] / horizon)
            # refit: elite action frequencies per step, with smoothing so no action dies
            for t in range(horizon):
                freq = np.bincount(seqs[elite, t], minlength=n_a).astype(np.float64)
                freq += 0.1
                probs[t] = freq / freq.sum()
        if best_seq is None:
            return PlanResult()
        # decode the winning imagined future into real predictions
        actions = [acts[i] for i in best_seq]
        predictions: List[Prediction] = []
        cur: Sequence[Any] = state
        total_reward = 0.0
        for a in actions:
            p = self.predict(cur, a)
            predictions.append(p)
            total_reward += p.reward
            cur = p.next_state
        confidence = max(0.0, min(1.0,
                                  math.exp(-best_dis / max(1e-6, self._epi_scale))
                                  * min(1.0, self.samples / float(self.experience_full))))
        return PlanResult(actions=actions, predictions=predictions,
                          expected_reward=total_reward, expected_energy=best_dis,
                          confidence=confidence)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "transitions": self._count,
            "backend": "jepa-hierarchical",
            "latent_dim": self.latent_dim,
            "coarse_dim": self.coarse_dim,
            "n_predictors": self.n_predictors,
            "horizons": list(self.horizons),
            "hierarchy": {"coarse_dim": self.coarse_dim,
                          "from_horizon": self.coarse_from_horizon},
            "train_steps": self._steps,
            "energy_scale": round(self.energy_scale, 4),
            "err_ema": round(self.err_ema, 5),
            "collapse_std": round(self._last_z_std, 4),
            "last_surprise": round(self.last_surprise, 4),
            "plan_capable": True,
            "actions": {str(a): n for a, n in self._own_samples.items()},
            "symbolic_transitions": len(self._knn),
        }

    # ------------------------------------------------------------------ #
    # persistence — the learned latent world survives restarts (Rule 7)
    # ------------------------------------------------------------------ #
    def to_dict(self, *, max_buffer: int = 6000, **_kw: Any) -> Dict[str, Any]:
        keep = min(len(self._S), max_buffer)
        return {
            "kind": "jepa", "state_dim": self._state_dim, "count": self._count,
            "latent_dim": self.latent_dim, "coarse_dim": self.coarse_dim,
            "enc_hidden": list(self.enc_hidden), "pred_hidden": list(self.pred_hidden),
            "n_predictors": self.n_predictors, "horizons": list(self.horizons),
            "embed_dim": self.embed_dim,
            "enc": self._enc.to_dict() if self._enc else None,
            "tgt": self._tgt.to_dict() if self._tgt else None,
            "preds": {str(h): [p.to_dict() for p in bank]
                      for h, bank in self._preds.items()},
            "head": self._head.to_dict() if self._head else None,
            "dec": self._dec.to_dict() if self._dec else None,
            "c_enc": self._c_enc.to_dict() if self._c_enc else None,
            "c_tgt": self._c_tgt.to_dict() if self._c_tgt else None,
            "c_preds": {str(h): [p.to_dict() for p in bank]
                        for h, bank in self._c_preds.items()},
            "c_dec": self._c_dec.to_dict() if self._c_dec else None,
            "embed": {repr(a): [float(x) for x in e] for a, e in self._embed.items()},
            "own_samples": {repr(a): int(n) for a, n in self._own_samples.items()},
            "lp": {repr(a): float(v) for a, v in self._lp.items()},
            "S": self._S[-keep:], "A": [repr(a) for a in self._A[-keep:]],
            "NS": self._NS[-keep:], "R": self._R[-keep:],
            "D": self._D[-keep:], "SEG": self._SEG[-keep:], "P": self._P[-keep:],
            "norm_n": self._norm_n,
            "norm_mean": self._norm_mean.tolist() if self._norm_mean is not None else None,
            "norm_M2": self._norm_M2.tolist() if self._norm_M2 is not None else None,
            "err_ema": self.err_ema, "energy_scale": self.energy_scale,
            "epi_scale": self._epi_scale, "samples": self.samples,
            "steps": self._steps, "seg_id": self._seg_id,
            "knn": self._knn.to_dict(),
        }

    def load_dict(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict) or data.get("kind") != "jepa":
            return False
        if int(data.get("latent_dim", -1)) != self.latent_dim or \
                int(data.get("coarse_dim", -1)) != self.coarse_dim or \
                list(data.get("enc_hidden", [])) != list(self.enc_hidden) or \
                list(data.get("pred_hidden", [])) != list(self.pred_hidden) or \
                int(data.get("n_predictors", -1)) != self.n_predictors or \
                list(data.get("horizons", [])) != list(self.horizons) or \
                int(data.get("embed_dim", -1)) != self.embed_dim:
            return False
        restored = False
        state_dim = data.get("state_dim")
        if state_dim is not None and data.get("enc"):
            self._build_nets(int(state_dim))
            nets = [(self._enc, data.get("enc")), (self._tgt, data.get("tgt")),
                    (self._head, data.get("head")),
                    (self._c_enc, data.get("c_enc")), (self._c_tgt, data.get("c_tgt"))]
            ok = all(n is not None and d is not None and n.load_dict(d) for n, d in nets)
            preds_d = data.get("preds") or {}
            c_preds_d = data.get("c_preds") or {}
            ok = ok and set(preds_d) == {str(h) for h in self._preds} and all(
                len(preds_d[str(h)]) == len(bank)
                and all(p.load_dict(d) for p, d in zip(bank, preds_d[str(h)]))
                for h, bank in self._preds.items())
            ok = ok and set(c_preds_d) == {str(h) for h in self._c_preds} and all(
                len(c_preds_d[str(h)]) == len(bank)
                and all(p.load_dict(d) for p, d in zip(bank, c_preds_d[str(h)]))
                for h, bank in self._c_preds.items())
            ok = ok and self._dec is not None and data.get("dec") is not None \
                and self._dec.load_dict(data["dec"])
            ok = ok and self._c_dec is not None and data.get("c_dec") is not None \
                and self._c_dec.load_dict(data["c_dec"])
            if ok:
                for key, e in (data.get("embed") or {}).items():
                    a = _unrepr_action(key)
                    self._embed[a] = np.asarray(e, dtype=np.float64)
                    self._embed_m[a] = np.zeros(self.embed_dim)
                    self._embed_v[a] = np.zeros(self.embed_dim)
                    self._embed_t[a] = 0
                self._own_samples = {_unrepr_action(k): int(v)
                                     for k, v in (data.get("own_samples") or {}).items()}
                self._lp = {_unrepr_action(k): float(v)
                            for k, v in (data.get("lp") or {}).items()}
                self._S = [list(map(float, r)) for r in data.get("S", [])]
                self._A = [_unrepr_action(k) for k in data.get("A", [])]
                self._NS = [list(map(float, r)) for r in data.get("NS", [])]
                self._R = [float(x) for x in data.get("R", [])]
                self._D = [bool(x) for x in data.get("D", [])]
                self._SEG = [int(x) for x in data.get("SEG", [])]
                self._P = [float(x) for x in data.get("P", [])] or [1.0] * len(self._S)
                self._norm_n = int(data.get("norm_n", 0))
                if data.get("norm_mean") is not None:
                    self._norm_mean = np.asarray(data["norm_mean"], dtype=np.float64)
                    self._norm_M2 = np.asarray(data["norm_M2"], dtype=np.float64)
                self.err_ema = float(data.get("err_ema", 1.0))
                self.energy_scale = float(data.get("energy_scale", 0.5))
                self._epi_scale = float(data.get("epi_scale", 0.5))
                self.samples = int(data.get("samples", len(self._S)))
                self._steps = int(data.get("steps", 0))
                self._seg_id = int(data.get("seg_id", 0))
                self._count = int(data.get("count", len(self._S)))
                restored = True
        if data.get("knn"):
            restored = self._knn.load_dict(data["knn"]) or restored
        return restored


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import json
    import random as _random
    import tempfile

    print("=" * 70)
    print("NYXARA hierarchical-JEPA world-model self-test")
    print("=" * 70)
    if not _HAS_NUMPY:
        raise SystemExit("numpy required for the self-test")

    MOVES = {"up": (0.0, 1.0), "down": (0.0, -1.0),
             "left": (-1.0, 0.0), "right": (1.0, 0.0)}

    def true_step(p: Tuple[float, float], a: str) -> Tuple[float, float]:
        dx, dy = MOVES[a]
        return (p[0] + dx, p[1] + dy)

    model = JEPAWorldModel(seed=0)
    rng = _random.Random(0)
    for _ in range(60):                                  # trajectories, not iid samples
        pos = (float(rng.randint(-6, 6)), float(rng.randint(-6, 6)))
        for t in range(20):
            a = rng.choice(list(MOVES))
            nxt = true_step(pos, a)
            done = t == 19
            model.observe(pos, a, nxt, reward=-abs(nxt[0]) - abs(nxt[1]), done=done)
            pos = nxt
    model.consolidate(iters=60)

    p = model.predict((4.0, 0.0), "left")
    print(f"1-step  predict((4,0),'left') → {tuple(round(v, 2) for v in p.next_state)}"
          f"  conf={p.confidence:.2f}  (want ≈(3,0))")
    assert abs(p.next_state[0] - 3.0) < 0.6 and abs(p.next_state[1]) < 0.6

    p4 = model.predict_horizon((4.0, 0.0), "left", 4)
    print(f"4-step  predict_horizon → {tuple(round(v, 2) for v in p4.next_state)}"
          f"  (want ≈(0,0))")
    p16 = model.predict_horizon((6.0, 0.0), "left", 16)
    print(f"16-step (coarse route)  → {tuple(round(v, 2) for v in p16.next_state)}")

    e_true = model.energy((4.0, 0.0), "left", (3.0, 0.0))
    e_false = model.energy((4.0, 0.0), "left", (6.0, 3.0))
    print(f"energy  true={e_true:.3f}  false={e_false:.3f}  "
          f"plausibility true={model.plausibility((4.0, 0.0), 'left', (3.0, 0.0)):.2f}")
    assert e_true < e_false

    print(f"anti-collapse latent std = {model.stats()['collapse_std']:.3f} (want > 0.1)")
    assert model.stats()["collapse_std"] > 0.1

    unknown = model.predict((4.0, 0.0), "teleport")
    print(f"unknown action conf = {unknown.confidence:.3f} (want 0)")
    assert unknown.confidence == 0.0

    plan = model.plan((5.0, 5.0), goal_state=(0.0, 0.0), horizon=8)
    print(f"plan (5,5)→(0,0): {[str(a) for a in plan.actions]}  "
          f"conf={plan.confidence:.2f}")
    lefts_downs = sum(1 for a in plan.actions if a in ("left", "down"))
    assert lefts_downs >= len(plan.actions) // 2

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(model.to_dict(), f)
        path = f.name
    clone = JEPAWorldModel(seed=0)
    with open(path) as f:
        assert clone.load_dict(json.load(f))
    p2 = clone.predict((4.0, 0.0), "left")
    assert all(abs(a - b) < 1e-6 for a, b in zip(p.next_state, p2.next_state))
    print("persistence roundtrip OK")
    print("ALL SELF-TESTS PASSED")
