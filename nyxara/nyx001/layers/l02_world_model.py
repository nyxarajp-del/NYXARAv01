"""NYXARA · nyx001/layers/l02_world_model.py — Layer 2, the dynamic world model (🌍, the organ).

The charter calls this "the central organ", and the loop it specifies is the whole of it:

    ŝ_{t+1} = F_θ(s_t, a_t)          predict
    e_t     = D(s_{t+1}, ŝ_{t+1})    observe reality, measure the error
    θ      ← θ - α ∇_θ e_t           learn from the error

Everything else in NYX V.001 is downstream of this. Episodic memory prioritises by ``e_t``.
Curiosity hunts where ``e_t`` is high and *falling*. Concepts form where the model has learned
that different inputs predict the same futures. The self-model's calibration is a claim about
``e_t``. If this layer does not actually learn, nothing above it means anything — which is why
:meth:`WorldModel.learning_curve` exists and why the tests assert the error *falls* rather than
asserting the code ran.

``F_θ`` is a gated recurrence, not a Transformer:

    h_t = a ⊙ h_{t-1} + W_in·[s_t ; a_t]        (diag_scan — a selective state-space step)
    ŝ_{t+1} = W_out·GELU(W_hid·h_t) + s_t       (a residual: predict the *change*)

Two choices worth defending. The recurrence is a diagonal gated scan because the charter forbids
copying the Transformer recipe and because a world model wants a persistent state, not an
all-pairs re-read of its history — the state *is* the model's belief about the world right now.
And the output is residual: it predicts ``Δs`` and adds it to ``s_t``. A world model that predicts
absolute next-state can score well by learning to copy its input, which looks like a low error and
is not a model of anything. Predicting the change makes copying score exactly zero.

:meth:`WorldModel.rollout` runs the model against its own predictions to give Layers 7 and 8 an
imagined future. Its honesty rule: it returns the horizon it *actually* reached and its own
per-step confidence decay, because an imagined future ten steps out from a model with a 0.4 error
is not foresight, and reporting it without that decay would be fabricating it.

Honest: this is a learned dynamics model over the perception layer's own feature space, trained
online by gradient descent. It models what NYX V.001 has *seen*, at the resolution Layer 1 gives
it. It is not a physics engine, it does not model the world, and outside its training distribution
its predictions degrade — which :attr:`Prediction.uncertainty` is there to say out loud.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nyxara.nyx001.layers.types import Prediction
from nyxara.nyx001.substrate import (
    AdaptiveOptimizer,
    CognitiveState,
    CompositeObjective,
    HAS_NUMPY,
    Tensor,
    backward,
    constant,
    he,
    ones,
    parameter,
    uniform_gate,
    xavier,
    zeros,
)
from nyxara.nyx001.substrate import autograd as A

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["WorldModel"]


class WorldModel:
    """Layer 2. Predicts the next state, measures its error, and learns from it."""

    def __init__(self, substrate: Any, *, lr: float = 3e-3, action_dim: int = 16,
                 hidden_mult: int = 2, history: int = 512, warmup: int = 4,
                 rehearsal_enabled: bool = True, rehearsal_steps: int = 2,
                 rehearsal_capacity: int = 512, ewc_enabled: bool = True,
                 ewc_lambda: float = 25.0) -> None:
        self.substrate = substrate
        self.width = int(getattr(substrate, "width", 256))
        self.action_dim = int(action_dim)
        self.hidden = int(self.width * max(1, hidden_mult))
        self.history = int(history)
        self.warmup = int(warmup)
        self.available = bool(HAS_NUMPY)
        self.steps = 0

        self._lr = float(lr)
        self.rehearsal_enabled = bool(rehearsal_enabled)
        self.rehearsal_steps = int(rehearsal_steps)
        self.rehearsal_capacity = int(rehearsal_capacity)
        self.rehearsal_steps_run = 0
        self.ewc_enabled = bool(ewc_enabled)
        self.ewc_lambda = float(ewc_lambda)
        self.consolidations = 0
        self._anchor = None
        self._fisher: list = []
        self._fisher_accum: list = []
        self._fisher_n = 0
        self._buffer: List[Any] = []
        self._seen = 0
        self._rng = (substrate.stream("world_model_replay")
                     if hasattr(substrate, "stream") else np.random.default_rng(0))
        self._errors: List[float] = []
        self._state: Any = None                  # h_t, the recurrent belief
        self._last_state: Any = None             # s_t, for the residual and for learning
        self._last_pred: Optional[Prediction] = None
        self._last_action: Any = None
        self._h_before: Any = None               # h as predict() saw it — observe() re-runs from it
        self._milestones: List[float] = []       # long-run downsampled curve; see learning_curve
        self._bucket: List[float] = []
        self._params: List[Tensor] = []
        self._built = False
        self.objective = CompositeObjective(w_concept=0.0, w_compression=0.0,
                                            w_calibration=0.0, w_curiosity=0.0)
        self.optimizer: Optional[AdaptiveOptimizer] = None
        self._build()

    # ---- the learning rate, as one number rather than two ---- #
    @property
    def lr(self) -> float:
        """The rate the optimizer is actually stepping at.

        A plain attribute here was a **dead knob**. ``_build`` passes it to the optimizer once
        and never reads it again, so Layer 17 — which lists ``world_model.lr`` as the first of
        its ten mutable targets — could set it, log a rollout, and change nothing about how the
        model learns. Layer 13 avoided the trap only by writing ``optimizer.lr`` directly. The
        property makes the two impossible to disagree.
        """
        opt = getattr(self, "optimizer", None)
        return float(opt.lr) if opt is not None else self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        self._lr = float(value)
        opt = getattr(self, "optimizer", None)
        if opt is not None:
            opt.lr = float(value)

    # ---- F_θ ---- #
    def _build(self) -> None:
        if not self.available:
            return
        try:
            rng = self.substrate.stream("world_model")
            d_in = self.width + self.action_dim
            self.w_in = xavier(rng, d_in, self.width)
            self.b_in = zeros(self.width)
            # The gate is stored as a LOGIT and squashed to (0,1) in the forward pass. Storing it
            # raw and clipping after each step would take it out of the graph; this keeps it a
            # first-class learned parameter that cannot leave (0,1) however large a step gets.
            # Initialised from the same (0.85, 0.999) band uniform_gate would have produced.
            g0 = uniform_gate(rng, self.width).data
            self.a_raw = parameter(np.log(g0 / (1.0 - g0)))
            self.w_hid = he(rng, self.width, self.hidden)
            self.b_hid = zeros(self.hidden)
            self.w_out = xavier(rng, self.hidden, self.width, gain=0.1)
            self.b_out = zeros(self.width)
            self.g_norm = ones(self.width)
            self._params = [self.w_in, self.b_in, self.a_raw, self.w_hid, self.b_hid,
                            self.w_out, self.b_out, self.g_norm]
            self.optimizer = AdaptiveOptimizer(self._params, lr=self._lr, clip=1.0)
            self._fisher_accum = [np.zeros_like(p.data) for p in self._params]
            self._fisher = [np.zeros_like(p.data) for p in self._params]
            if hasattr(self.substrate, "note_params"):
                self.substrate.note_params(*self._params)
            self._built = True
        except Exception:  # noqa: BLE001 — an unbuilt world model declines, it does not crash
            self._built = False

    def _encode_action(self, action: Optional[Any]) -> Any:
        """Actions share the state's space via a small fixed random code — untrained, stable."""
        v = np.zeros(self.action_dim, dtype=np.float64)
        if action is None:
            return v
        if hasattr(action, "__array__"):
            flat = np.asarray(action, dtype=np.float64).ravel()
            n = min(flat.size, self.action_dim)
            v[:n] = flat[:n]
            return v
        h = abs(hash(str(action)))
        v[h % self.action_dim] = 1.0
        return v

    def _forward(self, s_t: Any, a_t: Any, h_prev: Any) -> Tuple[Tensor, Any]:
        """One step of ``F_θ``. Returns (ŝ_{t+1} as a Tensor, new h) — the tape stays live."""
        x = constant(np.concatenate([s_t, a_t]).reshape(1, 1, -1))
        proj = A.add(A.matmul(x, self.w_in), self.b_in)              # (1,1,W)
        gate = A.sigmoid(self.a_raw)                                 # (W,) in (0,1), learned
        # Carry h across calls by hand: diag_scan starts each call from a zero state, so the
        # recurrence h_t = a⊙h_{t-1} + W_in·x_t is completed here. The multiply goes through the
        # GRAPH, not through ``.data`` — with h_prev detached but the gate live, ∂loss/∂gate is
        # h_prev, which is exactly the term that teaches the model how long to remember.
        #
        # Getting this wrong is not cosmetic. With the gate detached its gradient is identically
        # zero, so it stays frozen at its random init and the model has NO learnable way to
        # suppress h. The recurrent state then becomes a nuisance input that makes the same s_t
        # predict differently depending on history, and a Markovian system becomes unfittable.
        # That is what "selective" means in a selective state-space model: the gate must be able
        # to learn to shut.
        #
        # h_prev itself stays a constant: this is truncated BPTT at length 1. Credit for what the
        # state *contains* does not propagate back through earlier steps — a real limit, and the
        # reason this layer learns single-step dynamics far better than long dependencies.
        if h_prev is not None:
            proj = A.add(proj, A.mul(constant(h_prev.reshape(1, 1, -1)), gate))
        h = A.diag_scan(proj, gate)                                  # (1,1,W)
        hid = A.gelu(A.add(A.matmul(h, self.w_hid), self.b_hid))
        delta = A.add(A.matmul(hid, self.w_out), self.b_out)
        delta = A.mul(delta, self.g_norm)
        pred = A.add(delta, constant(s_t.reshape(1, 1, -1)))         # residual: predict the change
        return pred, h.data.reshape(-1)

    # ---- predict ---- #
    def predict(self, state: Any, action: Optional[Any] = None, *,
                horizon: int = 1) -> Prediction:
        """``ŝ_{t+1} = F_θ(s_t, a_t)``. Never raises; an unbuilt model returns a null prediction."""
        if not self._built or state is None:
            return Prediction(horizon=horizon, uncertainty=1.0)
        try:
            s = np.asarray(state, dtype=np.float64).ravel()[: self.width]
            if s.size < self.width:
                s = np.pad(s, (0, self.width - s.size))
            a = self._encode_action(action)
            # Keep the state the forward pass actually SAW. observe() re-runs the same forward to
            # build a fresh tape, and it must re-run it from this h — not from the updated one and
            # not from None. Training on a different function than the one that predicted is a
            # silent train/inference mismatch: it showed up here as a model that trained to a
            # 5e-05 error and still lost to copying its own input.
            self._h_before = None if self._state is None else self._state.copy()
            pred, new_h = self._forward(s, a, self._state)
            self._state = new_h
            self._last_state = s
            out = Prediction(state=pred.data.reshape(-1), horizon=horizon,
                             uncertainty=self.uncertainty())
            self._last_pred = out
            self._last_tensor = pred
            self._last_action = a
            return out
        except Exception:  # noqa: BLE001
            return Prediction(horizon=horizon, uncertainty=1.0)

    # ---- observe + learn ---- #
    def observe(self, actual: Any, *, cognitive: Optional[CognitiveState] = None) -> float:
        """``e_t = D(s_{t+1}, ŝ_{t+1})``, then one gradient step. Returns the error.

        Called with no outstanding prediction this returns 0.0 and learns nothing — there is no
        error to measure against a prediction that was never made.
        """
        if not self._built or self._last_pred is None or actual is None:
            return 0.0
        try:
            tgt = np.asarray(actual, dtype=np.float64).ravel()[: self.width]
            if tgt.size < self.width:
                tgt = np.pad(tgt, (0, self.width - tgt.size))
            # Re-run the forward pass so the tape is fresh: the graph from predict() has already
            # been read for its value and reusing it would accumulate gradients across calls.
            # From ``_h_before`` — the same recurrent state predict() saw — so the function being
            # trained is exactly the function that made the prediction.
            pred, _ = self._forward(self._last_state, self._last_action, self._h_before)
            loss = self.objective(prediction=self.objective.prediction_term(
                pred, tgt.reshape(1, 1, -1)))
            self.optimizer.zero_grad()
            backward(loss.total)
            self._accumulate_fisher()   # importance BEFORE the pull, so it measures the task
            self._ewc_pull()
            self.optimizer.step(cognitive)
            err = float(loss.value)
            self._errors.append(err)
            if len(self._errors) > self.history:
                self._errors = self._errors[-self.history:]
            # The long-run curve is kept separately, because ``_errors`` is a bounded *recent*
            # window: by the time anyone asks whether this model learned, the part where it did
            # has already scrolled out of that window. Milestones downsample instead of forgetting.
            self._bucket.append(err)
            if len(self._bucket) >= 32:
                self._milestones.append(float(sum(self._bucket) / len(self._bucket)))
                self._bucket = []
                if len(self._milestones) > 256:   # bounded too — halve, keeping the whole shape
                    self._milestones = self._milestones[::2]
            self.steps += 1
            self._last_pred.error = err
            self._remember_transition(self._last_state, self._last_action, tgt)
            self._rehearse(cognitive)
            return err
        except Exception:  # noqa: BLE001 — a failed update is a lost lesson, never a broken turn
            return 0.0

    # ---- rehearsal: the defence against catastrophic forgetting ---- #
    def _remember_transition(self, s: Any, a: Any, s_next: Any) -> None:
        """Keep a *reservoir* sample of every transition ever seen.

        Reservoir sampling, not a ring buffer, and the distinction is the whole point. A ring
        buffer holds the most RECENT ``n`` transitions — so after training on system B it contains
        only B, and rehearsing from it defends nothing. A reservoir holds a uniform sample of the
        entire history, so system A is still represented while B is being learned. That is what
        makes this a defence against forgetting rather than a slightly longer batch.
        """
        if not self.rehearsal_enabled or s is None or s_next is None:
            return
        try:
            self._seen += 1
            # The recurrent state is stored WITH the transition. Without it, rehearsal replays
            # every remembered step from a blank context — and the context is often the only
            # thing that distinguishes two systems presenting the same s_t and expecting
            # different s_{t+1}. Replaying stateless makes those samples mutually contradictory,
            # so the model learns their average, which is worse than either. Measured: with
            # h dropped, skill on A fell to 0.0 after learning B despite two-thirds of the
            # reservoir still being A.
            item = (np.asarray(s, dtype=np.float64).copy(),
                    None if a is None else np.asarray(a, dtype=np.float64).copy(),
                    np.asarray(s_next, dtype=np.float64).copy(),
                    None if self._h_before is None else self._h_before.copy())
            if len(self._buffer) < self.rehearsal_capacity:
                self._buffer.append(item)
                return
            # replace with probability capacity/seen — keeps the sample uniform over all history
            j = int(self._rng.integers(0, self._seen))
            if j < self.rehearsal_capacity:
                self._buffer[j] = item
        except Exception:  # noqa: BLE001 — a lost sample is never worth breaking a step for
            pass

    def _rehearse(self, cognitive: Optional[CognitiveState] = None) -> int:
        """Interleave gradient steps on remembered transitions. Returns how many ran.

        Each rehearsal step is a full predict-and-learn on an OLD transition, run from a fresh
        recurrent state so it does not disturb the live one. Without this the model is a pure
        online learner: every step moves it toward the current distribution and nothing holds it
        to the last one, which is exactly the shape of catastrophic forgetting.
        """
        if not self.rehearsal_enabled or not self._buffer:
            return 0
        n = 0
        saved_state, saved_h = self._state, self._h_before
        saved_last, saved_action = self._last_state, self._last_action
        try:
            for _ in range(max(0, int(self.rehearsal_steps))):
                idx = int(self._rng.integers(0, len(self._buffer)))
                s, a, s_next, h = self._buffer[idx]
                pred, _ = self._forward(s, a if a is not None else self._encode_action(None), h)
                loss = self.objective(prediction=self.objective.prediction_term(
                    pred, s_next.reshape(1, 1, -1)))
                self.optimizer.zero_grad()
                backward(loss.total)
                self._ewc_pull()
                self.optimizer.step(cognitive)
                self.rehearsal_steps_run += 1
                n += 1
            return n
        except Exception:  # noqa: BLE001 — rehearsal is best-effort, never fatal
            return n
        finally:
            # the live recurrent belief must survive rehearsal untouched
            self._state, self._h_before = saved_state, saved_h
            self._last_state, self._last_action = saved_last, saved_action

    # ---- EWC: anchor what mattered, so the next task cannot overwrite it ---- #
    def consolidate(self, *, task: str = "") -> Dict[str, Any]:
        """Snapshot θ and freeze an importance estimate — the memory *write* of EWC.

        Called at a task boundary. Importance is the accumulated squared gradient per weight (a
        diagonal empirical Fisher), which is the standard estimator: a weight whose gradient was
        consistently large mattered to what was just learned, and moving it is what destroys the
        skill.

        Rehearsal alone was measured as insufficient here — it improves retention monotonically
        with its budget (A-error 0.104 at 2 steps, 0.072 at 8) but trades away plasticity on the
        new task (0.67 → 0.53) and never got under the copy baseline. EWC attacks the same problem
        from the other side: instead of re-showing old data, it makes the weights that carried the
        old skill expensive to move.

        Not reused from ``memory/elastic_synapses.py`` despite that module implementing full EWC:
        its API is ``Mapping[str, float]`` — one named scalar per weight — and this model holds
        NumPy arrays with tens of thousands of entries. Adapting arrays to per-scalar names would
        have been slower and less clear than the dozen lines here.
        """
        if not self._built or not self.ewc_enabled:
            return {}
        try:
            total = max(1, self._fisher_n)
            self._anchor = [p.data.copy() for p in self._params]
            self._fisher = [f / total for f in self._fisher_accum]
            self._fisher_accum = [np.zeros_like(p.data) for p in self._params]
            self._fisher_n = 0
            self.consolidations += 1
            return {"task": task, "consolidations": self.consolidations,
                    "mean_importance": float(np.mean([float(f.mean()) for f in self._fisher]))}
        except Exception:  # noqa: BLE001
            return {}

    def _accumulate_fisher(self) -> None:
        """Running Σg² per weight — the importance signal ``consolidate`` freezes."""
        if not self.ewc_enabled:
            return
        try:
            for i, p in enumerate(self._params):
                if p.grad is not None:
                    self._fisher_accum[i] += p.grad * p.grad
            self._fisher_n += 1
        except Exception:  # noqa: BLE001
            pass

    def _ewc_pull(self) -> None:
        """Add λ·F·(θ−θ*) to each gradient: a force back toward the consolidated weights."""
        if not self.ewc_enabled or self._anchor is None:
            return
        try:
            for i, p in enumerate(self._params):
                if p.grad is None:
                    continue
                p.grad = p.grad + self.ewc_lambda * self._fisher[i] * (p.data - self._anchor[i])
        except Exception:  # noqa: BLE001
            pass

    def anchor_drift(self) -> Optional[float]:
        """How far the weights have travelled from their anchor. ``None`` before consolidation."""
        if self._anchor is None:
            return None
        try:
            return float(sum(float(np.linalg.norm(p.data - a))
                             for p, a in zip(self._params, self._anchor)))
        except Exception:  # noqa: BLE001
            return None

    # ---- imagination ---- #
    def rollout(self, state: Any, actions: Optional[List[Any]] = None, *,
                horizon: int = 4) -> List[Prediction]:
        """Run the model against its own predictions. Confidence decays, and says so.

        The decay is ``(1 - mean_error)^k`` at step ``k``: a model with a 0.4 mean error is
        reporting roughly 0.13 confidence four steps out. That number is the point — Layers 7 and
        8 are supposed to *see* that a long rollout from a bad model is worthless.
        """
        out: List[Prediction] = []
        if not self._built or state is None:
            return out
        try:
            s = np.asarray(state, dtype=np.float64).ravel()[: self.width]
            if s.size < self.width:
                s = np.pad(s, (0, self.width - s.size))
            h = self._state.copy() if self._state is not None else None
            # An UNTRAINED model must report maximum uncertainty, not zero. `mean_error() or 0.0`
            # collapsed None to 0.0 and produced confidence 1.0 at every horizon — a model that
            # has never seen a single example declaring perfect foresight. Layers 7 and 8 both
            # gate on this number, so the bug silently licensed causal claims and plans built on
            # nothing at all.
            mean_err = self.mean_error()
            if mean_err is None:
                base = 0.0                       # ⇒ uncertainty 1.0 at every horizon
            else:
                base = 1.0 - min(1.0, mean_err)
            acts = list(actions or [])
            for k in range(max(1, int(horizon))):
                a = self._encode_action(acts[k] if k < len(acts) else None)
                pred, h = self._forward(s, a, h)
                s = pred.data.reshape(-1)
                out.append(Prediction(state=s.copy(), horizon=k + 1,
                                      uncertainty=round(1.0 - base ** (k + 1), 4)))
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- what it knows about itself ---- #
    def mean_error(self, *, window: int = 32) -> Optional[float]:
        """Recent mean error, or ``None`` before ``warmup`` — never a misleading 0.0."""
        if len(self._errors) < self.warmup:
            return None
        w = self._errors[-window:]
        return float(sum(w) / len(w))

    def uncertainty(self) -> float:
        """How unsure this model is right now, in ``[0, 1]``. High before it has learned anything."""
        m = self.mean_error()
        if m is None:
            return 1.0
        return float(min(1.0, m / (1.0 + m)))

    def learning_curve(self, *, buckets: int = 8) -> List[float]:
        """Mean error per bucket over the *whole run* — the evidence this layer learns at all.

        Reads the downsampled milestones rather than the recent-error window, so the early steep
        part of the curve is still visible after thousands of steps. Falls back to the raw window
        only before the first milestone has been recorded.
        """
        source = self._milestones if len(self._milestones) >= buckets else self._errors
        if len(source) < buckets:
            return [round(e, 6) for e in source]
        size = len(source) // buckets
        return [round(float(sum(source[i * size:(i + 1) * size]) / size), 6)
                for i in range(buckets)]

    def is_learning(self) -> Optional[bool]:
        """Is the error actually falling? ``None`` when there is not yet enough evidence to say."""
        curve = self.learning_curve()
        if len(curve) < 4:
            return None
        first = sum(curve[:2]) / 2.0
        last = sum(curve[-2:]) / 2.0
        return bool(last < first * 0.95)

    def reset_state(self) -> None:
        """Drop the recurrent belief without touching the weights — a new episode, same model."""
        self._state = None
        self._last_pred = None

    def stats(self) -> Dict[str, Any]:
        return {"available": self.available, "built": self._built, "steps": self.steps,
                "width": self.width, "hidden": self.hidden,
                "mean_error": self.mean_error(), "uncertainty": round(self.uncertainty(), 4),
                "is_learning": self.is_learning(),
                "n_params": int(sum(int(p.data.size) for p in self._params))}

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"steps": self.steps, "errors": [round(e, 6) for e in self._errors[-128:]]}
        if self._built:
            d["weights"] = {f"p{i}": p.data.tolist() for i, p in enumerate(self._params)}
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.steps = int(d.get("steps", 0))
            self._errors = [float(e) for e in (d.get("errors") or [])]
            w = d.get("weights") or {}
            if self._built and w:
                for i, p in enumerate(self._params):
                    arr = w.get(f"p{i}")
                    if arr is not None:
                        cand = np.asarray(arr, dtype=np.float64)
                        if cand.shape == p.data.shape:
                            p.data = cand
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves the model as it was born
            pass
