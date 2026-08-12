"""NYXARA · nyx001/substrate/optim.py — the adaptive optimizer (⚙️, Δθ = F(E,N,U,M,T,…)).

Ordinary training computes ``Δθ = -lr · ∇L``: the update depends on the gradient and on nothing
else. The NYX V.001 charter asks for something different — an update *driven by cognition*, where
the system's own appraisal of the moment changes how hard it learns from it.

:class:`CognitiveState` is that appraisal, five signals the layers above report every step:

===  ================  ==================================================================
E    prediction error  how wrong the world model just was
N    novelty           how unlike anything already seen this observation is
U    uncertainty       how unsure the system is of its own answer
M    memory pressure   how full/contended the memory substrate is
T    time budget       how much compute this step is allowed (0 = starved, 1 = ample)
===  ================  ==================================================================

:class:`AdaptiveOptimizer` folds them into a single scalar ``gain`` over an RMSprop-style
per-parameter step:

    Δθ = -(lr · gain) · m̂ / (√v̂ + ε),  gain = f(E, N, U, M, T)

The shape of ``f`` is a set of *hypotheses about learning*, and each is written down here so it
can be argued with and ablated rather than absorbed:

* **error raises the step** — a large surprise is evidence the parameters are wrong, so it is
  worth moving further. Bounded, because a single outlier should not be allowed to wreck a model
  that is otherwise right.
* **novelty raises the step** — the first examples of a genuinely new regime deserve more weight
  than the thousandth example of a familiar one.
* **uncertainty *lowers* the step** — this is the one that reads backwards until you see why. When
  the system does not know whether its own answer was right, the gradient's *sign* is unreliable,
  and taking a big step along an unreliable direction is how a model talks itself into a
  confident mistake. Under uncertainty it should shuffle, not stride.
* **memory pressure lowers the step** — when the store is contended, the replay stream feeding the
  gradient is a biased sample of experience, so it deserves less trust.
* **time budget scales the step** — a step taken under a starved budget saw less of the data.

Every term is clamped and the product is clamped again to ``[gain_floor, gain_ceiling]``, so no
combination of signals can produce a step that destroys the model. This is a real safety property,
not a formality: an unbounded cognitive gain is a divergence waiting for its input.

:class:`AdaptiveOptimizer` degrades to plain RMSprop when handed a default
:class:`CognitiveState` (all signals neutral, gain ≡ 1.0), so the cognitive modulation can be
ablated by simply not reporting — which is how ``tests/nyx001/test_optim.py`` measures whether it
earns its place at all.

Honest: this is a *heuristic controller over a standard optimizer*, not a learned optimizer and
not a new convergence result. There is no proof it beats RMSprop in general; there is a measured
comparison on this system's own tasks, and it reports its own gain history so the modulation is
visible rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from nyxara.nyx001.substrate.autograd import Tensor

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["CognitiveState", "StepReport", "AdaptiveOptimizer"]


def _clamp(x: float, lo: float, hi: float) -> float:
    x = float(x)
    if x != x:                      # NaN — treat as neutral rather than propagating it
        return (lo + hi) * 0.5
    return lo if x < lo else (hi if x > hi else x)


def _clamp01(x: float) -> float:
    return _clamp(x, 0.0, 1.0)


@dataclass
class CognitiveState:
    """The five signals that modulate a step. All in ``[0, 1]``; all default to neutral."""

    error: float = 0.0              # E — prediction error, normalised
    novelty: float = 0.0            # N — how unlike anything seen before
    uncertainty: float = 0.0        # U — how unsure of its own answer
    memory_pressure: float = 0.0    # M — how contended the memory substrate is
    time_budget: float = 1.0        # T — 1.0 = ample compute, 0.0 = starved

    def __post_init__(self) -> None:
        self.error = _clamp01(self.error)
        self.novelty = _clamp01(self.novelty)
        self.uncertainty = _clamp01(self.uncertainty)
        self.memory_pressure = _clamp01(self.memory_pressure)
        self.time_budget = _clamp01(self.time_budget)

    @property
    def is_neutral(self) -> bool:
        """True when this state carries no information — the ablation baseline."""
        return (self.error == 0.0 and self.novelty == 0.0 and self.uncertainty == 0.0
                and self.memory_pressure == 0.0 and self.time_budget == 1.0)

    def to_dict(self) -> Dict[str, float]:
        return {"E": round(self.error, 4), "N": round(self.novelty, 4),
                "U": round(self.uncertainty, 4), "M": round(self.memory_pressure, 4),
                "T": round(self.time_budget, 4)}


@dataclass
class StepReport:
    """What one optimizer step actually did — so the modulation is inspectable, not assumed."""

    gain: float = 1.0
    grad_norm: float = 0.0
    update_norm: float = 0.0
    n_params: int = 0
    clipped: bool = False
    signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"gain": round(self.gain, 4), "grad_norm": round(self.grad_norm, 6),
                "update_norm": round(self.update_norm, 6), "n_params": self.n_params,
                "clipped": self.clipped, "signals": dict(self.signals)}


class AdaptiveOptimizer:
    """RMSprop with momentum, under a bounded cognitive gain.

    :param params: the tensors to learn. Only ``requires_grad`` leaves are touched.
    :param lr: base learning rate, before any modulation.
    :param beta1/beta2: momentum and second-moment decay (Adam's defaults, and Adam's bias
        correction, because without it the first few steps of a short run are badly scaled — and
        a developmental curriculum is *made* of short runs).
    :param clip: global gradient-norm ceiling. ``0`` disables clipping.
    """

    def __init__(self, params: Iterable[Tensor], *, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, clip: float = 1.0,
                 gain_floor: float = 0.1, gain_ceiling: float = 3.0,
                 w_error: float = 0.6, w_novelty: float = 0.4, w_uncertainty: float = 0.5,
                 w_memory: float = 0.3) -> None:
        self.params: List[Tensor] = [p for p in params if p is not None and p.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.clip = float(clip)
        self.gain_floor = float(gain_floor)
        self.gain_ceiling = float(gain_ceiling)
        self.w_error = float(w_error)
        self.w_novelty = float(w_novelty)
        self.w_uncertainty = float(w_uncertainty)
        self.w_memory = float(w_memory)
        self.t = 0
        self._m: List[Any] = [np.zeros_like(p.data) for p in self.params]
        self._v: List[Any] = [np.zeros_like(p.data) for p in self.params]
        self.gain_history: List[float] = []

    # ---- the cognitive gain ---- #
    def gain_for(self, state: Optional[CognitiveState]) -> float:
        """``F(E, N, U, M, T)`` — bounded, and exactly 1.0 for a neutral state.

        The neutral identity is a property the ablation depends on: with no signals reported this
        optimizer *is* RMSprop, so any measured difference is attributable to the modulation and
        not to a different base optimizer.
        """
        if state is None or state.is_neutral:
            return 1.0
        up = 1.0 + self.w_error * state.error + self.w_novelty * state.novelty
        down = 1.0 + self.w_uncertainty * state.uncertainty + self.w_memory * state.memory_pressure
        budget = 0.5 + 0.5 * state.time_budget      # T=0 halves the step; T=1 leaves it alone
        return _clamp(up / down * budget, self.gain_floor, self.gain_ceiling)

    # ---- the step ---- #
    def step(self, state: Optional[CognitiveState] = None) -> StepReport:
        """Apply one update. Returns what it did; never raises on an empty/degenerate graph."""
        report = StepReport(n_params=len(self.params), signals=state.to_dict() if state else {})
        live = [(i, p) for i, p in enumerate(self.params) if p.grad is not None]
        if not live:
            return report                            # nothing backpropagated — a no-op, honestly

        total_sq = 0.0
        for _, p in live:
            total_sq += float((p.grad * p.grad).sum())
        report.grad_norm = float(np.sqrt(total_sq))

        scale = 1.0
        if self.clip > 0.0 and report.grad_norm > self.clip:
            scale = self.clip / (report.grad_norm + 1e-12)
            report.clipped = True

        gain = self.gain_for(state)
        report.gain = gain
        self.gain_history.append(gain)
        if len(self.gain_history) > 4096:            # bounded, like every store in this repo
            self.gain_history = self.gain_history[-2048:]

        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        upd_sq = 0.0
        for i, p in live:
            g = p.grad * scale
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            delta = -(self.lr * gain) * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = p.data + delta
            upd_sq += float((delta * delta).sum())
        report.update_norm = float(np.sqrt(upd_sq))
        return report

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def mean_gain(self) -> Optional[float]:
        """Average modulation so far — ``None`` when no step has run, never a misleading 0.0."""
        if not self.gain_history:
            return None
        return float(sum(self.gain_history) / len(self.gain_history))

    def stats(self) -> Dict[str, Any]:
        return {"steps": self.t, "n_params": len(self.params),
                "n_values": int(sum(int(p.data.size) for p in self.params)),
                "lr": self.lr, "mean_gain": self.mean_gain()}
