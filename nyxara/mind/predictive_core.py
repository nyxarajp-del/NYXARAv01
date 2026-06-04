"""NYXARA · mind/predictive_core.py — the free-energy spine (✦★).

Perception, action, and emotion are not three systems. They are **one loop** that
minimises *prediction error* — the Free Energy Principle / Active Inference (Friston).

NYXARA carries a **generative model** ``g`` that predicts its sensations from a hidden
belief about their causes. The gap between prediction and reality is *prediction
error*; variational **free energy** bounds the surprise. From that single quantity:

* **Perception** = update the belief (``μ``) so predictions match what arrived
  (gradient descent on free energy — change the mind to fit the world).
* **Action** = choose the act whose *expected* free energy is lowest — i.e. that
  makes the future match the agent's **preferences** (change the world to fit the
  mind), while also reducing uncertainty (epistemic value).
* **Emotion** = a read-out of the error *dynamics*: free energy falling feels good
  (**valence**), large precision-weighted error feels activating (**arousal**), and
  the level of free energy is felt as **surprise**.

This is the beating heart the orchestrator can wire perception, the world-model,
and affect into. Pure standard library (small vectors, finite-difference Jacobian).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Vec",
    "EmotionReadout",
    "PerceptionResult",
    "Action",
    "ActionChoice",
    "PredictiveCore",
]

Vec = List[float]


# --------------------------------------------------------------------------- #
# tiny vector helpers (no numpy dependency)
# --------------------------------------------------------------------------- #
def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return [x - y for x, y in zip(a, b)]


def _add(a: Sequence[float], b: Sequence[float]) -> Vec:
    return [x + y for x, y in zip(a, b)]


def _scale(a: Sequence[float], s: float) -> Vec:
    return [x * s for x in a]


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _norm2(a: Sequence[float]) -> float:
    return sum(x * x for x in a)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _jacobian(g: Callable[[Vec], Vec], mu: Vec, eps: float = 1e-5) -> List[Vec]:
    """Finite-difference Jacobian J[i][j] = ∂g_i/∂μ_j."""
    base = g(mu)
    od, sd = len(base), len(mu)
    J = [[0.0] * sd for _ in range(od)]
    for j in range(sd):
        mp = list(mu)
        mp[j] += eps
        gp = g(mp)
        for i in range(od):
            J[i][j] = (gp[i] - base[i]) / eps
    return J


def _jt_vec(J: List[Vec], e: Vec) -> Vec:
    """Compute Jᵀ·e (state-dim vector)."""
    sd = len(J[0]) if J else 0
    out = [0.0] * sd
    for i, row in enumerate(J):
        for j in range(sd):
            out[j] += row[j] * e[i]
    return out


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #
@dataclass
class EmotionReadout:
    """Affect as a function of prediction-error dynamics."""

    valence: float    # [-1, 1] — free energy falling (good) vs rising (bad)
    arousal: float    # [0, 1]  — precision-weighted error magnitude
    surprise: float   # [0, 1]  — current free-energy level

    def to_dict(self) -> Dict[str, float]:
        return {"valence": round(self.valence, 4), "arousal": round(self.arousal, 4),
                "surprise": round(self.surprise, 4)}


@dataclass
class PerceptionResult:
    belief: Vec
    prediction: Vec
    error: Vec
    free_energy: float
    iterations: int

    @property
    def error_magnitude(self) -> float:
        return math.sqrt(_norm2(self.error))

    def to_dict(self) -> Dict[str, object]:
        return {"belief": [round(x, 4) for x in self.belief],
                "free_energy": round(self.free_energy, 6),
                "error_magnitude": round(self.error_magnitude, 6),
                "iterations": self.iterations}


# --------------------------------------------------------------------------- #
# Action (active inference)
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """An available action and the observation the agent *expects* if it acts.

    ``info_gain`` is the epistemic value (uncertainty it would resolve); higher is
    better and lowers expected free energy.
    """

    name: str
    predicted_observation: Vec
    info_gain: float = 0.0


@dataclass
class ActionChoice:
    action: Action
    expected_free_energy: float
    pragmatic: float
    epistemic: float

    def to_dict(self) -> Dict[str, object]:
        return {"action": self.action.name,
                "expected_free_energy": round(self.expected_free_energy, 4),
                "pragmatic": round(self.pragmatic, 4),
                "epistemic": round(self.epistemic, 4)}


# --------------------------------------------------------------------------- #
# Predictive core
# --------------------------------------------------------------------------- #
class PredictiveCore:
    """One prediction-error loop: perceive, act, feel — all from free energy."""

    def __init__(
        self,
        *,
        belief: Vec,
        generative_model: Optional[Callable[[Vec], Vec]] = None,
        prior: Optional[Vec] = None,
        preference: Optional[Vec] = None,
        sensory_precision: float = 1.0,
        prior_precision: float = 0.1,
        preference_precision: float = 1.0,
        learning_rate: float = 0.2,
        epistemic_weight: float = 1.0,
        history: int = 64,
    ) -> None:
        self.mu: Vec = list(belief)
        self.g = generative_model or (lambda s: list(s))   # default: identity model
        self.prior: Vec = list(prior) if prior is not None else list(belief)
        self.preference: Optional[Vec] = list(preference) if preference is not None else None
        self.sensory_precision = sensory_precision
        self.prior_precision = prior_precision
        self.preference_precision = preference_precision
        self.lr = learning_rate
        self.epistemic_weight = epistemic_weight
        self._F_history: Deque[float] = deque(maxlen=history)
        self._last_error_mag: float = 0.0

    # ---- prediction ---- #
    def predict(self) -> Vec:
        return self.g(self.mu)

    # ---- free energy ---- #
    def free_energy(self, observation: Sequence[float]) -> float:
        pred = self.predict()
        accuracy = 0.5 * self.sensory_precision * _dist2(observation, pred)
        complexity = 0.5 * self.prior_precision * _dist2(self.mu, self.prior)
        return accuracy + complexity

    # ---- perception: update belief to minimise free energy ---- #
    def perceive(self, observation: Sequence[float], *, iterations: int = 8) -> PerceptionResult:
        obs = list(observation)
        iters_done = 0
        for _ in range(iterations):
            pred = self.predict()
            error = _sub(obs, pred)                       # obs-dim
            J = _jacobian(self.g, self.mu)
            accuracy_grad = _scale(_jt_vec(J, error), self.sensory_precision)  # ∝ -dF/dμ
            complexity_grad = _scale(_sub(self.mu, self.prior), self.prior_precision)
            step = _sub(accuracy_grad, complexity_grad)   # ascent on -F == descent on F
            self.mu = _add(self.mu, _scale(step, self.lr))
            iters_done += 1

        pred = self.predict()
        error = _sub(obs, pred)
        F = self.free_energy(obs)
        self._F_history.append(F)
        self._last_error_mag = math.sqrt(_norm2(error))
        return PerceptionResult(belief=list(self.mu), prediction=pred, error=error,
                                free_energy=F, iterations=iters_done)

    # ---- action: minimise expected free energy ---- #
    def expected_free_energy(self, action: Action) -> ActionChoice:
        if self.preference is None:
            raise ValueError("preferences must be set to evaluate actions")
        pragmatic = 0.5 * self.preference_precision * _dist2(action.predicted_observation,
                                                             self.preference)
        epistemic = self.epistemic_weight * action.info_gain
        efe = pragmatic - epistemic   # want preferred outcome AND information
        return ActionChoice(action=action, expected_free_energy=efe,
                            pragmatic=pragmatic, epistemic=epistemic)

    def act(self, actions: Sequence[Action]) -> ActionChoice:
        """Active inference: pick the action with the lowest expected free energy."""
        if not actions:
            raise ValueError("no actions to choose from")
        choices = [self.expected_free_energy(a) for a in actions]
        choices.sort(key=lambda c: c.expected_free_energy)
        return choices[0]

    # ---- emotion: read out the error dynamics ---- #
    def emotion(self, *, scale: float = 1.0) -> EmotionReadout:
        F_now = self._F_history[-1] if self._F_history else 0.0
        F_prev = self._F_history[-2] if len(self._F_history) >= 2 else F_now
        # valence: free energy *dropping* (doing better than before) feels positive
        delta = F_prev - F_now
        valence = math.tanh(delta / max(1e-6, scale))
        # arousal: precision-weighted error magnitude, squashed to [0,1]
        arousal = _clamp(math.tanh(self.sensory_precision * self._last_error_mag / max(1e-6, scale)))
        # surprise: current free-energy level, squashed to [0,1]
        surprise = _clamp(math.tanh(F_now / max(1e-6, scale)))
        return EmotionReadout(valence=_clamp(valence, -1.0, 1.0), arousal=arousal,
                              surprise=surprise)

    # ---- combined step ---- #
    def step(self, observation: Sequence[float], *, iterations: int = 8,
             scale: float = 1.0) -> Tuple[PerceptionResult, EmotionReadout]:
        perception = self.perceive(observation, iterations=iterations)
        feeling = self.emotion(scale=scale)
        return perception, feeling

    # ---- precision learning (attention) ---- #
    def update_precision(self, *, floor: float = 0.05, ceil: float = 100.0) -> float:
        """Adapt sensory precision toward the inverse of recent error variance."""
        var = self._last_error_mag ** 2
        self.sensory_precision = _clamp(1.0 / (var + 1e-3), floor, ceil)
        return self.sensory_precision

    # ---- introspection ---- #
    def status(self) -> Dict[str, object]:
        return {"belief": [round(x, 4) for x in self.mu],
                "prediction": [round(x, 4) for x in self.predict()],
                "free_energy": round(self._F_history[-1], 6) if self._F_history else None,
                "sensory_precision": round(self.sensory_precision, 4),
                "emotion": self.emotion().to_dict()}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA predictive-core (free-energy) self-test")
    print("=" * 70)

    # PERCEPTION: belief should converge toward a stable observation, F should fall
    core = PredictiveCore(belief=[0.0, 0.0], prior=[0.0, 0.0],
                          sensory_precision=2.0, prior_precision=0.05, learning_rate=0.3)
    obs = [3.0, -1.0]
    Fs = []
    for t in range(20):
        p, e = core.step(obs)
        Fs.append(p.free_energy)
    print(f"\nbelief after 20 steps: {[round(x,3) for x in core.mu]} (target {obs})")
    print(f"free energy 0 -> 20  : {Fs[0]:.3f} -> {Fs[-1]:.4f}")
    assert _dist2(core.mu, obs) < 0.2          # perceived the cause
    assert Fs[-1] < Fs[0]                       # minimised free energy

    # EMOTION: with gradual (online) perception, valence is positive while error
    # falls and surprise drops as the belief catches up to the world.
    early = PredictiveCore(belief=[0.0], sensory_precision=2.0, learning_rate=0.15)
    p1, em1 = early.step([5.0], iterations=1)
    for _ in range(15):
        p2, em2 = early.step([5.0], iterations=1)
    print(f"\nemotion step1        : {em1.to_dict()}")
    print(f"emotion later        : {em2.to_dict()}")
    assert em2.valence >= 0                     # improving feels non-negative
    assert em2.surprise < em1.surprise          # less surprised as it learns

    # AROUSAL: a big surprising observation spikes arousal
    calm = PredictiveCore(belief=[5.0], sensory_precision=3.0)
    _, em_calm = calm.step([5.0])               # matches expectation
    _, em_shock = calm.step([50.0])             # violent surprise
    print(f"\narousal calm/shock   : {em_calm.arousal:.3f} / {em_shock.arousal:.3f}")
    assert em_shock.arousal > em_calm.arousal

    # ACTION (active inference): choose the act that fulfils preferences
    agent = PredictiveCore(belief=[0.0], preference=[10.0], preference_precision=1.0)
    actions = [
        Action("do_nothing", predicted_observation=[0.0]),
        Action("approach",   predicted_observation=[9.0], info_gain=0.1),
        Action("overshoot",  predicted_observation=[20.0]),
    ]
    choice = agent.act(actions)
    print(f"\nchosen action        : {choice.to_dict()}")
    assert choice.action.name == "approach"     # closest to the preferred state

    # NONLINEAR generative model handled via numeric Jacobian
    nl = PredictiveCore(belief=[1.0], generative_model=lambda s: [s[0] ** 2],
                        sensory_precision=1.0, prior_precision=0.01, learning_rate=0.05)
    for _ in range(60):
        nl.perceive([9.0])                       # wants s^2 == 9 -> s ≈ 3
    print(f"\nnonlinear belief     : {nl.mu[0]:.3f} (g(s)=s^2, target obs 9 -> s≈3)")
    assert abs(nl.mu[0] - 3.0) < 0.3

    # precision learning (attention): high error -> precision drops, stable -> rises
    core.update_precision()
    print(f"\nadapted precision    : {core.sensory_precision:.3f}")
    print(f"status               : {core.status()}")

    print("\nALL SELF-TESTS PASSED ✓")
