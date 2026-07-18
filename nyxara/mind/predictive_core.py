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
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Vec",
    "EmotionReadout",
    "PerceptionResult",
    "Action",
    "ActionChoice",
    "PredictiveCore",
    "HierarchicalPredictiveCore",
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
    """Affect as a function of prediction-error dynamics.

    Beyond the instantaneous triple, the deep read-outs track *slow* free-energy
    dynamics: **mood** is a slow EMA of valence (the trend of ΔF), **anxiety** is
    the level of *expected* free energy of the best available policy (how much
    surprise the future is predicted to hold), **relief** is anxiety falling, and
    **confidence_feeling** is the current precision over policies (γ), normalised.
    """

    valence: float    # [-1, 1] — free energy falling (good) vs rising (bad)
    arousal: float    # [0, 1]  — precision-weighted error magnitude
    surprise: float   # [0, 1]  — current free-energy level
    mood: float = 0.0                # [-1, 1] — slow EMA of valence
    anxiety: float = 0.0             # [0, 1]  — expected future free energy (best policy)
    relief: float = 0.0              # [0, 1]  — anxiety dropping
    confidence_feeling: float = 0.0  # [0, 1]  — policy precision γ, normalised

    def to_dict(self) -> Dict[str, float]:
        return {"valence": round(self.valence, 4), "arousal": round(self.arousal, 4),
                "surprise": round(self.surprise, 4), "mood": round(self.mood, 4),
                "anxiety": round(self.anxiety, 4), "relief": round(self.relief, 4),
                "confidence_feeling": round(self.confidence_feeling, 4)}


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
        # per-dimension precision (diagonal Π): None → the scalar path, bit-identical
        self.precision_vec: Optional[Vec] = None
        # Welford per-dim error statistics + volatility (variance-of-variance EMA)
        self._err_n: int = 0
        self._err_mean: Vec = []
        self._err_m2: Vec = []
        self._volatility: float = 0.0
        self._prev_var: Optional[Vec] = None
        # deep affect state (slow F dynamics)
        self._mood: float = 0.0
        self._anxiety: float = 0.0
        self._relief: float = 0.0
        self._confidence_feeling: float = 0.0

    # ---- prediction ---- #
    def predict(self) -> Vec:
        return self.g(self.mu)

    # ---- free energy ---- #
    def free_energy(self, observation: Sequence[float]) -> float:
        pred = self.predict()
        if self.precision_vec is not None:
            accuracy = 0.5 * sum(p * (o - q) ** 2 for p, o, q in
                                 zip(self.precision_vec, observation, pred))
        else:
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
            if self.precision_vec is not None:            # diagonal Π: weight per dim
                weighted = [p * e for p, e in zip(self.precision_vec, error)]
                accuracy_grad = _jt_vec(J, weighted)      # ∝ -dF/dμ
            else:
                accuracy_grad = _scale(_jt_vec(J, error), self.sensory_precision)
            complexity_grad = _scale(_sub(self.mu, self.prior), self.prior_precision)
            step = _sub(accuracy_grad, complexity_grad)   # ascent on -F == descent on F
            self.mu = _add(self.mu, _scale(step, self.lr))
            iters_done += 1

        pred = self.predict()
        error = _sub(obs, pred)
        F = self.free_energy(obs)
        F_prev = self._F_history[-1] if self._F_history else F
        self._F_history.append(F)
        self._last_error_mag = math.sqrt(_norm2(error))
        self._track_error_stats(error)
        # mood: slow EMA of instantaneous valence (the trend of ΔF)
        self._mood = 0.95 * self._mood + 0.05 * math.tanh(F_prev - F)
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

    def policy_posterior(self, choices: Sequence[ActionChoice], *,
                         gamma: float = 4.0) -> List[float]:
        """q(a) = softmax(−γ·EFE) — the posterior over actions (stable, sums to 1)."""
        if not choices:
            return []
        logits = [-gamma * c.expected_free_energy for c in choices]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        total = sum(exps) or 1.0
        return [x / total for x in exps]

    def act(self, actions: Sequence[Action], *, gamma: Optional[float] = None,
            sample: bool = False, rng: Optional[object] = None) -> ActionChoice:
        """Active inference: pick the action with the lowest expected free energy.

        Default behaviour (no ``gamma``/``sample``) is the exact argmin. With
        ``gamma`` set, the posterior softmax(−γ·EFE) is recorded on each choice;
        ``sample=True`` draws from that posterior instead of taking the argmin —
        low γ (an uncertain agent) then explores for free."""
        if not actions:
            raise ValueError("no actions to choose from")
        choices = [self.expected_free_energy(a) for a in actions]
        choices.sort(key=lambda c: c.expected_free_energy)
        best = choices[0]
        if gamma is not None:
            probs = self.policy_posterior(choices, gamma=gamma)
            if sample:
                import random as _random
                r = (rng or _random).random()
                acc = 0.0
                for c, p in zip(choices, probs):
                    acc += p
                    if r <= acc:
                        best = c
                        break
        # anxiety: the expected free energy of the best available future
        prev_anx = self._anxiety
        self._anxiety = _clamp(math.tanh(max(0.0, best.expected_free_energy)))
        self._relief = _clamp(prev_anx - self._anxiety)
        if gamma is not None:
            self._confidence_feeling = _clamp(gamma / 16.0)
        return best

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
                              surprise=surprise,
                              mood=_clamp(self._mood, -1.0, 1.0),
                              anxiety=self._anxiety, relief=self._relief,
                              confidence_feeling=self._confidence_feeling)

    # ---- combined step ---- #
    def step(self, observation: Sequence[float], *, iterations: int = 8,
             scale: float = 1.0) -> Tuple[PerceptionResult, EmotionReadout]:
        perception = self.perceive(observation, iterations=iterations)
        feeling = self.emotion(scale=scale)
        return perception, feeling

    # ---- preferences: the C target lives on the SAME instance perception runs on ---- #
    def set_preference(self, preference: Optional[Sequence[float]], *,
                       precision: Optional[float] = None) -> None:
        """Move the preferred observation (the C prior) — the one place goals enter."""
        self.preference = list(preference) if preference is not None else None
        if precision is not None:
            self.preference_precision = max(0.0, float(precision))

    # ---- precision learning (attention) ---- #
    def update_precision(self, *, floor: float = 0.05, ceil: float = 100.0) -> float:
        """Adapt sensory precision toward the inverse of recent error variance."""
        var = self._last_error_mag ** 2
        self.sensory_precision = _clamp(1.0 / (var + 1e-3), floor, ceil)
        return self.sensory_precision

    def _track_error_stats(self, error: Sequence[float]) -> None:
        """Welford per-dimension error statistics + a volatility EMA (how fast the
        error variance itself is drifting). Pure bookkeeping — outputs unchanged
        until :meth:`update_precision_vec` is called."""
        n = len(error)
        if len(self._err_mean) != n:
            self._err_mean = [0.0] * n
            self._err_m2 = [0.0] * n
            self._err_n = 0
            self._prev_var = None
        self._err_n += 1
        for i, e in enumerate(error):
            d = e - self._err_mean[i]
            self._err_mean[i] += d / self._err_n
            self._err_m2[i] += d * (e - self._err_mean[i])
        if self._err_n >= 2:
            var = [m2 / (self._err_n - 1) for m2 in self._err_m2]
            if self._prev_var is not None:
                drift = sum(abs(a - b) for a, b in zip(var, self._prev_var)) / n
                self._volatility = 0.9 * self._volatility + 0.1 * drift
            self._prev_var = var

    def update_precision_vec(self, *, floor: float = 0.05, ceil: float = 100.0) -> Optional[Vec]:
        """Per-dimension precision (diagonal Π): each dim's precision tracks the inverse
        of ITS error variance — noisy channels are trusted less, stable ones more (the
        real mechanism of attention). Volatility (a fast-drifting world) lowers all
        precisions: when the rules keep changing, hold beliefs more loosely."""
        if self._err_n < 2:
            return self.precision_vec
        damp = 1.0 / (1.0 + self._volatility)
        self.precision_vec = [
            _clamp(damp / (m2 / (self._err_n - 1) + 1e-3), floor, ceil)
            for m2 in self._err_m2]
        return self.precision_vec

    @property
    def volatility(self) -> float:
        return self._volatility

    def last_free_energy(self) -> Optional[float]:
        return self._F_history[-1] if self._F_history else None

    # ---- introspection ---- #
    def status(self) -> Dict[str, object]:
        return {"belief": [round(x, 4) for x in self.mu],
                "prediction": [round(x, 4) for x in self.predict()],
                "free_energy": round(self._F_history[-1], 6) if self._F_history else None,
                "sensory_precision": round(self.sensory_precision, 4),
                "emotion": self.emotion().to_dict()}


# --------------------------------------------------------------------------- #
# Hierarchical predictive core — empirical priors flow down, errors flow up
# --------------------------------------------------------------------------- #
class HierarchicalPredictiveCore:
    """Two-level hierarchical predictive coding over ONE free-energy objective.

    * The **lower** (fast, sensory) level perceives every observation exactly like
      a flat :class:`PredictiveCore`.
    * The **upper** (slow, conceptual) level perceives the lower level's *belief*
      every ``slow_every`` steps with a small learning rate — it extracts the slow
      regularities the fast level rides on.
    * The upper level's prediction becomes the lower level's **empirical prior**
      (canonical hierarchical predictive coding: priors flow down, prediction
      errors flow up), so transient noise is resisted while real change passes.

    The public surface mirrors :class:`PredictiveCore` (``mu``, ``perceive``,
    ``step``, ``free_energy``, ``act``, ``set_preference``, ``emotion``,
    ``status``…) so it is a drop-in for the orchestrator's predictive spine.
    """

    def __init__(self, *, belief: Vec, slow_every: int = 4, slow_lr: float = 0.05,
                 slow_prior_precision: float = 0.02, prior_coupling: float = 0.3,
                 **lower_kw: object) -> None:
        self.lower = PredictiveCore(belief=list(belief), **lower_kw)  # type: ignore[arg-type]
        self.upper = PredictiveCore(belief=list(belief), learning_rate=slow_lr,
                                    prior_precision=slow_prior_precision)
        self.slow_every = max(1, int(slow_every))
        self.prior_coupling = _clamp(prior_coupling)
        self._ticks = 0

    # ---- the shared belief surface (delegates to the fast level) ---- #
    @property
    def mu(self) -> Vec:
        return self.lower.mu

    @property
    def preference(self) -> Optional[Vec]:
        return self.lower.preference

    def predict(self) -> Vec:
        return self.lower.predict()

    def free_energy(self, observation: Sequence[float]) -> float:
        return self.lower.free_energy(observation)

    def last_free_energy(self) -> Optional[float]:
        return self.lower.last_free_energy()

    # ---- hierarchical perception ---- #
    def perceive(self, observation: Sequence[float], *, iterations: int = 8
                 ) -> PerceptionResult:
        result = self.lower.perceive(observation, iterations=iterations)
        self._ticks += 1
        if self._ticks % self.slow_every == 0:
            # errors flow up: the slow level perceives the fast level's belief
            self.upper.perceive(self.lower.mu, iterations=max(1, iterations // 2))
        # priors flow down: the slow prediction becomes the fast empirical prior
        top_down = self.upper.predict()
        c = self.prior_coupling
        self.lower.prior = [(1.0 - c) * p + c * t
                            for p, t in zip(self.lower.prior, top_down)]
        return result

    def step(self, observation: Sequence[float], *, iterations: int = 8,
             scale: float = 1.0) -> Tuple[PerceptionResult, EmotionReadout]:
        perception = self.perceive(observation, iterations=iterations)
        return perception, self.emotion(scale=scale)

    # ---- action + affect + precision: delegate to the fast level ---- #
    def expected_free_energy(self, action: Action) -> ActionChoice:
        return self.lower.expected_free_energy(action)

    def act(self, actions: Sequence[Action], **kw: object) -> ActionChoice:
        return self.lower.act(actions, **kw)  # type: ignore[arg-type]

    def policy_posterior(self, choices: Sequence[ActionChoice], *,
                         gamma: float = 4.0) -> List[float]:
        return self.lower.policy_posterior(choices, gamma=gamma)

    def set_preference(self, preference: Optional[Sequence[float]], *,
                       precision: Optional[float] = None) -> None:
        self.lower.set_preference(preference, precision=precision)

    def emotion(self, *, scale: float = 1.0) -> EmotionReadout:
        return self.lower.emotion(scale=scale)

    def update_precision(self, **kw: object) -> float:
        return self.lower.update_precision(**kw)  # type: ignore[arg-type]

    def update_precision_vec(self, **kw: object) -> Optional[Vec]:
        return self.lower.update_precision_vec(**kw)  # type: ignore[arg-type]

    def status(self) -> Dict[str, object]:
        s = self.lower.status()
        s["upper_belief"] = [round(x, 4) for x in self.upper.mu]
        s["ticks"] = self._ticks
        return s


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
