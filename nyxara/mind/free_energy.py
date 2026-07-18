"""NYXARA · mind/free_energy.py — the single objective (✦★★).

Active inference as the **one** objective. Perception and action do not carry
separate goals: both minimise (expected) free energy, computed here and only here.

* **Perception** minimises *variational* free energy — delegated to the shared
  :class:`~nyxara.mind.predictive_core.PredictiveCore` instance (same belief μ,
  same preferences, same precisions the action side reads).
* **Action** minimises *expected* free energy over imagined futures:

      EFE(π) = risk + ambiguity − epistemic

  - **risk** — how far the imagined outcomes fall from the preference prior C
    (Σ_t γ_d^t · −ln C(s_t)). Goals, drives and legacy reward functions all enter
    through ln C — there is no separate reward channel in the objective.
  - **ambiguity** — expected observation fog along the way (bounded proxy
    λ_amb·(1−confidence); the raw −ln conf would diverge on unseen actions and
    kill exploration, so ambiguity is deliberately kept below the epistemic pull).
  - **epistemic** — expected information gain, **computed** from the world model's
    honest uncertainty (:meth:`WorldModel.learning_progress`: novelty + ensemble
    disagreement) — never a hand-tuned constant. This term *is* curiosity: with a
    flat C the agent still seeks whatever it cannot yet predict. Exploration is
    emergent, not a bolted-on bonus.

* **Policy posterior** q(π) ∝ exp(−γ·EFE(π) + κ·ln E(π)) — a softmax with
  adaptive precision γ (an ignorant model holds a flat posterior, so behavioural
  exploration also emerges from low γ) and a habit prior E(π) learned from the
  agent's own past selections (habits emerge from repeated low-EFE choices).

* **Sophisticated planning** — :meth:`FreeEnergyEngine.plan` grows policies by
  depth-limited tree search over the world model's known actions, beam-pruned by
  partial EFE, so the engine does not depend on being handed candidate policies.

Pure Python (numpy fast path for the posterior when available, bit-compatible
fallback otherwise). No LLM anywhere. Fail-soft: a missing world model or
predictive core degrades gracefully, never raises out of ``select``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # numpy accelerates the posterior; the pure-Python path is always present
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001 — numpy is an optimisation, never a requirement
    _np = None  # type: ignore
    _HAS_NUMPY = False

from nyxara.mind.world_model import _dist

__all__ = [
    "PreferenceModel",
    "PolicyEvaluation",
    "FreeEnergyEngine",
]

State = Tuple[Any, ...]
Action = Any
Policy = Any  # Sequence[Action] | Callable[[State], Action]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _softmax(logits: Sequence[float], *, use_numpy: Optional[bool] = None) -> List[float]:
    """Stable softmax. ``use_numpy=None`` auto-selects (numpy for larger vectors);
    both paths produce the same numbers (tested to 1e-9)."""
    if not logits:
        return []
    if use_numpy is None:
        use_numpy = _HAS_NUMPY and len(logits) >= 16
    if use_numpy and _HAS_NUMPY:
        a = _np.asarray(logits, dtype=float)
        a = a - a.max()
        e = _np.exp(a)
        return [float(x) for x in (e / e.sum())]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps) or 1.0
    return [x / total for x in exps]


def _policy_key(policy: Policy) -> str:
    """A stable identity for a policy, used by the habit prior E(π)."""
    if callable(policy):
        return f"callable:{getattr(policy, '__name__', 'policy')}"
    try:
        return repr(list(policy))
    except Exception:  # noqa: BLE001
        return repr(policy)


# --------------------------------------------------------------------------- #
# The preference prior C — the ONE gate through which goals/rewards/drives enter
# --------------------------------------------------------------------------- #
@dataclass
class PreferenceModel:
    """The prior preference distribution C over outcomes (Gaussian form).

    ``ln C(s) = −0.5·precision·d²(s, target) + reward_weight·r + extra(s)``

    Rewards are *absorbed* into ln C (the standard active-inference reading of
    reward as log-preference) — they are not a separate objective. ``learn``
    lets C itself be shaped by experienced valence, so preferences are learned,
    not only declared.
    """

    target: Optional[List[float]] = None
    precision: float = 1.0
    reward_fn: Optional[Callable[[State, Any, State], float]] = None
    reward_weight: float = 1.0
    log_preference_extra: Optional[Callable[[State], float]] = None
    learning_rate: float = 0.05

    def set_target(self, vec: Optional[Sequence[float]], *,
                   precision: Optional[float] = None) -> None:
        self.target = list(vec) if vec is not None else None
        if precision is not None:
            self.precision = max(0.0, float(precision))

    def log_preference(self, state: Sequence[Any], *, action: Any = None,
                       prev_state: Optional[Sequence[Any]] = None,
                       reward: Optional[float] = None) -> float:
        lp = 0.0
        if self.target is not None:
            lp -= 0.5 * self.precision * _dist(state, self.target) ** 2
        r = reward
        if r is None and self.reward_fn is not None and prev_state is not None:
            try:
                r = float(self.reward_fn(tuple(prev_state), action, tuple(state)))
            except Exception:  # noqa: BLE001 — a broken reward_fn never breaks inference
                r = None
        if r is not None:
            lp += self.reward_weight * float(r)
        if self.log_preference_extra is not None:
            try:
                lp += float(self.log_preference_extra(tuple(state)))
            except Exception:  # noqa: BLE001
                pass
        return lp

    # ---- preference learning: C shaped by experienced valence ---- #
    def learn(self, observation: Sequence[Any], valence: float) -> None:
        """Positive valence pulls the preferred outcome toward what just happened
        (η·valence EMA on numeric dims); negative valence only softens precision —
        it never drags C toward a bad outcome."""
        v = max(-1.0, min(1.0, float(valence)))
        numeric = [float(x) if isinstance(x, (int, float)) and not isinstance(x, bool)
                   else None for x in observation]
        if v > 0.0:
            if self.target is None:
                if any(x is not None for x in numeric):
                    self.target = [x if x is not None else 0.0 for x in numeric]
                    self.precision = max(self.precision * 0.5, 0.05)
                return
            eta = self.learning_rate * v
            for i in range(min(len(self.target), len(numeric))):
                if numeric[i] is not None:
                    self.target[i] += eta * (numeric[i] - self.target[i])
            self.precision = min(self.precision * (1.0 + 0.1 * eta), 100.0)
        elif v < 0.0:
            self.precision = max(self.precision * (1.0 + 0.5 * self.learning_rate * v), 0.01)

    def to_dict(self) -> Dict[str, Any]:
        return {"target": ([round(float(x), 4) for x in self.target]
                           if self.target is not None else None),
                "precision": round(self.precision, 4),
                "reward_weight": round(self.reward_weight, 4),
                "has_reward_fn": self.reward_fn is not None}


# --------------------------------------------------------------------------- #
# Policy evaluation record
# --------------------------------------------------------------------------- #
@dataclass
class PolicyEvaluation:
    policy: Policy
    expected_free_energy: float       # risk + ambiguity − epistemic (lower = better)
    risk: float                       # Σ_t disc^t · (−ln C(s_t))
    ambiguity: float                  # Σ_t disc^t · λ_amb · (1 − conf_t)
    epistemic: float                  # Σ_t disc^t · λ_epi · info_gain(s_t, a_t) — COMPUTED
    total_reward: float = 0.0         # reporting/compat only — NOT part of the objective
    model_confidence: float = 0.0     # reporting only (Conclusion confidence, γ adaptation)
    final_state: Optional[State] = None
    trajectory: Optional[Any] = None
    posterior: float = 0.0            # q(π), filled by policy_posterior()

    def to_dict(self) -> Dict[str, Any]:
        return {"policy": (list(self.policy) if not callable(self.policy)
                           else _policy_key(self.policy)),
                "expected_free_energy": round(self.expected_free_energy, 4),
                "risk": round(self.risk, 4),
                "ambiguity": round(self.ambiguity, 4),
                "epistemic": round(self.epistemic, 4),
                "total_reward": round(self.total_reward, 4),
                "model_confidence": round(self.model_confidence, 4),
                "posterior": round(self.posterior, 4),
                "final_state": self.final_state}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class FreeEnergyEngine:
    """The single free-energy objective shared by perception and action."""

    def __init__(self, *, predictive: Any = None, world_model: Any = None,
                 preferences: Optional[PreferenceModel] = None,
                 epistemic_weight: float = 1.0, ambiguity_weight: float = 0.25,
                 discount: float = 0.97,
                 gamma: float = 4.0, gamma_min: float = 0.5, gamma_max: float = 16.0,
                 habit_weight: float = 0.3, habit_decay: float = 0.995) -> None:
        self.predictive = predictive
        self.world_model = world_model
        self.preferences = preferences if preferences is not None else PreferenceModel()
        self.epistemic_weight = epistemic_weight
        self.ambiguity_weight = ambiguity_weight
        self.discount = discount
        self.gamma = gamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.habit_weight = habit_weight
        self.habit_decay = habit_decay
        self._habit_counts: Dict[str, float] = {}
        self._conf_ema: Optional[float] = None
        self.last_selection: Optional[PolicyEvaluation] = None

    # ------------------------------------------------------------------ #
    # perception — the same F the action side plans against
    # ------------------------------------------------------------------ #
    def perceive(self, observation: Sequence[float], **kw: Any) -> Any:
        if self.predictive is None:
            return None
        return self.predictive.perceive(observation, **kw)

    def variational_free_energy(self, observation: Sequence[float]) -> float:
        if self.predictive is None:
            return 0.0
        return float(self.predictive.free_energy(observation))

    # ------------------------------------------------------------------ #
    # epistemic value — computed, never handed in
    # ------------------------------------------------------------------ #
    def epistemic_value(self, state: Sequence[Any], action: Action) -> float:
        """Expected information gain of ``action`` at ``state`` ∈ [0,1], from the
        world model's honest uncertainty (novelty + ensemble disagreement). When a
        JEPA backend also tracks per-action surprise EMA, the richer of the two
        signals wins. No world model → 0 (nothing to imagine, nothing to gain)."""
        wm = self.world_model
        if wm is None:
            return 0.0
        try:
            base = float(wm.learning_progress(tuple(state), action))
        except Exception:  # noqa: BLE001 — unpredictable region: maximal curiosity
            return 1.0
        # JEPA bridge: per-action surprise EMA (calibrated) can only raise curiosity
        try:
            by_action = getattr(wm, "learning_progress_by_action", None)
            if callable(by_action):
                lp = by_action().get(action)
                if lp is not None:
                    base = max(base, _clamp01(float(lp)))
        except Exception:  # noqa: BLE001
            pass
        return _clamp01(base)

    # ------------------------------------------------------------------ #
    # expected free energy over imagined futures
    # ------------------------------------------------------------------ #
    def evaluate_policy(self, start: Sequence[Any], policy: Policy, *,
                        horizon: int = 8,
                        preferences: Optional[PreferenceModel] = None
                        ) -> PolicyEvaluation:
        prefs = preferences if preferences is not None else self.preferences
        wm = self.world_model
        if wm is None:
            # no imagination available: risk against the start state, max ignorance
            risk = -prefs.log_preference(tuple(start))
            ambiguity = self.ambiguity_weight * max(1, horizon)
            return PolicyEvaluation(policy=policy,
                                    expected_free_energy=risk + ambiguity,
                                    risk=risk, ambiguity=ambiguity, epistemic=0.0,
                                    final_state=tuple(start))
        try:
            traj = wm.rollout(tuple(start), policy, steps=horizon,
                              reward_fn=prefs.reward_fn)
        except Exception:  # noqa: BLE001 — a failed rollout is maximal ignorance
            risk = -prefs.log_preference(tuple(start))
            ambiguity = self.ambiguity_weight * max(1, horizon)
            return PolicyEvaluation(policy=policy,
                                    expected_free_energy=risk + ambiguity,
                                    risk=risk, ambiguity=ambiguity, epistemic=0.0,
                                    final_state=tuple(start))
        risk = 0.0
        ambiguity = 0.0
        epistemic = 0.0
        for t in range(traj.length):
            disc = self.discount ** t
            s_prev = traj.states[t]
            s_next = traj.states[t + 1]
            a = traj.actions[t]
            conf = _clamp01(traj.confidences[t])
            # reward already realised by the rollout (reward_fn or learned model
            # reward) folds into ln C here — never counted as a second objective
            risk += disc * (-prefs.log_preference(s_next, action=a, prev_state=s_prev,
                                                  reward=traj.rewards[t]))
            ambiguity += disc * self.ambiguity_weight * (1.0 - conf)
            epistemic += disc * self.epistemic_weight * self.epistemic_value(s_prev, a)
        efe = risk + ambiguity - epistemic
        return PolicyEvaluation(policy=policy, expected_free_energy=efe, risk=risk,
                                ambiguity=ambiguity, epistemic=epistemic,
                                total_reward=traj.total_reward,
                                model_confidence=traj.mean_confidence,
                                final_state=traj.final_state, trajectory=traj)

    def evaluate_policies(self, start: Sequence[Any], policies: Sequence[Policy], *,
                          horizon: int = 8,
                          preferences: Optional[PreferenceModel] = None
                          ) -> List[PolicyEvaluation]:
        evals = [self.evaluate_policy(start, p, horizon=horizon, preferences=preferences)
                 for p in policies]
        evals.sort(key=lambda e: e.expected_free_energy)
        if evals:
            mean_conf = sum(e.model_confidence for e in evals) / len(evals)
            self._conf_ema = (mean_conf if self._conf_ema is None
                              else 0.9 * self._conf_ema + 0.1 * mean_conf)
        return self.policy_posterior(evals)

    # ------------------------------------------------------------------ #
    # posterior over policies — softmax(−γ·EFE + κ·ln E)
    # ------------------------------------------------------------------ #
    def _habit_logit(self, policy: Policy, n_policies: int) -> float:
        if self.habit_weight <= 0.0 or not self._habit_counts:
            return 0.0
        total = sum(self._habit_counts.values())
        count = self._habit_counts.get(_policy_key(policy), 0.0)
        # Laplace-smoothed E(π): unseen policies are never driven to −∞
        e = (count + 1.0) / (total + max(1, n_policies))
        return self.habit_weight * math.log(e)

    def policy_posterior(self, evals: Sequence[PolicyEvaluation], *,
                         use_numpy: Optional[bool] = None) -> List[PolicyEvaluation]:
        evals = list(evals)
        if not evals:
            return evals
        n = len(evals)
        logits = [-self.gamma * e.expected_free_energy + self._habit_logit(e.policy, n)
                  for e in evals]
        probs = _softmax(logits, use_numpy=use_numpy)
        for e, p in zip(evals, probs):
            e.posterior = p
        return evals

    def update_gamma(self) -> float:
        """Precision over policies: an ignorant model (low confidence EMA) keeps the
        posterior flat — exploration for free; a calibrated model commits."""
        if self._conf_ema is not None:
            self.gamma = self.gamma_min + (self.gamma_max - self.gamma_min) \
                * _clamp01(self._conf_ema)
        return self.gamma

    # ------------------------------------------------------------------ #
    # selection — argmax posterior (habit-aware) or posterior sampling
    # ------------------------------------------------------------------ #
    def select(self, start: Sequence[Any], policies: Sequence[Policy], *,
               horizon: int = 8, sample: bool = False,
               rng: Optional[random.Random] = None,
               preferences: Optional[PreferenceModel] = None
               ) -> Optional[PolicyEvaluation]:
        if not policies:
            return None
        try:
            evals = self.evaluate_policies(start, policies, horizon=horizon,
                                           preferences=preferences)
        except Exception:  # noqa: BLE001 — selection never raises; fall back flat
            evals = [PolicyEvaluation(policy=p, expected_free_energy=0.0, risk=0.0,
                                      ambiguity=0.0, epistemic=0.0,
                                      posterior=1.0 / len(policies))
                     for p in policies]
        if sample:
            r = (rng or random).random()
            acc = 0.0
            chosen = evals[-1]
            for e in evals:
                acc += e.posterior
                if r <= acc:
                    chosen = e
                    break
        else:
            chosen = max(evals, key=lambda e: e.posterior)
        self._reinforce_habit(chosen.policy)
        self.last_selection = chosen
        return chosen

    def _reinforce_habit(self, policy: Policy) -> None:
        for k in self._habit_counts:
            self._habit_counts[k] *= self.habit_decay
        key = _policy_key(policy)
        self._habit_counts[key] = self._habit_counts.get(key, 0.0) + 1.0
        if len(self._habit_counts) > 256:      # bounded memory
            weakest = min(self._habit_counts, key=self._habit_counts.get)
            del self._habit_counts[weakest]

    def habits(self, *, top: int = 8) -> List[Tuple[str, float]]:
        ranked = sorted(self._habit_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, round(v, 3)) for k, v in ranked[:top]]

    # ------------------------------------------------------------------ #
    # sophisticated planning — grow policies, beam-pruned by partial EFE
    # ------------------------------------------------------------------ #
    def plan(self, start: Sequence[Any], *, depth: int = 3, beam: int = 6,
             max_branch: int = 8,
             preferences: Optional[PreferenceModel] = None
             ) -> List[PolicyEvaluation]:
        """Depth-limited tree search over the world model's known actions. Each level
        extends the beam-best partial policies (by EFE) with every action — a
        depth-limited form of sophisticated active inference. Returns final-depth
        evaluations sorted by EFE (empty if no world model / no known actions)."""
        wm = self.world_model
        if wm is None:
            return []
        try:
            actions = list(wm.actions())[:max_branch]
        except Exception:  # noqa: BLE001
            return []
        if not actions:
            return []
        frontier: List[List[Action]] = [[a] for a in actions]
        evals: List[PolicyEvaluation] = []
        for level in range(1, max(1, depth) + 1):
            evals = [self.evaluate_policy(start, p, horizon=level,
                                          preferences=preferences) for p in frontier]
            evals.sort(key=lambda e: e.expected_free_energy)
            evals = evals[:max(1, beam)]
            if level == depth:
                break
            frontier = [list(e.policy) + [a] for e in evals for a in actions]
        return self.policy_posterior(evals)

    # ------------------------------------------------------------------ #
    # bridge into PredictiveCore.act — Actions with COMPUTED info gain
    # ------------------------------------------------------------------ #
    def actions_for_core(self, state: Sequence[Any], action_names: Sequence[Action], *,
                         predicted_obs_fn: Optional[Callable[[State], List[float]]] = None
                         ) -> List[Any]:
        """Build :class:`predictive_core.Action` objects whose ``info_gain`` is the
        engine's computed epistemic value and whose ``predicted_observation`` comes
        from the world model — the bridge that makes ``PredictiveCore.act`` live."""
        from nyxara.mind.predictive_core import Action as CoreAction
        out: List[Any] = []
        state = tuple(state)
        for name in action_names:
            info = self.epistemic_value(state, name)
            obs = self._predicted_core_obs(state, name, predicted_obs_fn)
            out.append(CoreAction(name=str(name), predicted_observation=obs,
                                  info_gain=info))
        return out

    def _predicted_core_obs(self, state: State, action: Action,
                            predicted_obs_fn: Optional[Callable[[State], List[float]]]
                            ) -> List[float]:
        next_state: State = state
        if self.world_model is not None:
            try:
                next_state = self.world_model.predict(state, action).next_state
            except Exception:  # noqa: BLE001
                next_state = state
        if predicted_obs_fn is not None:
            try:
                return [float(x) for x in predicted_obs_fn(tuple(next_state))]
            except Exception:  # noqa: BLE001
                pass
        # numeric dims as floats, padded/truncated to the core's belief dimension
        numeric = [float(x) for x in next_state
                   if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if self.predictive is not None:
            dim = len(self.predictive.mu)
            base = list(self.predictive.predict())
            for i in range(min(dim, len(numeric))):
                base[i] = numeric[i]
            return base
        return numeric or [0.0]

    # ------------------------------------------------------------------ #
    # persistence — learned habits and preferences survive restarts (Rule 7)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {"habit_counts": dict(self._habit_counts),
                "conf_ema": self._conf_ema, "gamma": self.gamma,
                "preferences": self.preferences.to_dict()}

    def load_dict(self, data: Dict[str, Any]) -> None:
        try:
            counts = data.get("habit_counts") or {}
            self._habit_counts = {str(k): float(v) for k, v in counts.items()}
            if data.get("conf_ema") is not None:
                self._conf_ema = float(data["conf_ema"])
            if data.get("gamma") is not None:
                self.gamma = float(data["gamma"])
            prefs = data.get("preferences") or {}
            if prefs.get("target") is not None:
                self.preferences.set_target(prefs["target"],
                                            precision=prefs.get("precision"))
        except Exception:  # noqa: BLE001 — a corrupt snapshot never blocks startup
            pass

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {"gamma": round(self.gamma, 4),
                "confidence_ema": (round(self._conf_ema, 4)
                                   if self._conf_ema is not None else None),
                "preferences": self.preferences.to_dict(),
                "habits": self.habits(top=4),
                "epistemic_weight": self.epistemic_weight,
                "ambiguity_weight": self.ambiguity_weight,
                "has_world_model": self.world_model is not None,
                "has_predictive": self.predictive is not None,
                "last_selection": (self.last_selection.to_dict()
                                   if self.last_selection is not None else None)}


# --------------------------------------------------------------------------- #
# Self-test / demo — the emergent-curiosity proofs
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.world_model import WorldModel

    print("=" * 70)
    print("NYXARA free-energy engine (single objective) self-test")
    print("=" * 70)

    def true_step(x: float, a: str) -> float:
        return {"left": x - 1.0, "right": x + 1.0, "stay": x}[a]

    # a ±1 walk world trained ONLY on x ∈ [−10, 0]: x > 0 is unknown territory
    wm = WorldModel(k=3, distance_scale=2.0)
    rng = random.Random(0)
    for _ in range(400):
        x = rng.uniform(-10.0, 0.0)
        a = rng.choice(["left", "right", "stay"])
        wm.observe((x,), a, (true_step(x, a),), reward=0.0)

    # 1) EMERGENT CURIOSITY: zero preferences, zero reward — epistemic term alone
    eng = FreeEnergyEngine(world_model=wm, preferences=PreferenceModel())
    left, right = ["left"] * 6, ["right"] * 6
    choice = eng.select((-1.0,), [left, right])
    ev = {(_policy_key(e.policy)): e for e in
          eng.evaluate_policies((-1.0,), [left, right])}
    print("\n[1] zero-preference agent (curiosity must be emergent)")
    for k, e in ev.items():
        print(f"    {k}: EFE={e.expected_free_energy:+.3f} risk={e.risk:+.3f} "
              f"amb={e.ambiguity:.3f} epi={e.epistemic:.3f}")
    assert choice is not None and list(choice.policy) == right, \
        "curiosity failed: agent did not explore the unknown region"
    print("    → chose RIGHT (into the unknown) — exploration is free ✓")

    # 2) PREFERENCE-ONLY EXPLOITS: epistemic off, target in the known region
    exploit = FreeEnergyEngine(world_model=wm, epistemic_weight=0.0,
                               preferences=PreferenceModel(target=[-5.0]))
    c2 = exploit.select((-1.0,), [left, right])
    print(f"\n[2] preference-only agent → {_policy_key(c2.policy)}")
    assert list(c2.policy) == left
    print("    → chose LEFT (toward the preferred state) ✓")

    # 3) POSTERIOR + γ: probabilities sum to 1; γ rises with model confidence
    evals = eng.evaluate_policies((-5.0,), [left, right, ["stay"] * 6])
    total_p = sum(e.posterior for e in evals)
    g0 = eng.gamma
    eng.update_gamma()
    print(f"\n[3] posterior mass={total_p:.6f}, γ {g0:.2f} → {eng.gamma:.2f}")
    assert abs(total_p - 1.0) < 1e-9

    # 4) HABITS EMERGE: repeated selection raises posterior mass, EFE unchanged
    before = {_policy_key(e.policy): e.posterior
              for e in eng.evaluate_policies((-5.0,), [left, ["stay"] * 6])}
    for _ in range(12):
        eng.select((-5.0,), [left, ["stay"] * 6])
    winner = _policy_key(eng.last_selection.policy)
    after = {_policy_key(e.policy): e.posterior
             for e in eng.evaluate_policies((-5.0,), [left, ["stay"] * 6])}
    print(f"\n[4] habit {winner}: posterior {before[winner]:.3f} → {after[winner]:.3f}")
    assert after[winner] >= before[winner]

    # 5) SOPHISTICATED PLANNING: no candidates handed in — the engine grows them
    plans = eng.plan((-1.0,), depth=3, beam=4)
    print(f"\n[5] plan() grew {len(plans)} policies; best = "
          f"{plans[0].to_dict()['policy']} (EFE {plans[0].expected_free_energy:+.3f})")
    assert plans and plans[0].epistemic > 0.0

    # 6) numpy path == pure-python path
    logits = [-1.0, 0.5, 2.0, -3.3]
    p_py = _softmax(logits, use_numpy=False)
    p_np = _softmax(logits, use_numpy=True)
    diff = max(abs(a - b) for a, b in zip(p_py, p_np))
    print(f"\n[6] softmax numpy-vs-python max diff = {diff:.2e}")
    assert diff < 1e-9

    print("\nALL SELF-TESTS PASSED ✓")
