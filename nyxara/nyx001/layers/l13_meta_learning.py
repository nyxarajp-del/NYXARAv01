"""NYXARA · nyx001/layers/l13_meta_learning.py — Layer 13, learning to learn (📈, strategy).

The charter: "Learning how to learn — Task failure → update learning strategy → better strategy."

A :class:`Strategy` here is a concrete, applied bundle of learning hyperparameters — learning rate,
replay batch size, the objective's term weights, and the optimizer's cognitive gain weights. Meta-
learning is a **bandit over strategies**, scored by the only thing that matters: how fast the world
model's error actually fell while a strategy was in play.

    reward(strategy) = (error_before - error_after) / max(1, steps_taken)

That is learning *rate of progress per step*, not final error — a strategy that reaches a good
error slowly is worse than one that reaches a slightly worse error quickly, because the slow one
costs experience the system does not have infinite amounts of.

Selection is UCB1, which is the part that keeps this honest. A greedy selector locks onto whichever
strategy happened to win its first trial and never learns that another was better; UCB1's
exploration bonus is proportional to ``sqrt(ln(total)/n_i)``, so a strategy tried twice is
revisited even while a strategy tried fifty times is winning. :meth:`confident` reports whether
enough trials have run for the ranking to mean anything, and it is False for a long time.

Mutation is bounded: a new strategy is an existing one perturbed within hard limits, so meta-
learning cannot wander into a configuration that destroys the learner. That bound is a real safety
property — an unbounded strategy search on a live system is a way to lose the model.

Honest: this optimises hyperparameters against a measured signal. It does not invent new learning
*algorithms*; the update rule stays the one in :mod:`~nyxara.nyx001.substrate.optim`. Calling
hyperparameter search "learning to learn" is a stretch the charter's phrasing invites, and this
docstring is where that stretch is admitted rather than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

__all__ = ["Strategy", "MetaLearner"]

# Hard bounds. Meta-learning may search inside these and nowhere else.
_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lr": (1e-5, 5e-2),
    "replay_batch": (1.0, 64.0),
    "w_prediction": (0.1, 2.0),
    "w_concept": (0.0, 1.0),
    "w_compression": (0.0, 0.5),
    "w_curiosity": (0.0, 0.3),
    "w_error": (0.0, 2.0),
    "w_uncertainty": (0.0, 2.0),
}


@dataclass
class Strategy:
    """One learning configuration, with its measured track record."""

    name: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    trials: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0

    @property
    def mean_reward(self) -> Optional[float]:
        """``None`` until tried — an untried strategy has no record, not a zero one."""
        return float(self.total_reward / self.trials) if self.trials else None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "trials": self.trials,
                "mean_reward": None if self.mean_reward is None else round(self.mean_reward, 8),
                "best_reward": round(self.best_reward, 8),
                "params": {k: round(v, 6) for k, v in self.params.items()}}


class MetaLearner:
    """Layer 13. A bounded UCB1 bandit over learning strategies, scored by learning speed."""

    def __init__(self, substrate: Any, *, max_strategies: int = 12, explore_c: float = 1.4,
                 min_trials: int = 3, mutation: float = 0.35) -> None:
        self.substrate = substrate
        self.max_strategies = int(max_strategies)
        self.explore_c = float(explore_c)
        self.min_trials = int(min_trials)
        self.mutation = float(mutation)
        self._strategies: Dict[str, Strategy] = {}
        self._current: Optional[str] = None
        self._trial_start: Optional[Tuple[float, int]] = None    # (error, steps) at trial start
        self.trials = 0
        self.mutations = 0
        self._rng = getattr(substrate, "stream", lambda _n: None)("meta_learning")
        self._seed_default()

    def _seed_default(self) -> None:
        self._strategies["baseline"] = Strategy(name="baseline", params={
            "lr": 3e-3, "replay_batch": 8.0, "w_prediction": 1.0, "w_concept": 0.2,
            "w_compression": 0.05, "w_curiosity": 0.05, "w_error": 0.6, "w_uncertainty": 0.5,
        })

    # ---- selection ---- #
    def select(self) -> Strategy:
        """UCB1. Untried strategies are taken first — an unknown is worth more than a known mean."""
        items = list(self._strategies.values())
        if not items:
            self._seed_default()
            items = list(self._strategies.values())
        untried = [s for s in items if s.trials == 0]
        chosen = untried[0] if untried else max(items, key=lambda s: self._ucb(s))
        self._current = chosen.name
        return chosen

    def _ucb(self, s: Strategy) -> float:
        if s.trials == 0:
            return float("inf")
        mean = s.mean_reward or 0.0
        return float(mean + self.explore_c * math.sqrt(math.log(max(2, self.trials)) / s.trials))

    # ---- the trial ---- #
    def begin(self, error: Optional[float], steps: int) -> Optional[Strategy]:
        """Start measuring a strategy. Needs a baseline error to score against."""
        if error is None:
            return None
        self._trial_start = (float(error), int(steps))
        return self.select()

    def end(self, error: Optional[float], steps: int) -> Optional[float]:
        """Close the trial and credit the strategy with the learning speed it achieved."""
        if self._trial_start is None or self._current is None or error is None:
            return None
        try:
            e0, s0 = self._trial_start
            took = max(1, int(steps) - s0)
            reward = float((e0 - float(error)) / took)
            s = self._strategies.get(self._current)
            if s is not None:
                s.trials += 1
                s.total_reward += reward
                s.best_reward = max(s.best_reward, reward)
            self.trials += 1
            self._trial_start = None
            return reward
        except Exception:  # noqa: BLE001
            return None

    # ---- mutation ---- #
    def mutate(self, base: Optional[Strategy] = None) -> Optional[Strategy]:
        """Perturb a strategy inside the hard bounds. Never produces an out-of-range learner."""
        try:
            src = base or self.best() or self._strategies.get("baseline")
            if src is None:
                return None
            if len(self._strategies) >= self.max_strategies:
                self._retire_worst()
            rng = self._rng
            params: Dict[str, float] = {}
            for k, v in src.params.items():
                lo, hi = _BOUNDS.get(k, (v, v))
                factor = 1.0 + (rng.uniform(-self.mutation, self.mutation) if rng is not None
                                else 0.0)
                params[k] = float(min(hi, max(lo, v * factor)))
            name = f"s{len(self._strategies)}"
            s = Strategy(name=name, params=params)
            self._strategies[name] = s
            self.mutations += 1
            return s
        except Exception:  # noqa: BLE001
            return None

    def _retire_worst(self) -> None:
        """Retire the worst *tried* strategy. Baseline and untried ones are never retired."""
        tried = [s for s in self._strategies.values()
                 if s.trials >= self.min_trials and s.name != "baseline"]
        if not tried:
            return
        victim = min(tried, key=lambda s: s.mean_reward or 0.0)
        self._strategies.pop(victim.name, None)

    # ---- read ---- #
    def best(self) -> Optional[Strategy]:
        """Best measured strategy, or ``None`` when none has been tried enough to say."""
        tried = [s for s in self._strategies.values() if s.trials >= self.min_trials]
        if not tried:
            return None
        return max(tried, key=lambda s: s.mean_reward or 0.0)

    def confident(self) -> bool:
        """Have enough trials run for the ranking to mean anything? False for a long time."""
        tried = [s for s in self._strategies.values() if s.trials >= self.min_trials]
        return len(tried) >= 2 and self.trials >= self.min_trials * 3

    def improved(self) -> Optional[bool]:
        """Has meta-learning beaten the baseline it started from? ``None`` while unknown."""
        base = self._strategies.get("baseline")
        best = self.best()
        if base is None or best is None or base.mean_reward is None or not self.confident():
            return None
        return bool((best.mean_reward or 0.0) > base.mean_reward)

    def apply_to(self, *, optimizer: Any = None, objective: Any = None,
                 memory: Any = None, strategy: Optional[Strategy] = None) -> Dict[str, Any]:
        """Push a strategy's parameters into the live learner. Returns what was actually changed."""
        s = strategy or self._strategies.get(self._current or "") or self.best()
        if s is None:
            return {}
        applied: Dict[str, Any] = {}
        try:
            p = s.params
            if optimizer is not None:
                for attr, key in (("lr", "lr"), ("w_error", "w_error"),
                                  ("w_uncertainty", "w_uncertainty")):
                    if key in p:
                        setattr(optimizer, attr, float(p[key]))
                        applied[attr] = float(p[key])
            if objective is not None:
                for attr in ("w_prediction", "w_concept", "w_compression", "w_curiosity"):
                    if attr in p:
                        setattr(objective, attr, float(p[attr]))
                        applied[attr] = float(p[attr])
            if memory is not None and "replay_batch" in p:
                memory.replay_batch = int(max(1, p["replay_batch"]))
                applied["replay_batch"] = memory.replay_batch
            return applied
        except Exception:  # noqa: BLE001
            return applied

    def stats(self) -> Dict[str, Any]:
        best = self.best()
        return {"strategies": len(self._strategies), "trials": self.trials,
                "mutations": self.mutations, "confident": self.confident(),
                "improved": self.improved(),
                "best": best.to_dict() if best is not None else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"trials": self.trials, "mutations": self.mutations,
                "strategies": [s.to_dict() for s in self._strategies.values()]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.trials = int(d.get("trials", 0))
            self.mutations = int(d.get("mutations", 0))
            for item in d.get("strategies") or []:
                name = str(item.get("name", ""))
                if not name:
                    continue
                s = Strategy(name=name, params={k: float(v) for k, v in
                                                (item.get("params") or {}).items()},
                             trials=int(item.get("trials", 0)),
                             best_reward=float(item.get("best_reward", 0.0)))
                mean = item.get("mean_reward")
                s.total_reward = float(mean) * s.trials if mean is not None else 0.0
                self._strategies[name] = s
        except Exception:  # noqa: BLE001
            pass
