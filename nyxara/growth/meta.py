"""NYXARA · growth/meta.py — meta-learning: learning to learn (🧠, self-tuning).

The growth layer learns skills; this module learns *how NYXARA learns* — and improves the
learning process itself. It treats each way of learning as a :class:`Strategy` (a bundle of
hyperparameters — learning rate, replay, anchoring) and, across many tasks, works out which
strategy pays off where, how to tune its hyperparameters, and how to carry what worked on one
skill over to a new one.

Four meta-capabilities:

* **Strategy selection (contextual bandit).** For a given kind of task, pick the strategy with
  the best expected improvement, exploring the untried ones via an upper-confidence bound — so
  NYXARA neither blindly repeats a mediocre method nor forgets to try a better one.
* **Hyperparameter self-tuning.** :meth:`MetaLearner.optimize` runs a gradient-free search,
  proposing perturbed strategy variants and keeping the ones that learn faster — the learner
  tunes itself.
* **Transfer.** A new task is matched to the most similar past task, and the strategy that
  worked there is warm-started — knowledge of *how to learn* transfers across skills.
* **Meta-rules.** Across the record it extracts generalisable rules ("for volatile tasks,
  prefer the cautious strategy") — explicit, inspectable meta-skills.

Self-improvement of the growth machinery is bounded like everything else: it tunes *how* to
learn, never *what NYXARA is*. It produces strategy configurations; it never touches the
character core.

Pure standard library.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Strategy",
    "Trial",
    "MetaLearner",
]

Features = Dict[str, float]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Strategy & trial
# --------------------------------------------------------------------------- #
@dataclass
class Strategy:
    name: str
    hyperparams: Dict[str, float] = field(default_factory=dict)
    kinds: frozenset = frozenset()        # applicable task kinds; empty = any

    def applies_to(self, kind: str) -> bool:
        return not self.kinds or kind in self.kinds

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "hyperparams": dict(self.hyperparams),
                "kinds": sorted(self.kinds)}


@dataclass
class Trial:
    strategy: str
    task_kind: str
    features: Features
    improvement: float
    cost: float = 1.0
    at: float = 0.0


# --------------------------------------------------------------------------- #
# Meta-learner
# --------------------------------------------------------------------------- #
class MetaLearner:
    """Learns which learning strategies work where, tunes them, and transfers them."""

    def __init__(self, *, ucb_c: float = 1.0, seed: int = 0) -> None:
        self.ucb_c = ucb_c
        self._strategies: Dict[str, Strategy] = {}
        self._trials: List[Trial] = []
        self._rng = random.Random(seed)

    # ---- registration ---- #
    def register(self, strategy: Strategy) -> Strategy:
        self._strategies[strategy.name] = strategy
        return strategy

    def strategies(self) -> List[Strategy]:
        return list(self._strategies.values())

    # ---- record outcomes ---- #
    def record(self, strategy: str, task_kind: str, features: Features,
               improvement: float, *, cost: float = 1.0) -> None:
        self._trials.append(Trial(strategy, task_kind, dict(features), improvement, cost))

    # ---- statistics ---- #
    def _ctx_trials(self, name: str, kind: str) -> List[Trial]:
        return [t for t in self._trials if t.strategy == name and t.task_kind == kind]

    def value(self, name: str, kind: str) -> float:
        ts = self._ctx_trials(name, kind)
        return sum(t.improvement for t in ts) / len(ts) if ts else 0.0

    def _ucb(self, name: str, kind: str, total: int) -> float:
        ts = self._ctx_trials(name, kind)
        if not ts:
            return float("inf")     # always try the untried at least once
        mean = sum(t.improvement for t in ts) / len(ts)
        return mean + self.ucb_c * math.sqrt(math.log(total + 1) / len(ts))

    # ---- strategy selection (contextual UCB) ---- #
    def select(self, task_kind: str, *, candidates: Optional[Sequence[str]] = None
               ) -> Optional[Strategy]:
        pool = [self._strategies[n] for n in (candidates or self._strategies)
                if n in self._strategies and self._strategies[n].applies_to(task_kind)]
        if not pool:
            return None
        total = sum(1 for t in self._trials if t.task_kind == task_kind)
        return max(pool, key=lambda s: self._ucb(s.name, task_kind, total))

    def best_for(self, task_kind: str, *, min_support: int = 1) -> Optional[Strategy]:
        """The strategy with the highest mean improvement on this kind (exploit only)."""
        scored = [(s, self.value(s.name, task_kind), len(self._ctx_trials(s.name, task_kind)))
                  for s in self._strategies.values() if s.applies_to(task_kind)]
        scored = [(s, v, n) for s, v, n in scored if n >= min_support]
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]

    # ---- transfer (warm start from a similar past task) ---- #
    @staticmethod
    def _similarity(a: Features, b: Features) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        num = sum(1.0 - abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
        return num / len(keys)

    def transfer(self, features: Features, *, task_kind: Optional[str] = None
                 ) -> Optional[Tuple[Strategy, float]]:
        """Recommend the strategy that worked on the most similar past task."""
        best: Optional[Tuple[Trial, float]] = None
        for t in self._trials:
            if task_kind is not None and t.task_kind != task_kind:
                continue
            sim = self._similarity(features, t.features) * (0.5 + 0.5 * _clamp(t.improvement))
            if best is None or sim > best[1]:
                best = (t, sim)
        if best is None or best[0].strategy not in self._strategies:
            return None
        return self._strategies[best[0].strategy], best[1]

    # ---- hyperparameter tuning (gradient-free) ---- #
    def propose_variant(self, strategy: Strategy, *, scale: float = 0.2) -> Strategy:
        """A perturbed copy of a strategy, ready to be tried and kept if it learns faster."""
        new_hp = {}
        for k, v in strategy.hyperparams.items():
            new_hp[k] = max(0.0, v + self._rng.gauss(0, scale * (abs(v) + 0.1)))
        variant = Strategy(name=f"{strategy.name}#{uuid.uuid4().hex[:4]}",
                           hyperparams=new_hp, kinds=strategy.kinds)
        return variant

    def optimize(self, task_kind: str, features: Features,
                 evaluate: Callable[[Strategy, Features], float], *,
                 rounds: int = 30, scale: float = 0.2, anneal: float = 0.96
                 ) -> Tuple[Strategy, List[float]]:
        """Self-tune by a (1+1) hill-climb: perturb the current best, keep what learns
        faster, anneal the step. Returns the tuned strategy and the best-so-far trace."""
        best = self.best_for(task_kind) or self.select(task_kind) \
            or next(iter(self._strategies.values()), None)
        if best is None:
            return None, []  # type: ignore[return-value]
        best_score = evaluate(best, features)
        self.record(best.name, task_kind, features, best_score)
        trace = [best_score]
        for _ in range(rounds):
            variant = self.propose_variant(best, scale=scale)
            self.register(variant)
            score = evaluate(variant, features)
            self.record(variant.name, task_kind, features, score)
            if score > best_score:
                best, best_score = variant, score
            scale *= anneal
            trace.append(best_score)
        return best, trace

    # ---- meta-rules (generalisable meta-skills) ---- #
    def meta_rules(self, *, min_support: int = 2) -> List[Dict[str, Any]]:
        kinds = {t.task_kind for t in self._trials}
        rules: List[Dict[str, Any]] = []
        for kind in sorted(kinds):
            best = None
            for s in self._strategies.values():
                ts = self._ctx_trials(s.name, kind)
                if len(ts) < min_support:
                    continue
                mean = sum(t.improvement for t in ts) / len(ts)
                if best is None or mean > best[1]:
                    best = (s.name, mean, len(ts))
            if best is not None:
                rules.append({"task_kind": kind, "prefer": best[0],
                              "mean_improvement": round(best[1], 4), "support": best[2]})
        return rules

    def report(self) -> Dict[str, Any]:
        return {"strategies": len(self._strategies), "trials": len(self._trials),
                "rules": len(self.meta_rules())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-learning self-test")
    print("=" * 70)

    ml = MetaLearner(seed=1)
    ml.register(Strategy("aggressive", {"lr": 0.5}))
    ml.register(Strategy("balanced", {"lr": 0.25}))
    ml.register(Strategy("cautious", {"lr": 0.05}))

    # a synthetic task landscape: volatile tasks want a LOW learning rate
    def evaluate(strategy: Strategy, features: Features) -> float:
        optimal_lr = 0.5 * (1 - features.get("volatility", 0.5))
        return _clamp(1.0 - 2.0 * abs(strategy.hyperparams.get("lr", 0.0) - optimal_lr))

    # STRATEGY SELECTION on a volatile task -> learns to prefer the cautious strategy
    volatile = {"volatility": 0.9}
    for _ in range(40):
        s = ml.select("dialogue")
        ml.record(s.name, "dialogue", volatile, evaluate(s, volatile))
    best = ml.best_for("dialogue")
    print(f"\nbest for volatile   : {best.name} (low lr wins)")
    assert best.name == "cautious"

    # a STABLE task wants a HIGH learning rate -> different best strategy
    stable = {"volatility": 0.0}
    for _ in range(40):
        s = ml.select("batch")
        ml.record(s.name, "batch", stable, evaluate(s, stable))
    best_stable = ml.best_for("batch")
    print(f"best for stable     : {best_stable.name} (high lr wins)")
    assert best_stable.name == "aggressive"

    # META-RULES: the generalisable lessons are extracted
    rules = ml.meta_rules()
    print("\nmeta-rules:")
    for r in rules:
        print(f"  task={r['task_kind']:9s} -> prefer {r['prefer']} "
              f"(mean {r['mean_improvement']})")
    by_kind = {r["task_kind"]: r["prefer"] for r in rules}
    assert by_kind["dialogue"] == "cautious" and by_kind["batch"] == "aggressive"

    # TRANSFER: a brand-new task that is similar to the volatile one warm-starts cautious
    new_task = {"volatility": 0.85}
    rec = ml.transfer(new_task)
    print(f"\ntransfer            : similar task -> warm-start {rec[0].name} (sim {rec[1]:.2f})")
    assert rec[0].name == "cautious"

    # HYPERPARAMETER SELF-TUNING: optimize discovers a near-optimal lr for a volatile task
    ml2 = MetaLearner(seed=2)
    ml2.register(Strategy("seed", {"lr": 0.4}))   # a poor starting point
    best_tuned, trace = ml2.optimize("dialogue", {"volatility": 0.9}, evaluate, rounds=60)
    print(f"\nself-tuned lr       : {best_tuned.hyperparams['lr']:.3f} "
          f"(optimal ~0.05); score {trace[-1]:.3f}")
    assert abs(best_tuned.hyperparams["lr"] - 0.05) < 0.15   # tuned toward the optimum
    assert trace[-1] > trace[0]                               # it improved

    print(f"\nreport              : {ml.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
