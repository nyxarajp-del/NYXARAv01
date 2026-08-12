"""NYXARA · nyx001/layers/l15_resource.py — Layer 15, resource-aware cognition (⏱, effort).

The charter: "Modulates computation based on task difficulty."

Difficulty is *estimated before* the work, from four signals the other layers already produce:
novelty (Layer 1), the world model's uncertainty (Layer 2), how weak the relevant recall was
(Layer 3), and whether any concept covers the input at all (Layer 4). A budget is then allocated
proportionally, and — the part that makes this a control loop rather than a heuristic — the
estimate is **scored against what the work actually cost**, so the estimator's own reliability is
measured and reported.

    difficulty = w·[novelty, uncertainty, recall_weakness, uncovered]
    budget     = floor + (ceiling - floor) · difficulty

:meth:`calibration` is the reliability of the difficulty estimate itself: the correlation between
predicted difficulty and observed cost over a sliding window. A metacontroller whose estimates do
not correlate with reality is spending compute at random, and that is worth knowing about. It
returns ``None`` until there are enough samples, like every other measured quantity here.

:meth:`should_stop` implements the anytime property: work stops when the budget is spent *or* when
marginal improvement per unit cost falls below ``min_gain_rate``. The second condition is what
prevents the common failure of spending a full large budget on a problem that stopped improving
after 5% of it.

Honest: cost is measured in wall-clock seconds and step counts, which are noisy on a shared
machine. The estimator is a fixed linear combination, not learned — Layer 13 can tune its weights,
but nothing here discovers what difficulty *is*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Difficulty", "Budget", "ResourceAwareCognition"]


@dataclass
class Difficulty:
    """An estimate made before the work, with the signals that produced it."""

    score: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    recall_weakness: float = 0.0
    uncovered: float = 0.0

    @property
    def band(self) -> str:
        return "hard" if self.score > 0.66 else ("moderate" if self.score > 0.33 else "easy")

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 4), "band": self.band,
                "novelty": round(self.novelty, 4), "uncertainty": round(self.uncertainty, 4),
                "recall_weakness": round(self.recall_weakness, 4),
                "uncovered": round(self.uncovered, 4)}


@dataclass
class Budget:
    """What this task is allowed to spend, and what it has spent."""

    seconds: float = 0.0
    steps: int = 0
    started: float = field(default_factory=time.time)
    spent_steps: int = 0
    best_value: Optional[float] = None
    difficulty: float = 0.0

    @property
    def elapsed(self) -> float:
        return float(time.time() - self.started)

    @property
    def exhausted(self) -> bool:
        return self.elapsed >= self.seconds or self.spent_steps >= self.steps

    @property
    def fraction_used(self) -> float:
        by_time = self.elapsed / max(1e-9, self.seconds)
        by_steps = self.spent_steps / max(1, self.steps)
        return float(min(1.0, max(by_time, by_steps)))

    def to_dict(self) -> Dict[str, Any]:
        return {"seconds": round(self.seconds, 3), "steps": self.steps,
                "spent_steps": self.spent_steps, "elapsed": round(self.elapsed, 3),
                "fraction_used": round(self.fraction_used, 4), "exhausted": self.exhausted,
                "difficulty": round(self.difficulty, 4)}


class ResourceAwareCognition:
    """Layer 15. Estimates difficulty, allocates a budget, and measures its own accuracy."""

    def __init__(self, substrate: Any, *, floor_s: float = 0.01, ceiling_s: float = 2.0,
                 floor_steps: int = 1, ceiling_steps: int = 64,
                 min_gain_rate: float = 1e-4, window: int = 128, min_samples: int = 16,
                 w_novelty: float = 0.3, w_uncertainty: float = 0.35,
                 w_recall: float = 0.2, w_uncovered: float = 0.15) -> None:
        self.substrate = substrate
        self.floor_s = float(floor_s)
        self.ceiling_s = float(ceiling_s)
        self.floor_steps = int(floor_steps)
        self.ceiling_steps = int(ceiling_steps)
        self.min_gain_rate = float(min_gain_rate)
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.w_novelty = float(w_novelty)
        self.w_uncertainty = float(w_uncertainty)
        self.w_recall = float(w_recall)
        self.w_uncovered = float(w_uncovered)
        self._samples: List[Tuple[float, float]] = []      # (predicted difficulty, observed cost)
        self.allocations = 0
        self.early_stops = 0

    def estimate(self, *, novelty: float = 0.0, uncertainty: float = 0.0,
                 recall_confidence: Optional[float] = None,
                 concept_covered: bool = True) -> Difficulty:
        """Difficulty before the work. Missing signals contribute their neutral value, not zero."""
        try:
            weak = 1.0 - float(recall_confidence) if recall_confidence is not None else 0.5
            unc = 0.0 if concept_covered else 1.0
            d = Difficulty(novelty=float(min(1.0, max(0.0, novelty))),
                           uncertainty=float(min(1.0, max(0.0, uncertainty))),
                           recall_weakness=float(min(1.0, max(0.0, weak))),
                           uncovered=unc)
            d.score = float(min(1.0, max(0.0,
                                         self.w_novelty * d.novelty
                                         + self.w_uncertainty * d.uncertainty
                                         + self.w_recall * d.recall_weakness
                                         + self.w_uncovered * d.uncovered)))
            return d
        except Exception:  # noqa: BLE001
            return Difficulty()

    def allocate(self, difficulty: Difficulty) -> Budget:
        """Budget proportional to difficulty, between the configured floor and ceiling."""
        d = float(min(1.0, max(0.0, difficulty.score)))
        self.allocations += 1
        return Budget(seconds=self.floor_s + (self.ceiling_s - self.floor_s) * d,
                      steps=int(self.floor_steps + (self.ceiling_steps - self.floor_steps) * d),
                      difficulty=d)

    def should_stop(self, budget: Budget, value: Optional[float] = None) -> bool:
        """Stop on an exhausted budget, or when improvement per unit cost has flattened.

        The second condition is the anytime property: a task that stopped improving after 5% of a
        large budget should not spend the other 95% to prove it.
        """
        try:
            if budget.exhausted:
                return True
            if value is None:
                return False
            if budget.best_value is None:
                budget.best_value = float(value)
                return False
            gain = float(value) - budget.best_value
            if gain > 0:
                budget.best_value = float(value)
            spent = max(1e-6, budget.fraction_used)
            if budget.spent_steps >= 2 and (gain / spent) < self.min_gain_rate:
                self.early_stops += 1
                return True
            return False
        except Exception:  # noqa: BLE001
            return budget.exhausted

    def record(self, budget: Budget) -> None:
        """Score the estimate against what the work actually cost."""
        try:
            self._samples.append((budget.difficulty, budget.fraction_used))
            if len(self._samples) > self.window:
                self._samples = self._samples[-self.window:]
        except Exception:  # noqa: BLE001
            pass

    def calibration(self) -> Optional[float]:
        """Correlation between predicted difficulty and observed cost. ``None`` when unknown."""
        if np is None or len(self._samples) < self.min_samples:
            return None
        try:
            a = np.asarray([p for p, _ in self._samples], dtype=np.float64)
            b = np.asarray([c for _, c in self._samples], dtype=np.float64)
            if float(a.std()) < 1e-9 or float(b.std()) < 1e-9:
                return None            # no variance to correlate — undefined, not zero
            return float(np.corrcoef(a, b)[0, 1])
        except Exception:  # noqa: BLE001
            return None

    def time_budget_signal(self, budget: Optional[Budget]) -> float:
        """The ``T`` signal the optimizer reads: 1.0 when ample, → 0.0 as the budget runs out."""
        if budget is None:
            return 1.0
        return float(max(0.0, 1.0 - budget.fraction_used))

    def stats(self) -> Dict[str, Any]:
        return {"allocations": self.allocations, "early_stops": self.early_stops,
                "samples": len(self._samples), "calibration": self.calibration(),
                "floor_s": self.floor_s, "ceiling_s": self.ceiling_s}

    def to_dict(self) -> Dict[str, Any]:
        return {"allocations": self.allocations, "early_stops": self.early_stops,
                "samples": [[round(a, 4), round(b, 4)] for a, b in self._samples[-128:]]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.allocations = int(d.get("allocations", 0))
            self.early_stops = int(d.get("early_stops", 0))
            self._samples = [(float(a), float(b)) for a, b in (d.get("samples") or [])]
        except Exception:  # noqa: BLE001
            pass
