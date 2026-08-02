"""NYXARA · mind/active_inference.py — predictive processing / free-energy (🧬, Rule 4, F2).

A passive mind waits for input; a living one **predicts** it and is driven by the gap. This module
gives NYXARA a small **generative model** that forecasts the next value of each stream it watches, so
the **prediction error is her surprise** — the free-energy signal biology minimises. From it she
chooses **epistemic actions**: among safe options she looks where her *uncertainty is highest*, because
that is where an observation buys the most understanding.

* **Predictor** — a bounded, online, torch-free next-value forecaster per source (persistence blended
  with a running mean; an EMA of its own error calibrates "how surprised should I be?").
* **GenerativeModel** — many predictors; ``observe`` returns a squashed surprise in ``[0,1)``, and
  ``expected_information_gain`` estimates how much probing a source would reduce uncertainty.
* **ActiveInference** — ties the model to the existing value-of-information engine
  (:class:`~nyxara.planning.voi.ValueOfInformation`): ``epistemic_action`` picks the source with the
  greatest expected free-energy reduction, and VOI reads its bounded cost/benefit — so perception and
  action minimise the *same* objective.
* **EpistemicScheduler** — the same principle turned inward: track how well she predicts her own
  success per task-family and steer effort toward the family she understands *least*, i.e. curiosity as
  uncertainty-seeking rather than random wandering.

Pure standard library; **no LLM**; it changes only *where she looks and what she practises*, never what
she values. Honest by construction: with no perception available it degrades to a no-op.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.planning.voi import Recommendation, ValueOfInformation

__all__ = [
    "Predictor",
    "GenerativeModel",
    "ActiveInference",
    "EpistemicScheduler",
]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class Predictor:
    """An online next-value forecaster whose own error calibrates its surprise."""

    alpha: float = 0.3
    n: int = 0
    mean: float = 0.0
    last: float = 0.0
    err_ema: float = 1.0

    def predict(self) -> float:
        if self.n == 0:
            return 0.0
        return 0.5 * self.last + 0.5 * self.mean

    def observe(self, x: float) -> float:
        """Fold in an observation; return the squashed surprise (prediction error) in ``[0,1)``."""
        pred = self.predict()
        err = abs(float(x) - pred)
        raw = err / (self.err_ema + 1e-9) if self.n > 0 else 1.0
        # update the model AFTER scoring surprise
        self.err_ema = (1.0 - self.alpha) * self.err_ema + self.alpha * err
        self.n += 1
        self.mean += (float(x) - self.mean) / self.n
        self.last = float(x)
        return _clamp01(raw / (raw + 1.0))

    def uncertainty(self) -> float:
        """Epistemic value of probing this source: unpredictability + novelty, in ``[0,1]``."""
        novelty = 1.0 / math.sqrt(self.n + 1.0)
        scale = self.err_ema + abs(self.mean) + 1.0
        unpredictability = self.err_ema / scale
        return _clamp01(0.6 * unpredictability + 0.4 * novelty)


@dataclass
class GenerativeModel:
    """A family of predictors — the mind's forward model of the streams it watches."""

    predictors: Dict[str, Predictor] = field(default_factory=dict)
    _last_surprise: Dict[str, float] = field(default_factory=dict)

    def _p(self, key: str) -> Predictor:
        return self.predictors.setdefault(key, Predictor())

    def observe(self, key: str, x: float) -> float:
        s = self._p(key).observe(x)
        self._last_surprise[key] = s
        return s

    def surprise(self, key: str) -> float:
        return self._last_surprise.get(key, 1.0)

    def expected_information_gain(self, key: str) -> float:
        p = self.predictors.get(key)
        return 1.0 if p is None or p.n < 2 else p.uncertainty()

    def free_energy(self) -> float:
        """Total current surprise across all watched streams (the quantity she acts to reduce)."""
        if not self._last_surprise:
            return 0.0
        return sum(self._last_surprise.values()) / len(self._last_surprise)


class ActiveInference:
    """Perceive by prediction; act to reduce surprise — over the existing VOI cost/benefit reading."""

    def __init__(self, voi: Optional[ValueOfInformation] = None) -> None:
        self.model = GenerativeModel()
        self.voi = voi or ValueOfInformation()

    def perceive(self, key: str, x: float) -> float:
        return self.model.observe(key, x)

    def free_energy(self) -> float:
        return self.model.free_energy()

    def most_informative(self, keys: Sequence[str]) -> Tuple[Optional[str], float]:
        if not keys:
            return None, 0.0
        gains = {k: self.model.expected_information_gain(k) for k in keys}
        best = max(keys, key=lambda k: gains[k])
        return best, gains[best]

    def epistemic_action(self, keys: Sequence[str], *, stakes: float = 0.4
                         ) -> Tuple[Optional[str], float, Recommendation]:
        """Pick the most informative source; VOI reads whether it is worth the effort."""
        best, gain = self.most_informative(keys)
        rec = self.voi.from_info_gain(gain, stakes=stakes)
        return best, gain, rec


class EpistemicScheduler:
    """Curiosity as uncertainty-seeking: steer practice toward the least-understood task-family."""

    def __init__(self, *, floor: float = 0.05) -> None:
        self.model = GenerativeModel()
        self.floor = floor

    def observe(self, family: str, solved: bool) -> float:
        """Record whether she solved a task in ``family``; returns her surprise at that outcome."""
        return self.model.observe(family, 1.0 if solved else 0.0)

    def weights(self, families: Sequence[str]) -> Dict[str, float]:
        """Sampling weights over families — higher where her competence is least predictable."""
        raw = {f: self.floor + self.model.expected_information_gain(f) for f in families}
        total = sum(raw.values()) or 1.0
        return {f: w / total for f, w in raw.items()}
