"""NYXARA · senses/predictive.py — predictive perception & surprise (👁, active inference).

Perception is not passive reception — it is *prediction checked against evidence*. NYXARA
keeps a running forecast of what she expects to sense next, and what actually arrives is
only interesting to the degree it **violates that forecast**. This module is that
predictive layer at the sensory edge, the perceptual complement to the deep generative
:mod:`mind.predictive_core`.

Two predictors, both online and adaptive:

* :class:`SensoryPredictor` — over **continuous feature vectors**. It tracks a constant-
  velocity EWMA model (level + drift) and per-dimension variance, predicts the next
  observation, and scores **precision-weighted surprise** (large errors on confident
  dimensions surprise most). It flags **novelty** and, further out in the tail,
  **anomaly**, against an adaptive threshold learned from its own recent surprise.
* :class:`SymbolPredictor` — over **discrete event streams** (which modality, which tag,
  which kind of thing). A smoothed unigram/bigram model gives ``-log₂ p`` surprise, so an
  out-of-pattern event registers as information.

Surprise drives **attention**: the more a percept defies prediction, the more attention it
earns (and the predictors habituate as variance grows, so the expected stops shouting).
:class:`PerceptualPredictor` fuses both over a :class:`~nyxara.senses.binding.Percept`
stream and can **boost a percept's salience** by its surprise — so the unexpected, not the
merely loud, rises to the mind.

Reuses :mod:`senses.binding` percepts. Pure standard library otherwise.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from nyxara.senses.binding import Modality, Percept

__all__ = [
    "SurpriseResult",
    "SensoryPredictor",
    "SymbolSurprise",
    "SymbolPredictor",
    "PerceptSurprise",
    "PerceptualPredictor",
]

Vec = List[float]
_EPS = 1e-9


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Continuous sensory prediction
# --------------------------------------------------------------------------- #
@dataclass
class SurpriseResult:
    predicted: Vec
    observation: Vec
    error: Vec
    surprise: float              # precision-weighted, >= 0
    attention: float             # [0, 1]
    novelty: bool
    anomaly: bool
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {"surprise": round(self.surprise, 5), "attention": round(self.attention, 4),
                "novelty": self.novelty, "anomaly": self.anomaly,
                "error": [round(e, 4) for e in self.error], "n": self.n}


class SensoryPredictor:
    """Online constant-velocity predictor with precision-weighted surprise."""

    def __init__(self, *, alpha: float = 0.3, novelty_k: float = 2.0,
                 anomaly_k: float = 3.5, warmup: int = 5, history: int = 64,
                 init_var: float = 1.0) -> None:
        self.alpha = alpha
        self.novelty_k = novelty_k
        self.anomaly_k = anomaly_k
        self.warmup = warmup
        self.init_var = init_var
        self._mean: Optional[Vec] = None
        self._vel: Optional[Vec] = None
        self._var: Optional[Vec] = None
        self._n = 0
        self._surprises: Deque[float] = deque(maxlen=history)

    @property
    def n(self) -> int:
        return self._n

    def predict(self) -> Optional[Vec]:
        """The currently expected next observation (level + drift), or None if untrained."""
        if self._mean is None:
            return None
        return [m + v for m, v in zip(self._mean, self._vel)]  # type: ignore[arg-type]

    def _thresholds(self) -> Optional[Any]:
        if len(self._surprises) < self.warmup:
            return None
        mu = sum(self._surprises) / len(self._surprises)
        var = sum((s - mu) ** 2 for s in self._surprises) / len(self._surprises)
        sd = math.sqrt(var)
        return mu + self.novelty_k * sd, mu + self.anomaly_k * sd

    def observe(self, obs: Sequence[float]) -> SurpriseResult:
        """Predict, score surprise against the observation, then update the model."""
        obs = [float(x) for x in obs]
        d = len(obs)
        if self._mean is None:
            self._mean = list(obs)
            self._vel = [0.0] * d
            self._var = [self.init_var] * d
            self._n = 1
            zero = [0.0] * d
            return SurpriseResult(predicted=list(obs), observation=obs, error=zero,
                                  surprise=0.0, attention=0.0, novelty=False,
                                  anomaly=False, n=1)
        if d != len(self._mean):
            raise ValueError(f"observation dim {d} != model dim {len(self._mean)}")

        pred = self.predict() or obs
        error = [o - p for o, p in zip(obs, pred)]
        # precision-weighted surprise (mean standardized squared error)
        pw = sum((e * e) / (v + _EPS) for e, v in zip(error, self._var)) / d  # type: ignore[arg-type]
        surprise = max(0.0, pw)

        thr = self._thresholds()
        novelty = anomaly = False
        if thr is not None:
            nov_t, ano_t = thr
            novelty = surprise > nov_t
            anomaly = surprise > ano_t

        attention = _clamp(surprise / (surprise + 1.0))

        # ---- update (predictive-coding step) ---- #
        a = self.alpha
        old_mean = list(self._mean)
        self._mean = [a * o + (1 - a) * m for o, m in zip(obs, self._mean)]
        self._vel = [a * (o - om) + (1 - a) * vv
                     for o, om, vv in zip(obs, old_mean, self._vel)]  # type: ignore[arg-type]
        self._var = [a * (e * e) + (1 - a) * vr
                     for e, vr in zip(error, self._var)]  # type: ignore[arg-type]
        self._n += 1
        self._surprises.append(surprise)

        return SurpriseResult(predicted=pred, observation=obs, error=error,
                              surprise=surprise, attention=attention,
                              novelty=novelty, anomaly=anomaly, n=self._n)

    def reset(self) -> None:
        self._mean = self._vel = self._var = None
        self._n = 0
        self._surprises.clear()


# --------------------------------------------------------------------------- #
# Discrete symbol prediction
# --------------------------------------------------------------------------- #
@dataclass
class SymbolSurprise:
    symbol: str
    probability: float
    surprise: float              # -log2 p, >= 0
    attention: float
    novelty: bool
    predicted: str
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "probability": round(self.probability, 4),
                "surprise": round(self.surprise, 4), "attention": round(self.attention, 4),
                "novelty": self.novelty, "predicted": self.predicted, "n": self.n}


class SymbolPredictor:
    """Smoothed unigram/bigram model over a discrete event stream (-log2 p surprise)."""

    def __init__(self, *, smoothing: float = 0.5, bigram_weight: float = 0.6,
                 novelty_bits: float = 0.0, warmup: int = 4, history: int = 64) -> None:
        self.smoothing = smoothing
        self.bigram_weight = bigram_weight
        self.novelty_bits = novelty_bits   # extra bits over running mean to flag novelty
        self.warmup = warmup
        self._uni: Counter = Counter()
        self._bi: Dict[str, Counter] = {}
        self._prev: Optional[str] = None
        self._n = 0
        self._surprises: Deque[float] = deque(maxlen=history)

    def _vocab(self) -> int:
        return max(1, len(self._uni))

    def probability(self, symbol: str, *, prev: Optional[str] = None) -> float:
        k = self.smoothing
        V = self._vocab() + 1   # +1 for an unseen symbol
        total = sum(self._uni.values())
        p_uni = (self._uni.get(symbol, 0) + k) / (total + k * V) if total else 1.0 / V
        prev = self._prev if prev is None else prev
        if prev is not None and prev in self._bi and sum(self._bi[prev].values()) > 0:
            bt = sum(self._bi[prev].values())
            p_bi = (self._bi[prev].get(symbol, 0) + k) / (bt + k * V)
            return self.bigram_weight * p_bi + (1 - self.bigram_weight) * p_uni
        return p_uni

    def predict(self) -> str:
        """The most likely next symbol given the current context."""
        if self._prev is not None and self._prev in self._bi and self._bi[self._prev]:
            return self._bi[self._prev].most_common(1)[0][0]
        return self._uni.most_common(1)[0][0] if self._uni else ""

    def observe(self, symbol: str) -> SymbolSurprise:
        predicted = self.predict() if self._n else symbol
        p = self.probability(symbol)
        surprise = -math.log2(max(p, _EPS))

        novelty = False
        if len(self._surprises) >= self.warmup:
            mu = sum(self._surprises) / len(self._surprises)
            novelty = surprise > mu + self.novelty_bits + 1.0

        attention = _clamp(surprise / (surprise + 4.0))

        # update
        self._uni[symbol] += 1
        if self._prev is not None:
            self._bi.setdefault(self._prev, Counter())[symbol] += 1
        self._prev = symbol
        self._n += 1
        self._surprises.append(surprise)

        return SymbolSurprise(symbol=symbol, probability=p, surprise=surprise,
                              attention=attention, novelty=novelty, predicted=predicted,
                              n=self._n)

    def reset(self) -> None:
        self._uni.clear()
        self._bi.clear()
        self._prev = None
        self._n = 0
        self._surprises.clear()


# --------------------------------------------------------------------------- #
# Perceptual predictor (fuses both over a percept stream)
# --------------------------------------------------------------------------- #
_MODALITIES = list(Modality)


def percept_features(p: Percept) -> Vec:
    """A small, stable feature vector for a percept (salience/trust/structure/modality)."""
    return [
        p.salience,
        p.provenance.trust,
        min(1.0, len(p.entities) / 5.0),
        min(1.0, len(p.tags) / 5.0),
        _MODALITIES.index(p.modality) / max(1, len(_MODALITIES) - 1),
        min(1.0, p.count / 5.0),
    ]


@dataclass
class PerceptSurprise:
    percept_id: str
    feature_surprise: float
    modality_surprise: float
    surprise: float
    attention: float
    novelty: bool
    anomaly: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"percept_id": self.percept_id, "surprise": round(self.surprise, 4),
                "feature_surprise": round(self.feature_surprise, 4),
                "modality_surprise": round(self.modality_surprise, 4),
                "attention": round(self.attention, 4), "novelty": self.novelty,
                "anomaly": self.anomaly}


class PerceptualPredictor:
    """Predicts the next percept's features and modality; surprise drives attention."""

    def __init__(self, *, feature_weight: float = 0.6, **kw: Any) -> None:
        self.feature_weight = feature_weight
        self.sensory = SensoryPredictor(**{k: v for k, v in kw.items()
                                           if k in {"alpha", "novelty_k", "anomaly_k",
                                                    "warmup", "history", "init_var"}})
        self.symbols = SymbolPredictor()

    def observe(self, percept: Percept, *, boost_salience: bool = False) -> PerceptSurprise:
        sr = self.sensory.observe(percept_features(percept))
        ss = self.symbols.observe(percept.modality.value)
        fw = self.feature_weight
        # blend the continuous and symbolic surprise into one attention signal
        combined_att = _clamp(fw * sr.attention + (1 - fw) * ss.attention)
        result = PerceptSurprise(
            percept_id=percept.id, feature_surprise=sr.surprise,
            modality_surprise=ss.surprise,
            surprise=fw * sr.surprise + (1 - fw) * ss.surprise,
            attention=combined_att, novelty=sr.novelty or ss.novelty,
            anomaly=sr.anomaly)
        if boost_salience:
            percept.salience = _clamp(max(percept.salience, combined_att))
        return result

    def reset(self) -> None:
        self.sensory.reset()
        self.symbols.reset()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA predictive-perception self-test")
    print("=" * 70)

    # ---- continuous: a predictable ramp, then a shock ---- #
    sp = SensoryPredictor(warmup=5)
    last = None
    for i in range(20):
        last = sp.observe([float(i), 10.0 - i * 0.5])   # smooth linear trend
    print(f"\nramp surprise (low) : {last.surprise:.3f} attention={last.attention:.2f}")
    assert last.surprise < 1.0   # the model learned the trend

    shock = sp.observe([100.0, -50.0])   # a sudden break in the pattern
    print(f"shock surprise (hi) : {shock.surprise:.2f} attention={shock.attention:.2f} "
          f"anomaly={shock.anomaly}")
    assert shock.surprise > last.surprise * 5
    assert shock.attention > 0.8 and (shock.anomaly or shock.novelty)
    print("prediction error    : a break in the pattern spikes surprise/attention ✓")

    # ---- discrete: a repetitive stream, then a rare symbol ---- #
    sy = SymbolPredictor()
    res = None
    for s in "AAAABAAAABAAAAB":     # mostly A, occasional B
        res = sy.observe(s)
    common = sy.observe("A")
    rare = sy.observe("Z")          # never seen before
    print(f"\nsymbol 'A' surprise : {common.surprise:.2f} bits")
    print(f"symbol 'Z' surprise : {rare.surprise:.2f} bits (novel={rare.novelty})")
    assert rare.surprise > common.surprise
    assert rare.novelty
    print(f"prediction          : after 'A', model expects {sy.predict()!r}")

    # ---- perceptual: an off-pattern percept earns attention ---- #
    from nyxara.senses.binding import Provenance

    pred = PerceptualPredictor(warmup=4)
    # a steady stream of low-salience trusted text percepts
    for i in range(10):
        p = Percept(Modality.TEXT, f"routine {i}", Provenance("owner", 0.9),
                    fingerprint=i + 1, salience=0.3)
        pred.observe(p)
    # then a sudden untrusted image with threat tags
    odd = Percept(Modality.IMAGE, "?!", Provenance("https://evil.com", 0.05),
                  fingerprint=999, salience=0.3, tags=["threat", "image"], entities=["X", "Y"])
    out = pred.observe(odd, boost_salience=True)
    print(f"\nodd percept surprise: {out.surprise:.3f} attention={out.attention:.2f} "
          f"novelty={out.novelty}")
    print(f"salience boosted    : {odd.salience:.2f} (was 0.30)")
    assert out.attention > 0.3
    assert odd.salience >= out.attention   # surprise lifted it above the routine
    print("attention gating    : the unexpected, not the loud, rose to the mind ✓")

    print("\nALL SELF-TESTS PASSED ✓")
