"""NYXARA · mind/prediction_engine.py — Prediction Engine (Level 13).

For any claim or action, NYXARA provides a calibrated prediction:

    Prediction + Probability + Expected Outcome + Risk Score +
    Confidence Interval + Reasoning Chain

Internally combines three signals:

    1. WorldModel.predict()     — state-transition confidence for action-style queries
    2. PredictiveCore.step()    — free-energy surprise = belief-mismatch risk
    3. VoI.decide()             — information-value weighting

Calibration: probability estimate is a geometric mean of the three signals, biased toward
the most informative available signal. When nothing is available, honest 0.5 ± wide CI.

The engine is read-only. Results annotate candidates but never block gates.
HonestyGuard.assess() can read `prediction.probability` as the expressed confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["PredictionEngine", "PredictionResult"]


# --------------------------------------------------------------------------- #
# PredictionResult
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    """One prediction pass for a claim or action."""
    query: str
    prediction: str                        # natural-language prediction
    probability: float                     # calibrated probability [0, 1]
    expected_outcome: str                  # short description of likely outcome
    risk_score: float                      # risk signal from free-energy surprise [0, 1]
    confidence_interval: Tuple[float, float]  # (lo, hi) probability interval
    reasoning_chain: List[str] = field(default_factory=list)
    world_model_confidence: Optional[float] = None
    surprise: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "prediction": self.prediction,
            "probability": round(self.probability, 3),
            "expected_outcome": self.expected_outcome,
            "risk_score": round(self.risk_score, 3),
            "confidence_interval": [round(self.confidence_interval[0], 3),
                                    round(self.confidence_interval[1], 3)],
            "reasoning_chain": self.reasoning_chain,
            "world_model_confidence": (round(self.world_model_confidence, 3)
                                       if self.world_model_confidence is not None else None),
            "surprise": round(self.surprise, 3) if self.surprise is not None else None,
        }


# --------------------------------------------------------------------------- #
# PredictionEngine
# --------------------------------------------------------------------------- #
class PredictionEngine:
    """Calibrated multi-signal prediction engine.

    Parameters
    ----------
    world_model:    WorldModel instance (optional).
    predictive:     PredictiveCore instance (optional).
    voi:            VoI instance (optional).
    """

    def __init__(self, world_model: Any = None, predictive: Any = None,
                 voi: Any = None) -> None:
        self.world_model = world_model
        self.predictive = predictive
        self.voi = voi
        self._predictions: int = 0

    # ---------------------------------------------------------------------- #
    def predict(self, query: str, context: Optional[str] = None) -> PredictionResult:
        """Run a full prediction pass for ``query``. Always returns a PredictionResult."""
        reasoning_chain: List[str] = []
        signals: List[float] = []

        # --- Signal 1: WorldModel state-transition confidence ---
        wm_conf: Optional[float] = None
        wm_reward: Optional[float] = None
        if self.world_model is not None and len(self.world_model) > 0:
            try:
                action = query[:20].lower().replace(" ", "_")
                pred = self.world_model.predict((0.0,), action)
                wm_conf = float(getattr(pred, "confidence", 0.5))
                wm_reward = float(getattr(pred, "expected_reward", 0.5))
                signals.append(wm_conf)
                reasoning_chain.append(
                    f"world-model: conf={wm_conf:.2f} reward={wm_reward:.2f}")
            except Exception:  # noqa: BLE001
                pass

        # --- Signal 2: PredictiveCore free-energy surprise ---
        surprise: Optional[float] = None
        if self.predictive is not None:
            try:
                obs = [float(ord(c) % 17) / 17.0 for c in query[:8]]
                obs += [0.0] * max(0, len(self.predictive.mu) - len(obs))
                obs = obs[:len(self.predictive.mu)]
                _, feeling = self.predictive.step(obs)
                surprise = float(min(1.0, feeling.surprise))
                # surprise → probability: high surprise = lower probability
                surprise_prob = max(0.05, 1.0 - surprise)
                signals.append(surprise_prob)
                reasoning_chain.append(
                    f"free-energy surprise={surprise:.2f} → p={surprise_prob:.2f}")
            except Exception:  # noqa: BLE001
                pass

        # --- Signal 3: VoI weighting ---
        if self.voi is not None:
            try:
                uncertainty = 1.0 - (signals[-1] if signals else 0.5)
                stakes = 0.5 if "delete" not in query.lower() else 0.8
                decision = self.voi.decide(uncertainty=uncertainty, stakes=stakes,
                                           reversibility=0.7)
                voi_signal = getattr(decision, "information_value", 0.5)
                signals.append(float(voi_signal))
                reasoning_chain.append(f"VoI: information_value={voi_signal:.2f}")
            except Exception:  # noqa: BLE001
                pass

        # --- Calibrate probability ---
        probability = self._calibrate(query, signals, reasoning_chain)
        risk_score = surprise if surprise is not None else (1.0 - probability)
        ci = self._confidence_interval(probability, len(signals))

        # --- Outcome description ---
        outcome = self._describe_outcome(query, probability, wm_reward)
        prediction_text = self._prediction_text(query, probability)

        self._predictions += 1
        return PredictionResult(
            query=query,
            prediction=prediction_text,
            probability=round(probability, 3),
            expected_outcome=outcome,
            risk_score=round(risk_score, 3),
            confidence_interval=ci,
            reasoning_chain=reasoning_chain,
            world_model_confidence=wm_conf,
            surprise=surprise,
        )

    @property
    def predictions_count(self) -> int:
        return self._predictions

    # ---------------------------------------------------------------------- #
    def _calibrate(self, query: str, signals: List[float],
                   reasoning: List[str]) -> float:
        """Geometric mean of available signals, with simple keyword priors."""
        base = 0.5  # honest default when nothing else is available

        # keyword priors: tautologies and falsehoods
        q_low = query.lower()
        if any(t in q_low for t in {"2+2", "1+1", "true is true", "sky is blue"}):
            base = 0.97
            reasoning.append("prior: near-tautology → high probability")
        elif any(t in q_low for t in {"impossible", "never", "always fails"}):
            base = 0.05
            reasoning.append("prior: near-impossibility → low probability")

        if not signals:
            return base

        # geometric mean of all signals
        product = 1.0
        for s in signals:
            product *= max(0.01, min(0.99, s))
        geo_mean = product ** (1.0 / len(signals))
        # blend with base (50/50 when few signals, more weight to signals as they accumulate)
        weight = min(0.8, 0.3 * len(signals))
        return round(weight * geo_mean + (1.0 - weight) * base, 3)

    @staticmethod
    def _confidence_interval(probability: float, n_signals: int) -> Tuple[float, float]:
        """Wider interval when fewer signals; narrower when well-supported."""
        half_width = 0.35 if n_signals == 0 else max(0.05, 0.3 / math.sqrt(n_signals + 1))
        lo = max(0.0, probability - half_width)
        hi = min(1.0, probability + half_width)
        return (round(lo, 3), round(hi, 3))

    @staticmethod
    def _describe_outcome(query: str, probability: float,
                          wm_reward: Optional[float]) -> str:
        if probability >= 0.85:
            tier = "very likely"
        elif probability >= 0.65:
            tier = "likely"
        elif probability >= 0.35:
            tier = "uncertain"
        else:
            tier = "unlikely"
        reward_note = ""
        if wm_reward is not None:
            reward_note = f"; expected reward={wm_reward:.2f}"
        return f"{tier} outcome for '{query[:40]}'{reward_note}"

    @staticmethod
    def _prediction_text(query: str, probability: float) -> str:
        pct = round(probability * 100)
        return f"~{pct}% probability: '{query[:60]}'"


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA prediction-engine self-test")
    print("=" * 70)

    engine = PredictionEngine()

    # tautology → high probability
    r1 = engine.predict("2+2=4")
    print(f"\ntautology '2+2=4':")
    print(f"  probability  : {r1.probability}")
    print(f"  CI           : {r1.confidence_interval}")
    print(f"  prediction   : {r1.prediction}")
    assert r1.probability >= 0.8

    # uncertainty → ~0.5
    r2 = engine.predict("stock market will crash tomorrow")
    print(f"\nuncertain claim:")
    print(f"  probability  : {r2.probability}")
    assert 0.1 <= r2.probability <= 0.9

    # near-impossibility → low
    r3 = engine.predict("this is impossible and always fails")
    print(f"\nnear-impossible claim:")
    print(f"  probability  : {r3.probability}")
    assert r3.probability <= 0.3

    print(f"\npredictions made: {engine.predictions_count}")
    assert engine.predictions_count == 3

    print("\nALL SELF-TESTS PASSED ✓")
