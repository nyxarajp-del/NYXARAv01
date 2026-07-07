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

__all__ = ["PredictionEngine", "PredictionResult", "LearnedPrior"]


# --------------------------------------------------------------------------- #
# LearnedPrior — a base rate NYXARA learns from real outcomes (no hardcoded strings)
# --------------------------------------------------------------------------- #
class LearnedPrior:
    """Beta-Bernoulli base-rate model over coarse query features.

    Replaces hand-written keyword priors. For each feature bucket of a query (arithmetic,
    negation, certainty, risky, question, length, and an always-on ``general`` bucket) NYXARA
    keeps a Beta(α, β) posterior over "a prediction of this kind turns out correct." The base
    rate for a query is the evidence-weighted mean of its observed buckets, so well-tried
    features dominate; a query with no observed feature yet gets an honest 0.5. Every real
    outcome (``observe``) sharpens it — the engine gets calibrated by living, not by a lookup
    table. Pure standard library; serialisable so it survives a restart.
    """

    _PRIOR_A = 1.0
    _PRIOR_B = 1.0
    _WH = {"who", "what", "when", "where", "why", "how", "which", "will",
           "is", "are", "does", "do", "can", "should", "would"}

    def __init__(self) -> None:
        self._arms: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def features(query: str) -> List[str]:
        q = (query or "").lower().strip()
        tags = ["general"]
        if any(ch.isdigit() for ch in q) and any(
                op in q for op in ("+", "-", "*", "/", "=", "sum", "plus", "minus", "times", "product")):
            tags.append("arithmetic")
        if any(w in q for w in ("not", "no ", "never", "impossible", "cannot",
                                "can't", "won't", "false", "n't")):
            tags.append("negation")
        if any(w in q for w in ("always", "definitely", "certainly", "guaranteed",
                                "must ", "true")):
            tags.append("certainty")
        if any(w in q for w in ("delete", "destroy", "irreversible", "overwrite",
                                "wipe", "remove", "drop ")):
            tags.append("risky")
        first = q.split()[0] if q.split() else ""
        if q.endswith("?") or first in LearnedPrior._WH:
            tags.append("question")
        tags.append("long" if len(q) > 80 else "short")
        return tags

    def _mean(self, tag: str) -> Optional[Tuple[float, int]]:
        arm = self._arms.get(tag)
        if not arm or int(arm.get("n", 0)) <= 0:
            return None
        a = float(arm.get("alpha", self._PRIOR_A))
        b = float(arm.get("beta", self._PRIOR_B))
        return (a / (a + b), int(arm.get("n", 0)))

    def base_rate(self, query: str) -> Tuple[float, Optional[str]]:
        """Return (base_rate, explanation) — an honest 0.5 with no explanation until learned."""
        observed = [(t, self._mean(t)) for t in self.features(query)]
        observed = [(t, mn) for t, mn in observed if mn is not None]
        if not observed:
            return 0.5, None
        acc = 0.0
        total_w = 0.0
        parts: List[str] = []
        for t, (m, n) in observed:
            w = float(n)
            acc += w * m
            total_w += w
            parts.append(f"{t}={m:.2f}(n={n})")
        base = acc / total_w if total_w else 0.5
        return max(0.02, min(0.98, base)), "learned base rate: " + ", ".join(parts)

    def observe(self, query: str, correct: bool, weight: float = 1.0) -> None:
        w = max(0.0, min(1.0, float(weight)))
        if w <= 0.0:
            return
        for t in self.features(query):
            arm = self._arms.setdefault(
                t, {"alpha": self._PRIOR_A, "beta": self._PRIOR_B, "n": 0})
            if correct:
                arm["alpha"] = float(arm.get("alpha", 1.0)) + w
            else:
                arm["beta"] = float(arm.get("beta", 1.0)) + w
            arm["n"] = int(arm.get("n", 0)) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": {k: {"alpha": round(float(v.get("alpha", 1.0)), 6),
                             "beta": round(float(v.get("beta", 1.0)), 6),
                             "n": int(v.get("n", 0))}
                         for k, v in self._arms.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        for k, v in (d.get("arms", {}) or {}).items():
            if isinstance(v, dict):
                self._arms[k] = {"alpha": float(v.get("alpha", 1.0) or 1.0),
                                 "beta": float(v.get("beta", 1.0) or 1.0),
                                 "n": int(v.get("n", 0) or 0)}


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
                 voi: Any = None, prior: Optional["LearnedPrior"] = None) -> None:
        self.world_model = world_model
        self.predictive = predictive
        self.voi = voi
        self.prior = prior if prior is not None else LearnedPrior()
        self._predictions: int = 0

    def observe_outcome(self, query: str, correct: bool, weight: float = 1.0) -> None:
        """Feed a real ground-truth outcome back so the base rate learns from experience.

        Called by the kernel when a prediction/claim is later confirmed or falsified; the same
        ground-truth signal the honesty calibrator consumes. Best-effort and never raises."""
        try:
            self.prior.observe(query, bool(correct), weight=weight)
        except Exception:  # noqa: BLE001
            pass

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
        """Geometric mean of available signals, anchored on a *learned* base rate.

        The base rate comes from :class:`LearnedPrior` — a Beta-Bernoulli model over query
        features that NYXARA updates from real outcomes — not from hardcoded keyword strings.
        Until a feature has been observed the base is an honest 0.5."""
        base, why = self.prior.base_rate(query)
        reasoning.append(why if why else "no learned prior yet → honest 0.5 base")

        if not signals:
            return round(base, 3)

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
    print("\ntautology '2+2=4':")
    print(f"  probability  : {r1.probability}")
    print(f"  CI           : {r1.confidence_interval}")
    print(f"  prediction   : {r1.prediction}")
    assert r1.probability >= 0.8

    # uncertainty → ~0.5
    r2 = engine.predict("stock market will crash tomorrow")
    print("\nuncertain claim:")
    print(f"  probability  : {r2.probability}")
    assert 0.1 <= r2.probability <= 0.9

    # near-impossibility → low
    r3 = engine.predict("this is impossible and always fails")
    print("\nnear-impossible claim:")
    print(f"  probability  : {r3.probability}")
    assert r3.probability <= 0.3

    print(f"\npredictions made: {engine.predictions_count}")
    assert engine.predictions_count == 3

    print("\nALL SELF-TESTS PASSED ✓")
