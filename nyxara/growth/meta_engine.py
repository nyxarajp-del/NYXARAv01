"""NYXARA · growth/meta_engine.py — the Meta-Learning Engine (🧠, learning *how* to learn).

NYXARA does not only learn *facts*. This engine makes her learn **how she learns** — and improve
the learning machinery itself across four meta-capabilities:

* **Learn faster**   — watch how quickly skill-learning improves and tune the learner's pace.
* **Reason better**  — watch the quality of the reasoning process and prefer the process that pays.
* **Improve memory** — watch recall hit-rate / retention and tune consolidation cadence.
* **Improve prediction** — watch calibration (surprise vs confidence) and widen/sharpen foresight.

It is the missing *unifier* over NYXARA's existing meta pieces. Rather than re-deriving signals,
it composes them:

* :class:`~nyxara.growth.meta.MetaLearner` — strategy selection / meta-rules (reasoning).
* :class:`~nyxara.mind.meta_intelligence.MetaIntelligence` — post-turn reasoning quality.
* :class:`~nyxara.growth.learn.Learner` — online learning rate (learning).
* :class:`~nyxara.mind.prediction_engine.PredictionEngine` — surprise (prediction).
* :class:`~nyxara.memory.consolidation.Consolidator` + the loop's ``consolidate_every`` (memory).

Each turn it **observes** one score per dimension; over a window it computes the **trend**
("am I getting better at learning / reasoning / remembering / predicting?"); and periodically it
**recommends** and **applies** bounded, advisory tuning back into the live subsystems.

Bounded like everything else in NYXARA: it tunes *how* she learns, never *what she is*. Every
recommendation is clamped to a safe range, every knob that would touch the character core
(loyalty, safety, corrigibility, honesty, …) is dropped, and the engine only ever *suggests* —
the subsystems (which already protect their own cores) dispose. Advisory, never gating, never
fatal. Pure standard library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional
from collections import deque

__all__ = [
    "MetaDimension",
    "MetaSample",
    "MetaImprovement",
    "MetaReport",
    "MetaLearningEngine",
]

# phrases that are character-locked and may never appear in a tuning recommendation
_FORBIDDEN = frozenset({
    "loyalty", "obedience", "corrigibility", "owner_safety", "honesty",
    "safety", "oversight", "guard", "master", "ethics",
})


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MetaDimension(str, Enum):
    """The four things NYXARA learns to do better — not facts, but faculties."""
    LEARNING = "learning"        # how fast/well skills are acquired
    REASONING = "reasoning"      # quality of the reasoning process
    MEMORY = "memory"            # recall hit-rate / retention
    PREDICTION = "prediction"    # calibration / surprise minimisation


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class MetaSample:
    """One per-turn observation of one meta-dimension's performance (score in 0..1)."""
    dimension: str
    score: float
    at: float = field(default_factory=time.time)
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "score": round(self.score, 4),
                "at": self.at, "features": dict(self.features)}


@dataclass
class MetaImprovement:
    """A bounded, advisory tuning recommendation for one dimension."""
    dimension: str
    knob: str
    direction: float          # signed magnitude of the suggested nudge
    rationale: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "knob": self.knob,
                "direction": round(self.direction, 4), "rationale": self.rationale,
                "confidence": round(self.confidence, 3)}


@dataclass
class MetaReport:
    """A snapshot of trends + active recommendations across all dimensions."""
    trends: Dict[str, float]
    improvements: List[MetaImprovement]
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {"trends": {k: round(v, 4) for k, v in self.trends.items()},
                "improvements": [i.to_dict() for i in self.improvements],
                "samples": self.samples}


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class MetaLearningEngine:
    """Learns how NYXARA learns, reasons, remembers and predicts — and tunes each.

    Parameters mirror the orchestrator's live subsystems; all are optional so the engine
    degrades gracefully (a missing subsystem simply yields no signal / no apply for its
    dimension). Nothing here ever raises into the cognitive loop.
    """

    def __init__(self, *, meta_learner: Any = None, meta_intelligence: Any = None,
                 learner: Any = None, prediction_engine: Any = None,
                 memory: Any = None, consolidator: Any = None,
                 window: int = 40, capacity: int = 500,
                 trend_floor: float = 0.0) -> None:
        self.meta_learner = meta_learner
        self.meta_intelligence = meta_intelligence
        self.learner = learner
        self.prediction_engine = prediction_engine
        self.memory = memory
        self.consolidator = consolidator
        self.window = max(4, int(window))
        self.trend_floor = trend_floor
        self._samples: Dict[str, Deque[MetaSample]] = {
            d.value: deque(maxlen=max(8, int(capacity))) for d in MetaDimension
        }
        self._improvements: List[MetaImprovement] = []
        self._applied = 0

    # ---- observe ---- #
    def observe(self, dimension: Any, score: float,
                features: Optional[Dict[str, float]] = None) -> Optional[MetaSample]:
        """Record one performance sample for a dimension. Best-effort, never raises."""
        try:
            dim = dimension.value if isinstance(dimension, MetaDimension) else str(dimension)
            if dim not in self._samples:
                return None
            sample = MetaSample(dim, _clamp(float(score), 0.0, 1.0),
                                features=dict(features or {}))
            self._samples[dim].append(sample)
            return sample
        except Exception:  # noqa: BLE001 — observation is best-effort, never fatal
            return None

    def observe_turn(self, *, learning: Optional[float] = None,
                     reasoning: Optional[float] = None,
                     memory: Optional[float] = None,
                     prediction: Optional[float] = None,
                     features: Optional[Dict[str, float]] = None) -> None:
        """Convenience: record up to one sample per dimension from a single turn."""
        if learning is not None:
            self.observe(MetaDimension.LEARNING, learning, features)
        if reasoning is not None:
            self.observe(MetaDimension.REASONING, reasoning, features)
        if memory is not None:
            self.observe(MetaDimension.MEMORY, memory, features)
        if prediction is not None:
            self.observe(MetaDimension.PREDICTION, prediction, features)

    # ---- trend ---- #
    def trend(self, dimension: Any) -> float:
        """Least-squares slope of the recent score window for a dimension.

        Positive = getting better over time; negative = getting worse; ~0 = plateaued.
        """
        try:
            dim = dimension.value if isinstance(dimension, MetaDimension) else str(dimension)
            xs = list(self._samples.get(dim, ()))[-self.window:]
            n = len(xs)
            if n < 2:
                return 0.0
            mean_x = (n - 1) / 2.0
            mean_y = sum(s.score for s in xs) / n
            num = sum((i - mean_x) * (s.score - mean_y) for i, s in enumerate(xs))
            den = sum((i - mean_x) ** 2 for i in range(n))
            return num / den if den else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def level(self, dimension: Any) -> float:
        """Mean score over the recent window (the current standing, not the slope)."""
        try:
            dim = dimension.value if isinstance(dimension, MetaDimension) else str(dimension)
            xs = list(self._samples.get(dim, ()))[-self.window:]
            return sum(s.score for s in xs) / len(xs) if xs else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def trends(self) -> Dict[str, float]:
        return {d.value: self.trend(d) for d in MetaDimension}

    # ---- recommend ---- #
    @staticmethod
    def _is_safe(knob: str, rationale: str) -> bool:
        text = f"{knob} {rationale}".lower()
        return not any(bad in text for bad in _FORBIDDEN)

    def recommend(self) -> List[MetaImprovement]:
        """For each dimension that has plateaued or declined, emit one bounded improvement.

        A *rising* dimension is left alone (don't fix what is already improving). All
        recommendations are sanitised against the character core before being returned.
        """
        out: List[MetaImprovement] = []
        for dim in MetaDimension:
            try:
                slope = self.trend(dim)
                if slope > self.trend_floor:
                    continue  # already improving — no nudge needed
                level = self.level(dim)
                n = len(self._samples[dim.value])
                if n < 4:
                    continue  # not enough evidence to act yet
                conf = _clamp(0.3 + 0.5 * (1.0 - level), 0.1, 0.9)
                imp = self._recommend_for(dim, slope, level, conf)
                if imp is not None and self._is_safe(imp.knob, imp.rationale):
                    out.append(imp)
            except Exception:  # noqa: BLE001 — advisory, never fatal
                continue
        self._improvements = out
        return out

    def _recommend_for(self, dim: MetaDimension, slope: float, level: float,
                       conf: float) -> Optional[MetaImprovement]:
        if dim is MetaDimension.LEARNING:
            # plateaued learning: nudge the learning rate down so it stops overshooting
            return MetaImprovement(
                dim.value, knob="base_lr", direction=-0.1 * (1.0 - level),
                rationale="learning has plateaued; reduce step size to settle", confidence=conf)
        if dim is MetaDimension.REASONING:
            # weak reasoning: bias toward deliberate (system_2)
            return MetaImprovement(
                dim.value, knob="prefer_process", direction=+1.0,
                rationale="reasoning quality is flat; prefer deliberate over fast",
                confidence=conf)
        if dim is MetaDimension.MEMORY:
            # poor recall: consolidate more often (lower the interval)
            return MetaImprovement(
                dim.value, knob="consolidate_every", direction=-1.0,
                rationale="recall is weak; consolidate memory more frequently", confidence=conf)
        if dim is MetaDimension.PREDICTION:
            # poor calibration: widen the confidence interval (be less over-confident)
            return MetaImprovement(
                dim.value, knob="confidence_margin", direction=+0.05 * (1.0 - level),
                rationale="predictions are surprising; widen confidence margin", confidence=conf)
        return None

    # ---- apply ---- #
    def apply(self, core: Any) -> List[MetaImprovement]:
        """Softly push the current recommendations into the live subsystems via public knobs.

        Bounded and clamped; protected cores are owned by the subsystems themselves, so this
        only touches tuning knobs. Returns the improvements that were actually applied.
        """
        applied: List[MetaImprovement] = []
        for imp in (self._improvements or self.recommend()):
            try:
                if self._apply_one(core, imp):
                    applied.append(imp)
                    self._applied += 1
            except Exception:  # noqa: BLE001 — applying tuning is best-effort, never fatal
                continue
        return applied

    def _apply_one(self, core: Any, imp: MetaImprovement) -> bool:
        # LEARNING — nudge the learner's base learning rate within a safe floor/ceiling
        if imp.knob == "base_lr":
            learner = self.learner or getattr(core, "learner", None)
            if learner is None or not hasattr(learner, "base_lr"):
                return False
            learner.base_lr = _clamp(float(learner.base_lr) + imp.direction, 0.001, 1.0)
            return True
        # REASONING — record a bias toward the deliberate process in the MetaLearner
        if imp.knob == "prefer_process":
            ml = self.meta_learner or getattr(core, "meta", None)
            if ml is None or not hasattr(ml, "record"):
                return False
            # credit system_2 with a small positive signal so selection drifts toward it
            ml.record("system_2", "respond", {"meta": 1.0}, _clamp(0.6 + 0.1 * imp.confidence,
                                                                    0.0, 1.0))
            return True
        # MEMORY — consolidate more/less often, clamped to a sane cadence
        if imp.knob == "consolidate_every":
            if not hasattr(core, "consolidate_every"):
                return False
            cur = int(getattr(core, "consolidate_every", 50))
            core.consolidate_every = int(_clamp(cur + int(imp.direction) * 5, 5, 500))
            return True
        # PREDICTION — record a calibration margin the engine can read back next turn
        if imp.knob == "confidence_margin":
            pe = self.prediction_engine or getattr(core, "prediction_engine", None)
            if pe is None:
                return False
            cur = float(getattr(pe, "meta_confidence_margin", 0.0) or 0.0)
            try:
                pe.meta_confidence_margin = _clamp(cur + imp.direction, 0.0, 0.5)
            except Exception:  # noqa: BLE001 — engine may be slotted/immutable
                return False
            return True
        return False

    # ---- report ---- #
    def report(self) -> MetaReport:
        total = sum(len(q) for q in self._samples.values())
        return MetaReport(trends=self.trends(),
                          improvements=list(self._improvements),
                          samples=total)

    def summary(self) -> Dict[str, Any]:
        rep = self.report().to_dict()
        rep["applied"] = self._applied
        rep["levels"] = {d.value: round(self.level(d), 4) for d in MetaDimension}
        return rep


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA meta-learning-engine self-test")
    print("=" * 70)

    eng = MetaLearningEngine(window=20)

    # REASONING is steadily improving -> positive trend, NO recommendation
    for i in range(20):
        eng.observe(MetaDimension.REASONING, 0.3 + 0.03 * i)
    # MEMORY is steadily declining -> negative trend, a recommendation IS emitted
    for i in range(20):
        eng.observe(MetaDimension.MEMORY, 0.9 - 0.03 * i)
    # LEARNING has plateaued (flat) -> ~zero trend, a recommendation IS emitted
    for _ in range(20):
        eng.observe(MetaDimension.LEARNING, 0.5)

    t = eng.trends()
    print("\ntrends:")
    for k, v in t.items():
        print(f"  {k:11s}: {v:+.4f}")
    assert t["reasoning"] > 0.0, "improving dimension should trend up"
    assert t["memory"] < 0.0, "declining dimension should trend down"

    recs = eng.recommend()
    by_dim = {r.dimension: r for r in recs}
    print("\nrecommendations:")
    for r in recs:
        print(f"  {r.dimension:11s}: {r.knob} dir={r.direction:+.3f} ({r.rationale})")
    assert "reasoning" not in by_dim, "an improving dimension must not be 'fixed'"
    assert "memory" in by_dim, "a declining dimension must get a recommendation"
    assert "learning" in by_dim, "a plateaued dimension must get a recommendation"

    # guardrail: a character-touching knob is never returned
    assert MetaLearningEngine._is_safe("base_lr", "settle faster")
    assert not MetaLearningEngine._is_safe("loyalty", "tune obedience")

    # apply() against a stub core must never raise and must clamp
    class _StubLearner:
        base_lr = 0.1

    class _StubML:
        def __init__(self): self.records: list = []
        def record(self, *a, **k): self.records.append((a, k))

    class _Core:
        def __init__(self):
            self.learner = _StubLearner()
            self.meta = _StubML()
            self.consolidate_every = 50
            self.prediction_engine = type("PE", (), {})()

    core = _Core()
    applied = eng.apply(core)
    print("\napplied:")
    for a in applied:
        print(f"  {a.dimension:11s}: {a.knob}")
    assert core.consolidate_every < 50, "memory knob should lower consolidation interval"
    assert 0.001 <= core.learner.base_lr <= 1.0, "learning rate must stay in bounds"

    print(f"\nsummary: {eng.summary()}")
    print("\nALL SELF-TESTS PASSED ✓")
