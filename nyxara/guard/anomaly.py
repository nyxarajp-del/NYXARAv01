"""NYXARA · guard/anomaly.py — behavioural & statistical anomaly detection (📈, watchtower).

What the shield catches in *content* and auth catches in *identity*, this module catches in
*behaviour*: the slow, quiet signatures of something wrong — a metric drifting out of its
normal range, a sudden spike in request rate, a never-before-seen event, an action sequence
that simply does not happen in healthy operation. It learns what "normal" looks like online,
with no labels, and flags departures from it.

Four complementary detectors, each adaptive and clock-injectable:

* :class:`ScalarBaseline` — an EWMA mean/variance per metric; a value beyond ``k`` standard
  deviations is a **statistical outlier** (predict-then-update, so the baseline tracks slow,
  legitimate drift without crying wolf).
* :class:`RateMonitor` — a sliding-window event rate against an EWMA baseline; a burst far
  above normal is a **rate spike** (the tell of a flood, a loop, or a scan).
* :class:`SequenceDetector` — reuses :class:`~nyxara.senses.predictive.SymbolPredictor`'s
  ``-log₂ p`` surprise so an **out-of-pattern event order** registers as information.
* :class:`NoveltyDetector` — flags the **first appearance** of a categorical value once the
  baseline is established.

Each finding is an :class:`Anomaly` with a 0–1 severity, and :class:`AnomalyDetector` can
hand the lot to :mod:`guard.guardian` as :class:`ThreatEvent` s — the watchtower feeding the
commander. Defensive sensing only; it never acts on its own.

Reuses :mod:`senses.predictive` and bridges to :mod:`guard.guardian`. Pure standard library.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from nyxara.guard.guardian import ThreatCategory, ThreatEvent
from nyxara.senses.predictive import SymbolPredictor

__all__ = [
    "AnomalyType",
    "Anomaly",
    "ScalarBaseline",
    "RateMonitor",
    "SequenceDetector",
    "NoveltyDetector",
    "AnomalyDetector",
]

_EPS = 1e-9


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Anomaly
# --------------------------------------------------------------------------- #
class AnomalyType(str, Enum):
    OUTLIER = "outlier"
    RATE_SPIKE = "rate_spike"
    SEQUENCE = "sequence"
    NOVELTY = "novelty"


_TYPE_CATEGORY: Dict[AnomalyType, ThreatCategory] = {
    AnomalyType.OUTLIER: ThreatCategory.RESOURCE_ABUSE,
    AnomalyType.RATE_SPIKE: ThreatCategory.RESOURCE_ABUSE,
    AnomalyType.SEQUENCE: ThreatCategory.INTRUSION,
    AnomalyType.NOVELTY: ThreatCategory.UNKNOWN,
}


@dataclass
class Anomaly:
    type: AnomalyType
    signal: str
    score: float                 # [0, 1] severity
    detail: str = ""
    value: Optional[float] = None
    expected: Optional[float] = None
    at: float = field(default_factory=time.time)

    def to_threat_event(self) -> ThreatEvent:
        return ThreatEvent(category=_TYPE_CATEGORY.get(self.type, ThreatCategory.UNKNOWN),
                           source=self.signal, description=f"{self.type.value}: {self.detail}",
                           severity=self.score, confidence=0.7,
                           data={"value": self.value, "expected": self.expected})

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type.value, "signal": self.signal, "score": round(self.score, 4),
                "detail": self.detail, "value": self.value, "expected": self.expected}


# --------------------------------------------------------------------------- #
# Scalar baseline (statistical outliers)
# --------------------------------------------------------------------------- #
class ScalarBaseline:
    """EWMA mean/variance for one metric; flags values beyond ``k`` std (predict-then-update)."""

    def __init__(self, name: str, *, alpha: float = 0.2, k: float = 3.0,
                 warmup: int = 8) -> None:
        self.name = name
        self.alpha = alpha
        self.k = k
        self.warmup = warmup
        self.mean = 0.0
        self.var = 1.0
        self.n = 0

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def zscore(self, x: float) -> float:
        return (x - self.mean) / (self.std + _EPS)

    def observe(self, x: float) -> Optional[Anomaly]:
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.var = 1.0
            return None
        z = self.zscore(x)
        anomaly: Optional[Anomaly] = None
        if self.n > self.warmup and abs(z) > self.k:
            score = _clamp((abs(z) - self.k) / (2 * self.k) + 0.5)
            anomaly = Anomaly(AnomalyType.OUTLIER, self.name, score,
                              detail=f"z={z:.2f} (|z|>{self.k})", value=x,
                              expected=round(self.mean, 4))
        # update after scoring
        err = x - self.mean
        self.mean += self.alpha * err
        self.var = (1 - self.alpha) * (self.var + self.alpha * err * err)
        return anomaly


# --------------------------------------------------------------------------- #
# Rate monitor (rate spikes)
# --------------------------------------------------------------------------- #
class RateMonitor:
    """Sliding-window event rate against an EWMA baseline; flags bursts."""

    def __init__(self, name: str, *, window: float = 10.0, spike_ratio: float = 3.0,
                 min_events: int = 5, alpha: float = 0.05, warmup: int = 3, clock=None) -> None:
        self.name = name
        self.window = window
        self.spike_ratio = spike_ratio
        self.min_events = min_events
        self.alpha = alpha
        self.warmup = warmup
        self._clock = clock or time.time
        self._events: Deque[float] = deque()
        self.baseline = 0.0
        self.samples = 0

    def rate(self, now: Optional[float] = None) -> int:
        now = self._clock() if now is None else now
        while self._events and self._events[0] < now - self.window:
            self._events.popleft()
        return len(self._events)

    def observe(self, now: Optional[float] = None) -> Optional[Anomaly]:
        now = self._clock() if now is None else now
        self._events.append(now)
        rate = self.rate(now)
        anomaly: Optional[Anomaly] = None
        if (self.samples >= self.warmup and rate >= self.min_events
                and self.baseline > 0 and rate > self.baseline * self.spike_ratio):
            score = _clamp(0.5 + (rate / (self.baseline * self.spike_ratio) - 1.0))
            anomaly = Anomaly(AnomalyType.RATE_SPIKE, self.name, score,
                              detail=f"rate={rate} vs baseline≈{self.baseline:.1f}",
                              value=float(rate), expected=round(self.baseline, 2))
        # learn the baseline from normal traffic only — never from a spike (no self-masking)
        if self.samples == 0:
            self.baseline = float(rate)
        elif anomaly is None:
            self.baseline = self.alpha * rate + (1 - self.alpha) * self.baseline
        self.samples += 1
        return anomaly


# --------------------------------------------------------------------------- #
# Sequence detector (out-of-pattern order) — reuses SymbolPredictor
# --------------------------------------------------------------------------- #
class SequenceDetector:
    """Wraps a symbol predictor: an unexpected next event is a sequence anomaly."""

    def __init__(self, name: str, **kw: Any) -> None:
        self.name = name
        self._sp = SymbolPredictor(**kw)

    def observe(self, symbol: str) -> Optional[Anomaly]:
        r = self._sp.observe(symbol)
        if r.novelty:
            return Anomaly(AnomalyType.SEQUENCE, self.name,
                           score=_clamp(r.surprise / 8.0),
                           detail=f"{symbol!r} unexpected after {self._sp.predict()!r} "
                                  f"({r.surprise:.1f} bits)")
        return None


# --------------------------------------------------------------------------- #
# Novelty detector (first appearance)
# --------------------------------------------------------------------------- #
class NoveltyDetector:
    """Flags the first appearance of a categorical value (after warmup)."""

    def __init__(self, name: str, *, warmup: int = 3) -> None:
        self.name = name
        self.warmup = warmup
        self._seen: set = set()
        self.n = 0

    def observe(self, value: str) -> Optional[Anomaly]:
        self.n += 1
        novel = self.n > self.warmup and value not in self._seen
        self._seen.add(value)
        if novel:
            return Anomaly(AnomalyType.NOVELTY, self.name, score=0.5,
                           detail=f"first sighting of {value!r}")
        return None


# --------------------------------------------------------------------------- #
# Detector facade
# --------------------------------------------------------------------------- #
class AnomalyDetector:
    """Online, label-free anomaly detection over metrics, rates, sequences and categories."""

    def __init__(self, *, clock=None, history: int = 256) -> None:
        self._clock = clock or time.time
        self._metrics: Dict[str, ScalarBaseline] = {}
        self._rates: Dict[str, RateMonitor] = {}
        self._sequences: Dict[str, SequenceDetector] = {}
        self._novelty: Dict[str, NoveltyDetector] = {}
        self._log: Deque[Anomaly] = deque(maxlen=history)

    # ---- registration (lazy) ---- #
    def metric_baseline(self, name: str, **kw: Any) -> ScalarBaseline:
        return self._metrics.setdefault(name, ScalarBaseline(name, **kw))

    def rate_monitor(self, name: str, **kw: Any) -> RateMonitor:
        if name not in self._rates:
            self._rates[name] = RateMonitor(name, clock=self._clock, **kw)
        return self._rates[name]

    def sequence_detector(self, name: str, **kw: Any) -> SequenceDetector:
        return self._sequences.setdefault(name, SequenceDetector(name, **kw))

    def novelty_detector(self, name: str, **kw: Any) -> NoveltyDetector:
        return self._novelty.setdefault(name, NoveltyDetector(name, **kw))

    # ---- observation ---- #
    def _record(self, a: Optional[Anomaly]) -> Optional[Anomaly]:
        if a is not None:
            self._log.append(a)
        return a

    def metric(self, name: str, value: float, **kw: Any) -> Optional[Anomaly]:
        return self._record(self.metric_baseline(name, **kw).observe(value))

    def rate_event(self, name: str, *, at: Optional[float] = None, **kw: Any) -> Optional[Anomaly]:
        return self._record(self.rate_monitor(name, **kw).observe(at))

    def sequence_event(self, name: str, symbol: str, **kw: Any) -> Optional[Anomaly]:
        return self._record(self.sequence_detector(name, **kw).observe(symbol))

    def categorical(self, name: str, value: str, **kw: Any) -> Optional[Anomaly]:
        return self._record(self.novelty_detector(name, **kw).observe(value))

    # ---- queries / bridge ---- #
    def recent(self, n: int = 20) -> List[Anomaly]:
        return list(self._log)[-n:]

    def to_threat_events(self, *, min_score: float = 0.0) -> List[ThreatEvent]:
        return [a.to_threat_event() for a in self._log if a.score >= min_score]

    def report(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for a in self._log:
            by_type[a.type.value] = by_type.get(a.type.value, 0) + 1
        return {"metrics": len(self._metrics), "rates": len(self._rates),
                "sequences": len(self._sequences), "novelty": len(self._novelty),
                "anomalies": len(self._log), "by_type": by_type}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA anomaly-detection self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    det = AnomalyDetector(clock=lambda: now["t"])

    # ---- statistical outlier: a metric living near 50, then a wild reading ---- #
    import random
    rng = random.Random(0)
    last = None
    for _ in range(30):
        last = det.metric("cpu_temp", 50 + rng.gauss(0, 1.5))
    assert last is None   # normal readings are not flagged
    spike = det.metric("cpu_temp", 95.0)
    print(f"\noutlier             : {spike.to_dict() if spike else None}")
    assert spike and spike.type is AnomalyType.OUTLIER

    # ---- rate spike: steady trickle, then a flood ---- #
    rm_name = "login_attempts"
    for i in range(8):           # one event every 5s -> low rate
        now["t"] += 5.0
        det.rate_event(rm_name, window=10.0, spike_ratio=3.0, min_events=5)
    flood = None
    for _ in range(12):          # a dozen events at the same instant -> spike
        flood = det.rate_event(rm_name, window=10.0, spike_ratio=3.0, min_events=5) or flood
    print(f"rate spike          : {flood.to_dict() if flood else None}")
    assert flood and flood.type is AnomalyType.RATE_SPIKE

    # ---- sequence anomaly: a regular A,B,A,B pattern, then a jolt ---- #
    for s in "ABABABABABAB":
        det.sequence_event("api_calls", s)
    jolt = det.sequence_event("api_calls", "ZZZ")
    print(f"sequence anomaly    : {jolt.to_dict() if jolt else None}")
    assert jolt and jolt.type is AnomalyType.SEQUENCE

    # ---- novelty: a new source IP never seen before ---- #
    for ip in ("10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.1"):
        det.categorical("source_ip", ip)
    new = det.categorical("source_ip", "203.0.113.9")
    print(f"novelty             : {new.to_dict() if new else None}")
    assert new and new.type is AnomalyType.NOVELTY

    # ---- bridge to the guardian: anomalies become threat events ---- #
    events = det.to_threat_events(min_score=0.0)
    print(f"\nthreat events       : {[(e.category.value, round(e.severity,2)) for e in events]}")
    assert len(events) >= 4
    assert all(isinstance(e, ThreatEvent) for e in events)

    # feed one into a real guardian end-to-end
    from nyxara.guard.guardian import Guardian
    g = Guardian()
    a, p = g.handle(spike.to_threat_event())
    print(f"guardian verdict    : level={a.level.label} actions={[x.value for x in p.actions]}")
    assert a.level.value >= 1

    print(f"\nreport              : {det.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
