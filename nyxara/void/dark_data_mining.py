"""NYXARA · void/dark_data_mining.py — the Dark-Data Miner (Void · 1).

Most systems only read the signal. NYXARA also reads the **void** — the places everyone
else dismisses as noise, garbage, or empty space — because truth hides there too:

* what a log is **not** saying (the silence between the lines);
* the **gap** where an event should have happened and didn't;
* the faint **anomaly buried in noise** that a mean-and-stddev glance would smear away;
* the **absent category** — the option no one mentioned, the question never asked;
* the **hidden rhythm** inside data that looks random.

This module turns that negative space into evidence. Every method is **robust** by design
(median / MAD rather than mean / stddev) so a few wild values can't blind it, and every
report carries an explicit *severity* so the orchestrator can weigh how loud a silence is.

Pure standard library — :mod:`statistics`, :mod:`math`, :class:`collections.Counter`. No
external dependencies, deterministic, auditable.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

__all__ = [
    "Anomaly",
    "AnomalyReport",
    "Gap",
    "GapReport",
    "Absence",
    "SilenceReport",
    "PeriodicityReport",
    "DarkDataMiner",
]

# 0.6745 = 1/1.4826; the constant that makes MAD a consistent estimator of stddev for
# normal data, so a "robust z-score" is comparable to an ordinary z-score.
_MAD_TO_SIGMA = 1.4826
_DEFAULT_Z = 3.5     # the standard robust-outlier cutoff (Iglewicz & Hoaglin)


def _robust_scale(series: Sequence[float], median: float) -> float:
    """A robust spread estimate: MAD·1.4826, falling back to stddev, then to 0."""
    mad = statistics.median(abs(x - median) for x in series)
    if mad > 1e-12:
        return mad * _MAD_TO_SIGMA
    # degenerate MAD (e.g. a majority of identical values) → fall back to stddev
    if len(series) >= 2:
        try:
            return statistics.pstdev(series)
        except statistics.StatisticsError:
            return 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Anomaly:
    index: int
    value: float
    robust_z: float
    direction: str            # "high" or "low"

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "value": round(self.value, 6),
                "robust_z": round(self.robust_z, 4), "direction": self.direction}


@dataclass
class AnomalyReport:
    anomalies: List[Anomaly]
    median: float
    scale: float
    n: int
    threshold: float

    @property
    def count(self) -> int:
        return len(self.anomalies)

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "median": round(self.median, 6),
                "scale": round(self.scale, 6), "n": self.n,
                "threshold": self.threshold,
                "anomalies": [a.to_dict() for a in self.anomalies]}


@dataclass
class Gap:
    start: float
    end: float
    duration: float
    severity: float           # robust z-score of this interval among all intervals

    def to_dict(self) -> Dict[str, Any]:
        return {"start": round(self.start, 6), "end": round(self.end, 6),
                "duration": round(self.duration, 6), "severity": round(self.severity, 4)}


@dataclass
class GapReport:
    gaps: List[Gap]
    typical_interval: float
    n_events: int

    @property
    def count(self) -> int:
        return len(self.gaps)

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "typical_interval": round(self.typical_interval, 6),
                "n_events": self.n_events, "gaps": [g.to_dict() for g in self.gaps]}


@dataclass
class Absence:
    item: Any
    expected: float
    observed: float
    deficit_ratio: float      # (expected − observed) / expected ∈ (0, 1]
    severity: float           # how loud this absence is [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {"item": self.item, "expected": round(self.expected, 4),
                "observed": round(self.observed, 4),
                "deficit_ratio": round(self.deficit_ratio, 4),
                "severity": round(self.severity, 4)}


@dataclass
class SilenceReport:
    absences: List[Absence]
    total_missing: int        # items expected but entirely unseen

    @property
    def count(self) -> int:
        return len(self.absences)

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "total_missing": self.total_missing,
                "absences": [a.to_dict() for a in self.absences]}


@dataclass
class PeriodicityReport:
    dominant_lag: Optional[int]
    strength: float           # normalised autocorrelation at the dominant lag [−1, 1]
    has_cycle: bool
    autocorr: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"dominant_lag": self.dominant_lag, "strength": round(self.strength, 4),
                "has_cycle": self.has_cycle,
                "autocorr": [round(a, 4) for a in self.autocorr]}


# --------------------------------------------------------------------------- #
# Miner
# --------------------------------------------------------------------------- #
class DarkDataMiner:
    """Extract hidden structure from noise, silence, and missing data.

    Parameters
    ----------
    z_threshold:    robust z-score cutoff for "this is a real anomaly / gap, not noise".
    cycle_threshold:  minimum normalised autocorrelation to call a rhythm a real cycle.
    """

    def __init__(self, *, z_threshold: float = _DEFAULT_Z,
                 cycle_threshold: float = 0.3) -> None:
        self.z_threshold = float(z_threshold)
        self.cycle_threshold = float(cycle_threshold)

    # ---- anomaly buried in noise ----------------------------------------- #
    def mine_anomalies(self, series: Sequence[float], *,
                       z_threshold: Optional[float] = None) -> AnomalyReport:
        """Find the faint outliers a mean/stddev pass would smear away (robust z-score)."""
        thr = self.z_threshold if z_threshold is None else float(z_threshold)
        data = [float(x) for x in series]
        if len(data) < 3:
            return AnomalyReport([], median=(data[0] if data else 0.0),
                                 scale=0.0, n=len(data), threshold=thr)
        med = statistics.median(data)
        scale = _robust_scale(data, med)
        anomalies: List[Anomaly] = []
        if scale > 1e-12:
            for i, x in enumerate(data):
                z = (x - med) / scale
                if abs(z) >= thr:
                    anomalies.append(Anomaly(index=i, value=x, robust_z=z,
                                             direction="high" if z > 0 else "low"))
        anomalies.sort(key=lambda a: abs(a.robust_z), reverse=True)
        return AnomalyReport(anomalies, median=med, scale=scale, n=len(data), threshold=thr)

    # ---- the silence where an event should have been --------------------- #
    def mine_gaps(self, timestamps: Sequence[float], *,
                  expected_interval: Optional[float] = None,
                  z_threshold: Optional[float] = None) -> GapReport:
        """Find unusually long silences between events — where something *should* have
        happened and didn't. A gap is an interval whose length is a robust-z outlier among
        all intervals (or, if ``expected_interval`` is given, far above it)."""
        thr = self.z_threshold if z_threshold is None else float(z_threshold)
        ts = sorted(float(t) for t in timestamps)
        if len(ts) < 3:
            return GapReport([], typical_interval=0.0, n_events=len(ts))
        intervals = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        typical = statistics.median(intervals)
        scale = _robust_scale(intervals, typical)
        gaps: List[Gap] = []
        for i, d in enumerate(intervals):
            z = (d - typical) / scale if scale > 1e-12 else 0.0
            is_gap = (z >= thr) if scale > 1e-12 else False
            if expected_interval is not None and expected_interval > 0:
                is_gap = is_gap or d >= 2.0 * expected_interval
            if is_gap and d > typical:
                gaps.append(Gap(start=ts[i], end=ts[i + 1], duration=d, severity=z))
        gaps.sort(key=lambda g: g.duration, reverse=True)
        return GapReport(gaps, typical_interval=typical, n_events=len(ts))

    # ---- the option no one mentioned ------------------------------------- #
    def mine_silence(self, observed: Union[Mapping[Any, float], Iterable[Any]],
                     expected: Union[Mapping[Any, float], Iterable[Any]]) -> SilenceReport:
        """Find expected things that are absent or conspicuously under-represented —
        what the data is *not* telling you.

        ``observed`` may be a mapping of counts or any iterable of items (counted here).
        ``expected`` may be a mapping of expected counts, or an iterable of items that
        *should* appear at least once.
        """
        if isinstance(observed, Mapping):
            obs_counts: Dict[Any, float] = {k: float(v) for k, v in observed.items()}
        else:
            obs_counts = {k: float(v) for k, v in Counter(observed).items()}

        if isinstance(expected, Mapping):
            exp_counts: Dict[Any, float] = {k: float(v) for k, v in expected.items()}
        else:
            exp_counts = {k: 1.0 for k in expected}

        absences: List[Absence] = []
        total_missing = 0
        for item, exp in exp_counts.items():
            if exp <= 0:
                continue
            obs = obs_counts.get(item, 0.0)
            if obs >= exp:
                continue
            deficit = (exp - obs) / exp
            # severity blends the relative deficit with how much was expected (loud absences
            # are large *and* of something we strongly expected).
            severity = deficit * (1.0 - math.exp(-exp))
            if obs == 0.0:
                total_missing += 1
            absences.append(Absence(item=item, expected=exp, observed=obs,
                                    deficit_ratio=deficit, severity=severity))
        absences.sort(key=lambda a: a.severity, reverse=True)
        return SilenceReport(absences, total_missing=total_missing)

    # ---- the rhythm inside the noise ------------------------------------- #
    def mine_periodicity(self, series: Sequence[float], *,
                         max_lag: Optional[int] = None,
                         cycle_threshold: Optional[float] = None) -> PeriodicityReport:
        """Find a hidden cycle in data that looks random (normalised autocorrelation)."""
        thr = self.cycle_threshold if cycle_threshold is None else float(cycle_threshold)
        data = [float(x) for x in series]
        n = len(data)
        if n < 4:
            return PeriodicityReport(None, 0.0, False, [])
        mean = statistics.fmean(data)
        denom = sum((x - mean) ** 2 for x in data)
        if denom <= 1e-12:
            return PeriodicityReport(None, 0.0, False, [])
        top = max_lag if max_lag is not None else n // 2
        top = max(1, min(top, n - 1))
        autocorr: List[float] = []
        for lag in range(1, top + 1):
            num = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n - lag))
            autocorr.append(num / denom)
        best_idx = max(range(len(autocorr)), key=lambda i: autocorr[i])
        strength = autocorr[best_idx]
        dominant_lag = best_idx + 1
        has_cycle = strength >= thr
        return PeriodicityReport(dominant_lag if has_cycle else None,
                                 strength, has_cycle, autocorr)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import random

    print("=" * 70)
    print("NYXARA dark-data-miner self-test")
    print("=" * 70)
    rng = random.Random(0)
    miner = DarkDataMiner()

    # 1) anomaly buried in noise: small gaussian noise + two implanted spikes
    series = [rng.gauss(0.0, 1.0) for _ in range(200)]
    series[50] = 12.0      # high spike
    series[120] = -11.0    # low spike
    ar = miner.mine_anomalies(series)
    found = {a.index for a in ar.anomalies}
    print(f"\nanomalies found      : {sorted(found)} (count={ar.count})")
    assert 50 in found and 120 in found
    assert ar.count <= 5    # robust: it does not drown in the noise

    # pure noise → essentially nothing flagged
    clean = miner.mine_anomalies([rng.gauss(0.0, 1.0) for _ in range(200)])
    print(f"pure-noise anomalies : {clean.count}")
    assert clean.count <= 2

    # 2) the silence where an event should have been
    times = [float(i) for i in range(20)]            # one event per unit time
    times = times[:10] + [t + 9.0 for t in times[10:]]  # a 10-unit silence after t=9
    gr = miner.mine_gaps(times, expected_interval=1.0)
    print(f"\ngaps found           : {gr.count}  biggest={gr.gaps[0].to_dict() if gr.gaps else None}")
    assert gr.count == 1 and gr.gaps[0].duration > 5.0

    # 3) the option no one mentioned
    observed = ["yes", "yes", "no", "yes", "no"]      # 'maybe' and 'abstain' never appear
    expected = ["yes", "no", "maybe", "abstain"]
    sr = miner.mine_silence(observed, expected)
    missing_items = {a.item for a in sr.absences}
    print(f"\nabsent options       : {sorted(missing_items)} total_missing={sr.total_missing}")
    assert "maybe" in missing_items and "abstain" in missing_items
    assert sr.total_missing == 2

    # 4) the rhythm inside the noise: period-5 signal + noise
    signal = [math.sin(2 * math.pi * i / 5.0) + rng.gauss(0.0, 0.3) for i in range(80)]
    pr = miner.mine_periodicity(signal)
    print(f"\nhidden cycle         : lag={pr.dominant_lag} strength={pr.strength:.2f} "
          f"has_cycle={pr.has_cycle}")
    assert pr.has_cycle and pr.dominant_lag == 5

    # pure noise → no cycle claimed (honest)
    noise = [rng.gauss(0.0, 1.0) for _ in range(80)]
    pr2 = miner.mine_periodicity(noise)
    print(f"noise cycle          : has_cycle={pr2.has_cycle} (honest)")
    assert not pr2.has_cycle

    print("\nALL SELF-TESTS PASSED ✓")
