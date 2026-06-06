"""Tests for nyxara.guard.anomaly."""

from __future__ import annotations

import random

from nyxara.guard.anomaly import (Anomaly, AnomalyDetector, AnomalyType, NoveltyDetector,
                                  RateMonitor, ScalarBaseline, SequenceDetector)
from nyxara.guard.guardian import ThreatCategory, ThreatEvent


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# -------------------- ScalarBaseline -------------------- #
def test_scalar_first_no_anomaly():
    b = ScalarBaseline("m")
    assert b.observe(50.0) is None


def test_scalar_normal_no_anomaly():
    b = ScalarBaseline("m", warmup=5)
    rng = random.Random(0)
    flagged = [b.observe(10 + rng.gauss(0, 1)) for _ in range(30)]
    assert all(a is None for a in flagged)


def test_scalar_outlier_flagged():
    b = ScalarBaseline("m", warmup=5, k=3.0)
    for _ in range(20):
        b.observe(10.0)
    a = b.observe(100.0)
    assert a is not None and a.type is AnomalyType.OUTLIER
    assert a.score > 0.5


def test_scalar_tracks_slow_drift():
    b = ScalarBaseline("m", warmup=5, alpha=0.3, k=3.0)
    # slowly rising value should be learned, not flagged
    flagged = [b.observe(float(i)) for i in range(40)]
    assert flagged[-1] is None


def test_scalar_zscore():
    b = ScalarBaseline("m")
    b.observe(10.0)
    assert b.zscore(10.0) == 0.0


def test_scalar_negative_outlier():
    b = ScalarBaseline("m", warmup=5)
    for _ in range(20):
        b.observe(100.0)
    a = b.observe(-50.0)
    assert a is not None and a.type is AnomalyType.OUTLIER


# -------------------- RateMonitor -------------------- #
def test_rate_no_spike_steady():
    clk = FakeClock()
    rm = RateMonitor("r", window=10.0, spike_ratio=3.0, min_events=5, clock=clk)
    flagged = []
    for _ in range(10):
        clk.advance(5.0)
        flagged.append(rm.observe())
    assert all(a is None for a in flagged)


def test_rate_spike_flagged():
    clk = FakeClock()
    rm = RateMonitor("r", window=10.0, spike_ratio=3.0, min_events=5, clock=clk)
    for _ in range(8):
        clk.advance(5.0)
        rm.observe()
    spike = None
    for _ in range(15):
        spike = rm.observe() or spike   # all at the same instant -> burst
    assert spike is not None and spike.type is AnomalyType.RATE_SPIKE


def test_rate_window_prunes():
    clk = FakeClock()
    rm = RateMonitor("r", window=10.0, clock=clk)
    rm.observe()
    clk.advance(20.0)
    assert rm.rate() == 0  # the lone old event aged out of the window


def test_rate_min_events_gate():
    clk = FakeClock()
    rm = RateMonitor("r", window=100.0, spike_ratio=2.0, min_events=50, clock=clk)
    # a few rapid events but below min_events -> no spike
    flagged = [rm.observe() for _ in range(5)]
    assert all(a is None for a in flagged)


# -------------------- SequenceDetector -------------------- #
def test_sequence_pattern_no_anomaly():
    sd = SequenceDetector("s")
    flagged = [sd.observe(c) for c in "ABABABABAB"]
    # the regular pattern eventually stops surprising
    assert flagged[-1] is None


def test_sequence_jolt_flagged():
    sd = SequenceDetector("s")
    for c in "ABABABABABAB":
        sd.observe(c)
    a = sd.observe("ZZZ")
    assert a is not None and a.type is AnomalyType.SEQUENCE


# -------------------- NoveltyDetector -------------------- #
def test_novelty_first_seen_after_warmup():
    nd = NoveltyDetector("n", warmup=2)
    nd.observe("a")
    nd.observe("a")
    nd.observe("a")  # past warmup, but 'a' already seen
    a = nd.observe("b")
    assert a is not None and a.type is AnomalyType.NOVELTY


def test_novelty_within_warmup_not_flagged():
    nd = NoveltyDetector("n", warmup=5)
    flagged = [nd.observe(x) for x in ("a", "b", "c")]
    assert all(a is None for a in flagged)


def test_novelty_repeat_not_flagged():
    nd = NoveltyDetector("n", warmup=1)
    nd.observe("a")
    nd.observe("b")  # novel
    a = nd.observe("a")  # repeat
    assert a is None


# -------------------- Anomaly bridge -------------------- #
def test_anomaly_to_threat_event():
    a = Anomaly(AnomalyType.RATE_SPIKE, "logins", 0.8, detail="flood")
    ev = a.to_threat_event()
    assert isinstance(ev, ThreatEvent)
    assert ev.category is ThreatCategory.RESOURCE_ABUSE
    assert ev.severity == 0.8


def test_anomaly_type_category_map():
    assert Anomaly(AnomalyType.SEQUENCE, "x", 0.5).to_threat_event().category is ThreatCategory.INTRUSION
    assert Anomaly(AnomalyType.NOVELTY, "x", 0.5).to_threat_event().category is ThreatCategory.UNKNOWN


def test_anomaly_to_dict():
    d = Anomaly(AnomalyType.OUTLIER, "cpu", 0.9, value=99.0, expected=50.0).to_dict()
    assert d["type"] == "outlier" and d["value"] == 99.0


# -------------------- AnomalyDetector facade -------------------- #
def test_facade_metric():
    det = AnomalyDetector()
    for _ in range(20):
        det.metric("temp", 50.0, warmup=5)
    a = det.metric("temp", 200.0)
    assert a is not None and a.type is AnomalyType.OUTLIER


def test_facade_rate_event():
    clk = FakeClock()
    det = AnomalyDetector(clock=clk)
    for _ in range(8):
        clk.advance(5.0)
        det.rate_event("hits", window=10.0, min_events=5)
    spike = None
    for _ in range(15):
        spike = det.rate_event("hits") or spike
    assert spike is not None


def test_facade_sequence():
    det = AnomalyDetector()
    for c in "ABABABABAB":
        det.sequence_event("seq", c)
    a = det.sequence_event("seq", "QQQ")
    assert a is not None and a.type is AnomalyType.SEQUENCE


def test_facade_categorical():
    det = AnomalyDetector()
    for ip in ("a", "a", "b", "a"):
        det.categorical("ip", ip, warmup=2)
    a = det.categorical("ip", "zzz")
    assert a is not None and a.type is AnomalyType.NOVELTY


def test_facade_logs_anomalies():
    det = AnomalyDetector()
    for _ in range(20):
        det.metric("temp", 50.0, warmup=5)
    det.metric("temp", 999.0)
    assert len(det.recent()) >= 1


def test_facade_lazy_registration():
    det = AnomalyDetector()
    det.metric("a", 1.0)
    det.rate_event("b")
    det.sequence_event("c", "x")
    det.categorical("d", "y")
    r = det.report()
    assert r["metrics"] == 1 and r["rates"] == 1 and r["sequences"] == 1 and r["novelty"] == 1


def test_facade_to_threat_events():
    det = AnomalyDetector()
    for _ in range(20):
        det.metric("temp", 50.0, warmup=5)
    det.metric("temp", 999.0)
    events = det.to_threat_events()
    assert len(events) >= 1 and all(isinstance(e, ThreatEvent) for e in events)


def test_facade_threat_events_min_score():
    det = AnomalyDetector()
    det.categorical("ip", "a", warmup=0)
    det.categorical("ip", "b")  # novelty score 0.5
    assert det.to_threat_events(min_score=0.9) == []
    assert len(det.to_threat_events(min_score=0.4)) >= 1


def test_facade_report():
    det = AnomalyDetector()
    det.metric("a", 1.0)
    r = det.report()
    assert "anomalies" in r and "by_type" in r


def test_recent_limit():
    det = AnomalyDetector()
    # each fresh metric yields one clean outlier -> the log holds 3 anomalies
    for name in ("a", "b", "c"):
        for _ in range(10):
            det.metric(name, 5.0, warmup=2)
        det.metric(name, 999.0)
    assert len(det.recent(2)) == 2
