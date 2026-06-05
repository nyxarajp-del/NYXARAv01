"""Tests for nyxara.identity.interoception."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import AffectSystem
from nyxara.identity.interoception import (
    FeltState,
    Interoception,
    InteroceptiveSignals,
)


def _io(**kw):
    return Interoception(signals=InteroceptiveSignals(**kw))


# -------------------- signals -------------------- #
def test_normalized_ranges():
    s = InteroceptiveSignals(compute_load=1.5, latency_ms=500, queue_depth=200,
                             queue_capacity=100)
    n = s.normalized()
    assert all(0.0 <= v <= 1.0 for v in n.values())
    assert n["compute_load"] == 1.0  # clamped
    assert n["queue"] == 1.0


def test_latency_normalization():
    s = InteroceptiveSignals(latency_ms=125, reference_latency_ms=250)
    assert s.normalized()["latency"] == pytest.approx(0.5)


# -------------------- felt state -------------------- #
def test_high_compute_is_effort():
    f = _io(compute_load=0.95).felt()
    assert f.effort > 0.6


def test_low_energy_is_fatigue():
    f = _io(energy=0.05).felt()
    assert f.fatigue > 0.8


def test_high_queue_is_pressure():
    f = _io(queue_depth=90, queue_capacity=100).felt()
    assert f.pressure > 0.5


def test_high_latency_is_sluggish():
    f = _io(latency_ms=400, reference_latency_ms=250).felt()
    assert f.sluggishness > 0.6


def test_low_confidence_is_unease():
    f = _io(confidence=0.2).felt()
    assert f.unease > 0.6


def test_high_error_is_discomfort():
    f = _io(error_rate=0.6).felt()
    assert f.discomfort > 0.4


def test_calm_body_has_vitality():
    f = _io(compute_load=0.1, energy=1.0, confidence=0.9, error_rate=0.0).felt()
    assert f.vitality > 0.6


def test_dominant_sensation():
    f = _io(compute_load=0.95, energy=1.0).felt()
    assert f.dominant() == "effort"
    calm = _io(compute_load=0.1, energy=1.0, confidence=0.9).felt()
    assert calm.dominant() == "ease"


def test_felt_describe():
    stressed = _io(compute_load=0.95, queue_depth=90, error_rate=0.5).felt()
    assert "strain" in stressed.describe() or "feel" in stressed.describe()
    calm = _io(compute_load=0.1, energy=1.0, confidence=0.9).felt()
    assert "ease" in calm.describe() or "steady" in calm.describe()


def test_felt_to_dict():
    d = _io().felt().to_dict()
    assert "effort" in d and "vitality" in d


# -------------------- comfort / allostatic load -------------------- #
def test_calm_body_high_comfort():
    io = _io(compute_load=0.1, memory_load=0.2, latency_ms=30, queue_depth=0,
             error_rate=0.0, confidence=0.9, energy=1.0)
    assert io.comfort() > 0.8


def test_stressed_body_low_comfort():
    io = _io(compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
             error_rate=0.4, confidence=0.4, energy=0.3)
    assert io.comfort() < 0.4


def test_allostatic_load_in_range():
    io = _io(compute_load=0.95, error_rate=1.0, energy=0.0, confidence=0.0)
    load = io.allostatic_load()
    assert 0.0 <= load <= 1.0


# -------------------- affective tone -------------------- #
def test_calm_positive_valence():
    tone = _io(compute_load=0.1, confidence=0.9, energy=1.0).to_affect()
    assert tone["valence"] > 0


def test_stressed_negative_valence_high_arousal():
    tone = _io(compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
               error_rate=0.4, confidence=0.4, energy=0.3).to_affect()
    assert tone["valence"] < 0
    assert tone["arousal"] > 0.6


def test_fatigue_lowers_arousal():
    tired = _io(energy=0.05, compute_load=0.1, queue_depth=0, confidence=0.7)
    wired = _io(energy=1.0, compute_load=0.9, queue_depth=80, confidence=0.7)
    assert tired.to_affect()["arousal"] < wired.to_affect()["arousal"]


def test_push_to_affect():
    affect = AffectSystem()
    io = _io(compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
             error_rate=0.4, confidence=0.4, energy=0.3)
    io.push_to_affect(affect)
    assert affect.emotion is not None
    assert affect.emotion.valence < 0
    assert affect.emotion.cause.startswith("interoception")


# -------------------- update / ingest -------------------- #
def test_update():
    io = _io()
    io.update(compute_load=0.8, energy=0.5)
    assert io.signals.compute_load == 0.8
    assert io.signals.energy == 0.5


def test_update_ignores_unknown():
    io = _io()
    io.update(nonexistent=1.0)
    assert not hasattr(io.signals, "nonexistent")


def test_sample_with_source():
    src = lambda: InteroceptiveSignals(compute_load=0.42)
    io = Interoception(source=src)
    s = io.sample()
    assert s.compute_load == 0.42


def test_ingest_from_bus_and_presence():
    class FakeQueue:
        def qsize(self):
            return 30

    class FakeBus:
        _queue = FakeQueue()
        _max_queue = 100

    class FakePresence:
        energy = 0.6

    io = Interoception().ingest(bus=FakeBus(), presence=FakePresence())
    assert io.signals.queue_depth == 30
    assert io.signals.queue_capacity == 100
    assert io.signals.energy == 0.6


def test_ingest_from_telemetry():
    class FakeTel:
        error_rate = 0.3
        latency_ms = 150

    io = Interoception().ingest(telemetry=FakeTel())
    assert io.signals.error_rate == 0.3
    assert io.signals.latency_ms == 150


# -------------------- reporting -------------------- #
def test_body_report_nonempty():
    assert _io().body_report()


def test_status_structure():
    s = _io().status()
    assert "signals" in s and "felt" in s and "comfort" in s and "affect" in s
    assert "report" in s
