"""Polymorphic sensorium: numeric streams -> spikes, auto-register capped, silent on empty."""

from __future__ import annotations

from nyxara.nyx5.sensorium import PolymorphicSensorium


def test_numeric_stream_produces_spikes():
    s = PolymorphicSensorium(n_neurons=64, max_channels=8)
    s.sense({"cpu": 0.0})           # establish range
    out = s.sense({"cpu": 1.0})     # high value -> more spikes
    assert out.spikes
    assert "cpu" in out.channels


def test_channel_cap_respected():
    s = PolymorphicSensorium(n_neurons=256, max_channels=2)
    s.sense({"a": 1.0, "b": 2.0, "c": 3.0})
    assert len(s.channels()) == 2   # third channel refused past the cap


def test_empty_readings_are_noop():
    s = PolymorphicSensorium(n_neurons=32)
    out = s.sense({})
    assert out.spikes == []


def test_non_numeric_channel_skipped():
    s = PolymorphicSensorium(n_neurons=32)
    out = s.sense({"good": 1.0, "bad": "not-a-number"})
    assert "good" in out.channels
    assert "bad" not in out.channels
