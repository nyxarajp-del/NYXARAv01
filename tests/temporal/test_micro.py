"""Tests for nyxara.temporal.micro (Layer 1 — hardware/network)."""

from __future__ import annotations

from nyxara.temporal.clock import VirtualClock
from nyxara.temporal.micro import MicroLayer
from nyxara.temporal.signals import HardwareSample, MicroAggregate, NetworkSample


def _layer(**kw) -> MicroLayer:
    return MicroLayer(clock=VirtualClock(), window=32, **kw)


# -------------------- sampling -------------------- #
def test_sample_returns_hardware_sample():
    m = _layer()
    s = m.sample()
    assert isinstance(s, HardwareSample)
    assert s.source in ("proc", "fallback")
    assert 0.0 <= s.cpu_percent <= 100.0
    assert 0.0 <= s.mem_percent <= 100.0


def test_tick_grows_window_and_counts():
    m = _layer()
    for _ in range(5):
        m.tick()
    assert m.ticks == 5
    agg = m.aggregate()
    assert isinstance(agg, MicroAggregate)
    assert agg.n_samples == 5


def test_empty_aggregate_is_safe():
    m = _layer()
    agg = m.aggregate()
    assert agg.n_samples == 0 and agg.healthy


# -------------------- anomaly detection (deterministic, injected) -------------------- #
def test_cpu_and_memory_anomalies_flagged():
    m = _layer(cpu_alert=80.0, mem_alert=80.0)
    clk = m.clock
    for cpu, mem in [(10.0, 10.0), (95.0, 92.0), (12.0, 11.0)]:
        m._hw.append(HardwareSample(at=clk.now(), cpu_percent=cpu, mem_percent=mem))
        clk.advance(0.001)
    agg = m.aggregate()
    assert not agg.healthy
    assert any("cpu spike" in a for a in agg.anomalies)
    assert any("memory pressure" in a for a in agg.anomalies)


def test_packet_storm_flagged():
    m = _layer(packet_spike_factor=5.0, packet_spike_floor=100.0)
    clk = m.clock
    # steady low traffic, then a sudden spike in the last interval
    cum = 0
    for step in (10, 10, 10, 5000):
        cum += step
        m._net.append(NetworkSample(at=clk.now(), rx_packets=cum, tx_packets=0))
        clk.advance(1.0)
    # also need at least one hardware sample so aggregate runs the body
    m._hw.append(HardwareSample(at=clk.now(), cpu_percent=5.0, mem_percent=5.0))
    agg = m.aggregate()
    assert any("packet storm" in a for a in agg.anomalies)


def test_healthy_window_has_no_anomalies():
    m = _layer(cpu_alert=99.0, mem_alert=99.0)
    clk = m.clock
    for _ in range(4):
        m._hw.append(HardwareSample(at=clk.now(), cpu_percent=10.0, mem_percent=20.0))
        clk.advance(0.001)
    agg = m.aggregate()
    assert agg.healthy and not agg.anomalies


# -------------------- report -------------------- #
def test_report_shape():
    m = _layer()
    m.tick()
    rep = m.report()
    assert rep["name"] == "micro" and "last" in rep and "has_proc" in rep
