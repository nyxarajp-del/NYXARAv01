"""Tests for nyxara.causal.thermo_inference."""

from __future__ import annotations

from nyxara.causal.thermo_inference import ThermodynamicMonitor


def test_calm_stream_does_not_spike():
    mon = ThermodynamicMonitor(window=32, spike_z=2.5, warmup=8)
    spikes = 0
    for _ in range(40):
        r = mon.observe(1.0 + 0.001 * _)
        spikes += int(r.spike)
    assert spikes == 0


def test_sudden_jump_spikes_and_heals():
    healed = {"n": 0}

    def healer(reading):
        healed["n"] += 1
        return True

    mon = ThermodynamicMonitor(window=32, spike_z=3.0, warmup=8, heal=healer)
    for _ in range(30):
        mon.observe(1.0)
    r = mon.observe(50.0)                             # a large, sudden surprise
    assert r.spike is True
    assert r.healed is True
    assert healed["n"] == 1
    assert r.z > 3.0


def test_vector_surprise_is_reduced_to_a_magnitude():
    mon = ThermodynamicMonitor(window=8)
    r = mon.observe([3.0, 4.0])                       # L2 norm == 5
    assert abs(r.surprise - 5.0) < 1e-9


def test_entropy_is_nonnegative_and_status_counts():
    mon = ThermodynamicMonitor(window=16, warmup=4, spike_z=2.0)
    for v in (1.0, 2.0, 1.0, 3.0, 1.0, 9.0, 1.0, 1.0):
        r = mon.observe(v)
        assert r.entropy >= 0.0
    st = mon.status()
    assert st["beats"] == 8
    assert st["spikes"] >= 1


def test_a_failing_healer_never_propagates():
    def boom(reading):
        raise RuntimeError("healer exploded")

    mon = ThermodynamicMonitor(window=8, warmup=2, spike_z=2.0, heal=boom)
    for _ in range(6):
        mon.observe(1.0)
    r = mon.observe(40.0)                             # spike → healer raises, but must be swallowed
    assert r.spike is True
    assert r.healed is False
