"""LIF neuron: integrates charge, fires at threshold, resets, honours the refractory window."""

from __future__ import annotations

from nyxara.nyx5.neuron import LIFNeuron, NeuronState, SpikeEvent


def test_fires_at_threshold_and_resets():
    n = LIFNeuron(neuron_id=0, threshold=1.0, v_reset=0.0, refractory=2.0, tau=20.0)
    assert n.integrate(0.0, 0.5) is False          # below threshold
    assert n.v == 0.5
    assert n.integrate(0.0, 0.6) is True            # crosses 1.0 -> fires
    assert n.v == 0.0                               # reset
    assert n.spike_count == 1


def test_no_fire_below_threshold():
    n = LIFNeuron(neuron_id=1, threshold=1.0)
    for _ in range(5):
        assert n.integrate(0.0, 0.1) is False
    assert n.spike_count == 0


def test_membrane_leaks_over_time():
    n = LIFNeuron(neuron_id=2, tau=10.0, threshold=100.0)
    n.integrate(0.0, 1.0)
    v0 = n.v
    n.integrate(10.0, 0.0)                          # one tau later, no new charge
    assert n.v < v0
    assert abs(n.v - v0 * 2.718281828 ** -1) < 1e-3


def test_refractory_window_blocks_input():
    n = LIFNeuron(neuron_id=3, threshold=1.0, refractory=5.0)
    assert n.integrate(0.0, 1.0) is True            # fires at t=0
    assert n.state(1.0) is NeuronState.REFRACTORY
    assert n.integrate(1.0, 5.0) is False           # within refractory: ignored even at huge charge
    assert n.integrate(6.0, 1.0) is True            # after refractory: fires again


def test_spike_event_roundtrip():
    e = SpikeEvent(time=3.0, neuron_id=7, weight=0.25)
    d = e.to_dict()
    assert d["neuron_id"] == 7 and d["weight"] == 0.25
