"""SpikingNetwork: bounded raster, energy = spike count, STDP mutates weights, no runaway growth."""

from __future__ import annotations

from nyxara.nyx5.snn import SpikingNetwork


def _net(**kw):
    return SpikingNetwork(n_neurons=4, threshold=0.5, initial_connectivity=0.0, seed=3, **kw)


def test_input_produces_bounded_raster():
    net = _net()
    net.topology.connect(0, 1, weight=0.6, delay=1.0)
    r = net.stimulate([(0, 0.0, 1.0)], duration=10.0)
    assert 0 in r.firing_neurons()      # the stimulated neuron fires
    assert 1 in r.firing_neurons()      # and the cascade reaches its target
    assert r.energy_proxy == float(r.n_spikes)


def test_stdp_potentiates_a_causal_synapse():
    net = _net()
    syn = net.topology.connect(0, 1, weight=0.6, delay=1.0)
    net.stimulate([(0, 0.0, 1.0)], duration=10.0)
    assert syn.weight > 0.6             # pre-before-post -> potentiated


def test_no_unbounded_growth():
    net = SpikingNetwork(n_neurons=16, threshold=0.4, max_synapses=64, seed=5)
    for _ in range(200):
        net.stimulate([(i, 0.0, 1.0) for i in range(4)], duration=20.0)
    assert len(net.topology) <= 64      # cap respected across many turns
    assert net.total_spikes > 0


def test_event_budget_bounds_work():
    net = SpikingNetwork(n_neurons=32, threshold=0.1, max_events=100,
                         initial_connectivity=0.3, seed=7)
    r = net.stimulate([(i, 0.0, 5.0) for i in range(8)], duration=1000.0)
    assert r.events <= 100              # never exceeds the budget even under dense feedback


def test_roundtrip_serialisation():
    net = _net()
    net.topology.connect(0, 1, weight=0.6)
    net.stimulate([(0, 0.0, 1.0)], duration=10.0)
    d = net.to_dict()
    fresh = SpikingNetwork(n_neurons=4, initial_connectivity=0.0, seed=3)
    fresh.load_dict(d)
    assert len(fresh.topology) == len(net.topology)
    assert fresh.ticks == net.ticks
