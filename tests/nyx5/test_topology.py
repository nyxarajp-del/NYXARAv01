"""LiveTopology: prune weak+stale, grow correlated pairs, respect the cap, stay consistent."""

from __future__ import annotations

from nyxara.nyx5.topology import LiveTopology


def test_connect_and_query():
    topo = LiveTopology()
    syn = topo.connect(0, 1, weight=0.5)
    assert syn is not None
    assert topo.has(0, 1)
    assert len(topo) == 1
    assert topo.connect(0, 1) is syn      # idempotent
    assert len(topo) == 1
    assert topo.connect(2, 2) is None     # no self-loops


def test_prune_removes_weak_and_stale():
    topo = LiveTopology(prune_threshold=0.1, prune_stale_ms=100.0)
    weak = topo.connect(0, 1, weight=0.05)
    strong = topo.connect(0, 2, weight=0.9)
    assert weak is not None and strong is not None
    weak.last_pre_t = 0.0                 # stale
    strong.last_pre_t = 0.0
    pruned = topo.prune(now=1000.0)       # long after -> weak is stale + weak
    assert pruned == 1
    assert not topo.has(0, 1)
    assert topo.has(0, 2)                 # strong survives


def test_prune_spares_weak_but_active():
    topo = LiveTopology(prune_threshold=0.1, prune_stale_ms=100.0)
    syn = topo.connect(0, 1, weight=0.05)
    assert syn is not None
    syn.last_pre_t = 950.0                # recently active
    assert topo.prune(now=1000.0) == 0    # not stale enough -> spared


def test_grow_and_cap():
    topo = LiveTopology(max_synapses=3)
    added = topo.grow([(0, 1), (0, 2), (1, 2), (1, 3)])
    assert added == 3                     # capped at max_synapses
    assert len(topo) == 3


def test_roundtrip_serialisation():
    topo = LiveTopology(max_synapses=100)
    topo.connect(0, 1, weight=0.3)
    topo.connect(1, 2, weight=0.7)
    d = topo.to_dict()
    fresh = LiveTopology()
    fresh.load_dict(d)
    assert len(fresh) == 2
    assert fresh.get(1, 2).weight == 0.7
