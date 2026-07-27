"""HD memory: encode→remember→recall round-trips through the real HDC; recall is non-degrading."""

from __future__ import annotations

from nyxara.nyx5.hd_memory import HDMemory, SpikeEncoder
from nyxara.cognition.hyper_dimensional_vectors import HyperSpace


def test_remember_and_recall_exact():
    m = HDMemory(dim=10000, seed=1)
    m.remember("alpha", [(0.0, 1), (1.0, 2), (2.0, 3)])
    m.remember("beta", [(0.0, 10), (1.0, 11), (2.0, 12)])
    r = m.recall([(0.0, 1), (1.0, 2), (2.0, 3)])
    assert r is not None and r.name == "alpha"
    assert r.score > 0.9


def test_recall_is_associative_not_exact():
    m = HDMemory(dim=10000, seed=2)
    m.remember("alpha", [(0.0, 1), (1.0, 2), (2.0, 3), (3.0, 4)])
    m.remember("beta", [(0.0, 20), (1.0, 21), (2.0, 22), (3.0, 23)])
    # a noisy version of alpha (one neuron different) still recalls alpha
    r = m.recall([(0.0, 1), (1.0, 2), (2.0, 3), (3.0, 99)])
    assert r is not None and r.name == "alpha"


def test_non_degrading_many_memories():
    m = HDMemory(dim=10000, seed=3)
    for i in range(50):
        m.remember(f"m{i}", [(0.0, i * 3), (1.0, i * 3 + 1), (2.0, i * 3 + 2)])
    # an early memory is still recalled correctly after 49 later writes
    r = m.recall([(0.0, 0), (1.0, 1), (2.0, 2)])
    assert r is not None and r.name == "m0"


def test_unrelated_query_has_low_similarity():
    m = HDMemory(dim=10000, seed=4)
    m.remember("alpha", [(0.0, 1), (1.0, 2), (2.0, 3)])
    r = m.recall([(0.0, 500), (1.0, 501), (2.0, 502)], threshold=0.5)
    assert r is None            # nothing similar enough crosses the threshold


def test_encoder_order_matters():
    space = HyperSpace(dim=10000, seed=5)
    enc = SpikeEncoder(space)
    a = enc.encode([(0.0, 1), (10.0, 2)])
    b = enc.encode([(0.0, 2), (10.0, 1)])   # same neurons, swapped time buckets
    assert space.similarity(a, b) < 0.9      # order is encoded, not commutative


def test_roundtrip():
    m = HDMemory(dim=10000, seed=6)
    m.remember("k", [(0.0, 7), (1.0, 8)])
    m2 = HDMemory()
    m2.load_dict(m.to_dict())
    r = m2.recall([(0.0, 7), (1.0, 8)])
    assert r is not None and r.name == "k"
