"""Holographic swarm: fragment recalls what it holds; merge is lossless; single-node zero-op."""

from __future__ import annotations

from nyxara.nyx5.holo_swarm import HolographicShard


def test_fragment_recalls_its_own_memories():
    a = HolographicShard("a", seed=1)
    a.remember("m1", [(0.0, 1), (1.0, 2)])
    a.remember("m2", [(0.0, 10), (1.0, 11)])
    frag = a.fragment(["m1"])
    r = frag.recall([(0.0, 1), (1.0, 2)])
    assert r is not None and r.name == "m1"
    assert len(frag) == 1               # a partition holds only its slice


def test_merge_is_lossless_union():
    a = HolographicShard("a", seed=1)
    b = HolographicShard("b", seed=1)
    a.remember("m1", [(0.0, 1), (1.0, 2)])
    b.remember("m2", [(0.0, 10), (1.0, 11)])
    a.merge(b)
    assert set(a.memory.keys()) == {"m1", "m2"}
    assert a.recall([(0.0, 10), (1.0, 11)]).name == "m2"


def test_merge_idempotent():
    a = HolographicShard("a", seed=2)
    b = HolographicShard("b", seed=2)
    b.remember("m", [(0.0, 5)])
    a.merge(b)
    changed = a.merge(b)                 # second merge changes nothing
    assert changed == 0


def test_higher_version_wins():
    a = HolographicShard("a", seed=3)
    b = HolographicShard("b", seed=3)
    a.remember("k", [(0.0, 1)], version=1)
    b.remember("k", [(0.0, 9)], version=5)
    a.merge(b)
    assert a._versions["k"] == 5         # newer version adopted
