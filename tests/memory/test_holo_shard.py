"""Tests for memory/holo_shard.py — k-of-n erasure-coded reconstruction (Change set O).

Fast + standalone. Pins: any k of n shards rebuild the exact bytes; fewer than k cannot; a corrupt/
missing subset still works as long as k survive.
"""

from __future__ import annotations

import itertools

from nyxara.memory.holo_shard import HoloSharder


def test_roundtrip_from_exactly_k_of_any_subset():
    data = b"NYXARA episodic + semantic memory blob \x00\x01\x02 ... " * 5
    sh = HoloSharder(k=3, n=6)
    shards = sh.encode(data)
    assert len(shards) == 6
    # every 3-shard subset must reconstruct the original exactly
    for combo in itertools.combinations(shards, 3):
        assert sh.decode(list(combo)) == data


def test_all_shards_also_reconstruct():
    data = b"the whole mind"
    sh = HoloSharder(k=2, n=5)
    shards = sh.encode(data)
    assert sh.decode(shards) == data


def test_fewer_than_k_cannot_reconstruct():
    data = b"cannot rebuild below threshold"
    sh = HoloSharder(k=4, n=7)
    shards = sh.encode(data)
    assert sh.decode(shards[:3]) is None       # only 3 < k=4


def test_survives_losing_up_to_n_minus_k_shards():
    data = bytes(range(256)) * 3
    sh = HoloSharder(k=3, n=6)                   # tolerate losing any 3
    shards = sh.encode(data)
    survivors = [shards[0], shards[2], shards[5]]   # lost 1, 3, 4
    assert sh.decode(survivors) == data


def test_empty_payload_roundtrips():
    sh = HoloSharder(k=2, n=4)
    assert sh.decode(sh.encode(b"")) == b""


def test_bad_params_rejected():
    import pytest
    with pytest.raises(ValueError):
        HoloSharder(k=5, n=3)
