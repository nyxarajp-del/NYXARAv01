"""Negentropy maintainer: dedup + prune stale, interval gating, fail-soft on malformed entries."""

from __future__ import annotations

import time

from nyxara.nyx5.negentropy import NegentropyMaintainer


def test_dedup_and_prune():
    now = time.time()
    store = {
        "a": {"value": 1, "ts": now},
        "b": {"value": 1, "ts": now},          # duplicate of a
        "c": {"value": 2, "ts": 0.0},          # stale
    }
    m = NegentropyMaintainer(stale_seconds=1000.0)
    rep = m.sweep(store, now=now)
    assert rep.deduped == 1
    assert rep.pruned == 1
    assert set(store.keys()) == {"a"}          # canonical copy kept, dup+stale gone


def test_interval_gating():
    m = NegentropyMaintainer(interval_ticks=3)
    due = [m.tick() for _ in range(6)]
    assert due == [False, False, True, False, False, True]


def test_fail_soft_on_malformed():
    store = {"ok": {"value": 1, "ts": time.time()}, "weird": "not-a-dict"}
    m = NegentropyMaintainer()
    rep = m.sweep(store)                        # must not raise
    assert "ok" in store
    assert rep.integrity


def test_integrity_changes_with_content():
    m = NegentropyMaintainer()
    s1 = {"a": {"value": 1, "ts": time.time()}}
    s2 = {"a": {"value": 2, "ts": time.time()}}
    assert m.sweep(s1).integrity != m.sweep(s2).integrity
