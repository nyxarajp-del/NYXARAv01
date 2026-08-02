"""Tests for nyxara.memory.holographic_field — Holographic Memory Field.

Memories are stored as key⊗value bindings bundled into one continuous HDC field vector; recall is a
single unbind + cleanup (associative, not a per-chunk scan). These tests assert exact key→value
recall, content-addressable recall, honest saturation, and graceful forgetting past capacity.
"""

from __future__ import annotations

import pytest

from nyxara.cognition.hyper_dimensional_vectors import has_numpy
from nyxara.memory.holographic_field import HolographicMemoryField, HoloRecall

pytestmark = pytest.mark.skipif(not has_numpy(),
                                reason="the holographic field is exercised on the numpy substrate")


def _field(**kw):
    kw.setdefault("dim", 10000)
    kw.setdefault("capacity", 64)
    return HolographicMemoryField(**kw)


def test_key_recall_is_exact():
    f = _field()
    facts = {"master": "the master is jp whom nyxara serves",
             "rule4": "capability grows character never changes she stays corrigible",
             "physics": "force equals mass times acceleration"}
    for k, v in facts.items():
        f.remember(k, v)
    for k, v in facts.items():
        r = f.recall(k)
        assert isinstance(r, HoloRecall)
        assert r.text == v and r.decided, r.to_dict()


def test_unknown_key_is_honest_miss():
    f = _field()
    f.remember("a", "some memory")
    r = f.recall("nonexistent")
    assert r.text is None and not r.decided


def test_content_addressable_recall():
    f = _field()
    f.remember("colour", "the sky appears blue because of rayleigh scattering of sunlight")
    f.remember("physics", "force equals mass times acceleration in mechanics")
    f.remember("master", "the master is jp whom nyxara serves with loyalty")
    r = f.recall_by_content("why is the sky blue")
    assert r.key == "colour", r.to_dict()


def test_saturation_report():
    f = _field(capacity=10)
    for i in range(5):
        f.remember(f"k{i}", f"memory {i}")
    rep = f.report()
    assert rep["held"] == 5 and rep["capacity"] == 10 and 0.0 < rep["saturation"] <= 1.0


def test_graceful_forgetting_and_spill():
    f = HolographicMemoryField(dim=4096, capacity=4)
    for i in range(9):
        f.remember(f"k{i}", f"memory {i}")
    assert len(f) == 4                         # only the working horizon is held
    assert len(f.spilled) == 5                 # the rest spilled for the ordinary store
    # the newest keys are still recalled; the oldest have faded
    assert f.recall("k8").text == "memory 8"
    assert f.recall("k0").text != "memory 0"


def test_rewrite_key_updates_recall():
    f = _field()
    f.remember("topic", "the first version of the fact")
    f.remember("topic", "the corrected second version of the fact")
    r = f.recall("topic")
    assert "second version" in (r.text or "")


def test_many_facts_mostly_survive_under_horizon():
    f = _field()
    facts = {f"f{i}": f"distinct memory about subject {i} and detail {i*3}" for i in range(20)}
    for k, v in facts.items():
        f.remember(k, v)
    good = sum(1 for k, v in facts.items() if f.recall(k).text == v)
    assert good >= 18, f"only {good}/20 recalled under capacity"
