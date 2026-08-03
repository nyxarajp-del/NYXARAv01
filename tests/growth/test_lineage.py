"""Tests for the signed evolution ledger — growth/lineage.py (Part H)."""
from __future__ import annotations

from nyxara.growth.lineage import LineageLedger


def test_records_chain_and_verify():
    led = LineageLedger(secret=b"test-secret")
    led.record("code_edit", "optimize loop", certificate="proven-equivalent")
    led.record("model_promotion", "promote v3", detail={"version": 3})
    led.record("axiom", "discovered law L")
    assert len(led) == 3
    assert led.verify_chain() is True
    # each record commits to its parent's hash (a real chain)
    hist = led.history()
    assert hist[1].parent_hash == hist[0].record_hash
    assert hist[2].parent_hash == hist[1].record_hash


def test_tampering_is_detected():
    led = LineageLedger(secret=b"s")
    led.record("code_edit", "a")
    led.record("code_edit", "b")
    led.history()[0].summary = "tampered"  # mutate a sealed record
    assert led.verify_chain() is False


def test_revert_truncates_and_records_and_runs_rollback():
    led = LineageLedger(secret=b"s")
    led.record("code_edit", "gen0")
    led.record("code_edit", "gen1")
    led.record("code_edit", "gen2")

    undone = []
    ok = led.revert_to(0, rollback=lambda rec: undone.append(rec.summary) or True)
    assert ok is True
    # rolled back gen2 then gen1 (newest first)
    assert undone == ["gen2", "gen1"]
    # ledger now holds gen0 + a signed 'revert' record, and still verifies
    assert led.verify_chain() is True
    kinds = [r.kind for r in led.history()]
    assert kinds == ["code_edit", "revert"]


def test_revert_is_fail_closed_when_rollback_fails():
    led = LineageLedger(secret=b"s")
    led.record("code_edit", "gen0")
    led.record("code_edit", "gen1")
    before = len(led)
    ok = led.revert_to(0, rollback=lambda rec: False)  # rollback refuses
    assert ok is False and len(led) == before  # unchanged


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "lineage.json"
    led = LineageLedger(path=p, secret=b"s")
    led.record("code_edit", "gen0", certificate="c0")
    led.record("model_promotion", "gen1")
    reloaded = LineageLedger(path=p, secret=b"s")
    assert len(reloaded) == 2 and reloaded.verify_chain() is True
    assert reloaded.history()[0].certificate == "c0"
