"""Tests for nyxara.memory.consolidation."""

from __future__ import annotations

import math
import time

import pytest

from nyxara.memory.consolidation import (
    Consolidator,
    ForgettingCurve,
    ReconsolidationMode,
)
from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.store import MemoryStore, MemoryType

_DAY = 86400.0


# -------------------- forgetting curve -------------------- #
def test_retention_at_zero_is_one():
    assert ForgettingCurve.retention(0, 10) == 1.0


def test_retention_at_one_stability_is_e_inverse():
    assert ForgettingCurve.retention(10, 10) == pytest.approx(math.exp(-1.0))


def test_retention_decays_with_time():
    assert ForgettingCurve.retention(5, 10) > ForgettingCurve.retention(20, 10)


def test_retention_zero_stability():
    assert ForgettingCurve.retention(1, 0) == 0.0


def test_next_stability_grows_and_caps():
    s = ForgettingCurve.next_stability(10)
    assert s > 10
    assert ForgettingCurve.next_stability(1e9) <= 3650.0


# -------------------- store fixture -------------------- #
def _store():
    store = MemoryStore()
    store.remember("The owner is JP.", mem_type=MemoryType.SEMANTIC,
                   provenance=Provenance(SourceType.OWNER, confidence=1.0),
                   importance=1.0, tags=["owner", "core"])
    for i in range(4):
        store.remember({"event": "ssh_login", "port": 22, "verdict": "blocked", "n": i},
                       mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.SENSOR, confidence=0.9),
                       importance=0.5, tags=["security", "login"])
    trivia = store.remember("idle nothing", mem_type=MemoryType.EPISODIC,
                            provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.3),
                            importance=0.1, tags=["idle"], half_life_days=1.0)
    return store, trivia


# -------------------- protection -------------------- #
def test_owner_memory_protected():
    store, _ = _store()
    con = Consolidator(store)
    owner = store.recall_one("owner is JP")
    assert con.is_protected(owner)


def test_high_importance_protected():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("important", importance=0.95)
    assert con.is_protected(rec)


def test_procedural_protected():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("how to do X", mem_type=MemoryType.PROCEDURAL, importance=0.1)
    assert con.is_protected(rec)


def test_trivia_not_protected():
    store, trivia = _store()
    con = Consolidator(store)
    assert not con.is_protected(trivia)


# -------------------- dream replay -------------------- #
def test_dream_replay_strengthens_stability():
    store, _ = _store()
    con = Consolidator(store, replay_top_k=10)
    now = time.time()
    target = store.recall_one("owner is JP")
    before = con.stability(target)
    con.dream_replay(now=now)
    after = con.stability(target)
    assert after > before


def test_dream_replay_touches_and_bumps_importance():
    store, _ = _store()
    con = Consolidator(store, replay_top_k=10)
    rec = store.by_type(MemoryType.EPISODIC)[0]
    imp_before = rec.importance
    acc_before = rec.access_count
    con.dream_replay()
    assert rec.importance >= imp_before
    assert rec.access_count > acc_before


def test_replay_priority_higher_for_important():
    store, trivia = _store()
    con = Consolidator(store)
    owner = store.recall_one("owner is JP")
    assert con.replay_priority(owner) > con.replay_priority(trivia)


# -------------------- compression -------------------- #
def test_compress_creates_abstraction():
    store, _ = _store()
    con = Consolidator(store, min_cluster=3, cluster_threshold=0.4)
    abstractions, pruned = con.compress()
    assert len(abstractions) >= 1
    gist = store.get(abstractions[0])
    assert gist.mem_type is MemoryType.SEMANTIC
    assert "port=22" in gist.content["summary"]


def test_compress_links_members():
    store, _ = _store()
    con = Consolidator(store, min_cluster=3, cluster_threshold=0.4)
    abstractions, _ = con.compress()
    members = store.neighbors(abstractions[0])
    assert len(members) >= 3


def test_compress_marks_consolidated():
    store, _ = _store()
    con = Consolidator(store, min_cluster=3, cluster_threshold=0.4)
    con.compress()
    logins = [r for r in store.by_type(MemoryType.EPISODIC) if "login" in r.tags]
    assert all(r.metadata.get("consolidated") for r in logins)
    # running again produces no new abstraction (already consolidated)
    abstractions2, _ = con.compress()
    assert abstractions2 == []


def test_compress_below_cluster_size_noop():
    store = MemoryStore()
    store.remember({"event": "x"}, mem_type=MemoryType.EPISODIC)
    con = Consolidator(store, min_cluster=3)
    assert con.compress() == ([], [])


def test_compress_with_pruning():
    store, _ = _store()
    con = Consolidator(store, min_cluster=3, cluster_threshold=0.4,
                       prune_consolidated=True, prune_importance=0.9)
    abstractions, pruned = con.compress()
    # low-importance duplicates pruned, one exemplar kept
    assert len(pruned) >= 1


# -------------------- forgetting -------------------- #
def test_forget_pass_removes_weak_future():
    store, trivia = _store()
    con = Consolidator(store, forget_threshold=0.05, grace_days=0.0)
    forgotten = con.forget_pass(now=time.time() + 90 * _DAY)
    assert trivia.mem_id in forgotten


def test_forget_pass_keeps_owner():
    store, _ = _store()
    con = Consolidator(store, grace_days=0.0)
    con.forget_pass(now=time.time() + 9999 * _DAY)
    assert store.recall_one("owner is JP") is not None


def test_grace_period_protects_new():
    store, trivia = _store()
    con = Consolidator(store, grace_days=365.0)
    forgotten = con.forget_pass(now=time.time() + 90 * _DAY)
    assert trivia.mem_id not in forgotten  # within grace


# -------------------- reconsolidation -------------------- #
def test_reconsolidate_update_changes_content():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("port 8080", mem_type=MemoryType.SEMANTIC, tags=["cfg"])
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.UPDATE,
                      new_content="port 9090")
    assert "9090" in rec.text()
    # re-embedded -> recall finds the new content
    assert store.recall_one("port 9090") is not None


def test_reconsolidate_strengthen():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("a fact", importance=0.4)
    s_before = con.stability(rec)
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.STRENGTHEN, delta=0.3)
    assert rec.importance == pytest.approx(0.7)
    assert con.stability(rec) > s_before


def test_reconsolidate_weaken():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("fading", importance=0.6)
    s_before = con.stability(rec)
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.WEAKEN, delta=0.3)
    assert rec.importance == pytest.approx(0.3)
    assert con.stability(rec) < s_before


def test_reconsolidate_merge_dict():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember({"a": 1}, mem_type=MemoryType.SEMANTIC)
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.MERGE, new_content={"b": 2})
    assert rec.content == {"a": 1, "b": 2}


def test_reconsolidate_corroborate():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("claim", provenance=Provenance(SourceType.WEB, confidence=0.7))
    now = time.time()
    before = rec.provenance.trust(now)
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.STRENGTHEN,
                      corroborate_with=SourceType.SENSOR, now=now)
    assert rec.provenance.trust(now) > before


def test_reconsolidate_records_history():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("x")
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.STRENGTHEN)
    con.reconsolidate(rec.mem_id, mode=ReconsolidationMode.WEAKEN)
    assert len(rec.metadata[con.HISTORY_KEY]) == 2


def test_reconsolidate_touches_access():
    store, _ = _store()
    con = Consolidator(store)
    rec = store.remember("x")
    before = rec.access_count
    con.reconsolidate(rec.mem_id)
    assert rec.access_count > before


# -------------------- full cycle -------------------- #
def test_run_cycle_report():
    store, trivia = _store()
    con = Consolidator(store, min_cluster=3, cluster_threshold=0.4, grace_days=0.0)
    rep = con.run_cycle(now=time.time() + 90 * _DAY)
    d = rep.to_dict()
    assert d["replayed"] >= 1
    assert d["abstractions"] >= 1
    assert "duration_s" in d


def test_retention_table():
    store, _ = _store()
    con = Consolidator(store)
    table = con.retention_table(now=time.time() + 10 * _DAY)
    assert table
    # sorted ascending by retention
    rets = [row["retention"] for row in table]
    assert rets == sorted(rets)
