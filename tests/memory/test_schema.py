"""Tests for nyxara.memory.schema."""

from __future__ import annotations

import pytest

from nyxara.memory.provenance import Provenance, SourceType
from nyxara.memory.schema import (
    Schema,
    SchemaLibrary,
    Slot,
    episode_from_record,
)
from nyxara.memory.store import MemoryRecord, MemoryType


# -------------------- Slot -------------------- #
def test_slot_modal_and_counts():
    s = Slot("x")
    for v in ["a", "a", "a", "b"]:
        s.observe(v)
    val, prob = s.modal()
    assert val == "a"
    assert prob == pytest.approx(0.75)
    assert s.present_count == 4
    assert s.distinct() == 2


def test_slot_constant():
    s = Slot("x")
    s.observe(5)
    s.observe(5)
    assert s.is_constant()
    assert s.variability() == 0.0


def test_slot_variability_increases():
    s = Slot("x")
    for v in ["a", "b", "c", "d"]:
        s.observe(v)
    assert s.variability() == pytest.approx(1.0)  # uniform -> max entropy


def test_slot_numeric_stats():
    s = Slot("n")
    for v in [10, 20, 30]:
        s.observe(v)
    st = s.numeric_stats()
    assert st["mean"] == pytest.approx(20.0)
    assert st["min"] == 10 and st["max"] == 30


def test_slot_fill_rate():
    s = Slot("x")
    s.observe("a")
    s.observe("b")
    assert s.fill_rate(4) == 0.5


def test_slot_supports():
    s = Slot("x")
    s.observe("a")
    assert s.supports("a")
    assert not s.supports("z")


# -------------------- Schema induction -------------------- #
def _logins():
    return [
        {"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"},
        {"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"},
        {"event": "ssh", "port": 22, "time": "day", "verdict": "blocked"},
        {"event": "ssh", "port": 2222, "time": "night", "verdict": "allowed"},
    ]


def test_induce_builds_slots():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    assert sch.support == 4
    assert set(sch.slots) == {"event", "port", "time", "verdict"}
    assert sch.constant_slots()["event"] == "ssh"


def test_induce_empty_raises():
    lib = SchemaLibrary()
    with pytest.raises(ValueError):
        lib.induce("x", [])


def test_modal_defaults():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    assert sch.predict_slot("port")[0] == 22
    assert sch.predict_slot("time")[0] == "night"
    assert sch.predict_slot("verdict")[0] == "blocked"


def test_predict_unknown_slot():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    assert sch.predict_slot("nonexistent") == (None, 0.0)


def test_required_vs_optional():
    lib = SchemaLibrary()
    eps = [{"a": 1, "b": 2}, {"a": 1}, {"a": 1}, {"a": 1}]  # 'a' always, 'b' rare
    sch = lib.induce("s", eps)
    assert "a" in sch.required_slots(0.8)
    assert "b" in sch.optional_slots(0.8)


# -------------------- matching & anomaly -------------------- #
def test_match_typical_higher_than_weird():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    typical = {"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"}
    weird = {"event": "ftp", "size": 999, "dest": "x"}
    assert sch.match(typical) > sch.match(weird)


def test_anomaly_score():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    typical = {"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"}
    weird = {"event": "ftp", "size": 999}
    assert sch.anomaly_score(weird) > sch.anomaly_score(typical)


def test_empty_schema_match_zero():
    sch = Schema(name="empty")
    assert sch.match({"a": 1}) == 0.0


# -------------------- slot filling -------------------- #
def test_fill_predicts_missing_slots():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    filled = sch.fill({"event": "ssh"})
    assert filled.values["port"] == 22
    assert filled.values["verdict"] == "blocked"
    assert "port" in filled.predicted


def test_fill_does_not_override_given():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    filled = sch.fill({"event": "ssh", "port": 9999})
    assert filled.values["port"] == 9999  # provided value kept
    assert "port" not in filled.predicted


def test_fill_respects_threshold():
    lib = SchemaLibrary()
    eps = [{"a": 1, "b": 2}, {"a": 1}, {"a": 1}, {"a": 1}]  # b fill_rate 0.25
    sch = lib.induce("s", eps)
    filled = sch.fill({}, fill_threshold=0.5)
    assert "a" in filled.values  # high fill rate predicted
    assert "b" not in filled.values  # below threshold, not predicted


# -------------------- clustering -------------------- #
def test_cluster_induce_separates_groups():
    lib = SchemaLibrary(match_threshold=0.5)
    mixed = _logins() + [
        {"event": "coffee", "drink": "espresso", "time": "morning"},
        {"event": "coffee", "drink": "latte", "time": "morning"},
    ]
    clusters = lib.cluster_induce(mixed)
    assert len(clusters) >= 2
    total = sum(c.support for c in clusters)
    assert total == len(mixed)


# -------------------- integration & retrieval -------------------- #
def test_best_match_finds_schema():
    lib = SchemaLibrary()
    lib.induce("login", _logins())
    best, score = lib.best_match({"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"})
    assert best is not None
    assert score > 0.5


def test_integrate_folds_into_existing():
    lib = SchemaLibrary()
    sch = lib.induce("login", _logins())
    before = sch.support
    lib.integrate({"event": "ssh", "port": 22, "time": "night", "verdict": "blocked"})
    assert sch.support == before + 1
    assert len(lib) == 1  # no new schema spawned


def test_integrate_spawns_new_for_novel():
    lib = SchemaLibrary(match_threshold=0.6)
    lib.induce("login", _logins())
    lib.integrate({"event": "earthquake", "magnitude": 7, "region": "pacific"})
    assert len(lib) == 2


def test_library_fill():
    lib = SchemaLibrary()
    lib.induce("login", _logins())
    filled = lib.fill({"event": "ssh", "port": 22})
    assert filled is not None
    assert filled.values["verdict"] == "blocked"


def test_library_fill_no_match_returns_none():
    lib = SchemaLibrary(match_threshold=0.9)
    lib.induce("login", _logins())
    assert lib.fill({"totally": "different", "wild": "stuff"}) is None


def test_stats():
    lib = SchemaLibrary()
    lib.induce("login", _logins())
    s = lib.stats()
    assert s["schemas"] == 1
    assert s["total_support"] == 4


# -------------------- from MemoryRecord -------------------- #
def test_episode_from_record_dict_content():
    rec = MemoryRecord(content={"event": "x", "n": 1}, mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.SENSOR), tags=frozenset({"t"}))
    ep = episode_from_record(rec)
    assert ep["event"] == "x"
    assert ep["_tags"] == ("t",)


def test_episode_from_record_text_content():
    rec = MemoryRecord(content="hello world", mem_type=MemoryType.EPISODIC,
                       provenance=Provenance(SourceType.SENSOR))
    ep = episode_from_record(rec)
    assert ep["_text"] == "hello world"
