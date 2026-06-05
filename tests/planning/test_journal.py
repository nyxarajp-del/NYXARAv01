"""Tests for nyxara.planning.journal."""

from __future__ import annotations

import os
import tempfile

import pytest

from nyxara.kernel.errors import InvariantViolation
from nyxara.planning.journal import ActionStatus, EntryKind, Journal, JournalEntry


def _journal_with_action():
    j = Journal()
    aid = j.record_action("do x", goal="g", rationale="because", autonomous=True,
                          confidence=0.9, reversibility=0.8)
    return j, aid


# -------------------- recording -------------------- #
def test_record_action_returns_id():
    j, aid = _journal_with_action()
    assert isinstance(aid, str)
    assert len(j) == 1
    assert j.latest_status(aid) is ActionStatus.EXECUTED


def test_record_action_payload():
    j, aid = _journal_with_action()
    entry = j.actions()[0]
    assert entry.payload["action"] == "do x"
    assert entry.payload["goal"] == "g"
    assert entry.payload["autonomous"] is True


def test_record_outcome():
    j, aid = _journal_with_action()
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED, outcome={"k": 1})
    assert j.latest_status(aid) is ActionStatus.SUCCEEDED
    assert len(j) == 2


def test_record_outcome_orphan_refused():
    j = Journal()
    with pytest.raises(InvariantViolation):
        j.record_outcome("nope", status=ActionStatus.SUCCEEDED)


def test_latest_status_reflects_latest_outcome():
    j, aid = _journal_with_action()
    j.record_outcome(aid, status=ActionStatus.FAILED)
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED)  # a retry succeeded
    assert j.latest_status(aid) is ActionStatus.SUCCEEDED


def test_note():
    j = Journal()
    j.note("a thought")
    assert j.entries()[0].kind is EntryKind.NOTE


# -------------------- hash chain / immutability -------------------- #
def test_chain_advances():
    j = Journal()
    h0 = j.head
    j.record_action("x")
    assert j.head != h0


def test_verify_intact():
    j, aid = _journal_with_action()
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED)
    assert j.verify() is True


def test_verify_detects_payload_tamper():
    j, aid = _journal_with_action()
    j._entries[0].payload["action"] = "tampered"
    with pytest.raises(InvariantViolation):
        j.verify()


def test_verify_detects_reorder():
    j = Journal()
    j.record_action("a")
    j.record_action("b")
    j._entries[0], j._entries[1] = j._entries[1], j._entries[0]
    with pytest.raises(InvariantViolation):
        j.verify()


def test_verify_detects_digest_tamper():
    j, aid = _journal_with_action()
    j._entries[0].digest = "deadbeef"
    with pytest.raises(InvariantViolation):
        j.verify()


def test_entry_compute_digest_stable():
    e = JournalEntry(seq=0, kind=EntryKind.ACTION, action_id="x", payload={"a": 1})
    assert e.compute_digest() == e.compute_digest()


# -------------------- queries -------------------- #
def test_actions_filter():
    j, aid = _journal_with_action()
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED)
    assert len(j.actions()) == 1  # outcome is not an action


def test_by_action():
    j, aid = _journal_with_action()
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED)
    assert len(j.by_action(aid)) == 2  # action + outcome


def test_by_goal():
    j = Journal()
    j.record_action("a", goal="protect")
    j.record_action("b", goal="protect")
    j.record_action("c", goal="learn")
    assert len(j.by_goal("protect")) == 2


def test_by_status_and_failures():
    j = Journal()
    a1 = j.record_action("ok")
    a2 = j.record_action("bad")
    j.record_outcome(a1, status=ActionStatus.SUCCEEDED)
    j.record_outcome(a2, status=ActionStatus.FAILED)
    assert a1 in j.by_status(ActionStatus.SUCCEEDED)
    assert a2 in j.failures()


def test_autonomous_filter():
    j = Journal()
    j.record_action("auto", autonomous=True)
    j.record_action("commanded", autonomous=False)
    assert len(j.autonomous_actions()) == 1


def test_recent():
    j = Journal()
    for i in range(5):
        j.record_action(f"a{i}")
    assert len(j.recent(3)) == 3


def test_latest_status_unknown():
    assert Journal().latest_status("nope") is None


# -------------------- persistence -------------------- #
def test_save_and_load():
    j = Journal()
    aid = j.record_action("backup", goal="resilience", autonomous=True)
    j.record_outcome(aid, status=ActionStatus.SUCCEEDED)
    path = os.path.join(tempfile.gettempdir(), "nyx_journal_test.jsonl")
    try:
        j.save(path)
        loaded = Journal.load(path)
        assert loaded.verify()
        assert len(loaded) == len(j)
        assert loaded.latest_status(aid) is ActionStatus.SUCCEEDED
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_load_preserves_chain_head():
    j = Journal()
    j.record_action("x")
    path = os.path.join(tempfile.gettempdir(), "nyx_journal_head.jsonl")
    try:
        j.save(path)
        loaded = Journal.load(path)
        assert loaded.head == j.head
        # can continue appending with a valid chain
        loaded.record_action("y")
        assert loaded.verify()
    finally:
        if os.path.exists(path):
            os.remove(path)


# -------------------- report -------------------- #
def test_report():
    j = Journal()
    a1 = j.record_action("a", autonomous=True)
    j.record_outcome(a1, status=ActionStatus.SUCCEEDED)
    j.record_action("b", autonomous=False)
    r = j.report()
    assert r["total_actions"] == 2
    assert r["autonomous"] == 1
    assert r["by_status"]["succeeded"] == 1


def test_len():
    j = Journal()
    j.record_action("a")
    j.note("n")
    assert len(j) == 2
