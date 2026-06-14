"""Tests for nyxara.kernel.replay."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from nyxara.kernel.replay import (
    Mode,
    Recorder,
    ReplayDivergence,
    StepKind,
)


def test_record_then_replay_random_matches():
    rec = Recorder(mode=Mode.RECORD, seed=1)
    with rec:
        a = [rec.random.random() for _ in range(5)]
        b = [rec.random.randint(0, 10) for _ in range(5)]

    rep = Recorder(mode=Mode.REPLAY, seed=1)
    rep.entries = rec.entries
    with rep:
        a2 = [rep.random.random() for _ in range(5)]
        b2 = [rep.random.randint(0, 10) for _ in range(5)]
    assert a == a2
    assert b == b2


def test_clock_is_deterministic_on_replay():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        t = rec.now()
    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with rep:
        assert rep.now() == t


def test_capture_not_executed_on_replay():
    calls = {"n": 0}

    def ext():
        calls["n"] += 1
        return calls["n"]

    rec = Recorder(mode=Mode.RECORD)
    with rec:
        v = rec.capture("ext", ext)
    assert calls["n"] == 1 and v == 1

    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with rep:
        v2 = rep.capture("ext", lambda: (_ for _ in ()).throw(AssertionError("ran!")))
    assert v2 == 1
    assert calls["n"] == 1  # not re-executed


def test_async_capture():
    calls = {"n": 0}

    async def ext():
        calls["n"] += 1
        return "async-result"

    async def record():
        rec = Recorder(mode=Mode.RECORD)
        with rec:
            v = await rec.acapture("a", ext)
        return rec, v

    rec, v = asyncio.run(record())
    assert v == "async-result" and calls["n"] == 1

    async def replay():
        rep = Recorder(mode=Mode.REPLAY)
        rep.entries = rec.entries
        with rep:
            return await rep.acapture("a", lambda: (_ for _ in ()).throw(AssertionError()))

    v2 = asyncio.run(replay())
    assert v2 == "async-result"
    assert calls["n"] == 1


def test_off_mode_is_passthrough():
    calls = {"n": 0}
    rec = Recorder(mode=Mode.OFF, seed=1)
    with rec:
        rec.random.random()
        v = rec.capture("x", lambda: calls.__setitem__("n", calls["n"] + 1) or 7)
    assert v == 7
    assert calls["n"] == 1
    assert rec.entries == []


def test_steps_record_enter_exit():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        with rec.step("outer"):
            with rec.step("inner"):
                pass
    kinds = [(e.kind, e.label) for e in rec.entries if e.kind in
             (StepKind.STEP_ENTER, StepKind.STEP_EXIT)]
    assert (StepKind.STEP_ENTER, "outer") in kinds
    assert (StepKind.STEP_EXIT, "inner") in kinds
    # nesting order: outer enter, inner enter, inner exit, outer exit
    labels = [l for _, l in kinds]
    assert labels == ["outer", "inner", "inner", "outer"]


def test_full_replay_reproduces_sequence():
    def run(rec):
        out = []
        with rec:
            with rec.step("s"):
                out.append(rec.random.uniform(0, 1))
                out.append(rec.capture("io", lambda: 99))
                out.append(rec.random.choice([10, 20, 30]))
            rec.decision("d", {"x": 1})
        return out

    rec = Recorder(mode=Mode.RECORD, seed=5)
    recorded = run(rec)

    rep = Recorder(mode=Mode.REPLAY, seed=5)
    rep.entries = rec.entries
    replayed = run_replay(rep)
    assert recorded == replayed


def run_replay(rep):
    out = []
    with rep:
        with rep.step("s"):
            out.append(rep.random.uniform(0, 1))
            out.append(rep.capture("io", lambda: -1))
            out.append(rep.random.choice([10, 20, 30]))
        rep.decision("d", {"x": 1})
    return out


def test_divergence_on_wrong_label():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.capture("expected", lambda: 1)

    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with pytest.raises(ReplayDivergence):
        with rep:
            rep.capture("DIFFERENT", lambda: 1)


def test_divergence_on_wrong_kind():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.capture("x", lambda: 1)

    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with pytest.raises(ReplayDivergence):
        with rep:
            rep.now()  # journal expects a CAPTURE/marker, got CLOCK


def test_replay_past_end_raises():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.capture("only", lambda: 1)

    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with pytest.raises(ReplayDivergence):  # ReplayExhausted is a subclass
        with rep:
            rep.capture("only", lambda: 1)
            rep.now()      # nothing recorded here -> diverges/exhausts
            rep.now()


def test_hash_chain_intact():
    rec = Recorder(mode=Mode.RECORD, seed=1)
    with rec:
        for i in range(5):
            rec.capture(f"c{i}", lambda i=i: i)
    assert rec.verify_chain() is True


def test_hash_chain_detects_tamper():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.capture("a", lambda: 1)
        rec.capture("b", lambda: 2)
    rec.entries[1].value = "tampered"
    with pytest.raises(ReplayDivergence):
        rec.verify_chain()


def test_hash_chain_detects_reorder():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.capture("a", lambda: 1)
        rec.capture("b", lambda: 2)
    rec.entries[1], rec.entries[2] = rec.entries[2], rec.entries[1]
    with pytest.raises(ReplayDivergence):
        rec.verify_chain()


def test_save_and_load_roundtrip():
    rec = Recorder(mode=Mode.RECORD, seed=11)
    with rec:
        rec.random.random()
        rec.capture("io", lambda: {"k": "v"})
        rec.decision("d", [1, 2, 3])

    path = os.path.join(tempfile.gettempdir(), "nyx_replay_test.jsonl")
    try:
        rec.save(path)
        loaded = Recorder.load(path, mode=Mode.REPLAY)
        assert loaded.verify_chain()
        assert len(loaded.entries) == len(rec.entries)
        assert loaded.meta.seed == 11
        # replay reproduces
        with loaded:
            loaded.random.random()
            assert loaded.capture("io", lambda: None) == {"k": "v"}
            assert loaded.decision("d", [1, 2, 3]) == [1, 2, 3]
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_input_and_output_recorded():
    rec = Recorder(mode=Mode.RECORD)
    with rec:
        rec.input("sensor", {"temp": 20})
        rec.output("reply", "hello")
    rep = Recorder(mode=Mode.REPLAY)
    rep.entries = rec.entries
    with rep:
        assert rep.input("sensor", {"temp": 999}) == {"temp": 20}  # replay returns recorded
        assert rep.output("reply", "ignored") == "hello"


def test_summary_keys():
    rec = Recorder(mode=Mode.RECORD, seed=1)
    with rec:
        rec.random.random()
        rec.capture("io", lambda: 1)
    s = rec.summary()
    assert s["mode"] == "record"
    assert s["entries"] >= 2
    assert "random" in s["by_kind"] and "capture" in s["by_kind"]


def test_shuffle_deterministic():
    rec = Recorder(mode=Mode.RECORD, seed=3)
    data = [1, 2, 3, 4, 5]
    with rec:
        d = data[:]
        rec.random.shuffle(d)
    rep = Recorder(mode=Mode.REPLAY, seed=999)  # different seed, must still match
    rep.entries = rec.entries
    with rep:
        d2 = data[:]
        rep.random.shuffle(d2)
    assert d == d2
