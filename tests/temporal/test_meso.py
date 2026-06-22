"""Tests for nyxara.temporal.meso (Layer 2 — turn observation)."""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.temporal.clock import VirtualClock
from nyxara.temporal.meso import MesoLayer, looks_like_code


def _result(disp, text, tool=None, conf=0.7):
    return SimpleNamespace(disposition=disp, response=text, tool=tool,
                           candidate=SimpleNamespace(confidence=conf, text=text))


def _meso() -> MesoLayer:
    return MesoLayer(clock=VirtualClock(start=1000.0))


# -------------------- code detection -------------------- #
def test_looks_like_code():
    assert looks_like_code("def f():\n    return 1")
    assert looks_like_code("```python\nx = 1\n```")
    assert not looks_like_code("just a plain english sentence")
    assert not looks_like_code("")


# -------------------- observation -------------------- #
def test_observe_records_turn_features():
    m = _meso()
    obs = m.observe("write a function", _result("act", "def add(a, b): return a + b"),
                    latency_s=0.5, authority="owner")
    assert obs.produced_code and obs.disposition == "act"
    assert obs.authority == "owner" and obs.latency_s == 0.5
    assert m.observed == 1


def test_observe_handles_enum_disposition():
    # duck-typed CycleResult whose disposition has a .value (like the real Disposition enum)
    disp = SimpleNamespace(value="escalate")
    m = _meso()
    obs = m.observe("do risky thing", _result(disp, "I should ask first."))
    assert obs.disposition == "escalate"


# -------------------- aggregation -------------------- #
def test_aggregate_counts():
    m = _meso()
    m.observe("a", _result("act", "def x(): pass", tool="shell"))
    m.observe("b", _result("act", "hello"))
    m.observe("c", _result("refuse", "no"))
    m.observe("d", _result("escalate", "ask"))
    agg = m.aggregate()
    assert agg.n_turns == 4
    assert agg.acted == 2 and agg.refused == 1 and agg.escalated == 1
    assert agg.code_turns == 1 and agg.tool_calls == 1


def test_empty_aggregate_is_safe():
    m = _meso()
    agg = m.aggregate()
    assert agg.n_turns == 0


def test_report_shape():
    m = _meso()
    m.observe("a", _result("act", "ok"))
    rep = m.report()
    assert rep["name"] == "meso" and rep["ticks"] == 1
