"""Tests for Pillar D3 — interoception wired into the cognitive cycle (felt body → mood/report)."""

from __future__ import annotations

from nyxara.identity.interoception import Interoception, InteroceptiveSignals
from nyxara.kernel.orchestrator import NyxaraCore


def _stressed() -> Interoception:
    return Interoception(signals=InteroceptiveSignals(
        compute_load=0.95, memory_load=0.9, latency_ms=400, queue_depth=90,
        queue_capacity=100, error_rate=0.4, confidence=0.4, energy=0.3))


def test_core_builds_interoception_and_reports_its_body():
    core = NyxaraCore()
    assert core.interoception is not None
    rep = core.report()
    # honest self-report of the internal state (Rule 6)
    assert "comfort" in rep and "body" in rep
    assert isinstance(rep["body"], str) and rep["body"]


def test_idle_surfaces_a_strained_body():
    core = NyxaraCore()
    core.interoception = _stressed()
    m = core.idle_maintenance()
    assert m["comfort"] < 0.5
    assert m["sensation"] != "ease"
    assert "straining" in m["body"] or "tired" in m["body"] or "backlog" in m["body"]


def test_strained_body_colours_mood_negative():
    core = NyxaraCore()
    before = core.affect.mood.valence
    core.interoception = _stressed()
    core.idle_maintenance()
    # a body under real strain pulls mood down (interoception → affect)
    assert core.affect.mood.valence < before


def test_easy_body_does_not_fight_mood_relaxation():
    # a calm body (the default) must not inject a tone every tick — it lets affect relax
    core = NyxaraCore()
    core.affect.feel(0.9, 0.8, cause="elated")
    high = core.affect.mood.valence
    for _ in range(20):
        core.idle_maintenance()              # default calm body: comfort high, no push
    assert core.affect.mood.valence < high
