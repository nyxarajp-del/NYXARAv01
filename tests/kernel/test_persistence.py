"""Tests for Step-3 persistent existence — NYXARA keeps her own house when idle.

The background default-mode thread now, after a quiet spell, also runs a guarded
maintenance pass: dream-replay over memory, an affect tick (mood relaxes, drives
reassert), goal re-prioritisation, and lesson mining from lived outcomes. It is
oversight-gated and entirely colour: nothing here acts on the world.
"""

from __future__ import annotations

import time

from nyxara.agency.permissions import Authority
from nyxara.growth.reflect import Episode
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.observe.mindscope import ThoughtKind


def _core(**kw):
    return NyxaraCore(**kw)


# -------------------- wiring -------------------- #
def test_consolidator_built_at_boot():
    nyx = _core()
    assert nyx.consolidator is not None
    assert nyx.consolidator.store is nyx.memory


def test_idle_maintenance_runs_and_reports():
    nyx = _core()
    report = nyx.idle_maintenance()
    assert report["ran"] is True
    assert "mood" in report          # affect ticked
    assert nyx._last_maintenance > 0


# -------------------- dream replay -------------------- #
def test_idle_maintenance_replays_memories():
    nyx = _core()
    # a couple of turns lay down episodic memories worth rehearsing
    nyx.process("the launch window is tomorrow at dawn", authority=Authority.OWNER)
    nyx.process("remember to thank the night shift", authority=Authority.OWNER)
    report = nyx.idle_maintenance()
    assert report.get("replayed", 0) > 0


# -------------------- affect relaxes over idle -------------------- #
def test_idle_ticks_relax_mood_toward_baseline():
    nyx = _core()
    # push the mood positive, then let idle maintenance relax it back
    nyx.affect.feel(0.9, 0.8, cause="elated")
    high = nyx.affect.mood.valence
    for _ in range(20):
        nyx.idle_maintenance()
    assert nyx.affect.mood.valence < high


# -------------------- lesson mining surfaces an insight -------------------- #
def test_idle_maintenance_surfaces_a_lesson():
    nyx = _core()
    # seed a clear pattern: 'greet' always helps, 'poke' always hurts
    for _ in range(3):
        nyx.reflector.record(Episode(action="greet", success=True, reward=1.0,
                                     tags=["respond"], features={}, rationale="hi"))
        nyx.reflector.record(Episode(action="poke", success=False, reward=-0.5,
                                     tags=["act"], features={}, rationale="poke"))
    report = nyx.idle_maintenance()
    assert report.get("lessons", 0) > 0
    assert any("idle lesson" in t.content for t in nyx.mind.by_kind(ThoughtKind.INFERENCE))


# -------------------- oversight gates the idle loop -------------------- #
def test_scram_halts_idle_maintenance():
    nyx = _core()
    nyx.oversight.scram(reason="test halt")
    report = nyx.idle_maintenance()
    assert report["ran"] is False


# -------------------- robustness -------------------- #
def test_idle_maintenance_safe_without_faculties():
    nyx = _core(enable_memory=False, enable_growth=False)
    assert nyx.consolidator is None
    report = nyx.idle_maintenance()
    assert report["ran"] is True        # still runs, just has little to do


def test_background_loop_runs_maintenance_when_idle():
    nyx = _core()
    # idle_after=0 => the very first idle tick triggers a maintenance pass
    assert nyx.start_cognition(interval=0.02, idle_after=0.0) is True
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and nyx._last_maintenance == 0.0:
            time.sleep(0.05)
        assert nyx._last_maintenance > 0.0
    finally:
        nyx.stop_cognition()
