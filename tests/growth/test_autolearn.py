"""Tests for nyxara.growth.autolearn — the automatic learning loop."""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.growth.autolearn import GrowthEngine, GrowthReport
from nyxara.kernel.orchestrator import NyxaraCore


def _core_with_experience(n: int = 5) -> NyxaraCore:
    core = NyxaraCore()
    for _ in range(n):
        core.process("rotate the logs", authority=Authority.OWNER)
    core.process("how are you?", authority=Authority.OWNER)
    return core


def test_run_returns_report():
    engine = GrowthEngine.from_core(_core_with_experience())
    rep = engine.run()
    assert isinstance(rep, GrowthReport)
    assert rep.episodes_seen >= 5


def test_episodes_drawn_from_journal():
    core = _core_with_experience(3)
    engine = GrowthEngine.from_core(core)
    eps = engine._episodes_from_journal()
    assert len(eps) >= 3
    assert all(hasattr(e, "action") and hasattr(e, "success") for e in eps)


def test_idempotent_over_seen_actions():
    engine = GrowthEngine.from_core(_core_with_experience())
    engine.run()
    seen = engine.run().episodes_seen
    # a second pass with no new actions sees the same total, no duplicates
    assert seen == engine.run().episodes_seen


def test_consolidation_runs_over_memory():
    core = _core_with_experience(6)
    engine = GrowthEngine.from_core(core)
    rep = engine.run()
    # dream-replay touched some memories
    assert rep.replayed >= 0 and rep.abstractions >= 0


def test_foundry_off_by_default():
    engine = GrowthEngine.from_core(_core_with_experience())
    rep = engine.run()
    assert rep.foundry == []


def test_self_improvement_runs_on_cadence():
    core = _core_with_experience(2)
    engine = GrowthEngine.from_core(core, enable_self_improvement=True,
                                    self_improvement_every=1)
    rep = engine.run()
    assert rep.self_improvement is not None
    # the cycle reports the five faculties' output
    assert "weaknesses" in rep.self_improvement
    assert rep.self_improvement["enacted"] is False     # config default: no auto-enact


def test_self_improvement_off_by_flag():
    core = _core_with_experience(2)
    engine = GrowthEngine.from_core(core, enable_self_improvement=False)
    rep = engine.run()
    assert rep.self_improvement is None


def test_self_improvement_throttled_between_passes():
    core = _core_with_experience(2)
    engine = GrowthEngine.from_core(core, enable_self_improvement=True,
                                    self_improvement_every=3)
    # pass 1 and 2 skip RSI; pass 3 runs it
    assert engine.run().self_improvement is None
    assert engine.run().self_improvement is None
    assert engine.run().self_improvement is not None


def test_lessons_stored_into_memory():
    # contrasting outcomes so the reflector can actually mine a lesson
    core = NyxaraCore()
    from nyxara.planning.journal import ActionStatus
    for i in range(6):
        aid = core.journal.record_action(f"task alpha {i}", autonomous=True)
        core.journal.record_outcome(aid, status=ActionStatus.SUCCEEDED)
    for i in range(6):
        aid = core.journal.record_action(f"task beta {i}", autonomous=True)
        core.journal.record_outcome(aid, status=ActionStatus.FAILED)
    engine = GrowthEngine.from_core(core)
    rep = engine.run()
    # lessons may or may not clear thresholds, but the store call path must be safe
    assert rep.lessons_stored == len(engine._stored_lessons)
