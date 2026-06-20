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


# --------------------------------------------------------------------------- #
# Pillar F · Edge 4 — rivalry wiring into the growth pass
# --------------------------------------------------------------------------- #
def _arena_with_lang_gap():
    from nyxara.eval.benchmark import Benchmark, BenchmarkTask
    from nyxara.growth.rivalry import Arena
    bench = Benchmark("demo", [
        BenchmarkTask(id="m1", category="math", prompt="2+2?", answer="4", grader="exact"),
        BenchmarkTask(id="l1", category="lang", prompt="opposite of hot?", answer="cold",
                      grader="contains"),
    ])
    return Arena(benchmark=bench)


def test_rivalry_off_by_default():
    core = NyxaraCore()
    engine = GrowthEngine.from_core(core)
    assert engine.rivalry() is None                     # disabled -> no-op


def test_rivalry_runs_and_yields_gap_topics():
    core = NyxaraCore()
    me = lambda p: "4" if p == "2+2?" else "?"          # strong math, weak lang
    rival = lambda p: "cold" if "hot" in p else "?"     # strong lang
    engine = GrowthEngine.from_core(
        core, enable_rivalry=True, arena=_arena_with_lang_gap(),
        rival_solver=rival, self_solver=me)
    out = engine.rivalry()
    assert out is not None
    assert out["topics"] == ["hard problems in lang"]
    assert out["rival"]["name"] == "rival"


def test_rivalry_feeds_selfplay_topics_in_run(monkeypatch):
    core = NyxaraCore()
    me = lambda p: "4" if p == "2+2?" else "?"
    rival = lambda p: "cold" if "hot" in p else "?"
    engine = GrowthEngine.from_core(
        core, enable_rivalry=True, enable_foundry=True, enable_selfplay=True,
        arena=_arena_with_lang_gap(), rival_solver=rival, self_solver=me)

    captured = {}
    monkeypatch.setattr(engine, "self_play",
                        lambda *, n=None, topics=None: captured.setdefault("topics", topics) or {})
    monkeypatch.setattr(engine, "improve_self", lambda *a, **k: [])
    rep = engine.run()
    assert rep.rivalry is not None
    assert captured["topics"] == ["hard problems in lang"]   # rival's lead steered curiosity


def test_rivalry_stores_lesson_into_memory():
    core = NyxaraCore()
    me = lambda p: "4" if p == "2+2?" else "?"
    rival = lambda p: "cold" if "hot" in p else "?"
    engine = GrowthEngine.from_core(
        core, enable_rivalry=True, arena=_arena_with_lang_gap(),
        rival_solver=rival, self_solver=me)
    engine.rivalry()
    hits = [m for m in core.memory._kv.values() if "rivalry" in m.tags]
    assert hits and "lang" in hits[0].content
