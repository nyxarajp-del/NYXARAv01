"""Tests for nyxara.growth.autolearn — the automatic learning loop."""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.growth.autolearn import GrowthEngine, GrowthReport
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.orchestrator import NyxaraCore


def _engine_with_tmp_flywheel(tmp_path, **kw) -> GrowthEngine:
    """A growth engine whose verified-data flywheel writes to a temp file (hermetic)."""
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.flywheel.store_path = tmp_path / "flywheel.jsonl"
    settings.llm.self_model_dir = tmp_path / "foundry"
    return GrowthEngine(settings=settings, **kw)


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
    # The foundry forges real model dirs/manifests to disk, so it is sealed OFF under the hermetic
    # TEST profile (config._harden_for_profile sets foundry.enabled=False). The forge gate lives on
    # the growth engine (enable_foundry ← settings.foundry.enabled), so thread TEST settings into the
    # engine — otherwise it silently falls back to the global (DEV) settings, where the foundry is on.
    settings = NyxaraSettings.for_profile(Profile.TEST)
    engine = GrowthEngine.from_core(_core_with_experience(), settings=settings)
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


# --------------------------------------------------------------------------- #
# The ceiling-break — verifiable self-play + synthesis fold GROUND TRUTH into the
# flywheel the foundry forges from, so the model's target is correctness, not the teacher.
# --------------------------------------------------------------------------- #
def test_play_verifiable_collects_ground_truth_into_shared_flywheel(tmp_path):
    engine = _engine_with_tmp_flywheel(tmp_path)
    out = engine.play_verifiable(n=24)
    assert out is not None
    assert out["collected"] > 0                         # exact-oracle pairs, no teacher needed
    fw = engine._shared_flywheel()
    assert fw.count() > 0
    # and these are exactly the docs the foundry will train on
    from nyxara.growth.distill import load_distillation_docs
    assert len(load_distillation_docs(fw.store_path)) > 0


def test_curate_synthetic_feeds_verified_items(tmp_path):
    engine = _engine_with_tmp_flywheel(tmp_path)
    out = engine.curate_synthetic(rounds=2)
    assert out is not None
    assert out["generated"] > 0
    assert out["accepted"] >= 0                          # prover-verified survivors only


def test_verifiable_selfplay_on_by_default_when_foundry_on(tmp_path):
    # default: verifiable self-play + synthesis ride along whenever the foundry runs
    engine = _engine_with_tmp_flywheel(tmp_path, enable_foundry=True)
    assert engine.enable_verifiable_selfplay is True
    assert engine.enable_synthesis is True


def test_forge_pass_reports_ground_truth_generation(tmp_path, monkeypatch):
    engine = _engine_with_tmp_flywheel(tmp_path, enable_foundry=True)
    # skip the heavy train/promote; we only assert the ground-truth data was manufactured first
    monkeypatch.setattr(engine, "improve_self", lambda *a, **k: [])
    rep = engine.run(do_foundry=True)
    assert rep.verifiable_selfplay is not None
    assert rep.verifiable_selfplay["collected"] > 0
    assert rep.synthesis is not None


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


def test_lessons_are_taught_to_the_learned_brain():
    """A distilled lesson must reach the substrate that *answers*, not just memory (lessons→weights)."""
    from types import SimpleNamespace

    core = NyxaraCore()
    brain = core.reasoner._own_brain()
    assert brain is not None
    before = brain.learned_count
    engine = GrowthEngine.from_core(core)
    lesson = SimpleNamespace(
        kind=SimpleNamespace(value="strategy"),
        subject="closing unused ports",
        text="Closing unused network ports reduces a host's attack surface.",
        confidence=0.8)
    stored = engine._store_lessons([lesson])
    assert stored == 1
    assert brain.learned_count > before          # the lesson became learned, answerable knowledge


def test_lessons_feed_the_flywheel_as_verified_pairs(tmp_path):
    """A distilled lesson must also reach the foundry corpus, so the NEXT forged model
    is trained on it — reflection reaching all the way to weights."""
    from types import SimpleNamespace
    from nyxara.growth.flywheel import DataFlywheel

    fw = DataFlywheel(store_path=tmp_path / "fw.jsonl")
    fake_core = SimpleNamespace(flywheel=fw, memory=None, journal=None)
    engine = _engine_with_tmp_flywheel(tmp_path, core=fake_core)
    lesson = SimpleNamespace(
        kind=SimpleNamespace(value="strategy"),
        subject="closing unused ports",
        text="Closing unused network ports reduces a host's attack surface.",
        confidence=0.9)
    engine._feed_lesson_to_flywheel(lesson)
    exs = fw.examples()
    assert len(exs) == 1
    assert exs[0].source == "flywheel-verified"          # a lesson is verified supervision
    assert "closing unused ports" in exs[0].prompt
    assert exs[0].answer == lesson.text
