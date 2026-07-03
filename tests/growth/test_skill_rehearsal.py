"""Tests for nyxara.growth.skill_rehearsal — never forget a learned skill."""

from __future__ import annotations

from nyxara.cognition.skill_induction import SkillInductionEngine
from nyxara.growth.skill_rehearsal import RehearsalReport, SkillRehearsalLibrary

_DEMOS = [("hello", "HELLO"), ("go now", "GO NOW"), ("ok", "OK")]


def _engine_with_skill(name: str = "shout") -> SkillInductionEngine:
    eng = SkillInductionEngine()
    assert eng.induce(name, _DEMOS) is not None
    return eng


# -------------------- sync -------------------- #
def test_sync_snapshots_verified_skill():
    lib = SkillRehearsalLibrary(_engine_with_skill())
    assert lib.sync() == 1
    assert lib.stats()["snapshots"] == 1


def test_sync_skips_unverified_skill():
    eng = _engine_with_skill()
    eng._skills["shout"].program.clear()      # the live skill no longer reproduces its demos
    lib = SkillRehearsalLibrary(eng)
    assert lib.sync() == 0                     # a broken skill is never snapshotted as good


def test_sync_is_idempotent_until_the_skill_changes():
    eng = _engine_with_skill()
    lib = SkillRehearsalLibrary(eng)
    assert lib.sync() == 1
    assert lib.sync() == 0                     # unchanged skill -> no new snapshot


# -------------------- rehearse / restore -------------------- #
def test_rehearse_healthy_engine_verifies():
    eng = _engine_with_skill()
    lib = SkillRehearsalLibrary(eng)
    lib.sync()
    report = lib.rehearse()
    assert isinstance(report, RehearsalReport)
    assert report.rehearsed == 1 and report.verified == 1 and report.regressed == 0


def test_rehearse_detects_and_restores_regression():
    eng = _engine_with_skill()
    lib = SkillRehearsalLibrary(eng)
    lib.sync()
    eng._skills["shout"].program.clear()      # sabotage: the skill now does nothing
    assert eng._skills["shout"].apply("hello") == "hello"
    report = lib.rehearse()
    assert report.regressed == 1 and report.restored == 1
    assert eng._skills["shout"].apply("hello") == "HELLO"   # known-good behavior is back


def test_rehearse_detects_missing_skill():
    eng = _engine_with_skill()
    lib = SkillRehearsalLibrary(eng)
    lib.sync()
    del eng._skills["shout"]                  # the skill vanished entirely
    report = lib.rehearse()
    assert report.regressed == 1 and report.restored == 1
    assert "shout" in eng._skills


def test_rehearse_round_robin_covers_all_skills():
    eng = _engine_with_skill("shout")
    assert eng.induce("greet", [("hi", "hi!"), ("yo", "yo!"), ("hey", "hey!")]) is not None
    lib = SkillRehearsalLibrary(eng, batch=1)
    lib.sync()
    seen = set()
    for _ in range(2):
        seen.update(lib.rehearse().skills)
    assert seen == {"shout", "greet"}         # batch=1 still reaches every skill in turn


def test_restore_unknown_skill_is_false():
    lib = SkillRehearsalLibrary(_engine_with_skill())
    assert lib.restore("never-learned") is False


# -------------------- persistence -------------------- #
def test_snapshots_persist_through_store():
    from nyxara.memory.store import MemoryStore

    store = MemoryStore()
    eng = SkillInductionEngine(store.embedder, store=store)
    assert eng.induce("shout", _DEMOS) is not None
    l1 = SkillRehearsalLibrary(eng, store=store)
    assert l1.sync() == 1
    l2 = SkillRehearsalLibrary(eng, store=store)   # a fresh library on the same store
    assert l2.rehearse().verified == 1              # hydrated snapshot verifies


def test_to_dict_load_dict_round_trip():
    lib = SkillRehearsalLibrary(_engine_with_skill())
    lib.sync()
    blob = lib.to_dict()
    fresh = SkillRehearsalLibrary(_engine_with_skill())
    fresh.load_dict(blob)
    assert fresh.stats()["snapshots"] == 1
    assert fresh.rehearse().verified == 1


def test_stats_shape():
    lib = SkillRehearsalLibrary(_engine_with_skill(), batch=3)
    lib.sync()
    lib.rehearse()
    stats = lib.stats()
    assert stats["batch"] == 3
    assert stats["lifetime"]["rehearsed"] == 1
    assert stats["last"]["verified"] == 1
