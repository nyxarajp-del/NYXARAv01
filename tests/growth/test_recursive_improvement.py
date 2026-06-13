"""Tests for nyxara.growth.recursive_improvement — the unified RSI cycle."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nyxara.growth.recursive_improvement import (RecursiveSelfImprovement,
                                                 SelfImprovementReport)


def test_dry_run_populates_all_sections():
    rep = RecursiveSelfImprovement().run(enact=False)
    assert isinstance(rep, SelfImprovementReport)
    assert rep.code and rep.code["files_scanned"] > 20
    assert rep.architecture and rep.architecture["n_modules"] > 50
    assert rep.weaknesses is not None
    assert rep.benchmark is not None


def test_dry_run_changes_nothing():
    rsi = RecursiveSelfImprovement()
    rep = rsi.run(enact=False)
    assert rep.enacted is False
    assert rep.kept == 0 and rep.rolled_back == 0
    assert rep.optimizations == []


def test_run_benchmarks_returns_real_fraction():
    bench = RecursiveSelfImprovement().run_benchmarks()
    assert 0.0 <= bench["accuracy"] <= 1.0
    assert "handoff" in bench
    assert "by_category" in bench


def test_dry_run_does_not_touch_source_files():
    targets = [Path("nyxara/kernel/orchestrator.py"),
               Path("nyxara/growth/autolearn.py")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in targets}
    RecursiveSelfImprovement().run(enact=False)
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in targets}
    assert before == after, "dry-run must not modify any source file"


def test_lessons_stored_in_memory_when_enacting():
    from nyxara.memory.store import MemoryStore
    mem = MemoryStore()
    rsi = RecursiveSelfImprovement(memory=mem)
    # enact the SAFE set only (lessons); no source edits without autonomous_enact in config
    rep = rsi.optimize(enact=True)
    assert rep.lessons_stored > 0
    assert len(mem) > 0


def test_tuning_stays_within_bounds():
    from nyxara.kernel.config import get_settings
    s = get_settings().model_copy(deep=True)
    s.self_improvement.allow_tuning = True
    s.self_improvement.benchmark_in_cycle = True
    rsi = RecursiveSelfImprovement(settings=s)
    rsi.run_benchmarks()
    rsi.detect_weaknesses()
    rsi._maybe_tune(rsi.detect_weaknesses(), enact=True)
    assert 1 <= s.llm.recursive_improvement_iterations <= 20
