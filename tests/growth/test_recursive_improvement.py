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


# --------------------------------------------------------------------------- #
# multi-edit safety: at most one *kept* edit per file per cycle (line numbers
# from a single review pass go stale once a kept edit shifts a file's lines)
# --------------------------------------------------------------------------- #
class _FakeEdit:
    def __init__(self, file: str) -> None:
        self.file = file

    def to_dict(self):
        return {"file": self.file}


class _FakeOutcome:
    def __init__(self, *, kept: bool) -> None:
        self.applied = True
        self.kept = kept
        self.rolled_back = not kept

    def to_dict(self):
        return {"kept": self.kept, "rolled_back": self.rolled_back}


class _FakeGen:
    """Maps each weakness to an edit targeting weakness.locus's file."""
    def generate(self, w):
        return _FakeEdit(getattr(w, "file", "?"))


class _FakeOptimizer:
    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.applied = []

    def apply(self, edit):
        self.applied.append(edit.file)
        return _FakeOutcome(kept=self.keep)


class _W:
    def __init__(self, file: str) -> None:
        self.file = file
        self.is_source_edit = True


def _rsi():
    return RecursiveSelfImprovement.__new__(RecursiveSelfImprovement)


def test_enact_edits_one_kept_edit_per_file():
    rsi = _rsi()
    opt = _FakeOptimizer(keep=True)
    report = SelfImprovementReport()
    # two weaknesses in fileA, one in fileB
    ranked = [_W("a.py"), _W("a.py"), _W("b.py")]
    rsi._enact_edits(ranked, _FakeGen(), opt, budget=5, report=report)
    # fileA edited once (second deferred), fileB edited once
    assert opt.applied == ["a.py", "b.py"]
    assert report.kept == 2
    assert len(report.optimizations) == 2


def test_enact_edits_rolled_back_does_not_claim_file():
    rsi = _rsi()
    opt = _FakeOptimizer(keep=False)        # every edit rolls back
    report = SelfImprovementReport()
    ranked = [_W("a.py"), _W("a.py")]
    rsi._enact_edits(ranked, _FakeGen(), opt, budget=5, report=report)
    # a rolled-back edit leaves the file unchanged, so the second same-file edit may proceed
    assert opt.applied == ["a.py", "a.py"]
    assert report.kept == 0 and report.rolled_back == 2


def test_enact_edits_respects_budget():
    rsi = _rsi()
    opt = _FakeOptimizer(keep=True)
    report = SelfImprovementReport()
    ranked = [_W("a.py"), _W("b.py"), _W("c.py")]
    rsi._enact_edits(ranked, _FakeGen(), opt, budget=2, report=report)
    assert opt.applied == ["a.py", "b.py"]   # third skipped by budget
    assert report.kept == 2


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


# --------------------------------------------------------------------------- #
# the intelligence index genuinely DRIVES behavior (not a dashboard number):
# it scopes the benchmark and caps reasoning depth by index + compute
# --------------------------------------------------------------------------- #
class _StubIntel:
    """A controllable IntelligenceIndex stand-in returning a fixed effort budget."""
    def __init__(self, budget):
        self.budget = budget

    def load(self):
        return None

    def effort_budget(self, state, compute):
        return self.budget


def _tuning_settings():
    from nyxara.kernel.config import get_settings
    s = get_settings().model_copy(deep=True)
    s.self_improvement.allow_tuning = True
    s.self_improvement.benchmark_in_cycle = True
    s.llm.recursive_improvement_iterations = 10
    return s


def test_weak_compute_scopes_the_benchmark():
    weak = _StubIntel({"benchmark_full": False, "recursion_depth": 3,
                       "max_edits_per_cycle": 1, "intensity": 0.2})
    rsi = RecursiveSelfImprovement(settings=_tuning_settings(), intelligence=weak)
    bench = rsi.run_benchmarks()
    assert bench["scope"] != "full"                 # probed a single category, not the battery
    assert len(bench["by_category"]) == 1


def test_strong_compute_runs_full_benchmark():
    strong = _StubIntel({"benchmark_full": True, "recursion_depth": 20,
                         "max_edits_per_cycle": 3, "intensity": 0.9})
    rsi = RecursiveSelfImprovement(settings=_tuning_settings(), intelligence=strong)
    bench = rsi.run_benchmarks()
    assert bench["scope"] == "full"
    assert len(bench["by_category"]) >= 2


def test_recursion_depth_capped_by_index():
    weak = _StubIntel({"benchmark_full": False, "recursion_depth": 3,
                       "max_edits_per_cycle": 1, "intensity": 0.2})
    s = _tuning_settings()
    rsi = RecursiveSelfImprovement(settings=s, intelligence=weak)
    rsi.run_benchmarks()
    rsi._bench["accuracy"] = 0.1                     # force the tune branch
    rsi._maybe_tune(rsi.detect_weaknesses(), enact=True)
    assert s.llm.recursive_improvement_iterations <= 3   # not permitted deeper than the index allows


def test_disabled_index_leaves_behavior_unscaled():
    s = _tuning_settings()
    s.self_improvement.intelligence_index_enabled = False
    rsi = RecursiveSelfImprovement(settings=s)
    bench = rsi.run_benchmarks()
    assert bench["scope"] == "full"                 # no index → full battery, full depth ceiling
