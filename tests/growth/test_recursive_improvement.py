"""Tests for nyxara.growth.recursive_improvement — the unified RSI cycle."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nyxara.growth.recursive_improvement import (RecursiveSelfImprovement,
                                                 SelfImprovementReport)
from nyxara.kernel.config import get_settings


def _rsi_with_parallel(parallel: bool) -> RecursiveSelfImprovement:
    s = get_settings().model_copy(deep=True)
    s.self_improvement.parallel_cycle = parallel
    return RecursiveSelfImprovement(settings=s)


def test_parallel_cycle_matches_sequential():
    rep_p = _rsi_with_parallel(True).run(enact=False)
    rep_s = _rsi_with_parallel(False).run(enact=False)
    # the three observation steps write disjoint caches, so order cannot change the result
    assert rep_p.code["files_scanned"] == rep_s.code["files_scanned"]
    assert rep_p.architecture["n_modules"] == rep_s.architecture["n_modules"]


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


# --------------------------------------------------------------------------- #
# external validation: held-out / adversarial scores never steer edit selection,
# and the ledger is credited by the TRANSFER delta (anti-Goodhart, the heart of
# this change)
# --------------------------------------------------------------------------- #
def test_validation_never_feeds_weakness_selection():
    """The validation block (held-out + adversarial) must not influence what gets edited."""
    rsi = _rsi()
    rsi.settings = get_settings()
    # the proxy benchmark (drives edits) has two weak training categories...
    rsi._bench = {"by_category": {"arithmetic": {"accuracy": 0.2},
                                  "logic": {"accuracy": 0.4}}}
    # ...and the validation block reports adversarial categories that exist NOWHERE in _bench
    rsi._validation = {"transfer_score": 0.1,
                       "adversarial_by_category": {"calibration": {"accuracy": 0.0},
                                                   "deduction": {"accuracy": 0.1}}}
    weak = rsi._weakest_categories(limit=5)
    blob = " ".join(weak)
    assert "calibration" not in blob and "deduction" not in blob   # validation can't steer edits
    assert "arithmetic" in blob                                    # only the proxy fold does


def test_credit_outcome_rewards_transfer_delta_not_proxy():
    from nyxara.memory.store import MemoryStore
    rsi = RecursiveSelfImprovement(memory=MemoryStore())
    rsi.settings.self_improvement.credit_on_transfer = True
    # held-out transfer rose from 0.40 -> 0.55 even though we won't look at the proxy index delta
    rsi._prior_transfer = 0.40
    rsi._prior_index = 0.90              # proxy FELL — if credit used the index it would be negative
    rsi._validation = {"transfer_score": 0.55}
    rsi._touched_arms = ["arm:deepen_reasoning"]
    report = SelfImprovementReport(enacted=True, intelligence_index=0.80)
    rsi._credit_outcome(report)
    assert report.ledger is not None
    assert report.ledger["basis"] == "transfer"
    assert abs(report.ledger["delta"] - 0.15) < 1e-6     # 0.55 - 0.40, NOT the index drop
    assert report.ledger["reward"] > 0                   # a real transferred gain is reinforced


def test_credit_falls_back_to_index_when_transfer_disabled():
    from nyxara.memory.store import MemoryStore
    rsi = RecursiveSelfImprovement(memory=MemoryStore())
    rsi.settings.self_improvement.credit_on_transfer = False
    rsi._prior_index = 0.50
    rsi._validation = {"transfer_score": 0.99}           # present but must be ignored
    rsi._touched_arms = ["arm:deepen_reasoning"]
    report = SelfImprovementReport(enacted=True, intelligence_index=0.60)
    rsi._credit_outcome(report)
    assert report.ledger["basis"] == "index"
    assert abs(report.ledger["delta"] - 0.10) < 1e-6     # 0.60 - 0.50


# --------------------------------------------------------------------------- #
# run_validation now includes the REAL, externally-true held-out corpus as the
# dominant transfer ruler — and degrades cleanly when it is disabled.
# --------------------------------------------------------------------------- #
def test_run_validation_includes_realworld_and_blends_transfer():
    s = get_settings().model_copy(deep=True)
    s.self_improvement.validation_realworld_enabled = True
    val = RecursiveSelfImprovement(settings=s).run_validation()
    assert val is not None and "error" not in val
    # the real corpus was measured and surfaced...
    for key in ("realworld_accuracy", "realworld_mean_score", "realworld_n"):
        assert key in val, val
    assert val["realworld_n"] >= 40
    assert val["realworld_source"] == "bundled"
    # ...and transfer_score is a clean convex blend of the three rulers (always in [0, 1]).
    assert 0.0 <= val["transfer_score"] <= 1.0


def test_run_validation_degrades_when_realworld_disabled():
    s = get_settings().model_copy(deep=True)
    s.self_improvement.validation_realworld_enabled = False
    val = RecursiveSelfImprovement(settings=s).run_validation()
    assert val is not None and "error" not in val
    assert "realworld_accuracy" not in val          # dropped cleanly
    assert "transfer_score" in val                  # the other two rulers still produce a score
    assert 0.0 <= val["transfer_score"] <= 1.0
