"""Tests for nyxara.growth.intelligence — the Intelligence Index: I_(t+1) = f(I_t, C_available).

Hermetic and offline: signals come from a synthetic RSI report, compute from constructed
ComputeReports, and persistence rides an in-process MemoryStore.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from nyxara.growth.intelligence import IntelligenceIndex, IntelligenceState
from nyxara.kernel.compute import ComputeReport, GPUInfo
from nyxara.memory.store import MemoryStore


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Report:
    """A minimal stand-in for SelfImprovementReport."""
    def __init__(self, accuracy=0.8, self_n=6, teacher_n=4, kept=1, n_weak=2, lessons=10):
        self.benchmark = {"accuracy": accuracy,
                          "handoff": {"self": self_n, "teacher": teacher_n, "none": 0}}
        self.weaknesses = {"n_weaknesses": n_weak}
        self.kept = kept
        self.lessons_stored = lessons


def _big() -> ComputeReport:
    return ComputeReport(cpu_count=16, memory={"available_mb": 32000, "total_mb": 64000},
                         gpu=GPUInfo(available=True, backend="cuda", count=1, names=["A100"]),
                         torch_available=True)


def _small() -> ComputeReport:
    return ComputeReport(cpu_count=1, memory={"available_mb": 512, "total_mb": 1024},
                         gpu=GPUInfo(), torch_available=False)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def test_signals_are_bounded_fractions():
    sig = IntelligenceIndex(memory=MemoryStore()).compute_signals(_Report())
    assert set(sig) == {"accuracy", "handoff", "weaknesses", "knowledge"}
    assert all(0.0 <= v <= 1.0 for v in sig.values())
    assert sig["accuracy"] == 0.8
    assert abs(sig["handoff"] - 0.6) < 1e-9          # 6 self / 10 total


def test_signals_survive_empty_report():
    sig = IntelligenceIndex().compute_signals(object())
    assert all(0.0 <= v <= 1.0 for v in sig.values())


# --------------------------------------------------------------------------- #
# compute capacity
# --------------------------------------------------------------------------- #
def test_capacity_is_bounded_and_ordered():
    cap_big = IntelligenceIndex.compute_capacity(_big())
    cap_small = IntelligenceIndex.compute_capacity(_small())
    assert 0.0 <= cap_small < cap_big <= 1.0


def test_capacity_never_raises_on_garbage():
    assert 0.0 <= IntelligenceIndex.compute_capacity(None) <= 1.0
    assert 0.0 <= IntelligenceIndex.compute_capacity(object()) <= 1.0


# --------------------------------------------------------------------------- #
# the equation
# --------------------------------------------------------------------------- #
def test_index_is_bounded_and_advances_t():
    idx = IntelligenceIndex()
    s0 = IntelligenceState()
    s1 = idx.update(s0, idx.compute_signals(_Report()), _big())
    assert 0.0 <= s1.index <= 1.0
    assert s1.t == 1
    assert s1.index > s0.index                       # a good cycle raises the index


def test_higher_accuracy_never_lowers_index():
    idx = IntelligenceIndex()
    base = idx.compute_signals(_Report())
    s0 = IntelligenceState()
    hi = idx.update(s0, {**base, "accuracy": 1.0}, _big())
    lo = idx.update(s0, {**base, "accuracy": 0.0}, _big())
    assert hi.index >= lo.index


def test_more_compute_yields_more_gain():
    idx = IntelligenceIndex()
    sig = idx.compute_signals(_Report())
    s0 = IntelligenceState()
    assert idx.update(s0, sig, _big()).index >= idx.update(s0, sig, _small()).index


def test_momentum_smooths_toward_target():
    idx = IntelligenceIndex()
    sig = idx.compute_signals(_Report())
    state = IntelligenceState()
    prev = state.index
    for _ in range(5):
        state = idx.update(state, sig, _big())
        assert state.index >= prev - 1e-9            # monotone non-decreasing toward target
        prev = state.index
    assert state.index <= 1.0


# --------------------------------------------------------------------------- #
# effort budget scales by compute, respects config bounds
# --------------------------------------------------------------------------- #
def test_effort_budget_respects_config_bounds():
    idx = IntelligenceIndex()
    state = idx.update(IntelligenceState(), idx.compute_signals(_Report()), _big())
    budget = idx.effort_budget(state, _big())
    assert 0 <= budget["max_edits_per_cycle"] <= idx.settings.self_improvement.max_edits_per_cycle
    assert 1 <= budget["recursion_depth"] <= 20


def test_effort_budget_bigger_machine_gets_bigger_budget():
    idx = IntelligenceIndex()
    state = idx.update(IntelligenceState(), idx.compute_signals(_Report()), _big())
    big = idx.effort_budget(state, _big())
    small = idx.effort_budget(state, _small())
    assert big["max_edits_per_cycle"] >= small["max_edits_per_cycle"]


def test_effort_unscaled_when_disabled():
    idx = IntelligenceIndex()
    idx.settings.self_improvement.scale_effort_by_compute = False
    try:
        b = idx.effort_budget(IntelligenceState(), _small())
        assert b["max_edits_per_cycle"] == idx.settings.self_improvement.max_edits_per_cycle
    finally:
        idx.settings.self_improvement.scale_effort_by_compute = True


# --------------------------------------------------------------------------- #
# persistence (rides memory, protected, single record)
# --------------------------------------------------------------------------- #
def test_persists_and_loads_via_memory():
    store = MemoryStore()
    idx = IntelligenceIndex(memory=store)
    state = idx.update(IntelligenceState(), idx.compute_signals(_Report()), _big())
    assert idx.save(state)
    loaded = IntelligenceIndex(memory=store).load()
    assert loaded.t == state.t
    assert abs(loaded.index - state.index) < 1e-6


def test_persisted_record_is_protected():
    store = MemoryStore()
    idx = IntelligenceIndex(memory=store)
    idx.save(idx.update(IntelligenceState(), idx.compute_signals(_Report()), _big()))
    rec = idx._find_record()
    assert rec is not None
    assert rec.importance >= 0.9
    assert idx.settings.memory.deep_synapse_tag in rec.tags


def test_save_keeps_a_single_record():
    store = MemoryStore()
    idx = IntelligenceIndex(memory=store)
    state = IntelligenceState()
    for _ in range(3):
        state = idx.update(state, idx.compute_signals(_Report()), _big())
        idx.save(state)
    matches = [r for r in store._kv.values() if "intelligence-index" in r.tags]
    assert len(matches) == 1
    assert IntelligenceState.from_dict(matches[0].metadata["state"]).t == state.t


def test_fresh_store_starts_at_I0():
    assert IntelligenceIndex(memory=MemoryStore()).load().t == 0
    assert IntelligenceIndex().load().index == 0.0     # no memory => in-process zero


# --------------------------------------------------------------------------- #
# integration with the RSI cycle (dry-run changes no source, sets the index)
# --------------------------------------------------------------------------- #
def test_rsi_run_populates_intelligence_without_touching_source():
    from nyxara.growth.recursive_improvement import RecursiveSelfImprovement

    pkg = Path(__import__("nyxara").__file__).resolve().parent
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in pkg.rglob("*.py")}

    rsi = RecursiveSelfImprovement(memory=MemoryStore())
    report = rsi.run(enact=False)

    assert report.intelligence_index is not None and 0.0 <= report.intelligence_index <= 1.0
    assert report.intelligence_t is not None and report.intelligence_t >= 1
    assert report.effort_budget is not None and "max_edits_per_cycle" in report.effort_budget
    # a dry-run must change no source file
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob("*.py")}
    assert before == after
    assert not report.optimizations
