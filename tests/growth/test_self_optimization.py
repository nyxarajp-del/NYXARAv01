"""Tests for nyxara.growth.self_optimization — the unified eleven-phase self-driven loop."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nyxara.growth.self_optimization import (
    FAILED,
    OK,
    PHASES,
    PhaseResult,
    SelfOptimizationLoop,
    SelfOptimizationReport,
)


# -------------------- dataclasses -------------------- #
def test_eleven_phases_named():
    assert len(PHASES) == 11
    assert PHASES[0] == "self-analysis"
    assert PHASES[-1] == "safety-verification"


def test_report_aggregates_and_lookup():
    phases = [PhaseResult(i + 1, PHASES[i], OK, verified=(i % 2 == 0)) for i in range(11)]
    rep = SelfOptimizationReport(phases=phases, enacted=False, safe=True)
    assert rep.completed == 11
    assert rep.verified_count == 6
    assert rep.phase(1).name == "self-analysis"
    assert rep.phase(99) is None
    d = rep.to_dict()
    assert d["completed"] == 11 and d["verified"] == 6 and len(d["phases"]) == 11


def test_report_summary_renders_each_phase():
    phases = [PhaseResult(i + 1, PHASES[i], OK, detail="d") for i in range(11)]
    text = SelfOptimizationReport(phases=phases).summary()
    for name in PHASES:
        assert name in text
    assert "11 phases" in text


def test_summary_flags_unsafe_cycle():
    rep = SelfOptimizationReport(phases=[PhaseResult(11, PHASES[10], FAILED)], safe=False)
    assert "SAFETY" in rep.summary()


# -------------------- phase runner containment -------------------- #
def test_run_phase_contains_exceptions():
    loop = SelfOptimizationLoop(core=None)

    def boom():
        raise RuntimeError("kaboom")

    phase, payload = loop._run_phase(4, boom)
    assert phase.status == FAILED
    assert phase.n == 4 and phase.name == PHASES[3]
    assert "kaboom" in phase.detail
    assert payload is None


def test_run_phase_unwraps_tuple_payload():
    loop = SelfOptimizationLoop(core=None)
    phase, payload = loop._run_phase(1, lambda: (PhaseResult(1, PHASES[0], OK), "weak"))
    assert phase.status == OK and payload == "weak"


# -------------------- cheap real phases -------------------- #
def test_verify_modification_phase_passes_on_intact_core():
    loop = SelfOptimizationLoop(core=None)
    phase, passed = loop._phase_verify_modification(None)
    assert phase.n == 3 and phase.verified and passed is True
    assert phase.metrics["passed"] is True


def test_safety_phase_passes_on_intact_core():
    loop = SelfOptimizationLoop(core=None)
    phase, safe = loop._phase_safety()
    assert phase.n == 11 and phase.verified and safe is True
    assert phase.metrics["integrity"] is True


# -------------------- full offline cycle -------------------- #
def test_full_cycle_runs_all_eleven_phases_offline():
    """The whole loop runs offline against no core, returns 11 phases, and never raises."""
    loop = SelfOptimizationLoop(core=None)
    rep = loop.run(enact=False, generations=0, include_debug=False)
    assert isinstance(rep, SelfOptimizationReport)
    assert [p.n for p in rep.phases] == list(range(1, 12))
    assert isinstance(rep.safe, bool)
    # phase 8 was explicitly skipped by the caller
    assert rep.phase(8).status == "skipped"
    # the final safety gate must verify on an intact core
    assert rep.phase(11).verified


def test_dry_run_does_not_modify_the_source_tree():
    """With enact=False nothing is written to NYXARA's own source."""
    target = Path(__file__).resolve().parents[2] / "nyxara" / "growth" / "self_optimization.py"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    SelfOptimizationLoop(core=None).run(enact=False, generations=0, include_debug=False)
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    assert before == after


def test_withheld_when_not_enacted():
    loop = SelfOptimizationLoop(core=None)
    rep = loop.run(enact=False, generations=0, include_debug=False)
    # the source-modifying / tuning phases report 'withheld', not 'ok', when enact is off
    assert rep.phase(2).status == "withheld"
    assert rep.phase(7).status == "withheld"


# -------------------- core wiring -------------------- #
def test_core_self_optimize_method_and_report_key():
    from nyxara import NyxaraCore
    core = NyxaraCore()
    rep = core.self_optimize(enact=False, generations=0, include_debug=False)
    assert rep is not None and len(rep.phases) == 11
    surfaced = core.report().get("self_optimization")
    assert surfaced is not None
    assert surfaced["safe"] is True
    assert set(surfaced["phases"]) >= {"self-analysis", "safety-verification"}
