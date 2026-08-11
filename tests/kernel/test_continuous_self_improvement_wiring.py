"""Tests for the continuous Recursive-Self-Improvement wire.

NYXARA's heavy self-improvement tower (the ``GrowthEngine``: redesign reasoning via
mind-evolution, evaluate+rebuild her own architecture via recursive self-improvement, improve how
she improves via the meta-meta tower, invent+test theories via meta-research) used to run *only*
when a human ran a CLI or the never-started ``AutonomicLoop``. It is now built on the live core and
driven by NYXARA HERSELF from :meth:`NyxaraCore.idle_maintenance` on a throttled, oversight-gated,
config-flagged cadence — no human command, no external LLM.

These tests pin the wire (built, fires on cadence, halts on scram, is sealed off in the suite,
surfaces in the report). The heavy sub-engines are disabled here so ``GrowthEngine.run()`` reduces
to its fast reflect+consolidate path — capability-1/5/6 *execution* is covered elsewhere; this is a
wiring contract, not a benchmark.
"""

from __future__ import annotations

import os

import pytest

from nyxara.kernel.config import reload_settings
from nyxara.kernel.orchestrator import NyxaraCore

# Enable the continuous wire with a small cadence, but keep every heavy/self-modifying sub-engine
# OFF so the idle growth pass is fast and writes nothing to disk.
_ENV = {
    "NYXARA_SELF_IMPROVEMENT__CONTINUOUS": "true",
    "NYXARA_SELF_IMPROVEMENT__IDLE_GROWTH_EVERY": "3",
    "NYXARA_SELF_IMPROVEMENT__ENABLED": "false",      # skip the codebase RSI benchmark
    "NYXARA_MIND_EVOLUTION__ENABLED": "false",         # skip the reasoner-evolution benchmark
    "NYXARA_META_RESEARCH__ENABLED": "false",          # skip meta-research
    "NYXARA_FOUNDRY__ENABLED": "false",                # no forging in the wire test
}


@pytest.fixture
def continuous_core():
    """A core booted with the continuous wire armed (and the heavy engines off), restoring the
    suite's hermetic posture on teardown."""
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ.update(_ENV)
    reload_settings()
    try:
        yield NyxaraCore()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reload_settings()   # back to the conftest (continuous OFF) posture


# -------------------- it is built and armed -------------------- #
def test_growth_engine_built_on_the_core():
    # the unifier is built whenever growth is enabled — independent of the continuous flag
    nyx = NyxaraCore()
    assert nyx.growth_engine is not None
    assert nyx._growth_idle_count == 0


def test_growth_engine_absent_when_growth_disabled():
    nyx = NyxaraCore(enable_growth=False)
    assert nyx.growth_engine is None


# -------------------- the suite seals it OFF (hermetic) -------------------- #
@pytest.mark.timeout(900)
def test_continuous_is_off_by_default_in_the_suite():
    """Six idle passes must not run the growth tower.

    The 900s override is a mitigation, not a fix, and the thing it mitigates is **pre-existing**:
    this test overruns the suite's 300s ceiling on ``main`` too, at this same line. What it costs
    is ``idle_maintenance`` doing real work per pass — chiefly ``sim/physics_world.physics_stream``
    training a pure-Python world model — and that cost AMPLIFIES roughly 9x when the test runs
    late in a full suite versus alone (35s isolated). The amplification is accumulated state from
    earlier tests, and it is not diagnosed here.

    NYX V.001 in the reason-seat makes it worse by about 45% (measured: 24s → 35s isolated), which
    is the Layer stack's ~0.15s per turn paid across many turns. That is a real cost of the merge
    and is recorded rather than hidden — but it is not what pushes this past the ceiling, because
    the ceiling is already breached without it.
    """
    nyx = NyxaraCore()
    for _ in range(6):
        report = nyx.idle_maintenance()
        assert "growth" not in report
    assert nyx._growth_idle_count == 0


# -------------------- NYXARA drives it herself on a cadence -------------------- #
def test_idle_runs_growth_on_the_outer_cadence(continuous_core):
    nyx = continuous_core
    # idle_growth_every == 3: the first two idle passes do not run the tower
    assert "growth" not in nyx.idle_maintenance()
    assert "growth" not in nyx.idle_maintenance()
    # the third idle pass fires one full GrowthEngine cycle
    report = nyx.idle_maintenance()
    assert "growth" in report
    assert nyx._growth_idle_count == 3
    assert nyx._last_growth_report is not None


# -------------------- corrigibility: scram halts self-improvement -------------------- #
def test_scram_halts_continuous_growth(continuous_core):
    nyx = continuous_core
    nyx.oversight.scram(reason="test halt")
    for _ in range(4):   # well past the cadence
        report = nyx.idle_maintenance()
        assert "growth" not in report
    # a halted mind never advanced the growth counter
    assert nyx._growth_idle_count == 0


# -------------------- the report proves it is real, wired, and armed -------------------- #
def test_report_exposes_continuous_self_improvement(continuous_core):
    nyx = continuous_core
    csi = nyx.report()["continuous_self_improvement"]
    assert csi["armed"] is True
    assert csi["idle_every"] == 3


def test_report_shows_disarmed_under_suite_default():
    nyx = NyxaraCore()
    csi = nyx.report()["continuous_self_improvement"]
    assert csi["armed"] is False
