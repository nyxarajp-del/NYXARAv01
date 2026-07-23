"""Tests for the self-optimization / self-healing wire in the always-on AutonomicLoop.

NYXARA's unified eleven-phase ``SelfOptimizationLoop`` (``core.self_optimize`` — self-analysis →
optimize → experiment → architecture → learning → self-debug → invent → safety-verify) used to run
*only* on demand (a turn or the CLI). It is now driven by the background mind itself: on a slow
periodic cadence, and reactively when the loop genuinely stalls. These tests pin that wire — fires on
cadence, off by default (back-compat), halts on scram, seals the heavy pytest self-debug OFF under the
hermetic TEST profile (so it can never recurse into the running suite), and surfaces in the report.

The real self-optimization cycle is stubbed here (``core.self_optimize`` is replaced by a recorder):
this is a wiring contract, not a benchmark of the eleven phases.
"""

from __future__ import annotations

from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.kernel.orchestrator import NyxaraCore


class _Rep:
    """A stand-in SelfOptimizationReport with the only surface the wire touches."""

    def to_dict(self):
        return {"phases": [], "enacted": False}


class _Recorder:
    """Replaces ``core.self_optimize`` and records how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return _Rep()


class _StubGrowth:
    """A no-op growth engine so ``_maybe_grow`` stays fast and writes nothing."""

    def run(self, **kw):
        return {}


def _loop(**kw):
    core = NyxaraCore()
    rec = _Recorder()
    core.self_optimize = rec   # instance attr shadows the bound method
    loop = AutonomicLoop(core, interval_s=0.0, growth_engine=_StubGrowth(), **kw)
    return loop, rec


# -------------------- fires on the periodic cadence -------------------- #
def test_periodic_self_optimize_fires_on_cadence():
    loop, rec = _loop(growth_every=1, self_optimize_every=2)
    # every tick is a growth pass (growth_every=1); self_optimize_every=2 → fire on passes 2, 4, ...
    loop.run_for(4)
    assert len(rec.calls) == 2
    assert loop.report()["self_optimizations"] == 2


# -------------------- off by default (back-compat) -------------------- #
def test_self_optimize_off_by_default():
    loop, rec = _loop(growth_every=1)   # self_optimize_every defaults to 0
    loop.run_for(6)
    assert rec.calls == []
    assert loop.report()["self_optimizations"] == 0


# -------------------- oversight halts self-modification -------------------- #
def test_scram_halts_self_optimize():
    loop, rec = _loop(growth_every=1, self_optimize_every=1)
    loop.core.scram(reason="stand down")
    # a scrammed tick no-ops before growth; and a forced pass is gated too
    loop.run_for(3)
    assert loop._maybe_self_optimize(force=True) is None
    assert rec.calls == []
    loop.core.resume()
    assert loop._maybe_self_optimize(force=True) is not None
    assert len(rec.calls) == 1


# -------------------- the heavy pytest self-debug is TEST-sealed -------------------- #
def test_debug_phase_sealed_under_test_profile():
    # even when include_debug is explicitly requested, the TEST profile forces it OFF so the
    # detection pytest can never recurse into the running suite.
    loop, rec = _loop()
    loop._maybe_self_optimize(force=True, include_debug=True)
    assert len(rec.calls) == 1
    assert rec.calls[0]["include_debug"] is False


# -------------------- reactive heal on a genuine stall -------------------- #
def test_reactive_heal_fires_on_stall_then_cools_down():
    loop, rec = _loop(stall_threshold=3)
    # simulate a stalled mind
    loop.consecutive_unproductive = 3
    loop._maybe_reactive_heal()
    assert len(rec.calls) == 1                 # fired one forced self-heal
    # cooldown blocks an immediate second attempt even while still stalled
    loop.consecutive_unproductive = 3
    loop._maybe_reactive_heal()
    assert len(rec.calls) == 1


def test_reactive_heal_quiet_when_not_stalled():
    loop, rec = _loop(stall_threshold=5)
    loop.consecutive_unproductive = 2          # below threshold
    loop._maybe_reactive_heal()
    assert rec.calls == []
