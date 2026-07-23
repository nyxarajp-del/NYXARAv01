"""Tests for the self-optimization / self-healing wire in the always-on AutonomicLoop.

NYXARA's unified eleven-phase ``SelfOptimizationLoop`` (``core.self_optimize`` — self-analysis →
optimize → experiment → architecture → learning → self-debug → invent → safety-verify) used to run
*only* on demand (a turn or the CLI). It is now driven by the background mind itself: on a slow
periodic cadence, reactively when the loop genuinely stalls, and pre-emptively on a free-energy spike.
These tests pin that wire — fires on cadence, off by default (back-compat), halts on scram, seals the
heavy pytest self-debug OFF under any pytest process (so it can never recurse into the running suite),
and surfaces in the report.

A lightweight fake core is used (``core.self_optimize`` is a recorder): this is a wiring contract, not a
benchmark of the eleven phases, and it keeps the suite hermetic + fast (no NyxaraCore boot).
"""

from __future__ import annotations

from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.kernel.orchestrator import Disposition


class _Rep:
    def to_dict(self):
        return {"phases": [], "enacted": False}


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return _Rep()


class _Oversight:
    def __init__(self):
        self._ok = True

    def gate(self):
        return self._ok

    def scram(self, reason=""):
        self._ok = False

    def resume(self):
        self._ok = True


class _Result:
    disposition = Disposition.ACT


class _FakeCore:
    """Just enough surface for the AutonomicLoop wire — no heavy NyxaraCore boot."""

    def __init__(self):
        self.oversight = _Oversight()
        self.self_optimize = _Recorder()

    def process(self, prompt, authority=None):
        return _Result()

    def scram(self, reason=""):
        self.oversight.scram(reason)

    def resume(self):
        self.oversight.resume()


class _StubGrowth:
    def run(self, **kw):
        return {}


def _loop(**kw):
    core = _FakeCore()
    loop = AutonomicLoop(core, interval_s=0.0, growth_engine=_StubGrowth(),
                         advance_missions=False, **kw)
    return loop, core.self_optimize


# -------------------- fires on the periodic cadence -------------------- #
def test_periodic_self_optimize_fires_on_cadence():
    loop, rec = _loop(growth_every=1, self_optimize_every=2)
    loop.run_for(4)                            # passes 2 and 4 fire
    assert len(rec.calls) == 2
    assert loop.report()["self_optimizations"] == 2


# -------------------- off by default (back-compat) -------------------- #
def test_self_optimize_off_by_default():
    loop, rec = _loop(growth_every=1)          # self_optimize_every defaults to 0
    loop.run_for(6)
    assert rec.calls == []
    assert loop.report()["self_optimizations"] == 0


# -------------------- oversight halts self-modification -------------------- #
def test_scram_halts_self_optimize():
    loop, rec = _loop(growth_every=1, self_optimize_every=1)
    loop.core.scram(reason="stand down")
    loop.run_for(3)
    assert loop._maybe_self_optimize(force=True) is None
    assert rec.calls == []
    loop.core.resume()
    assert loop._maybe_self_optimize(force=True) is not None
    assert len(rec.calls) == 1


# -------------------- the heavy pytest self-debug is sealed under pytest -------------------- #
def test_debug_phase_sealed_under_pytest():
    loop, rec = _loop()
    loop._maybe_self_optimize(force=True, include_debug=True)
    assert len(rec.calls) == 1
    assert rec.calls[0]["include_debug"] is False


# -------------------- reactive heal on a genuine stall -------------------- #
def test_reactive_heal_fires_on_stall_then_cools_down():
    loop, rec = _loop(stall_threshold=3)
    loop.consecutive_unproductive = 3
    loop._maybe_reactive_heal()
    assert len(rec.calls) == 1
    loop.consecutive_unproductive = 3
    loop._maybe_reactive_heal()
    assert len(rec.calls) == 1                 # cooldown blocks a second immediate attempt


def test_reactive_heal_quiet_when_not_stalled():
    loop, rec = _loop(stall_threshold=5)
    loop.consecutive_unproductive = 2
    loop._maybe_reactive_heal()
    assert rec.calls == []


# -------------------- N: free-energy pre-emptive heal -------------------- #
def test_free_energy_disabled_by_default():
    loop, rec = _loop()
    assert loop._genmodel is None
    loop._maybe_preemptive_heal()
    assert rec.calls == []


def test_free_energy_spike_triggers_preemptive_heal():
    loop, rec = _loop(free_energy_threshold=0.2, stall_threshold=3)
    if loop._genmodel is None:
        import pytest
        pytest.skip("active-inference model unavailable")
    for _ in range(3):
        loop._maybe_preemptive_heal()          # quiet baseline
    loop.errors += 25                          # a sudden error burst → high surprise
    loop.consecutive_unproductive = 4
    for _ in range(3):
        loop._maybe_preemptive_heal()
    assert len(rec.calls) >= 1
    assert "free_energy" in loop.report()
