"""Tests for nyxara.temporal.fractal (the nested-loop driver)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nyxara.temporal.clock import RealClock, VirtualClock
from nyxara.temporal.fractal import FractalTemporalHierarchy


def _fth(**kw) -> FractalTemporalHierarchy:
    defaults = dict(clock=VirtualClock(), meso_every_micro=4, macro_every_meso=2,
                    micro_interval_s=0.01)
    defaults.update(kw)
    return FractalTemporalHierarchy(**defaults)


# -------------------- nested synchronous driver (loops within loops) -------------------- #
def test_run_for_nesting_ratio():
    fth = _fth()
    fired = fth.run_for(20)
    assert fth.ticks == 20
    rollups = sum(1 for f in fired if "meso_aggregate" in f)
    awarenesses = sum(1 for f in fired if "awareness" in f)
    assert rollups == 5            # every 4 micro beats
    assert awarenesses == 2        # every 2 meso roll-ups


def test_tick_always_samples_micro():
    fth = _fth()
    out = fth.tick()
    assert "micro_sample" in out and fth.ticks == 1


# -------------------- oversight -------------------- #
def test_scram_halts_the_whole_hierarchy():
    ov = SimpleNamespace(up=False, gate=lambda: ov.up)
    core = SimpleNamespace(oversight=ov, memory=None, goals=None, soul=None,
                           affect=None, self_model=None, reflector=None, bus=None)
    fth = FractalTemporalHierarchy(core=core, clock=VirtualClock(),
                                   meso_every_micro=2, macro_every_meso=2)
    out = fth.tick()
    assert out == {"halted": True} and fth.ticks == 0
    ov.up = True
    out2 = fth.tick()
    assert "micro_sample" in out2 and fth.ticks == 1


# -------------------- live async driver -------------------- #
def test_async_start_stop_bounded():
    async def _run():
        fth = FractalTemporalHierarchy(clock=RealClock(), micro_interval_s=0.001,
                                       meso_interval_s=0.005, macro_interval_s=0.02,
                                       meso_every_micro=2)
        fth.start(max_ticks=5)
        await asyncio.sleep(0.05)
        await fth.stop()
        return fth.ticks
    total = asyncio.run(_run())
    assert total >= 5


# -------------------- from_core + report -------------------- #
def test_from_core_builds_with_defaults():
    core = SimpleNamespace(oversight=None, memory=None, goals=None, soul=None,
                           affect=None, self_model=None, reflector=None, bus=None)
    fth = FractalTemporalHierarchy.from_core(core)
    fth.run_for(8)
    rep = fth.report()
    assert rep["ticks"] == 8
    assert set(rep) >= {"micro", "meso", "macro", "nesting", "intervals_s"}
