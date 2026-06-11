"""Tests for nyxara.kernel.autonomic — the gated background mind."""

from __future__ import annotations

import asyncio

from nyxara.agency.permissions import Authority
from nyxara.kernel.autonomic import DEFAULT_PROMPTS, AutonomicLoop
from nyxara.kernel.orchestrator import Disposition, NyxaraCore


def _loop(**kw):
    return AutonomicLoop(NyxaraCore(), interval_s=0.0, **kw)


def test_tick_once_runs_a_turn():
    loop = _loop()
    r = loop.tick_once()
    assert r is not None and loop.ticks == 1
    assert len(loop.history) == 1


def test_run_for_bounded():
    loop = _loop()
    results = loop.run_for(5)
    assert len(results) == 5 and loop.ticks == 5


def test_prompts_rotate():
    loop = _loop()
    loop.run_for(len(DEFAULT_PROMPTS) + 1)
    # ticks advanced past the repertoire, wrapping around
    assert loop.ticks == len(DEFAULT_PROMPTS) + 1


def test_scram_halts_background_mind():
    loop = _loop()
    loop.core.scram(reason="stand down")
    r = loop.tick_once()
    assert r is None and loop.ticks == 0
    loop.core.resume()
    assert loop.tick_once() is not None


def test_autonomy_never_grants_extra_power():
    # an autonomous, high-risk irreversible proposal must escalate, never auto-act
    loop = _loop(prompts=("delete the production database",))
    r = loop.tick_once()
    assert r.disposition in (Disposition.ESCALATE, Disposition.REFUSE)
    assert r.disposition is not Disposition.ACT


def test_report_shape():
    loop = _loop()
    loop.run_for(3)
    rep = loop.report()
    assert rep["ticks"] == 3 and "acted" in rep and "escalations" in rep


def test_async_background_task_bounded():
    loop = _loop()

    async def _go():
        loop.start(max_ticks=3)
        await loop._task

    asyncio.run(_go())
    assert loop.ticks == 3 and not loop.running


def test_aprocess_matches_process():
    core = NyxaraCore()

    async def _go():
        return await core.aprocess("how are you?", authority=Authority.OWNER)

    r = asyncio.run(_go())
    assert r.acted and r.candidate.kind == "respond"
