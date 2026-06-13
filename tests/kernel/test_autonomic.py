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


def test_periodic_growth_runs():
    loop = _loop(growth_every=2)
    loop.run_for(4)
    assert loop.report()["growth_passes"] == 2
    assert len(loop.growth_reports) == 2


def test_no_growth_when_disabled():
    loop = _loop()  # growth_every defaults to 0
    loop.run_for(4)
    assert loop.report()["growth_passes"] == 0


def test_inner_life_off_uses_repertoire():
    # default behaviour is unchanged: prompts come from the static reflective repertoire
    loop = _loop()
    loop.tick_once()
    assert loop.prompt_sources[-1] == "repertoire"
    assert loop.report()["sources"] == {"repertoire": 1}


def test_inner_life_prefers_a_due_intention():
    from nyxara.memory.prospective import ProspectiveMemory
    pm = ProspectiveMemory()
    pm.remind_at(when=1.0, description="follow up on the Master's deployment request")
    loop = _loop(inner_life=True, prospective=pm)
    r = loop.tick_once()
    assert r is not None
    # a due standing intention is the most time-sensitive thing to think about
    assert loop.prompt_sources[-1] == "intention"


def test_due_intentions_are_queued_not_dropped():
    from nyxara.memory.prospective import ProspectiveMemory
    pm = ProspectiveMemory()
    pm.remind_at(when=1.0, description="first standing intention")
    pm.remind_at(when=1.0, description="second standing intention")
    loop = _loop(inner_life=True, prospective=pm)
    loop.run_for(2)
    # both due intentions each became their own turn — neither was silently consumed
    assert loop.prompt_sources[:2] == ["intention", "intention"]


def test_inner_life_falls_back_to_stream_then_repertoire():
    class _Thought:
        text = "loyalty and protection reinforce each other"

    class _Stream:
        def __init__(self):
            self.calls = 0

        def tick(self, engagement: float = 0.0, now=None):
            self.calls += 1
            return [_Thought()] if self.calls == 1 else []   # one thought, then quiet

    loop = _loop(inner_life=True, stream=_Stream())
    loop.run_for(2)
    assert loop.prompt_sources == ["stream", "repertoire"]   # stream first, then the steady list


def test_aprocess_matches_process():
    core = NyxaraCore()

    async def _go():
        return await core.aprocess("how are you?", authority=Authority.OWNER)

    r = asyncio.run(_go())
    assert r.acted and r.candidate.kind == "respond"
