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

    # disable the (higher-priority) proactive source so the stream→repertoire fallback
    # can be exercised in isolation
    loop = _loop(inner_life=True, stream=_Stream(), proactive_allowed=False)
    loop.run_for(2)
    assert loop.prompt_sources == ["stream", "repertoire"]   # stream first, then the steady list


def _code_core():
    # enable_memory=False keeps the research action offline (researcher is None), so the
    # code-driven decision+action path is exercised fully but hermetically (no network).
    return NyxaraCore(enable_memory=False)


def test_code_mode_decides_and_acts_without_the_llm():
    core = _code_core()
    calls = {"n": 0}
    orig = core.process
    def _spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    core.process = _spy

    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code")
    loop.run_for(5)

    assert loop.ticks == 5
    # the whole point: NYXARA decides in her own code; the LLM reasoner is never the decider
    assert calls["n"] == 0
    rep = loop.report()
    assert rep["decision_mode"] == "code"
    assert rep["intents_adopted"] >= 1      # adopted her own goals by active inference
    assert rep["code_acts"] >= 1            # cleared ACTs executed in code (via the scheduler)
    assert core.journal.verify()            # every autonomous act journaled + chain intact
    assert len(core.journal.actions()) >= 1


def test_code_mode_keeps_goals_and_scheduler_bounded():
    core = _code_core()
    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code")
    base = len(core.goals)
    loop.run_for(40)
    # re-adopting the same drive-goal every tick must not grow the objective space unbounded
    assert len(core.goals) <= base + 6
    # finished one-shot scheduler tasks are purged, so the task table stays bounded
    assert core.scheduler.report()["tasks"] <= 8


def test_code_mode_respects_scram():
    core = _code_core()
    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code")
    core.scram(reason="stand down")
    assert loop.tick_once() is None and loop.ticks == 0
    core.resume()
    assert loop.tick_once() is not None and loop.ticks == 1


def test_code_mode_persists_state_to_disk(tmp_path, monkeypatch):
    core = _code_core()
    # pin the persistence dir to a temp location (independent of the global settings/home)
    monkeypatch.setattr(core, "_autonomy_state_dir", lambda: str(tmp_path))
    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code")
    loop.run_for(4)
    saved = core.persist_autonomy_state()
    assert saved["goals"] and saved["motivation"]
    assert (tmp_path / "autonomy_goals.json").exists()
    assert (tmp_path / "autonomy_motivation.json").exists()
    # the emergent goals reload into a fresh GoalSystem seeded with the same core commitments
    from nyxara.planning.goals import GoalSystem
    gs = GoalSystem()
    reloaded = gs.load(str(tmp_path / "autonomy_goals.json"))
    assert reloaded >= 1


def test_presence_paces_but_never_silences_autonomy():
    # regression: an un-attended NYXARA decays to ASLEEP and only the Master/threat can wake
    # her, so presence must only modulate CADENCE — never disable self-initiated action, or
    # the background mind would go dormant exactly when it must stay autonomous.
    from nyxara.kernel.presence import Presence, PresenceState
    core = _code_core()
    pres = Presence(initial=PresenceState.ASLEEP)
    loop = AutonomicLoop(core, interval_s=0.0, decision_mode="code", presence=pres)
    loop._apply_presence()                       # advance the (asleep) arousal state machine
    assert loop.proactive_allowed is True        # low arousal must NOT gate proactivity off
    r = loop.tick_once()
    assert r["acted"] >= 1                        # still decides + acts while "asleep"
    # cadence still reflects the rested state (slower than an engaged tick)
    assert loop._current_interval() == pres.tick_interval()


def test_aprocess_matches_process():
    core = NyxaraCore()

    async def _go():
        return await core.aprocess("how are you?", authority=Authority.OWNER)

    r = asyncio.run(_go())
    assert r.acted and r.candidate.kind == "respond"
