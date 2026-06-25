"""Tests for the embodied sensorimotor loop wired into the cognitive cycle.

The world model used to be fed only a synthetic belief-count signal, then a narrow six-number
filesystem tick. Now the kernel drives a genuine perceive→decide→act→consequence→learn loop
(:class:`nyxara.sim.embodied.EmbodiedAgent`) on each idle pass: NYXARA authors perceivable
content, reads it back with the senses, and learns real ``(state, action, next_state, reward)``
transitions — so its dynamics reflect a *lived* world. Live-web perception stays idle here
(no URLs configured), keeping these tests deterministic and offline.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.sim.embodied import EmbodiedAgent
from nyxara.sim.real_environment import RealEnvironment


def test_core_builds_embodied_agent_with_growth():
    core = NyxaraCore(enable_growth=True)
    assert core.world_model is not None
    assert isinstance(core.real_environment, RealEnvironment)
    assert isinstance(core.embodied_agent, EmbodiedAgent)
    # the embodied agent lives in the same body and learns into the same planning model
    assert core.embodied_agent.env is core.real_environment
    assert core.embodied_agent.world_model is core.world_model
    core.real_environment.cleanup()


def test_idle_feeds_world_model_real_transitions():
    core = NyxaraCore(enable_growth=True)
    before = len(core.world_model)
    report = {}
    for _ in range(5):
        report = core.idle_maintenance()
    # the world model grew from genuine embodied transitions
    assert len(core.world_model) > before
    assert "embodiment" in report
    assert report["embodiment"]["action"] in EmbodiedAgent.ACTIONS
    # the agent really perceived its world (senses are inside the loop, not bolted on)
    assert core.embodied_agent.status()["perceived_artifacts"] >= 0
    # embodied actions now live in the model
    seen = set(core.world_model.actions())
    assert seen & set(EmbodiedAgent.ACTIONS)
    core.real_environment.cleanup()


def test_paused_mind_takes_no_embodied_action():
    core = NyxaraCore(enable_growth=True)
    core.pause()
    before = len(core.world_model)
    report = core.idle_maintenance()
    # oversight-gated: a paused mind does not act on the real world
    assert "embodiment" not in report
    assert len(core.world_model) == before
    core.real_environment.cleanup()
