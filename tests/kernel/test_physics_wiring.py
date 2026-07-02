"""Tests for the intuitive-physics loop wired into the cognitive cycle.

On each idle pass (when oversight permits) the kernel drives a real
perceive→act→consequence→learn loop inside a rigid-body micro-world
(:class:`nyxara.sim.physics_world.PhysicsAgent`): NYXARA shoves its body — push/lift/drop/poke —
and learns the physics from what actually happens, into a dedicated learned-dynamics model.
These tests assert the agent is built with growth, that idle passes feed it genuine transitions
and surface a ``physics`` report block, and that a paused mind takes no autonomous physics action.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.sim.physics_world import ACTIONS, PhysicsAgent


def test_core_builds_physics_agent_with_growth():
    core = NyxaraCore(enable_growth=True)
    assert isinstance(core.physics_agent, PhysicsAgent)
    # physics owns a DEDICATED model — its 11-D state space is distinct from the 8-D filesystem
    # embodiment, and the per-action learners lock to a single dimensionality
    assert core.physics_agent.world_model is not core.world_model
    if core.real_environment is not None:
        core.real_environment.cleanup()


def test_no_physics_agent_without_growth():
    core = NyxaraCore(enable_growth=False)
    assert getattr(core, "physics_agent", None) is None


def test_idle_feeds_physics_model_real_transitions():
    core = NyxaraCore(enable_growth=True)
    pa = core.physics_agent
    before = len(pa.world_model)
    report = {}
    for _ in range(4):
        report = core.idle_maintenance()
    # the physics model grew from genuine lived transitions
    assert len(pa.world_model) > before
    assert "physics" in report
    assert report["physics"]["action"] in ACTIONS
    # motor effects are now modelled
    assert set(pa.world_model.actions()) & set(ACTIONS)
    if core.real_environment is not None:
        core.real_environment.cleanup()


def test_paused_mind_takes_no_physics_action():
    core = NyxaraCore(enable_growth=True)
    core.pause()
    before = len(core.physics_agent.world_model)
    report = core.idle_maintenance()
    # oversight-gated: a paused mind does not act, even on a simulated body
    assert "physics" not in report
    assert len(core.physics_agent.world_model) == before
    if core.real_environment is not None:
        core.real_environment.cleanup()
