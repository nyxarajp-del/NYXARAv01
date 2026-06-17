"""Tests for the real-environment sensorimotor loop wired into the cognitive cycle.

The world model used to be fed only a synthetic belief-count signal. Now the kernel
drives a genuine (state, action, next_state, reward) transition from a real scratch-dir
environment on each idle pass, so its learned dynamics reflect a real environment.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.sim.real_environment import RealEnvironment


def test_core_builds_real_environment_with_growth():
    core = NyxaraCore(enable_growth=True)
    assert core.world_model is not None
    assert isinstance(core.real_environment, RealEnvironment)
    core.real_environment.cleanup()


def test_idle_feeds_world_model_real_transitions():
    core = NyxaraCore(enable_growth=True)
    before = len(core.world_model)
    report = {}
    for _ in range(5):
        report = core.idle_maintenance()
    # the world model grew from genuine sensorimotor transitions
    assert len(core.world_model) > before
    assert "sensorimotor" in report
    assert report["sensorimotor"]["action"] in RealEnvironment.ACTIONS
    # real-environment actions now live in the model, alongside the epistemic one
    seen = set(core.world_model.actions())
    assert seen & set(RealEnvironment.ACTIONS)
    core.real_environment.cleanup()


def test_paused_mind_takes_no_sensorimotor_action():
    core = NyxaraCore(enable_growth=True)
    core.pause()
    before = len(core.world_model)
    report = core.idle_maintenance()
    # oversight-gated: a paused mind does not act on the real filesystem
    assert "sensorimotor" not in report
    assert len(core.world_model) == before
    core.real_environment.cleanup()
