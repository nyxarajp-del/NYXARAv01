"""Integration: the CAUSAL engine is wired into NyxaraCore and the heartbeat, safely.

These exercise the *wiring* NYXARA adds (engine build, the per-turn hot-path hook, the heartbeat
thermodynamic tick, and the config flag) without driving the full heavy reasoning pipeline — that
keeps them fast and deterministic while still proving the integration points.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings


def _core():
    from nyxara.kernel.orchestrator import NyxaraCore
    return NyxaraCore()


def test_engine_is_built_and_seeded():
    core = _core()
    assert core.causal_engine is not None
    # seeded standing concepts mean a matching query resonates immediately (no cold start)
    hit = core.causal_engine.field.best("help me fix a python code bug in a function")
    assert hit is not None and hit.label == "code"


def test_hot_path_hook_resonates_and_records():
    core = _core()
    thoughts: list = []
    core._causal_engage("reasoning logic and inference to solve a problem", thoughts)
    et = core._last_causal
    assert et is not None
    assert et.resonance                                   # a seeded concept resonated
    assert thoughts                                       # an advisory thought was recorded


def test_hot_path_hook_phase_shifts_on_a_novel_gap():
    core = _core()
    thoughts: list = []
    core._causal_engage("qwibble frobnicate the zorbtar quuxatron gribble", thoughts)
    et = core._last_causal
    assert et.gap is True
    assert et.phase_shift is not None and et.phase_shift["installed"] is True


def test_hot_path_hook_never_raises_even_if_engine_is_broken():
    core = _core()

    class _Boom:
        def turn(self, *a, **k):
            raise RuntimeError("engine exploded")
    core.causal_engine = _Boom()
    # the helper must swallow any engine failure — a turn is never broken by the engine
    core._causal_engage("anything at all", [])


def test_heartbeat_beat_ticks_the_thermodynamic_monitor():
    core = _core()
    if core.heartbeat is None:
        pytest.skip("heartbeat unavailable in this build")
    before = core.causal_engine.thermo.status()["beats"]
    core.heartbeat.beat(dt=1.0)
    after = core.causal_engine.thermo.status()["beats"]
    assert after == before + 1


def test_config_flag_disables_the_engine(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core.causal_engine is None
    # the hot-path hook is a no-op when the engine is off — nothing raised, nothing recorded
    core._causal_engage("hello", [])
