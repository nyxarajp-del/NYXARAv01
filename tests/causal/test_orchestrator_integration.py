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


def test_the_knot_gate_actually_runs_on_a_real_turn():
    """CAUSAL·2 was built, tested and then never called: the hot path passed no claims at all,
    so a Knot Mutation Failure could not happen in production no matter what was said."""
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    first = core._last_causal
    assert first.committed == 1                 # the claim was mined and tied in

    thoughts: list = []
    core._causal_engage("Heavy rain prevents flooding in the valley.", thoughts)
    second = core._last_causal
    assert second.consistent is False           # contradicts the previous turn
    assert second.abstain is True
    assert thoughts                             # and she says so in the workspace


def test_turns_without_a_causal_claim_stay_clean():
    core = _core()
    assert core._causal_claims("hello, how are you today") == []
    core._causal_engage("hello, how are you today", [])
    assert core._last_causal.consistent is True
    assert core._last_causal.knot_check is None


def test_the_turn_gate_can_be_switched_off(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.knot_gate_on_turns = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core._causal_claims("Heavy rain causes flooding.") == []


def test_report_surfaces_the_engine_and_its_last_turn():
    """``_last_causal`` was written every turn and read by nothing — the comment beside it
    claimed report() surfaced it, and report() had never mentioned the engine at all."""
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    core._causal_engage("Heavy rain prevents flooding in the valley.", [])

    causal = core.report().get("causal")
    assert causal is not None
    assert causal["knot_gate_abstains"] is True
    assert causal["lattice"]["knots"] == 1
    assert causal["last_turn"]["consistent"] is False
    assert causal["last_turn"]["abstain"] is True


def test_report_omits_causal_when_the_engine_is_off(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert "causal" not in core.report()


def test_config_flag_disables_the_engine(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core.causal_engine is None
    # the hot-path hook is a no-op when the engine is off — nothing raised, nothing recorded
    core._causal_engage("hello", [])
