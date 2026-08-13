"""NYX V.001 wiring: she is the primary brain, and the old names point INTO her.

The genre test (like the other ``tests/kernel/test_*_wiring.py``): prove the faculty is present at
boot, that it took the seat it claims, that the brains it merged are reachable under their old
names, and that turning it off leaves the mind exactly as it was.

``test_nyx_and_nyx5_are_views_into_nyx001`` is the one that protects the 131 existing references
and the 73 test files that predate this merge. If those became copies rather than identity, there
would be two of each brain in the process, learning separately, and every one of those callers
would be talking to the wrong one.
"""

from __future__ import annotations

import os

import pytest

from nyxara.kernel.config import reload_settings


@pytest.fixture(autouse=True)
def _clean_env():
    """Isolate each test from NYX V.001 env overrides and reset the settings singleton."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("NYXARA_NYX001__")}
    for k in list(saved):
        os.environ.pop(k, None)
    # toy scale: these test WIRING, not learning capacity, and medium would pay minutes per boot
    # for nothing they assert. tests/nyx001/test_metrics.py is where the proving ground runs.
    os.environ["NYXARA_NYX001__SCALE"] = "toy"
    reload_settings()
    yield
    for k in list(os.environ):
        if k.startswith("NYXARA_NYX001__"):
            os.environ.pop(k, None)
    os.environ.update(saved)
    reload_settings()


def _core():
    from nyxara.kernel.orchestrator import NyxaraCore
    return NyxaraCore()


def _chain(reasoner):
    """Walk the reason-seat chain outermost-first. Bounded; the chain is short by construction."""
    names, node = [], reasoner
    for _ in range(8):
        names.append(type(node).__name__)
        node = getattr(node, "base", None)
        if node is None:
            break
    return names


def test_boots_with_nyx001_present():
    core = _core()
    assert core.nyx001 is not None
    stats = core.nyx001_stats()
    assert stats.get("brains", {}).get("stack") is True


def test_nyx_and_nyx5_are_views_into_nyx001():
    """Identity, not copies — otherwise there are two of each brain learning separately."""
    core = _core()
    assert core.nyx is core.nyx001.v03
    assert core.nyx5 is core.nyx001.snn


def test_nyx001_takes_the_outermost_reason_seat():
    core = _core()
    chain = _chain(core.reasoner)
    assert chain[0] == "NyxV001Reasoner", f"NYX V.001 is not outermost: {chain}"
    assert "NyxReasoner" in chain, f"V.01-.03 fell out of the chain: {chain}"


def test_boots_with_nyx001_disabled():
    """Cleanly absent, and the mind she was merging is left exactly as it was."""
    os.environ["NYXARA_NYX001__ENABLED"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001 is None
    assert core.nyx is not None, "disabling the merge destroyed the brain it was merging"
    assert type(core.reasoner).__name__ != "NyxV001Reasoner"
    r = core.process("hello")
    assert r.disposition is not None


def test_advisory_mode_leaves_the_seat_alone():
    os.environ["NYXARA_NYX001__AS_REASONER"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001 is not None, "the brain should still be built when only the seat is off"
    assert type(core.reasoner).__name__ != "NyxV001Reasoner"


def test_a_turn_still_reaches_a_disposition():
    """NYX V.001 proposes; the unchanged fail-closed gate disposes."""
    core = _core()
    r = core.process("what is two plus two")
    assert r.disposition is not None


def test_disabling_the_layer_stack_leaves_the_other_two_minds():
    os.environ["NYXARA_NYX001__LAYERS_ENABLED"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001 is not None
    assert core.nyx001.stack is None
    assert core.nyx is not None, "the V.01-.03 brain was lost with the layer stack"


def test_core_facade_methods_are_reachable():
    core = _core()
    assert isinstance(core.nyx001_stats(), dict)
    assert isinstance(core.nyx001_develop(), dict)
    assert isinstance(core.nyx001_tick(), dict)
    assert core.nyx001_is_learning() in (True, False, None)
    thought = core.nyx001_think("alpha beta gamma")
    assert thought is not None


def test_facade_methods_degrade_cleanly_when_absent():
    os.environ["NYXARA_NYX001__ENABLED"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001_stats() == {}
    assert core.nyx001_develop() == {}
    assert core.nyx001_think("x") is None
    assert core.nyx001_is_learning() is None


def test_config_defaults_make_her_primary():
    from nyxara.kernel.config import get_settings
    cfg = get_settings().nyx001
    assert cfg.enabled is True
    assert cfg.as_reasoner is True, "she is meant to be the PRIMARY brain by default"
    assert cfg.v03_enabled and cfg.snn_enabled and cfg.layers_enabled


def test_default_scale_is_medium():
    """The scale that was asked for. The fixture overrides it per-test; the default must not."""
    for k in list(os.environ):
        if k.startswith("NYXARA_NYX001__"):
            os.environ.pop(k, None)
    reload_settings()
    from nyxara.kernel.config import get_settings
    assert get_settings().nyx001.scale == "medium"


def test_the_heartbeat_actually_runs_in_the_live_loop():
    """``nyx001_tick`` existed and had no caller anywhere in the process.

    The dark core never pulsed, compression and meta-learning never ran outside a test, and the
    curriculum only re-measured on the every-16th-turn path inside ``think`` — so the three
    faculties that make NYX V.001 more than three brains in a bag were off in practice while
    every config flag said they were on.
    """
    core = _core()
    for i in range(4):
        core.process(f"alpha beta gamma {i}")
    before = core.nyx001.dark.pulses
    core.idle_maintenance()
    assert core.nyx001.dark.pulses > before, "idle maintenance did not beat NYX V.001's heart"


def test_the_heartbeat_does_not_beat_v03_twice():
    """Idle maintenance ticks core.nyx directly for its wondering report, then ticks NYX V.001."""
    core = _core()
    calls = []
    original = core.nyx001.v03.tick

    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    core.nyx001.v03.tick = counted
    core.idle_maintenance()
    assert len(calls) <= 1, f"V.01-.03 deliberated {len(calls)} times in one idle pass"


def test_track_b_is_reachable_from_a_booted_core():
    """The four Track B modules were constructed by nothing: no config gate, no owner, no route."""
    core = _core()
    cortex = core.nyx001.stack.lingua
    assert cortex is not None, "Track B is not wired into the stack"
    for i in range(8):
        core.nyx001.stack.cycle(f"alpha beta gamma delta {i}")
    st = cortex.stats()
    assert st["reads"] > 0, "Track B is present but the cycle never feeds it"
    assert st["tokenizer"]["observed"] > 0
    assert st["grounding"]["groundings"] > 0, "words were never bound to a concept"


def test_track_b_can_be_turned_off_cleanly():
    os.environ["NYXARA_NYX001__LINGUA_ENABLED"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001 is not None
    assert core.nyx001.stack.lingua is None
    assert core.nyx001.stack.cycle("alpha beta").failed == [], "losing Track B broke the stack"


def test_track_b_is_not_counted_as_an_eighteenth_layer():
    """Track A is Layers 0-17; Track B is a modality. The charter keeps them apart."""
    core = _core()
    stack = core.nyx001.stack
    assert "lingua" not in stack.layers()
    assert stack.stats()["active_layers"] == 18


def test_language_refuses_to_run_ahead_of_concepts():
    """Environment → Perception → Concepts → World Model → Language, enforced not documented."""
    from nyxara.nyx001.layers.l00_substrate import Substrate
    from nyxara.nyx001.lingua.cortex import LanguageCortex

    cortex = LanguageCortex(Substrate(seed=3, rung="toy"))
    for _ in range(10):
        cortex.read("alpha beta gamma", concept=None)      # no concept has formed yet
    assert cortex.vocabulary() == [], "grounded a word with no concept to ground it in"
    assert cortex.meaning("alpha") is None


def test_the_tick_config_is_actually_read():
    """tick_enabled/consolidate_every_s/develop_every_s were declared and read by nothing."""
    os.environ["NYXARA_NYX001__TICK_ENABLED"] = "false"
    reload_settings()
    core = _core()
    assert core.nyx001 is not None
    assert core.nyx001_tick() == {}, "the heartbeat ran with tick_enabled=false"


def test_tick_intervals_come_from_config():
    os.environ["NYXARA_NYX001__CONSOLIDATE_EVERY_S"] = "123.0"
    os.environ["NYXARA_NYX001__DEVELOP_EVERY_S"] = "456.0"
    reload_settings()
    core = _core()
    assert core.nyx001.consolidate_every_s == 123.0
    assert core.nyx001.develop_every_s == 456.0
