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
    # for nothing they assert. tests/nyx001/test_proving_ground.py is where scale is exercised.
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
