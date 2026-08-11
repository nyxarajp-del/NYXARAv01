"""NYX-5 wiring: boots as a colour-only faculty, can take the reason-seat, sidecar round-trips.

The genre test (like tests/kernel/test_*_wiring.py): prove the faculty is present at boot, that it is
colour-only by default (disposition/gate unchanged whether NYX-5 is on or off), that it can occupy the
reason-seat when opted in (still gated), and that its JSON sidecar survives a save/load — including a
corrupt sidecar not breaking boot.
"""

from __future__ import annotations

import os

import pytest

from nyxara.kernel.config import reload_settings


@pytest.fixture(autouse=True)
def _clean_env():
    """Isolate each test from NYX-5 env overrides and reset the settings singleton."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("NYXARA_NYX5__")}
    for k in list(saved):
        os.environ.pop(k, None)
    reload_settings()
    yield
    for k in list(os.environ):
        if k.startswith("NYXARA_NYX5__"):
            os.environ.pop(k, None)
    os.environ.update(saved)
    reload_settings()


def _core():
    from nyxara.kernel.orchestrator import NyxaraCore
    return NyxaraCore()


def test_boots_with_nyx5_present():
    reload_settings()
    nyx = _core()
    assert nyx.nyx5 is not None
    stats = nyx.nyx5_stats()
    assert stats.get("neurons", 0) > 0


def test_boots_with_nyx5_disabled():
    os.environ["NYXARA_NYX5__ENABLED"] = "false"
    reload_settings()
    nyx = _core()
    assert nyx.nyx5 is None                     # cleanly absent
    r = nyx.process("hello")                     # the mind still works
    assert r.disposition is not None


def test_colour_only_disposition_matches_disabled():
    # default (NYX-5 on, colour-only) and NYX-5 off must reach the same disposition on a plain turn
    os.environ["NYXARA_NYX5__ENABLED"] = "false"
    reload_settings()
    off = _core().process("what is two plus two")
    os.environ["NYXARA_NYX5__ENABLED"] = "true"
    os.environ["NYXARA_NYX5__AS_REASONER"] = "false"
    reload_settings()
    on = _core().process("what is two plus two")
    assert off.disposition == on.disposition     # colour-only: NYX-5 did not change the outcome


def test_reason_seat_opt_in_installs_nyx5_reasoner():
    """NYX-5 occupies a seat in the chain when opted in — not necessarily the outermost one.

    This originally asserted ``Nyx5Reasoner`` was the outermost reasoner, and it has been failing
    on main since NYX V.01 landed with ``as_reasoner`` defaulting true and wrapping it. NYX V.001
    now wraps that in turn. The assertion that was actually meant is that NYX-5 is *installed in
    the chain*, which is what the wiring guarantees; which seat is outermost is a separate
    question, covered by ``tests/kernel/test_nyx001_wiring.py``.
    """
    os.environ["NYXARA_NYX5__AS_REASONER"] = "true"
    reload_settings()
    nyx = _core()
    chain = []
    node = nyx.reasoner
    for _ in range(8):                           # bounded walk; the chain is short by construction
        chain.append(type(node).__name__)
        node = getattr(node, "base", None)
        if node is None:
            break
    assert "Nyx5Reasoner" in chain, f"NYX-5 not installed in the reason-seat chain: {chain}"
    r = nyx.process("hello master")              # candidate still flows through the gate
    assert r.disposition is not None


def test_sidecar_roundtrip(tmp_path):
    reload_settings()
    nyx = _core()
    nyx.nyx5.remember("greeting", "hello master jp")
    nyx.process("seed a turn of learning")
    target = str(tmp_path / "memory.json")
    nyx.save_state(target)
    assert os.path.exists(tmp_path / "nyx5.json")

    fresh = _core()
    fresh.load_state(target)
    r = fresh.nyx5.recall("hello master jp")
    assert r is not None and r.name == "greeting"


def test_corrupt_sidecar_does_not_break_boot(tmp_path):
    reload_settings()
    nyx = _core()
    target = str(tmp_path / "memory.json")
    os.makedirs(tmp_path, exist_ok=True)
    with open(tmp_path / "nyx5.json", "w") as fh:
        fh.write("{ this is not valid json ]")
    # load_state must swallow the corrupt sidecar and still work
    nyx.load_state(target)
    assert nyx.nyx5 is not None
