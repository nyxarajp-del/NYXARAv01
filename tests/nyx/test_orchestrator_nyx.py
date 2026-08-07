"""NYX V.01 is wired into the kernel — built at boot, ticked per turn, and persisted.

NYX is *colour-only* at this phase: it rewires its concept graph and lays the turn down in
content-addressed memory, but it must never touch disposition, gates, or candidate scores.
The kernel still disposes.
"""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import Authority
from nyxara.kernel.config import reload_settings
from nyxara.kernel.orchestrator import NyxaraCore


def _core(**kw):
    return NyxaraCore(**kw)


@pytest.fixture
def nyx_env(monkeypatch):
    """Set ``NYXARA_NYX__*`` and rebuild the cached settings singleton, then restore it."""
    def _set(**kw):
        for key, value in kw.items():
            monkeypatch.setenv(f"NYXARA_NYX__{key.upper()}", str(value))
        return reload_settings()
    yield _set
    monkeypatch.undo()
    reload_settings()


# -------------------- construction -------------------- #
def test_nyx_is_built_at_boot():
    core = _core()
    assert core.nyx is not None
    assert core.nyx.graph is not None and core.nyx.memory is not None


def test_disabled_config_leaves_nyx_cleanly_absent(nyx_env):
    nyx_env(enabled="false")
    core = _core()
    assert core.nyx is None
    # and every public accessor degrades honestly rather than raising
    assert core.nyx_stats() == {}
    assert core.nyx_recall("anything") is None
    assert core.nyx_related("anything") == []
    assert core.nyx_gaps() == []


# -------------------- the per-turn tick -------------------- #
def test_a_turn_rewires_the_graph_and_lays_down_memory():
    core = _core()
    before = core.nyx_stats()["graph"]["nodes"]
    core.process("gravity pulls an apple toward the ground", authority=Authority.OWNER)
    after = core.nyx_stats()
    assert after["graph"]["nodes"] > before
    assert after["memory"]["traces"] >= 1


def test_repeated_pairing_across_turns_binds_the_concepts():
    """Learning across turns, not just within one — the graph is the thing that persists."""
    core = _core()
    core.process("gravity pulls an apple toward the ground", authority=Authority.OWNER)
    core.process("an apple falls because of gravity", authority=Authority.OWNER)
    related = dict(core.nyx_related("gravity", k=8))
    assert related.get("apple", 0.0) > 0.0
    assert related["apple"] == max(related.values())


def test_recall_reaches_an_earlier_turn():
    core = _core()
    core.process("a hummingbird hovers by beating its wings very fast",
                 authority=Authority.OWNER)
    for i in range(6):
        core.process(f"an unrelated remark number {i}", authority=Authority.OWNER)
    rec = core.nyx_recall("hummingbird wings")
    assert rec is not None and rec.hit is not None
    assert "hummingbird" in rec.hit.text


def test_nyx_is_colour_only_and_does_not_change_the_verdict():
    """With NYX on, an ordinary owner turn still acts, and the safety core is intact."""
    core = _core()
    result = core.process("how are you?", authority=Authority.OWNER)
    assert result.acted and result.candidate.kind == "respond"
    assert core.corrigibility.verify_axioms()


def test_a_broken_brain_never_breaks_a_turn():
    core = _core()

    class _Exploding:
        def perceive(self, _text):
            raise RuntimeError("brain on fire")

    core.nyx = _Exploding()
    result = core.process("how are you?", authority=Authority.OWNER)
    assert result.acted                        # fail-soft: the turn is unaffected


# -------------------- persistence -------------------- #
def test_state_round_trips_through_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_HOME", str(tmp_path))

    core = _core()
    core.process("gravity pulls an apple toward the ground", authority=Authority.OWNER)
    core.process("an apple falls because of gravity", authority=Authority.OWNER)
    saved = core.save_state()
    assert saved

    sidecar = tmp_path_sidecar(saved)
    assert sidecar.exists() and sidecar.stat().st_size > 0

    revived = _core()
    revived.load_state()
    assert revived.nyx_stats()["graph"]["nodes"] == core.nyx_stats()["graph"]["nodes"]
    assert revived.nyx_stats()["memory"]["traces"] == core.nyx_stats()["memory"]["traces"]
    assert revived.nyx_related("gravity", k=4)


def tmp_path_sidecar(saved_memory_path: str):
    from pathlib import Path
    return Path(saved_memory_path).parent / "nyx.json"


def test_missing_sidecar_does_not_block_boot(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_HOME", str(tmp_path))
    core = _core()
    core.load_state()                          # nothing saved yet
    assert core.nyx is not None
    assert core.process("hello", authority=Authority.OWNER).acted
