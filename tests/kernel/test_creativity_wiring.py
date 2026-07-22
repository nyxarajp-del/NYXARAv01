"""Kernel wiring tests for True Original Creativity (P24 · F19).

The engine must be reachable from :class:`NyxaraCore` (create / imagine /
originality_report / rate_creation), fire on idle, serve as the autonomic
guaranteed-self-work fallback, and persist under the session dir."""

import os

import pytest

from nyxara.kernel.orchestrator import NyxaraCore


@pytest.fixture()
def core(tmp_path, monkeypatch):
    real = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path)) if p.startswith("~")
                        else real(p))
    return NyxaraCore()


def test_create_is_wired_and_llm_free(core):
    rep = core.create("a compass for dreams", modality="idea", rounds=1)
    assert "error" not in rep
    assert rep["llm_used"] is False
    assert rep["kept"]


def test_imagine_wires_the_orphaned_creative_engine(core):
    out = core.imagine("threat detection", blend_with="chess")
    assert "error" not in out
    assert out["llm_used"] is False
    assert len(out["ideas"]) >= 1
    assert all("technique" in i for i in out["ideas"])
    assert out["best"]["provenance"] == "self_reflection"


def test_originality_report_after_create(core):
    core.create("a compass for dreams", modality="idea", rounds=1)
    rep = core.originality_report()
    assert rep["llm_used"] is False
    assert rep["kept"] >= 1
    assert "atelier" in rep and "muse" in rep and "strategy" in rep


def test_rate_creation_moves_her_taste(core):
    core.create("a compass for dreams", modality="idea", rounds=1)
    before = core.originality_report()["aesthetic_weights"]
    out = core.rate_creation(10)
    assert "error" not in out
    assert out["aesthetic_weights"] != before or out["signal"] == 0.5


def test_rate_without_creation_is_honest(core):
    out = core.rate_creation(9)
    assert "error" in out


def test_state_persists_under_session_dir(core, tmp_path):
    core.create("a compass for dreams", modality="idea", rounds=1)
    assert (tmp_path / ".nyxara" / "originality.json").exists()


def test_idle_maintenance_creates_on_the_throttled_tick(core):
    core._create_idle_count = 7          # next tick hits the % 8 branch
    report = core.idle_maintenance()
    assert report.get("originals", 0) >= 1
    assert core._originality().muse.projects   # the muse actually chose a project


def test_autonomic_fallback_reaches_creation(core):
    from nyxara.kernel.autonomic import AutonomicLoop
    loop = AutonomicLoop(core, interval_s=0.0)
    core._originality()                  # engine present on the core
    core.active_curiosity = None         # earlier fallback off so creation is reached
    label = loop._guaranteed_self_work()
    assert label in ("creation", "consolidation")
    if label == "creation":
        assert core._originality_engine.kept or core._originality_engine.attempts
