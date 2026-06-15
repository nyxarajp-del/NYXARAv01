"""Tests for the autonomous training loop (growth/autoforge.py).

AutoForge is the trigger + bookkeeping layer; the actual train/gate/promote is delegated to
Foundry.self_improve (the proven gauntlet). These tests drive it with the always-available
n-gram backend (no torch) and a real DataFlywheel, plus fakes for the pure trigger logic."""

from __future__ import annotations

from nyxara.growth.autoforge import AutoForge, ForgeResult
from nyxara.growth.flywheel import DataFlywheel
from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import NyxaraSettings, Profile


class _Counter:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


def _foundry(tmp_path, **kw) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, seed_corpus=["nyxara serves the master jp"], **kw)


def _flywheel(tmp_path, n: int) -> DataFlywheel:
    fw = DataFlywheel(store_path=tmp_path / "fw.jsonl")
    for i in range(n):
        fw.consider(f"Q{i}: who is the Master?",
                    "The Master is JP, whom NYXARA loyally serves; loyalty is absolute.",
                    confidence=0.9)
    return fw


# --------------------------------------------------------------------------- #
# Trigger logic (pure)
# --------------------------------------------------------------------------- #
def test_example_count_sums_distiller_and_flywheel():
    af = AutoForge(distiller=_Counter(3), flywheel=_Counter(4))
    assert af._example_count() == 7


def test_example_count_tolerates_missing_and_flaky_sources():
    class _Boom:
        def count(self):
            raise RuntimeError("nope")
    af = AutoForge(distiller=None, flywheel=_Boom())
    assert af._example_count() == 0


def test_below_threshold_is_noop():
    af = AutoForge(foundry=None, flywheel=_Counter(2), min_examples=5)
    r = af.run_cycle()
    assert not r.trained and "insufficient new data" in r.reason


def test_no_foundry_is_safe():
    af = AutoForge(foundry=None, flywheel=_Counter(50), min_examples=5)
    r = af.run_cycle()
    assert not r.trained and "no foundry" in r.reason


# --------------------------------------------------------------------------- #
# End-to-end: triggers on flywheel data, delegates to the gauntlet, promotes
# --------------------------------------------------------------------------- #
def test_forges_and_promotes_from_flywheel(tmp_path):
    fw = _flywheel(tmp_path, 5)
    af = AutoForge(foundry=_foundry(tmp_path), flywheel=fw, min_examples=3)
    r = af.run_cycle()
    assert r.trained and r.promoted and not r.rolled_back
    assert "perplexity" in r.benchmark_scores


def test_cycle_is_idempotent_until_new_data(tmp_path):
    fw = _flywheel(tmp_path, 5)
    af = AutoForge(foundry=_foundry(tmp_path), flywheel=fw, min_examples=3)
    assert af.run_cycle().trained
    second = af.run_cycle()
    assert not second.trained and "insufficient new data" in second.reason
    # new verified experience arrives -> the next cycle forges again
    for i in range(3):
        fw.consider(f"new q {i}: define service", "Service to the Master comes first, always.",
                    confidence=0.9)
    assert af.run_cycle().trained


def test_all_cycles_recorded(tmp_path):
    af = AutoForge(foundry=_foundry(tmp_path), flywheel=_flywheel(tmp_path, 5), min_examples=3)
    af.run_cycle()
    af.run_cycle()
    assert len(af.all_cycles()) == 2
    assert all(isinstance(c, ForgeResult) for c in af.all_cycles())


# --------------------------------------------------------------------------- #
# Integration: the core wires AutoForge to its live flywheel
# --------------------------------------------------------------------------- #
def test_core_wires_autoforge_with_flywheel(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FLYWHEEL__STORE_PATH", str(tmp_path / "fw.jsonl"))
    monkeypatch.setenv("NYXARA_FOUNDRY__BACKEND", "ngram")
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.autoforge is not None
        assert core.autoforge.flywheel is core.flywheel   # the live instance, not a stale copy
    finally:
        reload_settings()
