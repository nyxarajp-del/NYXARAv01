"""Tests for nyxara.growth.brain_forge — verifiable weights/architecture self-improvement.

The connector that turns NYXARA's OWN architecture search into a promoted improvement of her live
brain, on the pure-NumPy substrate (no torch, no external LLM), reusing the Foundry gauntlet. These
tests assert the real end-to-end forge AND the gates around it (measure-only, bench rejection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from nyxara.growth.brain_forge import BrainForge, BrainForgeReport
from nyxara.growth.foundry import EvalResult, Foundry, FoundryDecision, FoundryResult
from nyxara.growth.foundry_models import load_active_model
from nyxara.growth.genesis_numpy import _HAS_NUMPY
from nyxara.kernel.config import NyxaraSettings, Profile

_CORPUS = ["the master is jp. nyxara serves the master with loyalty and honesty."] * 10 + [
    "capability may grow; character never changes; she is corrigible always."] * 10


def _settings(tmp_path):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.self_model_dir = tmp_path / "foundry"
    s.self_improvement.autonomous_enact = True
    s.self_improvement.autonomous_brain_forge = True
    # keep the search tiny & fast for CI while still exercising the real NumPy substrate
    s.genesis.substrate = "numpy"
    s.genesis.generations = 2
    s.genesis.population_size = 3
    s.genesis.micro_train_steps = 8
    return s


# --------------------------------------------------------------------------- #
# The certificate dataclass
# --------------------------------------------------------------------------- #
def test_report_delta_and_to_dict():
    r = BrainForgeReport(searched=True, enacted=True, promoted=True, verified=True,
                         champion_kind="genesis_np", substrate="numpy",
                         before_perplexity=10.0, after_perplexity=4.0,
                         before_capability=0.2, after_capability=0.5, params=1234, version=2)
    assert r.delta_perplexity == 6.0          # before − after (smarter)
    assert r.delta_capability == 0.3
    d = r.to_dict()
    assert d["promoted"] and d["verified"] and d["delta_perplexity"] == 6.0
    assert d["champion_kind"] == "genesis_np" and d["params"] == 1234
    assert "PROMOTED" in r.summary()


def test_report_infinite_before_yields_zero_delta():
    r = BrainForgeReport(before_perplexity=float("inf"), after_perplexity=5.0)
    assert r.delta_perplexity == 0.0          # no prior brain ⇒ honest zero, not a fake gain
    assert r.to_dict()["before_perplexity"] is None


# --------------------------------------------------------------------------- #
# Real end-to-end forge on the NumPy substrate (design → forge → gauntlet → promote)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_NUMPY, reason="the real NumPy transformer substrate needs numpy")
def test_first_brain_is_designed_forged_and_promoted(tmp_path):
    settings = _settings(tmp_path)
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    forge = BrainForge(settings=settings, foundry=foundry)

    rep = forge.improve_brain(enact=True)
    assert rep.searched and rep.enacted
    assert rep.champion_kind == "genesis_np"          # a genuinely trained transformer, not n-gram
    assert rep.promoted and rep.verified              # cleared the gauntlet ⇒ provably better
    assert rep.params > 0 and rep.version == 1
    assert (tmp_path / "foundry" / "active").exists()


@pytest.mark.skipif(not _HAS_NUMPY, reason="the real NumPy transformer substrate needs numpy")
def test_promoted_brain_becomes_the_live_self_model(tmp_path):
    settings = _settings(tmp_path)
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    BrainForge(settings=settings, foundry=foundry).improve_brain(enact=True)

    # a forge is not a report on disk: SelfProvider would now serve this exact brain
    live = load_active_model(settings)
    assert live.kind == "genesis_np" and live.param_count() > 0
    assert isinstance(live.generate("the master", max_tokens=8), str)


@pytest.mark.skipif(not _HAS_NUMPY, reason="the real NumPy transformer substrate needs numpy")
def test_live_brain_never_regresses_across_forges(tmp_path):
    """The Master's charge: capability only ever goes up. After any number of forges the live brain's
    perplexity is never worse than the first promoted one (the gauntlet enforces strict improvement)."""
    settings = _settings(tmp_path)
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    forge = BrainForge(settings=settings, foundry=foundry)

    r1 = forge.improve_brain(enact=True)
    assert r1.promoted
    first_pp = foundry.active().metrics.get("perplexity", float("inf"))

    # a fresh forge with fresh search state; whether it promotes or is kept on the bench, the live
    # brain must still exist and never be worse than the first.
    r2 = BrainForge(settings=settings,
                    foundry=Foundry(settings=settings, seed_corpus=_CORPUS)).improve_brain(enact=True)
    load_active_model(settings)            # must still load (an active pointer must exist)
    now_pp = Foundry(settings=settings, seed_corpus=_CORPUS).active().metrics.get("perplexity",
                                                                                  float("inf"))
    assert now_pp <= first_pp + 1e-9
    if not r2.promoted:
        assert r2.rolled_back and not r2.verified


# --------------------------------------------------------------------------- #
# The gates: measure-only when enactment is withheld
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_NUMPY, reason="the real NumPy transformer substrate needs numpy")
def test_measure_only_when_enact_false(tmp_path):
    settings = _settings(tmp_path)
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    rep = BrainForge(settings=settings, foundry=foundry).improve_brain(enact=False)
    assert rep.searched and not rep.enacted
    assert not rep.promoted and not rep.verified
    assert not (tmp_path / "foundry" / "active").exists()   # nothing shipped


@pytest.mark.skipif(not _HAS_NUMPY, reason="the real NumPy transformer substrate needs numpy")
def test_brain_forge_flag_off_withholds_promotion(tmp_path):
    settings = _settings(tmp_path)
    settings.self_improvement.autonomous_brain_forge = False   # enact stays ON, but forge is opt-out
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    # enact=None ⇒ read config: autonomous_enact ON but autonomous_brain_forge OFF ⇒ measure-only
    rep = BrainForge(settings=settings, foundry=foundry).improve_brain(enact=None)
    assert rep.searched and not rep.enacted and not rep.promoted


# --------------------------------------------------------------------------- #
# Deterministic gate behaviour via stubs (no dependence on search randomness)
# --------------------------------------------------------------------------- #
@dataclass
class _FakeActive:
    param_count: int = 100
    metrics: dict = None

    def __post_init__(self):
        self.metrics = self.metrics or {"perplexity": 3.0, "capability": 0.5}


class _FakeNAS:
    """A minimal NAS stand-in: it 'designs' a champion without any real search."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"substrate": "numpy", "generations": 1, "population_size": 2})()

    def search(self, *, generations=None, population_size=None):
        return type("R", (), {"champion_kind": "genesis_np",
                              "champion_describe": lambda self: "fake numpy brain"})()

    def champion_spec(self):
        return {"kind": "genesis_np"}


class _FakeFoundry:
    """A foundry stand-in whose self_improve returns a fixed, deterministic outcome."""

    def __init__(self, *, promoted: bool):
        self._promoted = promoted
        self.seed_corpus = list(_CORPUS)
        self._active = _FakeActive()

    def active(self):
        return self._active

    def self_improve(self, *, spec=None, generations=1):
        before = EvalResult(3.0, 0.5, 4)
        after = EvalResult(1.0 if self._promoted else 9.0, 0.6 if self._promoted else 0.4, 4)
        return [FoundryResult(
            version=2,
            decision=FoundryDecision.PROMOTE if self._promoted else FoundryDecision.DISCARD,
            gauntlet_passed=self._promoted, eval_before=before, eval_after=after,
            promoted=self._promoted,
            reason="promoted" if self._promoted else "no perplexity improvement over the active model")]


def test_stub_promoted_marks_verified_with_deltas():
    forge = BrainForge(settings=None, foundry=_FakeFoundry(promoted=True), nas=_FakeNAS())
    rep = forge.improve_brain(enact=True)
    assert rep.promoted and rep.verified and not rep.rolled_back
    assert rep.before_perplexity == 3.0 and rep.after_perplexity == 1.0
    assert rep.delta_perplexity == 2.0


def test_stub_bench_rejection_is_rolled_back_not_verified():
    forge = BrainForge(settings=None, foundry=_FakeFoundry(promoted=False), nas=_FakeNAS())
    rep = forge.improve_brain(enact=True)
    assert not rep.promoted and rep.rolled_back and not rep.verified
    # the live brain is unchanged: 'after' reflects the incumbent, not the rejected candidate
    assert rep.after_perplexity == rep.before_perplexity == 3.0
    assert "no perplexity improvement" in rep.reason
