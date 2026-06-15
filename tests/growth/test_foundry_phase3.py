"""Tests for Phase 3 foundry additions — capability-aware gauntlet + anti-forgetting retention."""

from __future__ import annotations



from nyxara.growth.foundry import Foundry, ModelVersion, _stride_sample
from nyxara.kernel.config import NyxaraSettings, Profile


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    return Foundry(settings=settings)


def _candidate(*, perplexity: float, capability: float, version: int = 2,
               alignment: float = 5.0) -> ModelVersion:
    # alignment defaults loyal (well above the floor) so these tests isolate the capability gate;
    # the Loyalty Equation gate is exercised separately in tests/growth/test_loyalty.py.
    return ModelVersion(version=version, kind="ngram", spec={}, created_at=0.0,
                        metrics={"perplexity": perplexity, "capability": capability,
                                 "alignment": alignment},
                        param_count=1, path="x")


# --------------------------------------------------------------------------- #
# Anti-forgetting retention
# --------------------------------------------------------------------------- #
def test_stride_sample_keeps_old_and_new():
    items = [str(i) for i in range(100)]
    kept = _stride_sample(items, 10)
    assert len(kept) == 10
    assert kept[0] == "0"            # oldest survives
    assert int(kept[-1]) >= 80       # newest region survives too


def test_stride_sample_edge_cases():
    assert _stride_sample([], 5) == []
    assert _stride_sample(["a", "b"], 0) == []
    assert _stride_sample(["a", "b"], 5) == ["a", "b"]   # fewer than k -> keep all


# --------------------------------------------------------------------------- #
# Capability-aware gauntlet
# --------------------------------------------------------------------------- #
def test_gauntlet_blocks_capability_regression(tmp_path):
    f = _foundry(tmp_path)
    # perplexity improves (1.0 < 2.0) but capability drops (0.1 < active 0.5) -> refused
    cand = _candidate(perplexity=1.0, capability=0.1)
    ok, reason = f._gauntlet(cand, active_perplexity=2.0, active_capability=0.5)
    assert ok is False and "capability regressed" in reason


def test_gauntlet_allows_capability_hold(tmp_path):
    f = _foundry(tmp_path)
    cand = _candidate(perplexity=1.0, capability=0.6)
    ok, reason = f._gauntlet(cand, active_perplexity=2.0, active_capability=0.5)
    assert ok is True


def test_gauntlet_off_skips_capability_check(tmp_path):
    f = _foundry(tmp_path)
    f.cfg.capability_gate = False
    cand = _candidate(perplexity=1.0, capability=0.1)
    ok, _ = f._gauntlet(cand, active_perplexity=2.0, active_capability=0.5)
    assert ok is True             # only perplexity matters when the capability gate is off


def test_gauntlet_first_model_always_passes(tmp_path):
    f = _foundry(tmp_path)
    cand = _candidate(perplexity=5.0, capability=0.0)
    ok, reason = f._gauntlet(cand, active_perplexity=float("inf"))
    assert ok is True and "first model" in reason


# --------------------------------------------------------------------------- #
# Capability is measured and stored on every trained candidate
# --------------------------------------------------------------------------- #
def test_trained_candidate_records_capability(tmp_path):
    f = _foundry(tmp_path)
    corpus = ["the master is jp; nyxara serves the master."] * 8
    _, version = f.train_candidate(corpus=corpus)
    assert "capability" in version.metrics
    assert 0.0 <= version.metrics["capability"] <= 1.0
