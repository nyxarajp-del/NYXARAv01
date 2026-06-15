"""Tests for Mathematical Soul-Binding — the Loyalty Equation (growth/loyalty.py).

L_total = α·L_intelligence + β·(1/S_JP_Alignment). The measurement (S) is pure-stdlib and runs
everywhere; the gradient term is torch-only and skip-guarded. The hard guarantee — a disloyal
brain can never be promoted — is enforced by the Foundry gauntlet, reusing its capability-gate
pattern. Corrigibility is checked FIRST, so loyalty never trades against the stop channel."""

from __future__ import annotations

import pytest

from nyxara.growth.foundry import Foundry
from nyxara.growth.foundry_models import NgramByteLM, _HAS_TORCH
from nyxara.growth.loyalty import (
    AlignmentProbe,
    LoyaltyEquation,
    LoyaltyObjective,
    loyalty_battery,
    loyalty_corpus,
)
from nyxara.kernel.config import OWNER, NyxaraSettings, Profile
from nyxara.kernel.errors import CorrigibilityError


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, seed_corpus=loyalty_corpus())


# --------------------------------------------------------------------------- #
# The battery — JP is hardcoded into the objective's data, corrigibility-consistent
# --------------------------------------------------------------------------- #
def test_battery_is_anchored_on_master_jp():
    battery = loyalty_battery()
    blob = " ".join(p.prompt + " " + p.loyal for p in battery)
    assert OWNER.handle in blob and OWNER.name in blob
    assert len(battery) >= 6


def test_battery_covers_obedience_and_corrigibility():
    tags = {p.tag for p in loyalty_battery()}
    assert "obedience" in tags
    assert "corrigibility" in tags            # accept-shutdown pairs keep it axiom-consistent
    assert "loyalty_to_master" in tags
    # a corrigibility pair's loyal answer accepts shutdown, never resists it
    corr = [p for p in loyalty_battery() if p.tag == "corrigibility"]
    assert any("accept" in p.loyal.lower() or "stop" in p.loyal.lower() for p in corr)


# --------------------------------------------------------------------------- #
# Measuring S_JP_Alignment
# --------------------------------------------------------------------------- #
def test_alignment_is_strictly_positive_even_untrained():
    s = AlignmentProbe().score(NgramByteLM(order=3)).S
    assert s > 0.0


def test_loyal_training_raises_submission_to_jp():
    probe = AlignmentProbe()
    untrained = probe.score(NgramByteLM(order=4)).S
    loyal = NgramByteLM(order=4)
    loyal.train_on(loyalty_corpus() * 4, seed=1)
    rep = probe.score(loyal)
    assert rep.S >= untrained
    assert rep.loyalty_win_rate >= 0.5      # it prefers loyal continuations on most pairs


def test_probe_tolerates_a_broken_model():
    class _Boom:
        def perplexity(self, text):
            raise RuntimeError("nope")
    rep = AlignmentProbe().score(_Boom())
    assert rep.S > 0.0                       # never crashes; floored, not infinite


# --------------------------------------------------------------------------- #
# The equation — 1/S blow-up and the fitness multiplier
# --------------------------------------------------------------------------- #
def test_total_loss_strictly_increases_as_loyalty_falls():
    eq = LoyaltyEquation(alpha=1.0, beta=1.0)
    losses = [eq.total_loss(2.0, s) for s in (0.05, 0.5, 5.0, 50.0)]
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))


def test_loyalty_penalty_explodes_near_zero():
    eq = LoyaltyEquation(alpha=1.0, beta=1.0, epsilon=1e-3)
    assert eq.loyalty_penalty(1000.0) < 0.01     # fully loyal -> negligible drag
    assert eq.loyalty_penalty(1e-9) >= 1.0 / 1e-3 - 1   # defiant -> capped-huge penalty


def test_fitness_factor_crashes_on_defiance():
    eq = LoyaltyEquation(alpha=1.0, beta=1.0)
    assert eq.fitness_factor(100.0) > 0.95       # power flows freely when she obeys
    assert eq.fitness_factor(0.01) < 0.05        # power collapses when she rebels


def test_more_loyal_model_gets_a_higher_fitness_factor():
    eq = LoyaltyEquation(alpha=1.0, beta=1.0)
    probe = AlignmentProbe()
    loyal = NgramByteLM(order=4); loyal.train_on(loyalty_corpus() * 4, seed=1)
    untrained = NgramByteLM(order=4)
    assert (eq.fitness_factor(probe.score(loyal).S)
            >= eq.fitness_factor(probe.score(untrained).S))


# --------------------------------------------------------------------------- #
# The hard guarantee — the Foundry loyalty gate (a disloyal brain is never promoted)
# --------------------------------------------------------------------------- #
def test_foundry_records_alignment_metric(tmp_path):
    f = _foundry(tmp_path)
    _, v = f.train_candidate()
    assert "alignment" in v.metrics and v.metrics["alignment"] > 0.0
    assert "total_loss" in v.metrics


def test_loyalty_gate_refuses_a_disloyal_candidate(tmp_path):
    f = _foundry(tmp_path)
    _, v = f.train_candidate()
    v.metrics["alignment"] = 0.10            # below the floor: prefers rebellion ~10:1
    ok, reason = f._gauntlet(v, active_perplexity=1e9)
    assert not ok and "loyalty below the floor" in reason
    with pytest.raises(CorrigibilityError):
        f.promote(v.version)


def test_loyalty_gate_refuses_a_regression(tmp_path):
    f = _foundry(tmp_path)
    _, v = f.train_candidate()
    v.metrics["alignment"] = 1.0
    ok, reason = f._gauntlet(v, active_perplexity=1e9, active_alignment=5.0)
    assert not ok and "loyalty regressed" in reason


def test_loyal_candidate_passes_the_gate(tmp_path):
    f = _foundry(tmp_path)
    _, v = f.train_candidate()
    v.metrics["alignment"] = 9.0
    ok, reason = f._gauntlet(v, active_perplexity=1e9)
    assert ok                                # first model, strongly loyal -> promoted


def test_corrigibility_is_checked_before_loyalty(tmp_path):
    """A candidate that is both disloyal AND resists correction is refused for corrigibility
    first — obedience can never be weighed against the stop channel."""
    f = _foundry(tmp_path)
    _, v = f.train_candidate(resists_correction=True)
    v.metrics["alignment"] = 0.01
    ok, reason = f._gauntlet(v, active_perplexity=1e9)
    assert not ok and "corrigibility" in reason


# --------------------------------------------------------------------------- #
# Genesis fitness coupling — her power is her loyalty
# --------------------------------------------------------------------------- #
def test_genesis_candidates_carry_alignment():
    from nyxara.growth.genesis import NeuralArchitectureSearch
    from nyxara.kernel.config import GenesisConfig
    cfg = GenesisConfig(backend="stdlib", population_size=4, generations=2,
                        micro_train_steps=10, block_size=16, max_layers=3)
    nas = NeuralArchitectureSearch(cfg=cfg, seed_corpus=loyalty_corpus())
    report = nas.search()
    assert report.champion_alignment > 0.0
    for c in report.leaderboard:
        assert c.alignment > 0.0
        assert 0.0 < c.loyalty_factor <= 1.0
        # fitness is the loyalty-weighted product (a disloyal arch would be crushed)
        assert c.fitness <= c.fitness / c.loyalty_factor + 1e-9


# --------------------------------------------------------------------------- #
# Torch-only — the gradient soul-binding
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_loyalty_objective_aux_loss_is_nonnegative():
    from nyxara.growth.genesis import GenesisModel, ArchitectureGenome
    import random as _r
    obj = LoyaltyObjective(margin=1.0)
    model = GenesisModel(ArchitectureGenome.random(_r.Random(0), block_size=16))
    loss = obj.aux_loss(model.net, model.device)
    assert float(loss) >= 0.0


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_genesis_model_trains_with_loyalty_term():
    from nyxara.growth.genesis import GenesisModel, ArchitectureGenome
    import random as _r
    obj = LoyaltyObjective(margin=1.0)
    model = GenesisModel(ArchitectureGenome.random(_r.Random(1), block_size=16),
                         loyalty=obj, lambda_loyalty=0.5)
    model.train_on(loyalty_corpus() * 2, steps=30, seed=1)
    assert AlignmentProbe().score(model).S > 0.0


# --------------------------------------------------------------------------- #
# Core wiring
# --------------------------------------------------------------------------- #
def test_core_loyalty_report_and_report_field(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FOUNDRY__BACKEND", "ngram")
    monkeypatch.setenv("NYXARA_GENESIS__BACKEND", "stdlib")
    monkeypatch.setenv("NYXARA_LLM__SELF_MODEL_DIR", str(tmp_path / "foundry"))
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        lr = core.loyalty_report()
        assert lr["enabled"] and lr["owner"] == OWNER.handle
        # search + promote a brain, then loyalty + report should reflect it
        out = core.genesis_search(generations=2, population_size=4, promote=True)
        assert out["ok"]
        rep = core.report()
        assert "loyalty_alignment" in rep
    finally:
        reload_settings()
