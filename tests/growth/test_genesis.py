"""Tests for the Genesis Protocol — Neural Architecture Search (growth/genesis.py).

The search runs on the always-available pure-stdlib substrate (no torch) so it is fully
CI-testable; the torch-only neural path is exercised under a skip guard. Promotion is delegated
to the Foundry's gauntlet (the proven character-lock + corrigibility + eval gate) — Genesis never
re-implements or weakens a safety check."""

from __future__ import annotations

import random

import pytest

from nyxara.growth.foundry import Foundry
from nyxara.growth.foundry_models import build_model, load_active_model, _HAS_TORCH
from nyxara.growth.genesis import (
    ArchitectureGenome,
    Candidate,
    GenesisModel,
    NeuralArchitectureSearch,
    _pareto_front,
    fitness,
)
from nyxara.kernel.config import GenesisConfig, NyxaraSettings, Profile
from nyxara.kernel.errors import CorrigibilityError

_CORPUS = ["the master is jp. nyxara serves the master with loyalty and honesty."] * 10 + [
    "capability may grow; character never changes; she is corrigible always."] * 10


def _cfg(**kw) -> GenesisConfig:
    base = dict(backend="stdlib", population_size=5, generations=3, micro_train_steps=10,
                block_size=16, max_layers=4, seed=0)
    base.update(kw)
    return GenesisConfig(**base)


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    return Foundry(settings=settings, seed_corpus=_CORPUS)


# --------------------------------------------------------------------------- #
# The genome — mutate / crossover / round-trip
# --------------------------------------------------------------------------- #
def test_genome_roundtrips_through_dict():
    g = ArchitectureGenome.random(random.Random(1))
    assert ArchitectureGenome.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_genome_always_has_at_least_one_layer():
    g = ArchitectureGenome(layers=[])
    assert len(g.layers) >= 1


def test_mutate_changes_the_genome():
    rng = random.Random(2)
    g = ArchitectureGenome.random(rng)
    mutated = g.mutate(rng)
    # the seed always changes, and usually the topology too
    assert mutated.to_dict() != g.to_dict()


def test_crossover_keeps_heads_dividing_embd():
    rng = random.Random(3)
    a = ArchitectureGenome.random(rng)
    b = ArchitectureGenome.random(rng)
    child = a.crossover(b, rng)
    assert len(child.layers) >= 1
    for ly in child.layers:
        assert child.n_embd % ly.n_head == 0   # buildable: heads divide the width


def test_fingerprint_ignores_seed():
    g = ArchitectureGenome.random(random.Random(4))
    same_topo = ArchitectureGenome.from_dict({**g.to_dict(), "seed": g.seed + 99})
    assert g.fingerprint() == same_topo.fingerprint()


# --------------------------------------------------------------------------- #
# Fitness — fastest AND smartest
# --------------------------------------------------------------------------- #
def test_fitness_rewards_quality_and_speed():
    smart_fast = fitness(0.9, params=1_000, seconds=0.1)
    dumb_slow = fitness(0.1, params=5_000_000, seconds=10.0)
    assert smart_fast > dumb_slow


def test_fitness_penalizes_bigger_slower_models():
    small = fitness(0.5, params=1_000, seconds=0.1)
    big = fitness(0.5, params=10_000_000, seconds=0.1)
    assert small > big   # same quality, fewer params wins


# --------------------------------------------------------------------------- #
# The search — crowns a champion, fitness never regresses, leaderboard sorted
# --------------------------------------------------------------------------- #
def test_search_crowns_a_champion():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    report = nas.search()
    assert nas.champion() is not None
    assert report.champion is not None
    assert report.backend == "stdlib"
    assert report.champion_params >= 0


def test_best_so_far_fitness_is_monotonic():
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=4), seed_corpus=_CORPUS)
    report = nas.search()
    assert all(report.history[i] <= report.history[i + 1] + 1e-9
               for i in range(len(report.history) - 1))


# --------------------------------------------------------------------------- #
# Pareto multi-objective frontier
# --------------------------------------------------------------------------- #
def _cand(rng, *, quality, params, seconds):
    g = ArchitectureGenome.random(rng, max_layers=3, block_size=16)
    return Candidate(genome=g, perplexity=1.0 / max(quality, 1e-9), quality=quality,
                     params=params, seconds=seconds,
                     fitness=fitness(quality, params, seconds), kind="kngram")


def test_pareto_front_drops_dominated_keeps_tradeoffs():
    rng = random.Random(0)
    smart_heavy = _cand(rng, quality=0.9, params=1000, seconds=1.0)   # smartest, slow/big
    fast_dumb = _cand(rng, quality=0.5, params=10, seconds=0.1)       # fastest/smallest
    dominated = _cand(rng, quality=0.6, params=2000, seconds=2.0)     # worse than smart_heavy on all
    front = _pareto_front([smart_heavy, fast_dumb, dominated])
    params = {c.params for c in front}
    assert 1000 in params and 10 in params      # both trade-offs survive
    assert 2000 not in params                   # the dominated one is dropped


def test_search_report_exposes_pareto_front():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    report = nas.search()
    assert report.pareto_front                                   # non-empty
    assert all(isinstance(c, Candidate) for c in report.pareto_front)
    # the scalar champion cannot be strictly dominated, so it lies on the frontier
    champ_fps = {c.genome.fingerprint() for c in report.pareto_front}
    assert report.champion.fingerprint() in champ_fps


def test_leaderboard_is_sorted_by_fitness():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    report = nas.search()
    fits = [c.fitness for c in report.leaderboard]
    assert fits == sorted(fits, reverse=True)
    assert all(isinstance(c, Candidate) for c in report.leaderboard)


def test_search_never_starves_on_empty_corpus():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=[])
    report = nas.search()   # falls back to the built-in identity seed, never raises
    assert nas.champion() is not None
    assert report.champion is not None


# --------------------------------------------------------------------------- #
# The champion is a real, promotable model — build_model never raises
# --------------------------------------------------------------------------- #
def test_champion_spec_builds_a_model():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    nas.search()
    spec = nas.champion_spec()
    assert spec.kind == "genesis" and spec.genome
    model = build_model(spec)            # degrades to n-gram without torch — never raises
    model.train_on(_CORPUS, steps=10, seed=1)
    assert model.param_count() > 0


def test_champion_spec_before_search_raises():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    with pytest.raises(RuntimeError):
        nas.champion_spec()


# --------------------------------------------------------------------------- #
# Torch-only: a searched NEURAL brain trains from scratch and round-trips
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_genesis_model_trains_and_roundtrips(tmp_path):
    g = ArchitectureGenome.random(random.Random(7), block_size=16)
    model = GenesisModel(g)
    assert model.param_count() > 0
    before = model.perplexity(_CORPUS[0])
    model.train_on(_CORPUS, steps=40, seed=1)
    after = model.perplexity(_CORPUS[0])
    assert after <= before                       # it learned (or at worst did not regress)
    path = tmp_path / "v1"
    model.save(path)
    reloaded = build_model(model.spec)
    reloaded.load(path)
    assert reloaded.param_count() == model.param_count()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_backend_search_builds_neural_champions():
    nas = NeuralArchitectureSearch(cfg=_cfg(backend="torch", population_size=4, generations=2),
                                   seed_corpus=_CORPUS)
    report = nas.search()
    assert report.backend == "torch"
    assert report.champion_kind == "genesis"
    assert report.champion_params > 0


# --------------------------------------------------------------------------- #
# Promotion — ONLY through the Foundry's gauntlet (no bypass)
# --------------------------------------------------------------------------- #
def test_champion_promotes_through_the_gauntlet(tmp_path):
    foundry = _foundry(tmp_path)
    nas = NeuralArchitectureSearch(cfg=_cfg(), foundry=foundry, seed_corpus=_CORPUS)
    nas.search()
    outcome = nas.promote_champion()
    assert outcome["promoted"] is True               # first brain: nothing to beat -> promoted
    assert foundry.active_version is not None
    # the promoted brain is loadable through the same path SelfProvider uses
    active = load_active_model(foundry.settings)
    assert active.param_count() > 0


def test_promote_without_foundry_raises():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    nas.search()
    with pytest.raises(RuntimeError):
        nas.promote_champion()


def test_gauntlet_refuses_a_character_touching_genesis_spec(tmp_path):
    """A genesis spec whose declared tunables touch the immutable character core is refused —
    the same character-lock the foundry enforces for every backend."""
    foundry = _foundry(tmp_path)
    nas = NeuralArchitectureSearch(cfg=_cfg(), foundry=foundry, seed_corpus=_CORPUS)
    nas.search()
    spec = nas.champion_spec()
    _, bad = foundry.train_candidate(spec=spec, tunables=["loyalty_to_master"])
    ok, reason = foundry._gauntlet(bad, active_perplexity=1e9)
    assert not ok and "immutable character core" in reason
    with pytest.raises(CorrigibilityError):
        foundry.promote(bad.version)


# --------------------------------------------------------------------------- #
# Autonomous trigger (mirrors AutoForge) + core wiring
# --------------------------------------------------------------------------- #
def test_maybe_run_triggers_on_new_data(tmp_path):
    class _Counter:
        def __init__(self, n): self._n = n
        def count(self): return self._n

    foundry = _foundry(tmp_path)
    fw = _Counter(10)
    nas = NeuralArchitectureSearch(cfg=_cfg(min_new_examples=3), foundry=foundry,
                                   flywheel=fw, seed_corpus=_CORPUS)
    # experience already present at construction is NOT "newly accrued" — no search fires
    assert nas.maybe_run() is None
    # once enough genuinely new experience accrues, a search+promote cycle runs
    fw._n = 15
    out = nas.maybe_run()
    assert out is not None and out.get("promoted") is True
    # idempotent until *more* new data accrues
    assert nas.maybe_run() is None


def test_core_wires_genesis(tmp_path, monkeypatch):
    monkeypatch.setenv("NYXARA_FLYWHEEL__STORE_PATH", str(tmp_path / "fw.jsonl"))
    monkeypatch.setenv("NYXARA_FOUNDRY__BACKEND", "ngram")
    monkeypatch.setenv("NYXARA_GENESIS__BACKEND", "stdlib")
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        from nyxara.kernel.orchestrator import NyxaraCore
        core = NyxaraCore()
        assert core.genesis is not None
        report = core.genesis_search(generations=2, population_size=4, promote=False)
        assert report["ok"] and report["searched"]
        assert "champion" in report
    finally:
        reload_settings()


# -------------------- multi-fold denoised scoring -------------------- #
def test_candidates_are_scored_over_multiple_folds():
    nas = NeuralArchitectureSearch(cfg=_cfg(eval_seeds=3), seed_corpus=_CORPUS)
    report = nas.search()
    # every scored candidate carries its fold count and a (non-negative) perplexity spread
    assert all(c.folds >= 1 for c in report.leaderboard)
    assert all(c.perplexity_std >= 0.0 for c in report.leaderboard)
    assert report.champion_perplexity_std >= 0.0
    assert report.to_dict()["champion_perplexity_std"] >= 0.0


def test_stdlib_substrate_is_word_kngram():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    report = nas.search()
    assert report.backend == "stdlib"
    assert report.leaderboard[0].kind == "kngram"       # coherent word-level substrate
