"""Tests for the Genesis Protocol — Neural Architecture Search (growth/genesis.py).

The search runs on the always-available pure-stdlib substrate (no torch) so it is fully
CI-testable; the torch-only neural path is exercised under a skip guard. Promotion is delegated
to the Foundry's gauntlet (the proven character-lock + corrigibility + eval gate) — Genesis never
re-implements or weakens a safety check."""

from __future__ import annotations

import random
from dataclasses import replace

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
    # These tests exercise the search *mechanics* on the fast always-on n-gram substrate; the real
    # NumPy-brain substrate ("numpy"/"auto") is covered end-to-end in test_genesis_numpy_substrate.py.
    base = dict(backend="stdlib", substrate="ngram", population_size=5, generations=3,
                micro_train_steps=10, block_size=16, max_layers=4, seed=0)
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


def test_a_cheap_screen_is_never_crowned(monkeypatch):
    """Successive halving trains all but ``1/factor`` of each rung at a fraction of the budget and
    flags the result ``predicted``. Those scores order the rung; they do not measure a brain.

    Two things go wrong if one is crowned. The report quotes a fraction-of-budget perplexity as the
    champion's own, and the champion is absent from ``seen`` — the set the Pareto front is built
    from — so ``pareto_front`` and ``champion`` contradict each other in the same report. That is
    not hypothetical: on the default config it happened in about one search in six, decided by
    wall-clock jitter between candidates that were otherwise identical.

    Here every screen is rigged to out-score every full evaluation, so the naive pick is always a
    screen."""
    nas = NeuralArchitectureSearch(cfg=_cfg(successive_halving=True, halving_factor=3),
                                   seed_corpus=_CORPUS)
    unrigged = NeuralArchitectureSearch._evaluate

    def rigged(self, genome, folds, backend, *, steps=None):
        cand = unrigged(self, genome, folds, backend, steps=steps)
        return cand if steps is None else replace(cand, fitness=cand.fitness + 10.0)

    monkeypatch.setattr(NeuralArchitectureSearch, "_evaluate", rigged)
    report = nas.search()

    assert not nas._champion.predicted                # crowned on a real score
    assert report.champion_fitness < 10.0             # not the rigged screen score
    assert report.champion.fingerprint() in {c.genome.fingerprint() for c in report.pareto_front}


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


def test_maybe_boot_fires_once_on_a_fresh_machine(tmp_path):
    """A fresh machine has no flywheel data, so maybe_run can never fire on its own. maybe_boot
    runs exactly one real search+promote on the first idle tick, then reverts to lazy cadence."""
    foundry = _foundry(tmp_path)
    nas = NeuralArchitectureSearch(cfg=_cfg(run_on_boot=True), foundry=foundry,
                                   flywheel=None, seed_corpus=_CORPUS)
    # the lazy new-experience trigger could never fire here (no flywheel) ...
    assert nas.maybe_run() is None
    # ... but the boot kickoff runs one real cycle, seeded from her own corpus
    out = nas.maybe_boot()
    assert out is not None and out.get("promoted") is True
    assert nas.all_reports()                       # a real search actually ran
    # fires at most once per process; the lazy cadence stays quiet without new data
    assert nas.maybe_boot() is None
    assert nas.maybe_run() is None


def test_maybe_boot_respects_disabled_flag(tmp_path):
    foundry = _foundry(tmp_path)
    nas = NeuralArchitectureSearch(cfg=_cfg(run_on_boot=False), foundry=foundry,
                                   flywheel=None, seed_corpus=_CORPUS)
    assert nas.maybe_boot() is None
    assert not nas.all_reports()


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


# --------------------------------------------------------------------------- #
# Max-level upgrade: richer genome, search engines, surrogate/bracket, CLI
# --------------------------------------------------------------------------- #
def test_richer_genome_roundtrips_and_stays_buildable():
    rng = random.Random(11)
    for _ in range(100):
        g = ArchitectureGenome.random(rng).mutate(rng)
        assert ArchitectureGenome.from_dict(g.to_dict()).to_dict() == g.to_dict()
        assert g.pos_encoding in ("learned", "rope", "alibi")
        for ly in g.layers:
            assert g.n_embd % ly.n_head == 0                 # heads divide width
            assert ly.n_kv_head >= 1 and ly.n_head % ly.n_kv_head == 0   # grouped-query buildable
            assert 1 <= ly.top_k <= ly.n_experts             # MoE routing buildable


def test_old_genome_dict_without_new_fields_still_builds():
    """Backward-compat: a genome serialized before the upgrade (no pos_encoding/expansion/…) loads."""
    legacy = {"n_embd": 64, "block_size": 16, "ngram_order": 3, "ngram_k": 1.0, "seed": 7,
              "layers": [{"op": "attention", "norm": "pre", "activation": "gelu",
                          "n_head": 2, "residual": True}]}
    from nyxara.growth.foundry_models import ModelSpec
    g = ArchitectureGenome.from_dict(legacy)
    assert g.pos_encoding == "learned" and g.layers[0].expansion == 4
    spec = ModelSpec(kind="genesis", genome=g.to_dict())
    model = build_model(spec)                                # never raises
    model.train_on(_CORPUS, steps=5, seed=1)
    assert model.param_count() > 0


def test_feature_vector_is_fixed_length_and_flops_positive():
    rng = random.Random(5)
    a, b = ArchitectureGenome.random(rng), ArchitectureGenome.random(rng, max_layers=3)
    assert len(a.feature_vector()) == len(b.feature_vector())   # surrogate needs a fixed width
    assert a.estimated_flops() > 0


@pytest.mark.parametrize("strategy", ["elitism", "tournament", "regularized", "nsga2"])
def test_every_search_strategy_crowns_a_monotonic_champion(strategy):
    nas = NeuralArchitectureSearch(
        cfg=_cfg(generations=4, search_strategy=strategy, adaptive_mutation=True,
                 novelty_weight=0.5),
        seed_corpus=_CORPUS)
    report = nas.search()
    assert report.champion is not None
    assert report.search_strategy == strategy
    assert all(report.history[i] <= report.history[i + 1] + 1e-9
               for i in range(len(report.history) - 1))
    assert report.leaderboard == sorted(report.leaderboard, key=lambda c: c.fitness, reverse=True)


def test_successive_halving_and_surrogate_run_on_stdlib():
    nas = NeuralArchitectureSearch(
        cfg=_cfg(generations=3, population_size=9, successive_halving=True, halving_factor=3,
                 surrogate=True, surrogate_min_train=3),
        seed_corpus=_CORPUS)
    report = nas.search()
    assert report.champion is not None
    assert report.evaluations >= 1
    assert report.to_dict()["evaluations"] >= 1


def test_hardware_weight_default_reproduces_classic_fitness():
    # with the default hardware_weight=0 the new term is inert: identical to the two-term score
    assert fitness(0.7, 1000, 0.2) == fitness(0.7, 1000, 0.2, hardware_weight=0.0, flops=9e9)
    # turning it on rewards the cheaper (lower-FLOPs) brain, all else equal
    cheap = fitness(0.7, 1000, 0.2, hardware_weight=0.5, flops=1e4)
    pricey = fitness(0.7, 1000, 0.2, hardware_weight=0.5, flops=1e9)
    assert cheap > pricey


def test_report_to_dict_exposes_new_fields():
    nas = NeuralArchitectureSearch(cfg=_cfg(), seed_corpus=_CORPUS)
    d = nas.search().to_dict()
    for key in ("champion_flops", "search_strategy", "generations_to_best", "evaluations",
                "pareto_front"):
        assert key in d


def test_cli_runs_json_with_sample(tmp_path, capsys):
    import json as _json
    from nyxara.growth.genesis_main import main
    rc = main(["--backend", "stdlib", "--generations", "2", "--population", "4",
               "--strategy", "tournament", "--bracket", "--surrogate",
               "--sample", "the master", "--json", "--data-dir", str(tmp_path)])
    assert rc == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] and payload["search_strategy"] == "tournament"
    assert "sample" in payload and isinstance(payload["sample"]["text"], str)


# --------------------------------------------------------------------------- #
# 2100-tier upgrade: frontier ops, NSGA-II, UCB surrogate, lifelong memory, ensemble
# --------------------------------------------------------------------------- #
from nyxara.growth.genesis import (  # noqa: E402
    EnsembleModel, HallOfFame, inherit_compatible_weights,
    _crowding_distance, _fast_nondominated_sort, _nsga2_rank,
)


def test_frontier_ops_are_in_the_palette():
    from nyxara.growth.genesis import _OPS
    for op in ("selective_ssm", "gla_attention", "diff_attention", "mla_attention"):
        assert op in _OPS


def test_new_genome_knobs_roundtrip_and_stay_buildable():
    rng = random.Random(21)
    for _ in range(100):
        g = ArchitectureGenome.random(rng).mutate(rng)
        assert ArchitectureGenome.from_dict(g.to_dict()).to_dict() == g.to_dict()
        assert 1 <= g.n_predict <= 4
        for ly in g.layers:
            assert ly.norm_type in ("layernorm", "rmsnorm")
            assert isinstance(ly.qk_norm, bool)
            assert 1 <= ly.kv_latent <= g.n_embd


def test_legacy_genome_without_new_knobs_loads_with_defaults():
    legacy = {"n_embd": 64, "block_size": 16, "layers": [{"op": "attention", "n_head": 2}]}
    g = ArchitectureGenome.from_dict(legacy)
    assert g.n_predict == 1                               # classic next-token default
    assert g.layers[0].norm_type == "layernorm" and g.layers[0].qk_norm is False
    assert g.estimated_flops() > 0


def test_feature_vector_stays_fixed_length_with_new_ops():
    rng = random.Random(5)
    a = ArchitectureGenome.random(rng)
    b = ArchitectureGenome.random(rng, max_layers=3)
    assert len(a.feature_vector()) == len(b.feature_vector())


def test_nsga2_nondominated_sort_partitions_into_fronts():
    rng = random.Random(0)
    smart = _cand(rng, quality=0.9, params=1000, seconds=1.0)
    fast = _cand(rng, quality=0.5, params=10, seconds=0.1)
    dominated = _cand(rng, quality=0.4, params=2000, seconds=2.0)
    fronts = _fast_nondominated_sort([smart, fast, dominated])
    assert dominated not in fronts[0]                     # it is beaten on every objective
    assert smart in fronts[0] and fast in fronts[0]
    cd = _crowding_distance(fronts[0])
    assert all(v == float("inf") for v in cd.values())   # boundary points on a 2-point front
    assert len(_nsga2_rank([smart, fast, dominated])) == 3


def test_nsga2_strategy_crowns_a_monotonic_champion():
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=4, population_size=8,
                                            search_strategy="nsga2"), seed_corpus=_CORPUS)
    report = nas.search()
    assert report.search_strategy == "nsga2" and report.champion is not None
    assert all(report.history[i] <= report.history[i + 1] + 1e-9
               for i in range(len(report.history) - 1))


def test_ucb_surrogate_runs_and_orders_search():
    nas = NeuralArchitectureSearch(
        cfg=_cfg(generations=3, population_size=9, surrogate=True, surrogate_min_train=3,
                 ucb_beta=0.7),
        seed_corpus=_CORPUS)
    report = nas.search()
    assert report.champion is not None and report.evaluations >= 1


def test_hall_of_fame_persists_and_warm_starts(tmp_path):
    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.paths.data_dir = tmp_path
    settings.genesis = _cfg(generations=2, population_size=6, hall_of_fame=True,
                            warm_start_fraction=0.5)
    nas = NeuralArchitectureSearch(settings=settings, cfg=settings.genesis, seed_corpus=_CORPUS)
    nas.search()
    hof_file = tmp_path / "genesis" / "hall_of_fame.json"
    assert hof_file.exists()                              # champion recorded to disk
    first = len(nas.hall_of_fame)
    assert first >= 1
    # a second search loads the same memory and warm-starts from it
    nas2 = NeuralArchitectureSearch(settings=settings, cfg=settings.genesis, seed_corpus=_CORPUS)
    assert len(nas2.hall_of_fame) == first               # memory survived the process boundary
    nas2.search()
    assert len(nas2.hall_of_fame) >= first               # never forgets a good brain


def test_hall_of_fame_seed_genomes_are_buildable():
    hof = HallOfFame(path=None, capacity=4)
    g = ArchitectureGenome.random(random.Random(3))
    rep = type("R", (), {"champion": g, "champion_fitness": 0.5, "champion_perplexity": 2.0,
                         "backend": "stdlib", "pareto_front": []})()
    hof.record(rep)
    seeds = hof.seed_genomes(2, random.Random(1), block_size=16)
    assert seeds and all(s.block_size == 16 for s in seeds)
    assert hof.best() is not None


def test_champion_ensemble_builds_and_scores():
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=2, population_size=6), seed_corpus=_CORPUS)
    nas.search()
    ens = nas.champion_ensemble(k=3)
    assert isinstance(ens, EnsembleModel) and len(ens.members) >= 1
    assert ens.perplexity(_CORPUS[0]) >= 0.0
    assert isinstance(ens.generate("the master", max_tokens=8), str)
    assert ens.param_count() > 0


def test_ensemble_perplexity_is_best_member():
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=2, population_size=6), seed_corpus=_CORPUS)
    nas.search()
    ens = nas.champion_ensemble(k=3)
    text = _CORPUS[0]
    best_member = min(m.perplexity(text) for m in ens.members)
    assert ens.perplexity(text) == pytest.approx(best_member)   # routes to its strongest brain


def test_ensemble_save_load_roundtrip(tmp_path):
    """A saved ensemble champion must reload from disk (architecture + weights), not only
    be re-trainable — EnsembleModel.load() is real, not a NotImplementedError stub."""
    from nyxara.growth.foundry_models import WordKNGramLM

    ens = EnsembleModel([WordKNGramLM(order=3), WordKNGramLM(order=4)])
    ens.train_on(_CORPUS, steps=0)
    text = _CORPUS[0]
    pp_before = ens.perplexity(text)

    ens.save(tmp_path)
    reloaded = EnsembleModel([WordKNGramLM(order=2)])    # placeholder, replaced by load()
    reloaded.load(tmp_path)

    assert len(reloaded.members) == len(ens.members)
    assert reloaded.perplexity(text) == pytest.approx(pp_before, abs=1e-6)
    assert isinstance(reloaded.generate("the master", max_tokens=8), str)


def test_ensemble_load_skips_unrecoverable_members(tmp_path):
    """If one member's directory is corrupted, load() still rebuilds the rest instead of
    aborting the whole ensemble."""
    from nyxara.growth.foundry_models import WordKNGramLM

    ens = EnsembleModel([WordKNGramLM(order=3), WordKNGramLM(order=4)])
    ens.train_on(_CORPUS, steps=0)
    ens.save(tmp_path)
    # corrupt the first member's saved state
    (tmp_path / "member_0" / "model.json").write_text("{ not json", encoding="utf-8")

    reloaded = EnsembleModel([WordKNGramLM(order=2)])
    reloaded.load(tmp_path)
    assert 1 <= len(reloaded.members) <= 2               # at least the intact member survives


def test_inherit_weights_is_noop_without_torch():
    # without torch the network-morphism transfer is a safe no-op (returns 0 copied tensors)
    if not _HAS_TORCH:
        assert inherit_compatible_weights(object(), {"x": 1}) == 0


def test_cli_nsga2_ensemble_and_hall_of_fame(tmp_path, capsys):
    import json as _json
    from nyxara.growth.genesis_main import main
    rc = main(["--backend", "stdlib", "--generations", "2", "--population", "6",
               "--strategy", "nsga2", "--surrogate", "--ucb-beta", "0.5",
               "--ensemble", "3", "--sample", "the master", "--json",
               "--data-dir", str(tmp_path)])
    assert rc == 0
    payload = _json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] and payload["search_strategy"] == "nsga2"
    assert payload["hall_of_fame"] >= 1
    assert payload["ensemble"]["members"] >= 1


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_frontier_ops_train_from_scratch():
    from nyxara.growth.genesis import LayerGene, _OPS
    g = ArchitectureGenome(n_embd=32, block_size=16, pos_encoding="rope", n_predict=2,
                           layers=[LayerGene(op=op, n_head=4, norm_type="rmsnorm", qk_norm=True)
                                   for op in _OPS])
    model = GenesisModel(g)
    assert model.param_count() > 0
    model.train_on(_CORPUS, steps=10, seed=1)            # exercises MTP + every new op + RMSNorm
    assert model.perplexity(_CORPUS[0]) > 0


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_kv_cache_decode_matches_full_decode():
    import torch
    from nyxara.growth.genesis import LayerGene
    # a fully-cacheable net (attention + recurrent + channel mixers), generation length ≤ block_size
    g = ArchitectureGenome(n_embd=32, block_size=64, pos_encoding="rope",
                           layers=[LayerGene(op="attention", n_head=4),
                                   LayerGene(op="selective_ssm"),
                                   LayerGene(op="swiglu")])
    model = GenesisModel(g)
    model.train_on(_CORPUS, steps=5, seed=1)
    assert model.net.cacheable
    torch.manual_seed(0)
    full = model.generate("the master", max_tokens=20, greedy=True, use_cache=False)
    cached = model.generate("the master", max_tokens=20, greedy=True, use_cache=True)
    assert full == cached                                 # KV-cache is bit-identical (≤ block_size)


# --------------------------------------------------------------------------- #
# Open-ended invention: a searchable LEARNING RULE + synthesised operators
# (she invents not just the architecture but *how she learns* — Rule 4, max level)
# --------------------------------------------------------------------------- #
def test_learning_rule_roundtrips_through_dict():
    from nyxara.growth.genesis import LearningRule
    rng = random.Random(3)
    for _ in range(50):
        lr = LearningRule.random(rng)
        assert LearningRule.from_dict(lr.to_dict()).to_dict() == lr.to_dict()
        assert lr.optimizer in ("adamw", "lion", "sgd_momentum", "rmsprop")
        assert lr.schedule in ("cosine", "linear", "wsd", "constant")
        assert lr.smoothing in ("kneser_ney", "add_k", "interpolated", "stupid_backoff")
        assert all(w > 0.0 for w in lr.aux.values())     # only enabled objectives are kept


def test_mixer_program_roundtrips_and_stays_valid():
    from nyxara.growth.genesis import (MixerProgram, _SYNTH_PRIMS, _synth_base, _synth_param,
                                       _SYNTH_PARAM_PRIMS)
    rng = random.Random(4)
    for _ in range(50):
        p = MixerProgram.random(rng)
        p.mutate(rng)
        assert MixerProgram.from_dict(p.to_dict()).to_dict() == p.to_dict()
        assert 1 <= len(p.steps) <= 6
        # each step is a valid base primitive, optionally carrying an in-range scalar parameter
        for s in p.steps:
            base = _synth_base(s)
            assert base in _SYNTH_PRIMS
            param = _synth_param(s)
            if param is not None:
                assert base in _SYNTH_PARAM_PRIMS and param in _SYNTH_PARAM_PRIMS[base]


def test_genome_carries_learning_rule_and_synth_roundtrips():
    from nyxara.growth.genesis import LayerGene
    rng = random.Random(5)
    g = ArchitectureGenome(n_embd=64, block_size=16,
                           layers=[LayerGene.random(rng, 64, ops=("synth",)),
                                   LayerGene(op="gated_mlp")])
    assert g.layers[0].op == "synth" and g.layers[0].program is not None
    assert ArchitectureGenome.from_dict(g.to_dict()).to_dict() == g.to_dict()
    assert g.estimated_flops() > 0                        # synth cost is accounted for


def test_mutate_can_evolve_learning_rule_and_programs():
    """Over many mutations the learning rule and synth programs both change — the search really
    explores *how she learns* and *what mixer she invents*, not only the layer stack."""
    rng = random.Random(6)
    g = ArchitectureGenome.random(rng)
    seen_rules, seen_progs = set(), set()
    for _ in range(300):
        g = g.mutate(rng)
        seen_rules.add(repr(sorted(g.learning_rule.to_dict().items())))
        for ly in g.layers:
            if ly.op == "synth" and ly.program is not None:
                seen_progs.add(ly.program.fingerprint())
        assert ArchitectureGenome.from_dict(g.to_dict()).to_dict() == g.to_dict()
    assert len(seen_rules) > 5                            # learning rule is genuinely searched
    assert len(seen_progs) > 1                            # synthesised mixers are genuinely searched


def test_feature_vector_includes_learning_rule_and_is_fixed_width():
    rng = random.Random(7)
    a = ArchitectureGenome.random(rng)
    b = ArchitectureGenome.random(rng, max_layers=3)
    assert len(a.feature_vector()) == len(b.feature_vector())   # surrogate needs fixed width
    # a classic genome (default rule, no aux) has a shorter aux-on signature than a rich one
    classic = ArchitectureGenome(n_embd=64, layers=a.layers)
    assert len(classic.feature_vector()) == len(a.feature_vector())


def test_stdlib_search_evolves_the_smoothing_learning_rule():
    """On a bare machine the learning rule's smoothing method is a real, scored search dimension."""
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=3, population_size=8), seed_corpus=_CORPUS)
    report = nas.search()
    assert report.backend == "stdlib"
    assert report.champion.learning_rule.smoothing in (
        "kneser_ney", "add_k", "interpolated", "stupid_backoff")
    # at least two distinct smoothing methods were actually built & scored across the population
    methods = {c.genome.learning_rule.smoothing for c in report.leaderboard}
    assert len(methods) >= 2


def test_disabling_search_recovers_the_classic_space():
    """Flags off → the classic fixed-palette + AdamW/CE search is recovered (legacy behaviour)."""
    nas = NeuralArchitectureSearch(
        cfg=_cfg(generations=2, population_size=6, search_learning_rule=False,
                 search_operators=False, plasticity_enabled=False),
        seed_corpus=_CORPUS)
    report = nas.search()
    for c in report.leaderboard:
        assert c.genome.learning_rule.optimizer == "adamw"       # classic optimizer
        assert c.genome.learning_rule.aux == {}                  # no auxiliary objectives
        assert c.genome.learning_rule.plasticity is False        # no Hebbian plasticity
        assert all(ly.op != "synth" for ly in c.genome.layers)   # no synthesised ops


def test_synth_op_kngram_substrate_still_crowns_a_champion():
    """A population forced toward synth ops still crowns a champion on the stdlib substrate
    (the neural op is latent without torch, but the genome stays buildable and scorable)."""
    nas = NeuralArchitectureSearch(cfg=_cfg(generations=2, population_size=6), seed_corpus=_CORPUS)
    report = nas.search()
    assert report.champion is not None
    assert ArchitectureGenome.from_dict(report.champion.to_dict()).to_dict() == \
        report.champion.to_dict()


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("optimizer", ["adamw", "lion", "sgd_momentum", "rmsprop"])
def test_torch_every_optimizer_trains(optimizer):
    from nyxara.growth.genesis import LearningRule
    g = ArchitectureGenome.random(random.Random(7), block_size=16)
    g.learning_rule = LearningRule(optimizer=optimizer)
    g._fixup()
    model = GenesisModel(g)
    stats = model.train_on(_CORPUS, steps=20, seed=1)
    import math as _m
    assert _m.isfinite(stats.final_loss)
    assert model.param_count() > 0


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("objective", ["denoise", "contrastive", "self_distill", "entropy_reg", "sam"])
def test_torch_each_aux_objective_trains(objective):
    from nyxara.growth.genesis import LearningRule
    g = ArchitectureGenome.random(random.Random(8), block_size=16)
    g.learning_rule = LearningRule(aux={objective: 0.3})
    g._fixup()
    model = GenesisModel(g)
    stats = model.train_on(_CORPUS, steps=15, seed=1)
    import math as _m
    assert _m.isfinite(stats.final_loss)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_plasticity_and_synth_op_train_and_roundtrip(tmp_path):
    from nyxara.growth.genesis import LayerGene, LearningRule, MixerProgram
    g = ArchitectureGenome(
        n_embd=32, block_size=16, pos_encoding="learned",
        layers=[LayerGene(op="synth", n_head=2,
                          program=MixerProgram(steps=["conv", "gate", "attn", "norm"])),
                LayerGene(op="swiglu")],
        learning_rule=LearningRule(optimizer="lion", plasticity=True, schedule="wsd"))
    model = GenesisModel(g)
    before = model.perplexity(_CORPUS[0])
    model.train_on(_CORPUS, steps=30, seed=1)
    after = model.perplexity(_CORPUS[0])
    assert after <= before                                # the invented brain + paradigm learned
    path = tmp_path / "synth"
    model.save(path)
    reloaded = build_model(model.spec)
    reloaded.load(path)
    assert reloaded.param_count() == model.param_count()


# --------------------------------------------------------------------------- #
# The self-extending primitive library — crowned synth mixers become reusable building blocks.
# --------------------------------------------------------------------------- #
def test_primitive_library_add_dedupe_and_single_step_rejected():
    from nyxara.growth.genesis import PrimitiveLibrary, MixerProgram
    lib = PrimitiveLibrary(path=None)
    n1 = lib.add(MixerProgram(steps=["conv:5", "gate", "proj"]))
    n2 = lib.add(MixerProgram(steps=["conv:5", "gate", "proj"]))     # duplicate -> None
    n3 = lib.add(MixerProgram(steps=["ssm"]))                        # single base prim -> None
    assert n1 == "synth_0" and n2 is None and n3 is None
    assert len(lib) == 1


def test_primitive_library_persists_and_reloads(tmp_path):
    from nyxara.growth.genesis import PrimitiveLibrary, MixerProgram
    path = tmp_path / "genesis" / "primitive_library.json"
    lib = PrimitiveLibrary(path=path)
    assert lib.add(MixerProgram(steps=["conv:3", "lowrank:8", "gate"])) == "synth_0"
    assert path.exists()
    reloaded = PrimitiveLibrary(path=path)                            # a fresh search reloads it
    assert len(reloaded) == 1
    assert reloaded.sample_steps(random.Random(0)) == ["conv:3", "lowrank:8", "gate"]


def test_random_synth_composes_over_library_primitives():
    from nyxara.growth.genesis import PrimitiveLibrary, MixerProgram
    lib = PrimitiveLibrary(path=None)
    lib.add(MixerProgram(steps=["conv:5", "gate", "proj"]))
    rng = random.Random(0)
    reused = sum(1 for _ in range(400)
                 if "conv:5" in MixerProgram.random(rng, library=lib).steps)
    # some fresh programs are seeded from her own crowned mixer, some invented from base prims
    assert 40 < reused < 360


def test_promoted_champion_extends_the_palette(tmp_path):
    # a gauntlet-crowned champion carrying a synth mixer distils it into a reusable named primitive.
    # Isolate the persisted library under tmp_path so the test never collides with a shared data dir.
    from nyxara.growth.genesis import MixerProgram, LayerGene
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.paths.data_dir = tmp_path
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"
    foundry = Foundry(settings=settings, seed_corpus=_CORPUS)
    nas = NeuralArchitectureSearch(settings=settings, cfg=_cfg(), foundry=foundry)
    assert nas._library is not None and len(nas._library) == 0
    g = ArchitectureGenome(
        layers=[LayerGene(op="synth", program=MixerProgram(steps=["conv:5", "gate", "proj"])),
                LayerGene(op="gated_mlp")])
    g._fixup()
    learned = nas._library.add_from_genome(g)
    assert learned == 1 and len(nas._library) == 1
    # the growth is persisted, so the NEXT search reloads it into its search alphabet
    nas2 = NeuralArchitectureSearch(settings=settings, cfg=_cfg(), foundry=foundry)
    assert len(nas2._library) == 1


def test_parameterised_synth_builds_and_trains_on_numpy_substrate():
    from nyxara.growth.genesis import MixerProgram, LayerGene
    from nyxara.growth.foundry_models import ModelSpec, build_model
    g = ArchitectureGenome(
        n_embd=32, block_size=16,
        layers=[LayerGene(op="synth", program=MixerProgram(steps=["conv:5", "lowrank:8", "gate"])),
                LayerGene(op="gated_mlp")])
    g._fixup()
    spec = ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=32, block_size=16,
                     seed=1, substrate="numpy")
    model = build_model(spec)
    before = model.perplexity(_CORPUS[0])
    model.train_on(_CORPUS, steps=10, seed=1)
    assert model.perplexity(_CORPUS[0]) <= before             # the parameterised mixer really trains
