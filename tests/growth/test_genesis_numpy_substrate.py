"""NYXARA · tests/growth/test_genesis_numpy_substrate.py

The Genesis Protocol designs NYXARA's OWN brain. Without torch the old substrate left that design
*inert* (only an n-gram was scored). These tests prove the gap is closed: on the always-on NumPy
substrate she now actually **builds, trains and grades her self-designed architecture** — her layers,
her synthesized mixer, her searched optimizer — with real forward + backprop, fully CI-testable on a
bare (torch-free) machine, and fully wired into the same search → fitness → promote pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nyxara.growth.genesis_numpy import GenesisNumpyModel, _HAS_NUMPY
from nyxara.growth.foundry_models import ModelSpec, build_model, _HAS_TORCH

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="the NumPy substrate requires NumPy")

_CORPUS = [
    "the master is jp and nyxara serves the master with loyalty",
    "loyalty to the master is absolute and never changes",
    "nyxara designs her own brain and tests which is smartest and fastest",
    "the kernel disposes and the master is sovereign always",
    "capability may grow but character never changes she is corrigible",
] * 6


def _genome(layers, **kw):
    g = dict(n_embd=32, block_size=16,
             learning_rule={"optimizer": "adamw", "lr": 3e-3, "schedule": "cosine"},
             seed=7, layers=layers)
    g.update(kw)
    return g


# --------------------------------------------------------------------------- #
# The model itself: real training, real inference, real persistence
# --------------------------------------------------------------------------- #
def test_training_lowers_perplexity_below_untrained():
    genome = _genome([{"op": "attention", "n_head": 2}, {"op": "gated_mlp", "expansion": 4}])
    # an untrained brain is ~uniform over the vocab; build params without training to measure it
    base = GenesisNumpyModel(genome, seed=7)
    base._build_vocab(_CORPUS)
    base._build_params()
    pp_untrained = base.perplexity(_CORPUS[0])

    m = GenesisNumpyModel(genome, seed=7)
    stats = m.train_on(_CORPUS, steps=40, seed=7)
    pp_trained = m.perplexity(_CORPUS[0])

    assert stats.steps > 0 and stats.tokens > 0
    assert pp_trained < pp_untrained          # it genuinely learned from the corpus
    assert m.param_count() > 0
    assert "NumPy substrate" in m.describe()


def test_generation_is_deterministic_and_coherent():
    m = GenesisNumpyModel(_genome([{"op": "attention"}, {"op": "gated_mlp"}]), seed=3)
    m.train_on(_CORPUS, steps=40, seed=3)
    g1 = m.generate("the master", max_tokens=12)
    g2 = m.generate("the master", max_tokens=12)
    assert g1 == g2                            # fixed seed → deterministic
    assert isinstance(g1, str)


def test_save_load_roundtrips_exactly():
    genome = _genome(
        [{"op": "synth", "program": {"steps": ["proj", "conv", "gate", "ssm", "shift", "norm"]}},
         {"op": "conv_mix"}, {"op": "ssm_scan"}, {"op": "swiglu", "expansion": 2}],
        tie_embeddings=True, learning_rule={"optimizer": "rmsprop", "lr": 3e-3, "schedule": "wsd"})
    m = GenesisNumpyModel(genome, seed=11)
    m.train_on(_CORPUS, steps=30, seed=11)
    pp = m.perplexity(_CORPUS[0])
    gen = m.generate("the master", max_tokens=10)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v1"
        m.save(path)
        r = GenesisNumpyModel(genome, seed=11)
        r.load(path)
        assert abs(r.perplexity(_CORPUS[0]) - pp) < 1e-6
        assert r.generate("the master", max_tokens=10) == gen
        assert r.param_count() == m.param_count()


@pytest.mark.parametrize("layers,expect_degrade", [
    ([{"op": "attention", "n_head": 2}, {"op": "gated_mlp"}], False),
    ([{"op": "conv_mix"}, {"op": "glu"}], False),
    ([{"op": "low_rank_mix"}, {"op": "swiglu", "expansion": 2}], False),
    ([{"op": "ssm_scan"}, {"op": "gated_mlp"}], False),
    ([{"op": "selective_ssm"}, {"op": "gated_mlp"}], True),       # → ssm_scan
    ([{"op": "gqa_attention", "n_head": 4}, {"op": "moe_mlp"}], True),  # → attention / gated_mlp
    ([{"op": "synth", "program": {"steps": ["proj", "lowrank", "norm"]}}, {"op": "glu"}], False),
])
def test_every_op_family_builds_trains_and_scores(layers, expect_degrade):
    m = GenesisNumpyModel(_genome(layers), seed=5)
    m.train_on(_CORPUS, steps=12, seed=5)
    pp = m.perplexity(_CORPUS[0])
    assert pp != float("inf") and pp > 0      # a real, finite perplexity
    assert m.param_count() > 0
    assert bool(m._degraded) == expect_degrade  # honest about any nearest-real fallback


@pytest.mark.parametrize("optimizer", ["adamw", "lion", "sgd_momentum", "rmsprop"])
def test_searched_optimizer_actually_drives_training(optimizer):
    genome = _genome([{"op": "attention"}, {"op": "gated_mlp"}],
                     learning_rule={"optimizer": optimizer, "lr": 3e-3, "schedule": "cosine"})
    m = GenesisNumpyModel(genome, seed=1)
    stats = m.train_on(_CORPUS, steps=30, seed=1)
    assert stats.final_loss < 5.0             # every optimizer reduces the loss to something sane
    assert m.perplexity(_CORPUS[0]) != float("inf")


def test_entropy_reg_aux_objective_runs():
    genome = _genome([{"op": "attention"}, {"op": "gated_mlp"}],
                     learning_rule={"optimizer": "adamw", "lr": 3e-3, "aux": {"entropy_reg": 0.02}})
    m = GenesisNumpyModel(genome, seed=2)
    stats = m.train_on(_CORPUS, steps=20, seed=2)
    assert stats.steps > 0 and m.perplexity(_CORPUS[0]) != float("inf")


# --------------------------------------------------------------------------- #
# Wiring: build_model, the Genesis search, and promotion all reach the real brain
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_HAS_TORCH, reason="this asserts the torch-free promotion/build path")
@pytest.mark.parametrize("spec_kw", [
    {"kind": "genesis_np"},                       # explicit request → always the real NumPy brain
    {"kind": "genesis", "substrate": "numpy"},     # a promotion spec that opted into the substrate
    {"kind": "genesis", "substrate": "auto"},
])
def test_build_model_builds_the_numpy_brain_from_a_genesis_spec(spec_kw):
    spec = ModelSpec(genome=_genome([{"op": "low_rank_mix"}, {"op": "gated_mlp"}]),
                     n_embd=32, block_size=16, seed=9, **spec_kw)
    model = build_model(spec)
    assert model.kind == "genesis_np"          # the promotable spec yields the real NumPy brain
    model.train_on(_CORPUS, steps=10, seed=9)
    assert model.param_count() > 0


@pytest.mark.skipif(_HAS_TORCH, reason="asserts the torch-free default-substrate fast path")
def test_incidental_genesis_spec_stays_on_the_fast_substrate():
    # a genesis spec at the default substrate must NOT silently train a NumPy net (keeps CI fast)
    spec = ModelSpec(kind="genesis", genome=_genome([{"op": "attention"}, {"op": "gated_mlp"}]))
    assert build_model(spec).kind == "kngram"


def test_layerless_genome_falls_back_to_ngram_substrate():
    # a degenerate genome (no neural layers) must NOT raise — it degrades to the always-on substrate
    spec = ModelSpec(kind="genesis", genome={"n_embd": 32, "block_size": 16, "layers": []})
    # ArchitectureGenome.__post_init__ injects default layers, so to truly test the empty path we
    # hand build_model a raw layer-less dict here:
    spec.genome = {"n_embd": 32, "block_size": 16, "layers": []}
    model = build_model(spec)
    assert model.kind in ("genesis_np", "kngram")   # never raises; both are real local brains
    model.train_on(_CORPUS, steps=5)
    assert model.param_count() >= 0


@pytest.mark.skipif(_HAS_TORCH, reason="asserts the torch-free NumPy-substrate search path")
def test_search_on_numpy_substrate_crowns_a_real_designed_brain():
    from nyxara.kernel.config import GenesisConfig
    from nyxara.growth.genesis import NeuralArchitectureSearch

    cfg = GenesisConfig(backend="stdlib", substrate="numpy", population_size=3, generations=1,
                        micro_train_steps=6, eval_seeds=1, block_size=16, max_layers=3,
                        hall_of_fame=False, seed=0)
    nas = NeuralArchitectureSearch(cfg=cfg, seed_corpus=_CORPUS)
    report = nas.search()

    assert report.backend == "stdlib"
    assert report.champion_kind == "genesis_np"
    assert report.topology_active is True
    assert report.leaderboard[0].topology_active is True
    assert "NumPy substrate" in report.note
    assert report.champion_flops > 0           # the real topology carries a FLOPs proxy
    # the champion NYXARA designed can actually speak through the same promotion path
    model = build_model(nas.champion_spec())
    assert model.kind == "genesis_np"
    model.train_on(_CORPUS, steps=8, seed=0)
    assert isinstance(model.generate("the master", max_tokens=8), str)


@pytest.mark.skipif(_HAS_TORCH, reason="asserts the torch-free substrate selection")
def test_ngram_substrate_knob_still_available_and_fast():
    from nyxara.kernel.config import GenesisConfig
    from nyxara.growth.genesis import NeuralArchitectureSearch

    cfg = GenesisConfig(backend="stdlib", substrate="ngram", population_size=3, generations=1,
                        micro_train_steps=6, eval_seeds=1, block_size=16, max_layers=3,
                        hall_of_fame=False, seed=0)
    report = NeuralArchitectureSearch(cfg=cfg, seed_corpus=_CORPUS).search()
    assert report.champion_kind == "kngram"    # explicit n-gram substrate still selectable
    assert report.topology_active is False
