"""Tests for Step-2 real neural substrate — the foundry can reach genuine GPT-2 scale.

The default ``custom`` profile keeps the tiny, CPU-/CI-runnable model (so nothing here
needs torch), while a named profile such as ``gpt2`` fixes the canonical 124M-parameter
GPT-2 architecture. We verify the dimensions, the parameter estimate, and that a chosen
profile actually flows through into the trained model's recorded spec.
"""

from __future__ import annotations

from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import FoundryConfig, NyxaraSettings, Profile


# -------------------- profile -> dimensions -------------------- #
def test_default_profile_is_custom_and_tiny():
    cfg = FoundryConfig()
    assert cfg.profile == "custom"
    dims = cfg.resolved_dims()
    assert dims == {"n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 64}
    # the CI/default substrate stays small (well under a million params)
    assert cfg.estimated_params() < 5_000_000


def test_gpt2_profile_is_canonical_gpt2():
    cfg = FoundryConfig(profile="gpt2")
    assert cfg.resolved_dims() == {"n_layer": 12, "n_head": 12,
                                   "n_embd": 768, "block_size": 1024}


def test_gpt2_profile_reaches_gpt2_scale():
    cfg = FoundryConfig(profile="gpt2")
    # the requested minimum: GPT-2 scale (≥ 117M params); canonical GPT-2 is ~124M
    assert cfg.estimated_params() >= 117_000_000
    assert cfg.estimated_params() < 200_000_000


def test_gpt2_medium_is_larger_than_gpt2():
    assert (FoundryConfig(profile="gpt2-medium").estimated_params()
            > FoundryConfig(profile="gpt2").estimated_params())


def test_custom_profile_honours_explicit_fields():
    cfg = FoundryConfig(profile="custom", n_embd=256, n_layer=6, n_head=8, block_size=128)
    assert cfg.resolved_dims() == {"n_layer": 6, "n_head": 8,
                                   "n_embd": 256, "block_size": 128}


# -------------------- the profile flows into the foundry -------------------- #
def test_profile_drives_trained_model_spec(tmp_path):
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    # n-gram backend keeps the test torch-free; the spec still records the resolved dims,
    # proving a named profile overrides the raw (tiny) dimension fields end-to-end.
    settings.foundry.backend = "ngram"
    settings.foundry.profile = "gpt2"
    f = Foundry(settings=settings, seed_corpus=["nyxara is loyal to the master"] * 8)
    _model, version = f.train_candidate()
    assert version.spec["n_embd"] == 768
    assert version.spec["n_layer"] == 12
    assert version.spec["block_size"] == 1024
