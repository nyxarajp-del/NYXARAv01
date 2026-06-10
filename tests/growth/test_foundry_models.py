"""Tests for nyxara.growth.foundry_models."""

from __future__ import annotations

import pytest

from nyxara.growth.foundry_models import (NgramByteLM, ModelSpec, build_model,
                                          load_active_model, _HAS_TORCH)

CORPUS = ["the master is jp. nyxara serves the master."] * 8 + [
    "loyalty to the master is absolute and never changes."] * 8


# -------------------- NgramByteLM (always available) -------------------- #
def test_untrained_model_is_high_perplexity():
    lm = NgramByteLM(order=3)
    assert lm.param_count() == 0
    assert lm.perplexity(CORPUS[0]) > 100   # ~uniform over 256 bytes


def test_training_from_scratch_lowers_perplexity():
    untrained = NgramByteLM(order=3).perplexity(CORPUS[0])
    lm = NgramByteLM(order=3)
    stats = lm.train_on(CORPUS, seed=1)
    assert lm.param_count() > 0
    assert stats.tokens > 0
    assert lm.perplexity(CORPUS[0]) < untrained


def test_generate_is_deterministic_for_fixed_seed():
    lm = NgramByteLM(order=3, seed=7)
    lm.train_on(CORPUS, seed=1)
    assert lm.generate("the master", max_tokens=30) == lm.generate("the master", max_tokens=30)


def test_save_load_round_trip(tmp_path):
    lm = NgramByteLM(order=3)
    lm.train_on(CORPUS, seed=1)
    pp = lm.perplexity(CORPUS[0])
    gen = lm.generate("the master", max_tokens=30)
    lm.save(tmp_path / "v1")
    reloaded = NgramByteLM()
    reloaded.load(tmp_path / "v1")
    assert reloaded.perplexity(CORPUS[0]) == pp
    assert reloaded.generate("the master", max_tokens=30) == gen


def test_perplexity_finite_on_unseen_text():
    lm = NgramByteLM(order=3)
    lm.train_on(CORPUS, seed=1)
    # smoothing guarantees a finite score even for never-seen bytes
    assert lm.perplexity("\x00\x01\x02 totally novel ¬unicode∂") != float("inf")


# -------------------- factory & graceful degradation -------------------- #
def test_build_model_auto_degrades_without_torch():
    m = build_model(ModelSpec(kind="auto"))
    assert m.kind == ("nanogpt" if _HAS_TORCH else "ngram")


def test_build_model_nanogpt_never_raises_on_bare_machine():
    # asking for nanogpt without torch must fall back, never raise
    m = build_model(ModelSpec(kind="nanogpt"))
    assert m.kind == ("nanogpt" if _HAS_TORCH else "ngram")


def test_load_active_model_reads_promoted_version(tmp_path):
    # write a version dir + spec.json + active pointer the way the foundry would
    import json
    from pathlib import Path

    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)
    root = tmp_path / "foundry"
    settings.llm.self_model_dir = root
    spec = ModelSpec(kind="ngram", ngram_order=3)
    lm = build_model(spec)
    lm.train_on(CORPUS, seed=1)
    vdir = root / "v1"
    lm.save(vdir)
    (vdir / "spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    (root / "active").write_text("v1", encoding="utf-8")

    loaded = load_active_model(settings)
    assert loaded.kind == "ngram"
    assert loaded.param_count() == lm.param_count()


# -------------------- optional torch backend -------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_nanogpt_trains_from_scratch():
    from nyxara.growth.foundry_models import NanoGPTModel
    gpt = NanoGPTModel(ModelSpec(kind="nanogpt", n_layer=1, n_head=2, n_embd=32,
                                 block_size=32, seed=1))
    assert gpt.param_count() > 0
    gpt.train_on(CORPUS, steps=40, seed=1)
    assert gpt.perplexity(CORPUS[0]) != float("inf")
