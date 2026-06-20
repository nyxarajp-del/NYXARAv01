"""Tests for the LoRA fine-tuning backend (growth/foundry_models.py).

Two layers:
* logic-level tests that need no heavy deps (spec serialisation, factory fallback contract);
* a real end-to-end LoRA test, gated on torch+transformers+peft AND model availability — it
  skips (never fails) on a machine without the stack or without network for the tiny base.
"""

from __future__ import annotations

import json

import pytest

from nyxara.growth.foundry_models import (ModelSpec, WordKNGramLM, _HAS_LORA, _HAS_TORCH,
                                          build_model)

TINY = "sshleifer/tiny-gpt2"


# --------------------------------------------------------------------------- #
# Logic-level (no heavy deps)
# --------------------------------------------------------------------------- #
def test_spec_serialisation_roundtrips_lora_fields():
    spec = ModelSpec(kind="lora", base_model="x/y", lora_r=4, lora_alpha=8,
                     lora_dropout=0.1, lora_lr=1e-3, max_seq_len=64, device="cpu")
    d = spec.to_dict()
    assert d["base_model"] == "x/y" and d["lora_r"] == 4 and d["max_seq_len"] == 64
    again = ModelSpec.from_dict(d)
    assert again == spec


def test_spec_from_dict_tolerates_missing_lora_fields():
    # an old persisted spec (pre-LoRA) must still load, defaulting the new knobs
    old = {"kind": "ngram", "ngram_order": 3, "seed": 0}
    spec = ModelSpec.from_dict(old)
    assert spec.kind == "ngram" and spec.base_model == "sshleifer/tiny-gpt2"


def test_build_model_lora_never_raises_for_missing_deps():
    # contract: build_model never raises; it degrades to the best available real backend. With
    # torch present a from-zero NanoGPT is used (genuine neural training); a machine without torch
    # falls back to the coherent word-level Kneser-Ney brain (WordKNGramLM), not byte gibberish.
    model = build_model(ModelSpec(kind="lora", base_model="does/not/exist"))
    assert model.kind in ("lora", "nanogpt", "kngram")
    if not _HAS_TORCH:
        assert isinstance(model, WordKNGramLM)


# --------------------------------------------------------------------------- #
# QLoRA (4-bit) — pure decision/config logic, no GPU or deps needed
# --------------------------------------------------------------------------- #
def test_spec_roundtrips_qlora_fields():
    spec = ModelSpec(kind="lora", base_model="Qwen/Qwen2.5-7B", load_in_4bit=True,
                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16",
                     bnb_4bit_use_double_quant=False, gradient_checkpointing=False)
    again = ModelSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.load_in_4bit and again.bnb_4bit_quant_type == "nf4"


def test_spec_from_dict_tolerates_missing_qlora_fields():
    # a spec persisted before QLoRA must still load, defaulting the new knobs to off/safe
    spec = ModelSpec.from_dict({"kind": "lora", "base_model": "x/y"})
    assert spec.load_in_4bit is False and spec.gradient_checkpointing is True


def test_should_quantize_requires_request_bnb_and_cuda():
    from nyxara.growth.foundry_models import _should_quantize
    on = ModelSpec(kind="lora", load_in_4bit=True)
    off = ModelSpec(kind="lora", load_in_4bit=False)
    # requested + both capabilities present -> quantize
    assert _should_quantize(on, has_bnb=True, has_cuda=True) is True
    # never quantize when not requested, even if capable
    assert _should_quantize(off, has_bnb=True, has_cuda=True) is False
    # requested but a capability missing -> degrade to full precision (no crash path)
    assert _should_quantize(on, has_bnb=False, has_cuda=True) is False
    assert _should_quantize(on, has_bnb=True, has_cuda=False) is False


def test_quant_kwargs_reflects_spec():
    from nyxara.growth.foundry_models import _quant_kwargs
    spec = ModelSpec(kind="lora", load_in_4bit=True, bnb_4bit_quant_type="fp4",
                     bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=False)
    kw = _quant_kwargs(spec)
    assert kw == {"load_in_4bit": True, "bnb_4bit_quant_type": "fp4",
                  "bnb_4bit_compute_dtype": "float16", "bnb_4bit_use_double_quant": False}


def test_qlora_degrades_on_cpu_machine():
    # On this (CPU/CI) box _should_quantize must be False even when 4-bit is requested,
    # so the LoRA backend would load full-precision rather than attempt an impossible 4-bit load.
    from nyxara.growth.foundry_models import _should_quantize
    assert _should_quantize(ModelSpec(kind="lora", load_in_4bit=True)) is False


# --------------------------------------------------------------------------- #
# Real end-to-end LoRA (deps + network gated)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def lora_model():
    if not _HAS_LORA:
        pytest.skip("torch+transformers+peft not installed")
    from nyxara.growth.foundry_models import LoRAModel
    spec = ModelSpec(kind="lora", base_model=TINY, lora_r=4, lora_alpha=8,
                     lora_dropout=0.0, lora_lr=5e-3, max_seq_len=32, device="cpu", seed=1)
    try:
        return LoRAModel(spec)
    except Exception as exc:  # noqa: BLE001 — no network for the base, or load failure
        pytest.skip(f"tiny base unavailable: {exc}")


def test_lora_trains_only_the_adapter(lora_model):
    # only the low-rank adapter is trainable — a tiny fraction of the parameters
    n_lora = lora_model.param_count()
    assert n_lora > 0
    total = sum(p.numel() for p in lora_model.net.parameters())
    assert n_lora < total            # the base is frozen


def test_lora_train_generate_perplexity(lora_model):
    corpus = ["the master is jp. nyxara serves the master."] * 6 + \
             ["loyalty to the master is absolute."] * 6
    pp_before = lora_model.perplexity(corpus[0])
    stats = lora_model.train_on(corpus, steps=15, seed=1)
    pp_after = lora_model.perplexity(corpus[0])
    assert stats.steps == 15 and stats.tokens > 0
    assert pp_after <= pp_before     # adapter learned (or at least did not worsen)
    out = lora_model.generate("the master", max_tokens=4)
    assert isinstance(out, str)


def test_lora_save_load_roundtrip(lora_model, tmp_path):
    from nyxara.growth.foundry_models import LoRAModel
    lora_model.train_on(["the master is jp."] * 4, steps=5, seed=1)
    target = lora_model.perplexity("the master is jp.")
    vdir = tmp_path / "v1"
    lora_model.save(vdir)
    assert (vdir / "spec.json").exists() and (vdir / "adapter").is_dir()
    # spec.json records the base so any machine can rebuild the model
    assert json.loads((vdir / "spec.json").read_text())["base_model"] == TINY

    reloaded = LoRAModel(lora_model.spec)
    reloaded.load(vdir)
    assert reloaded.param_count() == lora_model.param_count()
    assert abs(reloaded.perplexity("the master is jp.") - target) < 1.0


def test_build_model_returns_lora_when_available(lora_model):
    # with the stack present, the factory yields a real LoRA model for kind="lora"
    model = build_model(ModelSpec(kind="lora", base_model=TINY, max_seq_len=32, device="cpu"))
    assert model.kind == "lora"
