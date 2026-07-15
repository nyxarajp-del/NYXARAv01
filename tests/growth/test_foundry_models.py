"""Tests for nyxara.growth.foundry_models."""

from __future__ import annotations

import pytest

from nyxara.growth.foundry_models import (NgramByteLM, ModelSpec, build_model,
                                          load_active_model, strongest_backend, _HAS_TORCH)

try:
    import numpy  # noqa: F401
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001
    _HAS_NUMPY = False

# The strongest backend a plain "neural" request (auto/nanogpt/lora) resolves to on THIS machine:
# a real from-scratch NumPy transformer when torch is absent but NumPy is present, else the
# always-on word-level Kneser-Ney n-gram. (With torch it is a NanoGPT — kind "nanogpt".)
_NEURAL_KIND = "nanogpt" if _HAS_TORCH else ("genesis_np" if _HAS_NUMPY else "kngram")

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
def test_build_model_auto_builds_the_strongest_available_brain():
    # "auto" must build a REAL model, as neural as this machine allows: a NanoGPT with torch, a
    # from-scratch NumPy transformer on a torch-less-but-NumPy box (not a toy n-gram), else kngram.
    m = build_model(ModelSpec(kind="auto"))
    assert m.kind == _NEURAL_KIND
    if not _HAS_TORCH:
        assert strongest_backend() == _NEURAL_KIND


@pytest.mark.skipif(_HAS_TORCH or not _HAS_NUMPY, reason="covers the torch-less + NumPy box")
def test_neural_request_builds_real_numpy_transformer_without_torch():
    # THE upgrade: a neural request on a CPU/NumPy box builds a genuine trained neural net, not a
    # byte-count table. It has real parameters and its perplexity drops after training.
    m = build_model(ModelSpec(kind="nanogpt", seed=1))
    assert m.kind == "genesis_np"
    before = m.perplexity(CORPUS[0])
    m.train_on(CORPUS, steps=40, seed=1)
    after = m.perplexity(CORPUS[0])
    assert m.param_count() > 1000          # a real neural net, orders larger than an n-gram table
    assert after < before                  # it genuinely learned via backprop


def test_build_model_nanogpt_never_raises_on_bare_machine():
    # asking for nanogpt must never raise regardless of installed deps
    m = build_model(ModelSpec(kind="nanogpt"))
    assert m.kind == _NEURAL_KIND


def test_load_active_model_reads_promoted_version(tmp_path):
    # write a version dir + spec.json + active pointer the way the foundry would
    import json

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


# A small but REAL natural-English corpus with a held-out split (same distribution).
_REAL_TRAIN = [
    "the quick brown fox jumps over the lazy dog near the river bank",
    "she sells sea shells by the sea shore in the early morning light",
    "all that glitters is not gold and all who wander are not truly lost",
    "the rain in spain falls mainly on the wide and open plain at dawn",
    "knowledge is power but wisdom is knowing how to use it with care",
    "a journey of a thousand miles begins with a single careful step",
]
_REAL_HOLDOUT = [
    "the quick brown fox runs over the lazy dog by the river at dawn",
    "wisdom is knowing how to use knowledge with patience and care",
]


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_nanogpt_really_learns_lowers_heldout_perplexity():
    """Real neural training: a from-zero GPT must materially reduce perplexity on held-out
    real text it never trained on — i.e. it generalises, not just memorises."""
    import statistics

    from nyxara.growth.foundry_models import NanoGPTModel
    gpt = NanoGPTModel(ModelSpec(kind="nanogpt", n_layer=2, n_head=2, n_embd=64,
                                 block_size=64, seed=0))
    before = statistics.mean(gpt.perplexity(t) for t in _REAL_HOLDOUT)
    gpt.train_on(_REAL_TRAIN, steps=200, seed=0)
    after = statistics.mean(gpt.perplexity(t) for t in _REAL_HOLDOUT)
    assert after < before * 0.6, f"held-out perplexity barely moved: {before:.1f} -> {after:.1f}"


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_lora_request_uses_neural_model_when_torch_present():
    # asking for LoRA on a torch machine without the transformers/peft stack must still yield a
    # real neural model (from-zero NanoGPT), never a drop to the n-gram backend
    m = build_model(ModelSpec(kind="lora"))
    assert m.kind != "ngram"


def test_explicit_ngram_is_never_upgraded():
    # an explicit n-gram request stays pure-stdlib even when torch is available
    assert build_model(ModelSpec(kind="ngram")).kind == "ngram"


# -------------------- WordKNGramLM (coherent word-level Kneser-Ney) -------------------- #
from nyxara.growth.foundry_models import WordKNGramLM, _word_detokenize  # noqa: E402

_PROSE = [
    "NYXARA serves the Master with loyalty and honesty.",
    "The mind proposes; the kernel disposes; the Master is sovereign.",
    "NYXARA is loyal to the Master and always honest.",
    "The Master is sovereign and NYXARA serves with loyalty.",
] * 8


def test_kngram_training_lowers_perplexity():
    untrained = WordKNGramLM(order=3).perplexity(_PROSE[0])
    lm = WordKNGramLM(order=3)
    lm.train_on(_PROSE)
    assert lm.perplexity(_PROSE[0]) < untrained


def test_kngram_generates_coherent_known_words():
    lm = WordKNGramLM(order=3, seed=3)
    lm.train_on(_PROSE)
    text = lm.generate("The Master", max_tokens=20)
    assert text.strip()                                  # non-empty
    vocab = set(lm.tok2id)
    produced = set(text.replace(".", " ").replace(";", " ").split())
    # every emitted word is a real token the model learned (no byte gibberish)
    assert produced and produced <= vocab


def test_kngram_generation_deterministic_for_seed():
    a = WordKNGramLM(order=3, seed=11); a.train_on(_PROSE)
    b = WordKNGramLM(order=3, seed=11); b.train_on(_PROSE)
    assert a.generate("NYXARA", max_tokens=15) == b.generate("NYXARA", max_tokens=15)


def test_kngram_perplexity_finite_on_unseen_text():
    lm = WordKNGramLM(order=3)
    lm.train_on(_PROSE)
    p = lm.perplexity("an entirely unrelated sentence about quantum widgets")
    assert p == p and p != float("inf")                  # finite (KN never zero-probs)


def test_kngram_save_load_round_trip(tmp_path):
    lm = WordKNGramLM(order=3, seed=5)
    lm.train_on(_PROSE)
    lm.save(tmp_path)
    reloaded = WordKNGramLM()
    reloaded.load(tmp_path)
    assert abs(reloaded.perplexity(_PROSE[0]) - lm.perplexity(_PROSE[0])) < 1e-9
    assert reloaded.generate("The Master", max_tokens=10) == lm.generate("The Master",
                                                                         max_tokens=10)


def test_kngram_untrained_generate_is_empty():
    assert WordKNGramLM(order=3).generate("anything", max_tokens=10) == ""


def test_kngram_default_sampling_is_unchanged():
    # the new sampling knobs default to OFF, reproducing the historical pure-KN sampling exactly
    lm = WordKNGramLM(order=3, seed=7); lm.train_on(_PROSE)
    base = lm.generate("The Master", max_tokens=20)
    same = lm.generate("The Master", max_tokens=20, top_k=0, top_p=0.0, repetition_penalty=1.0)
    assert base == same


def test_kngram_hardened_sampling_stays_in_vocab_and_deterministic():
    lm = WordKNGramLM(order=3, seed=7); lm.train_on(_PROSE)
    a = lm.generate("The Master", max_tokens=20, top_k=8, top_p=0.92, repetition_penalty=1.3)
    b = lm.generate("The Master", max_tokens=20, top_k=8, top_p=0.92, repetition_penalty=1.3)
    assert a == b                                        # fixed seed -> deterministic
    if a.strip():
        produced = set(a.replace(".", " ").replace(";", " ").split())
        assert produced <= set(lm.tok2id)               # still only real learned tokens


def test_detokenizer_spacing():
    assert _word_detokenize(["Hello", ",", "world", "!"]) == "Hello, world!"


def test_factory_kngram_explicit_and_fallback():
    assert build_model(ModelSpec(kind="kngram")).kind == "kngram"
    assert build_model(ModelSpec(kind="ngram")).kind == "ngram"   # byte model stays explicit
    # a neural request resolves to the strongest brain this machine can run (neural, not a toy)
    assert build_model(ModelSpec(kind="auto")).kind == _NEURAL_KIND


def test_kngram_promotes_and_loads_as_self_brain(tmp_path):
    # mirror how the foundry persists + how SelfProvider reloads the own brain
    import json
    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)
    root = tmp_path / "foundry"
    settings.llm.self_model_dir = root
    spec = ModelSpec(kind="kngram", ngram_order=3)
    lm = build_model(spec)
    lm.train_on(_PROSE, seed=1)
    vdir = root / "v1"
    lm.save(vdir)
    (vdir / "spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    (root / "active").write_text("v1", encoding="utf-8")
    loaded = load_active_model(settings)
    assert loaded.kind == "kngram"
    assert loaded.generate("The Master", max_tokens=10) == lm.generate("The Master",
                                                                       max_tokens=10)


# --------------------------------------------------------------------------- #
# LoRA rank auto-scaling (no heavy deps — pure rank math + spec serialisation)
# --------------------------------------------------------------------------- #
from nyxara.growth.foundry_models import LoRAModel  # noqa: E402


def _fake_base(hidden):
    return type("B", (), {"config": type("C", (), {"hidden_size": hidden})()})()


def test_infer_lora_rank_scales_with_base_width():
    assert LoRAModel._infer_lora_rank(_fake_base(2), 8) == 8        # tiny base -> floor
    assert LoRAModel._infer_lora_rank(_fake_base(768), 8) == 24     # gpt2-scale
    assert LoRAModel._infer_lora_rank(_fake_base(4096), 8) == 128   # 7B-scale, clamped sane
    assert LoRAModel._infer_lora_rank(_fake_base(99999), 8) == 256  # hard cap


def test_infer_lora_rank_falls_back_when_width_unreadable():
    assert LoRAModel._infer_lora_rank(object(), 16) == 16           # no config -> explicit r


def test_modelspec_lora_r_auto_round_trip():
    spec = ModelSpec(kind="lora", lora_r_auto=True)
    assert ModelSpec.from_dict(spec.to_dict()).lora_r_auto is True
    assert ModelSpec().lora_r_auto is False                         # default off for direct use


def test_modelspec_trust_remote_code_round_trip():
    # needed for an exotic base that ships custom modeling code; must survive spec.json persistence
    spec = ModelSpec(kind="lora", base_model="exotic/custom-arch", trust_remote_code=True)
    assert spec.to_dict()["trust_remote_code"] is True
    assert ModelSpec.from_dict(spec.to_dict()).trust_remote_code is True
    assert ModelSpec().trust_remote_code is False                   # default off for a stock base
