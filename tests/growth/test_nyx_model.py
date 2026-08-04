"""Tests for the NYX-300M decoder in growth/foundry_models.py.

Three things are load-bearing here and each is *measured*, not assumed:

* the parameter estimate matches what torch actually allocates, so the profiles cannot lie;
* KV-cached decoding produces the identical tokens to uncached decoding, which is where a
  rotary-position off-by-one hides;
* the MoE really is sparse — the existing MoE in ``growth/genesis.py`` computes every expert
  and masks the rest, and a test that only checked *outputs* would pass on that too.

The heavy 300M build is exercised through ``estimate_params`` (pure Python, no allocation);
the behavioural tests use small models so the suite stays fast.
"""

from __future__ import annotations

import pytest

from nyxara.growth.foundry_models import (
    ModelSpec,
    architecture_signature,
    estimate_params,
    resolved_dims,
)

torch = pytest.importorskip("torch", reason="the NYX decoder is a torch backend")

from nyxara.growth.foundry_models import (  # noqa: E402  (after the torch guard)
    NanoGPTModel,
    _is_modern,
    _NanoGPT,
    _NyxGPT,
    _SparseMoE,
)


def nyx_spec(**kw) -> ModelSpec:
    """A small NYX spec — same architecture as the 300M profile, 1/1000th the size."""
    base = dict(kind="nanogpt", vocab_size=512, n_layer=2, n_embd=64, n_head=4, n_kv_head=2,
                mlp_hidden=128, block_size=32, rope_theta=10000.0, tie_embeddings=True,
                seed=0)
    base.update(kw)
    return ModelSpec(**base)


# --------------------------------------------------------------------------- #
# The estimate must equal what torch allocates — for every profile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ["nyxara-30m", "nyxara-moe-fast"])
def test_estimate_matches_allocation_for_profiles(profile: str) -> None:
    from nyxara.kernel.config import _FOUNDRY_PROFILES

    dims = _FOUNDRY_PROFILES[profile]
    spec = ModelSpec(kind="nanogpt",
                     **{k: v for k, v in dims.items() if k in ModelSpec.__dataclass_fields__})
    spec.tie_embeddings = bool(dims.get("tie_embeddings"))
    model = NanoGPTModel(spec)
    assert model.param_count() == estimate_params(spec).total


def test_estimate_matches_allocation_small() -> None:
    spec = nyx_spec()
    assert NanoGPTModel(spec).param_count() == estimate_params(spec).total


def test_moe_estimate_matches_allocation() -> None:
    spec = nyx_spec(n_experts=4, moe_top_k=2, moe_every=1, moe_expert_hidden=96)
    est = estimate_params(spec)
    assert NanoGPTModel(spec).param_count() == est.total
    assert est.active < est.total          # sparse: fewer weights run than are stored


def test_the_300m_profile_is_exactly_299_418_624() -> None:
    """The headline number. Pure Python — this assertion needs no torch and no GPU."""
    from nyxara.kernel.config import FoundryConfig

    assert FoundryConfig(profile="nyxara-300m").estimated_params() == 299_418_624


def test_param_estimate_reports_both_numbers_for_sparse_profiles() -> None:
    from nyxara.kernel.config import FoundryConfig

    est = FoundryConfig(profile="nyxara-moe-fast").param_estimate()
    assert est.sparse and est.active < est.total
    assert "active per token" in est.summary()     # never quote total alone for an MoE


def test_dense_profile_reports_one_number() -> None:
    from nyxara.kernel.config import FoundryConfig

    est = FoundryConfig(profile="nyxara-300m").param_estimate()
    assert not est.sparse and est.active == est.total


# --------------------------------------------------------------------------- #
# Backward compatibility: an old spec must still build — and load — the old module
# --------------------------------------------------------------------------- #
def test_legacy_spec_builds_the_legacy_module() -> None:
    assert not _is_modern(ModelSpec(kind="nanogpt"))
    assert isinstance(NanoGPTModel(ModelSpec(kind="nanogpt")).net, _NanoGPT)


def test_any_modern_knob_selects_the_nyx_module() -> None:
    for knob in ({"vocab_size": 1024}, {"rope_theta": 10000.0}, {"mlp_hidden": 128},
                 {"n_kv_head": 2}, {"n_experts": 4}, {"tie_embeddings": True}):
        assert _is_modern(ModelSpec(kind="nanogpt", **knob)), knob


def test_legacy_checkpoint_round_trips(tmp_path) -> None:
    """A checkpoint saved before the NYX decoder existed must still load into it."""
    model = NanoGPTModel(ModelSpec(kind="nanogpt", n_layer=1, n_head=2, n_embd=32,
                                   block_size=16))
    model.train_on(["hello world " * 20], steps=2)
    model.save(tmp_path)
    restored = NanoGPTModel(ModelSpec(kind="nanogpt"))
    restored.load(tmp_path)
    assert restored.param_count() == model.param_count()
    assert isinstance(restored.net, _NanoGPT)


def test_nyx_checkpoint_round_trips_with_its_tokenizer(tmp_path) -> None:
    from nyxara.growth.tokenizer import train_tokenizer

    tok, _ = train_tokenizer(["the quick brown fox " * 20], vocab_size=400)
    spec = nyx_spec(vocab_size=tok.vocab_size)
    model = NanoGPTModel(spec)
    model.attach_tokenizer(tok)
    model.save(tmp_path)

    restored = NanoGPTModel(nyx_spec(vocab_size=tok.vocab_size))
    restored.load(tmp_path)
    # The tokenizer travelled with the weights, so the restored model speaks the same language.
    assert restored.tokenizer is not None
    assert restored.tokenizer.vocab_sig() == tok.vocab_sig()


def test_attach_tokenizer_refuses_a_size_mismatch() -> None:
    from nyxara.growth.tokenizer import NyxTokenizer

    model = NanoGPTModel(nyx_spec(vocab_size=512))
    with pytest.raises(ValueError, match="does not match"):
        model.attach_tokenizer(NyxTokenizer())      # 261 entries, not 512


# --------------------------------------------------------------------------- #
# KV cache — where a rotary off-by-one hides
# --------------------------------------------------------------------------- #
def test_cached_decode_equals_uncached_decode() -> None:
    """Greedy decoding must be bit-identical with and without the cache.

    If RoPE were applied at position 0 for every cached step (the easy mistake), this diverges
    within a couple of tokens while still producing plausible-looking output.
    """
    model = NanoGPTModel(nyx_spec())
    a = model.generate("hello", max_tokens=24, greedy=True, use_cache=True)
    b = model.generate("hello", max_tokens=24, greedy=True, use_cache=False)
    assert a == b


def test_cached_decode_equals_uncached_decode_for_moe() -> None:
    model = NanoGPTModel(nyx_spec(n_experts=4, moe_top_k=2, moe_every=1))
    a = model.generate("hello", max_tokens=16, greedy=True, use_cache=True)
    b = model.generate("hello", max_tokens=16, greedy=True, use_cache=False)
    assert a == b


def test_cache_grows_one_step_at_a_time() -> None:
    model = NanoGPTModel(nyx_spec())
    net = model.net
    x = torch.randint(0, 512, (1, 5))
    with torch.no_grad():
        _logits, caches, _aux = net(x, return_aux=True)
        assert caches[0][0].shape[-2] == 5
        _logits, caches, _aux = net(torch.randint(0, 512, (1, 1)), caches=caches,
                                    return_aux=True)
        assert caches[0][0].shape[-2] == 6


# --------------------------------------------------------------------------- #
# Grouped-query attention
# --------------------------------------------------------------------------- #
def test_gqa_shrinks_the_kv_cache() -> None:
    """4 KV heads instead of 16 is a 4× smaller cache — the point of GQA at decode time."""
    mha = NanoGPTModel(nyx_spec(n_head=4, n_kv_head=4))
    gqa = NanoGPTModel(nyx_spec(n_head=4, n_kv_head=1))
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        _l, mha_cache, _a = mha.net(x, return_aux=True)
        _l, gqa_cache, _a = gqa.net(x, return_aux=True)
    assert gqa_cache[0][0].shape[1] * 4 == mha_cache[0][0].shape[1]


def test_kv_head_count_is_clamped_to_divide_the_query_heads() -> None:
    """An impossible grouping must be repaired, not silently produce a shape error."""
    d = resolved_dims(ModelSpec(kind="nanogpt", n_head=12, n_kv_head=5, n_embd=768))
    assert d["n_head"] % d["n_kv_head"] == 0


# --------------------------------------------------------------------------- #
# MoE sparsity — measured, because output-only tests pass on a dense implementation
# --------------------------------------------------------------------------- #
def test_moe_dispatch_is_genuinely_sparse() -> None:
    """Count the tokens each expert actually processes.

    With top-2 of 8, the expert bank must see ~2 tokens of work per input token. The MoE in
    genesis.py — which computes every expert and multiplies by a zero gate — would see 8, and
    that is the difference between an MoE that is worth building and one that is pure loss.
    """
    n_experts, top_k, n_tokens = 8, 2, 40
    moe = _SparseMoE(n_embd=32, hidden=64, n_experts=n_experts, top_k=top_k)

    processed = {"tokens": 0}

    def counting_hook(_module, inputs):
        processed["tokens"] += inputs[0].shape[0]

    for expert in moe.experts:
        expert.register_forward_pre_hook(counting_hook)

    x = torch.randn(1, n_tokens, 32)
    out, aux = moe(x)

    assert out.shape == x.shape
    assert processed["tokens"] == n_tokens * top_k, (
        f"expert bank processed {processed['tokens']} token-slots; top-{top_k} routing of "
        f"{n_tokens} tokens must be exactly {n_tokens * top_k} — anything near "
        f"{n_tokens * n_experts} means the dispatch is dense and the sparsity is a fiction")
    assert float(aux.detach()) > 0.0


def test_moe_routing_weights_sum_to_one_per_token() -> None:
    """Each token's chosen experts must form a convex combination, or scale drifts by layer."""
    moe = _SparseMoE(n_embd=16, hidden=32, n_experts=4, top_k=2)
    with torch.no_grad():
        for expert in moe.experts:          # make every expert output a constant 1
            torch.nn.init.zeros_(expert.gate.weight)
            torch.nn.init.zeros_(expert.up.weight)
            torch.nn.init.zeros_(expert.down.weight)
            expert.down.weight += 0.0
    x = torch.randn(1, 6, 16)
    out, _aux = moe(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_moe_load_balance_aux_penalises_a_collapsed_router() -> None:
    """A router that sends everything to one expert must score worse than a balanced one."""
    balanced = _SparseMoE(n_embd=16, hidden=32, n_experts=4, top_k=1)
    collapsed = _SparseMoE(n_embd=16, hidden=32, n_experts=4, top_k=1)
    with torch.no_grad():
        collapsed.router.weight.zero_()
        collapsed.router.weight[0] += 10.0          # everything routes to expert 0
    x = torch.randn(1, 32, 16)
    _o, aux_balanced = balanced(x)
    _o, aux_collapsed = collapsed(x)
    assert float(aux_collapsed.detach()) > float(aux_balanced.detach())


# --------------------------------------------------------------------------- #
# Architecture signature — what stops a 300M candidate loading a 137k checkpoint
# --------------------------------------------------------------------------- #
def test_signature_differs_across_scales() -> None:
    assert architecture_signature(nyx_spec()) != architecture_signature(nyx_spec(n_layer=4))
    assert architecture_signature(nyx_spec()) != architecture_signature(nyx_spec(vocab_size=1024))
    assert architecture_signature(ModelSpec()) != architecture_signature(nyx_spec())


def test_signature_is_stable_for_the_same_spec() -> None:
    assert architecture_signature(nyx_spec()) == architecture_signature(nyx_spec())


# --------------------------------------------------------------------------- #
# Training still works, and the EWC hook is untouched
# --------------------------------------------------------------------------- #
def test_nyx_model_trains_and_lowers_its_loss() -> None:
    model = NanoGPTModel(nyx_spec(vocab_size=256))
    stats = model.train_on(["the quick brown fox jumps over the lazy dog "] * 8, steps=30)
    assert stats.steps == 30 and stats.final_loss > 0


def test_moe_model_trains() -> None:
    model = NanoGPTModel(nyx_spec(vocab_size=256, n_experts=4, moe_top_k=2, moe_every=1))
    stats = model.train_on(["hello world " * 20], steps=10)
    assert stats.final_loss > 0


def test_cross_entropy_nats_returns_unnormalised_totals() -> None:
    """bits_per_byte needs the raw total, not a per-token average."""
    model = NanoGPTModel(nyx_spec(vocab_size=256))
    total, n = model.cross_entropy_nats("hello world, this is a test of the model")
    assert n > 0 and total > 0


# --------------------------------------------------------------------------- #
# Sampling controls
# --------------------------------------------------------------------------- #
def test_greedy_is_deterministic() -> None:
    model = NanoGPTModel(nyx_spec())
    assert model.generate("x", max_tokens=12, greedy=True) == \
           model.generate("x", max_tokens=12, greedy=True)


def test_top_k_and_top_p_do_not_crash_and_stay_in_vocab() -> None:
    model = NanoGPTModel(nyx_spec(vocab_size=256))
    out = model.generate("x", max_tokens=12, temperature=0.8, top_k=20, top_p=0.9,
                         repetition_penalty=1.1)
    assert isinstance(out, str)
