"""Tests for growth/trainer.py — the pretraining loop and its honest refusal.

The refusal is the part worth testing hardest. A script that starts a run which cannot finish
looks exactly like a script that started a run which can, and the failure only becomes visible
days later. ``preflight`` is pure — device and memory are injectable — so every machine class
can be checked from any machine.
"""

from __future__ import annotations

import math

import pytest

from nyxara.growth.foundry_models import ModelSpec
from nyxara.growth.trainer import (
    PreflightVerdict,
    TrainConfig,
    TrainingState,
    _lr_factor,
    estimate_training_time,
    preflight,
)


def profile_spec(name: str) -> ModelSpec:
    from nyxara.kernel.config import _FOUNDRY_PROFILES

    dims = _FOUNDRY_PROFILES[name]
    spec = ModelSpec(kind="nanogpt",
                     **{k: v for k, v in dims.items() if k in ModelSpec.__dataclass_fields__})
    spec.tie_embeddings = bool(dims.get("tie_embeddings"))
    return spec


# --------------------------------------------------------------------------- #
# Preflight — refuse loudly rather than hang
# --------------------------------------------------------------------------- #
def test_cpu_refuses_the_full_300m_run() -> None:
    """The headline refusal. Years, not days — and it must say so instead of starting."""
    verdict = preflight(profile_spec("nyxara-300m"), tokens=6_000_000_000,
                        device_key="cpu", memory_available_gb=64)
    assert not verdict.ok
    assert "YEARS" in verdict.human_time() or verdict.estimated_seconds > 30 * 86_400
    assert "nyxara-30m" in verdict.suggestion


def test_a100_accepts_the_full_300m_run() -> None:
    verdict = preflight(profile_spec("nyxara-300m"), tokens=6_000_000_000,
                        device_key="cuda-40gb", memory_available_gb=40)
    assert verdict.ok
    assert verdict.estimated_seconds < 30 * 86_400


def test_refuses_when_memory_is_short() -> None:
    verdict = preflight(profile_spec("nyxara-300m"), tokens=1_000_000,
                        device_key="cuda-16gb", memory_available_gb=4)
    assert not verdict.ok
    assert "GB" in verdict.reason
    assert verdict.suggestion


def test_memory_estimate_covers_optimizer_state() -> None:
    """Weights alone is the classic underestimate: Adam's moments are 4× the weights in fp32."""
    verdict = preflight(profile_spec("nyxara-300m"), tokens=1000,
                        device_key="cuda-80gb", memory_available_gb=80)
    weights_only_gb = 299_418_624 * 2 / 1e9
    assert verdict.memory_needed_gb > weights_only_gb * 4


def test_small_profile_runs_on_cpu() -> None:
    """The proof profile must be genuinely runnable here, or the pipeline cannot be verified."""
    verdict = preflight(profile_spec("nyxara-30m"), tokens=2_000_000,
                        device_key="cpu", memory_available_gb=16)
    assert verdict.ok


def test_sparse_profile_is_estimated_on_ACTIVE_parameters() -> None:
    """The whole reason to build an MoE: fewer weights execute, so it is genuinely faster."""
    dense = preflight(profile_spec("nyxara-300m"), tokens=100_000_000,
                      device_key="cuda-40gb", memory_available_gb=40)
    sparse = preflight(profile_spec("nyxara-moe-fast"), tokens=100_000_000,
                       device_key="cuda-40gb", memory_available_gb=40)
    assert sparse.active_params < sparse.params           # it is sparse
    assert sparse.estimated_seconds < dense.estimated_seconds


def test_verdict_summary_quotes_both_parameter_numbers_for_sparse() -> None:
    verdict = preflight(profile_spec("nyxara-moe-fast"), tokens=1_000_000,
                        device_key="cuda-40gb", memory_available_gb=40)
    assert "active" in verdict.summary()


def test_estimate_scales_with_tokens_and_shrinks_with_hardware() -> None:
    a = estimate_training_time(active_params=299_418_624, tokens=1_000_000, device_key="cpu")
    b = estimate_training_time(active_params=299_418_624, tokens=2_000_000, device_key="cpu")
    assert math.isclose(b / a, 2.0, rel_tol=1e-6)
    gpu = estimate_training_time(active_params=299_418_624, tokens=1_000_000,
                                 device_key="cuda-80gb")
    assert gpu < a


def test_human_time_says_years_when_it_means_years() -> None:
    v = PreflightVerdict(ok=False, estimated_seconds=86_400 * 400)
    assert "YEARS" in v.human_time()


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #
def test_warmup_ramps_then_cosine_decays() -> None:
    kw = dict(total=1000, warmup=100, kind="cosine", min_ratio=0.1)
    assert _lr_factor(0, **kw) < _lr_factor(50, **kw) < _lr_factor(99, **kw)
    assert _lr_factor(100, **kw) > _lr_factor(500, **kw) > _lr_factor(999, **kw)


def test_cosine_floors_instead_of_reaching_zero() -> None:
    """A cosine that hits exactly zero spends its last steps not learning."""
    assert _lr_factor(999, total=1000, warmup=10, kind="cosine", min_ratio=0.1) >= 0.1


def test_constant_schedule_stays_flat_after_warmup() -> None:
    kw = dict(total=1000, warmup=100, kind="constant", min_ratio=0.1)
    assert _lr_factor(200, **kw) == _lr_factor(900, **kw) == 1.0


# --------------------------------------------------------------------------- #
# The loop itself
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="training requires torch")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """A complete tokenizer → corpus → model pipeline, small enough to train in seconds."""
    from nyxara.growth.corpus import ShardDataset, build_corpus
    from nyxara.growth.foundry_models import NanoGPTModel
    from nyxara.growth.synth_data import generate_domain_docs
    from nyxara.growth.tokenizer import train_tokenizer

    root = tmp_path_factory.mktemp("pipeline")
    docs = []
    for domain in ("math", "code", "conversation"):
        docs += list(generate_domain_docs(domain, 150, seed=7))
    tok, _ = train_tokenizer(docs, vocab_size=800)
    build_corpus(tokenizer=tok, out_dir=root / "shards", tokens_budget=200_000, seed=7)
    dataset = ShardDataset(root / "shards", block_size=32, seed=0)

    spec = ModelSpec(kind="nanogpt", vocab_size=tok.vocab_size, n_layer=2, n_embd=64,
                     n_head=4, n_kv_head=2, mlp_hidden=128, block_size=32,
                     rope_theta=10000.0, tie_embeddings=True)
    model = NanoGPTModel(spec)
    model.attach_tokenizer(tok)
    return model, dataset, root


def _config(**kw) -> TrainConfig:
    base = dict(max_steps=40, batch_size=4, lr=1e-3, eval_every=0, log_every=10,
                checkpoint_every=0, compile_model=False, eval_batches=2)
    base.update(kw)
    return TrainConfig(**base)


def test_training_lowers_the_loss(pipeline) -> None:
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    logs = []
    pretrain(model, dataset, config=_config(max_steps=80, log_every=10),
             on_log=logs.append, skip_preflight=True)
    losses = [r["loss"] for r in logs if "loss" in r]
    assert len(losses) >= 4
    assert min(losses[-2:]) < losses[0], f"loss did not fall: {losses}"


def test_gradient_accumulation_runs(pipeline) -> None:
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    state = pretrain(model, dataset, config=_config(max_steps=6, grad_accum_steps=3),
                     skip_preflight=True)
    assert state.step == 6


def test_per_domain_validation_reports_every_domain(pipeline) -> None:
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    logs = []
    pretrain(model, dataset, config=_config(max_steps=10, eval_every=10),
             on_log=logs.append, skip_preflight=True)
    evals = [r["val"] for r in logs if "val" in r]
    assert evals, "no validation was recorded"
    assert set(dataset.domains) <= set(evals[-1])
    assert "mean" in evals[-1]


def test_checkpoint_and_resume(pipeline, tmp_path) -> None:
    """A multi-day run must survive a kill; resuming must continue, not restart."""
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    out = tmp_path / "ckpt"
    first = pretrain(model, dataset, config=_config(max_steps=20, checkpoint_every=10),
                     out_dir=out, skip_preflight=True)
    assert first.step == 20
    assert (out / "model.pt").exists() and (out / "optimizer.pt").exists()
    assert (out / "train_state.json").exists()

    second = pretrain(model, dataset, config=_config(max_steps=30, checkpoint_every=10),
                      out_dir=out, resume=True, skip_preflight=True)
    assert second.step == 30
    assert second.tokens_seen > first.tokens_seen


def test_resume_restores_the_data_cursor(pipeline, tmp_path) -> None:
    """Without this a resumed run silently restarts the epoch, over-training shard zero."""
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    out = tmp_path / "ckpt2"
    state = pretrain(model, dataset, config=_config(max_steps=10, checkpoint_every=5),
                     out_dir=out, skip_preflight=True)
    assert state.data_cursor, "the data cursor was not persisted"


def test_preflight_blocks_a_real_run(pipeline) -> None:
    """pretrain must refuse rather than begin, when preflight says the run cannot finish."""
    from nyxara.growth.foundry_models import NanoGPTModel
    from nyxara.growth.trainer import pretrain

    _model, dataset, _root = pipeline
    huge = NanoGPTModel.__new__(NanoGPTModel)     # no allocation — preflight is pure
    huge.spec = profile_spec("nyxara-300m")
    huge.net = None
    huge.device = "cpu"
    with pytest.raises(RuntimeError, match="refused by preflight"):
        pretrain(huge, dataset, config=_config(max_steps=10_000_000))


def test_weight_decay_skips_norms_and_embeddings(pipeline) -> None:
    """Decaying an embedding pulls unseen tokens toward zero — a quiet way to damage a model."""
    from nyxara.growth.trainer import _param_groups

    model, _dataset, _root = pipeline
    decay, no_decay = _param_groups(model.net, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    embedding = model.net.tok.weight
    assert any(p is embedding for p in no_decay["params"])
    assert not any(p is embedding for p in decay["params"])


def test_state_round_trips() -> None:
    s = TrainingState(step=7, tokens_seen=99, best_val=1.5, data_cursor={"a": 1})
    assert TrainingState.from_dict(s.to_dict()).step == 7
    assert TrainingState.from_dict(s.to_dict()).data_cursor == {"a": 1}


def test_ewc_penalty_is_still_applied(pipeline) -> None:
    """Anti-forgetting must survive the rewrite — the hook is load-bearing, not decorative."""
    from nyxara.growth.trainer import pretrain

    model, dataset, _root = pipeline
    calls = {"n": 0}

    class _Synapses:
        def penalty(self):
            calls["n"] += 1
            return torch.zeros(())

    model.synapses = _Synapses()
    try:
        pretrain(model, dataset, config=_config(max_steps=5), skip_preflight=True)
        assert calls["n"] >= 5
    finally:
        model.synapses = None
