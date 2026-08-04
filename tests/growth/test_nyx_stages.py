"""Tests for the stages after pretraining: SFT, DPO, latent heads, expansion, the oracle.

Each module has one property whose failure is invisible from the outside, and those are what
these tests are aimed at:

* SFT — the loss must cover the answer and *not* the prompt;
* DPO — the frozen reference must exist, or the objective has a degenerate optimum;
* latent head — VICReg must forbid the constant solution, which otherwise scores beautifully;
* expansion — a new expert must be reachable by the router and must be discarded if it hurts;
* oracle — only PROVEN items may enter the corpus, and the meta-loss may never touch the ruler.
"""

from __future__ import annotations

import json

import pytest

from nyxara.growth.foundry_models import ModelSpec
from nyxara.growth.oracle import (
    FrozenRuler,
    MetaLoss,
    OracleConfig,
    SelfPlayOracle,
)
from nyxara.growth.sft import IGNORE_INDEX, SFTConfig, build_examples, mask_to_answer

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from nyxara.growth.foundry_models import NanoGPTModel  # noqa: E402
from nyxara.growth.tokenizer import ASSISTANT_TAG, USER_TAG, train_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    from nyxara.growth.synth_data import generate_domain_docs

    docs = []
    for domain in ("math", "code", "conversation"):
        docs += list(generate_domain_docs(domain, 120, seed=11))
    tok, _ = train_tokenizer(docs, vocab_size=900)
    return tok


@pytest.fixture(scope="module")
def sft_docs():
    from nyxara.growth.synth_data import generate_domain_docs

    return list(generate_domain_docs("conversation", 60, seed=11))


def _model(tokenizer, **kw) -> NanoGPTModel:
    base = dict(kind="nanogpt", vocab_size=tokenizer.vocab_size, n_layer=2, n_embd=64,
                n_head=4, n_kv_head=2, mlp_hidden=128, block_size=128,
                rope_theta=10000.0, tie_embeddings=True)
    base.update(kw)
    model = NanoGPTModel(ModelSpec(**base))
    model.attach_tokenizer(tokenizer)
    return model


# --------------------------------------------------------------------------- #
# SFT — the mask is the whole point
# --------------------------------------------------------------------------- #
def test_mask_covers_the_answer_and_not_the_prompt(tokenizer) -> None:
    doc = f"{USER_TAG}\nwhat is two plus two\n\n{ASSISTANT_TAG}\nit is four\n"
    ids = tokenizer.encode(doc, add_eos=True)
    labels = mask_to_answer(ids, assistant_id=tokenizer.assistant_id)
    assert labels is not None

    cut = ids.index(tokenizer.assistant_id)
    assert all(v == IGNORE_INDEX for v in labels[:cut + 1]), "prompt tokens are supervised"
    assert any(v != IGNORE_INDEX for v in labels[cut + 1:]), "answer tokens are masked"
    # And the supervised span decodes back to the answer, not to the question.
    supervised = [v for v in labels if v != IGNORE_INDEX]
    assert "four" in tokenizer.decode(supervised)


def test_mask_returns_none_without_the_marker(tokenizer) -> None:
    """A guessed boundary would train her to emit her own role tag mid-reply."""
    assert mask_to_answer(tokenizer.encode("plain text, no marker"),
                          assistant_id=tokenizer.assistant_id) is None


def test_multi_turn_supervises_only_the_last_reply(tokenizer) -> None:
    doc = (f"{USER_TAG}\nfirst\n\n{ASSISTANT_TAG}\nreply one\n\n"
           f"{USER_TAG}\nsecond\n\n{ASSISTANT_TAG}\nreply two\n")
    ids = tokenizer.encode(doc, add_eos=True)
    labels = mask_to_answer(ids, assistant_id=tokenizer.assistant_id)
    supervised = tokenizer.decode([v for v in labels if v != IGNORE_INDEX])
    assert "two" in supervised and "reply one" not in supervised


def test_build_examples_counts_and_skips(tokenizer, sft_docs) -> None:
    from nyxara.growth.sft import SFTReport

    report = SFTReport()
    examples = build_examples(sft_docs + ["no marker here at all"], tokenizer=tokenizer,
                              block_size=256, report=report)
    assert examples
    assert report.skipped_no_marker == 1
    assert 0.0 < report.answer_fraction < 1.0, "some prompt must be masked, some answer kept"


def test_sft_runs_and_reports(tokenizer, sft_docs) -> None:
    from nyxara.growth.sft import supervised_finetune

    model = _model(tokenizer)
    report = supervised_finetune(model, sft_docs,
                                 config=SFTConfig(epochs=1, batch_size=2, lr=1e-4))
    assert report.examples > 0 and report.steps > 0
    assert report.answer_tokens > 0


def test_sft_refuses_without_a_tokenizer() -> None:
    from nyxara.growth.sft import supervised_finetune

    model = NanoGPTModel(ModelSpec(kind="nanogpt"))     # byte-level, no marker token
    report = supervised_finetune(model, ["anything"])
    assert report.examples == 0
    assert any("tokenizer" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# DPO — the reference model is what stops it destroying the model
# --------------------------------------------------------------------------- #
def test_dpo_runs_and_ranks(tokenizer) -> None:
    from nyxara.growth.dpo import DPOConfig, PreferencePair, preference_optimize

    model = _model(tokenizer)
    pairs = [PreferencePair(prompt=f"question {i}",
                            chosen=f"a careful, complete answer number {i}",
                            rejected="idk")
             for i in range(6)]
    report = preference_optimize(model, pairs,
                                 config=DPOConfig(epochs=1, batch_size=2, lr=1e-5))
    assert report.pairs == 6 and report.steps > 0
    assert 0.0 <= report.accuracy <= 1.0


def test_dpo_halts_when_the_policy_drifts_off_its_anchor(tokenizer) -> None:
    """Past a certain distance the model is no longer the one the gauntlet passed."""
    from nyxara.growth.dpo import DPOConfig, PreferencePair, preference_optimize

    model = _model(tokenizer)
    pairs = [PreferencePair(prompt=f"q{i}", chosen="yes " * 20, rejected="no " * 20)
             for i in range(8)]
    report = preference_optimize(model, pairs,
                                 config=DPOConfig(epochs=3, lr=1e-1, max_kl_drift=0.001))
    assert report.halted_on_drift
    assert any("gauntlet certified" in note for note in report.notes)


def test_dpo_needs_pairs_and_a_tokenizer(tokenizer) -> None:
    from nyxara.growth.dpo import preference_optimize

    assert preference_optimize(_model(tokenizer), []).pairs == 0


def test_pairs_from_flywheel_needs_both_sides(tmp_path) -> None:
    from nyxara.growth.dpo import pairs_from_flywheel

    store = tmp_path / "flywheel.jsonl"
    store.write_text("\n".join(json.dumps(r) for r in [
        {"prompt": "q1", "answer": "good", "provenance": "flywheel-verified"},
        {"prompt": "q1", "answer": "meh", "provenance": "flywheel"},
        {"prompt": "q2", "answer": "lonely", "provenance": "flywheel-verified"},
    ]), encoding="utf-8")
    pairs = pairs_from_flywheel(store)
    assert len(pairs) == 1
    assert pairs[0].chosen == "good" and pairs[0].rejected == "meh"


def test_pairs_from_flywheel_tolerates_a_missing_store(tmp_path) -> None:
    from nyxara.growth.dpo import pairs_from_flywheel

    assert pairs_from_flywheel(tmp_path / "nope.jsonl") == []


# --------------------------------------------------------------------------- #
# Latent head — VICReg must forbid the trivial solution
# --------------------------------------------------------------------------- #
def test_latent_head_produces_a_loss(tokenizer) -> None:
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, latent_enabled=True, horizon=2, latent_dim=32)
    hidden = torch.randn(2, 24, 64)
    loss, stats = head.auxiliary_loss(hidden)
    assert float(loss.detach()) > 0 and "latent" in stats


def test_vicreg_punishes_a_collapsed_latent(tokenizer) -> None:
    """Predicting a constant is the trivial optimum; without VICReg the head finds it."""
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, latent_enabled=True, horizon=2, latent_dim=32)
    varied = torch.randn(4, 24, 64)
    constant = torch.ones(4, 24, 64)
    _loss_v, stats_v = head.auxiliary_loss(varied)
    _loss_c, stats_c = head.auxiliary_loss(constant)
    assert stats_c["variance"] < stats_v["variance"], (
        "a collapsed latent must show lower variance — that is what the penalty keys on")


def test_ema_target_is_not_trained_by_gradient(tokenizer) -> None:
    """A target that trains alongside the predictor moves to meet it, and both collapse."""
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, latent_enabled=True, latent_dim=32)
    assert all(not p.requires_grad for p in head.latent.target.parameters())
    assert all(p.requires_grad for p in head.latent.encoder.parameters())


def test_ema_update_moves_the_target_toward_the_encoder(tokenizer) -> None:
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, latent_enabled=True, latent_dim=32, ema_decay=0.5)
    with torch.no_grad():
        for p in head.latent.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in head.latent.target.parameters()]
    head.step()
    after = list(head.latent.target.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_disabled_heads_cost_nothing(tokenizer) -> None:
    from nyxara.growth.latent_head import build_latent_head

    head = build_latent_head(_model(tokenizer))
    loss, stats = head.auxiliary_loss(torch.randn(2, 8, 64))
    assert float(loss) == 0.0 and not stats
    assert head.parameters() == []


def test_mtp_loss_aligns_each_head_with_its_offset(tokenizer) -> None:
    """Head d predicts d positions ahead; a misalignment here trains the heads on noise."""
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, mtp_enabled=True, n_predict=4)
    hidden = torch.randn(2, 12, 64)
    targets = torch.randint(0, tokenizer.vocab_size, (2, 12))
    loss = head.mtp.loss(hidden, targets)
    assert float(loss.detach()) > 0

    # A sequence shorter than the deepest offset must degrade, not raise.
    short = head.mtp.loss(torch.randn(1, 2, 64), torch.randint(0, 10, (1, 2)))
    assert float(short.detach()) >= 0.0


def test_mtp_contributes_to_the_auxiliary_loss(tokenizer) -> None:
    from nyxara.growth.latent_head import build_latent_head

    model = _model(tokenizer)
    head = build_latent_head(model, mtp_enabled=True, n_predict=3)
    hidden = torch.randn(2, 16, 64)
    targets = torch.randint(0, tokenizer.vocab_size, (2, 16))
    _no_targets, stats_none = head.auxiliary_loss(hidden)
    _with, stats = head.auxiliary_loss(hidden, targets)
    assert "mtp" not in stats_none and stats["mtp"] > 0


def test_speculative_report_measures_rather_than_asserts(tokenizer) -> None:
    """Whatever the speedup is, that is what gets reported — including below 1.0."""
    from nyxara.growth.latent_head import build_latent_head, measure_speculative_speedup

    model = _model(tokenizer)
    head = build_latent_head(model, mtp_enabled=True, n_predict=3)
    report = measure_speculative_speedup(model, head, "hello", max_tokens=8)
    assert report.baseline_tokens_per_sec > 0
    assert report.proposed > 0
    assert 0.0 <= report.acceptance_rate <= 1.0


def test_causal_grounding_reports_reduction_not_absence() -> None:
    from nyxara.growth.latent_head import measure_causal_grounding

    result = measure_causal_grounding(lambda a: "impossible" in a,
                                      answers_with=["fine", "fine", "impossible"],
                                      answers_without=["impossible"] * 3)
    assert result["relative_reduction"] > 0
    assert result["impossible_rate_with_bottleneck"] > 0.0
    assert "not their elimination" in result["note"]


# --------------------------------------------------------------------------- #
# Self-expanding MoE
# --------------------------------------------------------------------------- #
def test_weakest_domain_needs_a_real_gap() -> None:
    from nyxara.growth.expand import weakest_domain

    assert weakest_domain({"a": 1.0, "b": 1.05, "c": 1.02}) is None      # noise
    assert weakest_domain({"a": 1.0, "b": 3.0, "c": 1.02}) == "b"        # a real gap


def test_growing_adds_an_expert_and_its_router_row(tokenizer) -> None:
    """Without the router row the new expert is unreachable and silently dead."""
    from nyxara.growth.expand import SelfExpandingMoE

    model = _model(tokenizer, n_experts=4, moe_top_k=2, moe_every=1, moe_expert_hidden=64)
    expander = SelfExpandingMoE(model)
    _index, mixer = expander.moe_layers()[0]
    before_experts = len(mixer.experts)
    before_router = mixer.router.out_features

    expander._grow(mixer)
    assert len(mixer.experts) == before_experts + 1
    assert mixer.router.out_features == before_router + 1, "the router did not grow"
    assert mixer.n_experts == len(mixer.experts)

    # And the grown layer still runs, producing the same shape.
    out, aux = mixer(torch.randn(1, 6, 64))
    assert out.shape == (1, 6, 64) and float(aux.detach()) > 0


def test_restore_puts_a_rejected_expert_completely_back(tokenizer) -> None:
    from nyxara.growth.expand import SelfExpandingMoE

    model = _model(tokenizer, n_experts=4, moe_top_k=2, moe_every=1, moe_expert_hidden=64)
    expander = SelfExpandingMoE(model)
    _index, mixer = expander.moe_layers()[0]
    snapshot = expander._snapshot(mixer)
    expander._grow(mixer)
    expander._restore(mixer, snapshot)
    assert len(mixer.experts) == 4 and mixer.router.out_features == 4


def test_expansion_refuses_when_no_domain_is_behind(tokenizer) -> None:
    from nyxara.growth.expand import ExpansionConfig, SelfExpandingMoE

    model = _model(tokenizer, n_experts=2, moe_top_k=1, moe_every=1, moe_expert_hidden=32)
    expander = SelfExpandingMoE(model, config=ExpansionConfig(train_steps=1))
    report = expander.expand(dataset=None, evaluate=lambda: {"a": 1.0, "b": 1.01})
    assert not report.admitted and "far enough behind" in report.reason


def test_expansion_discards_an_expert_that_hurts_another_domain(tokenizer) -> None:
    """Capability bought by breaking something else is not a gain."""
    from nyxara.growth.expand import ExpansionConfig, SelfExpandingMoE

    model = _model(tokenizer, n_experts=2, moe_top_k=1, moe_every=1, moe_expert_hidden=32)
    expander = SelfExpandingMoE(model, config=ExpansionConfig(train_steps=1, batch_size=2))

    class _Dataset:
        block_size = 128
        domains = ["a", "b"]

        def batch(self, n, domain=None):
            return (np.zeros((n, 16), dtype=np.int64), np.zeros((n, 16), dtype=np.int64))

    scores = iter([{"a": 1.0, "b": 3.0}, {"a": 2.0, "b": 1.0}])   # b improves, a collapses
    report = expander.expand(_Dataset(), evaluate=lambda: next(scores))
    assert not report.admitted and "regressed" in report.reason
    assert expander.expert_count() == 2, "the discarded expert was left behind"


def test_expansion_on_a_dense_model_says_so(tokenizer) -> None:
    from nyxara.growth.expand import SelfExpandingMoE

    report = SelfExpandingMoE(_model(tokenizer)).expand(dataset=None)
    assert "no MoE layers" in report.reason


# --------------------------------------------------------------------------- #
# Oracle & meta-loss
# --------------------------------------------------------------------------- #
def test_oracle_keeps_only_proven_items() -> None:
    """An unproven item in the corpus teaches her to assert what nothing checked."""
    items, report = SelfPlayOracle(
        config=OracleConfig(batch=12, max_rounds=2, seconds_budget=20.0)).run()
    assert report.proposed > 0
    assert report.proven == len(items)
    assert all(item.certificate for item in items)
    assert report.proven + report.refuted + report.unprovable == report.proposed


def test_oracle_items_render_in_her_template() -> None:
    items, _report = SelfPlayOracle(
        config=OracleConfig(batch=8, max_rounds=1, seconds_budget=15.0)).run()
    if not items:
        pytest.skip("no items were certified in this budget")
    assert ASSISTANT_TAG in items[0].to_doc()


def test_oracle_saves_to_a_store(tmp_path) -> None:
    oracle = SelfPlayOracle(config=OracleConfig(batch=8, max_rounds=1, seconds_budget=15.0))
    items, _report = oracle.run()
    written = oracle.save(items, tmp_path / "proven.jsonl")
    assert written == len(items)


def test_difficulty_is_steered_toward_the_provable_band() -> None:
    oracle = SelfPlayOracle(config=OracleConfig(target_prove_rate=0.5, band=0.1))
    oracle.difficulty = 3
    oracle._steer_difficulty(0.95)      # everything provable -> ask harder questions
    assert oracle.difficulty == 4
    oracle._steer_difficulty(0.05)      # nothing provable -> pull back
    assert oracle.difficulty == 3


def test_meta_loss_raises_difficulty_on_mastery() -> None:
    meta = MetaLoss()
    meta.observe({"math": 0.9, "code": 0.85})
    assert meta.difficulty == 2
    meta.observe({"math": 0.2, "code": 0.3})
    assert meta.difficulty == 1


def test_meta_loss_reweights_toward_the_weakest_domain() -> None:
    meta = MetaLoss()
    meta.observe({"math": 0.9, "code": 0.2})
    assert meta.weights["domain:code"] > 1.0


def test_frozen_ruler_is_immune_to_mutating_its_source() -> None:
    """The first line of defence: the ruler copies, so the caller's list cannot reach it."""
    items = ["q1", "q2", "q3"]
    ruler = FrozenRuler(items)
    meta = MetaLoss(ruler=ruler)

    items.append("an easier question")
    assert len(ruler.items) == 3, "the ruler tracked a mutation of the list it was given"
    meta.assert_ruler_untouched()


def test_frozen_ruler_detects_direct_tampering() -> None:
    """'Capability rose' and 'the ruler shrank' must not be indistinguishable."""
    ruler = FrozenRuler(["q1", "q2", "q3"])
    meta = MetaLoss(ruler=ruler)
    meta.assert_ruler_untouched()

    ruler._items = ("q1", "an easier question")     # swap the battery out from under it
    with pytest.raises(RuntimeError, match="NEVER touch the ruler"):
        meta.assert_ruler_untouched()


def test_frozen_ruler_survives_difficulty_escalation() -> None:
    ruler = FrozenRuler(["a", "b"])
    meta = MetaLoss(ruler=ruler)
    for _ in range(10):
        meta.observe({"math": 0.95, "code": 0.95})
    assert meta.difficulty > 1
    meta.assert_ruler_untouched()          # the ruler did not move with it
