"""Tests for growth/corpus.py — the sharded, screened, domain-mixed pretraining corpus.

The properties worth guarding here are the ones whose failure is *silent*:

* a screen designed for scraped text quietly deleting her own verified data;
* domains being trained in sequence rather than interleaved (which causes forgetting);
* an absent domain being reported as success;
* eval contamination, which makes every downstream number look better than the model is.
"""

from __future__ import annotations

import pytest

from nyxara.growth.corpus import (
    DEFAULT_DOMAIN_WEIGHTS,
    DOMAINS,
    ContaminationFilter,
    QualityFilter,
    ShardDataset,
    ShardIndex,
    ShardWriter,
    build_corpus,
)
from nyxara.growth.synth_data import generate_domain_docs
from nyxara.growth.tokenizer import train_tokenizer

np = pytest.importorskip("numpy", reason="the memory-mapped reader requires numpy")


@pytest.fixture(scope="module")
def tokenizer():
    docs = []
    for domain in ("math", "code", "conversation"):
        docs += list(generate_domain_docs(domain, 80, seed=3))
    tok, _report = train_tokenizer(docs, vocab_size=2000)
    return tok


@pytest.fixture
def built(tmp_path, tokenizer):
    index, report = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "shards",
                                 tokens_budget=300_000, seed=3)
    return tmp_path / "shards", index, report


# --------------------------------------------------------------------------- #
# The quality screen must not eat her own verified data
# --------------------------------------------------------------------------- #
def test_quality_screen_rejects_scraped_junk() -> None:
    q = QualityFilter()
    assert not q.check("short")[0]
    assert not q.check("buy now! " * 60)[0]
    assert not q.check("=" * 500)[0]


def test_quality_screen_keeps_real_prose() -> None:
    q = QualityFilter()
    prose = ("The foundry trains a candidate model on a held-out split and promotes it only "
             "when the gauntlet agrees. Every promotion is reversible, and the character core "
             "is frozen at infinite importance so no capability gain can move it. " * 2)
    assert q.check(prose)[0]


def test_quality_screen_does_not_punish_code() -> None:
    """Source code looks like junk to a prose filter; punishing it guts the code domain."""
    dense = ("x = {'a': [1, 2], 'b': (3, 4)}; y = [(i, j) for i, j in x['b']]\n"
             "z = {**x, 'c': y[0][1] << 2 | 0xFF} if y else {}\n") * 5
    assert not QualityFilter().check(dense, domain="general")[0], (
        "this sample is symbol-dense enough that the prose screen should reject it")
    assert QualityFilter().check(dense, domain="code")[0], (
        "the code domain must be exempt from the prose symbol-ratio rule")


def test_trusted_sources_skip_the_quality_screen(built) -> None:
    """The regression: a 120-char verified code example must NOT be dropped as 'too_short'.

    Running the scraped-text screen over her own verified synthetic data deletes the entire
    code domain while the build still reports success — the exact kind of silent failure this
    module's screens are supposed to prevent.
    """
    _shard_dir, _index, report = built
    assert report.tokens.get("code", 0) > 0, (
        "the code domain is empty — a screen meant for scraped text has eaten verified data")


# --------------------------------------------------------------------------- #
# Shards: write, index, memory-map
# --------------------------------------------------------------------------- #
def test_build_produces_shards_and_an_index(built) -> None:
    shard_dir, index, _report = built
    assert (shard_dir / "index.json").exists()
    assert index.tokens() > 0
    for shard in index.shards:
        assert (shard_dir / shard["path"]).exists()


def test_index_round_trips(built) -> None:
    shard_dir, index, _report = built
    assert ShardIndex.load(shard_dir).to_dict() == index.to_dict()


def test_dataset_yields_correctly_shaped_shifted_windows(built) -> None:
    shard_dir, _index, _report = built
    ds = ShardDataset(shard_dir, block_size=32, seed=0)
    x, y = ds.sample()
    assert x.shape == (32,) and y.shape == (32,)
    assert np.array_equal(x[1:], y[:-1])       # next-token shift


def test_dataset_batches_stack(built) -> None:
    shard_dir, _index, _report = built
    x, y = ShardDataset(shard_dir, block_size=16, seed=0).batch(5)
    assert x.shape == (5, 16) and y.shape == (5, 16)


def test_ids_stay_inside_the_vocabulary(built, tokenizer) -> None:
    shard_dir, _index, _report = built
    x, _y = ShardDataset(shard_dir, block_size=32, seed=0).batch(16)
    assert int(x.max()) < tokenizer.vocab_size


def test_writer_refuses_ids_beyond_the_uint16_range(tmp_path) -> None:
    """A wider vocabulary must fail loudly, not wrap around into different tokens."""
    writer = ShardWriter(tmp_path, shard_tokens=8)
    with pytest.raises(ValueError, match="uint16"):
        writer.add("general", [70_000] * 8)


# --------------------------------------------------------------------------- #
# Domain mixing — per batch, not per phase
# --------------------------------------------------------------------------- #
def test_batches_interleave_domains(built) -> None:
    """Training general-then-code-then-math is how you get catastrophic forgetting."""
    shard_dir, _index, _report = built
    ds = ShardDataset(shard_dir, block_size=16, seed=0)
    seen = {ds._pick_domain() for _ in range(400)}
    assert len(seen) == len(ds.domains) > 1


def test_weights_follow_the_configured_mix(built) -> None:
    shard_dir, _index, _report = built
    ds = ShardDataset(shard_dir, block_size=16, seed=0)
    counts = {d: 0 for d in ds.domains}
    for _ in range(6000):
        counts[ds._pick_domain()] += 1
    for domain, weight in ds.weights.items():
        assert abs(counts[domain] / 6000 - weight) < 0.05


def test_absent_domain_weight_is_redistributed_not_sampled(built) -> None:
    """A weight for a domain with no shards must not become an empty draw."""
    shard_dir, _index, _report = built
    ds = ShardDataset(shard_dir, block_size=16, seed=0)
    assert abs(sum(ds.weights.values()) - 1.0) < 1e-9
    assert set(ds.weights) <= set(ds.domains)


def test_absent_domain_is_reported_not_hidden(built) -> None:
    """`general` has no offline source, and the report must say so rather than look complete."""
    _shard_dir, _index, report = built
    assert any("ABSENT" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
def test_build_resumes_instead_of_restarting(tmp_path, tokenizer) -> None:
    out = tmp_path / "shards"
    _index, first = build_corpus(tokenizer=tokenizer, out_dir=out, tokens_budget=40_000, seed=3)
    index2, second = build_corpus(tokenizer=tokenizer, out_dir=out, tokens_budget=80_000, seed=3)
    assert any("resuming" in note for note in second.notes)
    assert index2.tokens() >= first.total_tokens


def test_dataset_cursor_round_trips(built) -> None:
    """A resumed run must continue through the data, not silently restart the epoch."""
    shard_dir, _index, _report = built
    a = ShardDataset(shard_dir, block_size=16, seed=0)
    for _ in range(10):
        a.sample()
    state = a.state_dict()
    b = ShardDataset(shard_dir, block_size=16, seed=0)
    b.load_state_dict(state)
    assert np.array_equal(a.sample()[0], b.sample()[0])


def test_dataset_needs_a_built_corpus(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path, block_size=16)


# --------------------------------------------------------------------------- #
# Contamination — the screen whose absence fails silently and upward
# --------------------------------------------------------------------------- #
def test_contamination_filter_indexes_the_shipped_eval_sets() -> None:
    f = ContaminationFilter().load_eval_sets()
    assert f.loaded, "the eval sets could not be indexed — contamination would go unchecked"


def test_contamination_filter_catches_a_verbatim_eval_prompt() -> None:
    from nyxara.eval.datasets import build_realworld_benchmark

    f = ContaminationFilter().load_eval_sets()
    tasks = [t for t in build_realworld_benchmark().tasks() if len(t.prompt.split()) >= 15]
    if not tasks:
        pytest.skip("no eval prompt long enough for a 13-gram match")
    assert f.is_contaminated(tasks[0].prompt)


def test_contamination_filter_passes_unrelated_text() -> None:
    f = ContaminationFilter().load_eval_sets()
    assert not f.is_contaminated(
        "The shard writer appends token ids to per-domain binary files and records each one "
        "in an index so that a build which dies partway through keeps everything it finished.")


def test_empty_contamination_filter_is_a_no_op() -> None:
    assert not ContaminationFilter().is_contaminated("anything at all")


# --------------------------------------------------------------------------- #
# Report honesty
# --------------------------------------------------------------------------- #
def test_report_notes_when_streaming_is_unavailable(built) -> None:
    _shard_dir, _index, report = built
    try:
        import datasets  # noqa: F401
    except Exception:
        assert any("datasets" in note for note in report.notes)


def test_default_weights_cover_every_domain_and_sum_to_one() -> None:
    assert set(DEFAULT_DOMAIN_WEIGHTS) == set(DOMAINS)
    assert abs(sum(DEFAULT_DOMAIN_WEIGHTS.values()) - 1.0) < 1e-9
