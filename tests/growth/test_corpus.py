"""Tests for growth/corpus.py — the sharded, screened, domain-mixed pretraining corpus.

The properties worth guarding here are the ones whose failure is *silent*:

* a screen designed for scraped text quietly deleting her own verified data;
* domains being trained in sequence rather than interleaved (which causes forgetting);
* an absent domain being reported as success;
* eval contamination, which makes every downstream number look better than the model is.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

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
    """A corpus built with NO network access.

    `stream_sources={}` is not optional here. Once the `datasets` package is installed the
    default source list reaches HuggingFace, and the suite would depend on the network being up,
    on several datasets staying un-gated, and on minutes of download per test. Hermetic by
    construction beats hermetic by "the optional dependency happens to be missing".
    """
    index, report = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "shards",
                                 tokens_budget=300_000, seed=3, stream_sources={})
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
    """`general` has no offline source — it is streamed or it is absent, never fabricated.

    A synthetic general corpus would teach fluent nonsense, so `DEFAULT_SYNTHETIC_SHARE` sets it
    to zero on purpose. The report has to SAY the domain is missing rather than look complete.
    """
    _shard_dir, _index, report = built
    assert any("ABSENT" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #
def test_build_resumes_instead_of_restarting(tmp_path, tokenizer) -> None:
    out = tmp_path / "shards"
    _index, first = build_corpus(tokenizer=tokenizer, out_dir=out, tokens_budget=40_000, seed=3,
                                 stream_sources={})
    index2, second = build_corpus(tokenizer=tokenizer, out_dir=out, tokens_budget=80_000, seed=3,
                                  stream_sources={})
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


def test_an_empty_source_dict_really_disables_streaming(tmp_path, tokenizer) -> None:
    """`{}` means "no sources", not "use the defaults".

    `dict(stream_sources or DEFAULT)` treats an empty dict as falsy and silently restored the
    default source list — so `--no-stream` reached HuggingFace anyway and this suite depended on
    the network being up. Measured, the same build went from 18.6s (streaming) to 0.9s once the
    check became `is not None`.
    """
    from nyxara.growth.corpus import CorpusBuilder

    builder = CorpusBuilder(tokenizer=tokenizer, out_dir=tmp_path, stream_sources={})
    assert builder.stream_sources == {}, "an explicitly empty source list was overridden"

    default = CorpusBuilder(tokenizer=tokenizer, out_dir=tmp_path)
    assert default.stream_sources, "omitting the argument must still give the defaults"


def test_an_empty_weight_dict_is_not_silently_replaced(tmp_path, tokenizer) -> None:
    from nyxara.growth.corpus import CorpusBuilder

    builder = CorpusBuilder(tokenizer=tokenizer, out_dir=tmp_path, weights={},
                            synthetic_share={})
    assert builder.weights == {} and builder.synthetic_share == {}


def test_the_synthetic_share_scales_with_the_budget(tmp_path, tokenizer) -> None:
    """Synthetic data must be a designed fraction of the budget, not a fixed document count.

    The fixed count produced ~1.8M tokens against a 6B budget — 0.03% of the corpus — so the one
    source that is correct by construction was rounding error.
    """
    small, _ = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "a",
                            tokens_budget=40_000, seed=1, stream_sources={})
    large, _ = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "b",
                            tokens_budget=400_000, seed=1, stream_sources={})
    assert large.tokens("math") > small.tokens("math") * 3, (
        "a 10x budget produced barely more synthetic maths — the share is not scaling")


def test_the_validation_split_is_disjoint_and_content_keyed(tmp_path, tokenizer) -> None:
    """Two builds of the same corpus must put each document on the same side of the split."""
    first, _ = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "a",
                            tokens_budget=120_000, seed=7, stream_sources={})
    second, _ = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "b",
                             tokens_budget=120_000, seed=7, stream_sources={})
    assert first.tokens(split="val") > 0, "no validation split was produced"
    assert first.tokens(split="val") == second.tokens(split="val")
    assert first.tokens(split="train") == second.tokens(split="train")


def test_val_shards_are_not_served_to_the_training_reader(tmp_path, tokenizer) -> None:
    index, _ = build_corpus(tokenizer=tokenizer, out_dir=tmp_path / "s",
                            tokens_budget=120_000, seed=7, stream_sources={})
    train = ShardDataset(tmp_path / "s", block_size=16, seed=0)
    assert train.total_tokens() == index.tokens(split="train")
    assert train.total_tokens() < index.tokens(), "the reader is serving validation data"


def test_default_weights_cover_every_domain_and_sum_to_one() -> None:
    assert set(DEFAULT_DOMAIN_WEIGHTS) == set(DOMAINS)
    assert abs(sum(DEFAULT_DOMAIN_WEIGHTS.values()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Resume — the bug that made a crash at 80% cost 100% AND corrupt the rest
# --------------------------------------------------------------------------- #
def test_index_is_on_disk_after_every_shard_not_only_at_close(tmp_path) -> None:
    """A finished shard must be a finished file the index already knows about.

    `close()` was the only writer of index.json, so a build killed partway left orphaned .bin
    files and a stale index — and the next run restarted the shard counter and overwrote them.
    """
    writer = ShardWriter(tmp_path, shard_tokens=1024)
    writer.add("general", list(range(2048)))        # forces two full shards, no close()
    index = ShardIndex.load(tmp_path)
    assert index is not None, "index.json was not written until close()"
    assert index.tokens("general") == 2048
    for shard in index.shards:
        assert (tmp_path / shard["path"]).exists()


def test_a_crashed_build_does_not_overwrite_its_own_shards(tmp_path) -> None:
    """The regression: orphan reconciliation must not renumber onto a surviving shard."""
    first = ShardWriter(tmp_path, shard_tokens=1024)
    first.add("general", [7] * 2048)                 # two shards on disk + in the index
    del first                                        # simulate a kill: no close()

    second = ShardWriter(tmp_path, shard_tokens=1024)
    assert second.existing_tokens("general") == 2048, "resume lost the finished shards"
    second.add("general", [9] * 1024)
    second.close()

    index = ShardIndex.load(tmp_path)
    assert index.tokens("general") == 3072, "a new shard overwrote an existing one"
    names = [s["path"] for s in index.shards]
    assert len(names) == len(set(names)), f"duplicate shard names: {names}"


def test_orphaned_shards_with_no_index_entry_are_removed(tmp_path) -> None:
    """A .bin written but never indexed is unreferenced data — it must not linger and collide."""
    writer = ShardWriter(tmp_path, shard_tokens=1024)
    writer.add("general", list(range(1024)))
    writer.close()
    (tmp_path / "general-09999.bin").write_bytes(b"\x00\x01" * 8)   # orphan from a crash

    reopened = ShardWriter(tmp_path, shard_tokens=1024)
    assert reopened.orphans_removed == 1
    assert not (tmp_path / "general-09999.bin").exists()
    assert reopened.existing_tokens("general") == 1024


def test_writer_refuses_negative_ids(tmp_path) -> None:
    with pytest.raises(ValueError):
        ShardWriter(tmp_path, shard_tokens=8).add("general", [-1] * 8)


# --------------------------------------------------------------------------- #
# Contamination hashing must survive a process boundary
# --------------------------------------------------------------------------- #
def test_contamination_hash_is_stable_across_processes() -> None:
    """`hash(str)` is salted per process, so an index built in the parent means nothing in a
    spawned worker — which silently disables the screen in the multiprocessing build."""
    import subprocess
    import sys

    code = ("from nyxara.growth.corpus import ContaminationFilter;"
            "print(ContaminationFilter._hash_gram('the quick brown fox jumps'))")
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env=env, cwd=str(REPO_ROOT)).stdout.strip())
    assert len(outs) == 1, f"hash differs across PYTHONHASHSEED: {outs}"


def test_contamination_index_round_trips_through_disk(tmp_path) -> None:
    f = ContaminationFilter().load_eval_sets()
    path = f.save(tmp_path / "decontam.bin")
    back = ContaminationFilter.load(path)
    assert back.loaded and back._hashes == f._hashes


def test_strided_index_still_catches_a_long_overlap() -> None:
    """Striding the INDEX is safe only because the QUERY still walks every n-gram."""
    text = " ".join(f"w{i}" for i in range(60))
    dense, strided = ContaminationFilter(n=13), ContaminationFilter(n=13, stride=4)
    for f in (dense, strided):
        f._hashes.update(f._ngrams(text))
        f.loaded = True
    probe = "zzz " + " ".join(f"w{i}" for i in range(20, 40)) + " qqq"
    assert dense.is_contaminated(probe)
    assert strided.is_contaminated(probe), "a strided index missed a 20-word overlap"


# --------------------------------------------------------------------------- #
# Multi-turn conversations must survive the door
# --------------------------------------------------------------------------- #
def test_every_turn_of_a_transcript_is_rendered() -> None:
    """The conversation domain's whole reason for existing is the multi-turn structure."""
    from nyxara.growth.corpus import _row_text

    msgs = [{"role": "user", "content": f"question number {i} about apples"} for i in range(4)]
    turns = []
    for i in range(4):
        turns.append({"role": "user", "content": f"question number {i} about apples"})
        turns.append({"role": "assistant", "content": f"answer number {i} concerning pears"})

    text = _row_text({"messages": turns}, "messages")
    for i in range(4):
        assert f"question number {i}" in text, f"user turn {i} was dropped"
        assert f"answer number {i}" in text, f"assistant turn {i} was dropped"
    assert len(msgs) == 4      # guard against the fixture being edited into meaninglessness


def test_a_dangling_user_turn_is_not_paired_with_the_wrong_answer() -> None:
    from nyxara.growth.corpus import _row_text

    turns = [{"role": "user", "content": "first thing I asked about"},
             {"role": "user", "content": "second thing I asked about"},
             {"role": "assistant", "content": "the single reply given"}]
    text = _row_text({"messages": turns}, "messages")
    assert text.count("the single reply given") == 1


# --------------------------------------------------------------------------- #
# Shard sampling must follow token mass, not shard count
# --------------------------------------------------------------------------- #
def test_shards_are_sampled_in_proportion_to_length(tmp_path) -> None:
    """A short tail shard beside long ones must not be oversampled into prominence."""
    writer = ShardWriter(tmp_path, shard_tokens=4096)
    writer.add("general", [1] * 4096)          # one full shard of 1s
    writer.add("general", [2] * 4096)          # a second full shard of 1s
    writer.close(vocab_size=8)
    # A deliberately tiny tail shard of 2s.
    ShardWriter(tmp_path, shard_tokens=4096).add("general", [2] * 64)
    ShardWriter(tmp_path, shard_tokens=4096).close(vocab_size=8)

    ds = ShardDataset(tmp_path, block_size=8, seed=0)
    picks = [ds._pick_shard("general") for _ in range(4000)]
    tail = sum(1 for p in picks if p == 2) / len(picks)
    expected = 64 / (4096 + 4096 + 64)
    assert tail < 0.05, (
        f"the 64-token tail shard took {tail:.1%} of draws (mass share is {expected:.2%}) — "
        "shards are being chosen uniformly rather than by length")
