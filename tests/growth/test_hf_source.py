"""Tests for growth/hf_source.py and growth/progress.py — reading fast, and resuming at all.

The split filter gets the most attention here, because its failure was **silent contamination**:
asking `HuggingFaceH4/ultrachat_200k` for its `train_sft` split returned the `test_gen` files,
and evaluation data would have entered the training corpus through the very code meant to keep
it out. `_` is a word character, so a `\\btrain\\b` pattern matches neither `train_sft` nor
`test_gen`; the filter concluded "this repo has no splits" and returned everything.

Every test here is offline. Nothing in this module's logic needs the network, and a test that
reaches HuggingFace is a test that fails when a dataset is gated, renamed or slow.
"""

from __future__ import annotations

from nyxara.growth.hf_source import _filter_split, _norm
from nyxara.growth.progress import BuildProgress


# --------------------------------------------------------------------------- #
# Split filtering — the one whose failure is invisible
# --------------------------------------------------------------------------- #
_ULTRACHAT = [
    "data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
    "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
    "data/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet",
    "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
    "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
]


def test_an_underscored_split_name_selects_only_its_own_files() -> None:
    """The regression: `train_sft` must never return `test_gen`."""
    got = _filter_split(_ULTRACHAT, "train_sft")
    assert got, "the split filter returned nothing"
    assert all("train_sft" in f for f in got), f"eval data leaked into training: {got}"


def test_requesting_an_eval_split_still_returns_eval_files() -> None:
    got = _filter_split(_ULTRACHAT, "test_gen")
    assert got and all("test_gen" in f for f in got)


def test_an_unmatched_training_split_excludes_eval_files_rather_than_taking_everything() -> None:
    """Falling back to "everything" is exactly how test data gets in."""
    files = ["data/validation-0.parquet", "data/shard-0.parquet", "data/shard-1.parquet"]
    got = _filter_split(files, "train")
    assert got == ["data/shard-0.parquet", "data/shard-1.parquet"]
    assert not any("valid" in f for f in got)


def test_a_repo_with_no_split_markers_keeps_every_file() -> None:
    files = ["part_1_200000.parquet", "part_2.parquet"]
    assert _filter_split(files, "train") == files


def test_a_directory_layout_split_is_not_thrown_away() -> None:
    """FineWeb names its config in directories and its split nowhere."""
    files = ["sample/10BT/000_00000.parquet", "sample/10BT/001_00000.parquet"]
    assert len(_filter_split(files, "train")) == 2


def test_config_names_normalise_across_punctuation() -> None:
    """`sample-10BT` and `sample/10BT` name the same thing."""
    assert _norm("sample-10BT") == _norm("sample/10BT") == "sample10bt"


# --------------------------------------------------------------------------- #
# Progress — resume must survive the machine, not merely the process
# --------------------------------------------------------------------------- #
def test_progress_round_trips_through_disk(tmp_path) -> None:
    progress = BuildProgress()
    progress.mark("general:x", "file-a#0", tokens=1200)
    progress.mark("general:x", "file-a#1", tokens=900)
    progress.mark("math:y", "file-b#7", tokens=500)
    progress.save(tmp_path)

    back = BuildProgress.load(tmp_path)
    assert back.done_for("general:x") == {"file-a#0", "file-a#1"}
    assert back.tokens_from("general:x") == 2100
    assert back.tokens_from("math:y") == 500
    assert back.row_groups_done == 3


def test_a_missing_progress_file_is_a_fresh_build_not_a_crash(tmp_path) -> None:
    assert BuildProgress.load(tmp_path).row_groups_done == 0


def test_a_corrupt_progress_file_restarts_rather_than_raising(tmp_path) -> None:
    (tmp_path / "progress.json").write_text("{not json", encoding="utf-8")
    assert BuildProgress.load(tmp_path).row_groups_done == 0


def test_progress_is_written_atomically(tmp_path) -> None:
    """A truncated cursor is worse than none — it makes a resume point at a place never reached."""
    progress = BuildProgress()
    progress.mark("s", "f#0", tokens=10)
    progress.save(tmp_path)
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
    assert BuildProgress.load(tmp_path).tokens_from("s") == 10


def test_marking_the_same_row_group_twice_does_not_double_count_position() -> None:
    progress = BuildProgress()
    progress.mark("s", "f#0", tokens=100)
    progress.mark("s", "f#0", tokens=100)
    assert progress.done_for("s") == {"f#0"}


# --------------------------------------------------------------------------- #
# Dedup tables must survive with their coefficients
# --------------------------------------------------------------------------- #
def test_a_reloaded_minhash_still_recognises_what_it_stored(tmp_path) -> None:
    """The coefficients matter as much as the tables.

    They are drawn from a seeded RNG at construction, so a filter reloaded with fresh
    coefficients computes different band keys and matches nothing it already held — the tables
    present and useless, near-duplicate detection silently off across a resume.
    """
    import numpy as np

    from nyxara.growth.screen import MinHashLSH

    lsh = MinHashLSH(capacity_bits=14)
    document = " ".join(f"word{i}" for i in range(300))
    assert lsh.seen_or_add(document) is False

    back = MinHashLSH.load_from(lsh.save(tmp_path / "lsh.npz"))
    assert np.array_equal(back._mul, lsh._mul) and np.array_equal(back._add, lsh._add)
    assert back.seen_or_add(document.replace("word50", "CHANGED")) is True


def test_a_reloaded_exact_dedup_still_rejects_a_duplicate(tmp_path) -> None:
    from nyxara.growth.screen import ExactDedup, fingerprint

    dedup = ExactDedup(capacity_bits=14)
    assert dedup.add_if_new(fingerprint("hello there world")) is True
    back = ExactDedup.load_from(dedup.save(tmp_path / "exact.npy"))
    assert back.add_if_new(fingerprint("hello there world")) is False


def test_dedup_tables_are_sized_to_the_budget(tmp_path) -> None:
    """Fixed 2**25 slots is 268 MB — right at 6B tokens, absurd for a 600k-token build."""
    from nyxara.growth.corpus import CorpusBuilder

    class _Tok:
        vocab_size = 256

        def vocab_sig(self) -> str:
            return "x"

    small = CorpusBuilder(tokenizer=_Tok(), out_dir=tmp_path / "a", tokens_budget=600_000)
    large = CorpusBuilder(tokenizer=_Tok(), out_dir=tmp_path / "b", tokens_budget=6_000_000_000)
    assert small._exact.size < large._exact.size, "the table did not scale with the budget"
    assert small._exact.size * 8 < 8 * 1024 * 1024, "a small build still allocated megabytes"
