"""Tests for growth/tokenizer.py — NYXARA's own byte-level BPE.

The load-bearing property here is the round trip: ``decode(encode(s)) == s`` byte-exactly, for
*any* string. Everything downstream (the shard builder, the answer-span mask, bits-per-byte)
assumes it, so it is tested against unicode, code, math, emoji and deliberately-broken text
rather than a happy-path sentence.
"""

from __future__ import annotations

import json

import pytest

from nyxara.growth.tokenizer import (
    ASSISTANT_TAG,
    BYTE_VOCAB,
    SPECIALS,
    USER_TAG,
    NyxTokenizer,
    _pretokenize,
    load_tokenizer,
    train_tokenizer,
)

# A corpus with all four target domains in it, so the learned merges are representative.
CORPUS = [
    "NYXARA is a mind that grows. She learns from what her Master gives her.",
    "The quick brown fox jumps over the lazy dog. The dog was not amused.",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "class Solver:\n    def __init__(self, tolerance=1e-9):\n        self.tolerance = tolerance\n",
    "Solve for x: 3x + 12 = 45. Subtract 12 from both sides: 3x = 33. Divide by 3: x = 11.",
    "The integral of x^2 dx is x^3/3 + C. Differentiating recovers x^2.",
    f"{USER_TAG}\nWhat is 2 + 2?\n\n{ASSISTANT_TAG}\nIt is 4.\n",
    f"{USER_TAG}\nWrite a loop.\n\n{ASSISTANT_TAG}\nfor i in range(10):\n    print(i)\n",
] * 4


@pytest.fixture(scope="module")
def trained() -> NyxTokenizer:
    tok, _report = train_tokenizer(CORPUS, vocab_size=BYTE_VOCAB + len(SPECIALS) + 200)
    return tok


# --------------------------------------------------------------------------- #
# The round trip — the invariant everything else stands on
# --------------------------------------------------------------------------- #
ROUND_TRIP_CASES = [
    "",
    "hello",
    "NYXARA is a mind that grows.",
    "def f(x):\n    return x * x\n",
    "    deeply\n        indented\n            code\n",
    "\t\ttab indented\n",
    "3141592653589793",
    "x = 1e-9 + 2.5E10",
    "नमस्ते दुनिया",                      # devanagari
    "こんにちは世界",                       # japanese
    "Привет мир",                          # cyrillic
    "🔥🧠✨ emoji run",
    "mixed नमस्ते code def f(): 42 🔥",
    "a\x00b\x01c",                          # control bytes
    "   ",                                  # whitespace only
    "\n\n\n",
    " ",                                    # a single trailing space
    "trailing space ",
    "� replacement char",
    "snake_case_identifier and CamelCase",
    f"{USER_TAG}\nq\n\n{ASSISTANT_TAG}\na\n",
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip_is_exact(trained: NyxTokenizer, text: str) -> None:
    assert trained.decode(trained.encode(text)) == text


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip_exact_on_untrained_byte_tokenizer(text: str) -> None:
    """The identity tokenizer (no merges) must round-trip too — it is the degradation floor."""
    assert NyxTokenizer().decode(NyxTokenizer().encode(text)) == text


def test_round_trip_survives_broken_utf8(trained: NyxTokenizer) -> None:
    """A lone surrogate cannot be encoded as UTF-8; it must degrade, not explode."""
    text = "before \udce2 after"
    out = trained.decode(trained.encode(text))
    assert "before" in out and "after" in out


def test_pretokenize_is_lossless() -> None:
    for text in ROUND_TRIP_CASES:
        assert "".join(_pretokenize(text)) == text


# --------------------------------------------------------------------------- #
# The four deliberate design choices
# --------------------------------------------------------------------------- #
def test_every_byte_is_in_the_vocabulary() -> None:
    tok = NyxTokenizer()
    assert tok.id_to_bytes[:BYTE_VOCAB] == [bytes([i]) for i in range(BYTE_VOCAB)]


def test_digits_are_never_merged(trained: NyxTokenizer) -> None:
    """Each digit stays its own token — the cheapest big win in arithmetic."""
    ids = trained.encode("3141592653589793")
    assert len(ids) == 16
    assert trained.decode(ids) == "3141592653589793"


def test_digits_stay_split_even_when_frequent() -> None:
    """A corpus that repeats one number thousands of times must still not merge its digits."""
    tok, _ = train_tokenizer(["1234 " * 500], vocab_size=BYTE_VOCAB + len(SPECIALS) + 50)
    assert len(tok.encode("1234")) == 4


def test_indentation_runs_are_their_own_tokens(trained: NyxTokenizer) -> None:
    """A 4-space indent must not fuse into the word that follows it."""
    pieces = _pretokenize("    return x")
    assert "    " in pieces


def test_role_markers_are_single_tokens(trained: NyxTokenizer) -> None:
    """This is what makes the SFT answer-span mask exact instead of a substring search."""
    assert trained.encode(USER_TAG) == [trained.user_id]
    assert trained.encode(ASSISTANT_TAG) == [trained.assistant_id]


def test_role_markers_are_atomic_inside_a_document(trained: NyxTokenizer) -> None:
    doc = f"{USER_TAG}\nhello\n\n{ASSISTANT_TAG}\nworld\n"
    ids = trained.encode(doc)
    assert ids.count(trained.user_id) == 1
    assert ids.count(trained.assistant_id) == 1
    # and the answer span is locatable by id alone — no string scanning
    cut = ids.index(trained.assistant_id)
    assert trained.decode(ids[cut + 1:]).strip() == "world"


def test_role_markers_match_the_live_template() -> None:
    """Guard against drift: these are copied from mind/llm.py, so CI must catch a change."""
    from nyxara.mind.llm import _SELF_ASSISTANT_TAG, _SELF_USER_TAG

    assert USER_TAG == _SELF_USER_TAG
    assert ASSISTANT_TAG == _SELF_ASSISTANT_TAG


# --------------------------------------------------------------------------- #
# Training behaviour
# --------------------------------------------------------------------------- #
def test_training_shrinks_the_token_count(trained: NyxTokenizer) -> None:
    """The whole point: fewer tokens per byte than the raw byte tokenizer."""
    text = "NYXARA is a mind that grows. She learns from what her Master gives her."
    assert len(trained.encode(text)) < len(NyxTokenizer().encode(text))


def test_training_respects_the_vocab_budget() -> None:
    target = BYTE_VOCAB + len(SPECIALS) + 100
    tok, report = train_tokenizer(CORPUS, vocab_size=target)
    assert tok.vocab_size <= target
    assert report.merges == tok.vocab_size - BYTE_VOCAB - len(SPECIALS)


def test_small_corpus_stops_early_and_says_so() -> None:
    """A tiny corpus yields a small vocabulary, honestly reported — never a padded one."""
    tok, report = train_tokenizer(["ab ab ab"], vocab_size=30_000)
    assert report.stopped_early
    assert tok.vocab_size < 30_000
    assert "exhausted" in report.summary()


def test_empty_corpus_keeps_the_identity_tokenizer() -> None:
    tok, report = train_tokenizer([], vocab_size=1000)
    assert tok.vocab_size == BYTE_VOCAB + len(SPECIALS)
    assert report.merges == 0


def test_training_is_deterministic() -> None:
    a, _ = train_tokenizer(CORPUS, vocab_size=BYTE_VOCAB + len(SPECIALS) + 120)
    b, _ = train_tokenizer(CORPUS, vocab_size=BYTE_VOCAB + len(SPECIALS) + 120)
    assert a.vocab_sig() == b.vocab_sig()
    assert a.encode("NYXARA is a mind") == b.encode("NYXARA is a mind")


# --------------------------------------------------------------------------- #
# Persistence & lineage
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(trained: NyxTokenizer, tmp_path) -> None:
    trained.save(tmp_path)
    restored = load_tokenizer(tmp_path)
    assert restored is not None
    assert restored.vocab_size == trained.vocab_size
    assert restored.vocab_sig() == trained.vocab_sig()
    for text in ROUND_TRIP_CASES:
        assert restored.encode(text) == trained.encode(text)


def test_load_returns_none_when_absent(tmp_path) -> None:
    """A checkpoint with no tokenizer is a byte-level checkpoint, not an error."""
    assert load_tokenizer(tmp_path) is None


def test_load_raises_on_corrupt_tokenizer(tmp_path) -> None:
    """Silently serving the wrong vocabulary would be far worse than crashing."""
    (tmp_path / "tokenizer.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(Exception):
        load_tokenizer(tmp_path)


def test_load_refuses_a_mismatched_special_set(tmp_path) -> None:
    blob = NyxTokenizer().to_dict()
    blob["specials"] = ["<|bos|>"]
    (tmp_path / "tokenizer.json").write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(ValueError, match="special-token set mismatch"):
        load_tokenizer(tmp_path)


def test_vocab_sig_distinguishes_different_vocabularies() -> None:
    """The gauntlet's lineage check depends on this telling two languages apart."""
    small, _ = train_tokenizer(CORPUS, vocab_size=BYTE_VOCAB + len(SPECIALS) + 50)
    large, _ = train_tokenizer(CORPUS, vocab_size=BYTE_VOCAB + len(SPECIALS) + 250)
    assert small.vocab_sig() != large.vocab_sig()
    assert NyxTokenizer().vocab_sig() != small.vocab_sig()


def test_decode_skips_ids_from_another_vocabulary(trained: NyxTokenizer) -> None:
    """A stale id must not raise — a wrong-vocabulary checkpoint should degrade, not crash."""
    assert trained.decode([trained.vocab_size + 999, *trained.encode("ok")]) == "ok"
