"""Tests for eval/domains.py — the metric that survives a change of tokenizer.

``bits_per_byte`` exists because per-token perplexity is not comparable across vocabularies, and
the promotion gate depends on that comparison. The tests that matter here are the ones that
demonstrate the failure it fixes: two models that tokenize differently, ranked correctly by
bits-per-byte and *backwards* by perplexity.
"""

from __future__ import annotations

import math


from nyxara.eval.domains import (
    DOMAINS,
    LN2,
    DomainScores,
    bits_per_byte,
    corpus_bits_per_byte,
    evaluate_domains,
)

TEXT = ("NYXARA trains a candidate model on a held-out split and promotes it only when the "
        "gauntlet agrees. Every promotion is reversible.")


class _FakeModel:
    """A model with a known cross-entropy, so the arithmetic can be checked exactly."""

    def __init__(self, nats_per_token: float, tokens_per_text: int) -> None:
        self.nats_per_token = nats_per_token
        self.tokens_per_text = tokens_per_text

    def cross_entropy_nats(self, text: str):
        return self.nats_per_token * self.tokens_per_text, self.tokens_per_token(text)

    def tokens_per_token(self, _text: str) -> int:
        return self.tokens_per_text

    def perplexity(self, _text: str) -> float:
        return math.exp(self.nats_per_token)


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
def test_bits_per_byte_is_nats_over_bytes_in_base_two() -> None:
    model = _FakeModel(nats_per_token=1.0, tokens_per_text=10)
    n_bytes = len(TEXT.encode("utf-8"))
    assert math.isclose(bits_per_byte(model, TEXT), 10.0 / (n_bytes * LN2), rel_tol=1e-9)


def test_empty_text_is_infinite_not_an_error() -> None:
    assert bits_per_byte(_FakeModel(1.0, 5), "") == float("inf")


def test_a_model_that_cannot_score_is_infinitely_bad_not_a_crash() -> None:
    class _Broken:
        def cross_entropy_nats(self, _text):
            raise RuntimeError("no")

    assert bits_per_byte(_Broken(), TEXT) == float("inf")


def test_corpus_score_is_weighted_by_bytes_not_by_document() -> None:
    """Averaging per-document lets a hundred one-liners outvote a long document."""
    model = _FakeModel(nats_per_token=1.0, tokens_per_text=10)
    long_doc = TEXT * 10
    weighted = corpus_bits_per_byte(model, [TEXT, long_doc])
    naive = (bits_per_byte(model, TEXT) + bits_per_byte(model, long_doc)) / 2
    assert not math.isclose(weighted, naive, rel_tol=1e-6)


def test_corpus_score_skips_unscorable_documents() -> None:
    model = _FakeModel(nats_per_token=1.0, tokens_per_text=10)
    assert math.isfinite(corpus_bits_per_byte(model, ["", TEXT, ""]))


def test_empty_corpus_is_infinite() -> None:
    assert corpus_bits_per_byte(_FakeModel(1.0, 5), []) == float("inf")


# --------------------------------------------------------------------------- #
# The failure it exists to fix
# --------------------------------------------------------------------------- #
def test_bits_per_byte_ranks_correctly_where_perplexity_ranks_backwards() -> None:
    """The exact scenario the promotion gate would get wrong.

    A byte-level model splits the text into many easy tokens and reports a LOW per-token
    perplexity. A sub-word model splits it into few hard tokens and reports a HIGH one — while
    compressing the text better, which is what "a better language model" means. Perplexity
    prefers the worse model; bits-per-byte does not.
    """
    n_bytes = len(TEXT.encode("utf-8"))
    # Byte-level: one token per byte, 2.0 nats each -> 2.0 nats/byte.
    byte_model = _FakeModel(nats_per_token=2.0, tokens_per_text=n_bytes)
    # Sub-word: a quarter as many tokens, 4.0 nats each -> 1.0 nats/byte. Genuinely better.
    bpe_model = _FakeModel(nats_per_token=4.0, tokens_per_text=n_bytes // 4)

    assert byte_model.perplexity(TEXT) < bpe_model.perplexity(TEXT), (
        "setup: per-token perplexity must favour the byte model here")
    assert bits_per_byte(bpe_model, TEXT) < bits_per_byte(byte_model, TEXT), (
        "bits-per-byte must favour the model that actually compresses better")


def test_default_cross_entropy_derives_from_perplexity() -> None:
    """Every backend inherits a usable implementation, even without overriding it."""
    from nyxara.growth.foundry_models import WordKNGramLM

    model = WordKNGramLM(order=2)
    model.train_on(["the quick brown fox jumps over the lazy dog"] * 5)
    total, n = model.cross_entropy_nats("the quick brown fox")
    assert n > 0 and math.isfinite(total)
    assert math.isfinite(bits_per_byte(model, "the quick brown fox"))


def test_word_backend_counts_words_not_bytes() -> None:
    """Its perplexity is per word, so the total must be un-normalised by words."""
    from nyxara.growth.foundry_models import WordKNGramLM

    model = WordKNGramLM(order=2)
    text = "one two three four five"
    assert model.token_count(text) < len(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Domain scores
# --------------------------------------------------------------------------- #
def test_weakest_ignores_the_differently_scaled_general_score() -> None:
    """`general` is in bits and the rest are accuracies; comparing them is meaningless."""
    scores = DomainScores(general_bpb=9.9, code=0.8, math=0.2, conversation=0.6)
    assert scores.weakest == "math"


def test_summary_reports_all_four_domains() -> None:
    text = DomainScores(general_bpb=1.2, code=0.5, math=0.4, conversation=0.6).summary()
    for domain in DOMAINS:
        assert domain in text


def test_scores_round_trip() -> None:
    scores = DomainScores(general_bpb=1.25, code=0.5, math=0.4, conversation=0.6)
    assert scores.to_dict()["general_bpb"] == 1.25


def test_infinite_bpb_serialises_as_null() -> None:
    """JSON has no infinity; a metric that cannot round-trip is a metric nobody can store."""
    import json

    assert json.loads(json.dumps(DomainScores().to_dict()))["general_bpb"] is None


def test_evaluate_domains_notes_a_missing_holdout() -> None:
    scores = evaluate_domains(_FakeModel(1.0, 10), skip_generation=True)
    assert any("no holdout" in note for note in scores.notes)


def test_evaluate_domains_scores_general_from_a_holdout() -> None:
    scores = evaluate_domains(_FakeModel(1.0, 10), holdout=[TEXT, TEXT],
                              skip_generation=True)
    assert math.isfinite(scores.general_bpb)
    assert scores.counts["general"] == 2


def test_a_failing_battery_is_a_note_and_a_zero_not_a_crash() -> None:
    """This runs inside the forge; a metric that raises takes the whole thing down."""
    class _Exploding:
        def cross_entropy_nats(self, _text):
            return 1.0, 1

        def generate(self, *_a, **_kw):
            raise RuntimeError("generation is broken")

    scores = evaluate_domains(_Exploding(), holdout=[TEXT])
    assert scores.code == 0.0 and scores.math == 0.0
