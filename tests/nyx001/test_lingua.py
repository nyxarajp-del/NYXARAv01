"""Track B — language last. The tests here are mostly about what these modules REFUSE to do."""

from __future__ import annotations

import numpy as np
import pytest

from nyxara.nyx001.layers.l00_substrate import Substrate
from nyxara.nyx001.lingua import DynamicEmbedding, Grounding, SequenceModel, Tokenizer


@pytest.fixture
def sub():
    return Substrate(seed=17, rung="toy")


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_starts_with_no_vocabulary_at_all():
    """No corpus, no inherited merges — a byte tokenizer until experience says otherwise."""
    t = Tokenizer()
    assert t.vocab_size == 256, "started with a vocabulary it did not learn"
    assert t.learned == 0
    assert t.compression("hello world") == pytest.approx(1.0), "compressed before learning"


def test_tokenizer_learns_merges_from_what_it_is_shown():
    t = Tokenizer(max_vocab=512, merge_at=3)
    for _ in range(20):
        t.encode("alpha beta gamma")
    assert t.learned > 0, "saw the same text twenty times and learned nothing"
    assert t.compression("alpha beta gamma") > 1.0


def test_tokenizer_roundtrip_is_lossless():
    t = Tokenizer(max_vocab=512, merge_at=2)
    for _ in range(30):
        t.encode("the quick brown fox jumps")
    for text in ("the quick brown fox jumps", "unseen text entirely", "", "unicode: ñ é 中"):
        assert t.decode(t.encode(text, learn=False)) == text, f"lossy on {text!r}"


def test_tokenizer_vocabulary_is_bounded_and_evicts():
    t = Tokenizer(max_vocab=270, merge_at=2)
    for i in range(200):
        t.encode(f"phrase number {i} with padding text")
    assert t.vocab_size <= 270
    assert t.evicted > 0, "a bounded vocabulary that never reports eviction claims infinity"


def test_two_tokenizers_fed_different_experience_diverge():
    """Correct behaviour for a system forbidden from inheriting a vocabulary."""
    a, b = Tokenizer(merge_at=2), Tokenizer(merge_at=2)
    for _ in range(20):
        a.encode("alpha alpha alpha")
        b.encode("zulu zulu zulu")
    assert a._merges != b._merges


# --------------------------------------------------------------------------- #
# Dynamic embedding
# --------------------------------------------------------------------------- #
def test_embedding_actually_moves_with_context(sub):
    """~0 drift would mean this is a lookup table wearing five weight matrices."""
    emb = DynamicEmbedding(sub, vocab=256)
    contexts = [np.random.default_rng(i).normal(0, 1, sub.width) for i in range(4)]
    drift = emb.drift(7, contexts)
    assert drift is not None and drift > 0.01, f"embeddings barely move: {drift}"


def test_embedding_is_identical_without_any_signals(sub):
    emb = DynamicEmbedding(sub, vocab=256)
    a = emb.embed(5)
    b = emb.embed(5)
    assert np.allclose(a.data, b.data), "not deterministic with no situation supplied"


def test_uncertainty_reduces_the_tokens_own_contribution(sub):
    """Under high uncertainty, what a token meant last time is the least trustworthy thing held."""
    emb = DynamicEmbedding(sub, vocab=256)
    ctx = np.random.default_rng(0).normal(0, 1, sub.width)
    sure = emb.embed(9, context=ctx, uncertainty=0.0)
    unsure = emb.embed(9, context=ctx, uncertainty=1.0)
    assert not np.allclose(sure.data, unsure.data), "uncertainty changed nothing"


def test_embedding_reports_that_its_table_is_untrained(sub):
    """Honesty: with no language objective running, the table has not left its init — and says so."""
    emb = DynamicEmbedding(sub, vocab=256)
    assert emb.table_moved() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Sequence model
# --------------------------------------------------------------------------- #
def test_sequence_model_is_not_attention(sub):
    seq = SequenceModel(sub, depth=2)
    assert "state-space" in seq.stats()["mechanism"]
    assert seq.decay_range() is not None


def test_sequence_gates_stay_inside_zero_one(sub):
    seq = SequenceModel(sub, depth=2)
    d = seq.decay_range()
    assert 0.0 < d["min"] and d["max"] < 1.0


def test_sequence_carries_state_across_calls(sub):
    seq = SequenceModel(sub, depth=2)
    emb = DynamicEmbedding(sub, vocab=256)
    z = emb.embed_sequence([1, 2, 3], context=np.ones(sub.width))
    assert seq.state_norm() is None, "held state before anything ran"
    seq.forward(z)
    assert seq.state_norm() is not None, "a recurrent model that carries nothing is not recurrent"
    seq.reset_state()
    assert seq.state_norm() is None


def test_sequence_output_keeps_its_shape(sub):
    seq = SequenceModel(sub, depth=2)
    emb = DynamicEmbedding(sub, vocab=256)
    z = emb.embed_sequence([1, 2, 3, 4], context=np.ones(sub.width))
    out = seq.forward(z)
    assert out is not None and out.data.shape == (1, 4, sub.width)


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #
class _Concept:
    def __init__(self, name):
        self.name = name


def test_grounding_refuses_a_meaning_it_has_not_earned(sub):
    """A layer that always answers always looks like it understands."""
    g = Grounding(sub, min_support=3.0)
    assert g.meaning("never_seen") is None
    g.ground("once", _Concept("c0"))
    assert g.meaning("once") is None, "claimed a meaning from a single co-occurrence"


def test_grounding_earns_a_meaning_through_repetition(sub):
    g = Grounding(sub, min_support=3.0, min_exclusivity=0.5)
    for _ in range(6):
        g.ground("alpha", _Concept("c0"))
    assert g.meaning("alpha") == "c0"
    assert g.is_grounded("alpha")


def test_a_word_spread_across_concepts_is_grounded_in_none(sub):
    """Exclusivity is what stops a promiscuous word counting as understood."""
    g = Grounding(sub, min_support=2.0, min_exclusivity=0.6)
    for i in range(6):
        g.ground("vague", _Concept(f"c{i}"))
    assert g.meaning("vague") is None, "a word bound to six concepts was called grounded"


def test_describe_runs_the_ordering_backwards(sub):
    g = Grounding(sub, min_support=2.0)
    for _ in range(5):
        g.ground_text("alpha stream", _Concept("c0"))
    words = [w for w, _ in g.describe(_Concept("c0"))]
    assert "alpha" in words and "stream" in words


def test_grounding_coverage_is_none_before_anything_is_seen(sub):
    assert Grounding(sub).coverage() is None


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #
def test_tokenizer_ids_stay_a_bijection_after_eviction():
    """Eviction used to rewind ``_next_id`` onto ids that were still in use.

    The counter then walked forward over live symbols and reassigned them, so several pairs
    ended up mapped to one id. Measured before the fix at a bounded vocabulary: 81 pairs sharing
    6 ids.
    """
    t = Tokenizer(max_vocab=262, merge_at=2)
    rng = np.random.default_rng(0)
    for _ in range(400):
        t.encode("".join(rng.choice(list("abcdefghij"), 20)))
    assert t.evicted > 0, "the bound was never reached; this test proved nothing"
    assert sorted(t._merges.values()) == sorted(t._inverse), \
        "two pairs share one id — encode and decode no longer agree"


def test_tokenizer_roundtrip_survives_eviction():
    """The lossless claim has to hold at the bound, not only below it.

    The existing round-trip test uses a vocabulary large enough never to evict. With eviction
    running, every one of 50 round trips failed before the fix.
    """
    t = Tokenizer(max_vocab=262, merge_at=2)
    rng = np.random.default_rng(1)
    corpus = ["".join(rng.choice(list("abcdefghij"), 20)) for _ in range(400)]
    for s in corpus:
        t.encode(s)
    assert t.evicted > 0
    for s in corpus[:50]:
        assert t.decode(t.encode(s, learn=False)) == s, f"lossy after eviction on {s!r}"


def test_tokenizer_never_evicts_a_symbol_another_merge_is_built_from():
    """Expanding a merge whose component was dropped cannot recover the original bytes.

    Only roots of the merge forest — symbols nothing else expands into — may be evicted.
    """
    t = Tokenizer(max_vocab=262, merge_at=2)
    rng = np.random.default_rng(2)
    for _ in range(400):
        t.encode("".join(rng.choice(list("abcde"), 24)))
    referenced = {c for pair in t._inverse.values() for c in pair if c >= 256}
    assert referenced <= set(t._inverse), "a live merge references an evicted symbol"


def test_sequence_state_stays_bounded_across_many_calls(sub):
    """The carried state must enter the scan at t=0 only.

    Broadcasting it across all T timesteps re-injects it every step and the scan sums those
    geometrically, so ``forward(carry=True)`` compounds without bound. Measured before the fix:
    ``state_norm`` 2.6e+139 after 120 calls, with GELU overflowing to inf.
    """
    seq = SequenceModel(sub, depth=2)
    emb = DynamicEmbedding(sub, vocab=256)
    ctx = np.random.default_rng(0).normal(0, 1, sub.width)
    for _ in range(200):
        seq.forward(emb.embed_sequence([1, 2, 3, 4, 5, 6, 7, 8], context=ctx))
    norm = seq.state_norm()
    assert norm is not None and np.isfinite(norm), f"state diverged: {norm}"
    assert norm < 1e4, f"state is finite but exploding: {norm}"


def test_sequence_decay_gate_still_receives_gradient(sub):
    """Masking t>0 must remove the extra injections, not the term that teaches the gate."""
    from nyxara.nyx001.substrate import autograd as A
    from nyxara.nyx001.substrate import backward

    seq = SequenceModel(sub, depth=2)
    emb = DynamicEmbedding(sub, vocab=256)
    ctx = np.random.default_rng(0).normal(0, 1, sub.width)
    seq.forward(emb.embed_sequence([1, 2, 3], context=ctx))     # establish a carried state
    out = seq.forward(emb.embed_sequence([4, 5, 6], context=ctx))
    for b in seq._blocks:
        b["a_raw"].zero_grad()
    total = A.matmul(A.reshape(out, (3, sub.width)), A.constant(np.ones((sub.width, 1))))
    backward(A.reshape(total, (1, 3)))
    for b in seq._blocks:
        g = b["a_raw"].grad
        assert g is not None and float(np.abs(g).sum()) > 0.0, \
            "the decay gate is detached again — the recurrence is decorative"
