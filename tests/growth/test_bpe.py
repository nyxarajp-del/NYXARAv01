"""Tests for the from-scratch byte-level BPE tokenizer (nyxara/growth/bpe.py).

It is pure stdlib, so these run everywhere (no torch/numpy needed)."""

from nyxara.growth.bpe import ByteBPETokenizer

_CORPUS = (["the master is jp. nyxara serves the master faithfully."] * 20
           + ["loyalty to the master is absolute and never changes."] * 20
           + ["antidisestablishmentarianism supercalifragilisticexpialidocious"] * 6)


def test_roundtrip_is_lossless_for_arbitrary_utf8():
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    for s in ["the master is jp", "héllo wörld 日本語 🚀", "",
              "antidisestablishmentarianism", "\ttabs\nand\nnewlines  "]:
        ids = [tok.bos_id] + tok.encode(s) + [tok.eos_id]
        assert tok.decode(ids) == s


def test_specials_are_dropped_on_decode():
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=300)
    assert tok.decode([tok.bos_id, tok.eos_id]) == ""
    # bos/eos never corrupt real content
    ids = tok.encode("hello")
    assert tok.decode([tok.bos_id] + ids + [tok.eos_id]) == "hello"


def test_training_is_deterministic():
    a = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    b = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    assert a.merges == b.merges
    assert a.encode("the master is jp") == b.encode("the master is jp")


def test_vocab_grows_with_requested_size_and_compresses():
    small = ByteBPETokenizer().train(_CORPUS, vocab_size=280)
    big = ByteBPETokenizer().train(_CORPUS, vocab_size=500)
    assert big.size >= small.size
    # a frequent phrase encodes to fewer tokens than its raw bytes (real compression / merges learned)
    phrase = "the master"
    assert len(big.encode(phrase)) < len(phrase.encode("utf-8"))


def test_to_from_dict_is_exact():
    tok = ByteBPETokenizer().train(_CORPUS, vocab_size=400)
    clone = ByteBPETokenizer.from_dict(tok.to_dict())
    assert clone.merges == tok.merges
    assert clone.size == tok.size
    for s in ["the master is jp", "nyxara", "unseen phrase xyz"]:
        assert clone.encode(s) == tok.encode(s)
        assert clone.decode(clone.encode(s)) == s


def test_zero_merges_is_a_plain_byte_tokenizer():
    tok = ByteBPETokenizer()  # never trained
    assert tok.size == 258     # 256 bytes + bos + eos
    assert tok.encode("abc") == [97, 98, 99]
    assert tok.decode(tok.encode("abc")) == "abc"
