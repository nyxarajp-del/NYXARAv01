"""NYXARA · growth/bpe.py — a from-scratch byte-level BPE tokenizer she trains herself (🔤).

The pure-NumPy brain (:mod:`nyxara.growth.genesis_numpy`) used to tokenize at the *word* level
with a hard 600-word vocabulary: every rarer word folded onto ``<unk>``, so the model could
never even represent — let alone learn — most of what it read. That is a large, silent quality
ceiling on a sovereign brain that has no giant pretrained tokenizer to borrow.

This module removes it. :class:`ByteBPETokenizer` is a genuine **byte-level Byte-Pair-Encoding**
tokenizer, trained FROM ZERO on NYXARA's own corpus, pure standard library (no ``tokenizers``,
no ``sentencepiece``, no network):

* The base alphabet is the 256 raw byte values, so **every** possible input is representable and
  there is **no ``<unk>`` token, ever** — round-tripping is lossless for arbitrary UTF-8.
* Training greedily merges the most frequent adjacent symbol pair into a new symbol, repeatedly,
  until the requested ``vocab_size`` is reached or no pair repeats. Ties are broken by the lowest
  pair id, so training is **deterministic**: the same corpus + size always yields the same merges.
* ``to_dict``/``from_dict`` serialize the ordered merge list (integer id pairs) exactly, so a
  saved model reloads a bit-identical tokenizer.

It implements just what :class:`~nyxara.growth.genesis_numpy.GenesisNumpyModel` needs —
``train``, ``encode``, ``decode``, ``size`` — and nothing else. It never touches the frozen
word/byte tokenizers in :mod:`nyxara.growth.foundry_models`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = ["ByteBPETokenizer"]

# Special tokens live ABOVE the 256 byte values so a raw byte id is always < 256 and a special
# id is always >= 256 — the two id spaces never collide, and decode() can drop specials cleanly.
_BOS_ID = 256
_EOS_ID = 257
_N_SPECIAL = 2
_BASE = 258            # first id available for a learned merge
_BYTE_MAX = 256


class ByteBPETokenizer:
    """A deterministic, from-scratch byte-level BPE tokenizer (pure stdlib).

    Ids: ``0..255`` are the raw bytes, ``256`` is ``<bos>``, ``257`` is ``<eos>``, and every id
    ``>= 258`` is a learned merge of two lower ids. ``vocab`` maps each id to the byte string it
    expands to (specials expand to ``b""``), so :meth:`decode` is a pure concatenation.
    """

    def __init__(self) -> None:
        # ordered list of (id_a, id_b) merges; the i-th merge is assigned id (_BASE + i)
        self.merges: List[Tuple[int, int]] = []
        # merge lookup for encoding: (id_a, id_b) -> merged_id
        self._rank: Dict[Tuple[int, int], int] = {}
        # id -> raw bytes it expands to (built from the merges); specials -> b""
        self._expand: Dict[int, bytes] = {}
        self._rebuild_expand()

    # ---- ids ---- #
    @property
    def bos_id(self) -> int:
        return _BOS_ID

    @property
    def eos_id(self) -> int:
        return _EOS_ID

    @property
    def size(self) -> int:
        """Total vocabulary size: 256 bytes + 2 specials + learned merges."""
        return _BASE + len(self.merges)

    # ---- training ---- #
    def train(self, corpus: Sequence[str], vocab_size: int) -> "ByteBPETokenizer":
        """Learn merges from ``corpus`` until the vocabulary reaches ``vocab_size``.

        ``vocab_size`` counts bytes + specials + merges, so it must be at least ``_BASE`` (258);
        a smaller request simply learns zero merges (a plain byte tokenizer). Deterministic:
        equal inputs give equal merges, ties broken by the lowest pair. Returns ``self``."""
        target_merges = max(0, int(vocab_size) - _BASE)
        # Represent each document as a list of current symbol ids, starting from raw bytes.
        seqs: List[List[int]] = []
        for doc in corpus:
            if not doc:
                continue
            seqs.append(list(doc.encode("utf-8", errors="replace")))
        seqs = [s for s in seqs if s]
        self.merges = []
        self._rank = {}
        for _ in range(target_merges):
            pair = self._best_pair(seqs)
            if pair is None:
                break                     # no adjacent pair repeats — nothing left to merge
            new_id = _BASE + len(self.merges)
            self.merges.append(pair)
            self._rank[pair] = new_id
            seqs = [self._apply_merge(s, pair, new_id) for s in seqs]
        self._rebuild_expand()
        return self

    @staticmethod
    def _best_pair(seqs: Sequence[Sequence[int]]) -> Tuple[int, int] | None:
        """The most frequent adjacent (a, b) pair across all sequences; ties → lowest pair."""
        counts: Dict[Tuple[int, int], int] = {}
        for s in seqs:
            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                counts[p] = counts.get(p, 0) + 1
        best: Tuple[int, int] | None = None
        best_c = 0
        for p, c in counts.items():
            if c < 2:                       # a pair must repeat to be worth merging
                continue
            if best is None or c > best_c or (c == best_c and p < best):
                best, best_c = p, c
        return best

    @staticmethod
    def _apply_merge(seq: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        a, b = pair
        out: List[int] = []
        i, n = 0, len(seq)
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ---- encode / decode ---- #
    def encode(self, text: str) -> List[int]:
        """Encode ``text`` to token ids (no BOS/EOS — the caller adds those)."""
        ids: List[int] = list(text.encode("utf-8", errors="replace"))
        if not self._rank or len(ids) < 2:
            return ids
        # Greedily apply the learned merges in the order they were learned (lowest rank first),
        # matching the training dynamics so the segmentation is consistent.
        while True:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                mid = self._rank.get((ids[i], ids[i + 1]))
                if mid is not None and (best_rank is None or mid < best_rank):
                    best_rank, best_pos = mid, i
            if best_rank is None:
                break
            ids[best_pos:best_pos + 2] = [best_rank]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text, dropping specials. Lossless for real byte content."""
        buf = bytearray()
        for i in ids:
            b = self._expand.get(int(i))
            if b:
                buf.extend(b)
        return buf.decode("utf-8", errors="replace")

    def _rebuild_expand(self) -> None:
        """Rebuild the id -> bytes expansion table from the current merges."""
        exp: Dict[int, bytes] = {i: bytes([i]) for i in range(_BYTE_MAX)}
        exp[_BOS_ID] = b""
        exp[_EOS_ID] = b""
        for idx, (a, b) in enumerate(self.merges):
            new_id = _BASE + idx
            exp[new_id] = exp.get(a, b"") + exp.get(b, b"")
        self._expand = exp
        self._rank = {pair: _BASE + idx for idx, pair in enumerate(self.merges)}

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, object]:
        """A JSON-able snapshot: the ordered merge list is the whole learned state."""
        return {"merges": [[int(a), int(b)] for a, b in self.merges],
                "bos_id": _BOS_ID, "eos_id": _EOS_ID}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ByteBPETokenizer":
        tok = cls()
        tok.merges = [(int(a), int(b)) for a, b in (d.get("merges") or [])]  # type: ignore[union-attr]
        tok._rebuild_expand()
        return tok
