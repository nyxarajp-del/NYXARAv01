"""NYXARA · growth/tokenizer.py — her own byte-level BPE tokenizer, trained from zero (🔤→🧠).

Every from-scratch backend in :mod:`~nyxara.growth.foundry_models` speaks **raw bytes**
(``_VOCAB = 256``). That is honest and it never fails, but it is the single biggest thing
standing between her and a real brain at scale: at 256 symbols a 2048-token context is 2048
*bytes* — roughly 300 words — and every parameter spent on the embedding table is spent
learning that ``t`` often follows ``n``. A trained sub-word vocabulary buys back a ~4×
effective context and lets the weights learn *meaning* instead of spelling.

So this module gives her a tokenizer she trains herself, on her own corpus, from zero — no
downloaded vocabulary, no borrowed merges. Byte-level BPE, in the lineage of GPT-2/Llama, with
four deliberate choices for the four things the Master asked her to be good at:

* **Byte fallback — every one of the 256 bytes is in the vocabulary.** Merges only ever *combine*
  tokens that already exist, so there is no such thing as an out-of-vocabulary input and
  ``decode(encode(s)) == s`` holds byte-exactly for *any* string: Devanagari, emoji, mojibake,
  a truncated UTF-8 sequence, arbitrary binary. This is a property of the construction, not a
  hope, and :func:`NyxTokenizer.encode` asserts the pre-tokenizer half of it at runtime.
* **Digits are never merged.** Each digit is its own pre-token, so ``4096`` is four tokens and
  never a single opaque one. This is the cheapest large win available in arithmetic: a model
  cannot carry a column it cannot see.
* **Whitespace runs are their own tokens.** Indentation is *syntax* in Python, and a tokenizer
  that glues ``    return`` into one symbol makes every indent level a different word. Runs of
  two-or-more spaces (and tabs) are pre-tokenized apart; a single leading space still rides with
  its word, which is what keeps ordinary prose cheap.
* **Her template's role markers are atomic special tokens.** ``### User:`` and ``### NYXARA:``
  (:mod:`~nyxara.mind.llm`) each encode to exactly one id. That is what makes the supervised
  answer-span mask in ``growth/sft.py`` exact rather than a string search, and what lets
  generation stop on a token id instead of a substring scan.

**On speed, honestly.** This is a pure-standard-library implementation — no ``tokenizers``, no
``sentencepiece``, nothing to install. Training uses an incrementally-indexed pair counter
(only the words touched by a merge are rescored), and encoding memoizes per pre-token, which is
where essentially all the speed comes from since real text repeats its words. That is fast
enough to tokenize a large corpus in a background shard build; it is *not* competitive with a
Rust tokenizer, and this docstring is not going to pretend otherwise.

Depends on nothing inside NYXARA except, lazily and optionally, ``mind/llm`` for the exact role
marker strings — and it hard-codes safe copies so that import can never be required. Nothing
imports back.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "NyxTokenizer",
    "TokenizerReport",
    "train_tokenizer",
    "load_tokenizer",
    "BYTE_VOCAB",
    "SPECIALS",
]

BYTE_VOCAB = 256          # ids 0..255 are the raw bytes, always, in every trained tokenizer

# The role markers from mind/llm.py (_SELF_USER_TAG / _SELF_ASSISTANT_TAG). Copied literally
# rather than imported so a tokenizer can be built with no NYXARA imports at all; the values are
# verified against the live ones by tests/growth/test_tokenizer.py, which is the right place for
# that check — a drift there should fail loudly in CI, not silently at training time.
USER_TAG = "### User:"
ASSISTANT_TAG = "### NYXARA:"

# Order matters: these are appended to the vocabulary in exactly this sequence, so the ids are
# stable across every tokenizer trained by this module regardless of corpus or vocab size.
SPECIALS: Tuple[str, ...] = ("<|bos|>", "<|eos|>", "<|pad|>", USER_TAG, ASSISTANT_TAG)

# --------------------------------------------------------------------------- #
# Pre-tokenization
# --------------------------------------------------------------------------- #
# Ordered alternation, and every branch consumes at least one character, so the concatenation of
# all matches reconstructs the input exactly (encode() asserts this). The branches, in order:
#
#   [\r\n]+        newline runs — a paragraph break is one token, not twelve
#   [ ]{2,}|\t+    INDENTATION: runs of 2+ spaces, or any run of tabs, kept apart from the word
#   ?[^\W\d_]+     a word (unicode letters), optionally carrying ONE leading space
#   ?\d            a SINGLE digit, optionally carrying one leading space — never merged with
#                  its neighbours, which is what makes column arithmetic learnable
#   ?[^\s]         any other single character (punctuation, symbols, emoji, stray bytes)
#   \s             a lone whitespace character that nothing above claimed
#
# ``_`` is excluded from the word class so ``snake_case`` splits at the underscore — identifiers
# then share sub-words with the prose that names the same concept.
_PRETOKEN_RE = re.compile(
    r"[\r\n]+"
    r"|[ ]{2,}"
    r"|\t+"
    r"| ?[^\W\d_]+"
    r"| ?\d"
    r"| ?[^\s]"
    r"|\s",
    re.UNICODE,
)


def _pretokenize(text: str) -> List[str]:
    """Split ``text`` into pre-tokens. Lossless: ``"".join(_pretokenize(t)) == t``.

    The regex is total over any string, but a defensive character-wise fallback is kept for the
    impossible case: losing input here would silently break the round-trip guarantee that the
    rest of this module (and the corpus builder) relies on, so it is checked rather than assumed.
    """
    pieces = _PRETOKEN_RE.findall(text)
    if "".join(pieces) != text:     # pragma: no cover — defensive; the regex is total
        return list(text)
    return pieces


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class TokenizerReport:
    """A countable account of one training run — what she learned and where it stopped."""

    vocab_size: int = 0
    merges: int = 0
    requested_vocab: int = 0
    corpus_bytes: int = 0
    unique_pretokens: int = 0
    stopped_early: bool = False       # ran out of mergeable pairs before reaching the target
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def summary(self) -> str:
        head = (f"vocab {self.vocab_size} ({self.merges} merges) from "
                f"{self.corpus_bytes:,} bytes / {self.unique_pretokens:,} unique pre-tokens")
        if self.stopped_early:
            head += (f"; STOPPED EARLY — the corpus had no more mergeable pairs "
                     f"(asked for {self.requested_vocab})")
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"· {head}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"vocab_size": self.vocab_size, "merges": self.merges,
                "requested_vocab": self.requested_vocab, "corpus_bytes": self.corpus_bytes,
                "unique_pretokens": self.unique_pretokens,
                "stopped_early": self.stopped_early, "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# The tokenizer
# --------------------------------------------------------------------------- #
class NyxTokenizer:
    """A byte-level BPE tokenizer NYXARA trains on her own corpus.

    Construct it empty (``NyxTokenizer()``) for the identity byte tokenizer — vocab 256 plus the
    specials, no merges — which is exactly the behaviour every backend has today. Call
    :meth:`train` to learn merges, or :func:`load_tokenizer` to restore one from a checkpoint.
    """

    def __init__(self) -> None:
        # id -> the bytes that id stands for. 0..255 are the raw bytes; specials and merged
        # tokens are appended after, in that order, and never renumbered.
        self.id_to_bytes: List[bytes] = [bytes([i]) for i in range(BYTE_VOCAB)]
        self.special_to_id: Dict[str, int] = {}
        self.id_to_special: Dict[int, str] = {}
        for tag in SPECIALS:
            self.special_to_id[tag] = len(self.id_to_bytes)
            self.id_to_special[len(self.id_to_bytes)] = tag
            self.id_to_bytes.append(tag.encode("utf-8"))
        # (left_id, right_id) -> merged_id, in learned order. Rank is the insertion order, which
        # is what encode() applies greedily.
        self.merges: Dict[Tuple[int, int], int] = {}
        self._rank: Dict[Tuple[int, int], int] = {}
        # pre-token string -> its token ids. This memo is where the speed lives: real corpora
        # repeat their words, so the BPE inner loop runs once per DISTINCT pre-token, not once
        # per occurrence. Bounded so a pathological corpus cannot grow it without limit.
        self._cache: Dict[str, List[int]] = {}
        self._cache_max = 500_000

    # ---- sizes & identity ---- #
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_bytes)

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def user_id(self) -> int:
        return self.special_to_id[USER_TAG]

    @property
    def assistant_id(self) -> int:
        return self.special_to_id[ASSISTANT_TAG]

    def vocab_sig(self) -> str:
        """A stable fingerprint of this exact vocabulary — ``sha256`` over size + merges.

        The foundry's promotion gate uses this to tell whether a candidate speaks the *same*
        language as the active model. Two brains with different signatures cannot have their
        per-token perplexities compared at all (see ``eval/domains.bits_per_byte``), so this is
        load-bearing, not bookkeeping.
        """
        h = hashlib.sha256()
        h.update(str(self.vocab_size).encode())
        for (a, b), new in self.merges.items():
            h.update(f"{a},{b}->{new};".encode())
        return h.hexdigest()

    # ---- training ---- #
    def train(self, corpus: Iterable[str], *, vocab_size: int = 32768,
              min_frequency: int = 2, report: Optional[TokenizerReport] = None
              ) -> TokenizerReport:
        """Learn merges from ``corpus`` until the vocabulary reaches ``vocab_size``.

        The corpus is consumed once into a frequency table over *pre-tokens*, and every merge
        after that is scored on that table rather than on the text — so cost scales with the
        number of distinct words, not with corpus size. A merge is only ever between two
        adjacent ids inside one pre-token, which is what keeps digits (and the special tags)
        atomic no matter how often they co-occur.

        Stops early, and says so in the report, when no pair occurs ``min_frequency`` times —
        a small corpus produces a small vocabulary rather than a padded one.
        """
        report = report if report is not None else TokenizerReport()
        report.requested_vocab = int(vocab_size)

        # ---- pass 1: pre-token frequencies ---- #
        freqs: Dict[str, int] = defaultdict(int)
        total_bytes = 0
        for doc in corpus:
            if not doc:
                continue
            total_bytes += len(doc.encode("utf-8", "replace"))
            for piece in _pretokenize(doc):
                freqs[piece] += 1
        report.corpus_bytes = total_bytes
        report.unique_pretokens = len(freqs)
        if not freqs:
            report.note("empty corpus — kept the identity byte tokenizer")
            report.vocab_size = self.vocab_size
            return report

        # Each distinct pre-token becomes one symbol sequence. A pre-token that IS a special tag
        # is dropped here: it already has an id, and letting it merge would break its atomicity.
        words: List[List[int]] = []
        counts: List[int] = []
        for piece, count in freqs.items():
            if piece in self.special_to_id:
                continue
            words.append(list(piece.encode("utf-8", "replace")))
            counts.append(count)

        # ---- the incremental pair index ---- #
        # Rescoring every word on every merge is what makes naive BPE unusable at 32k merges, so
        # we keep a pair -> total-count map alongside pair -> which words contain it, and touch
        # only the affected words when a merge fires.
        # A max-heap over pair counts, with LAZY invalidation. Scanning ``pair_counts`` for the
        # best pair on every merge is the thing that makes naive BPE quadratic and unusable at
        # 32k merges over a real corpus: a large corpus has millions of distinct pairs, and
        # 32_000 × millions is not a number of operations that finishes. Entries are pushed
        # whenever a count changes and are checked against the live count when popped, so a
        # stale entry costs one comparison instead of an invariant to maintain.
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_words: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        heap: List[Tuple[int, Tuple[int, int]]] = []
        dirty: Set[Tuple[int, int]] = set()

        def index_word(idx: int) -> None:
            seq, count = words[idx], counts[idx]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += count
                pair_words[pair].add(idx)
                dirty.add(pair)

        def deindex_word(idx: int) -> None:
            seq, count = words[idx], counts[idx]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_words.pop(pair, None)
                else:
                    pair_words[pair].discard(idx)
                    dirty.add(pair)

        def flush() -> None:
            """Push every changed count onto the heap (one entry per change, not per pair)."""
            for pair in dirty:
                count = pair_counts.get(pair)
                if count:
                    heapq.heappush(heap, (-count, pair))
            dirty.clear()

        for i in range(len(words)):
            index_word(i)
        flush()

        target_merges = max(0, int(vocab_size) - self.vocab_size)
        min_freq = max(1, int(min_frequency))
        merged = 0
        while merged < target_merges:
            flush()
            # Pop until the top entry agrees with the live count. Ties break on the pair itself,
            # so a run is reproducible across processes and machines.
            best: Optional[Tuple[int, int]] = None
            while heap:
                neg, candidate = heap[0]
                if pair_counts.get(candidate, 0) != -neg:
                    heapq.heappop(heap)          # stale — the count moved since this was pushed
                    continue
                best = candidate
                break
            if best is None:
                report.stopped_early = True
                break
            if pair_counts[best] < min_freq:
                report.stopped_early = True
                break
            heapq.heappop(heap)

            new_id = len(self.id_to_bytes)
            self.id_to_bytes.append(self.id_to_bytes[best[0]] + self.id_to_bytes[best[1]])
            self._rank[best] = len(self.merges)
            self.merges[best] = new_id
            merged += 1

            for idx in list(pair_words.get(best, ())):
                deindex_word(idx)
                words[idx] = _apply_merge(words[idx], best, new_id)
                index_word(idx)

        report.merges = merged
        report.vocab_size = self.vocab_size
        if report.stopped_early:
            report.note(f"corpus exhausted at {self.vocab_size} tokens "
                        f"(min_frequency={min_freq})")
        self._cache.clear()
        return report

    # ---- encoding ---- #
    def _encode_piece(self, piece: str) -> List[int]:
        """BPE one pre-token into ids, memoized. Never fails, never emits an unknown."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        special = self.special_to_id.get(piece)
        if special is not None:
            ids = [special]
        else:
            ids = list(piece.encode("utf-8", "replace"))
            # Greedy by learned rank: repeatedly apply the earliest-learned applicable merge,
            # which is what reproduces the training-time segmentation.
            while len(ids) >= 2:
                best_pair, best_rank = None, None
                for pair in zip(ids, ids[1:]):
                    rank = self._rank.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_pair, best_rank = pair, rank
                if best_pair is None:
                    break
                ids = _apply_merge(ids, best_pair, self.merges[best_pair])

        if len(self._cache) < self._cache_max:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode ``text`` to token ids. Total — every string encodes, none can fail.

        The role markers are matched *before* the pre-tokenizer sees them, so each becomes
        exactly one id even though the pre-tokenizer would otherwise shred ``### NYXARA:`` into
        punctuation and words.
        """
        ids: List[int] = [self.bos_id] if add_bos else []
        for chunk, is_special in _split_specials(text):
            if is_special:
                ids.append(self.special_to_id[chunk])
            else:
                for piece in _pretokenize(chunk):
                    ids.extend(self._encode_piece(piece))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_specials: bool = False) -> str:
        """Decode ids back to text. Byte-exact for anything :meth:`encode` produced.

        Decoding concatenates *bytes* and only then decodes UTF-8, which is what makes a
        multi-byte character split across two tokens reassemble correctly instead of turning
        into two replacement characters.
        """
        out = bytearray()
        for i in ids:
            if i < 0 or i >= len(self.id_to_bytes):
                continue                       # an id from another vocabulary: skip, never raise
            if skip_specials and i in self.id_to_special:
                continue
            out += self.id_to_bytes[i]
        return out.decode("utf-8", errors="replace")

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "nyx-bpe-1",
            "vocab_size": self.vocab_size,
            "specials": list(SPECIALS),
            # Only the merges need storing: ids 0..255 and the specials are reconstructed by
            # __init__, and every merged id follows from applying the merges in order.
            "merges": [[a, b] for (a, b) in self.merges],
            "vocab_sig": self.vocab_sig(),
        }

    def save(self, directory: Any) -> Path:
        """Write ``tokenizer.json`` into ``directory`` and return its path.

        Lives beside ``spec.json`` in the model version directory on purpose: a checkpoint that
        cannot say which language it speaks is a checkpoint that cannot be served.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "tokenizer.json"
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "NyxTokenizer":
        tok = cls()
        specials = list(blob.get("specials") or SPECIALS)
        if specials != list(SPECIALS):
            # A tokenizer written by a build with a different special set cannot be replayed into
            # this one: the ids would silently shift under every merge. Refuse rather than
            # produce a tokenizer that decodes to the wrong text.
            raise ValueError(f"tokenizer special-token set mismatch: {specials} != {list(SPECIALS)}")
        for pair in blob.get("merges") or ():
            a, b = int(pair[0]), int(pair[1])
            new_id = len(tok.id_to_bytes)
            tok.id_to_bytes.append(tok.id_to_bytes[a] + tok.id_to_bytes[b])
            tok._rank[(a, b)] = len(tok.merges)
            tok.merges[(a, b)] = new_id
        return tok

    def load(self, directory: Any) -> "NyxTokenizer":
        """Restore in place from ``directory/tokenizer.json``."""
        other = load_tokenizer(directory)
        if other is None:
            raise FileNotFoundError(f"no tokenizer.json under {directory}")
        self.id_to_bytes = other.id_to_bytes
        self.special_to_id = other.special_to_id
        self.id_to_special = other.id_to_special
        self.merges = other.merges
        self._rank = other._rank
        self._cache = {}
        return self


def _apply_merge(seq: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``seq`` with ``new_id``."""
    out: List[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


# Longest-first so a tag that is a prefix of another can never mask it.
_SPECIAL_SPLIT_RE = re.compile("(" + "|".join(re.escape(s) for s in
                                              sorted(SPECIALS, key=len, reverse=True)) + ")")


def _split_specials(text: str) -> List[Tuple[str, bool]]:
    """Split ``text`` into ``(chunk, is_special)`` runs. Lossless by construction."""
    out: List[Tuple[str, bool]] = []
    for part in _SPECIAL_SPLIT_RE.split(text):
        if not part:
            continue
        out.append((part, part in SPECIALS))
    return out


# --------------------------------------------------------------------------- #
# Module-level helpers (the shape the rest of the foundry calls)
# --------------------------------------------------------------------------- #
def train_tokenizer(corpus: Iterable[str], *, vocab_size: int = 32768,
                    min_frequency: int = 2, save_to: Any = None
                    ) -> Tuple[NyxTokenizer, TokenizerReport]:
    """Train a tokenizer on ``corpus`` and optionally save it. Returns ``(tokenizer, report)``."""
    tok = NyxTokenizer()
    report = tok.train(corpus, vocab_size=vocab_size, min_frequency=min_frequency)
    if save_to is not None:
        tok.save(save_to)
    return tok, report


def load_tokenizer(directory: Any) -> Optional[NyxTokenizer]:
    """Load ``directory/tokenizer.json``, or ``None`` when there isn't one.

    Returns ``None`` rather than raising for a missing file: a checkpoint without a tokenizer is
    a byte-level checkpoint, which is a valid thing to be and exactly what every model trained
    before this module existed is. A *corrupt* tokenizer still raises — that is a real fault and
    silently serving the wrong vocabulary would be far worse than a crash.
    """
    if directory is None:
        return None
    path = Path(directory)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        return None
    return NyxTokenizer.from_dict(json.loads(path.read_text(encoding="utf-8")))
