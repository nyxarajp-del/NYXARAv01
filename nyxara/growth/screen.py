"""NYXARA · growth/screen.py — corpus screens whose memory does not grow with the corpus (🧹→📏).

:mod:`~nyxara.growth.corpus` screens every document through a ``Set[str]`` of content hashes. At
the scale it was written for that is right. At 6B tokens it is 15–25M entries of 64-character
hex — **3–5 GB of RAM, on a 16 GB box that is also holding shard buffers and a tokenizer** — and
it is lost entirely on restart, so a resumed build silently re-admits everything it already had.

Everything here is **flat in memory**: fixed-size numpy tables sized once, at startup, from a bit
budget. A screen whose cost scales with what it has already seen is a screen that fails exactly
when the corpus gets big enough to need it.

Four things live here:

* :class:`ExactDedup` — 268 MB open-addressed table replacing the unbounded set.
* :class:`MinHashLSH` — near-duplicate detection, which the corpus has none of today. Web text is
  heavily boilerplate-duplicated and an exact hash catches none of it.
* :func:`language_ok` and :func:`route_domain` — cheap screens for streamed text, so prose does
  not get filed as code and LaTeX-free arithmetic does not get filed as maths.
* :func:`split_for` — a deterministic train/val assignment, so a held-out split exists at all and
  survives a resumed build without leaking.

numpy is required (the corpus reader already requires it). Depends on nothing in ``nyxara``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "ExactDedup",
    "MinHashLSH",
    "DedupStats",
    "fingerprint",
    "shingles",
    "language_ok",
    "route_domain",
    "split_for",
]

_MASK64 = (1 << 64) - 1
_EMPTY = np.uint64(0)
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _blake64(data: bytes) -> int:
    """A stable 64-bit hash. Never ``hash()`` — that is salted per process."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def fingerprint(text: str) -> int:
    """A 64-bit fingerprint of ``text``, normalised the way ``dataset._content_hash`` normalises.

    Lowercased, whitespace-collapsed and truncated, so trivial reformatting does not present as
    a new document. Returns an int rather than hex: 8 bytes in a numpy table instead of 64
    characters in a Python string is the whole point.
    """
    norm = _WS_RE.sub(" ", (text or "").lower()).strip()[:2000]
    h = _blake64(norm.encode("utf-8", "replace"))
    return h or 1          # 0 is the empty slot marker; collapse it to 1 rather than lose it


def shingles(text: str, *, width: int = 5) -> List[int]:
    """Hashed word n-grams — the set whose Jaccard similarity MinHash estimates."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < width:
        return [_blake64(" ".join(words).encode())] if words else []
    seen: Dict[int, None] = {}
    for i in range(len(words) - width + 1):
        seen.setdefault(_blake64(" ".join(words[i:i + width]).encode()), None)
    return list(seen)


@dataclass
class DedupStats:
    """What each screen refused, and how loaded its table is."""

    exact_hits: int = 0
    near_hits: int = 0
    inserted: int = 0
    evictions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"exact_duplicates": self.exact_hits, "near_duplicates": self.near_hits,
                "inserted": self.inserted, "lsh_evictions": self.evictions}


class ExactDedup:
    """Exact-duplicate membership in a fixed-size open-addressed ``uint64`` table.

    2²⁵ slots × 8 bytes = **268 MB, constant**, versus 3–5 GB and growing for the set it
    replaces. At 12M documents the load factor is 0.36, where linear probing is still cheap.

    The one honest cost: two documents whose 64-bit fingerprints collide are treated as
    duplicates and the second is dropped. At 25M documents that is a birthday probability of
    ~1.7e-5 — a handful of documents across the whole corpus, against multiple GB of RAM. It is
    reported in :attr:`stats` rather than hidden.
    """

    def __init__(self, *, capacity_bits: int = 25) -> None:
        self.bits = max(10, int(capacity_bits))
        self.size = 1 << self.bits
        self._mask = self.size - 1
        self._table = np.zeros(self.size, dtype=np.uint64)
        self.count = 0
        self.stats = DedupStats()

    @property
    def load(self) -> float:
        return self.count / self.size

    def _slot(self, h: int) -> int:
        # Multiplicative scatter: blake2b is already uniform, but the low bits alone would
        # cluster for fingerprints that share a suffix.
        return int((h ^ (h >> 29)) & self._mask)

    def add_if_new(self, h: int) -> bool:
        """``True`` if ``h`` was unseen (and is now stored); ``False`` if it is a duplicate."""
        key = np.uint64(h or 1)
        idx = self._slot(int(key))
        table = self._table
        for _ in range(self.size):
            current = table[idx]
            if current == _EMPTY:
                table[idx] = key
                self.count += 1
                self.stats.inserted += 1
                return True
            if current == key:
                self.stats.exact_hits += 1
                return False
            idx = (idx + 1) & self._mask
        raise RuntimeError("ExactDedup table is full — raise capacity_bits")

    def __contains__(self, h: int) -> bool:
        key = np.uint64(h or 1)
        idx = self._slot(int(key))
        for _ in range(self.size):
            current = self._table[idx]
            if current == _EMPTY:
                return False
            if current == key:
                return True
            idx = (idx + 1) & self._mask
        return False

    def save(self, path: Any) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, self._table, allow_pickle=False)
        return out

    @classmethod
    def load_from(cls, path: Any) -> "ExactDedup":
        table = np.load(Path(path), allow_pickle=False)
        obj = cls(capacity_bits=int(math.log2(len(table))))
        obj._table = table
        obj.count = int(np.count_nonzero(table))
        return obj


class MinHashLSH:
    """Near-duplicate detection with banded LSH over bottom-k MinHash signatures.

    **Bottom-k, not k independent permutations.** The textbook construction hashes every shingle
    once per permutation — 128 hashes per shingle, which in Python is unaffordable at billions
    of tokens. Bottom-k hashes each shingle *once* and keeps the k smallest values; the result
    estimates Jaccard similarity just as well, at 1/128th the hashing.

    ``bands × rows`` sets the threshold: a pair is a candidate when some band matches entirely,
    with probability ``1 - (1 - J^rows)^bands``. The default 8×16 puts the S-curve's midpoint at
    ``(1/8)^(1/16) ≈ 0.88`` — where boilerplate republication lives, without collapsing
    documents that merely share a topic.

    Band tables are direct-mapped and **lossy on collision**: a new signature overwrites whatever
    occupied the slot. That keeps memory flat at ``bands × 2^bits × 8`` bytes regardless of
    corpus size (2.1 GB at the defaults), degrading gracefully into missed duplicates rather
    than into an OOM. Evictions are counted, not hidden.
    """

    def __init__(self, *, bands: int = 8, rows: int = 16, capacity_bits: int = 22,
                 seed: int = 0) -> None:
        self.bands = max(1, int(bands))
        self.rows = max(1, int(rows))
        self.k = self.bands * self.rows
        self.bits = max(10, int(capacity_bits))
        self.size = 1 << self.bits
        self._mask = self.size - 1
        self._tables = [np.zeros(self.size, dtype=np.uint64) for _ in range(self.bands)]
        self.stats = DedupStats()
        rng = np.random.default_rng(seed)
        # Odd multipliers keep the mixing bijective mod 2^64.
        self._mul = (rng.integers(1, 1 << 62, size=self.bands, dtype=np.uint64) * 2 + 1)
        self._add = rng.integers(0, 1 << 62, size=self.bands, dtype=np.uint64)

    @property
    def threshold(self) -> float:
        """The Jaccard similarity at which a pair becomes ~50% likely to be caught."""
        return float((1.0 / self.bands) ** (1.0 / self.rows))

    def save(self, path: Any) -> Path:
        """Persist the band tables AND the hash coefficients.

        The coefficients matter as much as the tables: they are drawn from a seeded RNG at
        construction, so a reloaded filter with fresh coefficients would compute different band
        keys for the same document and match nothing it had already stored. The tables would be
        present and useless — near-duplicate detection silently off across a resume.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, mul=self._mul, add=self._add,
                 meta=np.array([self.bands, self.rows, self.bits], dtype=np.int64),
                 **{f"band{i}": t for i, t in enumerate(self._tables)})
        return out

    @classmethod
    def load_from(cls, path: Any) -> "MinHashLSH":
        blob = np.load(Path(path), allow_pickle=False)
        bands, rows, bits = (int(v) for v in blob["meta"])
        obj = cls(bands=bands, rows=rows, capacity_bits=bits)
        obj._mul = blob["mul"]
        obj._add = blob["add"]
        obj._tables = [blob[f"band{i}"] for i in range(bands)]
        return obj

    def signature(self, text: str) -> Optional[np.ndarray]:
        """Bottom-k MinHash of ``text``'s shingles, or ``None`` when there is too little text."""
        hashed = shingles(text)
        if not hashed:
            return None
        arr = np.array(sorted(hashed)[:self.k], dtype=np.uint64)
        if len(arr) < self.k:
            # Pad by cycling: a short document's signature must still be k wide, and repeating
            # its own minima keeps near-identical short documents mapping together.
            arr = np.resize(arr, self.k)
        return arr

    def _band_keys(self, signature: np.ndarray) -> List[int]:
        keys: List[int] = []
        for b in range(self.bands):
            chunk = signature[b * self.rows:(b + 1) * self.rows]
            acc = np.uint64(0)
            for value in chunk:
                acc = np.uint64((int(acc) * 0x100000001B3 + int(value)) & _MASK64)
            keys.append(int((int(acc) * int(self._mul[b]) + int(self._add[b])) & _MASK64) or 1)
        return keys

    def seen_or_add(self, text: str) -> bool:
        """``True`` when ``text`` is a near-duplicate of something already added."""
        signature = self.signature(text)
        if signature is None:
            return False
        hit = False
        for b, key in enumerate(self._band_keys(signature)):
            slot = (key ^ (key >> 31)) & self._mask
            table = self._tables[b]
            current = int(table[slot])
            if current == key:
                hit = True
            elif current != 0:
                self.stats.evictions += 1
            table[slot] = np.uint64(key)
        if hit:
            self.stats.near_hits += 1
        else:
            self.stats.inserted += 1
        return hit


# --------------------------------------------------------------------------- #
# Cheap content screens for streamed text
# --------------------------------------------------------------------------- #
_EN_STOPWORDS = frozenset("""
the of and to in a is that it for as was with be by on not he this are or his from at which but
have an they you had were their has been we all will would there can what about when more if out
""".split())


def language_ok(text: str, *, min_stopword_rate: float = 0.06,
                min_latin_rate: float = 0.70) -> bool:
    """A cheap English screen: mostly-Latin script with a plausible stopword rate.

    No model, no dependency — this runs on every streamed document. It is deliberately permissive
    on the Latin-script test (code and maths are legitimately stopword-poor, and are screened by
    :func:`route_domain` instead) and exists mainly to keep whole non-English dumps out of a
    corpus whose evaluation batteries are all English.
    """
    sample = (text or "")[:2000]
    if not sample.strip():
        return False
    letters = [c for c in sample if c.isalpha()]
    if letters:
        latin = sum(1 for c in letters if ord(c) < 0x250)
        if latin / len(letters) < min_latin_rate:
            return False
    words = _WORD_RE.findall(sample.lower())
    if len(words) < 20:
        return True                     # too short to judge; other screens will handle it
    return (sum(1 for w in words if w in _EN_STOPWORDS) / len(words)) >= min_stopword_rate


_CODE_TOKENS = re.compile(
    r"\b(def|class|function|return|import|from|const|let|var|public|private|static|void|"
    r"if|else|elif|for|while|try|catch|except|struct|impl|fn|package)\b")
_MATH_TOKENS = re.compile(
    r"(\\frac|\\sum|\\int|\\begin\{|\\end\{|\\alpha|\\beta|\\theta|\$\$|\\\[|"
    r"\btheorem\b|\blemma\b|\bproof\b|\bintegral\b|\bderivative\b|\bequation\b)", re.I)
_TURN_TOKENS = re.compile(r"(^|\n)\s*(###\s*(user|assistant)|user:|assistant:|human:|q:|a:)",
                          re.I)


def route_domain(text: str, hint: str = "general") -> str:
    """Re-file a streamed document into the domain it actually belongs to.

    Only ever *narrows* ``general``. A document a source already declared as code, maths or
    conversation keeps that label — the source knows better than these regexes, and second-
    guessing it would be a way to quietly empty a domain. What this catches is the other
    direction: The Stack's READMEs arriving as ``code``, open-web-math's prose arriving as
    ``math``, and code samples inside a general web crawl arriving as prose.
    """
    if hint != "general":
        return hint
    sample = (text or "")[:4000]
    if not sample.strip():
        return hint

    # Maths is tested FIRST, and the order is not cosmetic. LaTeX is brace-dense — `\begin{...}`,
    # `\frac{a}{b}`, `$...$` — so a symbol-ratio rule evaluated first classifies a theorem as
    # source code and quietly moves the maths corpus into the code domain. `\frac` and `\begin{`
    # are far more specific evidence than punctuation density, so they get to answer first.
    math_hits = len(_MATH_TOKENS.findall(sample))
    if math_hits >= 3:
        return "math"

    lines = sample.splitlines() or [sample]
    indented = sum(1 for line in lines if line[:1] in (" ", "\t")) / max(1, len(lines))
    code_hits = len(_CODE_TOKENS.findall(sample))
    symbols = sum(1 for c in sample if c in "{}();=<>[]") / max(1, len(sample))
    # The bare symbol-density rule still needs *some* corroboration, so that a document which is
    # merely punctuation-heavy (tables, markup, citation lists) is not filed as code either.
    if (code_hits >= 4 and (indented > 0.20 or symbols > 0.04)) or (symbols > 0.09
                                                                    and code_hits >= 1):
        return "code"

    if len(_TURN_TOKENS.findall(sample)) >= 2:
        return "conversation"
    return "general"


def split_for(fp: int, *, val_permille: int = 5) -> str:
    """Deterministic train/val assignment from a document's own fingerprint.

    Keyed on content, not on arrival order, so a document always lands on the same side however
    many times the build is resumed, reordered or re-sharded. An order-keyed or random split
    would quietly leak validation documents into training across a restart — and the resulting
    val loss would look good for exactly the wrong reason.
    """
    if val_permille <= 0:
        return "train"
    return "val" if (fp >> 17) % 1000 < int(val_permille) else "train"
