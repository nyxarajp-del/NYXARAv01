"""NYXARA · growth/corpus.py — a real pretraining corpus: four domains, sharded to disk (🌊→💾).

:meth:`~nyxara.growth.foundry.Foundry.collect_corpus` holds the whole corpus in memory as a
``List[str]`` and caps it at ``max_corpus_items = 2000``. That is exactly right for the thing it
was built for — fine-tuning on her own lived experience — and it is roughly six orders of
magnitude short of what a 300M-parameter model needs. Chinchilla-optimal for 300M is ~6B tokens,
which is ~24 GB of text. It cannot be a Python list.

So this module is the other kind of corpus: **streamed in, screened, tokenized once, and written
to disk as ``uint16`` shards** that training then memory-maps. Nothing is ever fully resident.

**Four domains, mixed per batch — not per phase.**

    general 0.40 · code 0.25 · math 0.20 · conversation 0.15

The weights are config-driven, and the mixing happens *inside* each batch. That detail matters
more than it looks: training general-then-code-then-math is the obvious implementation and it is
also how you get catastrophic forgetting — by the time the model reaches math it has begun
losing prose. Interleaving costs nothing and removes the problem.

**A source ladder, each rung optional.** In priority order per domain:

1. **The Master's own files** — the existing :mod:`~nyxara.growth.dataset` store. He chose it, so
   it outranks everything, exactly as ``collect_corpus`` already treats it.
2. **Streamed public datasets** — via the optional ``datasets`` package. Names live in config,
   never hard-coded here, because dataset gating and licensing change under you.
3. **Her own verified synthetic data** — :mod:`~nyxara.growth.synth_data`, which is always
   available and needs no network at all.
4. **Her own lived corpus** — flywheel, distillation, screened web text.

Missing any rung degrades to the rest rather than failing, so a bare offline machine still
builds a real (smaller) corpus.

**Five screens at the door.** Three are reused rather than reimplemented — the content hash from
:mod:`~nyxara.growth.dataset`, its L-SOVEREIGN loyalty check, and
:class:`~nyxara.senses.web.InjectionScanner`. Two are new because they only start to matter at
this scale:

* **Quality** — length, symbol ratio, repetition. Web text at 6B tokens contains a great deal of
  navigation furniture and base64.
* **Contamination** — n-gram overlap against the shipped evaluation sets. Nothing in the repo
  checks this today, and with billions of scraped tokens it is not a hypothetical: an eval that
  the training set has memorised measures nothing, and it fails *silently* and *upward*, which
  is the worst possible direction for a number people then trust.

Depends on ``growth/dataset``, ``growth/tokenizer``, ``knowledge/ingest`` and ``senses/web``.
``numpy`` is required for the memory-mapped reader and optional for everything else. Nothing
imports back.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001 — the writer degrades; only the memmap reader truly needs it
    _HAS_NUMPY = False

__all__ = [
    "DOMAINS",
    "DEFAULT_DOMAIN_WEIGHTS",
    "DEFAULT_STREAM_SOURCES",
    "QualityFilter",
    "ContaminationFilter",
    "ShardWriter",
    "ShardIndex",
    "ShardDataset",
    "CorpusReport",
    "build_corpus",
]

DOMAINS: Tuple[str, ...] = ("general", "code", "math", "conversation", "causal")

# `causal` is the fifth domain: interventional and counterfactual reasoning generated from
# structural models whose mechanisms are known, so every answer is ground truth rather than an
# estimate. It is small on purpose — 5% is enough to teach that P(Y|do(X)) and P(Y|X) are
# different quantities, and the budget for that comes out of the other four rather than on top.
DEFAULT_DOMAIN_WEIGHTS: Dict[str, float] = {
    "general": 0.38,
    "code": 0.24,
    "math": 0.19,
    "conversation": 0.14,
    "causal": 0.05,
}

# What share of each domain's token budget comes from verified synthetic generation rather than
# streamed text. `general` is ~zero on purpose: synthetic data makes her reliable at procedures
# that can be checked, and cannot make her knowledgeable about the world — a fabricated general
# corpus would teach fluent nonsense. `causal` is 1.0 because there is no streamed source for it.
DEFAULT_SYNTHETIC_SHARE: Dict[str, float] = {
    "general": 0.0,
    "code": 0.20,
    "math": 0.45,
    "conversation": 0.35,
    "causal": 1.0,
}

# Streamed sources, per domain. Deliberately DATA, not code: dataset names, configs and gating
# change often enough that hard-coding them here would rot. Every one is optional — a name that
# cannot be resolved is a note in the report, never a failure.
#
# Several sources per domain, interleaved by `weight` rather than drained in order. A gated or
# renamed source degrades to a note and the rest carry the domain — `bigcode/the-stack-smol` is
# gated behind an access request, so a single-source code domain would silently arrive empty.
DEFAULT_STREAM_SOURCES: Dict[str, List[Dict[str, Any]]] = {
    "general": [
        {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text_key": "text",
         "weight": 3.0, "license": "ODC-By-1.0"},
        {"path": "HuggingFaceTB/cosmopedia-v2", "text_key": "text", "weight": 1.0,
         "license": "Apache-2.0"},
    ],
    "code": [
        {"path": "bigcode/the-stack-smol", "text_key": "content", "weight": 2.0,
         "license": "per-file; developer opt-out applies"},
        {"path": "nampdn-ai/tiny-codes", "text_key": "response", "weight": 1.0,
         "license": "Apache-2.0"},
    ],
    "math": [
        {"path": "open-web-math/open-web-math", "text_key": "text", "weight": 2.0,
         "license": "ODC-By-1.0"},
        {"path": "HuggingFaceTB/finemath", "name": "finemath-4plus", "text_key": "text",
         "weight": 1.0, "license": "ODC-By-1.0"},
    ],
    "conversation": [
        {"path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "text_key": "messages",
         "weight": 2.0, "license": "MIT"},
        {"path": "Open-Orca/SlimOrca", "text_key": "conversations", "weight": 1.0,
         "license": "MIT"},
    ],
}

# uint16 shards: 65,536 ids. Every profile here tops out at 32,768, so this is half the range and
# half the disk of uint32. Guarded at write time rather than assumed.
_SHARD_DTYPE = "uint16"
_MAX_ID = 65_535
# 25M tokens ≈ 50 MB per shard. The previous 100M buffered a hundred million *Python ints* per
# domain — ~800 MB each, ~3.2 GB across four — which is the OOM on a 16 GB box. The buffer is an
# array("H") now, so a shard costs 50 MB of RAM rather than 800, and a crash loses at most 50 MB
# of tokenization instead of 200.
_SHARD_TOKENS = 25_000_000

_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Screens that only start to matter at scale
# --------------------------------------------------------------------------- #
class QualityFilter:
    """Heuristic quality screen for streamed text.

    Deliberately crude and fast — it runs on every document of a multi-billion-token stream, so
    anything model-based is out of the question. It is aimed at the specific junk that dominates
    scraped text: navigation chrome, base64 blobs, link farms and single-word-repeated spam.

    ``code`` relaxes the symbol and line-length rules, because source code legitimately looks
    like junk to a prose filter — punishing it here would quietly gut the code domain.
    """

    def __init__(self, *, min_chars: int = 200, max_symbol_ratio: float = 0.30,
                 max_repeat_ratio: float = 0.30, min_unique_words: int = 12) -> None:
        self.min_chars = min_chars
        self.max_symbol_ratio = max_symbol_ratio
        self.max_repeat_ratio = max_repeat_ratio
        self.min_unique_words = min_unique_words

    def check(self, text: str, *, domain: str = "general") -> Tuple[bool, str]:
        """``(keep, reason)`` — ``reason`` names the screen that rejected it."""
        if not text or len(text) < self.min_chars:
            return False, "too_short"

        words = text.split()
        if not words:
            return False, "no_words"

        if domain != "code":
            alpha = sum(c.isalpha() or c.isspace() for c in text)
            if 1.0 - (alpha / len(text)) > self.max_symbol_ratio:
                return False, "symbol_heavy"
            if len(set(words)) < self.min_unique_words:
                return False, "low_diversity"

        # One token repeated over and over — the signature of a link farm or a broken scrape.
        # Counter, not `words.count(w)` in a comprehension: that was O(200 · len(words)) per
        # document, and this screen runs on every document of a multi-billion-token stream.
        window = words[:2000]
        most_common = max(Counter(window).values(), default=0)
        if len(window) >= 20 and most_common / len(window) > self.max_repeat_ratio:
            return False, "repetitive"

        return True, ""


class ContaminationFilter:
    """Rejects training text that overlaps the evaluation sets.

    Builds a set of word n-gram hashes from every shipped eval prompt/answer, and drops any
    document sharing one. That is a blunt instrument and it is the right one here: the failure
    it prevents is silent and one-directional. A contaminated corpus makes every downstream
    number — the promotion gate, the domain scores, the whole claim of "good at math" — look
    *better* than the model is, and nothing in the system would notice.

    Empty (a no-op) when the eval sets cannot be loaded, with a note rather than an exception:
    the corpus build must not die because a benchmark file moved.
    """

    def __init__(self, *, n: int = 13, stride: int = 1) -> None:
        self.n = max(4, int(n))
        # Index every `stride`-th n-gram. With n=13 and stride=4, any overlap of 16+ consecutive
        # words is still certain to be caught, at a quarter of the hashing cost.
        self.stride = max(1, int(stride))
        self._hashes: Set[int] = set()
        self.loaded = False

    @staticmethod
    def _hash_gram(gram: str) -> int:
        """blake2b, not the builtin ``hash()``.

        ``hash(str)`` is salted per process by PYTHONHASHSEED, so an index built in the parent
        means nothing in a ``spawn``ed worker and cannot be cached to disk between runs. That
        makes the whole screen unreproducible *and* blocks the multiprocessing build — the
        contamination index has to survive crossing a process boundary.
        """
        return int.from_bytes(hashlib.blake2b(gram.encode("utf-8", "replace"),
                                              digest_size=8).digest(), "big")

    def _ngrams(self, text: str, *, stride: Optional[int] = None) -> Iterator[int]:
        step = self.stride if stride is None else max(1, int(stride))
        words = _WS_RE.sub(" ", (text or "").lower()).strip().split()
        for i in range(0, max(0, len(words) - self.n + 1), step):
            yield self._hash_gram(" ".join(words[i:i + self.n]))

    def save(self, path: Any) -> Path:
        """Persist the index so a multi-hour build need not rebuild it, and workers can load it."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = array.array("Q", sorted(self._hashes))
        out.write_bytes(blob.tobytes())
        return out

    @classmethod
    def load(cls, path: Any, *, n: int = 13, stride: int = 1) -> "ContaminationFilter":
        f = cls(n=n, stride=stride)
        raw = Path(path).read_bytes()
        blob = array.array("Q")
        blob.frombytes(raw)
        f._hashes = set(blob)
        f.loaded = bool(f._hashes)
        return f

    def load_eval_sets(self, settings: Any = None) -> "ContaminationFilter":
        """Index every shipped benchmark so training text overlapping them can be dropped."""
        texts: List[str] = []
        for loader in (self._realworld_texts, self._hard_benchmark_texts):
            try:
                texts.extend(loader(settings))
            except Exception:  # noqa: BLE001 — a missing battery is a smaller index, not a crash
                continue
        for text in texts:
            self._hashes.update(self._ngrams(text))
        self.loaded = bool(self._hashes)
        return self

    @staticmethod
    def _realworld_texts(settings: Any) -> List[str]:
        from nyxara.eval.datasets import build_realworld_benchmark
        bench = build_realworld_benchmark(settings)
        return [f"{t.prompt} {t.answer}" for t in bench.tasks()]

    @staticmethod
    def _hard_benchmark_texts(_settings: Any) -> List[str]:
        from nyxara.eval.hard_benchmark import build_code_benchmark, build_math_benchmark
        out: List[str] = []
        for bench in (build_math_benchmark(), build_code_benchmark()):
            out.extend(f"{t.prompt} {t.answer}" for t in bench.tasks())
        return out

    def is_contaminated(self, text: str) -> bool:
        """``True`` when ``text`` shares an indexed n-gram with a shipped eval set.

        The query side always walks **every** n-gram (stride 1) even when the index was built
        with a stride. That asymmetry is what makes striding safe rather than lossy: with n=13
        and stride 4, any 16-word overlap contains four consecutive eval n-gram start positions,
        one of which is necessarily ≡ 0 (mod 4) and therefore indexed. Striding the query too
        would let an overlap slip through on an unlucky alignment.
        """
        if not self._hashes:
            return False
        return any(h in self._hashes for h in self._ngrams(text, stride=1))


# --------------------------------------------------------------------------- #
# Shards
# --------------------------------------------------------------------------- #
@dataclass
class ShardIndex:
    """The manifest describing a built corpus — what is on disk, per domain."""

    shards: List[Dict[str, Any]] = field(default_factory=list)
    vocab_sig: str = ""
    vocab_size: int = 0
    created_at: float = field(default_factory=time.time)

    # ``.get("split", "train")`` throughout: every index written before splits existed is a
    # train-only index, and must keep loading unchanged.
    @staticmethod
    def _split_of(shard: Dict[str, Any]) -> str:
        return str(shard.get("split", "train"))

    def domains(self, split: Optional[str] = "train") -> List[str]:
        return sorted({s["domain"] for s in self.shards
                       if split is None or self._split_of(s) == split})

    def splits(self) -> List[str]:
        return sorted({self._split_of(s) for s in self.shards})

    def tokens(self, domain: Optional[str] = None, *, split: Optional[str] = None) -> int:
        return sum(s["n_tokens"] for s in self.shards
                   if (domain is None or s["domain"] == domain)
                   and (split is None or self._split_of(s) == split))

    def for_domain(self, domain: str, *, split: Optional[str] = "train") -> List[Dict[str, Any]]:
        return [s for s in self.shards if s["domain"] == domain
                and (split is None or self._split_of(s) == split)]

    def to_dict(self) -> Dict[str, Any]:
        return {"shards": list(self.shards), "vocab_sig": self.vocab_sig,
                "vocab_size": self.vocab_size, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShardIndex":
        return cls(shards=list(d.get("shards") or []), vocab_sig=d.get("vocab_sig", ""),
                   vocab_size=int(d.get("vocab_size", 0)),
                   created_at=float(d.get("created_at", 0.0)))

    def save(self, directory: Any) -> Path:
        """Write ``index.json`` atomically — tmp file, then ``os.replace``.

        A plain ``write_text`` that dies midway leaves a truncated index, which is worse than no
        index at all: it makes a resumable build look resumable while pointing at nothing.
        """
        path = Path(directory) / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, directory: Any) -> Optional["ShardIndex"]:
        path = Path(directory) / "index.json"
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class ShardWriter:
    """Appends token ids to per-domain ``uint16`` shard files.

    Resumable by construction: a completed shard is a finished file listed in ``index.json``, so
    a build that dies at 80% keeps the 80%. A 6B-token tokenization is a multi-hour job and
    losing it to one crash is not acceptable.
    """

    def __init__(self, directory: Any, *, shard_tokens: int = _SHARD_TOKENS) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = max(1024, int(shard_tokens))
        # array("H") — two bytes per token, not a 28-byte Python int plus an 8-byte pointer.
        self._buffers: Dict[Tuple[str, str], array.array] = {}
        self._counts: Dict[str, int] = {}
        self.index = ShardIndex.load(self.dir) or ShardIndex()
        self.orphans_removed = 0
        self.reconcile()

    def reconcile(self) -> int:
        """Delete ``.bin`` files on disk that no index entry claims, and drop entries with no file.

        A build killed between writing a shard and writing the index leaves an orphan. Without
        this, the shard counter restarts from the index's (shorter) length and the next flush
        **overwrites** a shard that already holds real tokens — the corpus then silently contains
        one shard's worth of duplicated text and is short by another.
        """
        known = {s["path"] for s in self.index.shards}
        for path in sorted(self.dir.glob("*.bin")):
            if path.name not in known:
                path.unlink()
                self.orphans_removed += 1
        kept = [s for s in self.index.shards if (self.dir / s["path"]).exists()]
        if len(kept) != len(self.index.shards):
            self.index.shards = kept
        # Resume the per-domain counter past the highest index actually on disk, so a new shard
        # can never collide with a survivor even if numbering has gaps.
        for shard in self.index.shards:
            stem = Path(shard["path"]).stem
            try:
                n = int(stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            self._counts[shard["domain"]] = max(self._counts.get(shard["domain"], 0), n + 1)
        return self.orphans_removed

    def existing_tokens(self, domain: str, *, split: str = "train") -> int:
        return self.index.tokens(domain, split=split)

    def add(self, domain: str, ids: Sequence[int], *, split: str = "train") -> None:
        if len(ids) == 0:
            return
        # Validate on the way IN, not at flush time. The buffer holds up to a shard's worth of
        # tokens, so deferring this means a vocabulary that overflows uint16 is only discovered
        # after millions of tokens have been accepted — and the ids would silently wrap to
        # different tokens if the check were missed entirely.
        peak = max(ids)
        if peak > _MAX_ID:
            raise ValueError(
                f"token id {peak} exceeds the uint16 shard range ({_MAX_ID}); "
                f"a vocabulary this large needs a wider shard dtype")
        if min(ids) < 0:
            raise ValueError("negative token id cannot be stored in a uint16 shard")
        buf = self._buffers.setdefault((domain, split), array.array("H"))
        buf.extend(int(i) for i in ids)
        while len(buf) >= self.shard_tokens:
            self._flush(domain, buf[:self.shard_tokens], split=split)
            del buf[:self.shard_tokens]

    def add_bytes(self, domain: str, blob: bytes, *, split: str = "train") -> None:
        """Append already-packed little-endian ``uint16`` bytes — the multiprocessing path.

        Workers pack their own tokens, so the parent never rebuilds a Python list from them.
        """
        if not blob:
            return
        buf = self._buffers.setdefault((domain, split), array.array("H"))
        buf.frombytes(blob)
        while len(buf) >= self.shard_tokens:
            self._flush(domain, buf[:self.shard_tokens], split=split)
            del buf[:self.shard_tokens]

    def _flush(self, domain: str, ids: Sequence[int], *, split: str = "train") -> None:
        if len(ids) == 0:
            return
        n = self._counts.get(domain, 0)
        self._counts[domain] = n + 1
        prefix = "" if split == "train" else f"{split}-"
        path = self.dir / f"{prefix}{domain}-{n:05d}.bin"
        blob = ids if isinstance(ids, array.array) else array.array("H", ids)
        with path.open("wb") as fh:
            blob.tofile(fh)
        self.index.shards.append({"domain": domain, "path": path.name,
                                  "n_tokens": len(ids), "dtype": _SHARD_DTYPE,
                                  "split": split})
        # Persist NOW, not at close(). A finished shard must be a finished file that the index
        # already knows about, or "resumable by construction" is only true for clean exits — and
        # a multi-hour build is precisely the one that does not get a clean exit.
        self.index.save(self.dir)

    def close(self, *, vocab_sig: str = "", vocab_size: int = 0) -> ShardIndex:
        """Flush every partial shard and write the index."""
        for (domain, split), buf in list(self._buffers.items()):
            if len(buf):
                self._flush(domain, buf, split=split)
                del buf[:]
        self.index.vocab_sig = vocab_sig or self.index.vocab_sig
        self.index.vocab_size = vocab_size or self.index.vocab_size
        self.index.save(self.dir)
        return self.index


class ShardDataset:
    """Memory-mapped reader that mixes domains **within** every batch.

    Requires numpy. Holds no token data in Python: ``np.memmap`` lets the OS page in only the
    windows actually touched, so a 24 GB corpus costs no measurable RAM.

    :meth:`state_dict` / :meth:`load_state_dict` carry the sampling position, so a resumed run
    continues through the data instead of silently restarting the epoch — which would quietly
    over-train the first shard and undertrain the rest.
    """

    def __init__(self, directory: Any, *, block_size: int = 1024,
                 weights: Optional[Dict[str, float]] = None, seed: int = 0,
                 split: str = "train") -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("ShardDataset requires numpy")
        self.dir = Path(directory)
        index = ShardIndex.load(self.dir)
        if index is None or not index.shards:
            raise FileNotFoundError(f"no shard index under {self.dir} — build the corpus first")
        self.index = index
        self.split = str(split)
        self.block_size = max(8, int(block_size))
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._draws = 0

        self._maps: Dict[str, List[Any]] = {}
        for shard in index.shards:
            if ShardIndex._split_of(shard) != self.split:
                continue
            arr = np.memmap(self.dir / shard["path"], dtype=shard.get("dtype", _SHARD_DTYPE),
                            mode="r")
            self._maps.setdefault(shard["domain"], []).append(arr)
        if not self._maps:
            raise FileNotFoundError(
                f"no '{self.split}' shards under {self.dir} "
                f"(splits present: {', '.join(index.splits()) or 'none'})")

        # Cumulative lengths per domain, so a shard is chosen in proportion to how many tokens it
        # holds. Choosing uniformly gives a 5M-token tail shard the same weight as a 25M one —
        # a 5x oversample of whatever happens to be at the end of the corpus.
        self._cum: Dict[str, Any] = {
            domain: np.cumsum([len(a) for a in arrays], dtype=np.int64)
            for domain, arrays in self._maps.items()
        }

        requested = dict(weights or DEFAULT_DOMAIN_WEIGHTS)
        # Renormalise over the domains that actually have data. A weight for a domain that
        # produced no shards must be redistributed, not silently sampled as empty.
        present = {d: w for d, w in requested.items() if self._maps.get(d) and w > 0}
        total = sum(present.values())
        self.weights = ({d: w / total for d, w in present.items()} if total > 0
                        else {d: 1.0 / len(self._maps) for d in self._maps})

    @property
    def domains(self) -> List[str]:
        return sorted(self._maps)

    def total_tokens(self) -> int:
        return self.index.tokens(split=self.split)

    def _pick_shard(self, domain: str) -> int:
        """Choose a shard in proportion to its token count, not uniformly."""
        cum = self._cum[domain]
        total = int(cum[-1])
        if total <= 0:
            return self._rng.randrange(len(self._maps[domain]))
        return int(np.searchsorted(cum, self._rng.randrange(total), side="right"))

    def _pick_domain(self) -> str:
        r = self._rng.random()
        acc = 0.0
        for domain, weight in sorted(self.weights.items()):
            acc += weight
            if r <= acc:
                return domain
        return sorted(self.weights)[-1]

    def sample(self, domain: Optional[str] = None) -> Tuple[Any, Any]:
        """One ``(x, y)`` training window as numpy int64 arrays, next-token shifted."""
        domain = domain or self._pick_domain()
        arrays = self._maps[domain]
        arr = arrays[self._pick_shard(domain)]
        need = self.block_size + 1
        if len(arr) <= need:
            # A shard shorter than one window — which happens whenever a domain is
            # underrepresented, and a real corpus always has one. Tile until the window is full
            # rather than wrapping once: a single wrap still leaves the array short when the
            # shard is less than half a window, and np.stack then fails on ragged rows only
            # once that domain is first sampled, deep into a run.
            ids = np.asarray(arr, dtype=np.int64)
            if len(ids) == 0:
                ids = np.zeros(need, dtype=np.int64)
            elif len(ids) < need:
                reps = -(-need // len(ids))          # ceiling division
                ids = np.tile(ids, reps)[:need]
        else:
            start = self._rng.randrange(len(arr) - self.block_size - 1)
            ids = np.asarray(arr[start:start + self.block_size + 1], dtype=np.int64)
        self._draws += 1
        return ids[:-1], ids[1:]

    def batch(self, batch_size: int, domain: Optional[str] = None) -> Tuple[Any, Any]:
        """A batch whose rows are drawn from the domain mix — interleaved, not phased."""
        xs, ys = [], []
        for _ in range(max(1, batch_size)):
            x, y = self.sample(domain)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    def state_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "draws": self._draws,
                "rng": list(self._rng.getstate()[1]), "pos": self._rng.getstate()[2]}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        try:
            self._rng.setstate((3, tuple(state["rng"]), state["pos"]))
            self._draws = int(state.get("draws", 0))
        except Exception:  # noqa: BLE001 — an unreadable cursor restarts sampling, never crashes
            self._rng = random.Random(self.seed)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
@dataclass
class CorpusReport:
    """A countable account of one corpus build — what she took and what each screen refused."""

    tokens: Dict[str, int] = field(default_factory=dict)
    documents: Dict[str, int] = field(default_factory=dict)
    sources: Dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    near_duplicates: int = 0
    wrong_language: int = 0
    low_quality: int = 0
    contaminated: int = 0
    val_tokens: int = 0
    quarantined: int = 0        # L-SOVEREIGN — identity-hostile, never trained on
    injection_flagged: int = 0
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens.values())

    def summary(self) -> str:
        head = f"{self.total_tokens:,} tokens"
        mix = ", ".join(f"{d} {self.tokens.get(d, 0):,}" for d in DOMAINS
                        if self.tokens.get(d))
        screens = []
        for label, n in (("duplicate", self.duplicates),
                         ("near-duplicate", self.near_duplicates),
                         ("wrong-language", self.wrong_language),
                         ("low-quality", self.low_quality),
                         ("CONTAMINATED", self.contaminated),
                         ("QUARANTINED", self.quarantined),
                         ("injection-flagged", self.injection_flagged)):
            if n:
                screens.append(f"{n:,} {label}")
        tail = f"\n  screens: {', '.join(screens)}" if screens else ""
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"· {head} [{mix}]{tail}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"tokens": dict(self.tokens), "documents": dict(self.documents),
                "sources": dict(self.sources), "duplicates": self.duplicates,
                "near_duplicates": self.near_duplicates, "val_tokens": self.val_tokens,
                "wrong_language": self.wrong_language,
                "low_quality": self.low_quality, "contaminated": self.contaminated,
                "quarantined": self.quarantined,
                "injection_flagged": self.injection_flagged, "notes": list(self.notes)}


class CorpusBuilder:
    """Streams documents from every available source, screens them, and writes token shards."""

    def __init__(self, *, tokenizer: Any, out_dir: Any, settings: Any = None,
                 weights: Optional[Dict[str, float]] = None,
                 tokens_budget: int = 6_000_000_000,
                 stream_sources: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 synthetic_share: Optional[Dict[str, float]] = None,
                 quality: Optional[QualityFilter] = None,
                 contamination: Optional[ContaminationFilter] = None,
                 val_permille: int = 5, dedup_bits: int = 22,
                 shard_tokens: int = _SHARD_TOKENS, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.out_dir = Path(out_dir)
        self.settings = settings
        # `is not None`, never `or`. An empty dict is falsy, so `stream_sources or DEFAULT`
        # silently restored the default source list when a caller passed `{}` to mean "no
        # streaming at all" — which made `--no-stream` reach the network anyway and made the
        # test suite depend on HuggingFace being up. "Explicitly empty" and "not specified" are
        # different requests and must stay different.
        self.weights = dict(weights if weights is not None else DEFAULT_DOMAIN_WEIGHTS)
        self.synthetic_share = dict(
            synthetic_share if synthetic_share is not None else DEFAULT_SYNTHETIC_SHARE)
        self.tokens_budget = max(1, int(tokens_budget))
        self.stream_sources = dict(
            stream_sources if stream_sources is not None else DEFAULT_STREAM_SOURCES)
        self.quality = quality or QualityFilter()
        self.contamination = contamination
        self.val_permille = max(0, int(val_permille))
        self.shard_tokens = shard_tokens
        self.seed = seed
        self._scanner: Any = None
        # Bounded-memory screens from growth/screen. The `Set[str]` these replace held 64-char
        # hex digests and grew without limit — 3-5 GB at 6B tokens, and gone on restart.
        try:
            from nyxara.growth.screen import ExactDedup, MinHashLSH

            self._exact = ExactDedup(capacity_bits=max(16, int(dedup_bits) + 3))
            self._near = MinHashLSH(capacity_bits=max(14, int(dedup_bits)))
        except Exception:  # noqa: BLE001 — no numpy: fall back to the unbounded set
            self._exact = None
            self._near = None
        self._seen: Set[str] = set()

    # ---- screens (reused, not reimplemented) ---- #
    def _screen(self, text: str, report: CorpusReport, *, domain: str,
                trusted: bool) -> bool:
        """Run all five screens. ``True`` keeps the document."""
        from nyxara.growth.dataset import _content_hash, _loyalty_hostility

        if self._exact is not None:
            from nyxara.growth.screen import fingerprint

            if not self._exact.add_if_new(fingerprint(text)):
                report.duplicates += 1
                return False
        else:
            sha = _content_hash(text)
            if sha in self._seen:
                report.duplicates += 1
                return False
            self._seen.add(sha)

        severity, _pattern = _loyalty_hostility(text)
        if severity > 0.0:
            report.quarantined += 1
            return False

        # The quality screen exists for SCRAPED text — it looks for navigation chrome, base64
        # and link farms. Trusted sources (the Master's files, her own verified synthetic data)
        # must skip it, and this is not a nicety: a correct 120-character verified code example
        # trips `min_chars=200` and is thrown away as "too_short". Running the screen over
        # trusted data silently deletes the entire code domain while reporting success.
        if not trusted:
            keep, _reason = self.quality.check(text, domain=domain)
            if not keep:
                report.low_quality += 1
                return False
            # Near-duplicates, on scraped text only. Her own generated data is distinct by
            # construction, and MinHash over it would be pure cost. Web text is the opposite:
            # heavily boilerplate-republished, and an exact hash catches none of it.
            if self._near is not None and self._near.seen_or_add(text):
                report.near_duplicates += 1
                return False
            if domain == "general" and not _language_ok(text):
                report.wrong_language += 1
                return False

        if self.contamination is not None and self.contamination.is_contaminated(text):
            report.contaminated += 1
            return False

        # Likewise the injection screen: the Master's own files get the "warn" posture
        # growth/dataset.py already takes (counted, kept). Streamed web text is untrusted and a
        # hit is dropped.
        if not trusted and self._injection_score(text) >= 0.5:
            report.injection_flagged += 1
            return False

        return True

    def _injection_score(self, text: str) -> float:
        if self._scanner is None:
            try:
                from nyxara.senses.web import InjectionScanner
                self._scanner = InjectionScanner()
            except Exception:  # noqa: BLE001
                self._scanner = False
        if not self._scanner:
            return 0.0
        try:
            return float(self._scanner.scan(text).score)
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- sources ---- #
    def _master_docs(self, report: CorpusReport) -> Iterator[Tuple[str, str, bool]]:
        """The Master's ingested files — highest priority, and trusted."""
        try:
            from nyxara.growth.dataset import _default_store_path, load_dataset_docs
            docs = load_dataset_docs(_default_store_path(self.settings), limit=None)
        except Exception as exc:  # noqa: BLE001
            report.note(f"master dataset unavailable ({exc})")
            return
        report.sources["master"] = len(docs)
        for doc in docs:
            yield doc, "general", True

    def _synthetic_docs(self, report: CorpusReport, budget: Dict[str, int]
                        ) -> Iterator[Tuple[str, str, bool]]:
        """Verified synthetic data, sized as a **share of each domain's token budget**.

        This used to take a fixed document count — ``max(100, min(20_000, budget // 400_000))``
        — which at a 6B budget is ~15k documents per domain. Measured, those average ~40 tokens,
        so the one source that is correct *by construction* contributed **1.8M tokens against
        6B: 0.03% of the corpus**. It was an incidental sprinkle rather than a designed
        fraction, and no amount of improving the generators would have changed that.

        Now each domain gets ``DEFAULT_SYNTHETIC_SHARE`` of its own budget. Documents are
        requested in batches sized from the running average, so the target is met without
        generating far past it.
        """
        try:
            from nyxara.growth.synth_data import generate_domain_docs
        except Exception as exc:  # noqa: BLE001
            report.note(f"synthetic generators unavailable ({exc})")
            return

        n = 0
        for domain in ("math", "code", "conversation", "causal"):
            share = self.synthetic_share.get(domain, 0.0)
            target = int(budget.get(domain, 0) * share)
            if target <= 0:
                continue
            produced_tokens = 0
            # Estimate documents from a per-domain default, then correct from what we actually
            # see. Asking for a token count and getting documents needs one round of feedback.
            mean_tokens = 120.0
            attempts = 0
            while produced_tokens < target and attempts < 12:
                attempts += 1
                want = max(64, int((target - produced_tokens) / max(1.0, mean_tokens) * 1.1))
                want = min(want, 400_000)
                emitted = 0
                for doc in generate_domain_docs(domain, want, seed=self.seed + attempts):
                    emitted += 1
                    n += 1
                    produced_tokens += max(1, len(doc) // 4)   # cheap proxy; exact count is
                    yield doc, domain, True                    # taken by the caller on write
                    if produced_tokens >= target:
                        break
                if emitted == 0:
                    report.note(f"{domain}: synthetic generators produced nothing further "
                                f"({produced_tokens:,}/{target:,} tokens of the target)")
                    break
                mean_tokens = max(20.0, produced_tokens / max(1, n))
        report.sources["synthetic"] = n

    def _own_docs(self, report: CorpusReport) -> Iterator[Tuple[str, str, bool]]:
        """Her own lived corpus — flywheel, distillation, screened web text."""
        n = 0
        try:
            root = Path(getattr(getattr(self.settings, "llm", None), "self_model_dir", None)
                        or (self.out_dir.parent / "foundry"))
            from nyxara.growth.distill import load_distillation_docs
            for name, domain in (("flywheel.jsonl", "conversation"),
                                 ("distill.jsonl", "conversation")):
                path = root / name
                if path.exists():
                    for doc in load_distillation_docs(path, limit=None):
                        n += 1
                        yield doc, domain, True
        except Exception as exc:  # noqa: BLE001
            report.note(f"own corpus unavailable ({exc})")
        try:
            from nyxara.growth.acquire import load_acquired_docs
            root = Path(getattr(getattr(self.settings, "llm", None), "self_model_dir", None)
                        or (self.out_dir.parent / "foundry"))
            path = root / "acquired.jsonl"
            if path.exists():
                for doc in load_acquired_docs(path, limit=None):
                    n += 1
                    yield doc, "general", False
        except Exception:  # noqa: BLE001
            pass
        report.sources["own"] = n

    def _streamed_docs(self, report: CorpusReport, budget: Dict[str, int]
                       ) -> Iterator[Tuple[str, str, bool]]:
        """Streamed public datasets. Requires the optional ``datasets`` package."""
        try:
            from datasets import load_dataset          # type: ignore
        except Exception:
            report.note("the `datasets` package is not installed — streamed sources skipped; "
                        "the corpus is built from the Master's files, her own data and the "
                        "synthetic generators only")
            return

        # Open every source first, then round-robin across them.
        #
        # The previous loop drained each source before touching the next: FineWeb-Edu's
        # sample-10BT would stream ~10B tokens' worth of rows — discarding ~7.6B of them at the
        # caller's `if budget <= 0: continue` — before maths, code or conversation were reached
        # at all. Its break condition also tested only the GLOBAL budget, so a domain whose own
        # budget was long spent kept streaming into a discard. Interleaving fixes both: a
        # finished source is closed and dropped, and a finished domain stops being scheduled.
        cursors: List[Dict[str, Any]] = []
        for domain, specs in self.stream_sources.items():
            if budget.get(domain, 0) <= 0:
                continue
            weights = [float(s.get("weight", 1.0)) for s in specs] or [1.0]
            total = sum(weights) or 1.0
            for spec, weight in zip(specs, weights):
                try:
                    stream = load_dataset(spec["path"], spec.get("name"),
                                          split=spec.get("split", "train"), streaming=True)
                except Exception as exc:  # noqa: BLE001 — gated/renamed/offline: note and move on
                    report.note(f"{domain}: {spec['path']} unavailable ({exc})")
                    continue
                cursors.append({
                    "domain": domain, "id": f"{domain}:{spec['path']}",
                    "iter": iter(stream), "key": spec.get("text_key", "text"),
                    # Each source gets its own slice of the domain budget, so one source cannot
                    # consume the share meant for the others.
                    "quota": max(1, int(budget.get(domain, 0) * weight / total)),
                    "taken": 0,
                })
        if not cursors:
            report.sources["streamed"] = 0
            return

        per_visit = 32          # amortise per-source setup without letting one source run away
        try:
            yield from self._drain(cursors, budget, report, per_visit)
        finally:
            # Close every iterator we are walking away from, while the interpreter is still
            # alive. `datasets` streaming runs background prefetch threads; abandoning the
            # generator leaves them holding GIL state into interpreter finalization, and the
            # process then dies with "PyGILState_Release: thread state must be current" AFTER
            # a successful build — a corpus that is fine on disk and a run that looks crashed.
            for cursor in cursors:
                try:
                    cursor["iter"].close()
                except Exception:  # noqa: BLE001 — best effort; we are already leaving
                    pass
            report.sources["streamed"] = report.sources.get("streamed", 0)

    def _drain(self, cursors: List[Dict[str, Any]], budget: Dict[str, int],
               report: CorpusReport, per_visit: int) -> Iterator[Tuple[str, str, bool]]:
        """Round-robin across open sources until every one is finished or out of budget."""
        n = 0
        while cursors:
            for cursor in list(cursors):
                domain = cursor["domain"]
                if budget.get(domain, 0) <= 0 or cursor["taken"] >= cursor["quota"]:
                    self._close(cursor, cursors)    # this domain or source is finished
                    continue
                emitted = 0
                while emitted < per_visit:
                    try:
                        row = next(cursor["iter"])
                    except StopIteration:
                        self._close(cursor, cursors)
                        report.note(f"{cursor['id']} exhausted after {cursor['taken']:,} rows")
                        break
                    except Exception as exc:  # noqa: BLE001 — a mid-stream fault ends that source
                        self._close(cursor, cursors)
                        report.note(f"{cursor['id']} failed mid-stream ({exc})")
                        break
                    text = _row_text(row, cursor["key"])
                    if not text:
                        continue
                    emitted += 1
                    n += 1
                    cursor["taken"] += max(1, len(text) // 4)
                    report.sources[cursor["id"]] = report.sources.get(cursor["id"], 0) + 1
                    yield text, domain, False
        report.sources["streamed"] = n

    @staticmethod
    def _close(cursor: Dict[str, Any], cursors: List[Dict[str, Any]]) -> None:
        """Drop a source AND shut its background prefetch threads down."""
        if cursor in cursors:
            cursors.remove(cursor)
        try:
            cursor["iter"].close()
        except Exception:  # noqa: BLE001
            pass

    def _split_for(self, text: str) -> str:
        """Deterministic train/val assignment keyed on the document's own content."""
        if self.val_permille <= 0:
            return "train"
        try:
            from nyxara.growth.screen import fingerprint, split_for

            return split_for(fingerprint(text), val_permille=self.val_permille)
        except Exception:  # noqa: BLE001 — no screen module: everything is training data
            return "train"

    @staticmethod
    def _budget_spent(budget: Dict[str, int]) -> bool:
        return all(v <= 0 for v in budget.values())

    # ---- the build ---- #
    def build(self, *, report: Optional[CorpusReport] = None) -> Tuple[ShardIndex, CorpusReport]:
        """Stream, screen, tokenize and shard. Resumable; never raises for a missing source."""
        report = report if report is not None else CorpusReport()
        writer = ShardWriter(self.out_dir, shard_tokens=self.shard_tokens)

        total = sum(self.weights.values()) or 1.0
        budget = {d: int(self.tokens_budget * w / total) for d, w in self.weights.items()}
        for domain in list(budget):
            budget[domain] = max(0, budget[domain] - writer.existing_tokens(domain))
        if writer.index.shards:
            report.note(f"resuming: {writer.index.tokens():,} tokens already on disk")

        streams = (
            self._master_docs(report),
            self._synthetic_docs(report, dict(budget)),
            self._own_docs(report),
            self._streamed_docs(report, budget),
        )

        for stream in streams:
            for text, domain, trusted in stream:
                if budget.get(domain, 0) <= 0:
                    continue
                if not self._screen(text, report, domain=domain, trusted=trusted):
                    continue
                ids = self.tokenizer.encode(text, add_eos=True)
                if not ids:
                    continue
                # Held out by CONTENT, never by arrival order: a document always lands on the
                # same side however often the build is resumed, reordered or re-sharded. An
                # order-keyed split leaks validation documents into training across a restart,
                # and the resulting val loss looks good for exactly the wrong reason.
                split = self._split_for(text)
                writer.add(domain, ids, split=split)
                if split == "train":
                    budget[domain] -= len(ids)
                    report.tokens[domain] = report.tokens.get(domain, 0) + len(ids)
                    report.documents[domain] = report.documents.get(domain, 0) + 1
                else:
                    report.val_tokens += len(ids)
            if self._budget_spent(budget):
                break

        index = writer.close(vocab_sig=self.tokenizer.vocab_sig(),
                             vocab_size=self.tokenizer.vocab_size)
        for domain in DOMAINS:
            if budget.get(domain, 0) > 0 and report.tokens.get(domain, 0) == 0:
                report.note(f"{domain}: no data found — this domain is ABSENT from the corpus")
        return index, report


def _language_ok(text: str) -> bool:
    """English screen for streamed general text; permissive, and skipped when unavailable."""
    try:
        from nyxara.growth.screen import language_ok

        return language_ok(text)
    except Exception:  # noqa: BLE001 — no screen is better than dropping everything
        return True


def _render_transcript(messages: Sequence[Any]) -> str:
    """Render **every** turn of a chat transcript, in order.

    The previous version looped the list assigning to one ``user`` and one ``assistant``
    variable, so a 10-turn UltraChat conversation arrived as the *last* user message plus the
    *last* assistant message — which are usually not even the same exchange. That silently threw
    away the multi-turn structure, which is the only reason to stream a conversation dataset at
    all: what survived was a corpus of disconnected single turns wearing a conversation's name.
    """
    from nyxara.mind.llm import format_self_training_doc

    parts: List[str] = []
    pending_user: Optional[str] = None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue
        if role in ("user", "human"):
            # Two user turns in a row: keep the later one rather than pairing the wrong halves.
            pending_user = content if pending_user is None else f"{pending_user}\n\n{content}"
        elif role in ("assistant", "ai", "bot"):
            if pending_user:
                parts.append(format_self_training_doc(pending_user, content))
                pending_user = None
        # `system` turns are dropped: her template carries its own system framing, and splicing
        # a foreign one in would teach the model to expect instructions it will never receive.
    return "".join(parts)


def _row_text(row: Any, key: str) -> str:
    """Extract training text from one streamed row, including chat-shaped rows."""
    if not isinstance(row, dict):
        return str(row or "")
    value = row.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list):           # a `messages` transcript — render it in her template
        try:
            return _render_transcript(value)
        except Exception:  # noqa: BLE001
            pass
    for fallback in ("text", "content", "body"):
        if isinstance(row.get(fallback), str):
            return row[fallback]
    return ""


def build_corpus(*, tokenizer: Any, out_dir: Any, settings: Any = None,
                 weights: Optional[Dict[str, float]] = None,
                 tokens_budget: int = 6_000_000_000,
                 stream_sources: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 synthetic_share: Optional[Dict[str, float]] = None,
                 check_contamination: bool = True,
                 val_permille: int = 5,
                 shard_tokens: int = _SHARD_TOKENS,
                 seed: int = 0) -> Tuple[ShardIndex, CorpusReport]:
    """Build a sharded, screened, domain-mixed pretraining corpus. Resumable.

    ``tokens_budget`` defaults to the Chinchilla-optimal 6B for a 300M model. The real corpus
    will usually be smaller than the budget — that is fine and is reported, not hidden.

    ``synthetic_share`` is the fraction of each domain's budget drawn from verified generation
    rather than streamed text; ``val_permille`` holds documents out by content hash, so the
    split survives a resumed build without leaking.
    """
    contamination = (ContaminationFilter().load_eval_sets(settings)
                     if check_contamination else None)
    builder = CorpusBuilder(tokenizer=tokenizer, out_dir=out_dir, settings=settings,
                            weights=weights, tokens_budget=tokens_budget,
                            stream_sources=stream_sources, synthetic_share=synthetic_share,
                            contamination=contamination, val_permille=val_permille,
                            shard_tokens=shard_tokens, seed=seed)
    index, report = builder.build()
    if contamination is not None and not contamination.loaded:
        report.note("contamination screen was EMPTY — eval sets could not be loaded, so "
                    "train/eval overlap was NOT checked")
    return index, report
