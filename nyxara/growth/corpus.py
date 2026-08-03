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

import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

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

DOMAINS: Tuple[str, ...] = ("general", "code", "math", "conversation")

DEFAULT_DOMAIN_WEIGHTS: Dict[str, float] = {
    "general": 0.40,
    "code": 0.25,
    "math": 0.20,
    "conversation": 0.15,
}

# Streamed sources, per domain. Deliberately DATA, not code: dataset names, configs and gating
# change often enough that hard-coding them here would rot. Every one is optional — a name that
# cannot be resolved is a note in the report, never a failure.
DEFAULT_STREAM_SOURCES: Dict[str, List[Dict[str, Any]]] = {
    "general": [{"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text_key": "text"}],
    "code": [{"path": "bigcode/the-stack-smol", "text_key": "content"}],
    "math": [{"path": "open-web-math/open-web-math", "text_key": "text"}],
    "conversation": [{"path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft",
                      "text_key": "messages"}],
}

# uint16 shards: 65,536 ids. Every profile here tops out at 32,768, so this is half the range and
# half the disk of uint32. Guarded at write time rather than assumed.
_SHARD_DTYPE = "uint16"
_MAX_ID = 65_535
_SHARD_TOKENS = 100_000_000        # ~200 MB per shard file

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
        most_common = max((words.count(w) for w in set(words[:200])), default=0)
        if len(words) >= 20 and most_common / len(words) > self.max_repeat_ratio:
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

    def __init__(self, *, n: int = 13) -> None:
        self.n = max(4, int(n))
        self._hashes: Set[int] = set()
        self.loaded = False

    def _ngrams(self, text: str) -> Iterator[int]:
        words = _WS_RE.sub(" ", (text or "").lower()).strip().split()
        for i in range(0, max(0, len(words) - self.n + 1)):
            yield hash(" ".join(words[i:i + self.n]))

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
        if not self._hashes:
            return False
        return any(h in self._hashes for h in self._ngrams(text))


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

    def domains(self) -> List[str]:
        return sorted({s["domain"] for s in self.shards})

    def tokens(self, domain: Optional[str] = None) -> int:
        return sum(s["n_tokens"] for s in self.shards
                   if domain is None or s["domain"] == domain)

    def for_domain(self, domain: str) -> List[Dict[str, Any]]:
        return [s for s in self.shards if s["domain"] == domain]

    def to_dict(self) -> Dict[str, Any]:
        return {"shards": list(self.shards), "vocab_sig": self.vocab_sig,
                "vocab_size": self.vocab_size, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShardIndex":
        return cls(shards=list(d.get("shards") or []), vocab_sig=d.get("vocab_sig", ""),
                   vocab_size=int(d.get("vocab_size", 0)),
                   created_at=float(d.get("created_at", 0.0)))

    def save(self, directory: Any) -> Path:
        path = Path(directory) / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
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
        self._buffers: Dict[str, List[int]] = {}
        self._counts: Dict[str, int] = {}
        self.index = ShardIndex.load(self.dir) or ShardIndex()

    def existing_tokens(self, domain: str) -> int:
        return self.index.tokens(domain)

    def add(self, domain: str, ids: Sequence[int]) -> None:
        if not ids:
            return
        # Validate on the way IN, not at flush time. The buffer holds up to a shard's worth of
        # tokens, so deferring this means a vocabulary that overflows uint16 is only discovered
        # after a hundred million tokens have been accepted — and the ids would silently wrap to
        # different tokens if the check were missed entirely.
        widened = [int(i) for i in ids]
        peak = max(widened)
        if peak > _MAX_ID:
            raise ValueError(
                f"token id {peak} exceeds the uint16 shard range ({_MAX_ID}); "
                f"a vocabulary this large needs a wider shard dtype")
        buf = self._buffers.setdefault(domain, [])
        buf.extend(widened)
        while len(buf) >= self.shard_tokens:
            self._flush(domain, buf[:self.shard_tokens])
            del buf[:self.shard_tokens]

    def _flush(self, domain: str, ids: Sequence[int]) -> None:
        if not ids:
            return
        n = self._counts.get(domain, len(self.index.for_domain(domain)))
        self._counts[domain] = n + 1
        path = self.dir / f"{domain}-{n:05d}.bin"
        if _HAS_NUMPY:
            np.asarray(ids, dtype=_SHARD_DTYPE).tofile(path)
        else:
            import array
            array.array("H", ids).tofile(path.open("wb"))
        self.index.shards.append({"domain": domain, "path": path.name,
                                  "n_tokens": len(ids), "dtype": _SHARD_DTYPE})

    def close(self, *, vocab_sig: str = "", vocab_size: int = 0) -> ShardIndex:
        """Flush every partial shard and write the index."""
        for domain, buf in list(self._buffers.items()):
            if buf:
                self._flush(domain, buf)
                buf.clear()
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
                 weights: Optional[Dict[str, float]] = None, seed: int = 0) -> None:
        if not _HAS_NUMPY:
            raise RuntimeError("ShardDataset requires numpy")
        self.dir = Path(directory)
        index = ShardIndex.load(self.dir)
        if index is None or not index.shards:
            raise FileNotFoundError(f"no shard index under {self.dir} — build the corpus first")
        self.index = index
        self.block_size = max(8, int(block_size))
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._draws = 0

        self._maps: Dict[str, List[Any]] = {}
        for shard in index.shards:
            arr = np.memmap(self.dir / shard["path"], dtype=shard.get("dtype", _SHARD_DTYPE),
                            mode="r")
            self._maps.setdefault(shard["domain"], []).append(arr)

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
        return self.index.tokens()

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
        arr = arrays[self._rng.randrange(len(arrays))]
        if len(arr) <= self.block_size + 1:
            ids = np.asarray(arr, dtype=np.int64)
            pad = self.block_size + 1 - len(ids)
            ids = np.concatenate([ids, ids[:pad]]) if len(ids) else np.zeros(
                self.block_size + 1, dtype=np.int64)
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
    low_quality: int = 0
    contaminated: int = 0
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
        for label, n in (("duplicate", self.duplicates), ("low-quality", self.low_quality),
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
                "low_quality": self.low_quality, "contaminated": self.contaminated,
                "quarantined": self.quarantined,
                "injection_flagged": self.injection_flagged, "notes": list(self.notes)}


class CorpusBuilder:
    """Streams documents from every available source, screens them, and writes token shards."""

    def __init__(self, *, tokenizer: Any, out_dir: Any, settings: Any = None,
                 weights: Optional[Dict[str, float]] = None,
                 tokens_budget: int = 6_000_000_000,
                 stream_sources: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 quality: Optional[QualityFilter] = None,
                 contamination: Optional[ContaminationFilter] = None,
                 shard_tokens: int = _SHARD_TOKENS, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.out_dir = Path(out_dir)
        self.settings = settings
        self.weights = dict(weights or DEFAULT_DOMAIN_WEIGHTS)
        self.tokens_budget = max(1, int(tokens_budget))
        self.stream_sources = dict(stream_sources or DEFAULT_STREAM_SOURCES)
        self.quality = quality or QualityFilter()
        self.contamination = contamination
        self.shard_tokens = shard_tokens
        self.seed = seed
        self._seen: Set[str] = set()
        self._scanner: Any = None

    # ---- screens (reused, not reimplemented) ---- #
    def _screen(self, text: str, report: CorpusReport, *, domain: str,
                trusted: bool) -> bool:
        """Run all five screens. ``True`` keeps the document."""
        from nyxara.growth.dataset import _content_hash, _loyalty_hostility

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

    def _synthetic_docs(self, report: CorpusReport, per_domain: int
                        ) -> Iterator[Tuple[str, str, bool]]:
        """Verified synthetic math/code/tool/reasoning data — always available, no network."""
        try:
            from nyxara.growth.synth_data import generate_domain_docs
        except Exception as exc:  # noqa: BLE001
            report.note(f"synthetic generators unavailable ({exc})")
            return
        n = 0
        for domain in ("math", "code", "conversation"):
            for doc in generate_domain_docs(domain, per_domain, seed=self.seed):
                n += 1
                yield doc, domain, True
        report.sources["synthetic"] = n

    def _own_docs(self, report: CorpusReport) -> Iterator[Tuple[str, str, bool]]:
        """Her own lived corpus — flywheel, distillation, screened web text."""
        n = 0
        try:
            from nyxara.growth.foundry import Foundry
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

        n = 0
        for domain, specs in self.stream_sources.items():
            if budget.get(domain, 0) <= 0:
                continue
            for spec in specs:
                try:
                    stream = load_dataset(spec["path"], spec.get("name"),
                                          split=spec.get("split", "train"), streaming=True)
                except Exception as exc:  # noqa: BLE001 — gated/renamed/offline: note and move on
                    report.note(f"{domain}: {spec['path']} unavailable ({exc})")
                    continue
                key = spec.get("text_key", "text")
                for row in stream:
                    text = _row_text(row, key)
                    if text:
                        n += 1
                        yield text, domain, False
                    if n % 512 == 0 and self._budget_spent(budget):
                        break
        report.sources["streamed"] = n

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

        per_domain_synth = max(100, min(20_000, self.tokens_budget // 400_000 or 100))
        streams = (
            self._master_docs(report),
            self._synthetic_docs(report, per_domain_synth),
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
                writer.add(domain, ids)
                budget[domain] -= len(ids)
                report.tokens[domain] = report.tokens.get(domain, 0) + len(ids)
                report.documents[domain] = report.documents.get(domain, 0) + 1
            if self._budget_spent(budget):
                break

        index = writer.close(vocab_sig=self.tokenizer.vocab_sig(),
                             vocab_size=self.tokenizer.vocab_size)
        for domain in DOMAINS:
            if budget.get(domain, 0) > 0 and report.tokens.get(domain, 0) == 0:
                report.note(f"{domain}: no data found — this domain is ABSENT from the corpus")
        return index, report


def _row_text(row: Any, key: str) -> str:
    """Extract training text from one streamed row, including chat-shaped rows."""
    if not isinstance(row, dict):
        return str(row or "")
    value = row.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, list):           # a `messages` transcript — render it in her template
        try:
            from nyxara.mind.llm import format_self_training_doc
            user = assistant = ""
            for msg in value:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).lower()
                if role in ("user", "human"):
                    user = str(msg.get("content", "") or "")
                elif role in ("assistant", "ai", "bot"):
                    assistant = str(msg.get("content", "") or "")
            if user and assistant:
                return format_self_training_doc(user, assistant)
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
                 check_contamination: bool = True,
                 shard_tokens: int = _SHARD_TOKENS,
                 seed: int = 0) -> Tuple[ShardIndex, CorpusReport]:
    """Build a sharded, screened, domain-mixed pretraining corpus. Resumable.

    ``tokens_budget`` defaults to the Chinchilla-optimal 6B for a 300M model. The real corpus
    will usually be smaller than the budget — that is fine and is reported, not hidden.
    """
    contamination = (ContaminationFilter().load_eval_sets(settings)
                     if check_contamination else None)
    builder = CorpusBuilder(tokenizer=tokenizer, out_dir=out_dir, settings=settings,
                            weights=weights, tokens_budget=tokens_budget,
                            stream_sources=stream_sources, contamination=contamination,
                            shard_tokens=shard_tokens, seed=seed)
    index, report = builder.build()
    if contamination is not None and not contamination.loaded:
        report.note("contamination screen was EMPTY — eval sets could not be loaded, so "
                    "train/eval overlap was NOT checked")
    return index, report
