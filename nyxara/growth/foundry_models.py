"""NYXARA · growth/foundry_models.py — the trainable language models she owns (🛠, from zero).

This is where NYXARA's *own* brain is forged. Several backends, same interface, chosen by what
the machine can run — exactly the "works bare, sharper with better tools" philosophy of
:mod:`senses.vision`:

* :class:`WordKNGramLM` — **the coherent, always-available local brain (pure stdlib).** A
  word/subword-level n-gram with **interpolated Kneser-Ney** smoothing, trained FROM SCRATCH.
  KN's continuation probabilities make n-gram text read naturally (a word is likely because it
  appears in many distinct contexts, not merely because it is frequent), so on a GPU-less box
  this is the honest ceiling for a sovereign model — real words, not byte gibberish. It is the
  fallback :func:`build_model` chooses when ``torch`` is absent.
* :class:`NgramByteLM` — a byte-level n-gram with add-k smoothing (vocabulary fixed at 256, no
  tokenizer). Kept for exact backward-compatibility and selected only by an explicit
  ``kind="ngram"``.
* :class:`NanoGPTModel` — **optional, only when ``torch`` is installed.** A small byte-level
  GPT (decoder-only transformer) trained from zero with AdamW. Constructing it without torch
  raises a clearly-caught error so :func:`build_model` falls back to the word-level KN model.

Every model implements the same :class:`BaseLanguageModel` contract — ``train_on``,
``generate``, ``perplexity``, ``save``/``load``, ``param_count`` — so the foundry
(:mod:`growth.foundry`) and the :class:`~nyxara.mind.llm.SelfProvider` can treat them
interchangeably.

Pure standard library at the core; ``torch`` optional. Imports NOTHING from ``mind/`` —
the dependency only ever flows mind/llm.py -> here (lazily), never back.
"""

from __future__ import annotations

import json
import math
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # optional, never required — the n-gram backend always works without it
    import torch  # type: ignore
    from torch import nn  # type: ignore
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _has_lora() -> bool:
    """True iff the LoRA stack (torch + transformers + peft) is importable."""
    if not _HAS_TORCH:
        return False
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _has_bnb() -> bool:
    """True iff bitsandbytes is importable — the 4-bit quantization that makes QLoRA fit."""
    if not _HAS_LORA:
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


_HAS_LORA = _has_lora()
_HAS_BNB = _has_bnb()

__all__ = [
    "ModelSpec",
    "TrainStats",
    "BaseLanguageModel",
    "NgramByteLM",
    "NanoGPTModel",
    "LoRAModel",
    "build_model",
    "strongest_backend",
    "load_active_model",
    "_should_quantize",
    "_quant_kwargs",
    "_HAS_TORCH",
    "_HAS_LORA",
    "_HAS_BNB",
]

_VOCAB = 256          # byte-level: every model speaks raw bytes
_BOS = -1             # beginning-of-sequence padding marker for n-gram contexts


# --------------------------------------------------------------------------- #
# Spec & training stats
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """Declares which model to build and its (capability-only) hyper-parameters.

    Note: every field here is a *capability* knob (size, order, seed) — never a character
    value. The foundry's gauntlet refuses any spec that tries to smuggle an immutable-core
    name in here (see :meth:`growth.foundry.Foundry._gauntlet`)."""

    kind: str = "auto"          # "auto" | "ngram" | "nanogpt" | "lora" | "genesis" | "genesis_np"
    ngram_order: int = 3
    ngram_k: float = 1.0        # add-k smoothing (n-gram / genesis-stdlib substrate)
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    seed: int = 0
    # ---- Genesis architecture (kind="genesis"; the searched topology, growth/genesis.py) ---- #
    # The serialized ArchitectureGenome the Genesis Protocol crowned. When present, build_model
    # assembles a brand-new neural net from it: a torch GenesisModel when torch is installed, else
    # a REAL pure-NumPy GenesisNumpyModel (genesis_numpy.py) that actually builds and trains the
    # designed topology on the always-on substrate. Only with neither torch nor NumPy, or a
    # layer-less genome, does it degrade to the n-gram substrate. Never raises.
    genome: Optional[Dict[str, Any]] = None
    # Which substrate scores/serves a genome when there is no torch: "numpy"/"auto" build the REAL
    # pure-NumPy GenesisNumpyModel from the genome; "ngram" (the safe default) keeps the fast
    # always-on word-Kneser-Ney substrate. The search, promotion and topology paths set this from
    # the GenesisConfig; an incidental genesis spec left at the default stays fast. ``kind`` of
    # "genesis_np" forces the NumPy brain regardless of this field.
    substrate: str = "ngram"
    # ---- LoRA fine-tuning knobs (kind="lora"; needs torch+transformers+peft) ---- #
    # Low-level default is a tiny GPT-2 so direct/test instantiation stays instant and offline;
    # the runtime foundry threads in FoundryConfig.base_model (Qwen2.5-0.5B-Instruct) — the single
    # real base NYXARA fine-tunes. Override per-spec for any other HF causal-LM.
    base_model: str = "sshleifer/tiny-gpt2"   # the pretrained base to adapt
    # Load custom modeling code shipped with the base. Qwen2.5 (the runtime base) is a native
    # transformers arch needing no remote code; kept a knob for an exotic base that ships its own
    # modeling code. Threaded into every AutoTokenizer/AutoModel from_pretrained below.
    trust_remote_code: bool = False
    lora_r: int = 8
    lora_r_auto: bool = False    # True -> scale the rank to the base width (set by the foundry)
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_lr: float = 2e-4
    max_seq_len: int = 256
    device: str = ""            # "" -> auto (cuda if available, else cpu)
    # ---- QLoRA: 4-bit quantization of the base (needs bitsandbytes + CUDA) ---- #
    # This is what makes a 7B+ base fit and fine-tune on a single consumer GPU: the frozen base
    # is loaded in 4-bit and only the small adapter trains in higher precision. Requested via
    # ``load_in_4bit``; honoured only when bitsandbytes is importable AND CUDA is present —
    # otherwise the LoRA backend silently loads the base full-precision (so CPU/CI is unchanged).
    load_in_4bit: bool = False
    load_in_8bit: bool = False                  # 8-bit alternative (4-bit wins if both set)
    bnb_4bit_quant_type: str = "nf4"            # "nf4" (QLoRA default) or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"    # compute dtype for the de-quantized matmuls
    bnb_4bit_use_double_quant: bool = True      # nested quantization — a little more memory saved
    gradient_checkpointing: bool = True         # trade compute for memory (recommended with 4-bit)
    # ---- LoRA structure (full control over WHERE the adapter attaches) ---- #
    # Empty -> let peft infer the architecture defaults (with an all-linear fallback).
    lora_target_modules: Tuple[str, ...] = ()
    lora_bias: str = "none"                     # "none" | "all" | "lora_only"
    lora_use_rslora: bool = False               # rank-stabilised LoRA scaling
    lora_modules_to_save: Tuple[str, ...] = ()  # extra full-precision modules (e.g. lm_head)
    # ---- Optimiser / schedule (full control over HOW the adapter trains) ---- #
    batch_size: int = 1
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.03
    lr_scheduler: str = "constant"              # "constant" | "linear" | "cosine"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0                  # 0 -> no clipping
    train_epochs: int = 0                       # 0 -> use the caller's ``steps``

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ngram_order": self.ngram_order,
                "ngram_k": self.ngram_k, "block_size": self.block_size,
                "n_layer": self.n_layer, "n_head": self.n_head, "n_embd": self.n_embd,
                "seed": self.seed, "genome": self.genome, "substrate": self.substrate,
                "base_model": self.base_model,
                "trust_remote_code": self.trust_remote_code, "lora_r": self.lora_r,
                "lora_r_auto": self.lora_r_auto,
                "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
                "lora_lr": self.lora_lr, "max_seq_len": self.max_seq_len,
                "device": self.device, "load_in_4bit": self.load_in_4bit,
                "load_in_8bit": self.load_in_8bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "gradient_checkpointing": self.gradient_checkpointing,
                "lora_target_modules": list(self.lora_target_modules),
                "lora_bias": self.lora_bias,
                "lora_use_rslora": self.lora_use_rslora,
                "lora_modules_to_save": list(self.lora_modules_to_save),
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "warmup_ratio": self.warmup_ratio,
                "lr_scheduler": self.lr_scheduler,
                "weight_decay": self.weight_decay,
                "adam_beta1": self.adam_beta1, "adam_beta2": self.adam_beta2,
                "adam_eps": self.adam_eps, "max_grad_norm": self.max_grad_norm,
                "train_epochs": self.train_epochs}

    def __post_init__(self) -> None:
        # JSON round-trips tuples as lists — normalise so from_dict(to_dict()) == spec.
        if not isinstance(self.lora_target_modules, tuple):
            self.lora_target_modules = tuple(self.lora_target_modules or ())
        if not isinstance(self.lora_modules_to_save, tuple):
            self.lora_modules_to_save = tuple(self.lora_modules_to_save or ())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class TrainStats:
    steps: int = 0
    final_loss: float = 0.0     # cross-entropy (nats) on the training corpus
    seconds: float = 0.0
    tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "final_loss": round(self.final_loss, 5),
                "seconds": round(self.seconds, 4), "tokens": self.tokens}


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
class BaseLanguageModel(ABC):
    """The uniform interface every NYXARA-owned model implements."""

    kind: str = "base"

    @abstractmethod
    def train_on(self, corpus: Sequence[str], *, steps: int = 0, seed: int = 0) -> TrainStats: ...

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 128) -> str: ...

    @abstractmethod
    def perplexity(self, text: str) -> float: ...

    @abstractmethod
    def param_count(self) -> int: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @abstractmethod
    def load(self, directory: Path) -> None: ...


# --------------------------------------------------------------------------- #
# Always-on backend: a pure-stdlib byte-level n-gram model, trained from zero
# --------------------------------------------------------------------------- #
class NgramByteLM(BaseLanguageModel):
    """A from-scratch byte-level n-gram language model with interpolated back-off smoothing.

    Counts are exact (one pass over the corpus, accumulating every context order n..1). The model
    starts empty — it *is* trained from zero — and scores with recursive Jelinek-Mercer
    interpolation: a full-order context unseen for a given byte still profits from the shorter
    contexts it *did* see, backing off smoothly down to a smoothed unigram rather than collapsing
    straight to it. So every probability is finite, and held-out text is scored far better than
    plain add-k. Pure standard library: dict-of-dicts counts, unchanged on-disk schema.
    """

    kind = "ngram"

    def __init__(self, order: int = 3, *, k: float = 1.0, seed: int = 0) -> None:
        self.order = max(1, int(order))
        self.k = float(k)
        self.seed = int(seed)
        # context (tuple of the previous order-1 byte values) -> {next_byte: count}
        self.counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self.totals: Dict[Tuple[int, ...], int] = {}
        self.unigram: List[int] = [0] * _VOCAB
        self.unigram_total: int = 0

    # ---- training ---- #
    def _contexts(self, data: bytes):
        ctx_len = self.order - 1
        pad = (_BOS,) * ctx_len
        seq = pad + tuple(data)
        for i in range(ctx_len, len(seq)):
            yield seq[i - ctx_len:i], seq[i]

    def train_on(self, corpus: Sequence[str], *, steps: int = 0, seed: int = 0) -> TrainStats:
        import time
        start = time.monotonic()
        tokens = 0
        for doc in corpus:
            data = doc.encode("utf-8", errors="replace")
            for ctx, nxt in self._contexts(data):
                # count EVERY suffix of the context (orders n..1), so _prob can interpolate down
                # through the shorter contexts instead of dropping straight to the unigram. Same
                # dict-of-dicts schema — just more keys — so save/load round-trips unchanged.
                for j in range(len(ctx)):
                    sub = ctx[j:]
                    row = self.counts.get(sub)
                    if row is None:
                        row = {}
                        self.counts[sub] = row
                    row[nxt] = row.get(nxt, 0) + 1
                    self.totals[sub] = self.totals.get(sub, 0) + 1
                self.unigram[nxt] += 1
                self.unigram_total += 1
                tokens += 1
        loss = self._corpus_cross_entropy(corpus)
        # n-gram counting is single-pass and exact; we report it as one "step" per document.
        return TrainStats(steps=max(1, len(corpus)), final_loss=loss,
                          seconds=time.monotonic() - start, tokens=tokens)

    # ---- probability ---- #
    def _prob(self, ctx: Tuple[int, ...], nxt: int) -> float:
        # Recursive Jelinek-Mercer interpolation, highest order down to the smoothed unigram:
        #   p(nxt|ctx) = λ·p_ml(nxt|ctx) + (1-λ)·p(nxt|shorter ctx),  λ = N(ctx)/(N(ctx)+k·V).
        # λ grows with the evidence a context carries, so a well-seen (n-1)-gram rescues a full
        # context that never saw THIS byte — real back-off, not a hard fall to the unigram.
        p = (self.unigram[nxt] + self.k) / (self.unigram_total + self.k * _VOCAB)
        for i in range(1, len(ctx) + 1):
            sub = ctx[len(ctx) - i:]          # the i most-recent context bytes (a suffix)
            total = self.totals.get(sub, 0)
            if total <= 0:
                continue
            row = self.counts.get(sub) or {}
            p_ml = row.get(nxt, 0) / total
            lam = total / (total + self.k * _VOCAB)
            p = lam * p_ml + (1.0 - lam) * p
        return p

    def _corpus_cross_entropy(self, corpus: Sequence[str]) -> float:
        total_nll, n = 0.0, 0
        for doc in corpus:
            data = doc.encode("utf-8", errors="replace")
            for ctx, nxt in self._contexts(data):
                total_nll += -math.log(self._prob(ctx, nxt))
                n += 1
        return total_nll / n if n else 0.0

    def perplexity(self, text: str) -> float:
        ce = self._corpus_cross_entropy([text])
        return math.exp(ce) if ce < 700 else float("inf")

    # ---- generation ---- #
    def generate(self, prompt: str, *, max_tokens: int = 128) -> str:
        rng = random.Random(self.seed)
        ctx_len = self.order - 1
        out = bytearray()
        data = prompt.encode("utf-8", errors="replace")
        ctx = ((_BOS,) * ctx_len + tuple(data))[-ctx_len:] if ctx_len else ()
        for _ in range(max_tokens):
            row = self.counts.get(ctx)
            if row:
                bytes_, weights = zip(*row.items())
            elif self.unigram_total:
                bytes_ = [b for b in range(_VOCAB) if self.unigram[b]]
                weights = [self.unigram[b] for b in bytes_]
            else:
                break  # an untrained model has nothing to say
            nxt = rng.choices(list(bytes_), weights=list(weights), k=1)[0]
            out.append(nxt)
            if ctx_len:
                ctx = (ctx + (nxt,))[-ctx_len:]
        return out.decode("utf-8", errors="replace")

    # ---- introspection & persistence ---- #
    def param_count(self) -> int:
        return sum(len(row) for row in self.counts.values())

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # tuple keys aren't JSON-able; encode contexts as comma-joined ints.
        counts = {",".join(map(str, ctx)): {str(b): c for b, c in row.items()}
                  for ctx, row in self.counts.items()}
        blob = {"kind": self.kind, "order": self.order, "k": self.k, "seed": self.seed,
                "counts": counts,
                "totals": {",".join(map(str, ctx)): t for ctx, t in self.totals.items()},
                "unigram": self.unigram, "unigram_total": self.unigram_total}
        (directory / "model.json").write_text(json.dumps(blob), encoding="utf-8")

    def load(self, directory: Path) -> None:
        blob = json.loads((Path(directory) / "model.json").read_text(encoding="utf-8"))
        self.order = blob["order"]; self.k = blob["k"]; self.seed = blob.get("seed", 0)

        def _key(s: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in s.split(",")) if s else ()

        self.counts = {_key(s): {int(b): c for b, c in row.items()}
                       for s, row in blob["counts"].items()}
        self.totals = {_key(s): t for s, t in blob["totals"].items()}
        self.unigram = list(blob["unigram"])
        self.unigram_total = blob["unigram_total"]


# --------------------------------------------------------------------------- #
# Always-on backend: a word/subword-level Kneser-Ney n-gram (coherent, pure stdlib)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")   # words, numbers, or a single symbol
_BOS_TOK = "<bos>"
_EOS_TOK = "<eos>"
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}…’”'\"")
_NO_SPACE_AFTER = frozenset("([{“‘$#@")


def _word_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _word_detokenize(tokens: Sequence[str]) -> str:
    """Re-join word tokens into natural text (no space before punctuation / after openers)."""
    out: List[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        prev = tokens[i - 1]
        if tok in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER or tok == "'" or prev == "'":
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out)


class WordKNGramLM(BaseLanguageModel):
    """A word-level n-gram language model with **interpolated Kneser-Ney** smoothing.

    This is the coherent, fully-local "own brain": instead of sampling raw bytes (which yields
    locally-plausible gibberish), it models the distribution over *words*, and KN's continuation
    probabilities are exactly what make n-gram text read naturally — a word is likely not just
    because it is frequent, but because it appears in many *distinct* contexts. Trained from
    zero, pure standard library (no numpy/torch/network), so it runs on the barest machine and
    is the honest ceiling for a sovereign local model without a GPU.
    """

    kind = "kngram"

    # the searchable learning *method* — Genesis evolves which smoothing (i.e., how the model
    # generalizes from counts) wins, so the "new learning paradigm" dimension is real on a bare
    # machine too, not only on the torch path.
    _SMOOTHINGS: Tuple[str, ...] = ("kneser_ney", "add_k", "interpolated", "stupid_backoff")

    def __init__(self, order: int = 3, *, discount: float = 0.75, seed: int = 0,
                 smoothing: str = "kneser_ney", k: float = 1.0) -> None:
        self.order = max(2, int(order))
        self.D = min(0.99, max(0.1, float(discount)))
        self.smoothing = smoothing if smoothing in self._SMOOTHINGS else "kneser_ney"
        self.k = max(1e-3, float(k))               # add-k pseudo-count
        self.seed = int(seed)
        # raw m-gram counts (all orders) — used by add-k / interpolation / stupid-backoff smoothing
        self.raw: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {}
        self.raw_tot: Dict[int, Dict[Tuple[int, ...], int]] = {}
        self.tok2id: Dict[str, int] = {}
        self.id2tok: List[str] = []
        self._bos = self._intern(_BOS_TOK)
        self._eos = self._intern(_EOS_TOK)
        # highest-order raw counts and lower-order continuation counts, both context-keyed
        self.hi: Dict[Tuple[int, ...], Dict[int, int]] = {}
        self.lo: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {}
        self._hi_tot: Dict[Tuple[int, ...], int] = {}
        self._lo_tot: Dict[int, Dict[Tuple[int, ...], int]] = {}

    def _intern(self, tok: str) -> int:
        i = self.tok2id.get(tok)
        if i is None:
            i = len(self.id2tok)
            self.tok2id[tok] = i
            self.id2tok.append(tok)
        return i

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.id2tok))

    def _encode(self, text: str, *, intern: bool) -> List[int]:
        ids: List[int] = [self._bos] * (self.order - 1)
        for t in _word_tokenize(text):
            ids.append(self._intern(t) if intern else self.tok2id.get(t, self._unk()))
        ids.append(self._eos)
        return ids

    def _unk(self) -> int:
        return self._bos          # unseen tokens fold onto BOS (a never-emitted context slot)

    # ---- training ---- #
    def train_on(self, corpus: Sequence[str], *, steps: int = 0, seed: int = 0,
                 accumulate: bool = False) -> TrainStats:
        """Train from the corpus. With ``accumulate=True`` the new counts are *added on top* of the
        existing ones (continual learning, growth/foundry continual path): prior knowledge is
        retained instead of overwritten, so a later skill cannot erase an earlier one. The default
        (``accumulate=False``) rebuilds from scratch — the historical behaviour, byte-for-byte."""
        import time
        start = time.monotonic()
        raw: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
            m: defaultdict(lambda: defaultdict(int)) for m in range(1, self.order + 1)}
        # continual warm-start: seed the counters with the weights already learned, so this round
        # adds to (never replaces) prior knowledge.
        if accumulate and self.raw:
            for m in range(1, self.order + 1):
                for ctx, row in self.raw.get(m, {}).items():
                    for w, c in row.items():
                        raw[m][ctx][w] += int(c)
        tokens = 0
        for doc in corpus:
            ids = self._encode(doc, intern=True)
            for m in range(1, self.order + 1):
                for i in range(m - 1, len(ids)):
                    ctx = tuple(ids[i - m + 1:i])
                    raw[m][ctx][ids[i]] += 1
            tokens += len(ids)
        # keep the raw m-gram counts (all orders) for the non-KN smoothing methods
        self.raw = {m: {ctx: dict(row) for ctx, row in raw[m].items()}
                    for m in range(1, self.order + 1)}
        # derive the KN high/low tables (and all totals) from the raw counts
        self._rebuild_smoothing()
        loss = self._corpus_cross_entropy(corpus)
        return TrainStats(steps=max(1, len(corpus)), final_loss=loss,
                          seconds=time.monotonic() - start, tokens=tokens)

    def unlearn(self, corpus: Sequence[str], *, strength: int = 1) -> int:
        """The exact inverse of ``train_on(accumulate=True)``: decrement the m-gram counts for these
        docs (never below zero) and rebuild the smoothing tables — a real *negative* learning signal.

        This is how NYXARA learns from a mistake at the weight level, not just by refusing to repeat
        it: a continuation she was corrected on is genuinely un-learned from the distribution
        ``generate``/``perplexity`` read, across all orders (so KN backoff cannot resurrect it),
        deterministically and reversibly (re-teaching restores it). Unknown tokens are skipped —
        there is nothing to decrement — so it never invents or corrupts vocabulary. Returns the
        number of (context, word) counts it reduced. Only ever touches capability counts; the
        immutable character core is not representable in these tables, so it cannot be reached."""
        if not self.raw:
            return 0
        s = max(1, int(strength))
        reduced = 0
        for doc in corpus:
            seq = self._encode(doc, intern=False)          # same padded id stream training saw
            for m in range(1, self.order + 1):
                table = self.raw.get(m)
                if not table:
                    continue
                for i in range(m - 1, len(seq)):
                    ctx = tuple(seq[i - m + 1:i])
                    w = seq[i]
                    row = table.get(ctx)
                    if not row or w not in row:
                        continue
                    row[w] -= s
                    reduced += 1
                    if row[w] <= 0:
                        del row[w]
                        if not row:
                            table.pop(ctx, None)
        if reduced:
            self._rebuild_smoothing()
        return reduced

    def _rebuild_smoothing(self) -> None:
        """Re-derive the raw totals and the KN high/low continuation tables from ``self.raw``.

        Shared by :meth:`train_on` and :meth:`unlearn` so that after any count change (add or
        decrement) the smoothing tables and totals are consistent with the raw counts."""
        self.raw_tot = {m: {ctx: sum(row.values()) for ctx, row in self.raw.get(m, {}).items()}
                        for m in range(1, self.order + 1)}
        # highest order uses raw counts directly
        self.hi = {ctx: dict(row) for ctx, row in self.raw.get(self.order, {}).items()}
        # lower orders use Kneser-Ney *continuation* counts (distinct left-extensions)
        self.lo = {}
        for m in range(1, self.order):
            left: Dict[Tuple[int, ...], Dict[int, set]] = defaultdict(lambda: defaultdict(set))
            for big_ctx, row in self.raw.get(m + 1, {}).items():   # (m+1)-gram prefix of length m
                pre, suffix = big_ctx[0], big_ctx[1:]              # preceding token, lower context
                for w in row:
                    left[suffix][w].add(pre)
            self.lo[m] = {ctx: {w: len(s) for w, s in ws.items()} for ctx, ws in left.items()}
        self._recompute_totals()

    def _recompute_totals(self) -> None:
        self._hi_tot = {ctx: sum(row.values()) for ctx, row in self.hi.items()}
        self._lo_tot = {m: {ctx: sum(row.values()) for ctx, row in self.lo[m].items()}
                        for m in self.lo}

    # ---- interpolated Kneser-Ney probability ---- #
    def _p(self, m: int, ctx: Tuple[int, ...], w: int) -> float:
        if m == 1:
            row = self.lo.get(1, {}).get((), {})
            total = self._lo_tot.get(1, {}).get((), 0)
            if total <= 0:
                return 1.0 / self.vocab_size
            cc = row.get(w, 0)
            lam = self.D * len(row) / total
            return max(cc - self.D, 0.0) / total + lam * (1.0 / self.vocab_size)
        if m == self.order:
            total = self._hi_tot.get(ctx, 0)
            if total <= 0:
                return self._p(m - 1, ctx[1:], w)
            row = self.hi.get(ctx, {})
            lam = self.D * len(row) / total
            return max(row.get(w, 0) - self.D, 0.0) / total + lam * self._p(m - 1, ctx[1:], w)
        total = self._lo_tot.get(m, {}).get(ctx, 0)
        if total <= 0:
            return self._p(m - 1, ctx[1:], w)
        row = self.lo[m].get(ctx, {})
        lam = self.D * len(row) / total
        return max(row.get(w, 0) - self.D, 0.0) / total + lam * self._p(m - 1, ctx[1:], w)

    # ---- alternative searchable smoothing methods (all real, all from-scratch) ---- #
    def _p_addk(self, m: int, ctx: Tuple[int, ...], w: int) -> float:
        """Add-k (Lidstone) smoothing with shorten-context backoff when a context is unseen."""
        total = self.raw_tot.get(m, {}).get(ctx, 0)
        if total <= 0:
            return 1.0 / self.vocab_size if m == 1 else self._p_addk(m - 1, ctx[1:], w)
        cnt = self.raw.get(m, {}).get(ctx, {}).get(w, 0)
        return (cnt + self.k) / (total + self.k * self.vocab_size)

    def _p_interp(self, m: int, ctx: Tuple[int, ...], w: int) -> float:
        """Jelinek-Mercer linear interpolation across orders down to a uniform base."""
        base = 1.0 / self.vocab_size
        if m < 1:
            return base
        total = self.raw_tot.get(m, {}).get(ctx, 0)
        ml = (self.raw.get(m, {}).get(ctx, {}).get(w, 0) / total) if total > 0 else 0.0
        lam = 0.7
        lower = base if m == 1 else self._p_interp(m - 1, ctx[1:], w)
        return lam * ml + (1.0 - lam) * lower

    def _p_backoff(self, m: int, ctx: Tuple[int, ...], w: int) -> float:
        """Stupid backoff (Brants et al.): use the ML estimate where seen, else discount and back off."""
        total = self.raw_tot.get(m, {}).get(ctx, 0)
        cnt = self.raw.get(m, {}).get(ctx, {}).get(w, 0) if total > 0 else 0
        if cnt > 0:
            return cnt / total
        return 0.4 * (1.0 / self.vocab_size if m == 1 else self._p_backoff(m - 1, ctx[1:], w))

    def _prob(self, ctx_full: Tuple[int, ...], w: int) -> float:
        ctx = ctx_full[-(self.order - 1):] if self.order > 1 else ()
        if self.smoothing == "add_k":
            return self._p_addk(self.order, ctx, w)
        if self.smoothing == "interpolated":
            return self._p_interp(self.order, ctx, w)
        if self.smoothing == "stupid_backoff":
            return self._p_backoff(self.order, ctx, w)
        return self._p(self.order, ctx, w)          # interpolated Kneser-Ney (default)

    def _corpus_cross_entropy(self, corpus: Sequence[str]) -> float:
        total_nll, n = 0.0, 0
        for doc in corpus:
            ids = self._encode(doc, intern=False)
            for i in range(self.order - 1, len(ids)):
                ctx = tuple(ids[i - self.order + 1:i])
                p = self._prob(ctx, ids[i])
                total_nll += -math.log(p if p > 0 else 1e-12)
                n += 1
        return total_nll / n if n else 0.0

    def perplexity(self, text: str) -> float:
        ce = self._corpus_cross_entropy([text])
        return math.exp(ce) if ce < 700 else float("inf")

    # ---- generation ---- #
    def _candidates(self, ctx_full: Tuple[int, ...]) -> List[int]:
        cands: set = set()
        for m in range(self.order, 1, -1):
            ctx = ctx_full[-(m - 1):]
            row = self.hi.get(ctx) if m == self.order else self.lo.get(m, {}).get(ctx)
            if row:
                cands.update(row.keys())
            if len(cands) >= 48:
                break
        if not cands:
            cands.update(self.lo.get(1, {}).get((), {}).keys())
        cands.discard(self._bos)
        return list(cands)

    def generate(self, prompt: str, *, max_tokens: int = 128, top_k: int = 0,
                 top_p: float = 0.0, repetition_penalty: float = 1.0) -> str:
        """Sample a continuation. The defaults (``top_k=0, top_p=0.0,
        repetition_penalty=1.0``) reproduce the historical pure interpolated-KN sampling exactly;
        enabling them trims the long improbable tail (``top_k``/``top_p`` nucleus) and discourages
        immediate loops (``repetition_penalty`` > 1), so the n-gram reads more coherently."""
        if not self.id2tok or not self.hi:
            return ""
        rng = random.Random(self.seed)
        ctx = self._encode(prompt, intern=False)[:-1]      # drop the trailing EOS of the prompt
        out_ids: List[int] = []
        recent: List[int] = []                              # tail used for the repetition penalty
        for _ in range(max_tokens):
            cands = self._candidates(tuple(ctx))
            if not cands:
                break
            weights = [self._prob(tuple(ctx), w) for w in cands]
            if repetition_penalty and repetition_penalty != 1.0 and recent:
                penal = set(recent)
                weights = [(wt / repetition_penalty) if c in penal else wt
                           for c, wt in zip(cands, weights)]
            cands, weights = self._restrict(cands, weights, top_k, top_p)
            wsum = sum(weights)
            if wsum <= 0:
                break
            nxt = rng.choices(cands, weights=weights, k=1)[0]
            if nxt == self._eos:
                break
            out_ids.append(nxt)
            ctx.append(nxt)
            recent.append(nxt)
            if len(recent) > 8:                             # bounded window keeps the penalty local
                recent.pop(0)
        return _word_detokenize([self.id2tok[i] for i in out_ids
                                 if 0 <= i < len(self.id2tok)])

    @staticmethod
    def _restrict(cands: List[int], weights: List[float], top_k: int,
                  top_p: float) -> Tuple[List[int], List[float]]:
        """Keep only the most-probable candidates (top-k count and/or top-p nucleus mass).

        Disabled (``top_k<=0`` and ``top_p<=0``) it returns the inputs untouched, so the default
        sampling path is byte-for-byte the historical one. Order is preserved on ties so a fixed
        seed stays deterministic."""
        if (top_k <= 0 and top_p <= 0.0) or len(cands) <= 1:
            return cands, weights
        order = sorted(range(len(cands)), key=lambda i: weights[i], reverse=True)
        if top_k > 0:
            order = order[:top_k]
        if top_p and top_p > 0.0:
            total = sum(weights[i] for i in order) or 1.0
            kept, acc = [], 0.0
            for i in order:
                kept.append(i)
                acc += weights[i] / total
                if acc >= top_p:
                    break
            order = kept
        keep = sorted(order)                                # restore candidate order for determinism
        return [cands[i] for i in keep], [weights[i] for i in keep]

    # ---- introspection & persistence ---- #
    def param_count(self) -> int:
        return sum(len(r) for r in self.hi.values()) + sum(
            len(r) for m in self.lo for r in self.lo[m].values())

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        def _enc(table: Dict[Tuple[int, ...], Dict[int, int]]) -> Dict[str, Dict[str, int]]:
            return {",".join(map(str, ctx)): {str(w): c for w, c in row.items()}
                    for ctx, row in table.items()}

        blob = {"kind": self.kind, "order": self.order, "discount": self.D, "seed": self.seed,
                "smoothing": self.smoothing, "k": self.k,
                "id2tok": self.id2tok, "hi": _enc(self.hi),
                "lo": {str(m): _enc(self.lo[m]) for m in self.lo},
                "raw": {str(m): _enc(self.raw[m]) for m in self.raw}}
        (directory / "model.json").write_text(json.dumps(blob), encoding="utf-8")

    def load(self, directory: Path) -> None:
        blob = json.loads((Path(directory) / "model.json").read_text(encoding="utf-8"))
        self.order = int(blob["order"]); self.D = float(blob["discount"])
        self.seed = int(blob.get("seed", 0))
        self.smoothing = blob.get("smoothing", "kneser_ney")
        self.k = float(blob.get("k", 1.0))
        self.id2tok = list(blob["id2tok"])
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        self._bos = self.tok2id.get(_BOS_TOK, 0)
        self._eos = self.tok2id.get(_EOS_TOK, min(1, len(self.id2tok) - 1))

        def _key(s: str) -> Tuple[int, ...]:
            return tuple(int(x) for x in s.split(",")) if s else ()

        def _dec(table: Dict[str, Dict[str, int]]) -> Dict[Tuple[int, ...], Dict[int, int]]:
            return {_key(s): {int(w): c for w, c in row.items()} for s, row in table.items()}

        self.hi = _dec(blob["hi"])
        self.lo = {int(m): _dec(tbl) for m, tbl in blob["lo"].items()}
        self.raw = {int(m): _dec(tbl) for m, tbl in blob.get("raw", {}).items()}
        self.raw_tot = {m: {ctx: sum(row.values()) for ctx, row in self.raw[m].items()}
                        for m in self.raw}
        self._recompute_totals()


# --------------------------------------------------------------------------- #
# Optional backend: a from-scratch byte-level nano-GPT (torch only)
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class _Block(nn.Module):
        def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                     nn.Linear(4 * n_embd, n_embd))
            mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
            self.register_buffer("mask", mask)

        def forward(self, x):  # type: ignore[override]
            t = x.size(1)
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=self.mask[:t, :t], need_weights=False)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x

    class _NanoGPT(nn.Module):
        def __init__(self, *, n_embd: int, n_head: int, n_layer: int, block_size: int) -> None:
            super().__init__()
            self.block_size = block_size
            self.tok = nn.Embedding(_VOCAB, n_embd)
            self.pos = nn.Embedding(block_size, n_embd)
            self.blocks = nn.ModuleList(
                [_Block(n_embd, n_head, block_size) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, _VOCAB, bias=False)

        def forward(self, idx):  # type: ignore[override]
            t = idx.size(1)
            pos = torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x))


class NanoGPTModel(BaseLanguageModel):
    """A from-scratch byte-level GPT. Optional: requires ``torch`` (install ``.[foundry]``)."""

    kind = "nanogpt"

    def __init__(self, spec: Optional[ModelSpec] = None) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("NanoGPTModel requires torch (pip install -e .[foundry])")
        self.spec = spec or ModelSpec(kind="nanogpt")
        torch.manual_seed(self.spec.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = _NanoGPT(n_embd=self.spec.n_embd, n_head=self.spec.n_head,
                            n_layer=self.spec.n_layer, block_size=self.spec.block_size
                            ).to(self.device)
        # Optional Elastic Weight Consolidation: when set, the EWC penalty over previously
        # consolidated weights is added to the training loss, so retraining a new generation
        # does not catastrophically forget the last (memory/elastic_synapses.py). None ⇒
        # behaviour is exactly as before.
        self.synapses: Any = None

    def _encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def train_on(self, corpus: Sequence[str], *, steps: int = 200, seed: int = 0) -> TrainStats:
        import time
        start = time.monotonic()
        data = self._encode("\n".join(corpus))
        bs = self.spec.block_size
        if len(data) <= bs + 1:
            data = (data * (bs * 2 // max(1, len(data)) + 2))[: bs * 4]
        t = torch.tensor(data, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3)
        rng = random.Random(seed or self.spec.seed)
        self.net.train()
        last = 0.0
        for _ in range(max(1, steps)):
            i = rng.randint(0, len(data) - bs - 1)
            x = t[i:i + bs].unsqueeze(0)
            y = t[i + 1:i + 1 + bs].unsqueeze(0)
            logits = self.net(x)
            loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.view(-1))
            if self.synapses is not None:   # Elastic Weight Consolidation: protect old skills
                loss = loss + self.synapses.penalty()
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        # Lock this generation's weights in as a consolidated memory so the *next* retrain
        # treats them as important and resists overwriting them.
        if self.synapses is not None:
            try:
                self.synapses.consolidate(task=f"gen-{self.synapses._consolidations}")
            except Exception:  # noqa: BLE001 — forgetting-protection is best-effort
                pass
        return TrainStats(steps=max(1, steps), final_loss=last,
                          seconds=time.monotonic() - start, tokens=len(data))

    def perplexity(self, text: str) -> float:
        self.net.eval()
        data = self._encode(text)
        bs = self.spec.block_size
        if len(data) < 2:
            return float("inf")
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, max(1, len(data) - 1), bs):
                chunk = data[i:i + bs + 1]
                if len(chunk) < 2:
                    break
                x = torch.tensor(chunk[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
                y = torch.tensor(chunk[1:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.net(x)
                loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.view(-1))
                total += float(loss.item()) * y.numel(); n += y.numel()
        ce = total / n if n else float("inf")
        return math.exp(ce) if ce < 700 else float("inf")

    def generate(self, prompt: str, *, max_tokens: int = 128) -> str:
        self.net.eval()
        torch.manual_seed(self.spec.seed)
        idx = self._encode(prompt) or [ord("\n")]
        bs = self.spec.block_size
        with torch.no_grad():
            for _ in range(max_tokens):
                ctx = torch.tensor(idx[-bs:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.net(ctx)[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
                idx.append(nxt)
        return bytes(idx[len(self._encode(prompt)):]).decode("utf-8", errors="replace")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.json").write_text(json.dumps(self.spec.to_dict()), encoding="utf-8")
        torch.save(self.net.state_dict(), directory / "model.pt")
        # Persist the elastic-synapse anchors so forging generations keep their memory.
        if self.synapses is not None:
            try:
                (directory / "synapses.json").write_text(
                    json.dumps(self.synapses.state_dict()), encoding="utf-8")
            except Exception:  # noqa: BLE001 — synapse persistence is best-effort
                pass

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self.spec = ModelSpec.from_dict(
            json.loads((directory / "spec.json").read_text(encoding="utf-8")))
        self.net = _NanoGPT(n_embd=self.spec.n_embd, n_head=self.spec.n_head,
                            n_layer=self.spec.n_layer, block_size=self.spec.block_size
                            ).to(self.device)
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location=self.device))
        syn_path = directory / "synapses.json"
        if self.synapses is not None and syn_path.exists():
            try:
                self.synapses.load_state_dict(
                    json.loads(syn_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — synapse restore is best-effort
                pass


# --------------------------------------------------------------------------- #
# Optional backend: LoRA fine-tuning of a pretrained model (torch+transformers+peft)
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    """True iff a CUDA device is present — 4-bit quantization needs one."""
    try:
        return bool(_HAS_TORCH and torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _should_quantize(spec: "ModelSpec", *, has_bnb: Optional[bool] = None,
                     has_cuda: Optional[bool] = None) -> bool:
    """Decide whether to load the base quantized (4-bit QLoRA or 8-bit), honestly.

    Quantize only when it was *requested* (``spec.load_in_4bit`` / ``spec.load_in_8bit``)
    AND it can actually work — bitsandbytes importable and a CUDA device present. Neither
    has a CPU path, so on a keyless/CPU/CI machine this returns False and the LoRA backend
    loads the base full-precision instead of crashing. Pure (deps injectable) so the
    decision is unit-testable without a GPU.
    """
    has_bnb = _HAS_BNB if has_bnb is None else has_bnb
    has_cuda = _cuda_available() if has_cuda is None else has_cuda
    wanted = spec.load_in_4bit or getattr(spec, "load_in_8bit", False)
    return bool(wanted and has_bnb and has_cuda)


def _quant_kwargs(spec: "ModelSpec") -> Dict[str, Any]:
    """The BitsAndBytesConfig keyword arguments implied by ``spec`` — a pure dict, so the
    intended quantization can be asserted in tests without importing transformers/bnb.
    4-bit wins when both are requested (it is the stricter QLoRA path)."""
    if spec.load_in_4bit or not getattr(spec, "load_in_8bit", False):
        return {"load_in_4bit": True,
                "bnb_4bit_quant_type": spec.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": spec.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": spec.bnb_4bit_use_double_quant}
    return {"load_in_8bit": True}


class LoRAModel(BaseLanguageModel):
    """Adapt a *pretrained* model to NYXARA's lived experience via LoRA (optionally QLoRA).

    Unlike the from-scratch n-gram / nano-GPT backends, this stands on the shoulders of a
    real pretrained base (``spec.base_model``) and learns only a small **low-rank adapter**
    on top — the path to genuine capability, since the base already speaks the language and
    the foundry just teaches it *NYXARA's* voice, facts, and corrections from her own memory.
    Only the adapter weights (a tiny fraction of the parameters) are trained, so it is cheap
    enough to run repeatedly, and the gauntlet still gates every promotion.

    Token-level (the base's tokenizer), so its perplexity is in token space — compare a LoRA
    candidate against a LoRA active, not across backends. Requires ``.[foundry]`` (torch +
    transformers + peft); constructing it without them raises a clearly-caught error so
    :func:`build_model` falls back to the always-on n-gram backend.
    """

    kind = "lora"

    def __init__(self, spec: Optional[ModelSpec] = None) -> None:
        if not _HAS_LORA:
            raise RuntimeError("LoRAModel requires torch+transformers+peft "
                               "(pip install -e .[foundry])")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.spec = spec or ModelSpec(kind="lora")
        torch.manual_seed(self.spec.seed)
        self.device = self.spec.device or ("cuda" if torch.cuda.is_available() else "cpu")
        trc = bool(getattr(self.spec, "trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.base_model, trust_remote_code=trc)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.quantized = _should_quantize(self.spec)
        if self.quantized:
            # QLoRA: load the frozen base in 4-bit so a 7B+ model fits on one GPU, then prepare
            # it for k-bit training (casts norms/embeddings, enables input grads). Device is
            # placed by the HF loader via device_map, so we don't .to(self.device) the base.
            base = self._load_quantized_base()
            net = self._apply_lora(base)
            if self.spec.gradient_checkpointing:
                try:
                    net.gradient_checkpointing_enable()
                    if hasattr(net, "enable_input_require_grads"):
                        net.enable_input_require_grads()
                    if getattr(net, "config", None) is not None:
                        net.config.use_cache = False
                except Exception:  # noqa: BLE001 — checkpointing is an optimisation, never required
                    pass
            self.net = net
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.spec.base_model, trust_remote_code=trc)
            self.net = self._apply_lora(base).to(self.device)
        # Optional Elastic Weight Consolidation over the adapter weights (see NanoGPTModel).
        self.synapses: Any = None

    def _load_quantized_base(self) -> Any:
        """Load the base quantized (4-bit NF4 QLoRA, or 8-bit) and prepare it for
        k-bit LoRA training."""
        import torch as _torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        kw = _quant_kwargs(self.spec)
        if kw.get("load_in_8bit"):
            bnb = BitsAndBytesConfig(load_in_8bit=True)
        else:
            compute_dtype = getattr(_torch, kw["bnb_4bit_compute_dtype"], _torch.float16)
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=kw["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=kw["bnb_4bit_use_double_quant"])
        base = AutoModelForCausalLM.from_pretrained(
            self.spec.base_model, quantization_config=bnb, device_map="auto",
            trust_remote_code=bool(getattr(self.spec, "trust_remote_code", False)))
        return prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=self.spec.gradient_checkpointing)

    @staticmethod
    def _infer_lora_rank(base: Any, fallback: int) -> int:
        """Scale the LoRA rank to the base's hidden width: a wider base earns more adapter
        capacity (≈ width/32), clamped to peft's sane [8, 256]. A 4096-wide 7B base gets r≈128;
        a tiny CPU/CI base stays at the floor of 8. Falls back to ``fallback`` if width is
        unreadable, so this never breaks an unknown architecture."""
        try:
            cfg = getattr(base, "config", None)
            hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0)
        except Exception:  # noqa: BLE001
            hidden = 0
        if hidden <= 0:
            return fallback
        return max(8, min(256, hidden // 32))

    def _apply_lora(self, base: Any) -> Any:
        from peft import LoraConfig, TaskType, get_peft_model
        if getattr(self.spec, "lora_r_auto", False):
            r = self._infer_lora_rank(base, self.spec.lora_r)
            alpha = max(self.spec.lora_alpha, 2 * r)   # keep alpha ≈ 2·r as the rank scales
        else:
            r, alpha = self.spec.lora_r, self.spec.lora_alpha
        common = dict(r=r, lora_alpha=alpha,
                      lora_dropout=self.spec.lora_dropout, task_type=TaskType.CAUSAL_LM,
                      bias=getattr(self.spec, "lora_bias", "none"),
                      use_rslora=getattr(self.spec, "lora_use_rslora", False))
        modules_to_save = list(getattr(self.spec, "lora_modules_to_save", ()) or ())
        if modules_to_save:
            common["modules_to_save"] = modules_to_save
        targets = list(getattr(self.spec, "lora_target_modules", ()) or ())
        if targets:
            # Explicit placement: the spec says exactly which projections adapt.
            return get_peft_model(base, LoraConfig(target_modules=targets, **common))
        try:
            # Let peft infer target modules from the architecture (gpt2->c_attn, llama->q/v…).
            return get_peft_model(base, LoraConfig(**common))
        except Exception:  # noqa: BLE001 — unknown arch: target every linear layer instead
            return get_peft_model(base, LoraConfig(target_modules="all-linear", **common))

    def _attach(self, peft_model: Any) -> None:
        self.net = peft_model.to(self.device)

    # ---- training (adapter weights only) ---- #
    def _windows(self, corpus: Sequence[str]) -> List[List[int]]:
        ids = self.tokenizer("\n".join(corpus)).input_ids
        n = max(8, int(self.spec.max_seq_len))
        if len(ids) <= 1:
            return []
        return [ids[i:i + n] for i in range(0, len(ids), n) if len(ids[i:i + n]) >= 2]

    def _lr_lambda(self, total_steps: int):
        """Warmup then constant / linear-decay / cosine, per ``spec.lr_scheduler``.

        Hand-rolled (no transformers ``get_scheduler`` dependency) so the schedule is
        exact, inspectable, and works with any torch version."""
        warmup = int(total_steps * max(0.0, self.spec.warmup_ratio))
        kind = str(getattr(self.spec, "lr_scheduler", "constant")).lower()

        def factor(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            if kind == "constant" or total_steps <= warmup:
                return 1.0
            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(1.0, max(0.0, progress))
            if kind == "linear":
                return max(0.0, 1.0 - progress)
            if kind == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return factor

    def _batch(self, windows: List[List[int]], picks: List[int]) -> Tuple[Any, Any]:
        """Pad ``picks`` windows to the longest and mask padding out of the loss."""
        pad = self.tokenizer.pad_token_id or 0
        chosen = [windows[i] for i in picks]
        width = max(len(w) for w in chosen)
        ids = torch.full((len(chosen), width), pad, dtype=torch.long)
        labels = torch.full((len(chosen), width), -100, dtype=torch.long)
        for row, w in enumerate(chosen):
            t = torch.tensor(w, dtype=torch.long)
            ids[row, :len(w)] = t
            labels[row, :len(w)] = t
        return ids.to(self.device), labels.to(self.device)

    def train_on(self, corpus: Sequence[str], *, steps: int = 100, seed: int = 0,
                 accumulate: bool = False) -> TrainStats:
        """Fine-tune the LoRA adapter with the spec's full optimiser/schedule control.

        Honours ``batch_size``, ``grad_accum_steps``, ``warmup_ratio`` + ``lr_scheduler``,
        AdamW betas/eps/``weight_decay``, ``max_grad_norm`` clipping, and ``train_epochs``
        (>0 -> passes over the corpus windows override the caller's ``steps``). ``accumulate``
        is accepted for warm-start parity with the count-based backends: continuing to train
        the loaded adapter *is* accumulation for a neural model."""
        import time
        start = time.monotonic()
        windows = self._windows(corpus)
        if not windows:
            return TrainStats(steps=0, final_loss=0.0, seconds=0.0, tokens=0)
        batch = max(1, int(getattr(self.spec, "batch_size", 1)))
        accum = max(1, int(getattr(self.spec, "grad_accum_steps", 1)))
        epochs = int(getattr(self.spec, "train_epochs", 0))
        if epochs > 0:   # optimiser steps needed for `epochs` passes over the windows
            total = max(1, math.ceil(len(windows) * epochs / (batch * accum)))
        else:
            total = max(1, steps)
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=self.spec.lora_lr,
            betas=(getattr(self.spec, "adam_beta1", 0.9), getattr(self.spec, "adam_beta2", 0.999)),
            eps=getattr(self.spec, "adam_eps", 1e-8),
            weight_decay=getattr(self.spec, "weight_decay", 0.0))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, self._lr_lambda(total))
        clip = float(getattr(self.spec, "max_grad_norm", 1.0))
        rng = random.Random(seed or self.spec.seed)
        self.net.train()
        last, tokens = 0.0, 0
        for _ in range(total):
            opt.zero_grad()
            for _micro in range(accum):
                picks = [rng.randrange(len(windows)) for _ in range(batch)]
                ids, labels = self._batch(windows, picks)
                out = self.net(input_ids=ids, labels=labels)
                loss = out.loss
                if self.synapses is not None:   # Elastic Weight Consolidation: protect old skills
                    loss = loss + self.synapses.penalty()
                (loss / accum).backward()
                last = float(out.loss.item())
                tokens += sum(len(windows[i]) for i in picks)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            sched.step()
        if self.synapses is not None:
            try:
                self.synapses.consolidate(task=f"gen-{self.synapses._consolidations}")
            except Exception:  # noqa: BLE001 — forgetting-protection is best-effort
                pass
        return TrainStats(steps=total, final_loss=last,
                          seconds=time.monotonic() - start, tokens=tokens)

    def perplexity(self, text: str) -> float:
        self.net.eval()
        ids_all = self.tokenizer(text).input_ids
        n = max(8, int(self.spec.max_seq_len))
        if len(ids_all) < 2:
            return float("inf")
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids_all), n):
                chunk = ids_all[i:i + n]
                if len(chunk) < 2:
                    break
                ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                loss = float(self.net(input_ids=ids, labels=ids).loss.item())
                total += loss * (len(chunk) - 1); count += len(chunk) - 1
        ce = total / count if count else float("inf")
        return math.exp(ce) if ce < 700 else float("inf")

    def generate(self, prompt: str, *, max_tokens: int = 128) -> str:
        self.net.eval()
        torch.manual_seed(self.spec.seed)
        enc = self.tokenizer(prompt or "\n", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.net.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id)
        new = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def param_count(self) -> int:
        """LoRA-adapter parameters — the part NYXARA actually learns.

        Counted by name so it is stable whether the adapter is in training mode (requires
        grad) or freshly loaded for inference (frozen)."""
        n = sum(p.numel() for name, p in self.net.named_parameters() if "lora_" in name)
        return n or sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.json").write_text(json.dumps(self.spec.to_dict()), encoding="utf-8")
        self.net.save_pretrained(str(directory / "adapter"))   # adapter weights only
        self.tokenizer.save_pretrained(str(directory / "adapter"))

    def load(self, directory: Path) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        directory = Path(directory)
        self.spec = ModelSpec.from_dict(
            json.loads((directory / "spec.json").read_text(encoding="utf-8")))
        base = AutoModelForCausalLM.from_pretrained(self.spec.base_model)
        self._attach(PeftModel.from_pretrained(base, str(directory / "adapter")))


# --------------------------------------------------------------------------- #
# Factory & active-model loader
# --------------------------------------------------------------------------- #
def _numpy_available() -> bool:
    """True iff NumPy is importable — the substrate for a real CPU-only neural transformer."""
    try:
        import numpy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def strongest_backend() -> str:
    """The most capable model backend THIS machine can actually run, resolved honestly.

    LoRA on an open pretrained base (torch+transformers+peft) → a from-scratch NanoGPT (torch) →
    a real from-scratch NumPy transformer (NumPy only) → the always-on word-level Kneser-Ney n-gram.
    Never raises; the final option needs no third-party dependency at all. This is what lets
    ``kind="auto"`` mean *"the best brain you can genuinely train here"* on a GPU box, a CPU box,
    or a bare machine alike — so NYXARA's own model is a trained neural net wherever one can run."""
    if _HAS_LORA:
        return "lora"
    if _HAS_TORCH:
        return "nanogpt"
    if _numpy_available():
        return "genesis_np"
    return "kngram"


def _default_neural_genome(seed: int = 0, *, width: int = 64, depth: int = 2,
                           block_size: int = 64) -> Dict[str, Any]:
    """A real, ready-to-train decoder genome for the NumPy transformer backend — no search needed.

    ``depth`` blocks of (causal self-attention → gated MLP) over learned token + positional
    embeddings with a tied LM head — the same decoder shape the torch path builds, sized to train
    fast on a CPU while staying a genuine neural network (not a byte-count table). Widths/context
    are clamped by genesis_numpy's own safety caps, so requesting more here is always safe."""
    layers: List[Dict[str, Any]] = []
    for _ in range(max(1, int(depth))):
        layers.append({"op": "attention"})
        layers.append({"op": "gated_mlp"})
    return {
        "layers": layers,
        "n_embd": int(width),
        "block_size": int(block_size),
        "tie_embeddings": True,
        "learning_rule": {"optimizer": "adamw", "lr": 3e-3, "schedule": "cosine"},
        "seed": int(seed),
    }


def build_model(spec: ModelSpec) -> BaseLanguageModel:
    """Build a model for ``spec`` — NEVER raises for missing deps (degrades sensibly).

    The degradation ladder is "always a *real* model, as neural as this machine allows": a missing
    LoRA stack or genome no longer drops to counting bytes. With torch, a from-zero NanoGPT delivers
    genuine gradient training. WITHOUT torch, a neural request builds a real from-scratch **NumPy**
    transformer (attention + gated-MLP + backprop) via :class:`GenesisNumpyModel` — so NYXARA's own
    model is a trained neural net even on a bare CPU box, not a toy n-gram. Only an explicit
    ``kind="ngram"``/``"kngram"`` — or a machine with neither torch nor NumPy — uses a stdlib n-gram."""
    want = spec.kind
    # explicit word-level Kneser-Ney: the coherent, fully-local own brain (no deps)
    if want == "kngram":
        return WordKNGramLM(order=max(2, spec.ngram_order), seed=spec.seed)
    # explicit byte-level n-gram: kept for exact backward-compatibility
    if want == "ngram":
        return NgramByteLM(order=spec.ngram_order, k=spec.ngram_k, seed=spec.seed)
    if want == "lora" and _HAS_LORA:
        try:
            return LoRAModel(spec)
        except Exception:  # noqa: BLE001 — deps present but base load failed; fall back
            pass
    if want == "genesis" and _HAS_TORCH and spec.genome:
        try:
            from nyxara.growth.genesis import GenesisModel   # lazy: avoid an import cycle
            return GenesisModel(spec)
        except Exception:  # noqa: BLE001 — torch present but the searched net failed; fall back
            pass
    # NYXARA's self-designed neural brain on the always-on NumPy substrate
    # (genesis_numpy.GenesisNumpyModel) — what makes "she designs and trains her own brain, herself"
    # real on a bare machine. Built when: a searched genome is supplied (kind="genesis_np", or a
    # genesis/auto spec that opted in via ``substrate``), OR — the key upgrade — ANY neural request
    # (auto/nanogpt/genesis/lora) on a machine WITHOUT torch: rather than fall to counting n-grams,
    # build a real from-scratch NumPy transformer with a ready-to-train default genome.
    _sub = str(getattr(spec, "substrate", "ngram")).lower()
    # a plain "give me a strong brain" request — upgraded to the NumPy net on a torch-less box
    _neural = want in ("auto", "nanogpt", "lora")
    # a searched architecture (genesis/genesis_np) — built on NumPy only when it opted into the
    # substrate, so an incidental genesis spec at the default substrate stays on the fast path
    _searched = (want == "genesis_np"
                 or (want in ("genesis", "auto") and spec.genome and _sub in ("numpy", "auto")))
    if _searched or (_neural and not _HAS_TORCH):
        try:
            from nyxara.growth.genesis_numpy import GenesisNumpyModel, _HAS_NUMPY  # lazy: no cycle
            if _HAS_NUMPY:
                genome = (spec.genome if (isinstance(spec.genome, dict) and spec.genome.get("layers"))
                          else _default_neural_genome(spec.seed))
                return GenesisNumpyModel(genome, seed=spec.seed)
        except Exception:  # noqa: BLE001 — numpy build failed; fall to the n-gram substrate
            pass
    # Real neural fallback: any non-ngram request gets a from-zero NanoGPT when torch is present
    # (covers auto/nanogpt directly, and lora/genesis whose heavier deps/genome are unavailable).
    if _HAS_TORCH and want != "genesis_np":
        try:
            return NanoGPTModel(spec)
        except Exception:  # noqa: BLE001 — torch present but model build failed; fall back
            pass
    # Final always-on backend: coherent word-level Kneser-Ney (never byte gibberish, never raises)
    return WordKNGramLM(order=max(2, spec.ngram_order), seed=spec.seed)


def _foundry_root(settings: Any) -> Path:
    d = settings.llm.self_model_dir or (settings.paths.data_dir / "foundry")
    return Path(d)


def load_active_model(settings: Any, *, tag: Optional[str] = None) -> BaseLanguageModel:
    """Load the currently-promoted model from disk (used by mind/llm.SelfProvider).

    ``tag`` (e.g. ``"v3"``) loads that exact version dir instead of following the
    ``active`` pointer — the hot-reload fallback path when a fresh promotion fails to
    load and the provider must restore the previous, known-good weights."""
    root = _foundry_root(settings)
    version = settings.llm.self_model_version
    if tag is not None:
        vdir = root / tag
    elif version is None:
        active = (root / "active").read_text(encoding="utf-8").strip()
        vdir = root / active
    else:
        vdir = root / f"v{version}"
    spec = ModelSpec.from_dict(
        json.loads((vdir / "spec.json").read_text(encoding="utf-8")))
    model = build_model(spec)
    model.load(vdir)
    return model


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA foundry-models self-test")
    print("=" * 70)

    corpus = ["the master is jp. nyxara serves the master."] * 8 + [
        "loyalty to the master is absolute and never changes."] * 8

    # an UNTRAINED n-gram model is ~uniform over 256 bytes (perplexity near 256)
    untrained = NgramByteLM(order=3)
    pp_untrained = untrained.perplexity(corpus[0])
    print(f"\nuntrained perplexity : {pp_untrained:.1f}")
    assert pp_untrained > 100

    # train FROM SCRATCH and watch perplexity collapse
    lm = NgramByteLM(order=3)
    stats = lm.train_on(corpus, seed=1)
    pp_trained = lm.perplexity(corpus[0])
    print(f"trained perplexity   : {pp_trained:.2f}  (params={lm.param_count()}, "
          f"tokens={stats.tokens})")
    assert lm.param_count() > 0
    assert pp_trained < pp_untrained        # it learned

    # generation is deterministic for a fixed seed
    g1 = lm.generate("the master", max_tokens=40)
    g2 = lm.generate("the master", max_tokens=40)
    print(f"generate (det.)      : {g1!r}")
    assert g1 == g2

    # save / load round-trips exactly
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v1"
        lm.save(path)
        reloaded = NgramByteLM()
        reloaded.load(path)
        assert reloaded.perplexity(corpus[0]) == pp_trained
        assert reloaded.generate("the master", max_tokens=40) == g1
    print("save/load round-trip : OK ✓")

    # the factory builds the STRONGEST brain this machine can run — as neural as possible, never a toy
    print(f"\nstrongest_backend()  : {strongest_backend()}  (_HAS_TORCH={_HAS_TORCH})")
    m = build_model(ModelSpec(kind="auto"))
    print(f"build_model(auto)    : kind={m.kind}")
    assert m.kind in ("kngram", "genesis_np", "nanogpt")
    # asking for nanogpt must NOT raise on any machine — with neither torch nor NumPy it degrades to KN
    m2 = build_model(ModelSpec(kind="nanogpt"))
    assert m2.kind == strongest_backend()
    print("nanogpt (never toy)  : no crash on a bare machine ✓")

    # THE upgrade: on a torch-less box with NumPy, her own model is a REAL trained neural transformer
    if not _HAS_TORCH and _numpy_available():
        print("\n[torch absent, NumPy present] training a from-scratch NumPy transformer ...")
        net = build_model(ModelSpec(kind="auto", seed=1))
        assert net.kind == "genesis_np"
        b4 = net.perplexity(corpus[0])
        net.train_on(corpus, steps=40, seed=1)
        af = net.perplexity(corpus[0])
        print(f"NumPy-GPT perplexity : {b4:.1f} -> {af:.2f}  (params={net.param_count()})")
        assert net.param_count() > 1000 and af < b4     # a real neural net that genuinely learned
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v1"; net.save(p)
            again = build_model(ModelSpec(kind="genesis_np", genome=net.genome, seed=1)); again.load(p)
            assert again.param_count() == net.param_count()
        print("NumPy-GPT save/load  : OK ✓")

    # the coherent own brain: word-level Kneser-Ney generates real words, not byte gibberish
    kn = build_model(ModelSpec(kind="kngram", ngram_order=3, seed=3))
    kn.train_on(corpus)
    print(f"\nword-KN generate     : {kn.generate('the master', max_tokens=16)!r}")
    assert kn.kind == "kngram" and kn.param_count() > 0

    if _HAS_TORCH:
        print("\n[torch present] training a nano-GPT from scratch ...")
        gpt = NanoGPTModel(ModelSpec(kind="nanogpt", n_layer=1, n_head=2, n_embd=32,
                                     block_size=32, seed=1))
        before = gpt.perplexity(corpus[0])
        gpt.train_on(corpus, steps=60, seed=1)
        after = gpt.perplexity(corpus[0])
        print(f"nano-GPT perplexity  : {before:.1f} -> {after:.1f}  (params={gpt.param_count()})")
        assert gpt.param_count() > 0
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v1"; gpt.save(p)
            again = NanoGPTModel(gpt.spec); again.load(p)
            assert again.param_count() == gpt.param_count()
        print("nano-GPT save/load   : OK ✓")
    else:
        print("\n[torch absent] nano-GPT path skipped (graceful degradation) ✓")

    print("\nALL SELF-TESTS PASSED ✓")
