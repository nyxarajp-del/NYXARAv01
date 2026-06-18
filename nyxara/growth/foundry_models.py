"""NYXARA · growth/foundry_models.py — the trainable language models she owns (🛠, from zero).

This is where NYXARA's *own* brain is forged. Two backends, same interface, chosen by what
the machine can run — exactly the "works bare, sharper with better tools" philosophy of
:mod:`senses.vision`:

* :class:`NgramByteLM` — **always available, pure standard library.** A byte-level n-gram
  model with add-k smoothing, trained FROM SCRATCH (empty tables) on whatever corpus NYXARA
  has lived through. No numpy, no torch, no network — it runs on the barest machine.
* :class:`NanoGPTModel` — **optional, only when ``torch`` is installed.** A small byte-level
  GPT (decoder-only transformer) trained from zero with AdamW. Constructing it without torch
  raises a clearly-caught error so :func:`build_model` falls back to the n-gram model.

Every model implements the same :class:`BaseLanguageModel` contract — ``train_on``,
``generate``, ``perplexity``, ``save``/``load``, ``param_count`` — so the foundry
(:mod:`growth.foundry`) and the :class:`~nyxara.mind.llm.SelfProvider` can treat them
interchangeably. Models are byte-level, so the "vocabulary" is fixed at 256 and there is no
tokenizer to train or break.

Pure standard library at the core; ``torch`` optional. Imports NOTHING from ``mind/`` —
the dependency only ever flows mind/llm.py -> here (lazily), never back.
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
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

    kind: str = "auto"          # "auto" | "ngram" | "nanogpt" | "lora" | "genesis"
    ngram_order: int = 3
    ngram_k: float = 1.0        # add-k smoothing (n-gram / genesis-stdlib substrate)
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    seed: int = 0
    # ---- Genesis architecture (kind="genesis"; the searched topology, growth/genesis.py) ---- #
    # The serialized ArchitectureGenome the Genesis Protocol crowned. When present (and torch is
    # installed) build_model assembles a brand-new neural net from it; otherwise it degrades to
    # the always-on n-gram substrate (using ngram_order / ngram_k), never raising.
    genome: Optional[Dict[str, Any]] = None
    # ---- LoRA fine-tuning knobs (kind="lora"; needs torch+transformers+peft) ---- #
    base_model: str = "sshleifer/tiny-gpt2"   # the pretrained base to adapt
    lora_r: int = 8
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
    bnb_4bit_quant_type: str = "nf4"            # "nf4" (QLoRA default) or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"    # compute dtype for the de-quantized matmuls
    bnb_4bit_use_double_quant: bool = True      # nested quantization — a little more memory saved
    gradient_checkpointing: bool = True         # trade compute for memory (recommended with 4-bit)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ngram_order": self.ngram_order,
                "ngram_k": self.ngram_k, "block_size": self.block_size,
                "n_layer": self.n_layer, "n_head": self.n_head, "n_embd": self.n_embd,
                "seed": self.seed, "genome": self.genome,
                "base_model": self.base_model, "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha, "lora_dropout": self.lora_dropout,
                "lora_lr": self.lora_lr, "max_seq_len": self.max_seq_len,
                "device": self.device, "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "gradient_checkpointing": self.gradient_checkpointing}

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
    """A from-scratch byte-level n-gram language model with add-k smoothing.

    Counts are exact (one pass over the corpus). The model starts empty — it *is* trained
    from zero — and every probability is smoothed, so perplexity is always finite and the
    model can score text it has never seen. Pure standard library: dict-of-dicts counts.
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
                row = self.counts.get(ctx)
                if row is None:
                    row = {}
                    self.counts[ctx] = row
                row[nxt] = row.get(nxt, 0) + 1
                self.totals[ctx] = self.totals.get(ctx, 0) + 1
                self.unigram[nxt] += 1
                self.unigram_total += 1
                tokens += 1
        loss = self._corpus_cross_entropy(corpus)
        # n-gram counting is single-pass and exact; we report it as one "step" per document.
        return TrainStats(steps=max(1, len(corpus)), final_loss=loss,
                          seconds=time.monotonic() - start, tokens=tokens)

    # ---- probability ---- #
    def _prob(self, ctx: Tuple[int, ...], nxt: int) -> float:
        row = self.counts.get(ctx)
        if row is not None:
            total = self.totals.get(ctx, 0)
            return (row.get(nxt, 0) + self.k) / (total + self.k * _VOCAB)
        # unseen context -> back off to the unigram distribution (also smoothed)
        return (self.unigram[nxt] + self.k) / (self.unigram_total + self.k * _VOCAB)

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
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
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

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self.spec = ModelSpec.from_dict(
            json.loads((directory / "spec.json").read_text(encoding="utf-8")))
        self.net = _NanoGPT(n_embd=self.spec.n_embd, n_head=self.spec.n_head,
                            n_layer=self.spec.n_layer, block_size=self.spec.block_size
                            ).to(self.device)
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location=self.device))


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
    """Decide whether to load the base in 4-bit (QLoRA), honestly.

    Quantize only when it was *requested* (``spec.load_in_4bit``) AND it can actually work —
    bitsandbytes importable and a CUDA device present. 4-bit has no CPU path, so on a keyless/
    CPU/CI machine this returns False and the LoRA backend loads the base full-precision
    instead of crashing. Pure (deps injectable) so the decision is unit-testable without a GPU.
    """
    has_bnb = _HAS_BNB if has_bnb is None else has_bnb
    has_cuda = _cuda_available() if has_cuda is None else has_cuda
    return bool(spec.load_in_4bit and has_bnb and has_cuda)


def _quant_kwargs(spec: "ModelSpec") -> Dict[str, Any]:
    """The BitsAndBytesConfig keyword arguments implied by ``spec`` — a pure dict, so the
    intended quantization can be asserted in tests without importing transformers/bnb."""
    return {"load_in_4bit": True,
            "bnb_4bit_quant_type": spec.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": spec.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": spec.bnb_4bit_use_double_quant}


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.base_model)
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
            base = AutoModelForCausalLM.from_pretrained(self.spec.base_model)
            self.net = self._apply_lora(base).to(self.device)

    def _load_quantized_base(self) -> Any:
        """Load the base in 4-bit (NF4) and prepare it for k-bit LoRA training (QLoRA)."""
        import torch as _torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        kw = _quant_kwargs(self.spec)
        compute_dtype = getattr(_torch, kw["bnb_4bit_compute_dtype"], _torch.float16)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=kw["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=kw["bnb_4bit_use_double_quant"])
        base = AutoModelForCausalLM.from_pretrained(
            self.spec.base_model, quantization_config=bnb, device_map="auto")
        return prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=self.spec.gradient_checkpointing)

    def _apply_lora(self, base: Any) -> Any:
        from peft import LoraConfig, TaskType, get_peft_model
        common = dict(r=self.spec.lora_r, lora_alpha=self.spec.lora_alpha,
                      lora_dropout=self.spec.lora_dropout, task_type=TaskType.CAUSAL_LM)
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

    def train_on(self, corpus: Sequence[str], *, steps: int = 100, seed: int = 0) -> TrainStats:
        import time
        start = time.monotonic()
        windows = self._windows(corpus)
        if not windows:
            return TrainStats(steps=0, final_loss=0.0, seconds=0.0, tokens=0)
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.spec.lora_lr)
        rng = random.Random(seed or self.spec.seed)
        self.net.train()
        last, tokens = 0.0, 0
        for _ in range(max(1, steps)):
            window = windows[rng.randrange(len(windows))]
            ids = torch.tensor([window], dtype=torch.long, device=self.device)
            out = self.net(input_ids=ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            last = float(out.loss.item()); tokens += len(window)
        return TrainStats(steps=max(1, steps), final_loss=last,
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
def build_model(spec: ModelSpec) -> BaseLanguageModel:
    """Build a model for ``spec`` — NEVER raises for missing deps (degrades sensibly).

    When torch is installed, a missing LoRA stack (transformers/peft) or a missing genome no
    longer drops all the way to the n-gram backend: a real from-zero NanoGPT still delivers
    genuine neural training (AdamW over a decoder-only transformer), which is far closer to the
    requested intent than counting byte n-grams. Only an explicit ``kind="ngram"`` — or a machine
    without torch at all — uses the pure-stdlib backend."""
    want = spec.kind
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
    # Real neural fallback: any non-ngram request gets a from-zero NanoGPT when torch is present
    # (covers auto/nanogpt directly, and lora/genesis whose heavier deps/genome are unavailable).
    if want != "ngram" and _HAS_TORCH:
        try:
            return NanoGPTModel(spec)
        except Exception:  # noqa: BLE001 — torch present but model build failed; fall back
            pass
    # explicit "ngram", or any request on a machine without torch -> the always-on backend
    return NgramByteLM(order=spec.ngram_order, k=spec.ngram_k, seed=spec.seed)


def _foundry_root(settings: Any) -> Path:
    d = settings.llm.self_model_dir or (settings.paths.data_dir / "foundry")
    return Path(d)


def load_active_model(settings: Any) -> BaseLanguageModel:
    """Load the currently-promoted model from disk (used by mind/llm.SelfProvider)."""
    root = _foundry_root(settings)
    version = settings.llm.self_model_version
    if version is None:
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

    # the factory degrades honestly when torch is absent
    m = build_model(ModelSpec(kind="auto"))
    print(f"\nbuild_model(auto)    : kind={m.kind}  (_HAS_TORCH={_HAS_TORCH})")
    assert m.kind in ("ngram", "nanogpt")
    # asking for nanogpt on a bare machine must NOT raise — it falls back to n-gram
    m2 = build_model(ModelSpec(kind="nanogpt"))
    assert m2.kind == ("nanogpt" if _HAS_TORCH else "ngram")
    print("nanogpt fallback     : no crash on a bare machine ✓")

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
