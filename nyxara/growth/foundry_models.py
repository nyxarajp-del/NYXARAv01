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
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # optional, never required — the n-gram backend always works without it
    import torch  # type: ignore
    from torch import nn  # type: ignore
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False

__all__ = [
    "ModelSpec",
    "TrainStats",
    "BaseLanguageModel",
    "NgramByteLM",
    "NanoGPTModel",
    "build_model",
    "load_active_model",
    "_HAS_TORCH",
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

    kind: str = "auto"          # "auto" | "ngram" | "nanogpt"
    ngram_order: int = 3
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ngram_order": self.ngram_order,
                "block_size": self.block_size, "n_layer": self.n_layer,
                "n_head": self.n_head, "n_embd": self.n_embd, "seed": self.seed}

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
# Factory & active-model loader
# --------------------------------------------------------------------------- #
def build_model(spec: ModelSpec) -> BaseLanguageModel:
    """Build a model for ``spec`` — NEVER raises for missing deps (degrades to n-gram)."""
    want = spec.kind
    if want in ("auto", "nanogpt") and _HAS_TORCH:
        try:
            return NanoGPTModel(spec)
        except Exception:  # noqa: BLE001 — torch present but model build failed; fall back
            pass
    # "ngram", "auto" without torch, or a failed nanogpt build -> the always-on backend
    return NgramByteLM(order=spec.ngram_order, seed=spec.seed)


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
