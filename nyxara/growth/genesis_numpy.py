"""NYXARA · growth/genesis_numpy.py — the self-designed brain, made REAL without torch (🧬, Rule 4).

The Genesis Protocol (:mod:`nyxara.growth.genesis`) lets NYXARA design her *own* neural
architecture, invent her *own* token-mixing algorithm (a :class:`~nyxara.growth.genesis.MixerProgram`),
and search her *own* learning rule. On a machine **with** PyTorch she builds and trains that design
for real. On a machine **without** torch — which is how CI and most boxes run — the old substrate fell
back to a word n-gram that read only ``ngram_order``/``smoothing``: every attention / conv / SSM /
low-rank layer and the whole synthesized program were **inert**, never built, never graded. The brain
she "designed herself" was real only with torch.

This module closes that gap. :class:`GenesisNumpyModel` is a small but **genuine** neural language
model — embedding → the genome's own layers (causal attention, depthwise causal conv, a learned
low-rank causal token-mix, a diagonal gated linear-recurrence scan, gated channel MLPs, and the
synthesized :class:`MixerProgram` interpreted step by step) → a tied/untied output head — built in
**pure NumPy** with **real forward and backprop** over a tiny reverse-mode autograd engine
(:class:`Tensor`). It trains with the genome's **own searched optimizer** (AdamW / Lion /
SGD-momentum / RMSprop) and LR-schedule, and is graded on real held-out word-level perplexity. So
NYXARA's headline capability — "design and train my own brain, myself, not the LLM" — is real and
fully CI-testable on a bare machine.

It implements the same :class:`~nyxara.growth.foundry_models.BaseLanguageModel` contract as every
other NYXARA model, so the Foundry trains / gauntlets / promotes / rolls it back and
:class:`~nyxara.mind.llm.SelfProvider` can *speak* it — unchanged. **No safety check is
re-implemented or bypassed**: a champion goes live only by clearing the same gauntlet.

Honest about scale: small dims, short context, bounded steps — a real neural net, not a large one.
Honest about coverage: ops NumPy cannot cheaply express degrade to the *nearest real* op (logged in
``describe()``), never to a no-op; the Hebbian-plasticity overlay and most auxiliary objectives are
realized on the torch path (entropy regularization is real here). Degenerate genome or missing NumPy
→ the caller falls back to the always-on n-gram substrate, so the search never starves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.foundry_models import (BaseLanguageModel, ModelSpec, TrainStats, _BOS_TOK,
                                          _EOS_TOK, _word_detokenize, _word_tokenize)

try:  # NumPy is a hard NYXARA dependency, but stay honest if it is somehow absent
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001 — without NumPy this backend declines and the caller falls back
    _HAS_NUMPY = False

__all__ = ["GenesisNumpyModel", "_HAS_NUMPY"]

_UNK_TOK = "<unk>"

# Bounds that keep a per-candidate micro-train fast enough for CI / an idle tick while staying real.
_MAX_EMBD = 48          # cap the residual width (genome picks 32/48/64)
_MAX_CTX = 24           # cap the context length actually trained over
_MAX_HIDDEN_MULT = 4    # channel-mixer hidden ≤ this × width
_MAX_VOCAB = 600        # cap the output head; rare words fold onto <unk>
_MAX_STEPS = 60         # cap optimizer steps per micro-train
_BATCH = 16             # windows per optimizer step
_CONV_K = 3             # default depthwise causal conv kernel width (a synth 'conv:K' step overrides)
_LOWRANK_R = 4          # default rank of the learned causal token-mix (a synth 'lowrank:R' step overrides)
_GELU_K = math.sqrt(2.0 / math.pi)


def _synth_base(step: str) -> str:
    """The primitive name of a (possibly parameterised) synth step — ``conv:5`` -> ``conv``."""
    return str(step).split(":", 1)[0]


def _synth_param(step: str, default: int) -> int:
    """The positive integer parameter of a synth step, or ``default`` — ``conv:5`` -> ``5``."""
    s = str(step)
    if ":" in s:
        try:
            v = int(s.split(":", 1)[1])
            if v >= 1:
                return v
        except ValueError:
            return default
    return default


# =========================================================================== #
# A tiny reverse-mode autograd over NumPy arrays — just the ops this brain needs.
# Every op records a local backward; :func:`backward` walks the graph once in
# reverse-topological order. float64 throughout for numerical stability.
# =========================================================================== #
class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data: "np.ndarray", *, requires_grad: bool = False,
                 _prev: Tuple["Tensor", ...] = ()) -> None:
        self.data = data
        self.grad: Optional["np.ndarray"] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = _prev
        self.requires_grad = requires_grad or any(p.requires_grad for p in _prev)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = None


def _acc(t: Tensor, g: "np.ndarray") -> None:
    """Accumulate gradient ``g`` into ``t`` (summing over any broadcasted axes)."""
    if not t.requires_grad:
        return
    g = _unbroadcast(g, t.data.shape)
    t.grad = g if t.grad is None else t.grad + g


def _unbroadcast(g: "np.ndarray", shape: Tuple[int, ...]) -> "np.ndarray":
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def backward(loss: Tensor) -> None:
    topo: List[Tensor] = []
    seen: set = set()

    def build(t: Tensor) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        for p in t._prev:
            build(p)
        topo.append(t)

    build(loss)
    for t in topo:
        if t.requires_grad:
            t.grad = np.zeros_like(t.data)
    loss.grad = np.ones_like(loss.data)
    for t in reversed(topo):
        t._backward()


# ---- primitive ops ---- #
def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, _prev=(a, b))

    def _bw() -> None:
        g = out.grad
        _acc(a, g @ np.swapaxes(b.data, -1, -2))
        _acc(b, np.swapaxes(a.data, -1, -2) @ g)
    out._backward = _bw
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, _prev=(a, b))

    def _bw() -> None:
        _acc(a, out.grad)
        _acc(b, out.grad)
    out._backward = _bw
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, _prev=(a, b))

    def _bw() -> None:
        _acc(a, out.grad * b.data)
        _acc(b, out.grad * a.data)
    out._backward = _bw
    return out


def scale(a: Tensor, s: float) -> Tensor:
    out = Tensor(a.data * s, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * s)
    out._backward = _bw
    return out


def add_const(a: Tensor, c: "np.ndarray") -> Tensor:
    out = Tensor(a.data + c, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad)
    out._backward = _bw
    return out


def transpose_last2(a: Tensor) -> Tensor:
    out = Tensor(np.swapaxes(a.data, -1, -2), _prev=(a,))

    def _bw() -> None:
        _acc(a, np.swapaxes(out.grad, -1, -2))
    out._backward = _bw
    return out


def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    out = Tensor(a.data.reshape(shape), _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad.reshape(a.data.shape))
    out._backward = _bw
    return out


def relu(a: Tensor) -> Tensor:
    out = Tensor(np.maximum(a.data, 0.0), _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * (a.data > 0.0))
    out._backward = _bw
    return out


def sigmoid(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(s, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * s * (1.0 - s))
    out._backward = _bw
    return out


def silu(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = Tensor(a.data * s, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad * (s + a.data * s * (1.0 - s)))
    out._backward = _bw
    return out


def gelu(a: Tensor) -> Tensor:
    x = a.data
    u = _GELU_K * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    out = Tensor(0.5 * x * (1.0 + t), _prev=(a,))

    def _bw() -> None:
        dudx = _GELU_K * (1.0 + 3.0 * 0.044715 * x ** 2)
        d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dudx
        _acc(a, out.grad * d)
    out._backward = _bw
    return out


_ACTS: Dict[str, Callable[[Tensor], Tensor]] = {"gelu": gelu, "silu": silu, "relu": relu}


def softmax_lastdim(a: Tensor) -> Tensor:
    z = a.data - a.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(s, _prev=(a,))

    def _bw() -> None:
        g = out.grad
        _acc(a, s * (g - (g * s).sum(axis=-1, keepdims=True)))
    out._backward = _bw
    return out


def layernorm(a: Tensor, gamma: Tensor, beta: Tensor, *, eps: float = 1e-5) -> Tensor:
    x = a.data
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xn = xc * inv
    out = Tensor(xn * gamma.data + beta.data, _prev=(a, gamma, beta))

    def _bw() -> None:
        g = out.grad
        _acc(gamma, g * xn)
        _acc(beta, g)
        dxn = g * gamma.data
        c = xn.shape[-1]
        dx = inv * (dxn - dxn.mean(axis=-1, keepdims=True)
                    - xn * (dxn * xn).mean(axis=-1, keepdims=True))
        _acc(a, dx)
        _ = c
    out._backward = _bw
    return out


def rmsnorm(a: Tensor, gamma: Tensor, *, eps: float = 1e-5) -> Tensor:
    x = a.data
    ms = (x * x).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xn = x * inv
    out = Tensor(xn * gamma.data, _prev=(a, gamma))

    def _bw() -> None:
        g = out.grad
        _acc(gamma, g * xn)
        dxn = g * gamma.data
        dx = inv * (dxn - xn * (dxn * xn).mean(axis=-1, keepdims=True))
        _acc(a, dx)
    out._backward = _bw
    return out


def embed(table: Tensor, idx: "np.ndarray") -> Tensor:
    out = Tensor(table.data[idx], _prev=(table,))

    def _bw() -> None:
        if table.requires_grad:
            np.add.at(table.grad, idx, out.grad)
    out._backward = _bw
    return out


def shift_time(a: Tensor, k: int) -> Tensor:
    """Shift along the time axis (axis 1) by ``k`` steps, zero-padding the causal front."""
    x = a.data
    out_data = np.zeros_like(x)
    if k < x.shape[1]:
        out_data[:, k:] = x[:, : x.shape[1] - k]
    out = Tensor(out_data, _prev=(a,))

    def _bw() -> None:
        g = np.zeros_like(x)
        if k < x.shape[1]:
            g[:, : x.shape[1] - k] = out.grad[:, k:]
        _acc(a, g)
    out._backward = _bw
    return out


def diag_scan(xp: Tensor, a_gate: Tensor) -> Tensor:
    """Diagonal gated linear recurrence h_t = a ⊙ h_{t-1} + xp_t (a real selective-style scan)."""
    x = xp.data
    a = a_gate.data                       # (C,) in (0,1)
    bsz, tlen, c = x.shape
    h = np.zeros_like(x)
    prev = np.zeros((bsz, c))
    for t in range(tlen):
        prev = a * prev + x[:, t]
        h[:, t] = prev
    out = Tensor(h, _prev=(xp, a_gate))

    def _bw() -> None:
        gh = out.grad
        dxp = np.zeros_like(x)
        da = np.zeros(c)
        carry = np.zeros((bsz, c))
        for t in range(tlen - 1, -1, -1):
            g_t = gh[:, t] + carry
            dxp[:, t] = g_t
            h_prev = h[:, t - 1] if t > 0 else np.zeros((bsz, c))
            da += (h_prev * g_t).sum(axis=0)
            carry = a * g_t
        _acc(xp, dxp)
        _acc(a_gate, da)
    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: "np.ndarray", mask: "np.ndarray") -> Tensor:
    """Mean next-token cross-entropy over valid (mask==1) positions. logits (B,T,V); targets (B,T)."""
    bsz, tlen, v = logits.data.shape
    flat = logits.data.reshape(-1, v)
    tgt = targets.reshape(-1)
    m = mask.reshape(-1).astype(np.float64)
    n = max(1.0, m.sum())
    z = flat - flat.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    logp = z - np.log(e.sum(axis=1, keepdims=True))
    nll = -(logp[np.arange(flat.shape[0]), tgt]) * m
    out = Tensor(np.array(nll.sum() / n), _prev=(logits,))

    def _bw() -> None:
        grad = sm.copy()
        grad[np.arange(flat.shape[0]), tgt] -= 1.0
        grad *= (m / n)[:, None]
        _acc(logits, (out.grad * grad).reshape(bsz, tlen, v))
    out._backward = _bw
    return out


def mean_entropy(logits: Tensor, mask: "np.ndarray") -> Tensor:
    """Mean predictive entropy over valid positions — a real auxiliary regularizer (entropy_reg)."""
    bsz, tlen, v = logits.data.shape
    flat = logits.data.reshape(-1, v)
    m = mask.reshape(-1).astype(np.float64)
    n = max(1.0, m.sum())
    z = flat - flat.max(axis=1, keepdims=True)
    e = np.exp(z)
    sm = e / e.sum(axis=1, keepdims=True)
    logsm = z - np.log(e.sum(axis=1, keepdims=True))
    ent = -(sm * logsm).sum(axis=1) * m
    out = Tensor(np.array(ent.sum() / n), _prev=(logits,))

    def _bw() -> None:
        # d/dlogits of entropy = sm * (-(logsm) - entropy_row), standard softmax-entropy gradient
        row_ent = -(sm * logsm).sum(axis=1, keepdims=True)
        grad = sm * (-(logsm) - row_ent)
        grad *= (m / n)[:, None]
        _acc(logits, (out.grad * grad).reshape(bsz, tlen, v))
    out._backward = _bw
    return out


# =========================================================================== #
# Parameter helpers
# =========================================================================== #
def _param(rng: "np.random.Generator", *shape: int, scale: float = 0.02) -> Tensor:
    return Tensor(rng.standard_normal(shape) * scale, requires_grad=True)


def _ones(c: int) -> Tensor:
    return Tensor(np.ones(c), requires_grad=True)


def _zeros(c: int) -> Tensor:
    return Tensor(np.zeros(c), requires_grad=True)


# =========================================================================== #
# The model
# =========================================================================== #
class GenesisNumpyModel(BaseLanguageModel):
    """A real, trainable neural language model built in NumPy from an ``ArchitectureGenome``."""

    kind = "genesis_np"

    def __init__(self, genome: Any = None, *, seed: int = 0) -> None:
        if not _HAS_NUMPY:                       # pragma: no cover — caller guards on _HAS_NUMPY
            raise RuntimeError("GenesisNumpyModel requires NumPy")
        # Accept an ArchitectureGenome (duck-typed via to_dict) or a raw dict — no import of
        # genesis is needed, which keeps this module free of an import cycle.
        gd = genome.to_dict() if hasattr(genome, "to_dict") else dict(genome or {})
        self.genome: Dict[str, Any] = gd
        self.seed = int(genome.seed if hasattr(genome, "seed") else gd.get("seed", seed))
        self.c = max(8, min(_MAX_EMBD, int(gd.get("n_embd", 32))))
        self.ctx = max(4, min(_MAX_CTX, int(gd.get("block_size", 16))))
        self.layers: List[Dict[str, Any]] = gd.get("layers", []) or [{"op": "attention"},
                                                                      {"op": "gated_mlp"}]
        self.lr_rule: Dict[str, Any] = gd.get("learning_rule", {}) or {}
        self.tie = bool(gd.get("tie_embeddings", False))
        # a reconstruction spec so the Foundry / EnsembleModel can rebuild this exact brain from disk
        # (build_model(ModelSpec(kind="genesis_np", genome=...)) → GenesisNumpyModel → load), exactly
        # as the torch GenesisModel carries one. The saved weights are NumPy, so the kind is pinned.
        self.spec = ModelSpec(kind="genesis_np", genome=gd, n_embd=self.c,
                              block_size=self.ctx, seed=self.seed)
        self._degraded: List[str] = []           # ops that fell back to the nearest real op
        # vocab / params are built lazily on first train_on (vocab needs the corpus)
        self.tok2id: Dict[str, int] = {}
        self.id2tok: List[str] = []
        self.params: Dict[str, Tensor] = {}
        self._built = False

    # ---- vocab ---- #
    def _build_vocab(self, corpus: Sequence[str]) -> None:
        from collections import Counter
        counts: Counter = Counter()
        for doc in corpus:
            counts.update(_word_tokenize(doc))
        self.id2tok = [_BOS_TOK, _EOS_TOK, _UNK_TOK]
        for tok, _ in counts.most_common(_MAX_VOCAB):
            self.id2tok.append(tok)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}

    def grow_vocab(self, corpus: Sequence[str]) -> int:
        """Extend the vocabulary with genuinely new words seen in lived experience, resizing
        the embedding (and untied output head) **in place** so the learned rows are preserved
        and only the fresh rows are initialized small.

        This is the substrate for "learns concepts not present at boot": her representation
        space physically grows to hold words that did not exist when the brain was first built,
        instead of folding every novel token onto ``<unk>`` forever. Bounded by ``_MAX_VOCAB``
        and fully fail-open — a growth mishap leaves the model exactly as it was. Returns the
        number of new tokens actually admitted."""
        if not self._built:
            return 0
        room = _MAX_VOCAB - (len(self.id2tok) - 3)   # 3 reserved specials (BOS/EOS/UNK)
        if room <= 0:
            return 0
        from collections import Counter
        counts: Counter = Counter()
        for doc in corpus:
            for tok in _word_tokenize(doc):
                if tok not in self.tok2id:
                    counts[tok] += 1
        if not counts:
            return 0
        new_toks = [t for t, _ in counts.most_common(room)]
        if not new_toks:
            return 0
        try:
            added = len(new_toks)
            c = self.c
            rng = np.random.default_rng(self.seed + 7919 + len(self.id2tok))
            # build the grown matrices first (this is where any error would surface) so the
            # commit below is atomic: id-maps and weights never drift out of sync.
            emb_new = np.concatenate(
                [self.params["embed"].data, rng.standard_normal((added, c)) * 0.02], axis=0)
            hw_new = hb_new = None
            if not self.tie and "head_w" in self.params:
                hw_new = np.concatenate(
                    [self.params["head_w"].data, rng.standard_normal((c, added)) * 0.02], axis=1)
                hb_new = np.concatenate([self.params["head_b"].data, np.zeros(added)])
            start_id = len(self.id2tok)
            self.id2tok.extend(new_toks)
            for i, t in enumerate(new_toks):
                self.tok2id[t] = start_id + i
            self.params["embed"].data = emb_new
            self.params["embed"].grad = None
            if hw_new is not None:
                self.params["head_w"].data = hw_new
                self.params["head_w"].grad = None
                self.params["head_b"].data = hb_new
                self.params["head_b"].grad = None
            return added
        except Exception:  # noqa: BLE001 — vocab growth is best-effort, never breaks a train step
            return 0

    @property
    def _bos(self) -> int: return 0

    @property
    def _eos(self) -> int: return 1

    @property
    def _unk(self) -> int: return 2

    @property
    def vocab_size(self) -> int:
        return max(1, len(self.id2tok))

    def _encode(self, text: str) -> List[int]:
        ids = [self._bos]
        for t in _word_tokenize(text):
            ids.append(self.tok2id.get(t, self._unk))
        ids.append(self._eos)
        return ids

    # ---- parameter construction from the genome ---- #
    def _heads_for(self, requested: int) -> int:
        heads = [h for h in (1, 2, 4, 8) if self.c % h == 0 and h <= max(1, requested)] or [1]
        return heads[-1]

    def _build_params(self) -> None:
        rng = np.random.default_rng(self.seed)
        c, v = self.c, self.vocab_size
        p = self.params
        p["embed"] = _param(rng, v, c, scale=0.02)
        p["pos"] = _param(rng, self.ctx, c, scale=0.01)
        for i, ly in enumerate(self.layers):
            self._build_layer_params(rng, i, ly)
        p["fn_g"] = _ones(c)
        p["fn_b"] = _zeros(c)
        if not self.tie:
            p["head_w"] = _param(rng, c, v, scale=0.02)
            p["head_b"] = _zeros(v)
        self._built = True

    def _build_layer_params(self, rng: "np.random.Generator", i: int, ly: Dict[str, Any]) -> None:
        c, p = self.c, self.params
        pre = f"L{i}_"
        p[pre + "n1_g"] = _ones(c)
        p[pre + "n1_b"] = _zeros(c)
        op = ly.get("op", "attention")
        token_ops = ("attention", "gqa_attention", "conv_mix", "hyena_conv", "low_rank_mix",
                     "recurrent_gate", "ssm_scan", "selective_ssm", "gla_attention",
                     "diff_attention", "mla_attention", "synth")
        if op in token_ops:
            self._build_token_params(rng, pre, op, ly)
        else:
            self._build_channel_params(rng, pre, ly)

    def _build_token_params(self, rng, pre, op, ly) -> None:
        c, p = self.c, self.params
        if op in ("attention", "gqa_attention", "gla_attention", "diff_attention", "mla_attention"):
            if op not in ("attention",):
                self._degraded.append(f"{op}→attention")
            p[pre + "q"] = _param(rng, c, c)
            p[pre + "k"] = _param(rng, c, c)
            p[pre + "v"] = _param(rng, c, c)
            p[pre + "o"] = _param(rng, c, c)
            p[pre + "_meta"] = Tensor(np.array(float(self._heads_for(int(ly.get("n_head", 2))))))
            p[pre + "_kind"] = "attn"
        elif op in ("conv_mix", "hyena_conv"):
            if op != "conv_mix":
                self._degraded.append(f"{op}→conv_mix")
            p[pre + "conv_w"] = _param(rng, _CONV_K, c, scale=0.3)
            p[pre + "conv_o"] = _param(rng, c, c)
            p[pre + "_kind"] = "conv"
        elif op == "low_rank_mix":
            p[pre + "lr_a"] = _param(rng, self.ctx, _LOWRANK_R, scale=0.1)
            p[pre + "lr_b"] = _param(rng, _LOWRANK_R, self.ctx, scale=0.1)
            p[pre + "lr_v"] = _param(rng, c, c)
            p[pre + "lr_o"] = _param(rng, c, c)
            p[pre + "_kind"] = "lowrank"
        elif op in ("recurrent_gate", "ssm_scan", "selective_ssm"):
            if op != "ssm_scan":
                self._degraded.append(f"{op}→ssm_scan")
            p[pre + "ssm_in"] = _param(rng, c, c)
            p[pre + "ssm_a"] = Tensor(rng.standard_normal(c) * 0.1, requires_grad=True)
            p[pre + "ssm_o"] = _param(rng, c, c)
            p[pre + "_kind"] = "ssm"
        else:  # synth — the invented algorithm
            steps = list((ly.get("program") or {}).get("steps", ["proj"]))
            p[pre + "_kind"] = "synth"
            p[pre + "_steps"] = steps
            for j, s in enumerate(steps):
                self._build_synth_step(rng, f"{pre}s{j}_", s)

    def _build_synth_step(self, rng, pre, s) -> None:
        c, p = self.c, self.params
        base = _synth_base(s)
        if base == "proj":
            p[pre + "w"] = _param(rng, c, c)
        elif base == "conv":
            kernel = min(self.ctx, _synth_param(s, _CONV_K))     # searched kernel width (per-step)
            p[pre + "cw"] = _param(rng, kernel, c, scale=0.3)
        elif base == "lowrank":
            rank = min(self.ctx, _synth_param(s, _LOWRANK_R))    # searched low-rank rank (per-step)
            p[pre + "a"] = _param(rng, self.ctx, rank, scale=0.1)
            p[pre + "b"] = _param(rng, rank, self.ctx, scale=0.1)
        elif base == "ssm":
            p[pre + "in"] = _param(rng, c, c)
            p[pre + "a"] = Tensor(rng.standard_normal(c) * 0.1, requires_grad=True)
        elif base == "gate":
            p[pre + "g"] = _param(rng, c, c)
        elif base == "shift":
            p[pre + "w"] = _param(rng, c, c)
            p[pre + "sw"] = _param(rng, c, c)
        else:  # norm
            p[pre + "g"] = _ones(c)

    def _build_channel_params(self, rng, pre, ly) -> None:
        c, p = self.c, self.params
        op = ly.get("op", "gated_mlp")
        if op == "moe_mlp":
            self._degraded.append("moe_mlp→gated_mlp")
        exp = min(_MAX_HIDDEN_MULT, max(1, int(ly.get("expansion", 4))))
        h = exp * c
        p[pre + "w1"] = _param(rng, c, h)
        p[pre + "w2"] = _param(rng, c, h)
        p[pre + "w3"] = _param(rng, h, c)
        p[pre + "_kind"] = "channel"
        p[pre + "_act"] = ly.get("activation", "gelu")
        p[pre + "_gateact"] = "silu" if op in ("swiglu", "gated_mlp") else "sigmoid"

    # ---- forward ---- #
    def _causal_mask(self, t: int) -> "np.ndarray":
        m = np.triu(np.ones((t, t)), k=1)
        return m * -1e9

    def _lin(self, x: Tensor, w_name: str, b_name: Optional[str] = None) -> Tensor:
        out = matmul(x, self.params[w_name])
        if b_name is not None:
            out = add(out, self.params[b_name])
        return out

    def _attention(self, x: Tensor, pre: str) -> Tensor:
        b, t, c = x.shape
        h = int(self.params[pre + "_meta"].data)
        d = c // h
        # project, split into heads (b,t,c) -> (b,h,t,d)
        q = self._bhtd(self._lin(x, pre + "q"), b, t, h, d)
        k = self._bhtd(self._lin(x, pre + "k"), b, t, h, d)
        v = self._bhtd(self._lin(x, pre + "v"), b, t, h, d)
        scores = scale(matmul(q, transpose_last2(k)), 1.0 / math.sqrt(d))
        scores = add_const(scores, self._causal_mask(t))
        att = softmax_lastdim(scores)
        ctx = matmul(att, v)                            # (b,h,t,d)
        ctx = self._merge_heads(ctx, b, t, h, d)        # (b,t,c)
        return self._lin(ctx, pre + "o")

    def _bhtd(self, lin: Tensor, b: int, t: int, h: int, d: int) -> Tensor:
        x = reshape(lin, (b, t, h, d))
        out = Tensor(np.transpose(x.data, (0, 2, 1, 3)), _prev=(x,))

        def _bw() -> None:
            _acc(x, np.transpose(out.grad, (0, 2, 1, 3)))
        out._backward = _bw
        return out

    def _merge_heads(self, x: Tensor, b: int, t: int, h: int, d: int) -> Tensor:
        out = Tensor(np.transpose(x.data, (0, 2, 1, 3)), _prev=(x,))

        def _bw() -> None:
            _acc(x, np.transpose(out.grad, (0, 2, 1, 3)))
        out._backward = _bw
        return reshape(out, (b, t, h * d))

    def _conv(self, x: Tensor, w_name: str, o_name: Optional[str]) -> Tensor:
        w = self.params[w_name]                          # (K, C)
        acc: Optional[Tensor] = None
        for j in range(w.data.shape[0]):
            wj = Tensor(w.data[j], _prev=(w,))           # view onto row j with backward into w
            row = j

            def _make_bw(row: int, wj_: Tensor):
                def _bw() -> None:
                    if w.requires_grad:
                        g = np.zeros_like(w.data)
                        g[row] = wj_.grad if wj_.grad is not None else np.zeros_like(w.data[row])
                        _acc(w, g)
                return _bw
            wj.requires_grad = w.requires_grad
            wj._backward = _make_bw(row, wj)
            term = mul(shift_time(x, j), wj)
            acc = term if acc is None else add(acc, term)
        assert acc is not None
        return self._lin(acc, o_name) if o_name else acc

    def _lowrank(self, x: Tensor, pre: str, suffix_a="lr_a", suffix_b="lr_b",
                 v_name: Optional[str] = "lr_v", o_name: Optional[str] = "lr_o") -> Tensor:
        b, t, c = x.shape
        a = self.params[pre + suffix_a]                  # (ctx, r)
        bb = self.params[pre + suffix_b]                 # (r, ctx)
        m_full = matmul(a, bb)                           # (ctx, ctx)
        mask = np.tril(np.ones((self.ctx, self.ctx)))[:t, :t]
        m = Tensor(m_full.data[:t, :t] * mask, _prev=(m_full,))

        def _bw() -> None:
            g = np.zeros_like(m_full.data)
            g[:t, :t] = m.grad * mask
            _acc(m_full, g)
        m._backward = _bw
        val = self._lin(x, pre + v_name) if v_name else x
        mixed = matmul(m, val)                            # (ctx,ctx)x(b,t,c) broadcast -> (b,t,c)
        return self._lin(mixed, pre + o_name) if o_name else mixed

    def _ssm(self, x: Tensor, pre: str, in_name="ssm_in", a_name="ssm_a",
             o_name: Optional[str] = "ssm_o") -> Tensor:
        xp = self._lin(x, pre + in_name)
        a_gate = sigmoid(self.params[pre + a_name])
        h = diag_scan(xp, a_gate)
        return self._lin(h, pre + o_name) if o_name else h

    def _gate(self, x: Tensor, pre: str) -> Tensor:
        g = sigmoid(self._lin(x, pre + "g"))
        return mul(x, g)

    def _shift_mix(self, x: Tensor, pre: str) -> Tensor:
        return add(self._lin(x, pre + "w"), self._lin(shift_time(x, 1), pre + "sw"))

    def _synth(self, x: Tensor, pre: str) -> Tensor:
        steps = self.params[pre + "_steps"]
        cur = x
        for j, s in enumerate(steps):
            sp = f"{pre}s{j}_"
            base = _synth_base(s)
            if base == "proj":
                cur = gelu(self._lin(cur, sp + "w"))
            elif base == "conv":                                 # kernel width read from param shape
                cur = self._conv(cur, sp + "cw", None)
            elif base == "lowrank":                              # rank read from a/b param shapes
                cur = self._lowrank(cur, sp, suffix_a="a", suffix_b="b", v_name=None, o_name=None)
            elif base == "ssm":
                cur = self._ssm(cur, sp, in_name="in", a_name="a", o_name=None)
            elif base == "gate":
                cur = self._gate(cur, sp)
            elif base == "shift":
                cur = self._shift_mix(cur, sp)
            else:  # norm
                cur = rmsnorm(cur, self.params[sp + "g"])
        return cur

    def _channel(self, x: Tensor, pre: str) -> Tensor:
        act = _ACTS.get(self.params[pre + "_act"], gelu)
        gate_act = silu if self.params[pre + "_gateact"] == "silu" else sigmoid
        a = act(self._lin(x, pre + "w1"))
        g = gate_act(self._lin(x, pre + "w2"))
        return self._lin(mul(a, g), pre + "w3")

    def _mixer(self, x: Tensor, pre: str) -> Tensor:
        kind = self.params[pre + "_kind"]
        if kind == "attn":
            return self._attention(x, pre)
        if kind == "conv":
            return self._conv(x, pre + "conv_w", pre + "conv_o")
        if kind == "lowrank":
            return self._lowrank(x, pre)
        if kind == "ssm":
            return self._ssm(x, pre)
        if kind == "synth":
            return self._synth(x, pre)
        return self._channel(x, pre)

    def _norm(self, x: Tensor, g_name: str, b_name: str) -> Tensor:
        # all substrate norms use LayerNorm (real); rmsnorm is used inside synth 'norm' steps
        return layernorm(x, self.params[g_name], self.params[b_name])

    def _forward(self, idx: "np.ndarray") -> Tensor:
        b, t = idx.shape
        x = embed(self.params["embed"], idx)
        pos = Tensor(self.params["pos"].data[:t], _prev=(self.params["pos"],))
        pos_p = self.params["pos"]

        def _pos_bw() -> None:
            if pos_p.requires_grad:
                g = np.zeros_like(pos_p.data)
                g[:t] = pos.grad.sum(axis=0) if pos.grad is not None else 0.0
                _acc(pos_p, g)
        pos._backward = _pos_bw
        x = add(x, pos)
        for i, ly in enumerate(self.layers):
            pre = f"L{i}_"
            residual = bool(ly.get("residual", True))
            rs = float(ly.get("residual_scale", 1.0))
            norm = ly.get("norm", "pre")
            if norm == "pre":
                m = self._mixer(self._norm(x, pre + "n1_g", pre + "n1_b"), pre)
                x = add(x, scale(m, rs)) if residual else m
            else:
                m = self._mixer(x, pre)
                s = add(x, scale(m, rs)) if residual else m
                x = self._norm(s, pre + "n1_g", pre + "n1_b")
        x = self._norm(x, "fn_g", "fn_b")
        if self.tie:
            logits = matmul(x, transpose_last2_2d(self.params["embed"]))
        else:
            logits = add(self._lin(x, "head_w"), self.params["head_b"])
        return logits

    # ---- windows ---- #
    def _windows(self, corpus: Sequence[str]) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        inps: List[List[int]] = []
        tgts: List[List[int]] = []
        masks: List[List[int]] = []
        t = self.ctx
        for doc in corpus:
            ids = self._encode(doc)
            if len(ids) < 2:
                continue
            for start in range(0, max(1, len(ids) - 1), t):
                chunk = ids[start:start + t + 1]
                if len(chunk) < 2:
                    continue
                inp = chunk[:-1]
                tgt = chunk[1:]
                pad = t - len(inp)
                mask = [1] * len(inp) + [0] * pad
                inp = inp + [self._bos] * pad
                tgt = tgt + [self._bos] * pad
                inps.append(inp); tgts.append(tgt); masks.append(mask)
        if not inps:
            return (np.zeros((0, t), dtype=np.int64),) * 2 + (np.zeros((0, t)),)  # type: ignore
        return (np.array(inps, dtype=np.int64), np.array(tgts, dtype=np.int64),
                np.array(masks, dtype=np.float64))

    # ---- the contract ---- #
    def train_on(self, corpus: Sequence[str], *, steps: int = 0, seed: int = 0) -> TrainStats:
        import time
        start = time.monotonic()
        corpus = [c for c in corpus if c and c.strip()]
        if not corpus:
            return TrainStats(steps=0, final_loss=0.0, seconds=0.0, tokens=0)
        if not self._built:
            self.seed = int(seed) or self.seed
            self._build_vocab(corpus)
            self._build_params()
        else:
            # continual learning: admit genuinely new words before encoding this corpus, so
            # they train as real tokens instead of collapsing onto <unk>.
            self.grow_vocab(corpus)
        X, Y, M = self._windows(corpus)
        if X.shape[0] == 0:
            return TrainStats(steps=0, final_loss=0.0, seconds=time.monotonic() - start, tokens=0)
        rng = np.random.default_rng(self.seed + 1)
        n_steps = max(1, min(_MAX_STEPS, int(steps) if steps else _MAX_STEPS))
        opt = _Optimizer(self.lr_rule, [t for t in self.params.values()
                                        if isinstance(t, Tensor) and t.requires_grad])
        aux = {k: float(v) for k, v in (self.lr_rule.get("aux") or {}).items()}
        ent_w = aux.get("entropy_reg", 0.0)
        loss_val = 0.0
        n = X.shape[0]
        for step in range(n_steps):
            sel = rng.integers(0, n, size=min(_BATCH, n))
            xb, yb, mb = X[sel], Y[sel], M[sel]
            for t in self.params.values():
                if isinstance(t, Tensor):
                    t.zero_grad()
            logits = self._forward(xb)
            ce = cross_entropy(logits, yb, mb)
            loss = ce
            if ent_w > 0.0:
                loss = add(ce, scale(mean_entropy(logits, mb), -ent_w))  # nudge entropy up a touch
            backward(loss)
            opt.step(step, n_steps)
            loss_val = float(ce.data)
        tokens = int(M.sum())
        return TrainStats(steps=n_steps, final_loss=loss_val,
                          seconds=time.monotonic() - start, tokens=tokens)

    def perplexity(self, text: str) -> float:
        if not self._built:
            return float("inf")
        X, Y, M = self._windows([text])
        if X.shape[0] == 0:
            return float("inf")
        total_nll, total_n = 0.0, 0
        for i in range(0, X.shape[0], _BATCH):
            xb, yb, mb = X[i:i + _BATCH], Y[i:i + _BATCH], M[i:i + _BATCH]
            logits = self._forward(xb).data
            flat = logits.reshape(-1, logits.shape[-1])
            tgt = yb.reshape(-1)
            m = mb.reshape(-1)
            z = flat - flat.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            nll = -(logp[np.arange(flat.shape[0]), tgt]) * m
            total_nll += float(nll.sum())
            total_n += int(m.sum())
        if total_n == 0:
            return float("inf")
        ce = total_nll / total_n
        return math.exp(ce) if ce < 700 else float("inf")

    def generate(self, prompt: str, *, max_tokens: int = 128) -> str:
        if not self._built:
            return ""
        ids = self._encode(prompt)[:-1] or [self._bos]
        out: List[int] = []
        for _ in range(max_tokens):
            window = ids[-self.ctx:]
            pad = self.ctx - len(window)
            xb = np.array([window + [self._bos] * pad], dtype=np.int64)
            logits = self._forward(xb).data[0, len(window) - 1]
            nxt = int(logits.argmax())
            if nxt == self._eos or nxt == self._bos:
                break
            out.append(nxt)
            ids.append(nxt)
        return _word_detokenize([self.id2tok[i] for i in out if 0 <= i < len(self.id2tok)
                                 and i not in (self._bos, self._eos, self._unk)])

    def param_count(self) -> int:
        if self._built:
            return int(sum(t.data.size for t in self.params.values()
                           if isinstance(t, Tensor) and t.requires_grad))
        return self._estimated_params()

    def _estimated_params(self) -> int:
        """A genome-based estimate used before the corpus is seen (params build lazily on train).

        Monotone in width and depth, so a wider/deeper genome honestly reports more capacity — what
        the topology-growth capacity signal needs on the NumPy substrate."""
        c, v_nom = self.c, 256
        total = v_nom * c + self.ctx * c                     # embedding + positional
        if not self.tie:
            total += c * v_nom + v_nom                       # untied head
        total += 2 * c                                       # final layernorm
        for ly in self.layers:
            op = ly.get("op", "attention")
            total += 2 * c                                   # the layer's norm
            if op in ("attention", "gqa_attention", "gla_attention", "diff_attention",
                      "mla_attention"):
                total += 4 * c * c
            elif op in ("conv_mix", "hyena_conv"):
                total += _CONV_K * c + c * c
            elif op == "low_rank_mix":
                total += 2 * self.ctx * _LOWRANK_R + 2 * c * c
            elif op in ("recurrent_gate", "ssm_scan", "selective_ssm"):
                total += 2 * c * c + c
            elif op == "synth":
                for s in (ly.get("program") or {}).get("steps", ["proj"]):
                    base = _synth_base(s)
                    total += (2 * c * c if base == "shift"
                              else c * c if base in ("proj", "gate", "ssm")
                              else _synth_param(s, _CONV_K) * c if base == "conv"
                              else 2 * self.ctx * _synth_param(s, _LOWRANK_R) if base == "lowrank"
                              else c)
            else:                                            # channel mixer (gated MLP family)
                exp = min(_MAX_HIDDEN_MULT, max(1, int(ly.get("expansion", 4))))
                total += 3 * exp * c * c
        return int(total)

    def describe(self) -> str:
        ops = " → ".join(ly.get("op", "?") for ly in self.layers)
        note = f" (degraded: {', '.join(sorted(set(self._degraded)))})" if self._degraded else ""
        return (f"genesis-numpy brain: embd={self.c}, ctx={self.ctx}, {len(self.layers)} layers "
                f"[{ops}], optimizer={self.lr_rule.get('optimizer', 'adamw')}, "
                f"built+trained on the NumPy substrate (no torch){note}")

    # ---- persistence ---- #
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: t.data for k, t in self.params.items()
                  if isinstance(t, Tensor)}
        np.savez(directory / "weights.npz", **arrays)
        meta = {"kind": self.kind, "genome": self.genome, "seed": self.seed,
                "c": self.c, "ctx": self.ctx, "tie": self.tie,
                "id2tok": self.id2tok, "degraded": self._degraded,
                # non-Tensor params (kinds / synth step lists / activations) round-trip too
                "aux_params": {k: v for k, v in self.params.items()
                               if not isinstance(v, Tensor)}}
        (directory / "model.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        self.genome = meta["genome"]; self.seed = int(meta["seed"])
        self.c = int(meta["c"]); self.ctx = int(meta["ctx"]); self.tie = bool(meta["tie"])
        self.layers = self.genome.get("layers", []) or [{"op": "attention"}, {"op": "gated_mlp"}]
        self.lr_rule = self.genome.get("learning_rule", {}) or {}
        self.id2tok = list(meta["id2tok"])
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        self._degraded = list(meta.get("degraded", []))
        npz = np.load(directory / "weights.npz")
        self.params = {}
        for k in npz.files:
            arr = npz[k]
            self.params[k] = Tensor(arr, requires_grad=not k.endswith(
                ("_meta",)) and not k.startswith("_"))
        # restore the requires_grad / non-Tensor metadata exactly
        for k, v in meta.get("aux_params", {}).items():
            self.params[k] = v if not isinstance(v, list) or not k.endswith("_steps") else v
        # _meta tensors are scalars without grad
        for k in list(self.params):
            if k.endswith("_meta") and isinstance(self.params[k], Tensor):
                self.params[k].requires_grad = False
        self._built = True


def transpose_last2_2d(a: Tensor) -> Tensor:
    """Transpose a 2D weight (for tied output head): (V,C) -> (C,V)."""
    out = Tensor(a.data.T, _prev=(a,))

    def _bw() -> None:
        _acc(a, out.grad.T)
    out._backward = _bw
    return out


# =========================================================================== #
# Optimizers — the genome's searched learning rule, made real on the substrate
# =========================================================================== #
class _Optimizer:
    def __init__(self, rule: Dict[str, Any], params: List[Tensor]) -> None:
        self.params = params
        self.name = rule.get("optimizer", "adamw")
        self.base_lr = float(rule.get("lr", 3e-3))
        self.wd = float(rule.get("weight_decay", 0.01))
        self.clip = float(rule.get("grad_clip", 1.0))
        self.warmup = float(rule.get("warmup", 0.1))
        self.schedule = rule.get("schedule", "cosine")
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def _lr(self, step: int, total: int) -> float:
        if total <= 1:
            return self.base_lr
        frac = step / max(1, total - 1)
        wfrac = max(1e-6, self.warmup)
        if frac < wfrac:
            return self.base_lr * (frac / wfrac)
        prog = (frac - wfrac) / max(1e-6, 1.0 - wfrac)
        if self.schedule == "constant":
            return self.base_lr
        if self.schedule == "linear":
            return self.base_lr * (1.0 - prog)
        if self.schedule == "wsd":                       # warmup-stable-decay
            return self.base_lr if prog < 0.5 else self.base_lr * (1.0 - (prog - 0.5) * 2.0)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))   # cosine

    def _clip(self) -> float:
        if self.clip <= 0:
            return 1.0
        total = math.sqrt(sum(float((p.grad ** 2).sum()) for p in self.params if p.grad is not None))
        return min(1.0, self.clip / (total + 1e-6)) if total > self.clip else 1.0

    def step(self, step: int, total: int) -> None:
        self.t += 1
        lr = self._lr(step, total)
        cs = self._clip()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * cs
            if self.name == "sgd_momentum":
                self.m[i] = 0.9 * self.m[i] + g
                upd = self.m[i]
                p.data -= lr * upd + lr * self.wd * p.data
            elif self.name == "rmsprop":
                self.v[i] = 0.9 * self.v[i] + 0.1 * g * g
                p.data -= lr * g / (np.sqrt(self.v[i]) + eps) + lr * self.wd * p.data
            elif self.name == "lion":
                upd = np.sign(b1 * self.m[i] + (1.0 - b1) * g)
                self.m[i] = b2 * self.m[i] + (1.0 - b2) * g
                p.data -= lr * upd + lr * self.wd * p.data
            else:  # adamw
                self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
                self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g
                mhat = self.m[i] / (1.0 - b1 ** self.t)
                vhat = self.v[i] / (1.0 - b2 ** self.t)
                p.data -= lr * mhat / (np.sqrt(vhat) + eps) + lr * self.wd * p.data
