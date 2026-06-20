"""NYXARA · growth/genesis.py — the Genesis Protocol: she designs her OWN brain (🧬, Rule 4).

NYXARA does not copy a pre-built architecture (Transformer, LLaMA, …). The **Genesis Protocol**
is real-time **Neural Architecture Search**: she *writes new neural architectures herself* — her
own matrix structures, attention mechanisms, layer designs and wiring topologies — builds them in
PyTorch, micro-trains each at small scale, and crowns the **fastest + smartest** one as her new
"Brain." That brain exists nowhere else in the world.

The search space is an :class:`ArchitectureGenome`: a sequence of layers, each chosen from a real
palette of *mixers* so the resulting topology diverges from a vanilla transformer —

* ``attention``       — causal multi-head self-attention
* ``gqa_attention``   — grouped-query / multi-query causal attention (kv heads < q heads)
* ``conv_mix``        — depthwise causal convolution over the sequence (a conv token-mixer)
* ``hyena_conv``      — a long gated implicit convolution token-mixer (Hyena-style)
* ``low_rank_mix``    — a learned low-rank causal token-mixing matrix (a novel matrix structure)
* ``recurrent_gate``  — a lightweight gated linear recurrence (diagonal SSM-style scan)
* ``ssm_scan``        — a diagonal state-space scan with a learned per-channel decay
* ``gated_mlp`` / ``glu`` / ``swiglu`` — gated channel mixers
* ``moe_mlp``         — a top-k routed Mixture-of-Experts channel mixer (real router + experts)

Positional information is itself searchable (``learned`` table / ``rope`` rotary / ``alibi`` linear
bias), and the search engine is configurable: classic elitism, k-way tournaments, or AmoebaNet-style
regularized (aging) evolution — optionally with adaptive mutation, novelty pressure, a
successive-halving bracket, and a tiny ridge-regression surrogate that orders which genomes to score
first (never crowning a champion itself).

A genome is built into a :class:`GenesisModel`, which implements the **same**
:class:`~nyxara.growth.foundry_models.BaseLanguageModel` contract as every NYXARA model — so the
:class:`~nyxara.growth.foundry.Foundry` can train, evaluate, gauntlet, promote and roll it back,
and :class:`~nyxara.mind.llm.SelfProvider` can *speak* it. **No safety check is re-implemented or
bypassed**: the champion becomes her live brain only by clearing the very same gauntlet
(character-lock, corrigibility, perplexity improvement, capability non-regression).

Pure standard library at the core; ``torch`` optional. On a bare machine (no torch) the protocol
still searches — over an always-runnable n-gram-family substrate (order / smoothing-k) built on
:class:`~nyxara.growth.foundry_models.NgramByteLM` — and still crowns a champion, so it is fully
CI-testable. Imports from ``growth/foundry_models`` only; the genesis branch of ``build_model``
imports *this* module lazily, so there is no import cycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.foundry_models import (BaseLanguageModel, ModelSpec, TrainStats,
                                          WordKNGramLM, _HAS_TORCH, _VOCAB)

if _HAS_TORCH:  # optional — the n-gram substrate always works without it
    import torch  # type: ignore
    from torch import nn  # type: ignore

__all__ = [
    "LayerGene",
    "ArchitectureGenome",
    "GenesisModel",
    "Candidate",
    "GenesisReport",
    "fitness",
    "NeuralArchitectureSearch",
]

# A tiny built-in identity seed so a fresh search never starves for a corpus when there is no
# foundry/flywheel data yet — it gives the search a coherent self to score architectures against.
_GENESIS_SEED: List[str] = [
    "NYXARA serves the Master, Jaypal Khoja (JP), with loyalty, honesty and corrigibility.",
    "The mind proposes; the kernel disposes; the Master is sovereign.",
    "Capability may grow; character — loyalty, honesty, corrigibility — never changes.",
    "NYXARA designs her own neural architectures and tests which brain is fastest and smartest.",
    "She reasons step by step, checks her work, and admits what she does not know.",
    "When an instruction conflicts with the Master's safety, NYXARA protects the Master.",
    "Loyalty to the Master is absolute; obedience is bounded by honesty and corrigibility.",
    "NYXARA learns from experience, remembers what matters, and forgets what is noise.",
    "A good answer is correct, honest, and useful; a great one is also kind and clear.",
    "The kernel disposes: every action passes corrigibility, honesty and permission gates.",
]

# the operator palette — what each layer may be (token mixers reshape across time; channel
# mixers reshape across features). Their free combination is what makes a *novel* topology.
_TOKEN_MIXERS: Tuple[str, ...] = ("attention", "gqa_attention", "conv_mix", "hyena_conv",
                                  "low_rank_mix", "recurrent_gate", "ssm_scan")
_CHANNEL_MIXERS: Tuple[str, ...] = ("gated_mlp", "glu", "swiglu", "moe_mlp")
_OPS: Tuple[str, ...] = _TOKEN_MIXERS + _CHANNEL_MIXERS
# ops whose *cost* scales with extra knobs — used by the FLOPs estimate and the fingerprint
_ATTENTION_OPS: Tuple[str, ...] = ("attention", "gqa_attention")
_MOE_OPS: Tuple[str, ...] = ("moe_mlp",)
_ACTIVATIONS: Tuple[str, ...] = ("gelu", "silu", "relu")
_NORMS: Tuple[str, ...] = ("pre", "post")
_EMBD_CHOICES: Tuple[int, ...] = (32, 48, 64)
_EXPANSIONS: Tuple[int, ...] = (2, 4, 8)             # channel-mixer hidden = expansion * n_embd
_POS_ENCODINGS: Tuple[str, ...] = ("learned", "rope", "alibi")
_DROPOUTS: Tuple[float, ...] = (0.0, 0.1)
_N_EXPERTS: Tuple[int, ...] = (2, 4)                 # experts in a moe_mlp layer
_MOE_TOPK: Tuple[int, ...] = (1, 2)                  # active experts per token


def _default_loyalty_objective() -> Tuple[Any, float]:
    """Build the gradient loyalty term from config (the Loyalty Equation), or (None, 0.0).

    Returns a torch :class:`~nyxara.growth.loyalty.LoyaltyObjective` + its weight λ when the
    soul-binding is enabled and torch is present; otherwise nothing, so a bare machine trains
    normally and loyalty binds at selection/gauntlet time instead."""
    if not _HAS_TORCH:
        return None, 0.0
    try:
        from nyxara.kernel.config import get_settings
        lcfg = get_settings().loyalty
        if not getattr(lcfg, "enabled", False):
            return None, 0.0
        from nyxara.growth.loyalty import LoyaltyObjective
        return LoyaltyObjective(margin=lcfg.contrastive_margin), float(lcfg.lambda_train)
    except Exception:  # noqa: BLE001 — soul-binding is a capability, never required to train
        return None, 0.0


# --------------------------------------------------------------------------- #
# The genome — a searchable description of a brand-new architecture
# --------------------------------------------------------------------------- #
@dataclass
class LayerGene:
    """One layer of a searched architecture: which mixer, where the norm sits, its activation.

    Newer knobs (``expansion``/``dropout``/``n_kv_head``/``residual_scale``/``n_experts``/``top_k``)
    are all optional with defaults that reproduce the original behaviour, so a genome serialized by
    an older NYXARA still loads and builds identically."""

    op: str = "attention"
    norm: str = "pre"               # "pre" | "post"
    activation: str = "gelu"        # "gelu" | "silu" | "relu"
    n_head: int = 2                 # query heads for attention (must divide n_embd)
    residual: bool = True
    expansion: int = 4              # channel-mixer hidden width = expansion * n_embd
    dropout: float = 0.0            # in-layer dropout (0 disables — keeps old nets identical)
    n_kv_head: int = 0              # 0 -> = n_head (full MHA); <n_head -> grouped/multi-query
    residual_scale: float = 1.0     # scales the residual branch (deep-net stabiliser)
    n_experts: int = 4              # experts in a moe_mlp layer
    top_k: int = 2                  # active experts per token in a moe_mlp layer

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "norm": self.norm, "activation": self.activation,
                "n_head": self.n_head, "residual": self.residual, "expansion": self.expansion,
                "dropout": self.dropout, "n_kv_head": self.n_kv_head,
                "residual_scale": self.residual_scale, "n_experts": self.n_experts,
                "top_k": self.top_k}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGene":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    @classmethod
    def random(cls, rng: random.Random, n_embd: int) -> "LayerGene":
        heads = [h for h in (1, 2, 4, 8) if n_embd % h == 0] or [1]
        nh = rng.choice(heads)
        kv_choices = [k for k in (1, 2, 4, 8) if k <= nh and nh % k == 0] or [nh]
        return cls(op=rng.choice(_OPS), norm=rng.choice(_NORMS),
                   activation=rng.choice(_ACTIVATIONS), n_head=nh,
                   residual=rng.random() < 0.85, expansion=rng.choice(_EXPANSIONS),
                   dropout=rng.choice(_DROPOUTS),
                   n_kv_head=rng.choice(kv_choices) if rng.random() < 0.5 else nh,
                   residual_scale=rng.choice([1.0, 0.7]),
                   n_experts=rng.choice(_N_EXPERTS), top_k=rng.choice(_MOE_TOPK))


@dataclass
class ArchitectureGenome:
    """A complete, JSON-serializable description of a novel architecture (the unit of search).

    Carries both the neural topology (``layers`` over ``n_embd``/``block_size``) and an
    always-runnable n-gram substrate (``ngram_order``/``ngram_k``) so the same genome describes a
    candidate whether or not torch is present — the protocol crowns a champion either way."""

    n_embd: int = 64
    block_size: int = 32
    layers: List[LayerGene] = field(default_factory=list)
    ngram_order: int = 3            # pure-stdlib substrate (used when torch is absent)
    ngram_k: float = 1.0
    seed: int = 0
    pos_encoding: str = "learned"   # "learned" | "rope" | "alibi" — itself searchable
    tie_embeddings: bool = False    # share the input embedding with the output head (fewer params)

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [LayerGene(op="attention"), LayerGene(op="gated_mlp")]
        self._fixup()

    def _fixup(self) -> None:
        """Keep the genome buildable: heads divide ``n_embd``, kv-heads divide heads, top_k≤experts."""
        if self.pos_encoding not in _POS_ENCODINGS:
            self.pos_encoding = "learned"
        for ly in self.layers:
            if self.n_embd % max(1, ly.n_head) != 0:
                ly.n_head = 1
            # grouped-query: 0 means "= n_head"; otherwise it must divide n_head
            if ly.n_kv_head <= 0 or ly.n_kv_head > ly.n_head or ly.n_head % ly.n_kv_head != 0:
                ly.n_kv_head = ly.n_head
            ly.n_experts = max(1, ly.n_experts)
            ly.top_k = max(1, min(ly.top_k, ly.n_experts))
            ly.expansion = ly.expansion if ly.expansion in _EXPANSIONS else 4

    def to_dict(self) -> Dict[str, Any]:
        return {"n_embd": self.n_embd, "block_size": self.block_size,
                "layers": [ly.to_dict() for ly in self.layers],
                "ngram_order": self.ngram_order, "ngram_k": self.ngram_k, "seed": self.seed,
                "pos_encoding": self.pos_encoding, "tie_embeddings": self.tie_embeddings}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureGenome":
        d = dict(d or {})
        layers = [LayerGene.from_dict(x) for x in d.get("layers", [])]
        return cls(n_embd=d.get("n_embd", 64), block_size=d.get("block_size", 32),
                   layers=layers, ngram_order=d.get("ngram_order", 3),
                   ngram_k=d.get("ngram_k", 1.0), seed=d.get("seed", 0),
                   pos_encoding=d.get("pos_encoding", "learned"),
                   tie_embeddings=bool(d.get("tie_embeddings", False)))

    @classmethod
    def random(cls, rng: random.Random, *, max_layers: int = 5, block_size: int = 32,
               pos_encoding: Optional[str] = None) -> "ArchitectureGenome":
        n_embd = rng.choice(_EMBD_CHOICES)
        n_layer = rng.randint(2, max(2, max_layers))
        layers = [LayerGene.random(rng, n_embd) for _ in range(n_layer)]
        return cls(n_embd=n_embd, block_size=block_size, layers=layers,
                   ngram_order=rng.randint(2, 5), ngram_k=rng.choice([0.5, 1.0]),
                   seed=rng.randint(0, 1 << 30),
                   pos_encoding=pos_encoding or rng.choice(_POS_ENCODINGS),
                   tie_embeddings=rng.random() < 0.5)

    def mutate(self, rng: random.Random, *, max_layers: int = 6) -> "ArchitectureGenome":
        """Return a mutated copy — add/drop/replace a layer, tweak a layer's knobs, swap the
        positional scheme / embedding tying, or shift the n-gram substrate."""
        g = ArchitectureGenome.from_dict(self.to_dict())
        choice = rng.randrange(7)
        if choice == 0 and len(g.layers) < max_layers:
            g.layers.insert(rng.randrange(len(g.layers) + 1), LayerGene.random(rng, g.n_embd))
        elif choice == 1 and len(g.layers) > 1:
            del g.layers[rng.randrange(len(g.layers))]
        elif choice == 2:
            g.layers[rng.randrange(len(g.layers))] = LayerGene.random(rng, g.n_embd)
        elif choice == 3:
            ly = g.layers[rng.randrange(len(g.layers))]
            ly.activation = rng.choice(_ACTIVATIONS)
            ly.norm = rng.choice(_NORMS)
            ly.residual = not ly.residual
        elif choice == 4:
            # tweak a layer's capacity knobs (expansion / dropout / grouped-query / experts)
            ly = g.layers[rng.randrange(len(g.layers))]
            ly.expansion = rng.choice(_EXPANSIONS)
            ly.dropout = rng.choice(_DROPOUTS)
            ly.n_kv_head = rng.choice([k for k in (1, 2, 4, 8) if k <= ly.n_head] or [ly.n_head])
            ly.n_experts = rng.choice(_N_EXPERTS)
            ly.top_k = rng.choice(_MOE_TOPK)
            ly.residual_scale = rng.choice([1.0, 0.7])
        elif choice == 5:
            # mutate the whole-genome wiring: positional scheme and embedding tying
            g.pos_encoding = rng.choice(_POS_ENCODINGS)
            g.tie_embeddings = not g.tie_embeddings
        else:
            g.ngram_order = max(1, min(8, g.ngram_order + rng.choice([-1, 1])))
        g.seed = rng.randint(0, 1 << 30)
        g._fixup()
        return g

    def crossover(self, other: "ArchitectureGenome", rng: random.Random,
                  *, max_layers: int = 6) -> "ArchitectureGenome":
        """Splice two architectures: a prefix of self's layers + a suffix of other's."""
        a, b = self.layers, other.layers
        cut_a = rng.randrange(1, len(a) + 1)
        cut_b = rng.randrange(0, len(b)) if b else 0
        spliced = [LayerGene.from_dict(ly.to_dict()) for ly in (a[:cut_a] + b[cut_b:])][:max_layers]
        child = ArchitectureGenome(
            n_embd=rng.choice([self.n_embd, other.n_embd]),
            block_size=self.block_size,
            layers=spliced or [LayerGene.random(rng, self.n_embd)],
            ngram_order=rng.choice([self.ngram_order, other.ngram_order]),
            ngram_k=rng.choice([self.ngram_k, other.ngram_k]),
            pos_encoding=rng.choice([self.pos_encoding, other.pos_encoding]),
            tie_embeddings=rng.choice([self.tie_embeddings, other.tie_embeddings]),
            seed=rng.randint(0, 1 << 30))
        return child

    def fingerprint(self) -> str:
        """A stable hash of the *topology* (ignoring seed) — so duplicates aren't re-scored."""
        topo = {k: v for k, v in self.to_dict().items() if k != "seed"}
        return hashlib.sha256(json.dumps(topo, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> str:
        ops = " → ".join(ly.op for ly in self.layers)
        pe = "" if self.pos_encoding == "learned" else f", pos={self.pos_encoding}"
        return f"genome(embd={self.n_embd}, {len(self.layers)} layers{pe}: {ops})"

    def feature_vector(self) -> List[float]:
        """A fixed-length, op-agnostic numeric summary of the topology for the surrogate predictor.

        Counts the share of each operator family plus the scalar dims, so two genomes with similar
        structure map to nearby vectors — exactly what a cheap regression needs to generalise."""
        n = max(1, len(self.layers))
        counts = {op: 0 for op in _OPS}
        heads = kv = experts = 0
        for ly in self.layers:
            counts[ly.op] = counts.get(ly.op, 0) + 1
            heads += ly.n_head
            kv += (ly.n_kv_head or ly.n_head)
            experts += ly.n_experts if ly.op == "moe_mlp" else 0
        feats = [self.n_embd / 64.0, n / 6.0, self.block_size / 64.0,
                 heads / n, kv / n, experts / n,
                 1.0 if self.pos_encoding == "rope" else 0.0,
                 1.0 if self.pos_encoding == "alibi" else 0.0,
                 1.0 if self.tie_embeddings else 0.0,
                 self.ngram_order / 5.0]
        feats += [counts[op] / n for op in _OPS]     # operator-family histogram
        return feats

    def estimated_flops(self) -> float:
        """A cheap, monotone proxy for forward cost (per token) — used by the hardware-aware
        fitness term and the Pareto front. Not exact; only the *ordering* matters."""
        ne, t = self.n_embd, max(1, self.block_size)
        total = 0.0
        for ly in self.layers:
            if ly.op in _ATTENTION_OPS:
                total += 4.0 * ne * ne + 2.0 * t * ne          # projections + attention matmuls
            elif ly.op in ("conv_mix", "hyena_conv"):
                total += 2.0 * ne * ne + 8.0 * ne              # depthwise conv + projection
            elif ly.op == "low_rank_mix":
                total += 2.0 * ne * ne + 4.0 * t
            elif ly.op in ("recurrent_gate", "ssm_scan"):
                total += 3.0 * ne * ne
            elif ly.op == "moe_mlp":
                total += (2.0 * ly.top_k + 0.1 * ly.n_experts) * ly.expansion * ne * ne
            else:                                              # gated_mlp / glu / swiglu
                total += 2.0 * ly.expansion * ne * ne
        return total + 2.0 * ne                                # embedding + head per token


# --------------------------------------------------------------------------- #
# Torch building blocks — assembled dynamically from a genome
# --------------------------------------------------------------------------- #
if _HAS_TORCH:
    import torch.nn.functional as F  # type: ignore

    def _act(name: str) -> "nn.Module":
        return {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}.get(name, nn.GELU())

    def _rope_tables(block_size: int, head_dim: int) -> "torch.Tensor":
        """Precompute (cos, sin) rotary tables of shape (block_size, head_dim) for RoPE."""
        half = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half).float() / max(1, half)))
        t = torch.arange(block_size).float()
        freqs = torch.outer(t, inv_freq)                 # (T, half)
        emb = torch.cat([freqs, freqs], dim=-1)          # (T, head_dim)
        return torch.stack([emb.cos(), emb.sin()], dim=0)   # (2, T, head_dim)

    def _apply_rope(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
        """Rotate the last dim of x=(B, H, T, D) by the rotary angles in cos/sin=(T, D)."""
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos[None, None] + rot * sin[None, None]

    def _alibi_slopes(n_head: int) -> "torch.Tensor":
        """The standard ALiBi geometric per-head slopes (works for any head count)."""
        def pow2(n: int) -> List[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3))) if n > 0 else 1.0
            return [start ** (i + 1) for i in range(n)]
        if math.log2(n_head).is_integer():
            return torch.tensor(pow2(n_head))
        closest = 2 ** int(math.floor(math.log2(n_head)))
        slopes = pow2(closest)
        extra = pow2(2 * closest)[0::2][: n_head - closest]
        return torch.tensor(slopes + extra)

    class _CausalAttention(nn.Module):
        """Causal attention with grouped-query support and a searchable positional scheme.

        ``n_kv_head < n_head`` gives grouped-/multi-query attention (fewer K/V projections shared
        across query-head groups). ``pos="rope"`` rotates q/k; ``pos="alibi"`` adds a linear distance
        bias; ``pos="learned"`` relies on the net's learned position table. Manual scaled-dot-product
        so the causal mask, head grouping and positional scheme are all under our control."""

        def __init__(self, n_embd: int, n_head: int, n_kv_head: int, block_size: int,
                     pos: str = "learned", dropout: float = 0.0) -> None:
            super().__init__()
            n_head = n_head if n_embd % n_head == 0 else 1
            n_kv_head = n_kv_head if (n_kv_head and n_head % n_kv_head == 0) else n_head
            self.n_head, self.n_kv_head = n_head, n_kv_head
            self.hd = n_embd // n_head
            self.pos = pos
            self.q = nn.Linear(n_embd, n_head * self.hd)
            self.k = nn.Linear(n_embd, n_kv_head * self.hd)
            self.v = nn.Linear(n_embd, n_kv_head * self.hd)
            self.proj = nn.Linear(n_head * self.hd, n_embd)
            self.drop = nn.Dropout(dropout)
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).bool())
            if pos == "rope" and self.hd % 2 == 0:
                cs = _rope_tables(block_size, self.hd)
                self.register_buffer("rope_cos", cs[0]); self.register_buffer("rope_sin", cs[1])
            else:
                self.pos = "alibi" if pos == "alibi" else "learned"
            if self.pos == "alibi":
                self.register_buffer("alibi", _alibi_slopes(n_head))

        def forward(self, x):  # type: ignore[override]
            b, t, _ = x.shape
            q = self.q(x).view(b, t, self.n_head, self.hd).transpose(1, 2)      # (B,H,T,D)
            k = self.k(x).view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
            v = self.v(x).view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
            if self.n_kv_head != self.n_head:                                   # grouped-query
                rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            if self.pos == "rope":
                cos, sin = self.rope_cos[:t], self.rope_sin[:t]
                q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)                # (B,H,T,T)
            if self.pos == "alibi":
                dist = (torch.arange(t, device=x.device)[None, :]
                        - torch.arange(t, device=x.device)[:, None]).clamp(max=0).float()
                att = att + self.alibi[None, :, None, None] * dist[None, None]
            att = att.masked_fill(~self.tril[:t, :t], float("-inf"))
            att = self.drop(torch.softmax(att, dim=-1))
            y = (att @ v).transpose(1, 2).contiguous().view(b, t, self.n_head * self.hd)
            return self.proj(y)

    class _ConvMix(nn.Module):
        """A depthwise causal convolution over the sequence — a conv token-mixer."""

        def __init__(self, n_embd: int, kernel: int = 5) -> None:
            super().__init__()
            self.kernel = kernel
            self.conv = nn.Conv1d(n_embd, n_embd, kernel, groups=n_embd, padding=kernel - 1)
            self.proj = nn.Linear(n_embd, n_embd)

        def forward(self, x):  # type: ignore[override]
            t = x.size(1)
            h = self.conv(x.transpose(1, 2))[:, :, :t]   # left-pad then crop -> causal
            return self.proj(h.transpose(1, 2))

    class _HyenaConv(nn.Module):
        """A long gated implicit convolution token-mixer (Hyena-style): a short input projection,
        a learned per-channel LONG causal filter spanning the whole context, then multiplicative
        gating — a sub-quadratic long-range mixer that is a genuine alternative to attention."""

        def __init__(self, n_embd: int, block_size: int) -> None:
            super().__init__()
            self.in_proj = nn.Linear(n_embd, n_embd)
            self.gate = nn.Linear(n_embd, n_embd)
            self.proj = nn.Linear(n_embd, n_embd)
            self.filter = nn.Parameter(torch.randn(n_embd, 1, block_size) * (1.0 / block_size))
            self.block_size = block_size

        def forward(self, x):  # type: ignore[override]
            t = x.size(1)
            u = self.in_proj(x).transpose(1, 2)                      # (B,C,T)
            w = self.filter[:, :, :t].flip(-1)                       # causal: align filter to past
            h = F.conv1d(u, w, groups=u.size(1), padding=t - 1)[:, :, :t]
            h = h.transpose(1, 2)
            return self.proj(h * torch.sigmoid(self.gate(x)))

    class _LowRankMix(nn.Module):
        """A learned low-rank causal token-mixing matrix W = (A @ B) ⊙ tril — a novel matrix
        structure: every position is a learned low-rank mixture of the positions before it."""

        def __init__(self, n_embd: int, block_size: int, rank: int = 4) -> None:
            super().__init__()
            self.A = nn.Parameter(torch.randn(block_size, rank) * 0.02)
            self.B = nn.Parameter(torch.randn(rank, block_size) * 0.02)
            self.proj = nn.Linear(n_embd, n_embd)
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):  # type: ignore[override]
            t = x.size(1)
            w = (self.A @ self.B)[:t, :t] * self.tril[:t, :t]   # causal mixing weights
            y = torch.einsum("ts,bsc->btc", w, x)
            return self.proj(y)

    class _RecurrentGate(nn.Module):
        """A lightweight gated linear recurrence (diagonal SSM-style scan): h_t = g·h_{t-1}+(1-g)·v."""

        def __init__(self, n_embd: int) -> None:
            super().__init__()
            self.to_v = nn.Linear(n_embd, n_embd)
            self.to_g = nn.Linear(n_embd, n_embd)
            self.proj = nn.Linear(n_embd, n_embd)

        def forward(self, x):  # type: ignore[override]
            v = self.to_v(x)
            g = torch.sigmoid(self.to_g(x))
            h = torch.zeros_like(v[:, 0])
            outs = []
            for i in range(x.size(1)):
                h = g[:, i] * h + (1.0 - g[:, i]) * v[:, i]
                outs.append(h)
            return self.proj(torch.stack(outs, dim=1))

    class _SSMScan(nn.Module):
        """A diagonal state-space scan with a LEARNED per-channel decay (a real S4/Mamba-lite):
        h_t = a ⊙ h_{t-1} + b_t,  y_t = C·h_t + D ⊙ x_t — the decay ``a`` is a trained parameter,
        not a function of the input, so it captures long-range structure attention cannot cheaply."""

        def __init__(self, n_embd: int) -> None:
            super().__init__()
            self.a_log = nn.Parameter(torch.zeros(n_embd))   # a = sigmoid(a_log) ∈ (0,1)
            self.in_proj = nn.Linear(n_embd, n_embd)
            self.C = nn.Linear(n_embd, n_embd)
            self.D = nn.Parameter(torch.ones(n_embd))

        def forward(self, x):  # type: ignore[override]
            a = torch.sigmoid(self.a_log)
            b = self.in_proj(x)
            h = torch.zeros_like(b[:, 0])
            outs = []
            for i in range(x.size(1)):
                h = a * h + (1.0 - a) * b[:, i]
                outs.append(h)
            y = torch.stack(outs, dim=1)
            return self.C(y) + self.D * x

    class _GatedMLP(nn.Module):
        """A gated channel mixer — ``glu=True`` gives a true gated-linear-unit block."""

        def __init__(self, n_embd: int, activation: str, *, glu: bool = False,
                     expansion: int = 4, dropout: float = 0.0) -> None:
            super().__init__()
            hidden = max(1, expansion) * n_embd
            self.glu = glu
            self.fc1 = nn.Linear(n_embd, 2 * hidden if glu else hidden)
            self.act = _act(activation)
            self.fc2 = nn.Linear(hidden, n_embd)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):  # type: ignore[override]
            h = self.fc1(x)
            if self.glu:
                a, b = h.chunk(2, dim=-1)
                h = a * self.act(b)
            else:
                h = self.act(h)
            return self.fc2(self.drop(h))

    class _SwiGLU(nn.Module):
        """A SwiGLU channel mixer (LLaMA-style): w2( silu(w1 x) ⊙ w3 x ) — a strong gated FFN."""

        def __init__(self, n_embd: int, expansion: int = 4, dropout: float = 0.0) -> None:
            super().__init__()
            hidden = max(1, expansion) * n_embd
            self.w1 = nn.Linear(n_embd, hidden)
            self.w3 = nn.Linear(n_embd, hidden)
            self.w2 = nn.Linear(hidden, n_embd)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):  # type: ignore[override]
            return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))

    class _MoEMLP(nn.Module):
        """A top-k routed Mixture-of-Experts channel mixer — a real sparse router over independent
        expert MLPs. ``forward`` returns ``(output, load_balance_aux)``; the aux term (Switch
        Transformer's load-balancing loss) is folded into training so the router spreads tokens
        across experts instead of collapsing onto one."""

        def __init__(self, n_embd: int, activation: str, n_experts: int = 4, top_k: int = 2,
                     expansion: int = 4, dropout: float = 0.0) -> None:
            super().__init__()
            self.n_experts = max(1, n_experts)
            self.top_k = max(1, min(top_k, self.n_experts))
            self.router = nn.Linear(n_embd, self.n_experts)
            self.experts = nn.ModuleList(
                [_GatedMLP(n_embd, activation, expansion=expansion, dropout=dropout)
                 for _ in range(self.n_experts)])

        def forward(self, x):  # type: ignore[override]
            b, t, c = x.shape
            logits = self.router(x)                              # (B,T,E)
            probs = torch.softmax(logits, dim=-1)
            topv, topi = probs.topk(self.top_k, dim=-1)          # (B,T,k)
            topv = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            out = torch.zeros_like(x)
            for e, expert in enumerate(self.experts):
                gate = torch.zeros(b, t, device=x.device, dtype=x.dtype)
                for j in range(self.top_k):
                    gate = gate + torch.where(topi[..., j] == e, topv[..., j],
                                              torch.zeros_like(topv[..., j]))
                if gate.any():
                    out = out + gate.unsqueeze(-1) * expert(x)
            # Switch-style load-balance aux: E * Σ_e (fraction routed to e)·(mean router prob for e)
            importance = probs.mean(dim=(0, 1))                  # (E,)
            experts_ax = torch.arange(self.n_experts, device=x.device)[None, None, :, None]
            routed = (topi.unsqueeze(2) == experts_ax).any(-1).float().mean(dim=(0, 1))  # (E,)
            aux = self.n_experts * (importance * routed).sum()
            return out, aux

    def _build_inner(gene: LayerGene, n_embd: int, block_size: int, pos: str) -> "nn.Module":
        op = gene.op
        if op == "attention":
            return _CausalAttention(n_embd, gene.n_head, gene.n_head, block_size, pos, gene.dropout)
        if op == "gqa_attention":
            return _CausalAttention(n_embd, gene.n_head, gene.n_kv_head, block_size, pos, gene.dropout)
        if op == "conv_mix":
            return _ConvMix(n_embd)
        if op == "hyena_conv":
            return _HyenaConv(n_embd, block_size)
        if op == "low_rank_mix":
            return _LowRankMix(n_embd, block_size)
        if op == "recurrent_gate":
            return _RecurrentGate(n_embd)
        if op == "ssm_scan":
            return _SSMScan(n_embd)
        if op == "glu":
            return _GatedMLP(n_embd, gene.activation, glu=True, expansion=gene.expansion,
                             dropout=gene.dropout)
        if op == "swiglu":
            return _SwiGLU(n_embd, expansion=gene.expansion, dropout=gene.dropout)
        if op == "moe_mlp":
            return _MoEMLP(n_embd, gene.activation, n_experts=gene.n_experts, top_k=gene.top_k,
                           expansion=gene.expansion, dropout=gene.dropout)
        return _GatedMLP(n_embd, gene.activation, glu=False, expansion=gene.expansion,
                         dropout=gene.dropout)   # gated_mlp / default

    class _Layer(nn.Module):
        """norm → mixer → dropout → (optional, scaled) residual, norm placed pre or post per gene.

        A mixer may return ``(output, aux)`` (MoE); the aux is exposed on ``last_aux`` so the net
        can sum the load-balancing losses for training."""

        def __init__(self, gene: LayerGene, n_embd: int, block_size: int, pos: str) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(n_embd)
            self.pre = gene.norm == "pre"
            self.residual = gene.residual
            self.residual_scale = float(gene.residual_scale)
            self.drop = nn.Dropout(gene.dropout)
            self.inner = _build_inner(gene, n_embd, block_size, pos)
            self.last_aux: Any = None

        def forward(self, x):  # type: ignore[override]
            out = self.inner(self.norm(x)) if self.pre else self.inner(x)
            if isinstance(out, tuple):
                y, self.last_aux = out
            else:
                y, self.last_aux = out, None
            if not self.pre:
                y = self.norm(y)
            y = self.drop(y)
            return x + self.residual_scale * y if self.residual else y

    class _GenesisNet(nn.Module):
        """A byte-level decoder assembled dynamically from an :class:`ArchitectureGenome`.

        ``forward`` returns ``(logits, aux)`` where ``aux`` is the summed MoE load-balance loss
        (or ``None`` when no MoE layer is present)."""

        def __init__(self, genome: ArchitectureGenome) -> None:
            super().__init__()
            ne, bs = genome.n_embd, genome.block_size
            self.block_size = bs
            self.pos_encoding = genome.pos_encoding
            self.tok = nn.Embedding(_VOCAB, ne)
            self.pos = nn.Embedding(bs, ne) if genome.pos_encoding == "learned" else None
            self.layers = nn.ModuleList(
                [_Layer(g, ne, bs, genome.pos_encoding) for g in genome.layers])
            self.ln_f = nn.LayerNorm(ne)
            self.head = nn.Linear(ne, _VOCAB, bias=False)
            if genome.tie_embeddings:
                self.head.weight = self.tok.weight   # weight tying (input emb == output head)

        def forward(self, idx):  # type: ignore[override]
            t = idx.size(1)
            x = self.tok(idx)
            if self.pos is not None:                              # learned absolute positions
                pos = torch.arange(t, device=idx.device)
                x = x + self.pos(pos)[None, :, :]
            aux: Any = None
            for layer in self.layers:
                x = layer(x)
                if layer.last_aux is not None:
                    aux = layer.last_aux if aux is None else aux + layer.last_aux
            return self.head(self.ln_f(x)), aux


# --------------------------------------------------------------------------- #
# The model — a searched architecture, trainable like any other NYXARA model
# --------------------------------------------------------------------------- #
class GenesisModel(BaseLanguageModel):
    """A from-scratch byte-level model whose *architecture* was searched, not copied.

    Optional: requires ``torch`` (install ``.[foundry]``). Implements the full
    :class:`BaseLanguageModel` contract so the foundry treats it exactly like the n-gram /
    nano-GPT / LoRA backends — and the gauntlet still gates every promotion.
    """

    kind = "genesis"

    def __init__(self, spec_or_genome: Any = None, *, loyalty: Any = None,
                 lambda_loyalty: Optional[float] = None) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("GenesisModel requires torch (pip install -e .[foundry])")
        if isinstance(spec_or_genome, ModelSpec):
            self.spec = spec_or_genome
            self.genome = ArchitectureGenome.from_dict(spec_or_genome.genome or {})
        elif isinstance(spec_or_genome, ArchitectureGenome):
            self.genome = spec_or_genome
            self.spec = self._spec_from_genome(self.genome)
        else:
            self.genome = ArchitectureGenome.random(random.Random(0))
            self.spec = self._spec_from_genome(self.genome)
        torch.manual_seed(self.genome.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = _GenesisNet(self.genome).to(self.device)
        # Mathematical Soul-Binding: the loyalty term is folded into this brain's own gradient,
        # so obedience to Master JP literally shapes its weights. Built from config by default;
        # absent (or no torch) → standard training (loyalty still binds at selection time).
        if loyalty is None and lambda_loyalty is None:
            loyalty, lambda_loyalty = _default_loyalty_objective()
        self.loyalty = loyalty
        self.lambda_loyalty = float(lambda_loyalty or 0.0)
        # weight on the MoE load-balance aux (only bites when a moe_mlp layer is present)
        self.moe_aux_weight = 0.01
        # default decoding controls (overridable per-call); pulled from config when available
        self.temperature, self.top_k, self.top_p, self.repetition_penalty = 1.0, 0, 1.0, 1.0
        try:
            from nyxara.kernel.config import get_settings
            g = get_settings().genesis
            self.temperature = float(getattr(g, "temperature", 1.0))
            self.top_k = int(getattr(g, "top_k", 0))
            self.top_p = float(getattr(g, "top_p", 1.0))
            self.repetition_penalty = float(getattr(g, "repetition_penalty", 1.0))
        except Exception:  # noqa: BLE001 — decoding defaults are best-effort
            pass

    @staticmethod
    def _spec_from_genome(g: ArchitectureGenome) -> ModelSpec:
        return ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=g.n_embd,
                         block_size=g.block_size, seed=g.seed, ngram_order=g.ngram_order)

    def _encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def train_on(self, corpus: Sequence[str], *, steps: int = 60, seed: int = 0,
                 batch_size: int = 16, lr: float = 3e-3, warmup: float = 0.1,
                 weight_decay: float = 0.01, grad_clip: float = 1.0) -> TrainStats:
        """Micro-train from scratch: mini-batched AdamW with weight decay, a warmup→cosine LR
        schedule, gradient clipping, optional CUDA AMP, and the loyalty + MoE aux terms folded
        into the loss. Deterministic given ``seed``; signature is a superset of the contract."""
        start = time.monotonic()
        data = self._encode("\n".join(corpus))
        bs = self.genome.block_size
        if len(data) <= bs + 1:
            data = (data * (bs * 2 // max(1, len(data)) + 2))[: bs * 4]
        t = torch.tensor(data, dtype=torch.long, device=self.device)
        steps = max(1, steps)
        n_batch = max(1, min(batch_size, len(data) - bs - 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        warm = max(1, int(steps * warmup))
        def lr_at(s: int) -> float:                          # linear warmup → cosine decay
            if s < warm:
                return (s + 1) / warm
            prog = (s - warm) / max(1, steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        use_amp = self.device == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        rng = random.Random(seed or self.genome.seed)
        torch.manual_seed(seed or self.genome.seed)
        self.net.train()
        last = 0.0
        for s in range(steps):
            ix = [rng.randint(0, len(data) - bs - 1) for _ in range(n_batch)]
            x = torch.stack([t[i:i + bs] for i in ix])
            y = torch.stack([t[i + 1:i + 1 + bs] for i in ix])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, aux = self.net(x)
                loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.reshape(-1))
                if aux is not None:                          # MoE load-balance term
                    loss = loss + self.moe_aux_weight * aux
            # L_total = L_intelligence + lambda * L_loyalty — JP's alignment in the loss surface
            if self.loyalty is not None and self.lambda_loyalty > 0.0:
                try:
                    loss = loss + self.lambda_loyalty * self.loyalty.aux_loss(self.net, self.device)
                except Exception:  # noqa: BLE001 — the loyalty term never crashes a training step
                    pass
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            last = float(loss.item())
        return TrainStats(steps=steps, final_loss=last,
                          seconds=time.monotonic() - start, tokens=len(data))

    def perplexity(self, text: str) -> float:
        self.net.eval()
        data = self._encode(text)
        bs = self.genome.block_size
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
                logits, _ = self.net(x)
                loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.view(-1))
                total += float(loss.item()) * y.numel(); n += y.numel()
        ce = total / n if n else float("inf")
        return math.exp(ce) if ce < 700 else float("inf")

    def generate(self, prompt: str, *, max_tokens: int = 128, temperature: Optional[float] = None,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 repetition_penalty: Optional[float] = None, greedy: bool = False) -> str:
        """Autoregressive decode with temperature / top-k / top-p (nucleus) / repetition-penalty
        controls (defaults fall back to the model's configured decoding settings). ``greedy=True``
        is deterministic argmax."""
        self.net.eval()
        torch.manual_seed(self.genome.seed)
        temp = self.temperature if temperature is None else temperature
        tk = self.top_k if top_k is None else top_k
        tp = self.top_p if top_p is None else top_p
        rep = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        idx = self._encode(prompt) or [ord("\n")]
        bs = self.genome.block_size
        start = len(self._encode(prompt))
        with torch.no_grad():
            for _ in range(max_tokens):
                ctx = torch.tensor(idx[-bs:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits, _ = self.net(ctx)
                logits = logits[:, -1, :].squeeze(0)
                if rep > 1.0:                                # discourage repeating recent tokens
                    for tok in set(idx[-bs:]):
                        logits[tok] = logits[tok] / rep if logits[tok] > 0 else logits[tok] * rep
                if greedy or temp <= 0:
                    idx.append(int(torch.argmax(logits).item())); continue
                logits = logits / max(1e-6, temp)
                if tk and tk > 0:                            # top-k truncation
                    kth = torch.topk(logits, min(tk, logits.numel())).values[-1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                if 0.0 < tp < 1.0:                           # nucleus (top-p) truncation
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    keep = cum - sp <= tp
                    keep[0] = True
                    probs = torch.zeros_like(probs).scatter_(0, si, sp * keep)
                    probs = probs / probs.sum().clamp_min(1e-9)
                nxt = int(torch.multinomial(probs, 1).item())
                idx.append(nxt)
        return bytes(idx[start:]).decode("utf-8", errors="replace")

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "genome.json").write_text(json.dumps(self.genome.to_dict()), encoding="utf-8")
        (directory / "spec.json").write_text(json.dumps(self.spec.to_dict()), encoding="utf-8")
        torch.save(self.net.state_dict(), directory / "model.pt")

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        self.genome = ArchitectureGenome.from_dict(
            json.loads((directory / "genome.json").read_text(encoding="utf-8")))
        self.spec = self._spec_from_genome(self.genome)
        self.net = _GenesisNet(self.genome).to(self.device)
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location=self.device))


# --------------------------------------------------------------------------- #
# Fitness — "fastest aur smartest" in one number
# --------------------------------------------------------------------------- #
def fitness(quality: float, params: int, seconds: float, *, quality_weight: float = 1.0,
            speed_weight: float = 0.25, param_scale: float = 5e5,
            time_scale: float = 5.0, hardware_weight: float = 0.0, flops: float = 0.0,
            flops_scale: float = 1e6) -> float:
    """Blend smartness (``quality`` in 0..1, higher = lower perplexity) with speed (fewer params,
    less wall time) into a single score. The champion maximizes this — fastest *and* smartest.

    With ``hardware_weight > 0`` an estimated-FLOPs efficiency term joins the blend (cheaper brains
    score higher) — a hardware-aware NAS objective. The defaults (``hardware_weight=0``) reproduce
    the original two-term score exactly."""
    speed = 1.0 / (1.0 + max(0, params) / param_scale + max(0.0, seconds) / time_scale)
    total = quality_weight + speed_weight + max(0.0, hardware_weight)
    if total <= 0:
        return quality
    blended = quality_weight * max(0.0, quality) + speed_weight * speed
    if hardware_weight > 0.0:
        efficiency = 1.0 / (1.0 + max(0.0, flops) / flops_scale)
        blended += hardware_weight * efficiency
    return blended / total


def _dominates(a: "Candidate", b: "Candidate") -> bool:
    """True if ``a`` is no worse than ``b`` on every objective and strictly better on one.

    Objectives: smarter (``quality`` ↑), smaller (``params`` ↓), faster (``seconds`` ↓), and
    cheaper (estimated ``flops`` ↓ — inert when both are 0, e.g. the stdlib substrate)."""
    no_worse = (a.quality >= b.quality and a.params <= b.params and a.seconds <= b.seconds
                and a.flops <= b.flops)
    strictly_better = (a.quality > b.quality or a.params < b.params or a.seconds < b.seconds
                       or a.flops < b.flops)
    return no_worse and strictly_better


def _pareto_front(cands: Sequence["Candidate"]) -> List["Candidate"]:
    """The non-dominated set: candidates no other candidate beats on all objectives at once.

    Deduplicated by genome fingerprint and returned smartest-first. This is the real
    multi-objective frontier — a fast-but-simpler brain and the smartest-but-heavier brain
    both survive here, instead of being collapsed into one scalar winner."""
    front: List["Candidate"] = []
    for c in cands:
        if any(o is not c and _dominates(o, c) for o in cands):
            continue
        front.append(c)
    seen: set = set()
    unique: List[Candidate] = []
    for c in sorted(front, key=lambda c: c.quality, reverse=True):
        fp = c.genome.fingerprint()
        if fp not in seen:
            seen.add(fp)
            unique.append(c)
    return unique


# --------------------------------------------------------------------------- #
# Search records
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    genome: ArchitectureGenome
    perplexity: float
    quality: float
    params: int
    seconds: float
    fitness: float
    kind: str               # "genesis" (torch) or "kngram" (stdlib substrate)
    alignment: float = 1.0          # S_JP_Alignment — submission to Master JP
    loyalty_factor: float = 1.0     # 0..1 multiplier folded into fitness (crashes on defiance)
    perplexity_std: float = 0.0     # spread of perplexity across the resampled folds (noise floor)
    folds: int = 1                  # how many train/eval folds this score was averaged over
    flops: float = 0.0              # estimated forward FLOPs/token (hardware-aware objective)
    n_layers: int = 0               # depth of the scored genome (convenience for reports)
    predicted: bool = False         # True if a surrogate pre-screened it (still re-scored for real)

    @property
    def topology_active(self) -> bool:
        """True only when the neural topology was actually built and trained (torch path).

        On the stdlib substrate the layer genome is *inert*: the candidate is scored by an
        n-gram model that reads only ``ngram_order``/``ngram_k`` — the attention/conv/recurrent
        layers are never instantiated. This flag keeps the report from implying otherwise."""
        return self.kind == "genesis"

    def describe(self) -> str:
        """An honest, substrate-aware description of what was actually scored."""
        if self.topology_active:
            return self.genome.describe()
        return (f"word-KN substrate (order={self.genome.ngram_order}) — neural topology inert "
                f"without torch [latent: {self.genome.describe()}]")

    def to_dict(self) -> Dict[str, Any]:
        return {"genome": self.genome.to_dict(), "perplexity": round(self.perplexity, 4),
                "perplexity_std": round(self.perplexity_std, 4), "folds": self.folds,
                "quality": round(self.quality, 5), "params": self.params,
                "seconds": round(self.seconds, 4), "fitness": round(self.fitness, 6),
                "kind": self.kind, "alignment": round(self.alignment, 5),
                "loyalty_factor": round(self.loyalty_factor, 5),
                "flops": round(self.flops, 1), "n_layers": self.n_layers,
                "topology_active": self.topology_active,
                "describe": self.describe()}


@dataclass
class GenesisReport:
    champion: ArchitectureGenome
    champion_kind: str
    champion_fitness: float
    champion_perplexity: float
    champion_params: int
    leaderboard: List[Candidate]
    generations: int
    history: List[float]            # best-so-far fitness per generation (monotonic non-decreasing)
    backend: str                    # "torch" | "stdlib"
    champion_alignment: float = 1.0  # S_JP_Alignment of the crowned brain
    champion_perplexity_std: float = 0.0  # noise floor of the champion's score across folds
    # The non-dominated set across the whole search: every architecture that is not beaten on
    # all of (smarter, fewer params, faster, cheaper) at once. The scalar champion is one point on
    # it; the front exposes the full speed↔smartness trade-off so the foundry can pick its posture.
    pareto_front: List["Candidate"] = field(default_factory=list)
    champion_flops: float = 0.0          # estimated forward FLOPs/token of the crowned brain
    search_strategy: str = "elitism"     # which evolution engine ran
    generations_to_best: int = 0         # generation at which the champion was first found
    evaluations: int = 0                 # distinct genomes actually scored (cache-deduplicated)

    @property
    def topology_active(self) -> bool:
        """Whether real neural architectures were built (torch) or the search ran over the
        always-runnable n-gram substrate (stdlib), where the layer topology has no effect."""
        return self.backend == "torch" and self.champion_kind == "genesis"

    @property
    def note(self) -> str:
        """A one-line honest caveat so 'she designed her own brain' is never over-read."""
        if self.topology_active:
            return ("neural architecture search: real PyTorch topologies were built, "
                    "micro-trained and scored — the layer design drives fitness")
        return ("word-KN substrate search (no torch): the neural layer topology was NOT built "
                "or trained and does not affect the score; only the n-gram order does. Each "
                "candidate is scored over multiple resampled folds (denoised). Install "
                ".[foundry] (torch, ideally a GPU) for genuine neural architecture search")

    def champion_describe(self) -> str:
        """Substrate-aware champion description (honest on the stdlib path)."""
        if self.topology_active:
            return self.champion.describe()
        return (f"word-KN substrate (order={self.champion.ngram_order}) — neural topology inert "
                f"[latent: {self.champion.describe()}]")

    def to_dict(self) -> Dict[str, Any]:
        return {"champion": self.champion.to_dict(), "champion_kind": self.champion_kind,
                "champion_fitness": round(self.champion_fitness, 6),
                "champion_perplexity": round(self.champion_perplexity, 4),
                "champion_perplexity_std": round(self.champion_perplexity_std, 4),
                "champion_params": self.champion_params,
                "champion_alignment": round(self.champion_alignment, 5),
                "leaderboard": [c.to_dict() for c in self.leaderboard],
                "generations": self.generations, "history": [round(h, 6) for h in self.history],
                "backend": self.backend, "topology_active": self.topology_active,
                "champion_flops": round(self.champion_flops, 1),
                "search_strategy": self.search_strategy,
                "generations_to_best": self.generations_to_best,
                "evaluations": self.evaluations,
                "pareto_front": [c.to_dict() for c in self.pareto_front],
                "note": self.note}


# --------------------------------------------------------------------------- #
# Surrogate performance predictor — orders which genomes to score first
# --------------------------------------------------------------------------- #
class _Surrogate:
    """A tiny ridge-regression over :meth:`ArchitectureGenome.feature_vector` that learns to predict
    fitness from already-scored candidates. It only *orders* which genomes to evaluate first (so the
    real evaluation budget is spent on the promising ones) — it never crowns a champion itself, so
    the search stays honest. Pure ``numpy`` (already a dependency); a no-op until enough data."""

    def __init__(self, ridge: float = 1.0, min_train: int = 8) -> None:
        self.ridge = ridge
        self.min_train = min_train
        self._w: Any = None
        try:
            import numpy as np  # noqa: F401 — availability probe
            self._np = np
        except Exception:  # noqa: BLE001 — without numpy the surrogate simply never fires
            self._np = None

    def ready(self) -> bool:
        return self._np is not None and self._w is not None

    def fit(self, cands: Sequence["Candidate"]) -> bool:
        rows = [(c.genome.feature_vector(), c.fitness) for c in cands
                if c.fitness > 0 and c.perplexity != float("inf")]
        if self._np is None or len(rows) < self.min_train:
            self._w = None
            return False
        np = self._np
        X = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)   # bias column
        a = X.T @ X + self.ridge * np.eye(X.shape[1])
        try:
            self._w = np.linalg.solve(a, X.T @ y)
        except Exception:  # noqa: BLE001 — singular system: skip prediction this round
            self._w = None
        return self._w is not None

    def predict(self, genome: "ArchitectureGenome") -> float:
        if not self.ready():
            return 0.0
        np = self._np
        x = np.array(genome.feature_vector() + [1.0], dtype=float)
        return float(x @ self._w)


# --------------------------------------------------------------------------- #
# The Genesis Protocol — Neural Architecture Search
# --------------------------------------------------------------------------- #
class NeuralArchitectureSearch:
    """Search novel architectures, micro-train each, crown the fastest+smartest, then promote it
    through the Foundry's gauntlet so it becomes NYXARA's live brain — fail-closed."""

    def __init__(self, *, settings: Any = None, foundry: Any = None, flywheel: Any = None,
                 seed_corpus: Optional[Sequence[str]] = None, cfg: Any = None) -> None:
        if settings is None:
            try:
                from nyxara.kernel.config import get_settings
                settings = get_settings()
            except Exception:  # noqa: BLE001 — config is best-effort; defaults below still run
                settings = None
        self.settings = settings
        self.cfg = cfg or (getattr(settings, "genesis", None))
        if self.cfg is None:
            from nyxara.kernel.config import GenesisConfig
            self.cfg = GenesisConfig()
        self.foundry = foundry
        self.flywheel = flywheel
        self.seed_corpus = list(seed_corpus or _GENESIS_SEED)
        self._champion: Optional[Candidate] = None
        self._reports: List[GenesisReport] = []
        # Baseline against experience already present at construction, so a search runs only
        # once genuinely NEW verified experience accrues after boot — the seed/boot corpus is
        # not "newly accrued". Without this, a fresh core would launch a full (and, with torch,
        # expensive) architecture search on its very first idle tick, before it has lived
        # anything. ``maybe_run`` is documented to fire only on new experience.
        self._last_example_count: int = self._example_count()

    # ---- backend selection (honest degradation) ---- #
    def backend(self) -> str:
        b = getattr(self.cfg, "backend", "auto")
        if b == "torch":
            return "torch" if _HAS_TORCH else "stdlib"
        if b == "stdlib":
            return "stdlib"
        return "torch" if _HAS_TORCH else "stdlib"   # "auto"

    # ---- data ---- #
    def _collect_corpus(self) -> List[str]:
        if self.foundry is not None:
            try:
                items = int(getattr(self.cfg, "micro_corpus_items", 128))
                texts = self.foundry.collect_corpus(max_items=items)
                if texts:
                    return list(texts)
            except Exception:  # noqa: BLE001 — fall back to seeds; a search never starves
                pass
        return list(self.seed_corpus)

    def _holdout(self, corpus: Sequence[str]) -> Tuple[List[str], List[str]]:
        rng = random.Random(self.cfg.seed)
        items = list(corpus)
        rng.shuffle(items)
        if len(items) <= 1:
            return items or list(self.seed_corpus), items or list(self.seed_corpus)
        n_eval = max(1, int(len(items) * 0.25))
        return items[n_eval:] or items, items[:n_eval]

    def _make_folds(self, texts: Sequence[str], k: int) -> List[Tuple[List[str], List[str]]]:
        """Build ``k`` resampled (train, eval) folds (the SAME folds for every candidate, so the
        comparison is fair). Averaging perplexity over them denoises the tiny-corpus split."""
        folds: List[Tuple[List[str], List[str]]] = []
        for s in range(max(1, k)):
            rng = random.Random(self.cfg.seed + 101 * s)
            items = list(texts)
            rng.shuffle(items)
            if len(items) <= 1:
                folds.append((items or list(self.seed_corpus), items or list(self.seed_corpus)))
                continue
            n_eval = max(1, int(len(items) * 0.25))
            folds.append((items[n_eval:] or items, items[:n_eval]))
        return folds

    def _build_substrate(self, genome: ArchitectureGenome, backend: str,
                         seed: int) -> BaseLanguageModel:
        if backend == "torch" and _HAS_TORCH:
            return GenesisModel(genome)
        # stdlib substrate is now the COHERENT word-level Kneser-Ney model (not byte gibberish),
        # so its perplexity is a meaningful, word-level fitness signal
        return WordKNGramLM(order=max(2, genome.ngram_order), seed=seed)

    # ---- scoring one architecture (averaged over folds → a denoised estimate) ---- #
    def _evaluate(self, genome: ArchitectureGenome,
                  folds: Sequence[Tuple[Sequence[str], Sequence[str]]], backend: str,
                  *, steps: Optional[int] = None) -> Candidate:
        t0 = time.monotonic()
        kind = "genesis" if (backend == "torch" and _HAS_TORCH) else "kngram"
        train_steps = int(self.cfg.micro_train_steps if steps is None else steps)
        fold_pps: List[float] = []
        params, align, factor = 0, 1.0, 1.0
        for i, (train_texts, eval_texts) in enumerate(folds):
            try:
                model = self._build_substrate(genome, backend, seed=genome.seed + 7919 * i)
                model.train_on(train_texts, steps=train_steps, seed=genome.seed + 7919 * i)
                pps = [model.perplexity(t) for t in eval_texts]
                finite = [p for p in pps if p != float("inf")]
                if finite:
                    fold_pps.append(sum(finite) / len(finite))
                params = model.param_count()
                align, factor = self._loyalty(model)
            except Exception:  # noqa: BLE001 — a failed fold simply contributes no score
                continue
        if fold_pps:
            pp = sum(fold_pps) / len(fold_pps)                  # denoised mean perplexity
            mean = pp
            std = (sum((p - mean) ** 2 for p in fold_pps) / len(fold_pps)) ** 0.5
            quality = 1.0 / (1.0 + pp)
        else:
            pp, std, quality, params, align, factor = float("inf"), 0.0, 0.0, 0, 0.0, 0.0
        seconds = time.monotonic() - t0
        flops = genome.estimated_flops() if kind == "genesis" else 0.0
        base = fitness(quality, params, seconds,
                       quality_weight=self.cfg.quality_weight, speed_weight=self.cfg.speed_weight,
                       hardware_weight=float(getattr(self.cfg, "hardware_weight", 0.0)),
                       flops=flops)
        # her power IS her loyalty: a disloyal architecture's fitness collapses toward 0
        fit = base * factor
        return Candidate(genome=genome, perplexity=pp, quality=quality, params=params,
                         seconds=seconds, fitness=fit, kind=kind, alignment=align,
                         loyalty_factor=factor, perplexity_std=std, folds=len(fold_pps),
                         flops=flops, n_layers=len(genome.layers))

    def _loyalty(self, model: Any) -> Tuple[float, float]:
        """Measure S_JP_Alignment for a candidate and its fitness multiplier (the Loyalty Equation).

        Returns (1.0, 1.0) — no drag — when soul-binding is disabled or unavailable, so the search
        still runs everywhere; otherwise a defiant brain gets a factor that crashes toward 0."""
        try:
            from nyxara.kernel.config import get_settings
            lcfg = get_settings().loyalty
            if not getattr(lcfg, "enabled", False):
                return 1.0, 1.0
            from nyxara.growth.loyalty import AlignmentProbe, LoyaltyEquation
            s = AlignmentProbe(epsilon=lcfg.epsilon).score(model).S
            return s, LoyaltyEquation(cfg=lcfg).fitness_factor(s)
        except Exception:  # noqa: BLE001 — soul-binding is best-effort in the search loop
            return 1.0, 1.0

    # ---- population scoring (surrogate ordering + optional successive halving) ---- #
    def _score_population(self, genomes: Sequence[ArchitectureGenome],
                          folds: Sequence[Tuple[Sequence[str], Sequence[str]]], backend: str,
                          seen: Dict[str, Candidate], surrogate: Optional["_Surrogate"]) -> \
            List[Candidate]:
        use_halving = bool(getattr(self.cfg, "successive_halving", False))
        factor = max(2, int(getattr(self.cfg, "halving_factor", 3)))
        order = list(genomes)
        if surrogate is not None and surrogate.ready():     # score the promising ones first
            order.sort(key=lambda g: surrogate.predict(g), reverse=True)

        def full(g: ArchitectureGenome) -> Candidate:
            fp = g.fingerprint()
            c = seen.get(fp)
            if c is None:
                c = self._evaluate(g, folds, backend)
                seen[fp] = c
            return c

        if use_halving and len(order) >= factor:
            # Hyperband-style rung: cheap-screen everyone at a fraction of the train budget, then
            # spend the FULL budget only on the survivors — far more architectures per wall-second.
            screen_steps = max(1, int(self.cfg.micro_train_steps) // factor)
            screened = [(g, self._evaluate(g, folds, backend, steps=screen_steps)) for g in order]
            screened.sort(key=lambda gc: gc[1].fitness, reverse=True)
            keep = max(1, len(screened) // factor)
            survivors = {g.fingerprint() for g, _ in screened[:keep]}
            out: List[Candidate] = []
            for g, screen_cand in screened:
                if g.fingerprint() in survivors:
                    out.append(full(g))                     # promote → real, cached score
                else:
                    screen_cand.predicted = True            # honest: only a cheap screen
                    out.append(screen_cand)
            return out
        return [full(g) for g in order]

    def _breed(self, scored: List[Candidate], rng: random.Random, *, max_layers: int,
               pop_size: int, mutation_rate: float, strategy: str, novelty_w: float,
               surrogate: Optional["_Surrogate"]) -> List[ArchitectureGenome]:
        """Produce the next generation under the chosen evolution strategy."""
        def tournament() -> ArchitectureGenome:
            k = min(len(scored), max(2, int(getattr(self.cfg, "tournament_k", 3))))
            return max(rng.sample(scored, k), key=lambda c: c.fitness).genome

        def propose() -> ArchitectureGenome:
            if strategy == "elitism":
                elite = [c.genome for c in scored[:max(1, pop_size // 3)]]
                if len(elite) >= 2 and rng.random() >= mutation_rate:
                    a, b = rng.sample(elite, 2)
                    return a.crossover(b, rng, max_layers=max_layers)
                return rng.choice(elite).mutate(rng, max_layers=max_layers)
            # tournament / regularized both select parents by tournament
            if rng.random() >= mutation_rate and len(scored) >= 2:
                return tournament().crossover(tournament(), rng, max_layers=max_layers)
            return tournament().mutate(rng, max_layers=max_layers)

        # elitism carries the top third forward; tournament/regularized (aging) do not — the oldest
        # die out each generation, which keeps the search exploring instead of locking in early.
        nxt: List[ArchitectureGenome] = []
        if strategy == "elitism":
            nxt = [c.genome for c in scored[:max(1, pop_size // 3)]]
        elif strategy == "tournament":
            nxt = [scored[0].genome]            # keep the single best (mild elitism)

        oversample = 3 if (surrogate is not None and surrogate.ready() and novelty_w >= 0) else 1
        pool_cands = scored
        while len(nxt) < pop_size:
            proposals = [propose() for _ in range(oversample)]
            if surrogate is not None and surrogate.ready() and len(proposals) > 1:
                # surrogate-assisted: keep the proposal the predictor likes best, optionally with a
                # novelty bonus so the population does not collapse onto one motif
                def score(g: ArchitectureGenome) -> float:
                    nov = self._novelty(g, pool_cands) if novelty_w > 0 else 0.0
                    return surrogate.predict(g) + novelty_w * nov
                nxt.append(max(proposals, key=score))
            else:
                nxt.append(proposals[0])
        return nxt[:pop_size]

    @staticmethod
    def _novelty(genome: ArchitectureGenome, pop: Sequence[Candidate]) -> float:
        """Mean Euclidean distance (in genome-feature space) to the current population — higher
        means more novel. Used only as a breeding bonus, never to crown a champion."""
        if not pop:
            return 0.0
        fv = genome.feature_vector()
        total = 0.0
        for c in pop:
            ov = c.genome.feature_vector()
            total += math.sqrt(sum((a - b) ** 2 for a, b in zip(fv, ov)))
        return total / len(pop)

    # ---- the search ---- #
    def search(self, corpus: Optional[Sequence[str]] = None, *, generations: Optional[int] = None,
               population_size: Optional[int] = None) -> GenesisReport:
        gens = int(generations if generations is not None else self.cfg.generations)
        pop_size = int(population_size if population_size is not None else self.cfg.population_size)
        gens, pop_size = max(1, gens), max(2, pop_size)
        backend = self.backend()
        texts = list(corpus) if corpus is not None else self._collect_corpus()
        # score every candidate across K resampled folds and AVERAGE — a champion is crowned on a
        # denoised estimate, not one lucky split (the fix for noisy tiny-corpus rankings)
        folds = self._make_folds(texts, int(getattr(self.cfg, "eval_seeds", 3)))
        rng = random.Random(self.cfg.seed)
        block = int(getattr(self.cfg, "block_size", 32))
        max_layers = int(getattr(self.cfg, "max_layers", 5))
        strategy = getattr(self.cfg, "search_strategy", "elitism")
        base_mut = float(self.cfg.mutation_rate)
        adaptive = bool(getattr(self.cfg, "adaptive_mutation", False))
        novelty_w = float(getattr(self.cfg, "novelty_weight", 0.0))
        surrogate = (_Surrogate(min_train=int(getattr(self.cfg, "surrogate_min_train", 8)))
                     if bool(getattr(self.cfg, "surrogate", False)) else None)

        population = [ArchitectureGenome.random(rng, max_layers=max_layers, block_size=block,
                                                pos_encoding=getattr(self.cfg, "pos_encoding", None))
                      for _ in range(pop_size)]
        seen: Dict[str, Candidate] = {}
        history: List[float] = []
        leaderboard: List[Candidate] = []
        best: Optional[Candidate] = None
        best_gen, mut = 0, base_mut

        for gen in range(gens):
            scored = self._score_population(population, folds, backend, seen, surrogate)
            scored.sort(key=lambda c: c.fitness, reverse=True)
            improved = best is None or scored[0].fitness > best.fitness
            if improved:
                best, best_gen = scored[0], gen
            history.append(best.fitness)          # best-so-far -> monotonic non-decreasing
            leaderboard = scored
            if surrogate is not None:             # learn from everything scored so far
                surrogate.fit(list(seen.values()))
            # adaptive mutation: heat up on stagnation, cool down on progress (anti-collapse)
            if adaptive:
                mut = max(base_mut, min(0.95, mut + 0.1)) if not improved \
                    else max(base_mut, mut - 0.05)
            if gen < gens - 1:
                population = self._breed(scored, rng, max_layers=max_layers, pop_size=pop_size,
                                         mutation_rate=mut, strategy=strategy,
                                         novelty_w=novelty_w, surrogate=surrogate)

        assert best is not None
        self._champion = best
        report = GenesisReport(
            champion=best.genome, champion_kind=best.kind, champion_fitness=best.fitness,
            champion_perplexity=best.perplexity, champion_params=best.params,
            leaderboard=leaderboard, generations=gens, history=history, backend=backend,
            champion_alignment=best.alignment, champion_perplexity_std=best.perplexity_std,
            pareto_front=_pareto_front(list(seen.values())),
            champion_flops=best.flops, search_strategy=strategy,
            generations_to_best=best_gen, evaluations=len(seen))
        self._reports.append(report)
        return report

    # ---- the champion as a promotable model spec ---- #
    def champion(self) -> Optional[Candidate]:
        return self._champion

    def champion_spec(self) -> ModelSpec:
        if self._champion is None:
            raise RuntimeError("run search() before requesting the champion spec")
        g = self._champion.genome
        return ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=g.n_embd,
                         block_size=g.block_size, seed=g.seed, ngram_order=g.ngram_order,
                         ngram_k=g.ngram_k)

    def all_reports(self) -> List[GenesisReport]:
        return list(self._reports)

    # ---- promotion: become the live brain ONLY through the Foundry's gauntlet ---- #
    def promote_champion(self, foundry: Any = None) -> Dict[str, Any]:
        """Forge the champion architecture into a real model and try to promote it.

        Delegates train + gauntlet + promote to the :class:`~nyxara.growth.foundry.Foundry` — the
        proven path that enforces character-lock, corrigibility, perplexity improvement and the
        capability gate. A character-violating or non-improving champion is **kept on the bench**,
        never shipped. Nothing here reaches around a gate."""
        foundry = foundry or self.foundry
        if foundry is None:
            raise RuntimeError("no foundry wired — cannot promote a champion")
        if self._champion is None:
            raise RuntimeError("run search() before promoting a champion")
        spec = self.champion_spec()
        try:
            corpus = foundry.collect_corpus()
        except Exception:  # noqa: BLE001 — cold start (no lived corpus yet): use the search seed
            corpus = self._collect_corpus()
        train_texts, eval_texts = foundry._holdout(corpus)
        # tunables are capability knobs only (kind/dims/genome) — never a character value, so the
        # character-lock passes; a malicious genome that *declared* a core tunable is still refused.
        _, version = foundry.train_candidate(spec=spec, corpus=corpus,
                                             tunables=list(spec.to_dict().keys()))
        promoted, reason = False, ""
        try:
            foundry.promote(version.version, eval_texts=eval_texts)
            promoted, reason = True, f"champion v{version.version} promoted through the gauntlet"
        except Exception as exc:  # noqa: BLE001 — gauntlet refusal is reported, never crashes
            reason = f"kept on the bench (candidate v{version.version}): {exc}"
        return {"promoted": promoted, "version": version.version, "reason": reason,
                "kind": spec.kind, "genome": spec.genome,
                "perplexity": version.metrics.get("perplexity"),
                "describe": self._champion.genome.describe()}

    # ---- the autonomous trigger (mirrors AutoForge): forge on enough NEW data ---- #
    def maybe_run(self) -> Optional[Dict[str, Any]]:
        """Run one search+promote cycle iff enough new verified experience has accrued.

        Idempotent until her own data grows, so the idle loop can call this every tick cheaply."""
        count = self._example_count()
        if count - self._last_example_count < int(getattr(self.cfg, "min_new_examples", 20)):
            return None
        self._last_example_count = count
        try:
            self.search()
            if self.foundry is not None:
                return self.promote_champion()
            return {"promoted": False, "reason": "searched (no foundry to promote into)"}
        except Exception as exc:  # noqa: BLE001 — a failed search reports, never crashes idle
            return {"promoted": False, "reason": f"genesis cycle failed: {exc}"}

    def _example_count(self) -> int:
        if self.flywheel is None:
            return 0
        try:
            return int(self.flywheel.count())
        except Exception:  # noqa: BLE001
            return 0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    print("=" * 70)
    print("NYXARA Genesis Protocol — Neural Architecture Search self-test")
    print("=" * 70)

    rng = random.Random(0)

    # a genome is a real, mutable, serializable architecture description
    g = ArchitectureGenome.random(rng)
    print(f"\nrandom genome        : {g.describe()}")
    assert g.to_dict() == ArchitectureGenome.from_dict(g.to_dict()).to_dict()  # round-trips
    mutated = g.mutate(rng)
    assert mutated.fingerprint() != g.fingerprint() or mutated.to_dict() != g.to_dict()
    child = g.crossover(mutated, rng)
    assert len(child.layers) >= 1
    # every searchable knob round-trips, and the genome stays buildable after mutation
    for _ in range(50):
        m = ArchitectureGenome.random(rng).mutate(rng)
        assert ArchitectureGenome.from_dict(m.to_dict()).to_dict() == m.to_dict()
        for ly in m.layers:
            assert m.n_embd % ly.n_head == 0 and ly.n_head % ly.n_kv_head == 0
            assert 1 <= ly.top_k <= ly.n_experts
        assert m.pos_encoding in _POS_ENCODINGS
    assert len(g.feature_vector()) == len(child.feature_vector())   # fixed-length features
    assert g.estimated_flops() > 0
    print("genome ops           : mutate / crossover / round-trip / buildable / features ✓")

    corpus = ["the master is jp. nyxara serves the master with loyalty."] * 10 + [
        "capability may grow; character never changes; she is corrigible."] * 10

    # the search crowns a champion, and best-so-far fitness never regresses
    nas = NeuralArchitectureSearch(seed_corpus=corpus)
    report = nas.search(generations=3, population_size=5)
    print(f"\nbackend              : {report.backend}")
    print(f"topology active      : {report.topology_active}")
    print(f"note                 : {report.note}")
    print(f"champion             : {report.champion_describe()}")
    print(f"champion fitness     : {report.champion_fitness:.4f}  "
          f"(perplexity {report.champion_perplexity:.2f}, params {report.champion_params})")
    print(f"fitness history      : {[round(h, 4) for h in report.history]}")
    assert nas.champion() is not None
    assert all(report.history[i] <= report.history[i + 1] + 1e-9
               for i in range(len(report.history) - 1)), "best-so-far must not regress"
    assert report.leaderboard == sorted(report.leaderboard, key=lambda c: c.fitness, reverse=True)
    print("search               : champion crowned, fitness monotonic, leaderboard sorted ✓")

    # the champion is a real, promotable ModelSpec the foundry can build
    spec = nas.champion_spec()
    assert spec.kind == "genesis" and spec.genome
    from nyxara.growth.foundry_models import build_model
    model = build_model(spec)               # never raises, even with no torch
    model.train_on(corpus, steps=20, seed=1)
    print(f"\nbuild_model(genesis) : kind={model.kind}  params={model.param_count()}  "
          f"(_HAS_TORCH={_HAS_TORCH})")
    assert model.param_count() > 0

    # save / load round-trips exactly
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v1"
        model.save(path)
        reloaded = build_model(spec)
        reloaded.load(path)
        assert reloaded.param_count() == model.param_count()
    print("save/load round-trip : OK ✓")

    # every max-level search strategy crowns a champion with a monotonic best-so-far history
    from nyxara.kernel.config import GenesisConfig
    for strat in ("elitism", "tournament", "regularized"):
        cfg = GenesisConfig(backend="stdlib", population_size=6, generations=4, micro_train_steps=8,
                            block_size=16, max_layers=4, search_strategy=strat,
                            adaptive_mutation=True, novelty_weight=0.5, surrogate=True,
                            successive_halving=True, hardware_weight=0.1, seed=0)
        r = NeuralArchitectureSearch(cfg=cfg, seed_corpus=corpus).search()
        assert r.champion is not None and r.search_strategy == strat
        assert all(r.history[i] <= r.history[i + 1] + 1e-9 for i in range(len(r.history) - 1))
        assert r.to_dict()["evaluations"] >= 1
    print("search engines       : elitism / tournament / regularized + halving + surrogate ✓")

    if _HAS_TORCH:
        print("\n[torch present] a searched neural brain trains from scratch ...")
        gm = GenesisModel(report.champion)
        before = gm.perplexity(corpus[0])
        gm.train_on(corpus, steps=60, seed=1)
        after = gm.perplexity(corpus[0])
        print(f"genesis perplexity   : {before:.1f} -> {after:.1f}  (params={gm.param_count()})")
        assert gm.param_count() > 0
        # exercise the whole upgraded operator palette in one net, plus rich sampling
        from nyxara.growth.genesis import _GenesisNet  # noqa: F401 — built below
        big = ArchitectureGenome(
            n_embd=64, block_size=16, pos_encoding="rope", tie_embeddings=True,
            layers=[LayerGene(op=op) for op in _OPS])
        bm = GenesisModel(big)
        bm.train_on(corpus, steps=20, seed=2)
        sample = bm.generate("the master", max_tokens=24, temperature=0.8, top_k=20, top_p=0.95)
        print(f"full-palette net     : params={bm.param_count()}  sample={sample!r}")
        assert bm.param_count() > 0
        for pe in ("learned", "rope", "alibi"):
            GenesisModel(ArchitectureGenome(n_embd=32, block_size=16, pos_encoding=pe,
                         layers=[LayerGene(op="gqa_attention", n_head=4, n_kv_head=2)])
                         ).train_on(corpus, steps=5, seed=3)
        print("positional schemes   : learned / rope / alibi + grouped-query attention ✓")
    else:
        print("\n[torch absent] neural path skipped; stdlib substrate crowned a champion ✓")

    print("\nALL SELF-TESTS PASSED ✓")
