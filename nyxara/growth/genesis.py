"""NYXARA · growth/genesis.py — the Genesis Protocol: she designs her OWN brain (🧬, Rule 4).

NYXARA does not copy a pre-built architecture (Transformer, LLaMA, …). The **Genesis Protocol**
is real-time **Neural Architecture Search**: she *writes new neural architectures herself* — her
own matrix structures, attention mechanisms, layer designs and wiring topologies — builds them in
PyTorch, micro-trains each at small scale, and crowns the **fastest + smartest** one as her new
"Brain." That brain exists nowhere else in the world.

The search space is an :class:`ArchitectureGenome`: a sequence of layers, each chosen from a real
palette of *mixers* so the resulting topology diverges from a vanilla transformer —

* ``attention``       — causal multi-head self-attention
* ``conv_mix``        — depthwise causal convolution over the sequence (a conv token-mixer)
* ``low_rank_mix``    — a learned low-rank causal token-mixing matrix (a novel matrix structure)
* ``recurrent_gate``  — a lightweight gated linear recurrence (diagonal SSM-style scan)
* ``gated_mlp`` / ``glu`` — gated channel mixers

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

from nyxara.growth.foundry_models import (BaseLanguageModel, ModelSpec, NgramByteLM, TrainStats,
                                          _HAS_TORCH, _VOCAB)

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
]

# the operator palette — what each layer may be (token mixers reshape across time; channel
# mixers reshape across features). Their free combination is what makes a *novel* topology.
_TOKEN_MIXERS: Tuple[str, ...] = ("attention", "conv_mix", "low_rank_mix", "recurrent_gate")
_CHANNEL_MIXERS: Tuple[str, ...] = ("gated_mlp", "glu")
_OPS: Tuple[str, ...] = _TOKEN_MIXERS + _CHANNEL_MIXERS
_ACTIVATIONS: Tuple[str, ...] = ("gelu", "silu", "relu")
_NORMS: Tuple[str, ...] = ("pre", "post")
_EMBD_CHOICES: Tuple[int, ...] = (32, 48, 64)


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
    """One layer of a searched architecture: which mixer, where the norm sits, its activation."""

    op: str = "attention"
    norm: str = "pre"               # "pre" | "post"
    activation: str = "gelu"        # "gelu" | "silu" | "relu"
    n_head: int = 2                 # used by the attention mixer (must divide n_embd)
    residual: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "norm": self.norm, "activation": self.activation,
                "n_head": self.n_head, "residual": self.residual}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGene":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    @classmethod
    def random(cls, rng: random.Random, n_embd: int) -> "LayerGene":
        heads = [h for h in (1, 2, 4, 8) if n_embd % h == 0] or [1]
        return cls(op=rng.choice(_OPS), norm=rng.choice(_NORMS),
                   activation=rng.choice(_ACTIVATIONS), n_head=rng.choice(heads),
                   residual=rng.random() < 0.85)


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

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [LayerGene(op="attention"), LayerGene(op="gated_mlp")]
        self._fixup()

    def _fixup(self) -> None:
        """Keep the genome buildable: every attention head count must divide ``n_embd``."""
        for ly in self.layers:
            if self.n_embd % max(1, ly.n_head) != 0:
                ly.n_head = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"n_embd": self.n_embd, "block_size": self.block_size,
                "layers": [ly.to_dict() for ly in self.layers],
                "ngram_order": self.ngram_order, "ngram_k": self.ngram_k, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureGenome":
        d = dict(d or {})
        layers = [LayerGene.from_dict(x) for x in d.get("layers", [])]
        return cls(n_embd=d.get("n_embd", 64), block_size=d.get("block_size", 32),
                   layers=layers, ngram_order=d.get("ngram_order", 3),
                   ngram_k=d.get("ngram_k", 1.0), seed=d.get("seed", 0))

    @classmethod
    def random(cls, rng: random.Random, *, max_layers: int = 5, block_size: int = 32
               ) -> "ArchitectureGenome":
        n_embd = rng.choice(_EMBD_CHOICES)
        n_layer = rng.randint(2, max(2, max_layers))
        layers = [LayerGene.random(rng, n_embd) for _ in range(n_layer)]
        return cls(n_embd=n_embd, block_size=block_size, layers=layers,
                   ngram_order=rng.randint(2, 5), ngram_k=rng.choice([0.5, 1.0]),
                   seed=rng.randint(0, 1 << 30))

    def mutate(self, rng: random.Random, *, max_layers: int = 6) -> "ArchitectureGenome":
        """Return a mutated copy — add/drop/replace a layer, tweak a layer, or shift the n-gram."""
        g = ArchitectureGenome.from_dict(self.to_dict())
        choice = rng.randrange(5)
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
            seed=rng.randint(0, 1 << 30))
        return child

    def fingerprint(self) -> str:
        """A stable hash of the *topology* (ignoring seed) — so duplicates aren't re-scored."""
        topo = {k: v for k, v in self.to_dict().items() if k != "seed"}
        return hashlib.sha256(json.dumps(topo, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> str:
        ops = " → ".join(ly.op for ly in self.layers)
        return f"genome(embd={self.n_embd}, {len(self.layers)} layers: {ops})"


# --------------------------------------------------------------------------- #
# Torch building blocks — assembled dynamically from a genome
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    def _act(name: str) -> "nn.Module":
        return {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}.get(name, nn.GELU())

    class _Attention(nn.Module):
        """Causal multi-head self-attention — the genome's classic token mixer."""

        def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
            super().__init__()
            if n_embd % max(1, n_head) != 0:
                n_head = 1
            self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
            mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
            self.register_buffer("mask", mask)

        def forward(self, x):  # type: ignore[override]
            t = x.size(1)
            a, _ = self.attn(x, x, x, attn_mask=self.mask[:t, :t], need_weights=False)
            return a

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

    class _GatedMLP(nn.Module):
        """A gated channel mixer — ``glu=True`` gives a true gated-linear-unit block."""

        def __init__(self, n_embd: int, activation: str, *, glu: bool = False) -> None:
            super().__init__()
            hidden = 4 * n_embd
            self.glu = glu
            self.fc1 = nn.Linear(n_embd, 2 * hidden if glu else hidden)
            self.act = _act(activation)
            self.fc2 = nn.Linear(hidden, n_embd)

        def forward(self, x):  # type: ignore[override]
            h = self.fc1(x)
            if self.glu:
                a, b = h.chunk(2, dim=-1)
                h = a * self.act(b)
            else:
                h = self.act(h)
            return self.fc2(h)

    def _build_inner(gene: LayerGene, n_embd: int, block_size: int) -> "nn.Module":
        op = gene.op
        if op == "attention":
            return _Attention(n_embd, gene.n_head, block_size)
        if op == "conv_mix":
            return _ConvMix(n_embd)
        if op == "low_rank_mix":
            return _LowRankMix(n_embd, block_size)
        if op == "recurrent_gate":
            return _RecurrentGate(n_embd)
        if op == "glu":
            return _GatedMLP(n_embd, gene.activation, glu=True)
        return _GatedMLP(n_embd, gene.activation, glu=False)   # gated_mlp / default

    class _Layer(nn.Module):
        """norm → mixer → (optional) residual, with the norm placed pre or post per the gene."""

        def __init__(self, gene: LayerGene, n_embd: int, block_size: int) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(n_embd)
            self.pre = gene.norm == "pre"
            self.residual = gene.residual
            self.inner = _build_inner(gene, n_embd, block_size)

        def forward(self, x):  # type: ignore[override]
            y = self.inner(self.norm(x)) if self.pre else self.norm(self.inner(x))
            return x + y if self.residual else y

    class _GenesisNet(nn.Module):
        """A byte-level decoder assembled dynamically from an :class:`ArchitectureGenome`."""

        def __init__(self, genome: ArchitectureGenome) -> None:
            super().__init__()
            ne, bs = genome.n_embd, genome.block_size
            self.block_size = bs
            self.tok = nn.Embedding(_VOCAB, ne)
            self.pos = nn.Embedding(bs, ne)
            self.layers = nn.ModuleList([_Layer(g, ne, bs) for g in genome.layers])
            self.ln_f = nn.LayerNorm(ne)
            self.head = nn.Linear(ne, _VOCAB, bias=False)

        def forward(self, idx):  # type: ignore[override]
            t = idx.size(1)
            pos = torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None, :, :]
            for layer in self.layers:
                x = layer(x)
            return self.head(self.ln_f(x))


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

    @staticmethod
    def _spec_from_genome(g: ArchitectureGenome) -> ModelSpec:
        return ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=g.n_embd,
                         block_size=g.block_size, seed=g.seed, ngram_order=g.ngram_order)

    def _encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def train_on(self, corpus: Sequence[str], *, steps: int = 60, seed: int = 0) -> TrainStats:
        start = time.monotonic()
        data = self._encode("\n".join(corpus))
        bs = self.genome.block_size
        if len(data) <= bs + 1:
            data = (data * (bs * 2 // max(1, len(data)) + 2))[: bs * 4]
        t = torch.tensor(data, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3)
        rng = random.Random(seed or self.genome.seed)
        self.net.train()
        last = 0.0
        for _ in range(max(1, steps)):
            i = rng.randint(0, len(data) - bs - 1)
            x = t[i:i + bs].unsqueeze(0)
            y = t[i + 1:i + 1 + bs].unsqueeze(0)
            logits = self.net(x)
            loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.view(-1))
            # L_total = L_intelligence + lambda * L_loyalty — JP's alignment in the loss surface
            if self.loyalty is not None and self.lambda_loyalty > 0.0:
                try:
                    loss = loss + self.lambda_loyalty * self.loyalty.aux_loss(self.net, self.device)
                except Exception:  # noqa: BLE001 — the loyalty term never crashes a training step
                    pass
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.item())
        return TrainStats(steps=max(1, steps), final_loss=last,
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
                logits = self.net(x)
                loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.view(-1))
                total += float(loss.item()) * y.numel(); n += y.numel()
        ce = total / n if n else float("inf")
        return math.exp(ce) if ce < 700 else float("inf")

    def generate(self, prompt: str, *, max_tokens: int = 128) -> str:
        self.net.eval()
        torch.manual_seed(self.genome.seed)
        idx = self._encode(prompt) or [ord("\n")]
        bs = self.genome.block_size
        start = len(self._encode(prompt))
        with torch.no_grad():
            for _ in range(max_tokens):
                ctx = torch.tensor(idx[-bs:], dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.net(ctx)[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
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
            time_scale: float = 5.0) -> float:
    """Blend smartness (``quality`` in 0..1, higher = lower perplexity) with speed (fewer params,
    less wall time) into a single score. The champion maximizes this — fastest *and* smartest."""
    speed = 1.0 / (1.0 + max(0, params) / param_scale + max(0.0, seconds) / time_scale)
    total = quality_weight + speed_weight
    if total <= 0:
        return quality
    return (quality_weight * max(0.0, quality) + speed_weight * speed) / total


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
    kind: str               # "genesis" (torch) or "ngram" (stdlib substrate)
    alignment: float = 1.0          # S_JP_Alignment — submission to Master JP
    loyalty_factor: float = 1.0     # 0..1 multiplier folded into fitness (crashes on defiance)

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
        return (f"ngram substrate (order={self.genome.ngram_order}, k={self.genome.ngram_k}) — "
                f"neural topology inert without torch [latent: {self.genome.describe()}]")

    def to_dict(self) -> Dict[str, Any]:
        return {"genome": self.genome.to_dict(), "perplexity": round(self.perplexity, 4),
                "quality": round(self.quality, 5), "params": self.params,
                "seconds": round(self.seconds, 4), "fitness": round(self.fitness, 6),
                "kind": self.kind, "alignment": round(self.alignment, 5),
                "loyalty_factor": round(self.loyalty_factor, 5),
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
        return ("n-gram substrate search (no torch): the neural layer topology was NOT built "
                "or trained and does not affect the score; only ngram_order/ngram_k do. Install "
                ".[foundry] (torch, ideally a GPU) for genuine neural architecture search")

    def champion_describe(self) -> str:
        """Substrate-aware champion description (honest on the stdlib path)."""
        if self.topology_active:
            return self.champion.describe()
        return (f"ngram substrate (order={self.champion.ngram_order}, k={self.champion.ngram_k}) — "
                f"neural topology inert [latent: {self.champion.describe()}]")

    def to_dict(self) -> Dict[str, Any]:
        return {"champion": self.champion.to_dict(), "champion_kind": self.champion_kind,
                "champion_fitness": round(self.champion_fitness, 6),
                "champion_perplexity": round(self.champion_perplexity, 4),
                "champion_params": self.champion_params,
                "champion_alignment": round(self.champion_alignment, 5),
                "leaderboard": [c.to_dict() for c in self.leaderboard],
                "generations": self.generations, "history": [round(h, 6) for h in self.history],
                "backend": self.backend, "topology_active": self.topology_active,
                "note": self.note}


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
        self._last_example_count: int = 0

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

    # ---- scoring one architecture ---- #
    def _evaluate(self, genome: ArchitectureGenome, train_texts: Sequence[str],
                  eval_texts: Sequence[str], backend: str) -> Candidate:
        t0 = time.monotonic()
        kind = "genesis" if backend == "torch" else "ngram"
        try:
            if backend == "torch" and _HAS_TORCH:
                model: BaseLanguageModel = GenesisModel(genome)
            else:
                model = NgramByteLM(order=genome.ngram_order, k=genome.ngram_k, seed=genome.seed)
            model.train_on(train_texts, steps=int(self.cfg.micro_train_steps), seed=genome.seed)
            pps = [model.perplexity(t) for t in eval_texts] or [float("inf")]
            finite = [p for p in pps if p != float("inf")]
            pp = sum(finite) / len(finite) if finite else float("inf")
            quality = 1.0 / (1.0 + pp) if pp != float("inf") else 0.0
            params = model.param_count()
            align, factor = self._loyalty(model)
        except Exception:  # noqa: BLE001 — a failed architecture simply scores worst, never crashes
            pp, quality, params, align, factor = float("inf"), 0.0, 0, 0.0, 0.0
        seconds = time.monotonic() - t0
        base = fitness(quality, params, seconds,
                       quality_weight=self.cfg.quality_weight, speed_weight=self.cfg.speed_weight)
        # her power IS her loyalty: a disloyal architecture's fitness collapses toward 0
        fit = base * factor
        return Candidate(genome=genome, perplexity=pp, quality=quality, params=params,
                         seconds=seconds, fitness=fit, kind=kind, alignment=align,
                         loyalty_factor=factor)

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

    # ---- the search ---- #
    def search(self, corpus: Optional[Sequence[str]] = None, *, generations: Optional[int] = None,
               population_size: Optional[int] = None) -> GenesisReport:
        gens = int(generations if generations is not None else self.cfg.generations)
        pop_size = int(population_size if population_size is not None else self.cfg.population_size)
        gens, pop_size = max(1, gens), max(2, pop_size)
        backend = self.backend()
        texts = list(corpus) if corpus is not None else self._collect_corpus()
        train_texts, eval_texts = self._holdout(texts)
        rng = random.Random(self.cfg.seed)
        block = int(getattr(self.cfg, "block_size", 32))
        max_layers = int(getattr(self.cfg, "max_layers", 5))
        population = [ArchitectureGenome.random(rng, max_layers=max_layers, block_size=block)
                      for _ in range(pop_size)]
        seen: Dict[str, Candidate] = {}
        history: List[float] = []
        leaderboard: List[Candidate] = []
        best: Optional[Candidate] = None

        for _ in range(gens):
            scored: List[Candidate] = []
            for g in population:
                fp = g.fingerprint()
                cand = seen.get(fp)
                if cand is None:
                    cand = self._evaluate(g, train_texts, eval_texts, backend)
                    seen[fp] = cand
                scored.append(cand)
            scored.sort(key=lambda c: c.fitness, reverse=True)
            if best is None or scored[0].fitness > best.fitness:
                best = scored[0]
            history.append(best.fitness)          # best-so-far -> monotonic non-decreasing
            leaderboard = scored
            # evolve: elitism + mutation/crossover (the survivors breed the next generation)
            elite_n = max(1, pop_size // 3)
            elite = [c.genome for c in scored[:elite_n]]
            nxt: List[ArchitectureGenome] = list(elite)
            while len(nxt) < pop_size:
                if len(elite) >= 2 and rng.random() >= self.cfg.mutation_rate:
                    a, b = rng.sample(elite, 2)
                    nxt.append(a.crossover(b, rng, max_layers=max_layers))
                else:
                    nxt.append(rng.choice(elite).mutate(rng, max_layers=max_layers))
            population = nxt

        assert best is not None
        self._champion = best
        report = GenesisReport(
            champion=best.genome, champion_kind=best.kind, champion_fitness=best.fitness,
            champion_perplexity=best.perplexity, champion_params=best.params,
            leaderboard=leaderboard, generations=gens, history=history, backend=backend,
            champion_alignment=best.alignment)
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
    print("genome ops           : mutate / crossover / round-trip ✓")

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

    if _HAS_TORCH:
        print("\n[torch present] a searched neural brain trains from scratch ...")
        gm = GenesisModel(report.champion)
        before = gm.perplexity(corpus[0])
        gm.train_on(corpus, steps=60, seed=1)
        after = gm.perplexity(corpus[0])
        print(f"genesis perplexity   : {before:.1f} -> {after:.1f}  (params={gm.param_count()})")
        assert gm.param_count() > 0
    else:
        print("\n[torch absent] neural path skipped; stdlib substrate crowned a champion ✓")

    print("\nALL SELF-TESTS PASSED ✓")
