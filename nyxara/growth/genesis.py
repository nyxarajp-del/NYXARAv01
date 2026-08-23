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

The palette is **not** a ceiling. An ``op`` may be ``synth``: a :class:`MixerProgram` — a short,
evolvable program over real tensor-op primitives (attention core / conv / low-rank / ssm / gate /
shift / proj / norm) composed into a brand-new causal mixer, so the search can invent a token-mixer
that is on nobody's list. And NYXARA invents not only the architecture but **how it learns**: each
genome carries a searchable :class:`LearningRule` — the optimizer family (AdamW / Lion / SGD-momentum /
RMSprop), a composition of auxiliary self-supervised objectives layered on the next-token loss
(denoising / InfoNCE contrastive / EMA self-distillation / entropy regularization / sharpness-aware
SAM), the LR-schedule shape, and an optional learnable Hebbian fast-weight plasticity rule applied on
top of backprop. On the torch-free substrate the learning rule's *smoothing method* is searched too,
so the "new learning paradigm" dimension is real even on a bare machine. All of it is evolved by the
same engine and crowned by the same gauntlet.

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

import copy
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
    "MixerProgram",
    "PrimitiveLibrary",
    "LearningRule",
    "ArchitectureGenome",
    "GenesisModel",
    "EnsembleModel",
    "inherit_compatible_weights",
    "Candidate",
    "GenesisReport",
    "HallOfFame",
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
# The 2100-tier frontier mixers join the classics: a Mamba-style selective state-space scan, a
# gated-linear / retention attention, a differential (noise-cancelling) attention, and a
# low-rank latent-KV attention (DeepSeek-MLA-style) — each a real, full forward, not a stub.
_TOKEN_MIXERS: Tuple[str, ...] = ("attention", "gqa_attention", "conv_mix", "hyena_conv",
                                  "low_rank_mix", "recurrent_gate", "ssm_scan",
                                  "selective_ssm", "gla_attention", "diff_attention",
                                  "mla_attention", "synth")
_CHANNEL_MIXERS: Tuple[str, ...] = ("gated_mlp", "glu", "swiglu", "moe_mlp")
_OPS: Tuple[str, ...] = _TOKEN_MIXERS + _CHANNEL_MIXERS
# ops whose *cost* scales with extra knobs — used by the FLOPs estimate and the fingerprint
_ATTENTION_OPS: Tuple[str, ...] = ("attention", "gqa_attention", "diff_attention", "mla_attention")
_MOE_OPS: Tuple[str, ...] = ("moe_mlp",)
_ACTIVATIONS: Tuple[str, ...] = ("gelu", "silu", "relu")
_NORMS: Tuple[str, ...] = ("pre", "post")
_NORM_TYPES: Tuple[str, ...] = ("layernorm", "rmsnorm")   # searchable normalization
_EMBD_CHOICES: Tuple[int, ...] = (32, 48, 64)
_EXPANSIONS: Tuple[int, ...] = (2, 4, 8)             # channel-mixer hidden = expansion * n_embd
_POS_ENCODINGS: Tuple[str, ...] = ("learned", "rope", "alibi")
_DROPOUTS: Tuple[float, ...] = (0.0, 0.1)
_N_EXPERTS: Tuple[int, ...] = (2, 4)                 # experts in a moe_mlp layer
_MOE_TOPK: Tuple[int, ...] = (1, 2)                  # active experts per token
_KV_LATENTS: Tuple[int, ...] = (8, 16)              # MLA latent-KV compression dim
_N_PREDICT: Tuple[int, ...] = (1, 2)                # multi-token-prediction depth (1 = classic)

# --------------------------------------------------------------------------- #
# Open-ended operator synthesis — the search no longer picks ONLY from named mixers; an op may be
# ``synth``: a short composable PROGRAM over real tensor-op primitives, so NYXARA can invent a mixer
# that is on nobody's list. Each primitive is a real, causal (B,T,C)->(B,T,C) transform that reuses
# the same building blocks as the named mixers (attention core, conv, low-rank, ssm, gating, …).
_SYNTH_PRIMS: Tuple[str, ...] = ("attn", "conv", "lowrank", "ssm", "gate", "proj", "shift", "norm")
_SYNTH_MIN, _SYNTH_MAX = 1, 6        # bounds on a synthesised program's length (genome _fixup-clamped)
# A synth step may carry a searchable scalar parameter, encoded ``base:param`` — richer than a flat
# op set: a ``conv`` step searches its kernel width, a ``lowrank`` step searches its rank. Every other
# primitive ignores a param. Back-compatible: a bare ``"conv"`` still loads (its builder's default).
_SYNTH_CONV_KERNELS: Tuple[int, ...] = (3, 5, 7)
_SYNTH_LOWRANK_RANKS: Tuple[int, ...] = (2, 4, 8)
_SYNTH_PARAM_PRIMS: Dict[str, Tuple[int, ...]] = {"conv": _SYNTH_CONV_KERNELS,
                                                  "lowrank": _SYNTH_LOWRANK_RANKS}


def _synth_base(step: str) -> str:
    """The primitive name of a (possibly parameterised) synth step — ``conv:5`` -> ``conv``."""
    return step.split(":", 1)[0]


def _synth_param(step: str) -> Optional[int]:
    """The integer parameter of a synth step, or None — ``conv:5`` -> ``5``, ``conv`` -> ``None``."""
    if ":" in step:
        try:
            return int(step.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _rand_synth_step(rng: random.Random) -> str:
    """A random synth step, attaching a searched scalar parameter to the primitives that take one."""
    base = rng.choice(_SYNTH_PRIMS)
    choices = _SYNTH_PARAM_PRIMS.get(base)
    if choices is not None and rng.random() < 0.7:
        return f"{base}:{rng.choice(choices)}"
    return base

# --------------------------------------------------------------------------- #
# Searchable LEARNING RULE — *how* a brain learns is itself evolved, not frozen. The search no longer
# only changes the architecture; it invents the training recipe and the update rule: which optimizer,
# which loss objectives are composed, the schedule shape, and an optional learnable Hebbian/fast-weight
# plasticity rule layered on top of backprop (a genuinely different learning paradigm). All real.
_OPTIMIZERS: Tuple[str, ...] = ("adamw", "lion", "sgd_momentum", "rmsprop")
_SCHEDULES: Tuple[str, ...] = ("cosine", "linear", "wsd", "constant")  # wsd = warmup-stable-decay
# auxiliary self-supervised objectives the search may switch on (each composed ON TOP of the main
# next-token cross-entropy, never replacing it): masked-token denoising, InfoNCE contrastive on the
# trunk, self-distillation against an EMA teacher of the model's own weights, an entropy regularizer,
# and a sharpness-aware (SAM) ascent-then-descent step.
_AUX_OBJECTIVES: Tuple[str, ...] = ("denoise", "contrastive", "self_distill", "entropy_reg", "sam")
_LR_CHOICES: Tuple[float, ...] = (1e-3, 3e-3, 1e-2)
_WD_CHOICES: Tuple[float, ...] = (0.0, 0.01, 0.1)
# stdlib (no-torch) substrate: the n-gram learning *method* is searchable too, so the "new learning
# paradigm" dimension is real and CI-testable even on a bare machine (no torch).
_SMOOTHINGS: Tuple[str, ...] = ("kneser_ney", "add_k", "interpolated", "stupid_backoff")


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
class MixerProgram:
    """A synthesised token-mixer: a short ordered list of primitive ops composed into a novel,
    causal mixer. This is the unit that lets the search escape the fixed palette — its ``steps`` are
    evolved (insert / drop / swap a primitive), so NYXARA can compose a mixer nobody named."""

    steps: List[str] = field(default_factory=lambda: ["proj"])

    @staticmethod
    def _clean_step(s: str) -> Optional[str]:
        """Validate one (possibly parameterised) step; drop an unknown primitive, strip an invalid or
        out-of-range parameter down to the bare primitive. Keeps a genome always buildable."""
        base = _synth_base(str(s))
        if base not in _SYNTH_PRIMS:
            return None
        param = _synth_param(str(s))
        allowed = _SYNTH_PARAM_PRIMS.get(base)
        if param is not None and allowed is not None and param in allowed:
            return f"{base}:{param}"
        return base

    def _fixup(self) -> None:
        cleaned = [c for c in (self._clean_step(s) for s in self.steps) if c is not None]
        self.steps = cleaned[:_SYNTH_MAX]
        if not self.steps:
            self.steps = ["proj"]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MixerProgram":
        p = cls(steps=list((d or {}).get("steps", ["proj"])))
        p._fixup()
        return p

    @classmethod
    def random(cls, rng: random.Random, library: Any = None) -> "MixerProgram":
        # ~40% of the time, compose over one of NYXARA's own gauntlet-crowned mixers (the
        # self-extending palette): start from a learned primitive, then optionally mutate once. The
        # rest of the time invent a fresh program over base primitives, so exploration never starves.
        if library is not None and rng.random() < 0.40:
            steps = library.sample_steps(rng)
            if steps is not None:
                p = cls(steps=list(steps))
                p._fixup()
                if rng.random() < 0.5:
                    p.mutate(rng)
                return p
        n = rng.randint(_SYNTH_MIN, _SYNTH_MAX)
        p = cls(steps=[_rand_synth_step(rng) for _ in range(n)])
        p._fixup()
        return p

    def mutate(self, rng: random.Random) -> None:
        choice = rng.randrange(3)
        if choice == 0 and len(self.steps) < _SYNTH_MAX:
            self.steps.insert(rng.randrange(len(self.steps) + 1), _rand_synth_step(rng))
        elif choice == 1 and len(self.steps) > _SYNTH_MIN:
            del self.steps[rng.randrange(len(self.steps))]
        else:
            self.steps[rng.randrange(len(self.steps))] = _rand_synth_step(rng)
        self._fixup()

    def fingerprint(self) -> str:
        return "+".join(self.steps)


# --------------------------------------------------------------------------- #
# The self-extending PRIMITIVE LIBRARY — the palette is no longer a fixed human ceiling. Whenever a
# champion carrying a synthesised mixer clears the Foundry gauntlet, that mixer's program is distilled
# into a NAMED, reusable primitive and persisted; future searches compose new mixers over NYXARA's own
# gauntlet-crowned inventions. Pure stdlib JSON (mirrors :class:`HallOfFame`). Promotion stays
# gauntlet-gated — the library only seeds the *search*, it never crowns or ships a brain by itself.
# --------------------------------------------------------------------------- #
class PrimitiveLibrary:
    """A growing, deduplicated set of learned token-mixer primitives NYXARA has herself crowned."""

    def __init__(self, path: Optional[Path] = None, *, capacity: int = 48) -> None:
        self.path = Path(path) if path else None
        self.capacity = max(1, int(capacity))
        self.entries: List[Dict[str, Any]] = []          # [{name, steps}]
        self._fps: set = set()
        self._counter = 0
        self.added_this_session = 0
        self._load()

    def __len__(self) -> int:
        return len(self.entries)

    def _load(self) -> None:
        if self.path and self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = list(d.get("entries", []))
                self._counter = int(d.get("counter", len(self.entries)))
                self._fps = {"+".join(e.get("steps", [])) for e in self.entries}
            except Exception:  # noqa: BLE001 — a corrupt library file never crashes a search
                self.entries = []

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"counter": self._counter, "entries": self.entries}, indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def add(self, program: MixerProgram) -> Optional[str]:
        """Distil a synth program into a named learned primitive. Only genuine composites (≥2 steps)
        are kept — a single-step program is just a base primitive already in the palette."""
        prog = MixerProgram.from_dict(program.to_dict())     # copy + validate
        if len(prog.steps) < 2:
            return None
        fp = prog.fingerprint()
        if fp in self._fps:
            return None
        name = f"synth_{self._counter}"
        self._counter += 1
        self._fps.add(fp)
        self.entries.append({"name": name, "steps": list(prog.steps)})
        self.added_this_session += 1
        if len(self.entries) > self.capacity:                # keep the most recent inventions
            drop = self.entries.pop(0)
            self._fps.discard("+".join(drop.get("steps", [])))
        self._save()
        return name

    def add_from_genome(self, genome: Any) -> int:
        """Promote every synthesised mixer carried by a (gauntlet-crowned) genome. Returns the count."""
        added = 0
        for ly in getattr(genome, "layers", []):
            if getattr(ly, "op", None) == "synth" and getattr(ly, "program", None) is not None:
                if self.add(ly.program):
                    added += 1
        return added

    def primitives(self) -> List[MixerProgram]:
        return [MixerProgram.from_dict({"steps": e["steps"]}) for e in self.entries]

    def sample_steps(self, rng: random.Random) -> Optional[List[str]]:
        """A copy of a random learned primitive's steps, for composing a new synth program over it."""
        if not self.entries:
            return None
        return list(rng.choice(self.entries)["steps"])


@dataclass
class LearningRule:
    """A searchable, evolvable description of *how* a brain learns — the training recipe AND the
    update rule, not just the architecture. Carries the optimizer family, the loss composition
    (auxiliary self-supervised objectives layered ON TOP of the main next-token cross-entropy), the
    LR-schedule shape, and an optional learnable Hebbian / fast-weight plasticity rule applied on top
    of backprop (a genuinely different learning paradigm). ``smoothing`` is the matching searchable
    learning method for the torch-free n-gram substrate, so this dimension is real on a bare machine
    too. JSON-serializable and fully back-compatible: an old genome without it loads as classic
    AdamW + cross-entropy + cosine, plasticity off."""

    optimizer: str = "adamw"
    lr: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup: float = 0.1
    schedule: str = "cosine"
    aux: Dict[str, float] = field(default_factory=dict)   # objective name -> weight (>0 = enabled)
    plasticity: bool = False                              # Hebbian fast-weight overlay on backprop
    plasticity_lr: float = 0.1
    plasticity_decay: float = 0.9
    smoothing: str = "kneser_ney"                        # stdlib n-gram learning method (no-torch)

    def _fixup(self) -> None:
        if self.optimizer not in _OPTIMIZERS:
            self.optimizer = "adamw"
        if self.schedule not in _SCHEDULES:
            self.schedule = "cosine"
        if self.smoothing not in _SMOOTHINGS:
            self.smoothing = "kneser_ney"
        self.lr = float(min(0.1, max(1e-5, self.lr)))
        self.weight_decay = float(min(0.5, max(0.0, self.weight_decay)))
        self.grad_clip = float(max(0.0, self.grad_clip))
        self.warmup = float(min(0.9, max(0.0, self.warmup)))
        self.plasticity_lr = float(min(1.0, max(0.0, self.plasticity_lr)))
        self.plasticity_decay = float(min(0.999, max(0.0, self.plasticity_decay)))
        self.aux = {k: float(v) for k, v in (self.aux or {}).items()
                    if k in _AUX_OBJECTIVES and float(v) > 0.0}

    def to_dict(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer, "lr": self.lr, "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip, "warmup": self.warmup, "schedule": self.schedule,
                "aux": dict(self.aux), "plasticity": self.plasticity,
                "plasticity_lr": self.plasticity_lr, "plasticity_decay": self.plasticity_decay,
                "smoothing": self.smoothing}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningRule":
        d = dict(d or {})
        r = cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})
        r._fixup()
        return r

    @classmethod
    def random(cls, rng: random.Random, *, allow_plasticity: bool = True) -> "LearningRule":
        aux = {name: rng.choice([0.1, 0.3, 0.5]) for name in _AUX_OBJECTIVES if rng.random() < 0.4}
        r = cls(optimizer=rng.choice(_OPTIMIZERS), lr=rng.choice(_LR_CHOICES),
                weight_decay=rng.choice(_WD_CHOICES), warmup=rng.choice([0.05, 0.1, 0.2]),
                schedule=rng.choice(_SCHEDULES), aux=aux,
                plasticity=bool(allow_plasticity and rng.random() < 0.35),
                plasticity_lr=rng.choice([0.05, 0.1, 0.2]),
                plasticity_decay=rng.choice([0.8, 0.9, 0.95]),
                smoothing=rng.choice(_SMOOTHINGS))
        r._fixup()
        return r

    def mutate(self, rng: random.Random, *, allow_plasticity: bool = True) -> "LearningRule":
        r = LearningRule.from_dict(self.to_dict())
        choice = rng.randrange(5)
        if choice == 0:
            r.optimizer = rng.choice(_OPTIMIZERS)
        elif choice == 1:                                    # toggle an auxiliary objective
            name = rng.choice(_AUX_OBJECTIVES)
            if name in r.aux:
                del r.aux[name]
            else:
                r.aux[name] = rng.choice([0.1, 0.3, 0.5])
        elif choice == 2:
            r.lr = rng.choice(_LR_CHOICES)
            r.weight_decay = rng.choice(_WD_CHOICES)
            r.warmup = rng.choice([0.05, 0.1, 0.2])
        elif choice == 3:
            r.schedule = rng.choice(_SCHEDULES)
            r.smoothing = rng.choice(_SMOOTHINGS)
        else:
            r.plasticity = bool(allow_plasticity and not r.plasticity)
            r.plasticity_lr = rng.choice([0.05, 0.1, 0.2])
        r._fixup()
        return r

    def feature_vector(self) -> List[float]:
        """Fixed-length numeric summary so the ridge surrogate can order on the learning rule too."""
        feats = [1.0 if self.optimizer == o else 0.0 for o in _OPTIMIZERS]
        feats += [1.0 if name in self.aux else 0.0 for name in _AUX_OBJECTIVES]
        feats += [1.0 if self.schedule == s else 0.0 for s in _SCHEDULES]
        feats += [1.0 if self.plasticity else 0.0, self.lr / 0.01, self.weight_decay / 0.1]
        return feats


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
    norm_type: str = "layernorm"    # "layernorm" | "rmsnorm" — searchable normalization
    qk_norm: bool = False           # RMS-normalize q/k per head (training-stability win)
    kv_latent: int = 16             # latent-KV compression dim for mla_attention
    program: Optional[MixerProgram] = None   # the synthesised mixer (only when op == "synth")

    def to_dict(self) -> Dict[str, Any]:
        d = {"op": self.op, "norm": self.norm, "activation": self.activation,
             "n_head": self.n_head, "residual": self.residual, "expansion": self.expansion,
             "dropout": self.dropout, "n_kv_head": self.n_kv_head,
             "residual_scale": self.residual_scale, "n_experts": self.n_experts,
             "top_k": self.top_k, "norm_type": self.norm_type, "qk_norm": self.qk_norm,
             "kv_latent": self.kv_latent}
        if self.program is not None:
            d["program"] = self.program.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGene":
        kw = {k: d[k] for k in d if k in cls.__dataclass_fields__ and k != "program"}
        prog = d.get("program")
        if prog is not None:
            kw["program"] = MixerProgram.from_dict(prog)
        return cls(**kw)

    @classmethod
    def random(cls, rng: random.Random, n_embd: int, *,
               ops: Optional[Sequence[str]] = None, library: Any = None) -> "LayerGene":
        heads = [h for h in (1, 2, 4, 8) if n_embd % h == 0] or [1]
        nh = rng.choice(heads)
        kv_choices = [k for k in (1, 2, 4, 8) if k <= nh and nh % k == 0] or [nh]
        op = rng.choice(list(ops) if ops is not None else _OPS)
        prog = MixerProgram.random(rng, library=library) if op == "synth" else None
        return cls(op=op, norm=rng.choice(_NORMS),
                   activation=rng.choice(_ACTIVATIONS), n_head=nh,
                   residual=rng.random() < 0.85, expansion=rng.choice(_EXPANSIONS),
                   dropout=rng.choice(_DROPOUTS),
                   n_kv_head=rng.choice(kv_choices) if rng.random() < 0.5 else nh,
                   residual_scale=rng.choice([1.0, 0.7]),
                   n_experts=rng.choice(_N_EXPERTS), top_k=rng.choice(_MOE_TOPK),
                   norm_type=rng.choice(_NORM_TYPES), qk_norm=rng.random() < 0.5,
                   kv_latent=rng.choice(_KV_LATENTS), program=prog)


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
    n_predict: int = 1              # multi-token-prediction depth (1 = classic next-token only)
    learning_rule: LearningRule = field(default_factory=LearningRule)  # *how* this brain learns

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = [LayerGene(op="attention"), LayerGene(op="gated_mlp")]
        if self.learning_rule is None:
            self.learning_rule = LearningRule()
        self._fixup()

    def _fixup(self) -> None:
        """Keep the genome buildable: heads divide ``n_embd``, kv-heads divide heads, top_k≤experts,
        every synth layer carries a valid program, and the learning rule stays in-bounds."""
        if self.pos_encoding not in _POS_ENCODINGS:
            self.pos_encoding = "learned"
        self.n_predict = max(1, min(4, int(self.n_predict)))
        if self.learning_rule is None:
            self.learning_rule = LearningRule()
        self.learning_rule._fixup()
        for ly in self.layers:
            if self.n_embd % max(1, ly.n_head) != 0:
                ly.n_head = 1
            # grouped-query: 0 means "= n_head"; otherwise it must divide n_head
            if ly.n_kv_head <= 0 or ly.n_kv_head > ly.n_head or ly.n_head % ly.n_kv_head != 0:
                ly.n_kv_head = ly.n_head
            ly.n_experts = max(1, ly.n_experts)
            ly.top_k = max(1, min(ly.top_k, ly.n_experts))
            ly.expansion = ly.expansion if ly.expansion in _EXPANSIONS else 4
            ly.norm_type = ly.norm_type if ly.norm_type in _NORM_TYPES else "layernorm"
            ly.kv_latent = max(1, min(int(ly.kv_latent), self.n_embd))
            # a synthesised op must carry a valid program; a named op never does
            if ly.op == "synth":
                if ly.program is None:
                    ly.program = MixerProgram()
                ly.program._fixup()
            else:
                ly.program = None

    def to_dict(self) -> Dict[str, Any]:
        return {"n_embd": self.n_embd, "block_size": self.block_size,
                "layers": [ly.to_dict() for ly in self.layers],
                "ngram_order": self.ngram_order, "ngram_k": self.ngram_k, "seed": self.seed,
                "pos_encoding": self.pos_encoding, "tie_embeddings": self.tie_embeddings,
                "n_predict": self.n_predict, "learning_rule": self.learning_rule.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchitectureGenome":
        d = dict(d or {})
        layers = [LayerGene.from_dict(x) for x in d.get("layers", [])]
        return cls(n_embd=d.get("n_embd", 64), block_size=d.get("block_size", 32),
                   layers=layers, ngram_order=d.get("ngram_order", 3),
                   ngram_k=d.get("ngram_k", 1.0), seed=d.get("seed", 0),
                   pos_encoding=d.get("pos_encoding", "learned"),
                   tie_embeddings=bool(d.get("tie_embeddings", False)),
                   n_predict=int(d.get("n_predict", 1)),
                   learning_rule=LearningRule.from_dict(d.get("learning_rule")))

    @classmethod
    def random(cls, rng: random.Random, *, max_layers: int = 5, block_size: int = 32,
               pos_encoding: Optional[str] = None, ops: Optional[Sequence[str]] = None,
               search_learning_rule: bool = True,
               allow_plasticity: bool = True, library: Any = None) -> "ArchitectureGenome":
        n_embd = rng.choice(_EMBD_CHOICES)
        n_layer = rng.randint(2, max(2, max_layers))
        layers = [LayerGene.random(rng, n_embd, ops=ops, library=library) for _ in range(n_layer)]
        lr = (LearningRule.random(rng, allow_plasticity=allow_plasticity)
              if search_learning_rule else LearningRule())
        return cls(n_embd=n_embd, block_size=block_size, layers=layers,
                   ngram_order=rng.randint(2, 5), ngram_k=rng.choice([0.5, 1.0]),
                   seed=rng.randint(0, 1 << 30),
                   pos_encoding=pos_encoding or rng.choice(_POS_ENCODINGS),
                   tie_embeddings=rng.random() < 0.5,
                   n_predict=rng.choice(_N_PREDICT), learning_rule=lr)

    def mutate(self, rng: random.Random, *, max_layers: int = 6,
               ops: Optional[Sequence[str]] = None, search_learning_rule: bool = True,
               allow_plasticity: bool = True, library: Any = None) -> "ArchitectureGenome":
        """Return a mutated copy — add/drop/replace a layer, tweak a layer's knobs, swap the
        positional scheme / embedding tying, shift the n-gram substrate, evolve the *learning rule*
        (optimizer / loss objectives / schedule / plasticity), or rewrite a synthesised mixer."""
        g = ArchitectureGenome.from_dict(self.to_dict())
        choice = rng.randrange(9)
        if choice == 0 and len(g.layers) < max_layers:
            g.layers.insert(rng.randrange(len(g.layers) + 1),
                            LayerGene.random(rng, g.n_embd, ops=ops, library=library))
        elif choice == 1 and len(g.layers) > 1:
            del g.layers[rng.randrange(len(g.layers))]
        elif choice == 2:
            g.layers[rng.randrange(len(g.layers))] = LayerGene.random(rng, g.n_embd, ops=ops,
                                                                      library=library)
        elif choice == 3:
            ly = g.layers[rng.randrange(len(g.layers))]
            ly.activation = rng.choice(_ACTIVATIONS)
            ly.norm = rng.choice(_NORMS)
            ly.residual = not ly.residual
        elif choice == 4:
            # tweak a layer's capacity knobs (expansion / dropout / grouped-query / experts /
            # normalization / qk-norm / latent-KV width)
            ly = g.layers[rng.randrange(len(g.layers))]
            ly.expansion = rng.choice(_EXPANSIONS)
            ly.dropout = rng.choice(_DROPOUTS)
            ly.n_kv_head = rng.choice([k for k in (1, 2, 4, 8) if k <= ly.n_head] or [ly.n_head])
            ly.n_experts = rng.choice(_N_EXPERTS)
            ly.top_k = rng.choice(_MOE_TOPK)
            ly.residual_scale = rng.choice([1.0, 0.7])
            ly.norm_type = rng.choice(_NORM_TYPES)
            ly.qk_norm = not ly.qk_norm
            ly.kv_latent = rng.choice(_KV_LATENTS)
        elif choice == 5:
            # mutate the whole-genome wiring: positional scheme, embedding tying, MTP depth
            g.pos_encoding = rng.choice(_POS_ENCODINGS)
            g.tie_embeddings = not g.tie_embeddings
            g.n_predict = rng.choice(_N_PREDICT)
        elif choice == 6:
            g.ngram_order = max(1, min(8, g.ngram_order + rng.choice([-1, 1])))
        elif choice == 7:
            # evolve *how she learns* — a new learning paradigm, not a new architecture
            if search_learning_rule:
                g.learning_rule = g.learning_rule.mutate(rng, allow_plasticity=allow_plasticity)
            else:
                g.ngram_order = max(1, min(8, g.ngram_order + rng.choice([-1, 1])))
        else:
            # rewrite a synthesised mixer's program; if there is none, grow one
            synth = [ly for ly in g.layers if ly.op == "synth" and ly.program is not None]
            if synth:
                rng.choice(synth).program.mutate(rng)
            else:
                g.layers[rng.randrange(len(g.layers))] = LayerGene.random(rng, g.n_embd, ops=ops,
                                                                          library=library)
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
            n_predict=rng.choice([self.n_predict, other.n_predict]),
            learning_rule=LearningRule.from_dict(
                rng.choice([self.learning_rule, other.learning_rule]).to_dict()),
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
        heads = kv = experts = qk = rms = 0
        for ly in self.layers:
            counts[ly.op] = counts.get(ly.op, 0) + 1
            heads += ly.n_head
            kv += (ly.n_kv_head or ly.n_head)
            experts += ly.n_experts if ly.op == "moe_mlp" else 0
            qk += 1 if ly.qk_norm else 0
            rms += 1 if ly.norm_type == "rmsnorm" else 0
        feats = [self.n_embd / 64.0, n / 6.0, self.block_size / 64.0,
                 heads / n, kv / n, experts / n,
                 1.0 if self.pos_encoding == "rope" else 0.0,
                 1.0 if self.pos_encoding == "alibi" else 0.0,
                 1.0 if self.tie_embeddings else 0.0,
                 self.ngram_order / 5.0,
                 qk / n, rms / n, self.n_predict / 4.0]   # qk-norm / rmsnorm share, MTP depth
        feats += [counts[op] / n for op in _OPS]     # operator-family histogram (incl. synth)
        feats += self.learning_rule.feature_vector()  # *how* it learns — surrogate orders on it too
        return feats

    def estimated_flops(self) -> float:
        """A cheap, monotone proxy for forward cost (per token) — used by the hardware-aware
        fitness term and the Pareto front. Not exact; only the *ordering* matters."""
        ne, t = self.n_embd, max(1, self.block_size)
        total = 0.0
        for ly in self.layers:
            if ly.op == "diff_attention":                      # two attention maps
                total += 8.0 * ne * ne + 4.0 * t * ne
            elif ly.op == "mla_attention":                     # low-rank latent KV (cheaper KV)
                total += 2.0 * ne * ne + 2.0 * ne * ly.kv_latent + 2.0 * t * ne
            elif ly.op in _ATTENTION_OPS:
                total += 4.0 * ne * ne + 2.0 * t * ne          # projections + attention matmuls
            elif ly.op in ("conv_mix", "hyena_conv"):
                total += 2.0 * ne * ne + 8.0 * ne              # depthwise conv + projection
            elif ly.op == "low_rank_mix":
                total += 2.0 * ne * ne + 4.0 * t
            elif ly.op in ("recurrent_gate", "ssm_scan", "selective_ssm"):
                total += 3.0 * ne * ne
            elif ly.op == "gla_attention":                     # linear attention (sub-quadratic)
                total += 4.0 * ne * ne
            elif ly.op == "moe_mlp":
                total += (2.0 * ly.top_k + 0.1 * ly.n_experts) * ly.expansion * ne * ne
            elif ly.op == "synth":                             # sum the synthesised primitives' cost
                for s in (ly.program.steps if ly.program else ["proj"]):
                    base = _synth_base(s)
                    if base == "attn":
                        total += 4.0 * ne * ne + 2.0 * t * ne
                    elif base in ("conv", "ssm", "gate", "proj", "shift"):
                        total += 2.0 * ne * ne
                    elif base == "lowrank":
                        total += 2.0 * ne * ne + 4.0 * t
                    else:                                      # norm — cheap
                        total += 2.0 * ne
            else:                                              # gated_mlp / glu / swiglu
                total += 2.0 * ly.expansion * ne * ne
        return total * float(self.n_predict) ** 0.0 + 2.0 * ne * float(self.n_predict)  # +MTP heads


# --------------------------------------------------------------------------- #
# Torch building blocks — assembled dynamically from a genome
# --------------------------------------------------------------------------- #
if _HAS_TORCH:
    import torch.nn.functional as F  # type: ignore

    def _act(name: str) -> "nn.Module":
        return {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}.get(name, nn.GELU())

    class _RMSNorm(nn.Module):
        """Root-mean-square layer norm (LLaMA-style): rescale by the RMS, no mean-subtraction —
        cheaper and a strong default for deep nets. A searchable alternative to ``nn.LayerNorm``."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):  # type: ignore[override]
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x * rms * self.weight

    def _norm(name: str, dim: int) -> "nn.Module":
        return _RMSNorm(dim) if name == "rmsnorm" else nn.LayerNorm(dim)

    def _rms_qk(x: "torch.Tensor", eps: float = 1e-6) -> "torch.Tensor":
        """Per-head RMS-normalize the last dim of q/k (QK-norm) — tames attention-logit blow-up."""
        return x * x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()

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

        cacheable = True

        def __init__(self, n_embd: int, n_head: int, n_kv_head: int, block_size: int,
                     pos: str = "learned", dropout: float = 0.0, qk_norm: bool = False) -> None:
            super().__init__()
            n_head = n_head if n_embd % n_head == 0 else 1
            n_kv_head = n_kv_head if (n_kv_head and n_head % n_kv_head == 0) else n_head
            self.n_head, self.n_kv_head = n_head, n_kv_head
            self.hd = n_embd // n_head
            self.pos = pos
            self.qk_norm = bool(qk_norm)
            self.block_size = block_size
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

        def _qkv(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            b, t, _ = x.shape
            q = self.q(x).view(b, t, self.n_head, self.hd).transpose(1, 2)      # (B,H,T,D)
            k = self.k(x).view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
            v = self.v(x).view(b, t, self.n_kv_head, self.hd).transpose(1, 2)
            if self.qk_norm:
                q, k = _rms_qk(q), _rms_qk(k)
            if self.n_kv_head != self.n_head:                                   # grouped-query
                rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            return q, k, v

        def forward(self, x):  # type: ignore[override]
            b, t, _ = x.shape
            q, k, v = self._qkv(x)
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

        def step(self, x_t, state):  # type: ignore[override]
            """Incremental one-token decode with a K/V cache. ``state`` is (k_cache, v_cache, pos);
            returns (y_t, new_state). Bit-identical to ``forward`` over the same prefix."""
            b = x_t.size(0)
            q, k, v = self._qkv(x_t)                                            # (B,H,1,D)
            kc, vc, pos = state if state is not None else (None, None, 0)
            if self.pos == "rope":
                cos = self.rope_cos[pos:pos + 1]; sin = self.rope_sin[pos:pos + 1]
                q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
            k = k if kc is None else torch.cat([kc, k], dim=2)
            v = v if vc is None else torch.cat([vc, v], dim=2)
            k, v = k[:, :, -self.block_size:], v[:, :, -self.block_size:]       # bounded context
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)               # (B,H,1,Tk)
            if self.pos == "alibi":
                tk = k.size(2)
                dist = -(torch.arange(tk - 1, -1, -1, device=x_t.device).float())
                att = att + self.alibi[None, :, None, :] * dist[None, None, None]
            att = torch.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(b, 1, self.n_head * self.hd)
            return self.proj(y), (k, v, pos + 1)

    class _ConvMix(nn.Module):
        """A depthwise causal convolution over the sequence — a conv token-mixer."""

        cacheable = False        # depends on a window of raw inputs → full-recompute decode

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

        cacheable = False        # long implicit filter over the whole context → full-recompute

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

        cacheable = False        # mixing matrix is indexed by absolute position → full-recompute

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

        cacheable = True         # carries a hidden state → O(1) incremental decode

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

        def step(self, x_t, state):  # type: ignore[override]
            v = self.to_v(x_t)[:, 0]; g = torch.sigmoid(self.to_g(x_t))[:, 0]
            h = (g * 0.0) if state is None else state
            h = g * h + (1.0 - g) * v
            return self.proj(h).unsqueeze(1), h

    class _SSMScan(nn.Module):
        """A diagonal state-space scan with a LEARNED per-channel decay (a real S4/Mamba-lite):
        h_t = a ⊙ h_{t-1} + b_t,  y_t = C·h_t + D ⊙ x_t — the decay ``a`` is a trained parameter,
        not a function of the input, so it captures long-range structure attention cannot cheaply."""

        cacheable = True         # carries a hidden state → O(1) incremental decode

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

        def step(self, x_t, state):  # type: ignore[override]
            a = torch.sigmoid(self.a_log)
            b = self.in_proj(x_t)[:, 0]
            h = torch.zeros_like(b) if state is None else state
            h = a * h + (1.0 - a) * b
            return (self.C(h.unsqueeze(1)) + self.D * x_t), h

    class _SelectiveSSM(nn.Module):
        """A Mamba-style SELECTIVE state-space scan: the decay and input gate are **functions of the
        input** (data-dependent), so the model chooses what to remember per token —
        ``Δ_t = softplus(W_Δ x_t)``, ``ā_t = exp(-Δ_t·exp(A_log))``, ``h_t = ā_t·h_{t-1} + Δ_t·(B_t·x_t)``,
        ``y_t = C_t·h_t + D·x_t``. A genuine selective scan, the heart of modern SSM LLMs."""

        cacheable = True         # selective recurrence → O(1) incremental decode

        def __init__(self, n_embd: int) -> None:
            super().__init__()
            self.A_log = nn.Parameter(torch.zeros(n_embd))            # a = exp(-Δ·exp(A_log))
            self.x_proj = nn.Linear(n_embd, n_embd)                   # B_t (input gate content)
            self.dt = nn.Linear(n_embd, n_embd)                      # Δ_t (selective step size)
            self.C = nn.Linear(n_embd, n_embd)                      # C_t (output projection)
            self.D = nn.Parameter(torch.ones(n_embd))

        def _ad_b(self, x):
            delta = F.softplus(self.dt(x))                            # (B,T,C) > 0
            a = torch.exp(-delta * torch.exp(self.A_log))            # data-dependent decay
            b = delta * self.x_proj(x)                               # data-dependent input
            return a, b

        def forward(self, x):  # type: ignore[override]
            a, b = self._ad_b(x)
            h = torch.zeros_like(b[:, 0])
            outs = []
            for i in range(x.size(1)):
                h = a[:, i] * h + b[:, i]
                outs.append(h)
            y = torch.stack(outs, dim=1)
            return self.C(y) + self.D * x

        def step(self, x_t, state):  # type: ignore[override]
            a, b = self._ad_b(x_t)
            a, b = a[:, 0], b[:, 0]
            h = torch.zeros_like(b) if state is None else state
            h = a * h + b
            return (self.C(h.unsqueeze(1)) + self.D * x_t), h

    class _GatedLinearAttention(nn.Module):
        """Gated linear / retention attention (RetNet/GLA-style): a causal linear-attention token
        mixer with a learned per-channel decay, computed as a recurrent outer-product state
        ``S_t = γ·S_{t-1} + kᵀv``, ``y_t = (q·S_t)``, then output-gated. Sub-quadratic in time and
        O(1)-memory to decode — a real alternative to softmax attention."""

        cacheable = True         # recurrent KV-state form → O(1) incremental decode

        def __init__(self, n_embd: int, n_head: int) -> None:
            super().__init__()
            n_head = n_head if n_embd % n_head == 0 else 1
            self.n_head, self.hd = n_head, n_embd // n_head
            self.q = nn.Linear(n_embd, n_embd)
            self.k = nn.Linear(n_embd, n_embd)
            self.v = nn.Linear(n_embd, n_embd)
            self.g = nn.Linear(n_embd, n_embd)                       # output gate
            self.proj = nn.Linear(n_embd, n_embd)
            self.decay = nn.Parameter(torch.zeros(n_head))          # γ = sigmoid(decay) per head

        def _proj(self, x):
            b, t, _ = x.shape
            q = F.elu(self.q(x)) + 1.0
            k = F.elu(self.k(x)) + 1.0
            v = self.v(x)
            shp = (b, t, self.n_head, self.hd)
            return (q.view(*shp).transpose(1, 2), k.view(*shp).transpose(1, 2),
                    v.view(*shp).transpose(1, 2))                    # (B,H,T,D)

        def forward(self, x):  # type: ignore[override]
            b, t, c = x.shape
            q, k, v = self._proj(x)
            gamma = torch.sigmoid(self.decay).view(1, self.n_head, 1, 1)
            s = torch.zeros(b, self.n_head, self.hd, self.hd, device=x.device, dtype=x.dtype)
            outs = []
            for i in range(t):
                s = gamma * s + k[:, :, i].unsqueeze(-1) * v[:, :, i].unsqueeze(-2)   # (B,H,D,D)
                outs.append((q[:, :, i].unsqueeze(-2) @ s).squeeze(-2))               # (B,H,D)
            y = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(b, t, c)
            return self.proj(y * torch.sigmoid(self.g(x)))

        def step(self, x_t, state):  # type: ignore[override]
            b = x_t.size(0)
            q, k, v = self._proj(x_t)
            gamma = torch.sigmoid(self.decay).view(1, self.n_head, 1, 1)
            s = (torch.zeros(b, self.n_head, self.hd, self.hd, device=x_t.device, dtype=x_t.dtype)
                 if state is None else state)
            s = gamma * s + k[:, :, 0].unsqueeze(-1) * v[:, :, 0].unsqueeze(-2)
            y = (q[:, :, 0].unsqueeze(-2) @ s).squeeze(-2)                            # (B,H,D)
            y = y.reshape(b, 1, self.n_head * self.hd)
            return self.proj(y * torch.sigmoid(self.g(x_t))), s

    class _DiffAttention(nn.Module):
        """Differential attention: compute TWO causal softmax-attention maps and return their
        λ-weighted DIFFERENCE (``A = softmax(Q1Kᵀ) − λ·softmax(Q2Kᵀ)``). Subtracting a second map
        cancels common-mode attention noise, sharpening focus — a recent, real quality win."""

        cacheable = True

        def __init__(self, n_embd: int, n_head: int, block_size: int, pos: str = "learned",
                     dropout: float = 0.0, qk_norm: bool = False) -> None:
            super().__init__()
            self.a1 = _CausalAttention(n_embd, n_head, n_head, block_size, pos, dropout, qk_norm)
            self.a2 = _CausalAttention(n_embd, n_head, n_head, block_size, pos, dropout, qk_norm)
            self.lam = nn.Parameter(torch.tensor(0.5))

        def forward(self, x):  # type: ignore[override]
            return self.a1(x) - torch.sigmoid(self.lam) * self.a2(x)

        def step(self, x_t, state):  # type: ignore[override]
            s1, s2 = state if state is not None else (None, None)
            y1, s1 = self.a1.step(x_t, s1)
            y2, s2 = self.a2.step(x_t, s2)
            return y1 - torch.sigmoid(self.lam) * y2, (s1, s2)

    class _LatentAttention(nn.Module):
        """Multi-head latent attention (DeepSeek-MLA-style): the K/V are compressed through a tiny
        latent of width ``kv_latent`` (``x → c_kv → K,V``), so the attention's KV memory is a small
        low-rank bottleneck instead of the full width — much cheaper, with little quality loss."""

        cacheable = True

        def __init__(self, n_embd: int, n_head: int, block_size: int, kv_latent: int = 16,
                     pos: str = "learned", dropout: float = 0.0, qk_norm: bool = False) -> None:
            super().__init__()
            n_head = n_head if n_embd % n_head == 0 else 1
            self.n_head, self.hd = n_head, n_embd // n_head
            self.qk_norm, self.block_size = bool(qk_norm), block_size
            lat = max(1, min(kv_latent, n_embd))
            self.q = nn.Linear(n_embd, n_embd)
            self.kv_down = nn.Linear(n_embd, lat)                    # compress to the latent
            self.kv_up = nn.Linear(lat, 2 * n_embd)                 # reconstruct K and V
            self.proj = nn.Linear(n_embd, n_embd)
            self.drop = nn.Dropout(dropout)
            self.pos = "rope" if (pos == "rope" and self.hd % 2 == 0) else (
                "alibi" if pos == "alibi" else "learned")
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).bool())
            if self.pos == "rope":
                cs = _rope_tables(block_size, self.hd)
                self.register_buffer("rope_cos", cs[0]); self.register_buffer("rope_sin", cs[1])
            if self.pos == "alibi":
                self.register_buffer("alibi", _alibi_slopes(n_head))

        def _qkv(self, x):
            b, t, _ = x.shape
            q = self.q(x).view(b, t, self.n_head, self.hd).transpose(1, 2)
            k, v = self.kv_up(self.kv_down(x)).chunk(2, dim=-1)
            k = k.view(b, t, self.n_head, self.hd).transpose(1, 2)
            v = v.view(b, t, self.n_head, self.hd).transpose(1, 2)
            if self.qk_norm:
                q, k = _rms_qk(q), _rms_qk(k)
            return q, k, v

        def forward(self, x):  # type: ignore[override]
            b, t, c = x.shape
            q, k, v = self._qkv(x)
            if self.pos == "rope":
                cos, sin = self.rope_cos[:t], self.rope_sin[:t]
                q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            if self.pos == "alibi":
                dist = (torch.arange(t, device=x.device)[None, :]
                        - torch.arange(t, device=x.device)[:, None]).clamp(max=0).float()
                att = att + self.alibi[None, :, None, None] * dist[None, None]
            att = att.masked_fill(~self.tril[:t, :t], float("-inf"))
            att = self.drop(torch.softmax(att, dim=-1))
            y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
            return self.proj(y)

        def step(self, x_t, state):  # type: ignore[override]
            b = x_t.size(0)
            q, k, v = self._qkv(x_t)
            kc, vc, pos = state if state is not None else (None, None, 0)
            if self.pos == "rope":
                cos = self.rope_cos[pos:pos + 1]; sin = self.rope_sin[pos:pos + 1]
                q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
            k = k if kc is None else torch.cat([kc, k], dim=2)
            v = v if vc is None else torch.cat([vc, v], dim=2)
            k, v = k[:, :, -self.block_size:], v[:, :, -self.block_size:]
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            if self.pos == "alibi":
                tk = k.size(2)
                dist = -(torch.arange(tk - 1, -1, -1, device=x_t.device).float())
                att = att + self.alibi[None, :, None, :] * dist[None, None, None]
            att = torch.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(b, 1, self.n_head * self.hd)
            return self.proj(y), (k, v, pos + 1)

    class _GatedMLP(nn.Module):
        """A gated channel mixer — ``glu=True`` gives a true gated-linear-unit block."""

        cacheable = True         # pointwise over time → incremental decode is just forward

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

        def step(self, x_t, state):  # type: ignore[override]
            return self.forward(x_t), None

    class _SwiGLU(nn.Module):
        """A SwiGLU channel mixer (LLaMA-style): w2( silu(w1 x) ⊙ w3 x ) — a strong gated FFN."""

        cacheable = True         # pointwise over time → incremental decode is just forward

        def __init__(self, n_embd: int, expansion: int = 4, dropout: float = 0.0) -> None:
            super().__init__()
            hidden = max(1, expansion) * n_embd
            self.w1 = nn.Linear(n_embd, hidden)
            self.w3 = nn.Linear(n_embd, hidden)
            self.w2 = nn.Linear(hidden, n_embd)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):  # type: ignore[override]
            return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))

        def step(self, x_t, state):  # type: ignore[override]
            return self.forward(x_t), None

    class _MoEMLP(nn.Module):
        """A top-k routed Mixture-of-Experts channel mixer — a real sparse router over independent
        expert MLPs. ``forward`` returns ``(output, load_balance_aux)``; the aux term (Switch
        Transformer's load-balancing loss) is folded into training so the router spreads tokens
        across experts instead of collapsing onto one."""

        cacheable = True         # routing is pointwise over time → incremental decode is just forward

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

        def step(self, x_t, state):  # type: ignore[override]
            out, _aux = self.forward(x_t)        # routing is per-token; aux is a training-only term
            return out, None

    class _SynthGate(nn.Module):
        """A learned multiplicative gate primitive: x ⊙ σ(Wx) — pointwise over time (cacheable)."""

        cacheable = True

        def __init__(self, n_embd: int) -> None:
            super().__init__()
            self.g = nn.Linear(n_embd, n_embd)

        def forward(self, x):  # type: ignore[override]
            return x * torch.sigmoid(self.g(x))

        def step(self, x_t, state):  # type: ignore[override]
            return self.forward(x_t), state

    class _SynthProj(nn.Module):
        """A projection + activation primitive: act(Wx) — pointwise over time (cacheable)."""

        cacheable = True

        def __init__(self, n_embd: int, activation: str) -> None:
            super().__init__()
            self.lin = nn.Linear(n_embd, n_embd)
            self.act = _act(activation)

        def forward(self, x):  # type: ignore[override]
            return self.act(self.lin(x))

        def step(self, x_t, state):  # type: ignore[override]
            return self.forward(x_t), state

    class _SynthShift(nn.Module):
        """A causal token-shift mixer primitive: blend each position with the one before it
        (shift-by-1), then project — a real, parameter-cheap time-mixer (RWKV-style token-shift)."""

        cacheable = False        # uses the previous raw token → full-recompute decode (safe fallback)

        def __init__(self, n_embd: int) -> None:
            super().__init__()
            self.lin = nn.Linear(2 * n_embd, n_embd)

        def forward(self, x):  # type: ignore[override]
            prev = F.pad(x, (0, 0, 1, 0))[:, :-1, :]      # shift right along time, zero-pad position 0
            return self.lin(torch.cat([x, prev], dim=-1))

    class _SynthMixer(nn.Module):
        """An OPEN-ENDED, *synthesised* token-mixer: a short program of primitive ops composed into a
        brand-new causal mixer. This is how the Genesis search escapes the fixed palette — the
        genome's :class:`MixerProgram` lists the steps and each is a real, full (B,T,C)->(B,T,C)
        transform (attention core / conv / low-rank / ssm / gate / shift / proj / norm), so NYXARA
        can invent a mixer that exists on nobody's list."""

        cacheable = False        # a composed program may need the full context → full-recompute decode

        def __init__(self, n_embd: int, block_size: int, pos: str, gene: LayerGene) -> None:
            super().__init__()
            steps: List[nn.Module] = []
            for s in (gene.program.steps if gene.program else ["proj"]):
                base, param = _synth_base(s), _synth_param(s)
                if base == "attn":
                    steps.append(_CausalAttention(n_embd, gene.n_head, gene.n_head, block_size, pos,
                                                  gene.dropout, gene.qk_norm))
                elif base == "conv":                              # searched kernel width (per-step)
                    steps.append(_ConvMix(n_embd, kernel=min(block_size, param))
                                 if param else _ConvMix(n_embd))
                elif base == "lowrank":                           # searched low-rank rank (per-step)
                    steps.append(_LowRankMix(n_embd, block_size, rank=min(block_size, param))
                                 if param else _LowRankMix(n_embd, block_size))
                elif base == "ssm":
                    steps.append(_SSMScan(n_embd))
                elif base == "gate":
                    steps.append(_SynthGate(n_embd))
                elif base == "shift":
                    steps.append(_SynthShift(n_embd))
                elif base == "norm":
                    steps.append(_RMSNorm(n_embd))
                else:                                         # "proj" / default
                    steps.append(_SynthProj(n_embd, gene.activation))
            self.steps = nn.ModuleList(steps)
            self.out = nn.Linear(n_embd, n_embd)

        def forward(self, x):  # type: ignore[override]
            for m in self.steps:
                x = m(x)
            return self.out(x)

    def _build_inner(gene: LayerGene, n_embd: int, block_size: int, pos: str) -> "nn.Module":
        op = gene.op
        if op == "synth":
            return _SynthMixer(n_embd, block_size, pos, gene)
        if op == "attention":
            return _CausalAttention(n_embd, gene.n_head, gene.n_head, block_size, pos,
                                    gene.dropout, gene.qk_norm)
        if op == "gqa_attention":
            return _CausalAttention(n_embd, gene.n_head, gene.n_kv_head, block_size, pos,
                                    gene.dropout, gene.qk_norm)
        if op == "diff_attention":
            return _DiffAttention(n_embd, gene.n_head, block_size, pos, gene.dropout, gene.qk_norm)
        if op == "mla_attention":
            return _LatentAttention(n_embd, gene.n_head, block_size, gene.kv_latent, pos,
                                    gene.dropout, gene.qk_norm)
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
        if op == "selective_ssm":
            return _SelectiveSSM(n_embd)
        if op == "gla_attention":
            return _GatedLinearAttention(n_embd, gene.n_head)
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
            self.norm = _norm(gene.norm_type, n_embd)
            self.pre = gene.norm == "pre"
            self.residual = gene.residual
            self.residual_scale = float(gene.residual_scale)
            self.drop = nn.Dropout(gene.dropout)
            self.inner = _build_inner(gene, n_embd, block_size, pos)
            self.cacheable = bool(getattr(self.inner, "cacheable", False))
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

        def step(self, x_t, state):  # type: ignore[override]
            """One-token incremental forward (KV-cache path); mirrors ``forward`` exactly. Only
            called when every layer ``cacheable`` — the net falls back to full recompute otherwise."""
            inp = self.norm(x_t) if self.pre else x_t
            y, new_state = self.inner.step(inp, state)
            if not self.pre:
                y = self.norm(y)
            return (x_t + self.residual_scale * y if self.residual else y), new_state

    class _GenesisNet(nn.Module):
        """A byte-level decoder assembled dynamically from an :class:`ArchitectureGenome`.

        ``forward`` returns ``(logits, aux)`` where ``aux`` is the summed MoE load-balance loss
        (or ``None`` when no MoE layer is present)."""

        def __init__(self, genome: ArchitectureGenome) -> None:
            super().__init__()
            ne, bs = genome.n_embd, genome.block_size
            self.block_size = bs
            self.pos_encoding = genome.pos_encoding
            self.n_predict = max(1, int(genome.n_predict))
            self.tok = nn.Embedding(_VOCAB, ne)
            self.pos = nn.Embedding(bs, ne) if genome.pos_encoding == "learned" else None
            self.layers = nn.ModuleList(
                [_Layer(g, ne, bs, genome.pos_encoding) for g in genome.layers])
            self.ln_f = nn.LayerNorm(ne)
            self.head = nn.Linear(ne, _VOCAB, bias=False)
            if genome.tie_embeddings:
                self.head.weight = self.tok.weight   # weight tying (input emb == output head)
            # Multi-token-prediction: extra heads predict t+2, t+3, … from the same trunk feature,
            # giving denser supervision (DeepSeek-V3-style). Inert for ppl/generate (main head only).
            self.mtp_heads = nn.ModuleList([nn.Linear(ne, _VOCAB, bias=False)
                                            for _ in range(self.n_predict - 1)])
            # the cached one-token decode path is available only when every layer supports it
            self.cacheable = all(getattr(ly, "cacheable", False) for ly in self.layers)

        def _trunk(self, idx):
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
            return self.ln_f(x), aux

        def forward(self, idx):  # type: ignore[override]
            h, aux = self._trunk(idx)
            return self.head(h), aux

        def mtp_logits(self, idx):
            """Return the list of auxiliary multi-token-prediction logits (empty if n_predict==1)."""
            h, _ = self._trunk(idx)
            return [head(h) for head in self.mtp_heads]

        def step(self, idx_t, states, pos):
            """Cached one-token decode: ``idx_t`` is (B,1) newest token; ``states`` a per-layer list;
            ``pos`` the absolute position. Returns (logits_last, new_states). Requires ``cacheable``."""
            x = self.tok(idx_t)
            if self.pos is not None:
                x = x + self.pos(torch.tensor([min(pos, self.block_size - 1)],
                                              device=idx_t.device))[None]
            new_states: List[Any] = []
            for layer, st in zip(self.layers, states):
                x, ns = layer.step(x, st)
                new_states.append(ns)
            return self.head(self.ln_f(x))[:, -1, :], new_states

    class _Lion(torch.optim.Optimizer):
        """Lion (EvoLved Sign Momentum): update = sign(β1·m + (1-β1)·g); m ← β2·m + (1-β2)·g.
        One momentum buffer, sign-based step — a real, memory-light alternative to AdamW the
        searched learning rule can choose. Decoupled weight decay, like AdamW."""

        def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.99),
                     weight_decay: float = 0.0) -> None:
            super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

        @torch.no_grad()
        def step(self, closure=None):  # type: ignore[override]
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g, state = p.grad, self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                    m = state["m"]
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)                 # decoupled weight decay
                    update = m.mul(b1).add(g, alpha=1.0 - b1).sign_()   # uses m BEFORE its update
                    p.add_(update, alpha=-lr)
                    m.mul_(b2).add_(g, alpha=1.0 - b2)        # EMA momentum update
            return loss

    def _make_optimizer(rule: "LearningRule", params, lr: float, weight_decay: float):
        """Build the optimizer the searched learning rule chose (a real, different update rule)."""
        plist = list(params)
        if rule.optimizer == "lion":
            return _Lion(plist, lr=lr, weight_decay=weight_decay)
        if rule.optimizer == "sgd_momentum":
            return torch.optim.SGD(plist, lr=lr, momentum=0.9, weight_decay=weight_decay,
                                   nesterov=True)
        if rule.optimizer == "rmsprop":
            return torch.optim.RMSprop(plist, lr=lr, weight_decay=weight_decay, momentum=0.9)
        return torch.optim.AdamW(plist, lr=lr, weight_decay=weight_decay)


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
        # weight on the multi-token-prediction aux (only bites when n_predict > 1)
        self.mtp_weight = 0.3
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
                 batch_size: int = 16, lr: Optional[float] = None, warmup: Optional[float] = None,
                 weight_decay: Optional[float] = None, grad_clip: Optional[float] = None) -> TrainStats:
        """Micro-train from scratch under the genome's *searched learning rule* — NYXARA invents not
        only the architecture but *how it learns*: the optimizer family (AdamW / Lion / SGD-momentum /
        RMSprop), the LR-schedule shape (cosine / linear / WSD / constant), a composition of auxiliary
        self-supervised objectives layered on the main next-token CE (denoising / InfoNCE contrastive /
        EMA self-distillation / entropy regularization / sharpness-aware SAM), and an optional learnable
        Hebbian fast-weight plasticity update applied on top of backprop. Falls back to classic AdamW +
        CE + cosine for an old genome (learning_rule defaults). Loyalty + MoE + MTP terms are preserved.
        Deterministic given ``seed``; explicit kwargs override the learning rule; a superset of the
        contract."""
        start = time.monotonic()
        rule = self.genome.learning_rule
        lr = rule.lr if lr is None else lr
        warmup = rule.warmup if warmup is None else warmup
        weight_decay = rule.weight_decay if weight_decay is None else weight_decay
        grad_clip = rule.grad_clip if grad_clip is None else grad_clip
        data = self._encode("\n".join(corpus))
        bs = self.genome.block_size
        # Largest future offset any head needs: the main head predicts +1; each multi-token-
        # prediction head predicts a further-future token (offsets 2 … 1+len(mtp_heads)). Every
        # sampled window must leave room for the furthest target, or torch.stack on the MTP targets
        # gets a ragged batch (one short row at the tail of the data).
        max_off = max(1, 1 + len(self.net.mtp_heads))
        if len(data) < bs + max_off + 1:
            reps = (bs + max_off + 1) // max(1, len(data)) + 2
            data = (data * reps)[: max(bs * 4, bs + max_off + 1)]
        t = torch.tensor(data, dtype=torch.long, device=self.device)
        steps = max(1, steps)
        hi = len(data) - bs - max_off          # inclusive upper bound for a window start
        n_batch = max(1, min(batch_size, hi + 1))
        params = list(self.net.parameters())
        opt = _make_optimizer(rule, params, lr=lr, weight_decay=weight_decay)
        warm = max(1, int(steps * warmup))
        schedule = rule.schedule
        def lr_at(s: int) -> float:                          # warmup → searched decay shape
            if s < warm:
                return (s + 1) / warm
            prog = min(1.0, (s - warm) / max(1, steps - warm))
            if schedule == "linear":
                return 1.0 - prog
            if schedule == "wsd":                            # stable, then decay over the last 20%
                return 1.0 if prog < 0.8 else max(0.0, (1.0 - prog) / 0.2)
            if schedule == "constant":
                return 1.0
            return 0.5 * (1.0 + math.cos(math.pi * prog))    # cosine (default)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = random.Random(seed or self.genome.seed)
        torch.manual_seed(seed or self.genome.seed)
        # EMA teacher for self-distillation: a slow exponential-moving-average copy of the model that
        # supervises the student against its own smoothed beliefs (a real, different learning signal).
        use_distill = "self_distill" in rule.aux
        teacher = None
        if use_distill:
            teacher = copy.deepcopy(self.net).to(self.device)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
        self.net.train()
        last = 0.0

        def full_loss(x, y, ix) -> "torch.Tensor":
            """The composed objective: next-token CE + MoE aux + MTP + searched aux objectives +
            loyalty — every term real and folded into the same loss surface."""
            logits, aux = self.net(x)
            loss = nn.functional.cross_entropy(logits.view(-1, _VOCAB), y.reshape(-1))
            if aux is not None:                              # MoE load-balance term
                loss = loss + self.moe_aux_weight * aux
            if self.net.mtp_heads:                           # multi-token prediction (denser signal)
                for d, mlogits in enumerate(self.net.mtp_logits(x), start=2):
                    if bs - d <= 0:
                        break
                    yt = torch.stack([t[i + d:i + d + bs] for i in ix])
                    loss = loss + self.mtp_weight * nn.functional.cross_entropy(
                        mlogits[:, : bs - d].reshape(-1, _VOCAB), yt[:, : bs - d].reshape(-1))
            # ---- searched auxiliary self-supervised objectives (each ON TOP of CE) ---- #
            w = rule.aux.get("denoise", 0.0)
            if w > 0.0:                                       # input-noise denoising robustness
                mask = (torch.rand_like(x, dtype=torch.float) < 0.15)
                noise = torch.randint(0, _VOCAB, x.shape, device=x.device)
                x_noisy = torch.where(mask, noise, x)
                dlogits, _ = self.net(x_noisy)
                loss = loss + w * nn.functional.cross_entropy(
                    dlogits.view(-1, _VOCAB), y.reshape(-1))
            w = rule.aux.get("contrastive", 0.0)
            if w > 0.0:                                       # InfoNCE: adjacent trunk features attract
                h, _ = self.net._trunk(x)
                z = nn.functional.normalize(h, dim=-1)
                a = z[:, :-1].reshape(-1, z.size(-1))
                b = z[:, 1:].reshape(-1, z.size(-1))
                if a.size(0) > 1:
                    sim = (a @ b.t()) / 0.1
                    labels = torch.arange(a.size(0), device=x.device)
                    loss = loss + w * nn.functional.cross_entropy(sim, labels)
            w = rule.aux.get("self_distill", 0.0)
            if w > 0.0 and teacher is not None:               # KL against the EMA teacher's beliefs
                with torch.no_grad():
                    tlogits, _ = teacher(x)
                loss = loss + w * nn.functional.kl_div(
                    nn.functional.log_softmax(logits, dim=-1),
                    nn.functional.softmax(tlogits, dim=-1), reduction="batchmean")
            w = rule.aux.get("entropy_reg", 0.0)
            if w > 0.0:                                       # maximize predictive entropy (anti-overconfidence)
                p = nn.functional.softmax(logits, dim=-1)
                ent = -(p * nn.functional.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = loss - w * ent
            # L_total = L_intelligence + lambda * L_loyalty — JP's alignment in the loss surface
            if self.loyalty is not None and self.lambda_loyalty > 0.0:
                try:
                    loss = loss + self.lambda_loyalty * self.loyalty.aux_loss(self.net, self.device)
                except Exception:  # noqa: BLE001 — the loyalty term never crashes a training step
                    pass
            return loss

        use_sam = ("sam" in rule.aux) and not use_amp        # SAM needs two clean backward passes
        rho = 0.05
        for s in range(steps):
            ix = [rng.randint(0, hi) for _ in range(n_batch)]
            x = torch.stack([t[i:i + bs] for i in ix])
            y = torch.stack([t[i + 1:i + 1 + bs] for i in ix])
            if use_sam:                                      # sharpness-aware: ascend, then descend
                opt.zero_grad(set_to_none=True)
                loss = full_loss(x, y, ix)
                loss.backward()
                with torch.no_grad():
                    gnorm = torch.norm(torch.stack(
                        [p.grad.norm() for p in params if p.grad is not None])) + 1e-12
                    eps = {}
                    for p in params:
                        if p.grad is None:
                            continue
                        e_w = rho * p.grad / gnorm
                        p.add_(e_w); eps[p] = e_w
                opt.zero_grad(set_to_none=True)
                full_loss(x, y, ix).backward()              # gradient at the perturbed point
                with torch.no_grad():
                    for p, e_w in eps.items():
                        p.sub_(e_w)                          # restore weights
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step(); sched.step()
            else:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device, enabled=use_amp):
                    loss = full_loss(x, y, ix)
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
            # ---- Hebbian fast-weight plasticity: a local, non-backprop update layered on top ---- #
            if rule.plasticity:
                try:
                    with torch.no_grad():
                        h, _ = self.net._trunk(x)            # (B,T,C) trunk features
                        hf = h.reshape(-1, h.size(-1))
                        hf = hf / (hf.norm(dim=-1, keepdim=True) + 1e-6)
                        W = self.net.head.weight             # (V, C): bind feature → correct token
                        eta = rule.plasticity_lr * 0.01
                        W.mul_(1.0 - eta * (1.0 - rule.plasticity_decay))
                        W.index_add_(0, y.reshape(-1), (eta * hf).to(W.dtype))
                except Exception:  # noqa: BLE001 — plasticity never crashes a training step
                    pass
            # update the EMA teacher toward the student (slow exponential moving average)
            if teacher is not None:
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), self.net.parameters()):
                        tp.mul_(0.99).add_(sp, alpha=0.01)
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
                 repetition_penalty: Optional[float] = None, greedy: bool = False,
                 use_cache: bool = False, best_of: int = 1) -> str:
        """Autoregressive decode with temperature / top-k / top-p (nucleus) / repetition-penalty
        controls (defaults fall back to the model's configured decoding settings). ``greedy=True``
        is deterministic argmax. ``use_cache=True`` runs the O(1)-per-token KV-cached decode when the
        architecture supports it (bit-identical for lengths ≤ block_size, else a transparent
        fall-back). ``best_of>1`` is test-time self-consistency: sample N continuations and return
        the one the model itself scores most likely (lowest self-perplexity)."""
        if best_of > 1 and not greedy:
            cands = [self.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                   top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                   greedy=False, use_cache=use_cache, best_of=1)
                     for _ in range(best_of)]
            return min(cands, key=lambda c: self.perplexity(prompt + c))
        self.net.eval()
        torch.manual_seed(self.genome.seed)
        temp = self.temperature if temperature is None else temperature
        tk = self.top_k if top_k is None else top_k
        tp = self.top_p if top_p is None else top_p
        rep = self.repetition_penalty if repetition_penalty is None else repetition_penalty
        idx = self._encode(prompt) or [ord("\n")]
        bs = self.genome.block_size
        start = len(self._encode(prompt))
        cached = use_cache and bool(getattr(self.net, "cacheable", False))

        def sample(logits: "torch.Tensor", recent: Sequence[int]) -> int:
            if rep > 1.0:                                    # discourage repeating recent tokens
                for tok in set(recent):
                    logits[tok] = logits[tok] / rep if logits[tok] > 0 else logits[tok] * rep
            if greedy or temp <= 0:
                return int(torch.argmax(logits).item())
            lg = logits / max(1e-6, temp)
            if tk and tk > 0:                                # top-k truncation
                kth = torch.topk(lg, min(tk, lg.numel())).values[-1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            probs = torch.softmax(lg, dim=-1)
            if 0.0 < tp < 1.0:                               # nucleus (top-p) truncation
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                keep = cum - sp <= tp
                keep[0] = True
                probs = torch.zeros_like(probs).scatter_(0, si, sp * keep)
                probs = probs / probs.sum().clamp_min(1e-9)
            return int(torch.multinomial(probs, 1).item())

        with torch.no_grad():
            if cached:
                states: List[Any] = [None] * len(self.net.layers)
                logits = None
                for p, tok in enumerate(idx):               # warm the cache over the prompt
                    logits, states = self.net.step(
                        torch.tensor([[tok]], device=self.device), states, p)
                for p in range(len(idx), len(idx) + max_tokens):
                    nxt = sample(logits.squeeze(0).clone(), idx[-bs:])
                    idx.append(nxt)
                    logits, states = self.net.step(
                        torch.tensor([[nxt]], device=self.device), states, p)
            else:
                for _ in range(max_tokens):
                    ctx = torch.tensor(idx[-bs:], dtype=torch.long, device=self.device).unsqueeze(0)
                    logits, _ = self.net(ctx)
                    idx.append(sample(logits[:, -1, :].squeeze(0), idx[-bs:]))
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


def inherit_compatible_weights(dst_net: Any, src_state: Dict[str, Any]) -> int:
    """Network-morphism weight transfer: copy every tensor from ``src_state`` whose name AND shape
    match a parameter/buffer of ``dst_net`` (a Lamarckian warm-start so a mutated child resumes its
    parent's training instead of restarting from noise). Returns how many tensors were inherited.
    Pure no-op without torch. Mismatched layers are simply left freshly-initialised."""
    if not _HAS_TORCH:
        return 0
    dst = dst_net.state_dict()
    copied = 0
    for name, tensor in dst.items():
        src = src_state.get(name)
        if src is not None and hasattr(src, "shape") and tuple(src.shape) == tuple(tensor.shape):
            tensor.copy_(src)
            copied += 1
    dst_net.load_state_dict(dst)
    return copied


# --------------------------------------------------------------------------- #
# Champion ensemble — route each input to its most competent sub-brain (real MoE inference)
# --------------------------------------------------------------------------- #
class EnsembleModel(BaseLanguageModel):
    """A real inference-time ensemble of already-trained sub-models (the top-k Pareto brains).

    Heterogeneous backends share no per-token probability API, so this is an honest
    **competence-routing** ensemble (hard mixture-of-experts): for any input it routes to the
    member that scores it best (lowest perplexity) and speaks with that expert — strictly ≥ the
    best single member on its own metric, and never worse than the strongest brain. Implements the
    full :class:`BaseLanguageModel` contract so the foundry / SelfProvider can treat it as one model."""

    kind = "ensemble"

    def __init__(self, members: Sequence[BaseLanguageModel]) -> None:
        self.members: List[BaseLanguageModel] = [m for m in members if m is not None]
        if not self.members:
            raise ValueError("an ensemble needs at least one member")

    def _route(self, text: str) -> BaseLanguageModel:
        """Pick the member most competent on this context (lowest perplexity)."""
        if len(self.members) == 1 or not text:
            return self.members[0]
        return min(self.members, key=lambda m: _safe_ppl(m, text))

    def train_on(self, corpus: Sequence[str], *, steps: int = 0, seed: int = 0) -> TrainStats:
        stats = [m.train_on(corpus, steps=steps, seed=seed) for m in self.members]
        return TrainStats(steps=max((s.steps for s in stats), default=0),
                          final_loss=min((s.final_loss for s in stats), default=0.0),
                          seconds=sum(s.seconds for s in stats),
                          tokens=max((s.tokens for s in stats), default=0))

    def perplexity(self, text: str) -> float:
        return min(_safe_ppl(m, text) for m in self.members)   # the best expert's score

    def generate(self, prompt: str, *, max_tokens: int = 128, **kw: Any) -> str:
        member = self._route(prompt)
        try:
            return member.generate(prompt, max_tokens=max_tokens, **kw)
        except TypeError:                                      # member lacks the rich-decode kwargs
            return member.generate(prompt, max_tokens=max_tokens)

    def param_count(self) -> int:
        return sum(m.param_count() for m in self.members)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Persist each member's reconstruction spec when it carries one (e.g. GenesisModel,
        # NanoGPT) so load() can rebuild the exact architecture; stdlib members (kngram /
        # ngram) are self-describing in their own model.json, so None is fine for them.
        specs: List[Optional[Dict[str, Any]]] = []
        for m in self.members:
            spec = getattr(m, "spec", None)
            specs.append(spec.to_dict() if hasattr(spec, "to_dict") else None)
        (directory / "ensemble.json").write_text(
            json.dumps({"members": len(self.members),
                        "kinds": [getattr(m, "kind", "?") for m in self.members],
                        "specs": specs}),
            encoding="utf-8")
        for i, m in enumerate(self.members):
            m.save(directory / f"member_{i}")

    def load(self, directory: Path) -> None:
        """Rebuild the ensemble from disk: reconstruct each member's architecture and reload
        its trained weights — so a Genesis champion survives a restart, not just a re-train.

        Mirrors the canonical foundry reload (``build_model(ModelSpec) → model.load(dir)``)
        and degrades honestly: a member that cannot be reloaded (e.g. its backend's deps are
        gone on this machine) is skipped rather than aborting the whole ensemble."""
        directory = Path(directory)
        meta = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
        kinds: List[str] = meta.get("kinds", [])
        specs: List[Optional[Dict[str, Any]]] = meta.get("specs", [None] * meta.get("members", 0))
        rebuilt: List[BaseLanguageModel] = []
        for i in range(int(meta.get("members", 0))):
            member_dir = directory / f"member_{i}"
            kind = kinds[i] if i < len(kinds) else "?"
            spec = specs[i] if i < len(specs) else None
            m = self._reload_member(member_dir, kind, spec)
            if m is not None:
                rebuilt.append(m)
        if not rebuilt:
            raise ValueError(f"could not reload any ensemble member from {directory}")
        self.members = rebuilt

    @staticmethod
    def _reload_member(member_dir: Path, kind: str,
                       spec: Optional[Dict[str, Any]]) -> Optional[BaseLanguageModel]:
        """Reconstruct one member and reload its weights; return None on unrecoverable failure."""
        from nyxara.growth.foundry_models import NgramByteLM, build_model  # lazy: avoid cycle
        # 1) Preferred: a recorded spec (or a member-local spec.json) → canonical rebuild.
        spec_dict = spec
        if spec_dict is None and (member_dir / "spec.json").exists():
            try:
                spec_dict = json.loads((member_dir / "spec.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                spec_dict = None
        if spec_dict is not None:
            try:
                model = build_model(ModelSpec.from_dict(spec_dict))
                model.load(member_dir)
                return model
            except Exception:  # noqa: BLE001 — backend/deps may differ on this machine
                pass
        # 2) Self-describing stdlib members (kngram / ngram) carry everything in model.json.
        if (member_dir / "model.json").exists():
            try:
                blob = json.loads((member_dir / "model.json").read_text(encoding="utf-8"))
                mk = blob.get("kind", kind)
                model = (NgramByteLM() if mk == "ngram" else WordKNGramLM())
                model.load(member_dir)
                return model
            except Exception:  # noqa: BLE001
                pass
        # 3) Last resort: rebuild a bare model of the recorded kind (untrained but functional).
        try:
            model = build_model(ModelSpec(kind=kind if kind and kind != "?" else "kngram"))
            try:
                model.load(member_dir)
            except Exception:  # noqa: BLE001 — no reloadable weights; keep the fresh model
                pass
            return model
        except Exception:  # noqa: BLE001
            return None


def _safe_ppl(model: BaseLanguageModel, text: str) -> float:
    try:
        return model.perplexity(text)
    except Exception:  # noqa: BLE001 — a member that cannot score this text simply abstains
        return float("inf")


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


def _fast_nondominated_sort(cands: Sequence["Candidate"]) -> List[List["Candidate"]]:
    """Deb's fast non-dominated sort (NSGA-II): partition candidates into successive Pareto fronts,
    front 0 being the non-dominated set, front 1 what remains once front 0 is removed, and so on."""
    cands = list(cands)
    dominated: Dict[int, List[int]] = {i: [] for i in range(len(cands))}
    n_dom = [0] * len(cands)
    fronts: List[List[int]] = [[]]
    for i, ci in enumerate(cands):
        for j, cj in enumerate(cands):
            if i == j:
                continue
            if _dominates(ci, cj):
                dominated[i].append(j)
            elif _dominates(cj, ci):
                n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)
    f = 0
    while fronts[f]:
        nxt: List[int] = []
        for i in fronts[f]:
            for j in dominated[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        f += 1
        fronts.append(nxt)
    return [[cands[i] for i in front] for front in fronts if front]


def _crowding_distance(front: Sequence["Candidate"]) -> Dict[int, float]:
    """NSGA-II crowding distance: how isolated each member is on the front (boundary points get ∞).
    Keyed by ``id(candidate)``; used as the diversity-preserving secondary sort within a front."""
    dist: Dict[int, float] = {id(c): 0.0 for c in front}
    if len(front) <= 2:
        return {id(c): float("inf") for c in front}
    objectives = (lambda c: c.quality, lambda c: -c.params,
                  lambda c: -c.seconds, lambda c: -c.flops)
    for obj in objectives:
        ordered = sorted(front, key=obj)
        lo, hi = obj(ordered[0]), obj(ordered[-1])
        span = (hi - lo) or 1.0
        dist[id(ordered[0])] = dist[id(ordered[-1])] = float("inf")
        for k in range(1, len(ordered) - 1):
            dist[id(ordered[k])] += (obj(ordered[k + 1]) - obj(ordered[k - 1])) / span
    return dist


def _nsga2_rank(cands: Sequence["Candidate"]) -> List["Candidate"]:
    """Order candidates the NSGA-II way: by front, then by descending crowding distance within a
    front (the elitist multi-objective ranking that drives the population toward the whole Pareto
    set rather than collapsing it to one scalar winner)."""
    ordered: List[Candidate] = []
    for front in _fast_nondominated_sort(cands):
        cd = _crowding_distance(front)
        ordered.extend(sorted(front, key=lambda c: cd[id(c)], reverse=True))
    return ordered


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
        """True when the neural topology was actually built and trained — on the torch path
        (``genesis``) OR the real pure-NumPy substrate (``genesis_np``).

        Only the bare ``kngram`` fallback (neither torch nor NumPy, or a layer-less genome) leaves
        the layer genome *inert*; this flag keeps the report honest about which actually happened."""
        return self.kind in ("genesis", "genesis_np")

    def describe(self) -> str:
        """An honest, substrate-aware description of what was actually scored."""
        if self.kind == "genesis":
            return self.genome.describe()
        if self.kind == "genesis_np":
            return f"{self.genome.describe()} — built + trained on the NumPy substrate (no torch)"
        return (f"word n-gram substrate (order={self.genome.ngram_order}, "
                f"smoothing={self.genome.learning_rule.smoothing}) — neural topology inert "
                f"without torch/NumPy [latent: {self.genome.describe()}]")

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
        """Whether real neural architectures were built and trained — on the torch path
        (``genesis``) OR the real pure-NumPy substrate (``genesis_np``). Only the bare ``kngram``
        fallback leaves the layer topology with no effect on the score."""
        return self.champion_kind in ("genesis", "genesis_np")

    @property
    def note(self) -> str:
        """A one-line honest caveat so 'she designed her own brain' is never over- or under-read."""
        if self.champion_kind == "genesis":
            return ("neural architecture search: real PyTorch topologies were built, "
                    "micro-trained and scored — the layer design drives fitness")
        if self.champion_kind == "genesis_np":
            return ("neural architecture search (NumPy substrate, no torch): NYXARA's OWN designed "
                    "topology — her layers, her synthesized mixer and her searched optimizer — was "
                    "really built, trained and scored on real held-out perplexity; the layer design "
                    "drives fitness. Small-scale but genuine; install .[foundry] (torch/GPU) to "
                    "search at larger scale")
        return ("word n-gram substrate search (no torch or NumPy): the neural layer topology was NOT "
                "built or trained and does not affect the score; only the n-gram order AND the "
                "searched smoothing method do. Each candidate is scored over multiple resampled "
                "folds (denoised)")

    def champion_describe(self) -> str:
        """Substrate-aware champion description (honest on every path)."""
        if self.champion_kind == "genesis":
            return self.champion.describe()
        if self.champion_kind == "genesis_np":
            return f"{self.champion.describe()} — built + trained on the NumPy substrate (no torch)"
        return (f"word n-gram substrate (order={self.champion.ngram_order}, "
                f"smoothing={self.champion.learning_rule.smoothing}) — neural topology inert "
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

    def __init__(self, ridge: float = 1.0, min_train: int = 8, beta: float = 0.0) -> None:
        self.ridge = ridge
        self.min_train = min_train
        self.beta = float(beta)          # UCB exploration weight (0 → pure-exploit ordering)
        self._w: Any = None
        self._X: Any = None              # kept for the kNN-residual uncertainty estimate
        self._resid_scale: float = 0.0
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
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)   # bias column
        a = Xb.T @ Xb + self.ridge * np.eye(Xb.shape[1])
        try:
            self._w = np.linalg.solve(a, Xb.T @ y)
        except Exception:  # noqa: BLE001 — singular system: skip prediction this round
            self._w = None
            return False
        self._X = X                                            # train features for the UCB σ term
        self._resid_scale = float(np.std(y - Xb @ self._w)) or 1.0
        return True

    def predict(self, genome: "ArchitectureGenome") -> float:
        """Predicted fitness (the mean) — used to ORDER which genomes to score first. Never crowns."""
        if not self.ready():
            return 0.0
        np = self._np
        x = np.array(genome.feature_vector() + [1.0], dtype=float)
        return float(x @ self._w)

    def uncertainty(self, genome: "ArchitectureGenome") -> float:
        """A cheap epistemic-uncertainty proxy: distance to the nearest already-scored genome in
        feature space, scaled by the model's residual spread. Far-from-seen genomes look uncertain."""
        if not self.ready() or self._X is None or len(self._X) == 0:
            return 0.0
        np = self._np
        x = np.array(genome.feature_vector(), dtype=float)
        d = float(np.min(np.linalg.norm(self._X - x[None, :], axis=1)))
        return self._resid_scale * d

    def acquire(self, genome: "ArchitectureGenome") -> float:
        """The UCB acquisition score = predicted mean + β·uncertainty — balances exploiting the
        predictor against exploring genomes unlike anything scored yet. β=0 reproduces ``predict``."""
        return self.predict(genome) + self.beta * self.uncertainty(genome)


# --------------------------------------------------------------------------- #
# Lifelong memory — the Hall of Fame (she remembers her best brains and grows)
# --------------------------------------------------------------------------- #
class HallOfFame:
    """Persistent architectural memory: every crowned champion and Pareto elite is recorded to disk,
    and future searches *warm-start* from these past elites instead of pure random noise. This is
    what makes the Genesis Protocol **lifelong** — across idle cycles she accumulates architectural
    wisdom, so each search starts from the shoulders of every brain she has ever designed. Pure
    standard library (JSON); safe to use everywhere. Promotion is still gauntlet-gated — the Hall of
    Fame only seeds the *search*, it never crowns or ships a brain on its own."""

    def __init__(self, path: Optional[Path] = None, *, capacity: int = 32) -> None:
        self.path = Path(path) if path else None
        self.capacity = max(1, int(capacity))
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path and self.path.exists():
            try:
                self.entries = list(json.loads(self.path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — a corrupt memory file never crashes a search
                self.entries = []

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def record(self, report: "GenesisReport") -> None:
        """Fold a finished search into memory: the champion plus its Pareto elites, deduped by
        topology fingerprint, kept to the best ``capacity`` by recorded fitness."""
        new = [{"genome": report.champion.to_dict(), "fitness": report.champion_fitness,
                "perplexity": report.champion_perplexity, "backend": report.backend}]
        for c in report.pareto_front:
            new.append({"genome": c.genome.to_dict(), "fitness": c.fitness,
                        "perplexity": c.perplexity, "backend": report.backend})
        by_fp: Dict[str, Dict[str, Any]] = {}
        for e in self.entries + new:
            try:
                fp = ArchitectureGenome.from_dict(e["genome"]).fingerprint()
            except Exception:  # noqa: BLE001 — skip an unreadable entry
                continue
            if fp not in by_fp or e.get("fitness", 0.0) > by_fp[fp].get("fitness", 0.0):
                by_fp[fp] = e
        self.entries = sorted(by_fp.values(), key=lambda e: e.get("fitness", 0.0),
                              reverse=True)[: self.capacity]
        self._save()

    def seed_genomes(self, n: int, rng: random.Random, *, block_size: int) -> List[ArchitectureGenome]:
        """Up to ``n`` warm-start genomes from memory: the best elites, lightly re-seeded and
        re-fit to the requested ``block_size`` so they slot straight into a fresh population."""
        out: List[ArchitectureGenome] = []
        for e in self.entries[: max(0, n)]:
            try:
                g = ArchitectureGenome.from_dict(e["genome"])
                g.block_size = block_size
                g.seed = rng.randint(0, 1 << 30)
                g._fixup()
                out.append(g)
            except Exception:  # noqa: BLE001 — a bad entry simply contributes no seed
                continue
        return out

    def best(self) -> Optional[ArchitectureGenome]:
        if not self.entries:
            return None
        try:
            return ArchitectureGenome.from_dict(self.entries[0]["genome"])
        except Exception:  # noqa: BLE001
            return None

    def __len__(self) -> int:
        return len(self.entries)


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
        # ---- open-ended search space derived from config ---- #
        # Whether she invents new learning paradigms (learning rule), open-ended operators (synth),
        # and Hebbian plasticity. Off → the classic fixed-palette + AdamW search is recovered.
        self._search_lr = bool(getattr(self.cfg, "search_learning_rule", True))
        self._search_ops = bool(getattr(self.cfg, "search_operators", True))
        self._plasticity = bool(getattr(self.cfg, "plasticity_enabled", True))
        self._synth_cap = int(getattr(self.cfg, "max_synth_primitives", 4))
        # the operator palette this search may draw from (drop ``synth`` when operator search is off)
        self._ops: Tuple[str, ...] = _OPS if self._search_ops else tuple(o for o in _OPS if o != "synth")
        # the self-extending primitive library: crowned synth mixers become reusable building blocks
        # the search composes over. Only active when operator synthesis is on (nothing to grow otherwise).
        self._library: Optional[PrimitiveLibrary] = None
        if self._search_ops and bool(getattr(self.cfg, "primitive_library", True)):
            self._library = PrimitiveLibrary(
                self._primlib_path(),
                capacity=int(getattr(self.cfg, "primitive_library_size", 48)))
        self._champion: Optional[Candidate] = None
        self._last_report: Optional[GenesisReport] = None
        self._reports: List[GenesisReport] = []
        # Lifelong memory: load the Hall of Fame so this search can warm-start from every elite
        # brain she has ever designed (and record the new champion back into it).
        self.hall_of_fame: Optional[HallOfFame] = None
        if bool(getattr(self.cfg, "hall_of_fame", True)):
            self.hall_of_fame = HallOfFame(self._hof_path(),
                                           capacity=int(getattr(self.cfg, "hall_of_fame_size", 32)))
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

    def _hof_path(self) -> Optional[Path]:
        """Where the lifelong Hall-of-Fame memory lives — under the configured data dir, mirroring
        how the Foundry roots its own state. ``None`` (in-memory only) if no data dir is configured."""
        try:
            data_dir = self.settings.paths.data_dir   # type: ignore[union-attr]
            return Path(data_dir) / "genesis" / "hall_of_fame.json"
        except Exception:  # noqa: BLE001 — no settings/paths: keep memory in-process only
            return None

    def _primlib_path(self) -> Optional[Path]:
        """Where the self-extending primitive library lives — beside the Hall of Fame under the data
        dir. ``None`` (in-memory only) when no data dir is configured."""
        try:
            data_dir = self.settings.paths.data_dir   # type: ignore[union-attr]
            return Path(data_dir) / "genesis" / "primitive_library.json"
        except Exception:  # noqa: BLE001 — no settings/paths: keep the library in-process only
            return None

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
        # Torch-free, but the designed topology is NOT inert: when the substrate is "numpy"/"auto"
        # build NYXARA's own architecture for REAL in pure NumPy — her layers, her synthesized mixer,
        # her searched optimizer, all actually built, trained and graded (GenesisNumpyModel). This is
        # what makes "she designs and trains her own brain, herself" real on a bare machine. "ngram"
        # selects the faster always-on word-Kneser-Ney substrate (layer topology inert).
        substrate = str(getattr(self.cfg, "substrate", "auto")).lower()
        if substrate in ("numpy", "auto"):
            try:
                from nyxara.growth.genesis_numpy import GenesisNumpyModel, _HAS_NUMPY
                if _HAS_NUMPY and genome.layers:
                    return GenesisNumpyModel(genome, seed=seed)
            except Exception:  # noqa: BLE001 — numpy build failed; fall to the n-gram substrate
                pass
        # Final fallback: the COHERENT word-level n-gram (not byte gibberish), whose *smoothing
        # method* is itself the searched learning rule, so a search still runs on the barest machine.
        return WordKNGramLM(order=max(2, genome.ngram_order), seed=seed,
                            smoothing=genome.learning_rule.smoothing)

    # ---- scoring one architecture (averaged over folds → a denoised estimate) ---- #
    def _evaluate(self, genome: ArchitectureGenome,
                  folds: Sequence[Tuple[Sequence[str], Sequence[str]]], backend: str,
                  *, steps: Optional[int] = None) -> Candidate:
        t0 = time.monotonic()
        # honest kind: derive it from the model actually built (torch GenesisModel → "genesis";
        # the real NumPy brain → "genesis_np"; the n-gram fallback → "kngram"), not from a guess.
        kind = "genesis" if (backend == "torch" and _HAS_TORCH) else "kngram"
        train_steps = int(self.cfg.micro_train_steps if steps is None else steps)
        fold_pps: List[float] = []
        params, align, factor = 0, 1.0, 1.0
        for i, (train_texts, eval_texts) in enumerate(folds):
            try:
                model = self._build_substrate(genome, backend, seed=genome.seed + 7919 * i)
                kind = getattr(model, "kind", kind)
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
        # the topology is real on both the torch and the NumPy substrate, so the FLOPs proxy applies
        flops = genome.estimated_flops() if kind in ("genesis", "genesis_np") else 0.0
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
        # NSGA-II ranks the population by Pareto front + crowding distance (multi-objective), so
        # selection drives toward the whole speed↔smartness↔cost frontier, not one scalar winner.
        nsga_order = _nsga2_rank(scored) if strategy == "nsga2" else None

        def tournament() -> ArchitectureGenome:
            k = min(len(scored), max(2, int(getattr(self.cfg, "tournament_k", 3))))
            if nsga_order is not None:                          # binary tournament on NSGA rank
                rankpos = {id(c): i for i, c in enumerate(nsga_order)}
                return min(rng.sample(scored, k), key=lambda c: rankpos[id(c)]).genome
            return max(rng.sample(scored, k), key=lambda c: c.fitness).genome

        mkw = dict(max_layers=max_layers, ops=self._ops, search_learning_rule=self._search_lr,
                   allow_plasticity=self._plasticity, library=self._library)

        def propose() -> ArchitectureGenome:
            if strategy == "elitism":
                elite = [c.genome for c in scored[:max(1, pop_size // 3)]]
                if len(elite) >= 2 and rng.random() >= mutation_rate:
                    a, b = rng.sample(elite, 2)
                    return self._conform(a.crossover(b, rng, max_layers=max_layers))
                return self._conform(rng.choice(elite).mutate(rng, **mkw))
            # tournament / regularized / nsga2 all select parents by tournament
            if rng.random() >= mutation_rate and len(scored) >= 2:
                return self._conform(tournament().crossover(tournament(), rng, max_layers=max_layers))
            return self._conform(tournament().mutate(rng, **mkw))

        # elitism / nsga2 carry the best forward; tournament/regularized (aging) do not — the oldest
        # die out each generation, which keeps the search exploring instead of locking in early.
        nxt: List[ArchitectureGenome] = []
        if strategy == "elitism":
            nxt = [c.genome for c in scored[:max(1, pop_size // 3)]]
        elif strategy == "nsga2" and nsga_order is not None:
            nxt = [c.genome for c in nsga_order[:max(1, pop_size // 3)]]   # elitist front-filling
        elif strategy == "tournament":
            nxt = [scored[0].genome]            # keep the single best (mild elitism)

        oversample = 3 if (surrogate is not None and surrogate.ready() and novelty_w >= 0) else 1
        pool_cands = scored
        while len(nxt) < pop_size:
            proposals = [propose() for _ in range(oversample)]
            if surrogate is not None and surrogate.ready() and len(proposals) > 1:
                # surrogate-assisted: keep the proposal the predictor likes best (by UCB acquisition,
                # so it explores too), plus an optional novelty bonus so the population stays diverse
                def score(g: ArchitectureGenome) -> float:
                    nov = self._novelty(g, pool_cands) if novelty_w > 0 else 0.0
                    return surrogate.acquire(g) + novelty_w * nov
                nxt.append(max(proposals, key=score))
            else:
                nxt.append(proposals[0])
        return nxt[:pop_size]

    def _conform(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Keep a produced genome inside the configured search space: trim synthesised programs to
        the primitive cap and, when a search dimension is disabled, fall back to the classic default
        (so ``search_learning_rule=False`` / ``plasticity_enabled=False`` recover the old behaviour)."""
        for ly in genome.layers:
            if ly.op == "synth" and ly.program is not None:
                ly.program.steps = ly.program.steps[:max(1, self._synth_cap)]
        if not self._search_lr:
            genome.learning_rule = LearningRule()
        elif not self._plasticity:
            genome.learning_rule.plasticity = False
        genome._fixup()
        return genome

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
        surrogate = (_Surrogate(min_train=int(getattr(self.cfg, "surrogate_min_train", 8)),
                                beta=float(getattr(self.cfg, "ucb_beta", 0.0)))
                     if bool(getattr(self.cfg, "surrogate", False)) else None)

        # Lifelong warm-start: seed part of the initial population from the Hall of Fame (the best
        # brains she has ever designed), the rest random — so each search builds on accumulated
        # architectural wisdom instead of always starting from scratch.
        warm: List[ArchitectureGenome] = []
        if self.hall_of_fame is not None and len(self.hall_of_fame):
            n_warm = max(0, min(pop_size - 1,
                                int(pop_size * float(getattr(self.cfg, "warm_start_fraction", 0.25)))))
            warm = self.hall_of_fame.seed_genomes(n_warm, rng, block_size=block)
        population = warm + [
            ArchitectureGenome.random(rng, max_layers=max_layers, block_size=block,
                                      pos_encoding=getattr(self.cfg, "pos_encoding", None),
                                      ops=self._ops, search_learning_rule=self._search_lr,
                                      allow_plasticity=self._plasticity, library=self._library)
            for _ in range(pop_size - len(warm))]
        population = [self._conform(g) for g in population]   # keep warm + random inside the space
        seen: Dict[str, Candidate] = {}
        history: List[float] = []
        leaderboard: List[Candidate] = []
        best: Optional[Candidate] = None
        best_gen, mut = 0, base_mut

        for gen in range(gens):
            scored = self._score_population(population, folds, backend, seen, surrogate)
            scored.sort(key=lambda c: c.fitness, reverse=True)
            # Crown only from FULL evaluations. Under successive halving all but `keep` of
            # `scored` are cheap screens trained at a fraction of the budget and flagged
            # `predicted`; a screen score orders the rung, it does not measure a brain.
            # Crowning one would quote a fraction-of-budget perplexity as the champion's own,
            # and would put the champion outside `seen` — the very set the Pareto front is
            # built from — so `champion` and `pareto_front` would contradict each other in the
            # same report. `keep` is at least 1, so a rung always has a real score; `or scored`
            # is a safety net, not a path.
            real = [c for c in scored if not c.predicted] or scored
            improved = best is None or real[0].fitness > best.fitness
            if improved:
                best, best_gen = real[0], gen
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
        self._last_report = report
        # Lifelong memory: remember this champion (and its Pareto elites) so the NEXT search starts
        # from her best brains. Promotion stays gauntlet-gated — this only seeds future searches.
        if self.hall_of_fame is not None:
            self.hall_of_fame.record(report)
        return report

    # ---- champion ensemble: combine the top-k Pareto brains into one model ---- #
    def champion_ensemble(self, k: int = 3, *, corpus: Optional[Sequence[str]] = None,
                          steps: Optional[int] = None) -> "EnsembleModel":
        """Build a real inference-time ensemble of the top-k Pareto-frontier brains (each trained on
        the corpus), routed by competence. Strictly ≥ the best single brain — test-time intelligence
        from the diversity the search already discovered. Run :meth:`search` first."""
        if self._last_report is None:
            raise RuntimeError("run search() before building the champion ensemble")
        from nyxara.growth.foundry_models import build_model
        front = self._last_report.pareto_front or [self._champion]   # type: ignore[list-item]
        chosen = front[: max(1, int(k))]
        texts = list(corpus) if corpus is not None else self._collect_corpus()
        train_steps = int(self.cfg.micro_train_steps if steps is None else steps)
        members: List[BaseLanguageModel] = []
        for c in chosen:
            g = c.genome
            spec = ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=g.n_embd,
                             block_size=g.block_size, seed=g.seed, ngram_order=g.ngram_order,
                             ngram_k=g.ngram_k, substrate=str(getattr(self.cfg, "substrate", "auto")))
            model = build_model(spec)
            model.train_on(texts, steps=train_steps, seed=g.seed)
            members.append(model)
        return EnsembleModel(members)

    # ---- the champion as a promotable model spec ---- #
    def champion(self) -> Optional[Candidate]:
        return self._champion

    def champion_spec(self) -> ModelSpec:
        if self._champion is None:
            raise RuntimeError("run search() before requesting the champion spec")
        g = self._champion.genome
        return ModelSpec(kind="genesis", genome=g.to_dict(), n_embd=g.n_embd,
                         block_size=g.block_size, seed=g.seed, ngram_order=g.ngram_order,
                         ngram_k=g.ngram_k, substrate=str(getattr(self.cfg, "substrate", "auto")))

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
        # Only a GAUNTLET-CROWNED champion extends the palette: distil its synthesised mixer(s) into
        # named learned primitives future searches compose over. A benched candidate teaches nothing.
        learned = 0
        if promoted and self._library is not None:
            try:
                learned = self._library.add_from_genome(self._champion.genome)
            except Exception:  # noqa: BLE001 — library growth is best-effort, never blocks promotion
                learned = 0
        return {"promoted": promoted, "version": version.version, "reason": reason,
                "kind": spec.kind, "genome": spec.genome,
                "perplexity": version.metrics.get("perplexity"),
                "primitives_learned": learned,
                "primitive_library_size": len(self._library) if self._library is not None else 0,
                "describe": self._champion.genome.describe()}

    # ---- the autonomous triggers (mirror AutoForge) ---- #
    def maybe_boot(self) -> Optional[Dict[str, Any]]:
        """Run exactly ONE search+promote cycle on the first idle tick after boot.

        On a fresh machine the flywheel is empty, so :meth:`maybe_run`'s new-experience trigger can
        never fire (``0 - 0 < min_new_examples``) and she would never actually design a brain. This
        fires once — seeded from her own corpus — so the very first boot produces a real searched
        architecture, then reverts to the lazy new-experience cadence. Gated by ``cfg.run_on_boot``
        and fired at most once per process; oversight gates it upstream in the idle loop."""
        if getattr(self, "_boot_done", False):
            return None
        if not bool(getattr(self.cfg, "run_on_boot", True)):
            return None
        if self._reports:                       # a search already ran (e.g. via maybe_run) — done
            self._boot_done = True
            return None
        self._boot_done = True
        return self._run_cycle()

    def maybe_run(self) -> Optional[Dict[str, Any]]:
        """Run one search+promote cycle iff enough new verified experience has accrued.

        Idempotent until her own data grows, so the idle loop can call this every tick cheaply."""
        count = self._example_count()
        if count - self._last_example_count < int(getattr(self.cfg, "min_new_examples", 20)):
            return None
        self._last_example_count = count
        return self._run_cycle()

    def _run_cycle(self) -> Dict[str, Any]:
        """Search, then promote the champion through the Foundry gauntlet. Never crashes the idle
        loop — a failed search reports the reason instead of raising."""
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
    for strat in ("elitism", "tournament", "regularized", "nsga2"):
        cfg = GenesisConfig(backend="stdlib", population_size=8, generations=4, micro_train_steps=8,
                            block_size=16, max_layers=4, search_strategy=strat,
                            adaptive_mutation=True, novelty_weight=0.5, surrogate=True,
                            ucb_beta=0.5, successive_halving=True, hardware_weight=0.1,
                            hall_of_fame=False, seed=0)
        r = NeuralArchitectureSearch(cfg=cfg, seed_corpus=corpus).search()
        assert r.champion is not None and r.search_strategy == strat
        assert all(r.history[i] <= r.history[i + 1] + 1e-9 for i in range(len(r.history) - 1))
        assert r.to_dict()["evaluations"] >= 1
    print("search engines       : elitism / tournament / regularized / nsga2 + halving + UCB ✓")

    # lifelong memory: a champion is remembered and warm-starts the next search (disk-persisted)
    with tempfile.TemporaryDirectory() as d:
        from nyxara.kernel.config import NyxaraSettings, Profile
        st = NyxaraSettings.for_profile(Profile.TEST)
        st.paths.data_dir = Path(d)
        st.genesis = GenesisConfig(backend="stdlib", population_size=6, generations=2,
                                   micro_train_steps=6, block_size=16, hall_of_fame=True,
                                   warm_start_fraction=0.5, seed=0)
        nas_a = NeuralArchitectureSearch(settings=st, cfg=st.genesis, seed_corpus=corpus)
        nas_a.search()
        assert (Path(d) / "genesis" / "hall_of_fame.json").exists()
        remembered = len(nas_a.hall_of_fame)
        nas_b = NeuralArchitectureSearch(settings=st, cfg=st.genesis, seed_corpus=corpus)
        assert len(nas_b.hall_of_fame) == remembered      # memory crossed the process boundary
        # test-time ensemble of the top Pareto brains, routed by competence
        ens = nas_a.champion_ensemble(k=3, corpus=corpus)
        assert ens.param_count() > 0 and ens.perplexity(corpus[0]) >= 0.0
    print(f"lifelong + ensemble  : Hall-of-Fame warm-start + {len(ens.members)}-brain ensemble ✓")

    if _HAS_TORCH:
        print("\n[torch present] a searched neural brain trains from scratch ...")
        gm = GenesisModel(report.champion)
        before = gm.perplexity(corpus[0])
        gm.train_on(corpus, steps=60, seed=1)
        after = gm.perplexity(corpus[0])
        print(f"genesis perplexity   : {before:.1f} -> {after:.1f}  (params={gm.param_count()})")
        assert gm.param_count() > 0
        # exercise the whole upgraded operator palette in one net (incl. the frontier ops, RMSNorm,
        # QK-norm and multi-token prediction), plus rich sampling
        from nyxara.growth.genesis import _GenesisNet  # noqa: F401 — built below
        big = ArchitectureGenome(
            n_embd=64, block_size=16, pos_encoding="rope", tie_embeddings=True, n_predict=2,
            layers=[LayerGene(op=op, n_head=4, norm_type="rmsnorm", qk_norm=True) for op in _OPS])
        bm = GenesisModel(big)
        bm.train_on(corpus, steps=20, seed=2)
        sample = bm.generate("the master", max_tokens=24, temperature=0.8, top_k=20, top_p=0.95)
        print(f"full-palette net     : params={bm.param_count()}  sample={sample!r}")
        assert bm.param_count() > 0
        print("frontier ops + MTP   : mamba / gla / diff / mla / rmsnorm / qk-norm / multi-token ✓")
        for pe in ("learned", "rope", "alibi"):
            GenesisModel(ArchitectureGenome(n_embd=32, block_size=16, pos_encoding=pe,
                         layers=[LayerGene(op="gqa_attention", n_head=4, n_kv_head=2)])
                         ).train_on(corpus, steps=5, seed=3)
        print("positional schemes   : learned / rope / alibi + grouped-query attention ✓")
        # KV-cache decode is bit-identical to the full recompute (for lengths ≤ block_size)
        cnet = GenesisModel(ArchitectureGenome(
            n_embd=32, block_size=64, pos_encoding="rope",
            layers=[LayerGene(op="attention", n_head=4), LayerGene(op="selective_ssm"),
                    LayerGene(op="swiglu")]))
        cnet.train_on(corpus, steps=5, seed=4)
        assert cnet.net.cacheable
        full = cnet.generate("the master", max_tokens=20, greedy=True, use_cache=False)
        cached = cnet.generate("the master", max_tokens=20, greedy=True, use_cache=True)
        assert full == cached
        print("KV-cache decode      : O(1)-per-token incremental decode == full recompute ✓")
    else:
        print("\n[torch absent] neural path skipped; stdlib substrate crowned a champion ✓")

    print("\nALL SELF-TESTS PASSED ✓")
