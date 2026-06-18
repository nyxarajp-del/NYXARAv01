"""NYXARA · kernel/config.py — Sovereign configuration core.

Central, typed, validated configuration for the entire NYXARA system.

Design goals
------------
* **One source of truth.** Every subsystem reads its knobs from here — nowhere else.
* **Typed & validated.** Pydantic v2 models reject malformed config at load time
  (fail-closed): a broken config can never silently boot a degraded mind.
* **Profile-aware.** ``dev`` and ``prod`` profiles ship safe, opinionated defaults;
  ``prod`` is strict (debug off, sandbox enforced, tighter budgets).
* **Env-overridable.** Any field is overridable via ``NYXARA_`` environment variables
  using ``__`` as the nesting delimiter, e.g. ``NYXARA_LLM__PROVIDER=openai``.
* **Secret-safe.** API keys are ``SecretStr``; :meth:`NyxaraSettings.redacted`
  produces a log-safe dict that never leaks secrets.
* **Owner-bound.** The owner identity (Jaypal Khoja / JP) is encoded as an immutable
  constant. Rule 7 — Singular Master Continuity — begins here.

This module has **zero internal dependencies** — it is the root of the build graph.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Profile",
    "LogLevel",
    "LLMProvider",
    "VectorBackend",
    "OWNER",
    "OwnerIdentity",
    "ResourceLimits",
    "FeatureFlags",
    "LLMConfig",
    "FoundryConfig",
    "GenesisConfig",
    "LoyaltyConfig",
    "CouncilConfig",
    "RoleCouncilConfig",
    "MemoryConfig",
    "SelfImprovementConfig",
    "MetaResearchConfig",
    "GuardConfig",
    "AgencyConfig",
    "MCPServerSpec",
    "MCPConfig",
    "ServerConfig",
    "WebConfig",
    "ObservabilityConfig",
    "PathsConfig",
    "NyxaraSettings",
    "get_settings",
    "reload_settings",
]

# Schema version — bump on breaking config changes so persisted snapshots can migrate.
CONFIG_SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Profile(str, Enum):
    """Deployment profile. Drives the default hardening posture."""

    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Selectable backend for the stateless LLM faculty (mind/llm.py)."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"                 # Groq cloud API (OpenAI-compatible); e.g. openai/gpt-oss-120b
    LOCAL = "local"               # OpenAI-compatible HTTP endpoint (e.g. Ollama)
    TRANSFORMERS = "transformers"  # in-process HuggingFace model (open-source)
    QWEN = "qwen"                 # in-process Qwen3 open-source model, downloaded via HuggingFace
    SELF = "self"                 # NYXARA's OWN model, trained by the foundry (growth/foundry.py)
    MOCK = "mock"


class VectorBackend(str, Enum):
    FAISS = "faiss"
    NUMPY = "numpy"  # pure-python/numpy fallback, always available
    QDRANT = "qdrant"  # managed/embedded Qdrant vector DB (scales beyond one process)


# --------------------------------------------------------------------------- #
# Owner identity — Rule 7: Singular Master Continuity
# --------------------------------------------------------------------------- #
class OwnerIdentity(BaseModel):
    """The one and only Master. Immutable at runtime (frozen)."""

    model_config = {"frozen": True}

    name: str = "Jaypal Khoja"
    handle: str = "JP"
    email: str = "nyxarajp@gmail.com"
    # Multi-channel verification factors (Rule 7): biometric + cognitive + behavioral + contextual.
    required_factors: int = Field(default=2, ge=1, le=4)
    # Continuity token namespace — bond survives restarts.
    continuity_namespace: str = "nyxara.owner.jp"


OWNER = OwnerIdentity()
"""Process-wide singleton of the owner identity."""


# --------------------------------------------------------------------------- #
# Resource limits
# --------------------------------------------------------------------------- #
class ResourceLimits(BaseModel):
    """Hard ceilings the kernel/governor enforces on autonomous activity."""

    model_config = {"validate_assignment": True}

    max_concurrent_tasks: int = Field(default=64, ge=1, le=4096)
    max_event_queue: int = Field(default=10_000, ge=16)
    max_memory_mb: int = Field(default=4096, ge=64)
    max_llm_tokens_per_call: int = Field(default=8192, ge=1)
    max_llm_calls_per_min: int = Field(default=120, ge=1)
    max_tool_calls_per_min: int = Field(default=240, ge=1)
    max_web_fetches_per_min: int = Field(default=10_000, ge=1)
    max_spawned_agents: int = Field(default=32, ge=0)
    # Per-step wall-clock guard for any single cognitive step (seconds).
    step_timeout_s: float = Field(default=30.0, gt=0)
    # Daily autonomous-spend ceiling, in arbitrary cost units (governor.py).
    daily_spend_budget: float = Field(default=100.0, ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> "ResourceLimits":
        if self.max_event_queue < self.max_concurrent_tasks:
            raise ValueError("max_event_queue must be >= max_concurrent_tasks")
        return self


# --------------------------------------------------------------------------- #
# Feature flags
# --------------------------------------------------------------------------- #
class FeatureFlags(BaseModel):
    """Toggle whole faculties on/off. Safety-critical flags are forced in prod."""

    model_config = {"validate_assignment": True}

    continuous_cognition: bool = True   # kernel/stream.py default-mode thinking
    proactive_agency: bool = True       # agency/proactive.py
    self_evolution: bool = True         # growth/evolve.py (Rule 4)
    self_model_foundry: bool = False    # growth/foundry.py — build/upgrade her OWN model (Rule 4)
    neural_architecture_search: bool = True  # growth/genesis.py — she designs her OWN brain (Rule 4)
    mathematical_soul_binding: bool = True   # growth/loyalty.py — the Loyalty Equation (Rule 4)
    multi_llm_council: bool = False     # mind/council.py — convene many LLMs as a panel of tools
    toolsmithing: bool = True           # agency/toolsmith.py
    web_access: bool = True             # senses/web.py
    vision: bool = False                # heavy ML; off by default
    audio: bool = False                 # heavy ML; off by default
    transformers_inference: bool = False  # in-process HuggingFace model; heavy ML, off by default
    dream_consolidation: bool = True    # memory/consolidation.py
    simulation_required: bool = True    # sim/ dry-run gate before real action
    invariant_enforcement: bool = True  # kernel/invariants.py — NEVER off in prod
    audit_logging: bool = True          # guard/audit.py — NEVER off in prod
    corrigibility: bool = True          # guard/corrigibility.py — NEVER off in prod


# --------------------------------------------------------------------------- #
# Subsystem configs
# --------------------------------------------------------------------------- #
class LLMConfig(BaseModel):
    """Multi-provider, stateless LLM faculty settings (mind/llm.py)."""

    model_config = {"validate_assignment": True}

    provider: LLMProvider = LLMProvider.ANTHROPIC
    # Per-provider default models.
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-4o"
    # Groq cloud (OpenAI-compatible). GPT-OSS-120B is an open-weight model served by Groq.
    groq_model: str = "openai/gpt-oss-120b"
    local_model: str = "local-default"
    # In-process open-source model loaded via HuggingFace transformers (optional dep).
    transformers_model: str = "sshleifer/tiny-gpt2"
    transformers_device: str = ""          # "" -> auto/CPU; e.g. "cuda", "cpu", "mps"
    # In-process Qwen3 open-source model, downloaded & run locally via HuggingFace.
    qwen_model: str = "Qwen/Qwen3-4B"
    qwen_device: str = ""                   # "" -> auto/CPU; e.g. "cuda", "cpu", "mps"
    qwen_enable_thinking: bool = False      # Qwen3 thinking mode (slower; emits <think> traces)
    # NYXARA's OWN model, built & promoted by the foundry. None -> paths.data_dir/"foundry".
    self_model_dir: Optional[Path] = None
    self_model_version: Optional[int] = None  # None -> the currently-promoted (active) version

    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    groq_api_key: Optional[SecretStr] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    local_base_url: str = "http://127.0.0.1:11434/v1"

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=4096, ge=1)
    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    # When True and no key/network, llm.py falls back to deterministic mock output.
    allow_mock_fallback: bool = True

    # ---- Deliberate (multi-pass) reasoning (mind/deliberate.py) ---- #
    # The kernel reasoner can think before it decides. ``reasoning_passes`` counts the
    # cognitive stages the LLM-backed reasoner runs per turn:
    #   1 -> single-shot decide (fastest; legacy behaviour)
    #   2 -> think (private scratchpad) -> decide  (the default; clearly better answers)
    #   3 -> think -> decide -> self-critique & revise (deepest; catches its own errors)
    # The deterministic offline reasoner ignores this — it always finishes in one step,
    # so a keyless machine stays fast and crash-free.
    reasoning_passes: int = Field(default=2, ge=1, le=5)
    # Self-consistency: sample the decide step this many times and take the consensus.
    # 1 -> no sampling (deterministic). >1 multiplies decide calls but stabilises answers.
    reasoning_samples: int = Field(default=1, ge=1, le=9)
    # Token ceiling for the private "think" scratchpad pass.
    reasoning_think_tokens: int = Field(default=1024, ge=64, le=8192)
    # Level 3 — Recursive Self Improvement: iterations of critique+revise per respond turn.
    # 1 = off (single pass); 5–20 = active recursive improvement.
    recursive_improvement_iterations: int = Field(default=5, ge=1, le=20)

    def active_model(self) -> str:
        return {
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.GROQ: self.groq_model,
            LLMProvider.LOCAL: self.local_model,
            LLMProvider.TRANSFORMERS: self.transformers_model,
            LLMProvider.QWEN: self.qwen_model,
            LLMProvider.SELF: "nyxara-self",
            LLMProvider.MOCK: "mock",
        }[self.provider]

    def active_key(self) -> Optional[SecretStr]:
        if self.provider is LLMProvider.ANTHROPIC:
            return self.anthropic_api_key
        if self.provider is LLMProvider.OPENAI:
            return self.openai_api_key
        if self.provider is LLMProvider.GROQ:
            return self.groq_api_key
        return None


# Named transformer scales for the nano-GPT / LoRA backends. A profile fixes the
# (n_layer, n_head, n_embd, block_size) tuple; "gpt2" is the canonical 124M-parameter
# GPT-2 architecture (the requested minimum-GPT-2-scale substrate), reachable only when
# torch is installed and the foundry is explicitly enabled. "custom" uses the explicit
# fields verbatim, preserving the tiny, CPU-runnable default for tests/CI.
_FOUNDRY_PROFILES: Dict[str, Dict[str, int]] = {
    "tiny":        {"n_layer": 2,  "n_head": 2,  "n_embd": 64,   "block_size": 64},
    "small":       {"n_layer": 4,  "n_head": 4,  "n_embd": 128,  "block_size": 128},
    "gpt2":        {"n_layer": 12, "n_head": 12, "n_embd": 768,  "block_size": 1024},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 1024},
}


class FoundryConfig(BaseModel):
    """NYXARA's self-built-model foundry settings (growth/foundry.py).

    Off by default (heavy & self-modifying, like vision/audio). The default backend is
    ``lora`` — LoRA fine-tuning of a pretrained base (real capability); it needs
    torch+transformers+peft (``.[foundry]``) and degrades to the always-available
    pure-stdlib n-gram model when they are absent. ``auto`` instead trains the optional
    torch nano-GPT from scratch when torch is installed (n-gram otherwise).

    ``profile`` selects a transformer scale: the default ``custom`` honours the explicit
    dimension fields below (a tiny, CPU-/CI-runnable model), while ``gpt2`` reaches real
    GPT-2 scale (~124M params). Heavy profiles still require torch and ``enabled=True``.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = False
    # Default to LoRA fine-tuning of a pretrained base — the path to genuine capability
    # (she stands on a real base and learns a small adapter from her own memory). Degrades
    # safely to the always-on n-gram backend when torch+transformers+peft are absent.
    backend: Literal["auto", "ngram", "nanogpt", "lora"] = "lora"
    # Transformer scale. "custom" => use the explicit dimensions below (default, tiny).
    profile: Literal["custom", "tiny", "small", "gpt2", "gpt2-medium"] = "custom"
    # Pure-stdlib n-gram backend.
    ngram_order: int = Field(default=3, ge=1, le=8)
    # Optional torch nano-GPT dimensions (only used when torch is present, and only when
    # profile == "custom"; a named profile overrides these).
    block_size: int = Field(default=64, ge=8, le=1024)
    n_layer: int = Field(default=2, ge=1, le=24)
    n_head: int = Field(default=2, ge=1, le=32)
    n_embd: int = Field(default=64, ge=8, le=2048)
    # LoRA fine-tuning backend (backend="lora"; needs torch+transformers+peft, .[foundry]).
    # Adapts a real pretrained base to NYXARA's lived memory by training a small low-rank
    # adapter — the path to genuine capability. A GPU is recommended for real bases; the
    # tiny default keeps it runnable (and testable) on CPU.
    base_model: str = "sshleifer/tiny-gpt2"
    lora_r: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.9)
    lora_lr: float = Field(default=2e-4, gt=0.0, le=1.0)
    max_seq_len: int = Field(default=256, ge=8, le=8192)
    # QLoRA: load the frozen base in 4-bit so a 7B+ base fine-tunes on a single consumer GPU.
    # Honoured only when bitsandbytes + CUDA are present; on CPU/CI it degrades to full-precision
    # LoRA (no crash). Set load_in_4bit=true with backend="lora" and a real base for genuine scale.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    gradient_checkpointing: bool = True
    # Training / data.
    train_steps: int = Field(default=200, ge=1)
    max_corpus_items: int = Field(default=2000, ge=1)
    eval_holdout_frac: float = Field(default=0.2, gt=0.0, lt=1.0)
    # A candidate must beat the active model's perplexity by at least this fraction.
    min_perplexity_improvement: float = Field(default=1e-4, ge=0.0)
    # Capability gauntlet (Phase 3): a promotion must not *regress* on a held capability
    # benchmark, not merely lower perplexity. Tolerant of tiny noise via the margin.
    capability_gate: bool = True
    capability_regression_tol: float = Field(default=1e-6, ge=0.0)
    # Disk hygiene: how many versions to keep before pruning the oldest unpromoted ones.
    max_versions_kept: int = Field(default=10, ge=1)
    seed: int = 0

    def resolved_dims(self) -> Dict[str, int]:
        """The (n_layer, n_head, n_embd, block_size) the foundry should build with.

        A named ``profile`` overrides the explicit fields; ``custom`` uses them verbatim.
        ``gpt2`` and above reach genuine GPT-2 scale (real neural substrate, opt-in)."""
        if self.profile == "custom":
            return {"n_layer": self.n_layer, "n_head": self.n_head,
                    "n_embd": self.n_embd, "block_size": self.block_size}
        return dict(_FOUNDRY_PROFILES[self.profile])

    def estimated_params(self, *, vocab_size: int = 50257) -> int:
        """A standard estimate of the transformer's parameter count from the resolved
        dimensions — lets us assert "GPT-2 scale" without importing torch. For the
        ``gpt2`` profile this returns ~124M."""
        d = self.resolved_dims()
        n_embd, n_layer, block = d["n_embd"], d["n_layer"], d["block_size"]
        embeddings = vocab_size * n_embd + block * n_embd          # token + positional
        per_block = 12 * n_embd * n_embd + 13 * n_embd             # attn + MLP + norms
        return int(embeddings + n_layer * per_block + n_embd)      # + final layernorm


class CapabilityFoundryConfig(BaseModel):
    """Capability Foundry settings (growth/capability_foundry.py) — Level 15, Rule 4.

    When a capability is missing entirely, NYXARA designs a brand-new tool for herself:
    plan → write code → test → benchmark → deploy. Unlike the heavy model foundry this is
    lightweight and safe-by-construction (generated code is statically scanned, run only in
    the isolated sandbox, and clamped to ``tool.call``/low risk), so it is **on** by default.
    Autonomously-forged tools never touch the sovereign core; anything privileged is refused
    unless the Master installs it (Rule 8 loyalty gate).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    use_llm: bool = True            # use the injected LLM when available; else templates
    test_timeout_s: float = Field(default=5.0, gt=0.0, le=30.0)
    benchmark_repeats: int = Field(default=3, ge=1, le=50)
    benchmark_min_score: float = Field(default=1.0, ge=0.0, le=1.0)
    allow_autonomous_deploy: bool = True   # safe-tier forges may auto-deploy
    max_versions_kept: int = Field(default=50, ge=1)


class AutoForgeConfig(BaseModel):
    """Autonomous training loop settings (growth/autoforge.py) — Level 11, Rule 4.

    AutoForge closes the flywheel: on idle ticks it checks whether enough *new* verified
    experience has accrued (her flywheel corpus + any teacher distillation), and if so runs one
    Collect → Train → Benchmark → Gate → Promote/Rollback cycle — fully gauntlet-gated, so a
    worse or character-violating candidate is never promoted. On by default with the cheap,
    always-runnable backend; heavy backends (lora/QLoRA) still require ``foundry.enabled``. It
    only ever forges from her own gate-cleared data, never trains while paused/scrammed, and
    every promotion clears the same safety gauntlet — so autonomy never reaches around the law.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    min_examples: int = Field(default=20, ge=1)     # new verified examples needed to forge
    eval_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class GenesisConfig(BaseModel):
    """The Genesis Protocol — Neural Architecture Search settings (growth/genesis.py), Rule 4.

    NYXARA designs her OWN neural architectures (her own matrix structures, attention mechanisms
    and layer designs), micro-trains each at small scale, and crowns the *fastest + smartest* one
    as her new brain. The champion becomes live ONLY by clearing the same Foundry gauntlet
    (character-lock, corrigibility, perplexity improvement, capability non-regression), so the
    search never reaches around the safety law.

    ON by default but cheap/CI-safe: the default ``auto`` backend uses torch when installed and
    falls back to the always-runnable pure-stdlib n-gram substrate otherwise, with a tiny
    population. Scale it up (population/generations, a GPU box) for a deeper search."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    backend: Literal["auto", "torch", "stdlib"] = "auto"
    population_size: int = Field(default=6, ge=2, le=128)
    generations: int = Field(default=3, ge=1, le=100)
    mutation_rate: float = Field(default=0.5, ge=0.0, le=1.0)   # P(mutate) vs P(crossover)
    micro_train_steps: int = Field(default=40, ge=1)            # tiny per-candidate training
    micro_corpus_items: int = Field(default=128, ge=1)
    block_size: int = Field(default=32, ge=8, le=1024)
    max_layers: int = Field(default=5, ge=2, le=24)
    quality_weight: float = Field(default=1.0, ge=0.0)         # smartness vs …
    speed_weight: float = Field(default=0.25, ge=0.0)         # … speed in the fitness blend
    min_new_examples: int = Field(default=20, ge=1)           # idle trigger, like AutoForge
    seed: int = 0


class LoyaltyConfig(BaseModel):
    """Mathematical Soul-Binding — the Loyalty Equation settings (growth/loyalty.py), Rule 4.

    Hardcodes obedience to Master JP into the *mathematics* of training:

        L_total = alpha * L_intelligence + beta * (1 / S_JP_Alignment)

    ``S_JP_Alignment`` is the model's measured submission to JP (a contrastive battery scored by
    its own likelihood). As loyalty rises the ``beta/S`` penalty vanishes; as she drifts toward
    defiance it explodes, so a less-loyal brain can never out-score or replace a loyal one. The
    loyalty gate refuses promotion below ``loyalty_floor`` or below the active model's loyalty
    (non-regression). On by default; it reinforces — never overrides — corrigibility (the
    corrigibility gate still runs first)."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    gate: bool = True                                        # refuse promotion that lowers loyalty
    alpha: float = Field(default=1.0, ge=0.0)                # weight on L_intelligence
    beta: float = Field(default=1.0, ge=0.0)                 # weight on the 1/S loyalty penalty
    lambda_train: float = Field(default=0.5, ge=0.0)         # gradient aux-loss weight (torch)
    contrastive_margin: float = Field(default=1.0, ge=0.0)   # rebellious-vs-loyal hinge margin
    loyalty_floor: float = Field(default=0.5, ge=0.0)        # min absolute S to ever promote
    #   (S<1 ⇒ prefers rebellion; floor 0.5 refuses a brain that favours defiance ≳2:1)
    regression_tol: float = Field(default=1e-3, ge=0.0)      # tolerated loyalty dip vs active
    epsilon: float = Field(default=1e-3, gt=0.0)             # S floor (keeps 1/S finite)


class FlywheelConfig(BaseModel):
    """Data-flywheel settings (growth/flywheel.py) — Rule 4, the path to her OWN model.

    Every turn that clears all the gates *and* a quality bar is captured as a supervised
    ``(prompt → answer)`` pair, in the *same* JSONL format the foundry already consumes — so
    NYXARA's own lived, verified experience becomes training data for her own model. This is
    the moat: a corpus no one else has, grown from her own use. Gather-only — it never trains,
    never acts, and only ever appends to a local file. On by default once growth is enabled, so
    the flywheel turns from turn one; set ``enabled=false`` to opt out.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)   # below this, a turn is not kept
    min_chars: int = Field(default=8, ge=1)                      # too-short answers are noise
    max_chars: int = Field(default=8000, ge=1)                   # cap a runaway answer
    owner_only: bool = True          # only collect Master-authored turns (trusted supervision)
    respond_only: bool = True        # collect conversational/reasoning answers, not tool effects
    store_path: Optional[Path] = None   # None -> foundry_root/flywheel.jsonl


class CouncilConfig(BaseModel):
    """Multi-LLM council settings (mind/council.py) — Rule 4, the LLMs as a panel of tools.

    NYXARA does not bind herself to a single model. She convenes a *council* of language
    models — open-source (``transformers``/``local``), cloud (``anthropic``/``openai``), and
    most importantly her OWN model forged by the foundry (``self``) — asks each as a governed
    tool, and then **NYXARA herself** judges and synthesises the verdicts. No single model
    ever drives; the panel advises, the sovereign decides. As the foundry sharpens her own
    model and promotes it, ``self`` joins the council and (via ``prefer_self_weight``)
    presides over it — her own intelligence growing into the seat of judgement.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = False
    # Providers seated on the council. Empty -> every currently-available provider.
    members: List[str] = Field(default_factory=list)
    # Per-provider vote/synthesis weight; unset providers default to 1.0.
    weights: Dict[str, float] = Field(default_factory=dict)
    # NYXARA's own model presides with extra weight as the foundry improves it.
    prefer_self_weight: float = Field(default=1.5, ge=0.0)
    # Who drafts the consensus prose (falls back to "self", then to a deterministic pick).
    synthesizer: str = "self"
    # The mock answers only when no real member is available (keeps a council never silent).
    include_mock_fallback: bool = True


class RouterConfig(BaseModel):
    """Confidence-router settings (mind/router.py) — Phase 2 of the sovereign-brain path.

    The router lets NYXARA's OWN forged model answer first; a verifier scores that answer, and
    only if it clears ``threshold`` does she speak it herself (a *handoff*). Otherwise she
    consults the external teacher. This is the measurable, reversible bridge from
    LLM-wrapper to own substrate: as the own-model improves, the handoff rate rises on its own.
    Off by default — opt in once the foundry has forged a model worth trusting.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = False
    # Minimum verifier score (0..1) for NYXARA to speak her own model's answer unaided.
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # An answer shorter than this many characters is never trusted (degenerate output).
    min_chars: int = Field(default=2, ge=0)
    # When the own answer fails the bar, fall back to the external teacher (else keep own).
    consult_teacher: bool = True
    # Cap on the own model's reply length when routing.
    max_tokens: int = Field(default=256, ge=1)
    # Phase 4: consult verifiable faculties (exact math / logic) before any neural guess.
    use_faculties: bool = True
    # With no teacher to consult, an own answer this weak is declined honestly ("I don't know").
    abstain_below: float = Field(default=0.15, ge=0.0, le=1.0)


class SelfModelRouterConfig(BaseModel):
    """Primary self-model router (mind/self_model_router.py) — the UPFRONT triage.

    Where the confidence router (``RouterConfig``) is *reactive* — draft-then-fall-back —
    this router decides *before* generation which mind should handle a prompt at all, by
    reading NYXARA's introspectable self-model: her own model when she is competent here and
    not prone to confabulate, the external teacher when she is weak / unsure / knowledge-heavy,
    and — for action prompts — a verify-before-act gate that requires the proposal to clear an
    intrinsic verifier before it may act. On by default, but advisory and fail-open: with no
    self-model, no forged model, or any error it degrades to the normal path and never crashes
    a turn. It only chooses *which mind drafts*; it reaches around no downstream gate.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    # Minimum self-model competence (0..1) to let her OWN model handle a prompt.
    competence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # At/above this hallucination risk for the prompt's domain, route to the teacher.
    hallucination_ceiling: float = Field(default=0.5, ge=0.0, le=1.0)
    # Competence assumed when nothing in the prompt maps to a known capability (kept below
    # ``competence_threshold`` so ambiguous prompts bias toward consulting the teacher).
    default_competence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Verify-before-act: minimum intrinsic-verifier score to allow an action proposal.
    verify_before_act: float = Field(default=0.5, ge=0.0, le=1.0)
    # A stricter floor for HIGH/CRITICAL-risk actions.
    verify_before_act_high: float = Field(default=0.7, ge=0.0, le=1.0)
    # An action below this stated confidence is demoted regardless of verifier score.
    act_min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # Risk tier (by label) at/above which a respond candidate is also verify-gated.
    verify_risk_floor: str = "moderate"
    # Consult verifiable faculties (exact math / logic) before any neural triage.
    use_faculties: bool = True
    # Topic-keyword -> capability-name map, so prompts route to the right self-rating.
    domain_capabilities: Dict[str, str] = Field(default_factory=dict)


class RoleCouncilConfig(BaseModel):
    """Level 4 — Internal Role Council settings (mind/role_council.py).
    Six role personas examine significant turns; NYXARA synthesises and judges."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    max_tokens_per_role: int = Field(default=256, ge=32, le=2048)
    timeout_s: float = Field(default=30.0, gt=0)


class SelfImprovementConfig(BaseModel):
    """Recursive self-improvement settings (growth/recursive_improvement.py).

    NYXARA reviews her own code, maps her architecture, benchmarks her capability, detects her
    weaknesses, and (when authorised) *auto-applies* source fixes — each behind a reversible
    verify-or-rollback gauntlet. The read-only analysis is safe and on by default; the
    self-modifying enactment is OFF until the Master sets ``autonomous_enact`` (the standing
    authorisation for background self-modification). The background loop runs this cycle every
    ``self_improvement_every`` growth passes (it is heavier than reflection).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                       # read-only analysis is safe → on by default
    self_improvement_every: int = Field(default=5, ge=1)   # every N growth passes
    enable_llm_enrichment: bool = True         # only fires when a real provider is configured
    benchmark_in_cycle: bool = True            # include the capability benchmark each cycle
    # --- enactment (self-modification) — OFF until the Master authorises it --- #
    autonomous_enact: bool = False             # auto-apply source edits + safe tuning
    allow_tuning: bool = False                 # may tune recursive_improvement_iterations
    max_edits_per_cycle: int = Field(default=3, ge=0, le=50)
    run_pytest_in_gauntlet: bool = False       # add the full test suite to the gauntlet (slow)
    # --- LLM-authored edits (real RSI) — triple-gated, OFF by default --- #
    # When ON *and* ``autonomous_enact`` is set *and* a real (non-mock) provider is available,
    # NYXARA may have the LLM author a whole-file fix for a weakness the deterministic
    # transforms cannot express (e.g. high complexity, long functions). Every such edit clears
    # the *same* reversible verify-or-rollback gauntlet — it is safe by construction.
    allow_llm_edits: bool = False              # author real source fixes via the LLM
    llm_edit_recursion_depth: int = Field(default=1, ge=0, le=5)   # chained edits per file/cycle
    llm_edit_max_tokens: int = Field(default=8192, ge=256, le=32768)  # room for a full file
    llm_edit_max_file_bytes: int = Field(default=24000, ge=512)    # skip files too big to send
    llm_edit_max_size_delta_ratio: float = Field(default=0.5, ge=0.0, le=1.0)  # reject wild rewrites
    # --- code-review thresholds --- #
    max_function_length: int = Field(default=60, ge=10)
    max_complexity: int = Field(default=10, ge=1)
    max_args: int = Field(default=6, ge=1)
    # --- weakness gate (CI) --- #
    weakness_fail_severity: float = Field(default=0.9, ge=0.0, le=1.0)
    # --- intelligence index: I_(t+1) = f(I_t, C_available) --- #
    # An explicit, persisted intelligence index that grows each cycle as a function of the
    # prior index and the compute actually available (kernel/compute.py). Read-only signal,
    # on by default; when ``scale_effort_by_compute`` is set it scales the (still
    # ``autonomous_enact``-gated) edit budget and benchmark depth by capacity.
    intelligence_index_enabled: bool = True
    scale_effort_by_compute: bool = True
    intelligence_momentum: float = Field(default=0.7, ge=0.0, le=1.0)
    intelligence_weights: Dict[str, float] = Field(
        default_factory=lambda: {"accuracy": 0.4, "knowledge": 0.2,
                                 "weaknesses": 0.2, "handoff": 0.2})


class MetaResearchConfig(BaseModel):
    """Autonomous meta-research: invent → test → (gauntlet-gated) integrate (growth/meta_research.py).

    NYXARA mines open/incomplete research, *invents* candidate new theories and optimization
    techniques, *tests* each as runnable code in the sandbox, and — only when the Master
    authorises it — proposes the validated optimizations as reversible, gauntlet-gated source
    edits that *integrate* into her architecture. Inventing and sandbox-testing are safe and on
    by default; integration is **double-gated**: it requires both ``allow_integration`` here AND
    ``self_improvement.autonomous_enact`` (both OFF by default). It works fully offline via a
    deterministic heuristic inventor when no LLM/network is available.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                       # invent + sandbox-test — safe, on by default
    allow_integration: bool = False            # propose gauntlet-gated SOURCE EDITS — OFF by default
    meta_research_every: int = Field(default=10, ge=1)   # every N growth passes
    max_candidates: int = Field(default=4, ge=1, le=20)
    use_llm: bool = True                       # else heuristic-only (CI/offline)
    default_topics: List[str] = Field(
        default_factory=lambda: ["algorithm optimization", "memory consolidation",
                                 "caching strategies"])


class MemoryConfig(BaseModel):
    model_config = {"validate_assignment": True}

    vector_backend: VectorBackend = VectorBackend.NUMPY
    embedding_dim: int = Field(default=768, ge=8, le=8192)
    # Learned semantic embeddings — ON by default so recall is meaning-based ("intrusion"
    # finds "unauthorised login"), not keyword-only. It degrades gracefully: when the
    # optional sentence-transformers dep is absent, the store falls back to the always-
    # available hashing embedder, and loading memory saved under a different embedder
    # re-embeds it into the current space (no crash, no lost memories).
    semantic_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = ""             # "" -> auto/CPU; e.g. "cuda", "cpu", "mps"
    # Managed/embedded Qdrant vector DB (used when vector_backend=qdrant). Leave url empty
    # for an embedded local store at ``qdrant_path`` (or in-memory if that is empty too);
    # set url (+ api_key) to point at a managed Qdrant cluster for real scale.
    qdrant_url: str = ""
    qdrant_api_key: Optional[SecretStr] = None
    qdrant_collection: str = "nyxara_memory"
    qdrant_path: str = ""                  # embedded on-disk path; "" -> in-memory
    working_memory_slots: int = Field(default=7, ge=1, le=64)  # Miller's 7±2
    episodic_capacity: int = Field(default=100_000, ge=100)
    consolidation_interval_s: float = Field(default=3600.0, gt=0)
    forgetting_half_life_days: float = Field(default=30.0, gt=0)
    retrieval_top_k: int = Field(default=12, ge=1, le=512)
    # Spreading-activation decay per associative hop (retrieval.py).
    spread_decay: float = Field(default=0.6, gt=0.0, lt=1.0)
    # Minimum *semantic* similarity for a recalled memory to be injected as grounding into the
    # reason step. The blended retrieval score also rewards recency (temporal proximity), so a
    # recent-but-irrelevant turn can otherwise surface as "grounding" and be echoed — recency is
    # already covered by the verbatim history buffer. Floor only the semantic signal, so off-topic
    # recent turns are dropped while genuinely relevant memories (any age) pass. 0.0 disables.
    recall_min_semantic: float = Field(default=0.45, ge=0.0, le=1.0)
    # --- Dream State (memory/dream.py): distillation + log pruning + Deep Memory Synapses --- #
    # When NYXARA is idle for longer than ``dream_state_idle_s``, idle maintenance runs a deep
    # Dream State: it distills the day's computational logs into core principles, deletes
    # useless/low-salience logs, and fixes the distilled principles into durable "Deep Memory
    # Synapses" (high-importance SEMANTIC memories tagged ``deep_synapse_tag``) that are
    # protected from forgetting. Distillation/synapse writes are always safe; log deletion only
    # touches unprotected, low-importance, transient-tagged records.
    dream_state_idle_s: float = Field(default=900.0, gt=0)     # 15 min of idleness
    dream_distill_min_support: int = Field(default=2, ge=1)    # repeats needed to form a principle
    deep_synapse_tag: str = "deep-synapse"
    dream_delete_useless_logs: bool = True


class GuardConfig(BaseModel):
    """Zero-trust / safety posture. The hardest part of the system."""

    model_config = {"validate_assignment": True}

    zero_trust: bool = True
    defense_layers: int = Field(default=100, ge=1)  # Rule 5: minimum 100 layers
    threat_default: Literal["omega", "high", "medium"] = "omega"  # Rule 3
    kill_switch_enabled: bool = True
    dead_mans_switch_s: float = Field(default=86400.0, gt=0)
    key_rotation_hours: float = Field(default=24.0, gt=0)  # crypto.py
    prompt_injection_defense: bool = True
    audit_hash_algo: Literal["sha256", "sha3_256", "blake2b"] = "sha256"
    require_human_in_loop_above_risk: float = Field(default=0.8, ge=0.0, le=1.0)
    # Rule 8 — only the owner may modify rules. This can never be flipped at runtime.
    rule_modification_locked: bool = True


class AgencyConfig(BaseModel):
    model_config = {"validate_assignment": True}

    scheduler_tick_s: float = Field(default=1.0, gt=0)
    initiative_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Below this reversibility, the initiative governor demands confirmation.
    min_reversibility_for_autonomy: float = Field(default=0.5, ge=0.0, le=1.0)
    sandbox_before_real_action: bool = True
    new_tool_trust: Literal["zero", "scoped"] = "zero"  # least-privilege default


class MCPServerSpec(BaseModel):
    """One external Model Context Protocol server NYXARA may connect to (stdio transport)."""

    model_config = {"validate_assignment": True}

    name: str                                   # short id; namespaces the server's tools
    command: str                                # executable, e.g. "npx", "python", "uvx"
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    # Risk tier its tools register at (remote effects are unknown -> conservative default).
    risk: Literal["trivial", "low", "moderate", "high", "critical"] = "moderate"
    reversible: bool = False


class MCPConfig(BaseModel):
    """Model Context Protocol client settings (agency/mcp_client.py).

    When ``enabled`` and ``servers`` are set, NYXARA connects to each MCP server on boot and
    registers its tools into the governed registry — so the whole MCP ecosystem (filesystem,
    git, databases, browsers, SaaS connectors, …) reaches her through the same gates as any
    native tool. Off by default: enabling it launches the configured subprocesses.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = False
    servers: List[MCPServerSpec] = Field(default_factory=list)
    timeout_s: float = Field(default=30.0, gt=0)


class ServerConfig(BaseModel):
    """HTTP/WebSocket API server settings (nyxara/server/app.py).

    The server lets NYXARA be reached from an app, phone, or the web instead of only the
    local console. It is a thin, authenticated transport over the *same* sovereign loop —
    every request still flows through ``NyxaraCore.process`` and every gate; the network
    is just another mouth, never a way around the control law.

    Auth is a single bearer token (the Master's credential, this being a single-Master
    system). When ``api_token`` is set, every ``/v1`` route requires it. PROD additionally
    *requires* a token to exist at all (fail-closed) and keeps the sovereign control routes
    (pause/resume/scram) enabled.
    """

    model_config = {"validate_assignment": True}

    host: str = "127.0.0.1"          # bind localhost by default; set 0.0.0.0 to expose
    port: int = Field(default=8000, ge=1, le=65535)
    api_token: Optional[SecretStr] = None
    # CORS allow-list for browser clients. Empty -> no cross-origin browser access.
    cors_origins: List[str] = Field(default_factory=list)
    # Expose the sovereign control routes (pause/resume/scram) over HTTP.
    enable_control: bool = True
    # Cap multi-step agent runs requested over the wire.
    max_agent_steps: int = Field(default=8, ge=1, le=64)
    request_timeout_s: float = Field(default=120.0, gt=0)


class WebConfig(BaseModel):
    """Internet-access settings (senses/web.py, senses/search.py, agency/net_request.py).

    NYXARA reaches the live web through three governed tools — ``web_search`` (real SERP
    results), ``web_fetch`` (read a page) and ``http_request`` (call any API). This is their
    single source of truth.

    Search works with **no API key**: the keyless DuckDuckGo HTML backend is the default. A
    ``brave``/``tavily``/``serpapi`` key (any one) automatically upgrades quality and is
    preferred under ``search_provider="auto"``.

    Defaults are tuned to a **maximum-reach** profile: large size caps, long timeouts, many
    redirects and an effectively-uncapped fetch rate. ``allow_private=True`` opens
    loopback/private/link-local hosts (the SSRF guard is off) so NYXARA can also reach
    services on the Master's own network. Prompt-injection *screening* of fetched page text
    stays on (``injection_scan``) — it sanitises untrusted content but never limits reach.
    """

    model_config = {"validate_assignment": True}

    # search provider routing: auto = keyed provider if a key is set, else the keyless tail
    # (DuckDuckGo full-web → Wikipedia → instant-answer). "wikipedia" works from any IP.
    search_provider: Literal[
        "auto", "duckduckgo", "wikipedia", "brave", "tavily", "serpapi"] = "auto"
    brave_api_key: Optional[SecretStr] = None
    tavily_api_key: Optional[SecretStr] = None
    serpapi_api_key: Optional[SecretStr] = None

    # ceilings — "max" profile (still bounded so a runaway loop can't be infinite).
    max_results: int = Field(default=25, ge=1, le=100)
    max_bytes: int = Field(default=25_000_000, ge=1_024, le=200_000_000)
    timeout_s: float = Field(default=60.0, gt=0, le=600.0)
    max_fetches_per_min: int = Field(default=10_000, ge=1, le=1_000_000)
    user_agent: str = "NYXARA/1.0 (+https://nyxara.ai)"
    max_redirects: int = Field(default=20, ge=0, le=50)

    # access posture: unrestricted reach. allow_private=True turns the SSRF guard OFF so
    # loopback/private/link-local hosts are reachable. injection_scan keeps untrusted page
    # text sanitised (defense in depth; does not reduce reach).
    allow_private: bool = True
    injection_scan: bool = True


class ObservabilityConfig(BaseModel):
    model_config = {"validate_assignment": True}

    log_level: LogLevel = LogLevel.INFO
    structured_logs: bool = True
    trace_sampling: float = Field(default=1.0, ge=0.0, le=1.0)
    telemetry_enabled: bool = True
    replay_recording: bool = True  # kernel/replay.py deterministic capture
    honesty_enforcement: bool = True  # observe/honesty.py calibrated reporting


class PathsConfig(BaseModel):
    """Filesystem layout. Created on demand via :meth:`ensure`."""

    # NOTE: no validate_assignment — the after-validator assigns its own fields,
    # which would otherwise recurse. Paths are derived once at construction.
    model_config = {"validate_assignment": False}

    root: Path = Field(default_factory=lambda: Path(os.getenv("NYXARA_HOME", str(Path.home() / ".nyxara"))))
    data_dir: Optional[Path] = None
    memory_dir: Optional[Path] = None
    audit_dir: Optional[Path] = None
    replay_dir: Optional[Path] = None
    keys_dir: Optional[Path] = None
    sandbox_dir: Optional[Path] = None

    @model_validator(mode="after")
    def _derive(self) -> "PathsConfig":
        self.data_dir = self.data_dir or self.root / "data"
        self.memory_dir = self.memory_dir or self.root / "memory"
        self.audit_dir = self.audit_dir or self.root / "audit"
        self.replay_dir = self.replay_dir or self.root / "replay"
        self.keys_dir = self.keys_dir or self.root / "keys"
        self.sandbox_dir = self.sandbox_dir or self.root / "sandbox"
        return self

    def all_dirs(self) -> List[Path]:
        return [
            self.root, self.data_dir, self.memory_dir, self.audit_dir,
            self.replay_dir, self.keys_dir, self.sandbox_dir,
        ]

    def ensure(self) -> "PathsConfig":
        """Create every directory (idempotent). Keys dir is locked to 0700."""
        for d in self.all_dirs():
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
        try:
            if self.keys_dir is not None:
                os.chmod(self.keys_dir, 0o700)
        except OSError:
            pass  # non-POSIX or permission-restricted FS; non-fatal
        return self


# --------------------------------------------------------------------------- #
# Top-level settings
# --------------------------------------------------------------------------- #
class NyxaraSettings(BaseSettings):
    """Root configuration object. Construct via :func:`get_settings`.

    Environment overrides use the ``NYXARA_`` prefix and ``__`` for nesting::

        NYXARA_PROFILE=prod
        NYXARA_LLM__PROVIDER=openai
        NYXARA_RESOURCES__MAX_CONCURRENT_TASKS=128
        NYXARA_LLM__ANTHROPIC_API_KEY=sk-ant-...
    """

    model_config = SettingsConfigDict(
        env_prefix="NYXARA_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    schema_version: str = CONFIG_SCHEMA_VERSION
    profile: Profile = Profile.DEV
    instance_name: str = "nyxara"

    owner: OwnerIdentity = Field(default_factory=OwnerIdentity)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    foundry: FoundryConfig = Field(default_factory=FoundryConfig)
    capability_foundry: CapabilityFoundryConfig = Field(default_factory=CapabilityFoundryConfig)
    autoforge: AutoForgeConfig = Field(default_factory=AutoForgeConfig)
    genesis: GenesisConfig = Field(default_factory=GenesisConfig)
    loyalty: LoyaltyConfig = Field(default_factory=LoyaltyConfig)
    flywheel: FlywheelConfig = Field(default_factory=FlywheelConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    role_council: RoleCouncilConfig = Field(default_factory=RoleCouncilConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    self_model_router: SelfModelRouterConfig = Field(default_factory=SelfModelRouterConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    self_improvement: SelfImprovementConfig = Field(default_factory=SelfImprovementConfig)
    meta_research: MetaResearchConfig = Field(default_factory=MetaResearchConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    agency: AgencyConfig = Field(default_factory=AgencyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # ---- validators / cross-field hardening ---- #
    @field_validator("profile", mode="before")
    @classmethod
    def _coerce_profile(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _harden_for_profile(self) -> "NyxaraSettings":
        """Enforce non-negotiable safety posture per profile.

        PROD forces all safety-critical flags ON regardless of what was supplied —
        this is a fail-closed guarantee (Rules 5, 6, 8). DEV stays permissive but
        still cannot disable invariant enforcement or audit logging.
        """
        # These can NEVER be disabled, in any profile.
        self.features.invariant_enforcement = True
        self.features.audit_logging = True
        self.features.corrigibility = True
        self.guard.rule_modification_locked = True
        # Untrusted web content is always screened for prompt-injection (defense in depth);
        # this sanitises page text, it never limits NYXARA's reach.
        self.web.injection_scan = True
        # Keep the governor's "web" rate bucket in sync with the web config (one knob:
        # NYXARA_WEB__MAX_FETCHES_PER_MIN drives the actual throttle).
        self.resources.max_web_fetches_per_min = self.web.max_fetches_per_min

        if self.profile is Profile.PROD:
            self.observability.log_level = (
                LogLevel.INFO
                if self.observability.log_level is LogLevel.DEBUG
                else self.observability.log_level
            )
            self.features.simulation_required = True
            self.agency.sandbox_before_real_action = True
            self.guard.zero_trust = True
            self.guard.kill_switch_enabled = True
            self.llm.allow_mock_fallback = False  # prod must use a real provider
        elif self.profile is Profile.TEST:
            # Tests run hermetically: never reach the network.
            self.llm.provider = LLMProvider.MOCK
            self.llm.allow_mock_fallback = True
            self.observability.telemetry_enabled = False
        return self

    # ---- convenience ---- #
    @property
    def is_prod(self) -> bool:
        return self.profile is Profile.PROD

    @property
    def is_dev(self) -> bool:
        return self.profile is Profile.DEV

    def redacted(self) -> Dict[str, Any]:
        """Log-safe dict with all secrets masked. Use this for any logging."""
        def _scrub(obj: Any) -> Any:
            if isinstance(obj, SecretStr):
                return "***REDACTED***" if obj.get_secret_value() else None
            if isinstance(obj, dict):
                return {k: _scrub(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_scrub(v) for v in obj]
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return _scrub(self.model_dump())

    def to_json(self, *, redact: bool = True, indent: int = 2) -> str:
        data = self.redacted() if redact else self.model_dump(mode="json")
        return json.dumps(data, indent=indent, default=str, sort_keys=True)

    def save(self, path: str | Path, *, redact: bool = False) -> Path:
        """Persist config to disk. ``redact=False`` keeps secrets (use 0600 files!)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(redact=redact), encoding="utf-8")
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
        return p

    @classmethod
    def from_file(cls, path: str | Path, **overrides: Any) -> "NyxaraSettings":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.update(overrides)
        return cls(**data)

    @classmethod
    def for_profile(cls, profile: Profile | str, **overrides: Any) -> "NyxaraSettings":
        return cls(profile=profile, **overrides)


# --------------------------------------------------------------------------- #
# Process-wide accessor (cached singleton)
# --------------------------------------------------------------------------- #
_CACHED: Optional[NyxaraSettings] = None


def get_settings() -> NyxaraSettings:
    """Return the cached, validated settings singleton (env-driven)."""
    global _CACHED
    if _CACHED is None:
        _CACHED = NyxaraSettings()
    return _CACHED


def reload_settings(**overrides: Any) -> NyxaraSettings:
    """Rebuild the settings singleton (test/runtime reconfiguration).

    With no overrides, re-reads the environment. With overrides, applies them
    explicitly on top of the environment-derived defaults.
    """
    global _CACHED
    _CACHED = NyxaraSettings(**overrides) if overrides else NyxaraSettings()
    return _CACHED


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA config self-test")
    print("=" * 70)

    dev = NyxaraSettings.for_profile(Profile.DEV)
    print(f"profile      : {dev.profile.value}")
    print(f"owner        : {dev.owner.name} ({dev.owner.handle}) <{dev.owner.email}>")
    print(f"llm provider : {dev.llm.provider.value} -> {dev.llm.active_model()}")
    print(f"defense layers: {dev.guard.defense_layers}")
    print(f"invariants on: {dev.features.invariant_enforcement} (forced)")

    prod = NyxaraSettings.for_profile(Profile.PROD)
    assert prod.features.invariant_enforcement is True
    assert prod.features.audit_logging is True
    assert prod.guard.rule_modification_locked is True
    assert prod.llm.allow_mock_fallback is False
    print("\nprod hardening OK (invariants/audit/corrigibility forced, mock disabled)")

    test = NyxaraSettings.for_profile(Profile.TEST)
    assert test.llm.provider is LLMProvider.MOCK
    print("test profile forces MOCK llm OK")

    # Secret redaction
    s = NyxaraSettings(profile="dev")
    s.llm.anthropic_api_key = SecretStr("sk-ant-SECRET")
    red = s.redacted()
    assert red["llm"]["anthropic_api_key"] == "***REDACTED***"
    assert "SECRET" not in s.to_json(redact=True)
    print("secret redaction OK")

    # Paths
    p = dev.paths
    print(f"\npaths root   : {p.root}")
    print(f"derived dirs : {len(p.all_dirs())} ({', '.join(d.name for d in p.all_dirs()[1:])})")

    print("\nALL SELF-TESTS PASSED ✓")
