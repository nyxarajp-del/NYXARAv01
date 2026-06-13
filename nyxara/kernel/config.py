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
    "CouncilConfig",
    "RoleCouncilConfig",
    "MemoryConfig",
    "GuardConfig",
    "AgencyConfig",
    "MCPServerSpec",
    "MCPConfig",
    "ServerConfig",
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
    max_web_fetches_per_min: int = Field(default=60, ge=1)
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
    ``auto`` which uses the optional torch nano-GPT when torch is installed and falls
    back to the always-available pure-stdlib n-gram model otherwise.

    ``profile`` selects a transformer scale: the default ``custom`` honours the explicit
    dimension fields below (a tiny, CPU-/CI-runnable model), while ``gpt2`` reaches real
    GPT-2 scale (~124M params). Heavy profiles still require torch and ``enabled=True``.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = False
    backend: Literal["auto", "ngram", "nanogpt", "lora"] = "auto"
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


class RoleCouncilConfig(BaseModel):
    """Level 4 — Internal Role Council settings (mind/role_council.py).
    Six role personas examine significant turns; NYXARA synthesises and judges."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    max_tokens_per_role: int = Field(default=256, ge=32, le=2048)
    timeout_s: float = Field(default=30.0, gt=0)


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
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    role_council: RoleCouncilConfig = Field(default_factory=RoleCouncilConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    agency: AgencyConfig = Field(default_factory=AgencyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
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
