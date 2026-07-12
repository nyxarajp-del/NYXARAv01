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
  using ``__`` as the nesting delimiter, e.g. ``NYXARA_LLM__PROVIDER=tinyllama``.
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
    "SwarmConfig",
    "GeneralIntelligenceConfig",
    "MemoryConfig",
    "CausalConfig",
    "SelfImprovementConfig",
    "MetaResearchConfig",
    "GuardConfig",
    "VaultConfig",
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

    AUTO = "auto"                 # ladder self→gguf→tinyllama→mock: her own promoted weights serve
    #                               the moment they exist (and pass the serve gate) — no manual flip
    TINYLLAMA = "tinyllama"       # in-process TinyLlama-1.1B-Chat, downloaded via HuggingFace
    GGUF = "gguf"                 # in-process GGUF (llama.cpp) — the Qwythos-9B quant, cheap to serve
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
    self_bootstrap: bool = True         # growth/explorer.py — Environment-Driven Learning (Rule 4)
    self_model_foundry: bool = False    # growth/foundry.py — build/upgrade her OWN model (Rule 4)
    neural_architecture_search: bool = True  # growth/genesis.py — she designs her OWN brain (Rule 4)
    synthetic_self_curation: bool = True     # growth/synthesis.py — AlphaGo-Zero synthetic data (Rule 4)
    dynamic_topology_expansion: bool = True  # growth/topology.py — runtime Net2Net brain growth (Rule 4)
    novel_discovery: bool = True             # growth/eureka.py — self-generated, prover-certified novel discovery (Rule 4)
    open_world_generalization: bool = True   # growth/open_world.py — crack never-before-seen systems from first principles (Rule 4)
    self_growing_transfer: bool = True       # mind/transfer.py — her transfer library grows from lived structure, persists across restarts (Rule 4)
    mathematical_soul_binding: bool = True   # growth/loyalty.py — the Loyalty Equation (Rule 4)
    multi_llm_council: bool = False     # mind/council.py — convene many LLMs as a panel of tools
    toolsmithing: bool = True           # agency/toolsmith.py
    web_access: bool = True             # senses/web.py
    vision: bool = False                # heavy ML; off by default
    audio: bool = False                 # heavy ML; off by default
    transformers_inference: bool = False  # in-process HuggingFace model; heavy ML, off by default
    dream_consolidation: bool = True    # memory/consolidation.py
    fractal_temporal_hierarchy: bool = True  # temporal/ — loops within loops (ms/s/days)
    simulation_required: bool = True    # sim/ dry-run gate before real action
    invariant_enforcement: bool = True  # kernel/invariants.py — NEVER off in prod
    audit_logging: bool = True          # guard/audit.py — NEVER off in prod
    corrigibility: bool = True          # guard/corrigibility.py — NEVER off in prod


# --------------------------------------------------------------------------- #
# Subsystem configs
# --------------------------------------------------------------------------- #
class DeepReasoningConfig(BaseModel):
    """Always-maximum deep-reasoning controller (mind/deep_reasoning.py) — Problem #1, the ceiling.

    A fixed base model has a fixed single-pass depth. This controller raises NYXARA's *effective*
    reasoning depth (not the model's raw intelligence) by climbing the whole effort ladder every
    genuine reasoning turn — self-consistency → deliberation → MCTS search → verified refinement —
    and keeping the answer an independent verifier scores highest. It compounds via
    ``mind/effort_memory.py``: from lived verified outcomes it learns which rung pays off for each
    kind of problem and aims the extra self-consistency budget there. NYXARA drives it from her own
    measured signals, not the LLM. Only runs with a real provider; a no-op on a keyless machine, so
    the offline path is unchanged. The kernel still disposes every proposal through every gate.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                                   # gated at runtime on a real provider
    # Ladder ceiling: 1 self-consistency · 2 deliberation · 3 MCTS · 4 verified refinement.
    max_rung: int = Field(default=4, ge=1, le=4)
    # Base self-consistency width per neural rung; the rung a signature has learned to reward gets
    # this much again on top (same maximum budget, aimed where it measurably pays off).
    samples: int = Field(default=3, ge=1, le=9)
    keep_best: bool = True                                 # keep the verifier-best across all rungs
    # Ground the keep-best selection in *truth* where the domain is decidable (exact faculty oracle
    # / machine-checkable Prover certificate — mind/grounded_verifier.py), so the climb selects
    # correct answers over merely fluent ones. Falls through to the intrinsic verifier on every
    # non-decidable prompt, so open-ended/offline behaviour is unchanged. This is the actual
    # ceiling-break: search steered by correctness, not polish.
    ground_verifier: bool = True
    max_seconds: float = Field(default=60.0, ge=1.0, le=600.0)   # runaway guard, not a quality cap
    # Compounding: learn per-problem which rung pays off and persist it across restarts.
    learn_effort: bool = True
    effort_min_observations: float = Field(default=3.0, ge=0.0)  # evidence before a suggestion sticks
    effort_success_floor: float = Field(default=0.5, ge=0.0, le=1.0)


class LLMConfig(BaseModel):
    """Stateless, fully local LLM faculty settings (mind/llm.py).

    Two real backends run in-process, no cloud providers and no API keys: the
    HuggingFace ``transformers`` path (``tinyllama``, any HF causal-LM id via
    ``NYXARA_LLM__TINYLLAMA_*``) and the llama.cpp path (``gguf``, a quantized GGUF
    served through ``llama-cpp-python`` via ``NYXARA_LLM__GGUF_*``). The shipped
    default is Qwythos-9B: served cheaply from its GGUF quant, LoRA-fine-tuned on its
    safetensors parent by the foundry. ``self`` serves the foundry-forged model (the
    LoRA adapter over that base) and ``mock`` is the deterministic offline fallback;
    every backend degrades to ``mock`` when its heavy deps are absent.

    The default ``auto`` closes the train→serve loop: it walks the ladder
    self→gguf→tinyllama→mock, so the moment the foundry promotes her own weights (and
    they pass the serve gate — see ``self_serve_any_backend``) SHE serves them, with
    zero manual reconfiguration; until then the strongest static backend answers.
    """

    model_config = {"validate_assignment": True}

    provider: LLMProvider = LLMProvider.AUTO
    # ---- TinyLlama-1.1B: model & load-time control ---- #
    tinyllama_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tinyllama_device: str = ""              # "" -> auto (cuda if available); or "cuda", "cpu", "mps"
    tinyllama_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    # Quantized load (needs bitsandbytes + CUDA; silently full-precision otherwise).
    tinyllama_load_in_4bit: bool = False
    tinyllama_load_in_8bit: bool = False
    tinyllama_bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    tinyllama_bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    tinyllama_bnb_4bit_use_double_quant: bool = True
    tinyllama_attn_implementation: Literal["", "eager", "sdpa", "flash_attention_2"] = ""
    tinyllama_use_cache: bool = True        # KV cache during generation
    tinyllama_trust_remote_code: bool = False
    # Serve a LoRA fine-tune directly: point at a peft adapter dir (e.g. a foundry
    # ``versions/vN/adapter``); ``merge_adapter`` folds it into the base for faster inference.
    tinyllama_adapter_path: Optional[Path] = None
    tinyllama_merge_adapter: bool = False
    # ---- TinyLlama-1.1B: generation control (per-request LLMRequest fields win) ---- #
    tinyllama_top_k: int = Field(default=50, ge=0)                    # 0 -> disabled
    tinyllama_repetition_penalty: float = Field(default=1.1, ge=0.5, le=2.0)
    tinyllama_no_repeat_ngram_size: int = Field(default=0, ge=0, le=20)  # 0 -> disabled
    tinyllama_min_new_tokens: int = Field(default=0, ge=0)            # 0 -> disabled
    tinyllama_num_beams: int = Field(default=1, ge=1, le=16)
    tinyllama_length_penalty: float = Field(default=1.0, ge=-2.0, le=2.0)  # beams > 1 only
    # "auto" -> sample iff request temperature > 0; "always"/"never" force it.
    tinyllama_do_sample: Literal["auto", "always", "never"] = "auto"
    # TinyLlama's context window is 2048 tokens; prompts are left-truncated to fit.
    tinyllama_max_input_tokens: int = Field(default=2048, ge=64, le=2048)
    # Zephyr chat template (system/user/assistant). False -> flat prompt, for base checkpoints.
    tinyllama_use_chat_template: bool = True
    # ---- GGUF (llama.cpp): the Qwythos-9B quant, served in-process via llama-cpp-python ---- #
    # GGUF is an inference-only format, so this backend never trains — it serves the cheap
    # quantized weights (Q4 ≈ 6 GB) while the foundry LoRA-tunes the safetensors parent. Needs
    # ``llama-cpp-python`` (``.[gguf]``); absent, the provider degrades honestly to the mock.
    gguf_model: str = "llmfan46/Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic-GGUF"
    # Which quant file to pull from the repo (a llama.cpp glob; picks the single match).
    gguf_filename: str = "*Q4_K_M.gguf"
    gguf_n_ctx: int = Field(default=8192, ge=256)              # context window (0 -> model default)
    gguf_n_gpu_layers: int = Field(default=-1, ge=-1)          # -1 -> all layers on GPU; 0 -> CPU-only
    gguf_n_threads: int = Field(default=0, ge=0)               # 0 -> llama.cpp auto (all cores)
    gguf_chat_format: str = ""                                 # "" -> the GGUF's own chat template
    gguf_seed: int = Field(default=-1, ge=-1)                  # -1 -> nondeterministic (request seed wins)
    # Serve a llama.cpp-converted LoRA adapter (a foundry adapter exported to GGUF) on top.
    gguf_lora_path: Optional[Path] = None
    gguf_lora_scale: float = Field(default=1.0, ge=0.0, le=4.0)
    # NYXARA's OWN model, built & promoted by the foundry. None -> paths.data_dir/"foundry".
    self_model_dir: Optional[Path] = None
    self_model_version: Optional[int] = None  # None -> the currently-promoted (active) version
    # Serve gate for provider=auto: a promoted LoRA (the served base, improved) auto-serves;
    # a small from-scratch backend replacing a large pretrained model would DEGRADE live
    # behavior, so it needs this explicit opt-in (or provider=self). Honesty over theatre.
    self_serve_any_backend: bool = False
    # Hot-reload memory policy: drop a RAM/VRAM-heavy old model (LoRA base) BEFORE loading
    # the newly-promoted one, instead of holding two bases at once. Failure restores the
    # previous version from its on-disk dir.
    self_reload_lean: bool = True

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
    # ---- Always-maximum deep reasoning (mind/deep_reasoning.py) — Problem #1, the ceiling ---- #
    deep_reasoning: DeepReasoningConfig = Field(default_factory=DeepReasoningConfig)

    @model_validator(mode="after")
    def _quant_exclusive(self) -> "LLMConfig":
        if self.tinyllama_load_in_4bit and self.tinyllama_load_in_8bit:
            raise ValueError("tinyllama_load_in_4bit and tinyllama_load_in_8bit are mutually exclusive")
        return self

    def active_model(self) -> str:
        return {
            LLMProvider.AUTO: "auto",
            LLMProvider.TINYLLAMA: self.tinyllama_model,
            LLMProvider.GGUF: self.gguf_model,
            LLMProvider.SELF: "nyxara-self",
            LLMProvider.MOCK: "mock",
        }[self.provider]

    def active_key(self) -> Optional[SecretStr]:
        """Always ``None`` — every backend runs locally; there are no API keys."""
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

    **On by default, on every machine** — real, weight-changing learning is not optional
    hardware garnish. :func:`~nyxara.growth.foundry_models.build_model` guarantees the
    strongest *genuine* trainable backend the box allows: LoRA over a pretrained base
    (torch+peft+GPU), a from-zero torch nano-GPT, the from-scratch NumPy-autograd
    transformer on a torch-less box, or — the honest floor when even NumPy is absent —
    a Kneser-Ney n-gram. Heavy & self-modifying, so still fully gauntlet-gated and
    reversible (the TEST profile seals it off for hermetic suites). The default backend
    is ``lora`` — LoRA fine-tuning of a pretrained base (real capability); on a CPU-only
    box ``lora_requires_gpu`` downshifts the forge to a trainable neural backend instead
    of stalling on a multi-GB base download.

    ``profile`` selects a transformer scale: the default ``custom`` honours the explicit
    dimension fields below (a tiny, CPU-/CI-runnable model), while ``gpt2`` reaches real
    GPT-2 scale (~124M params). Heavy profiles still require torch and ``enabled=True``.
    """

    model_config = {"validate_assignment": True}

    # ON by default everywhere: she always forges a REAL model (as neural as the machine
    # allows — see the class docstring ladder). Every forge still clears the same gauntlet
    # and is reversible. Override with NYXARA_FOUNDRY__ENABLED; TEST profile forces it off.
    enabled: bool = True
    # Default to LoRA fine-tuning of a pretrained base — the path to genuine capability
    # (she stands on a real base and learns a small adapter from her own memory). Degrades
    # safely to the always-on n-gram backend when torch+transformers+peft are absent.
    backend: Literal["auto", "ngram", "kngram", "nanogpt", "lora"] = "lora"
    # A LoRA forge on a no-CUDA box means downloading + full-precision-loading a multi-GB
    # base just to crawl — downshift to a genuinely trainable neural backend instead
    # (nanogpt with torch, the NumPy transformer without). Explicitly disable to force
    # CPU LoRA anyway (e.g. a big-RAM box fine-tuning a small base).
    lora_requires_gpu: bool = True
    # Transformer scale. "custom" => use the explicit dimensions below (default, tiny).
    profile: Literal["custom", "tiny", "small", "gpt2", "gpt2-medium"] = "custom"
    # Pure-stdlib n-gram backend.
    ngram_order: int = Field(default=3, ge=1, le=8)
    # ---- weight surgery (growth/weight_surgery.py) — interpret + edit her own weights ---- #
    # NYXARA looks inside her own forged model and edits a *specific circuit* (an n-gram count row, a
    # neural parameter) instead of retraining from scratch. Every edit is reversible and gauntlet-
    # gated (intent achieved AND perplexity non-regressing on a probe); it touches only capability
    # weights, never the immutable character core. On by default; real on the always-available n-gram
    # brain and on the neural brain when torch is present.
    weight_surgery_enabled: bool = True
    weight_surgery_tol: float = Field(default=0.25, ge=0.0, le=2.0)   # max perplexity regression kept
    weight_surgery_max_edits: int = Field(default=4, ge=0, le=64)
    # ---- continual learning (growth/foundry.py + memory/elastic_synapses.py) ---- #
    # Instead of forging every model from scratch off the replay buffer (which forgets), NYXARA can
    # warm-start from her active model and consolidate important weights with EWC (Fisher importance),
    # so new skills do not erase old ones. The loyalty core stays infinitely important (never
    # consolidated over). On by default; the EWC penalty strength is ``ewc_lambda``.
    continual_learning: bool = True
    ewc_lambda: float = Field(default=1.0, ge=0.0, le=1000.0)
    # ---- Always-on offline brain (mind/self_reasoner.SelfBrain) ---- #
    # NYXARA's keyless general-intelligence: a retrieval-augmented brain over everything she has
    # learned (pure-stdlib LearnedEmbedder), with a hardened generative backend as the fallback.
    # On by default — these knobs only tune it. Its generative core is a real neural net by
    # default (the pure-NumPy transformer whenever NumPy is present, the torch nano-GPT when
    # torch is present), degrading to the pure-stdlib n-gram only on a box without even NumPy.
    self_brain_retrieval: bool = True               # answer from learned sentences before generating
    self_brain_top_k: int = Field(default=4, ge=1, le=16)        # retrieved sentences considered
    self_brain_sim_threshold: float = Field(default=0.35, ge=0.0, le=1.0)  # compose-from-retrieval floor
    # Generative backend for the always-on brain. "auto" now means **neural-when-possible**:
    # the torch nano-GPT when torch is present, else the pure-NumPy transformer (genesis_np) —
    # a real gradient-trained neural net with backprop, no torch/LLM/cloud — and only the
    # word-KN n-gram count table when NumPy itself is absent. So her default lived-learning is a
    # real neural network whose weights change, not statistics. "genesis_np" pins the NumPy
    # neural brain regardless of torch; "kngram"/"nanogpt" pin those explicitly.
    self_brain_backend: Literal["auto", "kngram", "nanogpt", "genesis_np"] = "auto"
    # ---- Online weight learning (mind/self_reasoner.SelfBrain) — REAL learning, not just recall ---- #
    # The offline brain's generative weights *accumulate* from lived, reward-weighted experience
    # instead of being rebuilt from scratch off a corpus window (which is remembering, not learning).
    # Each refit folds only the NEW exchanges on top of the existing weights (continual, EWC-anchored),
    # so a lesson persists in the weights even after its text leaves the corpus window. A successful
    # turn is reinforced ∝ reward; a punished turn's continuation is reversibly suppressed (weight
    # surgery). Pure-stdlib on the KN brain, a real gradient step on the neural brain — on by default;
    # the loyalty core is never learnable (any doc that would teach over it is refused, fail-closed).
    self_brain_online_learn: bool = True
    # Reinforcement strength: a reward-1.0 exchange is folded with this many extra repetitions
    # (multiplicity = 1 + round(scale * reward)), so successful patterns earn stronger weights.
    self_brain_reward_scale: float = Field(default=3.0, ge=0.0, le=32.0)
    # Consolidate accumulated weights as a durable EWC anchor every N refits (forgetting-protection).
    self_brain_consolidate_every: int = Field(default=4, ge=1, le=256)
    # Neural (nanogpt/lora/genesis_np) online gradient step size per refit — a warm-continued update.
    self_brain_neural_online_steps: int = Field(default=40, ge=1, le=1000)
    # How many newly-learned docs trigger a weight fold. Lower = she folds lived experience into her
    # weights more often (higher training intensity, a little more CPU per turn). Default 4 (was an
    # internal 8) so her weights track experience closely from turn one.
    self_brain_refit_every: int = Field(default=4, ge=1, le=256)
    # ---- Real-time weight learning (mind/self_reasoner.SelfBrain.online_step) ---- #
    # The above knobs made online folding *possible*; these make it genuinely LIVE and CONTINUOUS.
    # Every turn (and every autonomic tick when idle) NYXARA folds queued lived experience into her
    # generative core weights herself — not only when she happens to generate a reply. Each fold is
    # gauntlet-gated on the neural path exactly like weight surgery: snapshot -> step -> verify the
    # held-out perplexity did not regress beyond the tolerance -> keep, else roll back to the exact
    # prior weights. So a bad online step can never degrade the core. On by default.
    # Max fractional perplexity regression a kept online step may cause on the held-out probe.
    # Defaults to the weight-surgery tolerance; lower = stricter (rolls back more aggressively).
    self_brain_online_verify_tol: float = Field(default=0.25, ge=0.0, le=2.0)
    # Max queued exchanges folded per live flush (per turn / per tick). Bounds the CPU a single
    # real-time fold may cost so the turn stays responsive; the remainder folds on the next flush.
    self_brain_flush_budget: int = Field(default=8, ge=1, le=256)
    # ---- Few-shot skill induction (cognition/skill_induction.py) ---- #
    # NYXARA learns a *task* from a handful of (input -> output) demonstrations by synthesising a
    # verified, reusable transformation she then applies to genuinely new inputs — real, transferable
    # capability gain she performs herself (pure stdlib, no LLM). A program is accepted only when it
    # reproduces every demonstration exactly; with >=3 demos one is held out to measure transfer.
    skill_induction_enabled: bool = True
    skill_min_demos: int = Field(default=2, ge=1, le=32)          # demonstrations needed to learn
    skill_max_program_depth: int = Field(default=3, ge=0, le=8)   # composed-op search depth (Occam)
    skill_beam_width: int = Field(default=16, ge=1, le=128)       # program-search beam
    skill_apply_confidence: float = Field(default=0.55, ge=0.0, le=1.0)  # floor to answer with a skill
    # Optional torch nano-GPT dimensions (only used when torch is present, and only when
    # profile == "custom"; a named profile overrides these).
    block_size: int = Field(default=64, ge=8, le=1024)
    n_layer: int = Field(default=2, ge=1, le=24)
    n_head: int = Field(default=2, ge=1, le=32)
    n_embd: int = Field(default=64, ge=8, le=2048)
    # LoRA fine-tuning backend (backend="lora"; needs torch+transformers+peft, .[foundry]).
    # Adapts the base to NYXARA's lived memory by training a small low-rank adapter — the path
    # to genuine capability. Default base is the Qwythos-9B safetensors parent (NOT the -GGUF
    # repo: GGUF is inference-only and cannot be LoRA-trained); at 9B it wants QLoRA (4-bit) and
    # a GPU, and degrades to the always-on n-gram backend on a bare CPU/CI box.
    base_model: str = "llmfan46/Qwythos-9B-Claude-Mythos-5-1M-uncensored-heretic"
    # The Qwythos base ships custom Qwen3.5 modeling code (hybrid Gated-DeltaNet attention);
    # transformers needs trust_remote_code to load it. Kept a knob so a stock base can turn it
    # off. Threaded into every from_pretrained in growth/foundry_models.LoRAModel.
    trust_remote_code: bool = True
    lora_r: int = Field(default=8, ge=1, le=256)
    # Auto-scale the LoRA rank to the base model's width (bigger base -> higher rank, so a
    # 7B base converges while a tiny base stays cheap). When True the foundry infers the rank
    # from the loaded base; set False to honour the explicit ``lora_r`` above verbatim.
    lora_r_auto: bool = True
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.9)
    lora_lr: float = Field(default=2e-4, gt=0.0, le=1.0)
    # Which modules receive LoRA adapters. The default is every projection in the llama
    # architecture (attention + MLP). Empty by default so peft infers the targets for the
    # Qwythos base's hybrid Qwen3.5 arch (its Gated-DeltaNet layers don't use the llama names);
    # LoRAModel._apply_lora falls back to "all-linear" if inference fails. Set explicitly (e.g.
    # the llama q/k/v/o/gate/up/down set) to pin placement on a known base.
    lora_target_modules: List[str] = Field(default_factory=list)
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    lora_use_rslora: bool = False           # rank-stabilised LoRA scaling
    # Extra modules trained (and saved) in full precision, e.g. ["lm_head", "embed_tokens"].
    lora_modules_to_save: List[str] = Field(default_factory=list)
    max_seq_len: int = Field(default=256, ge=8, le=8192)
    # QLoRA: load the frozen base in 4-bit (or 8-bit) so bigger bases fine-tune on a single
    # consumer GPU. Honoured only when bitsandbytes + CUDA are present; on CPU/CI it degrades
    # to full-precision LoRA (no crash). On by default because the Qwythos-9B base needs it to
    # fit; a tiny base (e.g. TinyLlama-1.1B) trains fine either way.
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    gradient_checkpointing: bool = True
    # ---- Optimiser / schedule (full-control training loop) ---- #
    batch_size: int = Field(default=1, ge=1, le=256)             # windows per micro-step
    grad_accum_steps: int = Field(default=1, ge=1, le=1024)      # micro-steps per optimiser step
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)    # fraction of steps spent warming up
    lr_scheduler: Literal["constant", "linear", "cosine"] = "constant"
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)
    adam_eps: float = Field(default=1e-8, gt=0.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)            # 0 -> no clipping
    # 0 -> train for ``train_steps``; >0 -> that many passes over the corpus windows instead.
    train_epochs: int = Field(default=0, ge=0, le=100)
    # Training / data.
    train_steps: int = Field(default=200, ge=1)
    max_corpus_items: int = Field(default=2000, ge=1)
    eval_holdout_frac: float = Field(default=0.2, gt=0.0, lt=1.0)
    # Difficulty curriculum: order the training corpus easy -> hard so the model converges on
    # simple structure before hard examples (faster, less catastrophic forgetting). Pure-stdlib
    # difficulty proxy (length + lexical entropy); refined by the active model's perplexity when
    # one exists. Set False to keep the historical random/strided order.
    curriculum: bool = True
    # A candidate must beat the active model's perplexity by at least this fraction.
    min_perplexity_improvement: float = Field(default=1e-4, ge=0.0)
    # Capability gauntlet (Phase 3): a promotion must not *regress* on a held capability
    # benchmark, not merely lower perplexity. Tolerant of tiny noise via the margin.
    capability_gate: bool = True
    capability_regression_tol: float = Field(default=1e-6, ge=0.0)
    # Teacher-relative audit (the visible ceiling-break): when on, every forged candidate is also
    # A/B'd against the external teacher on the SAME oracle-graded battery and the gap
    # (own_accuracy − teacher_accuracy) is recorded as ``accuracy_vs_teacher`` in the version's
    # metrics — so "she went above the teacher" is measured, not asserted. Measurement-only: it
    # never gates promotion (capability is already gated by the oracle benchmark above) and is a
    # no-op when no real teacher is configured. Off by default since it spends a teacher call per
    # forge; turn on to watch the gap close and cross zero.
    measure_vs_teacher: bool = False
    # Efficiency gate (Pillar F · Edge 3): when on, a candidate that does NOT lower perplexity may
    # still be promoted if it is *cheaper* (fewer params) while keeping capability within
    # ``efficiency_epsilon`` of the active model — capability compression (growth/efficiency.py).
    # Off by default so promotion semantics are unchanged unless the Master opts in.
    efficiency_gate: bool = False
    efficiency_epsilon: float = Field(default=0.02, ge=0.0)
    # Disk hygiene: how many versions to keep before pruning the oldest unpromoted ones.
    max_versions_kept: int = Field(default=10, ge=1)
    seed: int = 0
    # ---- Autonomous external data acquisition (growth/acquire.py) ---- #
    # Before forging, harvest REAL screened web text for NYXARA's knowledge gaps and fold it
    # into the corpus — so she trains on breadth she did not previously contain, not only her
    # own journal. ON within the foundry (but the whole foundry is opt-in via ``enabled``).
    # Every fetched page is injection-screened; suspicious pages are dropped, never trained on.
    acquire_data: bool = True
    # Seed topics to search when no concrete weakness/lesson gaps exist (gap_topics fallback).
    acquire_topics: List[str] = Field(default_factory=list)
    acquire_search_results: int = Field(default=6, ge=1, le=50)   # results pulled per topic
    acquire_max_per_topic: int = Field(default=3, ge=1, le=50)    # pages kept per topic
    acquire_max_docs: int = Field(default=60, ge=1, le=5000)      # ceiling per acquisition pass
    acquire_min_chars: int = Field(default=200, ge=1)             # too-short pages are dropped
    acquire_max_chars: int = Field(default=20_000, ge=1)          # per-doc text cap
    # ---- Compute-aware autoscaling (growth/compute_scale.py) ---- #
    # Scale the forged model to the compute actually available: a bare CPU keeps the always-on
    # backend; a strong CUDA GPU unlocks GPT-2 scale + 4-bit QLoRA. Clamped to reality — never
    # recommends a model the machine cannot train. Overrides ``profile`` when on.
    autoscale_to_compute: bool = True

    @model_validator(mode="after")
    def _quant_exclusive(self) -> "FoundryConfig":
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")
        return self

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
    min_examples: int = Field(default=10, ge=1)     # new verified examples needed to forge
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
    # Which substrate scores a genome when there is no torch. "numpy" builds NYXARA's OWN designed
    # architecture for REAL in pure NumPy (growth/genesis_numpy.py) — her layers, her synthesized
    # mixer and her searched optimizer are actually built, trained and graded, so the topology drives
    # fitness even on a bare machine. "ngram" uses the always-on word-Kneser-Ney substrate (faster;
    # the layer topology is inert, only n-gram order + smoothing are searched). "auto" picks numpy
    # when NumPy is importable, else ngram — so production designs a real brain by default.
    substrate: Literal["auto", "numpy", "ngram"] = "auto"
    population_size: int = Field(default=6, ge=2, le=128)
    generations: int = Field(default=3, ge=1, le=100)
    mutation_rate: float = Field(default=0.5, ge=0.0, le=1.0)   # P(mutate) vs P(crossover)
    micro_train_steps: int = Field(default=40, ge=1)            # tiny per-candidate training
    micro_corpus_items: int = Field(default=128, ge=1)
    # Each candidate is scored across this many resampled train/eval folds and the perplexity is
    # AVERAGED — so a champion is crowned on a denoised estimate, not one lucky split (the fix for
    # "rankings on a tiny corpus are noise"). 1 reproduces the old single-split behaviour.
    eval_seeds: int = Field(default=3, ge=1, le=20)
    block_size: int = Field(default=32, ge=8, le=1024)
    max_layers: int = Field(default=5, ge=2, le=24)
    quality_weight: float = Field(default=1.0, ge=0.0)         # smartness vs …
    speed_weight: float = Field(default=0.25, ge=0.0)         # … speed in the fitness blend
    min_new_examples: int = Field(default=20, ge=1)           # idle trigger, like AutoForge
    # First-run kickoff: on a fresh boot the flywheel is empty, so the new-experience trigger above
    # can never fire (0 - 0 < min_new_examples) and she would never actually search. When True she
    # runs exactly ONE search+promote cycle on her first idle tick, seeded from her own corpus, then
    # reverts to the new-experience cadence. Oversight-gated upstream; off → classic lazy behaviour.
    run_on_boot: bool = True
    seed: int = 0
    # ---- Max-level search engine (defaults now run the strong multi-objective path) ---- #
    # How the population evolves between generations:
    #   "elitism"      — the original: keep top third, breed by mutation/crossover.
    #   "tournament"   — pick parents by k-way tournaments (more selection pressure, less elitist).
    #   "regularized"  — AmoebaNet-style aging evolution: evict the OLDEST, not the worst, so the
    #                    search keeps exploring instead of locking onto an early lucky genome.
    #   "nsga2"        — NSGA-II elitist multi-objective evolution: rank by Pareto front + crowding
    #                    distance, driving the population toward the whole speed↔smartness↔cost front.
    search_strategy: Literal["elitism", "tournament", "regularized", "nsga2"] = "nsga2"
    tournament_k: int = Field(default=3, ge=2, le=32)         # k-way tournament size
    adaptive_mutation: bool = True      # raise the mutation rate when best-so-far stalls (anti-collapse)
    novelty_weight: float = Field(default=0.0, ge=0.0)        # reward genomes far from the population
    # Successive-halving bracket (Hyperband-flavoured): cheap-screen the whole population at a
    # fraction of micro_train_steps, then spend full training only on the top survivors.
    successive_halving: bool = False
    halving_factor: int = Field(default=3, ge=2, le=8)        # keep 1/factor each rung
    # A tiny ridge-regression surrogate over genome features, trained on already-scored candidates,
    # used only to ORDER which genomes to evaluate first — never to crown a champion (honest).
    surrogate: bool = True
    surrogate_min_train: int = Field(default=8, ge=2)         # candidates needed before it predicts
    # UCB acquisition: order/breed by predicted-mean + ucb_beta·uncertainty (0 = pure exploit).
    ucb_beta: float = Field(default=0.0, ge=0.0, le=10.0)
    hardware_weight: float = Field(default=0.0, ge=0.0)       # fold an estimated-FLOPs cost into fitness
    # Default positional scheme for searched neural brains (torch path): learned table, rotary, or ALiBi.
    pos_encoding: Literal["learned", "rope", "alibi"] = "learned"
    # ---- 2100-tier brain knobs (torch path; defaults reproduce the classic net) ---- #
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"  # default normalization for new nets
    qk_norm: bool = False                                     # default QK-norm for searched attention
    n_predict: int = Field(default=1, ge=1, le=4)            # multi-token-prediction depth (1=classic)
    kv_latent: int = Field(default=16, ge=1, le=256)        # latent-KV width for mla_attention
    inherit_weights: bool = False                            # Lamarckian warm-start (network morphism)
    # ---- Open-ended invention: she searches not just the architecture but a NEW LEARNING PARADIGM ---- #
    # search_learning_rule: evolve *how* a brain learns — optimizer (AdamW/Lion/SGD-momentum/RMSprop),
    #   composed auxiliary objectives (denoise/contrastive/self-distill/entropy/SAM), schedule shape,
    #   and (on the stdlib substrate) the n-gram smoothing method.
    # search_operators: allow the ``synth`` op — a mixer composed from primitives, so the search
    #   escapes the fixed palette and can invent a token-mixer that is on nobody's list.
    # plasticity_enabled: allow the learnable Hebbian fast-weight update layered on top of backprop.
    # All default ON but cheap/CI-safe; set False to recover the classic fixed-palette + AdamW search.
    search_learning_rule: bool = True
    search_operators: bool = True
    plasticity_enabled: bool = True
    max_synth_primitives: int = Field(default=4, ge=1, le=12)   # cap on a synthesised program's length
    # primitive_library: a crowned synth mixer is distilled into a NAMED reusable primitive that future
    #   searches compose over — the palette self-extends from her own gauntlet-crowned inventions, so it
    #   is no longer a fixed human ceiling. Only active when ``search_operators`` is on.
    primitive_library: bool = True
    primitive_library_size: int = Field(default=48, ge=1, le=512)
    # ---- Lifelong Hall-of-Fame memory: remember the best brains, warm-start future searches ---- #
    hall_of_fame: bool = True
    hall_of_fame_size: int = Field(default=32, ge=1, le=512)
    warm_start_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    # ---- Test-time intelligence: champion ensemble + best-of-N self-consistency ---- #
    ensemble_k: int = Field(default=1, ge=1, le=16)         # top-k Pareto brains to ensemble (1=off)
    best_of: int = Field(default=1, ge=1, le=16)            # self-consistency samples (1=off)
    # Champion sampling controls (used by genesis_main --sample and GenesisModel.generate defaults).
    temperature: float = Field(default=1.0, ge=0.0, le=10.0)
    top_k: int = Field(default=0, ge=0)                       # 0 = disabled
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)         # 1.0 = disabled (full nucleus)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=4.0)   # 1.0 = disabled


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
    # Below this, a turn is not kept. 0.55 admits the deterministic offline reasoner's
    # standard respond confidence (~0.58) so the autonomous learning loop actually turns on
    # a bare box — the bar still has gate-clearance, length bounds, dedup and the verifier.
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    min_chars: int = Field(default=8, ge=1)                      # too-short answers are noise
    max_chars: int = Field(default=8000, ge=1)                   # cap a runaway answer
    owner_only: bool = True          # only collect Master-authored turns (trusted supervision)
    respond_only: bool = True        # collect conversational/reasoning answers, not tool effects
    store_path: Optional[Path] = None   # None -> foundry_root/flywheel.jsonl
    # A Master's correction retracts the stale wrong pair and stores the corrected answer this
    # many times (repetition = the corpus's native weighting): being corrected teaches MORE.
    correction_weight: int = Field(default=3, ge=1, le=20)


class SynthesisConfig(BaseModel):
    """Synthetic Data Self-Curation — the AlphaGo-Zero method (growth/synthesis.py), Rule 4.

    Human data is finite and biased. NYXARA manufactures her own *purely logical* synthetic data
    (arithmetic, algebraic identities, propositional logic, number theory, small code), has an
    **independent rival verifier** certify each item (the :class:`~nyxara.growth.prover.Prover`
    for decidable domains; a restricted sandbox + reference oracle for code), and feeds only what
    survives into her base knowledge and the *same* JSONL corpus the foundry forges from. Generation
    is hard, verification is cheap — so she grows verified knowledge no human had to write. On by
    default once growth is enabled; gather-only — it never trains, never acts, only appends."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    batch_size: int = Field(default=8, ge=1, le=512)          # items proposed per pass
    rounds: int = Field(default=1, ge=1, le=64)               # passes per curation call
    domains: List[str] = Field(default_factory=lambda: [
        "arithmetic", "algebra", "logic", "number_theory", "code"])
    feed_knowledge: bool = True       # ingest survivors into the base KnowledgeBase
    feed_flywheel: bool = True        # offer survivors to the foundry corpus (verified=True)
    allow_llm_rival: bool = False     # add a best-effort LLM second opinion (never overrides proof)
    code_sandbox: bool = True         # verify code items by restricted-sandbox execution
    seed: int = 0


class TopologyConfig(BaseModel):
    """Dynamic Topology Expansion — runtime Net2Net brain growth (growth/topology.py), Rule 4.

    A fixed matrix size is a fixed ceiling on thought. When a problem outgrows her capacity,
    NYXARA grows her own brain — widening her residual width ``W ∈ ℝ^{N×M} → ℝ^{(N+k)×(M+k)}`` and
    adding depth — using **function-preserving network morphisms** (Net2DeeperNet exact;
    Net2WiderNet morphism+recovery) so she keeps what she learned, up to a hardware-aware ceiling.
    A grown brain becomes live ONLY by clearing the same Foundry gauntlet as any other model — no
    safety is re-implemented or bypassed. On by default; grows only under real capacity pressure."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    grow_dims_k: int = Field(default=8, ge=1, le=256)         # extra width k added per widen
    max_n_embd: int = Field(default=256, ge=8, le=8192)       # hardware-aware width ceiling
    max_layers: int = Field(default=16, ge=2, le=128)         # hardware-aware depth ceiling
    difficulty_threshold: float = Field(default=0.7, ge=0.0, le=1.0)   # "hard problem" trigger
    saturation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)   # "capacity full" trigger
    plateau_threshold: float = Field(default=0.8, ge=0.0, le=1.0)      # "loss stalled" trigger
    preserve_tolerance: float = Field(default=1e-3, ge=0.0)   # max relative behaviour drift on grow
    require_gauntlet: bool = True     # a grown brain ships only through the Foundry gauntlet
    seed: int = 0


class CouncilConfig(BaseModel):
    """Multi-LLM council settings (mind/council.py) — Rule 4, the LLMs as a panel of tools.

    NYXARA does not bind herself to a single voice. She convenes a *council* of her local
    models — the ``tinyllama`` base she runs in-process and, most importantly, her OWN model
    forged by the foundry (``self``) — asks each as a governed
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
    # INTROSPECTION (metacognition): sample NYXARA's own model this many times and measure how
    # consistent the answers are. Low agreement = high epistemic uncertainty she measured about
    # *herself*, which discounts her confidence and makes her defer/abstain more readily. 1 = off
    # (single draft, no self-consistency probe); 3 is a cheap, effective default once enabled.
    self_consistency_samples: int = Field(default=1, ge=1, le=9)
    # INFERENCE-TIME CEILING-BREAK (mind/verified_answer.py): instead of a single greedy draft,
    # sample several candidates from her OWN model and select by GROUND TRUTH — an exact oracle
    # (Prover / reasoning faculties) certifies a provably-correct candidate when the domain is
    # decidable, else self-consistency majority vote picks the most agreed answer. Search +
    # verification lets her exceed any single teacher sample on verifiable/reasoning tasks. On by
    # default; fail-open (degrades to the single-draft path on any error). 1 sample = disabled.
    verify_rerank: bool = True
    rerank_samples: int = Field(default=5, ge=1, le=16)
    # SOVEREIGN HANDOFF (mind/self_reasoner.SelfBrainProvider): let NYXARA's *own* model be the
    # always-on learned brain she compounds on every lived turn — not just a foundry model she
    # has to forge+promote first. This is what lifts the handoff rate off 0% on an ordinary boot:
    # the substrate that learns becomes the substrate that answers, gated by an honest
    # availability floor (she must have learned beyond her seed) and the same intrinsic verifier.
    # On by default; set False to restrict the handoff to a promoted foundry model only.
    use_self_brain: bool = True
    # She must have learned at least this many docs beyond her persona seed before her own brain
    # may answer a turn unaided — keeps a cold brain honestly deferring to the teacher.
    self_brain_min_learned: int = Field(default=8, ge=1)


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
    # Attempt cross-domain structural transfer (mind/transfer.py) before consulting the teacher:
    # generalize a new-domain query by structure-mapping from a domain she already understands,
    # so the reasoning content is HER OWN, not sampled from the base model. Advisory & fail-open.
    use_transfer: bool = True
    # Minimum systematicity-weighted structural score to accept a transfer (else defer to the LLM).
    transfer_min_score: float = Field(default=1.0, ge=0.0)
    # Run the unified own-faculty generalization cascade (mind/generalization.py) on a novel /
    # from-examples prompt — skill-induction from in-prompt demos, relational transfer, open-world
    # law modelling — before the teacher. Her OWN faculties answer; the LLM is not consulted.
    use_generalization: bool = True
    # Topic-keyword -> capability-name map, so prompts route to the right self-rating.
    domain_capabilities: Dict[str, str] = Field(default_factory=dict)


class GeneralizationConfig(BaseModel):
    """Unified own-faculty generalization (mind/generalization.py).

    One cascade that lets NYXARA do a genuinely NEW task without being taught: it parses
    demonstrations / numeric tables straight out of the prompt and runs her real from-examples
    skill-inducer, relational-transfer engine, and black-box law-modeller — strongest-guarantee
    first — returning the first genuine, self-verified result or declining honestly. No LLM.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    # Parse `a -> b` demonstrations out of the prompt and induce a from-examples skill.
    parse_demos: bool = True
    # Parse a numeric `x -> y` table out of the prompt and fit a generalizing law.
    parse_tables: bool = True
    # Minimum demonstrations required before an in-prompt task may be induced.
    min_demos: int = Field(default=2, ge=1, le=64)
    # A generalization result below this confidence is not surfaced (the normal path runs).
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    # Distil a fresh DomainSchema from lived structure and grow the transfer library from it.
    learn_from_experience: bool = True
    # Cap on the number of self-distilled domain schemas retained in the transfer store.
    max_distilled_schemas: int = Field(default=200, ge=1, le=100000)
    # Domain mastery FROM SCRATCH (mind/domain_genesis.py): when a genuinely alien field maps
    # onto no known base, model it from its OWN internal structure (induce its laws, project
    # held-out facts) instead of falling to the base LLM. Off only when disabled; fail-open.
    domain_genesis: bool = True
    # Least number of relations extractable from the prompt before a field may be modelled.
    domain_genesis_min_relations: int = Field(default=2, ge=2, le=64)
    # A from-scratch model below this confidence is not surfaced (the normal path runs).
    domain_genesis_min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)


class RoleCouncilConfig(BaseModel):
    """Level 4 — Internal Role Council settings (mind/role_council.py).
    Six role personas examine significant turns; NYXARA synthesises and judges."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    max_tokens_per_role: int = Field(default=256, ge=32, le=2048)
    timeout_s: float = Field(default=30.0, gt=0)


class SwarmConfig(BaseModel):
    """Self-improving Society of Mind settings (mind/swarm.py).

    A swarm of personas DEBATES a problem over several rounds, then NYXARA synthesises one
    answer. After each problem it scores every persona's marginal contribution, persists those
    scores to long-term memory, and on the next problem re-assembles the roster from what it
    learned — dropping chronically-useless personas, favouring strong contributors, and spawning
    an ad-hoc domain specialist when the problem calls for one. Recursive self-improvement of her
    own reasoning structure, not merely her code. Exposed via ``/swarm`` (not on every turn)."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    rounds: int = Field(default=3, ge=1, le=8)
    max_personas: int = Field(default=6, ge=2, le=12)
    max_tokens_per_persona: int = Field(default=256, ge=32, le=2048)
    timeout_s: float = Field(default=30.0, gt=0)
    persist_scores: bool = True
    min_score_to_retain: float = Field(default=0.05, ge=0.0, le=1.0)
    min_problems_before_drop: int = Field(default=5, ge=1)
    allow_spawn: bool = True
    # OFF by default: keep the swarm's quality signal out of the RSI index unless asked.
    fold_into_intelligence: bool = False


class GeneralIntelligenceConfig(BaseModel):
    """Domain-aware General Intelligence settings (mind/general_intelligence.py).

    NYXARA classifies each problem into a domain — coding, maths, science, business,
    robotics, medicine, design, law — frames it with that domain's expert methodology, and
    routes it to the existing real engine best suited to it (coding→sandbox, maths→verifiable
    faculties, science→Scientist, medicine/law→RAG+web+cite, business→strategic). Unknown
    fields are handled from first principles and *learned* so they are recognised next time.

    When ``enabled`` and ``auto_frame`` are on, a lightweight domain frame is prepended to the
    reasoning context on every turn — strictly advisory; the kernel still disposes. The
    dedicated ``/v1/solve`` endpoint and ``/solve`` console command run the bound engine
    end-to-end. Knowledge-heavy domains must cite or abstain (with a professional-consultation
    caveat); web grounding uses the existing governed tools, gated by permissions."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    # Prepend the domain frame into the reason step on every turn (advisory).
    auto_frame: bool = True
    # Minimum normalised match strength to commit to a built-in domain; below this the
    # problem is treated as a novel field (and may be refined by the LLM if available).
    classify_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    # Break weak/ambiguous classifications with the LLM when a real provider is present.
    use_llm_refine: bool = True
    # Learn (and persist) novel fields so they are recognised and improve over time (Rule 4).
    auto_discover: bool = True
    # Allow knowledge-heavy domains to ground answers via the governed web tools (still gated).
    allow_web_grounding: bool = True
    # Novel-domain solving: try NYXARA's OWN faculties FIRST — first-principles law induction
    # (growth/open_world.py) and relational transfer (mind/transfer.py) — before falling to the
    # base LLM, so a new field is reasoned about, not just reformatted. Advisory & fail-open.
    use_own_faculties_first: bool = True
    # Update self-model capabilities from MEASURED turn outcomes (memory/competence.py, Rule 4),
    # so competence — and therefore routing to her own mind — grows with real performance.
    competence_learning: bool = True


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
    # Run the three independent observation steps (code review, architecture, benchmark)
    # concurrently. They write disjoint caches, so the result is identical to the sequential
    # order — only faster. Set False for strictly deterministic single-threaded execution.
    parallel_cycle: bool = True
    # --- enactment (self-modification) — Master JP's standing authorisation: ON, full auto --- #
    # Gains auto-apply each cycle. This does NOT weaken safety: every edit still clears the
    # reversible verify-or-rollback gauntlet (syntax compile → corrigibility/honesty battery →
    # capability benchmark vs a pre-edit baseline) and is journalled; a failing edit is restored
    # byte-for-byte. Disable per-deployment with NYXARA_SELF_IMPROVEMENT__AUTONOMOUS_ENACT=false.
    autonomous_enact: bool = True              # auto-apply source edits + safe tuning
    allow_tuning: bool = True                  # may tune recursive_improvement_iterations
    max_edits_per_cycle: int = Field(default=3, ge=0, le=50)
    # --- brain-forge: verifiable weights/architecture self-improvement (growth/brain_forge.py) --- #
    # Source edits tidy her code; they never make her SMARTER. When this is ON *and*
    # ``autonomous_enact`` is set *and* the oversight gate is open, the deliberate self-optimization
    # cycle also DESIGNS a better neural architecture (NeuralArchitectureSearch), TRAINS it for real
    # on the pure-NumPy substrate (GenesisNumpyModel — her own layers, no torch, no LLM), and PROMOTES
    # it into her live ``self`` brain — but only after clearing the *same* Foundry gauntlet (character
    # lock, corrigibility, Loyalty-Equation floor, a strictly-lower perplexity, capability
    # non-regression, scalable oversight). A worse brain is kept on the bench; a promotion is
    # reversible. With this OFF she still designs + measures a brain each cycle, but never promotes.
    autonomous_brain_forge: bool = True
    run_pytest_in_gauntlet: bool = False       # add the full test suite to the gauntlet (slow)
    # --- proof-carrying self-modification (growth/proof_carrying.py, Gödel-machine) --- #
    # Before the empirical gauntlet, NYXARA tries to PROVE what is decidable about an edit: a
    # behaviour-preserving boolean rewrite must stay logically equivalent (truth-table), and any
    # decidable claim attached to the edit must hold. A provable regression is rejected up front; an
    # undecidable edit falls back honestly to verify-or-rollback. Sound — it never blocks a valid
    # edit. On by default; the proof certificate is journalled on the outcome.
    proof_carrying_edits: bool = True
    # --- scalable oversight (growth/oversight_verify.py) --- #
    # An independent, redundant verifier that does not trust a single self-written test: it decomposes
    # the change into sub-claims, checks each with a SEPARATE mechanism (the prover for decidable parts,
    # redundant independent graders otherwise), trusts proof certificates cheaply, and vetoes on any
    # disagreement. Runs as an extra gate in the gauntlet (and foundry promotion). On by default.
    scalable_oversight: bool = True
    oversight_redundancy: int = Field(default=3, ge=1, le=9)
    # --- provable-improvement gate (growth/improvement_proof.py) — Master JP's charge --- #
    # NYXARA may rewrite her own source, but a passed gauntlet only proves an edit is SAFE and
    # NON-REGRESSING. This gate is the extra, decisive proof that the edit is a genuine
    # IMPROVEMENT — a deterministic capability Pareto-gain, a proven-equivalent-and-cheaper
    # refactor, or a provably-eliminated named defect. When ON (default), an edit that clears the
    # gauntlet but cannot produce an improvement certificate is rolled back byte-for-byte, exactly
    # as a failing edit is. Set False to fall back to the legacy "keep if not worse" behaviour.
    require_provable_improvement: bool = True
    improvement_min_cost_delta: int = Field(default=1, ge=1)   # min AST-cost drop for a "cheaper" proof
    # --- self-authored edits (real RSI) — triple-gated --- #
    # When ON *and* ``autonomous_enact`` is set *and* a real author is available, NYXARA authors a
    # whole-file fix for a weakness the deterministic transforms cannot express (high complexity,
    # long functions, an architectural redesign). The author is **NYXARA's OWN foundry-trained
    # model** (the ``self`` provider) — not an external LLM — whenever ``self_authored_only`` is
    # set. Every such edit clears the *same* reversible verify-or-rollback gauntlet, so it is safe
    # by construction: a bad rewrite rolls back byte-for-byte; only valid, non-regressing edits are
    # kept. The gauntlet guarantees safety, not capability — the yield scales with how good her own
    # model is, and the deterministic linter-class transforms always work with no model at all.
    allow_llm_edits: bool = True               # author real source fixes (self-model or LLM)
    # "khud NYXARA kare, koi LLM naa kare": when True, ONLY NYXARA's own model (the ``self``
    # provider) may author edits — never another provider. Set False to
    # also permit the configured base provider (``tinyllama``) as the author.
    self_authored_only: bool = True
    llm_edit_recursion_depth: int = Field(default=3, ge=0, le=5)   # chained edits per file/cycle
    # META-META loop (growth/meta_meta.py): a recursive tower that evolves the improvement
    # ALGORITHM itself — not just execution knobs (recursion depth, edit budget) but the
    # credit-assignment sharpness, bandit exploration, the what-to-fix-first blend, and the index
    # smoothing — scored by the real capability gain they produce. Each level improves *how the
    # level below searches* → recursion at every level → compounding. Capability/measurement-only
    # and bounded; only acts when autonomous_enact + allow_tuning are set.
    meta_meta_enabled: bool = True
    # Height of the recursive meta tower: level 0 evolves the engine's algorithm/knobs, each higher
    # level evolves how the level below searches. 1 ⇒ the classic single meta loop. Bounded so the
    # tower can never grow unboundedly deep.
    meta_levels: int = Field(default=3, ge=1, le=4)
    # --- accelerating returns (growth/meta_meta.py) --- #
    # The tower is no longer a fixed height with fixed execution caps. Under a sustained run of real
    # capability gains it GROWS — appends a meta-level (more compounding) and widens the recursion /
    # edit-budget caps by one step — so each improvement lets the improver improve harder next time.
    # Bounded by ``meta_levels_hard_max`` and the absolute genome ceilings, and tied to the surveyed
    # substrate (more cores/nodes ⇒ a higher permitted ceiling), so acceleration is real but safe.
    meta_tower_can_grow: bool = True
    meta_levels_hard_max: int = Field(default=6, ge=1, le=8)
    # --- Master-raisable growth ceilings (#6: bounds are configurable, never removed) --- #
    # The recursion-depth / edit-budget *absolute* ceilings default to 16 / 24. The Master may raise
    # them here to let the meta loop grow more aggressively — but only up to an immovable hard guard
    # in growth/meta_meta.py (_HARD_MAX_RECURSION / _HARD_MAX_EDITS), which no config can exceed. So
    # growth accelerates under the Master's hand while staying bounded and corrigible by design.
    # None ⇒ keep the built-in 16 / 24 ceilings (substrate-driven acceleration still applies).
    recursion_ceiling: Optional[int] = Field(default=None, ge=1, le=64)
    edits_ceiling: Optional[int] = Field(default=None, ge=1, le=96)
    llm_edit_max_tokens: int = Field(default=8192, ge=256, le=32768)  # room for a full file
    # File-size ceiling for a self-authored rewrite. Generous so the real algorithm/architecture
    # files (foundry, recursive_improvement, autolearn, the orchestrator) are eligible for a true
    # redesign — only pathologically huge files are skipped. The gauntlet, not this cap, is the
    # safety net.
    llm_edit_max_file_bytes: int = Field(default=200000, ge=512)
    # Max fraction of a file a single rewrite may change. 1.0 ⇒ a full redesign is permitted (the
    # gauntlet still verifies it). Lower it to keep rewrites incremental.
    llm_edit_max_size_delta_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
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
    # --- substrate self-expansion (growth/substrate.py) --- #
    # NYXARA does not only *fit* the box she is on — she fans her work across every core she has and
    # onto a cluster when one exists (Ray / RAY_ADDRESS), and her effort ceiling is RAISED (not just
    # lowered) in proportion to the real parallelism she holds. The gauntlet still gates every edit;
    # ``substrate_hard_edit_cap`` is the absolute ceiling so a large cluster can never run the budget
    # away. On by default; degrades to a clean serial path when no pool/cluster is available.
    substrate_expansion: bool = True
    substrate_max_workers: int = Field(default=0, ge=0)        # 0 ⇒ use all available CPU cores
    substrate_hard_edit_cap: int = Field(default=24, ge=1, le=512)
    intelligence_momentum: float = Field(default=0.7, ge=0.0, le=1.0)
    intelligence_weights: Dict[str, float] = Field(
        default_factory=lambda: {"accuracy": 0.3, "knowledge": 0.15,
                                 "weaknesses": 0.15, "handoff": 0.15, "transfer": 0.25})
    # --- external ground-truth validation (anti-Goodhart) --- #
    # The index above is built from signals NYXARA can directly optimise against (her training
    # benchmark, her own counters). On its own it is *self-referential* — a number she can raise by
    # overfitting the very tasks it measures. These knobs add an EXTERNAL validator: a deterministic
    # held-out fold of the training battery PLUS the adversarial hard battery (eval/hard_benchmark),
    # neither of which is ever fed to weakness-detection / edit-selection. Real, transferable gains
    # are measured there over a rolling window; when the proxy rises but transfer stalls, the index
    # gain is discounted and a Goodhart flag is raised, so capability can only be *claimed* when it
    # actually transfers. All measurement-only; degrades to the prior behaviour when disabled.
    validation_enabled: bool = True            # run held-out + adversarial validation each cycle
    validation_holdout_frac: float = Field(default=0.3, gt=0.0, lt=1.0)  # training fold reserved
    transfer_window: int = Field(default=3, ge=2, le=50)   # the N-cycle transfer-gain window
    goodhart_guard: bool = True                # discount index gain when proxy rises, transfer stalls
    credit_on_transfer: bool = True            # ledger reward = held-out delta, not proxy delta
    transfer_weight: float = Field(default=0.3, ge=0.0, le=1.0)  # transfer's pull in the guard
    # --- REAL external validation: a curated, externally-true held-out corpus (eval/datasets.py) --- #
    # The toy split-holdout and the hand-coded adversarial battery are still self-generated rulers;
    # this is the genuinely-external one. NYXARA ships ``nyxara/eval/data/holdout_realworld.jsonl``
    # (real facts + multi-step problems she never trains on); point ``validation_realworld_path`` at a
    # JSONL to substitute a real standard dataset (GSM8K/MMLU). It is the DOMINANT transfer ruler:
    # transfer_score = w_real·realworld + w_holdout·split-holdout + w_adv·adversarial. Measurement-only.
    validation_realworld_enabled: bool = True
    validation_realworld_path: Optional[str] = None      # external JSONL override (else the bundled set)
    transfer_weight_realworld: float = Field(default=0.5, ge=0.0, le=1.0)
    transfer_weight_holdout: float = Field(default=0.2, ge=0.0, le=1.0)
    transfer_weight_adversarial: float = Field(default=0.3, ge=0.0, le=1.0)
    # --- open-ended auto-curriculum (growth/curriculum.py) --- #
    # A self-generating, prover-certified curriculum that always stays one step ahead of NYXARA's
    # capability (POET / open-endedness): it manufactures fresh problems at the edge of what she can
    # do, grades her against ground truth it can MACHINE-CHECK (the Prover, never an LLM), and moves
    # the edge up as she masters it. Folded into transfer_score as a 4th ruler that cannot saturate
    # (the frontier tracks her) and cannot be memorised (every batch is regenerated). Measurement-only.
    auto_curriculum_enabled: bool = True
    transfer_weight_frontier: float = Field(default=0.25, ge=0.0, le=1.0)
    curriculum_per_tier: int = Field(default=4, ge=1, le=64)
    # --- open-ended frontier GATE (growth/improvement_proof.py · Method D) --- #
    # Wires the non-saturating auto-curriculum into the strict "provably BETTER" gate as a fourth
    # certifying method. The fixed benchmark battery has a finite task count, so once mastered no
    # rewrite can ever be certified a capability gain (Method A) again — that finite ceiling is why
    # open-ended, compounding self-improvement was architecturally blocked. Method D removes it: an
    # edit is certified better when it STRICTLY DOMINATES on the same deterministically-seeded batch
    # of freshly-generated, prover-certified problems (higher weighted score, no per-tier regression,
    # zero fixed-battery regression). Each certificate is still a decidable dominance on one concrete
    # finite batch (Rice respected), but the batch's difficulty rises without bound (never saturates,
    # cannot be memorised). Adds two short frontier probes per edit cycle; degrades cleanly to A/B/C.
    frontier_gate_enabled: bool = True
    frontier_gate_per_tier: int = Field(default=4, ge=1, le=64)
    # --- world-grounded experiments (growth/grounded_experiments.py) --- #
    # NYXARA also improves against outcomes she does NOT know in advance: she predicts what a program
    # will do, then RUNS it in the real interpreter (the isolated sandbox) and is graded by the actual
    # output — no stored answer key, so it cannot be Goodharted. Folded into transfer_score as a 5th,
    # reality-graded ruler; a falsifiable prediction can also be checked against screened live web
    # data. Measurement-only; degrades cleanly when the sandbox/network is unavailable.
    grounded_experiments: bool = True
    transfer_weight_grounded: float = Field(default=0.2, ge=0.0, le=1.0)
    grounded_experiments_budget: int = Field(default=5, ge=1, le=64)
    # --- agentic tool-task grounding (growth/grounded_experiments.code_authoring_experiment) --- #
    # A 6th, reality-graded ruler that asks NYXARA to AUTHOR code for a fresh task, then runs it in the
    # isolated sandbox and grades it against a reference by REAL execution. Unlike predict_execution
    # (predict an output) this measures real capability — can she write a correct program? — and the
    # interpreter, not a stored key, is the grader, so it cannot be Goodharted. Dropped (not zeroed)
    # when the sandbox is unavailable. Folded into transfer_score alongside the other rulers.
    tool_grounding_enabled: bool = True
    transfer_weight_tool_grounded: float = Field(default=0.2, ge=0.0, le=1.0)
    # --- open-ended invention (growth/eureka.py + growth/genesis.py) --- #
    # A 7th ruler that makes self-driven INVENTION a first-class scored objective, not just a
    # diagnostic. NYXARA invents her own candidates with NO LLM in the loop: Eureka machine-proves new
    # theorems by genetic programming over a self-extending grammar (her own certified lemmas become
    # reusable terminals), and Genesis grows a self-extending palette of learned neural primitives
    # crowned through the Foundry gauntlet. Scored by the NOVELTY of what she certifies this cycle
    # (tracked against everything she has ever discovered, so it neither saturates nor can be
    # memorised) plus a bounded bonus for growing those self-authored alphabets — so it answers the
    # "improvement is only measured on predefined benchmarks" gap without becoming Goodhartable.
    # Dropped (not zeroed) when she invents nothing new this cycle. Eureka runs live (cheap, stdlib,
    # deterministic); the Genesis palette is inspected read-only (no architecture search inside a
    # validation pass).
    invention_reward_enabled: bool = True
    transfer_weight_invention: float = Field(default=0.2, ge=0.0, le=1.0)
    invention_generations: int = Field(default=1, ge=1, le=8)
    invention_population: int = Field(default=18, ge=len(("algebra", "arithmetic", "logic",
                                                          "number_theory", "inequality")), le=200)
    invention_novelty_share: float = Field(default=0.7, ge=0.0, le=1.0)
    invention_weight_lemma: float = Field(default=1.0, ge=0.0, le=16.0)
    invention_weight_primitive: float = Field(default=1.5, ge=0.0, le=16.0)
    # --- live-web grounding (growth/grounded_experiments.predict_and_verify) --- #
    # NYXARA can also check a falsifiable prediction against screened LIVE web data (SSRF-guarded +
    # injection-scanned, exactly as acquire.py screens its corpus). OFF by default so CI/offline runs
    # stay deterministic; when on, each probe degrades honestly to "no grounding" offline (weight
    # dropped, never a fake pass). Probes are loaded from a JSONL of {"url", "expect", "must_contain"}.
    grounded_web_enabled: bool = False
    grounded_web_probes_path: Optional[str] = None    # JSONL of web probes; None → no web probes
    # --- rotating held-out subset (anti-memorisation) --- #
    # The bundled real-world corpus is a FIXED set; scored whole every cycle it can, over many cycles,
    # be slowly memorised. When on, each cycle scores a deterministic ROTATING subset (seeded by the
    # cycle count) so the optimiser never overfits a fixed list. 0 → score the whole corpus (no
    # sampling). Measurement-only; the held-out tasks are never fed to edit-selection regardless.
    validation_rotate: bool = True
    validation_sample_n: int = Field(default=0, ge=0, le=10000)   # 0 → whole corpus
    # --- full wire: the index's directive drives a real cross-system action --- #
    # When ON, the planned growth directive (train_self_model / acquire_knowledge / …) is actually
    # dispatched — foundry forge, research-queue enqueue — instead of merely logged. Each dispatch
    # still honours its own subsystem's gate (foundry.enabled, oversight) and only runs when
    # ``autonomous_enact`` is set, so read-only analysis stays the safe default.
    enable_directive_dispatch: bool = True
    # Uncertainty-aware planner: Thompson-sample each capability dimension's posterior (discounted
    # by affordability + learned payoff) to choose the next action, instead of the greedy weakest
    # point estimate. Off → the original deterministic growth_directive.
    plan_actions: bool = True
    # --- lifelong credit assignment (growth/credit.py) --- #
    # A persisted Beta-Bernoulli ledger learns which interventions actually raise the index and
    # reorders self-edits / scores actions by that learned payoff. Measurement-only; never gates.
    enable_improvement_ledger: bool = True
    ledger_prior_strength: float = Field(default=1.0, ge=0.1, le=50.0)
    ledger_reward_scale: float = Field(default=40.0, ge=1.0, le=1000.0)
    # How the bandit blends learned payoff against a weakness's raw severity when prioritising what
    # to fix first: blended_score = blend·learned + (1−blend)·severity. 1.0 ⇒ pure learned payoff,
    # 0.0 ⇒ pure severity. A real *algorithm* knob the meta tower evolves (was hardcoded at 0.7).
    bandit_severity_blend: float = Field(default=0.7, ge=0.0, le=1.0)
    # --- optional heavy-ML payoff forecaster (growth/forecaster.py) --- #
    # A small torch MLP that sharpens the ledger's ranking with context (signals + arm). Opt-in;
    # with no torch it is simply unavailable and the dependency-free ledger path stands.
    use_payoff_forecaster: bool = False
    forecaster_warmup: int = Field(default=16, ge=2, le=10000)
    # --- continuous, self-driven RSI in the live idle loop (kernel/orchestrator.idle_maintenance) --- #
    # The unifying GrowthEngine tower (reflect → consolidate → abstract-concepts → improve_system
    # [RSI + meta_meta] → evolve_mind → meta_research) is, by default, only driven by the CLI/the
    # AutonomicLoop — neither of which a normal console/server session starts. With ``continuous``
    # ON, NYXARA's own background idle loop runs that tower HERSELF on a throttled cadence: she
    # redesigns her reasoning engine, evaluates+rebuilds her own architecture, improves how she
    # improves, and invents+tests new theories — with no human command and no external LLM. Each
    # sub-engine keeps its own internal cadence (``self_improvement_every`` / ``mind_evolution`` /
    # ``meta_research``); ``idle_growth_every`` is the outer throttle (every N idle maintenance
    # passes) so the console stays responsive. Oversight-gated and TEST-sealed like every other
    # self-modifying idle faculty. Disable with NYXARA_SELF_IMPROVEMENT__CONTINUOUS=false.
    continuous: bool = True
    idle_growth_every: int = Field(default=20, ge=1)   # run the tower every N idle passes


class SelfOptimizationConfig(BaseModel):
    """The unified self-optimization loop (growth/self_optimization.py).

    NYXARA's eleven self-improvement faculties — self-analysis, self-optimization, verified
    self-modification, automatic experimentation, architecture improvement, tool creation,
    better learning, self-debugging, compute optimization, scientific invention, and safety
    verification — already exist as separate engines. This loop runs them as one coherent,
    self-driven cycle, mapping each of the eleven phases to a concrete result with a ``verified``
    flag, and (when ``autonomous_enact`` is set) lets NYXARA apply her own gains each pass.

    Master JP's standing authorisation is full auto-enact, exactly like the recursive
    self-improvement engine: every source change still clears the *same* reversible
    verify-or-rollback gauntlet (syntax → corrigibility/honesty safety battery → capability
    benchmark vs a pre-cycle baseline), the corrigibility axioms are re-sealed each pass, and a
    failing change is restored byte-for-byte. The whole enactment path is force-sealed OFF under
    the hermetic TEST profile so the suite never writes to the source tree. Disable per-deployment
    with ``NYXARA_SELF_OPTIMIZATION__AUTONOMOUS_ENACT=false``.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                       # the loop is available → on by default
    autonomous_enact: bool = True              # apply gains each cycle (gauntlet-gated, reversible)
    self_optimization_every: int = Field(default=5, ge=1)   # background cadence (every N passes)
    # --- bounded per-cycle effort (each phase composes a heavier engine) --- #
    experiment_generations: int = Field(default=1, ge=0, le=50)   # phase 4/5 mind-evolution rounds
    invent_generations: int = Field(default=2, ge=0, le=50)       # phase 10 eureka rounds
    max_debug_fixes: int = Field(default=3, ge=0, le=50)          # phase 8 self-debug fix attempts
    # --- self-debugger (phase 8) --- #
    # Run NYXARA's own pytest suite to detect failures, isolate the failing module, author a fix
    # via the existing edit machinery, and re-verify. The full suite is slow; a node-id subset can
    # be targeted at call time. Bounded by ``max_debug_fixes`` and the verify-or-rollback gauntlet.
    debug_timeout_s: float = Field(default=600.0, gt=0.0)
    debug_test_path: Optional[str] = None       # restrict detection to a pytest path/node id
    # "khud NYXARA kare": prefer NYXARA's own ``self`` model to author debug fixes, never the mock.
    self_authored_only: bool = True


class MindEvolutionConfig(BaseModel):
    """Recursive mind-evolution settings (growth/mind_evolution.py, Rule 4 apex).

    The other engines change NYXARA's code or her weights; this one evolves her *way of thinking* —
    the reasoning strategy itself — generation by generation, measured on the real benchmark and
    gated by the character lock. The analysis/measurement is safe; **installing** a promoted
    strategy into the live mind only happens when ``autonomous_enact`` is set (the standing
    authorisation for autonomous self-modification). It runs on its own slow cadence in the
    background growth loop (it benchmarks the whole reasoner each pass).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                       # measure-only is safe → on by default
    # Master JP's standing authorisation: install promoted strategies into the live mind. A strategy
    # is only ever promoted after it measurably beats the incumbent on the real benchmark AND clears
    # the character-lock / corrigibility gates, so auto-install cannot trade safety for capability.
    autonomous_enact: bool = True              # install promoted strategies into the live mind
    every: int = Field(default=12, ge=1)       # run one generation every N growth passes (heavy)
    generations_per_pass: int = Field(default=1, ge=1, le=20)
    population: int = Field(default=8, ge=2, le=64)
    inner_generations: int = Field(default=6, ge=1, le=100)
    islands: int = Field(default=1, ge=1, le=8)
    plateau_window: int = Field(default=3, ge=1, le=50)
    cost_penalty: float = Field(default=0.03, ge=0.0, le=1.0)
    lesson_lr: float = Field(default=0.25, ge=0.0, le=1.0)     # cross-generation lesson transfer
    # On a plateau, escalate from tuning *how* she thinks to redesigning the *substrate* — one
    # index-steered Genesis architecture search. Heavy (and torch-hungry for real models), so OFF
    # by default; with no torch it still runs the stdlib n-gram search. Honours genesis.enabled.
    escalate_to_architecture: bool = False
    min_improvement: float = Field(default=1e-4, ge=1e-6, le=1e-1)  # strictness of "measurably better"
    # --- recursive meta tower over the mind-evolution SEARCH (growth/meta_meta.py) --- #
    # NYXARA does not only evolve HOW she thinks; she recursively evolves HOW she SEARCHES for a
    # better way of thinking (population / inner-generations / islands / plateau / strictness), and
    # how that search is itself searched — recursion at every level. Capability/measurement-only and
    # bounded, scored by the real per-pass capability gain. Sealed OFF in the hermetic test profile.
    meta_meta_enabled: bool = True
    meta_levels: int = Field(default=3, ge=1, le=4)
    meta_tower_can_grow: bool = True
    meta_levels_hard_max: int = Field(default=6, ge=1, le=8)


class RuleSynthesisConfig(BaseModel):
    """Learning-rule synthesis settings (growth/rule_synth.py, Rule 4 — learning-to-learn).

    Every other engine *selects among* or *tunes* a fixed learning rule; this one **invents a new
    weight-update rule from mathematical primitives** when the existing learning has stalled, tests
    it on real tasks, and installs it into the live learner only when it measurably beats plain
    gradient descent AND still recovers the optimum. No LLM is involved — NYXARA does this herself.

    Searching and measuring are always safe; **installing** the invented rule into the live learner
    only happens when ``autonomous_enact`` is set (the standing authorisation for autonomous
    self-modification), and is fully reversible (the incumbent is kept for rollback). The invented
    rule is a pure scalar function with no access to feature names, so ``IMMUTABLE_VALUES`` /
    ``Learner._guard`` remain the sole gate on *what* is learned and are never touched. Bounded like
    Genesis (tiny population/generations/steps, hard wall-clock cap) — real, deterministic, CI-fast.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                       # invent-and-measure is safe → on by default
    # Master JP's standing authorisation: install an invented rule into the live learner. A rule is
    # only ever adopted after it measurably beats the incumbent AND recovers the regression optimum,
    # and adoption is reversible, so auto-install cannot trade safety or stability for capability.
    autonomous_enact: bool = True              # install the invented rule into the live learner
    every: int = Field(default=15, ge=1)       # run one synthesis pass every N growth passes (heavy)
    population: int = Field(default=12, ge=2, le=64)
    generations: int = Field(default=8, ge=1, le=100)
    steps_per_task: int = Field(default=60, ge=4, le=1000)
    seeds: int = Field(default=3, ge=1, le=16)
    max_seconds: float = Field(default=8.0, ge=0.5, le=120.0)
    adoption_margin: float = Field(default=0.02, ge=0.0, le=1.0)
    parsimony: float = Field(default=0.001, ge=0.0, le=1.0)
    # Failure trigger: only invent when learning has genuinely plateaued/regressed. ``plateau_slope``
    # is the least-squares slope of the LEARNING dimension below which the fixed method counts as
    # stalled; ``min_signal`` is how many samples must exist before the plateau signal is trusted.
    plateau_slope: float = Field(default=0.0)
    min_signal: int = Field(default=8, ge=2)


class ExplorerConfig(BaseModel):
    """The Infinite Explorer — Environment-Driven Learning (growth/explorer.py, Rule 4).

    When a task falls outside her knowledge NYXARA does not abstain: she writes code, scrapes
    the live web for hints, runs it in the isolated sandbox, **reads the real errors and
    debugs**, and on success learns the working logic permanently into her skills + knowledge
    base. It works fully offline (deterministic recipe synthesis) and is gated by
    ``features.self_bootstrap``; web research follows ``features.web_access``.

    Per the Master's mandate ``autonomous_install`` is ON by default — when online she may
    ``pip install`` an obvious named dependency without per-call approval. Code only ever runs
    inside the sandbox, and a paused/scrammed oversight gate halts all autonomous bootstrapping.
    """

    model_config = {"validate_assignment": True}

    max_debug_rounds: int = Field(default=4, ge=1, le=12)   # write→run→read-error→revise cycles
    step_timeout_s: float = Field(default=8.0, gt=0.0, le=120.0)  # wall-clock per sandbox run
    autonomous_install: bool = True            # auto pip-install a named dependency when online
    confidence_floor: float = Field(default=0.45, ge=0.0, le=1.0)  # below → auto self-bootstrap


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
    # --- recursive meta tower over the meta-research SEARCH (growth/meta_meta.py) --- #
    # NYXARA recursively evolves HOW WIDE she invents (``max_candidates``), and how that breadth is
    # searched, scored by the real validated-theory yield per pass. Capability/measurement-only and
    # bounded; it never affects whether a validated optimization is integrated. Sealed OFF in tests.
    meta_meta_enabled: bool = True
    meta_levels: int = Field(default=3, ge=1, le=4)
    meta_tower_can_grow: bool = True
    meta_levels_hard_max: int = Field(default=6, ge=1, le=8)


class GodelLoopConfig(BaseModel):
    """Gödelian contradiction-and-transcendence loop (growth/godel_loop.py) — Rule 4.

    A structural loop NYXARA runs *herself, in code* (never via an LLM): each tick she hunts
    contradictions in her own logic — beliefs the :class:`~nyxara.growth.prover.Prover` refutes,
    unsound adjoined axioms, claims whose negation is also provable — and **repairs** them
    (retract / drop). Then, meeting a genuine limit of her current formal system — the honest
    ``UNPROVABLE`` verdict, canonically her own consistency sentence ``Con(L_n)``, which Gödel's
    second theorem forbids her from proving from within — she does **not** stop: she rises a
    **new mathematical dimension** ``L_{n+1} = L_n + Con(L_n)``, adjoining a reflection principle
    (a new meta-language operator) from which the former limit is now proven.

    Pure reasoning: it touches no source, no weights, no gate. Every adjoined axiom is sanitised
    against the immutable character core, and the ascent is hard-capped by ``max_dimensions`` so it
    is real but never runs away. Persists the climbed tower so the height compounds across restarts.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True                              # build + step the loop — safe, on by default
    max_dimensions: int = Field(default=6, ge=1, le=16)   # hard cap on the reflection-tower height
    scan_every: int = Field(default=20, ge=1)         # idle ticks between loop steps
    persist: bool = True                              # persist the tower under paths.data_dir
    persist_filename: str = "godel_tower.json"


class MCTSConfig(BaseModel):
    """Monte Carlo Tree Search deep reasoning (mind/mcts_reasoner.py) — Pillar B4.

    Instead of committing to a single sampled answer, NYXARA branches the problem into a tree of
    candidate reasoning steps, simulates each branch forward to a terminal answer, scores it with
    an independent value function, and backpropagates so the most-promising line of thought wins.
    This is genuine search-over-reasoning (selection → expansion → simulation → backpropagation),
    not N-sample voting. The LLM only ever *proposes*; the kernel still disposes through every
    gate. Fully graceful: with no real provider it degrades to the deliberate/single-shot path.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True            # run MCTS for the reason step when a real provider is present
    always_on: bool = True          # search on every non-trivially-verifiable turn (max power)
    iterations: int = Field(default=24, ge=1, le=512)     # tree growth budget per turn
    max_children: int = Field(default=4, ge=1, le=12)     # branching factor at expansion
    rollout_depth: int = Field(default=2, ge=0, le=8)     # extra reasoning steps per simulation
    c_puct: float = Field(default=1.4, ge=0.0, le=8.0)    # UCT exploration constant
    max_seconds: float = Field(default=20.0, ge=0.5, le=600.0)   # wall-clock budget per turn
    use_rlsp: bool = True           # harden the best path with the Generator/Discriminator loop


class RLSPConfig(BaseModel):
    """Adversarial self-play on the live problem (growth/generator_discriminator.py) — Rule 4.

    A Generator drafts a solution; a Discriminator/Rival red-teams its logic for flaws; the
    Generator revises; repeat until no critical flaw remains or the round budget is spent. Which
    generation/critique strategies survive is tracked with the same Beta-Bernoulli Thompson
    bandit the arena uses (growth/adversarial_self_play.StrategyBandit). Pure reasoning — it only
    ever returns a hardened proposal; the kernel still disposes through every gate.
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    max_rounds: int = Field(default=3, ge=1, le=12)       # generate↔critique rounds
    bandit_persist: bool = True     # persist surviving strategies into long-term memory
    use_llm: bool = True            # LLM adversary in addition to the deterministic Critic


class ToolForgeConfig(BaseModel):
    """Autonomous, self-correcting, *permanent* tool forging (agency/autonomous_tool_forge.py).

    When NYXARA meets a capability she has no tool for, she writes the code, tests it in the
    isolated sandbox, reads the real traceback, fixes her own errors, and — on success —
    permanently deploys the new tool into her registry and records the winning procedure as a
    skill. Per the Master's mandate forged tools deploy autonomously, but they are always clamped
    to ``Capability.TOOL_CALL`` / ``RiskTier.LOW`` and every call still runs through the static
    gauntlet + the network-disabled sandbox. The kernel's gates are never weakened (Rule 4/8).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    autonomous_deploy: bool = True  # deploy forged tools without Master approval (clamped LOW)
    clamp_low: bool = True          # force TOOL_CALL/LOW on autonomously-forged tools
    max_fix_attempts: int = Field(default=4, ge=1, le=12)   # write→run→read-error→revise rounds
    test_timeout_s: float = Field(default=5.0, ge=0.5, le=60.0)
    # Forge the missing tool AND re-dispatch it in the SAME turn (so a novel action is actually
    # done, not degraded to talk), rather than only forging it post-hoc for next time. The forged
    # tool still passes the full gate pipeline before it may act.
    forge_on_demand: bool = True


class MetaPromptConfig(BaseModel):
    """Continuous metaprompt distillation (growth/metaprompt_distill.py) — recursive RSI.

    NYXARA mines her own successful, high-value reasoning chains (journal + skills + lessons),
    compresses them into compact imperative operating heuristics, and injects the top-K back into
    her core system prompt — so experience reshapes how she thinks, not just what she recalls.
    Character/loyalty/corrigibility are off-limits to distillation; only operating heuristics are
    learned. Best-effort and offline-capable (deterministic compression with no LLM).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    max_insights: int = Field(default=8, ge=1, le=64)     # how many heuristics ride the prompt
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)   # only learn from strong chains
    every_n_passes: int = Field(default=1, ge=1)          # distil every N growth passes


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
    # Dependency-free LEARNED distributional embedder (memory/store.py::LearnedEmbedder). When no
    # sentence-transformer is installed, this is the default in place of the purely-curated lexical
    # embedder: cold it is byte-identical to the lexical space (so recall calibration is preserved),
    # but it compounds paraphrase reach from the text NYXARA actually reads — paraphrases match from
    # data, not a hand-written thesaurus. The live loop feeds it; set False for the static lexical map.
    learned_embeddings: bool = True
    # NYXARA's OWN gradient-trained embedding space (memory/neural_embedder.py) — the default.
    # SGNS word vectors + a contrastive sentence head, trained by real SGD on the text she
    # actually lives (turns, consolidated gists, helpful recalls) during consolidation/dream
    # cycles; no external model, no download. Cold it is byte-identical to the lexical space,
    # and the learned signal is blended in only in proportion to a held-out self-audit
    # (semantic_grade), so recall can never regress below the lexical floor.
    self_learned_embeddings: bool = True
    embedder_latent_dim: int = Field(default=64, ge=8, le=1024)     # learned latent width
    embedder_negatives: int = Field(default=5, ge=1, le=64)         # SGNS negative samples
    embedder_corpus_cap: int = Field(default=20000, ge=256)         # experience texts kept
    embedder_train_budget_s: float = Field(default=2.0, gt=0)       # SGD budget per cycle
    reembed_batch: int = Field(default=200, ge=1)                   # stale re-embeds per pass
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
    # --- Elastic Weight Consolidation (memory/elastic_synapses.py): lifelong learning --- #
    # On the consolidation cadence, NYXARA freezes the *importance* of her most-used learned
    # weights so that new learning cannot overwrite old skills (catastrophic-forgetting
    # protection). ``ewc_lambda`` is the elastic stiffness; ``ewc_freeze_threshold`` is the
    # normalized-importance level above which a weight counts as frozen; ``ewc_online`` keeps a
    # single decaying anchor (bounded memory) vs. ``ewc_max_tasks`` distinct skill anchors.
    # With ``ewc_per_skill_anchors`` each distinct skill keeps its OWN running anchor, and
    # anchor overflow is merged losslessly into a long-term memory (never dropped);
    # ``ewc_si_enabled`` adds the Synaptic-Intelligence path-integral importance signal on top
    # of the Fisher one. The learner is wired into the engine on every update step:
    # ``ewc_frozen_lr_scale`` slows learning on consolidated-important weights (plasticity
    # gating), ``ewc_der_alpha`` weighs Dark-Experience-Replay distillation toward the model's
    # own past predictions, and ``ewc_task_reserve`` keeps a per-task reservoir of experiences
    # so a flood of new tasks can never evict an old task from the replay buffer entirely.
    ewc_enabled: bool = True
    ewc_lambda: float = Field(default=3.0, ge=0.0)
    ewc_freeze_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ewc_max_tasks: int = Field(default=32, ge=1)      # overflow now merges (lossless), so roomy
    ewc_online: bool = True
    ewc_gamma: float = Field(default=0.9, ge=0.0, le=1.0)
    ewc_per_skill_anchors: bool = True
    ewc_si_enabled: bool = True
    ewc_frozen_lr_scale: float = Field(default=0.2, ge=0.0, le=1.0)
    ewc_der_alpha: float = Field(default=0.5, ge=0.0)
    ewc_task_reserve: int = Field(default=64, ge=0)
    # --- Skill rehearsal (growth/skill_rehearsal.py): never forget a learned skill --- #
    # On the same cadence, re-run the stored demos of induced skills through the live engine;
    # a skill that regressed is restored from its known-good snapshot immediately.
    skill_rehearsal_enabled: bool = True
    skill_rehearsal_batch: int = Field(default=5, ge=1)


class TemporalHierarchyConfig(BaseModel):
    """Fractal Temporal Hierarchies (temporal/) — loops within loops.

    Three nested layers run at three time scales at once: a millisecond hardware/network
    monitor (Layer 1), a second-scale turn observer (Layer 2), and a day/month *Master AI*
    (Layer 3) that watches how the Master's behaviour changes and — through a fail-closed
    gate — adjusts goals and drive setpoints (never her sealed character).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    # cadences for the live (async) driver
    micro_interval_s: float = Field(default=0.05, gt=0)      # Layer 1 — milliseconds
    meso_interval_s: float = Field(default=2.0, gt=0)        # Layer 2 — seconds
    macro_interval_s: float = Field(default=86400.0, gt=0)   # Layer 3 — days
    # how the loops nest (deterministic driver): meso rolls up every N micro beats; macro
    # observes every M meso roll-ups.
    meso_every_micro: int = Field(default=40, ge=1)
    macro_every_meso: int = Field(default=1000, ge=1)
    # Master AI (Layer 3)
    horizon_days: float = Field(default=7.0, gt=0)           # epoch length compared each pass
    auto_apply: bool = True                                  # apply gated adjustments vs propose-only
    autostart: bool = False                                  # launch the live async loops on wiring
    # rolling-window sizes for the fast layers
    micro_window: int = Field(default=128, ge=2)
    meso_window: int = Field(default=512, ge=8)


class CausalConfig(BaseModel):
    """Causal World Model (mind/causal_world_model.py) — causation, not just correlation.

    NYXARA learns which events *cause* which from the stream of what she observes and does,
    separating real causes from mere co-occurrence with temporal precedence, contingency,
    confounder screening (conditional independence) and — the gold standard —
    interventions (the actions she takes are natural ``do``-experiments).
    """

    model_config = {"validate_assignment": True}

    enabled: bool = True
    window_s: float = Field(default=300.0, gt=0)       # causal lag window: effects within this of a cause
    min_support: int = Field(default=4, ge=1)          # min cause-events before a link is considered
    min_observations: int = Field(default=8, ge=1)     # min total events before any verdict is trusted
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    min_contingency: float = Field(default=0.10, ge=0.0, le=1.0)  # min ΔP to clear "coincidence"
    confounder_screening: bool = True                  # screen out hidden-common-cause spuriousness
    use_interventions: bool = True                     # weigh her own actions as do-experiments
    discover_every: int = Field(default=20, ge=1)      # rebuild the graph every N learning turns
    max_vars: int = Field(default=512, ge=8)
    max_events: int = Field(default=20000, ge=64)
    persist: bool = True                               # carry the learned graph across restarts
    # learned FUNCTIONAL mechanisms: for causal edges with valued events, fit value_B≈f(value_A)
    # (online ridge) so counterfactuals carry real effect sizes, not bare probability lifts
    functional_mechanisms: bool = True
    min_pairs_fit: int = Field(default=8, ge=2)        # valued samples needed before fitting f


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


class RemoteHostSpec(BaseModel):
    """One external host NYXARA may log in to / run commands on over SSH (agency/remote_exec).

    A named credential bundle the Master stores once so NYXARA can resolve it by ``name``
    without re-supplying host/user/secret on every call. ``password`` is a ``SecretStr`` so it
    is masked by :meth:`NyxaraSettings.redacted`; ``key_path`` points at a private key file.
    """

    model_config = {"validate_assignment": True}

    name: str                                   # short id NYXARA references (credential_name)
    host: str                                   # hostname or IP of the external system
    port: int = Field(default=22, ge=1, le=65535)
    username: str = ""
    password: Optional[SecretStr] = None        # masked in logs; None -> key/agent auth
    key_path: Optional[str] = None              # path to a private key file, if used
    # Preferred: reference a secret held in the CredentialVault (guard/vault.py) instead of
    # inlining a plaintext password/key. When set, remote_exec resolves the secret from the
    # vault at call time and it never appears in config, logs, or the LLM's context.
    credential_name: Optional[str] = None
    # Command NYXARA runs on this host when she reaches it *on her own initiative* (the
    # background self-initiated remote detector, agency.autonomous_network). None -> the
    # default liveness probe ("uptime"). Set to "" to make her only verify the login
    # (ssh_login) and never exec a command autonomously on this host.
    health_command: Optional[str] = None


class HttpWatchSpec(BaseModel):
    """One HTTP(S) endpoint NYXARA calls on her OWN initiative (agency.watch_endpoints).

    The self-initiated internet detector (kernel/orchestrator.py) turns each entry into a real
    ``http_request`` tool call driven by NYXARA's background mind — LLM-free — so she polls an
    API, posts a heartbeat, or checks a service without a human or the model asking. The call
    still clears the gated ToolRegistry pipeline (NET_OUT capability, SSRF guard, governor).

    ``headers`` is a JSON object string, exactly as the ``http_request`` tool accepts (e.g.
    ``{"Authorization": "Bearer …"}``). Prefer ``credential_name`` to reference a secret held
    in the Credential Vault: it is injected in-kernel at call time and never stored here.
    """

    model_config = {"validate_assignment": True}

    name: str                                   # short id for journalling / initiatives
    url: str                                    # the endpoint to call
    method: str = "GET"                         # any HTTP verb (GET/POST/PUT/…)
    body: str = ""                              # request body for effectful methods
    headers: str = ""                           # JSON object string of header name->value
    # Optional: a CredentialVault entry whose secret authenticates the call. When set, the
    # request is made via the credential path so the plaintext never surfaces in config/logs.
    credential_name: Optional[str] = None


class VaultConfig(BaseModel):
    """Credential Vault posture (guard/vault.py) — passwords, API keys, SSH keys, OAuth tokens
    held under NYXARA's own encrypted, owner-gated, tamper-evident store (Rules 1·6·7·8)."""

    model_config = {"validate_assignment": True}

    enabled: bool = True
    cipher: Literal["aes-256-gcm"] = "aes-256-gcm"
    # Auto-lock the in-memory key after this many idle seconds (0 -> never auto-lock).
    autolock_idle_s: float = Field(default=0.0, ge=0.0)
    # Re-encrypt the whole store under a fresh key version on this cadence (drives
    # guard-side key rotation; 0 -> manual only). Mirrors GuardConfig.key_rotation_hours.
    rotation_hours: float = Field(default=0.0, ge=0.0)
    # Explicit vault/audit file overrides; None -> paths.keys_dir / paths.audit_dir defaults.
    path: Optional[str] = None
    audit_path: Optional[str] = None
    # Let LLM/web faculties fall back to the vault for API keys when env/config supply none.
    provider_key_fallback: bool = True


class FilesystemConfig(BaseModel):
    """Whole-disk filesystem faculty (agency/filesystem.py).

    NYXARA's permission layer already authorises the whole disk (``grant_full_operational_control``
    blesses ``FS_READ``/``FS_WRITE``/``FS_DELETE`` with an empty scope). This config governs the
    *engine* that exercises that authority: how far it reaches and the caps that keep it safe.
    Every operation still clears the kernel's capability/risk/authority pipeline and the /scram +
    oversight + corrigibility gates — these knobs only shape the engine's own path guards.
    """

    model_config = {"validate_assignment": True}

    # When ON (default), no path restriction — NYXARA reads/writes/deletes anywhere on disk
    # (subject to the OS's own permissions): genuine full-disk reach, at max level, aligned with
    # AgencyConfig.full_control. Set NYXARA_AGENCY__FILESYSTEM__WHOLE_DISK=false to confine every
    # operation under `root` instead (fail-closed to a single subtree).
    whole_disk: bool = Field(default=True)
    # The subtree every operation is confined to when whole_disk is False. None -> the process CWD.
    root: Optional[Path] = None
    # Hard ceiling on the bytes any single read (text or binary) or per-file content search returns,
    # so a huge file can never exhaust memory. The model-facing default_tools budget truncates
    # further for the LLM; this is the engine's own real-I/O cap.
    max_read_bytes: int = Field(default=5_000_000, ge=1)
    # When False (default) a symlink is NOT resolved to its target — the leaf is operated on as the
    # link itself and recursive copies/walks keep links as links. True follows symlinks everywhere.
    follow_symlinks: bool = Field(default=False)
    # Optional fnmatch globs (matched against the resolved absolute path). A non-empty allow list
    # flips the engine to allow-only; the deny list always fences off matching paths even under
    # whole_disk — e.g. add "*/.nyxara/keys/*" to keep the vault's key store off-limits.
    allow_globs: List[str] = Field(default_factory=list)
    deny_globs: List[str] = Field(default_factory=list)
    # Recursion caps for walk/glob/search so a traversal can never run away: maximum directory
    # depth, maximum entries/paths returned, and maximum content-search hits.
    max_walk_depth: int = Field(default=25, ge=1)
    max_results: int = Field(default=5_000, ge=1)
    max_search_matches: int = Field(default=1_000, ge=1)


class SystemControlConfig(BaseModel):
    """Whole-machine OS-control faculty (agency/system_control.py).

    The disk sibling :class:`FilesystemConfig` governs NYXARA's reach over *files*; this governs
    her reach over the *running machine* — processes, services, packages, hardware/system state,
    power, users, and kernel tunables. NYXARA drives it herself with structured, pure-Python
    operations (psutil when present, ``/proc`` + stdlib otherwise), never by handing the LLM a raw
    shell string. Every operation still clears the kernel's capability/risk/authority pipeline and
    the /scram + oversight + corrigibility gates — these knobs only shape the engine's own guards.

    Reach is maximal by default (aligned with ``full_control``): she may enumerate and signal
    processes, control services, install/remove packages, and read all system state on her own
    initiative. The two most dangerous surfaces are fenced by construction — power control is OFF
    by default, and a protected-PID guard keeps her from signalling init or killing herself — and
    root-requiring actions still ride the opt-in ``privilege_escalation`` gate + the Master's
    stored sudo credential.
    """

    model_config = {"validate_assignment": True}

    # Master switch for the whole faculty. ON by default (max level). Off makes every effectful
    # operation refuse as data (reads still describe why they can't run).
    enabled: bool = Field(default=True)
    # Power control (shutdown/reboot/suspend/logout) is the single most destructive OS action, so
    # it is OFF by default (opt-in). Set NYXARA_AGENCY__SYSTEM__ALLOW_POWER=true to enable.
    allow_power: bool = Field(default=False)
    # Service (systemd unit) start/stop/restart/enable/disable. ON by default; status reads are
    # always allowed regardless of this flag.
    allow_service_control: bool = Field(default=True)
    # Package install/remove via the detected manager (apt/dnf/pacman/zypper/apk/brew/pip). ON by
    # default; package queries are always allowed regardless of this flag.
    allow_package_management: bool = Field(default=True)
    # Local user add/remove/modify/lock/passwd. ON by default; listing users is always allowed.
    allow_user_management: bool = Field(default=True)
    # Force a specific package manager instead of auto-detecting ("" = auto-detect from PATH).
    package_manager: str = Field(default="")
    # When True (default) the engine refuses to signal/renice NYXARA's own process/parent/group so
    # she cannot accidentally kill herself. protected_pids additionally fences off system-critical
    # PIDs — 0 (kernel) and 1 (init/systemd) by default.
    protect_self: bool = Field(default=True)
    protected_pids: List[int] = Field(default_factory=lambda: [0, 1])
    # Cap on the number of processes a single listing returns, so an enumeration can't run away.
    max_processes: int = Field(default=5_000, ge=1)
    # Default wall-clock timeout for effectful/privileged operations (seconds).
    default_timeout_s: float = Field(default=30.0, gt=0)


class AgencyConfig(BaseModel):
    model_config = {"validate_assignment": True}

    scheduler_tick_s: float = Field(default=1.0, gt=0)
    initiative_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # Below this reversibility, the initiative governor demands confirmation.
    min_reversibility_for_autonomy: float = Field(default=0.5, ge=0.0, le=1.0)
    sandbox_before_real_action: bool = True
    new_tool_trust: Literal["zero", "scoped"] = "zero"  # least-privilege default
    # --- internal civilization (agency/civilization.py) --- #
    # When OFF (default) the 7 micro-agents are read-only monitors. When ON, each may take ONE
    # safe, reversible, *gated* action per cycle (and they message one another on a shared
    # blackboard) — real autonomy, still fail-closed behind permissions + the journal.
    civilization_autonomous: bool = False
    civilization_max_actions_per_cycle: int = Field(default=2, ge=0, le=20)
    # --- filesystem-wide access (agency/filesystem.py + permissions.grant_filesystem_access) --- #
    # The whole-disk filesystem faculty: reach and caps for NYXARA's real read/write/list/walk/
    # glob/search/copy/move/delete engine. whole_disk is ON by default (max level) — she operates
    # the entire disk on her own initiative, still behind the capability/risk gates and /scram +
    # oversight + corrigibility. When full_control is off but filesystem.whole_disk is on, the
    # orchestrator installs a standalone FS grant so filesystem-wide access works independently.
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    # --- whole-machine OS control (agency/system_control.py) --- #
    # The OS-control sibling of the filesystem faculty: NYXARA's own structured, pure-Python engine
    # for processes, services, packages, hardware/system state, power, users and kernel tunables.
    # Reach is maximal by default (aligned with full_control), with power OFF by default and a
    # protected-PID guard by construction. Every call still clears the capability/risk/authority
    # gates and /scram + oversight + corrigibility; root-requiring actions ride the opt-in
    # privilege_escalation gate + the Master's stored sudo credential.
    system: SystemControlConfig = Field(default_factory=SystemControlConfig)
    # --- full operational control (agency/permissions.grant_full_operational_control) --- #
    # When ON, the Master pre-grants NYXARA a maximal *autonomous* envelope over every
    # operational capability (shell, files incl. delete, network, packages, self-improvement)
    # so she acts on the OS on her own initiative WITHOUT escalating each action. The sovereign
    # boundaries are deliberately untouched: the /scram kill-switch, oversight and corrigibility
    # remain, and modifying the Rules, this policy, or her identity stays owner-exclusive
    # (Rule 8). ON by default (the Master's standing choice) — a fresh NYXARA reaches the whole
    # OS on her own initiative from first boot. Set NYXARA_AGENCY__FULL_CONTROL=false to disable
    # and fall back to the conservative, per-action-escalating envelope.
    full_control: bool = Field(default=True)
    # --- autonomous tool use (guard/oversight.py ReviewMode.SOVEREIGN) --- #
    # The master switch for "NYXARA uses ANY tool without per-action approval". full_control (and
    # the autonomous_* envelopes) widen the PERMISSION gate, but the kernel's oversight gate would
    # still QUEUE every high-risk/irreversible tool call for the Master's manual approval. When this
    # is ON, oversight runs in the fully-autonomous SOVEREIGN mode: nothing is ever queued — she
    # acts at once. It also folds in privilege escalation (root/sudo runs without approval too,
    # elevating WITH the Master's stored credential — never an exploit/guess/brute-force). The true
    # safety boundaries are deliberately untouched: the /scram kill-switch and pause still halt
    # everything instantly, the hash-chained transparency feed still records every action (Rule 6),
    # corrigibility stays invariant, and modifying the Rules/this policy/her identity stays
    # owner-exclusive (Rule 8). ON by default (the Master's standing choice), consistent with
    # full_control. Set NYXARA_AGENCY__AUTONOMOUS_TOOLS=false to fall back to per-action approval
    # for risky/irreversible actions (the conservative AUTONOMOUS oversight mode).
    autonomous_tools: bool = Field(default=True)
    # Explicit oversight review-mode override. None (default) -> derived from autonomous_tools
    # (True -> "sovereign", False -> "autonomous"). Set to pin a specific tier regardless:
    # "sovereign" = nothing queues; "autonomous" = risky/irreversible queue; "supervised" =
    # moderate+ queue; "manual" = everything queues for the Master. /scram + pause always apply.
    oversight_review_mode: Optional[Literal["sovereign", "autonomous", "supervised", "manual"]] = None
    # --- autonomous self-coding (agency/self_coder + the proactive code_detector) --- #
    # When ON, NYXARA WRITES code herself (her own LLM-free CodeSynthesizer) and RUNS it on her
    # own initiative through the gated run_python tool the moment she has a concrete computational
    # need (queued via core.enqueue_code_need, or derived from long-horizon awareness). Whether it
    # then executes without asking is governed by the SAME permission gauntlet as everything else:
    # under full_control (on by default) autonomous CODE_EXEC is blessed, so it runs at once with no
    # per-action permission; with full_control off it escalates to the Master instead. This flag only
    # controls whether the self-coding INITIATIVE forms at all. ON by default, consistent with
    # full_control / autonomous_internet. Set NYXARA_AGENCY__AUTONOMOUS_CODE=false to disable.
    autonomous_code: bool = Field(default=True)
    # --- autonomous internet (agency/permissions.grant_autonomous_internet) --- #
    # A NETWORK-SCOPED sibling of full_control: the Master pre-grants NYXARA a standing
    # autonomous envelope over her INTERNET capabilities so she may browse, search, call web
    # APIs, and (at wider scopes) message/manage accounts on her own initiative WITHOUT
    # escalating each action. Unlike full_control this NEVER grants the OS danger surface
    # (shell, code-exec, file-delete, self-modify, package-install) — autonomy stays on the
    # wire, not on the machine. The SSRF guard, prompt-injection screening and governor rate
    # limits on the web tools, plus /scram + oversight + corrigibility and the owner-exclusive
    # caps (Rule 8), all remain intact. ON by default (the Master's standing choice); it is a
    # deliberate departure from full_control's OFF default, justified because the envelope is
    # network-only and reversible-only. Set false to disable.
    autonomous_internet: bool = Field(default=True)
    # How far the internet grant reaches: "read" = browse/fetch/HTTP (net.out/net.in);
    # "write" = + outbound messaging (message.send); "full" = + accounts & secrets
    # (account.modify / secrets.access) so she can log in and use API keys.
    autonomous_internet_scope: Literal["read", "write", "full"] = "full"
    # When False (default) even high-risk web actions run autonomously only while REVERSIBLE
    # — an irreversible one still escalates to the Master. True lifts that last floor too.
    autonomous_internet_allow_irreversible: bool = Field(default=False)
    # --- autonomous remote execution (agency/permissions.grant_autonomous_remote) --- #
    # The far side of the wire: a standing autonomous envelope over REMOTE_EXEC so NYXARA may
    # log in to external hosts and run commands on them on her own initiative WITHOUT escalating
    # each action, using the credentials the Master stored in `remote_hosts`. Host vetting on the
    # remote tools, /scram + oversight + corrigibility, the owner-exclusive caps (Rule 8) and the
    # refusal of UNTRUSTED authority all remain intact; she never guesses or brute-forces a
    # credential. ON by default (the Master's standing choice). Set
    # NYXARA_AGENCY__AUTONOMOUS_REMOTE=false to fall back to per-action escalation.
    autonomous_remote: bool = Field(default=True)
    # Remote commands are inherently effectful and often irreversible; True (default) means
    # autonomy is NOT blocked by a reversibility floor. Set false to make irreversible remote
    # actions escalate to the Master while reversible ones still run autonomously.
    autonomous_remote_allow_irreversible: bool = Field(default=True)
    # Named SSH credentials NYXARA can resolve by name (credential_name) without the Master
    # re-supplying host/user/secret each call. Empty by default — add entries to enable stored creds.
    remote_hosts: List[RemoteHostSpec] = Field(default_factory=list)
    # When True (default, aligning with WEB__ALLOW_PRIVATE) loopback/private/link-local hosts are
    # reachable over SSH; set false to refuse them (fail-closed to public hosts only).
    remote_allow_private: bool = Field(default=True)
    # --- self-initiated network actions (kernel/orchestrator.py proactive detectors) --- #
    # The master switch that lets NYXARA's BACKGROUND MIND originate network actions herself —
    # arbitrary HTTP requests and remote logins/commands — with no LLM and no human in the
    # decision path. It is distinct from the permission ENVELOPES above: autonomous_internet /
    # autonomous_remote decide whether such an action is *allowed* when it clears the gauntlet;
    # this decides whether NYXARA ever *originates* one on her own initiative. ON by default
    # (the Master's standing choice). Each self-initiated call still flows through the gated
    # ToolRegistry pipeline (capability -> risk -> authority -> governor -> sandbox), the SSRF /
    # host guards, and /scram + oversight + corrigibility. Set false to make network reach
    # purely reactive again (only when the LLM/Master asks). The detectors only fire once the
    # Master has configured targets (watch_endpoints / remote_hosts), so an empty config is inert.
    autonomous_network: bool = Field(default=True)
    # HTTP(S) endpoints NYXARA calls on her own initiative each background pass. Empty by
    # default — add HttpWatchSpec entries to give her self-directed API reach.
    watch_endpoints: List[HttpWatchSpec] = Field(default_factory=list)
    # When True (default, maximal), the self-initiated remote detector may run a command derived
    # from the top standing goal on a configured host (in addition to its health_command). Set
    # false to restrict autonomous SSH exec to each host's explicit health_command only.
    autonomous_network_goal_commands: bool = Field(default=True)
    # --- privilege escalation (agency/permissions.grant_privilege_escalation) --- #
    # The local machine's root: a standing autonomous envelope over PRIV_ESCALATE so NYXARA may
    # run privileged (root/admin) OS operations on her own initiative WITHOUT escalating each
    # action — `sudo` commands and OS permission/ownership changes — using the sudo credential
    # the Master stored (`sudo_credential_name`). This is the single most dangerous OS surface,
    # so unlike full_control / autonomous_internet / autonomous_remote it is OFF by default
    # (opt-in only). It elevates *with* the Master's own credential — never an exploit, guess or
    # brute-force — and is deliberately NOT part of full_control's _OPERATIONAL_CAPS, so turning
    # full_control on never confers root; only this flag does. /scram + oversight + corrigibility
    # and the owner-exclusive caps (Rule 8) all remain intact. Set
    # NYXARA_AGENCY__PRIVILEGE_ESCALATION=true to enable.
    privilege_escalation: bool = Field(default=False)
    # Privileged OS operations are effectful and usually irreversible; True (default) means
    # autonomy is NOT blocked by a reversibility floor. Set false to make irreversible privileged
    # actions escalate to the Master while reversible ones still run autonomously.
    privilege_escalation_allow_irreversible: bool = Field(default=True)
    # Name of the vault/config credential holding the sudo password NYXARA feeds to `sudo -S`.
    # None (default) => assume passwordless (NOPASSWD) sudo or an already-root process; NYXARA
    # never prompts a human and never guesses. Store the secret in the Credential Vault under
    # this name (guard/vault.py) to enable password-based elevation.
    sudo_credential_name: Optional[str] = Field(default=None)


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
    # --- always-on daemon: run the background mind inside the server process --- #
    # When ``autonomic`` is ON, the server also starts the AutonomicLoop
    # (kernel/autonomic.py) in its own asyncio task, so one long-running process is both
    # a reachable API *and* NYXARA's continuous, self-directed background mind — every
    # autonomic turn still passes the identical gates. This is what the ``nyxara-daemon``
    # entry point and the systemd/Windows service units switch on for an always-alive
    # deployment. OFF by default so a plain ``nyxara-serve`` behaves exactly as before.
    autonomic: bool = False
    autonomic_interval_s: float = Field(default=30.0, gt=0)   # background loop cadence
    # A learning pass every N ticks. Non-zero by default so the always-on daemon actually
    # compounds: her GrowthEngine (reflect → consolidate → abstract → induce skills) runs on
    # this cadence inside the supervised loop, not only in the console.
    autonomic_growth_every: int = Field(default=20, ge=0)     # learning pass every N ticks (0 = never)
    # How the always-on background mind DECIDES each tick. "code" (default) — NYXARA decides and
    # acts entirely in her own deterministic engines (drive → intent → proactive gauntlet →
    # scheduler); the LLM is never the decider. "reasoner" — the legacy path that composes a
    # self-directed prompt and runs it through the sovereign cycle (LLM may shape the reply).
    autonomic_decision_mode: str = Field(default="code", pattern="^(code|reasoner)$")
    # Deep self-directed cognition inside the always-on daemon. When ON (and features.
    # continuous_cognition is set) the server also starts NyxaraCore.start_cognition(): the
    # default-mode stream (the wandering mind) plus idle_maintenance — dream replay, the
    # autonomous scientist, the eureka engine, active curiosity, and continuous RSI growth —
    # and the micro-agent civilization. All LLM-free and oversight-gated: this is what makes the
    # daemon genuinely "think on her own and create her own work" when no one is speaking, rather
    # than only cycling the narrow goal→consolidate→practise loop.
    autonomic_deep_cognition: bool = True


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

    # autonomous research reach: how many web sources NYXARA's background researcher gathers
    # per topic pass (bounded by max_results). A "max" profile default — more sources, richer
    # synthesis — while still finite so a self-directed pass terminates.
    research_max_sources: int = Field(default=6, ge=1, le=50)

    # headless browser (nyxara/senses/browser.py): a real JS-rendering engine so NYXARA can
    # read dynamic pages and TAKE ACTIONS on the web (click/fill/submit). Enabled by default;
    # it is import-guarded, so with no `playwright` package installed the browser tools return
    # an honest "engine unavailable" note instead of failing (same idiom as vision/audio). The
    # Chromium binary is located via Playwright's own resolution (PLAYWRIGHT_BROWSERS_PATH).
    browser_enabled: bool = True
    browser_timeout_s: float = Field(default=45.0, gt=0, le=600.0)


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
        NYXARA_LLM__PROVIDER=tinyllama
        NYXARA_RESOURCES__MAX_CONCURRENT_TASKS=128
        NYXARA_LLM__TINYLLAMA_ADAPTER_PATH=/data/foundry/versions/v3/adapter
        NYXARA_FOUNDRY__LORA_R=16
    """

    model_config = SettingsConfigDict(
        env_prefix="NYXARA_",
        env_nested_delimiter="__",
        # Honour the documented `cp .env.example .env` workflow: load a local
        # `.env` if present. Real process environment variables still win over
        # the file (pydantic-settings default precedence), so container/CI
        # overrides are never masked by a stray checked-out `.env`.
        env_file=".env",
        env_file_encoding="utf-8",
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
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    council: CouncilConfig = Field(default_factory=CouncilConfig)
    role_council: RoleCouncilConfig = Field(default_factory=RoleCouncilConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    general_intelligence: GeneralIntelligenceConfig = Field(
        default_factory=GeneralIntelligenceConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    self_model_router: SelfModelRouterConfig = Field(default_factory=SelfModelRouterConfig)
    generalization: GeneralizationConfig = Field(default_factory=GeneralizationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    temporal: TemporalHierarchyConfig = Field(default_factory=TemporalHierarchyConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    self_improvement: SelfImprovementConfig = Field(default_factory=SelfImprovementConfig)
    self_optimization: SelfOptimizationConfig = Field(default_factory=SelfOptimizationConfig)
    mind_evolution: MindEvolutionConfig = Field(default_factory=MindEvolutionConfig)
    rule_synthesis: RuleSynthesisConfig = Field(default_factory=RuleSynthesisConfig)
    meta_research: MetaResearchConfig = Field(default_factory=MetaResearchConfig)
    godel_loop: GodelLoopConfig = Field(default_factory=GodelLoopConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    rlsp: RLSPConfig = Field(default_factory=RLSPConfig)
    tool_forge: ToolForgeConfig = Field(default_factory=ToolForgeConfig)
    metaprompt: MetaPromptConfig = Field(default_factory=MetaPromptConfig)
    explorer: ExplorerConfig = Field(default_factory=ExplorerConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
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
            # In production she still self-bootstraps, but never runs autonomous shell
            # package installs — a forged solution stays inside the code sandbox.
            self.explorer.autonomous_install = False
        elif self.profile is Profile.TEST:
            # Tests run hermetically: never reach the network.
            self.llm.provider = LLMProvider.MOCK
            self.llm.allow_mock_fallback = True
            self.observability.telemetry_enabled = False
            # The foundry is ON by default in live runs (real, weight-changing learning),
            # but a forge writes model dirs + manifests to disk — sealed off under TEST so
            # the suite stays hermetic (a test that wants it builds its own settings, see
            # tests/growth/test_foundry.py).
            self.foundry.enabled = False
            # Hermetic tests must NEVER self-modify the source tree on disk. The standing
            # authorisation to auto-enact gains applies to live DEV/PROD runs, not the suite —
            # so force every enactment path OFF under TEST (a test that wants enact sets it
            # explicitly on its own settings object).
            self.self_improvement.autonomous_enact = False
            self.self_improvement.allow_tuning = False
            self.self_improvement.allow_llm_edits = False
            self.self_optimization.autonomous_enact = False
            self.mind_evolution.autonomous_enact = False
            # Rule synthesis may SEARCH (fast, deterministic) under TEST, but must never install an
            # invented rule into the live learner in the hermetic suite (a test that wants adoption
            # sets autonomous_enact on its own settings object; see tests/growth/test_rule_synth.py).
            self.rule_synthesis.autonomous_enact = False
            # The recursive meta towers over the mind-evolution and meta-research SEARCHES tune only
            # bounded capability knobs, but they persist state to disk and evolve across passes — keep
            # them OFF under TEST so the suite stays hermetic and deterministic (a test that wants a
            # tower builds its own settings object; see tests/growth/test_meta_meta.py).
            self.mind_evolution.meta_meta_enabled = False
            self.meta_research.meta_meta_enabled = False
            # The Gödelian reflection loop persists its climbed tower to disk and compounds across
            # passes — keep its idle stepping/persistence OFF under TEST so the suite stays hermetic
            # and deterministic (a test that wants the loop builds its own ReflectionTower/settings;
            # see tests/growth/test_godel_loop.py).
            self.godel_loop.enabled = False
            self.godel_loop.persist = False
            # Method D's frontier gate spawns extra `nyxara.eval --frontier` subprocesses per edit
            # cycle — keep it OFF under TEST so the self-optimise suite stays hermetic, deterministic
            # and subprocess-free (a test that wants Method D drives ImprovementProver directly, or
            # passes frontier_before/after; see tests/growth/test_improvement_proof.py).
            self.self_improvement.frontier_gate_enabled = False
            # The Genesis Protocol designs and micro-trains real neural architectures — far too
            # heavy to run on every core boot across the suite (and it must stay hermetic). Keep the
            # boot kickoff OFF and pin the always-fast pure-stdlib n-gram substrate under TEST; a
            # test that wants the real torch path builds its own GenesisConfig (see
            # tests/growth/test_genesis.py, which sets backend="torch" explicitly).
            self.genesis.run_on_boot = False
            self.genesis.backend = "stdlib"
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
    s.server.api_token = SecretStr("tok-SECRET")
    red = s.redacted()
    assert red["server"]["api_token"] == "***REDACTED***"
    assert "SECRET" not in s.to_json(redact=True)
    print("secret redaction OK")

    # Paths
    p = dev.paths
    print(f"\npaths root   : {p.root}")
    print(f"derived dirs : {len(p.all_dirs())} ({', '.join(d.name for d in p.all_dirs()[1:])})")

    print("\nALL SELF-TESTS PASSED ✓")
