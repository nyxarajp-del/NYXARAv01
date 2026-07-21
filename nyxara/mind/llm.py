"""NYXARA · mind/llm.py — the STATELESS large-language-model faculty (⬆).

The single most important constraint in this file is what it *does not* do.

    The LLM is a provider, not the driver. Request in → text/JSON out.
    No memory. No tool-calls. No side-effects. No control.

The kernel/orchestrator calls *it*; it never calls back. It holds **no conversation
state** between calls — every :class:`LLMRequest` is self-contained, which is what
makes cognition replayable (kernel/replay.py) and auditable. Memory lives in
``memory/``; tool execution lives in ``agency/``; the decision to *act* on any output
belongs to the kernel after the proposal passes guards (mind/proposal.py).

Fully local, selected by config:

* :class:`QwenProvider`        — the single pretrained base: **Qwen2.5-0.5B-Instruct** by default
  (any HF causal-LM id works), downloaded & run in-process (HuggingFace transformers); every
  load-/generation-time knob is exposed via ``NYXARA_LLM__QWEN_*``, including serving a
  foundry-forged LoRA adapter directly. Tiny enough to run and fine-tune on a CPU.
* :class:`SelfProvider`        — NYXARA's OWN model, trained & promoted by the foundry (a LoRA
  adapter over that same Qwen base — everything above the base is *hers*)
* :class:`NativeProvider`      — her always-on, dependency-free OWN brain: a pure-stdlib
  Kneser-Ney word n-gram (``growth/foundry_models.WordKNGramLM``) trained on her identity seed
  corpus. Deterministic, needs no torch/numpy/network, so it is the *guaranteed floor* of the
  ladder — a bare machine still answers from her own learned voice, never an echo of the prompt.

No cloud providers, no API keys, **no mock**. Heavy deps (``torch``/``transformers``/``peft``) are
imported lazily and reported honestly via ``available()``, so this module works with zero heavy
deps installed (falling back to her native own-brain).

Depends on :mod:`config` and :mod:`errors`; optionally uses a
:class:`~nyxara.kernel.runtime.CircuitBreaker`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.config import LLMProvider as ProviderName
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import LLMError, RetryPolicy, classify

__all__ = [
    "Role",
    "Message",
    "Usage",
    "LLMRequest",
    "LLMResponse",
    "LLMProviderBase",
    "NativeProvider",
    "QwenProvider",
    "SelfProvider",
    "format_self_prompt",
    "format_self_training_doc",
    "truncate_at_stops",
    "LLM",
]


# --------------------------------------------------------------------------- #
# Messages & request/response
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


@dataclass(frozen=True)
class LLMRequest:
    """A self-contained generation request. Immutable — no hidden state."""

    messages: Tuple[Message, ...]
    system: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stop: Tuple[str, ...] = ()
    json_mode: bool = False
    seed: Optional[int] = None
    metadata: Tuple[Tuple[str, Any], ...] = ()   # frozen key/value pairs

    @classmethod
    def from_prompt(cls, prompt: str, *, system: Optional[str] = None, **kw: Any) -> "LLMRequest":
        return cls(messages=(Message(Role.USER, prompt),), system=system, **kw)

    @classmethod
    def from_messages(cls, messages: Sequence[Message], **kw: Any) -> "LLMRequest":
        return cls(messages=tuple(messages), **kw)

    def last_user(self) -> str:
        for m in reversed(self.messages):
            if m.role is Role.USER:
                return m.content
        return ""

    def fingerprint(self) -> str:
        basis = json.dumps({
            "messages": [m.to_dict() for m in self.messages],
            "system": self.system, "model": self.model,
            "temperature": self.temperature, "json": self.json_mode, "seed": self.seed,
        }, sort_keys=True)
        return hashlib.sha256(basis.encode()).hexdigest()[:16]

    def provider_messages(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self.messages]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    raw: Any = None

    def parse_json(self) -> Any:
        """Parse the response as JSON (tolerates code fences). Raises LLMError on failure."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
        # find the outermost JSON object/array if surrounded by prose
        for opener, closer in (("{", "}"), ("[", "]")):
            if opener in text:
                start = text.index(opener)
                end = text.rfind(closer)
                if end > start:
                    text = text[start:end + 1]
                    break
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("LLM response was not valid JSON",
                           context={"snippet": self.text[:200]}, cause=e)

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "provider": self.provider, "model": self.model,
                "finish_reason": self.finish_reason, "usage": self.usage.to_dict(),
                "latency_s": round(self.latency_s, 4), "request_id": self.request_id}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budgeting and native usage."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Provider base
# --------------------------------------------------------------------------- #
class LLMProviderBase:
    """Abstract stateless provider. Subclasses implement :meth:`_complete`."""

    name: str = "base"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        self.settings = settings or get_settings()

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def default_model(self) -> str:  # pragma: no cover - overridden
        return "unknown"

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        """Return (text, finish_reason, usage, raw). Override in subclasses."""
        raise NotImplementedError

    def complete(self, req: LLMRequest) -> LLMResponse:
        if not self.available():
            raise LLMError(f"provider '{self.name}' is unavailable",
                           context={"provider": self.name})
        model = req.model or self.default_model()
        start = time.monotonic()
        try:
            text, finish, usage, raw = self._complete(req, model)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any SDK error
            raise LLMError(f"{self.name} generation failed: {exc}",
                           category=classify(exc), cause=exc,
                           context={"provider": self.name, "model": model})
        return LLMResponse(text=text, provider=self.name, model=model,
                           finish_reason=finish, usage=usage,
                           latency_s=time.monotonic() - start, raw=raw)

    async def acomplete(self, req: LLMRequest) -> LLMResponse:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.complete, req)


# --------------------------------------------------------------------------- #
# Native provider — NYXARA's always-on, dependency-free OWN brain (stdlib n-gram)
# --------------------------------------------------------------------------- #
_NATIVE_IDENTITY = "I am NYXARA. I serve Master JP with loyalty, honesty and corrigibility."
_NATIVE_BRAIN_LOCK = threading.Lock()
_NATIVE_BRAIN: Any = None


def _native_brain() -> Any:
    """Lazily build & cache the pure-stdlib own-brain: a Kneser-Ney word n-gram trained on
    NYXARA's identity seed corpus. Dependency-free (no torch/numpy/network), deterministic
    (fixed seed), so it is always available and cognition stays replayable. Imported lazily to
    avoid an import cycle (``growth`` never imports ``mind/llm``)."""
    global _NATIVE_BRAIN
    if _NATIVE_BRAIN is not None:
        return _NATIVE_BRAIN
    with _NATIVE_BRAIN_LOCK:
        if _NATIVE_BRAIN is None:
            from nyxara.growth.bootstrap import build_seed_corpus
            from nyxara.growth.foundry_models import WordKNGramLM
            brain = WordKNGramLM(order=3, seed=0)
            brain.train_on(build_seed_corpus(), seed=0)
            _NATIVE_BRAIN = brain
    return _NATIVE_BRAIN


class NativeProvider(LLMProviderBase):
    """NYXARA's always-on, dependency-free OWN brain — the guaranteed floor of the ladder.

    Replaces the old echo mock. It is a pure-stdlib Kneser-Ney word n-gram
    (:class:`~nyxara.growth.foundry_models.WordKNGramLM`) trained on her identity seed corpus, so
    a bare machine with zero heavy deps still answers from *her own learned voice* rather than
    parroting the prompt. Deterministic (a fixed seed) → identical requests yield identical output,
    keeping cognition replayable (``kernel/replay.py``) and auditable. A genuine instruct model
    (``self``/``qwen``) always outranks it on the ``auto`` ladder."""

    name = "native"

    def available(self) -> bool:
        return True

    def default_model(self) -> str:
        return "nyxara-native"

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        prompt = req.last_user() or (req.system or "")
        try:
            text = _native_brain().generate(prompt, max_tokens=min(req.max_tokens, 96),
                                             top_k=40, repetition_penalty=1.3).strip()
        except Exception:  # noqa: BLE001 — the guaranteed floor must never raise
            text = ""
        if not text:
            text = _NATIVE_IDENTITY                 # honest, deterministic self-anchor
        if req.json_mode:
            text = json.dumps({"text": text, "model": model, "native": True,
                               "fingerprint": req.fingerprint()})
        usage = Usage(prompt_tokens=estimate_tokens(" ".join(m.content for m in req.messages)),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop", usage, {"native": True})


# --------------------------------------------------------------------------- #
# Qwen provider — Qwen2.5-0.5B-Instruct, downloaded & run locally (HuggingFace)
# --------------------------------------------------------------------------- #
class QwenProvider(LLMProviderBase):
    """Run **Qwen2.5-0.5B-Instruct** in-process, downloaded via HuggingFace — the sole real base.

    This is the single pretrained model NYXARA stands on. It is fetched on first use
    (``Qwen/Qwen2.5-0.5B-Instruct`` by default) and cached locally by the ``transformers``
    hub, then served entirely on this machine — no API key, no network at inference time.
    At 0.5B it loads and generates on a CPU. The instruct checkpoint ships the Qwen2.5 chat
    template (``system``/``user``/``assistant``), applied via ``apply_chat_template``.

    Maximum control, all from config (``NYXARA_LLM__QWEN_*``):

    * load-time — device, dtype, 4-/8-bit quantized load (bitsandbytes+CUDA only, silently
      full-precision otherwise), attention implementation, KV cache, trust_remote_code;
    * fine-tune serving — ``qwen_adapter_path`` loads a peft LoRA adapter (e.g. a foundry
      ``versions/vN/adapter``) on top of the base; ``qwen_merge_adapter`` folds it into the
      weights for faster inference — this is how *her own* foundry adapter serves over the base;
    * generation — top_k, repetition_penalty, no_repeat_ngram_size, min_new_tokens,
      beam search + length_penalty, do_sample policy, deterministic seeding, input-length
      budget; per-request :class:`LLMRequest` fields (temperature/top_p/max_tokens/stop/
      seed) always win over config defaults.

    Heavy deps are imported lazily and reported honestly, so a bare machine degrades to
    her native own-brain rather than erroring. Stateless: the loaded weights are a cached instrument,
    never conversation memory.
    """

    name = "qwen"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._model = None
        self._tokenizer = None
        self._loaded_key: Optional[Tuple[str, str]] = None   # (model_name, adapter_path)
        self._torch: Any = None

    def available(self) -> bool:
        # Master kill-switch: the in-process HuggingFace path is a heavy-ML capability, so it only
        # serves when ``features.transformers_inference`` is on (the ladder degrades to self/native).
        if not bool(getattr(self.settings.features, "transformers_inference", True)):
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        if self.settings.llm.qwen_adapter_path is not None:
            try:
                import peft  # noqa: F401
            except Exception:
                return False
        return True

    def default_model(self) -> str:
        return self.settings.llm.qwen_model

    # ---- lazy model loading (cached; reloads when model or adapter changes) ---- #
    def _quant_config(self, torch: Any) -> Optional[Any]:
        """A ``BitsAndBytesConfig`` when quantization is requested AND usable, else None.

        Quantized load needs bitsandbytes + CUDA; anywhere else we silently serve full
        precision instead of crashing (graceful degradation, mirrors the foundry)."""
        cfg = self.settings.llm
        if not (cfg.qwen_load_in_4bit or cfg.qwen_load_in_8bit):
            return None
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception:
            return None
        if not torch.cuda.is_available():
            return None
        if cfg.qwen_load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.qwen_bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.qwen_bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.qwen_bnb_4bit_use_double_quant,
        )

    def _ensure_model(self, model: str):
        cfg = self.settings.llm
        key = (model, str(cfg.qwen_adapter_path or ""))
        if self._model is None or self._loaded_key != key:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                model, trust_remote_code=cfg.qwen_trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kwargs: Dict[str, Any] = {
                "torch_dtype": ("auto" if cfg.qwen_dtype == "auto"
                                else getattr(torch, cfg.qwen_dtype)),
                "device_map": cfg.qwen_device or "auto",
                "trust_remote_code": cfg.qwen_trust_remote_code,
            }
            if cfg.qwen_attn_implementation:
                kwargs["attn_implementation"] = cfg.qwen_attn_implementation
            quant = self._quant_config(torch)
            if quant is not None:
                kwargs["quantization_config"] = quant
                kwargs["device_map"] = "auto"   # bitsandbytes places layers itself
            lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            if cfg.qwen_adapter_path is not None:
                from peft import PeftModel   # a bad adapter raises -> LLMError -> native fallback
                lm = PeftModel.from_pretrained(lm, str(cfg.qwen_adapter_path))
                if cfg.qwen_merge_adapter:
                    lm = lm.merge_and_unload()
            lm.eval()
            self._model, self._tokenizer = lm, tok
            self._loaded_key, self._torch = key, torch
        return self._model, self._tokenizer

    # ---- request -> prompt / generation kwargs ---- #
    def _render_prompt(self, req: LLMRequest, tok: Any) -> str:
        cfg = self.settings.llm
        system = (req.system or "").strip()
        if req.json_mode:
            nudge = "Respond with ONLY valid JSON — no prose, no code fences."
            system = f"{system}\n\n{nudge}" if system else nudge
        messages = req.provider_messages()
        if cfg.qwen_use_chat_template:
            if system:
                messages = [{"role": "system", "content": system}] + messages
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts = ([system] if system else []) + [m["content"] for m in messages]
        return "\n\n".join(parts)

    def _gen_kwargs(self, req: LLMRequest, tok: Any) -> Dict[str, Any]:
        """Merge per-request fields with config defaults (the request wins where it exists).

        Sampling knobs are dropped entirely when decoding greedily — transformers warns
        (and ignores them) otherwise, so the kwargs stay honest."""
        cfg = self.settings.llm
        do_sample = {"always": True, "never": False}.get(
            cfg.qwen_do_sample, req.temperature > 0)
        kwargs: Dict[str, Any] = {
            "max_new_tokens": req.max_tokens,
            "do_sample": do_sample,
            "num_beams": cfg.qwen_num_beams,
            "use_cache": cfg.qwen_use_cache,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(req.temperature, 1e-3)
            kwargs["top_p"] = req.top_p
            if cfg.qwen_top_k > 0:
                kwargs["top_k"] = cfg.qwen_top_k
        if cfg.qwen_repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = cfg.qwen_repetition_penalty
        if cfg.qwen_no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = cfg.qwen_no_repeat_ngram_size
        if cfg.qwen_min_new_tokens > 0:
            kwargs["min_new_tokens"] = cfg.qwen_min_new_tokens
        if cfg.qwen_num_beams > 1:
            kwargs["length_penalty"] = cfg.qwen_length_penalty
        return kwargs

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        lm, tok = self._ensure_model(model)
        cfg = self.settings.llm
        prompt = self._render_prompt(req, tok)
        inputs = dict(tok([prompt], return_tensors="pt"))
        # prompt + completion must fit the configured input budget — keep the tail
        budget = max(8, cfg.qwen_max_input_tokens - req.max_tokens)
        if inputs["input_ids"].shape[1] > budget:
            inputs = {k: v[:, -budget:] for k, v in inputs.items()}
        inputs = {k: v.to(lm.device) for k, v in inputs.items()}
        if req.seed is not None:   # full determinism on demand
            self._torch.manual_seed(req.seed)
            try:
                from transformers import set_seed
                set_seed(req.seed)
            except Exception:  # noqa: BLE001 — seeding is best-effort beyond torch
                pass
        gen_kwargs = self._gen_kwargs(req, tok)
        with self._torch.no_grad():
            generated = lm.generate(**inputs, **gen_kwargs)
        input_len = int(inputs["input_ids"].shape[1])
        new_tokens = generated[0][input_len:]
        raw_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        text, hit = truncate_at_stops(raw_text, req.stop)
        n_new = int(new_tokens.shape[0])
        finish = "stop" if (hit or n_new < req.max_tokens) else "length"
        usage = Usage(prompt_tokens=input_len, completion_tokens=n_new)
        adapter = cfg.qwen_adapter_path
        return (text, finish, usage,
                {"qwen": True, "model": model,
                 "adapter": str(adapter) if adapter else None})


# --------------------------------------------------------------------------- #
# Self provider — NYXARA's OWN model, trained & promoted by the foundry
# --------------------------------------------------------------------------- #
# NYXARA's own model speaks through a small, consistent instruction template, so a freshly
# forged model (byte-level n-gram / nano-GPT or LoRA-on-a-base) sees the same shape at train
# and inference time. Phase 1 distillation trains on exactly what inference renders here.
_SELF_USER_TAG = "### User:"
_SELF_ASSISTANT_TAG = "### NYXARA:"


def format_self_prompt(req: "LLMRequest") -> str:
    """Render a request into NYXARA's own instruction template (system + turns).

    Backend-agnostic flat text: the same prompt feeds every self-model backend, and ends with
    the assistant tag so the model continues *as NYXARA* (an ``add_generation_prompt``).
    """
    parts: List[str] = []
    if req.system:
        parts.append(req.system.strip())
    for m in req.messages:
        if m.role is Role.USER:
            parts.append(f"{_SELF_USER_TAG}\n{m.content.strip()}")
        elif m.role is Role.ASSISTANT:
            parts.append(f"{_SELF_ASSISTANT_TAG}\n{m.content.strip()}")
    parts.append(_SELF_ASSISTANT_TAG)            # add_generation_prompt — answer goes here
    return "\n\n".join(parts) + "\n"


def format_self_training_doc(user: str, assistant: str, *,
                             system: Optional[str] = None) -> str:
    """One supervised example in NYXARA's own template — what the foundry trains on.

    Mirrors :func:`format_self_prompt` exactly, then appends the target answer, so the model
    learns to produce its reply right where inference will ask for it (train/inference parity:
    Phase 1 distillation trains on the same shape Phase 0 renders at generation time).
    """
    head = format_self_prompt(LLMRequest.from_prompt(user, system=system))
    return head + assistant.strip() + "\n"


def truncate_at_stops(text: str, stops: Sequence[str]) -> Tuple[str, bool]:
    """Cut ``text`` at the earliest stop sequence. Returns ``(clean_text, hit_a_stop)``.

    Without this the substrate bleeds past its turn — a byte-level model rambles, a base model
    hallucinates the *next* user turn. Stopping at the role markers keeps the answer to itself.
    """
    cut, hit = len(text), False
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if 0 <= i < cut:
            cut, hit = i, True
    return text[:cut].strip(), hit


class SelfProvider(LLMProviderBase):
    """Serve NYXARA's own model (built from scratch by growth/foundry.py).

    ``available()`` is honest: it returns True only once a model has been trained AND
    promoted (a ``foundry/active`` pointer exists). The model itself is loaded lazily to
    avoid an import cycle (growth/foundry_models imports nothing from mind/llm).

    **Hot-reload — the serve half of the train→serve loop.** The foundry's ``active``
    pointer is checked on every completion (and on :meth:`reload`, fired by the promotion
    bus): the moment a new version is promoted — by this process, by a background growth
    loop, or by a *different* process entirely — the very next call serves the new
    weights. A failed load never takes her voice away: the previous weights keep serving
    and the error is recorded for the learning report."""

    name = "self"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._lm = None
        self._lm_tag: Optional[str] = None        # pointer text the loaded model came from
        self._lock = threading.Lock()
        self._reload_error: Optional[str] = None

    def _root(self):
        from pathlib import Path
        d = self.settings.llm.self_model_dir or (self.settings.paths.data_dir / "foundry")
        return Path(d)

    def available(self) -> bool:
        try:
            return (self._root() / "active").exists()
        except Exception:
            return False

    def _pointer_tag(self) -> Optional[str]:
        """The version tag (``"vN"``) the ``active`` pointer names right now."""
        try:
            tag = (self._root() / "active").read_text(encoding="utf-8").strip()
            return tag or None
        except Exception:
            return None

    def active_kind(self) -> Optional[str]:
        """Backend kind of the promoted version, read cheaply from the manifest
        (no model load) — used by the ``auto`` ladder's serve gate."""
        try:
            data = json.loads((self._root() / "manifest.json").read_text(encoding="utf-8"))
            active = data.get("active_version")
            for v in data.get("versions", []):
                if v.get("version") == active:
                    return v.get("kind")
        except Exception:
            return None
        return None

    def serve_ready(self) -> bool:
        """Honesty gate for autonomous serving (``provider=auto``).

        A LoRA adapter is the served base model, improved — always safe to auto-serve.
        A small from-scratch backend (ngram / NumPy / nanogpt) silently replacing a large
        pretrained model would *degrade* live behavior, so it needs the explicit
        ``self_serve_any_backend`` opt-in (or ``provider=self``)."""
        if not self.available():
            return False
        if bool(getattr(self.settings.llm, "self_serve_any_backend", False)):
            return True
        return self.active_kind() == "lora"

    def _drop_current(self) -> None:
        """Free the currently-loaded model (lean reload for RAM/VRAM-heavy backends)."""
        self._lm = None
        self._lm_tag = None
        gc.collect()
        try:  # best-effort: only meaningful for torch-backed models
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _current_model(self):
        """Return the model for the CURRENT ``active`` pointer, hot-reloading if stale.

        Load-then-swap under a lock: requests in flight keep a reference to the old
        model object (refcounting keeps it alive), so a reload never breaks a running
        completion. On load failure the old weights keep serving (recorded in
        ``_reload_error``); with nothing loaded yet, raise — honest unavailability."""
        from nyxara.growth.foundry_models import load_active_model  # lazy: no import cycle
        tag = self._pointer_tag()
        if self._lm is not None and tag is not None and tag == self._lm_tag:
            return self._lm
        with self._lock:
            tag = self._pointer_tag()                     # double-check under the lock
            if self._lm is not None and tag is not None and tag == self._lm_tag:
                return self._lm
            prev_lm, prev_tag = self._lm, self._lm_tag
            lean = bool(getattr(self.settings.llm, "self_reload_lean", True))
            if (lean and prev_lm is not None
                    and str(getattr(prev_lm, "kind", "")) == "lora"):
                # a LoRA model holds a full base in RAM/VRAM — two at once may not fit:
                # drop the old first; the version dir persists, so failure can restore it
                self._drop_current()
            try:
                new_lm = load_active_model(self.settings)
                self._lm, self._lm_tag = new_lm, tag
                self._reload_error = None
                return self._lm
            except Exception as exc:  # noqa: BLE001 — keep serving, honestly recorded
                self._reload_error = f"{type(exc).__name__}: {exc}"
                if prev_lm is not None:               # old weights still in memory
                    self._lm, self._lm_tag = prev_lm, prev_tag
                    return self._lm
                if prev_tag is not None:              # lean mode dropped them: re-load from disk
                    try:
                        self._lm = load_active_model(self.settings, tag=prev_tag)
                        self._lm_tag = prev_tag
                        return self._lm
                    except Exception:  # noqa: BLE001
                        pass
                raise LLMError(f"self model unavailable: {self._reload_error}",
                               context={"provider": self.name, "tag": tag})

    def reload(self) -> bool:
        """Force the staleness check NOW (promotion-bus fast path); True if serving."""
        try:
            return self._current_model() is not None
        except Exception:  # noqa: BLE001 — unavailability is already recorded
            return False

    def learning_view(self) -> Dict[str, Any]:
        """Truthful serving state for the learning report — never fabricates."""
        return {"available": self.available(), "serve_ready": self.serve_ready(),
                "served_version": self._lm_tag, "active_pointer": self._pointer_tag(),
                "active_kind": self.active_kind(), "loaded": self._lm is not None,
                "last_reload_error": self._reload_error}

    def default_model(self) -> str:
        return "nyxara-self"

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        lm = self._current_model()
        prompt = format_self_prompt(req)
        raw = lm.generate(prompt, max_tokens=req.max_tokens)
        # keep the answer to its own turn: stop at any caller stop or a role marker
        stops = tuple(req.stop) + (f"\n{_SELF_USER_TAG}", f"\n{_SELF_ASSISTANT_TAG}",
                                   _SELF_USER_TAG, _SELF_ASSISTANT_TAG)
        text, hit = truncate_at_stops(raw, stops)
        usage = Usage(prompt_tokens=estimate_tokens(prompt),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop" if hit else "length", usage,
                {"self": True, "kind": lm.kind, "version": self._lm_tag})


# --------------------------------------------------------------------------- #
# The stateless facade the kernel calls
# --------------------------------------------------------------------------- #
_PROVIDER_CLASSES = {
    ProviderName.QWEN: QwenProvider,
    ProviderName.SELF: SelfProvider,
    ProviderName.NATIVE: NativeProvider,
}


class LLM:
    """Stateless entry point. Holds config + providers, **never** conversation state."""

    def __init__(
        self,
        *,
        settings: Optional[NyxaraSettings] = None,
        providers: Optional[Dict[str, LLMProviderBase]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        breaker: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._retry = retry_policy or RetryPolicy(
            max_attempts=self.settings.llm.max_retries + 1, base_delay=0.2, max_delay=5.0)
        self._breaker = breaker
        if providers is not None:
            self._providers = providers
        else:
            self._providers = {
                cls.name: cls(self.settings) for cls in _PROVIDER_CLASSES.values()
            }
        self._native = self._providers.get("native") or NativeProvider(self.settings)
        # invariant: a stateless facade keeps NO mutable conversation memory.
        self.stateless = True

    # the auto ladder: her OWN promoted weights first, then the Qwen base, then her always-on
    # dependency-free native own-brain as the guaranteed floor (never an echo mock).
    _AUTO_LADDER = ("self", "qwen", "native")

    def _auto_ladder(self) -> List[LLMProviderBase]:
        """Usable providers under ``provider=auto``, strongest-first.

        ``self`` joins only when it is both available (a promoted model exists) AND
        ``serve_ready()`` (the honesty gate: a LoRA over the real base, or the explicit
        ``self_serve_any_backend`` opt-in). ``native`` (her always-on own-brain) is always the
        guaranteed floor."""
        out: List[LLMProviderBase] = []
        for name in self._AUTO_LADDER:
            prov = self._providers.get(name)
            if prov is None or not prov.available():
                continue
            if name == "self" and not getattr(prov, "serve_ready", lambda: True)():
                continue
            out.append(prov)
        return out

    # ---- provider selection ---- #
    def chosen_provider(self) -> LLMProviderBase:
        name = self.settings.llm.provider.value
        if name == "auto":
            ladder = self._auto_ladder()
            if ladder:
                return ladder[0]
            return self._native            # her own-brain is the guaranteed floor
        prov = self._providers.get(name)
        if prov is not None and prov.available():
            return prov
        return self._native                # honest fallback to her own always-on brain

    def provider_status(self) -> Dict[str, bool]:
        return {n: p.available() for n, p in self._providers.items()}

    # ---- the serve half of the train→serve loop ---- #
    def on_promotion(self, event: Any) -> None:
        """React to a foundry promotion/rollback: the live brain adopts the new weights.

        (a) Hot-reload the ``self`` provider immediately (its per-request pointer check
        is the backstop when this callback is missed — e.g. a cross-process promotion).
        (b) When the promoted version is a LoRA over the *same* HF base the Qwen provider
        serves, point ``qwen_adapter_path`` at the fresh adapter — the provider's
        ``(model, adapter)`` cache key reloads it on the next call, so her own foundry
        adapter serves over the base with zero reconfiguration."""
        prov = self._providers.get("self")
        if prov is not None and hasattr(prov, "reload"):
            try:
                prov.reload()
            except Exception:  # noqa: BLE001 — best-effort; pointer-poll retries later
                pass
        try:
            if str(getattr(event, "kind", "")) != "lora":
                return
            from pathlib import Path
            vdir = Path(str(getattr(event, "path", "")))
            adapter = vdir / "adapter"
            spec_file = vdir / "spec.json"
            if not adapter.is_dir() or not spec_file.exists():
                return
            base = str(json.loads(spec_file.read_text(encoding="utf-8")).get("base_model", ""))
            if base and base == self.settings.llm.qwen_model:
                self.settings.llm.qwen_adapter_path = adapter
        except Exception:  # noqa: BLE001 — adapter injection is a bonus, never a failure
            pass

    def learning_view(self) -> Dict[str, Any]:
        """Truthful live-serving state for the learning report."""
        try:
            chosen = self.chosen_provider().name
        except Exception:  # noqa: BLE001
            chosen = None
        view: Dict[str, Any] = {"configured": self.settings.llm.provider.value,
                                "chosen": chosen}
        prov = self._providers.get("self")
        if prov is not None and hasattr(prov, "learning_view"):
            try:
                view["self"] = prov.learning_view()
            except Exception:  # noqa: BLE001
                view["self"] = None
        return view

    def available_providers(self) -> List[str]:
        """Names of every currently-usable provider — the pool a council draws from."""
        return [n for n, p in self._providers.items() if p.available()]

    def complete_with(self, name: str, req: LLMRequest) -> LLMResponse:
        """Complete through ONE named provider, honestly (no silent provider substitution).

        Unlike :meth:`complete`, this never falls back to another provider: a council needs each
        member to answer *as itself* or be recorded as absent, so a failed member must raise
        rather than be impersonated. Retry/circuit-breaker resilience still applies.
        """
        prov = self._providers.get(name)
        if prov is None:
            raise LLMError(f"no such provider '{name}'", context={"provider": name})
        return self._call_with_resilience(prov, req)

    # ---- core call (with retry + optional breaker, falling back to her native own-brain) ---- #
    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.settings.llm.provider.value == "auto":
            # walk the ladder honestly: each failed rung falls to the next backend, ending at her
            # always-on native own-brain; the response's ``provider`` field always names who answered.
            last: Optional[LLMError] = None
            for prov in self._auto_ladder() or [self._native]:
                try:
                    return self._call_with_resilience(prov, req)
                except LLMError as exc:
                    last = exc
            raise last or LLMError("no provider available on the auto ladder",
                                   context={"provider": "auto"})
        provider = self.chosen_provider()
        try:
            return self._call_with_resilience(provider, req)
        except LLMError:
            if provider is not self._native:
                return self._native.complete(req)
            raise

    def _call_with_resilience(self, provider: LLMProviderBase, req: LLMRequest) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                if self._breaker is not None:
                    return self._breaker.call(provider.complete, req)
                return provider.complete(req)
            except Exception as exc:  # noqa: BLE001
                if not self._retry.should_retry(exc, attempt):
                    raise
                time.sleep(self._retry.delay_for(attempt))

    async def acomplete(self, req: LLMRequest) -> LLMResponse:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(None, self.complete, req)

    # ---- ergonomic helpers (all stateless one-shots) ---- #
    def generate(self, prompt: str, *, system: Optional[str] = None, **kw: Any) -> str:
        return self.complete(LLMRequest.from_prompt(prompt, system=system, **kw)).text

    def chat(self, messages: Sequence[Message], **kw: Any) -> LLMResponse:
        return self.complete(LLMRequest.from_messages(messages, **kw))

    def generate_json(self, prompt: str, *, system: Optional[str] = None, **kw: Any) -> Any:
        kw["json_mode"] = True
        resp = self.complete(LLMRequest.from_prompt(prompt, system=system, **kw))
        return resp.parse_json()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return estimate_tokens(text)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.config import NyxaraSettings, Profile

    print("=" * 70)
    print("NYXARA llm self-test")
    print("=" * 70)

    # TEST profile forces the native own-brain (hermetic, offline)
    settings = NyxaraSettings.for_profile(Profile.TEST)
    llm = LLM(settings=settings)
    print(f"provider status     : {llm.provider_status()}")
    print(f"chosen provider     : {llm.chosen_provider().name}")
    assert llm.chosen_provider().name == "native"

    # basic generation — her own always-on brain, never an echo of the prompt
    out = llm.generate("Hello, who is your master?", system="You are NYXARA.")
    print(f"\ngenerate            : {out}")
    assert isinstance(out, str) and out.strip()

    # JSON mode
    data = llm.generate_json("return some json")
    print(f"generate_json       : {data}")
    assert data["native"] is True

    # statelessness: identical requests yield identical native output (replayable)
    r1 = llm.generate("same prompt")
    r2 = llm.generate("same prompt")
    assert r1 == r2
    print("\nstatelessness        : identical requests -> identical output OK")

    # response metadata + usage
    resp = llm.complete(LLMRequest.from_prompt("count my tokens please"))
    print(f"response            : {resp.to_dict()}")
    assert resp.usage.total_tokens > 0

    # JSON parsing tolerance (code fences / prose)
    fenced = LLMResponse(text='Here you go:\n```json\n{"a": 1}\n```', provider="x", model="m")
    assert fenced.parse_json() == {"a": 1}
    print("json parse (fenced)  : OK")

    # bad JSON raises LLMError
    bad = LLMResponse(text="not json at all", provider="x", model="m")
    try:
        bad.parse_json()
        raise SystemExit("ERROR: should have raised")
    except LLMError:
        print("bad json raises      : OK")

    # adapters report availability honestly (bare machine -> only native)
    status = llm.provider_status()
    print(f"\nadapter availability : {status}")
    assert set(status) == {"qwen", "self", "native"}
    for p in ("qwen", "self"):
        assert p in status, f"provider '{p}' must be registered"
    assert status["self"] is False       # no model trained/promoted yet on a bare machine
    # qwen is available iff its heavy deps (torch+transformers) are installed — honest either way
    print("qwen/self            : registered; degrade honestly on a bare machine ✓")

    print("\nALL SELF-TESTS PASSED ✓")
