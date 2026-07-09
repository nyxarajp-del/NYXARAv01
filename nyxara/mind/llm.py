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

* :class:`TinyLlamaProvider`   — any HF causal-LM (TinyLlama-1.1B by default), downloaded & run
  in-process (HuggingFace transformers); every load-/generation-time knob is exposed via
  ``NYXARA_LLM__TINYLLAMA_*``, including serving a foundry-forged LoRA adapter directly
* :class:`LlamaCppProvider`    — a quantized **GGUF** (the Qwythos-9B quant by default) served
  in-process via ``llama-cpp-python``; cheap inference for a large base, configured by
  ``NYXARA_LLM__GGUF_*`` (GGUF is inference-only — the foundry trains the safetensors parent)
* :class:`SelfProvider`        — NYXARA's OWN model, trained & promoted by the foundry
* :class:`MockProvider`        — deterministic, offline; the always-available fallback

No cloud providers, no API keys. Heavy deps (``torch``/``transformers``/``peft``/``llama_cpp``)
are imported lazily and reported honestly via ``available()``, so this module works with zero
heavy deps installed (falling back to the mock).

Depends on :mod:`config` and :mod:`errors`; optionally uses a
:class:`~nyxara.kernel.runtime.CircuitBreaker`.
"""

from __future__ import annotations

import hashlib
import json
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
    "MockProvider",
    "TinyLlamaProvider",
    "LlamaCppProvider",
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
    """Rough token estimate (~4 chars/token) for budgeting and mock usage."""
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
# Mock provider — deterministic, always available
# --------------------------------------------------------------------------- #
class MockProvider(LLMProviderBase):
    """Offline, deterministic provider for tests, replay, and graceful fallback."""

    name = "mock"

    def available(self) -> bool:
        return True

    def default_model(self) -> str:
        return "mock"

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        last = req.last_user()
        if req.json_mode:
            text = json.dumps({"echo": last[:200], "model": model, "mock": True,
                               "fingerprint": req.fingerprint()})
        else:
            sys_note = f" (system: {req.system[:40]})" if req.system else ""
            text = f"[mock:{model}]{sys_note} {last[:400]}" if last else f"[mock:{model}] (empty)"
        usage = Usage(prompt_tokens=estimate_tokens(" ".join(m.content for m in req.messages)),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop", usage, {"mock": True})


# --------------------------------------------------------------------------- #
# TinyLlama provider — TinyLlama-1.1B, downloaded & run locally (HuggingFace)
# --------------------------------------------------------------------------- #
class TinyLlamaProvider(LLMProviderBase):
    """Run **TinyLlama-1.1B** in-process, downloaded via HuggingFace — the sole real backend.

    The model is fetched on first use (``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` by default)
    and cached locally by the ``transformers`` hub, then served entirely on this machine —
    no API key, no network at inference time. The chat checkpoint ships the Zephyr chat
    template (``system``/``user``/``assistant``), applied via ``apply_chat_template``.

    Maximum control, all from config (``NYXARA_LLM__TINYLLAMA_*``):

    * load-time — device, dtype, 4-/8-bit quantized load (bitsandbytes+CUDA only, silently
      full-precision otherwise), attention implementation, KV cache, trust_remote_code;
    * fine-tune serving — ``tinyllama_adapter_path`` loads a peft LoRA adapter (e.g. a
      foundry ``versions/vN/adapter``) on top of the base; ``tinyllama_merge_adapter``
      folds it into the weights for faster inference;
    * generation — top_k, repetition_penalty, no_repeat_ngram_size, min_new_tokens,
      beam search + length_penalty, do_sample policy, deterministic seeding, input-length
      budget; per-request :class:`LLMRequest` fields (temperature/top_p/max_tokens/stop/
      seed) always win over config defaults.

    Heavy deps are imported lazily and reported honestly, so a bare machine degrades to
    the mock rather than erroring. Stateless: the loaded weights are a cached instrument,
    never conversation memory.
    """

    name = "tinyllama"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._model = None
        self._tokenizer = None
        self._loaded_key: Optional[Tuple[str, str]] = None   # (model_name, adapter_path)
        self._torch: Any = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        if self.settings.llm.tinyllama_adapter_path is not None:
            try:
                import peft  # noqa: F401
            except Exception:
                return False
        return True

    def default_model(self) -> str:
        return self.settings.llm.tinyllama_model

    # ---- lazy model loading (cached; reloads when model or adapter changes) ---- #
    def _quant_config(self, torch: Any) -> Optional[Any]:
        """A ``BitsAndBytesConfig`` when quantization is requested AND usable, else None.

        Quantized load needs bitsandbytes + CUDA; anywhere else we silently serve full
        precision instead of crashing (graceful degradation, mirrors the foundry)."""
        cfg = self.settings.llm
        if not (cfg.tinyllama_load_in_4bit or cfg.tinyllama_load_in_8bit):
            return None
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception:
            return None
        if not torch.cuda.is_available():
            return None
        if cfg.tinyllama_load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.tinyllama_bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.tinyllama_bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.tinyllama_bnb_4bit_use_double_quant,
        )

    def _ensure_model(self, model: str):
        cfg = self.settings.llm
        key = (model, str(cfg.tinyllama_adapter_path or ""))
        if self._model is None or self._loaded_key != key:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                model, trust_remote_code=cfg.tinyllama_trust_remote_code)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kwargs: Dict[str, Any] = {
                "torch_dtype": ("auto" if cfg.tinyllama_dtype == "auto"
                                else getattr(torch, cfg.tinyllama_dtype)),
                "device_map": cfg.tinyllama_device or "auto",
                "trust_remote_code": cfg.tinyllama_trust_remote_code,
            }
            if cfg.tinyllama_attn_implementation:
                kwargs["attn_implementation"] = cfg.tinyllama_attn_implementation
            quant = self._quant_config(torch)
            if quant is not None:
                kwargs["quantization_config"] = quant
                kwargs["device_map"] = "auto"   # bitsandbytes places layers itself
            lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
            if cfg.tinyllama_adapter_path is not None:
                from peft import PeftModel   # a bad adapter raises -> LLMError -> mock fallback
                lm = PeftModel.from_pretrained(lm, str(cfg.tinyllama_adapter_path))
                if cfg.tinyllama_merge_adapter:
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
        if cfg.tinyllama_use_chat_template:
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
            cfg.tinyllama_do_sample, req.temperature > 0)
        kwargs: Dict[str, Any] = {
            "max_new_tokens": req.max_tokens,
            "do_sample": do_sample,
            "num_beams": cfg.tinyllama_num_beams,
            "use_cache": cfg.tinyllama_use_cache,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(req.temperature, 1e-3)
            kwargs["top_p"] = req.top_p
            if cfg.tinyllama_top_k > 0:
                kwargs["top_k"] = cfg.tinyllama_top_k
        if cfg.tinyllama_repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = cfg.tinyllama_repetition_penalty
        if cfg.tinyllama_no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = cfg.tinyllama_no_repeat_ngram_size
        if cfg.tinyllama_min_new_tokens > 0:
            kwargs["min_new_tokens"] = cfg.tinyllama_min_new_tokens
        if cfg.tinyllama_num_beams > 1:
            kwargs["length_penalty"] = cfg.tinyllama_length_penalty
        return kwargs

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        lm, tok = self._ensure_model(model)
        cfg = self.settings.llm
        prompt = self._render_prompt(req, tok)
        inputs = dict(tok([prompt], return_tensors="pt"))
        # prompt + completion must fit TinyLlama's 2048-token context — keep the tail
        budget = max(8, cfg.tinyllama_max_input_tokens - req.max_tokens)
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
        adapter = cfg.tinyllama_adapter_path
        return (text, finish, usage,
                {"tinyllama": True, "model": model,
                 "adapter": str(adapter) if adapter else None})


# --------------------------------------------------------------------------- #
# GGUF provider — a quantized GGUF served in-process via llama.cpp
# --------------------------------------------------------------------------- #
class LlamaCppProvider(LLMProviderBase):
    """Serve a **GGUF** model in-process through ``llama-cpp-python`` (llama.cpp).

    This is the cheap-inference twin of :class:`TinyLlamaProvider`: it runs the Qwythos-9B
    (or any) *quantized* GGUF — a few gigabytes instead of the full BF16 weights — so a large
    base is servable on a single consumer GPU or even CPU. GGUF is an inference-only format,
    so this backend **never trains**; the foundry LoRA-tunes the safetensors parent, and (once
    that adapter is exported to GGUF) ``gguf_lora_path`` applies it here at serve time.

    All from config (``NYXARA_LLM__GGUF_*``): which repo/quant to pull (``gguf_model`` +
    ``gguf_filename``), context window, GPU-offload layers, threads, an optional chat format
    override, and an optional llama.cpp LoRA. Per-request :class:`LLMRequest` fields
    (temperature/top_p/max_tokens/stop/seed) win over config. Stateless: the loaded model is a
    cached instrument, never conversation memory.

    Heavy dep imported lazily and reported honestly — with no ``llama-cpp-python`` installed
    ``available()`` is False and the facade degrades to the mock rather than erroring.
    """

    name = "gguf"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._llama = None
        self._loaded_key: Optional[Tuple[str, str, str]] = None  # (model, filename, lora_path)

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return False
        return True

    def default_model(self) -> str:
        return self.settings.llm.gguf_model

    # ---- lazy model loading (cached; reloads when model/quant/adapter changes) ---- #
    def _ensure_model(self, model: str):
        cfg = self.settings.llm
        lora = str(cfg.gguf_lora_path) if cfg.gguf_lora_path else ""
        key = (model, cfg.gguf_filename, lora)
        if self._llama is None or self._loaded_key != key:
            from llama_cpp import Llama
            kwargs: Dict[str, Any] = {
                "repo_id": model,
                "filename": cfg.gguf_filename,
                "n_ctx": cfg.gguf_n_ctx,
                "n_gpu_layers": cfg.gguf_n_gpu_layers,
                "verbose": False,
            }
            if cfg.gguf_n_threads > 0:
                kwargs["n_threads"] = cfg.gguf_n_threads
            if cfg.gguf_chat_format:
                kwargs["chat_format"] = cfg.gguf_chat_format
            if cfg.gguf_seed >= 0:
                kwargs["seed"] = cfg.gguf_seed
            if lora:
                kwargs["lora_path"] = lora
                kwargs["lora_scale"] = cfg.gguf_lora_scale
            self._llama = Llama.from_pretrained(**kwargs)
            self._loaded_key = key
        return self._llama

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        llama = self._ensure_model(model)
        cfg = self.settings.llm
        system = (req.system or "").strip()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(req.provider_messages())
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stop": list(req.stop) or None,
        }
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if req.seed is not None:      # per-request determinism wins over the config seed
            kwargs["seed"] = req.seed
        out = llama.create_chat_completion(**kwargs)
        choice = out["choices"][0]
        text = (choice.get("message", {}).get("content") or "").strip()
        text, hit = truncate_at_stops(text, req.stop)
        finish = "stop" if (hit or choice.get("finish_reason") == "stop") else "length"
        u = out.get("usage", {}) or {}
        usage = Usage(prompt_tokens=int(u.get("prompt_tokens", estimate_tokens(str(messages)))),
                      completion_tokens=int(u.get("completion_tokens", estimate_tokens(text))))
        return (text, finish, usage,
                {"gguf": True, "model": model, "filename": cfg.gguf_filename,
                 "lora": str(cfg.gguf_lora_path) if cfg.gguf_lora_path else None})


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
    avoid an import cycle (growth/foundry_models imports nothing from mind/llm)."""

    name = "self"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._lm = None

    def _root(self):
        from pathlib import Path
        d = self.settings.llm.self_model_dir or (self.settings.paths.data_dir / "foundry")
        return Path(d)

    def available(self) -> bool:
        try:
            return (self._root() / "active").exists()
        except Exception:
            return False

    def default_model(self) -> str:
        return "nyxara-self"

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        from nyxara.growth.foundry_models import load_active_model  # lazy: no import cycle
        if self._lm is None:
            self._lm = load_active_model(self.settings)
        prompt = format_self_prompt(req)
        raw = self._lm.generate(prompt, max_tokens=req.max_tokens)
        # keep the answer to its own turn: stop at any caller stop or a role marker
        stops = tuple(req.stop) + (f"\n{_SELF_USER_TAG}", f"\n{_SELF_ASSISTANT_TAG}",
                                   _SELF_USER_TAG, _SELF_ASSISTANT_TAG)
        text, hit = truncate_at_stops(raw, stops)
        usage = Usage(prompt_tokens=estimate_tokens(prompt),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop" if hit else "length", usage,
                {"self": True, "kind": self._lm.kind})


# --------------------------------------------------------------------------- #
# The stateless facade the kernel calls
# --------------------------------------------------------------------------- #
_PROVIDER_CLASSES = {
    ProviderName.TINYLLAMA: TinyLlamaProvider,
    ProviderName.GGUF: LlamaCppProvider,
    ProviderName.SELF: SelfProvider,
    ProviderName.MOCK: MockProvider,
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
        self._mock = self._providers.get("mock") or MockProvider(self.settings)
        # invariant: a stateless facade keeps NO mutable conversation memory.
        self.stateless = True

    # ---- provider selection ---- #
    def chosen_provider(self) -> LLMProviderBase:
        name = self.settings.llm.provider.value
        prov = self._providers.get(name)
        if prov is not None and prov.available():
            return prov
        if self.settings.llm.allow_mock_fallback:
            return self._mock
        raise LLMError(f"selected provider '{name}' unavailable and mock fallback disabled",
                       context={"provider": name})

    def provider_status(self) -> Dict[str, bool]:
        return {n: p.available() for n, p in self._providers.items()}

    def available_providers(self) -> List[str]:
        """Names of every currently-usable provider — the pool a council draws from."""
        return [n for n, p in self._providers.items() if p.available()]

    def complete_with(self, name: str, req: LLMRequest) -> LLMResponse:
        """Complete through ONE named provider, honestly (no silent mock substitution).

        Unlike :meth:`complete`, this never falls back to the mock: a council needs each
        member to answer *as itself* or be recorded as absent, so a failed member must raise
        rather than be impersonated. Retry/circuit-breaker resilience still applies.
        """
        prov = self._providers.get(name)
        if prov is None:
            raise LLMError(f"no such provider '{name}'", context={"provider": name})
        return self._call_with_resilience(prov, req)

    # ---- core call (with retry + optional breaker, falling back to mock) ---- #
    def complete(self, req: LLMRequest) -> LLMResponse:
        provider = self.chosen_provider()
        try:
            return self._call_with_resilience(provider, req)
        except LLMError:
            if provider is not self._mock and self.settings.llm.allow_mock_fallback:
                return self._mock.complete(req)
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

    # TEST profile forces the mock provider (hermetic, offline)
    settings = NyxaraSettings.for_profile(Profile.TEST)
    llm = LLM(settings=settings)
    print(f"provider status     : {llm.provider_status()}")
    print(f"chosen provider     : {llm.chosen_provider().name}")
    assert llm.chosen_provider().name == "mock"

    # basic generation
    out = llm.generate("Hello, who is your master?", system="You are NYXARA.")
    print(f"\ngenerate            : {out}")
    assert "Hello, who is your master?" in out

    # JSON mode
    data = llm.generate_json("return some json")
    print(f"generate_json       : {data}")
    assert data["mock"] is True

    # statelessness: identical requests yield identical mock output (replayable)
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

    # adapters report availability honestly (bare machine -> only mock)
    status = llm.provider_status()
    print(f"\nadapter availability : {status}")
    assert set(status) == {"tinyllama", "gguf", "self", "mock"}
    for p in ("tinyllama", "gguf", "self"):
        assert p in status, f"provider '{p}' must be registered"
    assert status["self"] is False       # no model trained/promoted yet on a bare machine
    # tinyllama/gguf are available iff their heavy deps are installed — honest either way
    print("tinyllama/gguf/self  : registered; degrade honestly on a bare machine ✓")

    print("\nALL SELF-TESTS PASSED ✓")
