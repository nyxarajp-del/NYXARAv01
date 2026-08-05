"""NYXARA · mind/llm.py — the STATELESS large-language-model faculty (⬆).

The single most important constraint in this file is what it *does not* do.

    The LLM is a provider, not the driver. Request in → text/JSON out.
    No memory. No tool-calls. No side-effects. No control.

The kernel/orchestrator calls *it*; it never calls back. It holds **no conversation
state** between calls — every :class:`LLMRequest` is self-contained, which is what
makes cognition replayable (kernel/replay.py) and auditable. Memory lives in
``memory/``; tool execution lives in ``agency/``; the decision to *act* on any output
belongs to the kernel after the proposal passes guards (mind/proposal.py).

Selected by config, strongest-reachable-first on the ``auto`` ladder:

* :class:`LiteRTLMProvider`    — her PRIMARY brain, and the only strong one that never leaves the
  machine: Gemma-4-E2B-it in Google's LiteRT-LM on-device format (a single ~2.4 GB INT4
  ``model.litertlm``), served IN-PROCESS through the ``litert_lm`` binding. No API key, no endpoint,
  no wire — so there is nothing for the isolation envelope to hide, and no outage can take it away.
  First rung whenever the weights are on disk, under exactly the same subordinate contract as every
  other rung: request in → text out, no persona, no state, no callback, output gated as a proposal.
* :class:`SelfProvider`        — NYXARA's OWN model, trained & promoted by the foundry (a LoRA
  adapter over the foundry base — everything above the base is *hers*).
* :class:`NativeProvider`      — her always-on, dependency-free OWN brain: a pure-stdlib
  Kneser-Ney word n-gram (``growth/foundry_models.WordKNGramLM``) trained on her identity seed
  corpus. Deterministic, needs no torch/numpy/network, so it is the *guaranteed floor* of the
  ladder — a bare machine still answers from her own learned voice, never an echo of the prompt.

**Every rung is hers now.** There are no cloud providers here at all: the aicredits/Groq/airouter
rungs were removed at the Master's instruction after all three proved dead in practice (no balance,
a daily token cap, a paid-plan wall) while still costing a wall of warnings per turn. What is left
is the ladder that never depended on anyone — ``litertlm`` on the machine, her forged ``self``
weights, her ``native`` own-brain as the floor. Nothing she says now leaves the host to be said,
which also makes ``guard/isolation_envelope.py`` inapplicable rather than merely optional.

Heavy/optional deps (the ``litert_lm`` binding, ``torch``/``peft`` for the foundry) are imported
lazily and reported honestly via ``available()``, so this module works with zero of them installed
(falling back to her native own-brain).

Depends on :mod:`config` and :mod:`errors`; optionally uses a
:class:`~nyxara.kernel.runtime.CircuitBreaker`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.config import LLMProvider as ProviderName
from nyxara.kernel.config import NyxaraSettings, get_settings
from nyxara.kernel.errors import LLMError, RetryPolicy, classify

log = logging.getLogger("nyxara.mind.llm")

__all__ = [
    "Role",
    "Message",
    "Usage",
    "LLMRequest",
    "LLMResponse",
    "LLMProviderBase",
    "NativeProvider",
    "SelfProvider",
    "LiteRTLMProvider",
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
    (her on-device ``litertlm`` brain or her forged ``self`` weights) always outranks it on the
    ``auto`` ladder."""

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
        A small from-scratch backend (ngram / NumPy / nanogpt) replacing a large pretrained
        model is a deliberate choice, so it goes through ``self_serve_any_backend`` — which
        ships **ON**, so her own forged brain serves from first boot. Set it to ``False`` for
        the conservative LoRA-only gate."""
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
# LiteRT-LM — her PRIMARY brain, running ON-DEVICE (Gemma-4-E2B-it, INT4)
# --------------------------------------------------------------------------- #
_JSON_NUDGE = "Respond with ONLY valid JSON — no prose, no code fences."

# Gemma 4's turn markers (vocabulary ids 105/106 — *not* Gemma 3's ``<start_of_turn>`` pair). A local
# model has no server to cut it off, so a reply that runs past its own turn keeps going into a
# hallucinated next one — stop at the first marker that appears in the text.
_GEMMA_STOPS = ("<turn|>", "<|turn>")


class LiteRTLMProvider(LLMProviderBase):
    """Serve a LiteRT-LM ``.litertlm`` model in-process — NYXARA's PRIMARY brain.

    The ladder's first rung, and the only strong model on it that **never touches the network**:
    Gemma-4-E2B-it quantized to INT4 and served by Google's LiteRT-LM runtime through the
    ``litert_lm`` binding. The weights sit on her own disk, inference happens in her own process, and
    the prompt is never transmitted anywhere.

    Two consequences worth stating plainly, because they are why this rung leads:

    * **No outage can take it.** Cloud rungs answer when reachable; this one answers when the file
      exists. An air-gapped machine now runs on a real instruct model rather than an n-gram.
    * **No isolation envelope.** ``guard/isolation_envelope.py`` exists to abstract her identity and
      the Master's secrets *before they leave for a cloud tool*. Nothing leaves here, so there is
      nothing to abstract — the envelope is not skipped by oversight, it is inapplicable.

    Nothing about *primary* changes the power relation. This is the same contract as every other
    provider in this module: stateless, persona-free, callback-free, and its output is a mere proposal
    the kernel gates (``kernel/orchestrator.py::_gate``). Being local makes it *reachable*, not
    authoritative — and an unaligned base model is precisely why the guards, not the model, decide
    what she acts on.

    **Statelessness with a KV cache.** A ``Conversation`` carries the model's attention state, so one
    is built *and closed* per request from the messages the caller supplied; only the ``Engine`` (a
    ~7 s, multi-GB load) is cached. Identical requests therefore stay replayable — with a seed pinned,
    the same request yields the same text, verified against the real weights.

    **Why the chat template is ours.** The template embedded in this model file calls ``.get`` on a
    map, which the runtime's Jinja engine does not implement; left alone, every ``send_message`` fails
    with *"Failed to apply template"*. ``settings.llm.litertlm_chat_template`` overrides it with
    Gemma's own turn format. Its two sharp edges are documented at that config field — the one that
    matters here is that a **bare string message renders to nothing**, so this provider always hands
    the runtime ``litert_lm.Message`` objects.
    """

    name = "litertlm"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._engine: Any = None
        self._engine_path: Optional[str] = None
        self._lock = threading.Lock()
        # Latched load failure. Once the runtime has proved it cannot start on this host, the rung
        # takes itself off the ladder instead of failing — and logging — on every single turn.
        self._dead: Optional[str] = None
        # Which chat template this build actually accepts, learned on the first successful turn.
        self._template_name: Optional[str] = None
        self._template: Optional[str] = None

    # ---- honest availability (cheap: never loads the engine, never downloads) ---- #
    def model_path(self) -> Any:
        from nyxara.mind.litertlm_assets import model_path
        return model_path(self.settings)

    def _binding(self) -> Any:
        """The ``litert_lm`` module, or None when the optional wheel is not installed."""
        try:
            import litert_lm
        except Exception:  # noqa: BLE001 — an unbuilt/ABI-mismatched wheel is just unavailability
            return None
        return litert_lm

    def native_library_error(self) -> Optional[str]:
        """``None`` if the runtime's shared library loads here, else why it does not.

        Importing ``litert_lm`` proves nothing: the Python package is pure wrapper and does not touch
        ``liblitert-lm.so`` until an ``Engine`` is constructed. That library is linked against the
        Vulkan loader — **even for the CPU backend** — so on a host without ``libvulkan.so.1`` the
        import succeeds, ``available()`` said yes, and every turn then failed with
        ``libvulkan.so.1: cannot open shared object file``. Checking the actual dlopen is the only
        honest probe. It costs ~0.15 s once per process (the library, never the 2.4 GB weights) and
        the result is cached by the binding itself.

        Before giving up, ``mind/vulkan_shim.py`` is given a chance to satisfy that one dependency
        in-process — the runtime imports no symbols from Vulkan, so on the CPU backend a stub is
        enough and no system package is needed. It declines for GPU/NPU backends, where a
        do-nothing Vulkan would be a silent lie.
        """
        try:
            from litert_lm import _ffi
        except Exception as exc:  # noqa: BLE001 — no binding at all
            return f"litert_lm is not installed ({exc})"
        # Before the shared library loads: the runtime writes its real errors to fd 2 and resolves
        # that fd once, at load time. Tee it now or never see why a turn failed.
        try:
            from nyxara.mind.runtime_log import install as _install_runtime_log
            _install_runtime_log()
        except Exception:  # noqa: BLE001 — extra detail is a bonus, never a requirement
            pass
        try:
            from nyxara.mind.vulkan_shim import ensure_vulkan_loader
            ensure_vulkan_loader(self.settings)
        except Exception:  # noqa: BLE001 — the shim is an aid, never a requirement
            pass
        try:
            _ffi._get_lib()
        except OSError as exc:                    # the missing-system-library case
            missing = str(exc).split(":")[0].strip()
            hint = ""
            if "vulkan" in str(exc).lower():
                # We get here only when the shim declined or could not help (a GPU/NPU backend, an
                # unknown architecture, a read-only data dir). Installing the loader always works.
                hint = (" — install the Vulkan loader: `sudo apt install libvulkan1` "
                        "(Debian/Ubuntu) or `sudo dnf install vulkan-loader` (Fedora/RHEL)")
            return f"litert_lm runtime cannot load {missing}{hint}"
        except Exception as exc:  # noqa: BLE001
            return f"litert_lm runtime failed to load: {exc}"
        return None

    def available(self) -> bool:
        if not bool(getattr(self.settings.llm, "litertlm_enabled", True)):
            return False
        if self._dead is not None:
            return False                          # already proved unusable on this host
        if self._binding() is None:
            return False
        if self.native_library_error() is not None:
            return False
        try:
            return self.model_path().is_file()
        except Exception:  # noqa: BLE001
            return False

    def default_model(self) -> str:
        return self.settings.llm.litertlm_model

    # ---- the engine (cached; the conversation deliberately is not) ---- #
    def _backend(self, litert_lm: Any) -> Any:
        kind = str(getattr(self.settings.llm, "litertlm_backend", "cpu")).lower()
        if kind == "gpu":
            return litert_lm.Backend.GPU()
        if kind == "npu":
            return litert_lm.Backend.NPU()
        threads = getattr(self.settings.llm, "litertlm_threads", None)
        return litert_lm.Backend.CPU(thread_count=threads) if threads else litert_lm.Backend.CPU()

    def _cache_dir(self) -> Optional[str]:
        from pathlib import Path
        d = getattr(self.settings.llm, "litertlm_cache_dir", None)
        if d is None:
            d = self.model_path().parent / "litertlm-cache"
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 — a read-only cache dir only costs recompilation
            return None
        return str(d)

    def _get_engine(self, litert_lm: Any) -> Any:
        """Load once, reuse forever. Multi-GB and ~7 s, so a per-request load is not an option."""
        path = str(self.model_path())
        if self._engine is not None and self._engine_path == path:
            return self._engine
        with self._lock:
            if self._engine is not None and self._engine_path == path:
                return self._engine
            try:   # the runtime is chatty on stderr; keep her console readable
                litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
            except Exception:  # noqa: BLE001
                pass
            try:
                engine = litert_lm.Engine(path, backend=self._backend(litert_lm),
                                          cache_dir=self._cache_dir())
            except Exception as exc:  # noqa: BLE001
                # A host that cannot start the runtime will not start it on the next turn either
                # (a missing system library, an unsupported backend, too little RAM). Latch it so
                # the rung leaves the ladder rather than failing — and logging — once per turn,
                # forever. ``reset()`` clears this after the operator fixes the host.
                self._dead = str(exc)
                log.warning("litertlm engine could not start; taking the rung off the ladder "
                            "until reset: %s", self._dead)
                raise
            self._engine, self._engine_path = engine, path
            log.info("litertlm engine loaded: %s", path)
            return engine

    def learning_view(self) -> Dict[str, Any]:
        """Truthful serving state for the learning report — never fabricates.

        The interesting field is ``reason``: when her primary is *not* serving, an operator should be
        able to see at a glance whether that is a missing wheel, a missing 2.4 GB file, or a switch
        someone turned off — rather than only noticing that answers got worse.
        """
        try:
            path = self.model_path()
            weights = path.is_file()
        except Exception:  # noqa: BLE001
            path, weights = None, False
        enabled = bool(getattr(self.settings.llm, "litertlm_enabled", True))
        binding = self._binding() is not None
        native = self.native_library_error() if binding else None
        if not enabled:
            reason = "disabled by config"
        elif not binding:
            reason = "litert_lm not installed (pip install litert-lm-api)"
        elif native is not None:
            reason = native
        elif self._dead is not None:
            reason = f"unusable on this host: {self._dead}"
        elif not weights:
            reason = f"weights missing at {path} (python scripts/fetch_litertlm_model.py)"
        else:
            reason = None
        return {"available": reason is None, "enabled": enabled,
                "binding_installed": binding, "runtime_loadable": binding and native is None,
                "weights_present": weights,
                "model_path": str(path) if path is not None else None,
                "model": self.settings.llm.litertlm_model,
                "backend": getattr(self.settings.llm, "litertlm_backend", "cpu"),
                "engine_loaded": self._engine is not None, "reason_unavailable": reason}

    def close(self) -> None:
        """Release the engine and its weights (used by the tests and by a lean reload)."""
        with self._lock:
            engine, self._engine, self._engine_path = self._engine, None, None
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001 — releasing must never raise
                pass

    def reset(self) -> None:
        """Forget a latched failure and try the host again.

        The latch exists so a broken host costs one warning instead of one per turn; it must not
        mean an operator has to restart her after installing the missing library.
        """
        self._dead = None
        self._template_name = self._template = None

    # ---- request → litert_lm.Message objects ---- #
    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text out of ``{"role": ..., "content": [{"type": "text", "text": ...}]}``.

        Tolerant on purpose: it joins every text part it finds and ignores non-text content, so a
        future multi-modal part cannot turn a good answer into a crash.
        """
        if isinstance(response, str):
            return response
        try:
            parts = response["content"]
        except Exception:  # noqa: BLE001
            return ""
        if isinstance(parts, str):
            return parts
        out: List[str] = []
        for part in parts or ():
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "".join(out)

    def _history(self, req: LLMRequest, litert_lm: Any) -> Tuple[List[Any], str]:
        """Split the request into (prior turns, the final user text to send).

        Gemma 4 has a real ``system`` role, so ``req.system`` becomes a genuine system turn rather
        than being folded into the user's words. Every element is a real ``litert_lm.Message``: a
        bare string would normalise to a *string* ``content``, the chat template would iterate its
        characters and render nothing, and the model would answer an empty prompt with a canned
        self-introduction.

        Note what is NOT here: no "you are NYXARA" persona. The system text is whatever the caller
        passed — her identity is composed in ``identity/soul.py``, never injected by a provider.
        """
        system = (req.system or "").strip()
        if req.json_mode:
            system = f"{system}\n\n{_JSON_NUDGE}" if system else _JSON_NUDGE

        turns: List[Tuple[Role, str]] = [(m.role, m.content) for m in req.messages
                                         if m.role in (Role.USER, Role.ASSISTANT)]
        if not turns:
            turns = [(Role.USER, "")]

        # the last user turn is what we SEND; everything before it is conversation history
        send_at = max((i for i, (r, _) in enumerate(turns) if r is Role.USER), default=len(turns) - 1)
        history: List[Any] = []
        if system:
            history.append(litert_lm.Message.system(system))
        for role, content in turns[:send_at]:
            if role is Role.USER:
                history.append(litert_lm.Message.user(content))
            else:
                history.append(litert_lm.Message.model(litert_lm.Contents.of(content)))
        return history, turns[send_at][1]

    def _templates(self) -> List[Tuple[str, Optional[str]]]:
        """The chat templates to try, best-known first. ``None`` means the model's embedded one.

        The template is the most fragile joint in this whole rung, and it fails in the most
        expensive way: ``send_message`` returns a bare
        ``RuntimeError: litert_lm_conversation_send_message failed`` while the *real* reason
        ("Failed to apply template: ...") goes to the runtime's own stderr, where a caller cannot
        see it. Rather than guess which formulation a given build's Jinja accepts, try them:

        1. the configured template — Gemma 4's turn format, escaped ``\\n``;
        2. the same shape with a literal newline inside the string, which some builds prefer;
        3. the model's own embedded template — broken on litert-lm-api 0.15 (it calls ``.get`` on a
           map) but correct on any build that implements that method, and the best answer there.

        Whichever works first is remembered for the process, so this costs one extra attempt once.
        """
        configured = self.settings.llm.litertlm_chat_template
        return [("configured", configured),
                ("literal-newline", configured.replace('{{ "\\n" }}', "{{ '\n' }}")),
                ("model-embedded", None)]

    def _send(self, engine: Any, litert_lm: Any, history: List[Any], prompt: str,
              sampler: Any, max_tokens: int) -> Any:
        """One turn through the runtime, falling across template variants until one works."""
        candidates = ([(self._template_name, self._template)] if self._template_name is not None
                      else self._templates())
        errors: List[str] = []
        for name, template in candidates:
            kwargs: Dict[str, Any] = {"messages": history or None, "sampler_config": sampler}
            if template is not None:
                kwargs["chat_template"] = template
            try:
                conv = engine.create_conversation(**kwargs)
            except Exception as exc:  # noqa: BLE001 — a rejected template is a candidate failing
                errors.append(f"{name}: {exc}")
                continue
            try:
                raw = conv.send_message(litert_lm.Message.user(prompt),
                                        max_output_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                continue
            finally:
                try:
                    conv.close()          # drop the KV cache: the facade stays stateless
                except Exception:  # noqa: BLE001
                    pass
            if self._template_name != name:
                log.info("litertlm chat template in use: %s", name)
                self._template_name, self._template = name, template
            return raw

        # Every template failed. On a host where that is structural, retrying it on every turn
        # produces one warning per turn forever (observed in the wild) — so take the rung off the
        # ladder and say why, exactly as an engine that will not start does.
        detail = "; ".join(errors)
        try:    # the runtime's own words beat our wrapper's every time
            from nyxara.mind.runtime_log import interesting
            said = interesting()
            if said:
                detail += " | runtime said: " + " / ".join(said)
        except Exception:  # noqa: BLE001
            pass
        if self._template_name is None:
            self._dead = f"no usable chat template ({detail})"
            log.warning("litertlm: no chat template this build accepts — taking the rung off the "
                        "ladder until reset: %s", detail)
        else:
            self._template_name = self._template = None   # re-probe from scratch next time
        raise LLMError(f"litertlm could not render a turn — {detail}",
                       context={"provider": self.name})

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        litert_lm = self._binding()
        if litert_lm is None:
            raise LLMError("litert_lm is not installed (pip install litert-lm-api)",
                           context={"provider": self.name})
        engine = self._get_engine(litert_lm)
        history, prompt = self._history(req, litert_lm)
        cfg = self.settings.llm
        sampler = litert_lm.SamplerConfig(
            top_k=int(getattr(cfg, "litertlm_top_k", 40)),
            top_p=float(req.top_p),
            temperature=float(req.temperature),
            seed=req.seed,
        )
        # One engine, serialized: the runtime holds a single set of weights, and a 2.4 GB INT4 model
        # on CPU is the bottleneck anyway — concurrent turns queue rather than corrupt each other.
        with self._lock:
            raw = self._send(engine, litert_lm, history, prompt, sampler, int(req.max_tokens))
        text, hit = truncate_at_stops(self._extract_text(raw), tuple(req.stop) + _GEMMA_STOPS)
        completion = estimate_tokens(text)
        # Honest finish reason. A stop marker in the text means the model ran past its turn and we
        # cut it. Otherwise the runtime stopped on EOS *or* on the token cap — and only a reply that
        # actually reached the cap is a truncation, so don't report one that plainly didn't.
        if hit:
            finish = "stop"
        else:
            finish = "length" if completion >= int(req.max_tokens) else "stop"
        usage = Usage(prompt_tokens=estimate_tokens(prompt), completion_tokens=completion)
        return (text, finish, usage,
                {"litertlm": True, "model": model, "on_device": True})


_PROVIDER_CLASSES = {
    ProviderName.LITERTLM: LiteRTLMProvider,
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
        # Degradation surface (operational health, not conversation state): the last provider
        # failure that forced a fallback, so an operator can see WHY the cloud went quiet
        # (billing, network, 4xx) instead of a silent drop to the native floor.
        self.last_fallback: Optional[Dict[str, Any]] = None
        self._warned_at: Dict[str, float] = {}
        # invariant: a stateless facade keeps NO mutable conversation memory.
        self.stateless = True

    # The auto ladder, and every rung of it runs on her own hardware: her PRIMARY on-device brain
    # (litertlm — Gemma-4-E2B-it in-process), then her OWN promoted foundry weights, and finally her
    # always-on dependency-free native own-brain as the guaranteed floor (never an echo mock). The
    # cloud rungs that used to sit in the middle are gone; nothing on this ladder can be taken away
    # by an outage, a bill, or a rate limit, because nothing on it is reached over a wire.
    _AUTO_LADDER = ("litertlm", "self", "native")

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

        Hot-reload the ``self`` provider immediately (its per-request pointer check is the
        backstop when this callback is missed — e.g. a cross-process promotion). The foundry's
        own LoRA serves through ``self`` directly, so no external provider needs reconfiguring."""
        prov = self._providers.get("self")
        if prov is not None and hasattr(prov, "reload"):
            try:
                prov.reload()
            except Exception:  # noqa: BLE001 — best-effort; pointer-poll retries later
                pass

    def learning_view(self) -> Dict[str, Any]:
        """Truthful live-serving state for the learning report."""
        try:
            chosen = self.chosen_provider().name
        except Exception:  # noqa: BLE001
            chosen = None
        view: Dict[str, Any] = {"configured": self.settings.llm.provider.value,
                                "chosen": chosen,
                                "last_fallback": self.last_fallback}
        for name in ("litertlm", "self"):
            prov = self._providers.get(name)
            if prov is not None and hasattr(prov, "learning_view"):
                try:
                    view[name] = prov.learning_view()
                except Exception:  # noqa: BLE001
                    view[name] = None
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
    @staticmethod
    def _is_silence(resp: LLMResponse) -> bool:
        """True when a rung 'succeeded' without actually saying anything.

        A rung can fail in two ways, and only one of them raises. A freshly-promoted ``self`` model
        whose sampler dead-ends, or a local runtime that decodes straight to a stop token, returns a
        perfectly well-formed response whose ``text`` is empty. That is not an answer — but because
        it is not an exception either, the ladder used to stop there and hand the caller silence,
        skipping the rungs (including her guaranteed native floor) that would have spoken.
        """
        return not (resp.text or "").strip()

    def complete(self, req: LLMRequest) -> LLMResponse:
        if self.settings.llm.provider.value == "auto":
            # walk the ladder honestly: each failed rung falls to the next backend, ending at her
            # always-on native own-brain; the response's ``provider`` field always names who answered.
            last: Optional[LLMError] = None
            silent: Optional[LLMResponse] = None
            for prov in self._auto_ladder() or [self._native]:
                try:
                    resp = self._call_with_resilience(prov, req)
                except LLMError as exc:
                    self._warn_fallback(prov.name, exc)
                    last = exc
                    continue
                if self._is_silence(resp):
                    # an empty answer is a failed rung, not a cheap one — keep walking
                    self._warn_fallback(prov.name, LLMError(
                        "provider returned an empty answer",
                        context={"provider": prov.name, "model": resp.model}))
                    silent = silent or resp
                    continue
                return resp
            if silent is not None:
                return silent          # every rung was mute: return one honestly rather than raise
            raise last or LLMError("no provider available on the auto ladder",
                                   context={"provider": "auto"})
        provider = self.chosen_provider()
        try:
            resp = self._call_with_resilience(provider, req)
        except LLMError as exc:
            if provider is not self._native:
                self._warn_fallback(provider.name, exc)
                return self._native.complete(req)
            raise
        if self._is_silence(resp) and provider is not self._native:
            # same treatment for a pinned rung: silence falls to her floor, which never is
            self._warn_fallback(provider.name, LLMError(
                "provider returned an empty answer",
                context={"provider": provider.name, "model": resp.model}))
            fallback = self._native.complete(req)
            return fallback if not self._is_silence(fallback) else resp
        return resp

    def _warn_fallback(self, provider_name: str, exc: Exception) -> None:
        """Record + surface a provider failure that forces a fallback — never silently.

        The warning carries the real upstream reason (billing_error, timeout, 4xx …) so an
        operator immediately sees why the primary model went quiet. Throttled to one log line
        per provider+reason per 60s: a deep-reasoning turn can retry dozens of times, and the
        point is visibility, not spam. Best-effort — this must never break a completion."""
        try:
            reason = str(exc) or exc.__class__.__name__
            self.last_fallback = {"provider": provider_name, "error": reason,
                                  "ts": time.time()}
            key = f"{provider_name}:{reason}"
            now = time.monotonic()
            if now - self._warned_at.get(key, -1e9) >= 60.0:
                self._warned_at[key] = now
                log.warning("LLM provider '%s' failed — falling back to her own brain: %s",
                            provider_name, reason)
        except Exception:  # noqa: BLE001
            pass

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
    assert set(status) == {"litertlm", "self", "native"}
    for p in ("litertlm", "self", "native"):
        assert p in status, f"provider '{p}' must be registered"
    assert status["self"] is False       # no model trained/promoted yet on a bare machine
    # TEST seals her on-device primary too: no 2.4 GB load inside a hermetic run
    assert status["litertlm"] is False
    # her PRIMARY on-device brain leads the ladder, her own brain is always its floor
    assert LLM._AUTO_LADDER == ("litertlm", "self", "native")
    assert LLM._AUTO_LADDER[0] == "litertlm" and LLM._AUTO_LADDER[-1] == "native"
    # the rungs that are HERS are the ones that run in-process — top and bottom of the ladder
    from nyxara.kernel.config import OWN_PROVIDERS
    # every rung is hers now — the cloud providers were removed entirely
    assert set(OWN_PROVIDERS) == set(LLM._AUTO_LADDER)
    print("litertlm/self/native : registered; every rung in-process; degrade honestly ✓")

    # a missing weights file is honest unavailability, never a crash (the bare-machine path)
    probe = LiteRTLMProvider(settings)
    assert probe.available() is False
    try:
        probe.complete(LLMRequest.from_prompt("should not run"))
        raise SystemExit("ERROR: unavailable provider must raise")
    except LLMError:
        print("litertlm unavailable      : raises LLMError, ladder moves on ✓")

    print("\nALL SELF-TESTS PASSED ✓")
