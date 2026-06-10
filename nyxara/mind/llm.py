"""NYXARA · mind/llm.py — the STATELESS large-language-model faculty (⬆).

The single most important constraint in this file is what it *does not* do.

    The LLM is a provider, not the driver. Request in → text/JSON out.
    No memory. No tool-calls. No side-effects. No control.

The kernel/orchestrator calls *it*; it never calls back. It holds **no conversation
state** between calls — every :class:`LLMRequest` is self-contained, which is what
makes cognition replayable (kernel/replay.py) and auditable. Memory lives in
``memory/``; tool execution lives in ``agency/``; the decision to *act* on any output
belongs to the kernel after the proposal passes guards (mind/proposal.py).

Multi-provider, selected by config:

* :class:`AnthropicProvider` — Claude (anthropic SDK + ``ANTHROPIC_API_KEY``)
* :class:`OpenAIProvider`    — GPT (openai SDK + ``OPENAI_API_KEY``)
* :class:`LocalProvider`     — any OpenAI-compatible endpoint (e.g. Ollama, via httpx)
* :class:`MockProvider`      — deterministic, offline; the always-available fallback

Each adapter imports its SDK lazily and reports ``available()`` honestly, so this
module works with zero heavy deps installed (falling back to the mock).

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
    "AnthropicProvider",
    "OpenAIProvider",
    "LocalProvider",
    "TransformersProvider",
    "SelfProvider",
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
# Anthropic provider
# --------------------------------------------------------------------------- #
class AnthropicProvider(LLMProviderBase):
    name = "anthropic"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._client = None

    def _key(self) -> Optional[str]:
        import os
        k = self.settings.llm.anthropic_api_key
        return (k.get_secret_value() if k else None) or os.getenv("ANTHROPIC_API_KEY")

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return bool(self._key())

    def default_model(self) -> str:
        return self.settings.llm.anthropic_model

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._key())
        return self._client

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "model": model, "max_tokens": req.max_tokens,
            "temperature": req.temperature, "top_p": req.top_p,
            "messages": req.provider_messages(),
        }
        if req.system:
            kwargs["system"] = req.system
        if req.stop:
            kwargs["stop_sequences"] = list(req.stop)
        resp = client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        usage = Usage(prompt_tokens=getattr(resp.usage, "input_tokens", 0),
                      completion_tokens=getattr(resp.usage, "output_tokens", 0))
        return (text, getattr(resp, "stop_reason", "stop") or "stop", usage, resp)


# --------------------------------------------------------------------------- #
# OpenAI provider
# --------------------------------------------------------------------------- #
class OpenAIProvider(LLMProviderBase):
    name = "openai"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._client = None

    def _key(self) -> Optional[str]:
        import os
        k = self.settings.llm.openai_api_key
        return (k.get_secret_value() if k else None) or os.getenv("OPENAI_API_KEY")

    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except Exception:
            return False
        return bool(self._key())

    def default_model(self) -> str:
        return self.settings.llm.openai_model

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._key())
        return self._client

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        client = self._ensure_client()
        messages = req.provider_messages()
        if req.system:
            messages = [{"role": "system", "content": req.system}] + messages
        kwargs: Dict[str, Any] = {
            "model": model, "messages": messages, "temperature": req.temperature,
            "max_tokens": req.max_tokens, "top_p": req.top_p,
        }
        if req.stop:
            kwargs["stop"] = list(req.stop)
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if req.seed is not None:
            kwargs["seed"] = req.seed
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = getattr(resp, "usage", None)
        usage = Usage(prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
                      completion_tokens=getattr(u, "completion_tokens", 0) if u else 0)
        return (text, choice.finish_reason or "stop", usage, resp)


# --------------------------------------------------------------------------- #
# Local provider — any OpenAI-compatible HTTP endpoint
# --------------------------------------------------------------------------- #
class LocalProvider(LLMProviderBase):
    name = "local"

    def available(self) -> bool:
        try:
            import httpx  # noqa: F401
        except Exception:
            return False
        return True

    def default_model(self) -> str:
        return self.settings.llm.local_model

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        import httpx
        messages = req.provider_messages()
        if req.system:
            messages = [{"role": "system", "content": req.system}] + messages
        payload = {"model": model, "messages": messages, "temperature": req.temperature,
                   "max_tokens": req.max_tokens, "top_p": req.top_p, "stream": False}
        url = self.settings.llm.local_base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.settings.llm.request_timeout_s) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        u = data.get("usage", {})
        usage = Usage(prompt_tokens=u.get("prompt_tokens", 0),
                      completion_tokens=u.get("completion_tokens", 0))
        return (text, choice.get("finish_reason", "stop"), usage, data)


# --------------------------------------------------------------------------- #
# Transformers provider — in-process open-source model (HuggingFace)
# --------------------------------------------------------------------------- #
class TransformersProvider(LLMProviderBase):
    """Run an open-source model in-process via HuggingFace ``transformers``.

    The LLM stays a tool NYXARA *uses*: request in -> text out, no state, no control.
    Heavy deps (``transformers`` + ``torch``) are imported lazily and reported honestly,
    so a bare machine degrades to the mock rather than erroring.
    """

    name = "transformers"

    def __init__(self, settings: Optional[NyxaraSettings] = None) -> None:
        super().__init__(settings)
        self._pipe = None
        self._pipe_model: Optional[str] = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    def default_model(self) -> str:
        return self.settings.llm.transformers_model

    def _ensure_pipe(self, model: str):
        if self._pipe is None or self._pipe_model != model:
            from transformers import pipeline
            device = self.settings.llm.transformers_device or None
            self._pipe = pipeline("text-generation", model=model,
                                  device_map=device if device else None)
            self._pipe_model = model
        return self._pipe

    def _complete(self, req: LLMRequest, model: str) -> Tuple[str, str, Usage, Any]:
        pipe = self._ensure_pipe(model)
        prompt = (req.system + "\n\n" if req.system else "") + req.last_user()
        out = pipe(prompt, max_new_tokens=req.max_tokens, temperature=max(req.temperature, 1e-3),
                   top_p=req.top_p, do_sample=req.temperature > 0,
                   return_full_text=False)
        text = out[0].get("generated_text", "") if out else ""
        usage = Usage(prompt_tokens=estimate_tokens(prompt),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop", usage, out)


# --------------------------------------------------------------------------- #
# Self provider — NYXARA's OWN model, trained & promoted by the foundry
# --------------------------------------------------------------------------- #
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
        prompt = (req.system + "\n" if req.system else "") + req.last_user()
        text = self._lm.generate(prompt, max_tokens=req.max_tokens)
        usage = Usage(prompt_tokens=estimate_tokens(prompt),
                      completion_tokens=estimate_tokens(text))
        return (text, "stop", usage, {"self": True, "kind": self._lm.kind})


# --------------------------------------------------------------------------- #
# The stateless facade the kernel calls
# --------------------------------------------------------------------------- #
_PROVIDER_CLASSES = {
    ProviderName.ANTHROPIC: AnthropicProvider,
    ProviderName.OPENAI: OpenAIProvider,
    ProviderName.LOCAL: LocalProvider,
    ProviderName.TRANSFORMERS: TransformersProvider,
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

    # adapters report availability honestly (no keys in TEST -> only mock/local)
    status = llm.provider_status()
    print(f"\nadapter availability : {status}")
    # the open-source + self-built providers are registered and degrade honestly
    assert "transformers" in status and "self" in status
    assert status["self"] is False     # no model trained/promoted yet on a bare machine
    print("transformers/self    : registered; both unavailable on a bare machine ✓")

    print("\nALL SELF-TESTS PASSED ✓")
