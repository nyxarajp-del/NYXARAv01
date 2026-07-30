"""Network-free tests for the Groq (OpenAI-compatible) provider — mind/llm.py::GroqProvider.

Groq is NYXARA's PRIMARY reachable model and still a SUBORDINATE tool: request in → text out, no
persona, no control. These tests use a fake ``openai`` module so nothing ever touches the network,
and assert (a) honest availability, (b) request/response mapping onto the wire, and (c) that it
answers *as a tool* (``provider == "groq"``).

Several of the wire assertions here encode behaviour verified against the real endpoint: the
``max_tokens`` ceiling of 32768 (above it Groq returns a 400), and that ``response_format``,
``seed`` and ``stop`` are all accepted.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest
from pydantic import SecretStr

from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, GroqProvider, LLMRequest, NativeProvider


# --------------------------------------------------------------------------- #
# A fake OpenAI SDK (installed into sys.modules) — no network, captures the call
# --------------------------------------------------------------------------- #
class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish


class _FakeUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(13, 5)


class _FakeCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, **kwargs):
        self._captured.clear()
        self._captured.update(kwargs)
        return _FakeResp("groq-answer")


class _FakeChat:
    def __init__(self, captured: dict) -> None:
        self.completions = _FakeCompletions(captured)


class _FakeOpenAI:
    captured: dict = {}
    init_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeOpenAI.init_kwargs = kwargs
        self.chat = _FakeChat(_FakeOpenAI.captured)


@pytest.fixture()
def fake_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    _FakeOpenAI.captured = {}
    _FakeOpenAI.init_kwargs = {}
    yield mod


def _settings(**over) -> NyxaraSettings:
    """A hermetic settings object with the Groq cloud tool explicitly enabled (a test opting in)."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.groq_enabled = True
    s.llm.groq_api_key = SecretStr("test-key-123")
    s.llm.provider = LLMProvider.GROQ
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# Availability honesty — a bare machine degrades, it never errors
# --------------------------------------------------------------------------- #
def test_available_true_when_enabled_key_and_sdk(fake_openai):
    assert GroqProvider(_settings()).available() is True


def test_unavailable_when_disabled(fake_openai):
    assert GroqProvider(_settings(groq_enabled=False)).available() is False


def test_unavailable_without_key(fake_openai):
    assert GroqProvider(_settings(groq_api_key=None)).available() is False


def test_unavailable_with_blank_key(fake_openai):
    assert GroqProvider(_settings(groq_api_key=SecretStr("   "))).available() is False


def test_unavailable_without_sdk(monkeypatch):
    # No fake_openai fixture → ensure the import fails deterministically.
    monkeypatch.setitem(sys.modules, "openai", None)
    assert GroqProvider(_settings()).available() is False


def test_test_profile_disables_groq():
    """The hermetic TEST profile must never reach the cloud."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.llm.groq_enabled is False
    assert GroqProvider(s).available() is False


# --------------------------------------------------------------------------- #
# Request → wire mapping
# --------------------------------------------------------------------------- #
def test_client_points_at_the_groq_endpoint(fake_openai):
    GroqProvider(_settings()).complete(LLMRequest.from_prompt("hi"))
    assert _FakeOpenAI.init_kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert _FakeOpenAI.init_kwargs["api_key"] == "test-key-123"


def test_complete_maps_request_and_response(fake_openai):
    prov = GroqProvider(_settings())
    resp = prov.complete(LLMRequest.from_prompt("2+2?", temperature=0.25, max_tokens=256,
                                                top_p=0.9))
    assert resp.provider == "groq"
    assert resp.model == "llama-3.3-70b-versatile"
    assert resp.text == "groq-answer"
    assert resp.usage.prompt_tokens == 13 and resp.usage.completion_tokens == 5
    sent = _FakeOpenAI.captured
    assert sent["model"] == "llama-3.3-70b-versatile"
    assert sent["temperature"] == 0.25
    assert sent["max_tokens"] == 256
    assert sent["top_p"] == 0.9
    assert sent["messages"] == [{"role": "user", "content": "2+2?"}]


def test_default_model_follows_config(fake_openai):
    prov = GroqProvider(_settings(groq_model="llama-3.1-8b-instant"))
    assert prov.default_model() == "llama-3.1-8b-instant"
    assert prov.complete(LLMRequest.from_prompt("hi")).model == "llama-3.1-8b-instant"


def test_complete_clamps_absurd_max_tokens(fake_openai):
    """Groq 400s on ``max_tokens`` above 32768 — clamp so no caller can silently kill the rung."""
    prov = GroqProvider(_settings())
    prov.complete(LLMRequest.from_prompt("hi", max_tokens=4_096_000))
    assert _FakeOpenAI.captured["max_tokens"] == GroqProvider._MAX_TOKENS_CEILING
    assert GroqProvider._MAX_TOKENS_CEILING == 32768


def test_json_mode_requests_json_object(fake_openai):
    prov = GroqProvider(_settings())
    prov.complete(LLMRequest.from_prompt("give me json", json_mode=True))
    assert _FakeOpenAI.captured["response_format"] == {"type": "json_object"}
    # the json nudge rides in the SYSTEM turn — it is an instruction, never a persona
    system = [m for m in _FakeOpenAI.captured["messages"] if m["role"] == "system"]
    assert len(system) == 1 and "JSON" in system[0]["content"]


def test_seed_and_stop_are_forwarded(fake_openai):
    prov = GroqProvider(_settings())
    prov.complete(LLMRequest.from_prompt("hi", seed=7, stop=("###",)))
    assert _FakeOpenAI.captured["seed"] == 7
    assert _FakeOpenAI.captured["stop"] == ["###"]


def test_raw_payload_names_the_answering_rung(fake_openai):
    resp = GroqProvider(_settings()).complete(LLMRequest.from_prompt("hi"))
    assert resp.raw["groq"] is True
    assert resp.raw["model"] == "llama-3.3-70b-versatile"


# --------------------------------------------------------------------------- #
# Resilience — the shared retry-without-extras fallback
# --------------------------------------------------------------------------- #
def test_complete_retries_without_extras_on_rejection(fake_openai):
    """A server that rejects the optional extras gets one retry without them, not a dropped call."""
    orig_create = _FakeCompletions.create
    calls: list[dict] = []

    def flaky(self, **kwargs):
        calls.append(dict(kwargs))
        if "response_format" in kwargs or "seed" in kwargs:
            raise RuntimeError("400: unsupported parameter")
        self._captured.clear()
        self._captured.update(kwargs)
        return _FakeResp("recovered")

    _FakeCompletions.create = flaky
    try:
        prov = GroqProvider(_settings())
        resp = prov.complete(LLMRequest.from_prompt("hi", json_mode=True, seed=3))
        assert resp.text == "recovered"
        assert len(calls) == 2
        assert "response_format" in calls[0] and "response_format" not in calls[1]
    finally:
        _FakeCompletions.create = orig_create


def test_cloud_failure_falls_back_loudly_not_silently(fake_openai, caplog):
    """A dead Groq rung must be *recorded* — an operator sees why the primary model went quiet."""
    orig_create = _FakeCompletions.create

    def dead(self, **kwargs):
        raise RuntimeError("billing_error: quota exhausted")

    _FakeCompletions.create = dead
    try:
        s = _settings()
        llm = LLM(settings=s, providers={"groq": GroqProvider(s),
                                         "native": NativeProvider(s)})
        with caplog.at_level(logging.WARNING, logger="nyxara.mind.llm"):
            resp = llm.complete(LLMRequest.from_prompt("hi"))
        assert resp.provider == "native"          # she keeps answering, from her own brain
        assert llm.last_fallback["provider"] == "groq"
        assert "billing_error" in llm.last_fallback["error"]
        assert any("billing_error" in r.getMessage() for r in caplog.records)
        assert any("groq" in r.getMessage() for r in caplog.records)
    finally:
        _FakeCompletions.create = orig_create
