"""Network-free tests for the airouter (OpenAI-compatible) provider — mind/llm.py::AIRouterProvider.

The provider is a SUBORDINATE tool: request in → text out, no persona, no control. These tests use a
fake ``openai`` module so nothing ever touches the network, and assert (a) honest availability, (b)
request/response mapping, and (c) that it answers *as a tool* (``provider == "airouter"``).
"""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, AIRouterProvider, LLMRequest


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
        self.usage = _FakeUsage(11, 7)


class _FakeCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, **kwargs):
        self._captured.clear()
        self._captured.update(kwargs)
        return _FakeResp("forty-two")


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
    """A hermetic settings object with the cloud tool explicitly enabled (a test opting in)."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.airouter_enabled = True
    s.llm.airouter_api_key = SecretStr("test-key-123")
    s.llm.provider = LLMProvider.AIROUTER
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# Availability honesty
# --------------------------------------------------------------------------- #
def test_available_true_when_enabled_key_and_sdk(fake_openai):
    assert AIRouterProvider(_settings()).available() is True


def test_unavailable_when_disabled(fake_openai):
    assert AIRouterProvider(_settings(airouter_enabled=False)).available() is False


def test_unavailable_without_key(fake_openai):
    assert AIRouterProvider(_settings(airouter_api_key=None)).available() is False


def test_unavailable_without_sdk(monkeypatch):
    # No fake_openai fixture → ensure the import fails deterministically.
    monkeypatch.setitem(sys.modules, "openai", None)
    assert AIRouterProvider(_settings()).available() is False


def test_test_profile_disables_airouter():
    """The hermetic TEST profile must never reach the cloud."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.llm.airouter_enabled is False
    assert AIRouterProvider(s).available() is False


# --------------------------------------------------------------------------- #
# Request / response mapping — answers AS A TOOL
# --------------------------------------------------------------------------- #
def test_complete_maps_request_and_response(fake_openai):
    prov = AIRouterProvider(_settings())
    resp = prov.complete(LLMRequest.from_prompt("what is 6*7?", system="Be terse.",
                                                temperature=0.3, max_tokens=64, top_p=0.9))
    # response mapped correctly, and it answered as the airouter TOOL (not a persona)
    assert resp.text == "forty-two"
    assert resp.provider == "airouter"
    assert resp.model == "zai/glm-5"
    assert resp.usage.prompt_tokens == 11 and resp.usage.completion_tokens == 7

    # the outgoing call carried system + user messages and the per-request knobs
    cap = _FakeOpenAI.captured
    assert cap["model"] == "zai/glm-5"
    assert cap["messages"][0] == {"role": "system", "content": "Be terse."}
    assert cap["messages"][-1]["role"] == "user"
    assert cap["temperature"] == 0.3 and cap["max_tokens"] == 64 and cap["top_p"] == 0.9

    # client built against the configured cloud endpoint + key
    assert _FakeOpenAI.init_kwargs["base_url"] == "https://api.airouter.in/v1"
    assert _FakeOpenAI.init_kwargs["api_key"] == "test-key-123"


def test_json_mode_requests_json_object(fake_openai):
    prov = AIRouterProvider(_settings())
    prov.complete(LLMRequest.from_prompt("give json", json_mode=True))
    assert _FakeOpenAI.captured.get("response_format") == {"type": "json_object"}


def test_auto_ladder_prefers_airouter_when_available(fake_openai):
    s = _settings()
    s.llm.provider = LLMProvider.AUTO
    llm = LLM(settings=s)
    assert "airouter" in llm.provider_status()
    assert llm.chosen_provider().name == "airouter"
