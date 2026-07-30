"""Sovereignty invariants for the LLM faculty — mind/llm.py.

These tests encode, as executable invariants, the single constraint the Master set:

    NYXARA controls the LLM — not the LLM controlling NYXARA.

Concretely that means three things, all network-free (a fake ``openai`` module, so nothing
ever leaves the machine):

1.  **GLM-5 is a reachable cloud tool** — the airouter provider is chosen whenever it is reachable
    and no stronger rung is. (Groq leads the ``auto`` ladder; see ``test_groq_sovereignty.py`` for
    the primary-rung invariants. These tests pin airouter's own behaviour as the FALLBACK rung, so
    each test here disables groq explicitly rather than relying on the TEST profile to do it.)
2.  **She is never cloud-dependent** — the instant the cloud tool is unavailable, ``complete()``
    keeps answering from a *local* provider (down to her always-on native own-brain) and never
    raises. She uses the cloud; she never needs it.
3.  **The cloud model injects no persona** — the provider forwards only the caller's own
    system/messages. Her identity lives in ``identity/soul.py``, never in the external model.
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
        self.usage = _FakeUsage(5, 3)


class _FakeCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, **kwargs):
        self._captured.clear()
        self._captured.update(kwargs)
        return _FakeResp("cloud-answer")


class _FakeChat:
    def __init__(self, captured: dict) -> None:
        self.completions = _FakeCompletions(captured)


class _FakeOpenAI:
    captured: dict = {}

    def __init__(self, **kwargs):
        self.chat = _FakeChat(_FakeOpenAI.captured)


@pytest.fixture()
def fake_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    _FakeOpenAI.captured = {}
    yield mod


def _settings(**over) -> NyxaraSettings:
    """Hermetic settings with the airouter cloud tool explicitly enabled (a test opting in).

    ``groq`` is disabled explicitly, not left to the TEST profile: these tests are about airouter's
    own behaviour, and a precondition that important should be stated in the test rather than
    inherited — otherwise they would silently start exercising a different rung."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.provider = LLMProvider.AUTO
    s.llm.groq_enabled = False
    s.llm.airouter_enabled = True
    s.llm.airouter_api_key = SecretStr("test-key-123")
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# 1. GLM-5 is a reachable cloud tool (the fallback rung)
# --------------------------------------------------------------------------- #
def test_airouter_is_the_cloud_fallback_rung():
    """Structural guarantee: airouter sits on the ladder, directly behind her primary cloud tool."""
    assert LLM._AUTO_LADDER.index("airouter") == LLM._AUTO_LADDER.index("groq") + 1
    # and always ahead of her own local brains, which remain the floor
    assert LLM._AUTO_LADDER.index("airouter") < LLM._AUTO_LADDER.index("native")


def test_glm5_is_chosen_when_reachable(fake_openai):
    llm = LLM(settings=_settings())
    assert llm.chosen_provider().name == "airouter"
    resp = llm.complete(LLMRequest.from_prompt("2+2?"))
    assert resp.provider == "airouter" and resp.model == "zai/glm-5"
    assert resp.text == "cloud-answer"


# --------------------------------------------------------------------------- #
# 2. She is never cloud-dependent — the cloud serves her, it never steers her
# --------------------------------------------------------------------------- #
def test_degrades_to_her_own_brain_when_cloud_unavailable(fake_openai):
    """Kill the cloud tool: she must keep answering from a LOCAL provider, never raise."""
    llm = LLM(settings=_settings(airouter_enabled=False))
    # airouter is unavailable, so it is not even chosen...
    assert llm.chosen_provider().name != "airouter"
    # ...and a real completion still succeeds, answered by her own (local) brain.
    resp = llm.complete(LLMRequest.from_prompt("who is your master?"))
    assert resp.provider != "airouter"
    assert isinstance(resp.text, str) and resp.text.strip()


def test_bare_machine_floor_is_her_native_own_brain(fake_openai):
    """No key at all (bare machine): the guaranteed floor is her native own-brain, not an error."""
    s = _settings(airouter_api_key=None)
    # Model a truly bare machine: no self model promoted (fresh TEST profile) and the heavy-ML
    # inference path off, so only her always-on native own-brain remains — the guaranteed floor.
    s.features.transformers_inference = False
    llm = LLM(settings=s)
    resp = llm.complete(LLMRequest.from_prompt("hello"))
    assert resp.provider == "native"
    assert resp.text.strip()


# --------------------------------------------------------------------------- #
# 3. The cloud model injects NO persona — identity lives in identity/soul.py
# --------------------------------------------------------------------------- #
def test_cloud_provider_injects_no_persona(fake_openai):
    """With no caller system prompt, the outgoing call carries no 'you are NYXARA' system turn."""
    prov = AIRouterProvider(_settings())
    prov.complete(LLMRequest.from_prompt("say hi"))          # system=None
    messages = _FakeOpenAI.captured["messages"]
    assert all(m["role"] != "system" for m in messages)      # provider added no persona
    assert [m["role"] for m in messages] == ["user"]


def test_cloud_provider_forwards_only_callers_system(fake_openai):
    """When the kernel supplies a system prompt, the provider forwards it verbatim — nothing else."""
    prov = AIRouterProvider(_settings())
    prov.complete(LLMRequest.from_prompt("say hi", system="Be terse."))
    messages = _FakeOpenAI.captured["messages"]
    assert messages[0] == {"role": "system", "content": "Be terse."}
    assert sum(1 for m in messages if m["role"] == "system") == 1
