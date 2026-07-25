"""Tests for the air-gapped mind — guard/isolation_envelope.py (Part K).

Verifies (a) the envelope abstracts named identity/secrets bijectively and re-hydrates losslessly, and
(b) end-to-end through AIRouterProvider only ABSTRACT tokens cross the wire while the caller still gets a
concrete, locally re-hydrated answer.
"""
from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from nyxara.guard.isolation_envelope import IsolationEnvelope
from nyxara.kernel.config import OWNER, LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import AIRouterProvider, LLMRequest


def _settings(**over) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.airouter_enabled = True
    s.llm.airouter_api_key = SecretStr("k")
    s.llm.provider = LLMProvider.AIROUTER
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


# --------------------------------------------------------------------------- #
# Unit — bijective abstraction / rehydration
# --------------------------------------------------------------------------- #
def test_abstract_hides_identity_and_secrets():
    env = IsolationEnvelope(_settings(), extra_secrets=["CorePassphrase42"])
    text = (f"I am NYXARA, serving {OWNER.name} ({OWNER.handle}, {OWNER.email}). "
            f"secret=CorePassphrase42")
    abstract = env.abstract(text)
    for leaked in ("NYXARA", OWNER.name, OWNER.handle, OWNER.email, "CorePassphrase42"):
        assert leaked not in abstract, f"{leaked!r} leaked past the air-gap"
    # round-trip is lossless
    assert env.rehydrate(abstract) == text


def test_rehydrate_maps_tokens_back():
    env = IsolationEnvelope(_settings())
    abstract = env.abstract("NYXARA")
    assert abstract != "NYXARA" and abstract in env.mapping().values()
    assert env.rehydrate(abstract) == "NYXARA"


def test_disabled_is_passthrough():
    env = IsolationEnvelope(_settings(airouter_isolation=False))
    assert env.enabled() is False
    assert env.abstract("NYXARA and Jaypal Khoja") == "NYXARA and Jaypal Khoja"


def test_word_boundary_avoids_false_hits():
    # "JP" (the handle) must not be abstracted inside an unrelated word like "JPEG".
    env = IsolationEnvelope(_settings())
    out = env.abstract("Open the JPEG file")
    assert out == "Open the JPEG file"


# --------------------------------------------------------------------------- #
# Integration — only abstract tokens cross the wire; caller gets concrete answer
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, c): self.content = c


class _Choice:
    def __init__(self, c): self.message = _Msg(c); self.finish_reason = "stop"


class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]; self.usage = None


class _Echo:
    """Echoes the abstracted user content back — lets us inspect the wire and the rehydration."""
    captured: dict = {}

    def __init__(self, **kw): self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        _Echo.captured = kwargs
        return _Resp(f"Regarding {kwargs['messages'][-1]['content']}, done.")


@pytest.fixture()
def fake_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _Echo
    monkeypatch.setitem(sys.modules, "openai", mod)
    _Echo.captured = {}
    yield


def test_only_abstract_tokens_cross_the_wire(fake_openai):
    prov = AIRouterProvider(_settings())
    resp = prov.complete(LLMRequest.from_prompt(f"As NYXARA, greet {OWNER.name}."))

    wire = str(_Echo.captured["messages"])
    # nothing sensitive left the process
    assert "NYXARA" not in wire and OWNER.name not in wire
    # but the caller gets a concrete, locally re-hydrated answer
    assert "NYXARA" in resp.text and OWNER.name in resp.text
    assert resp.provider == "airouter"
