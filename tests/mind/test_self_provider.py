"""Tests for the self-built (SelfProvider) LLM backend."""

from __future__ import annotations


from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, SelfProvider


def _settings():
    return NyxaraSettings.for_profile(Profile.TEST)


# -------------------- providers are registered & honest -------------------- #
def test_self_provider_registered():
    status = LLM(settings=_settings()).provider_status()
    assert "self" in status


def test_self_provider_unavailable_without_promoted_model(tmp_path):
    settings = _settings()
    settings.llm.self_model_dir = tmp_path / "foundry"   # no active pointer yet
    assert SelfProvider(settings).available() is False


def test_self_enum_value_exists():
    assert LLMProvider.SELF.value == "self"


# -------------------- self-built model becomes a usable provider -------------------- #
def test_self_provider_serves_promoted_model(tmp_path):
    from nyxara.growth.foundry import Foundry
    from nyxara.growth.learn import Experience, ReplayBuffer

    settings = _settings()
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"   # fast, torch-independent
    rb = ReplayBuffer(capacity=50)
    for _ in range(20):
        rb.add(Experience(action="serve the master", features={}, reward=1.0,
                          context="nyxara is loyal to jp the master always"))
    f = Foundry(settings=settings, replay=rb)
    f.self_improve(generations=1)          # trains from scratch + promotes

    from nyxara.mind.llm import LLMRequest
    prov = SelfProvider(settings)
    assert prov.available() is True        # a model was trained AND promoted
    resp = prov.complete(LLMRequest.from_prompt("nyxara is", max_tokens=16))
    assert isinstance(resp.text, str) and resp.provider == "self"
