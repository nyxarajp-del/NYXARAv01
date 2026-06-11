"""Tests for the Qwen3 (local, open-source) and Groq (cloud, OpenAI-compatible) backends.

These mirror the conventions in ``test_transformers_provider.py``: providers must be
registered, report availability *honestly* (deps + keys), and never silently impersonate
each other. Real inference is skipped unless the heavy deps / keys are actually present.
"""

from __future__ import annotations

import pytest

from nyxara.growth.foundry_models import _HAS_TORCH
from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, GroqProvider, LLMRequest, QwenProvider


def _settings():
    return NyxaraSettings.for_profile(Profile.TEST)


# -------------------- registration & enum wiring -------------------- #
def test_new_providers_registered():
    status = LLM(settings=_settings()).provider_status()
    assert "qwen" in status and "groq" in status


def test_qwen_groq_enum_values_exist():
    assert LLMProvider.QWEN.value == "qwen"
    assert LLMProvider.GROQ.value == "groq"


def test_active_model_mapping():
    s = _settings()
    s.llm.provider = LLMProvider.QWEN
    assert s.llm.active_model() == s.llm.qwen_model == "Qwen/Qwen3-4B"
    s.llm.provider = LLMProvider.GROQ
    assert s.llm.active_model() == s.llm.groq_model == "openai/gpt-oss-120b"


# -------------------- Groq: honest availability + key routing -------------------- #
def test_groq_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = _settings()
    settings.llm.groq_api_key = None
    assert GroqProvider(settings).available() is False


def test_groq_available_with_key(monkeypatch):
    try:
        import openai  # noqa: F401
    except Exception:
        pytest.skip("openai SDK not installed")
    from pydantic import SecretStr

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = _settings()
    settings.llm.groq_api_key = SecretStr("gsk-test-key")
    assert GroqProvider(settings).available() is True


def test_groq_default_model_and_base_url():
    settings = _settings()
    prov = GroqProvider(settings)
    assert prov.default_model() == "openai/gpt-oss-120b"
    assert settings.llm.groq_base_url.startswith("https://api.groq.com")


def test_groq_key_routed_by_active_key(monkeypatch):
    from pydantic import SecretStr

    settings = _settings()
    settings.llm.provider = LLMProvider.GROQ
    settings.llm.groq_api_key = SecretStr("gsk-secret")
    assert settings.llm.active_key().get_secret_value() == "gsk-secret"


def test_groq_secret_redacted(monkeypatch):
    from pydantic import SecretStr

    settings = _settings()
    settings.llm.groq_api_key = SecretStr("gsk-SECRET")
    assert "SECRET" not in settings.to_json(redact=True)
    assert settings.redacted()["llm"]["groq_api_key"] == "***REDACTED***"


# -------------------- Qwen: honest availability gated on torch -------------------- #
@pytest.mark.skipif(_HAS_TORCH, reason="torch IS installed; bare-machine path not applicable")
def test_qwen_unavailable_without_torch():
    assert QwenProvider(_settings()).available() is False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_qwen_available_with_torch():
    try:
        import transformers  # noqa: F401
    except Exception:
        pytest.skip("transformers not installed")
    assert QwenProvider(_settings()).available() is True


def test_qwen_default_model():
    assert QwenProvider(_settings()).default_model() == "Qwen/Qwen3-4B"


# -------------------- the facade never fakes a missing provider -------------------- #
def test_complete_with_groq_raises_when_unavailable(monkeypatch):
    """complete_with must answer *as the named provider* or fail — never silently mock."""
    from nyxara.kernel.errors import LLMError

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = _settings()
    settings.llm.groq_api_key = None
    llm = LLM(settings=settings)
    with pytest.raises(LLMError):
        llm.complete_with("groq", LLMRequest.from_prompt("hi"))
