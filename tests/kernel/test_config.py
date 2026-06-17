"""Tests for nyxara.kernel.config."""

from __future__ import annotations

import json

import pytest
from pydantic import SecretStr, ValidationError

from nyxara.kernel.config import (
    CONFIG_SCHEMA_VERSION,
    OWNER,
    FeatureFlags,
    LLMProvider,
    NyxaraSettings,
    Profile,
    ResourceLimits,
    VectorBackend,
    get_settings,
    reload_settings,
)


def test_defaults_and_owner():
    s = NyxaraSettings()
    assert s.profile is Profile.DEV
    assert s.schema_version == CONFIG_SCHEMA_VERSION
    assert s.owner.name == "Jaypal Khoja"
    assert s.owner.handle == "JP"
    assert OWNER.email == "nyxarajp@gmail.com"


def test_owner_is_frozen():
    with pytest.raises(ValidationError):
        OWNER.name = "someone else"  # type: ignore[misc]


def test_prod_hardening_forces_safety():
    s = NyxaraSettings.for_profile(Profile.PROD)
    # Safety-critical flags are forced ON regardless of input.
    assert s.features.invariant_enforcement is True
    assert s.features.audit_logging is True
    assert s.features.corrigibility is True
    assert s.guard.rule_modification_locked is True
    assert s.llm.allow_mock_fallback is False
    assert s.features.simulation_required is True
    assert s.is_prod and not s.is_dev


def test_safety_flags_cannot_be_disabled_even_in_dev():
    s = NyxaraSettings(
        profile=Profile.DEV,
        features=FeatureFlags(invariant_enforcement=False, audit_logging=False, corrigibility=False),
    )
    # post-validation hardening re-enables them
    assert s.features.invariant_enforcement is True
    assert s.features.audit_logging is True
    assert s.features.corrigibility is True


def test_test_profile_forces_mock_llm():
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.llm.provider is LLMProvider.MOCK
    assert s.observability.telemetry_enabled is False


def test_llm_active_model_and_key():
    s = NyxaraSettings()
    s.llm.provider = LLMProvider.ANTHROPIC
    assert s.llm.active_model() == s.llm.anthropic_model
    s.llm.provider = LLMProvider.OPENAI
    assert s.llm.active_model() == s.llm.openai_model
    s.llm.anthropic_api_key = SecretStr("k")
    s.llm.provider = LLMProvider.ANTHROPIC
    assert s.llm.active_key().get_secret_value() == "k"
    s.llm.provider = LLMProvider.LOCAL
    assert s.llm.active_key() is None


def test_resource_limits_validation():
    with pytest.raises(ValidationError):
        ResourceLimits(max_concurrent_tasks=0)
    with pytest.raises(ValidationError):
        # event queue smaller than concurrency is incoherent
        ResourceLimits(max_concurrent_tasks=100, max_event_queue=16)


def test_secret_redaction_never_leaks():
    s = NyxaraSettings()
    s.llm.anthropic_api_key = SecretStr("sk-ant-TOPSECRET")
    red = s.redacted()
    assert red["llm"]["anthropic_api_key"] == "***REDACTED***"
    blob = s.to_json(redact=True)
    assert "TOPSECRET" not in blob
    # redacted output must be valid JSON
    json.loads(blob)


def test_save_and_from_file_roundtrip(tmp_path):
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.memory.embedding_dim = 1024
    p = s.save(tmp_path / "cfg.json", redact=False)
    assert p.exists()
    loaded = NyxaraSettings.from_file(p)
    assert loaded.memory.embedding_dim == 1024


def test_env_override(monkeypatch):
    monkeypatch.setenv("NYXARA_PROFILE", "prod")
    monkeypatch.setenv("NYXARA_LLM__PROVIDER", "openai")
    monkeypatch.setenv("NYXARA_RESOURCES__MAX_CONCURRENT_TASKS", "128")
    s = NyxaraSettings()
    assert s.profile is Profile.PROD
    assert s.llm.provider is LLMProvider.OPENAI
    assert s.resources.max_concurrent_tasks == 128


def test_paths_ensure_creates_dirs(tmp_path):
    s = NyxaraSettings()
    s.paths.root = tmp_path / "nyx"
    # re-derive children for the new root
    from nyxara.kernel.config import PathsConfig

    s.paths = PathsConfig(root=tmp_path / "nyx")
    s.paths.ensure()
    for d in s.paths.all_dirs():
        assert d.exists()


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reload_settings_with_overrides():
    s = reload_settings(profile=Profile.PROD)
    assert s.profile is Profile.PROD
    assert get_settings().profile is Profile.PROD
    # restore default for other tests
    reload_settings(profile=Profile.DEV)


def test_vector_backend_default_is_numpy():
    assert NyxaraSettings().memory.vector_backend is VectorBackend.NUMPY


def test_web_config_defaults_are_max_and_unrestricted():
    web = NyxaraSettings().web
    assert web.search_provider == "auto"
    assert web.allow_private is True          # SSRF guard off — unrestricted reach
    assert web.injection_scan is True
    assert web.max_fetches_per_min == 10_000


def test_web_config_drives_governor_web_bucket():
    # the harden step syncs the governor's "web" rate bucket to the web config
    s = NyxaraSettings()
    assert s.resources.max_web_fetches_per_min == s.web.max_fetches_per_min


def test_web_env_override_and_secret_redaction(monkeypatch):
    monkeypatch.setenv("NYXARA_WEB__SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("NYXARA_WEB__ALLOW_PRIVATE", "false")
    monkeypatch.setenv("NYXARA_WEB__BRAVE_API_KEY", "brv-TOPSECRET")
    s = NyxaraSettings()
    assert s.web.search_provider == "brave" and s.web.allow_private is False
    assert s.web.brave_api_key.get_secret_value() == "brv-TOPSECRET"
    # the key must never leak through redacted output
    blob = s.to_json(redact=True)
    assert "TOPSECRET" not in blob
