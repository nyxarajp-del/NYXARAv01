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
    WorldModelBackend,
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


def test_test_profile_forces_native_llm():
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.llm.provider is LLMProvider.NATIVE
    assert s.observability.telemetry_enabled is False


def test_no_provider_can_reach_the_network():
    """The cloud rungs were removed, so the old 'every keyed provider is off under TEST' sweep has
    nothing to sweep. What it guarded is now structural — assert THAT, so a re-introduced cloud
    provider fails here instead of quietly sending the suite out over the wire."""
    from nyxara.kernel.config import LLMConfig

    keyed = [f for f in LLMConfig.model_fields if f.endswith("_api_key")]
    assert not keyed, f"a provider grew an API key again: {keyed} — give it a TEST kill-switch"
    assert not [f for f in LLMConfig.model_fields if f.endswith("_base_url")]
    assert NyxaraSettings.for_profile(Profile.TEST).llm.active_key() is None

def test_test_profile_seals_the_on_device_primary():
    """Her primary rung is local, so the key-derived sweep above cannot see it — check it by name.

    It is not a network risk; it is a 2.4 GB model load and, on a machine where the weights happen
    to be present, a real Gemma answering in place of a test's scripted provider."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.llm.litertlm_enabled is False
    assert s.llm.litertlm_auto_download is False


def test_own_providers_are_the_in_process_rungs():
    """One list answers 'is this hers or a teacher?' everywhere — keep it in step with the ladder."""
    from nyxara.kernel.config import OWN_PROVIDERS
    from nyxara.mind.llm import LLM

    assert OWN_PROVIDERS == ("litertlm", "self", "native")
    assert set(OWN_PROVIDERS) <= set(LLM._AUTO_LADDER)
    for name in OWN_PROVIDERS:            # nothing of hers is reached over a wire, so none has a key
        s = NyxaraSettings()
        s.llm.provider = LLMProvider(name)
        assert s.llm.active_key() is None
    # OWN_PROVIDERS is now the whole ladder — that IS the claim: nothing she runs is remote.
    assert set(OWN_PROVIDERS) == set(LLM._AUTO_LADDER)


def test_real_learning_defaults_on():
    """Real, weight-changing learning is the default posture (the closed loop)."""
    # Assert the CODE defaults, not a live settings object: tests/conftest.py deliberately pins
    # NYXARA_FOUNDRY__ENABLED=false for the whole suite (the foundry trains, and it was the source
    # of CI's multi-hour runs), so reading a constructed settings here was asserting against the
    # harness rather than against the shipped posture. This failed before the max-power change and
    # is fixed here rather than left red.
    from nyxara.kernel.config import FoundryConfig
    assert FoundryConfig.model_fields["enabled"].default is True   # on EVERY machine, not torch-gated
    s = NyxaraSettings.for_profile(Profile.DEV)
    # lora_requires_gpu is now ON by explicit instruction. It is a RESTRICTION, not a capability:
    # DistilGPT-2 LoRA-tunes fine on a CPU, and with this on a CPU-only box will refuse to.
    assert s.foundry.lora_requires_gpu is True
    assert s.autoforge.enabled is True
    assert s.autoforge.min_examples == 10
    assert s.flywheel.correction_weight == 3
    assert s.llm.self_serve_any_backend is True      # she serves her OWN forged brain from first boot
    assert s.llm.self_reload_lean is True


def test_test_profile_seals_the_foundry():
    """The hermetic suite never forges to disk; tests that want it opt in explicitly."""
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.foundry.enabled is False


def test_llm_active_model_and_key():
    """Every provider resolves to a model name, and none of them carries a key any more."""
    s = NyxaraSettings()
    s.llm.provider = LLMProvider.LITERTLM
    assert s.llm.active_model() == s.llm.litertlm_model == "gemma-4-E2B-it-litertlm"
    assert s.llm.active_key() is None
    s.llm.provider = LLMProvider.SELF
    assert s.llm.active_model() == "nyxara-self"
    assert s.llm.active_key() is None
    s.llm.provider = LLMProvider.NATIVE
    assert s.llm.active_model() == "nyxara-native"
    assert s.llm.active_key() is None

def test_litertlm_is_primary_and_foundry_base_is_local():
    """The shipped defaults put her ON-DEVICE brain first while the foundry LoRA-tunes its own base."""
    s = NyxaraSettings()
    # serving: ``auto`` ships as the default, which makes her on-device ``litertlm`` brain her PRIMARY
    # model while keeping the degradation path intact (a *pinned* rung would skip the other rungs and
    # drop straight to her native own-brain). Either way the facade degrades to her own brains when
    # nothing above is reachable, keeping her sovereign.
    assert s.llm.provider is LLMProvider.AUTO
    assert s.llm.active_model() == "auto"
    # her primary needs no endpoint and no key — that is what makes it un-outage-able.
    # Asserted against the SCHEMA default, not the instance: the suite-wide conftest pins
    # ``NYXARA_LLM__LITERTLM_ENABLED=false`` for hermeticity, so a live instance here is honestly
    # off. What ships enabled is the field's default, and that is what this line is about.
    from nyxara.kernel.config import LLMConfig
    assert LLMConfig.model_fields["litertlm_enabled"].default is True
    assert s.llm.litertlm_repo_id == "jamarag/gemma-4-E2B-it-ultra-uncensored-heretic-litertlm"
    assert s.llm.litertlm_filename == "model.litertlm"
    assert s.llm.litertlm_backend == "cpu"
    # Gemma 4's turn markers, not Gemma 3's — the wrong pair makes long prompts echo forever
    assert "<|turn>" in s.llm.litertlm_chat_template
    assert "<start_of_turn>" not in s.llm.litertlm_chat_template
    s.llm.provider = LLMProvider.LITERTLM
    assert s.llm.active_model() == "gemma-4-E2B-it-litertlm"
    assert s.llm.active_key() is None
    # nothing sits beneath it but her own brains — no endpoints, no keys, no cloud
    from nyxara.kernel.config import OWN_PROVIDERS
    assert set(OWN_PROVIDERS) == {"litertlm", "self", "native"}
    s.llm.provider = LLMProvider.AUTO
    # training: the foundry LoRA-tunes its own DistilGPT-2 base — everything above it is hers
    assert s.foundry.base_model == "distilgpt2"
    # Quantisation and remote code are now ON by default: max_power is the default posture and
    # its crank sets both. 8-bit stays off because the validator forbids it alongside 4-bit.
    assert s.foundry.load_in_4bit is True
    assert s.foundry.load_in_8bit is False
    assert s.foundry.trust_remote_code is True
    # DistilGPT-2 (GPT-2 arch) uses Conv1D names — the default pins attention (c_attn) + MLP (c_fc/c_proj)
    assert s.foundry.lora_target_modules == ["c_attn", "c_proj", "c_fc"]


def test_resource_limits_validation():
    with pytest.raises(ValidationError):
        ResourceLimits(max_concurrent_tasks=0)
    with pytest.raises(ValidationError):
        # event queue smaller than concurrency is incoherent
        ResourceLimits(max_concurrent_tasks=100, max_event_queue=16)


def test_secret_redaction_never_leaks():
    s = NyxaraSettings()
    s.server.api_token = SecretStr("tok-TOPSECRET")
    red = s.redacted()
    assert red["server"]["api_token"] == "***REDACTED***"
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
    monkeypatch.setenv("NYXARA_LLM__PROVIDER", "litertlm")
    monkeypatch.setenv("NYXARA_RESOURCES__MAX_CONCURRENT_TASKS", "128")
    s = NyxaraSettings()
    assert s.profile is Profile.PROD
    assert s.llm.provider is LLMProvider.LITERTLM
    assert s.resources.max_concurrent_tasks == 128


def test_native_reasoning_defaults_and_env_override(monkeypatch):
    s = NyxaraSettings()
    assert s.native_reasoning.enabled is True
    assert 0.0 <= s.native_reasoning.min_confidence <= 1.0
    assert s.native_reasoning.multi_hop and s.native_reasoning.self_improve
    assert s.causal.incremental_discovery is True
    assert s.causal.observe_stimulus_topics is True
    monkeypatch.setenv("NYXARA_NATIVE_REASONING__ENABLED", "false")
    monkeypatch.setenv("NYXARA_NATIVE_REASONING__MIN_CONFIDENCE", "0.6")
    monkeypatch.setenv("NYXARA_CAUSAL__MAX_STIMULUS_TOPICS", "5")
    s2 = NyxaraSettings()
    assert s2.native_reasoning.enabled is False
    assert s2.native_reasoning.min_confidence == 0.6
    assert s2.causal.max_stimulus_topics == 5


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


def test_world_model_defaults_are_jepa_max():
    wm = NyxaraSettings().world_model
    assert wm.enabled is True
    assert wm.backend is WorldModelBackend.AUTO      # auto → jepa when numpy present
    assert wm.latent_dim == 64 and wm.coarse_dim == 16
    assert wm.n_predictors == 5
    assert wm.horizons == [1, 2, 4, 8, 16]
    assert wm.coarse_from_horizon == 8
    assert wm.persist is True


def test_world_model_env_override(monkeypatch):
    monkeypatch.setenv("NYXARA_WORLD_MODEL__BACKEND", "knn")
    monkeypatch.setenv("NYXARA_WORLD_MODEL__LATENT_DIM", "32")
    s = NyxaraSettings()
    assert s.world_model.backend is WorldModelBackend.KNN
    assert s.world_model.latent_dim == 32


def test_world_model_backend_enum_members():
    assert {b.value for b in WorldModelBackend} == {
        "auto", "jepa", "transfer", "ensemble", "neural", "knn"}


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
