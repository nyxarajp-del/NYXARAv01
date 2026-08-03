"""SelfProvider hot-reload — the serve half of the train→serve loop (mind/llm.py).

The SAME provider instance must serve v2 the moment the pointer moves (in-process or
cross-process), keep serving the old weights when a fresh load fails, and report its
state truthfully."""

from __future__ import annotations

import pytest

from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import LLMError, LLMRequest, SelfProvider

CORPUS = ["the master is jp. nyxara serves the master with absolute loyalty."] * 6 + [
    "loyalty to the master is absolute and never changes, in every world."] * 6


def _settings(tmp_path) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.self_model_dir = tmp_path / "foundry"
    s.foundry.backend = "ngram"
    s.foundry.enabled = True
    return s


def _promote_generation(settings) -> int:
    f = Foundry(settings=settings, seed_corpus=CORPUS)
    _, v = f.train_candidate()
    f.promote(v.version)
    return v.version


def test_serves_promoted_model_with_version_meta(tmp_path):
    s = _settings(tmp_path)
    v1 = _promote_generation(s)
    prov = SelfProvider(s)
    assert prov.available()
    resp = prov.complete(LLMRequest.from_prompt("who is the master?"))
    assert resp.raw["version"] == f"v{v1}"
    assert resp.raw["self"] is True


def test_same_instance_hot_reloads_v2_on_next_call(tmp_path):
    s = _settings(tmp_path)
    v1 = _promote_generation(s)
    prov = SelfProvider(s)
    assert prov.complete(LLMRequest.from_prompt("hello")).raw["version"] == f"v{v1}"
    # a SECOND generation is promoted while the provider is live (any process could do it)
    v2 = _promote_generation(s)
    resp = prov.complete(LLMRequest.from_prompt("hello again"))
    assert resp.raw["version"] == f"v{v2}"          # the same instance now serves v2


def test_forced_reload_reports_serving_state(tmp_path):
    s = _settings(tmp_path)
    prov = SelfProvider(s)
    assert prov.reload() is False                    # nothing promoted yet — honest
    _promote_generation(s)
    assert prov.reload() is True
    view = prov.learning_view()
    assert view["available"] and view["loaded"]
    assert view["served_version"] == view["active_pointer"]
    assert view["last_reload_error"] is None


def test_failed_reload_keeps_serving_old_weights(tmp_path):
    s = _settings(tmp_path)
    v1 = _promote_generation(s)
    prov = SelfProvider(s)
    prov.complete(LLMRequest.from_prompt("warm up"))
    # simulate a corrupt promotion: pointer moves to a version dir that cannot load
    (tmp_path / "foundry" / "active").write_text("v999", encoding="utf-8")
    resp = prov.complete(LLMRequest.from_prompt("still there?"))
    assert resp.raw["version"] == f"v{v1}"           # old weights kept serving
    assert prov.learning_view()["last_reload_error"] is not None


def test_no_model_at_all_raises_honestly(tmp_path):
    s = _settings(tmp_path)
    prov = SelfProvider(s)
    (tmp_path / "foundry").mkdir(parents=True)
    (tmp_path / "foundry" / "active").write_text("v42", encoding="utf-8")  # dangling pointer
    with pytest.raises(LLMError):
        prov.complete(LLMRequest.from_prompt("anyone home?"))


def test_serve_ready_gate(tmp_path):
    """The gate is `self_serve_any_backend`, and it swings both ways.

    Its DEFAULT is ON (config.py: the Master's standing choice — her own forged brain serves
    the moment it exists). Turned off, the conservative LoRA-only policy comes back: an ngram
    brain must not silently replace a large pretrained model under `provider=auto`."""
    s = _settings(tmp_path)
    prov = SelfProvider(s)
    assert prov.serve_ready() is False               # nothing promoted — no gate can help
    _promote_generation(s)
    assert prov.active_kind() == "ngram"

    assert s.llm.self_serve_any_backend is True      # the shipped default
    assert prov.serve_ready() is True                # her own weights serve from first boot

    s.llm.self_serve_any_backend = False             # opt back out
    assert prov.serve_ready() is False               # ngram no longer auto-serves
