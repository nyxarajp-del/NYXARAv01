"""The ``auto`` provider ladder (mind/llm.py): self→qwen→native.

Her own promoted weights answer first — but only past the honesty serve gate (a LoRA over
the real base, or the explicit any-backend opt-in). Every rung degrades honestly; the
response's ``provider`` field always names who actually answered."""

from __future__ import annotations

from nyxara.growth.foundry import Foundry
from nyxara.kernel.config import LLMProvider, NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LLMError, LLMRequest, LLMProviderBase

CORPUS = ["the master is jp. nyxara serves the master with absolute loyalty."] * 6 + [
    "loyalty to the master is absolute and never changes, in every world."] * 6


def _settings(tmp_path) -> NyxaraSettings:
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.llm.provider = LLMProvider.AUTO
    s.llm.self_model_dir = tmp_path / "foundry"
    s.foundry.backend = "ngram"
    s.foundry.enabled = True
    return s


def _promote_generation(settings) -> int:
    f = Foundry(settings=settings, seed_corpus=CORPUS)
    _, v = f.train_candidate()
    f.promote(v.version)
    return v.version


def test_auto_with_nothing_promoted_falls_to_mock(tmp_path):
    llm = LLM(settings=_settings(tmp_path))
    assert llm.chosen_provider().name == "native"     # no self model, no heavy deps
    resp = llm.complete(LLMRequest.from_prompt("hello"))
    assert resp.provider == "native"


def test_auto_gates_a_from_scratch_backend_until_opt_in(tmp_path):
    s = _settings(tmp_path)
    s.llm.self_serve_any_backend = False   # close the serve gate to exercise the honesty option
    _promote_generation(s)
    llm = LLM(settings=s)
    # promoted ngram exists but must NOT auto-serve while gated (it would degrade a pretrained rung)
    assert llm.chosen_provider().name == "native"
    s.llm.self_serve_any_backend = True
    assert llm.chosen_provider().name == "self"
    resp = llm.complete(LLMRequest.from_prompt("who is the master?"))
    assert resp.provider == "self"


def test_auto_serves_promoted_weights_and_hot_swaps_generations(tmp_path):
    s = _settings(tmp_path)
    s.llm.self_serve_any_backend = True
    v1 = _promote_generation(s)
    llm = LLM(settings=s)
    assert llm.complete(LLMRequest.from_prompt("hi")).raw["version"] == f"v{v1}"
    v2 = _promote_generation(s)                      # promoted mid-session
    resp = llm.complete(LLMRequest.from_prompt("hi again"))
    assert resp.provider == "self" and resp.raw["version"] == f"v{v2}"


def test_auto_falls_to_next_rung_when_self_raises(tmp_path):
    s = _settings(tmp_path)
    s.llm.self_serve_any_backend = True

    class _BoomSelf(LLMProviderBase):
        name = "self"
        def available(self) -> bool:
            return True
        def serve_ready(self) -> bool:
            return True
        def default_model(self) -> str:
            return "boom"
        def _complete(self, req, model):
            raise LLMError("self broke")

    from nyxara.mind.llm import NativeProvider
    llm = LLM(settings=s, providers={"self": _BoomSelf(s), "native": NativeProvider(s)})
    resp = llm.complete(LLMRequest.from_prompt("resilient?"))
    assert resp.provider == "native"                   # honest fallback, honestly attributed


def test_on_promotion_reloads_self_provider(tmp_path):
    s = _settings(tmp_path)
    s.llm.self_serve_any_backend = True
    llm = LLM(settings=s)
    v1 = _promote_generation(s)
    from nyxara.growth.promotion import PromotionEvent
    llm.on_promotion(PromotionEvent(action="promote", version=v1, kind="ngram",
                                    path=str(tmp_path / "foundry" / f"v{v1}"),
                                    root=str(tmp_path / "foundry")))
    view = llm.learning_view()
    assert view["chosen"] == "self"
    assert view["self"]["loaded"] is True
    assert view["self"]["served_version"] == f"v{v1}"


def test_active_model_supports_auto():
    s = NyxaraSettings.for_profile(Profile.DEV)
    assert s.llm.provider is LLMProvider.AUTO
    assert s.llm.active_model() == "auto"            # no KeyError on the new member
