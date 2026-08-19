"""The cortex rung — mind/llm.py::GGUFProvider.

A fake ``llama_cpp`` is injected into ``sys.modules``, so nothing here loads 5.6 GB or needs the
wheel installed. What is pinned is everything that decides whether the rung is on the ladder, and
everything that decides whether she is honest about what she can do:

* ``available()`` is cheap and never lies — no binding, no weights, or switched off, all say no;
* a host that cannot load the weights takes the rung OFF the ladder instead of failing every turn;
* a projector she cannot drive is reported with its reason, never degraded into answering about an
  image from the surrounding text;
* the provider stays stateless across requests even though llama.cpp keeps a KV cache;
* a build with a different signature loses a parameter, not the whole cortex.
"""
from __future__ import annotations

import sys
import types

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import GGUFProvider, LLMError, LLMRequest

_MODEL = "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"
_MMPROJ = "mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf"


# --------------------------------------------------------------------------- #
# A llama.cpp that does what a test needs
# --------------------------------------------------------------------------- #
class _FakeLlama:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.resets = 0
        self.calls = []
        self.reply = "the disk filled up"
        self.finish = "stop"
        type(self).instances.append(self)

    def reset(self):
        self.resets += 1

    def create_chat_completion(self, *, messages, max_tokens=256, temperature=0.7, top_p=1.0,
                               top_k=40, seed=0, stop=None, response_format=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "seed": seed,
                           "stop": stop, "response_format": response_format})
        return {"choices": [{"message": {"content": self.reply}, "finish_reason": self.finish}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7}}

    def close(self):
        pass


def _install_binding(monkeypatch, *, chat_handlers=()):
    module = types.ModuleType("llama_cpp")
    module.Llama = _FakeLlama
    chat_format = types.ModuleType("llama_cpp.llama_chat_format")
    for name in chat_handlers:
        setattr(chat_format, name, lambda **kw: types.SimpleNamespace(**kw))
    module.llama_chat_format = chat_format
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat_format)
    return module


@pytest.fixture(autouse=True)
def _clean_process_cache():
    GGUFProvider._MODELS.clear()
    _FakeLlama.instances.clear()
    yield
    GGUFProvider._MODELS.clear()
    _FakeLlama.instances.clear()


def _settings(tmp_path, *, weights=True, projector=False, **over):
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.qwythos_model_path = tmp_path / _MODEL
    s.llm.qwythos_mmproj_path = tmp_path / _MMPROJ
    s.llm.qwythos_enabled = True
    s.llm.qwythos_auto_download = False
    s.llm.qwythos_vision_enabled = False
    for k, v in over.items():
        setattr(s.llm, k, v)
    if weights:
        (tmp_path / _MODEL).write_bytes(b"weights")
    if projector:
        (tmp_path / _MMPROJ).write_bytes(b"projector")
    return s


# --------------------------------------------------------------------------- #
# Availability is cheap and honest
# --------------------------------------------------------------------------- #
def test_no_binding_means_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    provider = GGUFProvider(_settings(tmp_path))
    assert provider.available() is False
    assert "llama_cpp not installed" in provider.learning_view()["reason_unavailable"]


def test_no_weights_means_unavailable(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path, weights=False))
    assert provider.available() is False
    assert "weights missing" in provider.learning_view()["reason_unavailable"]
    assert "fetch_qwythos_model.py" in provider.learning_view()["reason_unavailable"]


def test_switched_off_means_unavailable(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path, qwythos_enabled=False))
    assert provider.available() is False
    assert provider.learning_view()["reason_unavailable"] == "disabled by config"


def test_available_never_loads_the_weights(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path))
    assert provider.available() is True
    assert _FakeLlama.instances == [], "availability must stay a cheap file check"
    assert provider.learning_view()["model_loaded"] is False


# --------------------------------------------------------------------------- #
# A host that cannot load takes the rung off the ladder
# --------------------------------------------------------------------------- #
def test_a_load_failure_latches_and_removes_the_rung(tmp_path, monkeypatch):
    module = _install_binding(monkeypatch)

    def _explodes(**kwargs):
        raise MemoryError("not enough RAM for 5.6 GB")

    module.Llama = _explodes
    provider = GGUFProvider(_settings(tmp_path))
    assert provider.available() is True
    with pytest.raises(LLMError):
        provider.complete(LLMRequest.from_prompt("why?"))
    assert provider.available() is False, "a broken host must cost one warning, not one per turn"
    assert "unusable on this host" in provider.learning_view()["reason_unavailable"]


def test_reset_clears_the_latch_so_a_fixed_host_recovers(tmp_path, monkeypatch):
    module = _install_binding(monkeypatch)
    module.Llama = lambda **kw: (_ for _ in ()).throw(MemoryError("no RAM"))
    provider = GGUFProvider(_settings(tmp_path))
    with pytest.raises(LLMError):
        provider.complete(LLMRequest.from_prompt("why?"))
    assert provider.available() is False
    module.Llama = _FakeLlama
    provider.reset()
    assert provider.available() is True


def test_warm_never_raises_on_a_host_that_cannot_load(tmp_path, monkeypatch):
    module = _install_binding(monkeypatch)
    module.Llama = lambda **kw: (_ for _ in ()).throw(MemoryError("no RAM"))
    assert GGUFProvider(_settings(tmp_path)).warm() is False


# --------------------------------------------------------------------------- #
# One process, one set of weights
# --------------------------------------------------------------------------- #
def test_two_providers_share_one_loaded_model(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    settings = _settings(tmp_path)
    GGUFProvider(settings).complete(LLMRequest.from_prompt("a"))
    GGUFProvider(settings).complete(LLMRequest.from_prompt("b"))
    assert len(_FakeLlama.instances) == 1, "5.6 GB must not be held twice for one model"


def test_a_different_context_size_is_a_different_model(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    GGUFProvider(_settings(tmp_path)).complete(LLMRequest.from_prompt("a"))
    GGUFProvider(_settings(tmp_path, qwythos_context_tokens=4096)).complete(
        LLMRequest.from_prompt("b"))
    assert len(_FakeLlama.instances) == 2


# --------------------------------------------------------------------------- #
# Statelessness, with a KV cache underneath
# --------------------------------------------------------------------------- #
def test_each_request_resets_the_kv_cache(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path))
    provider.complete(LLMRequest.from_prompt("a"))
    provider.complete(LLMRequest.from_prompt("b"))
    assert _FakeLlama.instances[0].resets == 2, "whatever the last turn left must not leak in"


def test_identical_requests_give_identical_text(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path))
    first = provider.complete(LLMRequest.from_prompt("why?"))
    second = provider.complete(LLMRequest.from_prompt("why?"))
    assert first.text == second.text
    assert _FakeLlama.instances[0].calls[0]["seed"] == _FakeLlama.instances[0].calls[1]["seed"]


# --------------------------------------------------------------------------- #
# The request as chat messages
# --------------------------------------------------------------------------- #
def test_no_persona_is_injected(tmp_path, monkeypatch):
    """Her identity is composed in identity/soul.py — a provider must never add one."""
    _install_binding(monkeypatch)
    GGUFProvider(_settings(tmp_path)).complete(LLMRequest.from_prompt("hello"))
    messages = _FakeLlama.instances[0].calls[0]["messages"]
    assert not any("NYXARA" in (m.get("content") or "") for m in messages)


def test_the_system_prompt_becomes_a_real_system_turn(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    GGUFProvider(_settings(tmp_path)).complete(
        LLMRequest.from_prompt("hello", system="be brief"))
    messages = _FakeLlama.instances[0].calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[-1]["role"] == "user"


def test_json_mode_asks_for_json_both_ways(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    GGUFProvider(_settings(tmp_path)).complete(
        LLMRequest.from_prompt("give me json", json_mode=True))
    call = _FakeLlama.instances[0].calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert "JSON" in call["messages"][0]["content"]


def test_an_oversized_prompt_is_trimmed_rather_than_taking_the_rung_down(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    settings = _settings(tmp_path, qwythos_context_tokens=512)
    request = LLMRequest.from_prompt("x" * 40_000, system="y" * 40_000, max_tokens=64)
    provider = GGUFProvider(settings)
    provider.complete(request)                       # must not raise
    sent = _FakeLlama.instances[0].calls[0]["messages"]
    assert sum(len(m["content"]) for m in sent) < 40_000


# --------------------------------------------------------------------------- #
# Honest finish reasons
# --------------------------------------------------------------------------- #
def test_a_reply_that_ran_past_its_turn_is_cut_and_reported_as_stop(tmp_path, monkeypatch):
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path))
    provider.complete(LLMRequest.from_prompt("a"))    # builds the cached model
    _FakeLlama.instances[0].reply = "the answer<|im_end|>and then some rubbish"
    resp = provider.complete(LLMRequest.from_prompt("a"))
    assert resp.text == "the answer"
    assert resp.finish_reason == "stop"


def test_an_empty_reply_is_silence_the_ladder_can_move_past(tmp_path, monkeypatch):
    from nyxara.mind.llm import LLM
    _install_binding(monkeypatch)
    provider = GGUFProvider(_settings(tmp_path))
    provider.complete(LLMRequest.from_prompt("a"))
    _FakeLlama.instances[0].reply = "   "
    resp = provider.complete(LLMRequest.from_prompt("a"))
    assert LLM._is_silence(resp) is True


# --------------------------------------------------------------------------- #
# Sight is reported, never disguised
# --------------------------------------------------------------------------- #
def test_vision_off_says_so(tmp_path, monkeypatch):
    _install_binding(monkeypatch, chat_handlers=("Qwen25VLChatHandler",))
    can_see, why = GGUFProvider(_settings(tmp_path, projector=True)).vision_available()
    assert can_see is False and "disabled by config" in why


def test_a_missing_projector_says_so(tmp_path, monkeypatch):
    _install_binding(monkeypatch, chat_handlers=("Qwen25VLChatHandler",))
    can_see, why = GGUFProvider(
        _settings(tmp_path, qwythos_vision_enabled=True)).vision_available()
    assert can_see is False and "projector is missing" in why


def test_a_build_with_no_usable_handler_says_so_rather_than_answering_blind(tmp_path, monkeypatch):
    """The one failure a multimodal rung can have that looks exactly like success."""
    _install_binding(monkeypatch, chat_handlers=())
    can_see, why = GGUFProvider(
        _settings(tmp_path, projector=True, qwythos_vision_enabled=True)).vision_available()
    assert can_see is False and "no chat handler" in why


def test_vision_works_when_the_build_ships_a_handler(tmp_path, monkeypatch):
    _install_binding(monkeypatch, chat_handlers=("Qwen25VLChatHandler",))
    can_see, why = GGUFProvider(
        _settings(tmp_path, projector=True, qwythos_vision_enabled=True)).vision_available()
    assert can_see is True and why == ""


# --------------------------------------------------------------------------- #
# A binding that moved its parameters costs a parameter, not the cortex
# --------------------------------------------------------------------------- #
def test_unknown_kwargs_are_dropped_rather_than_killing_the_rung(tmp_path, monkeypatch):
    module = _install_binding(monkeypatch)

    class _Picky(_FakeLlama):
        def __init__(self, *, model_path, n_ctx, verbose=False):   # no seed, no n_gpu_layers
            super().__init__(model_path=model_path, n_ctx=n_ctx, verbose=verbose)

    module.Llama = _Picky
    provider = GGUFProvider(_settings(tmp_path))
    resp = provider.complete(LLMRequest.from_prompt("why?"))
    assert resp.text
    assert "seed" not in _FakeLlama.instances[0].kwargs


def test_a_kwargs_taking_build_gets_everything(tmp_path, monkeypatch):
    module = _install_binding(monkeypatch)

    class _Open(_FakeLlama):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    module.Llama = _Open
    GGUFProvider(_settings(tmp_path)).complete(LLMRequest.from_prompt("why?"))
    assert _FakeLlama.instances[0].kwargs["seed"] == 0
    assert _FakeLlama.instances[0].kwargs["n_gpu_layers"] == 0


def test_a_budget_too_small_for_the_system_prompt_drops_it_rather_than_keeping_all_of_it(
        tmp_path, monkeypatch):
    """``text[-0:]`` is the whole string, not the empty one — so a head/tail trim with a zero-sized
    tail silently keeps everything it was trying to cut."""
    _install_binding(monkeypatch)
    settings = _settings(tmp_path, qwythos_context_tokens=512)
    request = LLMRequest.from_prompt("u" * 4_000, system="s" * 200_000, max_tokens=300)
    GGUFProvider(settings).complete(request)
    sent = _FakeLlama.instances[0].calls[0]["messages"]
    assert sum(len(m["content"]) for m in sent) < 20_000
