"""Weight-free tests for her PRIMARY on-device brain — mind/llm.py::LiteRTLMProvider.

The real thing is a 2.4 GB INT4 Gemma-4-E2B-it served in-process by ``litert_lm``. Loading that
inside a test suite is out of the question, so these tests install a **fake ``litert_lm`` module**
into ``sys.modules`` and a stub weights file on ``tmp_path``. Nothing here touches the network, the
disk budget, or a real model.

What is pinned, and why each one matters:

* **Honest availability** — the flag, the binding and the weights file are three independent reasons
  to be unavailable, and each must degrade the ladder rather than raise.
* **The ladder head** — with the weights present, ``litertlm`` drafts; without them, aicredits leads
  exactly as it used to. This is the whole point of the change.
* **Statelessness** — one cached ``Engine``, a FRESH ``Conversation`` per request, always closed.
  A leaked conversation would silently give the facade a memory it is documented not to have.
* **Message objects, never bare strings** — a bare string normalises to a *string* ``content``, the
  chat template renders nothing, and the model answers an empty prompt with a canned
  self-introduction. That failure is invisible (a plausible-looking reply to every question), so it
  gets a test of its own.
* **Gemma 4's real turn markers** (``<|turn>`` / ``<turn|>``, not Gemma 3's ``<start_of_turn>``).
* **No persona** — the provider forwards the caller's system text and adds no identity of its own.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from nyxara.kernel.config import OWN_PROVIDERS, LLMProvider, NyxaraSettings, Profile
from nyxara.kernel.errors import LLMError
from nyxara.mind.llm import LLM, LiteRTLMProvider, LLMRequest, Message, Role


# --------------------------------------------------------------------------- #
# A fake litert_lm (installed into sys.modules) — records everything it is asked
# --------------------------------------------------------------------------- #
class _Text:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_json(self):
        return {"type": "text", "text": self.text}


class _Contents:
    def __init__(self, contents) -> None:
        self.contents = list(contents)

    @classmethod
    def of(cls, arg):
        return cls([_Text(arg)] if isinstance(arg, str) else [arg])

    def to_json(self):
        return [c.to_json() for c in self.contents]


class _Role:
    SYSTEM, USER, MODEL = "system", "user", "model"


class _Message:
    """Mirrors litert_lm.Message closely enough to catch a bare-string regression."""

    def __init__(self, role, contents=None) -> None:
        self.role = role
        self.contents = contents or _Contents([])

    @classmethod
    def system(cls, text):
        return cls(_Role.SYSTEM, _Contents.of(text))

    @classmethod
    def user(cls, text):
        return cls(_Role.USER, _Contents.of(text))

    @classmethod
    def model(cls, contents=None):
        return cls(_Role.MODEL, contents or _Contents([]))

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.contents.contents)


class _SamplerConfig:
    def __init__(self, top_k=None, top_p=None, temperature=None, seed=None) -> None:
        self.top_k, self.top_p, self.temperature, self.seed = top_k, top_p, temperature, seed


class _Backend:
    class CPU:
        def __init__(self, thread_count=None) -> None:
            self.kind, self.thread_count = "cpu", thread_count

    class GPU:
        def __init__(self) -> None:
            self.kind = "gpu"

    class NPU:
        def __init__(self) -> None:
            self.kind = "npu"


class _Conversation:
    def __init__(self, engine, **kw) -> None:
        self.engine = engine
        self.kwargs = kw
        self.closed = False
        self.sent = []

    def send_message(self, message, *, max_output_tokens=None, **kw):
        if self.engine.raises is not None:
            raise self.engine.raises
        self.sent.append((message, max_output_tokens))
        self.engine.sent.append((message, max_output_tokens))
        return {"role": "assistant",
                "content": [{"type": "text", "text": self.engine.reply}]}

    def close(self):
        self.closed = True


class _Engine:
    instances = []

    def __init__(self, model_path, backend=None, cache_dir=None, **kw) -> None:
        self.model_path, self.backend, self.cache_dir = model_path, backend, cache_dir
        self.conversations = []
        self.sent = []
        self.reply = "on-device answer"
        self.raises = None
        self.closed = False
        _Engine.instances.append(self)

    def create_conversation(self, **kw):
        conv = _Conversation(self, **kw)
        self.conversations.append(conv)
        return conv

    def close(self):
        self.closed = True


def _install_fake(monkeypatch):
    """Put a fake ``litert_lm`` on ``sys.modules`` and hand back the module."""
    _Engine.instances = []
    mod = types.ModuleType("litert_lm")
    mod.Engine = _Engine
    mod.Message = _Message
    mod.Contents = _Contents
    mod.Role = _Role
    mod.SamplerConfig = _SamplerConfig
    mod.Backend = _Backend
    mod.LogSeverity = types.SimpleNamespace(ERROR=3, SILENT=5)
    mod.set_min_log_severity = lambda severity: None
    monkeypatch.setitem(sys.modules, "litert_lm", mod)
    return mod


@pytest.fixture
def weights(tmp_path):
    """A stub weights file — ``available()`` only checks that the path is a file."""
    path = tmp_path / "model.litertlm"
    path.write_bytes(b"not really 2.4GB")
    return path


def _settings(weights_path, **over):
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.litertlm_enabled = True
    s.llm.litertlm_auto_download = False
    s.llm.litertlm_model_path = weights_path
    for k, v in over.items():
        setattr(s.llm, k, v)
    return s


def _only_local(s, foundry_dir=None):
    """Silence every cloud rung so the ladder is litertlm -> native.

    ``foundry_dir`` points ``self`` at an empty directory so a promoted model that happens to exist
    on the developer's machine cannot change which rung the fallback lands on.
    """
    s.llm.aicredits_enabled = False
    s.llm.groq_enabled = False
    s.llm.airouter_enabled = False
    if foundry_dir is not None:
        s.llm.self_model_dir = foundry_dir
    return s


# --------------------------------------------------------------------------- #
# 1. Honest availability — three independent reasons to be unavailable
# --------------------------------------------------------------------------- #
def test_available_when_flag_binding_and_weights_are_all_present(monkeypatch, weights):
    _install_fake(monkeypatch)
    assert LiteRTLMProvider(_settings(weights)).available() is True


def test_unavailable_when_disabled(monkeypatch, weights):
    _install_fake(monkeypatch)
    assert LiteRTLMProvider(_settings(weights, litertlm_enabled=False)).available() is False


def test_unavailable_without_the_binding(monkeypatch, weights):
    """A machine that never ran ``pip install litert-lm-api`` degrades, it does not crash."""
    monkeypatch.setitem(sys.modules, "litert_lm", None)   # import raises -> treated as absent
    assert LiteRTLMProvider(_settings(weights)).available() is False


def test_unavailable_without_weights(monkeypatch, tmp_path):
    _install_fake(monkeypatch)
    missing = tmp_path / "nope.litertlm"
    assert LiteRTLMProvider(_settings(missing)).available() is False


def test_unavailable_provider_raises_rather_than_answering(monkeypatch, tmp_path):
    """Unavailability must be loud at the provider so the facade can fall to the next rung."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(tmp_path / "nope.litertlm"))
    with pytest.raises(LLMError):
        prov.complete(LLMRequest.from_prompt("hello"))


def test_availability_never_downloads(monkeypatch, tmp_path):
    """``available()`` is a cheap file check — it must not reach the asset fetcher."""
    _install_fake(monkeypatch)
    import nyxara.mind.litertlm_assets as assets

    def _boom(*a, **k):
        raise AssertionError("available() must never trigger a download")

    monkeypatch.setattr(assets, "ensure_litertlm_model", _boom)
    assert LiteRTLMProvider(_settings(tmp_path / "nope.litertlm")).available() is False


# --------------------------------------------------------------------------- #
# 2. The ladder head — this is what "primary" means
# --------------------------------------------------------------------------- #
def test_litertlm_leads_the_auto_ladder():
    assert LLM._AUTO_LADDER[0] == "litertlm"
    assert LLM._AUTO_LADDER == ("litertlm", "aicredits", "groq", "airouter", "self", "native")


def test_it_drafts_the_turn_when_the_weights_are_present(monkeypatch, weights):
    _install_fake(monkeypatch)
    llm = LLM(settings=_only_local(_settings(weights)))
    assert llm.chosen_provider().name == "litertlm"
    resp = llm.complete(LLMRequest.from_prompt("who are you?"))
    assert resp.provider == "litertlm"
    assert resp.text == "on-device answer"


def test_absent_weights_hand_the_lead_back_to_the_cloud(monkeypatch, tmp_path):
    """The degradation promise: no weights simply means the ladder starts one rung lower."""
    _install_fake(monkeypatch)
    s = _settings(tmp_path / "nope.litertlm")
    llm = LLM(settings=s)
    assert llm.provider_status()["litertlm"] is False
    assert [p.name for p in llm._auto_ladder()][0] != "litertlm"


def test_it_is_one_of_her_own_brains_not_a_teacher():
    """Sovereignty classification: local means hers, so nothing may treat it as an outside voice."""
    assert "litertlm" in OWN_PROVIDERS
    assert "aicredits" not in OWN_PROVIDERS


def test_it_needs_no_api_key():
    s = NyxaraSettings(profile=Profile.DEV)
    s.llm.provider = LLMProvider.LITERTLM
    assert s.llm.active_key() is None
    assert s.llm.active_model() == s.llm.litertlm_model


# --------------------------------------------------------------------------- #
# 3. Statelessness — one engine, a fresh conversation per request
# --------------------------------------------------------------------------- #
def test_engine_is_cached_and_conversations_are_not(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    for _ in range(3):
        prov.complete(LLMRequest.from_prompt("hi"))
    assert len(_Engine.instances) == 1, "the multi-GB engine must load exactly once"
    engine = _Engine.instances[0]
    assert len(engine.conversations) == 3, "each request needs its own KV cache"
    assert all(c.closed for c in engine.conversations), "a leaked conversation is leaked memory"


def test_conversation_is_closed_even_when_generation_fails(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    _Engine.instances.clear()
    prov.complete(LLMRequest.from_prompt("warm the engine"))
    engine = _Engine.instances[0]
    engine.raises = RuntimeError("decode blew up")
    with pytest.raises(LLMError):
        prov.complete(LLMRequest.from_prompt("boom"))
    assert all(c.closed for c in engine.conversations)


def test_close_releases_the_engine(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("hi"))
    engine = _Engine.instances[0]
    prov.close()
    assert engine.closed is True
    prov.complete(LLMRequest.from_prompt("again"))
    assert len(_Engine.instances) == 2, "after close() the next call reloads"


def test_concurrent_requests_are_serialized(monkeypatch, weights):
    """One set of weights, one caller at a time — concurrency must queue, not interleave."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("warm"))
    errors = []

    def go():
        try:
            prov.complete(LLMRequest.from_prompt("hi"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(_Engine.instances) == 1


# --------------------------------------------------------------------------- #
# 4. Message mapping — Message objects, a real system role, Gemma 4's markers
# --------------------------------------------------------------------------- #
def test_the_sent_message_is_a_message_object_never_a_bare_string(monkeypatch, weights):
    """A bare string renders to an EMPTY prompt through the chat template (silent, plausible junk)."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("what is the capital of France?"))
    sent, _ = _Engine.instances[0].sent[0]
    assert isinstance(sent, _Message), "a str here silently empties the prompt"
    assert sent.role == _Role.USER
    assert sent.text == "what is the capital of France?"


def test_system_text_becomes_a_real_system_turn(monkeypatch, weights):
    """Gemma 4 has a system role, so the system text is not folded into the user's words."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("hi", system="You serve Master JP."))
    history = _Engine.instances[0].conversations[0].kwargs["messages"]
    assert [m.role for m in history] == [_Role.SYSTEM]
    assert history[0].text == "You serve Master JP."
    sent, _ = _Engine.instances[0].sent[0]
    assert sent.text == "hi", "the user turn must stay the user's own words"


def test_prior_turns_become_history_and_the_last_user_turn_is_sent(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_messages([
        Message(Role.USER, "My name is JP."),
        Message(Role.ASSISTANT, "Nice to meet you, JP."),
        Message(Role.USER, "What is my name?"),
    ]))
    history = _Engine.instances[0].conversations[0].kwargs["messages"]
    assert [m.role for m in history] == [_Role.USER, _Role.MODEL]
    assert history[1].text == "Nice to meet you, JP."
    sent, _ = _Engine.instances[0].sent[0]
    assert sent.text == "What is my name?"


def test_no_persona_is_injected(monkeypatch, weights):
    """Identity lives in identity/soul.py. A provider that adds one is a provider that steers."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("hello"))
    conv = _Engine.instances[0].conversations[0]
    assert conv.kwargs.get("messages") is None
    sent, _ = _Engine.instances[0].sent[0]
    assert "NYXARA" not in sent.text


def test_the_chat_template_uses_gemma_4_turn_markers(monkeypatch, weights):
    """Gemma 3's ``<start_of_turn>`` has no token here; using it makes long prompts echo forever."""
    _install_fake(monkeypatch)
    s = _settings(weights)
    prov = LiteRTLMProvider(s)
    prov.complete(LLMRequest.from_prompt("hi"))
    template = _Engine.instances[0].conversations[0].kwargs["chat_template"]
    assert template == s.llm.litertlm_chat_template
    assert "<|turn>" in template and "<turn|>" in template
    assert "<start_of_turn>" not in template


def test_sampling_and_token_budget_reach_the_runtime(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights, litertlm_top_k=17))
    prov.complete(LLMRequest.from_prompt("hi", temperature=0.25, top_p=0.9,
                                         max_tokens=321, seed=99))
    conv = _Engine.instances[0].conversations[0]
    sampler = conv.kwargs["sampler_config"]
    assert (sampler.top_k, sampler.top_p, sampler.temperature, sampler.seed) == (17, 0.9, 0.25, 99)
    assert _Engine.instances[0].sent[0][1] == 321


def test_backend_selection(monkeypatch, weights):
    _install_fake(monkeypatch)
    LiteRTLMProvider(_settings(weights, litertlm_backend="gpu")).complete(
        LLMRequest.from_prompt("hi"))
    assert _Engine.instances[0].backend.kind == "gpu"

    _Engine.instances.clear()
    LiteRTLMProvider(_settings(weights, litertlm_backend="cpu", litertlm_threads=3)).complete(
        LLMRequest.from_prompt("hi"))
    assert _Engine.instances[0].backend.kind == "cpu"
    assert _Engine.instances[0].backend.thread_count == 3


# --------------------------------------------------------------------------- #
# 5. Response handling — stops, JSON mode, finish reasons
# --------------------------------------------------------------------------- #
def test_reply_is_cut_at_gemma_turn_markers(monkeypatch, weights):
    """A local model has no server to stop it: past its turn it invents the next one."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].reply = "Paris.<turn|>\n<|turn>user\nand then I made this up"
    resp = prov.complete(LLMRequest.from_prompt("capital of France?"))
    assert resp.text == "Paris."
    assert resp.finish_reason == "stop"


def test_caller_stops_are_honoured(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].reply = "keep this###drop that"
    resp = prov.complete(LLMRequest.from_prompt("q", stop=("###",)))
    assert resp.text == "keep this"


def test_json_mode_nudges_the_system_turn(monkeypatch, weights):
    """No native JSON mode on this runtime, so we ask for it the way the cloud rungs do."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("give me json", json_mode=True))
    history = _Engine.instances[0].conversations[0].kwargs["messages"]
    assert history and history[0].role == _Role.SYSTEM
    assert "ONLY valid JSON" in history[0].text


def test_multi_part_replies_are_joined(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    assert prov._extract_text(
        {"role": "assistant",
         "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}
    ) == "Hello world"


def test_non_text_content_is_ignored_not_fatal(monkeypatch, weights):
    """A future multi-modal part must not turn a good answer into a crash."""
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    assert prov._extract_text(
        {"role": "assistant",
         "content": [{"type": "image", "path": "/x.png"}, {"type": "text", "text": "ok"}]}
    ) == "ok"


def test_a_reply_that_hits_the_cap_reports_length(monkeypatch, weights):
    _install_fake(monkeypatch)
    prov = LiteRTLMProvider(_settings(weights))
    prov.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].reply = "x" * 4000          # ~1000 estimated tokens
    resp = prov.complete(LLMRequest.from_prompt("q", max_tokens=4))
    assert resp.finish_reason == "length"


# --------------------------------------------------------------------------- #
# 6. The learning report — an operator must be able to see WHY it is quiet
# --------------------------------------------------------------------------- #
def test_learning_view_reports_a_healthy_primary(monkeypatch, weights):
    _install_fake(monkeypatch)
    view = LiteRTLMProvider(_settings(weights)).learning_view()
    assert view["available"] is True
    assert view["reason_unavailable"] is None
    assert view["weights_present"] is True and view["binding_installed"] is True
    assert view["engine_loaded"] is False, "reporting state must not load the weights"


def test_learning_view_names_the_missing_piece(monkeypatch, tmp_path, weights):
    """Three ways to be down, three distinguishable reasons — 'it got worse' is not a diagnosis."""
    _install_fake(monkeypatch)
    off = LiteRTLMProvider(_settings(weights, litertlm_enabled=False)).learning_view()
    assert "disabled" in off["reason_unavailable"]

    no_weights = LiteRTLMProvider(_settings(tmp_path / "gone.litertlm")).learning_view()
    assert "weights missing" in no_weights["reason_unavailable"]
    assert "fetch_litertlm_model" in no_weights["reason_unavailable"]

    monkeypatch.setitem(sys.modules, "litert_lm", None)
    no_binding = LiteRTLMProvider(_settings(weights)).learning_view()
    assert "litert-lm-api" in no_binding["reason_unavailable"]


def test_facade_learning_view_includes_the_primary(monkeypatch, weights):
    _install_fake(monkeypatch)
    view = LLM(settings=_only_local(_settings(weights))).learning_view()
    assert view["chosen"] == "litertlm"
    assert view["litertlm"]["available"] is True
    assert "self" in view, "the forged brain must still be reported alongside it"


def test_runtime_errors_are_normalised_to_llm_error(monkeypatch, weights, tmp_path):
    """A broken binding costs a rung, never the process."""
    _install_fake(monkeypatch)
    empty_foundry = tmp_path / "no-foundry"
    empty_foundry.mkdir()
    s = _only_local(_settings(weights), foundry_dir=empty_foundry)
    llm = LLM(settings=s)
    llm.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].raises = RuntimeError("litert_lm_conversation_send_message failed")
    resp = llm.complete(LLMRequest.from_prompt("now what?"))
    assert resp.provider == "native", "the ladder must carry on to her own floor"
    assert llm.last_fallback and llm.last_fallback["provider"] == "litertlm"


def test_a_failed_primary_falls_to_her_next_own_brain(monkeypatch, weights):
    """With the clouds off, a failed primary must land on another rung of HERS — and that rung talks.

    The non-empty assertion is the interesting half. A rung below can be promoted yet degenerate and
    return an empty string; ``LLM.complete`` treats that silence as a failed rung and keeps walking
    to her native floor, so a failed primary can never leave the caller with nothing.
    """
    _install_fake(monkeypatch)
    llm = LLM(settings=_only_local(_settings(weights)))
    llm.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].raises = RuntimeError("decode failed")
    resp = llm.complete(LLMRequest.from_prompt("and now?"))
    assert resp.provider in OWN_PROVIDERS and resp.provider != "litertlm"
    assert resp.text.strip(), "a degraded rung must still say something"


def test_a_mute_primary_is_treated_as_a_failed_rung(monkeypatch, weights, tmp_path):
    """The on-device runtime can decode straight to a stop token — well-formed, and empty.

    The foundry is pointed at an empty directory so the ladder is exactly litertlm -> native and
    ``last_fallback`` still names the primary (a silent ``self`` rung in between would overwrite it).
    """
    _install_fake(monkeypatch)
    empty_foundry = tmp_path / "no-foundry"
    empty_foundry.mkdir()
    llm = LLM(settings=_only_local(_settings(weights), foundry_dir=empty_foundry))
    llm.complete(LLMRequest.from_prompt("warm"))
    _Engine.instances[0].reply = ""
    resp = llm.complete(LLMRequest.from_prompt("say something"))
    assert resp.provider == "native"
    assert resp.text.strip()
    assert llm.last_fallback["provider"] == "litertlm" and "empty" in llm.last_fallback["error"]
