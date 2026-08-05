"""Tests for nyxara.mind.llm."""

from __future__ import annotations

import asyncio

import pytest

from nyxara.kernel.config import LLMProvider as ProviderName
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.errors import LLMError
from nyxara.mind.llm import (
    LLM,
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    Message,
    NativeProvider,
    Role,
    Usage,
    estimate_tokens,
)


def _native_llm():
    return LLM(settings=NyxaraSettings.for_profile(Profile.TEST))


# -------------------- request / message -------------------- #
def test_request_from_prompt():
    r = LLMRequest.from_prompt("hi", system="sys")
    assert r.messages[0].content == "hi"
    assert r.system == "sys"
    assert r.last_user() == "hi"


def test_request_from_messages_last_user():
    r = LLMRequest.from_messages([
        Message(Role.USER, "first"),
        Message(Role.ASSISTANT, "reply"),
        Message(Role.USER, "second"),
    ])
    assert r.last_user() == "second"


def test_request_fingerprint_stable():
    a = LLMRequest.from_prompt("x", temperature=0.5)
    b = LLMRequest.from_prompt("x", temperature=0.5)
    assert a.fingerprint() == b.fingerprint()
    c = LLMRequest.from_prompt("y", temperature=0.5)
    assert a.fingerprint() != c.fingerprint()


def test_request_is_immutable():
    r = LLMRequest.from_prompt("x")
    with pytest.raises(Exception):
        r.temperature = 1.0  # frozen


# -------------------- usage -------------------- #
def test_usage_total():
    u = Usage(prompt_tokens=10, completion_tokens=5)
    assert u.total_tokens == 15


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10


# -------------------- native own-brain provider -------------------- #
def test_native_always_available():
    assert NativeProvider().available() is True


def test_native_deterministic():
    p = NativeProvider()
    req = LLMRequest.from_prompt("hello")
    assert p.complete(req).text == p.complete(req).text


def test_native_answers_from_own_brain_not_echo():
    p = NativeProvider()
    resp = p.complete(LLMRequest.from_prompt("remember me"))
    # her own always-on brain drafts real text — it never parrots the prompt back
    assert isinstance(resp.text, str) and resp.text.strip()
    assert resp.provider == "native"
    assert resp.usage.total_tokens > 0


def test_native_json_mode():
    p = NativeProvider()
    resp = p.complete(LLMRequest.from_prompt("data", json_mode=True))
    assert resp.parse_json()["native"] is True


# -------------------- response parsing -------------------- #
def test_parse_json_plain():
    assert LLMResponse('{"a": 1}', "x", "m").parse_json() == {"a": 1}


def test_parse_json_fenced():
    r = LLMResponse('```json\n{"a": 1}\n```', "x", "m")
    assert r.parse_json() == {"a": 1}


def test_parse_json_with_prose():
    r = LLMResponse('Sure! {"a": 1, "b": 2} hope that helps', "x", "m")
    assert r.parse_json() == {"a": 1, "b": 2}


def test_parse_json_array():
    r = LLMResponse("[1, 2, 3]", "x", "m")
    assert r.parse_json() == [1, 2, 3]


def test_parse_json_bad_raises():
    with pytest.raises(LLMError):
        LLMResponse("definitely not json", "x", "m").parse_json()


# -------------------- LLM facade -------------------- #
def test_facade_uses_native_in_test_profile():
    llm = _native_llm()
    assert llm.chosen_provider().name == "native"


def test_facade_generate():
    llm = _native_llm()
    out = llm.generate("hello world", system="be nice")
    assert isinstance(out, str) and out.strip()


def test_facade_generate_json():
    llm = _native_llm()
    data = llm.generate_json("give me json")
    assert data["native"] is True


def test_facade_chat():
    llm = _native_llm()
    resp = llm.chat([Message(Role.USER, "hi")])
    assert isinstance(resp, LLMResponse)


def test_facade_statelessness():
    llm = _native_llm()
    # identical requests -> identical output; no carried-over state
    assert llm.generate("same") == llm.generate("same")
    assert llm.stateless is True


def test_provider_status():
    llm = _native_llm()
    status = llm.provider_status()
    assert status["native"] is True
    assert set(status) == {"litertlm", "aicredits", "groq", "airouter", "self", "native"}
    # Under TEST every rung but her native floor is honestly unavailable — including the on-device
    # primary, which the profile seals so the suite never loads 2.4 GB of weights.
    assert status["litertlm"] is False


def test_async_complete():
    llm = _native_llm()

    async def go():
        return await llm.acomplete(LLMRequest.from_prompt("async hi"))

    resp = asyncio.run(go())
    assert isinstance(resp.text, str) and resp.text.strip()


# -------------------- fallback + resilience -------------------- #
class _FlakyProvider(LLMProviderBase):
    name = "flaky"

    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.calls = 0

    def available(self):
        return True

    def default_model(self):
        return "flaky-1"

    def _complete(self, req, model):
        self.calls += 1
        if self.calls <= self.fail_times:
            from nyxara.kernel.errors import ExternalServiceError
            raise ExternalServiceError("503")
        return ("recovered", "stop", Usage(1, 1), None)


def test_retry_then_success():
    from nyxara.kernel.errors import RetryPolicy
    flaky = _FlakyProvider(fail_times=2)
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AIROUTER
    llm = LLM(settings=settings, providers={"airouter": flaky, "native": NativeProvider()},
              retry_policy=RetryPolicy(max_attempts=5, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.text == "recovered"
    assert flaky.calls == 3


def test_falls_back_to_native_when_provider_dead():
    from nyxara.kernel.errors import RetryPolicy

    class _Dead(LLMProviderBase):
        name = "dead"

        def available(self):
            return True

        def default_model(self):
            return "dead"

        def _complete(self, req, model):
            from nyxara.kernel.errors import ExternalServiceError
            raise ExternalServiceError("always down")

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AIROUTER
    llm = LLM(settings=settings, providers={"airouter": _Dead(), "native": NativeProvider()},
              retry_policy=RetryPolicy(max_attempts=2, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("fallback please"))
    # her always-on native own-brain is the guaranteed floor — it answers, never an echo
    assert resp.provider == "native"
    assert isinstance(resp.text, str) and resp.text.strip()


def test_native_floor_is_always_available():
    # There is no way to disable the native own-brain floor: a dead configured provider
    # falls back to her own always-on brain rather than raising — she is never voiceless.
    class _Dead(LLMProviderBase):
        name = "dead"

        def available(self):
            return True

        def default_model(self):
            return "dead"

        def _complete(self, req, model):
            from nyxara.kernel.errors import ExternalServiceError
            raise ExternalServiceError("down")

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AIROUTER
    from nyxara.kernel.errors import RetryPolicy
    llm = LLM(settings=settings, providers={"airouter": _Dead(), "native": NativeProvider()},
              retry_policy=RetryPolicy(max_attempts=1, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.provider == "native" and resp.text.strip()


def test_unavailable_provider_falls_back_to_native():
    class _Unavailable(LLMProviderBase):
        name = "x"

        def available(self):
            return False

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AIROUTER
    llm = LLM(settings=settings, providers={"airouter": _Unavailable(), "native": NativeProvider()})
    assert llm.chosen_provider().name == "native"


# -------------------- self-model prompt formatting (Phase 0) -------------------- #
def test_format_self_prompt_renders_system_and_turns():
    from nyxara.mind.llm import (format_self_prompt, _SELF_USER_TAG,
                                 _SELF_ASSISTANT_TAG)
    req = LLMRequest.from_messages(
        [Message(Role.USER, "first"), Message(Role.ASSISTANT, "reply"),
         Message(Role.USER, "second")], system="You are NYXARA.")
    prompt = format_self_prompt(req)
    assert prompt.startswith("You are NYXARA.")
    assert f"{_SELF_USER_TAG}\nfirst" in prompt
    assert f"{_SELF_ASSISTANT_TAG}\nreply" in prompt
    # ends primed for NYXARA to continue (an add_generation_prompt)
    assert prompt.endswith(_SELF_ASSISTANT_TAG + "\n")


def test_format_self_prompt_without_system():
    from nyxara.mind.llm import format_self_prompt, _SELF_USER_TAG
    prompt = format_self_prompt(LLMRequest.from_prompt("hi"))
    assert prompt.startswith(f"{_SELF_USER_TAG}\nhi")


def test_truncate_at_stops_cuts_at_earliest_marker():
    from nyxara.mind.llm import truncate_at_stops, _SELF_USER_TAG
    raw = f"The answer is 42.\n{_SELF_USER_TAG}\nnext question"
    text, hit = truncate_at_stops(raw, (f"\n{_SELF_USER_TAG}",))
    assert text == "The answer is 42."
    assert hit is True


def test_truncate_at_stops_no_marker_keeps_text():
    from nyxara.mind.llm import truncate_at_stops
    text, hit = truncate_at_stops("a clean answer", ("### User:",))
    assert text == "a clean answer"
    assert hit is False


def test_self_provider_unavailable_without_promoted_model(tmp_path):
    from nyxara.mind.llm import SelfProvider
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"   # empty -> no active model
    assert SelfProvider(settings).available() is False


def test_format_self_training_doc_appends_answer():
    from nyxara.mind.llm import (format_self_prompt, format_self_training_doc,
                                 _SELF_ASSISTANT_TAG)
    head = format_self_prompt(LLMRequest.from_prompt("2+2?", system="be NYXARA"))
    doc = format_self_training_doc("2+2?", "It is 4.", system="be NYXARA")
    # the training doc is the inference head plus the target answer (train/inference parity)
    assert doc.startswith(head)
    assert doc.endswith("It is 4.\n")
    assert f"{_SELF_ASSISTANT_TAG}\nIt is 4." in doc
