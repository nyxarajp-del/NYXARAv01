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
    MockProvider,
    Role,
    Usage,
    estimate_tokens,
)


def _mock_llm():
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


# -------------------- mock provider -------------------- #
def test_mock_always_available():
    assert MockProvider().available() is True


def test_mock_deterministic():
    p = MockProvider()
    req = LLMRequest.from_prompt("hello")
    assert p.complete(req).text == p.complete(req).text


def test_mock_echoes_prompt():
    p = MockProvider()
    resp = p.complete(LLMRequest.from_prompt("remember me"))
    assert "remember me" in resp.text
    assert resp.provider == "mock"
    assert resp.usage.total_tokens > 0


def test_mock_json_mode():
    p = MockProvider()
    resp = p.complete(LLMRequest.from_prompt("data", json_mode=True))
    assert resp.parse_json()["mock"] is True


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
def test_facade_uses_mock_in_test_profile():
    llm = _mock_llm()
    assert llm.chosen_provider().name == "mock"


def test_facade_generate():
    llm = _mock_llm()
    out = llm.generate("hello world", system="be nice")
    assert "hello world" in out


def test_facade_generate_json():
    llm = _mock_llm()
    data = llm.generate_json("give me json")
    assert data["mock"] is True


def test_facade_chat():
    llm = _mock_llm()
    resp = llm.chat([Message(Role.USER, "hi")])
    assert isinstance(resp, LLMResponse)


def test_facade_statelessness():
    llm = _mock_llm()
    # identical requests -> identical output; no carried-over state
    assert llm.generate("same") == llm.generate("same")
    assert llm.stateless is True


def test_provider_status():
    llm = _mock_llm()
    status = llm.provider_status()
    assert status["mock"] is True
    assert set(status) == {"qwen", "self", "mock"}


def test_async_complete():
    llm = _mock_llm()

    async def go():
        return await llm.acomplete(LLMRequest.from_prompt("async hi"))

    resp = asyncio.run(go())
    assert "async hi" in resp.text


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
    settings.llm.provider = ProviderName.QWEN
    llm = LLM(settings=settings, providers={"qwen": flaky, "mock": MockProvider()},
              retry_policy=RetryPolicy(max_attempts=5, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.text == "recovered"
    assert flaky.calls == 3


def test_falls_back_to_mock_when_provider_dead():
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

    settings = NyxaraSettings.for_profile(Profile.DEV)  # allows mock fallback
    settings.llm.provider = ProviderName.QWEN
    llm = LLM(settings=settings, providers={"qwen": _Dead(), "mock": MockProvider()},
              retry_policy=RetryPolicy(max_attempts=2, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("fallback please"))
    assert resp.provider == "mock"
    assert "fallback please" in resp.text


def test_no_fallback_raises_when_disabled():
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
    settings.llm.provider = ProviderName.QWEN
    settings.llm.allow_mock_fallback = False
    from nyxara.kernel.errors import RetryPolicy
    llm = LLM(settings=settings, providers={"qwen": _Dead()},
              retry_policy=RetryPolicy(max_attempts=1, base_delay=0))
    with pytest.raises(LLMError):
        llm.complete(LLMRequest.from_prompt("x"))


def test_unavailable_provider_falls_back_to_mock():
    class _Unavailable(LLMProviderBase):
        name = "x"

        def available(self):
            return False

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.QWEN
    llm = LLM(settings=settings, providers={"qwen": _Unavailable(), "mock": MockProvider()})
    assert llm.chosen_provider().name == "mock"


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
