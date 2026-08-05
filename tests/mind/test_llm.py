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
    assert set(status) == {"litertlm", "self", "native"}
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
    settings.llm.provider = ProviderName.LITERTLM
    llm = LLM(settings=settings, providers={"litertlm": flaky, "native": NativeProvider()},
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
    settings.llm.provider = ProviderName.LITERTLM
    llm = LLM(settings=settings, providers={"litertlm": _Dead(), "native": NativeProvider()},
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
    settings.llm.provider = ProviderName.LITERTLM
    from nyxara.kernel.errors import RetryPolicy
    llm = LLM(settings=settings, providers={"litertlm": _Dead(), "native": NativeProvider()},
              retry_policy=RetryPolicy(max_attempts=1, base_delay=0))
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.provider == "native" and resp.text.strip()


def test_unavailable_provider_falls_back_to_native():
    class _Unavailable(LLMProviderBase):
        name = "x"

        def available(self):
            return False

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.LITERTLM
    llm = LLM(settings=settings, providers={"litertlm": _Unavailable(), "native": NativeProvider()})
    assert llm.chosen_provider().name == "native"


# -------------------- silence is a failed rung, not a cheap answer -------------------- #
class _Mute(LLMProviderBase):
    """Succeeds without saying anything — the failure mode that does NOT raise.

    A promoted-but-degenerate ``self`` model does exactly this: a well-formed response whose text
    is empty. Nothing throws, so nothing used to stop the ladder handing that silence to the caller.
    """

    def __init__(self, name="mute", text=""):
        super().__init__()
        self.name = name
        self._text = text
        self.calls = 0

    def available(self):
        return True

    def default_model(self):
        return f"{self.name}-1"

    def _complete(self, req, model):
        self.calls += 1
        return (self._text, "stop", Usage(1, 0), None)


def _auto(providers):
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.AUTO
    llm = LLM(settings=settings, providers=providers)
    llm._AUTO_LADDER = tuple(providers)          # per-instance ladder over the fakes
    return llm


def test_an_empty_answer_falls_through_to_the_next_rung():
    mute, native = _Mute(), NativeProvider()
    llm = _auto({"mute": mute, "native": native})
    resp = llm.complete(LLMRequest.from_prompt("say something"))
    assert mute.calls == 1, "the mute rung must actually be tried first"
    assert resp.provider == "native"
    assert resp.text.strip(), "her floor must speak when the rung above it does not"


def test_whitespace_only_counts_as_silence():
    mute = _Mute(text="   \n\t ")
    llm = _auto({"mute": mute, "native": NativeProvider()})
    assert llm.complete(LLMRequest.from_prompt("x")).provider == "native"


def test_a_silent_rung_is_recorded_not_hidden():
    """An operator has to be able to see WHY the primary went quiet — silence included."""
    llm = _auto({"mute": _Mute(), "native": NativeProvider()})
    llm.complete(LLMRequest.from_prompt("x"))
    assert llm.last_fallback and llm.last_fallback["provider"] == "mute"
    assert "empty" in llm.last_fallback["error"]


def test_a_speaking_rung_is_still_preferred():
    """The fix must not make the ladder skip a rung that had something to say."""
    talker = _Mute(name="talker", text="a real answer")
    llm = _auto({"talker": talker, "native": NativeProvider()})
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.provider == "talker" and resp.text == "a real answer"
    assert llm.last_fallback is None


def test_all_rungs_mute_returns_a_response_rather_than_raising():
    """Degrade, never explode: if literally nothing speaks, the caller still gets a response."""
    llm = _auto({"mute": _Mute(), "mute2": _Mute(name="mute2")})
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.provider in {"mute", "mute2"} and resp.text == ""


def test_a_pinned_rung_that_goes_silent_falls_to_her_floor():
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.LITERTLM
    llm = LLM(settings=settings, providers={"litertlm": _Mute(name="litertlm"),
                                            "native": NativeProvider()})
    resp = llm.complete(LLMRequest.from_prompt("x"))
    assert resp.provider == "native" and resp.text.strip()


def test_complete_with_still_reports_silence_faithfully():
    """A council member that says nothing must be recorded as saying nothing, not substituted."""
    llm = _auto({"mute": _Mute(), "native": NativeProvider()})
    resp = llm.complete_with("mute", LLMRequest.from_prompt("x"))
    assert resp.provider == "mute" and resp.text == ""


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


# -------------------- a degraded turn must not be a silent turn -------------------- #
def test_reasoning_degradation_is_recorded_and_logged(caplog):
    """The bug this pins: ``LLMReasoner.__call__`` swallowed every exception from the reasoning
    stack and answered from her small own brain instead — no log, no record, no visible difference
    except worse replies. Reported from a live session where ``/learning`` said ``chosen: litertlm``
    with ``engine_loaded: false``: the ladder looked healthy and a 532-parameter n-gram was talking.

    The fallback itself is correct and stays. Only the silence was the defect.
    """
    import logging

    from nyxara.mind.llm_reasoner import LLMReasoner

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.LITERTLM
    reasoner = LLMReasoner(LLM(settings=settings, providers={"native": NativeProvider()}),
                           settings=settings)
    reasoner._is_real = lambda: True                     # a strong model is nominally reachable
    def _boom(stimulus, focus=None):
        raise RuntimeError("litert_lm_conversation_send_message failed")
    reasoner._reason = _boom

    with caplog.at_level(logging.WARNING, logger="nyxara.mind.llm_reasoner"):
        candidate = reasoner("hello")

    assert candidate is not None and candidate.text, "she must still answer — the fallback is right"
    assert reasoner.last_degradation is not None, "the degradation must be recorded"
    assert "send_message" in reasoner.last_degradation["reason"], "the real reason must survive"
    assert any("own small brain" in r.message for r in caplog.records), "and be said out loud"


def test_an_unreachable_model_is_also_recorded():
    """The other silent path: no real model at all went straight to the small brain, unremarked."""
    from nyxara.mind.llm_reasoner import LLMReasoner

    settings = NyxaraSettings.for_profile(Profile.DEV)
    reasoner = LLMReasoner(LLM(settings=settings, providers={"native": NativeProvider()}),
                           settings=settings)
    reasoner._is_real = lambda: False
    reasoner("hello")
    assert reasoner.last_degradation is not None
    assert "no real model" in reasoner.last_degradation["reason"]


def test_a_healthy_turn_clears_the_degradation_flag():
    """It reports the LAST turn, not the worst turn ever — a recovered model must look recovered."""
    from nyxara.mind.llm_reasoner import LLMReasoner

    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.LITERTLM
    reasoner = LLMReasoner(LLM(settings=settings, providers={"native": NativeProvider()}),
                           settings=settings)
    reasoner.last_degradation = {"reason": "stale", "ts": 0}
    reasoner._is_real = lambda: True
    reasoner._reason = lambda stimulus, focus=None: _default_candidate()
    reasoner("hello")
    assert reasoner.last_degradation is None


def _default_candidate():
    from nyxara.kernel.orchestrator import Candidate
    return Candidate(text="a real answer", kind="respond", confidence=0.9, reversible=True)


# -------------------- a pointer is not a model -------------------- #
def _self_provider(root):
    from nyxara.mind.llm import SelfProvider
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.self_model_dir = root
    return SelfProvider(settings)


def test_self_is_unavailable_when_the_pointer_names_nothing(tmp_path):
    """Reported live: ``self: available: true`` sitting beside, turn after turn,

        self model unavailable: FileNotFoundError: .../foundry/v1/model.pt

    ``available()`` checked only that the ``active`` pointer file existed. A pointer is not a model.
    A rung that cannot serve must not advertise itself — the ladder is built to step over it, and
    it cannot step over a lie.
    """
    (tmp_path / "active").write_text("v1", encoding="utf-8")
    assert _self_provider(tmp_path).available() is False, "a dangling pointer is not a model"

    (tmp_path / "v1").mkdir()
    assert _self_provider(tmp_path).available() is False, "an empty version dir is not a model"


def test_self_is_available_once_a_version_is_actually_there(tmp_path):
    (tmp_path / "active").write_text("v1", encoding="utf-8")
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "model.json").write_text("{}", encoding="utf-8")
    assert _self_provider(tmp_path).available() is True


def test_every_backend_artifact_counts_as_present(tmp_path):
    """Whatever the backend wrote, that version exists — the check must not favour one of them."""
    for i, artifact in enumerate(("adapter", "model.pt", "weights.npz", "model.json")):
        root = tmp_path / f"root{i}"
        vdir = root / "v1"
        vdir.mkdir(parents=True)
        (root / "active").write_text("v1", encoding="utf-8")
        if artifact == "adapter":
            (vdir / artifact).mkdir()
        else:
            (vdir / artifact).write_bytes(b"x")
        assert _self_provider(root).available() is True, f"{artifact} should count"


def test_an_empty_pointer_is_not_available(tmp_path):
    (tmp_path / "active").write_text("   \n", encoding="utf-8")
    assert _self_provider(tmp_path).available() is False
