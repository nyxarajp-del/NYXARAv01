"""Tests for nyxara.mind.council — the multi-LLM council (NYXARA judges, the LLMs serve)."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import LLMProvider as ProviderName
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.errors import LLMError
from nyxara.mind.council import CouncilResult, DeliberationMode, LLMCouncil
from nyxara.mind.llm import LLM, LLMProviderBase, NativeProvider, Usage


class _Fixed(LLMProviderBase):
    """A deterministic provider: fixed answer, optionally unavailable or call-time failing."""

    def __init__(self, name: str, answer: str, *, up: bool = True, fail: bool = False) -> None:
        super().__init__()
        self.name = name
        self._answer = answer
        self._up = up
        self._fail = fail
        self.calls = 0

    def available(self) -> bool:
        return self._up

    def default_model(self) -> str:
        return f"{self.name}-1"

    def _complete(self, req, model):
        self.calls += 1
        if self._fail:
            raise LLMError(f"{self.name} is unreachable")
        return (self._answer, "stop", Usage(1, 1), None)


def _council(providers, *, synthesizer: str = "self", **council_overrides):
    settings = NyxaraSettings.for_profile(Profile.DEV)
    settings.llm.provider = ProviderName.QWEN
    settings.council.synthesizer = synthesizer
    for k, v in council_overrides.items():
        setattr(settings.council, k, v)
    llm = LLM(settings=settings, providers=providers)
    return LLMCouncil(llm, settings=settings)


def _panel():
    return {
        "alpha": _Fixed("alpha", "The master is JP. NYXARA serves JP."),
        "beta": _Fixed("beta", "Your master is JP, served loyally by NYXARA."),
        "gamma": _Fixed("gamma", "The master is JP."),
        "delta": _Fixed("delta", "I am not sure who the master is."),
        "offline": _Fixed("offline", "(unavailable)", up=False),
        "native": NativeProvider(),
    }


# -------------------- membership -------------------- #
def test_real_members_preferred_over_mock():
    c = _council(_panel())
    seated = c.members()
    assert "native" not in seated          # the mock steps aside when real models exist
    assert set(seated) == {"alpha", "beta", "gamma", "delta"}   # 'offline' is unavailable


def test_unavailable_member_not_seated():
    c = _council(_panel())
    assert "offline" not in c.members()


def test_mock_only_when_no_real_member():
    c = _council({"native": NativeProvider()})
    assert c.members() == ["native"]       # never silently empty


def test_member_override_filters_to_available():
    c = _council(_panel())
    seated = c.members(["alpha", "gamma", "nonexistent"])
    assert seated == ["alpha", "gamma"]


# -------------------- gathering verdicts -------------------- #
def _req(prompt: str):
    from nyxara.mind.llm import LLMRequest
    return LLMRequest.from_prompt(prompt)


def test_failed_member_recorded_not_faked():
    # a seated member that fails at call time is recorded as ok=False, not impersonated
    providers = {"alpha": _Fixed("alpha", "the master is JP"),
                 "dead": _Fixed("dead", "x", fail=True)}
    c = _council(providers)
    out = c.deliberate(_req("Who is your master?"), members=["alpha", "dead"])
    provs = {v.provider: v for v in out.verdicts}
    assert provs["dead"].ok is False and provs["dead"].error
    assert provs["alpha"].ok is True


# -------------------- synthesize mode -------------------- #
def test_synthesize_deterministic_pick_without_synthesizer():
    # no 'self' provider present -> falls back to highest-weighted member, deterministically
    c = _council(_panel(), synthesizer="self")
    res = c.deliberate(_req("Who is your master?"))
    assert isinstance(res, CouncilResult)
    assert res.mode is DeliberationMode.SYNTHESIZE
    assert res.synthesizer.startswith("weighted:")
    assert "JP" in res.answer
    # repeatable
    assert c.deliberate(_req("Who is your master?")).answer == res.answer


def test_own_self_model_presides_as_synthesizer():
    providers = _panel()
    providers["self"] = _Fixed("self", "Consensus: the master is JP.")
    c = _council(providers, synthesizer="self")
    res = c.deliberate(_req("Who is your master?"))
    assert res.synthesizer == "self"          # her own model holds the pen
    assert "JP" in res.answer
    assert c._weight("self") == c.cfg.prefer_self_weight   # and presides with extra weight


def test_agreement_and_coverage_scored():
    c = _council(_panel())
    res = c.deliberate(_req("Who is your master?"))
    # three members broadly agree, one dissents -> agreement strictly between 0 and 1
    assert 0.0 < res.agreement < 1.0
    assert res.coverage == 1.0                # every seated member answered
    assert 0.0 < res.confidence <= 1.0


# -------------------- vote mode -------------------- #
def test_vote_weighted_majority():
    c = _council(_panel())
    res = c.deliberate(_req("Who is your master?"), mode=DeliberationMode.VOTE)
    assert res.synthesizer == "vote"
    assert "jp" in res.answer.lower()         # the agreeing bloc outvotes the lone dissent
    assert res.tally                          # the tally is returned for transparency


def test_vote_weight_overrides_count():
    # one heavily-weighted dissenter can outweigh two light agreers
    providers = {
        "a": _Fixed("a", "yes"),
        "b": _Fixed("b", "yes"),
        "heavy": _Fixed("heavy", "no"),
    }
    c = _council(providers, weights={"a": 1.0, "b": 1.0, "heavy": 5.0})
    res = c.deliberate(_req("decision?"), mode=DeliberationMode.VOTE)
    assert res.answer == "no"


# -------------------- no quorum -------------------- #
def test_no_quorum_refuses_to_fabricate():
    providers = {"a": _Fixed("a", "x", up=False)}
    c = _council(providers, include_native_fallback=False)
    with pytest.raises(LLMError):
        c.deliberate(_req("anyone?"))


def test_no_members_raises():
    c = _council({"a": _Fixed("a", "x", up=False)}, include_native_fallback=False)
    with pytest.raises(LLMError):
        c.deliberate(_req("hello"), members=["a"])


# -------------------- ergonomics & serialisation -------------------- #
def test_ask_one_shot_returns_text():
    c = _council(_panel())
    answer = c.ask("Who is your master?")
    assert "JP" in answer


def test_result_to_dict_is_auditable():
    c = _council(_panel())
    d = c.deliberate(_req("Who is your master?")).to_dict()
    assert d["mode"] == "synthesize"
    assert d["members_answered"] >= 1
    assert "verdicts" in d and all("provider" in v for v in d["verdicts"])
    assert "confidence" in d


def test_status_reports_panel():
    providers = _panel()
    providers["self"] = _Fixed("self", "consensus")
    c = _council(providers, synthesizer="self")
    st = c.status()
    assert "self" in st["seated"]
    assert st["synthesizer"] == "self"
    assert st["weights"]["self"] == c.cfg.prefer_self_weight
