"""Tests for nyxara.mind.router — Phase 2 confidence router (offline, scripted minds).

No network: scripted own/teacher providers stand in for real models, so we assert the
intrinsic verifier, the own-first / teacher-fallback decision, and the honest empty path."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.router import Router, RouterResult, answer_quality, default_verifier


# --------------------------------------------------------------------------- #
# Intrinsic verifier
# --------------------------------------------------------------------------- #
def test_verifier_zero_on_empty_and_too_short():
    assert answer_quality("q", "") == 0.0
    assert answer_quality("q", "a", min_chars=5) == 0.0


def test_verifier_punishes_degenerate_repetition():
    good = answer_quality("Explain entropy.", "Entropy measures disorder in a system.")
    degen = answer_quality("Explain entropy.", "the the the the the the")
    assert degen < 0.4 < good


def test_verifier_punishes_pure_echo():
    prompt = "What is the capital of France today?"
    echo = answer_quality(prompt, "what is the capital of france today")
    real = answer_quality(prompt, "It is Paris.")
    assert echo < real


def test_default_verifier_respects_min_chars():
    v = default_verifier(min_chars=10)
    assert v("q", "short") == 0.0
    assert v("q", "a clearly longer and coherent answer") > 0.0


# --------------------------------------------------------------------------- #
# Scripted minds
# --------------------------------------------------------------------------- #
class _Own:
    def __init__(self, text, ok=True):
        self._t, self._ok = text, ok

    def available(self):
        return self._ok

    def complete(self, req):
        return type("_R", (), {"text": self._t})()


class _Teacher:
    def __init__(self, providers=("anthropic", "self", "mock")):
        self._p = list(providers)

    def available_providers(self):
        return list(self._p)

    def complete_with(self, name, req):
        return type("_R", (), {"text": f"[teacher:{name}] a careful, correct answer"})()


def _settings(**router):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.router.enabled = True
    s.router.threshold = 0.5
    for k, v in router.items():
        setattr(s.router, k, v)
    return s


# --------------------------------------------------------------------------- #
# Routing decisions
# --------------------------------------------------------------------------- #
def test_confident_own_answer_is_handed_off():
    r = Router(_Teacher(), settings=_settings(),
               self_provider=_Own("The capital is Paris, Master."))
    res = r.draft("What is the capital of France?")
    assert res.source == "self" and res.handed_off is True
    assert res.confidence >= 0.5


def test_weak_own_answer_defers_to_teacher():
    r = Router(_Teacher(), settings=_settings(),
               self_provider=_Own("the the the the the"))
    res = r.draft("Explain entropy.")
    assert res.source == "teacher" and res.handed_off is False
    assert res.text.startswith("[teacher:anthropic]")


def test_high_threshold_forces_teacher():
    r = Router(_Teacher(), settings=_settings(threshold=0.99),
               self_provider=_Own("A perfectly fine and coherent answer about France."))
    assert r.draft("Tell me about France.").source == "teacher"


def test_no_self_model_uses_teacher():
    r = Router(_Teacher(), settings=_settings(), self_provider=_Own("x", ok=False))
    assert r.draft("hi").source == "teacher"


def test_no_self_no_teacher_is_honest_none():
    r = Router(_Teacher(providers=["mock"]), settings=_settings(),
               self_provider=_Own("x", ok=False))
    res = r.draft("hi")
    assert res.source == "none" and res.text == "" and res.handed_off is False


def test_no_teacher_keeps_a_confident_own_answer():
    r = Router(_Teacher(providers=["mock"]), settings=_settings(consult_teacher=False),
               self_provider=_Own("A coherent, confident answer, Master."))
    res = r.draft("status?")
    assert res.source == "self" and res.handed_off is True


# --------------------------------------------------------------------------- #
# Reasoner gating
# --------------------------------------------------------------------------- #
def test_reasoner_does_not_route_when_disabled():
    from nyxara.mind.llm_reasoner import LLMReasoner
    settings = NyxaraSettings.for_profile(Profile.TEST)   # router disabled by default
    reasoner = LLMReasoner(settings=settings)
    assert reasoner._maybe_route("hello") is None


def test_router_result_serialises():
    d = RouterResult("hi there", "self", 0.81, True).to_dict()
    assert d["source"] == "self" and d["handed_off"] is True and d["confidence"] == 0.81
