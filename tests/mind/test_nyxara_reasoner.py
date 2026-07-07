"""Tests for nyxara.mind.nyxara_reasoner — the sovereign-brain handoff (Step D).

When a real teacher LLM is configured AND NYXARA's own forged model is promoted, the
integrated reasoner must let her OWN model draft first and hand off when it clears the
verifier — instead of always deferring to the teacher (which kept the handoff rate at 0%).
Scripted minds stand in for real models, so no network and no trained weights are needed.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.nyxara_reasoner import NyxaraReasoner
from nyxara.mind.router import Router


class _OwnProvider:
    """A scripted stand-in for NYXARA's forged model (SelfProvider)."""

    def __init__(self, text, ok=True):
        self._t, self._ok = text, ok

    def available(self):
        return self._ok

    def complete(self, req):
        return type("_R", (), {"text": self._t})()


class _Teacher:
    def available_providers(self):
        return ["tinyllama", "self", "mock"]

    def complete_with(self, name, req):
        return type("_R", (), {"text": f"[teacher:{name}] a careful answer"})()


class _FakeProvider:
    name = "tinyllama"  # not "mock" → _real_llm() is True, so the teacher path is live


class _FakeLLM:
    """A real-provider facade whose teacher reply is clearly tagged."""

    def __init__(self, settings):
        self.settings = settings

    def chosen_provider(self):
        return _FakeProvider()

    def generate(self, prompt, system=None, **kw):
        return "[teacher-fallback] a careful answer from the external model"


def _router(self_text, ok=True):
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.router.enabled = True
    s.router.threshold = 0.5
    return Router(_Teacher(), settings=s, self_provider=_OwnProvider(self_text, ok=ok))


def _reasoner():
    s = NyxaraSettings.for_profile(Profile.TEST)
    return NyxaraReasoner(llm=_FakeLLM(s), settings=s), s


def test_confident_own_model_is_handed_off_not_the_teacher():
    r, _ = _reasoner()
    # inject an own-model-first router whose forged model answers well
    r._router = _router("The capital of France is Paris, Master.")
    cand = r("What is the capital of France?")
    assert cand.kind == "respond"
    assert "own model" in cand.rationale
    assert "teacher-fallback" not in cand.text


def test_weak_own_model_falls_through_to_teacher():
    r, _ = _reasoner()
    r._router = _router("the the the the the")  # degenerate → verifier rejects → no handoff
    cand = r("Explain thermodynamics in depth.")
    assert cand.kind == "respond"
    assert "own model" not in cand.rationale
    assert "teacher-fallback" in cand.text


def test_no_promoted_model_still_uses_teacher():
    r, _ = _reasoner()
    r._router = _router("anything", ok=False)  # SelfProvider.available() == False
    cand = r("Tell me about the sovereign loop.")
    assert "own model" not in cand.rationale
    assert "teacher-fallback" in cand.text


def test_verifiable_faculty_still_short_circuits_before_any_model():
    # math is answered by her own faculty, ahead of own-model and teacher alike (handoff)
    r, _ = _reasoner()
    r._router = _router("a wrong distracting answer")
    cand = r("what is 12 * 12?")
    assert cand.kind == "respond"
    assert "verifiable faculty" in cand.rationale
    assert "144" in cand.text
