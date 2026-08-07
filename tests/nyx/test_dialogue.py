"""NYX V.01 · speech — her content, honest about the voice, and never a script."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.dialogue import Dialogue, Surface


class _FakeLLM:
    """A stand-in language surface, so these tests never touch a real model."""

    def __init__(self, *, provider: str = "litertlm", text: str = "", raises: bool = False):
        self._provider, self._text, self._raises = provider, text, raises
        self.calls = 0

    def provider_status(self):
        return {"litertlm": self._provider == "litertlm", "self": True, "native": True}

    def chosen_provider(self):
        return type("P", (), {"name": self._provider})()

    def generate(self, prompt, *, system=None, max_tokens=None):
        self.calls += 1
        if self._raises:
            raise RuntimeError("model on fire")
        return self._text


@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "holo_capacity": 64})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return b


# -------------------- honesty about the voice -------------------- #
def test_a_degraded_surface_is_reported_not_disguised(brain):
    """The n-gram floor emits noise on a real prompt; passing that off as her reply is a lie."""
    d = Dialogue(llm=_FakeLLM(provider="native", text="::::::::::::::::"))
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert "::::" not in reply.text
    assert reply.rendered is False
    assert reply.surface.fluent is False and reply.surface.note


def test_the_note_says_how_to_fix_it(brain):
    d = Dialogue(llm=_FakeLLM(provider="native"))
    assert "fetch_litertlm_model" in d.surface().note


def test_a_degraded_surface_still_answers_from_her_own_cognition(brain):
    d = Dialogue(llm=_FakeLLM(provider="native", text="::::"))
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert "gravity" in reply.text.lower()          # her recall, in her own words


def test_a_fluent_surface_is_used_to_phrase_her_content(brain):
    llm = _FakeLLM(provider="litertlm",
                   text="Gravity is what pulls an apple toward the ground.")
    d = Dialogue(llm=llm)
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert reply.rendered is True and llm.calls == 1
    assert "gravity" in reply.text.lower()


def test_no_language_model_at_all_still_answers(brain):
    d = Dialogue(llm=None)
    d._tried = True                                  # pretend construction failed
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert reply.text and reply.rendered is False


# -------------------- the model may phrase, not invent -------------------- #
def test_a_rendering_that_wandered_off_is_rejected(brain):
    """A fluent model must not smuggle in an answer NYX never reached."""
    llm = _FakeLLM(provider="litertlm",
                   text="The capital of France is Paris and I recommend the Louvre.")
    d = Dialogue(llm=llm)
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert reply.rendered is False
    assert "paris" not in reply.text.lower()


def test_a_faithful_rendering_is_accepted(brain):
    llm = _FakeLLM(provider="litertlm",
                   text="Gravity pulls an apple down toward the ground.")
    d = Dialogue(llm=llm)
    assert d.respond(brain.think("what pulls an apple down"), brain=brain).rendered is True


def test_a_model_that_raises_falls_back_to_her_own_words(brain):
    d = Dialogue(llm=_FakeLLM(provider="litertlm", raises=True))
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert reply.rendered is False and "gravity" in reply.text.lower()


# -------------------- saying nothing, honestly -------------------- #
def test_with_nothing_to_say_she_says_so_and_names_what_is_ungrounded(brain):
    d = Dialogue(llm=_FakeLLM(provider="litertlm", text="whatever"))
    reply = d.respond(brain.think("zorblax quixotry vremkul"), brain=brain)
    if reply.source:
        pytest.skip("a specialist had something to say about this")
    assert "do not have anything" in reply.text.lower()


def test_an_undecided_answer_says_it_is_undecided():
    d = Dialogue(llm=_FakeLLM(provider="native"))
    thought = type("T", (), {"answer": "maybe this", "stimulus": "q", "winner": None,
                             "confidence": 0.3, "decided": False, "verified": False,
                             "percept": None, "assessment": None})()
    assert "not settled" in d.respond(thought).text.lower()


def test_a_none_thought_is_an_empty_reply():
    assert Dialogue(llm=_FakeLLM()).respond(None).text == ""


# -------------------- not a script -------------------- #
def test_the_same_question_answers_differently_as_she_learns():
    """A script cannot do this: the reply depends on what she knows."""
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "holo_capacity": 64})
    b = NyxBrain(cfg)
    d = Dialogue(llm=_FakeLLM(provider="native"))

    before = d.respond(b.think("what pulls an apple down"), brain=b).text
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    after = d.respond(b.think("what pulls an apple down"), brain=b).text

    assert after != before and "gravity" in after.lower()


def test_every_reply_carries_provenance(brain):
    d = Dialogue(llm=_FakeLLM(provider="native"))
    reply = d.respond(brain.think("what pulls an apple down"), brain=brain)
    assert reply.source and (reply.evidence or reply.why)


def test_replies_are_not_drawn_from_a_fixed_phrase_table(brain):
    """Different questions must produce genuinely different text, not slots in a template."""
    d = Dialogue(llm=_FakeLLM(provider="native"))
    texts = {d.respond(brain.think(q), brain=brain).text
             for q in ("what pulls an apple down",
                       "prove that (x+1)^2 = x^2 + 2x + 1",
                       "zorblax quixotry vremkul")}
    assert len(texts) == 3


# -------------------- wired into the brain -------------------- #
def test_converse_returns_a_reply(brain):
    reply = brain.converse("what pulls an apple down")
    assert reply.text and reply.surface is not None


def test_a_brain_without_dialogue_still_answers():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "dialogue_enabled": False})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    assert b.dialogue is None
    assert "gravity" in b.converse("what pulls an apple down").text.lower()


def test_surface_reports_honestly_when_it_cannot_be_inspected():
    class _Broken:
        def provider_status(self):
            raise RuntimeError("nope")

    got = Dialogue(llm=_Broken()).surface()
    assert isinstance(got, Surface) and got.fluent is False and got.note
