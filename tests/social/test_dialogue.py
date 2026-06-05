"""Tests for nyxara.social.dialogue."""

from __future__ import annotations

import pytest

from nyxara.social.dialogue import (
    Conversation,
    DialogueManager,
    ImplicatureEngine,
    ImplicatureType,
    SpeechAct,
    SubtextDetector,
    Utterance,
    detect_speech_act,
)


# -------------------- speech act detection -------------------- #
@pytest.mark.parametrize("text,act", [
    ("What is the status?", SpeechAct.QUESTION),
    ("Close the door.", SpeechAct.COMMAND),
    ("Could you check the logs?", SpeechAct.REQUEST),
    ("Please help me.", SpeechAct.REQUEST),
    ("Hello there", SpeechAct.GREETING),
    ("Thanks for that", SpeechAct.ACKNOWLEDGMENT),
    ("The network is secure.", SpeechAct.ASSERTION),
])
def test_detect_speech_act(text, act):
    assert detect_speech_act(text) is act


def test_empty_text_is_assertion():
    assert detect_speech_act("") is SpeechAct.ASSERTION


# -------------------- conversation / multi-party -------------------- #
def test_add_utterance():
    c = Conversation(["A", "B"])
    u = c.add("A", "Hello B")
    assert isinstance(u, Utterance)
    assert u.speaker == "A"
    assert len(c.turns) == 1


def test_participant_auto_added():
    c = Conversation()
    c.add("NewPerson", "hi")
    assert "NewPerson" in c.participants


def test_addressee_explicit():
    c = Conversation(["A", "B"])
    u = c.add("A", "anything", addressee="B")
    assert u.addressee == "B"


def test_addressee_at_mention():
    c = Conversation(["A", "Bob"])
    u = c.add("A", "Bob, can you help?")
    assert u.addressee == "Bob"


def test_addressee_default_previous_speaker():
    c = Conversation(["A", "B"])
    c.add("A", "first")
    u = c.add("B", "reply")
    assert u.addressee == "A"


def test_by_speaker():
    c = Conversation()
    c.add("A", "1")
    c.add("B", "2")
    c.add("A", "3")
    assert len(c.by_speaker("A")) == 2


def test_floor():
    c = Conversation()
    c.add("A", "x")
    assert c.floor() == "A"


# -------------------- turn-taking / interruption -------------------- #
def test_expected_speaker_after_question():
    c = Conversation(["JP", "NYXARA"])
    c.add("JP", "NYXARA, what is the status?")
    assert c.expected_speaker() == "NYXARA"


def test_no_expected_speaker_after_assertion():
    c = Conversation(["A", "B"])
    c.add("A", "The sky is blue.")
    assert c.expected_speaker() is None


def test_interruption_detected():
    c = Conversation(["JP", "NYXARA", "Guest"])
    c.add("JP", "NYXARA, what is the status?")  # expects NYXARA
    u = c.add("Guest", "Let me jump in.")        # Guest interrupts
    assert u.is_interruption


def test_no_interruption_when_expected_speaks():
    c = Conversation(["JP", "NYXARA"])
    c.add("JP", "NYXARA, status?")
    u = c.add("NYXARA", "All secure.")
    assert not u.is_interruption


def test_interruptions_list():
    c = Conversation(["A", "B", "C"])
    c.add("A", "B, answer me?")
    c.add("C", "interrupting")
    assert len(c.interruptions()) == 1


# -------------------- implicature -------------------- #
def test_scalar_implicature_some():
    imp = ImplicatureEngine().analyze("Some students passed.")
    assert any(i.type is ImplicatureType.SCALAR and i.implied == "not all" for i in imp)


def test_scalar_implicature_or():
    imp = ImplicatureEngine().analyze("You can have tea or coffee.")
    assert any(i.type is ImplicatureType.SCALAR for i in imp)


def test_relevance_implicature_cold():
    imp = ImplicatureEngine().analyze("It's cold in here.")
    assert any(i.type is ImplicatureType.RELEVANCE for i in imp)
    assert any("window" in i.implied or "warm" in i.implied for i in imp)


def test_relevance_implicature_thirsty():
    imp = ImplicatureEngine().analyze("I'm thirsty.")
    assert any(i.type is ImplicatureType.RELEVANCE for i in imp)


def test_irony_implicature():
    imp = ImplicatureEngine().analyze("Oh great, another error.")
    assert any(i.type is ImplicatureType.QUALITY for i in imp)


def test_no_implicature_plain():
    imp = ImplicatureEngine().analyze("The server is running normally.")
    assert imp == []


# -------------------- subtext -------------------- #
def test_indirect_request_could_you():
    st = SubtextDetector().detect("Could you check the logs?")
    assert st.indirect_request


def test_indirect_request_environmental():
    st = SubtextDetector().detect("It's cold in here.")
    assert st.indirect_request
    assert st.request_hint == "environmental hint"


def test_politeness_level():
    polite = SubtextDetector().detect("Could you please kindly help, thank you?")
    blunt = SubtextDetector().detect("Do it now.")
    assert polite.politeness > blunt.politeness


def test_hedging():
    st = SubtextDetector().detect("Maybe we could sort of try this, I think.")
    assert st.hedged


def test_no_hedging():
    st = SubtextDetector().detect("Do it now.")
    assert not st.hedged


def test_sarcasm():
    st = SubtextDetector().detect("Yeah right, that'll work.")
    assert st.sarcasm


def test_emotion_subtext():
    assert SubtextDetector().detect("I'm worried about this.").emotion == "afraid"
    assert SubtextDetector().detect("I'm so happy today.").emotion == "happy"
    assert SubtextDetector().detect("This is frustrating and annoying.").emotion == "angry"


def test_no_emotion():
    assert SubtextDetector().detect("The report is ready.").emotion is None


def test_subtext_to_dict():
    d = SubtextDetector().detect("Could you please help?").to_dict()
    assert "indirect_request" in d and "politeness" in d


# -------------------- dialogue manager -------------------- #
def test_dialogue_manager_add():
    dm = DialogueManager(["JP", "NYXARA"])
    r = dm.add("JP", "Could you check the logs please?")
    assert isinstance(r["utterance"], Utterance)
    assert r["subtext"].indirect_request
    assert "implicatures" in r


def test_dialogue_manager_tracks_turns():
    dm = DialogueManager()
    dm.add("A", "hi")
    dm.add("B", "hello")
    assert len(dm.conversation.turns) == 2


def test_dialogue_manager_status():
    dm = DialogueManager(["JP", "NYXARA"])
    dm.add("JP", "NYXARA, status?")
    s = dm.status()
    assert s["expected_next"] == "NYXARA"
    assert s["floor"] == "JP"
