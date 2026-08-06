"""Tests for observe/turn_ledger.py — what happened, as opposed to what was configured.

The bug this module exists for was not a crash. ``/learning`` reported::

    "chosen": "litertlm",  "engine_loaded": false

...while a 532-parameter n-gram answered every turn. Nothing was lying: ``chosen_provider()``
honestly reports which rung the ladder *would* pick. The question nobody could ask was which brain
actually spoke, because no event recorded it.

So the property under test is narrow and absolute: **a rung may only be credited with an answer it
produced.** Not configured, not selected, not available — produced.
"""
from __future__ import annotations

import threading

import pytest

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import LLM, LLMProviderBase, LLMRequest, NativeProvider, Usage
from nyxara.observe.turn_ledger import TurnLedger, get_ledger, reset_ledger


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_ledger()
    yield
    reset_ledger()


# --------------------------------------------------------------------------- #
# The invariant: credit is earned, never assigned
# --------------------------------------------------------------------------- #
def test_answered_by_names_the_rung_that_produced_text():
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("litertlm", "gemma", ok=False, error="send_message failed")
    ledger.record_generation("native", "nyxara-native", ok=True, chars=42)
    turn = ledger.end()
    assert turn.answered_by == "native"


def test_a_failed_attempt_is_never_credited():
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("litertlm", "gemma", ok=False, error="boom")
    assert ledger.end().answered_by is None


def test_an_empty_reply_is_not_an_answer():
    """A rung can return successfully and say nothing. Silence is not speech."""
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("self", "nyxara-self", ok=True, chars=0)
    assert ledger.end().answered_by is None


def test_answered_by_cannot_be_assigned():
    """The asymmetry IS the guarantee: no setting can make this claim on a rung's behalf."""
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("native", "nyxara-native", ok=True, chars=5)
    turn = ledger.end()
    with pytest.raises(AttributeError):
        turn.answered_by = "litertlm"        # type: ignore[misc]


def test_the_last_speaker_wins_not_the_first():
    """A turn can retry; the answer the Master read came from the last rung that produced one."""
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("a", "m", ok=True, chars=10)
    ledger.record_generation("b", "m", ok=True, chars=10)
    assert ledger.end().answered_by == "b"


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_a_generation_outside_a_turn_is_still_recorded():
    """Dropping unbracketed events would reintroduce exactly the blindness this removes."""
    ledger = TurnLedger()
    ledger.record_generation("native", "nyxara-native", ok=True, chars=4)
    assert ledger.last() is not None and ledger.last().answered_by == "native"


def test_begin_is_reentrant():
    """A nested caller must not split one turn into two and halve every count."""
    ledger = TurnLedger()
    first = ledger.begin()
    second = ledger.begin()
    assert first is second


def test_the_ring_is_bounded():
    ledger = TurnLedger(capacity=3)
    for _ in range(10):
        ledger.begin()
        ledger.record_generation("native", "m", ok=True, chars=1)
        ledger.end()
    assert len(ledger.recent(limit=99)) == 3


def test_tokens_and_attempts_are_summed():
    ledger = TurnLedger()
    ledger.begin()
    ledger.record_generation("a", "m", ok=True, prompt_tokens=5, completion_tokens=7, chars=3)
    ledger.record_generation("b", "m", ok=True, prompt_tokens=1, completion_tokens=2, chars=3)
    turn = ledger.end()
    assert turn.attempts == 2 and turn.tokens == 15


def test_concurrent_recording_loses_nothing():
    """The LLM facade is called from several threads; a dropped event is a blind spot."""
    ledger = TurnLedger()
    ledger.begin()

    def go():
        for _ in range(50):
            ledger.record_generation("native", "m", ok=True, chars=1)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ledger.end().attempts == 400


# --------------------------------------------------------------------------- #
# Wired into the facade — the whole point
# --------------------------------------------------------------------------- #
class _Broken(LLMProviderBase):
    name = "litertlm"

    def available(self):
        return True

    def default_model(self):
        return "gemma-4-E2B-it-litertlm"

    def _complete(self, req, model):
        raise RuntimeError("litert_lm_conversation_send_message failed")


def _ladder(providers):
    settings = NyxaraSettings.for_profile(Profile.DEV)
    llm = LLM(settings=settings, providers=providers)
    llm._AUTO_LADDER = tuple(providers)
    return llm


def test_the_facade_records_who_actually_spoke():
    llm = _ladder({"litertlm": _Broken(), "native": NativeProvider()})
    llm.complete(LLMRequest.from_prompt("Hii"))

    view = llm.learning_view()
    assert view["would_choose"] == "litertlm", "the ladder still favours the strong rung"
    assert view["actual"]["last_turn"]["answered_by"] == "native", "but this is who spoke"


def test_the_failed_attempt_is_recorded_with_its_reason():
    llm = _ladder({"litertlm": _Broken(), "native": NativeProvider()})
    llm.complete(LLMRequest.from_prompt("Hii"))

    gens = llm.learning_view()["actual"]["last_turn"]["generations"]
    assert [g["provider"] for g in gens] == ["litertlm", "native"]
    assert gens[0]["ok"] is False and "send_message" in gens[0]["error"]
    assert gens[1]["ok"] is True


def test_a_degradation_is_recorded_alongside():
    llm = _ladder({"litertlm": _Broken(), "native": NativeProvider()})
    llm.complete(LLMRequest.from_prompt("Hii"))
    degradations = llm.learning_view()["actual"]["last_turn"]["degradations"]
    assert degradations and "litertlm" in degradations[0]["reason"]


def test_a_healthy_turn_credits_the_rung_that_answered():
    llm = _ladder({"native": NativeProvider()})
    llm.complete(LLMRequest.from_prompt("Hii"))
    assert llm.learning_view()["actual"]["last_turn"]["answered_by"] == "native"


def test_an_unavailable_rung_is_recorded_as_an_attempt():
    """It was tried and produced nothing — that is a fact about the turn, not a non-event."""
    class _Down(LLMProviderBase):
        name = "litertlm"

        def available(self):
            return False

    prov = _Down()
    try:
        prov.complete(LLMRequest.from_prompt("x"))
    except Exception:  # noqa: BLE001 — expected
        pass
    turn = get_ledger().last()
    assert turn is not None
    assert any(g.provider == "litertlm" and not g.ok for g in turn.generations)


def test_counts_accumulate_across_turns():
    llm = _ladder({"native": NativeProvider()})
    for _ in range(3):
        get_ledger().begin()
        llm.complete(LLMRequest.from_prompt("Hii"))
        get_ledger().end()
    assert llm.learning_view()["actual"]["answered_by_counts"]["native"] == 3


def test_observability_never_breaks_a_turn(monkeypatch):
    """A broken ledger must cost visibility, never an answer."""
    import nyxara.observe.turn_ledger as tl

    def _boom(*a, **k):
        raise RuntimeError("ledger is on fire")

    monkeypatch.setattr(tl.TurnLedger, "record_generation", _boom)
    llm = _ladder({"native": NativeProvider()})
    resp = llm.complete(LLMRequest.from_prompt("Hii"))
    assert resp.text.strip(), "she must still answer"


def test_usage_is_carried_through_from_the_response():
    class _Counted(LLMProviderBase):
        name = "counted"

        def available(self):
            return True

        def default_model(self):
            return "m"

        def _complete(self, req, model):
            return ("hello there", "stop", Usage(prompt_tokens=11, completion_tokens=4), None)

    llm = _ladder({"counted": _Counted()})
    llm.complete(LLMRequest.from_prompt("Hii"))
    gen = llm.learning_view()["actual"]["last_turn"]["generations"][0]
    assert gen["prompt_tokens"] == 11 and gen["completion_tokens"] == 4
    assert gen["chars"] == len("hello there")
