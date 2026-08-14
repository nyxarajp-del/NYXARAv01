"""The on-device model's context window is budgeted, not discovered at send time.

Gemma-4-E2B has a 4096-token window and the runtime enforces it hard: over the limit it returns
INVALID_ARGUMENT, the rung is taken off the ladder for the whole process, and every later turn
silently falls back to the n-gram floor. Nothing budgeted against that, so a normal turn — system
prompt plus grounded context plus history — sailed past it and a 2.4 GB model was unusable for
real work while appearing installed and healthy.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.mind.llm import LiteRTLMProvider, LLMRequest, estimate_tokens

litert_lm = pytest.importorskip("litert_lm")


@pytest.fixture()
def provider() -> LiteRTLMProvider:
    """The provider without engine construction — only the budgeting is under test."""
    return LiteRTLMProvider.__new__(LiteRTLMProvider)


def _fit(provider: LiteRTLMProvider, system: str, prompt: str, *, max_tokens: int = 64):
    provider.settings = get_settings()
    history = [litert_lm.Message.system(system)] if system else []
    request = LLMRequest.from_prompt(prompt, system=system, max_tokens=max_tokens)
    return provider._fit_context(litert_lm, history, prompt, request)


def _tokens(provider: LiteRTLMProvider, history, prompt: str) -> int:
    return sum(estimate_tokens(provider._message_text(m)) for m in history) + estimate_tokens(prompt)


# ---- reading a message ------------------------------------------------------- #
def test_message_text_is_read_from_the_real_shape(provider: LiteRTLMProvider):
    """The payload is under `contents`, not `content`. Guessing the attribute returned "" for
    every message, so the budget measured zero, passed its own check, and trimmed nothing."""
    message = litert_lm.Message.system("hello world this is the system text")
    assert provider._message_text(message) == "hello world this is the system text"


def test_the_system_turn_is_identified(provider: LiteRTLMProvider):
    assert provider._is_system(litert_lm.Message.system("s"))
    assert not provider._is_system(litert_lm.Message.user("u"))


# ---- budgeting ---------------------------------------------------------------- #
def test_a_small_prompt_is_left_alone(provider: LiteRTLMProvider):
    history, prompt = _fit(provider, "You are NYXARA.", "Say hello.")
    assert prompt == "Say hello."
    assert provider._message_text(history[0]) == "You are NYXARA."


def test_an_oversized_system_prompt_is_trimmed_to_fit(provider: LiteRTLMProvider):
    huge = "You are NYXARA. " + ("Context detail about the system state. " * 900)
    history, prompt = _fit(provider, huge, "Say hello to the Master.")
    window = get_settings().llm.litertlm_context_tokens
    assert _tokens(provider, history, prompt) < window
    assert estimate_tokens(provider._message_text(history[0])) < estimate_tokens(huge)


def test_the_head_and_tail_of_the_system_prompt_survive(provider: LiteRTLMProvider):
    """The head carries her identity and the tail carries the task instructions; losing either
    changes what she is being asked to do."""
    huge = "IDENTITY-HEAD. " + ("filler. " * 4000) + " TASK-TAIL."
    history, _prompt = _fit(provider, huge, "go")
    trimmed = provider._message_text(history[0])
    assert "IDENTITY-HEAD" in trimmed
    assert "TASK-TAIL" in trimmed


def test_history_is_dropped_before_the_system_prompt(provider: LiteRTLMProvider):
    """Oldest turns are the least load-bearing thing in the window, and conversation state lives
    in njp/memory rather than here."""
    provider.settings = get_settings()
    system = "You are NYXARA."
    history = [litert_lm.Message.system(system)]
    for i in range(60):
        history.append(litert_lm.Message.user("an old conversational turn " * 40 + str(i)))
    request = LLMRequest.from_prompt("now", system=system, max_tokens=64)

    trimmed, prompt = provider._fit_context(litert_lm, history, "now", request)
    assert len(trimmed) < len(history)
    assert provider._message_text(trimmed[0]) == system, "the system turn was dropped first"
    assert _tokens(provider, trimmed, prompt) < get_settings().llm.litertlm_context_tokens


def test_a_message_larger_than_the_window_is_itself_trimmed(provider: LiteRTLMProvider):
    """The last resort: nothing else left to cut."""
    monster = "word " * 20000
    history, prompt = _fit(provider, "", monster)
    assert estimate_tokens(prompt) < get_settings().llm.litertlm_context_tokens
    assert len(prompt) < len(monster)


def test_room_is_reserved_for_the_completion(provider: LiteRTLMProvider):
    """The limit is on input PLUS output — budgeting only the input still overflows."""
    huge = "filler. " * 4000
    small = _fit(provider, huge, "go", max_tokens=64)
    large = _fit(provider, huge, "go", max_tokens=2048)
    assert _tokens(provider, *large) < _tokens(provider, *small)


def test_budgeting_never_raises_on_a_strange_message(provider: LiteRTLMProvider):
    """A failed trim must send the original and let the runtime say so, not break the turn."""
    provider.settings = get_settings()
    request = LLMRequest.from_prompt("hi", max_tokens=64)
    history, prompt = provider._fit_context(litert_lm, [object()], "hi", request)
    assert prompt == "hi"
