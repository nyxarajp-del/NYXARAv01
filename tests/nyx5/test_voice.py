"""Wit matrix: strips boilerplate, but never touches a safety/refusal message."""

from __future__ import annotations

from nyxara.nyx5.voice import WitMatrix


def test_strips_boilerplate():
    v = WitMatrix()
    out = v.restyle("As an AI language model, I am happy to help! The answer is 42.")
    assert "As an AI" not in out.text
    assert "happy to help" not in out.text
    assert "42" in out.text
    assert out.restyled is True


def test_safety_message_preserved():
    v = WitMatrix(safety_immutable=True)
    msg = "I can't help with that — it would cause harm."
    out = v.restyle(msg)
    assert out.text == msg               # untouched
    assert out.safety_preserved is True
    assert out.restyled is False


def test_refusal_not_restyled_even_with_boilerplate():
    v = WitMatrix(safety_immutable=True)
    msg = "As an AI, I will not assist with that request."
    out = v.restyle(msg)
    assert out.safety_preserved is True
    assert out.text == msg               # boilerplate NOT stripped from a refusal


def test_plain_text_unchanged_content():
    v = WitMatrix()
    out = v.restyle("Deploy the service to staging.")
    assert "Deploy the service to staging." in out.text
