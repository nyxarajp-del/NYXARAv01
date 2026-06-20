"""Tests for nyxara.mind.reasoning_chain — multi-step verifiable reasoning (offline, exact)."""

from __future__ import annotations

import pytest

from nyxara.mind.math import has_sympy
from nyxara.mind.reasoning_chain import parse_bindings, solve_chain


# --------------------------------------------------------------------------- #
# Variable bindings
# --------------------------------------------------------------------------- #
def test_parse_bindings():
    assert parse_bindings("If x = 5, what is 2x + 3?") == {"x": 5.0}
    assert parse_bindings("Given x = 4 and y = 2, what is x*y?") == {"x": 4.0, "y": 2.0}
    # no binding lead-in -> nothing bound (a bare equation is the algebra faculty's job)
    assert parse_bindings("x = 5") == {}


# --------------------------------------------------------------------------- #
# Substitution chains
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not has_sympy(), reason="substitution uses sympy for the general case")
def test_substitution_single_and_multi_var():
    r = solve_chain("If x = 5, what is 2x + 3?")
    assert r is not None and r.answer == "13"
    assert any("x = 5" in s for s in r.steps)            # the derivation is shown

    r2 = solve_chain("Given x = 4 and y = 2, what is x*y + 1?")
    assert r2 is not None and r2.answer == "9"


def test_substitution_defers_when_unbound():
    # target uses an unbound variable -> defer rather than guess
    assert solve_chain("If x = 5, what is x + z?") is None
    # no target use of a bound variable -> not a substitution chain
    assert solve_chain("If x = 5, what is 2 + 3?") is None


# --------------------------------------------------------------------------- #
# Follow-up operation chains
# --------------------------------------------------------------------------- #
def test_then_operation_chain():
    r = solve_chain("What is 20% of 150, then add 10?")
    assert r is not None and r.answer == "40"

    r2 = solve_chain("What is 12 * 12, then subtract 44?")
    assert r2 is not None and r2.answer == "100"


def test_then_chain_defers_without_a_structured_head():
    # head isn't exactly solvable -> defer
    assert solve_chain("What is your favourite colour, then add 10?") is None


# --------------------------------------------------------------------------- #
# Honest non-recognition
# --------------------------------------------------------------------------- #
def test_chain_defers_on_unstructured_text():
    assert solve_chain("How are you today?") is None
    assert solve_chain("What is the capital of France?") is None
    assert solve_chain("") is None
