"""Tests for nyxara.mind.reasoning_faculties — Phase 4 verifiable engines (offline, exact)."""

from __future__ import annotations

import pytest

from nyxara.mind.reasoning_faculties import (extract_expression, extract_formula,
                                             solve_with_faculties, tautology)


# --------------------------------------------------------------------------- #
# Arithmetic extraction + exact compute
# --------------------------------------------------------------------------- #
def test_extract_expression_finds_real_expressions():
    assert extract_expression("What is 12 * 12?") == "12 * 12"
    assert extract_expression("compute 2+3*4") == "2+3*4"
    assert extract_expression("100 / 4 please") == "100 / 4"


def test_extract_expression_ignores_word_problems():
    # numbers without an operator between them must NOT look like an expression
    assert extract_expression("A crate holds 3 boxes and 2 apples sit beside it.") is None
    assert extract_expression("hello there") is None


def test_math_faculty_computes_exactly():
    assert solve_with_faculties("What is 12 * 12?") == ("144", 1.0)
    assert solve_with_faculties("2+3*4") == ("14", 1.0)        # precedence honoured
    assert solve_with_faculties("100/4") == ("25", 1.0)


# --------------------------------------------------------------------------- #
# Propositional logic
# --------------------------------------------------------------------------- #
def test_tautology_modus_ponens():
    valid, counter = tautology("(A -> B) and A -> B")
    assert valid is True and counter is None


def test_tautology_finds_counterexample():
    valid, counter = tautology("A -> B")
    assert valid is False
    assert counter == {"A": True, "B": False}


def test_tautology_iff_and_or():
    assert tautology("A or not A")[0] is True
    assert tautology("(A <-> B) <-> (B <-> A)")[0] is True
    assert tautology("A and not A")[0] is False


def test_extract_formula_requires_a_connective():
    assert extract_formula("Is A -> B a tautology?") == "A -> B"
    assert extract_formula("just some words") is None
    assert extract_formula("A and B") is None        # no implication/iff signal -> not formal


def test_logic_faculty_via_solve():
    assert solve_with_faculties("Is (A -> B) and A -> B a tautology?") == \
        ("valid (a tautology)", 1.0)
    text, conf = solve_with_faculties("Is A -> B valid?")
    assert "not valid" in text and conf == 1.0


# --------------------------------------------------------------------------- #
# Honest non-recognition
# --------------------------------------------------------------------------- #
def test_solve_returns_none_for_unstructured_text():
    assert solve_with_faculties("Write a poem about loyalty.") is None
    assert solve_with_faculties("How are you today?") is None


def test_registry_prefers_verifiable():
    from nyxara.mind.reasoning_faculties import build_default_faculties
    from nyxara.mind.faculties import Task, TaskType
    _, selector = build_default_faculties()
    chosen = selector.select(Task(TaskType.ARITHMETIC, "what is 9*9", payload="9*9"))
    assert chosen is not None and chosen.name == "math" and chosen.verifiable
