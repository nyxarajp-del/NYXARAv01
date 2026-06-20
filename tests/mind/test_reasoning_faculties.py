"""Tests for nyxara.mind.reasoning_faculties — Phase 4 verifiable engines (offline, exact)."""

from __future__ import annotations


import pytest

from nyxara.mind.math import has_sympy
from nyxara.mind.reasoning_faculties import (extract_expression, extract_formula,
                                             parse_word_problem, solve_calculus,
                                             solve_comparative, solve_date, solve_equation,
                                             solve_percent, solve_sequence, solve_syllogism,
                                             solve_unit_convert, solve_with_faculties,
                                             tautology)


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
# Natural-language arithmetic word problems (parsed, then computed exactly)
# --------------------------------------------------------------------------- #
def test_word_problem_rate_plus_remainder():
    # the canonical "groups of items, plus loose ones" structure -> a*b + c
    assert parse_word_problem(
        "A crate holds 3 boxes and each box holds 2 apples. Another 1 apple sits "
        "loose beside the crate. How many apples are there in total?") == "3*2 + 1"


def test_word_problem_generalises_beyond_one_template():
    # different wording, same structure class — proves it parses structure, not a fixed sentence
    assert parse_word_problem(
        "There are 5 shelves with 4 books on every shelf. How many books in all?") == "5*4"
    assert parse_word_problem(
        "A team has 6 players and each player scored 3 goals. They were also awarded 2 "
        "extra goals. What is the total number of goals?") == "6*3 + 2"


def test_word_problem_subtraction_cue():
    expr = parse_word_problem(
        "Sam had 10 marbles and gave 3 away. How many marbles does he have left?")
    # 10 (no product) minus 3 — order of standalone terms follows the text
    assert expr in ("10 - 3", "-3 + 10")


def test_word_problem_solved_exactly_end_to_end():
    assert solve_with_faculties(
        "A crate holds 7 boxes and each box holds 6 apples. Another 1 apple sits loose. "
        "How many apples are there in total?") == ("43", 0.95)


def test_word_problem_abstains_when_unclear():
    # a quantity question but no arithmetic cue and unassignable structure -> defer (None)
    assert parse_word_problem("I have 3 apples and 2 oranges. What colour is the sky?") is None
    # decimals / money -> defer (not this faculty's whole-quantity class)
    assert parse_word_problem(
        "A bat and a ball cost 1.10 in total. How much is the ball?") is None
    # not a quantity question at all
    assert parse_word_problem("The 3 musketeers met 4 friends and chatted.") is None
    # a bare expression is the math faculty's job, not this one
    assert parse_word_problem("What is 12 * 12?") is None


def test_word_problem_does_not_break_bare_expression_path():
    # bare expressions still go to the exact math faculty at full confidence
    assert solve_with_faculties("2+3*4") == ("14", 1.0)


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
# Categorical syllogisms (transitive closure, deductive)
# --------------------------------------------------------------------------- #
def test_syllogism_transitive_yes():
    assert solve_syllogism(
        "All bloops are razzies, and all razzies are lazzies. Are all bloops "
        "definitely lazzies? Options: A) yes  B) no") == "yes"
    assert solve_syllogism(
        "All cats are mammals. All mammals are animals. Are all cats animals?") == "yes"


def test_syllogism_not_entailed_is_no():
    # X is a known category but Y is not reachable -> not provable -> "no"
    assert solve_syllogism(
        "All cats are mammals. All dogs are mammals. Are all cats dogs?") == "no"


def test_syllogism_abstains_on_hedged_or_unknown():
    # 'some' changes the logic -> defer
    assert solve_syllogism("Some bloops are razzies. Are all bloops razzies?") is None
    # no premises about the subject at all -> defer
    assert solve_syllogism("Are all wuzzles glorks?") is None
    # not a syllogism question
    assert solve_syllogism("All cats are mammals. What is a cat?") is None


def test_syllogism_via_solve_with_faculties():
    assert solve_with_faculties(
        "All A are B. All B are C. Are all A C?") == ("yes", 1.0)


# --------------------------------------------------------------------------- #
# Transitive comparisons (single-scale ordering, exact)
# --------------------------------------------------------------------------- #
def test_comparative_minimum_and_maximum():
    assert solve_comparative(
        "Tom is taller than Sam. Sam is taller than Ada. Who is shortest? "
        "A) Tom  B) Sam  C) Ada") == "Ada"
    assert solve_comparative(
        "Anu is older than Bob. Bob is older than Cy. Who is the oldest?") == "Anu"


def test_comparative_handles_mixed_directions_on_one_scale():
    # "shorter" and "taller" are the same scale (height) in opposite directions
    assert solve_comparative(
        "Sam is shorter than Tom. Ada is shorter than Sam. Who is tallest?") == "Tom"


def test_comparative_abstains_when_unsafe():
    # two different scales must not be combined -> defer
    assert solve_comparative(
        "Tom is taller than Sam. Sam is heavier than Ada. Who is tallest?") is None
    # not a 'who is the -est' question
    assert solve_comparative("Tom is taller than Sam. Is Tom tall?") is None
    # ambiguous / no comparisons
    assert solve_comparative("Who is the tallest person here?") is None


def test_comparative_via_solve_with_faculties():
    assert solve_with_faculties(
        "X is bigger than Y. Y is bigger than Z. Who is smallest?") == ("Z", 1.0)


def test_logic_word_problems_that_must_defer():
    # algebra trick and semantic odd-one-out are the neural mind's job, not a logic faculty
    assert solve_with_faculties(
        "A bat and a ball cost 1.10. The bat costs 1.00 more than the ball. "
        "How much is the ball, in cents? A) 10  B) 5  C) 1") is None
    assert solve_with_faculties(
        "Which word does not belong? A) rose  B) tulip  C) hammer  D) daisy") is None


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


# --------------------------------------------------------------------------- #
# Percentages (exact ratio)
# --------------------------------------------------------------------------- #
def test_percent_of_and_what_percent():
    assert solve_percent("What is 20% of 150?") == "30"
    assert solve_percent("15 percent of 200") == "30"
    assert solve_percent("30 is what percent of 120?") == "25%"


def test_percent_defers_on_noise():
    assert solve_percent("the project is 50 done") is None
    assert solve_percent("how are you") is None


def test_percent_end_to_end():
    assert solve_with_faculties("What is 20% of 150?") == ("30", 1.0)


# --------------------------------------------------------------------------- #
# Unit conversion (exact factors, incl. temperature)
# --------------------------------------------------------------------------- #
def test_unit_convert_length_and_temperature():
    assert solve_unit_convert("convert 5 km to miles").startswith("3.106856")
    assert solve_unit_convert("100 C to F") == "212°F"
    assert solve_unit_convert("32 F to C") == "0°C"


def test_unit_convert_defers_on_mismatch_or_unknown():
    assert solve_unit_convert("convert 5 km to kg") is None          # different dimensions
    assert solve_unit_convert("convert 5 widgets to gadgets") is None  # unknown units
    assert solve_unit_convert("how are you") is None


# --------------------------------------------------------------------------- #
# Algebra (solve for the unknown, SymPy)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not has_sympy(), reason="algebra needs sympy")
def test_algebra_linear_and_quadratic():
    assert solve_equation("solve 2x + 3 = 9") == "x = 3"
    assert solve_equation("x^2 = 4") == "x = -2, 2"
    assert solve_equation("x^2 - 5x + 6 = 0") == "x = 2, 3"


@pytest.mark.skipif(not has_sympy(), reason="algebra needs sympy")
def test_algebra_defers_without_equation_or_with_logic():
    assert solve_equation("what is the meaning of life") is None
    assert solve_equation("A -> B") is None                # propositional logic, not algebra
    assert solve_equation("2 + 2 = 4") is None             # no unknown to solve for


@pytest.mark.skipif(not has_sympy(), reason="algebra needs sympy")
def test_algebra_end_to_end():
    assert solve_with_faculties("solve 2x + 3 = 9") == ("x = 3", 1.0)


# --------------------------------------------------------------------------- #
# Calculus (derivative / integral / limit, SymPy)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not has_sympy(), reason="calculus needs sympy")
def test_calculus_derivative_integral_limit():
    assert solve_calculus("what is the derivative of x^2?") == "d/dx = 2*x"
    assert solve_calculus("differentiate x^3 + 2x") == "d/dx = 3*x**2 + 2"
    assert solve_calculus("integrate x^2 dx") == "∫ = x**3/3 + C"
    assert solve_calculus("limit of 1/x as x -> oo") == "0"


@pytest.mark.skipif(not has_sympy(), reason="calculus needs sympy")
def test_calculus_defers_without_a_cue():
    assert solve_calculus("what is x squared") is None
    assert solve_calculus("how are you") is None


# --------------------------------------------------------------------------- #
# Number sequences (constant-law progression)
# --------------------------------------------------------------------------- #
def test_sequence_arithmetic_and_geometric():
    assert solve_sequence("what comes next in 2, 4, 6, 8, ?") == "10"
    assert solve_sequence("next number: 3, 9, 27, ?") == "81"
    assert solve_sequence("continue 5, 10, 15, 20 ...") == "25"


def test_sequence_defers_on_no_pattern_or_no_cue():
    assert solve_sequence("2, 3, 5, 7, 11, ?") is None      # primes: no constant law
    assert solve_sequence("the numbers 2 4 6 are here") is None  # no 'next/sequence' cue


def test_sequence_end_to_end():
    assert solve_with_faculties("what comes next in 2, 4, 6, 8, ?") == ("10", 1.0)


# --------------------------------------------------------------------------- #
# Calendar arithmetic (stdlib, exact)
# --------------------------------------------------------------------------- #
def test_date_span_weekday_and_shift():
    assert solve_date("how many days between 2020-01-01 and 2020-02-01?") == "31"
    assert solve_date("what day of the week is 2024-01-01?") == "Monday"
    assert solve_date("5 days after 2024-01-01") == "2024-01-06"
    assert solve_date("3 days before 2024-01-10") == "2024-01-07"


def test_date_defers_without_iso_date():
    assert solve_date("what day is it today?") is None
    assert solve_date("how many days in a week?") is None


def test_date_does_not_get_eaten_by_arithmetic():
    # the ISO-date guard keeps "2020-01-01" from being computed as 2020-1-1 = 2018
    assert extract_expression("how many days between 2020-01-01 and 2020-02-01?") is None
    assert solve_with_faculties("how many days between 2020-01-01 and 2020-02-01?") == ("31", 1.0)
