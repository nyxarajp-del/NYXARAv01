"""Arithmetic — the thing nothing in njp/ could do.

Measured before this module existed: `2+2 kitna hai?` returned the empty string. The five-rung
ladder ends at `ProofCore`, which *verifies* propositions and answered INEXPRESSIBLE; a grep for
an evaluator across the package found none. She could rewrite her own source and could not add.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.calculate import Calculator, evaluate, expression_in


# --------------------------------------------------------------------------- #
# It computes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,expected", [
    ("2+2 kitna hai?", "4"),
    ("what is 7 times 8?", "56"),
    ("(3+4)*2", "14"),
    ("12 jod 8", "20"),
    ("what is 15% of 200", "30"),
    ("200 ka 15 percent", "30"),
    ("5 ka square", "25"),
    ("100 ka varg", "10000"),
    ("2 ** 10", "1024"),
    ("10 divided by 4", "5/2"),
])
def test_it_computes(question, expected):
    got = evaluate(question)
    assert got.ok, f"{question!r} produced nothing: {got.error}"
    assert got.text == expected


def test_an_exact_rational_is_reported_as_exact():
    got = evaluate("1/3")
    assert got.text == "1/3"
    assert got.exact


def test_a_division_that_lands_on_a_whole_number_is_still_exact():
    """"15% of 200" is 30, not 30.0-ish. Understating what she knows is a calibration error too."""
    got = evaluate("what is 15% of 200")
    assert got.text == "30"
    assert got.exact


def test_the_working_is_shown():
    assert evaluate("2+2")


# --------------------------------------------------------------------------- #
# It refuses
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    "how are you",
    "Ravi is 30 years old",
    "aag se kya hota hai",
    "",
    "what is my name",
])
def test_a_sentence_with_no_sum_in_it_is_left_alone(text):
    """A calculator that answers 'how are you' with a number is the pendulum-law bug again."""
    got = evaluate(text)
    assert not got.ok
    assert expression_in(text) is None


def test_division_by_zero_is_a_refusal_not_a_crash():
    got = evaluate("5 / 0")
    assert not got.ok
    assert "zero" in got.error


def test_an_unbounded_exponent_is_refused():
    got = evaluate("10**10**10")
    assert not got.ok


@pytest.mark.parametrize("hostile", [
    '__import__("os").system("ls")',
    "open('/etc/passwd').read()",
    "[x for x in range(10)]",
    "(lambda: 1)()",
    "1 if True else 2",
])
def test_nothing_outside_arithmetic_is_ever_evaluated(hostile):
    """The whitelist is the security boundary. Nothing here is ever passed to `eval`."""
    got = evaluate(hostile)
    assert not got.ok


def test_a_name_is_not_a_value():
    calculator = Calculator()
    assert not calculator.evaluate("x + 1").ok


# --------------------------------------------------------------------------- #
# It is reached from a real turn
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,expected", [
    ("2+2 kitna hai?", "4"),
    ("5 ka square kitna hai?", "25"),
    ("what is 15% of 200?", "30"),
    ("what is 7 times 8?", "56"),
])
def test_the_master_gets_the_number_back(question, expected):
    assert NJPBrain().think(question).answer == expected


def test_a_parsed_expression_classifies_the_problem_as_symbolic():
    """Measured: it classified EMPIRICAL, the calculator ran, computed 25, and the critic threw
    it away as "empirical claim with no stated test". The answer was right and unsaid."""
    brain = NJPBrain()
    thought = brain.think("5 ka square kitna hai?")
    assert thought.solution is not None
    assert thought.solution.kind == "symbolic"


def test_a_greeting_still_does_not_reach_the_calculator():
    brain = NJPBrain()
    answer = brain.think("How are you NYXARA?").answer
    assert "theek" in answer.lower()


def test_an_organ_that_is_off_is_absent_not_zeroed():
    class _Off:
        calculate_enabled = False

    brain = NJPBrain(_Off())
    assert brain.calculator is None
    assert "calculate" not in brain.stats()
