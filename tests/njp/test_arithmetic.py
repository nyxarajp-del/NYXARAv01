"""Worked sums recomputed rather than believed, and the honest size of what she can solve."""

from __future__ import annotations

from fractions import Fraction

import pytest

from nyxara.njp.arithmetic import (Arithmetic, Problem, Shape, _evaluate_expression,
                                   read_problems)
from nyxara.njp.arithmeticschool import audit, solve_paper, split


@pytest.fixture(scope="module")
def problems():
    return read_problems()


@pytest.fixture(scope="module")
def marked(problems):
    return audit(problems)


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def test_the_whole_submix_was_read_not_a_sample(problems):
    assert len(problems) > 20000


def test_every_problem_carries_the_chain_that_was_worked(problems):
    assert all(p.steps for p in problems)
    assert any(len(p.steps) > 2 for p in problems)


def test_a_step_is_a_whole_expression_not_a_pair_of_operands(problems):
    """Cutting `3/10 * 20/11 = 6/11` in two reported the corpus wrong about arithmetic it had
    got right."""
    assert all(len(step) == 2 for p in problems[:200] for step in p.steps)
    assert any(sum(c in "+-*/" for c in step[0]) > 1
               for p in problems[:2000] for step in p.steps)


# --------------------------------------------------------------------------------------------- #
#  the audit
# --------------------------------------------------------------------------------------------- #
def test_sums_go_through_her_own_calculator_and_never_through_eval():
    assert _evaluate_expression("3 * 40") == Fraction(120)
    assert _evaluate_expression("134400 / 3360") == Fraction(40)
    assert _evaluate_expression("__import__('os')") is None


def test_the_bracket_form_her_calculator_refuses_is_not_used():
    """With defensive brackets every one of 23,371 audits came back with nothing computed —
    a clean sheet that meant the checker had never run."""
    from nyxara.njp.calculate import evaluate

    assert not evaluate("(3) * (40)").ok
    assert evaluate("3 * 40").ok


def test_a_wrong_sum_is_caught():
    reader = Arithmetic()
    got = reader.check(Problem(steps=(("2 + 2", "5"),), answer="5"))
    assert not got.sound and "but it is 4" in got.wrong


def test_a_rounded_sum_is_not_called_wrong():
    """`45/4 = 11` has not made a mistake about arithmetic; it has rounded."""
    reader = Arithmetic()
    got = reader.check(Problem(steps=(("45 / 4", "11"),), answer="11"))
    assert not got.sound
    assert got.sound_allowing_rounding and not got.wrong


def test_a_letter_answer_is_not_counted_as_a_chain_that_missed():
    """Multiple-choice rows answer "(D)". Counting those as failures understated the corpus."""
    reader = Arithmetic()
    got = reader.check(Problem(steps=(("2 + 2", "4"),), answer="(D)"))
    assert got.sound and not got.numeric and not got.reaches


def test_most_of_the_corpus_computes_as_written(marked):
    row = marked.to_dict()
    assert row["exact"] > 0.85
    assert row["allowing_rounding"] >= row["exact"]
    assert row["reaches_its_answer"] > 0.75


# --------------------------------------------------------------------------------------------- #
#  shapes
# --------------------------------------------------------------------------------------------- #
def test_a_chain_becomes_a_program_over_the_question_s_own_numbers():
    reader = Arithmetic()
    problem = Problem(numbers=("3", "40", "60"), answer="220",
                      steps=(("3 * 40", "120"), ("120 + 40", "160"), ("60 + 160", "220")))
    shape = reader.shape_of(problem)
    assert shape is not None
    assert shape.render() == "q0 * q1; r0 + q1; q2 + r1"


def test_a_shape_runs_on_numbers_it_was_not_built_from():
    shape = Shape(moves=(("q0", "*", "q1"), ("r0", "+", "q2")))
    assert shape.run([Fraction(2), Fraction(5), Fraction(1)]) == Fraction(11)


def test_a_shape_with_nothing_to_bind_to_returns_nothing():
    shape = Shape(moves=(("q0", "*", "q9"),))
    assert shape.run([Fraction(2)]) is None


def test_a_step_with_more_than_one_operator_has_no_simple_shape():
    reader = Arithmetic()
    assert reader.shape_of(Problem(numbers=("3",), steps=(("3 * 4 + 5", "17"),))) is None


# --------------------------------------------------------------------------------------------- #
#  solving, and what it is worth
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def solving(problems):
    learn, held = split(problems)
    return solve_paper(learn[:4000], held[:1500])


def test_picking_a_shape_by_frequency_is_worth_nothing(solving):
    """The honest result: shape frequency carries no information about which problem needs which.

    Ranking the shapes and taking the first that binds scores what applying one shape to
    everything scores, and both are near zero. Solving needs reading the question.
    """
    assert solving["shapes"].accuracy < 0.10
    assert abs(solving["shapes"].accuracy - solving["one_shape"].accuracy) < 0.02


def test_with_no_shape_she_answers_nothing(solving):
    assert solving["nothing"].accuracy == 0.0
    assert solving["nothing"].answered == 0


def test_the_shapes_are_many_and_each_is_rare(problems):
    """2,315 distinct shapes and the commonest covers half a percent — that is the diagnosis."""
    learn, _held = split(problems)
    counts = Arithmetic().learn_shapes(learn[:6000])
    assert len(counts) > 300
    assert max(counts.values()) / sum(counts.values()) < 0.05
