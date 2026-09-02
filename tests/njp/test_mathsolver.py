"""Problems she has never seen — and the difference between dispatch and solving.

`nyxara.njp.mathematics` scores 410/410 on its own examination, and that number says almost
nothing: every item on it is a shape it already knows. Measured on thirty problems written to
match no skill — multi-step commerce, a set-up-and-solve, a modular exponent, a Diophantine count,
a draw without replacement:

    right                    1 / 30
    confidently wrong        9 / 30
    silent                  18 / 30

The nine wrong are the half that matters. "Marks up 40% then discounts 25%" answered **30** — the
discount skill firing on a percentage it recognised, inside a problem it did not. "The remainder
when 2^100 is divided by 7" returned 2^100 in full. "Two drawn without replacement" answered the
with-replacement number. A trigger that matches half a problem answers half a problem, and nothing
in a regex can notice the other half.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.mathematics import MathError, numbers_in
from nyxara.njp.mathsolver import (
    Expr,
    is_prime,
    looks_like_a_task,
    Problem,
    Solver,
    combinations,
    count_integer_solutions,
    crt_smallest,
    divisor_sum,
    factorial_valuation,
    harmonic_mean_speed,
    infinite_geometric_sum,
    read_number_relations,
    search_integers,
    shoelace_area,
    solve_algebraic,
    solve_linear,
)
from nyxara.njp.school import ExamConditions


@pytest.fixture(scope="module")
def solver() -> Solver:
    return Solver()


# --------------------------------------------------------------------------- #
# Expr — several unknowns, exactly
# --------------------------------------------------------------------------- #

def test_a_polynomial_in_several_unknowns():
    x, y = Expr.variable("x"), Expr.variable("y")
    assert ((x + y) * (x - y)).text() == "x^2 - y^2"
    assert (x * 2 + y * 3 - 12).is_linear
    assert ((x + y) ** 2).text() == "x^2 + 2xy + y^2"
    assert (x * x + y).at({"x": 3, "y": 4}) == 13


def test_substitution_is_how_an_unknown_leaves():
    x, y = Expr.variable("x"), Expr.variable("y")
    assert (x + y).substitute({"y": x * 2}).text() == "3x"
    assert (x * y).substitute({"y": 5}).text() == "5x"


def test_dividing_by_an_unknown_is_refused():
    """A rational function is not a polynomial, and this type cannot hold one."""
    with pytest.raises(MathError):
        Expr.variable("x") / Expr.variable("y")


# --------------------------------------------------------------------------- #
# The algebra engine
# --------------------------------------------------------------------------- #

def test_a_linear_system_is_solved_by_elimination():
    x, y = Expr.variable("x"), Expr.variable("y")
    found = solve_linear([x + y - 10, x - y - 2], ["x", "y"])
    assert found == {"x": 6, "y": 4}


def test_an_underdetermined_system_is_not_partly_answered():
    """One unknown out of three, handed on as though the system were solved, is a wrong number."""
    x, y = Expr.variable("x"), Expr.variable("y")
    assert solve_linear([x + y - 10], ["x", "y"]) is None


def test_an_inconsistent_system_has_no_solution():
    x, y = Expr.variable("x"), Expr.variable("y")
    assert solve_linear([x + y - 10, x + y - 12], ["x", "y"]) is None


def test_a_nonlinear_system_is_solved_by_substitution():
    problem = Problem()
    x, y = Expr.variable("x"), Expr.variable("y")
    problem.require(x + y, 20)
    problem.require(x * y, 96)
    found = {tuple(sorted(a.values())) for a in solve_algebraic(problem)}
    assert found == {(8, 12)}


# --------------------------------------------------------------------------- #
# The search and counting engines
# --------------------------------------------------------------------------- #

def test_search_is_exhaustive_and_says_so():
    assert search_integers(lambda n: n % 7 == 3 and n % 5 == 1, low=1, high=200) == 31
    assert search_integers(lambda n: n < 0, low=1, high=50) is None


@pytest.mark.parametrize("call,expected", [
    (lambda: factorial_valuation(100, 5), 24),
    (lambda: factorial_valuation(50, 5), 12),
    (lambda: divisor_sum(28), 56),
    (lambda: combinations(7, 3), 35),
    (lambda: crt_smallest([3, 4, 5], [2, 2, 2]), 62),
    (lambda: len(count_integer_solutions([2, 3], 12)), 1),
])
def test_the_counting_engines(call, expected):
    assert call() == expected


def test_a_convention_is_taken_openly_and_can_be_declined():
    """62 is the textbook answer; 2 satisfies every stated condition. The caller may have either."""
    assert crt_smallest([3, 4, 5], [2, 2, 2]) == 62
    assert crt_smallest([3, 4, 5], [2, 2, 2], low=1) == 2


def test_a_series_that_does_not_converge_has_no_sum():
    from fractions import Fraction
    assert infinite_geometric_sum(Fraction(3), Fraction(1, 2)) == 6
    with pytest.raises(MathError):
        infinite_geometric_sum(Fraction(2), Fraction(2))


def test_the_average_speed_is_harmonic_and_never_arithmetic():
    from fractions import Fraction
    assert harmonic_mean_speed([Fraction(30), Fraction(60)]) == 40


def test_the_area_of_a_triangle_from_its_vertices():
    from fractions import Fraction
    assert shoelace_area([(Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)),
                          (Fraction(0), Fraction(3))]) == 6


# --------------------------------------------------------------------------- #
# The problems — thirty built against, thirty written afterwards
# --------------------------------------------------------------------------- #

#: Written *first*, and the solver was built until they passed. Three of their expected answers
#: were wrong when this list was written and the solver was right — a work-rate that is 6 days and
#: not 7, an age that is 12 and not 24, a bridge crossing of 100/3 seconds nobody had worked out.
BUILT_AGAINST = [
    ("a shopkeeper marks up the price by 40% and then gives a discount of 25%. "
     "what is the overall profit percent?", "5%"),
    ("the sum of three consecutive even numbers is 78. what is the largest of them?", "28"),
    ("a train 150 m long crosses a pole in 10 seconds. how long will it take to cross a "
     "bridge 350 m long?", "100/3"),
    ("find the smallest number which when divided by 3, 4 and 5 leaves a remainder of 2 "
     "in each case", "62"),
    ("in how many ways can 5 people be seated in a row?", "120"),
    ("the average of 5 numbers is 20. when one number is removed the average becomes 18. "
     "what number was removed?", "28"),
    ("A can do a piece of work in 12 days and B in 15 days. they work together for 4 days "
     "and then A leaves. how many more days does B take?", "6"),
    ("if x + 1/x = 3, find the value of x^2 + 1/x^2", "7"),
    ("what is the remainder when 2^100 is divided by 7?", "2"),
    ("the sum of the digits of a two digit number is 9. reversing the digits increases the "
     "number by 27. find the number", "36"),
    ("a pipe fills a tank in 6 hours and another empties it in 8 hours. if both are opened "
     "how long will the tank take to fill?", "24"),
    ("what is the greatest number that divides 43, 91 and 183 leaving the same remainder "
     "in each case?", "4"),
    ("how many trailing zeros does 100 factorial have?", "24"),
    ("find the area of the triangle with vertices (0,0), (4,0) and (0,3)", "6"),
    ("the ratio of the ages of two people is 4:3. after 6 years it becomes 6:5. "
     "what is the present age of the elder?", "12"),
    ("what is the sum of the infinite geometric series with first term 3 and "
     "common ratio 1/2?", "6"),
    ("what is the probability of drawing a red king from a standard pack of 52 cards?", "1/26"),
    ("how many positive integer solutions does 2x + 3y = 12 have?", "1"),
    ("the compound interest on a sum for 2 years at 10 percent per annum is 420. "
     "find the sum", "2000"),
    ("one number is 3 more than twice another and their sum is 27. find the smaller number", "8"),
    ("how many diagonals does a polygon with 12 sides have?", "54"),
    ("what is the sum of all the factors of 28?", "56"),
    ("if 3 men can build a wall in 8 days, how many days will 4 men take?", "6"),
    ("a bag has 4 red and 6 blue balls. two are drawn without replacement. "
     "what is the probability both are red?", "2/15"),
    ("what is the last digit of 7^100?", "1"),
    ("the perimeter of a rectangle is 36 and its length is twice its width. "
     "what is the area?", "72"),
    ("what is the value of 1 + 2 + 3 + ... + 99 + 100 minus the sum of the "
     "first 50 natural numbers?", "3775"),
    ("find two numbers whose sum is 20 and product is 96. what is the larger?", "12"),
    ("what is the hcf of 2^4 times 3^2 and 2^2 times 3^3?", "36"),
    ("a car travels 60 km at 30 km/h and returns at 60 km/h. what is the average speed "
     "for the whole journey?", "40"),
]

#: Written **after** the solver was finished and not tuned against, with three shapes deliberately
#: chosen as ones nothing had been built for. First measurement: 23 right, 1 wrong, 6 silent.
WRITTEN_AFTER = [
    ("the sum of four consecutive odd numbers is 80. find the smallest", "17"),
    ("two numbers differ by 4 and their product is 96. what is the larger?", "12"),
    ("a shopkeeper marks his goods up by 20% and allows a discount of 10%. "
     "find his gain percent", "8%"),
    ("what is the remainder when 3^50 is divided by 5?", "4"),
    ("how many zeros are at the end of 50 factorial?", "12"),
    ("the greatest number which divides 62, 132 and 237 leaving the same remainder", "35"),
    ("in how many ways can 4 books be arranged on a shelf?", "24"),
    ("how many diagonals are there in a polygon of 8 sides?", "20"),
    ("the average of 6 numbers is 15. if a number is added the average becomes 16. "
     "what number was added?", "22"),
    ("a tap fills a tank in 4 hours and a leak empties it in 12 hours. "
     "how long will it take to fill?", "6"),
    ("5 workers dig a trench in 12 days. how many days will 10 workers take?", "6"),
    ("if y + 1/y = 4 find y^2 + 1/y^2", "14"),
    ("find the area of the triangle with vertices (1,1), (5,1) and (1,4)", "6"),
    ("the ratio of the ages of A and B is 5:2. after 8 years the ratio is 7:4. "
     "what is the present age of the younger?", "8"),
    ("what is the sum to infinity of the geometric series 4, 2, 1, ...", "8"),
    ("what is the probability of getting a black queen from a pack of cards?", "1/26"),
    ("a box has 5 white and 3 black balls. two are drawn without replacement. "
     "find the probability that both are white", "5/14"),
    ("how many positive integer solutions does 3x + 5y = 30 have?", "1"),
    ("the simple interest on a sum for 3 years at 5 percent is 300. find the sum", "2000"),
    ("the sum of the digits of a two digit number is 12 and the difference of the digits "
     "is 4. find the number", "84"),
    ("what is the sum of all the divisors of 36?", "91"),
    ("how many factors does 60 have?", "12"),
    ("the perimeter of a rectangle is 48 and the length is 3 times the width. "
     "find the area", "108"),
    ("a man walks 5 km at 5 km/h and returns at 10 km/h. what is his average speed?", "20/3"),
    ("what is the last digit of 3^2019?", "7"),
    ("find the smallest number which when divided by 6, 8 and 12 leaves remainder 5 "
     "in each case", "29"),
    ("the sum of two numbers is 15 and the sum of their squares is 125. find the larger", "10"),
    ("in how many ways can a committee of 3 be chosen from 7 people?", "35"),
    ("what is the hcf of 2^3 times 5^2 and 2^5 times 5?", "40"),
    ("how many arrangements are there of the letters of the word banana?", "60"),
]


@pytest.mark.parametrize("question,expected", BUILT_AGAINST)
def test_the_problems_it_was_built_against(solver, question, expected):
    got = solver.solve(question)
    assert got.ok, f"{question!r}: {got.error}"
    assert got.answer == expected


@pytest.mark.parametrize("question,expected", WRITTEN_AFTER)
def test_the_problems_written_after_it_was_finished(solver, question, expected):
    got = solver.solve(question)
    assert got.ok, f"{question!r}: {got.error}"
    assert got.answer == expected


# --------------------------------------------------------------------------- #
# Every defect the exam found, pinned
# --------------------------------------------------------------------------- #

def test_two_rates_in_sequence_compose_rather_than_subtract(solver):
    """The dispatcher answered 30: the discount skill, on a percentage inside another problem."""
    assert solver.solve("a shopkeeper marks up the price by 40% and then gives a discount "
                        "of 25%. what is the overall profit percent?").answer == "5%"


def test_a_modulus_is_not_a_decoration_on_an_exponent(solver):
    """`\\D` could not skip the words between "remainder" and "divided" — the thing being divided
    is a number, so the question failed on its own subject and came back silent."""
    assert solver.solve("what is the remainder when 2^100 is divided by 7?").answer == "2"


def test_a_number_at_the_end_of_a_sentence_is_a_number():
    """Written `(?![\\w.])`, the reader saw **no number at all** in "the sum is 78." — the full
    stop failed the lookahead, and every multi-sentence problem lost its last quantity."""
    assert numbers_in("the sum is 78. what is the largest") == [78]
    assert numbers_in("v1.2 is a version") == []


def test_a_run_is_read_to_its_last_term(solver):
    """Non-greedy, "… + 99 + 100" captured 99, and the whole series was the wrong one."""
    assert solver.solve("what is the value of 1 + 2 + 3 + ... + 99 + 100 minus the sum of "
                        "the first 50 natural numbers?").answer == "3775"


def test_a_factor_written_without_an_exponent_is_still_a_factor(solver):
    """Reading only the powers dropped the bare 5 out of "2^5 times 5", and the hcf came out 8."""
    assert solver.solve("what is the hcf of 2^3 times 5^2 and 2^5 times 5?").answer == "40"


def test_whose_is_not_an_interrogative(solver):
    """"Find two numbers **whose** sum is 7" was refused as a question asking for a non-quantity —
    twenty problems in a hundred, silently, by a guard meant for "what colour"."""
    assert solver.solve("find two numbers whose sum is 7 and product is 10. "
                        "what is the larger?").answer == "5"


def test_a_question_that_asks_for_no_quantity_is_refused(solver):
    got = solver.solve("the sum of three consecutive numbers is 78. "
                       "what is the colour of the largest?")
    assert not got.ok


def test_a_solution_that_is_negative_is_named_rather_than_hidden(solver):
    """(12, 8) and (-8, -12) both satisfy it. The convention is taken and the other is shown."""
    got = solver.solve("two numbers differ by 4 and their product is 96. what is the larger?")
    assert got.answer == "12"
    assert any("also satisfies" in step for step in got.steps)


def test_the_solver_shows_its_working(solver):
    got = solver.solve("a shopkeeper marks up the price by 40% and then gives a discount of 25%. "
                       "what is the overall profit percent?")
    assert len(got.steps) >= 3 and got.verified


# --------------------------------------------------------------------------- #
# Refusal, and the verification that makes it possible
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "the sum of two numbers is 20. what is the larger?",
    "a pipe fills a tank in 8 hours and a leak empties it in 4 hours. how long to fill?",
    "what is the sum to infinity of the geometric series 2, 4, 8, ...",
    "a bag has 1 red and 5 blue balls. two are drawn without replacement. "
    "what is the probability that both are red?",
    "how many diagonals does a polygon with 2 sides have?",
    "how are you today?",
    "what is the capital of France?",
])
def test_a_problem_with_no_answer_gets_no_answer(solver, question):
    assert not solver.solve(question).ok


def test_a_recognised_refusal_says_it_was_recognised(solver):
    """The flag the downstream ordering rests on — see the brain's `_mathematical`."""
    got = solver.solve("what is the sum to infinity of the geometric series 2, 4, 8, ...")
    assert not got.ok and got.recognised
    unread = solver.solve("what is the capital of France?")
    assert not unread.ok and not unread.recognised


def test_an_answer_that_fails_its_own_constraints_is_not_stated():
    """The property a pattern cannot have: a misreading here produces silence, not a number."""
    problem = read_number_relations("the sum of the numbers is 10 and their product is 1000")
    assert problem is not None
    assert solve_algebraic(problem) == [] or not any(
        all(c.holds(a) for c in problem.constraints) for a in solve_algebraic(problem))


# --------------------------------------------------------------------------- #
# It is reached from a real turn, and it is reached first
# --------------------------------------------------------------------------- #

def test_the_master_gets_the_hard_answers_back():
    brain = NJPBrain(ExamConditions())
    for question, expected in BUILT_AGAINST[:8] + WRITTEN_AFTER[:8]:
        assert brain.think(question, remember=False).answer == expected, question


def test_the_solver_is_asked_before_the_skill_table():
    """Asked the other way round, the discount skill answers 30 and the solver never runs."""
    brain = NJPBrain(ExamConditions())
    assert brain.think("a shopkeeper marks up the price by 40% and then gives a discount of 25%. "
                       "what is the overall profit percent?", remember=False).answer == "5%"


def test_a_recognised_refusal_blocks_the_skill_table():
    """Without it the turn falls through and the skill table adds the three terms it can see."""
    brain = NJPBrain(ExamConditions())
    assert brain.think("what is the sum to infinity of the geometric series 2, 4, 8, ...",
                       remember=False).answer == ""
    assert brain.think("a bag has 1 red and 5 blue balls. two are drawn without replacement. "
                       "what is the probability that both are red?", remember=False).answer == ""


def test_the_single_step_syllabus_is_untouched():
    """The solver in front must not shadow what already worked."""
    brain = NJPBrain(ExamConditions())
    for question, expected in (("what is the gcd of 48 and 18?", "6"),
                               ("expand (x+2)(x+3)", "x^2 + 5x + 6"),
                               ("what is the area of a triangle with base 10 and height 6?", "30"),
                               ("what is 24 + 18?", "42")):
        assert brain.think(question, remember=False).answer == expected, question


def test_a_hard_problem_is_never_filed_as_a_fact():
    brain = NJPBrain(ExamConditions())
    before = len(brain.grounder.facts)
    for question, _ in BUILT_AGAINST[:10]:
        brain.think(question)
    assert len(brain.grounder.facts) == before


def test_an_organ_that_is_off_is_absent_not_zeroed():
    class _Off:
        mathsolver_enabled = False

    brain = NJPBrain(_Off())
    assert brain.solver is None
    assert brain.think("the sum of three consecutive even numbers is 78. "
                       "what is the largest of them?", remember=False).answer == ""


# --------------------------------------------------------------------------- #
# The second tier — a third bank measured 2 right of 25 against the first
# --------------------------------------------------------------------------- #

#: Written after the first tier was finished and passing both earlier banks. First measurement:
#: **2 right, 12 wrong, 11 silent** — and three of the twelve came back as `noted:`, filed into
#: the knowledge store as facts.
HARDER_STILL = [
    ("find the smallest positive integer n such that n^2 + n + 41 is not prime", "40"),
    ("what is the sum of the digits of 2^10?", "7"),
    ("how many integers between 1 and 100 are divisible by 3 or 5?", "47"),
    ("find the last two digits of 7^2019", "43"),
    ("what is the gcd of 2^30 - 1 and 2^20 - 1?", "1023"),
    ("in how many ways can 8 identical balls be put into 3 distinct boxes?", "45"),
    ("what is the units digit of 1! + 2! + 3! + ... + 100!?", "3"),
    ("if the roots of x^2 - 5x + 6 = 0 are a and b, find a^2 + b^2", "13"),
    ("how many zeros does 1000 factorial end with?", "249"),
    ("what is the remainder when 1! + 2! + 3! + ... + 50! is divided by 15?", "3"),
    ("what is the sum of the first 20 terms of the ap 3, 7, 11, ...?", "820"),
    ("a number when divided by 7 leaves 3 and when divided by 11 leaves 5. "
     "find the smallest such number", "38"),
    ("how many prime numbers are there below 50?", "15"),
    ("what is the value of (1 - 1/2)(1 - 1/3)(1 - 1/4)(1 - 1/5)?", "1/5"),
    ("if 2^x = 32 find x", "5"),
    ("the angles of a triangle are in the ratio 2:3:4. find the largest angle", "80"),
    ("what is the area of a circle inscribed in a square of side 10?", "25π ≈ 78.5398"),
    ("how many terms are there in the ap 5, 9, 13, ..., 101?", "25"),
    ("what is the sum of all two digit numbers divisible by 7?", "728"),
    ("find the value of x if 3^(x+1) = 81", "3"),
    ("how many numbers between 1 and 200 are divisible by neither 2 nor 3?", "67"),
    ("what is the 100th term of the ap 7, 12, 17, ...?", "502"),
    ("if a:b = 2:3 and b:c = 4:5, find a:c", "8 : 15"),
    ("what is the sum of the squares of the first 10 natural numbers?", "385"),
    ("find the number of ways to arrange the letters of the word mississippi", "34650"),
]


@pytest.mark.parametrize("question,expected", HARDER_STILL)
def test_the_third_bank(solver, question, expected):
    got = solver.solve(question)
    assert got.ok, f"{question!r}: {got.error}"
    assert got.answer == expected


def test_a_predicate_search_takes_any_polynomial_and_any_property(solver):
    """The most general reading here, and nothing about the pair is enumerated in advance."""
    for question, expected in (
            ("find the smallest positive integer n such that n^2 + n + 41 is not prime", "40"),
            ("find the smallest positive integer n such that n^2 + 1 is divisible by 5", "2"),
            ("find the smallest positive integer n such that 2n + 1 is prime", "1"),
            ("find the smallest positive integer n such that n^2 is greater than 500", "23")):
        assert solver.solve(question).answer == expected, question


def test_a_property_it_cannot_read_is_not_approximated(solver):
    """A search whose predicate is *approximately* the question returns an exactly wrong number."""
    assert not solver.solve("find the smallest positive integer n such that n is a happy "
                            "number").ok


def test_the_unknown_in_an_exponent_may_be_negative(solver):
    """3^(x+3) = 9 has x = -1; searching from zero reported that there was none."""
    assert solver.solve("find the value of x if 3^(x+3) = 9").answer == "-1"


def test_a_gcd_is_taken_of_the_numbers_not_of_the_digits_in_the_sentence(solver):
    """The dispatcher answered 1048576: it found 30, 1, 20 and 1, and never saw a subtraction."""
    assert solver.solve("what is the gcd of 2^30 - 1 and 2^20 - 1?").answer == "1023"


def test_juxtaposed_brackets_are_a_product(solver):
    """`(a)(b)` is a *call* to `ast.parse`, so the calculator could not evaluate it at all."""
    assert solver.solve("what is the value of (1 - 1/2)(1 - 1/3)(1 - 1/4)(1 - 1/5)?").answer \
        == "1/5"


def test_the_roots_are_used_without_being_found(solver):
    """Vieta rather than the quadratic formula: the roots are often irrational, the answer is not."""
    got = solver.solve("if the roots of x^2 - 7x + 11 = 0 are a and b, find a^2 + b^2")
    assert got.answer == "27"


@pytest.mark.parametrize("n,expected", [(2, True), (41, True), (91, False), (1, False)])
def test_the_primality_predicate(n, expected):
    assert is_prime(n) is expected


# --------------------------------------------------------------------------- #
# A problem she cannot solve is still not a fact about the world
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("find the smallest positive integer n such that n^7 + 3 is a happy number", True),
    ("how many wibbles are there in 12 wobbles?", True),
    ("solve x @@ 3 = 9", True),
    ("scientists find water on mars", False),
    ("the sun is 150 million km away", False),
    ("a factor is a divisor of a number", False),
])
def test_an_unsolved_task_is_told_apart_from_a_statement(text, expected):
    """Both halves are required, and that is what keeps the guard narrow."""
    assert looks_like_a_task(text) is expected


def test_a_problem_she_cannot_solve_is_not_written_down():
    """Measured: three of the third bank's misses were filed as triples at confidence 0.75 —
    `('find', 'the') → 'smallest positive integer n'` and two more like it."""
    brain = NJPBrain(ExamConditions())
    before = len(brain.grounder.facts)
    for question in ("find the smallest positive integer n such that n^7 + 3 is a happy number",
                     "solve the riemann hypothesis for 42",
                     "find x if x @@ 3 = 9"):
        assert brain.think(question).answer == ""
    assert len(brain.grounder.facts) == before


def test_a_statement_is_still_learned():
    """The guard must not cost her the ability to be told something."""
    brain = NJPBrain(ExamConditions())
    before = len(brain.grounder.facts)
    brain.think("a factor is a divisor of a number")
    assert len(brain.grounder.facts) > before


# --------------------------------------------------------------------------- #
# The third tier — a fourth bank measured 2 right of 20 against the second
# --------------------------------------------------------------------------- #

#: Written after the second tier was finished and passing three banks. First measurement:
#: **2 right, 5 wrong, 13 silent**.
FOURTH_BANK = [
    ("a boat goes 20 km downstream in 2 hours and returns in 4 hours. "
     "find the speed of the stream", "5/2"),
    ("the mean of 5 numbers is 12. if each number is increased by 3, "
     "what is the new mean?", "15"),
    ("a sum doubles in 8 years at simple interest. in how many years will it triple?", "16"),
    ("what is the probability of getting at least one head in two tosses of a coin?", "3/4"),
    ("if 20% of a number is 45, what is 30% of the same number?", "135/2"),
    ("the sum of the first n natural numbers is 210. find n", "20"),
    ("how many three digit numbers are divisible by 13?", "69"),
    ("the lcm of two numbers is 180 and their hcf is 6. if one number is 30, "
     "find the other", "36"),
    ("a shopkeeper sells an article at a loss of 10%. if he had sold it for 50 more "
     "he would have gained 15%. find the cost price", "200"),
    ("find the value of x if (x-1)/2 + (x+1)/3 = 4", "x = 5"),
    ("the area of a square is 144. what is its perimeter?", "48"),
    ("two dice are thrown. what is the probability that the sum is 7?", "1/6"),
    ("how many odd numbers are there between 20 and 60?", "20"),
    ("if f(x) = 2x + 3, find f(5)", "13"),
    ("the difference between simple and compound interest on a sum for 2 years "
     "at 10 percent is 20. find the sum", "2000"),
    ("a man is 3 times as old as his son. 10 years ago he was 5 times as old. "
     "find the son's present age", "20"),
    ("a rectangle has area 60 and perimeter 32. find its length", "10"),
    ("how many squares are there on a chessboard?", "204"),
    ("what is the 10th term of the gp 2, 6, 18, ...?", "39366"),
]


@pytest.mark.parametrize("question,expected", FOURTH_BANK)
def test_the_fourth_bank(solver, question, expected):
    got = solver.solve(question)
    assert got.ok, f"{question!r}: {got.error}"
    assert got.answer == expected


def test_an_evaluation_frame_is_not_an_equation(solver):
    """"The value of 2x² + 9 **when** x = 5" contains an `=` and is not a thing to solve.

    Read as one it says x = 5 and answers 5 — the number in the question rather than the answer
    to it — and because a recognised refusal blocks, it took a passing paper down with it.
    """
    assert not solver.solve("what is the value of 2x^2 + 9 when x = 5").ok
    assert NJPBrain(ExamConditions()).think(
        "what is the value of 2x^2 + 9 when x = 5", remember=False).answer == "59"


def test_every_root_is_reported_not_one_of_them(solver):
    """Naming one root of a quadratic is choosing rather than solving."""
    assert solver.solve("solve x^2 - 17x + 72 = 0").answer == "x = 8, x = 9"


def test_a_quantity_is_read_by_where_it_is_written(solver):
    """Excluding the rate and the term *by value* deleted the gap when it equalled one of them."""
    assert solver.solve("the difference between simple and compound interest on a sum for "
                        "2 years at 10 percent is 10. find the sum").answer == "1000"


def test_a_question_asked_in_decimals_is_answered_in_decimals():
    """`0.1 + 0.02 + 0.003` is exactly 123/1000, and 123/1000 is not what was asked for."""
    from nyxara.njp.calculate import evaluate

    assert evaluate("0.1 + 0.02 + 0.003").text == "0.123"
    assert evaluate("1/3").text == "1/3"
    assert evaluate("10 divided by 4").text == "5/2"


def test_at_least_one_is_a_complement_never_a_sum(solver):
    assert solver.solve("what is the probability of getting at least one head in "
                        "3 tosses of a coin?").answer == "7/8"
