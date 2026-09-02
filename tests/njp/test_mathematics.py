"""Mathematics — the syllabus nothing in njp/ could do, and the store it was corrupting.

Measured before this module existed, on twenty-five ordinary school questions through
`NJPBrain.think` with every organ built: **3 right, 17 silent, and 5 filed as facts about the
world** — reproducible by building the brain with `mathematics_enabled = False`. `nyxara.njp.calculate` says of itself that it "does not do algebra, does not solve for
unknowns", and it was telling the truth — a closed expression was the whole of her mathematics.

The five that were not silent are why this is a defect rather than a gap. "simplify the fraction
18/24", "expand (x+2)(x+3)", "factorise x^2 + 5x + 6", "convert 5 km to metres" and "solve
x^2 - 5x + 6 = 0" are imperatives; nothing read them as tasks, so the semantic compiler read them
as assertions and filed all five at confidence 0.75 into the store inheritance and the puzzle
solver walk.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.mathematics import (
    MathError,
    Mathematician,
    Poly,
    normalise,
    numbers_in,
    solve,
)


@pytest.fixture(scope="module")
def maths() -> Mathematician:
    return Mathematician()


# --------------------------------------------------------------------------- #
# Poly — algebra without an evaluator
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("(x+2)(x+3)", "x^2 + 5x + 6"),
    ("x(x+1)", "x^2 + x"),
    ("2x^2 - 3x + 1", "2x^2 - 3x + 1"),
    ("(x+1)(x+2)(x+3)", "x^3 + 6x^2 + 11x + 6"),
    ("-3x + 7", "-3x + 7"),
    ("5", "5"),
    ("2y^2 - 3y + 1", "2y^2 - 3y + 1"),
])
def test_the_parser_reads_the_algebra_a_textbook_writes(text, expected):
    assert Poly.parse(text).text() == expected


def test_a_fractional_coefficient_is_written_where_it_cannot_be_misread():
    """`1/3x^3` reads as 1/(3x³) as easily as (1/3)x³. The denominator goes after the term."""
    assert Poly.parse("x^2").integral().text() == "x^3/3"
    assert Poly.parse("2x").integral().text() == "x^2"


def test_the_symbol_is_discovered_not_configured():
    """A constant has no unknown, so `2y^2` is about y however the 2 was spelled.

    Fixing the symbol from the first token — which the first version did — made "2y^2 - 3y + 1"
    refuse itself as "two different unknowns, x and y".
    """
    assert Poly.parse("2y^2 - 3y + 1").symbol == "y"
    assert Poly.parse("2y^2-3y+1").rational_roots() == [0.5, 1]


def test_two_genuinely_different_unknowns_are_refused():
    with pytest.raises(MathError):
        Poly.parse("x + y")


@pytest.mark.parametrize("attack", [
    "__import__('os')", "open('/etc/passwd')", "x.__class__", "[1,2][0]", "lambda: 1",
])
def test_nothing_that_is_not_arithmetic_can_be_parsed(attack):
    """The whitelist is the *representation*: a `Dict[int, Fraction]` cannot name a call."""
    with pytest.raises(MathError):
        Poly.parse(attack)


def test_exact_division_only():
    """A remainder is a refusal — a truncated quotient is a wrong answer in the right shape."""
    assert (Poly.parse("x^2 + 5x + 6") / Poly.parse("x + 2")).text() == "x + 3"
    with pytest.raises(MathError):
        Poly.parse("x^2 + 1") / Poly.parse("x + 2")


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #

def test_hinglish_and_english_reach_the_same_skill(maths):
    assert maths.solve("48 aur 18 ka hcf kitna hai?").answer == "6"
    assert maths.solve("what is the hcf of 48 and 18?").answer == "6"


def test_a_number_word_is_only_read_where_there_are_no_digits():
    """Every number word is also an ordinary word in one of the two languages.

    Measured: "91 ek prime number hai?" became "91 1 prime number hai" and was answered about
    **1**; "which one is bigger, 3/4 or 5/8?" grew a third number to compare.
    """
    assert "1 prime" not in normalise("91 ek prime number hai?")
    assert numbers_in(normalise("which one is bigger, 3/4 or 5/8?")) == \
        numbers_in("3/4 5/8")
    assert normalise("what is twenty plus thirty") == "what is 20 plus 30"


def test_the_factorial_operator_survives_normalisation():
    """`?` is never mathematics; `!` is, with a number in front of it."""
    assert solve("what is 5!?").answer == "120"


# --------------------------------------------------------------------------- #
# The syllabus
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,expected", [
    # number theory
    ("is 91 a prime number?", "no, 91 is not a prime number"),
    ("is 97 prime?", "yes, 97 is a prime number"),
    ("what are the prime factors of 84?", "2^2 × 3 × 7"),
    ("how many factors does 36 have?", "9"),
    ("what is the gcd of 48 and 18?", "6"),
    ("what is the lcm of 4 and 6?", "12"),
    ("is 91 divisible by 7?", "yes, 91 is divisible by 7"),
    ("is 64 a perfect square?", "yes, 64 is a perfect square"),
    ("what is 7 factorial?", "5040"),
    # powers, roots, logarithms
    ("what is the square root of 144?", "12"),
    ("what is the cube root of 27?", "3"),
    ("what is 2 to the power 10?", "1024"),
    ("what is log base 2 of 8?", "3"),
    # fractions
    ("simplify the fraction 18/24", "3/4"),
    ("write 3/4 as a decimal", "0.75"),
    ("write 3/8 as a percent", "37.5%"),
    # percentage and commerce
    ("15 is what percent of 60?", "25%"),
    ("12 is 25% of what number?", "48"),
    ("percent increase from 40 to 50", "25% increase"),
    ("an article was bought for 400 and sold for 500, what is the profit percent?",
     "25% profit"),
    ("an item marked 800 has a 25% discount, what is the price?", "600"),
    ("what is the simple interest on 5000 at 8% for 3 years?", "1200"),
    # ratio
    ("simplify the ratio 12:18", "2 : 3"),
    ("divide 600 in the ratio 2:3", "240 , 360"),
    ("if 3 : 4 = 9 : x, find x", "12"),
    # algebra
    ("solve for x: 2x + 5 = 17", "x = 6"),
    ("solve x^2 - 5x + 6 = 0", "x = 2, x = 3"),
    ("solve x + y = 10 and x - y = 2", "x = 6, y = 4"),
    ("expand (x+2)(x+3)", "x^2 + 5x + 6"),
    ("factorise x^2 + 5x + 6", "(x + 3)(x + 2)"),
    ("simplify 2x + 3x - 4 + 1", "5x - 3"),
    ("what is the value of x^2 + 1 when x = 3", "10"),
    # sequences
    ("what is the next term in 2, 4, 8, 16?", "32"),
    ("what is the 10th term of 3, 7, 11?", "39"),
    ("what is the sum of the first 100 natural numbers?", "5050"),
    ("what is the sum of the first 10 odd numbers?", "100"),
    # geometry and mensuration
    ("what is the area of a triangle with base 10 and height 6?", "30"),
    ("what is the perimeter of a rectangle of length 8 and width 3?", "22"),
    ("what is the hypotenuse of a right angled triangle with sides 3 and 4?", "5"),
    ("two angles of a triangle are 50 and 60, what is the third angle?", "70°"),
    ("what is the sum of the interior angles of a hexagon?", "720°"),
    ("what is the volume of a cube of side 4?", "64"),
    ("what is the area of a triangle with sides 3, 4 and 5?", "6"),
    # units
    ("convert 5 km to metres", "5000 metres"),
    ("how many minutes are in 3 hours?", "180 minutes"),
    # statistics and probability
    ("what is the mean of 4, 8, 15, 16, 23, 42?", "18"),
    ("what is the median of 3, 1, 4, 1, 5?", "3"),
    ("what is the range of 4, 9, 1, 12?", "11"),
    ("what is the probability of getting a 3 on a die?", "1/6"),
    ("a bag has 3 red and 5 blue balls, what is the probability of red?", "3/8"),
    # calculus
    ("what is the derivative of x^3?", "3x^2"),
    ("the integral of x^2", "x^3/3 + C"),
    ("integrate 2x from 0 to 3", "9"),
    ("the limit of x^2 + 1 as x -> 2", "5"),
    # word problems
    ("a train travels 180 km in 3 hours, what is its speed?", "60"),
    ("a car goes at 60 km/h for 2 hours, what distance does it cover?", "120"),
    ("A does a job in 6 days and B in 12 days, how long working together?", "4"),
])
def test_the_whole_syllabus(maths, question, expected):
    got = maths.solve(question)
    assert got.ok, f"{question!r} produced nothing: {got.error}"
    assert got.answer == expected


def test_an_answer_is_reported_with_pi_rather_than_rounded_silently():
    """Rounding π and calling the result an area overstates what she knows."""
    got = solve("what is the area of a circle with radius 7?")
    assert got.answer.startswith("49π")
    assert not got.exact


def test_every_answer_shows_its_working(maths):
    for question in ("what is the gcd of 48 and 18?", "solve x^2 - 5x + 6 = 0",
                     "what is the simple interest on 5000 at 8% for 3 years?"):
        assert maths.solve(question).steps, question


# --------------------------------------------------------------------------- #
# Refusal is a first-class answer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "how are you?",
    "what is the capital of France?",
    "tell me a story about a triangle",
    "expand your answer please",
])
def test_a_sentence_with_no_mathematics_in_it_is_not_recognised(maths, question):
    got = maths.solve(question)
    assert not got.ok
    assert got.topic == "", f"{question!r} was claimed by {got.method}"


@pytest.mark.parametrize("question,fragment", [
    ("convert 5 km to kilograms", "no conversion"),
    ("what is the mode of 1, 2, 3, 4, 5?", "no mode"),
    ("what is the next term in 2, 7, 1?", "cannot see the rule"),
    ("what is the hcf of red and blue?", "at least two numbers"),
])
def test_a_recognised_question_with_no_answer_is_refused_with_its_reason(maths, question,
                                                                        fragment):
    """The harder half of restraint: each of these reaches a skill and is declined *by* it."""
    got = maths.solve(question)
    assert not got.ok
    assert got.topic, f"{question!r} was not recognised at all"
    assert fragment in got.error


def test_english_is_never_parsed_as_algebra(maths):
    """A word is not an unknown.

    Measured: "a factor is a number that divides another number" reached the factoriser and came
    back "two different unknowns, is and a". The sentence was then a *recognised* task, so the
    brain refused to learn the definition it was being taught — a lesson silently discarded by
    the reader meant to be helping.
    """
    for sentence in ("a factor is a divisor of a number",
                     "a derivative is a rate of change",
                     "an equation is a statement that two expressions are equal"):
        got = maths.solve(sentence)
        assert not got.ok and got.topic == "", f"{sentence!r} was claimed by {got.method}"


def test_every_named_skill_exists():
    """A missing skill would be swallowed by the dispatcher's own robustness."""
    maths = Mathematician()
    assert [name for _, name in Mathematician.SKILLS if not hasattr(maths, name)] == []


# --------------------------------------------------------------------------- #
# It is reached from a real turn — and the store is left alone
# --------------------------------------------------------------------------- #

_FLOOR = [
    ("what is 24 + 18?", "42"),
    ("solve for x: 2x + 5 = 17", "x = 6"),
    ("what is the gcd of 48 and 18?", "6"),
    ("is 91 a prime number?", "no, 91 is not a prime number"),
    ("what are the prime factors of 84?", "2^2 × 3 × 7"),
    ("simplify the fraction 18/24", "3/4"),
    ("what is 1/2 + 1/3?", "5/6"),
    ("what is 20% of 250?", "50"),
    ("15 is what percent of 60?", "25%"),
    ("what is the area of a triangle with base 10 and height 6?", "30"),
    ("what is the mean of 4, 8, 15, 16, 23, 42?", "18"),
    ("what is the square root of 144?", "12"),
    ("what is the next term in 2, 4, 8, 16?", "32"),
    ("expand (x+2)(x+3)", "x^2 + 5x + 6"),
    ("factorise x^2 + 5x + 6", "(x + 3)(x + 2)"),
    ("what is the derivative of x^3?", "3x^2"),
    ("convert 5 km to metres", "5000 metres"),
    ("solve x^2 - 5x + 6 = 0", "x = 2, x = 3"),
    ("what is the perimeter of a rectangle of length 8 and width 3?", "22"),
]


@pytest.mark.parametrize("question,expected", _FLOOR)
def test_the_master_gets_the_answer_back(question, expected):
    assert NJPBrain().think(question).answer == expected


def test_a_maths_instruction_is_never_filed_as_a_fact_about_the_world():
    """The defect this module exists for, pinned from both sides.

    Before: five maths instructions among twenty-five questions became five triples at
    confidence 0.75 — `('simplify fraction', '18') → '24'`, `('expand', 'x') → '2 x 3'`,
    `('factorise', 'x') → '2 5x 6'`, `('convert', '5') → 'km metres'` and
    `('solve', 'x') → '2 5x 6 0'` — in the store inheritance and the puzzle solver walk.
    """
    brain = NJPBrain()
    before = len(brain.grounder.facts)
    for instruction in ("simplify the fraction 18/24", "expand (x+2)(x+3)",
                        "convert 5 km to metres", "factorise x^2 + 5x + 6",
                        "differentiate 3x^2 + 2x - 5"):
        brain.think(instruction)
    assert len(brain.grounder.facts) == before


def test_a_recognised_task_that_is_refused_is_also_not_filed():
    """The original defect surviving inside its own fix, on the one input where the fix declines.

    "convert 5 km to kilograms" is recognised by the conversion skill and refused because length
    and mass are different quantities. Gating the store on *solved* rather than on *claimed* let
    it fall through and be filed as `('convert', '5') → 'km kilograms'`.
    """
    brain = NJPBrain()
    before = len(brain.grounder.facts)
    assert brain.think("convert 5 km to kilograms").answer == ""
    assert len(brain.grounder.facts) == before


def test_a_worked_answer_is_not_mistaken_for_an_echo():
    """A maths answer is made of the question's own symbols, which is what correct looks like.

    `is_meta_commentary` scores word overlap and blanked both of these — "is 91 a prime number?"
    → "no, 91 is not a prime number" at 0.83, and "factorise x^2 + 5x + 6" → "(x + 3)(x + 2)" at
    0.67. Each was right, each was deleted, and the turn went out silent.
    """
    brain = NJPBrain()
    assert brain.think("is 91 a prime number?").answer.startswith("no, 91")
    assert brain.think("factorise x^2 + 5x + 6").answer == "(x + 3)(x + 2)"


def test_bare_arithmetic_keeps_the_route_it_already_had():
    """The calculator's own wiring is not replaced, and a test would not otherwise see that.

    Answering `2+2` from the mathematician would leave the strategy registration, the
    parsed-expression classifier flag and `_closed_arithmetic` in the source and unreachable.
    """
    brain = NJPBrain()
    thought = brain.think("5 ka square kitna hai?")
    assert thought.answer == "25"
    assert thought.mathematics is None
    assert thought.solution is not None and thought.solution.kind == "symbolic"


def test_the_working_travels_with_the_turn():
    thought = NJPBrain().think("what is the simple interest on 5000 at 8% for 3 years?")
    assert thought.mathematics is not None
    assert thought.mathematics.topic == "commerce"
    assert thought.mathematics.steps


def test_a_question_with_no_answer_gets_no_substitute():
    """`_closed_arithmetic`'s rule at its limit: a refusal is not an invitation to guess.

    Measured in `mathschool`: after a lesson stating "a mode is the value that appears most often
    in a list", the control "what is the mode of 1, 2, 3, 4, 5?" — a list with no mode — was
    answered with that definition, because deliberation and recall run whenever the answer is
    empty.
    """
    brain = NJPBrain()
    brain.think("a mode is the value that appears most often in a list")
    assert brain.think("what is the mode of 1, 2, 3, 4, 5?").answer == ""


def test_the_public_entry_point_returns_the_working():
    solution = NJPBrain().do_maths("solve x^2 - 5x + 6 = 0")
    assert solution.ok and solution.topic == "algebra"
    assert "discriminant" in " ".join(solution.steps)


def test_the_floor_is_reproducible_with_the_organ_switched_off():
    """The "before" number is a measurement anybody can repeat, not a story about the past.

    With `mathematics_enabled = False` the brain is exactly what it was: the same twenty-five
    questions come back 3 right, 17 silent and 5 filed as facts about the world. Asserted as the
    weaker, stable claim — *some* instruction is filed with the organ off and *none* is with it on
    — so the test pins the fix rather than freezing the grounder's exact reading of a sentence
    nobody should be grounding.
    """
    class _Off:
        mathematics_enabled = False

    instructions = ("simplify the fraction 18/24", "expand (x+2)(x+3)",
                    "factorise x^2 + 5x + 6", "convert 5 km to metres",
                    "solve x^2 - 5x + 6 = 0")

    before = NJPBrain(_Off())
    assert before.mathematician is None
    for instruction in instructions:
        before.think(instruction)
    assert len(before.grounder.facts) > 0, "the floor no longer reproduces — check the grounder"

    after = NJPBrain()
    for instruction in instructions:
        after.think(instruction)
    assert len(after.grounder.facts) == 0


def test_an_organ_that_is_off_is_absent_not_zeroed():
    class _Off:
        mathematics_enabled = False

    brain = NJPBrain(_Off())
    assert brain.mathematician is None
    assert brain.think("what is the gcd of 48 and 18?").answer == ""
