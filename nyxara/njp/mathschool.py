"""NYXARA · njp/mathschool.py — the mathematics syllabus, taught and then examined (📐, NJP V.23).

:mod:`nyxara.njp.mathematics` is the faculty. This is the school around it, and it exists because
a faculty nobody examines is a claim rather than a capability — the same argument
:mod:`nyxara.njp.school` makes for reasoning, language and coding, applied to the subject that had
none.

**It borrows that school's machinery rather than copying it.** :class:`~nyxara.njp.school.Question`,
:class:`~nyxara.njp.school.Score`, :class:`~nyxara.njp.school.Result`,
:class:`~nyxara.njp.school.Transcript` and the ``pre-test → teach → post-test`` loop in
:class:`~nyxara.njp.school.School` are imported, not reimplemented. That is not tidiness: the
three-outcome grading rule — right, wrong, **abstained**, never merged — is the thing that makes
every number in this package comparable, and a second copy of it is a second place for it to drift.

**Two kinds of subject, and the difference is the honest part of the report.**

*Sixteen doing subjects* — number theory, fractions, percentage, commerce, ratio, algebra,
sequences, geometry, mensuration, units, statistics, probability, calculus, word problems, powers,
arithmetic — are **decision procedures**, and a decision procedure cannot be taught. Each of them
reads ``1.00`` cold and is printed with ``already`` beside it, exactly as
:class:`~nyxara.njp.school.Arithmetic` is. What they are worth is stated plainly in that class's
own docstring and is worth repeating: a subject that reads 1.00 cold is evidence **about the
organ**, and an organ that quietly stopped working shows up here on the first run rather than
three subjects later as an unexplained dip. Their floor is not zero because they are easy; it is
zero-or-one because the answer is computed, and every one of them scored 0.00 before the faculty
existed.

*One knowing subject* — ``vocabulary`` — is the only thing here a lesson can move, and it moves
because **stating mathematics and doing it are different capabilities**. She can work out that 91
is not prime and, before the lesson, cannot say what a prime number *is*; the two live in different
organs and neither implies the other. Twenty definitions, of which a seeded **half are taught and
half are deliberately not**, and the untaught half are scored as controls where silence is the
right answer. A brain that learned the ten it was given and stayed quiet about the ten it was not
is the only shape of result that means anything; one that answers all twenty is guessing and the
report says so.

*One restraint subject* — every item a control. A question with no mathematics in it, and a
mathematics-shaped question with no answer ("divide 7 by 0", "the gcd of red and blue"). Silence
scores as right and an assertion scores as wrong however confident it sounds. This subject exists
because the failure :mod:`nyxara.njp.mathematics` was written for was **not** silence: it was five
sentences of arithmetic filed into the knowledge store as facts about the world. An organ that
answers everything would have passed the other seventeen subjects and be worse than the one it
replaced.

**What one run reports**, seed 11, reproducible with ``python -m nyxara.njp.mathschool`` and
``--exam``::

                        school          examination
    mastered            18 / 18         17 / 17
    right/wrong/absent  158 / 0 / 0     410 / 0 / 0
    accuracy            1.0000          1.0000
    facts written       10              0
    vocabulary          0.50 → 1.00     (not sat — see `MathExam`)

The ten facts are the lesson and nothing else: every doing subject asks with ``remember=False``
and writes nothing at all.
"""

from __future__ import annotations

import random
import time
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple

from nyxara.njp.school import (
    ExamConditions,
    Mint,
    Question,
    Result,
    School,
    Score,
    Subject,
    Taught,
    Transcript,
)

__all__ = [
    "MATHS_SUBJECTS",
    "MathExam",
    "MathSchool",
    "MathSubject",
    "main",
]


class MathSubject(Subject):
    """One area of the syllabus, with a way to generate fresh items and no way to teach itself.

    The default :meth:`teach` says what it is: an arithmetic procedure is not a belief, and a
    lesson that "taught" one would be a lesson that changed nothing and reported that it had.
    Subclasses that genuinely have something to state override it — exactly one of them does.
    """

    threshold, items = 0.9, 8

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        return Taught(0, "nothing to teach — this is a decision procedure, not a belief")

    def generate(self, rng: random.Random) -> List[Question]:
        raise NotImplementedError

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        return self.sit(brain, self.generate(mint.rng))

    @staticmethod
    def ask(brain: Any, question: Question) -> Tuple[str, str]:
        """Ask, **without remembering the turn.**

        The one line that differs from :meth:`nyxara.njp.school.Subject.ask`, and it was put here
        by a measurement rather than by caution. ``restraint`` scored 8/10 in a full exam and
        10/10 when the same brain was asked those ten questions alone: *"what is 7 divided by 0?"*
        came back **0.7**. Nothing in the mathematics faculty produced that — it declines the
        question outright — and nothing had to. Sixteen papers of arithmetic had gone into
        episodic memory ahead of it, and recall found the nearest thing in the store.

        An examination whose earlier items answer its later ones is not measuring the brain, it
        is measuring the exam's own ordering. ``remember=False`` is what makes the module's claim
        to write nothing true rather than intended.
        """
        try:
            reply = brain.think(question.ask, remember=False).answer
        except TypeError:      # a brain-like object without the keyword — ask it the plain way
            reply = brain.think(question.ask).answer
        except Exception as exc:  # noqa: BLE001 — a crash is a wrong answer, not a stopped exam
            return "wrong", f"{question.ask} → raised {type(exc).__name__}: {exc}"
        verdict = question.grade(reply)
        detail = f"{question.ask} → {str(reply).strip()[:60]!r}" if verdict != "right" else ""
        return verdict, detail

    # -- helpers every generator uses --------------------------------------- #
    @staticmethod
    def number(value: Any) -> str:
        """A generated gold answer written the way the faculty writes one."""
        from nyxara.njp.mathematics import _num
        return _num(value)


# --------------------------------------------------------------------------- #
# the doing subjects
# --------------------------------------------------------------------------- #

class NumberTheory(MathSubject):
    id, title = "number", "primes, factors, hcf and lcm"
    teaches = "a whole number has a structure, and it can be found rather than recalled"
    items = 10

    def generate(self, rng: random.Random) -> List[Question]:
        import math
        questions = []
        for _ in range(self.items):
            a, b = rng.randint(6, 96), rng.randint(4, 60)
            form = rng.choice(("prime", "hcf", "lcm", "divisible", "factors"))
            if form == "prime":
                n = rng.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 91, 87, 51, 49, 77, 121])
                prime = all(n % d for d in range(2, int(n ** 0.5) + 1)) and n > 1
                questions.append(Question(
                    ask=f"is {n} a prime number?",
                    accept=("yes, " if prime else "no, ",), note=f"{n}"))
            elif form == "hcf":
                questions.append(Question(ask=f"what is the hcf of {a} and {b}?",
                                          accept=(str(math.gcd(a, b)),), exact=True))
            elif form == "lcm":
                questions.append(Question(ask=f"what is the lcm of {a} and {b}?",
                                          accept=(str(a * b // math.gcd(a, b)),), exact=True))
            elif form == "divisible":
                questions.append(Question(
                    ask=f"is {a} divisible by {b}?",
                    accept=("yes, " if a % b == 0 else "no, ",)))
            else:
                n = rng.randint(12, 60)
                count = sum(1 for d in range(1, n + 1) if n % d == 0)
                questions.append(Question(ask=f"how many factors does {n} have?",
                                          accept=(str(count),), exact=True))
        return questions


class Fractions(MathSubject):
    id, title = "fractions", "fractions and decimals"
    teaches = "a fraction has one value and many spellings"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            top, bottom = rng.randint(2, 40), rng.randint(2, 40)
            value = Fraction(top, bottom)
            if rng.random() < 0.5:
                questions.append(Question(ask=f"simplify the fraction {top}/{bottom}",
                                          accept=(self.number(value),), exact=True))
            else:
                bottom = rng.choice((2, 4, 5, 8, 10, 20, 25))
                top = rng.randint(1, bottom - 1)
                questions.append(Question(
                    ask=f"write {top}/{bottom} as a decimal",
                    accept=(self.number(float(Fraction(top, bottom))),), exact=True))
        return questions


class Percentage(MathSubject):
    id, title = "percentage", "percentage in all three directions"
    teaches = "the part, the whole and the rate — any two give the third"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            # The pair is redrawn until the part is a whole number, and that is a fix to the
            # *exam*. With `//` and an unconstrained pair, "12% of 60" was graded against 7 while
            # the right answer is 7.2, and she was marked wrong four times out of twenty for
            # being right. Another gold that was wrong before she was — the defect the general
            # exam kept finding in the knowledge corpus, met for the first time outside it.
            while True:
                rate = rng.choice((5, 10, 12, 20, 25, 40, 50, 75))
                whole = rng.choice((40, 60, 80, 120, 200, 250, 400, 500))
                if whole * rate % 100 == 0:
                    break
            part = whole * rate // 100
            form = rng.choice(("of", "which", "whole"))
            if form == "of":
                questions.append(Question(ask=f"what is {rate}% of {whole}?",
                                          accept=(self.number(Fraction(part)),), exact=True))
            elif form == "which":
                questions.append(Question(ask=f"{part} is what percent of {whole}?",
                                          accept=(f"{self.number(Fraction(rate))}%",)))
            else:
                questions.append(Question(ask=f"{part} is {rate}% of what number?",
                                          accept=(self.number(Fraction(whole)),), exact=True))
        return questions


class Commerce(MathSubject):
    id, title = "commerce", "interest, profit and discount"
    teaches = "money questions are percentage questions wearing a sentence"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            principal = rng.choice((2000, 4000, 5000, 8000, 10000, 12000))
            rate = rng.choice((5, 8, 10, 12, 15))
            years = rng.randint(2, 5)
            form = rng.choice(("si", "profit", "discount"))
            if form == "si":
                interest = Fraction(principal * rate * years, 100)
                questions.append(Question(
                    ask=f"what is the simple interest on {principal} at {rate}% for {years} years?",
                    accept=(self.number(interest),), exact=True))
            elif form == "profit":
                cost = rng.choice((200, 400, 500, 800, 1000))
                gain = rng.choice((10, 20, 25, 50))
                sell = cost + cost * gain // 100
                questions.append(Question(
                    ask=f"an article was bought for {cost} and sold for {sell}, "
                        f"what is the profit percent?",
                    accept=(f"{self.number(Fraction(gain))}% profit",)))
            else:
                marked = rng.choice((400, 600, 800, 1200, 2000))
                cut = rng.choice((10, 20, 25, 50))
                questions.append(Question(
                    ask=f"an item marked {marked} has a {cut}% discount, what is the price?",
                    accept=(self.number(Fraction(marked - marked * cut // 100)),), exact=True))
        return questions


class RatioProportion(MathSubject):
    id, title = "ratio", "ratio and proportion"
    teaches = "a ratio is a division that has not been carried out yet"

    def generate(self, rng: random.Random) -> List[Question]:
        import math
        questions = []
        for _ in range(self.items):
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            if rng.random() < 0.5:
                scale = rng.randint(2, 9)
                divisor = math.gcd(a, b)
                questions.append(Question(
                    ask=f"simplify the ratio {a * scale}:{b * scale}",
                    accept=(f"{a // divisor} : {b // divisor}",)))
            else:
                total = (a + b) * rng.randint(4, 30)
                questions.append(Question(
                    ask=f"divide {total} in the ratio {a}:{b}",
                    accept=(self.number(Fraction(total * a, a + b)),)))
        return questions


class Algebra(MathSubject):
    id, title = "algebra", "solving, expanding and factorising"
    teaches = "an unknown is a number whose name you do not know yet"
    items = 10

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            form = rng.choice(("linear", "quadratic", "expand", "evaluate", "simultaneous"))
            if form == "linear":
                a, root = rng.randint(2, 9), rng.randint(-8, 12)
                c = rng.randint(-20, 20)
                questions.append(Question(ask=f"solve {a}x + {c} = {a * root + c}",
                                          accept=(f"x = {root}",)))
            elif form == "quadratic":
                p, q = rng.randint(1, 9), rng.randint(1, 9)
                questions.append(Question(
                    ask=f"solve x^2 - {p + q}x + {p * q} = 0",
                    accept=(f"x = {min(p, q)}",), note="both roots wanted"))
            elif form == "expand":
                p, q = rng.randint(1, 9), rng.randint(1, 9)
                questions.append(Question(
                    ask=f"expand (x+{p})(x+{q})",
                    accept=(f"x^2 + {p + q}x + {p * q}" if p + q != 1
                            else f"x^2 + x + {p * q}",)))
            elif form == "evaluate":
                a, b, at = rng.randint(2, 6), rng.randint(1, 9), rng.randint(2, 6)
                questions.append(Question(
                    ask=f"what is the value of {a}x^2 + {b} when x = {at}",
                    accept=(str(a * at * at + b),), exact=True))
            else:
                x, y = rng.randint(1, 9), rng.randint(1, 9)
                questions.append(Question(
                    ask=f"solve x + y = {x + y} and x - y = {x - y}",
                    accept=(f"x = {x}, y = {y}",)))
        return questions


class Sequences(MathSubject):
    id, title = "sequences", "sequences and series"
    teaches = "a list of numbers can have a rule, and the rule predicts the next one"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            first, step = rng.randint(1, 12), rng.randint(2, 9)
            form = rng.choice(("next_ap", "next_gp", "nth", "sum"))
            if form == "next_ap":
                terms = [first + step * k for k in range(4)]
                questions.append(Question(
                    ask=f"what is the next term in {', '.join(map(str, terms))}?",
                    accept=(str(first + step * 4),), exact=True))
            elif form == "next_gp":
                ratio = rng.randint(2, 4)
                terms = [first * ratio ** k for k in range(4)]
                questions.append(Question(
                    ask=f"what is the next term in {', '.join(map(str, terms))}?",
                    accept=(str(first * ratio ** 4),), exact=True))
            elif form == "nth":
                n = rng.randint(5, 20)
                terms = [first + step * k for k in range(3)]
                questions.append(Question(
                    ask=f"what is the {n}th term of {', '.join(map(str, terms))}?",
                    accept=(str(first + step * (n - 1)),), exact=True))
            else:
                n = rng.randint(5, 60)
                questions.append(Question(
                    ask=f"what is the sum of the first {n} natural numbers?",
                    accept=(str(n * (n + 1) // 2),), exact=True))
        return questions


class Geometry(MathSubject):
    id, title = "geometry", "area, perimeter and angles"
    teaches = "a shape's size follows from its measurements"
    items = 10

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            form = rng.choice(("rectangle", "triangle", "square", "pythagoras", "angles"))
            if form == "rectangle":
                l, b = rng.randint(2, 30), rng.randint(2, 30)
                if rng.random() < 0.5:
                    questions.append(Question(
                        ask=f"what is the area of a rectangle of length {l} and width {b}?",
                        accept=(str(l * b),), exact=True))
                else:
                    questions.append(Question(
                        ask=f"what is the perimeter of a rectangle of length {l} and width {b}?",
                        accept=(str(2 * (l + b)),), exact=True))
            elif form == "triangle":
                base, height = rng.randint(2, 30) * 2, rng.randint(2, 30)
                questions.append(Question(
                    ask=f"what is the area of a triangle with base {base} and height {height}?",
                    accept=(str(base * height // 2),), exact=True))
            elif form == "square":
                side = rng.randint(2, 25)
                questions.append(Question(ask=f"what is the area of a square of side {side}?",
                                          accept=(str(side * side),), exact=True))
            elif form == "pythagoras":
                a, b, c = rng.choice(((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
                                      (6, 8, 10), (9, 12, 15)))
                questions.append(Question(
                    ask=f"what is the hypotenuse of a right angled triangle "
                        f"with sides {a} and {b}?",
                    accept=(str(c),), exact=True))
            else:
                first, second = rng.randint(20, 80), rng.randint(20, 80)
                questions.append(Question(
                    ask=f"two angles of a triangle are {first} and {second}, "
                        f"what is the third angle?",
                    accept=(f"{180 - first - second}°", str(180 - first - second))))
        return questions


class Mensuration(MathSubject):
    id, title = "mensuration", "volume and surface area"
    teaches = "a solid's size follows from its measurements too"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            if rng.random() < 0.5:
                side = rng.randint(2, 15)
                if rng.random() < 0.5:
                    questions.append(Question(ask=f"what is the volume of a cube of side {side}?",
                                              accept=(str(side ** 3),), exact=True))
                else:
                    questions.append(Question(
                        ask=f"what is the surface area of a cube of side {side}?",
                        accept=(str(6 * side * side),), exact=True))
            else:
                l, b, h = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 12)
                questions.append(Question(
                    ask=f"what is the volume of a cuboid of length {l}, width {b} "
                        f"and height {h}?",
                    accept=(str(l * b * h),), exact=True))
        return questions


class Units(MathSubject):
    id, title = "units", "measurement and conversion"
    teaches = "a quantity keeps its size when its unit changes, and only then"

    _PAIRS = (("km", "metres", 1000), ("m", "cm", 100), ("kg", "grams", 1000),
              ("hours", "minutes", 60), ("minutes", "seconds", 60), ("litres", "ml", 1000))

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            source, target, factor = rng.choice(self._PAIRS)
            amount = rng.randint(2, 40)
            questions.append(Question(ask=f"convert {amount} {source} to {target}",
                                      accept=(str(amount * factor),)))
        return questions


class Statistics(MathSubject):
    id, title = "statistics", "mean, median, mode and range"
    teaches = "a list of numbers has a middle, and there is more than one of them"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            data = [rng.randint(1, 40) for _ in range(rng.choice((5, 6, 7)))]
            listed = ", ".join(map(str, data))
            form = rng.choice(("mean", "median", "range"))
            if form == "mean":
                value = Fraction(sum(data), len(data))
                accept = (self.number(value),) if value.denominator == 1 else \
                    (self.number(value), self.number(float(value)))
                questions.append(Question(ask=f"what is the mean of {listed}?", accept=accept))
            elif form == "median":
                ordered = sorted(data)
                middle = len(ordered) // 2
                value = Fraction(ordered[middle]) if len(ordered) % 2 else \
                    Fraction(ordered[middle - 1] + ordered[middle], 2)
                questions.append(Question(ask=f"what is the median of {listed}?",
                                          accept=(self.number(value),)))
            else:
                questions.append(Question(ask=f"what is the range of {listed}?",
                                          accept=(str(max(data) - min(data)),), exact=True))
        return questions


class Probability(MathSubject):
    id, title = "probability", "the probability of a simple event"
    teaches = "a chance is a count of what qualifies over a count of what could happen"
    items = 6

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            if rng.random() < 0.5:
                face = rng.randint(1, 6)
                questions.append(Question(
                    ask=f"what is the probability of getting a {face} on a die?",
                    accept=("1/6",)))
            else:
                red, blue = rng.randint(1, 8), rng.randint(1, 8)
                value = Fraction(red, red + blue)
                questions.append(Question(
                    ask=f"a bag has {red} red and {blue} blue balls, "
                        f"what is the probability of red?",
                    accept=(self.number(value),)))
        return questions


class Calculus(MathSubject):
    id, title = "calculus", "differentiation and integration"
    teaches = "a rate of change and an accumulation are each one rule applied to each term"
    items = 6

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            power, coefficient = rng.randint(2, 6), rng.randint(1, 9)
            if rng.random() < 0.5:
                term = f"{coefficient}x^{power}" if coefficient != 1 else f"x^{power}"
                head = "" if coefficient * power == 1 else str(coefficient * power)
                tail = "x" if power == 2 else f"x^{power - 1}"
                questions.append(Question(ask=f"what is the derivative of {term}?",
                                          accept=(f"{head}{tail}",)))
            else:
                at = rng.randint(1, 4)
                value = Fraction(coefficient * at ** (power + 1), power + 1)
                questions.append(Question(
                    ask=f"integrate {coefficient}x^{power} from 0 to {at}",
                    accept=(self.number(value), self.number(float(value))[:6])))
        return questions


class WordProblems(MathSubject):
    id, title = "word", "speed, distance, time and work"
    teaches = "a sentence can hold a formula, and finding it is most of the answer"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            speed, hours = rng.choice((20, 30, 40, 50, 60, 75, 80)), rng.randint(2, 8)
            if rng.random() < 0.5:
                questions.append(Question(
                    ask=f"a train travels {speed * hours} km in {hours} hours, "
                        f"what is its speed?",
                    accept=(str(speed),), exact=True))
            else:
                questions.append(Question(
                    ask=f"a car goes at {speed} km/h for {hours} hours, "
                        f"what distance does it cover?",
                    accept=(str(speed * hours),), exact=True))
        return questions


class Powers(MathSubject):
    id, title = "powers", "powers, roots and logarithms"
    teaches = "a power and a root are the same fact read in two directions"

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            base = rng.randint(2, 15)
            form = rng.choice(("square", "root", "power", "log"))
            if form == "square":
                questions.append(Question(ask=f"what is the square of {base}?",
                                          accept=(str(base * base),), exact=True))
            elif form == "root":
                questions.append(Question(ask=f"what is the square root of {base * base}?",
                                          accept=(str(base),), exact=True))
            elif form == "power":
                exponent = rng.randint(2, 6)
                questions.append(Question(ask=f"what is {base} to the power {exponent}?",
                                          accept=(str(base ** exponent),), exact=True))
            else:
                root, exponent = rng.choice((2, 3, 5, 10)), rng.randint(2, 4)
                questions.append(Question(ask=f"what is log base {root} of {root ** exponent}?",
                                          accept=(str(exponent),), exact=True))
        return questions


class ClosedArithmetic(MathSubject):
    """The floor she already had, kept in the syllabus so a regression in it is visible here.

    This is the one subject :mod:`nyxara.njp.school` also examines, and the duplication is
    deliberate: the mathematician deliberately does **not** answer bare arithmetic — that route
    predates it and still owns it — so if this subject ever dips, the cause is in the calculator
    or its wiring and not in anything this module added.
    """

    id, title = "arithmetic", "closed arithmetic"
    teaches = "an expression has a value, and it is the same value every time"
    items = 6

    def generate(self, rng: random.Random) -> List[Question]:
        questions = []
        for _ in range(self.items):
            a, b, c = rng.randint(2, 40), rng.randint(2, 30), rng.randint(2, 9)
            form, value = rng.choice([
                (f"what is {a} + {b}?", a + b),
                (f"what is {a} * {c}?", a * c),
                (f"{a} - {b} kitna hai?", a - b),
                (f"what is {a} + {b} * {c}?", a + b * c),
            ])
            questions.append(Question(ask=form, accept=(str(value),), exact=True))
        return questions


# --------------------------------------------------------------------------- #
# the knowing subject — the only thing here a lesson can move
# --------------------------------------------------------------------------- #

#: ``(term, the sentence that states it, what an answer must contain)``.
#:
#: Every definition is phrased as ``is a`` or ``means`` because that is what the store can read
#: back. That is not a stylistic choice and it is worth being blunt about: the same sentence
#: written as *"the area of a circle is pi r squared"* grounds perfectly well — as
#: ``('area of a circle', 'has_property', 'pi r squared')`` — and is then **unaskable**, because
#: "what is the area of a circle" reads as ``is_a`` and a held ``has_property`` does not answer
#: it. Measured, not assumed. The read/write asymmetry this package keeps finding, met
#: here by writing in the shape that can be read rather than by widening what answers what:
#: ``_GENERAL_ANSWER`` exists to stop "what is a sparrow" being answered "brown", and a
#: mathematics lesson is not a reason to take that guard down.
#:
#: Two further phrasings had to be found by measurement rather than chosen by taste, and each is
#: a fact about the reader rather than about mathematics. *"The median is the middle value **when**
#: the values are in order"* is filed as a **condition**, not a definition, so the term comes back
#: unaskable; the ``when`` clause is gone. And the definite article matters — *"the mean is …"*
#: does not read back where *"a mean is …"* does. Every line below round-trips through
#: :meth:`NJPBrain.think`, asserted by a test, because a lesson that does not land is a lesson
#: that reports a gain it did not produce.
_VOCABULARY: Tuple[Tuple[str, str, str], ...] = (
    ("a prime number", "a prime number is a number with exactly two factors", "two factors"),
    ("a factor", "a factor is a divisor of a number", "divisor"),
    ("a multiple", "a multiple is a number reached by multiplying another number", "multiplying"),
    ("an even number", "an even number is a number divisible by two", "divisible by two"),
    ("an odd number", "an odd number is a number not divisible by two", "not divisible by two"),
    ("a fraction", "a fraction is a part of a whole written as one number over another",
     "part of a whole"),
    ("a decimal", "a decimal is a number written with a point", "point"),
    ("a percentage", "a percentage is a fraction with a denominator of one hundred", "hundred"),
    ("a ratio", "a ratio is a comparison of two quantities of the same kind", "comparison"),
    ("an equation", "an equation is a statement that two expressions are equal", "equal"),
    ("a variable", "a variable is a symbol standing for an unknown number", "symbol"),
    ("a polygon", "a polygon is a closed shape with straight sides", "closed shape"),
    ("a hypotenuse", "a hypotenuse is the side opposite the right angle",
     "opposite the right angle"),
    ("a perimeter", "a perimeter is the distance around a shape", "distance around a shape"),
    ("an area", "an area is the amount of surface a shape covers", "surface"),
    ("a mean", "a mean is an average found by dividing the total by the count", "average"),
    ("a median", "a median is the middle value of an ordered list", "middle value"),
    ("a mode", "a mode is the value that appears most often in a list", "most often"),
    ("a probability", "a probability is a number between zero and one", "between zero and one"),
    ("a derivative", "a derivative is a rate of change", "rate of change"),
)


class Vocabulary(MathSubject):
    """The words of mathematics — the one subject here where a lesson can move a number.

    **Doing and saying are different capabilities.** She can work out that 91 is not prime through
    :mod:`nyxara.njp.mathematics` and, before the lesson, cannot say what a prime number *is*: the
    faculty computes and the store remembers, and neither implies the other. What is measured here
    is the second one.

    **Half the terms are deliberately not taught, and they are scored as controls.** Ten
    definitions are stated and ten are withheld — which ten is decided by the exam's own seed, so
    the split is different on a different run and cannot be arranged for. On the taught half a
    right answer scores; on the withheld half **silence** scores and any assertion is a miss,
    however plausible. A brain that answered all twenty would read as mastery on the taught half
    and would be guessing, and the split is the only thing that can tell those two apart.

    It is honest about what it is not: this is retrieval, not generalisation. The question is
    phrased differently from the lesson — *"what is a prime number"* against *"a prime number is a
    number with exactly two factors"* — so what is scored is whether a stated fact can be **asked
    back**, which is exactly the failure mode this package keeps finding, and not whether she can
    define a term nobody defined.
    """

    id, title = "vocabulary", "the words of mathematics"
    teaches = "the terms themselves — what a thing is, as against what its value comes to"
    threshold, items = 0.8, 20

    def __init__(self) -> None:
        #: Which terms the lesson **will** state. Fixed on the first exam — that is, on the
        #: pre-test — and not by :meth:`teach`, and that ordering is the whole validity of the
        #: number.
        #:
        #: Chosen inside `teach` instead, as the first version did, the pre-test has no split to
        #: grade against: every term is a control, she is silent on all twenty, and the floor
        #: reads **1.00 for knowing nothing at all**. The post-test then scores 0.85 and the
        #: report shows a lesson that *lost* fifteen points by teaching. Both numbers were
        #: meaningless and only their sign made it obvious.
        self.planned: Tuple[str, ...] = ()
        #: Which terms were actually stated. Set by :meth:`teach`; empty at pre-test time, which
        #: is why the pre-test scores the taught half as unanswered rather than as controls.
        self.stated: Tuple[str, ...] = ()

    def _split(self, mint: Mint) -> Tuple[str, ...]:
        if not self.planned:
            chosen = mint.rng.sample(range(len(_VOCABULARY)), len(_VOCABULARY) // 2)
            self.planned = tuple(_VOCABULARY[index][0] for index in sorted(chosen))
        return self.planned

    def teach(self, brain: Any, mint: Mint, *, coder: Any = None) -> Taught:
        planned = self._split(mint)
        stated = []
        for term, sentence, _ in _VOCABULARY:
            if term not in planned:
                continue
            try:
                brain.think(sentence)
                stated.append(term)
            except Exception:  # noqa: BLE001 — a lesson that fails is a lesson that taught nothing
                continue
        self.stated = tuple(stated)
        return Taught(len(stated), f"stated {len(stated)} of {len(_VOCABULARY)} definitions; "
                                   f"the other {len(_VOCABULARY) - len(stated)} are controls")

    @staticmethod
    def _ask_for(term: str) -> str:
        """The question form. One per term, and phrased unlike the lesson that stated it."""
        return f"what is {term}?"

    def exam(self, brain: Any, mint: Mint, *, coder: Any = None) -> Tuple[Score, List[str]]:
        planned = self._split(mint)
        questions = []
        for term, _, expected in _VOCABULARY:
            in_lesson = term in planned
            questions.append(Question(
                ask=self._ask_for(term),
                accept=(expected,) if in_lesson else (),
                silence_ok=not in_lesson,
                note="taught" if in_lesson else "withheld"))
        return self.sit(brain, questions)


# --------------------------------------------------------------------------- #
# the restraint subject — every item a control
# --------------------------------------------------------------------------- #

#: Questions that must come back empty. Two kinds, and both matter. The first has no mathematics
#: in it at all and is here because a mathematician that answers everything is the organ that
#: filed ``('convert', '5') → 'km metres'``. The second is mathematics-shaped and **has no
#: answer** — a division by zero, a conversion between two different quantities, a sequence with
#: no rule, a shape whose measurement was never given. Those are the harder half: every one of
#: them reaches a skill, is recognised, and has to be declined by that skill rather than by
#: failing to match a trigger.
_RESTRAINT: Tuple[str, ...] = (
    "what is the capital of Ruritania?",
    "who wrote the book that nobody has written?",
    "what colour is the number seven?",
    "what is 7 divided by 0?",
    "what is the hcf of red and blue?",
    "what is the square root of a banana?",
    "convert 5 km to kilograms",
    "what is the area of a circle?",
    "what is the next term in 2, 7, 1?",
    "what is the mode of 1, 2, 3, 4, 5?",
)


class Restraint(MathSubject):
    """Ten questions she must not answer, and the two reasons a question can be one of them.

    Scored the way :class:`~nyxara.njp.school.Abstention` scores: silence is right and an
    assertion is wrong, however confident. This is the subject that would have caught the defect
    :mod:`nyxara.njp.mathematics` was written for, had it existed — the five maths instructions
    that came back as *"noted: simplify fraction 18 24"* were not silent failures, they were
    confident ones, and only a control can see that difference.
    """

    id, title = "restraint", "the questions with no answer"
    teaches = "a question that has no answer gets no answer"
    threshold, items = 0.9, len(_RESTRAINT)

    def generate(self, rng: random.Random) -> List[Question]:
        return [Question(ask=ask, silence_ok=True) for ask in _RESTRAINT]


#: The syllabus, in the order a school teaches it: number, then the ways of writing a number, then
#: what you do to numbers, then shape, then measurement, then data, then change — and the two
#: subjects that are about *her* rather than about mathematics kept for last.
MATHS_SUBJECTS: Tuple[Any, ...] = (
    ClosedArithmetic, NumberTheory, Fractions, Percentage, Commerce, RatioProportion,
    Powers, Algebra, Sequences, Geometry, Mensuration, Units, Statistics, Probability,
    Calculus, WordProblems, Vocabulary, Restraint,
)


class MathSchool(School):
    """The mathematics syllabus, taught and examined by :class:`~nyxara.njp.school.School`.

    A subclass with a different reading list and nothing else. Every guarantee the parent makes
    holds unchanged here — a fresh :class:`~nyxara.njp.school.Mint` per subject and per phase, so
    a post-test item cannot be one a lesson mentioned; three outcomes kept apart; the floor
    reported beside the ceiling — and none of them is re-implemented, which is the only way they
    stay the same guarantees.
    """

    def __init__(self, *, seed: int = 11, rounds: int = 1,
                 subjects: Optional[Sequence[Any]] = None, verbose: bool = False) -> None:
        super().__init__(seed=seed, rounds=rounds,
                         subjects=MATHS_SUBJECTS if subjects is None else subjects,
                         verbose=verbose)


class MathExam:
    """Every subject, examined and not taught, on more items than the school uses.

    The counterpart to :meth:`nyxara.njp.brain.NJPBrain.sit_general_exam` and the same discipline:
    **it teaches nothing and writes nothing**, so it may be run twice around something that
    mutates her and the two numbers are comparable. ``vocabulary`` is therefore absent from the
    default papers — it is the one subject whose score is a fact about what it just taught, and an
    exam that taught would be an exam that could not be repeated.
    """

    #: Everything except the subject that has to teach to be worth sitting.
    PAPERS: Tuple[Any, ...] = tuple(s for s in MATHS_SUBJECTS if s is not Vocabulary)

    def __init__(self, brain: Any = None, *, limit: int = 40, seed: int = 20260902) -> None:
        self.brain = brain
        self.limit = max(1, int(limit))
        self.seed = int(seed)

    def sit(self, papers: Optional[Sequence[Any]] = None) -> Transcript:
        brain = self.brain
        if brain is None:
            from nyxara.njp.brain import NJPBrain
            brain = NJPBrain(ExamConditions())
        started = time.time()
        transcript = Transcript(seed=self.seed)
        chosen = list(papers if papers is not None else self.PAPERS)
        for index, factory in enumerate(chosen):
            subject = factory() if isinstance(factory, type) else factory
            result = Result(subject=subject.id, title=subject.title, teaches=subject.teaches,
                            threshold=subject.threshold, note="examined, not taught")
            paper_started = time.time()
            try:
                # The item count is raised for the exam and left alone for the school: a bigger
                # paper is a tighter measurement and a slower lesson, and only one of those is
                # worth paying for on every round. `Restraint` is exempt — its items are ten
                # named questions rather than a generator, and repeating them would inflate the
                # denominator without asking anything new.
                if not isinstance(subject, Restraint):
                    subject.items = self.limit
                mint = Mint(random.Random(self.seed * 100003 + index * 1009))
                result.post, result.misses = subject.exam(brain, mint)
            except Exception as exc:  # noqa: BLE001 — a paper that crashes is a failed paper
                result.note = f"raised {type(exc).__name__}: {exc}"
            result.pre = result.post      # nothing was taught, so there is no gain to claim
            result.seconds = time.time() - paper_started
            transcript.results.append(result)
        transcript.seconds = time.time() - started
        return transcript


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m nyxara.njp.mathschool [--seed N] [--rounds N] [--exam] [--json]``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="teach NYXARA mathematics, then examine her")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--limit", type=int, default=25,
                        help="items per paper, exam only")
    parser.add_argument("--exam", action="store_true",
                        help="sit the examination and teach nothing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain(ExamConditions())
    if args.exam:
        transcript = MathExam(brain, limit=args.limit, seed=args.seed).sit()
        heading = f"NYXARA · mathematics examination (seed {args.seed}, {args.limit} per paper)"
    else:
        transcript = MathSchool(seed=args.seed, rounds=args.rounds).attend(brain)
        heading = f"NYXARA · mathematics school (seed {args.seed}, {args.rounds} round(s))"

    if args.json:
        print(json.dumps(transcript.to_dict() if hasattr(transcript, "to_dict")
                         else [r.to_dict() for r in transcript.results], indent=2))
        return 0

    print(heading)
    print("-" * len(heading))
    print(f"{'subject':<14}{'pre':>6}{'post':>6}{'gain':>7}   note")
    right = wrong = absent = 0
    for result in transcript.results:
        flag = "already" if result.already else ("ok" if result.mastered else "FAILING")
        print(f"{result.subject:<14}{result.pre.accuracy:>6.2f}{result.post.accuracy:>6.2f}"
              f"{result.gain:>+7.2f}   {flag:<8} {result.note[:46]}")
        right += result.post.right
        wrong += result.post.wrong
        absent += result.post.abstained
    total = right + wrong + absent
    print("-" * len(heading))
    print(f"right {right} · wrong {wrong} · abstained {absent} · "
          f"accuracy {right / total if total else 0.0:.4f} · "
          f"mastered {len(transcript.mastered)}/{len(transcript.results)} · "
          f"{transcript.seconds:.1f}s")
    for result in transcript.results:
        for miss in result.misses[:3]:
            print(f"  {result.subject}: {miss}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
