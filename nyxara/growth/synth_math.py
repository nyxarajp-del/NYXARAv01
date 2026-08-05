"""NYXARA · growth/synth_math.py — worked mathematics, correct by construction (🔢→📐).

Every generator here **builds the problem from a known solution**, then writes the steps that
recover it. Nothing is guessed and then checked; there is no model in the loop that could be
wrong, and the answer is a fact about how the problem was constructed rather than a claim about
it. That is a stronger guarantee than verification, and it is why this can run at corpus scale.

Two things the old four generators got right and are kept: digits never merge in the tokenizer,
so column arithmetic is learnable — and the working is *shown*, because a small model that has
only ever seen answers has no mechanism for producing one, while a model that has seen the
derivation can reproduce the procedure.

One thing they got wrong: state space. ``_sympy_derivative`` could emit **240 distinct documents,
ever**, and nothing noticed. Every generator here declares its space and
:func:`~nyxara.growth.synth_common.draw` enforces it, so a narrow generator is a loud failure.

Surface diversity is deliberate and load-bearing. The old generators produced 232 unique
pre-tokens across 12,000 documents, which trains a BPE to 681 tokens of a requested 16,384 — a
vocabulary that cannot support a 300M model. Names, units, objects and phrasings are drawn from
wide pools here for exactly that reason: a corpus that says "Solve for x" ten million times
teaches the tokenizer one phrase.

Pure standard library plus the optional ``sympy`` (already a dependency).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Callable, Optional, Tuple

from nyxara.growth.synth_common import Generator

__all__ = ["GENERATORS", "generators"]

Built = Optional[Tuple[str, str]]

# --------------------------------------------------------------------------- #
# Surface pools — the difference between a corpus and one sentence repeated
# --------------------------------------------------------------------------- #
_NAMES = ("Aarav", "Priya", "Rahul", "Meera", "Kabir", "Ananya", "Vikram", "Nisha", "Arjun",
          "Diya", "Rohan", "Sana", "Ishaan", "Tara", "Dev", "Kiara", "Manav", "Riya",
          "Yusuf", "Leila", "Omar", "Zara", "Mateo", "Sofia", "Hana", "Kenji", "Amara",
          "Noor", "Elena", "Tomas", "Grace", "Jonas", "Ada", "Ravi", "Fatima", "Luca")
_OBJECTS = ("notebooks", "apples", "bricks", "seedlings", "batteries", "tickets", "marbles",
            "bolts", "sensors", "candles", "envelopes", "beads", "planks", "vials", "crates",
            "spools", "tiles", "coins", "stamps", "cartridges", "washers", "lanterns")
_PLACES = ("the workshop", "the depot", "the greenhouse", "the library", "the market",
           "the lab", "the warehouse", "the observatory", "the clinic", "the studio")
_ASK_SOLVE = ("Solve for {v}:", "Find {v}:", "What value of {v} satisfies", "Determine {v}:",
              "Solve the equation for {v}:", "For which {v} does this hold?")
_ASK_VALUE = ("What is", "Compute", "Evaluate", "Work out", "Calculate", "Find the value of")


def _pick(rng: random.Random, pool) -> str:
    return pool[rng.randrange(len(pool))]


def _ask_value(rng: random.Random, expr: str) -> str:
    return f"{_pick(rng, _ASK_VALUE)} {expr}?"


def _signed(n: int) -> str:
    return f"+ {n}" if n >= 0 else f"- {abs(n)}"


# --------------------------------------------------------------------------- #
# Arithmetic, shown column by column
# --------------------------------------------------------------------------- #
def _long_addition(rng: random.Random) -> Built:
    a, b = rng.randint(100, 99_999), rng.randint(100, 99_999)
    da, db = str(a)[::-1], str(b)[::-1]
    carry, cols = 0, []
    for i in range(max(len(da), len(db))):
        x = int(da[i]) if i < len(da) else 0
        y = int(db[i]) if i < len(db) else 0
        total = x + y + carry
        cols.append(f"  column {i + 1}: {x} + {y} + carry {carry} = {total} "
                    f"-> digit {total % 10}, carry {total // 10}")
        carry = total // 10
    steps = "Add column by column from the right:\n" + "\n".join(cols)
    if carry:
        steps += f"\n  final carry: {carry}"
    return _ask_value(rng, f"{a} + {b}"), f"{steps}\n{a} + {b} = {a + b}"


def _long_subtraction(rng: random.Random) -> Built:
    a = rng.randint(1000, 99_999)
    b = rng.randint(100, a)                      # constructed so the result is non-negative
    da, db = str(a)[::-1], str(b)[::-1]
    borrow, cols = 0, []
    for i in range(len(da)):
        x = int(da[i]) - borrow
        y = int(db[i]) if i < len(db) else 0
        if x < y:
            x, borrow = x + 10, 1
            cols.append(f"  column {i + 1}: borrow 10 -> {x} - {y} = {x - y}")
        else:
            borrow = 0
            cols.append(f"  column {i + 1}: {x} - {y} = {x - y}")
    steps = "Subtract column by column from the right:\n" + "\n".join(cols)
    return _ask_value(rng, f"{a} - {b}"), f"{steps}\n{a} - {b} = {a - b}"


def _long_multiplication(rng: random.Random) -> Built:
    a, b = rng.randint(12, 999), rng.randint(12, 99)
    rows = []
    for i, digit in enumerate(str(b)[::-1]):
        partial = a * int(digit) * (10 ** i)
        rows.append(f"  {a} x {digit}{'0' * i} = {partial}")
    partials = " + ".join(str(a * int(d) * 10 ** i) for i, d in enumerate(str(b)[::-1]))
    steps = ("Multiply by each digit of the second number, then add:\n"
             + "\n".join(rows) + f"\n  sum: {partials}")
    return _ask_value(rng, f"{a} x {b}"), f"{steps}\n{a} x {b} = {a * b}"


def _long_division(rng: random.Random) -> Built:
    divisor = rng.randint(3, 97)
    quotient = rng.randint(11, 9999)
    remainder = rng.randrange(divisor)
    dividend = divisor * quotient + remainder    # built from the answer, so it is exact
    steps = (f"Divide {dividend} by {divisor}.\n"
             f"  {divisor} x {quotient} = {divisor * quotient}\n"
             f"  {dividend} - {divisor * quotient} = {remainder}\n"
             f"  {remainder} < {divisor}, so the division stops here.")
    answer = (f"{steps}\n{dividend} = {divisor} x {quotient} + {remainder}\n"
              f"quotient {quotient}, remainder {remainder}")
    return f"Divide {dividend} by {divisor}, giving quotient and remainder.", answer


def _gcd_euclid(rng: random.Random) -> Built:
    a, b = rng.randint(12, 9000), rng.randint(12, 9000)
    x, y = max(a, b), min(a, b)
    rows = []
    while y:
        rows.append(f"  {x} = {y} x {x // y} + {x % y}")
        x, y = y, x % y
    lcm = a * b // x
    steps = "Euclid's algorithm:\n" + "\n".join(rows)
    return (f"Find the greatest common divisor of {a} and {b}, and their lowest common multiple.",
            f"{steps}\nThe last non-zero remainder is the gcd.\n"
            f"gcd({a}, {b}) = {x}\nlcm = {a} x {b} / {x} = {lcm}")


def _prime_factorisation(rng: random.Random) -> Built:
    n = rng.randint(24, 99_999)
    remaining, factors, rows = n, [], []
    d = 2
    while d * d <= remaining:
        while remaining % d == 0:
            rows.append(f"  {remaining} / {d} = {remaining // d}")
            remaining //= d
            factors.append(d)
        d += 1 if d == 2 else 2
    if remaining > 1:
        factors.append(remaining)
        rows.append(f"  {remaining} is prime")
    pretty = " x ".join(f"{p}^{factors.count(p)}" if factors.count(p) > 1 else str(p)
                        for p in sorted(set(factors)))
    return (f"Give the prime factorisation of {n}.",
            "Divide by the smallest prime that fits, repeatedly:\n" + "\n".join(rows)
            + f"\n{n} = {pretty}")


def _base_conversion(rng: random.Random) -> Built:
    n = rng.randint(20, 65_535)
    base = _pick(rng, ("2", "8", "16"))
    b = int(base)
    digits, rows, remaining = "", [], n
    alphabet = "0123456789ABCDEF"
    while remaining:
        rows.append(f"  {remaining} / {b} = {remaining // b} remainder {remaining % b}")
        digits = alphabet[remaining % b] + digits
        remaining //= b
    return (f"Convert {n} from base 10 to base {b}.",
            "Divide by the base repeatedly and read the remainders upward:\n" + "\n".join(rows)
            + f"\n{n} in base {b} is {digits}")


# --------------------------------------------------------------------------- #
# Fractions and percentages
# --------------------------------------------------------------------------- #
def _fraction_sum(rng: random.Random) -> Built:
    a, b = rng.randint(1, 20), rng.randint(2, 20)
    c, d = rng.randint(1, 20), rng.randint(2, 20)
    op = _pick(rng, ("+", "-"))
    result = Fraction(a, b) + Fraction(c, d) if op == "+" else Fraction(a, b) - Fraction(c, d)
    return (f"What is {a}/{b} {op} {c}/{d}? Give the answer in lowest terms.",
            f"Common denominator: {b} x {d} = {b * d}\n"
            f"{a}/{b} = {a * d}/{b * d}\n{c}/{d} = {c * b}/{b * d}\n"
            f"{'Sum' if op == '+' else 'Difference'}: {a * d} {op} {c * b} = "
            f"{a * d + (c * b if op == '+' else -c * b)}, over {b * d}\n"
            f"Lowest terms: {result.numerator}/{result.denominator}")


def _percentage(rng: random.Random) -> Built:
    name, obj, place = _pick(rng, _NAMES), _pick(rng, _OBJECTS), _pick(rng, _PLACES)
    total = rng.randint(20, 4000)
    pct = _pick(rng, ("5", "10", "12", "15", "20", "25", "30", "40", "45", "60", "75", "80"))
    p = int(pct)
    part = Fraction(total * p, 100)
    return (f"{name} counted {total} {obj} at {place}. {p}% of them were damaged. "
            f"How many were damaged?",
            f"{p}% means {p}/100.\n"
            f"  {p}/100 x {total} = {total * p}/100\n"
            f"  = {part.numerator}/{part.denominator}\n"
            f"{p}% of {total} is {part if part.denominator != 1 else part.numerator}")


def _compound_interest(rng: random.Random) -> Built:
    principal = rng.randrange(1000, 100_000, 500)
    rate = rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    years = rng.randint(2, 6)
    rows, amount = [], Fraction(principal)
    for y in range(1, years + 1):
        interest = amount * Fraction(rate, 100)
        amount += interest
        rows.append(f"  year {y}: interest = {float(interest):.2f}, balance = {float(amount):.2f}")
    return (f"{_pick(rng, _NAMES)} deposits {principal} at {rate}% compound interest per year. "
            f"What is the balance after {years} years?",
            "Each year the balance is multiplied by "
            f"(1 + {rate}/100):\n" + "\n".join(rows)
            + f"\nBalance after {years} years = {float(amount):.2f}")


def _ratio_share(rng: random.Random) -> Built:
    a, b = rng.randint(1, 12), rng.randint(1, 12)
    unit = rng.randint(3, 200)
    total = (a + b) * unit                        # divisible by construction
    n1, n2 = _pick(rng, _NAMES), _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS)
    return (f"{total} {obj} are shared between {n1} and {n2} in the ratio {a}:{b}. "
            f"How many does each receive?",
            f"Total parts: {a} + {b} = {a + b}\n"
            f"One part: {total} / {a + b} = {unit}\n"
            f"{n1}: {a} x {unit} = {a * unit}\n{n2}: {b} x {unit} = {b * unit}\n"
            f"Check: {a * unit} + {b * unit} = {total}")


def _unit_conversion(rng: random.Random) -> Built:
    table = (("km", "m", 1000), ("m", "cm", 100), ("kg", "g", 1000), ("hours", "minutes", 60),
             ("minutes", "seconds", 60), ("L", "mL", 1000), ("tonnes", "kg", 1000),
             ("days", "hours", 24), ("cm", "mm", 10), ("GB", "MB", 1024))
    src, dst, factor = table[rng.randrange(len(table))]
    n = rng.randint(2, 9999)
    return (f"Convert {n} {src} to {dst}.",
            f"1 {src} = {factor} {dst}, so multiply by {factor}.\n"
            f"  {n} x {factor} = {n * factor}\n{n} {src} = {n * factor} {dst}")


def _speed_distance_time(rng: random.Random) -> Built:
    speed = rng.randint(5, 120)
    hours = rng.randint(2, 12)
    distance = speed * hours
    name = _pick(rng, _NAMES)
    missing = rng.randrange(3)
    if missing == 0:
        return (f"{name} travels at {speed} km/h for {hours} hours. How far?",
                f"distance = speed x time\n  = {speed} x {hours}\n= {distance} km")
    if missing == 1:
        return (f"{name} covers {distance} km at {speed} km/h. How long does it take?",
                f"time = distance / speed\n  = {distance} / {speed}\n= {hours} hours")
    return (f"{name} covers {distance} km in {hours} hours. What is the average speed?",
            f"speed = distance / time\n  = {distance} / {hours}\n= {speed} km/h")


# --------------------------------------------------------------------------- #
# Algebra, constructed from the root
# --------------------------------------------------------------------------- #
def _linear_equation(rng: random.Random) -> Built:
    a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15])
    x = rng.randint(-40, 40)
    b = rng.randint(-90, 90)
    c = a * x + b
    var = _pick(rng, ("x", "y", "t", "n", "u", "w", "p", "k"))
    ask = _pick(rng, _ASK_SOLVE).format(v=var)
    return (f"{ask} {a}{var} {_signed(b)} = {c}",
            f"Start: {a}{var} {_signed(b)} = {c}\n"
            f"{'Add' if b < 0 else 'Subtract'} {abs(b)} {'to' if b < 0 else 'from'} both sides: "
            f"{a}{var} = {c - b}\n"
            f"Divide both sides by {a}: {var} = {(c - b) // a}\n"
            f"Check: {a}({x}) {_signed(b)} = {c}\n{var} = {x}")


def _two_step_both_sides(rng: random.Random) -> Built:
    x = rng.randint(-30, 30)
    a, c = rng.randint(2, 15), rng.randint(2, 15)
    if a == c:
        return None                               # no unique solution to construct
    b = rng.randint(-60, 60)
    d = (a - c) * x + b
    var = _pick(rng, ("x", "y", "m", "q"))
    return (f"Solve for {var}: {a}{var} {_signed(b)} = {c}{var} {_signed(d)}",
            f"Collect {var} on the left: {a}{var} - {c}{var} = {d} - ({b})\n"
            f"  {a - c}{var} = {d - b}\n"
            f"Divide by {a - c}: {var} = {(d - b) // (a - c)}\n"
            f"Check: {a}({x}) {_signed(b)} = {a * x + b}, and {c}({x}) {_signed(d)} = {c * x + d}\n"
            f"{var} = {x}")


def _quadratic_from_roots(rng: random.Random) -> Built:
    r1, r2 = rng.randint(-30, 30), rng.randint(-30, 30)
    a = rng.choice([1, 1, 1, 2, 3, 4, 5, 6])
    b, c = -a * (r1 + r2), a * r1 * r2
    lead = "" if a == 1 else str(a)
    return (f"Solve {lead}x^2 {_signed(b)}x {_signed(c)} = 0",
            f"Factor: {lead}(x {_signed(-r1)})(x {_signed(-r2)}) = 0\n"
            f"A product is zero when a factor is zero.\n"
            f"  x {_signed(-r1)} = 0  ->  x = {r1}\n"
            f"  x {_signed(-r2)} = 0  ->  x = {r2}\n"
            f"Check by expanding: {lead}x^2 {_signed(b)}x {_signed(c)}\n"
            f"x = {r1} or x = {r2}")


def _linear_system(rng: random.Random) -> Built:
    x, y = rng.randint(-15, 15), rng.randint(-15, 15)
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c, d = rng.randint(1, 9), rng.randint(1, 9)
    det = a * d - b * c
    if det == 0:
        return None                               # not uniquely solvable
    e, f = a * x + b * y, c * x + d * y
    return (f"Solve the system:\n  {a}x + {b}y = {e}\n  {c}x + {d}y = {f}",
            f"Eliminate x: multiply the first by {c} and the second by {a}.\n"
            f"  {a * c}x + {b * c}y = {e * c}\n  {a * c}x + {a * d}y = {a * f}\n"
            f"Subtract: {b * c - a * d}y = {e * c - a * f}\n"
            f"  y = {y}\n"
            f"Substitute into the first: {a}x + {b}({y}) = {e}\n"
            f"  {a}x = {e - b * y}\n  x = {x}\n"
            f"Check in the second: {c}({x}) + {d}({y}) = {f}\nx = {x}, y = {y}")


def _expand_binomial(rng: random.Random) -> Built:
    a, b = rng.randint(1, 9), rng.randint(-12, 12)
    c, d = rng.randint(1, 9), rng.randint(-12, 12)
    return (f"Expand ({a}x {_signed(b)})({c}x {_signed(d)}).",
            f"Multiply each term (FOIL):\n"
            f"  first:  {a}x x {c}x = {a * c}x^2\n"
            f"  outer:  {a}x x ({d}) = {a * d}x\n"
            f"  inner:  ({b}) x {c}x = {b * c}x\n"
            f"  last:   ({b}) x ({d}) = {b * d}\n"
            f"Collect the x terms: {a * d}x {_signed(b * c)}x = {a * d + b * c}x\n"
            f"{a * c}x^2 {_signed(a * d + b * c)}x {_signed(b * d)}")


def _arithmetic_sequence(rng: random.Random) -> Built:
    first, step = rng.randint(-40, 60), rng.randint(2, 25)
    n = rng.randint(6, 60)
    nth = first + (n - 1) * step
    total = n * (first + nth) // 2
    return (f"An arithmetic sequence starts at {first} and increases by {step} each term. "
            f"What is term {n}, and the sum of the first {n} terms?",
            f"term(n) = first + (n - 1) x step\n"
            f"  = {first} + ({n} - 1) x {step}\n  = {first} + {(n - 1) * step}\n  = {nth}\n"
            f"sum = n x (first + last) / 2\n"
            f"  = {n} x ({first} + {nth}) / 2\n  = {total}\n"
            f"term {n} = {nth}, sum = {total}")


def _geometric_sequence(rng: random.Random) -> Built:
    first = rng.randint(1, 400)
    ratio = rng.randint(2, 9)
    n = rng.randint(3, 12)
    nth = first * ratio ** (n - 1)
    total = first * (ratio ** n - 1) // (ratio - 1)
    return (f"A geometric sequence starts at {first} with common ratio {ratio}. "
            f"Give term {n} and the sum of the first {n} terms.",
            f"term(n) = first x ratio^(n-1)\n"
            f"  = {first} x {ratio}^{n - 1} = {first} x {ratio ** (n - 1)} = {nth}\n"
            f"sum = first x (ratio^n - 1) / (ratio - 1)\n"
            f"  = {first} x ({ratio ** n} - 1) / {ratio - 1} = {total}\n"
            f"term {n} = {nth}, sum = {total}")


# --------------------------------------------------------------------------- #
# Number theory, counting, statistics, geometry
# --------------------------------------------------------------------------- #
def _modular_arithmetic(rng: random.Random) -> Built:
    base, exp, mod = rng.randint(2, 40), rng.randint(2, 24), rng.randint(3, 97)
    rows, acc = [], 1
    for i in range(1, exp + 1):
        acc = acc * base % mod
        if i <= 4 or i == exp:
            rows.append(f"  {base}^{i} mod {mod} = {acc}")
        elif i == 5:
            rows.append("  ...")
    return (f"Compute {base}^{exp} mod {mod}.",
            "Multiply and reduce at each step, so the numbers stay small:\n" + "\n".join(rows)
            + f"\n{base}^{exp} mod {mod} = {pow(base, exp, mod)}")


def _combinations(rng: random.Random) -> Built:
    n = rng.randint(5, 45)
    k = rng.randint(2, min(8, n - 1))
    obj, name, place = _pick(rng, _OBJECTS), _pick(rng, _NAMES), _pick(rng, _PLACES)
    numer = " x ".join(str(n - i) for i in range(k))
    denom = " x ".join(str(i) for i in range(1, k + 1))
    return (f"{name} must choose {k} of the {n} {obj} at {place}. Order does not matter. "
            f"How many choices are there?",
            f"This is C({n}, {k}) = n! / (k! (n-k)!)\n"
            f"  = ({numer}) / ({denom})\n"
            f"  = {math.perm(n, k)} / {math.factorial(k)}\n"
            f"C({n}, {k}) = {math.comb(n, k)}")


def _permutations(rng: random.Random) -> Built:
    n = rng.randint(4, 40)
    k = rng.randint(2, min(7, n))
    obj, name = _pick(rng, _OBJECTS), _pick(rng, _NAMES)
    return (f"{name} lines up {k} of {n} distinct {obj} on a shelf. Order matters. "
            f"How many arrangements are possible?",
            f"P({n}, {k}) = n! / (n-k)! = " + " x ".join(str(n - i) for i in range(k))
            + f"\nP({n}, {k}) = {math.perm(n, k)}")


def _dice_probability(rng: random.Random) -> Built:
    faces = rng.choice([4, 6, 6, 8, 10, 12, 20])
    name = _pick(rng, _NAMES)
    kind = rng.randrange(3)
    outcomes = faces * faces
    if kind == 0:
        target = rng.randint(2, 2 * faces)
        ways = sum(1 for a in range(1, faces + 1) for b in range(1, faces + 1)
                   if a + b == target)
        question = f"the total is exactly {target}"
        detail = f"Outcomes summing to {target}: {ways}"
    elif kind == 1:
        target = rng.randint(2, 2 * faces)
        ways = sum(1 for a in range(1, faces + 1) for b in range(1, faces + 1)
                   if a + b >= target)
        question = f"the total is at least {target}"
        detail = f"Outcomes with a total of {target} or more: {ways}"
    else:
        target = rng.randint(0, faces - 1)
        ways = sum(1 for a in range(1, faces + 1) for b in range(1, faces + 1)
                   if abs(a - b) == target)
        question = f"the two dice differ by exactly {target}"
        detail = f"Outcomes with a difference of {target}: {ways}"
    if ways == 0:
        return None                               # a probability of zero teaches nothing here
    prob = Fraction(ways, outcomes)
    return (f"{name} rolls two fair {faces}-sided dice. What is the probability that "
            f"{question}?",
            f"Total outcomes: {faces} x {faces} = {outcomes}\n{detail}\n"
            f"Probability = {ways}/{outcomes} = {prob.numerator}/{prob.denominator}")


def _summary_statistics(rng: random.Random) -> Built:
    count = rng.randint(5, 11)
    values = sorted(rng.randint(1, 200) for _ in range(count))
    total = sum(values)
    mean = Fraction(total, count)
    mid = count // 2
    median = values[mid] if count % 2 else Fraction(values[mid - 1] + values[mid], 2)
    return (f"Find the mean and median of: {', '.join(map(str, values))}",
            f"Sorted: {', '.join(map(str, values))}\n"
            f"Sum = {total}, count = {count}\n"
            f"mean = {total}/{count} = "
            f"{mean if mean.denominator != 1 else mean.numerator}\n"
            f"The middle value{'s are averaged' if count % 2 == 0 else ' is taken'}: "
            f"median = {median}\n"
            f"mean = {mean if mean.denominator != 1 else mean.numerator}, median = {median}")


def _rectangle_geometry(rng: random.Random) -> Built:
    w, h = rng.randint(2, 90), rng.randint(2, 90)
    return (f"A rectangle measures {w} by {h}. Give its area, perimeter and diagonal squared.",
            f"area = {w} x {h} = {w * h}\n"
            f"perimeter = 2 x ({w} + {h}) = {2 * (w + h)}\n"
            f"diagonal^2 = {w}^2 + {h}^2 = {w * w} + {h * h} = {w * w + h * h}")


def _pythagorean_triple(rng: random.Random) -> Built:
    m = rng.randint(2, 30)
    n = rng.randint(1, m - 1)
    scale = rng.randint(1, 20)
    a, b, c = (m * m - n * n) * scale, 2 * m * n * scale, (m * m + n * n) * scale
    if rng.random() < 0.5:
        a, b = b, a                               # the legs are interchangeable; say so in data
    return (f"A right triangle has legs {a} and {b}. What is the hypotenuse?",
            f"By Pythagoras, c^2 = a^2 + b^2.\n"
            f"  {a}^2 = {a * a}\n  {b}^2 = {b * b}\n"
            f"  sum = {a * a + b * b}\n"
            f"  c = sqrt({a * a + b * b}) = {c}, since {c}^2 = {c * c}\n"
            f"hypotenuse = {c}")


def _determinant_2x2(rng: random.Random) -> Built:
    a, b, c, d = (rng.randint(-12, 12) for _ in range(4))
    det = a * d - b * c
    return (f"Find the determinant of [[{a}, {b}], [{c}, {d}]].",
            f"det = ad - bc\n  = ({a})({d}) - ({b})({c})\n"
            f"  = {a * d} - {b * c}\ndet = {det}")


def _polynomial_derivative(rng: random.Random) -> Built:
    terms = [(rng.randint(-12, 12), p) for p in sorted(
        rng.sample(range(1, 8), rng.randint(2, 4)), reverse=True)]
    terms = [(c, p) for c, p in terms if c]
    if not terms:
        return None
    poly = " ".join(f"{_signed(c)}x^{p}" for c, p in terms).lstrip("+ ")
    rows = [f"  d/dx[{c}x^{p}] = {c * p}x^{p - 1}" for c, p in terms]
    deriv = " ".join(f"{_signed(c * p)}x^{p - 1}" for c, p in terms).lstrip("+ ")
    return (f"Differentiate with respect to x: {poly}",
            "Apply the power rule d/dx[cx^n] = c n x^(n-1) term by term:\n"
            + "\n".join(rows) + f"\nd/dx = {deriv}")


def _polynomial_integral(rng: random.Random) -> Built:
    terms = [(rng.randint(-9, 9), p) for p in sorted(
        rng.sample(range(0, 6), rng.randint(1, 3)), reverse=True)]
    terms = [(c, p) for c, p in terms if c]
    if not terms:
        return None
    lo, hi = sorted(rng.sample(range(-6, 7), 2))
    poly = " ".join(f"{_signed(c)}x^{p}" for c, p in terms).lstrip("+ ")
    total = sum(Fraction(c, p + 1) * (Fraction(hi) ** (p + 1) - Fraction(lo) ** (p + 1))
                for c, p in terms)
    rows = [f"  integral of {c}x^{p} is {c}x^{p + 1}/{p + 1}" for c, p in terms]
    return (f"Evaluate the definite integral of {poly} from x = {lo} to x = {hi}.",
            "Integrate term by term, raising each power by one:\n" + "\n".join(rows)
            + f"\nEvaluate at {hi} and subtract the value at {lo}.\n"
            + f"integral = {total if total.denominator != 1 else total.numerator}")


def _sympy_derivative(rng: random.Random) -> Built:
    """Symbolic differentiation with the rule named.

    Broadened from the original, whose real state space was **240 documents** — a handful of
    coefficients times a handful of powers. Sums of terms, products and a chain-rule inner
    function widen it by four orders of magnitude.
    """
    try:
        import sympy as sp
    except Exception:  # noqa: BLE001
        return None
    x = sp.Symbol("x")
    shape = rng.randrange(3)
    if shape == 0:
        coeff, power = rng.randint(2, 30), rng.randint(2, 9)
        inner = x + rng.randint(-9, 9)
        expr, rule = coeff * inner ** power, "the power and chain rules"
    elif shape == 1:
        expr = sum(rng.randint(-15, 15) * x ** p for p in range(1, rng.randint(2, 6)))
        rule = "the power rule, term by term"
    else:
        expr = (rng.randint(2, 12) * x ** rng.randint(1, 4)) * (x + rng.randint(1, 9))
        rule = "the product rule"
    if expr == 0:
        return None
    derivative = sp.simplify(sp.diff(expr, x))
    return (f"Differentiate with respect to x: {sp.sstr(sp.expand(expr))}",
            f"Expression: {sp.sstr(sp.expand(expr))}\nApply {rule}.\n"
            f"d/dx = {sp.sstr(sp.expand(derivative))}")


# --------------------------------------------------------------------------- #
# The registry. `space` is an honest count of distinct reachable documents.
# --------------------------------------------------------------------------- #
def _g(name: str, fn: Callable[[random.Random], Built], space: int,
       weight: float = 1.0, proof_kind: str = "") -> Generator:
    return Generator(name=name, domain="math", fn=fn, space=space, weight=weight,
                     proof_kind=proof_kind)


GENERATORS: Tuple[Generator, ...] = (
    # arithmetic — the phrasing pool multiplies each space by len(_ASK_VALUE)
    _g("long_addition", _long_addition, 99_900 * 99_900 * 6, 1.4, "arithmetic"),
    _g("long_subtraction", _long_subtraction, 99_000 * 50_000 * 6, 1.2, "arithmetic"),
    _g("long_multiplication", _long_multiplication, 988 * 88 * 6, 1.2, "arithmetic"),
    _g("long_division", _long_division, 95 * 9989 * 97, 1.2, "arithmetic"),
    _g("gcd_euclid", _gcd_euclid, 8989 * 8989, 1.0, "number_theory"),
    _g("prime_factorisation", _prime_factorisation, 99_976, 1.0, "number_theory"),
    _g("base_conversion", _base_conversion, 65_516 * 3, 0.8),
    # fractions, percentages, rates
    _g("fraction_sum", _fraction_sum, 20 * 19 * 20 * 19 * 2, 1.1),
    _g("percentage", _percentage, 3981 * 12 * len(_NAMES) * len(_OBJECTS), 1.2),
    _g("compound_interest", _compound_interest, 198 * 8 * 5 * len(_NAMES), 0.9),
    _g("ratio_share", _ratio_share, 12 * 12 * 198 * len(_NAMES) ** 2, 1.0),
    _g("unit_conversion", _unit_conversion, 9998 * 10, 0.9),
    _g("speed_distance_time", _speed_distance_time, 116 * 11 * 3 * len(_NAMES), 1.0),
    # algebra
    _g("linear_equation", _linear_equation, 12 * 81 * 181 * 8 * 6, 1.5, "algebra"),
    _g("two_step_both_sides", _two_step_both_sides, 61 * 14 * 14 * 121 * 4, 1.2, "algebra"),
    _g("quadratic_from_roots", _quadratic_from_roots, 61 * 61 * 8, 1.2, "algebra"),
    _g("linear_system", _linear_system, 31 * 31 * 9 ** 4, 1.2),
    _g("expand_binomial", _expand_binomial, 9 * 25 * 9 * 25, 1.0, "algebra"),
    _g("arithmetic_sequence", _arithmetic_sequence, 101 * 24 * 55, 1.0),
    _g("geometric_sequence", _geometric_sequence, 400 * 8 * 10, 0.7),
    # number theory, counting, statistics, geometry
    _g("modular_arithmetic", _modular_arithmetic, 39 * 23 * 95, 0.9, "number_theory"),
    _g("combinations", _combinations, 41 * 7 * len(_OBJECTS) * len(_NAMES), 0.8),
    _g("permutations", _permutations, 37 * 6 * len(_OBJECTS) * len(_NAMES), 0.7),
    _g("dice_probability", _dice_probability, 7 * 40 * 3 * len(_NAMES), 0.8),
    _g("summary_statistics", _summary_statistics, 200 ** 5, 1.0),
    _g("rectangle_geometry", _rectangle_geometry, 89 * 89, 0.7),
    _g("pythagorean_triple", _pythagorean_triple, 29 * 29 * 20, 0.8),
    _g("determinant_2x2", _determinant_2x2, 25 ** 4, 0.8),
    # calculus
    _g("polynomial_derivative", _polynomial_derivative, 25 ** 4 * 35, 1.1),
    _g("polynomial_integral", _polynomial_integral, 19 ** 3 * 20 * 13 * 13, 1.0),
    _g("sympy_derivative", _sympy_derivative, 29 * 8 * 19 * 3 * 1000, 0.9),
)


def generators() -> Tuple[Generator, ...]:
    """Every maths generator. Kept a function so callers cannot mutate the tuple in place."""
    return GENERATORS
