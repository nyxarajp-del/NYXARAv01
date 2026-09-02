"""NYXARA · njp/mathematics.py — the mathematics she can actually *do* (∑, NJP V.23).

Measured before a line of this was written, on twenty-five questions from an ordinary school
syllabus, asked through :meth:`NJPBrain.think` on a brain with every organ built:

    right                              3 / 25
    silent                            17 / 25
    filed as a fact about the world    5 / 25

Reproducible now that the organ has a gate: build ``NJPBrain`` with ``mathematics_enabled =
False`` and ask the same twenty-five questions.

Three of the twenty-five are the same question in different clothes — ``24 + 18``, ``1/2 + 1/3``,
``20% of 250`` — and they are the three :mod:`nyxara.njp.calculate` was written for. That module
says of itself, in its own second paragraph, that it "does not do algebra, does not solve for
unknowns", and it was telling the truth: **a closed expression was the whole of her mathematics.**
Every gcd, every prime, every mean, every area, every derivative, every ``solve for x`` came back
as the empty string.

**The five that were not silent are the reason this module exists rather than being nice to
have.** They are imperatives, and nothing in the package read them as *tasks*, so the semantic
compiler read them as *assertions* and filed all five, at confidence 0.75 and source
``semantics``::

    ('simplify fraction', '18') → '24'
    ('expand', 'x')             → '2 x 3'
    ('factorise', 'x')          → '2 5x 6'
    ('convert', '5')            → 'km metres'
    ('solve', 'x')              → '2 5x 6 0'

Asking her to do arithmetic **wrote five sentences of nonsense into the knowledge store she
reasons from**, at the same confidence as a stated fact, where inheritance and the puzzle solver
would later walk over them. A silent failure costs a turn. That one costs the store. It is the
direction of failure this package keeps finding — a capability that can be exercised
and cannot be *asked* — pointed the other way for the first time: not knowledge that cannot be
reached, but a question that reaches the wrong organ and is written down.

**What this module is.** A mathematician, in the sense in which :mod:`nyxara.njp.calculate` is a
calculator: it reads a mathematics question in English or Hinglish, decides which of fifty
skills it is, applies that skill, and reports the working. It spans the school syllabus —
number theory, fractions, percentage and commerce, ratio, algebra, sequences, geometry and
mensuration, units, statistics, probability, powers and logarithms, elementary calculus, and the
word problems that dress any of them in a sentence.

**Exactness is carried, not recovered.** Every value inside is a :class:`fractions.Fraction`, so
``1/3`` is a third for as long as it is a third and becomes ``0.333…`` only when something is
asked for it in decimal. The one place exactness genuinely ends is a root or a π, and those are
reported as both — ``49π ≈ 153.938`` — because rounding π silently and calling it an area is the
same dishonesty :mod:`nyxara.njp.calculate` refuses when it says whether a value is exact.

**Nothing is ever `eval`\\ ed, here or downstream.** Algebra is parsed by :class:`Poly`'s own
tokeniser and recursive-descent parser into a polynomial over :class:`~fractions.Fraction`, which
can represent a sum of powers of one symbol and *cannot* represent a function call, a name, an
attribute or a subscript — the whitelist is the type, so there is nothing to widen. **sympy is
never called from this module at all**: exactness comes from :class:`~fractions.Fraction` and
roots from integer arithmetic, so every one of the fifty skills runs identically with it absent.
The one place it is reached is the shared :class:`~nyxara.njp.calculate.Calculator`, which uses it
to sharpen a closed expression to an exact rational; :meth:`Mathematician.stats` reports whether
it is installed and that is the whole of the dependency.

**Refusal is a first-class answer.** :meth:`Mathematician.solve` returns an empty
:class:`Solution` for anything it cannot close, which is what keeps it from becoming the failure
it was written to fix: a mathematician that answers "how are you" with a number is exactly the
organ that filed ``('convert', '5') → 'km metres'``. Every skill states its own trigger, and no
skill has a fallback.

Pure standard library. sympy optional and only ever an upgrade.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MathError",
    "Poly",
    "Solution",
    "Mathematician",
    "solve",
]


class MathError(ValueError):
    """A refusal with a reason attached. Never raised out of a public entry point."""


# --------------------------------------------------------------------------- #
# Numbers, said the way she should say them
# --------------------------------------------------------------------------- #

#: Above this an exponent stops being arithmetic and starts being a memory-exhaustion bug — the
#: same ceiling, for the same reason, as :data:`nyxara.njp.calculate._MAX_POW`.
_MAX_POW = 512
#: Past ~4300 digits Python's own ``int.__str__`` raises, so an answer this wide would fail at the
#: moment it was rendered rather than at the moment it was computed.
_MAX_DIGITS = 4300
#: A sieve above this is not a question, it is a denial of service written in three digits.
_MAX_SIEVE = 1_000_000


def _frac(value: Any) -> Fraction:
    """Anything numeric as an exact rational. A float becomes the rational it *displays* as.

    ``Fraction(0.1)`` is 3602879701896397/36028797018963968, which is the true value of the double
    and is never what somebody who typed ``0.1`` meant. Going through the string keeps the
    question the one that was asked.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(str(value).strip())


def _num(value: Any, *, places: int = 6) -> str:
    """A number as she should say it: whole where it is whole, a fraction where it is exact."""
    try:
        if isinstance(value, Fraction):
            if value.denominator == 1:
                return str(value.numerator)
            return f"{value.numerator}/{value.denominator}"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer() and abs(value) < 1e15:
                return str(int(value))
            text = f"{value:.{places}f}".rstrip("0").rstrip(".")
            return text or "0"
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def _dec(value: Fraction, *, places: int = 6) -> str:
    """The same rational written as a decimal, for the questions that ask in decimals."""
    return _num(float(value), places=places)


def _both(value: Fraction, *, places: int = 4) -> str:
    """``3/4`` where that is the whole story, ``3/4 = 0.75`` where the decimal is the answer.

    A fraction whose denominator divides a power of ten terminates, and for those two spellings
    of the same number one line is enough. For 1/3 the decimal is a *different, worse* claim and
    is offered beside the exact one rather than instead of it.
    """
    if value.denominator == 1:
        return str(value.numerator)
    return f"{_num(value)} = {_dec(value, places=places)}"


def _pi_text(coefficient: Fraction, *, places: int = 4) -> str:
    """``49π ≈ 153.9380``. Both halves, always — see the module docstring on exactness."""
    exact = "π" if coefficient == 1 else f"{_num(coefficient)}π"
    return f"{exact} ≈ {_num(float(coefficient) * math.pi, places=places)}"


def _root_text(value: Fraction, index: int = 2, *, places: int = 6) -> Tuple[str, bool]:
    """The root of ``value``, exact where it is exact, and honest about it where it is not."""
    if value < 0 and index % 2 == 0:
        raise MathError(f"a negative number has no real {'square' if index == 2 else 'even'} root")
    exact = _exact_root(value, index)
    if exact is not None:
        return _num(exact), True
    magnitude = float(abs(value)) ** (1.0 / index)
    approx = -magnitude if value < 0 else magnitude
    sign = "√" if index == 2 else f"{index}√"
    return f"{sign}{_num(value)} ≈ {_num(approx, places=places)}", False


def _exact_root(value: Fraction, index: int = 2) -> Optional[Fraction]:
    """``Fraction(4)`` under a square root is 2 exactly; ``Fraction(5)`` is not a rational at all."""
    try:
        negative = value < 0
        if negative and index % 2 == 0:
            return None
        top = _integer_root(abs(value.numerator), index)
        bottom = _integer_root(abs(value.denominator), index)
        if top is None or bottom is None:
            return None
        result = Fraction(top, bottom)
        return -result if negative else result
    except Exception:  # noqa: BLE001
        return None


def _integer_root(n: int, index: int) -> Optional[int]:
    """The exact integer ``index``-th root of ``n``, or ``None``. Integer-only, so never wrong."""
    if n < 0:
        return None
    if n in (0, 1):
        return n
    low, high = 1, 1 << ((n.bit_length() + index - 1) // index + 1)
    while low <= high:
        mid = (low + high) // 2
        power = mid ** index
        if power == n:
            return mid
        if power < n:
            low = mid + 1
        else:
            high = mid - 1
    return None


# --------------------------------------------------------------------------- #
# Poly — algebra without an evaluator
# --------------------------------------------------------------------------- #
#
# The whole of the algebra half rests on this type, and the reason it is a type rather than a call
# into sympy is the security note in the module docstring: a polynomial over one symbol is a
# `Dict[int, Fraction]`, and there is no arrangement of that dict that names a function, imports a
# module or reads an attribute. The parser below can only ever *produce* one of these, so the
# whitelist is the representation and cannot be widened by a Python release.

_TOKEN = re.compile(r"\s*(\d+\.\d+|\d+|[A-Za-z]+|\*\*|[-+*/^()])")


class Poly:
    """A polynomial in one symbol with exact rational coefficients.

    Immutable in practice — every operation returns a new one — and canonical: a zero coefficient
    is dropped on construction, so ``x^2 - x^2`` and ``0`` are the same object by ``==`` and print
    the same way. That matters more than it sounds: half the algebra skills below decide what to
    do by looking at :attr:`degree`, and a stored zero would make a linear equation look quadratic
    and be solved by a formula that divides by its leading coefficient.
    """

    __slots__ = ("coefficients", "symbol")

    def __init__(self, coefficients: Optional[Dict[int, Any]] = None, symbol: str = "x") -> None:
        cleaned: Dict[int, Fraction] = {}
        for power, coefficient in (coefficients or {}).items():
            power = int(power)
            if power < 0:
                raise MathError("a negative power is not a polynomial")
            value = _frac(coefficient)
            if value:
                cleaned[power] = cleaned.get(power, Fraction(0)) + value
        self.coefficients = {p: c for p, c in cleaned.items() if c}
        self.symbol = symbol or "x"

    # -- construction -------------------------------------------------------- #
    @classmethod
    def constant(cls, value: Any, symbol: str = "x") -> "Poly":
        return cls({0: _frac(value)}, symbol)

    @classmethod
    def variable(cls, symbol: str = "x") -> "Poly":
        return cls({1: Fraction(1)}, symbol)

    # -- shape --------------------------------------------------------------- #
    @property
    def degree(self) -> int:
        return max(self.coefficients) if self.coefficients else 0

    @property
    def is_constant(self) -> bool:
        return not self.coefficients or set(self.coefficients) == {0}

    @property
    def is_zero(self) -> bool:
        return not self.coefficients

    def coefficient(self, power: int) -> Fraction:
        return self.coefficients.get(int(power), Fraction(0))

    def value(self) -> Fraction:
        if not self.is_constant:
            raise MathError("that still has an unknown in it")
        return self.coefficient(0)

    # -- arithmetic ---------------------------------------------------------- #
    def __add__(self, other: Any) -> "Poly":
        other = self._lift(other)
        merged = dict(self.coefficients)
        for power, coefficient in other.coefficients.items():
            merged[power] = merged.get(power, Fraction(0)) + coefficient
        return Poly(merged, self._symbol_with(other))

    def __sub__(self, other: Any) -> "Poly":
        return self + (self._lift(other) * Poly.constant(-1, self.symbol))

    def __mul__(self, other: Any) -> "Poly":
        other = self._lift(other)
        product: Dict[int, Fraction] = {}
        for p1, c1 in self.coefficients.items():
            for p2, c2 in other.coefficients.items():
                if p1 + p2 > 64:
                    raise MathError("that polynomial is too large to expand")
                product[p1 + p2] = product.get(p1 + p2, Fraction(0)) + c1 * c2
        return Poly(product, self._symbol_with(other))

    def __truediv__(self, other: Any) -> "Poly":
        """Division, and only where it is exact.

        A polynomial divided by a non-constant is in general a *rational function*, which this
        type cannot represent. Returning a truncated quotient would be a wrong answer stated in
        the right shape, so a remainder is a refusal.
        """
        other = self._lift(other)
        if other.is_zero:
            raise MathError("division by zero")
        if other.is_constant:
            divisor = other.value()
            return Poly({p: c / divisor for p, c in self.coefficients.items()},
                        self._symbol_with(other))
        quotient, remainder = self.divmod(other)
        if not remainder.is_zero:
            raise MathError("that division does not come out exactly")
        return quotient

    def __pow__(self, exponent: int) -> "Poly":
        exponent = int(exponent)
        if exponent < 0:
            raise MathError("a negative power is not a polynomial")
        if exponent > _MAX_POW:
            raise MathError(f"exponent {exponent} is above the {_MAX_POW} ceiling")
        result = Poly.constant(1, self.symbol)
        for _ in range(exponent):
            result = result * self
        return result

    def __neg__(self) -> "Poly":
        return self * Poly.constant(-1, self.symbol)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Poly):
            return NotImplemented
        return self.coefficients == other.coefficients

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.coefficients.items())))

    def divmod(self, other: "Poly") -> Tuple["Poly", "Poly"]:
        """Long division. Exact throughout, because the coefficients are rationals."""
        if other.is_zero:
            raise MathError("division by zero")
        remainder, quotient = Poly(dict(self.coefficients), self.symbol), Poly({}, self.symbol)
        divisor_power, divisor_lead = other.degree, other.coefficient(other.degree)
        while not remainder.is_zero and remainder.degree >= divisor_power:
            shift = remainder.degree - divisor_power
            scale = remainder.coefficient(remainder.degree) / divisor_lead
            step = Poly({shift: scale}, self.symbol)
            quotient = quotient + step
            remainder = remainder - step * other
        return quotient, remainder

    def _lift(self, other: Any) -> "Poly":
        if isinstance(other, Poly):
            return other
        return Poly.constant(other, self.symbol)

    def _symbol_with(self, other: "Poly") -> str:
        """Which unknown the result is about.

        A constant has no unknown, so ``2 * y`` is about ``y`` however the 2 was spelled — without
        this, the parser's first token fixes the symbol before the expression has said what it is
        about, and ``2y^2 - 3y + 1`` refuses itself as "two different unknowns, x and y". Two
        genuinely different letters is still a refusal, because this type holds one symbol and
        answering about the wrong one silently is worse than declining.
        """
        if self.is_constant:
            return other.symbol
        if other.is_constant or other.symbol == self.symbol:
            return self.symbol
        raise MathError(f"two different unknowns, {self.symbol} and {other.symbol}")

    # -- calculus and evaluation --------------------------------------------- #
    def at(self, value: Any) -> Fraction:
        """Evaluate. Horner's rule, so a degree-20 polynomial is 20 multiplications, not 210."""
        point, total = _frac(value), Fraction(0)
        for power in range(self.degree, -1, -1):
            total = total * point + self.coefficient(power)
        return total

    def derivative(self) -> "Poly":
        return Poly({p - 1: c * p for p, c in self.coefficients.items() if p > 0}, self.symbol)

    def integral(self) -> "Poly":
        """The antiderivative with constant zero. The ``+ C`` is the caller's to say out loud."""
        return Poly({p + 1: c / (p + 1) for p, c in self.coefficients.items()}, self.symbol)

    # -- roots --------------------------------------------------------------- #
    def rational_roots(self) -> List[Fraction]:
        """Every rational root, exactly, by the rational-root theorem then division.

        Degree 1 and 2 are closed forms; above that the candidates are ``±p/q`` over the divisors
        of the constant and leading terms. That finds *rational* roots only, and a cubic with one
        rational and two irrational roots reports the one it can prove — which is the honest half
        rather than a numerical approximation of the other two dressed as exact values.
        """
        if self.is_zero or self.is_constant:
            return []
        scaled = self._integer_coefficients()
        roots: List[Fraction] = []
        working = scaled
        while working.degree >= 1:
            if working.degree == 1:
                root = -working.coefficient(0) / working.coefficient(1)
                roots.append(root)
                break
            if working.degree == 2:
                found = _quadratic_rational_roots(working)
                if found is None:
                    break
                roots.extend(found)
                break
            root = _first_rational_root(working)
            if root is None:
                break
            roots.append(root)
            working, _ = working.divmod(Poly({1: Fraction(1), 0: -root}, self.symbol))
        seen, unique = set(), []
        for root in roots:
            if root not in seen:
                seen.add(root)
                unique.append(root)
        return sorted(unique)

    def _integer_coefficients(self) -> "Poly":
        """The same polynomial scaled so every coefficient is an integer. Roots are unchanged."""
        denominators = [c.denominator for c in self.coefficients.values()] or [1]
        multiplier = 1
        for denominator in denominators:
            multiplier = multiplier * denominator // math.gcd(multiplier, denominator)
        return Poly({p: c * multiplier for p, c in self.coefficients.items()}, self.symbol)

    # -- rendering ----------------------------------------------------------- #
    def text(self) -> str:
        """``x^2 + 5x + 6`` — highest power first, signs joined, no ``1x`` and no ``+ -3``."""
        if self.is_zero:
            return "0"
        pieces: List[str] = []
        for power in sorted(self.coefficients, reverse=True):
            coefficient = self.coefficients[power]
            sign = "-" if coefficient < 0 else "+"
            magnitude = abs(coefficient)
            if power == 0:
                body = _num(magnitude)
            else:
                tail = self.symbol if power == 1 else f"{self.symbol}^{power}"
                if magnitude.denominator == 1:
                    head = "" if magnitude == 1 else _num(magnitude)
                    body = f"{head}{tail}"
                else:
                    # A fraction written *in front of* the symbol is ambiguous — "1/3x^3" reads as
                    # 1/(3x^3) as easily as it reads as (1/3)x^3 — so the denominator goes after
                    # the whole term, which is how it is written by hand.
                    head = "" if magnitude.numerator == 1 else str(magnitude.numerator)
                    body = f"{head}{tail}/{magnitude.denominator}"
            pieces.append(f"{sign} {body}" if pieces else (f"-{body}" if sign == "-" else body))
        return " ".join(pieces)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"Poly({self.text()!r})"

    # -- parsing ------------------------------------------------------------- #
    @classmethod
    def parse(cls, text: str, symbol: Optional[str] = None) -> "Poly":
        """Read ``(x+2)(x+3)``, ``2x^2 - 3x + 1`` or ``5`` into a polynomial.

        Implicit multiplication is accepted in all four places English writes it — ``2x``,
        ``x(x+1)``, ``(x+1)(x+2)`` and ``(x+1)x`` — because every one of them appears in an
        ordinary textbook and none of them is valid Python, which is precisely why this parser
        exists instead of :func:`ast.parse`.
        """
        parser = _Parser(str(text or ""), symbol)
        return parser.parse()


def _quadratic_rational_roots(poly: Poly) -> Optional[List[Fraction]]:
    """The two roots of a quadratic, when both are rational. ``None`` when they are not.

    ``None`` is not "no roots": ``x^2 - 2`` has two perfectly good roots and neither of them is a
    number this type can hold. The caller that wants a decimal asks for one; the caller that
    wanted to *factorise over the rationals* is being told correctly that it cannot.
    """
    a, b, c = poly.coefficient(2), poly.coefficient(1), poly.coefficient(0)
    if not a:
        return None
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    root = _exact_root(discriminant, 2)
    if root is None:
        return None
    return sorted({(-b + root) / (2 * a), (-b - root) / (2 * a)})


def _divisors(n: int) -> List[int]:
    n, out, step = abs(int(n)), [], 1
    if n == 0:
        return [1]
    while step * step <= n:
        if n % step == 0:
            out.append(step)
            out.append(n // step)
        step += 1
    return sorted(set(out))


def _first_rational_root(poly: Poly) -> Optional[Fraction]:
    """One rational root by the rational-root theorem, or ``None``. Integer coefficients assumed."""
    constant = poly.coefficient(0)
    lead = poly.coefficient(poly.degree)
    if not lead:
        return None
    if not constant:
        return Fraction(0)
    if abs(constant.numerator) > 10 ** 9 or abs(lead.numerator) > 10 ** 9:
        return None
    for p in _divisors(int(constant)):
        for q in _divisors(int(lead)):
            for candidate in (Fraction(p, q), Fraction(-p, q)):
                if poly.at(candidate) == 0:
                    return candidate
    return None


class _Parser:
    """Recursive descent over the four levels English algebra actually uses.

    ``expression → term (('+'|'-') term)*`` · ``term → power (('*'|'/'|juxtaposition) power)*`` ·
    ``power → atom ('^' integer)?`` · ``atom → number | symbol | '(' expression ')' | '-' atom``.

    The symbol is *discovered* rather than configured: the first letter run that is not a known
    function word becomes the unknown, so ``2y + 6 = 0`` is solved for ``y`` without anybody
    telling it that this question is about ``y``. Two different letters in one expression is a
    refusal, because this type holds one symbol and pretending otherwise would silently answer a
    different question.
    """

    #: Letter runs that are part of the arithmetic rather than the unknown.
    _WORDS = {"pi", "e"}

    def __init__(self, text: str, symbol: Optional[str] = None) -> None:
        self.tokens = self._tokenise(text)
        self.position = 0
        self.symbol = symbol

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        cleaned = str(text or "").replace("−", "-").replace("–", "-").replace("×", "*")
        cleaned = cleaned.replace("÷", "/").replace("**", "^")
        tokens, at = [], 0
        while at < len(cleaned):
            match = _TOKEN.match(cleaned, at)
            if match is None:
                if cleaned[at].isspace():
                    at += 1
                    continue
                raise MathError(f"{cleaned[at]!r} does not belong in an expression")
            tokens.append(match.group(1))
            at = match.end()
        return tokens

    # -- the grammar ---------------------------------------------------------- #
    def parse(self) -> Poly:
        if not self.tokens:
            raise MathError("there is no expression there")
        result = self._expression()
        if self.position < len(self.tokens):
            raise MathError(f"{self.tokens[self.position]!r} is left over")
        return result

    def _expression(self) -> Poly:
        result = self._term()
        while self._peek() in ("+", "-"):
            operator = self._take()
            right = self._term()
            result = result + right if operator == "+" else result - right
        return result

    def _term(self) -> Poly:
        result = self._power()
        while True:
            token = self._peek()
            if token in ("*", "/"):
                self._take()
                right = self._power()
                result = result * right if token == "*" else result / right
            elif token is not None and (token == "(" or self._is_number(token)
                                        or self._is_letters(token)):
                # Juxtaposition: `2x`, `x(x+1)`, `(x+1)(x+2)`, `(x+1)x`.
                result = result * self._power()
            else:
                return result

    def _power(self) -> Poly:
        base = self._atom()
        if self._peek() == "^":
            self._take()
            exponent = self._atom()
            if not exponent.is_constant or exponent.value().denominator != 1:
                raise MathError("a power has to be a whole number here")
            return base ** int(exponent.value())
        return base

    def _atom(self) -> Poly:
        token = self._take()
        if token is None:
            raise MathError("the expression stops in the middle")
        if token == "-":
            return -self._atom()
        if token == "+":
            return self._atom()
        if token == "(":
            inner = self._expression()
            if self._take() != ")":
                raise MathError("a bracket is never closed")
            return inner
        if self._is_number(token):
            return Poly.constant(_frac(token), self.symbol or "x")
        if self._is_letters(token):
            low = token.lower()
            if low == "pi":
                raise MathError("π is not a rational coefficient")
            # **A word is not an unknown.** Without this an ordinary English sentence parses as
            # algebra in several unknowns: "a factor is a number that divides another number"
            # reached the factoriser and came back "two different unknowns, is and a". The
            # sentence was then a *recognised* maths task, so the brain declined to learn the
            # definition it was being taught — a lesson silently discarded by the reader that was
            # supposed to be helping. An unknown is one letter, which is what an unknown is.
            if len(low) > 1:
                raise MathError(f"{token!r} is a word, not an unknown")
            if self.symbol is None:
                self.symbol = low
            if low != self.symbol:
                raise MathError(f"two different unknowns, {self.symbol} and {low}")
            return Poly.variable(self.symbol)
        raise MathError(f"{token!r} does not belong in an expression")

    # -- token helpers -------------------------------------------------------- #
    def _peek(self) -> Optional[str]:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self) -> Optional[str]:
        token = self._peek()
        if token is not None:
            self.position += 1
        return token

    @staticmethod
    def _is_number(token: str) -> bool:
        return bool(token) and (token[0].isdigit() or token[0] == ".")

    @staticmethod
    def _is_letters(token: str) -> bool:
        return bool(token) and token[0].isalpha()


# --------------------------------------------------------------------------- #
# What one solved question looks like
# --------------------------------------------------------------------------- #

@dataclass
class Solution:
    """One worked question: the answer, the topic it belonged to, and the working.

    ``steps`` is not decoration and is not a log. It is the answer to "how do you know", and every
    skill below fills it, because an organ whose output the gauntlet downstream treats as
    verifiable has to be able to show the derivation when asked. A :class:`Solution` with steps
    and no answer is a refusal that got some distance; a :class:`Solution` with neither is a
    question this module declined to recognise, which is the commonest and most important case.
    """

    question: str = ""
    topic: str = ""
    method: str = ""
    answer: str = ""
    value: Any = None
    exact: bool = False
    steps: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """Did this produce something she may state?"""
        return bool(self.answer) and not self.error

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "topic": self.topic, "method": self.method,
                "answer": self.answer, "exact": self.exact, "steps": self.steps[:12],
                "error": self.error, "ok": self.ok}

    def __str__(self) -> str:
        return self.answer if self.ok else (self.error or "")


# --------------------------------------------------------------------------- #
# Reading the question
# --------------------------------------------------------------------------- #

# Hinglish and Hindi rewritten to the one vocabulary the skills below match on. This is the same
# decision `calculate._REWRITES` makes and for the same reason: one grammar reaches the solver, so
# a skill cannot be reachable in English and unreachable in Hinglish by an oversight nobody sees.
_HINGLISH: Tuple[Tuple[str, str], ...] = (
    (r"\b(?:hal|hall)\s+kar(?:o|iye|na)\b", " solve "),
    (r"\b(?:nikaalo|nikalo|nikaliye|pata\s+karo|maalum\s+karo)\b", " find "),
    (r"\b(?:batao|bataao|bataiye|btao|bata)\b", " "),
    (r"\b(?:kitna|kitne|kitni)\s+(?:hai|hoga|hoti|hota|honge|hain)\b", " "),
    (r"\b(?:kya|kyaa)\s+(?:hai|hoga)\b", " "),
    (r"\bvargmul\b", " square root "),
    (r"\bghanmul\b", " cube root "),
    (r"\bvarg\b", " square "),
    (r"\bghan\b", " cube "),
    (r"\b(?:abhajya|abhaajya)\b", " prime "),
    (r"\b(?:gunankhand|gunanakhand)\b", " factor "),
    (r"\b(?:masavi|mahattam\s+samapavartak)\b", " hcf "),
    (r"\b(?:lasavi|laghuttam\s+samapavartya)\b", " lcm "),
    (r"\b(?:ausat|madhyaman)\b", " average "),
    (r"\bmadhyika\b", " median "),
    (r"\bbahulak\b", " mode "),
    (r"\b(?:kshetrafal|chetrafal)\b", " area "),
    (r"\bparimap\b", " perimeter "),
    (r"\b(?:aayatan|ayatan)\b", " volume "),
    (r"\b(?:pratishat|prtishat)\b", " percent "),
    (r"\b(?:sadharan\s+byaj|saral\s+byaj)\b", " simple interest "),
    (r"\b(?:chakravriddhi\s+byaj|chakravrddhi\s+byaj)\b", " compound interest "),
    (r"\bbyaj\b", " interest "),
    (r"\b(?:mooldhan|muldhan)\b", " principal "),
    (r"\b(?:chaal|gati)\b", " speed "),
    (r"\b(?:doori|duri)\b", " distance "),
    (r"\bsamay\b", " time "),
    (r"\banupat\b", " ratio "),
    (r"\bsankhya\b", " number "),
    (r"\b(?:tribhuj|trikon)\b", " triangle "),
    (r"\bvritt\b", " circle "),
    (r"\b(?:aayat|ayat)\b", " rectangle "),
    (r"\b(?:vargakar|varg\s*akar)\b", " square "),
    (r"\btrijya\b", " radius "),
    (r"\bvyas\b", " diameter "),
    (r"\b(?:aadhar|adhar)\b", " base "),
    (r"\b(?:unchai|uncha[ai]|oonchai)\b", " height "),
    (r"\b(?:lambai|lambaai)\b", " length "),
    (r"\b(?:chaudai|chaudaai|chaurai)\b", " width "),
)

#: Digits, decimals and the fraction forms she is asked in. Ordered longest-first so ``3/4`` is one
#: number rather than a 3 and a 4 — a mean over "1/2, 1/4" is otherwise an average of four numbers.
_NUMBER = re.compile(r"(?<![\w.])(-?\d+\s*/\s*\d+|-?\d+\.\d+|-?\d+)(?![\w.])")

#: Number words, because "half of forty" is a question and "0.5 of 40" is the same question typed
#: by somebody who already did the reading.
_WORD_NUMBERS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "lakh": "100000", "crore": "10000000",
    "ek": "1", "do": "2", "teen": "3", "char": "4", "paanch": "5", "panch": "5", "chhe": "6",
    "saat": "7", "aath": "8", "nau": "9", "das": "10", "sau": "100", "hazaar": "1000",
}


def normalise(text: str) -> str:
    """The question in the one vocabulary every skill below matches against."""
    low = " ".join(str(text or "").strip().lower().split())
    low = low.replace("−", "-").replace("–", "-").replace("—", "-")
    low = low.replace("×", "*").replace("✕", "*").replace("✖", "*").replace("÷", "/")
    for pattern, replacement in _HINGLISH:
        low = re.sub(pattern, replacement, low)
    # **Number words are read only in a sentence that has no digits in it.** Every one of them is
    # also an ordinary word in one of the two languages this reads — "ek" is Hindi's indefinite
    # article, "do" is an English verb, "one" is an English pronoun — and substituting inside a
    # sentence that already counts in digits invents a number that is not in the question. Measured:
    # "91 ek prime number hai?" became "91 1 prime number hai" and was answered about **1**, and
    # "which one is bigger, 3/4 or 5/8?" grew a third number to compare. A question spelled in
    # words has no digits to be confused with, which is exactly the case this is for.
    if not re.search(r"\d", low):
        for word, digits in _WORD_NUMBERS.items():
            low = re.sub(rf"\b{word}\b", digits, low)
    # `?` is never mathematics; `!` is, when a number is sitting in front of it. Stripping both
    # unconditionally — as the first version did — deleted the factorial operator out of "5!" and
    # made the question unanswerable by the skill written for it.
    low = re.sub(r"\?+", " ", low)
    low = re.sub(r"(?<!\d)!+", " ", low)
    return " ".join(low.split())


def numbers_in(text: str) -> List[Fraction]:
    """Every number in the sentence, exactly, in the order they were written."""
    out: List[Fraction] = []
    for raw in _NUMBER.findall(str(text or "")):
        try:
            out.append(_frac(raw.replace(" ", "")))
        except Exception:  # noqa: BLE001 — a token that does not parse is not a number
            continue
    return out


# --------------------------------------------------------------------------- #
# The mathematician
# --------------------------------------------------------------------------- #

class Mathematician:
    """Reads a mathematics question, decides which skill it is, and works it out.

    Stateless apart from three counters, so it is safe to share between turns and between brains.
    The counters exist for the reason :class:`nyxara.njp.calculate.Calculator`'s do: the defect
    this module was written for was not a wrong answer, it was an organ nothing reached, and
    ``asked`` staying at zero across a session is the only way that shows up before somebody
    notices the silence by hand.

    **The dispatch order is part of the contract.** Skills are tried in the order
    :attr:`SKILLS` lists them and the first that both *matches* and *closes* wins. Specific
    triggers come before general ones — ``"the square root of 144"`` must not be read as the
    arithmetic ``144`` — and closed arithmetic is deliberately last, because it is the only skill
    whose trigger is a shape rather than a word and would otherwise swallow the numbers out of a
    geometry question.
    """

    #: ``(topic, method name)``, tried in this order. See the note on ordering above.
    SKILLS: Tuple[Tuple[str, str], ...] = (
        # number theory — a word each, so none of them can be reached by accident
        ("number", "_prime_test"),
        ("number", "_prime_factors"),
        ("number", "_factors"),
        ("number", "_gcd"),
        ("number", "_lcm"),
        ("number", "_divisible"),
        ("number", "_parity"),
        ("number", "_perfect"),
        ("number", "_factorial"),
        # powers, roots and logarithms
        ("power", "_root"),
        ("power", "_power"),
        ("power", "_logarithm"),
        # fractions and decimals
        ("fraction", "_fraction_simplify"),
        ("fraction", "_fraction_convert"),
        ("fraction", "_fraction_compare"),
        # percentage and the commerce that is percentage in a sentence
        ("percent", "_percent_change"),
        ("percent", "_percent_whole"),
        ("percent", "_percent_which"),
        ("commerce", "_profit_loss"),
        ("commerce", "_discount"),
        ("commerce", "_simple_interest"),
        ("commerce", "_compound_interest"),
        ("percent", "_percent_of"),
        # ratio and proportion
        ("ratio", "_ratio_divide"),
        ("ratio", "_ratio_simplify"),
        ("ratio", "_proportion"),
        # algebra
        ("algebra", "_evaluate_at"),
        ("algebra", "_simultaneous"),
        ("algebra", "_solve_equation"),
        ("algebra", "_expand"),
        ("algebra", "_factorise"),
        ("algebra", "_simplify_expression"),
        # sequences and series
        ("sequence", "_series_sum"),
        ("sequence", "_sequence_nth"),
        ("sequence", "_sequence_next"),
        # geometry and mensuration
        ("geometry", "_pythagoras"),
        ("geometry", "_angles"),
        ("geometry", "_area"),
        ("geometry", "_perimeter"),
        ("geometry", "_volume"),
        ("geometry", "_surface_area"),
        # measurement
        ("units", "_convert"),
        # statistics and probability
        ("statistics", "_statistic"),
        ("probability", "_probability"),
        # calculus
        ("calculus", "_derivative"),
        ("calculus", "_integral"),
        ("calculus", "_limit"),
        # word problems that are a formula wearing a sentence
        ("word", "_speed"),
        ("word", "_work_rate"),
        # last, and last on purpose — see the class docstring
        ("arithmetic", "_arithmetic"),
    )

    def __init__(self, *, calculator: Any = None, prefer_exact: bool = True) -> None:
        self.prefer_exact = bool(prefer_exact)
        self._calculator = calculator
        self.asked = 0
        self.solved = 0
        self.declined = 0
        self._topics: Dict[str, int] = {}
        #: The last question and what it came to. A turn asks twice — once to decide whether the
        #: sentence is a task at all (so nothing is filed as a fact about the world) and once to
        #: answer it — and running fifty triggers twice for one sentence is the same work done
        #: for no reason. One entry deep on purpose: this is a memo, not a cache of answers, and
        #: a mathematician that remembered its answers would be a store of facts about arithmetic.
        self._last: Optional[Tuple[str, Solution]] = None

    # -- the public call ------------------------------------------------------ #
    def solve(self, text: str) -> Solution:
        """Work out what this sentence asks, or come back empty-handed.

        Empty-handed is the important half of the contract, exactly as it is for
        :func:`nyxara.njp.calculate.expression_in`. A question with no mathematics in it must
        return a :class:`Solution` that is not ``ok``, so the turn goes on to an organ that might
        know — and so that an imperative like "expand your answer" is never mistaken for algebra.
        """
        question = str(text or "").strip()
        if not question:
            return Solution(error="there is no question there")
        if self._last is not None and self._last[0] == question:
            return self._last[1]
        low = normalise(question)
        self.asked += 1
        for topic, name in self.SKILLS:
            try:
                skill = getattr(self, name)
                found = skill(low)
            except MathError as exc:
                # A skill that *recognised* the question and then could not do it is a refusal
                # with a reason, and the reason is more useful than trying the next skill — which
                # would, by construction, be a skill that does not understand the question.
                self.declined += 1
                refusal = Solution(question=question, topic=topic, method=name.strip("_"),
                                   error=str(exc))
                self._last = (question, refusal)
                return refusal
            except Exception:  # noqa: BLE001 — a broken skill must not take the turn down
                continue
            if found is not None and found.ok:
                found.question = question
                found.topic = found.topic or topic
                found.method = found.method or name.strip("_")
                self.solved += 1
                self._topics[found.topic] = self._topics.get(found.topic, 0) + 1
                self._last = (question, found)
                return found
        self.declined += 1
        refusal = Solution(question=question,
                           error="that is not a mathematics question I can close")
        self._last = (question, refusal)
        return refusal

    def stats(self) -> Dict[str, Any]:
        return {"asked": self.asked, "solved": self.solved, "declined": self.declined,
                "topics": dict(sorted(self._topics.items())), "skills": len(self.SKILLS),
                "exact_backend": _has_sympy()}

    # -- shared helpers ------------------------------------------------------- #
    @staticmethod
    def _answer(topic: str, method: str, answer: str, steps: Sequence[str],
                *, value: Any = None, exact: bool = True) -> Solution:
        return Solution(topic=topic, method=method, answer=str(answer), value=value,
                        exact=bool(exact), steps=[str(s) for s in steps])

    @staticmethod
    def _ints(values: Sequence[Fraction], *, what: str = "a whole number") -> List[int]:
        """The same values as integers, refusing rather than rounding."""
        out = []
        for value in values:
            if value.denominator != 1:
                raise MathError(f"{_num(value)} is not {what}")
            out.append(int(value))
        return out

    def _calculate(self, text: str) -> Any:
        """The shared arithmetic calculator, built on first use and never required."""
        if self._calculator is None:
            try:
                from nyxara.njp.calculate import Calculator
                self._calculator = Calculator(prefer_exact=self.prefer_exact)
            except Exception:  # noqa: BLE001
                return None
        return self._calculator

    # ---- number theory ------------------------------------------------------ #
    #
    # Every trigger here names an operation in words. That is deliberate: a number-theory question
    # is a question *about* a number rather than an expression containing one, so there is nothing
    # in its shape to recognise it by, and a trigger loose enough to catch "84" would catch every
    # sentence in the language that mentions a quantity.

    # Each of these needs a copula or an imperative, never the bare adjacency of a number and a
    # word. "the sum of the first 5 prime numbers" contains "5 prime" and is not a question about
    # whether 5 is prime — measured, not imagined: the first version of this trigger answered that
    # question with "yes, 5 is a prime number" and never reached the series skill at all.
    _RE_PRIME = re.compile(
        r"\bis\s+(-?\d+)\s+(?:an?\s+)?prime\b"
        r"|\b(?:check|test)\s+(?:if|whether)\s+(-?\d+)\s+is\s+prime\b"
        r"|\b(-?\d+)\s+(?:ek\s+)?(?:a\s+)?prime\s+(?:number\s+)?(?:hai|h)\b")
    _RE_PRIME_FACTORS = re.compile(
        r"\bprime\s+(?:factor(?:s|isation|ization)?|factorise|factorize)\b[^\d]{0,20}(\d+)")
    _RE_FACTORS = re.compile(
        r"\b(?:how\s+many\s+)?(factors|divisors)\s+(?:of|does)\s+(\d+)|"
        r"\b(\d+)\s+(?:ke|ka)\s+(factors|divisors)\b")
    _RE_GCD = re.compile(r"\b(?:gcd|hcf|highest\s+common\s+factor|greatest\s+common\s+(?:factor|divisor))\b")
    _RE_LCM = re.compile(r"\b(?:lcm|l\.c\.m|lowest\s+common\s+multiple|least\s+common\s+multiple)\b")
    _RE_DIVISIBLE = re.compile(r"\b(-?\d+)\s+(?:is\s+)?divisible\s+by\s+(-?\d+)")
    _RE_PARITY = re.compile(
        r"\bis\s+(-?\d+)\s+(?:an?\s+)?(even|odd)\b"
        r"|\b(-?\d+)\s+(even|odd)\s+(?:hai|h)\b")
    _RE_PERFECT = re.compile(
        r"\bis\s+(-?\d+)\s+(?:a\s+)?perfect\s+(square|cube)\b"
        r"|\b(-?\d+)\s+(?:ek\s+)?perfect\s+(square|cube)\s+(?:hai|h)\b"
        r"|\bperfect\s+(square|cube)\s+(?:hai|h)\b[^\d]{0,4}(-?\d+)")
    _RE_FACTORIAL = re.compile(r"\b(\d+)\s*!|\bfactorial\s+of\s+(\d+)\b|\b(\d+)\s+factorial\b")

    def _prime_test(self, low: str) -> Optional[Solution]:
        """Prime, and *why* — a composite is answered with the divisor that settles it.

        "91 is not prime" is a claim she should be able to defend, and the defence is one number:
        7. Reporting the witness rather than the verdict alone is the same discipline the truth
        gauntlet applies to everything else she asserts.
        """
        if "prime factor" in low or "prime factorisation" in low or "prime factorization" in low:
            return None
        match = self._RE_PRIME.search(low)
        if match is None:
            return None
        raw = match.group(1) or match.group(2) or match.group(3)
        if raw is None:
            return None
        n = int(raw)
        divisor = _smallest_divisor(n)
        if divisor is None:
            return self._answer("number", "prime_test", f"yes, {n} is a prime number",
                                [f"{n} has no divisor from 2 up to √{n} ≈ "
                                 f"{_num(math.isqrt(abs(n)))}"], value=True)
        if n < 2:
            return self._answer("number", "prime_test", f"no, {n} is not a prime number",
                                ["a prime is a whole number greater than 1"], value=False)
        return self._answer("number", "prime_test", f"no, {n} is not a prime number",
                            [f"{n} = {divisor} × {n // divisor}, so it has a divisor "
                             "other than 1 and itself"], value=False)

    def _prime_factors(self, low: str) -> Optional[Solution]:
        match = self._RE_PRIME_FACTORS.search(low)
        if match is None:
            return None
        n = int(match.group(1))
        if n < 2:
            raise MathError(f"{n} has no prime factorisation")
        factors = _factorise(n)
        powers = _as_powers(factors)
        return self._answer("number", "prime_factors", powers,
                            [f"{n} = " + " × ".join(str(f) for f in factors),
                             f"in index form: {powers}"], value=factors)

    def _factors(self, low: str) -> Optional[Solution]:
        match = self._RE_FACTORS.search(low)
        if match is None:
            return None
        raw = match.group(2) or match.group(3)
        if raw is None:
            return None
        n = int(raw)
        if n < 1:
            raise MathError("factors are counted for a positive whole number")
        divisors = _divisors(n)
        if "how many" in low:
            return self._answer("number", "factor_count", str(len(divisors)),
                                [f"the factors of {n} are " + ", ".join(map(str, divisors)),
                                 f"that is {len(divisors)} of them"], value=len(divisors))
        return self._answer("number", "factors", ", ".join(map(str, divisors)),
                            [f"every whole number that divides {n} exactly"], value=divisors)

    def _gcd(self, low: str) -> Optional[Solution]:
        if not self._RE_GCD.search(low):
            return None
        values = self._ints(numbers_in(low))
        if len(values) < 2:
            raise MathError("an hcf needs at least two numbers")
        result = values[0]
        for value in values[1:]:
            result = math.gcd(result, value)
        return self._answer("number", "gcd", str(result),
                            [f"{v} = " + " × ".join(map(str, _factorise(abs(v)) or [v]))
                             for v in values if abs(v) > 1] +
                            [f"the common factors multiply to {result}"], value=result)

    def _lcm(self, low: str) -> Optional[Solution]:
        if not self._RE_LCM.search(low):
            return None
        values = self._ints(numbers_in(low))
        if len(values) < 2:
            raise MathError("an lcm needs at least two numbers")
        if any(v == 0 for v in values):
            raise MathError("zero has no lowest common multiple")
        result = abs(values[0])
        for value in values[1:]:
            result = result * abs(value) // math.gcd(result, abs(value))
        hcf = values[0]
        for value in values[1:]:
            hcf = math.gcd(hcf, value)
        return self._answer("number", "lcm", str(result),
                            [f"hcf({', '.join(map(str, values))}) = {hcf}",
                             f"the product divided by what they share gives {result}"],
                            value=result)

    def _divisible(self, low: str) -> Optional[Solution]:
        match = self._RE_DIVISIBLE.search(low)
        if match is None:
            return None
        n, by = int(match.group(1)), int(match.group(2))
        if by == 0:
            raise MathError("nothing is divisible by zero")
        if n % by == 0:
            return self._answer("number", "divisible", f"yes, {n} is divisible by {by}",
                                [f"{n} ÷ {by} = {n // by} with nothing left over"], value=True)
        return self._answer("number", "divisible", f"no, {n} is not divisible by {by}",
                            [f"{n} ÷ {by} = {n // by} remainder {n % by}"], value=False)

    def _parity(self, low: str) -> Optional[Solution]:
        match = self._RE_PARITY.search(low)
        if match is None:
            return None
        raw = match.group(1) or match.group(3)
        asked = match.group(2) or match.group(4)
        if raw is None or asked is None:
            return None
        n = int(raw)
        is_even = n % 2 == 0
        actual = "even" if is_even else "odd"
        verdict = "yes" if actual == asked else "no"
        return self._answer("number", "parity", f"{verdict}, {n} is {actual}",
                            [f"{n} ÷ 2 leaves remainder {abs(n) % 2}"], value=actual)

    def _perfect(self, low: str) -> Optional[Solution]:
        match = self._RE_PERFECT.search(low)
        if match is None:
            return None
        raw = match.group(1) or match.group(3) or match.group(6)
        which = match.group(2) or match.group(4) or match.group(5)
        if raw is None or which is None:
            return None
        n, index = int(raw), 2 if which == "square" else 3
        root = _integer_root(n, index) if n >= 0 else None
        if root is None:
            return self._answer("number", "perfect", f"no, {n} is not a perfect {which}",
                                [f"no whole number raised to the power {index} gives {n}"],
                                value=False)
        return self._answer("number", "perfect", f"yes, {n} is a perfect {which}",
                            [f"{root}^{index} = {n}"], value=root)

    def _factorial(self, low: str) -> Optional[Solution]:
        match = self._RE_FACTORIAL.search(low)
        if match is None:
            return None
        raw = match.group(1) or match.group(2) or match.group(3)
        if raw is None:
            return None
        n = int(raw)
        if n > 2000:
            raise MathError(f"{n}! has more digits than she will print")
        value = math.factorial(n)
        if len(str(value)) > _MAX_DIGITS:
            raise MathError(f"{n}! has more than {_MAX_DIGITS} digits")
        working = " × ".join(str(k) for k in range(n, max(n - 4, 0), -1))
        if n > 4:
            working += " × … × 1"
        return self._answer("number", "factorial", str(value),
                            [f"{n}! = {working or '1'}"], value=value)

    # ---- powers, roots and logarithms ---------------------------------------- #

    _RE_ROOT = re.compile(
        r"\b(square|cube|fourth|fifth|(\d+)(?:st|nd|rd|th))?\s*root\s+of\s+(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)"
        r"|√\s*(-?\d+(?:\.\d+)?)")
    _RE_POWER = re.compile(
        r"\b(-?\d+(?:\.\d+)?)\s*(?:\^|\*\*)\s*(-?\d+)\b"
        r"|\b(-?\d+(?:\.\d+)?)\s+to\s+the\s+power(?:\s+of)?\s+(-?\d+)"
        r"|\b(square|cube)\s+of\s+(-?\d+(?:\.\d+)?)")
    _RE_LOG = re.compile(
        r"\b(?:log|ln)\b\s*(?:base\s*(\d+(?:\.\d+)?)\s*)?(?:of\s+)?(\d+(?:\.\d+)?)"
        r"|\blog\s*_?\s*(\d+)\s*\(?\s*(\d+(?:\.\d+)?)")

    def _root(self, low: str) -> Optional[Solution]:
        match = self._RE_ROOT.search(low)
        if match is None:
            return None
        raw = match.group(3) or match.group(4)
        if raw is None:
            return None
        word, digits = match.group(1), match.group(2)
        index = {"square": 2, "cube": 3, "fourth": 4, "fifth": 5}.get(word or "", 0)
        if not index:
            index = int(digits) if digits else 2
        if index < 2 or index > 32:
            raise MathError(f"a {index}th root is outside what she will take")
        value = _frac(raw.replace(" ", ""))
        text, exact = _root_text(value, index)
        exact_value = _exact_root(value, index)
        return self._answer("power", "root", text,
                            [f"{text.split(' ≈ ')[0]} is the number whose power {index} is "
                             f"{_num(value)}"] +
                            ([] if exact else ["no rational number has that power, so this is the "
                                               "closest decimal rather than the value"]),
                            value=exact_value if exact else float(text.split("≈")[-1]),
                            exact=exact)

    def _power(self, low: str) -> Optional[Solution]:
        match = self._RE_POWER.search(low)
        if match is None:
            return None
        if match.group(1) is not None:
            base, exponent = _frac(match.group(1)), int(match.group(2))
        elif match.group(3) is not None:
            base, exponent = _frac(match.group(3)), int(match.group(4))
        else:
            base, exponent = _frac(match.group(6)), 2 if match.group(5) == "square" else 3
        if abs(exponent) > _MAX_POW:
            raise MathError(f"exponent {exponent} is above the {_MAX_POW} ceiling")
        if exponent < 0 and base == 0:
            raise MathError("division by zero")
        value = base ** exponent
        if isinstance(value, Fraction) and len(str(value.numerator)) > _MAX_DIGITS:
            raise MathError(f"the answer has more than {_MAX_DIGITS} digits")
        return self._answer("power", "power", _num(value),
                            [f"{_num(base)}^{exponent} = " +
                             (" × ".join([_num(base)] * exponent) if 0 < exponent <= 4
                              else f"{_num(base)} multiplied by itself {exponent} times")],
                            value=value)

    def _logarithm(self, low: str) -> Optional[Solution]:
        if "log" not in low and "ln" not in low:
            return None
        match = self._RE_LOG.search(low)
        if match is None:
            return None
        base_raw = match.group(1) or match.group(3)
        value_raw = match.group(2) or match.group(4)
        if value_raw is None:
            return None
        value = _frac(value_raw)
        if value <= 0:
            raise MathError("a logarithm needs a positive number")
        if base_raw is None:
            base = Fraction(math.e).limit_denominator() if re.search(r"\bln\b", low) \
                else Fraction(10)
            natural = bool(re.search(r"\bln\b", low))
        else:
            base, natural = _frac(base_raw), False
        if natural:
            result = math.log(float(value))
            return self._answer("power", "logarithm", _num(result),
                                [f"ln {_num(value)} is the power e must be raised to"],
                                value=result, exact=False)
        if base <= 0 or base == 1:
            raise MathError("a logarithm base has to be positive and not 1")
        # An exact integer logarithm is a *different and better* answer than a decimal one, and it
        # is the one a school question is asking for: log base 2 of 8 is 3, not 2.9999999999999996.
        whole = _integer_log(value, base)
        if whole is not None:
            return self._answer("power", "logarithm", str(whole),
                                [f"{_num(base)}^{whole} = {_num(value)}"], value=whole)
        result = math.log(float(value), float(base))
        return self._answer("power", "logarithm", _num(result),
                            [f"the power {_num(base)} must be raised to to give {_num(value)}"],
                            value=result, exact=False)

    # ---- fractions and decimals ---------------------------------------------- #

    _RE_SIMPLIFY_FRACTION = re.compile(
        r"\b(?:simplify|reduce|lowest\s+terms?|simplest\s+form)\b[^\d]{0,24}(-?\d+)\s*/\s*(\d+)")
    _RE_TO_DECIMAL = re.compile(r"\b(-?\d+)\s*/\s*(\d+)\b[^\d]{0,24}\b(?:as|in|to|into)\s+"
                                r"(?:a\s+)?(decimal|percent(?:age)?)")
    _RE_TO_FRACTION = re.compile(r"\b(-?\d+\.\d+|\d+)\s*(%|percent)?\s*"
                                 r"\b(?:as|in|to|into)\s+(?:a\s+)?fraction")
    _RE_COMPARE = re.compile(r"\b(?:which\s+is\s+(bigger|larger|greater|smaller|less)|"
                             r"(bigger|larger|greater|smaller|less))\b")

    def _fraction_simplify(self, low: str) -> Optional[Solution]:
        match = self._RE_SIMPLIFY_FRACTION.search(low)
        if match is None:
            return None
        top, bottom = int(match.group(1)), int(match.group(2))
        if bottom == 0:
            raise MathError("division by zero")
        value = Fraction(top, bottom)
        divisor = math.gcd(abs(top), abs(bottom))
        return self._answer("fraction", "simplify", _num(value),
                            [f"hcf({abs(top)}, {abs(bottom)}) = {divisor}",
                             f"{top} ÷ {divisor} = {top // divisor}, "
                             f"{bottom} ÷ {divisor} = {bottom // divisor}"], value=value)

    def _fraction_convert(self, low: str) -> Optional[Solution]:
        match = self._RE_TO_DECIMAL.search(low)
        if match is not None:
            top, bottom = int(match.group(1)), int(match.group(2))
            if bottom == 0:
                raise MathError("division by zero")
            value = Fraction(top, bottom)
            if match.group(3).startswith("percent"):
                percent = value * 100
                return self._answer("fraction", "to_percent", f"{_maybe_decimal(percent)}%",
                                    [f"{top}/{bottom} = {_dec(value)}",
                                     f"× 100 gives {_maybe_decimal(percent)}%"], value=percent)
            return self._answer("fraction", "to_decimal", _dec(value),
                                [f"{top} ÷ {bottom} = {_dec(value)}"], value=value,
                                exact=_terminates(value))
        match = self._RE_TO_FRACTION.search(low)
        if match is None:
            return None
        value = _frac(match.group(1))
        if match.group(2):
            value = value / 100
        return self._answer("fraction", "to_fraction", _num(value),
                            [f"{match.group(1)}{match.group(2) or ''} written over its place value",
                             f"cancelled to lowest terms: {_num(value)}"], value=value)

    def _fraction_compare(self, low: str) -> Optional[Solution]:
        if not self._RE_COMPARE.search(low):
            return None
        values = numbers_in(low)
        if len(values) != 2:
            return None
        want_small = bool(re.search(r"\b(?:smaller|less|smallest)\b", low))
        first, second = values
        if first == second:
            return self._answer("fraction", "compare", f"they are equal — both are {_num(first)}",
                                [f"{_dec(first)} = {_dec(second)}"], value=None)
        chosen = min(first, second) if want_small else max(first, second)
        return self._answer("fraction", "compare", _num(chosen),
                            [f"{_num(first)} = {_dec(first)}", f"{_num(second)} = {_dec(second)}",
                             f"so {_num(chosen)} is the "
                             f"{'smaller' if want_small else 'larger'}"], value=chosen)

    # ---- percentage, and the commerce that is percentage in a sentence -------- #
    #
    # Five different questions share the word "percent" and three of them share the word "of", so
    # the order these are tried in is the whole of their correctness. `_percent_of` — the one that
    # simply multiplies — is tried *last* of the five, because it matches the surface of all of
    # them and would answer "15 is what percent of 60" with 9.

    _RE_PERCENT_OF = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)")
    _RE_PERCENT_WHICH = re.compile(
        r"(-?\d+(?:\.\d+)?)\s+is\s+what\s+(?:percent|percentage)\s+of\s+(-?\d+(?:\.\d+)?)"
        r"|what\s+(?:percent|percentage)\s+of\s+(-?\d+(?:\.\d+)?)\s+is\s+(-?\d+(?:\.\d+)?)")
    _RE_PERCENT_WHOLE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s+is\s+(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+what")
    _RE_PERCENT_CHANGE = re.compile(
        r"(?:percent(?:age)?\s+(increase|decrease|change)|(increase|decrease)\s+"
        r"(?:in\s+)?percent(?:age)?)\b[^\d]{0,30}(-?\d+(?:\.\d+)?)\D+(-?\d+(?:\.\d+)?)")

    def _percent_of(self, low: str) -> Optional[Solution]:
        match = self._RE_PERCENT_OF.search(low)
        if match is None:
            return None
        percent, whole = _frac(match.group(1)), _frac(match.group(2))
        value = whole * percent / 100
        return self._answer("percent", "percent_of", _num(value),
                            [f"{_num(percent)}% = {_num(percent)}/100",
                             f"{_num(percent)}/100 × {_num(whole)} = {_num(value)}"], value=value)

    def _percent_which(self, low: str) -> Optional[Solution]:
        match = self._RE_PERCENT_WHICH.search(low)
        if match is None:
            return None
        if match.group(1) is not None:
            part, whole = _frac(match.group(1)), _frac(match.group(2))
        else:
            whole, part = _frac(match.group(3)), _frac(match.group(4))
        if whole == 0:
            raise MathError("nothing is a percentage of zero")
        value = part / whole * 100
        return self._answer("percent", "percent_which", f"{_maybe_decimal(value)}%",
                            [f"{_num(part)} ÷ {_num(whole)} = {_num(part / whole)}",
                             f"× 100 gives {_maybe_decimal(value)}%"], value=value)

    def _percent_whole(self, low: str) -> Optional[Solution]:
        match = self._RE_PERCENT_WHOLE.search(low)
        if match is None:
            return None
        part, percent = _frac(match.group(1)), _frac(match.group(2))
        if percent == 0:
            raise MathError("0% of every number is 0, so the whole cannot be found")
        value = part * 100 / percent
        return self._answer("percent", "percent_whole", _num(value),
                            [f"{_num(percent)}% of x = {_num(part)}",
                             f"x = {_num(part)} × 100 ÷ {_num(percent)} = {_num(value)}"],
                            value=value)

    def _percent_change(self, low: str) -> Optional[Solution]:
        match = self._RE_PERCENT_CHANGE.search(low)
        if match is None:
            return None
        direction = match.group(1) or match.group(2)
        start, end = _frac(match.group(3)), _frac(match.group(4))
        if start == 0:
            raise MathError("a change from zero has no percentage")
        change = end - start
        value = change / start * 100
        word = "increase" if change > 0 else "decrease"
        if direction in ("increase", "decrease") and direction != word and change:
            # The question named a direction and the numbers disagree with it. Answering the
            # arithmetic silently would confirm a premise the numbers refute.
            article = "an" if direction[0] in "aeiou" else "a"
            return self._answer("percent", "percent_change",
                                f"{_maybe_decimal(abs(value))}% {word}, "
                                f"not {article} {direction}",
                                [f"from {_num(start)} to {_num(end)} is a change of "
                                 f"{_num(change)}"], value=value)
        return self._answer("percent", "percent_change", f"{_maybe_decimal(abs(value))}% {word}",
                            [f"change = {_num(end)} - {_num(start)} = {_num(change)}",
                             f"{_num(change)} ÷ {_num(start)} × 100 = {_num(value)}%"], value=value)

    _RE_PROFIT = re.compile(r"\b(?:cost\s+price|cp|bought\s+(?:it\s+)?for|buys?\s+.*?\s+for)\b")
    _RE_SELL = re.compile(r"\b(?:selling\s+price|sp|sold\s+(?:it\s+)?for|sells?\s+.*?\s+for)\b")
    _RE_DISCOUNT = re.compile(
        r"\bdiscount\b[^\d]{0,30}(\d+(?:\.\d+)?)\s*(?:%|percent)|(\d+(?:\.\d+)?)\s*"
        r"(?:%|percent)\s+discount")
    _RE_SI = re.compile(r"\bsimple\s+interest\b")
    _RE_CI = re.compile(r"\bcompound\s+interest\b")

    def _profit_loss(self, low: str) -> Optional[Solution]:
        """Profit and loss, from a sentence that names both prices.

        The two prices are told apart by *which phrase they follow*, never by which came first: "he
        sold for 500 an article he bought for 400" is an ordinary sentence and reading it by
        position gives a 25% loss on a 25% profit.
        """
        if not (self._RE_PROFIT.search(low) and self._RE_SELL.search(low)):
            return None
        if "profit" not in low and "loss" not in low and "gain" not in low:
            return None
        cost = _number_after(low, self._RE_PROFIT)
        sell = _number_after(low, self._RE_SELL)
        if cost is None or sell is None:
            return None
        if cost == 0:
            raise MathError("a profit percentage needs a cost price above zero")
        change = sell - cost
        value = abs(change) / cost * 100
        word = "profit" if change > 0 else ("loss" if change < 0 else "neither profit nor loss")
        if change == 0:
            return self._answer("commerce", "profit_loss", "neither profit nor loss — 0%",
                                [f"cost {_num(cost)} = selling {_num(sell)}"], value=Fraction(0))
        return self._answer("commerce", "profit_loss", f"{_maybe_decimal(value)}% {word}",
                            [f"{word} = {_num(abs(change))}",
                             f"{_num(abs(change))} ÷ {_num(cost)} × 100 = "
                             f"{_maybe_decimal(value)}%"],
                            value=value)

    def _discount(self, low: str) -> Optional[Solution]:
        match = self._RE_DISCOUNT.search(low)
        if match is None:
            return None
        percent = _frac(match.group(1) or match.group(2))
        prices = [n for n in numbers_in(low) if n != percent]
        if not prices:
            return None
        marked = prices[0]
        cut = marked * percent / 100
        final = marked - cut
        return self._answer("commerce", "discount", _num(final),
                            [f"discount = {_num(percent)}% of {_num(marked)} = {_num(cut)}",
                             f"{_num(marked)} - {_num(cut)} = {_num(final)}"], value=final)

    def _simple_interest(self, low: str) -> Optional[Solution]:
        if not self._RE_SI.search(low):
            return None
        principal, rate, years = _pri_rate_time(low)
        interest = principal * rate * years / 100
        return self._answer("commerce", "simple_interest", _num(interest),
                            ["SI = P × R × T ÷ 100",
                             f"= {_num(principal)} × {_num(rate)} × {_num(years)} ÷ 100 "
                             f"= {_num(interest)}",
                             f"amount = {_num(principal + interest)}"], value=interest)

    def _compound_interest(self, low: str) -> Optional[Solution]:
        if not self._RE_CI.search(low):
            return None
        principal, rate, years = _pri_rate_time(low)
        if years.denominator != 1:
            raise MathError("she compounds over whole periods only")
        growth = (1 + rate / 100) ** int(years)
        amount = principal * growth
        interest = amount - principal
        return self._answer("commerce", "compound_interest", _both(interest),
                            ["A = P(1 + R/100)^T",
                             f"= {_num(principal)} × (1 + {_num(rate)}/100)^{int(years)} "
                             f"= {_both(amount)}",
                             f"interest = A - P = {_both(interest)}"], value=interest)

    # ---- ratio and proportion ------------------------------------------------ #

    _RE_RATIO = re.compile(r"(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)(?:\s*:\s*(-?\d+(?:\.\d+)?))?")
    _RE_DIVIDE_RATIO = re.compile(r"\b(?:divide|share|split|distribute)\b")
    _RE_PROPORTION = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*(?:::|=)\s*"
                                r"(\d+(?:\.\d+)?)\s*:\s*([a-z?]|\d+(?:\.\d+)?)")

    def _ratio_simplify(self, low: str) -> Optional[Solution]:
        if "ratio" not in low or not re.search(r"\b(?:simplify|simplest|lowest|reduce)\b", low):
            return None
        match = self._RE_RATIO.search(low)
        if match is None:
            return None
        parts = [_frac(g) for g in match.groups() if g is not None]
        whole = self._ints([p * _common_denominator(parts) for p in parts])
        divisor = 0
        for part in whole:
            divisor = math.gcd(divisor, abs(part))
        if divisor == 0:
            raise MathError("a ratio of zeroes has no simplest form")
        reduced = [p // divisor for p in whole]
        return self._answer("ratio", "ratio_simplify", " : ".join(map(str, reduced)),
                            [f"hcf({', '.join(map(str, whole))}) = {divisor}",
                             "divide every term by it"], value=reduced)

    def _ratio_divide(self, low: str) -> Optional[Solution]:
        if "ratio" not in low or not self._RE_DIVIDE_RATIO.search(low):
            return None
        match = self._RE_RATIO.search(low)
        if match is None:
            return None
        parts = [_frac(g) for g in match.groups() if g is not None]
        total = _number_before(low, self._RE_RATIO, exclude=parts)
        if total is None:
            return None
        share_total = sum(parts, Fraction(0))
        if share_total == 0:
            raise MathError("the shares add up to zero")
        shares = [total * part / share_total for part in parts]
        return self._answer("ratio", "ratio_divide", " , ".join(_num(s) for s in shares),
                            [f"the shares add up to {_num(share_total)} parts",
                             f"one part = {_num(total)} ÷ {_num(share_total)} = "
                             f"{_num(total / share_total)}"] +
                            [f"{_num(part)} parts = {_num(share)}"
                             for part, share in zip(parts, shares)], value=shares)

    def _proportion(self, low: str) -> Optional[Solution]:
        match = self._RE_PROPORTION.search(low)
        if match is None:
            return None
        a, b, c, d = match.groups()
        if d.isalpha() or d == "?":
            value = _frac(b) * _frac(c) / _frac(a)
            if _frac(a) == 0:
                raise MathError("division by zero")
            return self._answer("ratio", "proportion", _num(value),
                                [f"{a} : {b} = {c} : x means {a}x = {b} × {c}",
                                 f"x = {b} × {c} ÷ {a} = {_num(value)}"], value=value)
        return None

    # ---- algebra ------------------------------------------------------------- #
    #
    # Everything here goes through `Poly`, which is why none of it can execute anything: the
    # parser's only output type is a dict of powers to rationals. See the module docstring.

    _RE_EQUATION = re.compile(r"=")
    _RE_SOLVE = re.compile(r"\b(?:solve|find|what\s+is|value\s+of|roots?\s+of)\b")
    _RE_EXPAND = re.compile(r"\b(?:expand|multiply\s+out)\b\s*(?:the\s+)?(?:expression\s+)?(.+)$")
    _RE_FACTORISE = re.compile(
        r"\b(?:factorise|factorize|factor)\b\s*(?:the\s+)?(?:expression\s+)?(.+)$")
    _RE_SIMPLIFY_EXPR = re.compile(
        r"\b(?:simplify|collect(?:\s+like\s+terms)?)\b\s*(?:the\s+)?(?:expression\s+)?(.+)$")
    _RE_EVALUATE_AT = re.compile(
        r"\b(?:value|evaluate|find|what\s+is)\b\s*(?:the\s+value\s+of\s+)?(?:of\s+)?(.+?)\s+"
        r"(?:when|at|if|for|given)\s+([a-z])\s*=\s*(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)")

    def _solve_equation(self, low: str) -> Optional[Solution]:
        """Solve one equation in one unknown, of any degree whose roots are findable exactly.

        Degree 1 and 2 are closed forms. Above that the rational roots are found by the rational
        root theorem and reported as the roots she can *prove*, with the leftover factor named —
        an approximation of an irrational root, presented among exact ones, would be the one wrong
        answer in a list of right ones and indistinguishable from them.
        """
        if "=" not in low:
            return None
        equation = _equation_body(low)
        if equation is None:
            return None
        left_text, _, right_text = equation.partition("=")
        symbols = _symbols_in(equation)
        if len(symbols) != 1:
            return None
        symbol = symbols[0]
        left, right = Poly.parse(left_text, symbol), Poly.parse(right_text, symbol)
        poly = left - right
        if poly.is_zero:
            raise MathError("that equation is true for every value — it is an identity")
        if poly.is_constant:
            raise MathError("that equation is never true — the two sides differ by a constant")
        steps = [f"{left.text()} = {right.text()}",
                 f"bring everything to one side: {poly.text()} = 0"]
        if poly.degree == 1:
            root = -poly.coefficient(0) / poly.coefficient(1)
            steps.append(f"{symbol} = {_num(-poly.coefficient(0))} ÷ "
                         f"{_num(poly.coefficient(1))} = {_num(root)}")
            return self._answer("algebra", "solve_linear", f"{symbol} = {_num(root)}", steps,
                                value=root)
        if poly.degree == 2:
            return self._quadratic(poly, symbol, steps)
        roots = poly.rational_roots()
        if not roots:
            raise MathError("that equation has no rational root she can state exactly")
        found = ", ".join(f"{symbol} = {_num(r)}" for r in roots)
        steps.append(f"the rational roots are {found}")
        if len(roots) < poly.degree:
            steps.append(f"a degree-{poly.degree} equation has {poly.degree} roots counted with "
                         "multiplicity; the rest are not rational and are not stated")
        return self._answer("algebra", "solve_polynomial", found, steps, value=roots)

    def _quadratic(self, poly: Poly, symbol: str, steps: List[str]) -> Solution:
        a, b, c = poly.coefficient(2), poly.coefficient(1), poly.coefficient(0)
        discriminant = b * b - 4 * a * c
        steps.append(f"a = {_num(a)}, b = {_num(b)}, c = {_num(c)}")
        steps.append(f"discriminant = b² - 4ac = {_num(discriminant)}")
        if discriminant < 0:
            return self._answer("algebra", "solve_quadratic",
                                "no real solution — the discriminant is negative",
                                steps + ["a negative discriminant has no real square root"],
                                value=[])
        root = _exact_root(discriminant, 2)
        if root is not None:
            values = sorted({(-b + root) / (2 * a), (-b - root) / (2 * a)})
            steps.append(f"√{_num(discriminant)} = {_num(root)}")
            found = ", ".join(f"{symbol} = {_num(v)}" for v in values)
            steps.append(f"({_num(-b)} ± {_num(root)}) ÷ {_num(2 * a)} gives {found}")
            return self._answer("algebra", "solve_quadratic", found, steps, value=values)
        surd = math.sqrt(float(discriminant))
        values = sorted({(float(-b) + surd) / float(2 * a), (float(-b) - surd) / float(2 * a)})
        found = ", ".join(f"{symbol} ≈ {_num(v)}" for v in values)
        steps.append(f"√{_num(discriminant)} is irrational, so the roots are decimals")
        return self._answer("algebra", "solve_quadratic", found, steps, value=values, exact=False)

    def _simultaneous(self, low: str) -> Optional[Solution]:
        """Two linear equations in two unknowns, by elimination, exactly.

        Kept ahead of :meth:`_solve_equation` in the dispatch table because a pair of equations
        contains a single equation, and the single-equation skill would happily solve the first of
        them for one unknown in terms of nothing and call it an answer.
        """
        equations = _equation_pair(low)
        if equations is None:
            return None
        first, second = equations
        symbols = sorted(set(_symbols_in(first)) | set(_symbols_in(second)))
        if len(symbols) != 2:
            return None
        x, y = symbols
        a1, b1, c1 = _linear_form(first, x, y)
        a2, b2, c2 = _linear_form(second, x, y)
        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            raise MathError("those two equations do not fix a single point")
        x_value = (c1 * b2 - c2 * b1) / determinant
        y_value = (a1 * c2 - a2 * c1) / determinant
        return self._answer("algebra", "simultaneous",
                            f"{x} = {_num(x_value)}, {y} = {_num(y_value)}",
                            [f"{_num(a1)}{x} + {_num(b1)}{y} = {_num(c1)}",
                             f"{_num(a2)}{x} + {_num(b2)}{y} = {_num(c2)}",
                             f"eliminating {y}: {x} = {_num(x_value)}",
                             f"substituting back: {y} = {_num(y_value)}"],
                            value=(x_value, y_value))

    def _expand(self, low: str) -> Optional[Solution]:
        match = self._RE_EXPAND.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        if not _looks_algebraic(body):
            return None
        poly = Poly.parse(body)
        return self._answer("algebra", "expand", poly.text(),
                            ["multiply every term of one bracket by every term of the other",
                             f"{body} = {poly.text()}"], value=poly)

    def _factorise(self, low: str) -> Optional[Solution]:
        """Factorise over the rationals, and say so when it does not factorise there.

        ``x^2 + 1`` has no factorisation with rational coefficients, and "it does not factorise" is
        the correct answer rather than a failure — the same distinction :func:`_quadratic_rational_roots`
        draws between "no roots" and "no roots *of this kind*".
        """
        match = self._RE_FACTORISE.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        if not _looks_algebraic(body):
            return None
        poly = Poly.parse(body)
        if poly.degree < 1:
            return None
        symbol = poly.symbol
        roots = poly.rational_roots()
        common = _content(poly)
        remaining, pieces = poly, []
        if common != 1:
            remaining = Poly({p: c / common for p, c in poly.coefficients.items()}, symbol)
            pieces.append(_num(common))
        for root in roots:
            factor = Poly({1: Fraction(root.denominator), 0: -root.numerator}, symbol)
            quotient, rest = remaining.divmod(factor)
            while rest.is_zero:
                remaining = quotient
                pieces.append(_bracketed(factor))
                quotient, rest = remaining.divmod(factor)
        if not remaining.is_constant:
            pieces.append(_bracketed(remaining))
        elif remaining.value() != 1:
            pieces.insert(0, _num(remaining.value()))
        # A one-term factor needs no brackets and reads best next to the coefficient:
        # `2x(x + 2)`, not `2(x + 2)(x)`. Sorting by whether a piece is bracketed puts the
        # coefficient and the bare symbol first without disturbing the order of the rest.
        pieces.sort(key=lambda piece: piece.startswith("("))
        if len(pieces) <= 1:
            return self._answer("algebra", "factorise",
                                f"{poly.text()} does not factorise over the rationals",
                                [f"no rational number makes {poly.text()} zero"], value=poly)
        product = "".join(p if p.startswith("(") else f"{p}" for p in pieces)
        return self._answer("algebra", "factorise", product,
                            [f"the roots are {', '.join(_num(r) for r in roots)}"
                             if roots else "taking out the common factor",
                             f"{poly.text()} = {product}"], value=pieces)

    def _evaluate_at(self, low: str) -> Optional[Solution]:
        match = self._RE_EVALUATE_AT.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        symbol, point = match.group(2), _frac(match.group(3).replace(" ", ""))
        if not _looks_algebraic(body) or symbol not in _symbols_in(body):
            return None
        poly = Poly.parse(body, symbol)
        value = poly.at(point)
        return self._answer("algebra", "evaluate_at", _num(value),
                            [f"put {symbol} = {_num(point)} into {poly.text()}",
                             f"that gives {_num(value)}"], value=value)

    def _simplify_expression(self, low: str) -> Optional[Solution]:
        match = self._RE_SIMPLIFY_EXPR.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        if not _looks_algebraic(body):
            return None
        poly = Poly.parse(body)
        return self._answer("algebra", "simplify_expression", poly.text(),
                            ["collect the terms of each power",
                             f"{body} = {poly.text()}"], value=poly)

    # ---- sequences and series ------------------------------------------------ #

    _RE_NEXT = re.compile(r"\bnext\s+(?:\d+\s+)?(?:term|number|element)s?\b")
    _RE_NTH = re.compile(r"\b(\d+)(?:st|nd|rd|th)\s+term\b")
    _RE_SUM = re.compile(r"\bsum\b")
    _RE_FIRST_N = re.compile(r"\bfirst\s+(\d+)\s+(natural|counting|even|odd|square|prime)?\s*"
                             r"(?:numbers?|terms?)?")

    def _sequence_next(self, low: str) -> Optional[Solution]:
        if not self._RE_NEXT.search(low):
            return None
        terms = _series_terms(low)
        if len(terms) < 3:
            return None
        rule = _rule_of(terms)
        if rule is None:
            raise MathError("she cannot see the rule in that sequence")
        kind, step, nxt = rule
        return self._answer("sequence", "next_term", _num(nxt),
                            [f"the terms are {', '.join(_num(t) for t in terms)}",
                             kind, f"so the next term is {_num(nxt)}"], value=nxt)

    def _sequence_nth(self, low: str) -> Optional[Solution]:
        match = self._RE_NTH.search(low)
        if match is None:
            return None
        n = int(match.group(1))
        if n < 1 or n > 100000:
            raise MathError(f"a {n}th term is outside what she will work out")
        # The ordinal is removed **by where it is written**, never by its value. Excluding it by
        # value — the first version of this — deletes the term as well whenever the sequence
        # happens to contain the ordinal: "the 15th term of 12, 15, 18" lost the 15, was left with
        # two terms, and came back silent. Found by the exam, on a question nobody wrote by hand.
        terms = _series_terms(low[:match.start()] + " " + low[match.end():])
        if len(terms) < 3:
            return None
        rule = _rule_of(terms)
        if rule is None:
            raise MathError("she cannot see the rule in that sequence")
        kind, step, _ = rule
        first = terms[0]
        if kind.startswith("each term is the one before plus"):
            value = first + step * (n - 1)
            formula = f"a + (n-1)d = {_num(first)} + ({n}-1)×{_num(step)}"
        else:
            value = first * step ** (n - 1)
            formula = f"a·r^(n-1) = {_num(first)} × {_num(step)}^{n - 1}"
        return self._answer("sequence", "nth_term", _num(value),
                            [f"the terms are {', '.join(_num(t) for t in terms)}", kind,
                             f"{formula} = {_num(value)}"], value=value)

    def _series_sum(self, low: str) -> Optional[Solution]:
        if not self._RE_SUM.search(low):
            return None
        match = self._RE_FIRST_N.search(low)
        if match is not None:
            n, kind = int(match.group(1)), (match.group(2) or "natural")
            return self._closed_sum(n, kind)
        terms = _series_terms(low)
        if len(terms) < 3:
            return None
        total = sum(terms, Fraction(0))
        return self._answer("sequence", "series_sum", _num(total),
                            [f"{' + '.join(_num(t) for t in terms)} = {_num(total)}"], value=total)

    def _closed_sum(self, n: int, kind: str) -> Solution:
        """The standard series sums, by formula rather than by adding ``n`` things up.

        The formula is the answer to a school question about the *formula*; adding a million
        numbers would give the same value and demonstrate nothing, and for ``n`` above the ceiling
        it would not finish.
        """
        if n < 1 or n > 10 ** 9:
            raise MathError(f"{n} terms is outside what she will sum")
        if kind in ("natural", "counting"):
            total = Fraction(n * (n + 1), 2)
            return self._answer("sequence", "sum_natural", _num(total),
                                [f"n(n+1)/2 = {n}×{n + 1}/2 = {_num(total)}"], value=total)
        if kind == "even":
            total = Fraction(n * (n + 1))
            return self._answer("sequence", "sum_even", _num(total),
                                [f"n(n+1) = {n}×{n + 1} = {_num(total)}"], value=total)
        if kind == "odd":
            total = Fraction(n * n)
            return self._answer("sequence", "sum_odd", _num(total),
                                [f"n² = {n}² = {_num(total)}"], value=total)
        if kind == "square":
            total = Fraction(n * (n + 1) * (2 * n + 1), 6)
            return self._answer("sequence", "sum_squares", _num(total),
                                [f"n(n+1)(2n+1)/6 = {_num(total)}"], value=total)
        if kind == "prime":
            if n > 10000:
                raise MathError("that many primes is outside what she will sieve")
            primes = _primes_upto_count(n)
            return self._answer("sequence", "sum_primes", str(sum(primes)),
                                [f"the first {n} primes are "
                                 f"{', '.join(map(str, primes[:8]))}"
                                 f"{' …' if n > 8 else ''}"], value=sum(primes))
        raise MathError(f"she has no closed form for the first {n} {kind} numbers")

    # ---- geometry and mensuration -------------------------------------------- #
    #
    # A shape question is a formula plus a reading problem, and the reading problem is the hard
    # half: "a rectangle 8 long and 3 wide" and "a rectangle of length 8 and width 3" name the same
    # two numbers in two orders, and taking them by position gets one of the two wrong. Every
    # dimension below is found by *its own name*, never by where it sits in the sentence.

    _RE_PYTHAGORAS = re.compile(r"\b(?:hypotenuse|pythagoras|right[- ]angled?\s+triangle)\b")
    _RE_TRIANGLE_ANGLE = re.compile(r"\b(?:third|remaining|other)\s+angle\b")
    _RE_POLYGON_ANGLES = re.compile(
        r"\b(?:interior|inside)\s+angles?\b|\bangles?\s+of\s+a\s+(?:polygon|"
        r"triangle|quadrilateral|pentagon|hexagon|heptagon|octagon|nonagon|decagon)\b")
    _RE_COMPLEMENT = re.compile(r"\b(complement|supplement)(?:ary)?\b")

    def _pythagoras(self, low: str) -> Optional[Solution]:
        if not self._RE_PYTHAGORAS.search(low):
            return None
        legs = numbers_in(low)
        if len(legs) < 2:
            return None
        hypotenuse = _dimension(low, ("hypotenuse",))
        if hypotenuse is not None and len(legs) >= 2:
            other = [v for v in legs if v != hypotenuse]
            if not other:
                return None
            leg = other[0]
            if hypotenuse <= leg:
                raise MathError("the hypotenuse is the longest side of a right-angled triangle")
            squared = hypotenuse * hypotenuse - leg * leg
            text, exact = _root_text(squared)
            return self._answer("geometry", "pythagoras_leg", text,
                                [f"a² = c² - b² = {_num(hypotenuse)}² - {_num(leg)}² "
                                 f"= {_num(squared)}", f"a = {text}"], exact=exact)
        a, b = legs[0], legs[1]
        squared = a * a + b * b
        text, exact = _root_text(squared)
        return self._answer("geometry", "pythagoras", text,
                            [f"c² = a² + b² = {_num(a)}² + {_num(b)}² = {_num(squared)}",
                             f"c = {text}"], exact=exact)

    def _angles(self, low: str) -> Optional[Solution]:
        match = self._RE_COMPLEMENT.search(low)
        if match is not None:
            values = numbers_in(low)
            if not values:
                return None
            angle = values[0]
            total = Fraction(90) if match.group(1) == "complement" else Fraction(180)
            if angle >= total:
                raise MathError(f"an angle of {_num(angle)}° has no "
                                f"{match.group(1)} — they add to {_num(total)}°")
            return self._answer("geometry", "angle_pair", f"{_num(total - angle)}°",
                                [f"{match.group(1)}ary angles add to {_num(total)}°",
                                 f"{_num(total)} - {_num(angle)} = {_num(total - angle)}"],
                                value=total - angle)
        if self._RE_TRIANGLE_ANGLE.search(low):
            values = numbers_in(low)
            if len(values) < 2:
                return None
            known = sum(values[:2], Fraction(0))
            if known >= 180:
                raise MathError("two angles of a triangle cannot add to 180° or more")
            return self._answer("geometry", "triangle_angle", f"{_num(180 - known)}°",
                                ["the angles of a triangle add to 180°",
                                 f"180 - ({_num(values[0])} + {_num(values[1])}) "
                                 f"= {_num(180 - known)}"], value=180 - known)
        if self._RE_POLYGON_ANGLES.search(low):
            sides = _polygon_sides(low)
            if sides is None:
                return None
            if sides < 3:
                raise MathError("a polygon has at least three sides")
            total = Fraction((sides - 2) * 180)
            if "each" in low or "one" in low:
                return self._answer("geometry", "polygon_angle", f"{_num(total / sides)}°",
                                    [f"(n-2)×180 = {_num(total)}°",
                                     f"a regular one divides that by {sides}"],
                                    value=total / sides)
            return self._answer("geometry", "polygon_angles", f"{_num(total)}°",
                                [f"(n-2)×180 with n = {sides}",
                                 f"= {sides - 2}×180 = {_num(total)}°"], value=total)
        return None

    def _area(self, low: str) -> Optional[Solution]:
        if "area" not in low or "surface area" in low:
            return None
        shape = _shape_in(low)
        if shape is None:
            return None
        if shape == "circle":
            radius = _radius(low)
            if radius is None:
                return None
            return self._answer("geometry", "area_circle", _pi_text(radius * radius),
                                [f"A = πr² with r = {_num(radius)}",
                                 f"= {_pi_text(radius * radius)}"],
                                value=radius * radius, exact=False)
        if shape == "square":
            side = _dimension(low, ("side", "edge", "length"))
            if side is None:
                return None
            return self._answer("geometry", "area_square", _num(side * side),
                                [f"A = s² = {_num(side)}² = {_num(side * side)}"],
                                value=side * side)
        if shape in ("rectangle", "parallelogram"):
            first = _dimension(low, ("length", "long", "base"))
            second = _dimension(low, ("width", "breadth", "wide", "height"))
            if first is None or second is None:
                return None
            name = "l × b" if shape == "rectangle" else "base × height"
            return self._answer("geometry", f"area_{shape}", _num(first * second),
                                [f"A = {name} = {_num(first)} × {_num(second)} "
                                 f"= {_num(first * second)}"], value=first * second)
        if shape == "triangle":
            base = _dimension(low, ("base",))
            height = _dimension(low, ("height", "altitude"))
            if base is not None and height is not None:
                area = base * height / 2
                return self._answer("geometry", "area_triangle", _num(area),
                                    [f"A = ½ × base × height = ½ × {_num(base)} × {_num(height)}",
                                     f"= {_num(area)}"], value=area)
            sides = numbers_in(low)
            if len(sides) == 3 and all(s > 0 for s in sides):
                return self._heron(sides)
            return None
        if shape == "trapezium":
            parallels = _dimensions(low, ("parallel", "sides"))
            height = _dimension(low, ("height", "distance", "altitude"))
            if len(parallels) < 2 or height is None:
                return None
            area = (parallels[0] + parallels[1]) * height / 2
            return self._answer("geometry", "area_trapezium", _num(area),
                                [f"A = ½ × (a + b) × h = ½ × ({_num(parallels[0])} + "
                                 f"{_num(parallels[1])}) × {_num(height)}",
                                 f"= {_num(area)}"], value=area)
        return None

    def _heron(self, sides: Sequence[Fraction]) -> Solution:
        """Heron's formula. The exact case is common enough in a textbook to be worth keeping."""
        a, b, c = sides
        if a + b <= c or a + c <= b or b + c <= a:
            raise MathError("those three lengths do not close into a triangle")
        s = (a + b + c) / 2
        squared = s * (s - a) * (s - b) * (s - c)
        text, exact = _root_text(squared)
        return self._answer("geometry", "area_heron", text,
                            [f"s = ({_num(a)} + {_num(b)} + {_num(c)})/2 = {_num(s)}",
                             f"A = √(s(s-a)(s-b)(s-c)) = {text}"], exact=exact)

    def _perimeter(self, low: str) -> Optional[Solution]:
        if "perimeter" not in low and "circumference" not in low:
            return None
        shape = _shape_in(low) or ("circle" if "circumference" in low else None)
        if shape is None:
            return None
        if shape == "circle":
            radius = _radius(low)
            if radius is None:
                return None
            return self._answer("geometry", "circumference", _pi_text(radius * 2),
                                [f"C = 2πr with r = {_num(radius)}",
                                 f"= {_pi_text(radius * 2)}"], value=radius * 2, exact=False)
        if shape == "square":
            side = _dimension(low, ("side", "edge", "length"))
            if side is None:
                return None
            return self._answer("geometry", "perimeter_square", _num(side * 4),
                                [f"P = 4s = 4 × {_num(side)} = {_num(side * 4)}"], value=side * 4)
        if shape in ("rectangle", "parallelogram"):
            first = _dimension(low, ("length", "long", "base"))
            second = _dimension(low, ("width", "breadth", "wide", "height"))
            if first is None or second is None:
                return None
            perimeter = (first + second) * 2
            return self._answer("geometry", "perimeter_rectangle", _num(perimeter),
                                [f"P = 2(l + b) = 2({_num(first)} + {_num(second)})",
                                 f"= {_num(perimeter)}"], value=perimeter)
        if shape == "triangle":
            sides = numbers_in(low)
            if len(sides) != 3:
                return None
            return self._answer("geometry", "perimeter_triangle", _num(sum(sides, Fraction(0))),
                                [f"P = {' + '.join(_num(s) for s in sides)} "
                                 f"= {_num(sum(sides, Fraction(0)))}"],
                                value=sum(sides, Fraction(0)))
        return None

    def _volume(self, low: str) -> Optional[Solution]:
        if "volume" not in low and "capacity" not in low:
            return None
        shape = _shape_in(low)
        if shape == "cube":
            side = _dimension(low, ("side", "edge", "length"))
            if side is None:
                return None
            return self._answer("geometry", "volume_cube", _num(side ** 3),
                                [f"V = s³ = {_num(side)}³ = {_num(side ** 3)}"], value=side ** 3)
        if shape == "cuboid":
            l = _dimension(low, ("length", "long"))
            b = _dimension(low, ("width", "breadth", "wide"))
            h = _dimension(low, ("height", "tall", "depth"))
            if None in (l, b, h):
                return None
            return self._answer("geometry", "volume_cuboid", _num(l * b * h),
                                [f"V = l × b × h = {_num(l)} × {_num(b)} × {_num(h)}",
                                 f"= {_num(l * b * h)}"], value=l * b * h)
        if shape == "cylinder":
            radius, height = _radius(low), _dimension(low, ("height", "tall", "long"))
            if radius is None or height is None:
                return None
            coefficient = radius * radius * height
            return self._answer("geometry", "volume_cylinder", _pi_text(coefficient),
                                [f"V = πr²h = π × {_num(radius)}² × {_num(height)}",
                                 f"= {_pi_text(coefficient)}"], value=coefficient, exact=False)
        if shape == "sphere":
            radius = _radius(low)
            if radius is None:
                return None
            coefficient = Fraction(4, 3) * radius ** 3
            return self._answer("geometry", "volume_sphere", _pi_text(coefficient),
                                [f"V = 4/3 πr³ = 4/3 × π × {_num(radius)}³",
                                 f"= {_pi_text(coefficient)}"], value=coefficient, exact=False)
        if shape == "cone":
            radius, height = _radius(low), _dimension(low, ("height", "tall"))
            if radius is None or height is None:
                return None
            coefficient = radius * radius * height / 3
            return self._answer("geometry", "volume_cone", _pi_text(coefficient),
                                [f"V = ⅓πr²h = ⅓ × π × {_num(radius)}² × {_num(height)}",
                                 f"= {_pi_text(coefficient)}"], value=coefficient, exact=False)
        return None

    def _surface_area(self, low: str) -> Optional[Solution]:
        if "surface area" not in low:
            return None
        shape = _shape_in(low)
        if shape == "cube":
            side = _dimension(low, ("side", "edge", "length"))
            if side is None:
                return None
            return self._answer("geometry", "surface_cube", _num(side * side * 6),
                                [f"A = 6s² = 6 × {_num(side)}² = {_num(side * side * 6)}"],
                                value=side * side * 6)
        if shape == "cuboid":
            l = _dimension(low, ("length", "long"))
            b = _dimension(low, ("width", "breadth", "wide"))
            h = _dimension(low, ("height", "tall", "depth"))
            if None in (l, b, h):
                return None
            area = 2 * (l * b + b * h + h * l)
            return self._answer("geometry", "surface_cuboid", _num(area),
                                [f"A = 2(lb + bh + hl) = 2({_num(l * b)} + {_num(b * h)} + "
                                 f"{_num(h * l)})", f"= {_num(area)}"], value=area)
        if shape == "sphere":
            radius = _radius(low)
            if radius is None:
                return None
            coefficient = 4 * radius * radius
            return self._answer("geometry", "surface_sphere", _pi_text(coefficient),
                                [f"A = 4πr² = 4 × π × {_num(radius)}²",
                                 f"= {_pi_text(coefficient)}"], value=coefficient, exact=False)
        if shape == "cylinder":
            radius, height = _radius(low), _dimension(low, ("height", "tall", "long"))
            if radius is None or height is None:
                return None
            coefficient = 2 * radius * (radius + height)
            return self._answer("geometry", "surface_cylinder", _pi_text(coefficient),
                                [f"A = 2πr(r + h) = 2π × {_num(radius)} × "
                                 f"({_num(radius)} + {_num(height)})",
                                 f"= {_pi_text(coefficient)}"], value=coefficient, exact=False)
        return None

    # ---- units ---------------------------------------------------------------- #

    _RE_CONVERT = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*([a-z²³]+)\s*(?:=|in|into|to|as)\s+([a-z²³]+)")

    def _convert(self, low: str) -> Optional[Solution]:
        """Convert between units of one quantity, and refuse across two.

        The refusal is the part worth having. Kilometres into kilograms is not a hard conversion,
        it is a question with no answer, and a converter that multiplies by *something* rather
        than declining is the organ that filed ``('convert', '5') → 'km metres'``.
        """
        if not re.search(r"\b(?:convert|change|express|how\s+many)\b", low) \
                and " in " not in low and " into " not in low:
            return None
        match = self._RE_CONVERT.search(low)
        if match is None:
            match = re.search(r"how\s+many\s+([a-z²³]+)\s+(?:are\s+)?in\s+(-?\d+(?:\.\d+)?)\s*"
                              r"([a-z²³]+)", low)
            if match is None:
                return None
            amount, source, target = _frac(match.group(2)), match.group(3), match.group(1)
        else:
            amount, source, target = _frac(match.group(1)), match.group(2), match.group(3)
        source_unit, target_unit = _unit(source), _unit(target)
        if source_unit is None or target_unit is None:
            return None
        source_quantity, source_factor = source_unit
        target_quantity, target_factor = target_unit
        if source_quantity != target_quantity:
            raise MathError(f"{source} measures {source_quantity} and {target} measures "
                            f"{target_quantity} — there is no conversion between them")
        value = amount * source_factor / target_factor
        return self._answer("units", "convert", f"{_both(value)} {target}",
                            [f"1 {source} = {_num(source_factor / target_factor)} {target}",
                             f"{_num(amount)} × {_num(source_factor / target_factor)} "
                             f"= {_both(value)}"], value=value)

    # ---- statistics ----------------------------------------------------------- #

    _RE_STATISTIC = re.compile(
        r"\b(mean|average|median|mode|range|variance|standard\s+deviation|sd)\b")

    def _statistic(self, low: str) -> Optional[Solution]:
        """Mean, median, mode, range, variance, standard deviation — over the listed data.

        Mode is the one that can honestly have no answer: a list where every value appears once has
        no most-common value, and reporting the first one would be a wrong answer that looks
        exactly like a right one.
        """
        match = self._RE_STATISTIC.search(low)
        if match is None:
            return None
        which = match.group(1).replace("standard deviation", "sd")
        data = numbers_in(re.sub(r"\b(?:mean|average|median|mode|range|variance|"
                                 r"standard\s+deviation|sd)\b", " ", low))
        if len(data) < 2:
            return None
        ordered = sorted(data)
        if which in ("mean", "average"):
            total = sum(data, Fraction(0))
            value = total / len(data)
            return self._answer("statistics", "mean", _both(value),
                                [f"{' + '.join(_num(d) for d in data)} = {_num(total)}",
                                 f"{_num(total)} ÷ {len(data)} = {_both(value)}"], value=value)
        if which == "median":
            middle = len(ordered) // 2
            if len(ordered) % 2:
                value = ordered[middle]
                working = f"the middle value of {len(ordered)} sorted values"
            else:
                value = (ordered[middle - 1] + ordered[middle]) / 2
                working = (f"the mean of the middle two, {_num(ordered[middle - 1])} and "
                           f"{_num(ordered[middle])}")
            return self._answer("statistics", "median", _both(value),
                                [f"sorted: {', '.join(_num(d) for d in ordered)}", working],
                                value=value)
        if which == "mode":
            counts: Dict[Fraction, int] = {}
            for value in data:
                counts[value] = counts.get(value, 0) + 1
            best = max(counts.values())
            if best == 1:
                raise MathError("every value appears once, so there is no mode")
            modes = sorted(v for v, c in counts.items() if c == best)
            return self._answer("statistics", "mode", ", ".join(_num(v) for v in modes),
                                [f"the commonest value appears {best} times"], value=modes)
        if which == "range":
            value = ordered[-1] - ordered[0]
            return self._answer("statistics", "range", _num(value),
                                [f"{_num(ordered[-1])} - {_num(ordered[0])} = {_num(value)}"],
                                value=value)
        mean = sum(data, Fraction(0)) / len(data)
        variance = sum(((d - mean) ** 2 for d in data), Fraction(0)) / len(data)
        if which == "variance":
            return self._answer("statistics", "variance", _both(variance),
                                [f"mean = {_both(mean)}",
                                 f"the mean of the squared deviations = {_both(variance)}"],
                                value=variance)
        text, exact = _root_text(variance)
        return self._answer("statistics", "sd", text,
                            [f"mean = {_both(mean)}", f"variance = {_both(variance)}",
                             f"sd = √variance = {text}"], exact=exact)

    # ---- probability ---------------------------------------------------------- #

    _RE_PROBABILITY = re.compile(r"\bprobabilit(?:y|ies)\b|\bchance\b|\blikelihood\b")

    def _probability(self, low: str) -> Optional[Solution]:
        """The three probability questions a school actually asks: a die, a coin, and a bag.

        Everything else — conditional probability, without replacement, more than one draw — is
        deliberately absent rather than approximated. A probability that is plausible and wrong is
        the least detectable kind of wrong answer there is.
        """
        if not self._RE_PROBABILITY.search(low):
            return None
        if re.search(r"\b(?:die|dice|dice)\b", low):
            favourable = _die_outcomes(low)
            if favourable is None:
                return None
            value = Fraction(len(favourable), 6)
            return self._answer("probability", "die", _num(value),
                                ["a die has 6 equally likely faces",
                                 f"{len(favourable)} of them qualify: "
                                 f"{', '.join(map(str, favourable))}",
                                 f"P = {len(favourable)}/6 = {_num(value)}"], value=value)
        if re.search(r"\b(?:coin|heads?|tails?)\b", low):
            tosses = _dimension(low, ("tosses", "times", "coins")) or Fraction(1)
            if tosses != 1:
                raise MathError("she does the single-toss case only")
            return self._answer("probability", "coin", "1/2",
                                ["a fair coin has 2 equally likely outcomes",
                                 "P = 1/2"], value=Fraction(1, 2))
        counts = _coloured_counts(low)
        if not counts:
            return None
        chosen = _wanted_colour(low, counts)
        if chosen is None:
            return None
        total = sum(counts.values())
        if total == 0:
            raise MathError("an empty bag has no probability to report")
        value = Fraction(counts[chosen], total)
        return self._answer("probability", "bag", _num(value),
                            [f"the bag holds {total} things in all",
                             f"{counts[chosen]} of them are {chosen}",
                             f"P = {counts[chosen]}/{total} = {_num(value)}"], value=value)

    # ---- calculus ------------------------------------------------------------- #

    _RE_DERIVATIVE = re.compile(
        r"\b(?:derivative|differentiate|d\s*/\s*d[a-z])\b\s*(?:of\s+)?(.*)$")
    _RE_INTEGRAL = re.compile(r"\b(?:integral|integrate|antiderivative)\b\s*(?:of\s+)?(.*)$")
    _RE_LIMIT = re.compile(r"\blimit\b\s*(?:of\s+)?(.*?)\s*(?:as|when)\s+([a-z])\s*"
                           r"(?:->|→|tends?\s+to|approaches)\s*(-?\d+(?:\.\d+)?)")
    _RE_BETWEEN = re.compile(r"\b(?:from|between)\s+(-?\d+(?:\.\d+)?)\s+(?:to|and)\s+"
                             r"(-?\d+(?:\.\d+)?)")

    def _derivative(self, low: str) -> Optional[Solution]:
        match = self._RE_DERIVATIVE.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        if not _looks_algebraic(body):
            return None
        poly = Poly.parse(body)
        derivative = poly.derivative()
        return self._answer("calculus", "derivative", derivative.text(),
                            ["bring each power down and reduce it by one",
                             f"d/d{poly.symbol} ({poly.text()}) = {derivative.text()}"],
                            value=derivative)

    def _integral(self, low: str) -> Optional[Solution]:
        match = self._RE_INTEGRAL.search(low)
        if match is None:
            return None
        limits = self._RE_BETWEEN.search(low)
        body = _clean_expression(self._RE_BETWEEN.sub(" ", match.group(1)))
        if not _looks_algebraic(body):
            return None
        poly = Poly.parse(body)
        integral = poly.integral()
        if limits is None:
            return self._answer("calculus", "integral", f"{integral.text()} + C",
                                ["raise each power by one and divide by the new power",
                                 f"∫ {poly.text()} d{poly.symbol} = {integral.text()} + C"],
                                value=integral)
        lower, upper = _frac(limits.group(1)), _frac(limits.group(2))
        value = integral.at(upper) - integral.at(lower)
        return self._answer("calculus", "definite_integral", _both(value),
                            [f"∫ {poly.text()} d{poly.symbol} = {integral.text()}",
                             f"F({_num(upper)}) - F({_num(lower)}) = "
                             f"{_num(integral.at(upper))} - {_num(integral.at(lower))}",
                             f"= {_both(value)}"], value=value)

    def _limit(self, low: str) -> Optional[Solution]:
        match = self._RE_LIMIT.search(low)
        if match is None:
            return None
        body = _clean_expression(match.group(1))
        symbol, point = match.group(2), _frac(match.group(3))
        if not _looks_algebraic(body):
            return None
        if "/" in body and _symbols_in(body):
            return self._quotient_limit(body, symbol, point)
        poly = Poly.parse(body, symbol)
        value = poly.at(point)
        return self._answer("calculus", "limit", _num(value),
                            ["a polynomial is continuous, so the limit is the value there",
                             f"{poly.text()} at {symbol} = {_num(point)} is {_num(value)}"],
                            value=value)

    def _quotient_limit(self, body: str, symbol: str, point: Fraction) -> Optional[Solution]:
        """A quotient whose denominator vanishes: cancel the factor, then substitute.

        This is the only limit worth having a rule for, and it is worth having because
        substituting first gives 0/0 — which is not an answer and is not a failure either.
        """
        top_text, _, bottom_text = body.partition("/")
        top, bottom = Poly.parse(top_text, symbol), Poly.parse(bottom_text, symbol)
        if bottom.at(point) != 0:
            value = top.at(point) / bottom.at(point)
            return self._answer("calculus", "limit", _both(value),
                                ["the denominator is not zero there, so substitute",
                                 f"= {_both(value)}"], value=value)
        if top.at(point) != 0:
            raise MathError("the denominator goes to zero and the numerator does not — "
                            "there is no finite limit")
        factor = Poly({1: Fraction(1), 0: -point}, symbol)
        top_quotient, top_rest = top.divmod(factor)
        bottom_quotient, bottom_rest = bottom.divmod(factor)
        if not top_rest.is_zero or not bottom_rest.is_zero:
            raise MathError("she cannot cancel that quotient exactly")
        if bottom_quotient.at(point) == 0:
            raise MathError("cancelling once is not enough for that one")
        value = top_quotient.at(point) / bottom_quotient.at(point)
        return self._answer("calculus", "limit", _both(value),
                            [f"both sides vanish at {symbol} = {_num(point)}, so "
                             f"({symbol} - {_num(point)}) is a common factor",
                             f"after cancelling: ({top_quotient.text()}) / "
                             f"({bottom_quotient.text()})",
                             f"substituting gives {_both(value)}"], value=value)

    # ---- word problems -------------------------------------------------------- #

    _RE_SPEED = re.compile(r"\bspeed\b|\bkm\s*/\s*h\b|\bkmph\b|\bm\s*/\s*s\b")

    def _speed(self, low: str) -> Optional[Solution]:
        """Speed, distance and time — whichever of the three the sentence leaves out.

        The formula is one line; the whole of the difficulty is deciding *which* of the three is
        being asked for, and that is read from the question word rather than from the count of
        numbers in the sentence.
        """
        if not (self._RE_SPEED.search(low) or ("distance" in low and "time" in low)):
            return None
        distance = _quantity(low, ("km", "kilometres", "kilometers", "metres", "meters", "m",
                                   "miles"), ("distance", "travels", "covers", "goes"))
        time = _quantity(low, ("hours", "hour", "hrs", "hr", "minutes", "min", "seconds", "sec"),
                         ("time", "in"))
        speed = _dimension(low, ("speed", "at"))
        wants = _asked_for(low, {"speed": ("speed", "how fast"),
                                 "distance": ("distance", "how far"),
                                 "time": ("time", "how long")})
        if wants == "speed" and distance is not None and time is not None:
            if time == 0:
                raise MathError("no time has passed, so there is no speed")
            value = distance / time
            return self._answer("word", "speed", _both(value),
                                ["speed = distance ÷ time",
                                 f"= {_num(distance)} ÷ {_num(time)} = {_both(value)}"],
                                value=value)
        if wants == "distance" and speed is not None and time is not None:
            value = speed * time
            return self._answer("word", "distance", _both(value),
                                ["distance = speed × time",
                                 f"= {_num(speed)} × {_num(time)} = {_both(value)}"], value=value)
        if wants == "time" and speed is not None and distance is not None:
            if speed == 0:
                raise MathError("at no speed the journey never ends")
            value = distance / speed
            return self._answer("word", "time", _both(value),
                                ["time = distance ÷ speed",
                                 f"= {_num(distance)} ÷ {_num(speed)} = {_both(value)}"],
                                value=value)
        return None

    _RE_WORK = re.compile(r"\b(?:working\s+together|together|both\s+work)\b")

    def _work_rate(self, low: str) -> Optional[Solution]:
        """Two workers, their separate times, and how long they take together.

        The rate is the reciprocal of the time and rates add — that one sentence is the whole
        subject, and it is stated in the working because a school answer to this question is
        marked on the reasoning rather than on the number.
        """
        if not self._RE_WORK.search(low):
            return None
        if "day" not in low and "hour" not in low and "minute" not in low:
            return None
        times = [t for t in numbers_in(low) if t > 0]
        if len(times) < 2:
            return None
        rates = [Fraction(1) / t for t in times[:2]]
        together = Fraction(1) / sum(rates, Fraction(0))
        return self._answer("word", "work_rate", _both(together),
                            [f"in one unit of time they do {_num(rates[0])} and {_num(rates[1])} "
                             "of the work",
                             f"together {_num(sum(rates, Fraction(0)))} of it",
                             f"so the whole job takes {_both(together)}"], value=together)

    # ---- closed arithmetic, last of all --------------------------------------- #

    def _arithmetic(self, low: str) -> Optional[Solution]:
        """The calculator, reached last.

        Kept last for the reason given in the class docstring: its trigger is the *shape* of an
        expression rather than a word, and "the area of a rectangle 8 by 3" contains the shape of
        one. Every skill above states what it is about; this one does not, so it goes at the end
        where nothing it swallows was wanted by anything else.
        """
        calculator = self._calculate(low)
        if calculator is None:
            return None
        evaluation = calculator.evaluate(low)
        if not evaluation.ok:
            return None
        return self._answer("arithmetic", "arithmetic", evaluation.text, evaluation.steps,
                            value=evaluation.value, exact=evaluation.exact)


def _has_sympy() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _smallest_divisor(n: int) -> Optional[int]:
    """The smallest divisor above 1, or ``None`` when there is none — i.e. when ``n`` is prime."""
    n = int(n)
    if n < 2:
        return 1
    if n % 2 == 0:
        return 2 if n != 2 else None
    step = 3
    while step * step <= n:
        if n % step == 0:
            return step
        step += 2
    return None


def _factorise(n: int) -> List[int]:
    """The prime factors of ``n`` with multiplicity, smallest first. Trial division, exact."""
    n, factors = abs(int(n)), []
    if n < 2:
        return []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    step = 3
    while step * step <= n:
        while n % step == 0:
            factors.append(step)
            n //= step
        step += 2
    if n > 1:
        factors.append(n)
    return factors


def _as_powers(factors: Sequence[int]) -> str:
    """``[2, 2, 3, 7]`` as ``2^2 × 3 × 7`` — the form the question actually asked for."""
    if not factors:
        return ""
    counts: Dict[int, int] = {}
    for factor in factors:
        counts[factor] = counts.get(factor, 0) + 1
    return " × ".join(f"{p}^{c}" if c > 1 else str(p) for p, c in sorted(counts.items()))


def _integer_log(value: Fraction, base: Fraction) -> Optional[int]:
    """The exact whole-number logarithm, when there is one. Integer arithmetic, so never off by
    a float's worth — ``math.log(1000, 10)`` is 2.9999999999999996 and this is 3."""
    if value <= 0 or base <= 0 or base == 1:
        return None
    power, guess = Fraction(1), 0
    while power < value and guess < 4096:
        power *= base
        guess += 1
    return guess if power == value else None


def _maybe_decimal(value: Fraction) -> str:
    """A percentage the way percentages are written: ``37.5``, not ``75/2``.

    Only where the decimal is *exact*. A third of a class is 100/3 percent and no decimal is that
    number, so that one keeps its fraction rather than being rounded into a claim it is not.
    """
    return _dec(value) if _terminates(value) else _num(value)


def _terminates(value: Fraction) -> bool:
    """Does this rational have a finite decimal expansion? Only 2s and 5s downstairs, if so."""
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    return denominator == 1


def _common_denominator(values: Sequence[Fraction]) -> int:
    multiplier = 1
    for value in values:
        multiplier = multiplier * value.denominator // math.gcd(multiplier, value.denominator)
    return multiplier


def _number_after(text: str, marker: Any) -> Optional[Fraction]:
    """The first number written after ``marker`` matches. ``None`` when the phrase has none.

    Reading a price by *the phrase it follows* rather than by its position is what lets
    "he sold for 500 what he bought for 400" be read correctly — see :meth:`_profit_loss`.
    """
    match = marker.search(text)
    if match is None:
        return None
    tail = numbers_in(text[match.start():])
    return tail[0] if tail else None


def _number_before(text: str, marker: Any, *, exclude: Sequence[Fraction] = ()) -> Optional[Fraction]:
    """The last number written before ``marker`` matches, skipping the ones it is made of."""
    match = marker.search(text)
    if match is None:
        return None
    head = [n for n in numbers_in(text[:match.start()]) if n not in exclude]
    return head[-1] if head else None


def _pri_rate_time(low: str) -> Tuple[Fraction, Fraction, Fraction]:
    """Principal, rate and time out of an interest question, each read by what marks it.

    Position is not enough and never was: "at 8% per annum for 3 years on 5000" names them in the
    reverse of the order the formula wants. The rate is the number wearing a percent sign, the
    time is the number in front of "year", and the principal is the one that is neither.
    """
    rate_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    time_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|saal|varsh)", low)
    if rate_match is None or time_match is None:
        raise MathError("an interest question needs a rate and a time")
    rate, years = _frac(rate_match.group(1)), _frac(time_match.group(1))
    rest = [n for n in numbers_in(low) if n != rate and n != years]
    if not rest:
        raise MathError("an interest question needs a principal")
    return max(rest), rate, years


#: Words that appear inside an algebra question and are not part of the expression. A letter run
#: that is not one of these and is one character long is the unknown; see :func:`_symbols_in`.
_ALGEBRA_NOISE = re.compile(
    r"\b(?:solve|find|for|the|equation|expression|value|of|please|now|and|then|where|"
    r"give|show|working|answer|in|terms|simplest|form|over|rationals?)\b")


def _clean_expression(tail: str) -> str:
    """The algebra out of the rest of the sentence.

    Deliberately conservative: it strips the words a question is *phrased* with and then requires
    what is left to parse. Anything cleverer — guessing where an expression ends by punctuation,
    say — would sometimes guess wrong, and a wrong guess here is an answer to a different sum.
    """
    body = _ALGEBRA_NOISE.sub(" ", str(tail or "").strip())
    body = body.replace(":", " ").replace(",", " ")
    body = re.sub(r"[^0-9a-z+\-*/^(). ]+", " ", body)
    return " ".join(body.split())


def _looks_algebraic(body: str) -> bool:
    """Is this cleaned body an algebraic expression at all, as against a piece of English?

    The distinction that matters here is **not recognised** against **recognised and refused**,
    because the brain treats those differently: a refused task is still a task and is never filed
    as a fact, while an unrecognised sentence is handed back to the grounder to learn from. So an
    algebra skill handed a sentence must return ``None`` rather than raise, and this is the test.

    A run of two or more letters is a word. One letter is an unknown. An expression may contain
    any number of the second and none of the first — which is exactly the rule
    :meth:`_Parser._atom` enforces one level down, applied here early enough that the skill can
    decline instead of the parser having to object.
    """
    cleaned = str(body or "").strip()
    if not cleaned:
        return False
    if re.search(r"[a-z]{2,}", cleaned):
        return False
    return bool(_symbols_in(cleaned))


def _symbols_in(text: str) -> List[str]:
    """The unknowns in an expression: single letters that are not a stray word."""
    stripped = _ALGEBRA_NOISE.sub(" ", str(text or ""))
    found = {m for m in re.findall(r"(?<![a-z])([a-z])(?![a-z])", stripped)}
    return sorted(found)


_RE_FOR_UNKNOWN = re.compile(r"\bfor\s+([a-z])\s*[:,]?\s*")


def _equation_body(low: str) -> Optional[str]:
    """The equation out of a sentence that asks for it to be solved.

    "solve **for x:** 2x + 5 = 17" names the unknown before stating the equation, and the naming
    is not part of it. Left in, the ``x`` juxtaposes with the ``2x`` that follows and a linear
    equation is read as ``2x² + 5 = 17`` — which is a perfectly well-formed quadratic, solved
    correctly, and the answer to a question nobody asked.
    """
    if low.count("=") != 1:
        return None
    low = _RE_FOR_UNKNOWN.sub(" ", low, count=1)
    body = _clean_expression(low)
    if body.count("=") != 1:
        # `_clean_expression` drops the `=` sign, so it is put back from the original text.
        left, _, right = low.partition("=")
        body = f"{_clean_expression(left)} = {_clean_expression(right)}"
    left, _, right = body.partition("=")
    if not left.strip() or not right.strip():
        return None
    if re.search(r"[a-z]{2,}", body):
        return None       # English with an "=" in it is not an equation — see `_looks_algebraic`
    return body


def _equation_pair(low: str) -> Optional[Tuple[str, str]]:
    """Two equations out of one sentence, split on the connective rather than on the ``=``."""
    if low.count("=") != 2:
        return None
    parts = re.split(r"\s*(?:,|;|\band\b)\s*", low)
    equations = [p for p in parts if "=" in p]
    if len(equations) != 2:
        return None
    cleaned = []
    for equation in equations:
        left, _, right = equation.partition("=")
        cleaned.append(f"{_clean_expression(left)} = {_clean_expression(right)}")
    if any(not e.replace("=", "").strip() for e in cleaned):
        return None
    if any(re.search(r"[a-z]{2,}", e) for e in cleaned):
        return None
    return cleaned[0], cleaned[1]


def _linear_form(equation: str, x: str, y: str) -> Tuple[Fraction, Fraction, Fraction]:
    """``ax + by = c`` out of one linear equation in two unknowns.

    Read by substitution rather than by parsing two symbols at once: the coefficient of ``x`` is
    what the expression changes by when ``x`` goes up by one with ``y`` held at zero. That is
    exactly true for a linear expression and needs no second parser — and where the equation is
    *not* linear it comes back inconsistent and the caller's determinant refuses it.
    """
    left, _, right = equation.partition("=")
    at00 = _two_variable_value(left, x, y, 0, 0) - _two_variable_value(right, x, y, 0, 0)
    at10 = _two_variable_value(left, x, y, 1, 0) - _two_variable_value(right, x, y, 1, 0)
    at01 = _two_variable_value(left, x, y, 0, 1) - _two_variable_value(right, x, y, 0, 1)
    return at10 - at00, at01 - at00, -at00


def _two_variable_value(expression: str, x: str, y: str, at_x: Any, at_y: Any) -> Fraction:
    """Evaluate a two-unknown linear expression at a point, using the one-symbol parser twice."""
    substituted = re.sub(rf"(?<![a-z]){re.escape(y)}(?![a-z])", f"({_num(_frac(at_y))})",
                         str(expression))
    poly = Poly.parse(substituted, x)
    return poly.at(at_x)


def _bracketed(poly: Poly) -> str:
    """A factor as it is written: bracketed when it is a sum, bare when it is one term."""
    text = poly.text()
    return text if len(poly.coefficients) <= 1 else f"({text})"


def _content(poly: Poly) -> Fraction:
    """The rational common factor of every coefficient — what comes out in front on factorising."""
    numerators, denominators = [], []
    for coefficient in poly.coefficients.values():
        numerators.append(abs(coefficient.numerator))
        denominators.append(coefficient.denominator)
    if not numerators:
        return Fraction(1)
    top = 0
    for numerator in numerators:
        top = math.gcd(top, numerator)
    bottom = 1
    for denominator in denominators:
        bottom = bottom * denominator // math.gcd(bottom, denominator)
    common = Fraction(top, bottom)
    lead = poly.coefficient(poly.degree)
    return -common if lead < 0 else common


def _series_terms(low: str) -> List[Fraction]:
    """The listed terms of a sequence, taken from the run of numbers separated by commas.

    The ordinal in "the 10th term of 3, 7, 11" is not a term, and nothing in its *shape* says so —
    so the caller cuts it out of the sentence before calling this, by position. See
    :meth:`Mathematician._sequence_nth` for what happens when it is cut out by value instead.
    """
    listed = re.search(r"((?:-?\d+(?:\.\d+)?\s*[,+]\s*){2,}-?\d+(?:\.\d+)?)", low)
    if listed is not None:
        return numbers_in(listed.group(1))
    return numbers_in(low)


def _rule_of(terms: Sequence[Fraction]) -> Optional[Tuple[str, Fraction, Fraction]]:
    """What rule generates this sequence: arithmetic, geometric, squares or Fibonacci-like.

    Tried in that order, and the order is the claim: 1, 2, 4 is geometric *and* is the start of
    plenty of other sequences, and the one a school question means is the simplest that fits. A
    sequence that fits none of them returns ``None`` rather than the sequence's own last term,
    because "I cannot see the rule" is the honest answer and repeating a term is not.
    """
    if len(terms) < 3:
        return None
    differences = [b - a for a, b in zip(terms, terms[1:])]
    if len(set(differences)) == 1:
        step = differences[0]
        return (f"each term is the one before plus {_num(step)}", step, terms[-1] + step)
    if all(term != 0 for term in terms[:-1]):
        ratios = [b / a for a, b in zip(terms, terms[1:])]
        if len(set(ratios)) == 1:
            ratio = ratios[0]
            return (f"each term is the one before times {_num(ratio)}", ratio, terms[-1] * ratio)
    seconds = [b - a for a, b in zip(differences, differences[1:])]
    if len(seconds) >= 2 and len(set(seconds)) == 1:
        second = seconds[0]
        nxt = terms[-1] + differences[-1] + second
        return (f"the differences go up by {_num(second)} each time", second, nxt)
    if len(terms) >= 4 and all(a + b == c for a, b, c in zip(terms, terms[1:], terms[2:])):
        return ("each term is the sum of the two before it", Fraction(0), terms[-1] + terms[-2])
    return None


def _primes_upto_count(count: int) -> List[int]:
    """The first ``count`` primes, by sieving a bound that is provably big enough."""
    if count < 1:
        return []
    limit = max(16, int(count * (math.log(count) + math.log(max(math.log(count), 1.1))) + 8)) \
        if count > 5 else 16
    while True:
        primes = _sieve(limit)
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def _sieve(limit: int) -> List[int]:
    limit = int(limit)
    if limit > _MAX_SIEVE:
        raise MathError(f"a sieve above {_MAX_SIEVE} is not a question she will answer")
    if limit < 2:
        return []
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for candidate in range(2, math.isqrt(limit) + 1):
        if flags[candidate]:
            flags[candidate * candidate::candidate] = bytearray(
                len(flags[candidate * candidate::candidate]))
    return [n for n in range(limit + 1) if flags[n]]


#: The shapes she knows a formula for. Order matters only in that a longer name is tried first, so
#: "right-angled triangle" is a triangle rather than an unrecognised phrase.
_SHAPES: Tuple[Tuple[str, str], ...] = (
    ("parallelogram", "parallelogram"), ("trapezium", "trapezium"), ("trapezoid", "trapezium"),
    ("rectangle", "rectangle"), ("triangle", "triangle"), ("circle", "circle"),
    ("square", "square"), ("cuboid", "cuboid"), ("cube", "cube"), ("cylinder", "cylinder"),
    ("sphere", "sphere"), ("cone", "cone"),
)

_POLYGON_SIDES: Dict[str, int] = {
    "triangle": 3, "quadrilateral": 4, "rectangle": 4, "square": 4, "pentagon": 5, "hexagon": 6,
    "heptagon": 7, "octagon": 8, "nonagon": 9, "decagon": 10,
}

#: ``unit → (quantity, how many of the base unit it is)``. The quantity is the half that matters:
#: it is what makes kilometres-into-kilograms a refusal rather than a multiplication.
_UNITS: Dict[str, Tuple[str, Fraction]] = {
    # length, base metre
    "mm": ("length", Fraction(1, 1000)), "millimetre": ("length", Fraction(1, 1000)),
    "millimeter": ("length", Fraction(1, 1000)), "millimetres": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)), "centimetre": ("length", Fraction(1, 100)),
    "centimeter": ("length", Fraction(1, 100)), "centimetres": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1)), "metre": ("length", Fraction(1)),
    "meter": ("length", Fraction(1)), "metres": ("length", Fraction(1)),
    "meters": ("length", Fraction(1)),
    "km": ("length", Fraction(1000)), "kilometre": ("length", Fraction(1000)),
    "kilometer": ("length", Fraction(1000)), "kilometres": ("length", Fraction(1000)),
    "kilometers": ("length", Fraction(1000)),
    "inch": ("length", Fraction(254, 10000)), "inches": ("length", Fraction(254, 10000)),
    "foot": ("length", Fraction(3048, 10000)), "feet": ("length", Fraction(3048, 10000)),
    # mass, base gram
    "mg": ("mass", Fraction(1, 1000)), "milligram": ("mass", Fraction(1, 1000)),
    "g": ("mass", Fraction(1)), "gram": ("mass", Fraction(1)), "grams": ("mass", Fraction(1)),
    "kg": ("mass", Fraction(1000)), "kilogram": ("mass", Fraction(1000)),
    "kilograms": ("mass", Fraction(1000)),
    "quintal": ("mass", Fraction(100000)), "tonne": ("mass", Fraction(1000000)),
    "ton": ("mass", Fraction(1000000)),
    # volume, base litre
    "ml": ("volume", Fraction(1, 1000)), "millilitre": ("volume", Fraction(1, 1000)),
    "l": ("volume", Fraction(1)), "litre": ("volume", Fraction(1)),
    "liter": ("volume", Fraction(1)), "litres": ("volume", Fraction(1)),
    "kl": ("volume", Fraction(1000)), "kilolitre": ("volume", Fraction(1000)),
    # time, base second
    "second": ("time", Fraction(1)), "seconds": ("time", Fraction(1)),
    "sec": ("time", Fraction(1)), "minute": ("time", Fraction(60)),
    "minutes": ("time", Fraction(60)), "min": ("time", Fraction(60)),
    "hour": ("time", Fraction(3600)), "hours": ("time", Fraction(3600)),
    "hr": ("time", Fraction(3600)), "day": ("time", Fraction(86400)),
    "days": ("time", Fraction(86400)), "week": ("time", Fraction(604800)),
    # area, base square metre
    "cm2": ("area", Fraction(1, 10000)), "cm²": ("area", Fraction(1, 10000)),
    "m2": ("area", Fraction(1)), "m²": ("area", Fraction(1)),
    "hectare": ("area", Fraction(10000)), "km2": ("area", Fraction(1000000)),
    "km²": ("area", Fraction(1000000)),
}


def _unit(name: str) -> Optional[Tuple[str, Fraction]]:
    return _UNITS.get(str(name or "").strip().lower())


def _shape_in(low: str) -> Optional[str]:
    for word, shape in _SHAPES:
        if re.search(rf"\b{word}s?\b", low):
            return shape
    return None


def _polygon_sides(low: str) -> Optional[int]:
    for name, sides in _POLYGON_SIDES.items():
        if re.search(rf"\b{name}\b", low):
            return sides
    match = re.search(r"\b(\d+)[- ](?:sided|side)\b", low)
    return int(match.group(1)) if match else None


#: How a dimension is written in a question: after its name, or before it. Both are ordinary
#: English and both appear in textbooks, so both are read — and neither is read by position.
def _dimension(low: str, names: Sequence[str]) -> Optional[Fraction]:
    """The value of a named dimension, or ``None`` when the question does not give it."""
    for name in names:
        match = re.search(rf"\b{name}\s*(?:of|is|are|=|:|,)?\s*(-?\d+(?:\.\d+)?)", low)
        if match:
            return _frac(match.group(1))
        match = re.search(rf"(-?\d+(?:\.\d+)?)\s*(?:cm|mm|m|km|units?)?\s+(?:{name})\b", low)
        if match:
            return _frac(match.group(1))
    return None


def _dimensions(low: str, names: Sequence[str]) -> List[Fraction]:
    """Two numbers named together — "parallel sides 8 and 12" — in the order written."""
    for name in names:
        match = re.search(rf"\b{name}\b[^\d]{{0,16}}(-?\d+(?:\.\d+)?)\s*(?:and|,|&)\s*"
                          rf"(-?\d+(?:\.\d+)?)", low)
        if match:
            return [_frac(match.group(1)), _frac(match.group(2))]
    return []


def _radius(low: str) -> Optional[Fraction]:
    """The radius, from a radius or from a diameter. A question may name either."""
    radius = _dimension(low, ("radius", "r"))
    if radius is not None:
        return radius
    diameter = _dimension(low, ("diameter",))
    return diameter / 2 if diameter is not None else None


#: Colour and object words a bag-of-things probability question is written with.
_THINGS = ("red", "blue", "green", "yellow", "black", "white", "orange", "pink", "purple",
           "brown", "grey", "gray", "boys", "girls", "men", "women", "defective", "good")


def _coloured_counts(low: str) -> Dict[str, int]:
    """``{"red": 3, "blue": 5}`` out of "a bag has 3 red and 5 blue balls"."""
    counts: Dict[str, int] = {}
    for colour in _THINGS:
        match = re.search(rf"(\d+)\s+{colour}\b", low)
        if match:
            counts[colour] = int(match.group(1))
    return counts


def _wanted_colour(low: str, counts: Dict[str, int]) -> Optional[str]:
    """Which of the named things the question asks the probability *of*.

    Read from the tail of the sentence — after "probability of", where the asked-for thing always
    sits — rather than from the list, which names all of them and cannot say which one is wanted.
    """
    tail = low.split("probability", 1)[-1]
    for colour in counts:
        if re.search(rf"\b{colour}\b", tail):
            return colour
    return None


def _die_outcomes(low: str) -> Optional[List[int]]:
    """Which faces of a die the question counts as a success."""
    faces = list(range(1, 7))
    tail = low.split("probability", 1)[-1] if "probability" in low else low
    if re.search(r"\beven\b", tail):
        return [f for f in faces if f % 2 == 0]
    if re.search(r"\bodd\b", tail):
        return [f for f in faces if f % 2]
    if re.search(r"\bprime\b", tail):
        return [f for f in faces if _smallest_divisor(f) is None]
    match = re.search(r"\b(?:greater|more)\s+than\s+(\d)\b", tail)
    if match:
        return [f for f in faces if f > int(match.group(1))]
    match = re.search(r"\b(?:less|smaller)\s+than\s+(\d)\b", tail)
    if match:
        return [f for f in faces if f < int(match.group(1))]
    match = re.search(r"\b(?:getting|rolling|showing|of)\s+(?:a\s+)?(\d)\b", tail)
    if match and 1 <= int(match.group(1)) <= 6:
        return [int(match.group(1))]
    return None


def _quantity(low: str, units: Sequence[str], markers: Sequence[str]) -> Optional[Fraction]:
    """A number identified by the unit it is written in, falling back to the word that marks it."""
    for unit in units:
        match = re.search(rf"(-?\d+(?:\.\d+)?)\s*{unit}\b", low)
        if match:
            return _frac(match.group(1))
    return _dimension(low, markers)


def _asked_for(low: str, options: Dict[str, Sequence[str]]) -> Optional[str]:
    """Which of several quantities the sentence is asking for, by where its question word is.

    The *last* marker wins, not the first: "a train travels 180 km in 3 hours, what is its speed"
    mentions distance and time before naming what it wants, and every word problem in this shape
    does the same — the question comes at the end of the sentence.
    """
    best, best_at = None, -1
    for name, markers in options.items():
        for marker in markers:
            for match in re.finditer(rf"\b{marker}\b", low):
                if match.start() > best_at:
                    best, best_at = name, match.start()
    return best


#: A module-level convenience for callers that do not want to hold an instance — the counterpart
#: to :data:`nyxara.njp.calculate._DEFAULT`, and shared for the same reason: the object is
#: stateless apart from its counters, so one of it is enough.
_DEFAULT = Mathematician()


def solve(text: str) -> Solution:
    """Work out the mathematics in ``text`` using the shared mathematician."""
    return _DEFAULT.solve(text)
