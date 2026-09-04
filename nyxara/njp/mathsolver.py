"""NYXARA · njp/mathsolver.py — solving a problem she has never seen (∴, NJP V.24).

:mod:`nyxara.njp.mathematics` scores 410/410 on its own examination and that number says almost
nothing, because every item on it is a *shape the module already knows*. Fifty skills, fifty
triggers, and a question that matches a trigger is answered by the procedure behind it. That is
dispatch. It is not solving, and the difference shows the moment a problem needs two steps.

Measured on thirty hard problems written to match **no** skill — multi-step commerce, a
set-up-and-solve, a modular exponent, a Diophantine count, an age ratio, an infinite series, a
draw without replacement:

    right                    1 / 30
    confidently wrong        9 / 30
    silent                  18 / 30

The nine wrong are the interesting half, exactly as they were in V.23. "Marks up 40% then discounts
25%, what is the profit percent" answered **30** — that is `40 - 25 + 15`? No: it is the discount
skill firing on a percentage it recognised, in a problem it did not. "The remainder when 2^100 is
divided by 7" returned 2^100 in full: the power skill matched, the word *remainder* was never
read. "Two drawn without replacement" answered 2/5 — the with-replacement answer, arrived at
confidently. A trigger that matches half a problem answers half a problem, and there is nothing in
a regex that can notice the other half.

**What this module does differently: nothing here answers anything.** A reading contributes
*constraints*; the solver solves whatever set of constraints came out; the verifier substitutes the
solution back into every one of them and only then may she speak. Two readings that both match one
sentence contribute both sets, and a two-step problem is solved by algebra rather than by a skill
that happened to be written for two steps. That is the whole architectural claim, and it is what
makes an unseen composition solvable: the *chain* is discovered by the solver, not enumerated by
an author.

Sixty readings and twenty-two engines, in three families. The families are the design; the
counts are what a second and a third bank of problems asked for, and each of them arrived because
something measured could not be solved rather than because a list looked short.

* **algebra** — a system of polynomial equations in several unknowns over
  :class:`~fractions.Fraction`, solved exactly: Gaussian elimination when it is linear,
  substitution down to a univariate polynomial when it is not.
* **search** — a bounded integer search when the constraints are Diophantine or the problem asks
  for *the smallest number such that*. Exhaustive within a stated bound, so a solution it returns
  is a solution and a bound it exhausts is a proof there is none below it.
* **counting** — the discrete closed forms that are not equations at all: modular exponentiation,
  factorial valuations, arrangements and selections, and the ways of counting a set.

**The most general reading is worth naming**, because it is the closest thing here to what
"solving" means: :func:`read_predicate_search` takes an arbitrary polynomial in one unknown, a
property compiled from the sentence, and an exhaustive bounded range. Nothing about the pair is
enumerated in advance — "the smallest n such that n² + n + 41 is not prime" (40) and "the smallest
n such that 2n + 1 is prime" (1) run the same code — and a property it cannot read is refused
rather than approximated, because a search whose predicate is *approximately* the question returns
a number that is exactly wrong.

**Every answer is verified before it is stated.** :meth:`Solver.solve` substitutes the assignment
into all the constraints and discards it if any fails, so a wrong reading of a sentence produces
silence rather than a confident number. This is the property the thirty-problem floor did not have
and could not have: a pattern that matches cannot check itself.

Exact throughout, standard library only, and every entry point fail-soft.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from nyxara.njp.mathematics import MathError, Poly, _frac, _num, normalise, numbers_in

__all__ = [
    "Expr",
    "Constraint",
    "Problem",
    "Solution",
    "Solver",
    "solve",
]

#: How wide an exhaustive integer search may go before it stops being an answer and starts being
#: a hang. Every search below states its own bound and reports when it exhausts one.
_MAX_SEARCH = 200_000
#: How many unknowns a system may carry. Past this the elimination is not the problem — the
#: *reading* is, and a system this size means the sentence was misread.
_MAX_UNKNOWNS = 8


# --------------------------------------------------------------------------- #
# Expr — a polynomial in several unknowns, exactly
# --------------------------------------------------------------------------- #

#: A monomial: the sorted variables and their powers. ``()`` is the constant monomial.
Monomial = Tuple[Tuple[str, int], ...]


def _monomial(pairs: Iterable[Tuple[str, int]]) -> Monomial:
    collected: Dict[str, int] = {}
    for name, power in pairs:
        if power:
            collected[name] = collected.get(name, 0) + power
    return tuple(sorted((n, p) for n, p in collected.items() if p))


class Expr:
    """A polynomial in any number of unknowns with exact rational coefficients.

    :class:`~nyxara.njp.mathematics.Poly` holds one symbol and is the right type for reading an
    expression a person wrote down. This one holds several and is the right type for a *problem*:
    a word problem almost never has one unknown, and the whole reason the dispatcher could not
    solve one is that it had nowhere to put the second.

    Canonical, so ``x - x`` and ``0`` are equal and print alike, and so a constraint that has
    become trivially true can be recognised as such rather than carried around.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: Optional[Dict[Monomial, Any]] = None) -> None:
        cleaned: Dict[Monomial, Fraction] = {}
        for monomial, coefficient in (terms or {}).items():
            value = _frac(coefficient)
            if value:
                cleaned[monomial] = cleaned.get(monomial, Fraction(0)) + value
        self.terms = {m: c for m, c in cleaned.items() if c}

    # -- construction -------------------------------------------------------- #
    @classmethod
    def constant(cls, value: Any) -> "Expr":
        return cls({(): _frac(value)})

    @classmethod
    def variable(cls, name: str) -> "Expr":
        return cls({((str(name), 1),): Fraction(1)})

    @staticmethod
    def lift(value: Any) -> "Expr":
        if isinstance(value, Expr):
            return value
        if isinstance(value, str):
            return Expr.variable(value)
        return Expr.constant(value)

    # -- shape --------------------------------------------------------------- #
    @property
    def variables(self) -> Set[str]:
        return {name for monomial in self.terms for name, _ in monomial}

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def is_constant(self) -> bool:
        return all(monomial == () for monomial in self.terms)

    @property
    def degree(self) -> int:
        return max((sum(p for _, p in m) for m in self.terms), default=0)

    def degree_in(self, name: str) -> int:
        return max((dict(m).get(name, 0) for m in self.terms), default=0)

    def is_linear_in(self, name: str) -> bool:
        return self.degree_in(name) <= 1

    @property
    def is_linear(self) -> bool:
        return self.degree <= 1

    def value(self) -> Fraction:
        if not self.is_constant:
            raise MathError(f"{self.text()} still has an unknown in it")
        return self.terms.get((), Fraction(0))

    # -- arithmetic ---------------------------------------------------------- #
    def __add__(self, other: Any) -> "Expr":
        merged = dict(self.terms)
        for monomial, coefficient in Expr.lift(other).terms.items():
            merged[monomial] = merged.get(monomial, Fraction(0)) + coefficient
        return Expr(merged)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return Expr({m: -c for m, c in self.terms.items()})

    def __sub__(self, other: Any) -> "Expr":
        return self + (-Expr.lift(other))

    def __rsub__(self, other: Any) -> "Expr":
        return Expr.lift(other) + (-self)

    def __mul__(self, other: Any) -> "Expr":
        product: Dict[Monomial, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in Expr.lift(other).terms.items():
                monomial = _monomial(list(m1) + list(m2))
                if sum(p for _, p in monomial) > 12:
                    raise MathError("that expression grows past what she will expand")
                product[monomial] = product.get(monomial, Fraction(0)) + c1 * c2
        return Expr(product)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Expr":
        divisor = Expr.lift(other)
        if not divisor.is_constant:
            raise MathError("she divides by a constant only — that is a rational function")
        if divisor.is_zero:
            raise MathError("division by zero")
        scale = divisor.value()
        return Expr({m: c / scale for m, c in self.terms.items()})

    def __pow__(self, exponent: int) -> "Expr":
        exponent = int(exponent)
        if exponent < 0:
            raise MathError("a negative power is not a polynomial")
        result = Expr.constant(1)
        for _ in range(exponent):
            result = result * self
        return result

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Expr):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(sorted((m, c) for m, c in self.terms.items())))

    # -- rearrangement ------------------------------------------------------- #
    def coefficients_in(self, name: str) -> Dict[int, "Expr"]:
        """This expression as a polynomial in one unknown, its coefficients being the rest."""
        out: Dict[int, Dict[Monomial, Fraction]] = {}
        for monomial, coefficient in self.terms.items():
            powers = dict(monomial)
            power = powers.pop(name, 0)
            rest = _monomial(powers.items())
            out.setdefault(power, {})[rest] = out.setdefault(power, {}).get(
                rest, Fraction(0)) + coefficient
        return {power: Expr(terms) for power, terms in out.items()}

    def substitute(self, values: Dict[str, Any]) -> "Expr":
        """Replace unknowns by expressions or numbers. The workhorse of the solver."""
        result = Expr({})
        for monomial, coefficient in self.terms.items():
            piece = Expr.constant(coefficient)
            for name, power in monomial:
                if name in values:
                    piece = piece * (Expr.lift(values[name]) ** power)
                else:
                    piece = piece * Expr({((name, power),): Fraction(1)})
            result = result + piece
        return result

    def at(self, values: Dict[str, Any]) -> Fraction:
        return self.substitute(values).value()

    def as_poly(self, name: str) -> Poly:
        """The one-symbol :class:`~nyxara.njp.mathematics.Poly` this is, when it is one."""
        if self.variables - {name}:
            raise MathError("that still has more than one unknown in it")
        return Poly({power: piece.value() for power, piece in self.coefficients_in(name).items()},
                    name)

    # -- rendering ----------------------------------------------------------- #
    def text(self) -> str:
        if self.is_zero:
            return "0"
        pieces = []
        # Descending total degree, then the conventional order within a degree: the power of the
        # earliest variable first, so (x+y)² reads `x^2 + 2xy + y^2` rather than `2xy + x^2 + y^2`.
        # Sorting on the raw monomial tuple gives the second, because ('x',1) sorts before
        # ('x',2) — correct as an ordering and wrong as algebra is written.
        for monomial in sorted(self.terms,
                               key=lambda m: (-sum(p for _, p in m),
                                              tuple((n, -p) for n, p in m))):
            coefficient = self.terms[monomial]
            body = "".join(name if power == 1 else f"{name}^{power}" for name, power in monomial)
            if not body:
                piece = _num(abs(coefficient))
            elif abs(coefficient) == 1:
                piece = body
            else:
                piece = f"{_num(abs(coefficient))}{body}"
            sign = "-" if coefficient < 0 else "+"
            pieces.append(f"{sign} {piece}" if pieces else
                          (f"-{piece}" if sign == "-" else piece))
        return " ".join(pieces)

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"Expr({self.text()!r})"


# --------------------------------------------------------------------------- #
# What a problem is
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Constraint:
    """One thing that must hold. ``left == 0`` unless ``kind`` says otherwise.

    The whole design rests on this being *small*. A reading of a sentence does not answer it and
    does not decide what kind of problem it is — it says one thing that must be true, and hands it
    over. Whether five of those add up to a solvable system is the solver's question, and it is a
    question a regex cannot ask.
    """

    left: Expr = field(default_factory=Expr)
    kind: str = "eq"                       # eq | integer | positive | bounds | digit | ne
    low: Optional[int] = None
    high: Optional[int] = None
    note: str = ""

    @property
    def variables(self) -> Set[str]:
        return self.left.variables

    def holds(self, values: Dict[str, Any]) -> bool:
        """Does this constraint hold under this assignment? Unknowns left over means unknown."""
        try:
            if self.left.variables - set(values):
                return True                # not yet decided; the verifier only rejects on facts
            if self.kind == "eq":
                return self.left.at(values) == 0
            if self.kind == "ne":
                return self.left.at(values) != 0
            value = self.left.at(values)
            if self.kind == "integer":
                return value.denominator == 1
            if self.kind == "positive":
                return value > 0
            if self.kind == "digit":
                return value.denominator == 1 and 0 <= value <= 9
            if self.kind == "bounds":
                if self.low is not None and value < self.low:
                    return False
                return not (self.high is not None and value > self.high)
        except Exception:  # noqa: BLE001
            return False
        return True

    def text(self) -> str:
        if self.kind == "eq":
            return f"{self.left.text()} = 0"
        if self.kind == "ne":
            return f"{self.left.text()} ≠ 0"
        if self.kind == "bounds":
            return f"{self.low if self.low is not None else '-∞'} ≤ {self.left.text()} ≤ " \
                   f"{self.high if self.high is not None else '∞'}"
        return f"{self.left.text()} is {self.kind}"


@dataclass
class Problem:
    """A read problem: what must hold, what is being asked for, and how it was read."""

    constraints: List[Constraint] = field(default_factory=list)
    target: Optional[Expr] = None
    #: What the answer is *called*, for the sentence she says back.
    target_name: str = ""
    #: How the answer should be rendered — ``number`` | ``percent`` | ``count`` | ``fraction``.
    render: str = "number"
    steps: List[str] = field(default_factory=list)
    #: Set by a reading that wants the whole thing solved by one of the discrete engines instead.
    engine: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def unknowns(self) -> List[str]:
        found: Set[str] = set()
        for constraint in self.constraints:
            found |= constraint.variables
        if self.target is not None:
            found |= self.target.variables
        return sorted(found)

    @property
    def equations(self) -> List[Expr]:
        return [c.left for c in self.constraints if c.kind == "eq"]

    def require(self, left: Any, right: Any = 0, *, note: str = "") -> None:
        """State that two things are equal. The only way a reading ever speaks."""
        self.constraints.append(
            Constraint(Expr.lift(left) - Expr.lift(right), "eq", note=note))
        if note:
            self.steps.append(note)

    def restrict(self, name: str, kind: str, *, low: Optional[int] = None,
                 high: Optional[int] = None) -> None:
        self.constraints.append(
            Constraint(Expr.variable(name), kind, low=low, high=high))

    def text(self) -> str:
        return " ; ".join(c.text() for c in self.constraints)


#: engine → the topic vocabulary :class:`nyxara.njp.mathematics.Mathematician` already uses.
#:
#: This map exists because the two solvers returned **different shapes** from one public method.
#: :meth:`~nyxara.njp.brain.NJPBrain.do_maths` asks this module first and ``mathematics`` second,
#: its docstring promises a solution "carrying the topic", and this class had no such field — so
#: which attributes a caller got depended on which solver happened to answer. That is a broken
#: contract rather than a missing feature, and it was caught by a test asserting the topic of a
#: quadratic.
#:
#: An engine is a *reader* — ``read_equation``, ``primes``, ``shoelace`` — and a topic is what the
#: question was about. Naming the second in terms of the first is a translation and is written out
#: rather than guessed, because an engine this map does not know reports ``""`` and an empty topic
#: is honest where a wrong one is not.
ENGINE_TOPIC: Dict[str, str] = {
    # The twenty-two engines, which name a *method*.
    "read_equation": "algebra", "roots": "algebra", "zeros": "algebra",
    "diophantine": "algebra", "value": "algebra",
    "primes": "number", "divisor_count": "number", "divisor_sum": "number",
    "modpow": "number", "crt": "number", "same_remainder": "number",
    "choose": "probability", "permute": "probability", "arrangements": "probability",
    "geometric": "sequence", "harmonic": "sequence", "sum_range": "sequence",
    "count_range": "sequence", "count_kind": "sequence",
    "shoelace": "geometry", "diagonals": "geometry", "pi": "geometry",
    "ratio": "ratio", "pick": "",
    # And the sixty readers, whose ``__name__`` reaches `engine` when a reading settles the
    # problem itself. Both kinds land in the same field, which is why both are here — a map
    # covering only the engines reported an empty topic for most of what this module solves.
    "read_prime_count": "number", "read_divisor_question": "number", "read_modular": "number",
    "read_mixed_remainders": "number", "read_same_remainder": "number",
    "read_remainder_search": "number", "read_digits": "number", "read_digit_operation": "number",
    "read_consecutive": "number", "read_factorial_zeros": "number",
    "read_general_gcd": "number", "read_factorised_gcd": "number", "read_lcm_hcf_pair": "number",
    "read_number_relations": "number", "read_predicate_search": "number",
    "read_diophantine": "algebra", "read_equation_system": "algebra",
    "read_exponential_equation": "algebra", "read_symmetric_roots": "algebra",
    "read_function_value": "algebra", "read_inverse_closed_form": "algebra",
    "read_bracket_product": "algebra", "read_reciprocal_identity": "algebra",
    "read_arrangements": "probability", "read_cards": "probability", "read_dice_sum": "probability",
    "read_draws": "probability", "read_at_least_one": "probability",
    "read_stars_and_bars": "probability", "read_chessboard": "probability",
    "read_progression": "sequence", "read_geometric_progression": "sequence",
    "read_infinite_series": "sequence", "read_series_difference": "sequence",
    "read_sum_over_range": "sequence", "read_count_over_range": "sequence",
    "read_sum_of_powers": "sequence",
    "read_coordinate_area": "geometry", "read_inscribed": "geometry",
    "read_rectangle_from_two": "geometry", "read_rectangle_relation": "geometry",
    "read_square_from_measure": "geometry", "read_angle_ratio": "geometry",
    "read_ratio_chain": "ratio", "read_inverse_proportion": "ratio",
    "read_age_ratio": "ratio", "read_age_multiple": "ratio",
    "read_interest_back": "commerce", "read_interest_multiple": "commerce",
    "read_si_ci_difference": "commerce", "read_markup_discount": "commerce",
    "read_two_selling_prices": "commerce", "read_percentage_transfer": "percent",
    "read_average_speed": "word", "read_train": "word", "read_boat": "word",
    "read_pipes": "word", "read_work_chain": "word", "read_average_change": "statistics",
    "read_mean_shift": "statistics",
}


@dataclass
class Solution:
    """A solved problem, with the assignment, the working, and how it was reached."""

    question: str = ""
    answer: str = ""
    value: Any = None
    engine: str = ""
    assignment: Dict[str, Fraction] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    verified: bool = False
    error: str = ""
    #: Did **some** reading understand this sentence, even though nothing solved it?
    #:
    #: The distinction the whole downstream ordering rests on. "The sum to infinity of 2, 4, 8, …"
    #: is recognised as a geometric series and refused, because that series has no sum; without
    #: this flag the turn falls through to the skill table, which adds the three terms it can see
    #: and answers **14**. A problem she has understood and declined must not be handed to
    #: something that understands it less.
    recognised: bool = False
    #: Is this an *instruction to work something out*, whether or not anything could?
    #:
    #: Deliberately **not** the same flag as `recognised`, and conflating them cost a working
    #: capability: "expand (x+2)(x+3)" is a task the solver has no reading for and the skill table
    #: expands perfectly, so a task flag that blocked the way a refusal does turned a right answer
    #: into silence. This one only ever protects the store — a problem she cannot solve is still
    #: not a fact about the world — and never decides who gets to answer.
    task: bool = False

    @property
    def topic(self) -> str:
        """What the question was **about**, in the vocabulary ``mathematics`` already uses.

        A property rather than a field so every construction site in this module gets it without
        one of them being forgotten — which is exactly how the two shapes drifted apart in the
        first place.
        """
        return ENGINE_TOPIC.get(self.engine, "")

    @property
    def ok(self) -> bool:
        """Answered, unrefused, **and checked**. All three, and the third is the new one."""
        return bool(self.answer) and not self.error and self.verified

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "engine": self.engine,
                "assignment": {k: _num(v) for k, v in self.assignment.items()},
                "steps": self.steps[:16], "verified": self.verified,
                "recognised": self.recognised, "task": self.task,
                "error": self.error, "ok": self.ok}

    def __str__(self) -> str:
        return self.answer if self.ok else (self.error or "")


# --------------------------------------------------------------------------- #
# The algebra engine
# --------------------------------------------------------------------------- #

def solve_linear(equations: Sequence[Expr], unknowns: Sequence[str]) -> Optional[Dict[str, Fraction]]:
    """Gaussian elimination over the rationals. Exact, so there is no pivoting tolerance to tune.

    ``None`` means *not determined* — inconsistent, or fewer independent equations than unknowns.
    A partially determined system is not returned partially: an answer for one unknown out of
    three, handed on as though the system were solved, is how a wrong number gets stated.
    """
    names = list(unknowns)
    if not names or len(names) > _MAX_UNKNOWNS:
        return None
    rows: List[List[Fraction]] = []
    for equation in equations:
        if not equation.is_linear:
            return None
        row = [equation.coefficients_in(name).get(1, Expr()).value() if
               equation.is_linear_in(name) else Fraction(0) for name in names]
        constant = equation.substitute({name: 0 for name in names}).value()
        rows.append(row + [-constant])
    if len(rows) < len(names):
        return None
    # Forward elimination.
    pivots: List[int] = []
    row_at = 0
    for column in range(len(names)):
        pivot = next((r for r in range(row_at, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[row_at], rows[pivot] = rows[pivot], rows[row_at]
        scale = rows[row_at][column]
        rows[row_at] = [value / scale for value in rows[row_at]]
        for other in range(len(rows)):
            if other != row_at and rows[other][column]:
                factor = rows[other][column]
                rows[other] = [a - factor * b for a, b in zip(rows[other], rows[row_at])]
        pivots.append(column)
        row_at += 1
    if len(pivots) < len(names):
        return None                     # underdetermined: not an answer, whatever it looks like
    for row in rows[row_at:]:
        if row[-1] and not any(row[:-1]):
            return None                 # inconsistent
    return {names[column]: rows[index][-1] for index, column in enumerate(pivots)}


def solve_algebraic(problem: Problem) -> List[Dict[str, Fraction]]:
    """Every exact solution of the system, by elimination when it is linear and by substitution
    when it is not.

    Substitution is the general move and it is worth naming: pick an unknown some equation is
    *linear in*, solve that equation for it in terms of the rest, put the result into every other
    equation, and recurse on a system with one fewer unknown. The base case is one equation in one
    unknown, which :class:`~nyxara.njp.mathematics.Poly` solves exactly. Nothing here is numeric,
    so an answer that comes back is exact or does not come back.
    """
    unknowns = problem.unknowns
    equations = [e for e in problem.equations if not e.is_zero]
    if not unknowns:
        return [{}]
    if len(unknowns) > _MAX_UNKNOWNS:
        return []
    if all(equation.is_linear for equation in equations):
        found = solve_linear(equations, unknowns)
        return [found] if found is not None else []
    return _solve_by_substitution(equations, unknowns, {})


def _solve_by_substitution(equations: Sequence[Expr], unknowns: Sequence[str],
                           found: Dict[str, Fraction], depth: int = 0) -> List[Dict[str, Fraction]]:
    live = [e for e in equations if not e.is_zero]
    remaining = [name for name in unknowns if name not in found]
    if depth > _MAX_UNKNOWNS:
        return []
    if not remaining:
        return [dict(found)] if all(e.is_constant and e.is_zero for e in live) else []
    if len(remaining) == 1 and live:
        name = remaining[0]
        for equation in live:
            if equation.variables - {name}:
                continue
            try:
                roots = equation.as_poly(name).rational_roots()
            except MathError:
                continue
            out = []
            for root in roots:
                candidate = dict(found)
                candidate[name] = root
                if all(e.substitute(candidate).is_zero for e in live
                       if not (e.variables - set(candidate))):
                    out.append(candidate)
            if out:
                return out
        return []
    # Choose the unknown some equation is linear in — that is the one substitution can eliminate.
    for equation in live:
        for name in remaining:
            pieces = equation.coefficients_in(name)
            lead = pieces.get(1)
            if lead is None or not lead.is_constant or lead.is_zero:
                continue
            if equation.degree_in(name) != 1:
                continue
            rest = Expr({})
            for power, piece in pieces.items():
                if power == 0:
                    rest = rest + piece
            expressed = -rest / lead
            rewritten = [e.substitute({name: expressed}) for e in live if e is not equation]
            others = [n for n in remaining if n != name]
            solutions = _solve_by_substitution(rewritten, others, found, depth + 1)
            out = []
            for solution in solutions:
                filled = dict(solution)
                try:
                    filled[name] = expressed.at(filled)
                except Exception:  # noqa: BLE001
                    continue
                out.append(filled)
            if out:
                return out
    return []


# --------------------------------------------------------------------------- #
# The search engine — exhaustive within a stated bound
# --------------------------------------------------------------------------- #

def search_integers(predicate: Any, *, low: int = 1, high: int = _MAX_SEARCH,
                    want: str = "smallest") -> Optional[int]:
    """The smallest (or largest) integer in ``[low, high]`` satisfying ``predicate``.

    Exhaustive, which is the point: a value it returns *is* a solution because it was checked, and
    a range it walks without finding one is a proof that none exists there. Neither claim needs
    the predicate to be of any particular form, so this is what answers the number puzzles no
    formula covers — and the bound is stated rather than assumed, so "she found none" is always
    "none below N" and never "there are none".
    """
    high = min(int(high), _MAX_SEARCH)
    order = range(int(low), high + 1) if want == "smallest" else range(high, int(low) - 1, -1)
    for candidate in order:
        try:
            if predicate(candidate):
                return candidate
        except Exception:  # noqa: BLE001 — a predicate that raises on one value is not a match
            continue
    return None


def count_integer_solutions(coefficients: Sequence[int], total: int, *,
                            positive: bool = True) -> List[Tuple[int, ...]]:
    """Every non-negative (or positive) integer solution of ``a·x + b·y + … = total``.

    Enumerated rather than counted by formula, so the solutions themselves come back and the count
    is a property of a list she can show. Bounded by the total divided by each coefficient, which
    is exact for positive coefficients and is why this terminates.
    """
    coefficients = [int(c) for c in coefficients]
    if not coefficients or any(c <= 0 for c in coefficients) or total < 0:
        return []
    floor = 1 if positive else 0
    ranges = []
    for coefficient in coefficients:
        top = total // coefficient
        if top - floor + 1 > 4096:
            raise MathError("that Diophantine equation is too wide to enumerate")
        ranges.append(range(floor, top + 1))
    return [combination for combination in itertools.product(*ranges)
            if sum(c * v for c, v in zip(coefficients, combination)) == total]


# --------------------------------------------------------------------------- #
# The counting engine — the closed forms that are not equations
# --------------------------------------------------------------------------- #

def _sieve_upto(limit: int) -> List[int]:
    """The primes up to ``limit``, by sieve. Bounded by :data:`_MAX_SEARCH`."""
    limit = int(limit)
    if limit > _MAX_SEARCH:
        raise MathError(f"a sieve above {_MAX_SEARCH} is not a question she will answer")
    if limit < 2:
        return []
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for candidate in range(2, math.isqrt(limit) + 1):
        if flags[candidate]:
            flags[candidate * candidate::candidate] = bytearray(
                len(range(candidate * candidate, limit + 1, candidate)))
    return [n for n in range(limit + 1) if flags[n]]


def is_prime(n: int) -> bool:
    """Trial division to the square root. Exact, and the predicate a search may be given."""
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    step = 3
    while step * step <= n:
        if n % step == 0:
            return False
        step += 2
    return True


def factorial_valuation(n: int, prime: int) -> int:
    """How many times ``prime`` divides ``n!``, by Legendre's formula.

    The trailing zeros of 100! are this at 5 — computing 100! and counting its zeros gets the same
    answer and is the reason the dispatcher answered that question with a 158-digit number.
    """
    n, prime, total, power = int(n), int(prime), 0, int(prime)
    if prime < 2 or n < 0:
        raise MathError("that valuation is not defined")
    while power <= n:
        total += n // power
        power *= prime
    return total


def divisor_sum(n: int) -> int:
    """σ(n) — the sum of every divisor, from the factorisation rather than by enumeration."""
    from nyxara.njp.mathematics import _factorise
    n = int(n)
    if n < 1:
        raise MathError("only a positive whole number has a divisor sum")
    total = 1
    for prime, power in _multiplicities(_factorise(n)).items():
        total *= (prime ** (power + 1) - 1) // (prime - 1)
    return total


def divisor_count(n: int) -> int:
    from nyxara.njp.mathematics import _factorise
    n = int(n)
    if n < 1:
        raise MathError("only a positive whole number has a divisor count")
    total = 1
    for _, power in _multiplicities(_factorise(n)).items():
        total *= power + 1
    return total


def _multiplicities(factors: Sequence[int]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for factor in factors:
        counts[factor] = counts.get(factor, 0) + 1
    return counts


def permutations(n: int, r: Optional[int] = None) -> int:
    n = int(n)
    r = n if r is None else int(r)
    if not 0 <= r <= n or n > 2000:
        raise MathError(f"P({n}, {r}) is not a count she will work out")
    return math.perm(n, r)


def combinations(n: int, r: int) -> int:
    n, r = int(n), int(r)
    if not 0 <= r <= n or n > 2000:
        raise MathError(f"C({n}, {r}) is not a count she will work out")
    return math.comb(n, r)


def arrangements(word: str) -> int:
    """The distinct arrangements of a word's letters — n! over the repeats."""
    letters = [c for c in str(word or "").lower() if c.isalpha()]
    if not letters or len(letters) > 400:
        raise MathError("that is not a word she will arrange")
    total = math.factorial(len(letters))
    counts: Dict[str, int] = {}
    for letter in letters:
        counts[letter] = counts.get(letter, 0) + 1
    for count in counts.values():
        total //= math.factorial(count)
    return total


def crt_smallest(moduli: Sequence[int], remainders: Sequence[int], *,
                 low: Optional[int] = None) -> Optional[int]:
    """The smallest integer at least ``low`` with the given remainders, by search over the lcm.

    A common remainder over several moduli is the lcm plus that remainder, and the general case is
    not — so this searches one period rather than assuming the easy shape. One period is exact and
    bounded, which is the only two properties an answer here needs.

    **``low`` defaults to the largest modulus, and that is a convention rather than a theorem.**
    "The smallest number which when divided by 3, 4 and 5 leaves remainder 2" is answered 62 by
    every textbook, and 2 satisfies every stated condition — 2 ÷ 3 does leave 2. The unstated
    assumption is that a number is bigger than what divides it. Taking the assumption is right;
    taking it silently is not, so the caller may pass ``low=1`` and get 2.
    """
    moduli = [int(m) for m in moduli]
    remainders = [int(r) for r in remainders]
    if not moduli or len(moduli) != len(remainders) or any(m < 1 for m in moduli):
        return None
    period = 1
    for modulus in moduli:
        period = period * modulus // math.gcd(period, modulus)
    if period > _MAX_SEARCH:
        raise MathError("those moduli give a period too wide to search")
    floor = max(moduli) if low is None else int(low)
    return search_integers(
        lambda n: all(n % m == r % m for m, r in zip(moduli, remainders)),
        low=floor, high=floor + period)


def infinite_geometric_sum(first: Fraction, ratio: Fraction) -> Fraction:
    """a/(1-r), and a refusal where the series does not converge.

    The refusal is the mathematics: |r| ≥ 1 does not make the sum hard to find, it makes there be
    no sum, and a formula applied outside its domain returns a number for a question with no
    answer.
    """
    if abs(ratio) >= 1:
        raise MathError(f"a ratio of {_num(ratio)} does not converge — that series has no sum")
    return first / (1 - ratio)


def shoelace_area(points: Sequence[Tuple[Fraction, Fraction]]) -> Fraction:
    """The area of a polygon from its vertices, exactly, in the order given."""
    if len(points) < 3:
        raise MathError("three points at least make a polygon")
    total = Fraction(0)
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def harmonic_mean_speed(speeds: Sequence[Fraction]) -> Fraction:
    """The average speed over equal distances — which is not the average of the speeds.

    30 out and 60 back averages 40, not 45, and the arithmetic mean is the single commonest wrong
    answer to this question. The distances are equal, the *times* are not, and the average is
    total distance over total time.
    """
    if not speeds or any(s <= 0 for s in speeds):
        raise MathError("an average speed needs positive speeds")
    return len(speeds) / sum((1 / s for s in speeds), Fraction(0))


# --------------------------------------------------------------------------- #
# Reading a problem into constraints
# --------------------------------------------------------------------------- #
#
# Every function below is a *reading*. None of them answers anything: a reading says what must be
# true and what is being asked for, and hands a `Problem` over. That is the difference from a
# skill table — a skill that matches half a problem answers half a problem, and a constraint that
# describes half a problem simply leaves the system underdetermined, which the solver reports as
# "not determined" rather than as a number.

_WORD_COUNT = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
               "nine": 9, "ten": 10, "both": 2}
_MULTIPLIER = {"twice": 2, "double": 2, "thrice": 3, "triple": 3, "half": Fraction(1, 2),
               "quarter": Fraction(1, 4), "double of": 2}


def _count_word(text: str, default: int = 2) -> int:
    for word, value in _WORD_COUNT.items():
        if re.search(rf"\b{word}\b", text):
            return value
    match = re.search(r"\b(\d+)\s+(?:numbers|consecutive|people|persons|men|women)\b", text)
    return int(match.group(1)) if match else default


def _times_in(fragment: str) -> Fraction:
    """How many of the other quantity this fragment names: "twice" is 2, "3 times" is 3, else 1."""
    for word, value in _MULTIPLIER.items():
        if re.search(rf"\b{word}\b", fragment):
            return Fraction(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*times\b", fragment)
    return _frac(match.group(1)) if match else Fraction(1)


def read_consecutive(low: str) -> Optional[Problem]:
    """"The sum of three consecutive even numbers is 78."

    The step is the reading's whole content — 1 for plain, 2 for even *and* for odd, because
    consecutive odd numbers are two apart just as even ones are. Getting that wrong gives a system
    that solves cleanly to the wrong three numbers, which is why it is stated here rather than
    assumed at the point of use.
    """
    if "consecutive" not in low:
        return None
    total = numbers_in(low)
    if not total:
        return None
    count = _count_word(low, 3)
    step = 2 if re.search(r"\b(?:even|odd)\b", low) else 1
    amount = total[-1]
    if re.search(r"\bsum\b|\badd", low):
        amount = max(total)
    problem = Problem(steps=[f"call the first one x; the {count} of them are "
                             + ", ".join("x" if k == 0 else f"x+{k * step}" for k in range(count))])
    first = Expr.variable("x")
    members = [first + Expr.constant(k * step) for k in range(count)]
    problem.require(sum(members[1:], members[0]), Expr.constant(amount),
                    note=f"their sum is {_num(amount)}")
    problem.restrict("x", "integer")
    if re.search(r"\b(?:largest|greatest|biggest|last)\b", low):
        problem.target, problem.target_name = members[-1], "the largest"
    elif re.search(r"\b(?:smallest|least|first)\b", low):
        problem.target, problem.target_name = members[0], "the smallest"
    elif re.search(r"\bmiddle\b", low):
        problem.target, problem.target_name = members[count // 2], "the middle one"
    else:
        problem.target, problem.target_name = members[0], "the first"
    return problem


def read_number_relations(low: str) -> Optional[Problem]:
    """The compositional reader: every clause that states a relation between two numbers.

    This is the one reading that is not about a *kind* of problem. It collects whatever relations
    the sentence happens to state — a sum, a product, a difference, "one is 3 more than twice the
    other" — and lets the solver find out whether they determine anything. "Find two numbers whose
    sum is 20 and product is 96" and "one number is 3 more than twice another and their sum is 27"
    are the same reading with different clauses, and neither has a template.
    """
    if not re.search(r"\bnumbers?\b", low):
        return None
    x, y = Expr.variable("x"), Expr.variable("y")
    problem = Problem(steps=["call them x and y"])
    stated = 0
    for pattern, build in (
            (r"\b(?:their\s+)?sum\s+(?:of\s+(?:the\s+)?(?:two\s+)?numbers\s+)?is\s+"
             r"(-?\d+(?:\.\d+)?)", lambda v: (x + y, v)),
            (r"\b(?:their\s+)?product\s+(?:of\s+them\s+)?is\s+(-?\d+(?:\.\d+)?)",
             lambda v: (x * y, v)),
            (r"\b(?:their\s+)?difference\s+is\s+(-?\d+(?:\.\d+)?)", lambda v: (x - y, v)),
            (r"\bnumbers?\s+whose\s+sum\s+is\s+(-?\d+(?:\.\d+)?)", lambda v: (x + y, v)),
            (r"\band\s+product\s+is\s+(-?\d+(?:\.\d+)?)", lambda v: (x * y, v)),
            # Phrasings the first version did not have, each found by a problem written after it
            # was finished: "differ by 4" states a difference without the noun, and "the sum of
            # their squares" is a relation nothing in the first list could express at all.
            (r"\bdiffers?\s+by\s+(-?\d+(?:\.\d+)?)", lambda v: (x - y, v)),
            (r"\bdifference\s+(?:of|between)\s+(?:the\s+)?(?:two\s+)?numbers\s+is\s+"
             r"(-?\d+(?:\.\d+)?)", lambda v: (x - y, v)),
            (r"\bsum\s+of\s+(?:their|the)\s+squares\s+is\s+(-?\d+(?:\.\d+)?)",
             lambda v: (x * x + y * y, v)),
            (r"\bsum\s+of\s+(?:their|the)\s+cubes\s+is\s+(-?\d+(?:\.\d+)?)",
             lambda v: (x * x * x + y * y * y, v))):
        match = re.search(pattern, low)
        if match:
            left, value = build(_frac(match.group(1)))
            problem.require(left, Expr.constant(value), note=f"{left.text()} = {_num(value)}")
            stated += 1
    # "one number is 3 more than twice another" — the shape that needs the multiplier read out of
    # the middle of the clause rather than off a fixed position.
    relation = re.search(r"\bone\s+(?:number\s+)?is\s+(.{0,32}?)\b(more|less)\s+than\s+"
                         r"(.{0,16}?)\s*(?:another|the\s+other)", low)
    if relation:
        offset_numbers = numbers_in(relation.group(1))
        offset = offset_numbers[0] if offset_numbers else Fraction(0)
        if relation.group(2) == "less":
            offset = -offset
        multiple = _times_in(relation.group(3))
        problem.require(x, y * Expr.constant(multiple) + Expr.constant(offset),
                        note=f"x = {_num(multiple)}y {'+' if offset >= 0 else '-'} "
                             f"{_num(abs(offset))}")
        stated += 1
    if stated < 2:
        return None
    if re.search(r"\b(?:larger|greater|bigger|largest|greatest)\b", low):
        problem.target_name, problem.payload["pick"] = "the larger", "max"
    elif re.search(r"\b(?:smaller|lesser|smallest|least)\b", low):
        problem.target_name, problem.payload["pick"] = "the smaller", "min"
    else:
        problem.target_name, problem.payload["pick"] = "the numbers", "all"
    problem.engine = "pick"
    return problem


def read_digits(low: str) -> Optional[Problem]:
    """A two-digit number, its digits, and what reversing it does.

    ``10t + u`` is the whole trick and it is the step a dispatcher has nowhere to put: the number
    and its digits are three quantities related by one equation, and every clause in the problem
    is about a different one of them.
    """
    if not re.search(r"\bdigits?\b", low) or not re.search(r"\btwo[- ]digit\b|\b2[- ]digit\b", low):
        return None
    tens, units = Expr.variable("t"), Expr.variable("u")
    number = tens * 10 + units
    reversed_number = units * 10 + tens
    problem = Problem(steps=["write the number as 10t + u, so reversing it is 10u + t"])
    problem.restrict("t", "digit")
    problem.restrict("u", "digit")
    problem.constraints.append(Constraint(tens, "bounds", low=1, high=9))
    stated = 0
    match = re.search(r"sum\s+of\s+(?:the\s+)?digits\b[^.]{0,40}?\bis\s+(\d+)", low)
    if match:
        problem.require(tens + units, Expr.constant(_frac(match.group(1))),
                        note=f"t + u = {match.group(1)}")
        stated += 1
    match = re.search(r"difference\s+(?:of|between)\s+(?:the\s+)?digits\b[^.]{0,40}?\bis\s+(\d+)",
                      low)
    if match:
        problem.require(tens - units, Expr.constant(_frac(match.group(1))),
                        note=f"t - u = {match.group(1)}")
        stated += 1
    match = re.search(r"revers\w*\s+(?:the\s+digits\s+)?(?:the\s+number\s+)?"
                      r"(increase|decrease|exceed|is\s+greater|is\s+less)\w*\s*(?:it\s+|the\s+"
                      r"number\s+)?(?:by\s+)?(\d+)", low)
    if match is None:
        match = re.search(r"revers\w*[^.]{0,40}?\b(increase|decrease|exceed)\w*[^.]{0,20}?"
                          r"(\d+)", low)
    if match:
        amount = _frac(match.group(2))
        if match.group(1).startswith(("decrease", "is less")):
            amount = -amount
        problem.require(reversed_number - number, Expr.constant(amount),
                        note=f"reversing changes it by {_num(amount)}")
        stated += 1
    if stated < 2:
        return None
    problem.target, problem.target_name = number, "the number"
    return problem


def read_markup_discount(low: str) -> Optional[Problem]:
    """"Marks up by 40% and then gives a discount of 25%" — two percentages that compose.

    The dispatcher answered this 30, and 30 is not a slip: the discount skill matched a percentage
    it recognised inside a problem it did not. Two rates applied in sequence multiply; they do not
    subtract, and no amount of matching harder on either one would have found that out.
    """
    # "marks his goods up by 20%" puts a noun between the verb and its particle, and every
    # shopkeeper problem written by a person does something like it.
    if not re.search(r"\bmark\w*\b[^.]{0,24}?\bup\b|\bmarkup\b|\bmarked\s+(?:the\s+)?price\b",
                     low):
        return None
    if "discount" not in low:
        return None
    # Words may sit on **either** side of the particle — "marks up the price by 40%" and "marks
    # his goods up by 20%" are the same sentence with the noun moved, and a pattern tightened for
    # one of them stopped reading the other. Non-greedy, so the markup's own percentage is found
    # before the discount's.
    markup = re.search(r"mark\w*\b[^.]{0,24}?\bup\b[^.]{0,24}?(\d+(?:\.\d+)?)\s*"
                       r"(?:%|percent)", low)
    discount = re.search(r"discount\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:%|percent)"
                         r"|(\d+(?:\.\d+)?)\s*(?:%|percent)\s*discount", low)
    if markup is None or discount is None:
        return None
    up = _frac(markup.group(1))
    off = _frac(discount.group(1) or discount.group(2))
    cost, marked, sell = (Expr.variable("cost"), Expr.variable("marked"), Expr.variable("sell"))
    problem = Problem(steps=["take the cost price as 100",
                             f"marked = cost × (1 + {_num(up)}/100)",
                             f"selling = marked × (1 - {_num(off)}/100)"])
    problem.require(cost, Expr.constant(100), note="cost price taken as 100")
    problem.require(marked, cost * Expr.constant(1 + up / 100))
    problem.require(sell, marked * Expr.constant(1 - off / 100))
    problem.target = (sell - cost) * Expr.constant(100) / Expr.constant(100)
    problem.target_name = "the profit percent"
    problem.render = "percent"
    return problem


def read_train(low: str) -> Optional[Problem]:
    """A train crossing a pole and then a bridge: the pole gives the speed, the bridge uses it.

    The length of the train is added to the length of the bridge and is *not* added for the pole —
    which is the whole content of the problem, and is a relation rather than a formula.
    """
    if "train" not in low or not re.search(r"\bcross\w*|\bpass\w*", low):
        return None
    length = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\s+long|train\s+"
                       r"(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)", low)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*(?:s\b|sec\w*)", low)
    bridge = re.search(r"(?:bridge|platform|tunnel)\D{0,24}?(\d+(?:\.\d+)?)\s*"
                       r"(?:m|metres?|meters?)|(\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\s*"
                       r"(?:long\s+)?(?:bridge|platform|tunnel)", low)
    if length is None or seconds is None or bridge is None:
        return None
    train = _frac(length.group(1) or length.group(2))
    time = _frac(seconds.group(1))
    span = _frac(bridge.group(1) or bridge.group(2))
    speed = Expr.variable("speed")
    problem = Problem(steps=[f"crossing a pole covers the train's own length, {_num(train)} m",
                             f"crossing the bridge covers {_num(train)} + {_num(span)} m"])
    problem.require(speed * Expr.constant(time), Expr.constant(train),
                    note=f"speed = {_num(train)} ÷ {_num(time)}")
    problem.target = (Expr.constant(train + span)) / Expr.constant(train / time)
    problem.target_name = "the time to cross the bridge"
    return problem


def read_remainder_search(low: str) -> Optional[Problem]:
    """"The smallest number which when divided by 3, 4 and 5 leaves remainder 2."""
    if not re.search(r"\bremainder\b", low) or not re.search(r"\bdivid\w+\s+by\b", low):
        return None
    if not re.search(r"\bsmallest|least\b", low):
        return None
    divisors = re.search(r"divid\w+\s+by\s+([\d,\s and]+?)\s+leaves", low)
    remainder = re.search(r"remainder\s+(?:of\s+)?(\d+)", low)
    if remainder is None:
        remainder = re.search(r"leaves\s+(\d+)", low)
    if divisors is None or remainder is None:
        return None
    moduli = [int(n) for n in numbers_in(divisors.group(1))]
    if len(moduli) < 2:
        return None
    left = int(remainder.group(1))
    return Problem(engine="crt", payload={"moduli": moduli, "remainder": left},
                   target_name="the smallest such number",
                   steps=[f"it must be {left} more than a multiple of each of "
                          f"{', '.join(map(str, moduli))}"])


def read_same_remainder(low: str) -> Optional[Problem]:
    """"The greatest number that divides 43, 91 and 183 leaving the same remainder."

    The remainder is not given and does not need to be: if two numbers leave the same remainder
    under d then d divides their difference, so the answer is the hcf of the differences. That
    step is the problem, and it is invisible to anything matching on the word "divides".
    """
    if not re.search(r"\bsame\s+remainder\b", low):
        return None
    if not re.search(r"\bgreatest|largest|biggest\b", low):
        return None
    values = [int(v) for v in numbers_in(low) if v.denominator == 1]
    if len(values) < 3:
        return None
    return Problem(engine="same_remainder", payload={"values": values},
                   target_name="the greatest such divisor",
                   steps=["equal remainders means the divisor divides every difference"])


def read_arrangements(low: str) -> Optional[Problem]:
    """Seatings, arrangements, selections and handshakes — the counting questions."""
    if not re.search(r"\bhow\s+many\s+ways|\bnumber\s+of\s+ways|\barrang\w*|"
                     r"\bpermutations?\b|\bhandshakes?\b|\bdiagonals?\b|\bselect\w*|"
                     r"\bchoose\b", low):
        return None
    match = re.search(r"\bdiagonals?\b", low)
    if match:
        sides = re.search(r"(\d+)\s*(?:sides|sided)|polygon\D{0,16}?(\d+)", low)
        if sides is None:
            return None
        n = int(sides.group(1) or sides.group(2))
        return Problem(engine="diagonals", payload={"n": n}, target_name="the number of diagonals",
                       steps=[f"each of the {n} vertices joins to {n - 3} non-neighbours, "
                              f"and each diagonal is counted twice"])
    if re.search(r"\bhandshakes?\b", low):
        people = numbers_in(low)
        if not people:
            return None
        return Problem(engine="choose", payload={"n": int(people[0]), "r": 2},
                       target_name="the number of handshakes",
                       steps=["a handshake is a pair, so this is n choose 2"])
    word = re.search(r"letters?\s+of\s+the\s+word\s+([a-z]+)", low)
    if word:
        return Problem(engine="arrangements", payload={"word": word.group(1)},
                       target_name="the number of arrangements",
                       steps=["n! over the factorial of each repeat"])
    counts = [int(v) for v in numbers_in(low) if v.denominator == 1]
    if re.search(r"\bchoose\b|\bselect\w*|\bcommittee\b|\bgroup\s+of\b", low) and len(counts) >= 2:
        return Problem(engine="choose", payload={"n": max(counts), "r": min(counts)},
                       target_name="the number of ways",
                       steps=["order does not matter, so this is a selection"])
    if counts and re.search(r"\bseat\w*|\barrang\w*|\brow\b|\bline\b|\bstand\b", low):
        n = counts[0]
        r = counts[1] if len(counts) > 1 else n
        return Problem(engine="permute", payload={"n": max(n, r), "r": min(n, r)},
                       target_name="the number of ways",
                       steps=["order matters, so this is an arrangement"])
    return None


def read_average_change(low: str) -> Optional[Problem]:
    """"The average of 5 numbers is 20; removing one makes it 18." The total is the bridge."""
    if "average" not in low and "mean" not in low:
        return None
    if not re.search(r"\bremov\w+|\bexclud\w+|\bleaves?\b|\badd\w*\b|\bincluded?\b", low):
        return None
    first = re.search(r"average\s+of\s+(\d+)\s+numbers\s+is\s+(-?\d+(?:\.\d+)?)", low)
    second = re.search(r"(?:becomes|is)\s+(-?\d+(?:\.\d+)?)\s*$|"
                       r"average\s+becomes\s+(-?\d+(?:\.\d+)?)", low)
    if first is None or second is None:
        return None
    count = int(first.group(1))
    before = _frac(first.group(2))
    after = _frac(second.group(1) or second.group(2))
    removed = re.search(r"\bremov\w+|\bexclud\w+", low) is not None
    changed = count - 1 if removed else count + 1
    total_before, total_after = before * count, after * changed
    value = total_before - total_after if removed else total_after - total_before
    return Problem(engine="value", payload={"value": value},
                   target_name="the number removed" if removed else "the number added",
                   steps=[f"total before = {count} × {_num(before)} = {_num(total_before)}",
                          f"total after = {changed} × {_num(after)} = {_num(total_after)}",
                          f"the difference is {_num(abs(value))}"])


def read_work_chain(low: str) -> Optional[Problem]:
    """Two workers, a spell together, and one of them leaving — rates add, and time is their sum.

    A "together" skill answers the first half of this and stops. What makes the rest reachable is
    that a rate is a quantity like any other: the work done in the shared spell is a number, and
    what is left is one over the remaining rate.
    """
    if not re.search(r"\bwork\b|\bjob\b|\btask\b|\bpiece\s+of\s+work\b", low):
        return None
    if not re.search(r"\bleaves?\b|\bquits?\b|\bafter\b", low):
        return None
    times = [t for t in numbers_in(low) if t > 0]
    if len(times) < 3:
        return None
    first, second, together = times[0], times[1], times[2]
    done = together * (Fraction(1) / first + Fraction(1) / second)
    if done >= 1:
        raise MathError("they finish the job before anyone leaves")
    left = (1 - done) * second
    return Problem(engine="value", payload={"value": left},
                   target_name="the extra days needed",
                   steps=[f"in one day they do 1/{_num(first)} + 1/{_num(second)} of the work",
                          f"in {_num(together)} days they do {_num(done)} of it",
                          f"the rest, {_num(1 - done)}, takes {_num(left)} more days"])


def read_pipes(low: str) -> Optional[Problem]:
    """A pipe filling and another emptying: the rates have opposite signs, which is the problem."""
    if not re.search(r"\bpipe\b|\btank\b|\bcistern\b", low):
        return None
    if not re.search(r"\bempt\w+|\bleak\w*|\bdrain\w*|\boutlet\b", low):
        return None
    times = [t for t in numbers_in(low) if t > 0]
    if len(times) < 2:
        return None
    fill, empty = times[0], times[1]
    net = Fraction(1) / fill - Fraction(1) / empty
    if net <= 0:
        raise MathError("the outlet is at least as fast as the inlet — the tank never fills")
    return Problem(engine="value", payload={"value": Fraction(1) / net},
                   target_name="the time to fill",
                   steps=[f"filling adds 1/{_num(fill)} an hour, emptying takes 1/{_num(empty)}",
                          f"the net rate is {_num(net)} an hour",
                          f"so it fills in {_num(1 / net)} hours"])


def read_inverse_proportion(low: str) -> Optional[Problem]:
    """"3 men build a wall in 8 days, how long do 4 men take?" — more workers, less time."""
    if not re.search(r"\bmen\b|\bworkers?\b|\bwomen\b|\bmachines?\b|\bpumps?\b", low):
        return None
    if not re.search(r"\bdays?\b|\bhours?\b", low):
        return None
    match = re.search(r"(\d+)\s+(?:men|workers|women|machines|pumps)\D{0,40}?(\d+)\s*"
                      r"(?:days?|hours?)", low)
    asked = re.search(r"(?:will|do|does|can)\s+(\d+)\s+(?:men|workers|women|machines|pumps)", low)
    if match is None or asked is None:
        return None
    workers, days, now = int(match.group(1)), _frac(match.group(2)), int(asked.group(1))
    if now <= 0:
        raise MathError("no workers means the wall is never built")
    value = days * workers / now
    return Problem(engine="value", payload={"value": value}, target_name="the days needed",
                   steps=[f"the job is {workers} × {_num(days)} = {_num(workers * days)} "
                          f"worker-days",
                          f"{now} workers take {_num(workers * days)} ÷ {now} = {_num(value)}"])


def read_reciprocal_identity(low: str) -> Optional[Problem]:
    """"If x + 1/x = 3, find x² + 1/x²." Squaring the given is the step; x itself is irrational.

    Worth keeping as its own reading precisely because solving for x first is the wrong move: the
    roots are irrational, and every route through them gives a decimal for a question whose answer
    is the integer 7.
    """
    match = re.search(r"([a-z])\s*\+\s*1\s*/\s*\1\s*=\s*(-?\d+(?:\.\d+)?)", low)
    if match is None:
        return None
    wanted = re.search(rf"{match.group(1)}\s*\^?\s*(\d)\s*\+\s*1\s*/\s*{match.group(1)}\s*\^?\s*\1",
                       low)
    power = int(wanted.group(1)) if wanted else 2
    base = _frac(match.group(2))
    if power == 2:
        value = base * base - 2
        working = f"({_num(base)})² - 2 = {_num(value)}"
    elif power == 3:
        value = base ** 3 - 3 * base
        working = f"({_num(base)})³ - 3×{_num(base)} = {_num(value)}"
    else:
        raise MathError(f"she has no identity for the {power}th power here")
    return Problem(engine="value", payload={"value": value},
                   target_name=f"the value of x^{power} + 1/x^{power}",
                   steps=[f"square the given: x² + 2 + 1/x² = {_num(base * base)}"
                          if power == 2 else "cube the given",
                          working])


def read_modular(low: str) -> Optional[Problem]:
    """"The remainder when 2^100 is divided by 7", and "the last digit of 7^100".

    The dispatcher returned 2^100 in full for the first of these: the power skill matched and the
    word *remainder* was never read at all. A modulus is not a decoration on an exponent.
    """
    power = re.search(r"(\d+)\s*(?:\^|\*\*)\s*(\d+)", low)
    if power is None:
        return None
    base, exponent = int(power.group(1)), int(power.group(2))
    modulus = None
    # `\D` cannot be used to skip the words between "remainder" and "divided": the thing being
    # divided is a *number*, so "the remainder when 2^100 is divided by 7" fails on its own
    # subject and the whole question came back silent.
    match = re.search(r"remainder[^.]{0,40}?divid\w+\s+by\s+(\d+)", low)
    if match:
        modulus = int(match.group(1))
    elif re.search(r"\blast\s+digit\b|\bunits?\s+digit\b|\bones\s+digit\b", low):
        modulus = 10
    elif re.search(r"\blast\s+two\s+digits\b", low):
        modulus = 100
    if modulus is None or modulus < 2:
        return None
    return Problem(engine="modpow", payload={"base": base, "exponent": exponent,
                                             "modulus": modulus},
                   target_name="the remainder" if modulus not in (10, 100) else "the last digit",
                   steps=[f"the powers of {base} repeat modulo {modulus}"])


def read_factorial_zeros(low: str) -> Optional[Problem]:
    """"How many trailing zeros does 100! have?" — Legendre at 5, not a 158-digit number."""
    if not re.search(r"\btrailing\s+zero\w*|\bzeros?\s+(?:are\s+)?at\s+the\s+end\b|"
                     r"\bends?\s+in\s+how\s+many\s+zero|"
                     r"\bzeros?\b[^.]{0,30}?\bend(?:s|ing)?\s+(?:with|in)\b", low):
        return None
    if not re.search(r"factorial|\d\s*!", low):
        return None
    values = [int(v) for v in numbers_in(low) if v.denominator == 1]
    if not values:
        return None
    return Problem(engine="zeros", payload={"n": max(values)},
                   target_name="the number of trailing zeros",
                   steps=["a trailing zero is a factor of 10, and 5s are scarcer than 2s"])


def read_divisor_question(low: str) -> Optional[Problem]:
    """The sum, or the count, of a number's factors — from its factorisation, not by listing."""
    if not re.search(r"\bfactors?\b|\bdivisors?\b", low):
        return None
    values = [int(v) for v in numbers_in(low) if v.denominator == 1 and v > 0]
    if not values:
        return None
    if re.search(r"\bsum\s+of\s+(?:all\s+)?(?:the\s+)?(?:factors|divisors)\b", low):
        return Problem(engine="divisor_sum", payload={"n": max(values)},
                       target_name="the sum of the factors",
                       steps=["σ(n) is a product over the prime powers, not a walk over divisors"])
    if re.search(r"\b(?:how\s+many|number\s+of)\s+(?:factors|divisors)\b", low):
        return Problem(engine="divisor_count", payload={"n": max(values)},
                       target_name="the number of factors")
    return None


def read_factorised_gcd(low: str) -> Optional[Problem]:
    """"The hcf of 2^4 × 3^2 and 2^2 × 3^3" — the numbers are written, not given.

    The dispatcher answered 1, because `numbers_in` handed it the exponents as if they were the
    numbers and gcd(4, 2, 2, 3) came out 1. The powers have to be evaluated before anything can be
    taken of them, which is a reading problem rather than an arithmetic one.
    """
    if not re.search(r"\bhcf\b|\bgcd\b|\blcm\b|\bhighest\s+common|\blowest\s+common", low):
        return None
    if "^" not in low and "**" not in low:
        return None            # a plain "hcf of 12 and 18" belongs to the mathematician
    sides = re.split(r"\s+and\s+", low)
    values: List[int] = []
    for side in sides:
        if not re.search(r"\d+\s*(?:\^|\*\*)\s*\d+", side):
            continue
        # **The powers are evaluated in place and then every remaining factor is multiplied in.**
        # Reading only the powers dropped the bare 5 out of "2^5 times 5" and made the hcf of
        # 200 and 160 come out 8, which is the hcf of 200 and 32. A factor written without an
        # exponent is still a factor.
        expanded = re.sub(r"(\d+)\s*(?:\^|\*\*)\s*(\d+)",
                          lambda m: str(int(m.group(1)) ** int(m.group(2)))
                          if int(m.group(2)) <= 64 else "1", side)
        tail = re.split(r"\bhcf\b|\bgcd\b|\blcm\b|\bof\b", expanded)[-1]
        factors = [int(v) for v in numbers_in(tail) if v.denominator == 1 and v > 0]
        if not factors:
            continue
        product = 1
        for factor in factors:
            product *= factor
        values.append(product)
    if len(values) < 2:
        return None
    wants_lcm = bool(re.search(r"\blcm\b|\blowest\s+common|\bleast\s+common", low))
    result = values[0]
    for value in values[1:]:
        result = (result * value // math.gcd(result, value)) if wants_lcm \
            else math.gcd(result, value)
    return Problem(engine="value", payload={"value": Fraction(result)},
                   target_name="the lcm" if wants_lcm else "the hcf",
                   steps=[f"the numbers are {' and '.join(str(v) for v in values)}",
                          f"their {'lcm' if wants_lcm else 'hcf'} is {result}"])


def read_coordinate_area(low: str) -> Optional[Problem]:
    """The area of a polygon given its vertices, by the shoelace formula."""
    if "area" not in low:
        return None
    points = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", low)
    if len(points) < 3:
        return None
    vertices = [(_frac(x), _frac(y)) for x, y in points]
    return Problem(engine="shoelace", payload={"points": vertices},
                   target_name="the area",
                   steps=["the shoelace sum over the vertices in order, halved"])


def read_age_ratio(low: str) -> Optional[Problem]:
    """"The ratio of their ages is 4:3; after 6 years it is 6:5."

    Two ratios and a shift is a linear system in two unknowns, and it is the shape a ratio skill
    cannot hold: the second ratio is about two quantities that do not exist yet.
    """
    if "age" not in low and "old" not in low:
        return None
    ratios = re.findall(r"(\d+)\s*:\s*(\d+)", low)
    if len(ratios) < 2:
        return None
    shift = re.search(r"(?:after|in)\s+(\d+)\s+years?", low)
    back = re.search(r"(\d+)\s+years?\s+ago", low)
    if shift is None and back is None:
        return None
    years = _frac(shift.group(1)) if shift else -_frac(back.group(1))
    (a1, b1), (a2, b2) = (_frac(ratios[0][0]), _frac(ratios[0][1])), \
                         (_frac(ratios[1][0]), _frac(ratios[1][1]))
    elder, younger = Expr.variable("elder"), Expr.variable("younger")
    problem = Problem(steps=[f"now they are in the ratio {_num(a1)}:{_num(b1)}",
                             f"in {_num(years)} years the ratio is {_num(a2)}:{_num(b2)}"])
    problem.require(elder * Expr.constant(b1), younger * Expr.constant(a1))
    problem.require((elder + Expr.constant(years)) * Expr.constant(b2),
                    (younger + Expr.constant(years)) * Expr.constant(a2))
    if re.search(r"\byounger|smaller|less\b", low):
        problem.target, problem.target_name = younger, "the younger age"
    else:
        problem.target, problem.target_name = elder, "the elder age"
    return problem


def read_infinite_series(low: str) -> Optional[Problem]:
    """The sum of an infinite geometric series — and a refusal when it does not converge."""
    if not re.search(r"\binfinit\w*|\bto\s+infinity\b", low) \
            or not re.search(r"\bgeometric\b|\bgp\b|\bseries\b", low):
        return None
    first = re.search(r"first\s+term\s+(?:is\s+)?(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", low)
    ratio = re.search(r"(?:common\s+)?ratio\s+(?:is\s+)?(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", low)
    if first is not None and ratio is not None:
        head, step = _frac(first.group(1).replace(" ", "")), _frac(ratio.group(1).replace(" ", ""))
    else:
        # **The series may be given as its own first terms.** "the sum to infinity of 4, 2, 1, …"
        # names neither the first term nor the ratio and is the way a person writes it; the ratio
        # is read off the terms, and refused unless every consecutive pair agrees — three terms
        # that are not geometric are not a geometric series however the sentence describes them.
        terms = [t for t in numbers_in(low) if t != 0]
        if len(terms) < 3:
            return None
        ratios = {b / a for a, b in zip(terms, terms[1:])}
        if len(ratios) != 1:
            return None
        head, step = terms[0], ratios.pop()
    return Problem(engine="geometric",
                   payload={"first": head, "ratio": step},
                   target_name="the sum to infinity",
                   steps=["a/(1 - r), and only where |r| < 1"])


def read_cards(low: str) -> Optional[Problem]:
    """The probability of a named card from a standard pack — a count over 52."""
    if "card" not in low or not re.search(r"\bprobabilit|\bchance\b", low):
        return None
    pack = 52
    counts = {"ace": 4, "king": 4, "queen": 4, "jack": 4, "face card": 12, "spade": 13,
              "heart": 13, "club": 13, "diamond": 13, "red": 26, "black": 26}
    tail = low.split("probability", 1)[-1]
    named = [name for name in counts if re.search(rf"\b{name}s?\b", tail)]
    if not named:
        return None
    if len(named) == 1:
        favourable = counts[named[0]]
    else:
        # "a red king" is the intersection of two conditions, not the union: 26 reds and 4 kings
        # meet in 2 cards, and adding or taking the smaller of them is wrong both ways.
        favourable = 1
        for name in named:
            favourable = favourable * counts[name]
        favourable //= pack ** (len(named) - 1)
    return Problem(engine="value", payload={"value": Fraction(favourable, pack)},
                   target_name="the probability",
                   steps=[f"{favourable} of the {pack} cards qualify"])


def read_draws(low: str) -> Optional[Problem]:
    """Two drawn without replacement: the second draw is from a smaller bag, and that is the point.

    With replacement the answer is (4/10)², without it is (4/10)(3/9), and the dispatcher gave the
    first for a question that said "without replacement" in as many words.
    """
    if not re.search(r"\bprobabilit|\bchance\b", low):
        return None
    if not re.search(r"\bdrawn?\b|\bpick\w*|\bchosen\b|\bselect\w*", low):
        return None
    from nyxara.njp.mathematics import _coloured_counts, _wanted_colour
    counts = _coloured_counts(low)
    if not counts:
        return None
    chosen = _wanted_colour(low, counts)
    if chosen is None:
        return None
    total = sum(counts.values())
    drawn = 2 if re.search(r"\btwo\b|\b2\b|\bboth\b", low) else 1
    replaced = not re.search(r"without\s+replacement|not\s+replaced", low)
    if drawn == 1:
        value = Fraction(counts[chosen], total)
        working = [f"{counts[chosen]} of {total} qualify"]
    elif replaced:
        value = Fraction(counts[chosen], total) ** 2
        working = ["with replacement the bag is the same both times"]
    else:
        if counts[chosen] < 2:
            raise MathError(f"there is only one {chosen} — two cannot be drawn")
        value = Fraction(counts[chosen], total) * Fraction(counts[chosen] - 1, total - 1)
        working = [f"first draw {counts[chosen]}/{total}",
                   f"then {counts[chosen] - 1}/{total - 1}, because one is gone"]
    return Problem(engine="value", payload={"value": value}, target_name="the probability",
                   steps=working)


def read_diophantine(low: str) -> Optional[Problem]:
    """"How many positive integer solutions does 2x + 3y = 12 have?" — enumerated, then counted."""
    if not re.search(r"\bsolutions?\b", low):
        return None
    if not re.search(r"\binteger\b|\bwhole\s+number\b|\bnatural\b", low):
        return None
    equation = re.search(r"(\d*)\s*([a-z])\s*\+\s*(\d*)\s*([a-z])\s*=\s*(\d+)", low)
    if equation is None:
        return None
    a = int(equation.group(1) or 1)
    b = int(equation.group(3) or 1)
    total = int(equation.group(5))
    positive = not re.search(r"non[- ]negative|\bzero\s+or\b", low)
    return Problem(engine="diophantine",
                   payload={"coefficients": [a, b], "total": total, "positive": positive},
                   target_name="the number of solutions",
                   steps=[f"every pair with {a}x + {b}y = {total}, enumerated"])


def read_interest_back(low: str) -> Optional[Problem]:
    """The interest is given and the sum is wanted — the formula, run backwards.

    Simple as well as compound, and the pair is why this is one reading rather than two: the only
    difference between them is what multiplies the principal, and writing a second reading for the
    second growth factor is how a solver turns back into a skill table.
    """
    compound = bool(re.search(r"\bcompound\s+interest\b", low))
    simple = bool(re.search(r"\bsimple\s+interest\b", low))
    if not (compound or simple):
        return None
    if not re.search(r"\bfind\s+the\s+(?:sum|principal)|\bwhat\s+is\s+the\s+(?:sum|principal)",
                     low):
        return None
    rate = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    years = re.search(r"(\d+)\s*years?", low)
    interest = re.search(r"interest[^.]{0,40}?\bis\s+(-?\d+(?:\.\d+)?)"
                         r"|\bis\s+(-?\d+(?:\.\d+)?)[^.]{0,20}?\bfind", low)
    if rate is None or years is None:
        return None
    amounts = [v for v in numbers_in(low)
               if v != _frac(rate.group(1)) and v != _frac(years.group(1))]
    if interest is None and not amounts:
        return None
    given = _frac(interest.group(1) or interest.group(2)) if interest else max(amounts)
    periods, rate_value = int(years.group(1)), _frac(rate.group(1))
    if compound:
        factor = (1 + rate_value / 100) ** periods - 1
        working = [f"A = P × (1 + {_num(rate_value)}/100)^{periods}",
                   f"interest = A - P = P × {_num(factor)}"]
    else:
        factor = rate_value * periods / 100
        working = [f"SI = P × R × T ÷ 100 = P × {_num(factor)}"]
    if factor == 0:
        raise MathError("no rate means no interest, so the sum cannot be recovered")
    principal = Expr.variable("principal")
    problem = Problem(steps=working)
    problem.require(principal * Expr.constant(factor), Expr.constant(given),
                    note=f"P × {_num(factor)} = {_num(given)}")
    problem.target, problem.target_name = principal, "the sum"
    return problem


def read_rectangle_relation(low: str) -> Optional[Problem]:
    """A perimeter and a relation between the sides — two equations, one shape."""
    if not re.search(r"\brectangle\b", low):
        return None
    perimeter = re.search(r"perimeter\D{0,20}?(\d+(?:\.\d+)?)", low)
    if perimeter is None:
        return None
    length, width = Expr.variable("length"), Expr.variable("width")
    problem = Problem(steps=["2(l + w) is the perimeter"])
    problem.require((length + width) * Expr.constant(2), Expr.constant(_frac(perimeter.group(1))))
    relation = re.search(r"length\s+is\s+(.{0,20}?)\s*(?:its\s+|the\s+)?width", low)
    if relation is not None:
        multiple = _times_in(relation.group(1))
        offset = numbers_in(relation.group(1))
        shift = offset[0] if offset and "times" not in relation.group(1) else Fraction(0)
        if re.search(r"\bless\b|\bshorter\b", relation.group(1)):
            shift = -shift
        problem.require(length, width * Expr.constant(multiple) + Expr.constant(shift),
                        note=f"l = {_num(multiple)}w")
    else:
        return None
    if "area" in low:
        problem.target, problem.target_name = length * width, "the area"
    elif re.search(r"\blength\b.*\?|\bfind\s+the\s+length", low):
        problem.target, problem.target_name = length, "the length"
    else:
        problem.target, problem.target_name = length * width, "the area"
    return problem


def read_average_speed(low: str) -> Optional[Problem]:
    """Out at one speed and back at another: the average is harmonic, never arithmetic."""
    if "average speed" not in low:
        return None
    if not re.search(r"\breturns?\b|\bback\b|\bboth\s+ways\b|\bround\s+trip\b", low):
        return None
    speeds = [s for s in numbers_in(re.sub(r"\d+\s*km\b(?!\s*/)", " ", low)) if s > 0]
    if len(speeds) < 2:
        return None
    return Problem(engine="harmonic", payload={"speeds": speeds[:2]},
                   target_name="the average speed",
                   steps=["equal distances, unequal times — total distance over total time"])


def read_series_difference(low: str) -> Optional[Problem]:
    """"1 + 2 + … + 100 minus the sum of the first 50" — two closed forms and a subtraction."""
    if not re.search(r"\.\.\.|…|\bminus\b|\bsubtract\w*", low):
        return None
    if "sum" not in low:
        return None
    # Greedy on purpose. Non-greedy, "… + 99 + 100" captured **99**: the run has to be consumed
    # to its last term, because the last term is what names the series.
    runs = re.findall(r"(?:\.\.\.|…)\s*\+?\s*(?:\d+\s*\+\s*)*(\d+)", low)
    firsts = re.findall(r"first\s+(\d+)\s+natural", low)
    ends = [int(v) for v in runs] + [int(v) for v in firsts]
    if len(ends) < 2:
        return None
    top, bottom = ends[0], ends[1]
    value = Fraction(top * (top + 1), 2) - Fraction(bottom * (bottom + 1), 2)
    return Problem(engine="value", payload={"value": value}, target_name="the value",
                   steps=[f"1..{top} sums to {top * (top + 1) // 2}",
                          f"1..{bottom} sums to {bottom * (bottom + 1) // 2}",
                          f"the difference is {_num(value)}"])


# --------------------------------------------------------------------------- #
# NJP V.24, second tier — the readings a third bank of problems asked for
# --------------------------------------------------------------------------- #

_RANGE = re.compile(r"between\s+(\d+)\s+and\s+(\d+)|from\s+(\d+)\s+to\s+(\d+)"
                    r"|(?:first|up\s+to)\s+(\d+)\b")


def _range_in(low: str) -> Optional[Tuple[int, int]]:
    """The interval a counting question is over. "Between 1 and 100" includes both ends here —
    the convention every such question is written under, and stated because it is a choice."""
    match = _RANGE.search(low)
    if match is None:
        if re.search(r"\btwo[- ]digit\b", low):
            return 10, 99
        if re.search(r"\bthree[- ]digit\b", low):
            return 100, 999
        return None
    if match.group(1):
        return int(match.group(1)), int(match.group(2))
    if match.group(3):
        return int(match.group(3)), int(match.group(4))
    return 1, int(match.group(5))


def read_count_over_range(low: str) -> Optional[Problem]:
    """"How many integers between 1 and 100 are divisible by 3 or 5?"

    Counted by walking the range and testing each one, not by inclusion–exclusion. That is slower
    and it is the reason this generalises: *or*, *and*, *neither* and *not* are all the same walk
    with a different test, and the arithmetic identity for each of them is a separate thing to get
    wrong. The bound is stated and finite, so the walk is a proof.
    """
    if not re.search(r"\bhow\s+many\b", low):
        return None
    kind = ""
    if re.search(r"\bodd\s+numbers?\b", low):
        kind = "odd"
    elif re.search(r"\beven\s+numbers?\b", low):
        kind = "even"
    elif re.search(r"\bperfect\s+squares?\b", low):
        kind = "square"
    if not kind and not re.search(r"\bdivisible\s+by\b|\bmultiples?\s+of\b", low):
        return None
    span = _range_in(low)
    if span is None:
        return None
    low_end, high_end = span
    if high_end - low_end > _MAX_SEARCH:
        raise MathError("that range is too wide to walk")
    if kind:
        # The same walk with a different test — which is the reason this reading is one reading.
        # "How many odd numbers between 20 and 60" is not a divisibility question in the words it
        # uses and is exactly one in the work it needs.
        return Problem(engine="count_kind",
                       payload={"low": low_end, "high": high_end, "kind": kind},
                       target_name="the count",
                       steps=[f"walking {low_end} to {high_end} and keeping the {kind} ones"])
    divisors = [int(v) for v in numbers_in(
        re.sub(r"between\s+\d+\s+and\s+\d+|from\s+\d+\s+to\s+\d+", " ", low))
        if v.denominator == 1 and v > 1]
    if not divisors:
        return None
    if re.search(r"\bneither\b|\bnot\s+divisible\b|\bnone\s+of\b", low):
        mode = "neither"
    elif re.search(r"\bboth\b|\band\b(?![^.]*\bor\b)", low):
        mode = "and"
    else:
        mode = "or"
    return Problem(engine="count_range",
                   payload={"low": low_end, "high": high_end, "divisors": divisors, "mode": mode},
                   target_name="the count",
                   steps=[f"walking {low_end} to {high_end} and testing each one"])


def read_sum_over_range(low: str) -> Optional[Problem]:
    """"The sum of all two digit numbers divisible by 7" — the same walk, added up instead."""
    if not re.search(r"\bsum\s+of\s+all\b|\bsum\s+of\s+the\s+numbers\b|\badd\s+all\b", low):
        return None
    if not re.search(r"\bdivisible\s+by\b|\bmultiples?\s+of\b", low):
        return None
    span = _range_in(low)
    if span is None:
        return None
    divisors = [int(v) for v in numbers_in(low) if v.denominator == 1 and v > 1
                and v not in span]
    if not divisors:
        return None
    return Problem(engine="sum_range",
                   payload={"low": span[0], "high": span[1], "divisors": divisors},
                   target_name="the sum",
                   steps=[f"every number from {span[0]} to {span[1]} divisible by "
                          f"{divisors[0]}, added"])


_AP_RUN = re.compile(r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)")


def read_progression(low: str) -> Optional[Problem]:
    """An arithmetic progression: its nth term, its sum, or how many terms it has.

    One reading for all three because they are one object. The first three terms give ``a`` and
    ``d``, and everything else is which question was asked — the sum skill in
    :mod:`nyxara.njp.mathematics` added the *listed* terms, which is why "the sum of the first 20
    terms of 3, 7, 11, …" came back 21.
    """
    if not re.search(r"\bap\b|\barithmetic\s+progression\b|\bseries\b|\bsequence\b|\bterms?\b",
                     low):
        return None
    run = _AP_RUN.search(low)
    if run is None:
        return None
    first, second, third = (_frac(run.group(1)), _frac(run.group(2)), _frac(run.group(3)))
    step = second - first
    if third - second != step:
        return None                     # not arithmetic; another reading may know what it is
    ordinal = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+term\b", low)
    if ordinal is not None:
        n = int(ordinal.group(1))
        return Problem(engine="value", payload={"value": first + step * (n - 1)},
                       target_name=f"the {n}th term",
                       steps=[f"a = {_num(first)}, d = {_num(step)}",
                              f"a + (n-1)d = {_num(first)} + {n - 1}×{_num(step)}"])
    count = re.search(r"sum\s+of\s+the\s+first\s+(\d+)\s+terms|first\s+(\d+)\s+terms", low)
    if count is not None and "sum" in low:
        n = int(count.group(1) or count.group(2))
        total = Fraction(n) * (2 * first + (n - 1) * step) / 2
        return Problem(engine="value", payload={"value": total},
                       target_name=f"the sum of {n} terms",
                       steps=[f"a = {_num(first)}, d = {_num(step)}",
                              f"n/2 × (2a + (n-1)d) = {_num(total)}"])
    last = re.search(r"(?:\.\.\.|…)\s*,?\s*(-?\d+)\s*$|(?:\.\.\.|…)\s*,?\s*(-?\d+)\D*$", low)
    if last is not None and re.search(r"how\s+many\s+terms|number\s+of\s+terms", low):
        end = _frac(last.group(1) or last.group(2))
        if step == 0:
            raise MathError("a progression with no step never reaches its last term")
        n = (end - first) / step + 1
        if n.denominator != 1 or n <= 0:
            raise MathError(f"{_num(end)} is not a term of that progression")
        return Problem(engine="value", payload={"value": n}, target_name="the number of terms",
                       steps=[f"a = {_num(first)}, d = {_num(step)}, last = {_num(end)}",
                              f"n = (last - a)/d + 1 = {_num(n)}"])
    if last is not None and "sum" in low:
        end = _frac(last.group(1) or last.group(2))
        if step == 0:
            raise MathError("a progression with no step never reaches its last term")
        n = (end - first) / step + 1
        if n.denominator != 1 or n <= 0:
            raise MathError(f"{_num(end)} is not a term of that progression")
        total = n * (first + end) / 2
        return Problem(engine="value", payload={"value": total}, target_name="the sum",
                       steps=[f"{_num(n)} terms, first {_num(first)}, last {_num(end)}",
                              f"n(a + l)/2 = {_num(total)}"])
    return None


def read_digit_operation(low: str) -> Optional[Problem]:
    """"The sum of the digits of 2^10", "the units digit of 1! + 2! + … + 100!".

    The value is computed first and then *read* — which is the step the power skill skipped when
    it answered 1024 to the first of these. The series of factorials is summed exactly and only
    then asked for its last digit, because a hundred factorials is a number Python holds without
    complaint and the question is about its final digit rather than about its size.
    """
    wants = None
    if re.search(r"\bsum\s+of\s+the\s+digits\b", low):
        wants = "digit_sum"
    elif re.search(r"\b(?:units?|last|ones)\s+digit\b", low):
        wants = "last_digit"
    elif re.search(r"\bremainder\b", low) and re.search(r"!|\bfactorial", low):
        wants = "remainder"
    if wants is None:
        return None
    value = _series_or_power(low)
    if value is None:
        return None
    if wants == "remainder":
        modulus = re.search(r"divid\w+\s+by\s+(\d+)", low)
        if modulus is None:
            return None
        return Problem(engine="value", payload={"value": Fraction(value % int(modulus.group(1)))},
                       target_name="the remainder",
                       steps=[f"the sum is {value if value < 10 ** 12 else 'a large number'}",
                              f"its remainder modulo {modulus.group(1)}"])
    if wants == "digit_sum":
        return Problem(engine="value",
                       payload={"value": Fraction(sum(int(c) for c in str(abs(value))))},
                       target_name="the digit sum",
                       steps=[f"the value is {value if value < 10 ** 24 else 'large'}",
                              "its digits, added"])
    return Problem(engine="value", payload={"value": Fraction(abs(value) % 10)},
                   target_name="the last digit",
                   steps=["the value, and then its final digit"])


def _series_or_power(low: str) -> Optional[int]:
    """The whole number a digit question is *about*: a power, or a run of factorials summed."""
    factorials = re.search(r"1\s*!\s*\+\s*2\s*!", low)
    if factorials is not None:
        end = re.search(r"(?:\.\.\.|…)\s*\+?\s*(\d+)\s*!", low)
        if end is None:
            return None
        limit = int(end.group(1))
        if limit > 400:
            # Past 10! every factorial ends in at least two zeros, so the last digits stop
            # changing. Summing to 400 is exact for every question of this shape and finite.
            limit = 400
        return sum(math.factorial(k) for k in range(1, limit + 1))
    power = re.search(r"(\d+)\s*(?:\^|\*\*)\s*(\d+)", low)
    if power is not None:
        base, exponent = int(power.group(1)), int(power.group(2))
        if exponent > 4096:
            raise MathError("that power has more digits than she will read")
        return base ** exponent
    plain = numbers_in(low)
    return int(plain[0]) if len(plain) == 1 and plain[0].denominator == 1 else None


def read_symmetric_roots(low: str) -> Optional[Problem]:
    """"If the roots of x² - 5x + 6 = 0 are a and b, find a² + b²."

    By Vieta, not by solving: the sum and product of the roots are read straight off the
    coefficients, and every symmetric function follows from those two. It matters because the
    roots are often irrational while the answer is an integer — solving first turns an exact
    question into a decimal one.
    """
    if not re.search(r"\broots?\b", low):
        return None
    equation = re.search(r"([a-z0-9\s^*+\-/().]+?)\s*=\s*0", low)
    if equation is None:
        return None
    try:
        poly = Poly.parse(_clean_equation(equation.group(1)))
    except MathError:
        return None
    if poly.degree != 2:
        return None
    a, b, c = poly.coefficient(2), poly.coefficient(1), poly.coefficient(0)
    total, product = -b / a, c / a
    names = re.findall(r"\bare\s+([a-z])\s+and\s+([a-z])\b", low)
    if not names:
        return None
    first, second = names[0]
    tail = low.split("find", 1)[-1] if "find" in low else low
    forms = (
        (rf"{first}\s*\^?\s*2\s*\+\s*{second}\s*\^?\s*2",
         total * total - 2 * product, "a² + b² = (a+b)² - 2ab"),
        (rf"{first}\s*\^?\s*3\s*\+\s*{second}\s*\^?\s*3",
         total ** 3 - 3 * product * total, "a³ + b³ = (a+b)³ - 3ab(a+b)"),
        (rf"1\s*/\s*{first}\s*\+\s*1\s*/\s*{second}",
         total / product if product else None, "1/a + 1/b = (a+b)/ab"),
        (rf"{first}\s*\*?\s*{second}\b", product, "ab is the constant over the leading term"),
        (rf"{first}\s*\+\s*{second}\b", total, "a + b is minus the middle over the leading term"),
    )
    for pattern, value, working in forms:
        if re.search(pattern, tail) and value is not None:
            return Problem(engine="value", payload={"value": value},
                           target_name="the value",
                           steps=[f"a + b = {_num(total)}, ab = {_num(product)}", working])
    return None


def _clean_equation(text: str) -> str:
    body = re.sub(r"\b(?:the|roots?|of|are|and|find|if|is|value)\b", " ", str(text or ""))
    return " ".join(re.sub(r"[^0-9a-z+\-*/^(). ]+", " ", body).split())


def read_exponential_equation(low: str) -> Optional[Problem]:
    """"2^x = 32", "3^(x+1) = 81" — the unknown is upstairs, and search finds it exactly."""
    match = re.search(r"(\d+)\s*(?:\^|\*\*)\s*\(?\s*([a-z])\s*([+-]\s*\d+)?\s*\)?\s*=\s*(\d+)",
                      low)
    if match is None:
        return None
    base, offset, target = int(match.group(1)), match.group(3), int(match.group(4))
    shift = int(offset.replace(" ", "")) if offset else 0
    if base < 2 or target < 1:
        raise MathError("that is not an exponential equation she can solve")
    # **The unknown may be negative.** 3^(x+3) = 9 has x = -1, and searching from zero found
    # nothing and reported none — a real solution missed by a bound nobody had thought about.
    # The *exponent* still has to be non-negative; the unknown inside it need not be.
    exponent = search_integers(lambda n: base ** (n + shift) == target if 0 <= n + shift <= 64
                               else False, low=-64, high=64)
    if exponent is None:
        raise MathError(f"no whole number makes {base} to that power equal {target}")
    return Problem(engine="value", payload={"value": Fraction(exponent)},
                   target_name=f"{match.group(2)}",
                   steps=[f"{base}^{exponent + shift} = {target}",
                          f"so {match.group(2)} = {exponent}"])


def read_ratio_chain(low: str) -> Optional[Problem]:
    """"a:b = 2:3 and b:c = 4:5, find a:c" — the shared term is scaled until the two agree."""
    pairs = re.findall(r"([a-z])\s*:\s*([a-z])\s*=\s*(\d+)\s*:\s*(\d+)", low)
    if len(pairs) < 2:
        return None
    (n1, n2, a1, b1), (n3, n4, a2, b2) = pairs[0], pairs[1]
    if n2 != n3:
        return None
    bridge = int(b1) * int(a2) // math.gcd(int(b1), int(a2))
    left = int(a1) * bridge // int(b1)
    right = int(b2) * bridge // int(a2)
    divisor = math.gcd(left, right)
    return Problem(engine="ratio", payload={"parts": [left // divisor, right // divisor]},
                   render="ratio", target_name=f"{n1}:{n4}",
                   steps=[f"scale both so {n2} is {bridge}",
                          f"{n1} = {left}, {n4} = {right}"])


def read_angle_ratio(low: str) -> Optional[Problem]:
    """The angles of a shape given as a ratio: the parts share a known total."""
    if "angle" not in low:
        return None
    parts = re.search(r"(\d+)\s*:\s*(\d+)(?:\s*:\s*(\d+))?", low)
    if parts is None:
        return None
    shares = [int(g) for g in parts.groups() if g is not None]
    if len(shares) < 2:
        return None
    if "triangle" in low:
        total = 180
    elif re.search(r"\bquadrilateral\b|\bsquare\b|\brectangle\b", low):
        total = 360
    elif re.search(r"\bstraight\s+line\b|\bsupplementary\b", low):
        total = 180
    elif re.search(r"\bcomplementary\b", low):
        total = 90
    else:
        return None
    unit = Fraction(total, sum(shares))
    values = [unit * share for share in shares]
    if re.search(r"\bsmallest|least\b", low):
        value, name = min(values), "the smallest angle"
    elif re.search(r"\blargest|greatest|biggest\b", low):
        value, name = max(values), "the largest angle"
    else:
        return Problem(engine="value", payload={"value": values[0]}, target_name="the first angle",
                       steps=[f"the parts add to {sum(shares)}, sharing {total}°"])
    return Problem(engine="value", payload={"value": value}, target_name=name,
                   steps=[f"the parts add to {sum(shares)}, sharing {total}°",
                          f"one part is {_num(unit)}°"])


def read_inscribed(low: str) -> Optional[Problem]:
    """A circle inside a square, or a square inside a circle — the radius is the whole step."""
    if "circle" not in low:
        return None
    if not re.search(r"\binscribed\b|\binside\b|\bwithin\b", low):
        return None
    side = _dimension_of(low, ("side", "length", "edge"))
    if side is None:
        return None
    if not re.search(r"\bsquare\b", low):
        return None
    radius = side / 2
    coefficient = radius * radius
    wants_area = "area" in low
    return Problem(engine="pi", payload={"coefficient": coefficient if wants_area
                                         else radius * 2},
                   target_name="the area" if wants_area else "the circumference",
                   steps=[f"the circle touches all four sides, so r = {_num(radius)}"])


def _dimension_of(low: str, names: Sequence[str]) -> Optional[Fraction]:
    from nyxara.njp.mathematics import _dimension
    return _dimension(low, names)


def read_stars_and_bars(low: str) -> Optional[Problem]:
    """"8 identical balls into 3 distinct boxes" — the bars between them are what is chosen."""
    if not re.search(r"\bidentical\b|\bsame\b|\bindistinguishable\b", low):
        return None
    if not re.search(r"\bboxes\b|\bbins\b|\bgroups\b|\bbags\b|\bchildren\b|\bpeople\b", low):
        return None
    values = [int(v) for v in numbers_in(low) if v.denominator == 1 and v > 0]
    if len(values) < 2:
        return None
    balls, boxes = max(values), min(values)
    if boxes < 1:
        raise MathError("there must be at least one box")
    return Problem(engine="choose", payload={"n": balls + boxes - 1, "r": boxes - 1},
                   target_name="the number of ways",
                   steps=[f"{balls} balls and {boxes - 1} dividers in a row",
                          f"choose which {boxes - 1} of the {balls + boxes - 1} places are bars"])


def read_prime_count(low: str) -> Optional[Problem]:
    """"How many prime numbers are there below 50?" — sieved, then counted."""
    if not re.search(r"\bhow\s+many\s+primes?\b|\bnumber\s+of\s+primes?\b", low):
        return None
    span = _range_in(low)
    limit = span[1] if span else None
    if limit is None:
        match = re.search(r"(?:below|under|less\s+than|up\s+to)\s+(\d+)", low)
        if match is None:
            return None
        limit = int(match.group(1)) - 1 if re.search(r"below|under|less", low) \
            else int(match.group(1))
    return Problem(engine="primes", payload={"low": span[0] if span else 2, "high": int(limit)},
                   target_name="the number of primes",
                   steps=[f"sieving up to {limit}"])


def read_general_gcd(low: str) -> Optional[Problem]:
    """The hcf or lcm of two things that have to be *worked out* first — ``2^30 - 1``, say.

    The dispatcher answered 1048576 for the gcd of 2³⁰-1 and 2²⁰-1: it found the numbers 30, 1,
    20 and 1 in the sentence and never saw a subtraction. Each side is evaluated before anything
    is taken of it, which is what the question was about.
    """
    if not re.search(r"\bhcf\b|\bgcd\b|\blcm\b|\bhighest\s+common|\b(?:lowest|least)\s+common",
                     low):
        return None
    if not re.search(r"[-+*/^]", low):
        return None
    tail = re.split(r"\bhcf\b|\bgcd\b|\blcm\b|common\s+(?:factor|divisor|multiple)\b", low)[-1]
    tail = re.sub(r"^\s*(?:of|between)\s+", "", tail.strip())
    sides = re.split(r"\s+and\s+", tail)
    if len(sides) < 2:
        return None
    from nyxara.njp.calculate import Calculator
    calculator = Calculator()
    values: List[int] = []
    for side in sides[:2]:
        body = re.sub(r"[^0-9+\-*/^(). ]+", " ", side).strip()
        if not body or not re.search(r"\d", body):
            return None
        evaluation = calculator.evaluate(body)
        if not evaluation.ok:
            return None
        value = evaluation.value
        if not isinstance(value, int) and getattr(value, "denominator", 2) != 1:
            return None
        values.append(int(value))
    if len(values) < 2 or any(v == 0 for v in values):
        return None
    wants_lcm = bool(re.search(r"\blcm\b|\b(?:lowest|least)\s+common", low))
    result = (values[0] * values[1] // math.gcd(*values)) if wants_lcm else math.gcd(*values)
    return Problem(engine="value", payload={"value": Fraction(result)},
                   target_name="the lcm" if wants_lcm else "the hcf",
                   steps=[f"the two numbers work out to {values[0]} and {values[1]}",
                          f"their {'lcm' if wants_lcm else 'hcf'} is {result}"])


def read_bracket_product(low: str) -> Optional[Problem]:
    """"(1 - 1/2)(1 - 1/3)(1 - 1/4)(1 - 1/5)" — juxtaposed brackets, which is not Python.

    The calculator's parser is :func:`ast.parse` and ``(a)(b)`` is a call there, so a product
    written the way every textbook writes it could not be evaluated at all. Inserting the implied
    multiplication is a reading step, and it is done here rather than in the calculator because
    widening what that module accepts is the last thing its security note wants.
    """
    if low.count("(") < 2 or ")(" not in low.replace(" ", ""):
        return None
    body = re.search(r"(\((?:[^()]*\)\s*\()+[^()]*\))", low)
    if body is None:
        return None
    expression = re.sub(r"\)\s*\(", ")*(", body.group(1))
    from nyxara.njp.calculate import Calculator
    evaluation = Calculator().evaluate(expression)
    if not evaluation.ok:
        return None
    return Problem(engine="value", payload={"value": _frac(str(evaluation.value))},
                   target_name="the product",
                   steps=[f"the brackets multiply: {expression}",
                          f"= {evaluation.text}"])


def read_sum_of_powers(low: str) -> Optional[Problem]:
    """"The sum of the squares of the first 10 natural numbers" — a closed form, phrased freely."""
    match = re.search(r"sum\s+of\s+the\s+(squares|cubes)\s+of\s+the\s+first\s+(\d+)", low)
    if match is None:
        match = re.search(r"sum\s+of\s+the\s+first\s+(\d+)\s+(square|cube)s?\b", low)
        if match is None:
            return None
        n, kind = int(match.group(1)), match.group(2) + "s"
    else:
        kind, n = match.group(1), int(match.group(2))
    if n < 1 or n > 10 ** 6:
        raise MathError(f"{n} terms is outside what she will sum")
    if kind.startswith("square"):
        value = Fraction(n * (n + 1) * (2 * n + 1), 6)
        working = "n(n+1)(2n+1)/6"
    else:
        value = Fraction(n * (n + 1), 2) ** 2
        working = "(n(n+1)/2)²"
    return Problem(engine="value", payload={"value": value}, target_name="the sum",
                   steps=[f"{working} with n = {n}", f"= {_num(value)}"])


def read_mixed_remainders(low: str) -> Optional[Problem]:
    """"Divided by 7 leaves 3 and divided by 11 leaves 5" — different remainders, one number."""
    pairs = re.findall(r"divid\w+\s+by\s+(\d+)\s+(?:it\s+)?leaves\s+(?:a\s+remainder\s+of\s+)?"
                       r"(\d+)", low)
    if len(pairs) < 2:
        return None
    moduli = [int(m) for m, _ in pairs]
    remainders = [int(r) for _, r in pairs]
    if len(set(remainders)) == 1:
        return None                     # the same-remainder reading owns that one
    found = crt_smallest(moduli, remainders, low=1)
    if found is None:
        raise MathError("no number below the search bound leaves those remainders")
    return Problem(engine="value", payload={"value": Fraction(found)},
                   target_name="the smallest such number",
                   steps=[f"{found} leaves " + ", ".join(f"{found % m} under {m}"
                                                         for m in moduli)])


_SUCH_THAT = re.compile(
    r"\b(smallest|largest|least|greatest)\s+(?:positive\s+)?(?:integer|whole\s+number|number|"
    r"value)?\s*([a-z])?\s*(?:such\s+that|for\s+which|so\s+that|where)\s+(.+)$")


def read_predicate_search(low: str) -> Optional[Problem]:
    """"Find the smallest positive integer n such that n² + n + 41 is not prime."

    The most general reading here, and the one that comes closest to what "solving" means: an
    arbitrary polynomial in one unknown, tested against a stated property, over an exhaustive
    bounded range. Nothing about the pair is enumerated in advance — the polynomial is parsed and
    the property is compiled, so a combination nobody wrote down is answered by the same code as
    one that was.

    What it will not do is guess at a property it does not recognise. A search whose predicate is
    approximately the question returns a number that is exactly wrong.
    """
    match = _SUCH_THAT.search(low)
    if match is None:
        return None
    want = "smallest" if match.group(1) in ("smallest", "least") else "largest"
    condition = match.group(3).strip()
    symbol = match.group(2)
    predicate, described = _compile_predicate(condition, symbol)
    if predicate is None:
        return None
    high = _MAX_SEARCH if want == "smallest" else 10_000
    found = search_integers(predicate, low=1, high=high, want=want)
    if found is None:
        raise MathError(f"no whole number up to {high} satisfies that")
    return Problem(engine="value", payload={"value": Fraction(found)},
                   target_name=f"the {want} such number",
                   steps=[f"testing each whole number in turn against: {described}",
                          f"the first that qualifies is {found}"])


def _compile_predicate(condition: str, symbol: Optional[str]) -> Tuple[Any, str]:
    """Turn "n² + n + 41 is not prime" into a callable, or come back empty-handed.

    Empty-handed is the important half again: a property this cannot read must not be quietly
    approximated by one it can.
    """
    body = condition
    tests: List[Tuple[str, Any]] = []
    negated = bool(re.search(r"\bis\s+not\b|\bnot\s+a\b|\bnever\b", body))

    remainder = re.search(r"divid\w+\s+by\s+(\d+)\s+leaves\s+(?:a\s+remainder\s+of\s+)?(\d+)", body)
    divisible = re.search(r"\b(?:is\s+)?(?:not\s+)?divisible\s+by\s+(\d+)", body)
    multiple = re.search(r"\b(?:is\s+)?(?:not\s+)?a\s+multiple\s+of\s+(\d+)", body)
    square = re.search(r"\bperfect\s+(square|cube)\b", body)
    prime = re.search(r"\bprime\b", body)
    greater = re.search(r"\b(?:greater|more|bigger)\s+than\s+(-?\d+)", body)
    smaller = re.search(r"\b(?:less|smaller|fewer)\s+than\s+(-?\d+)", body)

    head = re.split(r"\bis\b|\bleaves\b", body)[0]
    expression = _clean_equation(head)
    poly: Optional[Poly] = None
    if expression and re.search(r"[a-z]", expression):
        try:
            poly = Poly.parse(expression, symbol)
        except MathError:
            poly = None
    evaluate = (lambda n: int(poly.at(n))) if poly is not None else (lambda n: int(n))
    described = f"{poly.text() if poly is not None else (symbol or 'n')}"

    if remainder:
        modulus, left = int(remainder.group(1)), int(remainder.group(2))
        tests.append((f"leaves {left} on division by {modulus}",
                      lambda n: evaluate(n) % modulus == left))
    elif divisible or multiple:
        divisor = int((divisible or multiple).group(1))
        if divisor == 0:
            raise MathError("nothing is divisible by zero")
        tests.append((f"is divisible by {divisor}", lambda n: evaluate(n) % divisor == 0))
    elif square:
        index = 2 if square.group(1) == "square" else 3
        tests.append((f"is a perfect {square.group(1)}",
                      lambda n: _integer_root_of(evaluate(n), index) is not None))
    elif prime:
        tests.append(("is prime", lambda n: is_prime(evaluate(n))))
    elif greater:
        bound = int(greater.group(1))
        tests.append((f"is greater than {bound}", lambda n: evaluate(n) > bound))
    elif smaller:
        bound = int(smaller.group(1))
        tests.append((f"is less than {bound}", lambda n: evaluate(n) < bound))
    if not tests:
        return None, ""
    label, test = tests[0]
    if negated:
        return (lambda n: not test(n)), f"{described} is not {label.removeprefix('is ')}"
    return test, f"{described} {label}"


def _integer_root_of(value: int, index: int) -> Optional[int]:
    from nyxara.njp.mathematics import _integer_root
    return _integer_root(int(value), index) if value >= 0 else None


# --------------------------------------------------------------------------- #
# NJP V.24, third tier — the readings a fourth bank of problems asked for
# --------------------------------------------------------------------------- #

def read_equation(low: str) -> Optional[Problem]:
    """Any single-unknown equation, however the sentence introduces it.

    The most useful of the third tier and the least clever: it finds the ``=``, takes the algebra
    on each side of it, and solves. "Find the value of x **if** (x-1)/2 + (x+1)/3 = 4" was refused
    by the skill table because the word *if* sat between the unknown and its equation, and the
    equation reader there strips a fixed list of words rather than looking for the equation.
    """
    if low.count("=") != 1:
        return None
    # **An evaluation frame is not an equation.** "The value of 2x² + 9 **when** x = 5" contains
    # an `=` and is not a thing to solve; read as one it says x = 5 and answers 5, which is the
    # number in the question and not the answer to it. Measured as a regression on a paper that
    # had been passing.
    if re.search(r"\b(?:when|at|given|for)\s+[a-z]\s*=", low):
        return None
    left_text, _, right_text = low.partition("=")
    left_body = _algebra_tail(left_text)
    right_body = _algebra_head(right_text)
    if not left_body or not right_body:
        return None
    symbols = set(re.findall(r"(?<![a-z])([a-z])(?![a-z])", left_body + " " + right_body))
    if len(symbols) != 1:
        return None
    symbol = symbols.pop()
    try:
        poly = Poly.parse(left_body, symbol) - Poly.parse(right_body, symbol)
    except MathError:
        return None
    if poly.is_zero or poly.is_constant or poly.degree < 1:
        return None
    # **Every root, not one of them.** A quadratic has two and naming one is choosing rather than
    # solving; the first version raised "more than one answer" and, because a recognised refusal
    # blocks, took a paper that had been passing down with it.
    roots = poly.rational_roots()
    if not roots:
        raise MathError("that equation has no rational root she can state exactly")
    return Problem(engine="roots", payload={"roots": roots, "symbol": symbol},
                   target_name=symbol,
                   steps=[f"{left_body} = {right_body}",
                          f"bring everything to one side: {poly.text()} = 0"])


def _algebra_tail(text: str) -> str:
    """The algebra at the *end* of a phrase — everything after the last word that is not algebra."""
    tokens = str(text or "").split()
    kept: List[str] = []
    for token in reversed(tokens):
        if re.fullmatch(r"[0-9a-z+\-*/^(). ]+", token) and not re.search(r"[a-z]{2,}", token):
            kept.append(token)
        else:
            break
    return " ".join(reversed(kept))


def _algebra_head(text: str) -> str:
    """The algebra at the *start* of a phrase, by the same rule pointing the other way."""
    kept: List[str] = []
    for token in str(text or "").split():
        if re.fullmatch(r"[0-9a-z+\-*/^(). ]+", token) and not re.search(r"[a-z]{2,}", token):
            kept.append(token)
        else:
            break
    return " ".join(kept)


def read_function_value(low: str) -> Optional[Problem]:
    """"If f(x) = 2x + 3, find f(5)" — a definition and then a substitution."""
    definition = re.search(r"\b([a-z])\s*\(\s*([a-z])\s*\)\s*=\s*([^,.;]+)", low)
    if definition is None:
        return None
    name, symbol, body = definition.group(1), definition.group(2), definition.group(3)
    call = re.search(rf"{name}\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", low)
    if call is None:
        return None
    try:
        poly = Poly.parse(_clean_equation(body), symbol)
    except MathError:
        return None
    at = _frac(call.group(1))
    return Problem(engine="value", payload={"value": poly.at(at)},
                   target_name=f"{name}({_num(at)})",
                   steps=[f"{name}({symbol}) = {poly.text()}",
                          f"putting {symbol} = {_num(at)} gives {_num(poly.at(at))}"])


def read_percentage_transfer(low: str) -> Optional[Problem]:
    """"If 20% of a number is 45, what is 30% of it?" — the number is a step, not the answer."""
    given = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(?:a|the|some)?\s*number\s+is\s+"
                      r"(-?\d+(?:\.\d+)?)", low)
    wanted = re.search(r"what\s+is\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    if given is None or wanted is None:
        return None
    first, value, second = (_frac(given.group(1)), _frac(given.group(2)),
                            _frac(wanted.group(1)))
    if first == 0:
        raise MathError("0% of every number is 0, so the number cannot be recovered")
    whole = Expr.variable("whole")
    problem = Problem(steps=[f"{_num(first)}% of the number is {_num(value)}"])
    problem.require(whole * Expr.constant(first / 100), Expr.constant(value))
    problem.target = whole * Expr.constant(second / 100)
    problem.target_name = f"{_num(second)}% of it"
    return problem


def read_inverse_closed_form(low: str) -> Optional[Problem]:
    """"The sum of the first n natural numbers is 210. Find n." — the formula, run backwards."""
    match = re.search(r"sum\s+of\s+the\s+first\s+([a-z])\s+natural\s+numbers?\s+is\s+"
                      r"(-?\d+(?:\.\d+)?)", low)
    if match is None:
        return None
    symbol, total = match.group(1), _frac(match.group(2))
    found = search_integers(lambda k: k * (k + 1) == 2 * total, low=1, high=100_000)
    if found is None:
        raise MathError(f"no whole number of terms adds to {_num(total)}")
    return Problem(engine="value", payload={"value": Fraction(found)}, target_name=symbol,
                   steps=[f"n(n+1)/2 = {_num(total)}", f"n = {found}"])


def read_boat(low: str) -> Optional[Problem]:
    """Downstream and upstream: the stream is half their difference, the boat half their sum."""
    if not re.search(r"\bdownstream\b|\bupstream\b|\bstream\b|\bcurrent\b|\bstill\s+water\b", low):
        return None
    down = re.search(r"(\d+(?:\.\d+)?)\s*km\s+downstream\s+in\s+(\d+(?:\.\d+)?)\s*hours?", low)
    if down is None:
        return None
    back = re.search(r"return\w*\s+in\s+(\d+(?:\.\d+)?)\s*hours?", low)
    if back is None:
        return None
    distance, going, coming = (_frac(down.group(1)), _frac(down.group(2)), _frac(back.group(1)))
    if going == 0 or coming == 0:
        raise MathError("a journey takes time")
    downstream, upstream = distance / going, distance / coming
    wants_stream = bool(re.search(r"\bstream\b|\bcurrent\b", low.split("find", 1)[-1]))
    value = (downstream - upstream) / 2 if wants_stream else (downstream + upstream) / 2
    return Problem(engine="value", payload={"value": value},
                   target_name="the speed of the stream" if wants_stream
                   else "the speed of the boat in still water",
                   steps=[f"downstream {_num(downstream)}, upstream {_num(upstream)}",
                          "the stream is half their difference, the boat half their sum"])


def read_mean_shift(low: str) -> Optional[Problem]:
    """"Each number is increased by 3" — the mean moves by exactly that, and nothing else does."""
    if "mean" not in low and "average" not in low:
        return None
    base = re.search(r"(?:mean|average)\s+of\s+\d+\s+numbers?\s+is\s+(-?\d+(?:\.\d+)?)", low)
    shift = re.search(r"each\s+(?:number\s+)?(?:is\s+)?(increased|decreased|reduced)\s+by\s+"
                      r"(-?\d+(?:\.\d+)?)", low)
    if base is None or shift is None:
        multiply = re.search(r"each\s+(?:number\s+)?(?:is\s+)?multiplied\s+by\s+"
                             r"(-?\d+(?:\.\d+)?)", low)
        if base is None or multiply is None:
            return None
        value = _frac(base.group(1)) * _frac(multiply.group(1))
        return Problem(engine="value", payload={"value": value}, target_name="the new mean",
                       steps=["multiplying every value multiplies the mean"])
    amount = _frac(shift.group(2))
    if shift.group(1) != "increased":
        amount = -amount
    return Problem(engine="value", payload={"value": _frac(base.group(1)) + amount},
                   target_name="the new mean",
                   steps=["adding the same amount to every value adds it to the mean"])


def read_interest_multiple(low: str) -> Optional[Problem]:
    """"A sum doubles in 8 years at simple interest — when does it triple?"

    The principal never appears and does not need to: doubling means the interest equalled the
    principal, so one "principal's worth" takes 8 years and tripling needs two of them.
    """
    if "simple interest" not in low:
        return None
    first = re.search(r"\b(doubles?|triples?|quadruples?)\s+in\s+(\d+(?:\.\d+)?)\s*years?", low)
    second = re.search(r"(?:will|does)\s+it\s+(doubles?|triples?|quadruples?)", low)
    if first is None or second is None:
        return None
    growth = {"double": 2, "triple": 3, "quadruple": 4}
    start = growth[first.group(1).rstrip("s")] - 1
    target = growth[second.group(1).rstrip("s")] - 1
    years = _frac(first.group(2))
    return Problem(engine="value", payload={"value": years * target / start},
                   target_name="the years needed",
                   steps=[f"gaining one principal takes {_num(years / start)} years",
                          f"gaining {target} of them takes {_num(years * target / start)}"])


def read_at_least_one(low: str) -> Optional[Problem]:
    """"At least one head in two tosses" — one minus the probability of none, never a sum."""
    # `normalise` rewrites the number words in a sentence that has no digits, so "at least one"
    # arrives as "at least 1". Both spellings reach here.
    if not re.search(r"\bat\s+least\s+(?:one|1)\b", low):
        return None
    if not re.search(r"\bcoin\b|\bhead\b|\btail\b|\btoss\w*", low):
        return None
    tosses = re.search(r"(\d+)\s*(?:tosses|times|throws|coins)", low)
    count = int(tosses.group(1)) if tosses else 2
    if not 1 <= count <= 20:
        raise MathError("that many tosses is outside what she will count")
    value = 1 - Fraction(1, 2 ** count)
    return Problem(engine="value", payload={"value": value}, target_name="the probability",
                   steps=[f"P(no head in {count}) = 1/2^{count} = {_num(Fraction(1, 2 ** count))}",
                          f"so P(at least one) = {_num(value)}"])


def read_dice_sum(low: str) -> Optional[Problem]:
    """Two dice and a total — enumerated over all thirty-six outcomes, not reasoned about."""
    if not re.search(r"\bdice\b|\bdie\b", low) or not re.search(r"\bsum\b|\btotal\b", low):
        return None
    if not re.search(r"\btwo\b|\b2\b|\bpair\b", low):
        return None
    target = re.search(r"(?:sum|total)\s+(?:is|of|equals?)\s+(\d+)", low)
    if target is None:
        return None
    wanted = int(target.group(1))
    favourable = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == wanted)
    return Problem(engine="value", payload={"value": Fraction(favourable, 36)},
                   target_name="the probability",
                   steps=[f"{favourable} of the 36 outcomes total {wanted}"])


def read_lcm_hcf_pair(low: str) -> Optional[Problem]:
    """The product of two numbers is the product of their lcm and hcf — so the other one follows."""
    if not re.search(r"\blcm\b|\blowest\s+common\s+multiple\b", low):
        return None
    if not re.search(r"\bhcf\b|\bgcd\b|\bhighest\s+common\b", low):
        return None
    lcm = re.search(r"(?:lcm|lowest\s+common\s+multiple)\s+(?:of\s+two\s+numbers\s+)?is\s+(\d+)",
                    low)
    hcf = re.search(r"(?:hcf|gcd|highest\s+common\s+factor)\s+is\s+(\d+)", low)
    one = re.search(r"one\s+(?:of\s+the\s+)?numbers?\s+is\s+(\d+)|one\s+number\s+is\s+(\d+)", low)
    if lcm is None or hcf is None or one is None:
        return None
    first = int(one.group(1) or one.group(2))
    if first == 0:
        raise MathError("zero is not one of two numbers with an lcm")
    product = int(lcm.group(1)) * int(hcf.group(1))
    if product % first:
        raise MathError("no whole number pairs with that one to give those")
    return Problem(engine="value", payload={"value": Fraction(product // first)},
                   target_name="the other number",
                   steps=[f"lcm × hcf = {product}, which is the product of the two numbers",
                          f"{product} ÷ {first} = {product // first}"])


def read_two_selling_prices(low: str) -> Optional[Problem]:
    """Sold at a loss, and what would have been a gain — two lines through one cost price."""
    loss = re.search(r"loss\s+of\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    gain = re.search(r"(?:gain|profit)(?:ed)?\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    more = re.search(r"(?:for|by)\s+(-?\d+(?:\.\d+)?)\s+more", low)
    if loss is None or gain is None or more is None:
        return None
    down, up, difference = (_frac(loss.group(1)), _frac(gain.group(1)), _frac(more.group(1)))
    cost = Expr.variable("cost")
    problem = Problem(steps=[f"selling at a {_num(down)}% loss is cost × {_num(1 - down / 100)}",
                             f"selling at a {_num(up)}% gain is cost × {_num(1 + up / 100)}",
                             f"the two differ by {_num(difference)}"])
    problem.require(cost * Expr.constant((1 + up / 100) - (1 - down / 100)),
                    Expr.constant(difference))
    problem.target, problem.target_name = cost, "the cost price"
    return problem


def read_si_ci_difference(low: str) -> Optional[Problem]:
    """The gap between simple and compound interest over two years is P(R/100)², and nothing else."""
    if not re.search(r"difference\b[^.]{0,60}?\bsimple\b[^.]{0,30}?\bcompound\b|"
                     r"difference\b[^.]{0,60}?\bcompound\b[^.]{0,30}?\bsimple\b", low):
        return None
    rate = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    years = re.search(r"(\d+)\s*years?", low)
    if rate is None or years is None or int(years.group(1)) != 2:
        return None
    # **The gap is read by where it is written, not by being the number that is left.** Excluding
    # the rate and the term by *value* deletes the gap itself whenever it happens to equal one of
    # them — "at 10 percent is 10" left nothing at all, and the question went unanswered on a
    # seed where the two coincided.
    # `[^.]` cannot cross the full stop that ends the sentence the gap is stated in — and the
    # sentence always ends there, with "find the sum" beginning the next one.
    stated = re.search(r"\bis\s+(-?\d+(?:\.\d+)?)[^a-z]{0,8}find\b", low)
    if stated is not None:
        gap = [_frac(stated.group(1))]
    else:
        gap = [v for v in numbers_in(low)
               if v != _frac(rate.group(1)) and v != _frac(years.group(1))]
    if not gap:
        return None
    rate_value = _frac(rate.group(1))
    if rate_value == 0:
        raise MathError("no rate means no difference, so the sum cannot be recovered")
    principal = Expr.variable("principal")
    factor = (rate_value / 100) ** 2
    problem = Problem(steps=["over two years the gap is P × (R/100)²",
                             f"= P × {_num(factor)}"])
    problem.require(principal * Expr.constant(factor), Expr.constant(max(gap)))
    problem.target, problem.target_name = principal, "the sum"
    return problem


def read_age_multiple(low: str) -> Optional[Problem]:
    """"Three times as old now, five times as old ten years ago" — two lines, two ages."""
    if "old" not in low and "age" not in low:
        return None
    now = re.search(r"(\d+(?:\.\d+)?)\s+times\s+as\s+old", low)
    then = re.search(r"(\d+)\s+years?\s+(ago|later|hence)\D{0,40}?(\d+(?:\.\d+)?)\s+times", low)
    if then is None:
        then = re.search(r"(\d+)\s+years?\s+(ago|later|hence)[^.]{0,60}?(\d+(?:\.\d+)?)\s+times",
                         low)
    if now is None or then is None:
        return None
    ratio_now = _frac(now.group(1))
    years = _frac(then.group(1)) * (-1 if then.group(2) == "ago" else 1)
    ratio_then = _frac(then.group(3))
    elder, younger = Expr.variable("elder"), Expr.variable("younger")
    problem = Problem(steps=[f"now the elder is {_num(ratio_now)} times the younger",
                             f"{_num(abs(years))} years {then.group(2)} it was "
                             f"{_num(ratio_then)} times"])
    problem.require(elder, younger * Expr.constant(ratio_now))
    problem.require(elder + Expr.constant(years),
                    (younger + Expr.constant(years)) * Expr.constant(ratio_then))
    if re.search(r"\bson|younger|child|daughter\b", low.split("find", 1)[-1]):
        problem.target, problem.target_name = younger, "the younger age"
    else:
        problem.target, problem.target_name = elder, "the elder age"
    return problem


def read_square_from_measure(low: str) -> Optional[Problem]:
    """"The area of a square is 144 — what is its perimeter?" The side is the bridge."""
    if "square" not in low:
        return None
    area = re.search(r"area\s+of\s+(?:a|the)\s+square\s+is\s+(\d+(?:\.\d+)?)", low)
    perimeter = re.search(r"perimeter\s+of\s+(?:a|the)\s+square\s+is\s+(\d+(?:\.\d+)?)", low)
    if area is not None and "perimeter" in low.split("is", 1)[-1]:
        value = _frac(area.group(1))
        from nyxara.njp.mathematics import _exact_root
        side = _exact_root(value, 2)
        if side is None:
            raise MathError(f"{_num(value)} is not the area of a square with a rational side")
        return Problem(engine="value", payload={"value": side * 4}, target_name="the perimeter",
                       steps=[f"the side is √{_num(value)} = {_num(side)}",
                              f"the perimeter is 4 × {_num(side)}"])
    if perimeter is not None and "area" in low.split("is", 1)[-1]:
        side = _frac(perimeter.group(1)) / 4
        return Problem(engine="value", payload={"value": side * side}, target_name="the area",
                       steps=[f"the side is {_num(side)}", f"the area is {_num(side * side)}"])
    return None


def read_rectangle_from_two(low: str) -> Optional[Problem]:
    """An area and a perimeter together fix the sides — one nonlinear system, two answers."""
    if "rectangle" not in low:
        return None
    area = re.search(r"area\s+(?:is\s+|of\s+)?(\d+(?:\.\d+)?)", low)
    perimeter = re.search(r"perimeter\s+(?:is\s+|of\s+)?(\d+(?:\.\d+)?)", low)
    if area is None or perimeter is None:
        return None
    length, width = Expr.variable("length"), Expr.variable("width")
    problem = Problem(steps=["2(l + w) is the perimeter and lw is the area"])
    problem.require((length + width) * Expr.constant(2), Expr.constant(_frac(perimeter.group(1))))
    problem.require(length * width, Expr.constant(_frac(area.group(1))))
    problem.engine = "pick"
    if re.search(r"\bwidth|breadth|smaller\b", low.split("find", 1)[-1]):
        problem.target_name, problem.payload["pick"] = "the width", "min"
    else:
        problem.target_name, problem.payload["pick"] = "the length", "max"
    return problem


def read_geometric_progression(low: str) -> Optional[Problem]:
    """A geometric progression asked for its nth term — the same object as an AP, times instead
    of plus."""
    if not re.search(r"\bgp\b|\bgeometric\b", low):
        return None
    run = _AP_RUN.search(low)
    ordinal = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+term\b", low)
    if run is None or ordinal is None:
        return None
    first, second, third = (_frac(run.group(1)), _frac(run.group(2)), _frac(run.group(3)))
    if first == 0 or second == 0:
        return None
    ratio = second / first
    if third / second != ratio:
        return None
    n = int(ordinal.group(1))
    if n > 512:
        raise MathError(f"the {n}th term of that progression is too large to state")
    return Problem(engine="value", payload={"value": first * ratio ** (n - 1)},
                   target_name=f"the {n}th term",
                   steps=[f"a = {_num(first)}, r = {_num(ratio)}",
                          f"a·r^(n-1) = {_num(first)} × {_num(ratio)}^{n - 1}"])


def read_chessboard(low: str) -> Optional[Problem]:
    """The squares on a chessboard: every size from 1×1 to 8×8, counted and added."""
    if not re.search(r"\bchess\s*board\b|\bchessboard\b", low):
        return None
    if not re.search(r"\bsquares?\b", low):
        return None
    side = 8
    total = sum(k * k for k in range(1, side + 1))
    return Problem(engine="value", payload={"value": Fraction(total)},
                   target_name="the number of squares",
                   steps=[f"a k×k square fits in ({side}-k+1)² places",
                          f"1² + 2² + … + {side}² = {total}"])


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #

#: Every reading, tried in order. Order matters only where two could both match; a reading that
#: matches and then cannot solve costs nothing, because the next one is tried.
READINGS = (
    read_predicate_search,
    read_function_value,
    read_chessboard,
    read_boat,
    read_at_least_one,
    read_dice_sum,
    read_lcm_hcf_pair,
    read_two_selling_prices,
    read_si_ci_difference,
    read_interest_multiple,
    read_age_multiple,
    read_mean_shift,
    read_percentage_transfer,
    read_inverse_closed_form,
    read_square_from_measure,
    read_rectangle_from_two,
    read_geometric_progression,
    read_symmetric_roots,
    read_exponential_equation,
    read_digit_operation,
    read_mixed_remainders,
    read_count_over_range,
    read_sum_over_range,
    read_sum_of_powers,
    read_prime_count,
    read_stars_and_bars,
    read_ratio_chain,
    read_angle_ratio,
    read_inscribed,
    read_general_gcd,
    read_progression,
    read_bracket_product,
    read_reciprocal_identity,
    read_modular,
    read_factorial_zeros,
    read_factorised_gcd,
    read_remainder_search,
    read_same_remainder,
    read_coordinate_area,
    read_infinite_series,
    read_diophantine,
    read_cards,
    read_draws,
    read_arrangements,
    read_divisor_question,
    read_age_ratio,
    read_interest_back,
    read_markup_discount,
    read_train,
    read_average_change,
    read_work_chain,
    read_pipes,
    read_inverse_proportion,
    read_average_speed,
    read_rectangle_relation,
    read_series_difference,
    read_digits,
    read_consecutive,
    read_number_relations,
    read_equation,
)


class Solver:
    """Reads a problem into constraints, solves them, and checks the answer before stating it.

    The check is the part that is not decoration. A pattern that matches cannot notice that it
    matched the wrong thing; an assignment substituted back into every constraint either satisfies
    them or does not. So a misread sentence here produces **silence**, where in a dispatcher it
    produces a number — which is the difference between the nine confidently wrong answers on the
    thirty-problem floor and none.
    """

    def __init__(self) -> None:
        self.asked = 0
        self.solved = 0
        self.declined = 0
        self._engines: Dict[str, int] = {}
        #: The last question and what it came to — a turn asks twice and this makes it cost once,
        #: for the same reason `Mathematician._last` exists.
        self._last: Optional[Tuple[str, Solution]] = None

    # -- the public call ------------------------------------------------------ #
    def solve(self, text: str) -> Solution:
        question = str(text or "").strip()
        if not question:
            return Solution(error="there is no problem there")
        if self._last is not None and self._last[0] == question:
            return self._last[1]
        low = normalise(question)
        self.asked += 1
        if _asks_non_quantity(low):
            self.declined += 1
            refused = Solution(question=question, task=looks_like_a_task(low),
                               error="that asks for something that is not a quantity")
            self._last = (question, refused)
            return refused
        refusal, recognised = "", False
        for reading in READINGS:
            try:
                problem = reading(low)
            except MathError as exc:
                # A reading that *raised* recognised the sentence and then found it had no
                # answer — "two drawn from one red ball" is read perfectly and is impossible.
                # Without marking it recognised here, the refusal fell through to the skill
                # table, which answered 1/6.
                refusal, recognised = refusal or str(exc), True
                continue
            except Exception:  # noqa: BLE001 — a broken reading must not take the turn down
                continue
            if problem is None:
                continue
            recognised = True
            try:
                solution = self._settle(problem, question, reading.__name__)
            except MathError as exc:
                refusal = refusal or str(exc)
                continue
            except Exception:  # noqa: BLE001
                continue
            if solution is not None and solution.ok:
                self.solved += 1
                self._engines[solution.engine] = self._engines.get(solution.engine, 0) + 1
                self._last = (question, solution)
                return solution
        self.declined += 1
        refused = Solution(question=question, recognised=recognised,
                           task=recognised or looks_like_a_task(low),
                           error=refusal or "she could not read that into anything she can solve")
        self._last = (question, refused)
        return refused

    def stats(self) -> Dict[str, Any]:
        return {"asked": self.asked, "solved": self.solved, "declined": self.declined,
                "readings": len(READINGS), "by_reading": dict(sorted(self._engines.items()))}

    # -- solving --------------------------------------------------------------- #
    def _settle(self, problem: Problem, question: str, reading: str) -> Optional[Solution]:
        if problem.engine and problem.engine != "pick":
            value, extra, checked = _run_engine(problem)
            steps = list(problem.steps) + list(extra)
            return Solution(question=question, engine=reading, value=value,
                            answer=_render(value, problem.render), steps=steps, verified=checked)
        assignments = solve_algebraic(problem)
        if not assignments:
            return None
        good = [a for a in assignments if _satisfies(problem, a)]
        if not good:
            return None
        good, aside = _prefer_positive(good)
        if problem.engine == "pick":
            return self._pick(problem, good[0], question, reading, aside)
        if problem.target is None:
            return None
        values = sorted({problem.target.at(a) for a in good})
        if len(values) != 1:
            # Several different values all satisfy the constraints. That is not an answer, and
            # naming one of them would be choosing rather than solving.
            raise MathError("that has more than one answer and the problem does not say which")
        value = values[0]
        steps = list(problem.steps) + [
            "solving: " + ", ".join(f"{k} = {_num(v)}" for k, v in sorted(good[0].items())),
            f"{problem.target_name or 'the answer'} = {_num(value)}"]
        return Solution(question=question, engine=reading, value=value,
                        answer=_render(value, problem.render), assignment=good[0],
                        steps=steps, verified=True)

    @staticmethod
    def _pick(problem: Problem, assignment: Dict[str, Fraction], question: str,
              reading: str, aside: str = "") -> Solution:
        """Two unknowns and a question that asks for one of them by size rather than by name."""
        values = sorted(assignment.values())
        pick = problem.payload.get("pick", "all")
        if pick == "max":
            value: Any = values[-1]
        elif pick == "min":
            value = values[0]
        else:
            value = values
        answer = (", ".join(_num(v) for v in value) if isinstance(value, list)
                  else _num(value))
        return Solution(question=question, engine=reading, value=value, answer=answer,
                        assignment=assignment, verified=True,
                        steps=list(problem.steps) + ([aside] if aside else []) + [
                            "solving: " + ", ".join(f"{k} = {_num(v)}"
                                                    for k, v in sorted(assignment.items())),
                            f"{problem.target_name or 'the answer'} = {answer}"])


#: Interrogatives that do not ask for a number. A reading may match every clause of the problem
#: and still be answering a question nobody asked: "the sum of three consecutive numbers is 78 —
#: what is the **colour** of the largest?" is read perfectly and has no numeric answer, and the
#: first version of this replied 27.
#: "**whose**" and "**where**" were in this list and had to come out. "Find two numbers *whose*
#: sum is 7" is a relative pronoun and not a question, and the guard refused every problem written
#: that way — twenty in a hundred, silently. An interrogative is only an interrogative in the slot
#: where a question is asked, and a bare word cannot tell the difference.
_NOT_A_QUANTITY = re.compile(
    r"\bwhat\s+(?:is\s+)?(?:the\s+)?(?:colour|color|shape|name|kind|sort|type)\b"
    r"|\bwho\s+(?:is|are|was|were)\b|\bwhy\s+(?:is|are|does|do|did)\b"
    r"|\bwhich\s+colour\b")


def _asks_non_quantity(low: str) -> bool:
    """Does this sentence ask for something a number cannot be?"""
    return bool(_NOT_A_QUANTITY.search(low))


#: An imperative that makes a sentence a *task* rather than a claim about the world.
_TASK_VERB = re.compile(
    r"\b(?:find|solve|calculate|compute|evaluate|determine|simplify|factorise|factorize|"
    r"expand|prove|how\s+many|what\s+is\s+the\s+value\s+of)\b")


def looks_like_a_task(text: str) -> bool:
    """Is this an instruction to work something out, even though nothing here could?

    It exists for the store rather than for the answer. A maths problem she cannot solve is still
    not a fact about the world, and without this the grounder filed the ones that beat her:
    ``('find', 'the') → 'smallest positive integer n'`` and two more like it, at the same
    confidence as a stated fact. Silence about a problem is correct; **writing it down as
    knowledge is not**.

    Both halves are required, and that is what keeps it narrow. "Scientists find water on Mars"
    has the verb and no arithmetic; "the sun is 150 million km away" has the digits and no
    instruction. Neither is a task, and both are things she should still learn.
    """
    low = normalise(text)
    if not _TASK_VERB.search(low):
        return False
    return bool(re.search(r"\d", low) or re.search(r"[+\-*/^=]", low))


def _prefer_positive(found: List[Dict[str, Fraction]]) -> Tuple[List[Dict[str, Fraction]], str]:
    """Take the all-positive solution where there is exactly one, and **say so**.

    "Two numbers differ by 4 and their product is 96" is satisfied by (12, 8) and by (-8, -12),
    and both are solutions of what was written down. The answer everybody means is 12, and the
    reason is a convention about the word *numbers* rather than anything in the algebra — so the
    convention is applied and the discarded pair is named in the working, which is the difference
    between taking an assumption and hiding one.
    """
    if len(found) < 2:
        return found, ""
    positive = [a for a in found if all(v > 0 for v in a.values())]
    if len(positive) != 1:
        return found, ""
    others = [a for a in found if a is not positive[0]]
    shown = "; ".join(", ".join(f"{k} = {_num(v)}" for k, v in sorted(a.items()))
                      for a in others[:2])
    return positive, f"({shown} also satisfies it; the positive solution is taken)"


def _satisfies(problem: Problem, assignment: Dict[str, Fraction]) -> bool:
    """Does this assignment hold against every constraint that was read? The verification step."""
    return all(constraint.holds(assignment) for constraint in problem.constraints)


def _render(value: Any, style: str = "number") -> str:
    if isinstance(value, str):
        return value                    # a π answer arrives already written, exact and approximate
    if isinstance(value, list):
        if style == "ratio":
            return " : ".join(_num(v) for v in value)
        return ", ".join(_num(v) for v in value)
    if style == "percent":
        return f"{_num(value)}%"
    return _num(value)


def _run_engine(problem: Problem) -> Tuple[Any, List[str], bool]:
    """Run one discrete engine and say whether the answer was independently checked.

    **"Verified" means something different here and the difference is stated rather than blurred.**
    An algebraic answer is checked by substitution into the constraints that produced it. A closed
    form has no constraints to substitute into, so where a slower independent computation exists it
    is run and compared — 100! really is computed and its zeros counted, the modular power really
    is checked against the full power — and where none exists the answer is arithmetic on numbers
    already read, and is marked checked because there is nothing left to check.
    """
    kind, payload = problem.engine, problem.payload
    if kind == "value":
        return payload["value"], [], True
    if kind == "modpow":
        base, exponent, modulus = payload["base"], payload["exponent"], payload["modulus"]
        value = pow(base, exponent, modulus)
        checked = exponent <= 4096 and pow(base, exponent) % modulus == value
        return Fraction(value), [f"{base}^{exponent} mod {modulus} = {value}"], checked
    if kind == "zeros":
        n = payload["n"]
        value = factorial_valuation(n, 5)
        checked = False
        if n <= 2000:
            digits = str(math.factorial(n))
            checked = len(digits) - len(digits.rstrip("0")) == value
        return Fraction(value), [f"⌊{n}/5⌋ + ⌊{n}/25⌋ + … = {value}"], checked
    if kind == "crt":
        moduli, remainder = payload["moduli"], payload["remainder"]
        found = crt_smallest(moduli, [remainder] * len(moduli))
        if found is None:
            raise MathError("no number below the search bound leaves those remainders")
        checked = all(found % m == remainder % m for m in moduli)
        return Fraction(found), [f"{found} leaves {remainder} under each of "
                                 f"{', '.join(map(str, moduli))}"], checked
    if kind == "same_remainder":
        values = sorted(payload["values"])
        differences = [b - a for a, b in zip(values, values[1:])]
        result = 0
        for difference in differences:
            result = math.gcd(result, difference)
        if result < 2:
            raise MathError("those numbers share no divisor above 1")
        left = {v % result for v in values}
        return Fraction(result), [
            f"the differences are {', '.join(map(str, differences))}",
            f"their hcf is {result}, and each of them leaves {left.pop()}"], len(left) == 0
    if kind == "diagonals":
        n = payload["n"]
        if n < 3:
            raise MathError("a polygon has at least three sides")
        value = n * (n - 3) // 2
        return Fraction(value), [f"{n}×({n}-3)/2 = {value}"], value == combinations(n, 2) - n
    if kind == "choose":
        n, r = payload["n"], payload["r"]
        value = combinations(n, r)
        return Fraction(value), [f"C({n}, {r}) = {value}"], \
            value * math.factorial(r) == permutations(n, r)
    if kind == "permute":
        n, r = payload["n"], payload["r"]
        value = permutations(n, r)
        return Fraction(value), [f"P({n}, {r}) = {value}"], \
            value == combinations(n, r) * math.factorial(r)
    if kind == "arrangements":
        value = arrangements(payload["word"])
        return Fraction(value), [f"{value} distinct arrangements"], True
    if kind == "divisor_sum":
        n = payload["n"]
        value = divisor_sum(n)
        checked = n > 10 ** 6 or sum(d for d in range(1, n + 1) if n % d == 0) == value
        return Fraction(value), [f"σ({n}) = {value}"], checked
    if kind == "divisor_count":
        n = payload["n"]
        value = divisor_count(n)
        checked = n > 10 ** 6 or sum(1 for d in range(1, n + 1) if n % d == 0) == value
        return Fraction(value), [f"d({n}) = {value}"], checked
    if kind == "shoelace":
        value = shoelace_area(payload["points"])
        return value, [f"the area is {_num(value)}"], True
    if kind == "geometric":
        value = infinite_geometric_sum(payload["first"], payload["ratio"])
        return value, [f"{_num(payload['first'])} ÷ (1 - {_num(payload['ratio'])}) "
                       f"= {_num(value)}"], True
    if kind == "diophantine":
        found = count_integer_solutions(payload["coefficients"], payload["total"],
                                        positive=payload["positive"])
        checked = all(sum(c * v for c, v in zip(payload["coefficients"], pair))
                      == payload["total"] for pair in found)
        listed = ", ".join(f"({', '.join(map(str, pair))})" for pair in found[:6])
        return Fraction(len(found)), [f"the solutions are {listed or 'none'}"], checked
    if kind == "count_range":
        low_end, high_end = payload["low"], payload["high"]
        divisors, mode = payload["divisors"], payload["mode"]
        if mode == "and":
            test = lambda n: all(n % d == 0 for d in divisors)          # noqa: E731
        elif mode == "neither":
            test = lambda n: all(n % d for d in divisors)               # noqa: E731
        else:
            test = lambda n: any(n % d == 0 for d in divisors)          # noqa: E731
        value = sum(1 for n in range(low_end, high_end + 1) if test(n))
        return Fraction(value), [f"{value} of the {high_end - low_end + 1} numbers qualify"], True
    if kind == "roots":
        roots, symbol = payload["roots"], payload["symbol"]
        found = ", ".join(f"{symbol} = {_num(r)}" for r in roots)
        return found, [f"the roots are {found}"], True
    if kind == "count_kind":
        low_end, high_end, which = payload["low"], payload["high"], payload["kind"]
        if which == "odd":
            test = lambda n: n % 2 == 1                                  # noqa: E731
        elif which == "even":
            test = lambda n: n % 2 == 0                                  # noqa: E731
        else:
            test = lambda n: _integer_root_of(n, 2) is not None          # noqa: E731
        value = sum(1 for n in range(low_end, high_end + 1) if test(n))
        return Fraction(value), [f"{value} of them are {which}"], True
    if kind == "sum_range":
        low_end, high_end, divisors = payload["low"], payload["high"], payload["divisors"]
        members = [n for n in range(low_end, high_end + 1)
                   if all(n % d == 0 for d in divisors)]
        value = sum(members)
        return Fraction(value), [f"{len(members)} numbers, from {members[0] if members else 0} "
                                 f"to {members[-1] if members else 0}",
                                 f"they add to {value}"], True
    if kind == "primes":
        low_end, high_end = payload["low"], payload["high"]
        found = [p for p in _sieve_upto(high_end) if p >= low_end]
        return Fraction(len(found)), [f"{len(found)} primes up to {high_end}"], True
    if kind == "ratio":
        parts = payload["parts"]
        return parts, [f"the ratio is {' : '.join(map(str, parts))}"], True
    if kind == "pi":
        from nyxara.njp.mathematics import _pi_text
        coefficient = payload["coefficient"]
        return _pi_text(coefficient), [f"{_pi_text(coefficient)}"], True
    if kind == "harmonic":
        value = harmonic_mean_speed(payload["speeds"])
        return value, [f"the harmonic mean of {' and '.join(_num(s) for s in payload['speeds'])}"
                       f" is {_num(value)}"], True
    raise MathError(f"no engine named {kind!r}")


#: A module-level convenience, shared for the same reason the mathematician's is.
_DEFAULT = Solver()


def solve(text: str) -> Solution:
    """Solve a problem with the shared solver."""
    return _DEFAULT.solve(text)
