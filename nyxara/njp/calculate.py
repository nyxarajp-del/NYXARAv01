"""NYXARA · njp/calculate.py — the arithmetic she can actually do (🔢, NJP V.05).

Measured gap, not a suspected one: asked ``"2+2 kitna hai?"`` NJP returned the empty string. Every
stage worked and none of them could add. The ladder in :mod:`nyxara.njp.reason` has five rungs and
the deepest of them, ``PROOF``, calls :class:`~nyxara.njp.prove.ProofCore` — which **verifies a
proposition and never computes a value**, so an arithmetic question could at best be told that it
was not a well-formed claim. :mod:`nyxara.njp.prove` is the package's only sympy consumer;
``grep`` for an evaluator across ``nyxara/njp/`` returned nothing. She could store that fire causes
heat, cluster her own concepts, and rewrite her own source, and she could not work out two plus
two, because nothing in the brain had ever been asked to.

**This is a calculator, and it is deliberately only a calculator.** It computes the value of a
closed arithmetic expression and reports how it got there. It does not do algebra, does not solve
for unknowns, and does not know what any number means — :mod:`nyxara.njp.universe` owns quantities
that stand for something in the world, and this module never touches it.

**Nothing here is ever `eval`\\ ed.** The expression is parsed by :func:`ast.parse` and then walked
by :func:`_evaluate`, which accepts exactly seven node types and eight operators and raises on
everything else. A name, an attribute, a call, a subscript, a comprehension and a lambda are all
rejected at the node level — so a "question" like ``__import__('os').system('rm -rf /')`` is a
:class:`SyntaxError`-shaped refusal rather than a shell command. The whitelist is the security
boundary, and it is a whitelist precisely so that a Python version adding a new node type cannot
silently widen it.

**Exactness is reported, never assumed.** With sympy present ``1/3`` is the rational ``1/3`` and
says so; without it the answer is the float ``0.333…`` and says *that*. The two are different
claims about the same question and the caller is told which one it has, because a brain whose
whole point is calibrated honesty should not round silently.

**Hinglish counts as a first-class question form.** ``"2+2 kitna hai"``, ``"15 ka 20 percent"``,
``"5 ka square"`` and ``"what is 15% of 200"`` are the same request in different clothes, and the
extractor in :func:`expression_in` reads all of them. What it will not do is guess: a sentence with
no closed expression in it returns ``None`` and the turn goes on to something that might know.

Pure standard library, with sympy as an optional upgrade. Every public entry point is fail-soft.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["Evaluation", "Calculator", "expression_in", "evaluate"]


# --------------------------------------------------------------------------- #
# The whitelist. This is the security boundary — see the module docstring.
# --------------------------------------------------------------------------- #
#
# Binary operators she may apply. `ast.Pow` is here and is the one that needs a guard rather than
# a ban: exponentiation is ordinary arithmetic ("5 ka square") but it is also the one operator
# whose *output size* grows without bound in the exponent, so `10**10**10` is a denial of service
# written in four characters. `_MAX_POW` bounds it below the point where that matters.
_BINARY: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Above this the result stops being an answer and starts being a memory-exhaustion bug.
_MAX_POW = 512
#: The widest integer she will return in full. Python's own limit on int→str conversion is the
#: real reason: past ~4300 digits `str(n)` raises, so a "successful" evaluation would then fail
#: at the moment it was rendered — a much more confusing failure than declining up front.
_MAX_DIGITS = 4300


class CalculationError(ValueError):
    """A refusal with a reason attached. Never raised out of a public entry point."""


# --------------------------------------------------------------------------- #
# Natural language → a closed expression
# --------------------------------------------------------------------------- #

# Hinglish and English number-word forms that are arithmetic wearing a sentence. Each rewrites to
# ordinary infix, so exactly one evaluator sees exactly one grammar. Order matters: the percent
# forms must be tried before the bare "X ka Y" possessive, or "15 ka 20 percent" reads as a
# multiplication of 15 by 20.
_REWRITES: Tuple[Tuple[str, str], ...] = (
    # "15% of 200" / "15 percent of 200" / "200 ka 15 percent" / "200 ka 15%"
    (r"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent|pratishat|प्रतिशत)\s*(?:of|ka|ki|के|का)\s+"
     r"(\d+(?:\.\d+)?)", r"(\2 * \1 / 100)"),
    (r"(\d+(?:\.\d+)?)\s+(?:ka|ki|ke|का|की|के)\s+(\d+(?:\.\d+)?)\s*"
     r"(?:%|percent|per\s*cent|pratishat|प्रतिशत)", r"(\1 * \2 / 100)"),
    # "square of 5" / "5 ka square" / "5 ka varg"
    (r"(?:square|varg|वर्ग)\s+(?:of|ka|का)\s+(\d+(?:\.\d+)?)", r"(\1 ** 2)"),
    (r"(\d+(?:\.\d+)?)\s+(?:ka|ki|ke|का|की|के)\s+(?:square|varg|वर्ग)", r"(\1 ** 2)"),
    (r"(?:cube|ghan|घन)\s+(?:of|ka|का)\s+(\d+(?:\.\d+)?)", r"(\1 ** 3)"),
    (r"(\d+(?:\.\d+)?)\s+(?:ka|ki|ke|का|की|के)\s+(?:cube|ghan|घन)", r"(\1 ** 3)"),
    # Spelled-out operators, English and Hinglish alike.
    (r"\s+(?:plus|add(?:ed)?\s+to|jod|जोड़)\s+", r" + "),
    (r"\s+(?:minus|less|ghata|घटा)\s+", r" - "),
    (r"\s+(?:times|multiplied\s+by|into|guna|गुणा)\s+", r" * "),
    (r"\s+(?:divided\s+by|over|bhag|भाग)\s+", r" / "),
    (r"\s*(?:×|✕|✖)\s*", r" * "),
    (r"\s*(?:÷)\s*", r" / "),
    (r"\s*(?:−|–|—)\s*", r" - "),
)

# What a closed arithmetic expression looks like once the rewrites have run: numbers, operators,
# brackets and spaces, nothing else. Anchored on a digit at each end so a trailing "hai" or a
# leading "what is" cannot be swallowed into it, and requiring at least one operator so a
# sentence that merely *mentions* a number ("Ravi is 30") is not mistaken for a sum.
_EXPRESSION = re.compile(
    r"(?<![\w.])"
    r"(\(*\s*-?\s*\d+(?:\.\d+)?\s*(?:[-+*/%^]|\*\*|//)\s*[-+*/%^(). \d]*\d\s*\)*)"
    r"(?![\w.])")

#: Stripped before parsing — the sentence around the sum, in both languages.
_FRAME = re.compile(
    r"\b(?:what|whats|what's|how\s+much|calculate|compute|evaluate|solve|tell\s+me|"
    r"kitna|kitne|kitni|batao|bataao|bata|bataiye|btao|hota|hoti|hai|hain|ka|ki|ke|"
    r"कितना|कितने|बताओ|होता|होती|है|हैं)\b",
    re.IGNORECASE)


def expression_in(text: str) -> Optional[str]:
    """The closed arithmetic expression this sentence contains, or ``None``.

    ``None`` is the important half of the contract. A sentence with no sum in it must come back
    empty-handed so the turn can go on to an organ that might know — a calculator that answers
    "how are you" with a number is the same class of failure as a physics law answering a
    greeting, which :mod:`nyxara.njp.relevance` exists to prevent.
    """
    try:
        low = str(text or "").strip().lower()
        if not low:
            return None
        # `^` means exponent everywhere outside Python; it is XOR inside it, and quietly
        # returning 5 XOR 2 for "5^2" would be a wrong answer stated confidently.
        for pattern, replacement in _REWRITES:
            low = re.sub(pattern, replacement, low, flags=re.IGNORECASE)
        low = low.replace("^", "**")
        match = _EXPRESSION.search(low)
        if match is None:
            return None
        expression = _FRAME.sub(" ", match.group(1)).strip()
        return expression or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #

@dataclass
class Evaluation:
    """One computed value, with the working shown and its exactness named."""

    expression: str = ""
    value: Any = None
    text: str = ""
    exact: bool = False
    steps: List[str] = field(default_factory=list)
    backend: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Did this produce a number she may state?"""
        return self.value is not None and not self.error

    def to_dict(self) -> Dict[str, Any]:
        return {"expression": self.expression, "value": self.text, "exact": self.exact,
                "steps": self.steps[:8], "backend": self.backend, "error": self.error,
                "ok": self.ok}


def _check_pow(base: Any, exponent: Any) -> None:
    """Refuse an exponentiation whose *result* would be the problem."""
    try:
        if abs(float(exponent)) > _MAX_POW:
            raise CalculationError(f"exponent {exponent} is above the {_MAX_POW} ceiling")
        if abs(float(base)) > 1 and abs(float(base)) ** abs(float(exponent)) == math.inf:
            raise CalculationError("that overflows to infinity")
    except CalculationError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise CalculationError("that is too large to compute")


def _evaluate(node: ast.AST) -> Any:
    """Walk the parsed tree over the whitelist. Anything unlisted is a refusal, not a default."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    # `ast.Num` is deprecated but still produced by older Pythons; `ast.Constant` covers current.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, complex)):
            raise CalculationError("only numbers can be computed with")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINARY.get(type(node.op))
        if op is None:
            raise CalculationError(f"{type(node.op).__name__} is not an arithmetic operator")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculationError("division by zero")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise CalculationError(f"{type(node.op).__name__} is not an arithmetic operator")
        return op(_evaluate(node.operand))
    raise CalculationError(f"{type(node).__name__} is not allowed in an arithmetic expression")


class Calculator:
    """Evaluates closed arithmetic, exactly where it can and approximately where it must.

    Stateless and therefore safe to share. It keeps counters only so that ``stats()`` can say
    whether it has ever actually been reached on a turn — the failure this module was written for
    was not a wrong answer, it was an organ nothing called.
    """

    def __init__(self, *, prefer_exact: bool = True) -> None:
        self.prefer_exact = bool(prefer_exact)
        self.asked = 0
        self.evaluated = 0
        self.refused = 0
        self._sympy: Any = None
        self._sympy_tried = False

    # ---- optional exact backend ------------------------------------------- #
    def _exact(self) -> Any:
        """sympy if it is installed, else ``None``. Imported once, lazily, never required."""
        if not self._sympy_tried:
            self._sympy_tried = True
            try:
                import sympy
                self._sympy = sympy
            except Exception:  # noqa: BLE001
                self._sympy = None
        return self._sympy

    # ---- the public call --------------------------------------------------- #
    def evaluate(self, text: str) -> Evaluation:
        """Compute what this sentence asks for, or say why she will not.

        Fail-soft: every refusal comes back as an :class:`Evaluation` carrying ``error``, so a
        caller that forgets to check ``ok`` gets an unusable value rather than an exception in
        the middle of a turn.
        """
        out = Evaluation()
        try:
            expression = expression_in(text)
            if not expression:
                out.error = "no arithmetic expression in that"
                return out
            out.expression = expression
            self.asked += 1

            try:
                tree = ast.parse(expression, mode="eval")
            except SyntaxError:
                self.refused += 1
                out.error = "that is not a well-formed expression"
                return out

            # Float first, always — it is the value that decides whether the answer is even
            # representable, and it is the one that cannot hang. sympy is then asked to sharpen
            # it, never to replace the check.
            try:
                value = _evaluate(tree)
            except CalculationError as exc:
                self.refused += 1
                out.error = str(exc)
                return out
            except (OverflowError, ZeroDivisionError, ValueError) as exc:
                self.refused += 1
                out.error = str(exc) or "that cannot be computed"
                return out

            if isinstance(value, int) and len(str(abs(value))) > _MAX_DIGITS:
                self.refused += 1
                out.error = f"the answer has more than {_MAX_DIGITS} digits"
                return out

            out.value, out.backend, out.exact = value, "python", isinstance(value, int)
            out.text = self._render(value)
            out.steps = [f"{expression} = {out.text}"]

            # Sharpen to an exact rational where one exists and the float was not already exact.
            if self.prefer_exact and not out.exact:
                sharpened = self._sharpen(expression)
                if sharpened is not None:
                    exact_text, exact_value = sharpened
                    out.exact = True
                    out.backend = "sympy"
                    out.value = exact_value
                    out.steps.append(f"exactly {exact_text}")
                    # **A question asked in decimals is answered in decimals.** `0.1 + 0.02 +
                    # 0.003` is exactly 123/1000, and 123/1000 is a strange thing to say back to
                    # somebody who wrote three decimals — the fraction is the same claim in a
                    # spelling nobody asked for. The exactness is unchanged; only the rendering
                    # follows the question, and only where the decimal terminates.
                    out.text = self._as_asked(expression, exact_text, exact_value)
            self.evaluated += 1
            return out
        except Exception as exc:  # noqa: BLE001
            out.error = f"calculation failed: {exc}"
            return out

    @staticmethod
    def _as_asked(expression: str, exact_text: str, exact_value: Any) -> str:
        """The exact value written the way the question was written, where the two can agree."""
        if "." not in expression or "/" not in exact_text:
            return exact_text
        try:
            top, _, bottom = exact_text.partition("/")
            numerator, denominator = int(top), int(bottom)
            trimmed = denominator
            for prime in (2, 5):
                while trimmed % prime == 0:
                    trimmed //= prime
            if trimmed != 1:
                return exact_text        # a repeating decimal is a worse claim than the fraction
            digits = len(str(denominator)) + 4
            rendered = f"{numerator / denominator:.{digits}f}".rstrip("0").rstrip(".")
            return rendered or exact_text
        except Exception:  # noqa: BLE001
            return exact_text

    def _sharpen(self, expression: str) -> Optional[Tuple[str, Any]]:
        """The exact rational for this expression, when sympy is present and it has one."""
        sympy = self._exact()
        if sympy is None:
            return None
        try:
            # `evaluate=False` is deliberately NOT used: it defers the arithmetic and returns an
            # unevaluated tree. What makes this call safe is not a sympy flag but the fact that
            # `_evaluate` has already walked this exact string over the whitelist above — sympy
            # never sees an expression the AST walker has not already accepted.
            exact = sympy.sympify(expression, rational=True)
            if not exact.is_number:
                return None
            simplified = sympy.nsimplify(exact, rational=True)
            if not simplified.is_Rational:
                return None
            # A rational whose denominator is 1 is an exact *integer* that only looked
            # inexact because a division produced it. "15% of 200" is 30, not 30.0-ish, and
            # reporting it as approximate understates what she actually knows — the same
            # dishonesty as overstating it, pointed the other way.
            if simplified.q == 1:
                return str(simplified.p), int(simplified.p)
            return f"{simplified.p}/{simplified.q}", simplified
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _render(value: Any) -> str:
        """A number as she should say it: no trailing zeros, no float noise on a whole number."""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer() and abs(value) < 1e15:
                return str(int(value))
            # 12 significant figures is past double precision's honest reach, so anything the
            # repr would show beyond it is noise rather than information.
            return f"{value:.12g}"
        return str(value)

    def stats(self) -> Dict[str, Any]:
        return {"asked": self.asked, "evaluated": self.evaluated, "refused": self.refused,
                "exact_backend": self._exact() is not None, "prefer_exact": self.prefer_exact}


#: A module-level convenience for callers that do not want to hold an instance.
_DEFAULT = Calculator()


def evaluate(text: str) -> Evaluation:
    """Compute the arithmetic in ``text`` using the shared calculator."""
    return _DEFAULT.evaluate(text)
