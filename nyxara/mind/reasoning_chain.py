"""NYXARA · mind/reasoning_chain.py — multi-step verifiable reasoning (✦, Phase 4+).

A single faculty answers a single structured question. Real reasoning often needs *steps*:
bind a value, substitute it, compute; or compute a base and then apply a follow-up operation.
This module gives the sovereign offline mind a small, **exact** chain engine so she can solve

    "If x = 5, what is 2x + 3?"            → x=5 → 2·5+3 → 13
    "What is 20% of 150, then add 10?"     → 30 → 30+10 → 40

herself, with a shown derivation — not a neural guess. Like every faculty here it is honest:
it fires only on a fully-determined parse and returns ``None`` to defer otherwise, and the
arithmetic is exact (SymPy when present, a safe evaluator otherwise). The chain proposes only;
the kernel still disposes.

Pure/stdlib + optional SymPy; depends only on :mod:`mind.reasoning_faculties` (for the
implicit-multiplication parser and the bare-arithmetic recogniser) and :mod:`mind.math`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = ["ChainResult", "parse_bindings", "solve_chain"]


@dataclass
class ChainResult:
    """A worked multi-step answer: the final value plus the derivation that produced it."""

    answer: str
    confidence: float
    steps: List[str] = field(default_factory=list)

    def render(self) -> str:
        """A compact, human-readable derivation ending in the answer."""
        body = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        return (f"{body}\n  ⇒ {self.answer}" if self.steps else self.answer)


# --------------------------------------------------------------------------- #
# 1. Variable bindings + substitution  ("if x = 5 and y = 2, what is x*y + 1?")
# --------------------------------------------------------------------------- #
_BIND = re.compile(r"\b([a-z])\s*=\s*(-?\d+(?:\.\d+)?)\b", re.I)
_BIND_LEAD = re.compile(r"\b(?:if|let|given|suppose|assume|when|where|with)\b", re.I)
_TARGET = re.compile(r"\b(?:what\s+is|compute|evaluate|find|calculate|value\s+of|equals?)\b"
                     r"\s+(.+)$", re.I)


def parse_bindings(text: str) -> Dict[str, float]:
    """Variable→value bindings declared in ``text`` ('if x = 5 and y = 2' → {x:5, y:2})."""
    if not _BIND_LEAD.search(text or ""):
        return {}
    out: Dict[str, float] = {}
    for name, val in _BIND.findall(text or ""):
        out[name.lower()] = float(val)
    return out


def _eval_with(expr: str, env: Dict[str, float]):
    """Evaluate ``expr`` with the bindings substituted in; exact via SymPy when available."""
    from nyxara.mind.math import has_sympy
    if has_sympy():
        import sympy as sp
        from nyxara.mind.reasoning_faculties import _parse_algebra
        sub = {sp.Symbol(k): sp.nsimplify(v) for k, v in env.items()}
        result = _parse_algebra(expr).subs(sub)
        if not result.free_symbols:                       # fully determined → a number
            return sp.nsimplify(result)
        return None                                       # still has unknowns → defer
    # no SymPy: only the simplest "k*var" / "var (+|-) n" style, via the safe arithmetic path
    from nyxara.mind.math import MathEngine
    filled = re.sub(r"[a-z]", lambda m: str(env.get(m.group(0).lower(), m.group(0))), expr,
                    flags=re.I)
    if re.search(r"[a-z]", filled, re.I):                 # an unbound symbol remains → defer
        return None
    try:
        return MathEngine().evaluate(filled.replace("^", "**"))
    except Exception:  # noqa: BLE001
        return None


def _format(value) -> str:
    from nyxara.mind.reasoning_faculties import _format_number
    return _format_number(value)


def _substitution_chain(text: str) -> Optional[ChainResult]:
    env = parse_bindings(text)
    if not env:
        return None
    tm = _TARGET.search(text)
    if tm is None:
        return None
    target = tm.group(1).strip().rstrip("?.! ")
    # the target must actually use a bound variable — matched as a standalone letter, so "2x"
    # counts but "box" does not (a letter not flanked by other letters)
    if not any(re.search(rf"(?<![a-z]){re.escape(v)}(?![a-z])", target, re.I) for v in env):
        return None
    try:
        value = _eval_with(target, env)
    except Exception:  # noqa: BLE001 — anything unparseable defers to the neural mind
        return None
    if value is None:
        return None
    steps = [f"{k} = {_format(v)}" for k, v in env.items()]
    steps.append(f"{target} = {_format(value)}")
    return ChainResult(_format(value), 1.0, steps)


# --------------------------------------------------------------------------- #
# 2. Sequential follow-up op  ("what is 20% of 150, then add 10?")
# --------------------------------------------------------------------------- #
_THEN = re.compile(r"[,;]?\s*(?:and\s+)?then\s+(.+)$", re.I)
_OP = re.compile(r"\b(add|plus|increase\s+by|subtract|minus|less|decrease\s+by|"
                 r"times|multiply\s+by|multiplied\s+by|divide\s+by|divided\s+by)\b\s*"
                 r"(-?\d+(?:\.\d+)?)", re.I)
_OP_FUN = {
    "add": lambda a, b: a + b, "plus": lambda a, b: a + b, "increase by": lambda a, b: a + b,
    "subtract": lambda a, b: a - b, "minus": lambda a, b: a - b, "less": lambda a, b: a - b,
    "decrease by": lambda a, b: a - b,
    "times": lambda a, b: a * b, "multiply by": lambda a, b: a * b,
    "multiplied by": lambda a, b: a * b,
    "divide by": lambda a, b: a / b, "divided by": lambda a, b: a / b,
}


def _then_chain(text: str) -> Optional[ChainResult]:
    tm = _THEN.search(text)
    if tm is None:
        return None
    head = text[:tm.start()].strip().rstrip(",;")
    tail = tm.group(1).strip().rstrip("?.! ")
    op = _OP.search(tail)
    if op is None:
        return None
    # the head must itself be a structured, exactly-solvable question
    from nyxara.mind.reasoning_faculties import solve_with_faculties
    base = solve_with_faculties(head)
    if base is None:
        return None
    base_txt, _ = base
    m = re.search(r"-?\d+(?:\.\d+)?", base_txt)
    if m is None:                                         # head wasn't numeric → defer
        return None
    base_val = float(m.group(0))
    key = re.sub(r"\s+", " ", op.group(1).lower())
    fn = _OP_FUN.get(key)
    operand = float(op.group(2))
    if fn is None or (operand == 0 and "divide" in key):
        return None
    result = fn(base_val, operand)
    steps = [f"{head.strip()} = {_format(base_val)}",
             f"{_format(base_val)} {key} {_format(operand)} = {_format(result)}"]
    return ChainResult(_format(result), 0.97, steps)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def solve_chain(text: str) -> Optional[ChainResult]:
    """Solve a two-step structured problem exactly, or ``None`` to defer to the neural mind.

    Handles (1) variable binding + substitution and (2) a computed base with a follow-up
    arithmetic operation. Conservative: only a fully-determined parse fires."""
    for engine in (_substitution_chain, _then_chain):
        try:
            res = engine(text or "")
        except Exception:  # noqa: BLE001 — a chain failure is never fatal; defer
            res = None
        if res is not None:
            return res
    return None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA reasoning-chain self-test")
    print("=" * 70)

    r = solve_chain("If x = 5, what is 2x + 3?")
    print("\nsubstitution :", r.render() if r else None)
    assert r is not None and r.answer == "13"

    r2 = solve_chain("Given x = 4 and y = 2, what is x*y + 1?")
    print("multi-var    :", r2.render() if r2 else None)
    assert r2 is not None and r2.answer == "9"

    r3 = solve_chain("What is 20% of 150, then add 10?")
    print("then-op      :", r3.render() if r3 else None)
    assert r3 is not None and r3.answer == "40"

    assert solve_chain("How are you today?") is None
    assert solve_chain("What is the capital of France?") is None
    print("\nALL SELF-TESTS PASSED ✓")
