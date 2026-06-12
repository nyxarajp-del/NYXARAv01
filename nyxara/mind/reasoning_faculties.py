"""NYXARA · mind/reasoning_faculties.py — concrete verifiable faculties (✦, Phase 4).

``mind/faculties.py`` is the registry + the law *"verifiable beats probabilistic."* What it
lacked was *engines*. This module supplies the real ones, so NYXARA is genuinely "more than an
LLM": exact arithmetic is **computed** (via the sympy-backed :class:`~nyxara.mind.math.MathEngine`),
and propositional validity is **proven** by truth table — not guessed by a neural net. The
router (Phase 2) consults these first: if a verifiable engine fits, its exact answer wins over
both the own model and the teacher. The neural mind proposes; the symbolic faculty checks.

Dependency-light: the math faculty degrades to a safe AST evaluator without sympy; the logic
faculty is pure standard library. Both produce a :class:`~nyxara.mind.proposal.Proposal`, never
an action — the kernel still disposes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from nyxara.mind.faculties import (Faculty, FacultyRegistry, FacultySelector, Task, TaskType)
from nyxara.mind.proposal import ProposalKind
from nyxara.memory.provenance import SourceType

__all__ = [
    "extract_expression", "MathFaculty",
    "tautology", "extract_formula", "LogicFaculty",
    "build_default_faculties", "solve_with_faculties",
]


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
_EXPR = re.compile(r"[0-9][0-9\s+\-*/().^%]*[0-9)]")


def _format_number(value) -> str:
    """Render a numeric result cleanly: an integral float prints without the .0."""
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def extract_expression(text: str) -> Optional[str]:
    """Pull a pure arithmetic expression out of ``text`` (None if it is not really one).

    Triggers only when there is a real binary operator between numbers — so word problems
    ("3 boxes and 2 apples …") do NOT masquerade as expressions and reach the neural mind."""
    norm = (text or "").replace("×", "*").replace("÷", "/").replace("^", "**")
    cands = [c.strip() for c in _EXPR.findall(norm.replace("**", "^"))]
    cands = [c.replace("^", "**") for c in cands if any(op in c for op in "+-*/%^")]
    if not cands:
        return None
    expr = max(cands, key=len)
    return expr if re.search(r"\d\s*[-+*/%]|\*\*", expr) else None


class MathFaculty(Faculty):
    """Exact arithmetic via :class:`~nyxara.mind.math.MathEngine` — verifiable, cheap, certain."""

    name = "math"
    handles = frozenset({TaskType.ARITHMETIC})
    verifiable = True
    reliability = 1.0
    cost = 0.2

    def _expr(self, task: Task) -> Optional[str]:
        return extract_expression(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.ARITHMETIC and self._expr(task)) else 0.0

    def handle(self, task: Task):
        from nyxara.mind.math import MathEngine
        expr = self._expr(task)
        value = MathEngine().evaluate(expr)
        return self._propose(_format_number(value), kind=ProposalKind.ANSWER,
                             confidence=1.0, rationale=f"computed {expr} = {value}",
                             source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Propositional logic (pure stdlib truth table)
# --------------------------------------------------------------------------- #
_LOGIC_TOKENS = re.compile(r"\s*(<->|->|[()&|~!]|\band\b|\bor\b|\bnot\b|\bimplies\b|\biff\b|"
                           r"[A-Za-z][A-Za-z0-9_]*)")


def _tokenize_logic(s: str) -> List[str]:
    out, i = [], 0
    while i < len(s):
        m = _LOGIC_TOKENS.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"bad token at {i}: {s[i]!r}")
        tok = m.group(1)
        low = tok.lower()
        # lowercase only the word connectives — variable names keep their case (A != a)
        out.append(low if low in ("and", "or", "not", "implies", "iff") else tok)
        i = m.end()
    return out


_CONNECTIVES = {"and", "&", "or", "|", "not", "~", "!", "->", "implies", "<->", "iff"}


class _Parser:
    """Recursive-descent: iff < implies < or < and < not < atom (standard precedence)."""

    def __init__(self, tokens: List[str]) -> None:
        self.t = tokens
        self.i = 0

    def _peek(self) -> Optional[str]:
        return self.t[self.i] if self.i < len(self.t) else None

    def _eat(self, tok: Optional[str] = None) -> str:
        cur = self._peek()
        if cur is None or (tok is not None and cur != tok):
            raise ValueError(f"expected {tok!r}, got {cur!r}")
        self.i += 1
        return cur

    def parse(self):
        node = self._iff()
        if self._peek() is not None:
            raise ValueError(f"trailing tokens: {self.t[self.i:]}")
        return node

    def _iff(self):
        left = self._implies()
        while self._peek() in ("<->", "iff"):
            self._eat()
            left = ("iff", left, self._implies())
        return left

    def _implies(self):
        left = self._or()
        if self._peek() in ("->", "implies"):
            self._eat()
            return ("implies", left, self._implies())   # right-associative
        return left

    def _or(self):
        left = self._and()
        while self._peek() in ("or", "|"):
            self._eat()
            left = ("or", left, self._and())
        return left

    def _and(self):
        left = self._not()
        while self._peek() in ("and", "&"):
            self._eat()
            left = ("and", left, self._not())
        return left

    def _not(self):
        if self._peek() in ("not", "~", "!"):
            self._eat()
            return ("not", self._not())
        return self._atom()

    def _atom(self):
        cur = self._peek()
        if cur == "(":
            self._eat("(")
            node = self._iff()
            self._eat(")")
            return node
        if cur is None or cur in _CONNECTIVES or cur in (")",):
            raise ValueError(f"expected a variable, got {cur!r}")
        self._eat()
        return ("var", cur)


def _vars(node) -> List[str]:
    if node[0] == "var":
        return [node[1]]
    return sorted({v for child in node[1:] for v in _vars(child)})


def _eval(node, env: dict) -> bool:
    op = node[0]
    if op == "var":
        return env[node[1]]
    if op == "not":
        return not _eval(node[1], env)
    a = _eval(node[1], env)
    b = _eval(node[2], env)
    if op == "and":
        return a and b
    if op == "or":
        return a or b
    if op == "implies":
        return (not a) or b
    if op == "iff":
        return a == b
    raise ValueError(f"unknown op {op}")


def tautology(formula: str) -> Tuple[bool, Optional[dict]]:
    """True iff ``formula`` holds under every assignment; else (False, a counterexample)."""
    node = _Parser(_tokenize_logic(formula)).parse()
    names = _vars(node)
    for mask in range(2 ** len(names)):
        env = {n: bool((mask >> k) & 1) for k, n in enumerate(names)}
        if not _eval(node, env):
            return False, env
    return True, None


# allow a thin natural-language wrapper around a bare formula
_LOGIC_WRAP = re.compile(r"^\s*(?:is|prove|show)?\s*(?:that\s+)?(?:the\s+)?"
                         r"(?:tautology|valid)?\s*:?\s*", re.I)
_LOGIC_TAIL = re.compile(r"\s*(?:a\s+)?(?:tautology|valid|always\s+true)\s*\??\s*$", re.I)


def extract_formula(text: str) -> Optional[str]:
    """Pull a propositional formula out of ``text`` — only if it really parses as one."""
    s = (text or "").strip()
    if "->" not in s and "<->" not in s and not re.search(r"\b(implies|iff)\b", s, re.I):
        return None                     # require a connective that signals formal logic
    s = _LOGIC_TAIL.sub("", _LOGIC_WRAP.sub("", s)).strip().rstrip("?").strip()
    try:
        _Parser(_tokenize_logic(s)).parse()
    except Exception:  # noqa: BLE001
        return None
    return s


class LogicFaculty(Faculty):
    """Propositional validity by truth table — a verifiable proof, not a guess."""

    name = "logic"
    handles = frozenset({TaskType.LOGIC})
    verifiable = True
    reliability = 1.0
    cost = 0.3

    def _formula(self, task: Task) -> Optional[str]:
        return extract_formula(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.LOGIC and self._formula(task)) else 0.0

    def handle(self, task: Task):
        formula = self._formula(task)
        valid, counter = tautology(formula)
        if valid:
            content = "valid (a tautology)"
        else:
            assign = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
            content = f"not valid (counterexample: {assign})"
        return self._propose(content, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale=f"truth table over {formula}", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Default registry + the router entry point
# --------------------------------------------------------------------------- #
def build_default_faculties(llm: object = None) -> Tuple[FacultyRegistry, FacultySelector]:
    """A registry of the real verifiable engines (+ the LLM generalist when given)."""
    reg = FacultyRegistry()
    reg.register(MathFaculty())
    reg.register(LogicFaculty())
    if llm is not None:
        from nyxara.mind.faculties import LLMFaculty
        reg.register(LLMFaculty(llm))
    return reg, FacultySelector(reg)


_DEFAULT: Optional[FacultySelector] = None


def solve_with_faculties(text: str) -> Optional[Tuple[str, float]]:
    """If a verifiable faculty can answer ``text`` exactly, return ``(answer, confidence)``.

    The router's neuro-symbolic short-circuit: exact computation/proof beats any neural guess.
    Returns ``None`` when nothing structured is recognised (the neural mind then handles it)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_default_faculties()[1]
    for ttype in (TaskType.ARITHMETIC, TaskType.LOGIC):
        task = Task(ttype, description=text, payload=text)
        fac = _DEFAULT.select(task)
        if fac is not None and fac.verifiable and fac.suitability(task) > 0:
            try:
                prop = _DEFAULT.dispatch(task)
            except Exception:  # noqa: BLE001 — a faculty failure defers to the neural mind
                continue
            return str(prop.content), float(prop.confidence)
    return None
