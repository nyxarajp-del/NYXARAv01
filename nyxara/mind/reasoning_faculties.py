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
from typing import Dict, List, Optional, Set, Tuple

from nyxara.mind.faculties import (Faculty, FacultyRegistry, FacultySelector, Task, TaskType)
from nyxara.mind.proposal import ProposalKind
from nyxara.memory.provenance import SourceType

__all__ = [
    "extract_expression", "MathFaculty",
    "parse_word_problem", "WordProblemFaculty",
    "solve_percent", "PercentFaculty",
    "solve_unit_convert", "UnitConvertFaculty",
    "solve_equation", "AlgebraFaculty",
    "solve_calculus", "CalculusFaculty",
    "solve_sequence", "SequenceFaculty",
    "solve_date", "DateFaculty",
    "tautology", "extract_formula", "LogicFaculty",
    "solve_syllogism", "SyllogismFaculty",
    "solve_comparative", "ComparativeFaculty",
    "build_default_faculties", "solve_with_faculties", "solve_verifiable",
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
    # An ISO date (2020-01-31) is the DateFaculty's job — never let it look like "2020-1-31".
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text or ""):
        return None
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
# Natural-language elementary arithmetic (word problems)
# --------------------------------------------------------------------------- #
# The bare-expression faculty above deliberately ignores word problems. Offline (no teacher),
# that left NYXARA unable to solve even "3 boxes of 2 apples plus 1 more" herself. This faculty
# closes that gap WITHOUT bluffing: it reads the numbers and the cue words around them to build
# an exact expression for the general "rate × count (± loose amounts)" class — not by matching
# any fixed sentence. It fires only on a fully-assigned, unambiguous parse; on anything unclear
# it returns None so the question reaches the neural mind. The arithmetic itself is exact
# (MathEngine), so a fired answer is genuinely verifiable.
_WP_NUM = re.compile(r"\d[\d,]*\.?\d*")
_WP_MUL = re.compile(r"\b(?:each|every|per|apiece)\b", re.I)
_WP_SUB = re.compile(r"\b(?:left|remain\w*|fewer|lost|loses?|gave|gives?|given|ate|eats?|"
                     r"sold|sells?|spent|spend\w*|removed?|removes?|away|minus|drops?|"
                     r"dropped|fell|used|takes?\saway|took\saway)\b", re.I)
_WP_ADD = re.compile(r"\b(?:another|more|additional|extra|also|plus|added|adds?|joins?|"
                     r"joined|beside\w*|gains?|gained|receives?|received|bought|buys?|"
                     r"finds?|found)\b", re.I)
_WP_QUESTION = re.compile(r"\b(?:how many|how much|in total|altogether|in all|combined|"
                          r"total number|what is the total)\b", re.I)


def _wp_numbers(text: str) -> List[Tuple[int, float]]:
    """(position, value) for every integer-looking number in ``text``."""
    return [(m.start(), float(m.group().replace(",", ""))) for m in _WP_NUM.finditer(text)]


def parse_word_problem(text: str) -> Optional[str]:
    """Parse an elementary arithmetic word problem into an exact expression, or ``None``.

    Handles the general "rate × count, plus/minus standalone amounts" class by reading the
    numbers and the cue words near them ("each/per" → ×, "another/more" → +, "left/fewer" → −).
    Conservative by design: it requires a quantity question, 2–5 whole numbers, and that every
    number gets a role; if a multiplicative cue can't be paired cleanly, or any number is left
    unassigned, it returns ``None`` and defers — a verifiable faculty must never bluff."""
    s = (text or "").strip()
    low = s.lower()
    if not _WP_QUESTION.search(low):
        return None
    nums = _wp_numbers(low)
    if not 2 <= len(nums) <= 5:
        return None
    if any(v != int(v) for _, v in nums):       # decimals/percentages → defer
        return None

    used = [False] * len(nums)
    terms: List[Tuple[str, str]] = []           # (sign, expr_fragment)

    # 1) one multiplicative grouping — the numbers flanking an "each/per" cue multiply
    mul = _WP_MUL.search(low)
    if mul is not None:
        cue = mul.start()
        before = [i for i, (pos, _) in enumerate(nums) if pos < cue]
        after = [i for i, (pos, _) in enumerate(nums) if pos > cue]
        if before and after:
            i, j = before[-1], after[0]         # cue sits between the two factors
        elif len(before) >= 2 and not after:
            i, j = before[-2], before[-1]       # trailing cue ("... on every shelf")
        else:
            return None                         # "each" with nothing to pair → ambiguous
        a, b = int(nums[i][1]), int(nums[j][1])
        terms.append(("+", f"{a}*{b}"))
        used[i] = used[j] = True

    # 2) every remaining number is a standalone +/- term by its nearest preceding cue
    for i, (pos, val) in enumerate(nums):
        if used[i]:
            continue
        window = low[max(0, pos - 40):pos]
        sign = "-" if (_WP_SUB.search(window) and not _WP_ADD.search(window)) else "+"
        terms.append((sign, str(int(val))))
        used[i] = True

    if not all(used) or not terms:
        return None
    # without a product, require an explicit +/- cue so we never "solve" an incidental list
    if mul is None and not (_WP_ADD.search(low) or _WP_SUB.search(low)):
        return None

    expr = (terms[0][1] if terms[0][0] == "+" else f"-{terms[0][1]}")
    for sign, frag in terms[1:]:
        expr += f" {sign} {frag}"
    return expr


class WordProblemFaculty(Faculty):
    """Exact elementary arithmetic word problems — parsed, then computed (verifiable)."""

    name = "word-arithmetic"
    handles = frozenset({TaskType.ARITHMETIC})
    verifiable = True
    reliability = 0.95
    cost = 0.4

    def _expr(self, task: Task) -> Optional[str]:
        return parse_word_problem(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.9 if (task.type is TaskType.ARITHMETIC and self._expr(task)) else 0.0

    def handle(self, task: Task):
        from nyxara.mind.math import MathEngine
        expr = self._expr(task)
        value = MathEngine().evaluate(expr)
        return self._propose(_format_number(value), kind=ProposalKind.ANSWER,
                             confidence=0.95, rationale=f"parsed word problem -> {expr} = {value}",
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
# Categorical syllogisms — transitive closure over "all X are Y" (deductive, exact)
# --------------------------------------------------------------------------- #
# "All bloops are razzies, all razzies are lazzies. Are all bloops lazzies?" is a deduction,
# not a vibe: subset is transitive, so the answer is *proven* by reachability. We parse only
# universal-affirmative premises ("all X are Y" / "every X is a Y") and a yes/no "are all X Y?"
# question; anything with some/no/not (which changes the logic) is left to the neural mind.
_SYL_PREMISE = re.compile(r"(?<!not )\ball\s+([a-z]+)\s+are\s+([a-z]+)\b", re.I)
_SYL_PREMISE2 = re.compile(r"\bevery\s+([a-z]+)\s+is\s+(?:an?\s+)?([a-z]+)\b", re.I)
_SYL_QUESTION = re.compile(r"\b(?:are\s+all|is\s+every)\s+([a-z]+)\s+"
                           r"(?:definitely\s+|really\s+|always\s+|necessarily\s+)?"
                           r"(?:an?\s+)?([a-z]+)\b", re.I)


def _reachable(graph: Dict[str, Set[str]], start: str) -> Set[str]:
    """Every node reachable from ``start`` (transitive closure of a directed graph)."""
    seen: Set[str] = set()
    stack = list(graph.get(start, ()))
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(graph.get(n, ()))
    return seen


def solve_syllogism(text: str) -> Optional[str]:
    """Answer 'yes'/'no' to a categorical-syllogism question, or None to defer.

    Deductive: 'are all X (definitely) Y?' is *yes* iff Y is reachable from X through the
    'all … are …' subset edges, and otherwise *no* (not provable) — but only when X is a real
    category in the premises. Returns None on anything it cannot parse as a clean universal
    syllogism, so a verifiable faculty never overclaims."""
    low = (text or "").lower()
    q = _SYL_QUESTION.search(low)
    if q is None:
        return None
    subj, pred = q.group(1), q.group(2)
    graph: Dict[str, Set[str]] = {}
    for rx in (_SYL_PREMISE, _SYL_PREMISE2):
        for a, b in rx.findall(low):
            if a == b:
                continue
            graph.setdefault(a, set()).add(b)
    if not graph:
        return None
    # premises that hedge or weaken the universal (some/no/not) are out of this faculty's scope
    if re.search(r"\b(some|no|none|not|n't|most|few|only)\b",
                 re.sub(r"(?:options?|answer)\b.*$", "", low)):
        return None
    if pred in _reachable(graph, subj):
        return "yes"
    if subj in graph:                      # we know this category, but Y is not entailed
        return "no"
    return None                            # unknown subject — defer rather than guess


class SyllogismFaculty(Faculty):
    """Categorical syllogisms answered by transitive closure — a proof, not a guess."""

    name = "syllogism"
    handles = frozenset({TaskType.LOGIC})
    verifiable = True
    reliability = 1.0
    cost = 0.3

    def _answer(self, task: Task) -> Optional[str]:
        return solve_syllogism(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.LOGIC and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="categorical syllogism by transitive closure",
                             source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Transitive comparisons — "Tom is taller than Sam …; who is shortest?" (exact ordering)
# --------------------------------------------------------------------------- #
# A chain of "X is <adj>er than Y" defines a strict order on one scale; the superlative is its
# unique extremum. We pin each adjective to a (scale, polarity) so "taller" and "shortest" are
# understood as opposite ends of *height*. We fire only when every comparison shares one scale
# and the extremum is unique — otherwise we defer.
_ADJ: Dict[str, Tuple[str, int]] = {
    "tall": ("height", 1), "short": ("height", -1),
    "big": ("size", 1), "large": ("size", 1), "small": ("size", -1), "little": ("size", -1),
    "old": ("age", 1), "young": ("age", -1),
    "fast": ("speed", 1), "quick": ("speed", 1), "slow": ("speed", -1),
    "heavy": ("weight", 1), "light": ("weight", -1),
    "strong": ("strength", 1), "weak": ("strength", -1),
    "rich": ("wealth", 1), "poor": ("wealth", -1),
    "long": ("length", 1),
    "high": ("elevation", 1), "low": ("elevation", -1),
    "hot": ("temperature", 1), "cold": ("temperature", -1),
    "wide": ("width", 1), "narrow": ("width", -1),
    "deep": ("depth", 1), "shallow": ("depth", -1),
}
_CMP_STMT = re.compile(r"\b([A-Za-z][\w]*)\s+(?:is|was)\s+(?:much\s+|far\s+|a\s+bit\s+)?"
                       r"([a-z]+er)\s+than\s+([A-Za-z][\w]*)", re.I)
_CMP_QUESTION = re.compile(r"\bwho\s+(?:is|was)\s+(?:the\s+)?([a-z]+est)\b", re.I)


def _adj_info(word: str) -> Optional[Tuple[str, int]]:
    """Map a comparative/superlative adjective to its (scale, polarity), or None."""
    w = re.sub(r"[^a-z]", "", word.lower())
    base = w[:-3] if w.endswith("est") else (w[:-2] if w.endswith("er") else w)
    cands = {base, base + "e"}
    if len(base) >= 2 and base[-1] == base[-2]:
        cands.add(base[:-1])                       # bigg -> big
    for c in cands:
        if c in _ADJ:
            return _ADJ[c]
    return None


def solve_comparative(text: str) -> Optional[str]:
    """Answer 'who is the most/least …' from a chain of comparisons, or None to defer.

    Builds a strict order on a single scale ('X is taller than Y' → X > Y), then returns the
    unique top (for a '+' superlative like tallest) or bottom (for a '-' one like shortest).
    Conservative: one scale only, and a unique extremum, or it defers."""
    qm = _CMP_QUESTION.search(text or "")
    if qm is None:
        return None
    q_info = _adj_info(qm.group(1))
    if q_info is None:
        return None
    q_scale, q_pol = q_info

    greater: Dict[str, Set[str]] = {}              # canonical "is greater than" edges
    display: Dict[str, str] = {}
    nodes: Set[str] = set()
    found = False
    for x, adj, y in _CMP_STMT.findall(text or ""):
        info = _adj_info(adj)
        if info is None or info[0] != q_scale:     # unknown adj or a different scale → defer
            return None
        found = True
        _, pol = info
        hi, lo = (x, y) if pol == 1 else (y, x)
        hk, lk = hi.lower(), lo.lower()
        greater.setdefault(hk, set()).add(lk)
        display.setdefault(hk, hi); display.setdefault(lk, lo)
        nodes.update((hk, lk))
    if not found or len(nodes) < 2:
        return None

    closure = {n: _reachable(greater, n) for n in nodes}
    if q_pol == 1:                                  # want the maximum: greater than all others
        tops = [n for n in nodes if closure[n] == nodes - {n}]
    else:                                           # want the minimum: below all others
        tops = [n for n in nodes if all(n in closure[m] for m in nodes if m != n)]
    return display[tops[0]] if len(tops) == 1 else None


class ComparativeFaculty(Faculty):
    """Transitive comparison ordering — the superlative is the order's unique extremum."""

    name = "comparison"
    handles = frozenset({TaskType.LOGIC})
    verifiable = True
    reliability = 1.0
    cost = 0.3

    def _answer(self, task: Task) -> Optional[str]:
        return solve_comparative(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.LOGIC and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="transitive ordering over a single scale",
                             source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Percentages — "what is 20% of 150", "30 is what percent of 120" (exact, pure)
# --------------------------------------------------------------------------- #
# Pure-Python and exact: a percentage is a ratio, not a guess. We fire only on the two
# unambiguous canonical forms and defer otherwise, so we never "solve" an incidental "%".
_PCT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of|on)\s+(\d+(?:\.\d+)?)", re.I)
_PCT_WHAT = re.compile(r"(\d+(?:\.\d+)?)\s+is\s+what\s+(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", re.I)


def solve_percent(text: str) -> Optional[str]:
    """``P% of N`` → the amount; ``A is what % of B`` → the percentage. Else ``None``."""
    s = text or ""
    m = _PCT_WHAT.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b == 0:
            return None
        return _format_number(a / b * 100) + "%"
    m = _PCT_OF.search(s)
    if m:
        p, n = float(m.group(1)), float(m.group(2))
        return _format_number(p / 100.0 * n)
    return None


class PercentFaculty(Faculty):
    """Exact percentage arithmetic — a ratio computed, never estimated."""

    name = "percent"
    handles = frozenset({TaskType.ARITHMETIC})
    verifiable = True
    reliability = 1.0
    cost = 0.2

    def _answer(self, task: Task) -> Optional[str]:
        return solve_percent(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.96 if (task.type is TaskType.ARITHMETIC and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="exact percentage ratio", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Unit conversion — "convert 5 km to miles" (exact factors, pure)
# --------------------------------------------------------------------------- #
# A small table of exact (or standard) linear factors and the affine temperature pair. Each
# entry maps a unit to a canonical base; convert = value · from_factor / to_factor. We fire
# only when both units are known and share a dimension, and defer otherwise.
_UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "mile": "mile", "miles": "mile", "mi": "mile",
    "foot": "foot", "feet": "foot", "ft": "foot",
    "inch": "inch", "inches": "inch", "in": "inch",
    "yard": "yard", "yards": "yard", "yd": "yard",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
}
# canonical unit -> (dimension, factor-to-base). length base = metre, mass base = gram,
# volume base = litre.
_UNIT_FACTOR = {
    "km": ("length", 1000.0), "m": ("length", 1.0), "cm": ("length", 0.01),
    "mm": ("length", 0.001), "mile": ("length", 1609.344), "foot": ("length", 0.3048),
    "inch": ("length", 0.0254), "yard": ("length", 0.9144),
    "kg": ("mass", 1000.0), "g": ("mass", 1.0), "lb": ("mass", 453.59237),
    "oz": ("mass", 28.349523125),
    "l": ("volume", 1.0), "ml": ("volume", 0.001), "gallon": ("volume", 3.785411784),
}
_CONVERT = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s+(?:in|to|into|=)\s+([A-Za-z]+)", re.I)
_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?\s*)?\s*([cf])\b\s+(?:in|to|into)\s+"
                   r"(?:°|degrees?\s*)?\s*([cf])\b", re.I)


def solve_unit_convert(text: str) -> Optional[str]:
    """Convert between known units of one dimension (incl. °C↔°F), or ``None`` to defer."""
    s = text or ""
    tm = _TEMP.search(s)
    if tm:
        val, src, dst = float(tm.group(1)), tm.group(2).lower(), tm.group(3).lower()
        if src == dst:
            out = val
        elif src == "c":
            out = val * 9.0 / 5.0 + 32.0
        else:
            out = (val - 32.0) * 5.0 / 9.0
        return f"{_format_number(round(out, 4))}°{dst.upper()}"
    m = _CONVERT.search(s)
    if not m:
        return None
    val = float(m.group(1))
    src = _UNIT_ALIASES.get(m.group(2).lower())
    dst = _UNIT_ALIASES.get(m.group(3).lower())
    if src is None or dst is None:
        return None
    sd, sf = _UNIT_FACTOR[src]
    dd, df = _UNIT_FACTOR[dst]
    if sd != dd:                                   # different dimensions → defer
        return None
    out = val * sf / df
    return f"{_format_number(round(out, 6))} {m.group(3).lower()}"


class UnitConvertFaculty(Faculty):
    """Exact unit conversion over a fixed factor table — computed, not approximated by a net."""

    name = "unit-convert"
    handles = frozenset({TaskType.ARITHMETIC})
    verifiable = True
    reliability = 1.0
    cost = 0.2

    def _answer(self, task: Task) -> Optional[str]:
        return solve_unit_convert(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.94 if (task.type is TaskType.ARITHMETIC and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="exact unit conversion", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Algebra — solve an equation for its single unknown (SymPy, exact)
# --------------------------------------------------------------------------- #
# "solve 2x + 3 = 9", "x^2 = 4". We require an '=' and exactly one free variable, and we
# parse with implicit multiplication so "2x" means 2·x. Logic formulas (-> / <->) are not
# equations and are excluded. Degrades cleanly to None without SymPy.
_EQ_SPAN = re.compile(r"[0-9A-Za-z_.\s+\-*/^()]*=[0-9A-Za-z_.\s+\-*/^()]*")
# leading natural-language filler that is not part of the equation itself ("solve …", "find …")
_ALG_FILLER = frozenset((
    "solve", "find", "what", "is", "the", "value", "values", "of", "for", "roots",
    "root", "equation", "compute", "calculate", "determine", "where", "so", "that",
))


def _strip_filler(side: str) -> str:
    """Isolate the real expression on one side of '=': drop leading filler ('solve …'), then
    cut at the next filler word ('… = 9 for x' → '9'), so trailing prose never corrupts it."""
    toks = side.split()
    while toks and toks[0].lower().strip(".,:") in _ALG_FILLER:
        toks.pop(0)
    kept: List[str] = []
    for t in toks:
        if t.lower().strip(".,:") in _ALG_FILLER:
            break
        kept.append(t)
    return " ".join(kept)


def _parse_algebra(expr: str):
    """Parse one side of an equation with implicit multiplication; return a SymPy expr."""
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application,
                                            convert_xor)
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(expr, transformations=tr, evaluate=True)


def solve_equation(text: str) -> Optional[str]:
    """Solve ``text``'s embedded ``lhs = rhs`` for its single unknown, or ``None`` to defer."""
    from nyxara.mind.math import has_sympy
    if not has_sympy():
        return None
    s = text or ""
    if "->" in s or "<->" in s:                    # propositional logic, not algebra
        return None
    m = _EQ_SPAN.search(s)
    if not m or "=" not in m.group(0):
        return None
    span = m.group(0).strip().rstrip("?.! ")
    lhs, _, rhs = span.partition("=")
    lhs, rhs = _strip_filler(lhs.strip()), _strip_filler(rhs.strip())
    if not lhs or not rhs:
        return None
    try:
        import sympy as sp
        left, right = _parse_algebra(lhs), _parse_algebra(rhs)
        syms = sorted((left - right).free_symbols, key=str)
        if len(syms) != 1:                         # exactly one unknown, else defer
            return None
        var = syms[0]
        sols = sp.solve(sp.Eq(left, right), var)
        if not sols:
            return None
        rendered = ", ".join(_format_number(_pyify_sol(v)) for v in sols)
        return f"{var} = {rendered}"
    except Exception:  # noqa: BLE001 — anything unparseable defers to the neural mind
        return None


def _pyify_sol(v):
    """Render a SymPy solution cleanly (int when integral, else its exact form)."""
    try:
        if getattr(v, "is_integer", False):
            return int(v)
        if getattr(v, "is_rational", False) and not getattr(v, "is_integer", True):
            return str(v)
        if getattr(v, "is_real", False):
            f = float(v)
            return f if not float(f).is_integer() else int(f)
    except Exception:  # noqa: BLE001
        pass
    return str(v)


class AlgebraFaculty(Faculty):
    """Solve equations for their unknown via SymPy — an exact root, not a guess."""

    name = "algebra"
    handles = frozenset({TaskType.ALGEBRA})
    verifiable = True
    reliability = 1.0
    cost = 0.5

    def _answer(self, task: Task) -> Optional[str]:
        return solve_equation(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.ALGEBRA and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="solved the equation symbolically", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Calculus — derivative / integral / limit (SymPy, exact)
# --------------------------------------------------------------------------- #
_CALC_DERIV = re.compile(r"\b(?:derivative|differentiate|d/dx)\b", re.I)
_CALC_INTEG = re.compile(r"\b(?:integral|integrate|antiderivative)\b", re.I)
_CALC_LIMIT = re.compile(r"\blimit\b", re.I)
_WRT = re.compile(r"with\s+respect\s+to\s+([a-z])", re.I)
_LIMIT_AS = re.compile(r"as\s+([a-z])\s*(?:->|→|approaches?|to)\s*([0-9a-z.+\-]+)", re.I)
_CALC_CUE = re.compile(r"\b(?:derivative|differentiate|integral|integrate|antiderivative|"
                       r"limit|d/dx)\b", re.I)


def _calc_expr_tail(text: str, cue_end: int) -> Optional[str]:
    """The expression after the calculus cue word, stripped of 'of', 'dx', 'with respect to …'."""
    tail = text[cue_end:]
    tail = re.sub(r"^\s*(?:of|the)\b", "", tail, flags=re.I)   # "derivative OF x^2"
    if ":" in tail:                                            # "differentiate: x^2"
        tail = tail.split(":", 1)[1]
    tail = _WRT.sub("", tail)
    tail = _LIMIT_AS.sub("", tail)
    tail = re.sub(r"\bd[a-z]\b", "", tail)         # drop the 'dx' differential
    tail = tail.strip().rstrip("?.! ").strip()
    return tail or None


def solve_calculus(text: str) -> Optional[str]:
    """Differentiate / integrate / take a limit of an embedded expression, or ``None``.

    Parses the expression with implicit multiplication ("2x" → 2·x) so natural phrasings work,
    then computes exactly with SymPy. Anything unparseable returns ``None`` and defers."""
    from nyxara.mind.math import has_sympy
    if not has_sympy():
        return None
    import sympy as sp
    s = text or ""
    wrt = _WRT.search(s)
    var = sp.Symbol(wrt.group(1) if wrt else "x")
    try:
        lim = _CALC_LIMIT.search(s)
        if lim:
            la = _LIMIT_AS.search(s)
            expr = _calc_expr_tail(s, lim.end())
            if expr is None or la is None:
                return None
            pt = sp.oo if la.group(2) in ("oo", "inf", "infinity") else _parse_algebra(la.group(2))
            return str(sp.limit(_parse_algebra(expr), sp.Symbol(la.group(1)), pt))
        der = _CALC_DERIV.search(s)
        if der:
            expr = _calc_expr_tail(s, der.end())
            return None if expr is None else f"d/d{var} = " + str(sp.diff(_parse_algebra(expr), var))
        ig = _CALC_INTEG.search(s)
        if ig:
            expr = _calc_expr_tail(s, ig.end())
            return None if expr is None else "∫ = " + str(sp.integrate(_parse_algebra(expr), var)) + " + C"
    except Exception:  # noqa: BLE001 — unparseable calculus defers to the neural mind
        return None
    return None


class CalculusFaculty(Faculty):
    """Derivatives, integrals and limits via SymPy — exact symbolic calculus."""

    name = "calculus"
    handles = frozenset({TaskType.CALCULUS})
    verifiable = True
    reliability = 1.0
    cost = 0.6

    def _answer(self, task: Task) -> Optional[str]:
        return solve_calculus(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.CALCULUS and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="exact symbolic calculus", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Number sequences — the next term of an arithmetic / geometric progression (exact)
# --------------------------------------------------------------------------- #
# A run of ≥3 numbers with a constant difference (arithmetic) or constant ratio (geometric)
# *determines* its next term — that is deduction, not pattern-vibing. We fire only on a clean
# constant-difference or constant-ratio fit and defer on anything else.
_SEQ_NUMS = re.compile(r"-?\d+(?:\.\d+)?")
_SEQ_CUE = re.compile(r"\b(?:next|sequence|series|continue|comes?\s+next|pattern)\b|,\s*\?|…|\.\.\.",
                      re.I)


def solve_sequence(text: str) -> Optional[str]:
    """Next term of an arithmetic or geometric run of ≥3 numbers, or ``None`` to defer."""
    s = text or ""
    if not _SEQ_CUE.search(s):
        return None
    nums = [float(x) for x in _SEQ_NUMS.findall(s)]
    if len(nums) < 3:
        return None
    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    if all(abs(d - diffs[0]) < 1e-9 for d in diffs):       # arithmetic
        return _format_number(nums[-1] + diffs[0])
    if all(n != 0 for n in nums[:-1]):
        ratios = [nums[i + 1] / nums[i] for i in range(len(nums) - 1)]
        if all(abs(r - ratios[0]) < 1e-9 for r in ratios):  # geometric
            return _format_number(nums[-1] * ratios[0])
    return None


class SequenceFaculty(Faculty):
    """Next term of an arithmetic/geometric progression — deduced from a constant law."""

    name = "sequence"
    handles = frozenset({TaskType.SEQUENCE})
    verifiable = True
    reliability = 1.0
    cost = 0.3

    def _answer(self, task: Task) -> Optional[str]:
        return solve_sequence(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.9 if (task.type is TaskType.SEQUENCE and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="constant-law progression", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Calendar arithmetic — day-of-week, day spans, date shifts (stdlib, exact)
# --------------------------------------------------------------------------- #
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DAYS_BETWEEN = re.compile(r"\bdays?\s+between\b", re.I)
_WHAT_DAY = re.compile(r"\bwhat\s+(?:day|weekday)\b", re.I)
_SHIFT = re.compile(r"(\d+)\s+days?\s+(after|before|from)\b", re.I)


def solve_date(text: str) -> Optional[str]:
    """Day-of-week, span in days, or an N-day shift over ISO dates, or ``None`` to defer."""
    import datetime as _dt
    s = text or ""
    dates = [_dt.date(int(y), int(m), int(d)) for y, m, d in _ISO.findall(s)]
    if not dates:
        return None
    try:
        # two dates + a "between"/"days …" intent ⇒ a day span (tolerant of "days are there
        # between …" phrasings, not only the adjacent "days between").
        if len(dates) >= 2 and (re.search(r"\bbetween\b", s, re.I) or _DAYS_BETWEEN.search(s)):
            return str(abs((dates[1] - dates[0]).days))
        sh = _SHIFT.search(s)
        if sh:
            n = int(sh.group(1))
            delta = _dt.timedelta(days=n if sh.group(2).lower() != "before" else -n)
            return (dates[0] + delta).isoformat()
        if _WHAT_DAY.search(s):
            return dates[0].strftime("%A")
    except (ValueError, OverflowError):
        return None
    return None


class DateFaculty(Faculty):
    """Calendar arithmetic over ISO dates — exact via the standard library."""

    name = "date"
    handles = frozenset({TaskType.DATE})
    verifiable = True
    reliability = 1.0
    cost = 0.2

    def _answer(self, task: Task) -> Optional[str]:
        return solve_date(task.description or str(task.payload or ""))

    def suitability(self, task: Task) -> float:
        return 0.95 if (task.type is TaskType.DATE and self._answer(task)) else 0.0

    def handle(self, task: Task):
        ans = self._answer(task)
        return self._propose(ans, kind=ProposalKind.ANSWER, confidence=1.0,
                             rationale="exact calendar arithmetic", source=SourceType.TOOL)


# --------------------------------------------------------------------------- #
# Default registry + the router entry point
# --------------------------------------------------------------------------- #
def build_default_faculties(llm: object = None,
                            intuition: object = None) -> Tuple[FacultyRegistry, FacultySelector]:
    """A registry of the real verifiable engines (+ the LLM generalist when given).

    When an :class:`~nyxara.mind.intuition.IntuitionCore` is supplied, its non-algorithmic
    leap faculty is registered too — a *probabilistic* engine (so the verifiable bonus keeps
    exact engines ahead of it), giving the selector a real 'Aha!' fallback for sequence /
    analogy / reasoning tasks the exact engines cannot crack."""
    reg = FacultyRegistry()
    reg.register(MathFaculty())
    reg.register(WordProblemFaculty())
    reg.register(PercentFaculty())
    reg.register(UnitConvertFaculty())
    reg.register(AlgebraFaculty())
    reg.register(CalculusFaculty())
    reg.register(SequenceFaculty())
    reg.register(DateFaculty())
    reg.register(LogicFaculty())
    reg.register(SyllogismFaculty())
    reg.register(ComparativeFaculty())
    from nyxara.mind.first_principles import FirstPrinciplesFaculty
    reg.register(FirstPrinciplesFaculty())
    # her OWN generative engine handles GENERATION — the LLM is outranked there
    # (suitability 0.85 × reliability 0.75 / cost 0.5 = 1.275 vs the LLM's 0.18)
    from nyxara.mind.originality import CreativeFaculty
    reg.register(CreativeFaculty())
    if intuition is not None:
        from nyxara.mind.intuition import IntuitionFaculty
        reg.register(IntuitionFaculty(intuition))
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
    # specific structured types first, generic arithmetic last, so a date / equation / sequence
    # is never mis-grabbed by the bare-expression engine. First-principles derivation
    # (physics/chemistry/maths-proof) is tried before the bare engines so "balance …" or
    # "derive …" reaches the derivation faculty rather than being mis-parsed as arithmetic.
    for ttype in (TaskType.DATE, TaskType.CHEMISTRY, TaskType.PHYSICS, TaskType.DERIVATION,
                  TaskType.CALCULUS, TaskType.ALGEBRA, TaskType.SEQUENCE,
                  TaskType.ARITHMETIC, TaskType.LOGIC):
        task = Task(ttype, description=text, payload=text)
        fac = _DEFAULT.select(task)
        if fac is not None and fac.verifiable and fac.suitability(task) > 0:
            try:
                prop = _DEFAULT.dispatch(task)
            except Exception:  # noqa: BLE001 — a faculty failure defers to the neural mind
                continue
            return str(prop.content), float(prop.confidence)
    return None


def solve_verifiable(text: str) -> Optional[Tuple[str, float]]:
    """Exact, verifiable answer to ``text`` — a multi-step chain first, a single faculty second.

    This is the *whole* of NYXARA's verifiable reasoning in one place, so every caller (the
    integrated reasoner AND the confidence router) reaches the same capability. A multi-step
    reasoning chain (:func:`~nyxara.mind.reasoning_chain.solve_chain`) is tried first — it fires
    only on a binding/'then' and defers otherwise, so it never steals a single-shot question and a
    follow-up step is never lost to its head — then a single faculty (:func:`solve_with_faculties`).
    Returns ``(answer, confidence)`` or ``None`` to defer to the neural mind. Never raises."""
    try:
        from nyxara.mind.reasoning_chain import solve_chain
        chain = solve_chain(text)
        if chain is not None:
            return chain.render(), float(chain.confidence)
    except Exception:  # noqa: BLE001 — faculties are advisory; never crash a turn
        pass
    try:
        return solve_with_faculties(text)
    except Exception:  # noqa: BLE001
        return None
