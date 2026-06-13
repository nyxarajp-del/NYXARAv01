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
    "tautology", "extract_formula", "LogicFaculty",
    "solve_syllogism", "SyllogismFaculty",
    "solve_comparative", "ComparativeFaculty",
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
# Default registry + the router entry point
# --------------------------------------------------------------------------- #
def build_default_faculties(llm: object = None) -> Tuple[FacultyRegistry, FacultySelector]:
    """A registry of the real verifiable engines (+ the LLM generalist when given)."""
    reg = FacultyRegistry()
    reg.register(MathFaculty())
    reg.register(WordProblemFaculty())
    reg.register(LogicFaculty())
    reg.register(SyllogismFaculty())
    reg.register(ComparativeFaculty())
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
