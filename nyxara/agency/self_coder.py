"""NYXARA · agency/self_coder.py — the local, LLM-free code synthesiser (✎, deterministic).

When NYXARA needs a computation she has no pre-built tool for, she does what an engineer
does — she **writes a program for it** — but she writes it *herself*, with her own
deterministic engine, **not** by prompting an external LLM. This module is that engine: it
turns a task stated in words into **real, runnable Python source** for the families it can
genuinely handle, and **fails honestly** for anything else (never a fake "solved" stub).

The honest contract — *real, not fake*:

* It covers a concrete, tested, extensible set of task families (arithmetic, number theory,
  sequences/aggregation, string/text). For each, it emits source that actually computes the
  answer when run.
* For a task outside that set it returns ``ok=False`` with a plain reason — it does **not**
  pretend. A caller with an LLM wired may then fall back to the model; a keyless NYXARA
  simply reports she cannot synthesise it locally.

The emitted programs are deliberately **import-free, standard-Python** wherever possible so
they pass the ephemeral synthesiser's fail-closed static scan
(:func:`~nyxara.agency.dynamic_tool_creator._scan_source`, which forbids OS imports and
``eval``/``exec``/``open``) *and* run unchanged through the isolated sandbox
(:func:`~nyxara.agency.code_sandbox.run_python`). Each program assigns its answer to the
conventional ``result`` variable (and prints it), which both execution paths surface.

For the arithmetic family it reuses the exact, LLM-free :class:`~nyxara.mind.math.MathEngine`
to *validate* the expression and pre-compute the expected value — so the source it hands back
is known-good before it ever runs. Pure standard library otherwise; MathEngine is imported
lazily and its absence is handled gracefully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

__all__ = [
    "SynthesisResult",
    "CodeSynthesizer",
    "synthesize",
]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class SynthesisResult:
    """The outcome of a local synthesis attempt — real source, or an honest miss."""
    ok: bool
    source: str = ""
    origin: str = "local"          # local:<family> on success, "none" on an honest miss
    family: str = ""               # arithmetic | number_theory | sequence | string
    expected: Any = None           # the locally-computed answer, when known ahead of the run
    note: str = ""                 # a plain reason on failure

    def to_dict(self) -> dict:
        return {"ok": self.ok, "source": self.source, "origin": self.origin,
                "family": self.family, "expected": self.expected, "note": self.note}


# --------------------------------------------------------------------------- #
# Small parsing helpers (bounded, deterministic)
# --------------------------------------------------------------------------- #
_INT_RE = re.compile(r"-?\d+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LIST_RE = re.compile(r"\[([^\]]*)\]")
_QUOTED_RE = re.compile(r"[\"'`]([^\"'`]*)[\"'`]")
# A pure arithmetic expression: digits, the four ops, power/mod/floordiv, parens, spaces, dot.
_ARITH_RE = re.compile(r"^[\s\d+\-*/%().]+$")

# Guardrails so a synthesised program can never blow up the sandbox with a pathological size.
_MAX_FACTORIAL_N = 5000
_MAX_FIB_N = 100_000
_MAX_PRIME_LIMIT = 5_000_000
_MAX_NTH_PRIME = 200_000


def _ints(text: str) -> List[int]:
    return [int(m) for m in _INT_RE.findall(text)]


def _nums(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUM_RE.findall(text):
        f = float(m)
        out.append(int(f) if f.is_integer() else f)
    return out


def _list_literal(text: str) -> Optional[List[Any]]:
    """Extract a ``[...]`` list of numbers if the task carries one, else None."""
    m = _LIST_RE.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    vals: List[Any] = []
    for p in parts:
        try:
            f = float(p)
            vals.append(int(f) if f.is_integer() else f)
        except ValueError:
            return None
    return vals


def _quoted(text: str) -> Optional[str]:
    m = _QUOTED_RE.search(text)
    return m.group(1) if m else None


def _py_list(values: List[Any]) -> str:
    """Render a list of numbers as a Python literal, import-free and scan-safe."""
    return "[" + ", ".join(repr(v) for v in values) + "]"


# --------------------------------------------------------------------------- #
# The synthesiser
# --------------------------------------------------------------------------- #
class CodeSynthesizer:
    """Deterministically write real Python for the families NYXARA can handle herself."""

    def synthesize(self, task: str) -> SynthesisResult:
        """Turn *task* (words) into runnable source, or an honest miss. Never raises."""
        if not task or not task.strip():
            return SynthesisResult(False, origin="none", note="empty task")
        t = task.strip()
        low = t.lower()
        # Ordered so the most specific families are tried before the generic arithmetic path.
        for family in (self._number_theory, self._sequence, self._string, self._arithmetic):
            try:
                res = family(t, low)
            except Exception as exc:  # noqa: BLE001 — a bad parse is a miss, never a crash
                res = None
                _ = exc
            if res is not None and res.ok:
                return res
        return SynthesisResult(
            False, origin="none",
            note="no local synthesiser matched this task; it needs an LLM or a new template")

    # ------------------------------------------------------------------ #
    # Family: arithmetic expressions — emit `result = (<expr>)`, MathEngine-validated
    # ------------------------------------------------------------------ #
    def _arithmetic(self, task: str, low: str) -> Optional[SynthesisResult]:
        # Pull the arithmetic core out of a phrase like "calculate 2 + 3 * 4".
        expr = self._extract_arith(task)
        if expr is None:
            return None
        # Validate + pre-compute with the exact, LLM-free math engine when available; fall back
        # to Python's own evaluation of a strictly-arithmetic AST otherwise.
        expected, ok = self._eval_arith(expr)
        if not ok:
            return None
        source = (f"# NYXARA self-authored program (arithmetic)\n"
                  f"result = ({expr})\n"
                  f"print(result)\n")
        return SynthesisResult(True, source=source, origin="local:arithmetic",
                               family="arithmetic", expected=expected)

    @staticmethod
    def _extract_arith(task: str) -> Optional[str]:
        # Prefer a trailing arithmetic expression; strip a leading verb like "calculate".
        cleaned = re.sub(r"(?i)\b(calculate|compute|evaluate|what\s+is|solve|the\s+value\s+of)\b",
                         " ", task)
        cleaned = cleaned.strip(" ?.=\t\n")
        if not cleaned or not _ARITH_RE.match(cleaned):
            return None
        if not any(op in cleaned for op in "+-*/%"):
            return None  # a bare number is not a computation worth a program
        # must contain at least one digit
        if not any(c.isdigit() for c in cleaned):
            return None
        return cleaned

    @staticmethod
    def _eval_arith(expr: str) -> Tuple[Any, bool]:
        try:
            from nyxara.mind.math import MathEngine, MathError
            try:
                return MathEngine().evaluate(expr), True
            except MathError:
                return None, False
        except Exception:  # noqa: BLE001 — MathEngine optional; fall back to a safe AST eval
            pass
        import ast
        import operator as op
        ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
               ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
               ast.USub: op.neg, ast.UAdd: op.pos}

        def ev(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.operand))
            raise ValueError("unsupported")

        try:
            return ev(ast.parse(expr, mode="eval")), True
        except Exception:  # noqa: BLE001
            return None, False

    # ------------------------------------------------------------------ #
    # Family: number theory — import-free algorithms with the parsed integer args
    # ------------------------------------------------------------------ #
    def _number_theory(self, task: str, low: str) -> Optional[SynthesisResult]:
        ints = _ints(task)

        if "factorial" in low and ints:
            n = ints[0]
            if not (0 <= n <= _MAX_FACTORIAL_N):
                return None
            expected = 1
            for i in range(2, n + 1):
                expected *= i
            src = (f"# NYXARA self-authored program (factorial of {n})\n"
                   f"n = {n}\n"
                   f"result = 1\n"
                   f"for i in range(2, n + 1):\n"
                   f"    result *= i\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if "fibonacci" in low and ints:
            n = ints[0]
            if not (0 <= n <= _MAX_FIB_N):
                return None
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            src = (f"# NYXARA self-authored program (fibonacci number {n})\n"
                   f"n = {n}\n"
                   f"a, b = 0, 1\n"
                   f"for _ in range(n):\n"
                   f"    a, b = b, a + b\n"
                   f"result = a\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", a)

        if ("prime" in low and ("factor" in low or "factorize" in low
                                or "factorise" in low)) and ints:
            n = abs(ints[0])
            expected = _prime_factors(n)
            src = (f"# NYXARA self-authored program (prime factors of {n})\n"
                   f"n = {n}\n"
                   f"result = []\n"
                   f"d = 2\n"
                   f"while d * d <= n:\n"
                   f"    while n % d == 0:\n"
                   f"        result.append(d)\n"
                   f"        n //= d\n"
                   f"    d += 1\n"
                   f"if n > 1:\n"
                   f"    result.append(n)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if ("nth prime" in low or re.search(r"\b\d+(st|nd|rd|th)\s+prime", low)) and ints:
            k = ints[0]
            if not (1 <= k <= _MAX_NTH_PRIME):
                return None
            expected = _nth_prime(k)
            src = (f"# NYXARA self-authored program ({k}th prime)\n"
                   f"k = {k}\n"
                   f"count = 0\n"
                   f"num = 1\n"
                   f"result = None\n"
                   f"while count < k:\n"
                   f"    num += 1\n"
                   f"    is_p = num > 1\n"
                   f"    d = 2\n"
                   f"    while d * d <= num:\n"
                   f"        if num % d == 0:\n"
                   f"            is_p = False\n"
                   f"            break\n"
                   f"        d += 1\n"
                   f"    if is_p:\n"
                   f"        count += 1\n"
                   f"        result = num\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if "primes" in low and ("up to" in low or "below" in low or "under" in low) and ints:
            limit = ints[0]
            if not (0 <= limit <= _MAX_PRIME_LIMIT):
                return None
            expected = _primes_up_to(limit)
            src = (f"# NYXARA self-authored program (primes up to {limit})\n"
                   f"limit = {limit}\n"
                   f"sieve = [True] * (limit + 1)\n"
                   f"result = []\n"
                   f"for p in range(2, limit + 1):\n"
                   f"    if sieve[p]:\n"
                   f"        result.append(p)\n"
                   f"        for q in range(p * p, limit + 1, p):\n"
                   f"            sieve[q] = False\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        # "is N prime" — checked AFTER nth-prime / primes-up-to / prime-factors so a bare
        # "prime" mention in those phrasings can't be misread as a primality test.
        if ("prime" in low and "primes" not in low and "factor" not in low
                and re.search(r"\bis\b", low) and ints):
            n = ints[0]
            expected = _is_prime(n)
            src = (f"# NYXARA self-authored program (is {n} prime?)\n"
                   f"n = {n}\n"
                   f"if n < 2:\n"
                   f"    result = False\n"
                   f"else:\n"
                   f"    result = True\n"
                   f"    d = 2\n"
                   f"    while d * d <= n:\n"
                   f"        if n % d == 0:\n"
                   f"            result = False\n"
                   f"            break\n"
                   f"        d += 1\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if ("gcd" in low or "greatest common" in low) and len(ints) >= 2:
            a, b = abs(ints[0]), abs(ints[1])
            expected = _gcd(a, b)
            src = (f"# NYXARA self-authored program (gcd of {a} and {b})\n"
                   f"a, b = {a}, {b}\n"
                   f"while b:\n"
                   f"    a, b = b, a % b\n"
                   f"result = a\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if ("lcm" in low or "least common" in low) and len(ints) >= 2:
            a, b = abs(ints[0]), abs(ints[1])
            g = _gcd(a, b)
            expected = 0 if g == 0 else a * b // g
            src = (f"# NYXARA self-authored program (lcm of {a} and {b})\n"
                   f"a, b = {a}, {b}\n"
                   f"x, y = a, b\n"
                   f"while y:\n"
                   f"    x, y = y, x % y\n"
                   f"result = 0 if x == 0 else a * b // x\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        if re.search(r"sum of (the )?first (\d+)", low):
            n = int(re.search(r"sum of (the )?first (\d+)", low).group(2))
            if not (0 <= n <= 10_000_000):
                return None
            expected = n * (n + 1) // 2
            src = (f"# NYXARA self-authored program (sum of first {n} integers)\n"
                   f"n = {n}\n"
                   f"result = 0\n"
                   f"for i in range(1, n + 1):\n"
                   f"    result += i\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:number_theory", "number_theory", expected)

        return None

    # ------------------------------------------------------------------ #
    # Family: sequences / aggregation over an explicit list
    # ------------------------------------------------------------------ #
    def _sequence(self, task: str, low: str) -> Optional[SynthesisResult]:
        values = _list_literal(task)
        if values is None:
            return None
        lit = _py_list(values)

        if "sort" in low:
            reverse = "desc" in low or "reverse" in low or "largest" in low
            expected = sorted(values, reverse=reverse)
            src = (f"# NYXARA self-authored program (sort a list)\n"
                   f"data = {lit}\n"
                   f"result = sorted(data, reverse={reverse})\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        if not values:
            return None  # the aggregations below need at least one element

        if "sum" in low or "total" in low:
            expected = sum(values)
            src = (f"# NYXARA self-authored program (sum a list)\n"
                   f"data = {lit}\n"
                   f"result = sum(data)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        if "max" in low or "largest" in low or "maximum" in low:
            expected = max(values)
            src = (f"# NYXARA self-authored program (max of a list)\n"
                   f"data = {lit}\n"
                   f"result = max(data)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        if "min" in low or "smallest" in low or "minimum" in low:
            expected = min(values)
            src = (f"# NYXARA self-authored program (min of a list)\n"
                   f"data = {lit}\n"
                   f"result = min(data)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        if "median" in low:
            expected = _median(values)
            src = (f"# NYXARA self-authored program (median of a list)\n"
                   f"data = sorted({lit})\n"
                   f"m = len(data)\n"
                   f"if m % 2:\n"
                   f"    result = data[m // 2]\n"
                   f"else:\n"
                   f"    result = (data[m // 2 - 1] + data[m // 2]) / 2\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        if "mean" in low or "average" in low or "avg" in low:
            expected = sum(values) / len(values)
            src = (f"# NYXARA self-authored program (mean of a list)\n"
                   f"data = {lit}\n"
                   f"result = sum(data) / len(data)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:sequence", "sequence", expected)

        return None

    # ------------------------------------------------------------------ #
    # Family: string / text transforms over a quoted literal
    # ------------------------------------------------------------------ #
    def _string(self, task: str, low: str) -> Optional[SynthesisResult]:
        s = _quoted(task)
        if s is None:
            return None
        lit = repr(s)

        if "reverse" in low:
            expected = s[::-1]
            src = (f"# NYXARA self-authored program (reverse a string)\n"
                   f"s = {lit}\n"
                   f"result = s[::-1]\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        if "palindrome" in low:
            norm = "".join(ch.lower() for ch in s if ch.isalnum())
            expected = norm == norm[::-1]
            src = (f"# NYXARA self-authored program (palindrome check)\n"
                   f"s = {lit}\n"
                   f"norm = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
                   f"result = norm == norm[::-1]\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        if "upper" in low:
            expected = s.upper()
            src = (f"# NYXARA self-authored program (uppercase a string)\n"
                   f"s = {lit}\n"
                   f"result = s.upper()\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        if "lower" in low:
            expected = s.lower()
            src = (f"# NYXARA self-authored program (lowercase a string)\n"
                   f"s = {lit}\n"
                   f"result = s.lower()\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        if "word" in low and "count" in low:
            expected = len(s.split())
            src = (f"# NYXARA self-authored program (word count)\n"
                   f"s = {lit}\n"
                   f"result = len(s.split())\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        if ("char" in low or "letter" in low or "length" in low) and (
                "count" in low or "length" in low or "how many" in low):
            expected = len(s)
            src = (f"# NYXARA self-authored program (character count)\n"
                   f"s = {lit}\n"
                   f"result = len(s)\n"
                   f"print(result)\n")
            return SynthesisResult(True, src, "local:string", "string", expected)

        return None


# --------------------------------------------------------------------------- #
# Reference implementations (for pre-computing `expected`, kept in-process)
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _prime_factors(n: int) -> List[int]:
    out: List[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def _nth_prime(k: int) -> Optional[int]:
    count, num, last = 0, 1, None
    while count < k:
        num += 1
        if _is_prime(num):
            count += 1
            last = num
    return last


def _primes_up_to(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    out: List[int] = []
    for p in range(2, limit + 1):
        if sieve[p]:
            out.append(p)
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _median(values: List[Any]) -> Any:
    data = sorted(values)
    m = len(data)
    if m % 2:
        return data[m // 2]
    return (data[m // 2 - 1] + data[m // 2]) / 2


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #
def synthesize(task: str) -> SynthesisResult:
    """Synthesise real Python for *task* with the default :class:`CodeSynthesizer`."""
    return CodeSynthesizer().synthesize(task)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.code_sandbox import run_python

    print("=" * 70)
    print("NYXARA self-coder self-test (local, LLM-free synthesis → real execution)")
    print("=" * 70)

    syn = CodeSynthesizer()

    def _check(task: str, want: Any) -> None:
        r = syn.synthesize(task)
        assert r.ok, f"expected a synthesis for {task!r}, got miss: {r.note}"
        run = run_python(r.source, timeout_s=10.0)
        assert run.ok, f"synthesised program for {task!r} failed: {run.error}\n{r.source}"
        assert run.value == want, (f"{task!r}: ran to {run.value!r}, expected {want!r}")
        assert r.expected == want, (f"{task!r}: pre-computed {r.expected!r}, expected {want!r}")
        print(f"  ok  [{r.origin:22}] {task!r} -> {run.value!r}")

    # arithmetic
    _check("calculate 2 + 3 * 4", 14)
    _check("what is (10 - 4) * 5", 30)
    _check("compute 100 // 7", 14)
    # number theory
    _check("factorial of 6", 720)
    _check("the 10th fibonacci number", 55)
    _check("is 97 prime", True)
    _check("is 91 a prime number", False)
    _check("prime factors of 360", [2, 2, 2, 3, 3, 5])
    _check("find the 8th prime", 19)
    _check("primes up to 20", [2, 3, 5, 7, 11, 13, 17, 19])
    _check("gcd of 48 and 36", 12)
    _check("lcm of 4 and 6", 12)
    _check("sum of the first 100 integers", 5050)
    # sequences
    _check("sum of [1, 2, 3, 4, 5]", 15)
    _check("sort the list [3, 1, 2]", [1, 2, 3])
    _check("max of [4, 9, 2]", 9)
    _check("mean of [2, 4, 6]", 4.0)
    _check("median of [1, 3, 2, 5]", 2.5)
    # strings
    _check("reverse 'nyxara'", "araxyn")
    _check("uppercase 'loyal'", "LOYAL")
    _check("word count in 'the master is sovereign'", 4)
    _check("is 'racecar' a palindrome", True)

    # honest miss — no fake success
    miss = syn.synthesize("design a distributed consensus protocol")
    assert not miss.ok and miss.origin == "none", "an unsupported task must miss honestly"
    print(f"\n  honest miss         : ok={miss.ok} note={miss.note!r}")

    print("\nALL SELF-TESTS PASSED ✓")
