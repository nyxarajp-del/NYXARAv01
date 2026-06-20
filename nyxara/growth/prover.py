"""NYXARA · growth/prover.py — proof-carrying answers (✔, Pillar F · Edge 2, Rule 6).

When every mind is an AGI, *sounding* confident is worthless — everyone sounds confident. The
edge belongs to the mind whose answers come with a **machine-checkable certificate**: not "trust
me," but "here is the proof, check it yourself." Generation is hard; **verification is cheap** —
so NYXARA can certify her own answers, and just as importantly *check a rival mind's* answer,
keeping only what survives.

This module gives her a small, honest theorem-checker over the domains where truth is decidable:

* **arithmetic** — exact evaluation over the rationals (``fractions.Fraction``), so ``2+2=5``
  is *refuted*, never argued.
* **algebra** — symbolic identity / equation-solution checking via ``sympy`` when present,
  otherwise polynomial-identity testing at random integer points (Schwartz–Zippel), and exact
  substitution to *verify a candidate solution*.
* **logic** — propositional validity by full truth-table enumeration (pure stdlib, exact for
  small formulas), with ``z3`` as an optional accelerator.
* **inequality** — linear inequalities via ``z3`` when present, else candidate-substitution.
* **number_theory** — primality / gcd / divisibility by exact stdlib computation.

The contract is the same discipline the :class:`~nyxara.growth.scientist.Scientist` already keeps:
a verifiable check **never bluffs**. If a claim is outside what can be decided here, the verdict is
``UNPROVABLE`` — honest abstention, not a guess. Pure-stdlib core; ``sympy`` / ``z3`` are optional
and only ever *strengthen* a result. Touches no source, no weights, no gate.
"""

from __future__ import annotations

import ast
import math
import operator
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["ProofVerdict", "ProofClaim", "ProofResult", "Prover"]


class ProofVerdict(str, Enum):
    """The outcome of a proof attempt — honest about the third state."""

    PROVEN = "proven"
    REFUTED = "refuted"
    UNPROVABLE = "unprovable"   # outside what this checker can decide — abstain, never bluff


@dataclass
class ProofClaim:
    """A claim to certify. ``kind`` selects the decision procedure."""

    kind: str                                   # arithmetic | algebra | logic | inequality | number_theory
    statement: str
    candidate_answer: Optional[str] = None      # e.g. "x = 2" — verified by substitution

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "statement": self.statement,
                "candidate_answer": self.candidate_answer}


@dataclass
class ProofResult:
    """A verdict plus the *certificate* — the witness anyone can re-check independently."""

    verdict: ProofVerdict
    certificate: str = ""
    method: str = ""
    confidence: float = 1.0
    claim: Optional[ProofClaim] = None
    at: float = field(default_factory=lambda: __import__("time").time())

    @property
    def proven(self) -> bool:
        return self.verdict is ProofVerdict.PROVEN

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "certificate": self.certificate,
                "method": self.method, "confidence": round(float(self.confidence), 4),
                "claim": self.claim.to_dict() if self.claim else None, "at": self.at}

    def summary(self) -> str:
        return (f"[{self.verdict.value.upper()}] {self.method}"
                + (f" — {self.certificate}" if self.certificate else ""))


class Prover:
    """Certify claims in decidable domains; abstain (``UNPROVABLE``) everywhere else."""

    def __init__(self, *, seed: int = 1337, identity_points: int = 24) -> None:
        self.rng = random.Random(seed)
        self.identity_points = max(4, int(identity_points))

    # ------------------------------------------------------------------ #
    # Public entry point — dispatch by kind, never raise.
    # ------------------------------------------------------------------ #
    def prove(self, claim: ProofClaim) -> ProofResult:
        try:
            dispatch: Dict[str, Callable[[ProofClaim], ProofResult]] = {
                "arithmetic": self._prove_arithmetic,
                "algebra": self._prove_algebra,
                "logic": self._prove_logic,
                "inequality": self._prove_inequality,
                "number_theory": self._prove_number_theory,
            }
            fn = dispatch.get((claim.kind or "").strip().lower())
            if fn is None:
                return self._abstain(claim, f"no decision procedure for kind={claim.kind!r}")
            return fn(claim)
        except Exception as exc:  # noqa: BLE001 — a checker never throws; it abstains
            return self._abstain(claim, f"checker error: {exc}")

    def check_answer(self, kind: str, statement: str, candidate: str) -> ProofResult:
        """Convenience: certify that ``candidate`` is correct for ``statement``."""
        return self.prove(ProofClaim(kind=kind, statement=statement, candidate_answer=candidate))

    # ------------------------------------------------------------------ #
    # arithmetic — exact over the rationals
    # ------------------------------------------------------------------ #
    def _prove_arithmetic(self, claim: ProofClaim) -> ProofResult:
        stmt = claim.statement.strip()
        if "=" in stmt and not any(s in stmt for s in ("<=", ">=", "==", "<", ">")):
            lhs, rhs = stmt.split("=", 1)
            lv, rv = _eval_rational(lhs), _eval_rational(rhs)
            if lv is None or rv is None:
                return self._abstain(claim, "could not evaluate both sides exactly")
            ok = lv == rv
            return ProofResult(
                ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                certificate=f"{_fmt(lv)} {'=' if ok else '≠'} {_fmt(rv)}",
                method="exact rational evaluation", confidence=1.0, claim=claim)
        # expression + candidate answer
        val = _eval_rational(stmt)
        if val is None:
            return self._abstain(claim, "not a closed arithmetic expression")
        if claim.candidate_answer is not None:
            cand = _eval_rational(_rhs_of(claim.candidate_answer))
            if cand is None:
                return self._abstain(claim, "candidate answer is not numeric")
            ok = cand == val
            return ProofResult(
                ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                certificate=f"{stmt} = {_fmt(val)}; candidate {_fmt(cand)}",
                method="exact rational evaluation", confidence=1.0, claim=claim)
        return ProofResult(ProofVerdict.PROVEN, certificate=f"{stmt} = {_fmt(val)}",
                           method="exact rational evaluation", confidence=1.0, claim=claim)

    # ------------------------------------------------------------------ #
    # algebra — identity (sympy / PIT) or candidate-solution check (substitution)
    # ------------------------------------------------------------------ #
    def _prove_algebra(self, claim: ProofClaim) -> ProofResult:
        stmt = claim.statement
        if "=" not in stmt:
            return self._abstain(claim, "no equation/identity to check")
        lhs, rhs = stmt.split("=", 1)
        var = _first_var(stmt)

        # Path 1: a candidate solution -> verify exactly by substitution (always available).
        if claim.candidate_answer is not None and var is not None:
            cand = _eval_rational(_rhs_of(claim.candidate_answer))
            if cand is None:
                return self._abstain(claim, "candidate answer is not numeric")
            lv = _eval_rational(lhs, {var: cand})
            rv = _eval_rational(rhs, {var: cand})
            if lv is None or rv is None:
                return self._abstain(claim, "could not substitute candidate")
            ok = lv == rv
            return ProofResult(
                ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                certificate=f"{var}={_fmt(cand)}: {_fmt(lv)} {'=' if ok else '≠'} {_fmt(rv)}",
                method="exact substitution", confidence=1.0, claim=claim)

        # Path 2: an identity over a variable -> sympy if present, else PIT at random points.
        if var is not None:
            sym = self._sympy_identity(lhs, rhs, var)
            if sym is not None:
                sym.claim = claim
                return sym
            return self._pit_identity(lhs, rhs, var, claim)

        # No variable, no candidate -> it is really arithmetic.
        return self._prove_arithmetic(ProofClaim("arithmetic", stmt))

    def _sympy_identity(self, lhs: str, rhs: str, var: str) -> Optional[ProofResult]:
        try:
            import sympy  # type: ignore
        except Exception:  # noqa: BLE001
            return None
        try:
            x = sympy.symbols(var)
            le = sympy.sympify(_to_sympy(lhs), locals={var: x})
            re_ = sympy.sympify(_to_sympy(rhs), locals={var: x})
            zero = sympy.simplify(le - re_) == 0
            return ProofResult(
                ProofVerdict.PROVEN if zero else ProofVerdict.REFUTED,
                certificate=f"sympy.simplify(({lhs})-({rhs})) {'== 0' if zero else '!= 0'}",
                method="symbolic identity (sympy)", confidence=1.0)
        except Exception:  # noqa: BLE001
            return None

    def _pit_identity(self, lhs: str, rhs: str, var: str, claim: ProofClaim) -> ProofResult:
        tested = 0
        for _ in range(self.identity_points):
            pt = Fraction(self.rng.randint(-50, 50))
            lv = _eval_rational(lhs, {var: pt})
            rv = _eval_rational(rhs, {var: pt})
            if lv is None or rv is None:
                continue
            if lv != rv:
                return ProofResult(
                    ProofVerdict.REFUTED,
                    certificate=f"counter-example {var}={_fmt(pt)}: {_fmt(lv)} ≠ {_fmt(rv)}",
                    method="polynomial identity testing", confidence=1.0, claim=claim)
            tested += 1
        if tested == 0:
            return self._abstain(claim, "could not evaluate the identity at any test point")
        # All points agreed: Schwartz–Zippel — overwhelming, but not a closed symbolic proof.
        return ProofResult(
            ProofVerdict.PROVEN,
            certificate=f"agrees at {tested} random integer points (Schwartz–Zippel)",
            method="polynomial identity testing", confidence=round(1.0 - 0.5 ** tested, 6),
            claim=claim)

    # ------------------------------------------------------------------ #
    # logic — propositional validity by truth-table enumeration (exact, stdlib)
    # ------------------------------------------------------------------ #
    def _prove_logic(self, claim: ProofClaim) -> ProofResult:
        expr = claim.statement
        names = sorted(set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expr))
                       - {"and", "or", "not", "True", "False", "implies", "iff"})
        if len(names) > 16:
            return self._abstain(claim, "too many variables for exhaustive check")
        try:
            node = _parse_bool(expr)
        except Exception as exc:  # noqa: BLE001
            return self._abstain(claim, f"unparseable formula: {exc}")
        falsifier: Optional[Dict[str, bool]] = None
        any_true = False
        for i in range(2 ** len(names)):
            env = {name: bool((i >> b) & 1) for b, name in enumerate(names)}
            val = _eval_bool(node, env)
            any_true = any_true or val
            if not val and falsifier is None:
                falsifier = env
        if falsifier is None:
            return ProofResult(ProofVerdict.PROVEN,
                               certificate=f"valid over all {2 ** len(names)} assignments",
                               method="truth-table enumeration", confidence=1.0, claim=claim)
        if not any_true:
            return ProofResult(ProofVerdict.REFUTED,
                               certificate="contradiction: false under every assignment",
                               method="truth-table enumeration", confidence=1.0, claim=claim)
        return ProofResult(ProofVerdict.REFUTED,
                           certificate=f"not valid: falsified by {falsifier}",
                           method="truth-table enumeration", confidence=1.0, claim=claim)

    # ------------------------------------------------------------------ #
    # inequality — z3 if present, else candidate-substitution
    # ------------------------------------------------------------------ #
    def _prove_inequality(self, claim: ProofClaim) -> ProofResult:
        z3res = self._z3_inequality(claim)
        if z3res is not None:
            return z3res
        m = re.match(r"^(.*?)(<=|>=|<|>)(.*)$", claim.statement)
        if not m:
            return self._abstain(claim, "not a recognisable inequality")
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        env: Dict[str, Fraction] = {}
        if claim.candidate_answer is not None:
            var = _first_var(claim.statement)
            cand = _eval_rational(_rhs_of(claim.candidate_answer))
            if var is not None and cand is not None:
                env[var] = cand
        lv, rv = _eval_rational(lhs, env), _eval_rational(rhs, env)
        if lv is None or rv is None:
            return self._abstain(claim, "could not evaluate inequality (try z3 / give a candidate)")
        cmp = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}[op]
        ok = cmp(lv, rv)
        where = f" at {env}" if env else ""
        return ProofResult(ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                           certificate=f"{_fmt(lv)} {op} {_fmt(rv)}{where}",
                           method="exact evaluation", confidence=1.0, claim=claim)

    def _z3_inequality(self, claim: ProofClaim) -> Optional[ProofResult]:
        if claim.candidate_answer is not None:
            return None   # a concrete candidate is better checked exactly above
        try:
            import z3  # type: ignore
        except Exception:  # noqa: BLE001
            return None
        m = re.match(r"^(.*?)(<=|>=|<|>)(.*)$", claim.statement)
        if not m:
            return None
        try:
            names = sorted(set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", claim.statement)))
            syms = {n: z3.Real(n) for n in names}
            lhs = eval(_to_py(m.group(1)), {"__builtins__": {}}, syms)   # noqa: S307 — names sandboxed
            rhs = eval(_to_py(m.group(3)), {"__builtins__": {}}, syms)   # noqa: S307
            rel = {"<=": lhs <= rhs, ">=": lhs >= rhs, "<": lhs < rhs, ">": lhs > rhs}[m.group(2)]
            s = z3.Solver()
            s.add(z3.Not(rel))
            if s.check() == z3.unsat:
                return ProofResult(ProofVerdict.PROVEN, certificate="z3: negation unsatisfiable",
                                   method="z3 (universal)", confidence=1.0, claim=claim)
            return ProofResult(ProofVerdict.REFUTED,
                               certificate=f"z3 counter-model: {s.model()}",
                               method="z3 (universal)", confidence=1.0, claim=claim)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # number theory — exact stdlib
    # ------------------------------------------------------------------ #
    def _prove_number_theory(self, claim: ProofClaim) -> ProofResult:
        stmt = claim.statement.strip().lower()
        m = re.search(r"(?:is\s+)?(\d+)\s+(?:a\s+)?prime", stmt)
        if m:
            n = int(m.group(1))
            isp = _is_prime(n)
            cert = (f"{n} has no divisor in 2..√{n}" if isp
                    else f"{n} = {_smallest_factor(n)} × {n // _smallest_factor(n)}")
            return ProofResult(ProofVerdict.PROVEN if isp else ProofVerdict.REFUTED,
                               certificate=cert, method="trial division", confidence=1.0, claim=claim)
        m = re.search(r"gcd\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)", stmt)
        if m:
            a, b, g = int(m.group(1)), int(m.group(2)), int(m.group(3))
            real = math.gcd(a, b)
            ok = real == g
            return ProofResult(ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                               certificate=f"gcd({a},{b}) = {real}", method="Euclid",
                               confidence=1.0, claim=claim)
        m = re.search(r"(\d+)\s+divides\s+(\d+)", stmt)
        if m:
            d, n = int(m.group(1)), int(m.group(2))
            ok = d != 0 and n % d == 0
            cert = f"{n} = {d} × {n // d}" if ok else f"{n} mod {d} = {n % d if d else 'NaN'}"
            return ProofResult(ProofVerdict.PROVEN if ok else ProofVerdict.REFUTED,
                               certificate=cert, method="divisibility", confidence=1.0, claim=claim)
        return self._abstain(claim, "no number-theory pattern recognised")

    # ------------------------------------------------------------------ #
    def _abstain(self, claim: ProofClaim, why: str) -> ProofResult:
        return ProofResult(ProofVerdict.UNPROVABLE, certificate=why, method="abstain",
                           confidence=0.0, claim=claim)


# --------------------------------------------------------------------------- #
# Safe exact arithmetic evaluator (ast-based; numbers + operators only)
# --------------------------------------------------------------------------- #
_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_rational(expr: str, env: Optional[Dict[str, Fraction]] = None) -> Optional[Fraction]:
    """Evaluate ``expr`` exactly as a ``Fraction``; ``None`` if it isn't pure arithmetic."""
    env = env or {}
    try:
        node = ast.parse(_to_py(expr).strip(), mode="eval").body
    except Exception:  # noqa: BLE001
        return None
    try:
        return _eval_node(node, env)
    except Exception:  # noqa: BLE001
        return None


def _eval_node(node: ast.AST, env: Dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left, env), _eval_node(node.right, env)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1:
                raise ValueError("non-integer exponent")
            exp = int(right)
            if abs(exp) > 64:
                raise ValueError("exponent too large")
            return left ** exp
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(node.value).limit_denominator(10 ** 9) if isinstance(node.value, float) \
            else Fraction(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"free variable {node.id}")
    raise ValueError(f"disallowed node {type(node).__name__}")


def _to_py(expr: str) -> str:
    """Normalise human maths to Python: ``^`` -> ``**``, implicit ``2x`` -> ``2*x``."""
    s = (expr or "").replace("^", "**")
    s = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", s)       # 2x -> 2*x, 3( -> 3*(
    return s


def _to_sympy(expr: str) -> str:
    return _to_py(expr)


def _rhs_of(answer: str) -> str:
    """From 'x = 2' or '= 2' or '2' return the value side."""
    return answer.split("=", 1)[1] if "=" in answer else answer


def _first_var(expr: str) -> Optional[str]:
    for name in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expr or ""):
        if name not in ("e", "pi"):
            return name
    return None


def _fmt(v: Fraction) -> str:
    return str(int(v)) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# --------------------------------------------------------------------------- #
# Tiny boolean expression engine (truth-table enumeration)
# --------------------------------------------------------------------------- #
def _parse_bool(expr: str) -> ast.AST:
    s = expr
    s = re.sub(r"<->|iff|↔", " __IFF__ ", s)
    s = re.sub(r"->|implies|→", " __IMP__ ", s)
    s = s.replace("&", " and ").replace("|", " or ").replace("!", " not ")
    s = s.replace("¬", " not ").replace("∧", " and ").replace("∨", " or ")
    s = s.replace("__IFF__", "==").replace("__IMP__", "<=")   # A<=B models A->B over booleans
    return ast.parse(s.strip(), mode="eval").body


def _eval_bool(node: ast.AST, env: Dict[str, bool]) -> bool:
    if isinstance(node, ast.BoolOp):
        vals = [_eval_bool(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_bool(node.operand, env)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left = _eval_bool(node.left, env)
        right = _eval_bool(node.comparators[0], env)
        if isinstance(node.ops[0], ast.Eq):      # iff
            return left == right
        if isinstance(node.ops[0], ast.LtE):     # implies (A -> B  ==  A <= B)
            return (not left) or right
        raise ValueError("unsupported comparison")
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        return bool(env.get(node.id, False))
    if isinstance(node, ast.Constant):
        return bool(node.value)
    raise ValueError(f"unsupported boolean node {type(node).__name__}")


# --------------------------------------------------------------------------- #
# number-theory helpers
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _smallest_factor(n: int) -> int:
    if n % 2 == 0:
        return 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    p = Prover()
    print("=" * 70)
    print("NYXARA prover (proof-carrying answers) self-test")
    print("=" * 70)
    for kind, stmt, cand in [
        ("arithmetic", "2 + 2 = 4", None),
        ("arithmetic", "2 + 2 = 5", None),
        ("algebra", "(x+1)^2 = x^2 + 2*x + 1", None),
        ("algebra", "2*x + 3 = 7", "x = 2"),
        ("logic", "A or not A", None),
        ("logic", "A and not A", None),
        ("number_theory", "is 17 prime", None),
        ("number_theory", "gcd(12, 18) = 6", None),
    ]:
        r = p.check_answer(kind, stmt, cand) if cand else p.prove(ProofClaim(kind, stmt))
        print(f"  {stmt:34s} -> {r.summary()}")
