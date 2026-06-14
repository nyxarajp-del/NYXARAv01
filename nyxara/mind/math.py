"""NYXARA · mind/math.py — the exact, verifiable math faculty.

When the answer can be *computed* or *proven*, NYXARA does not guess. This faculty is
the symbolic/exact engine the kernel prefers over the probabilistic LLM whenever the
problem admits certainty (control law: verifiable > probabilistic).

Capabilities:

* **Symbolic algebra & calculus** (SymPy) — evaluate, simplify, expand, factor, solve
  equations, differentiate, integrate, limits.
* **Constraint solving & theorem proving** (Z3) — satisfiability with models, and
  proving a claim valid (by showing its negation is unsatisfiable).
* **Statistics** — mean/median/variance/stdev, correlation, linear regression.
* **Optimization** — golden-section scalar minimisation and symbolic critical points.
* **Game theory** — pure Nash equilibria, dominance, 2×2 zero-sum value (mixed).

SymPy/Z3 are imported lazily; arithmetic still works via a safe evaluator if SymPy is
absent. Results are exact (rationals where possible) and carry full confidence.

Optional acceleration; depends on :mod:`mind.faculties`/:mod:`mind.proposal` only for
the faculty adapter.
"""

from __future__ import annotations

import ast
import math
import operator as _op
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MathError",
    "MathEngine",
    "has_sympy",
    "has_z3",
]

try:
    import sympy as _sp  # type: ignore
    _HAS_SYMPY = True
except Exception:  # pragma: no cover
    _sp = None  # type: ignore
    _HAS_SYMPY = False

try:
    import z3 as _z3  # type: ignore
    _HAS_Z3 = True
except Exception:  # pragma: no cover
    _z3 = None  # type: ignore
    _HAS_Z3 = False


def has_sympy() -> bool:
    return _HAS_SYMPY


def has_z3() -> bool:
    return _HAS_Z3


class MathError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Safe arithmetic fallback (no sympy)
# --------------------------------------------------------------------------- #
_AST_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
            ast.Div: _op.truediv, ast.Pow: _op.pow, ast.Mod: _op.mod,
            ast.FloorDiv: _op.floordiv, ast.USub: _op.neg, ast.UAdd: _op.pos}


def _safe_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _AST_OPS:
        return _AST_OPS[type(node.op)](_safe_arith(node.left), _safe_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _AST_OPS:
        return _AST_OPS[type(node.op)](_safe_arith(node.operand))
    raise MathError("unsupported arithmetic expression")


def _pyify(value: Any) -> Any:
    """Convert a SymPy result to a plain python number when exact, else str."""
    if not _HAS_SYMPY:
        return value
    try:
        v = _sp.nsimplify(value) if getattr(value, "is_number", False) else value
    except Exception:
        v = value
    if getattr(v, "is_number", False):
        if v.is_integer:
            return int(v)
        if v.is_rational:
            return _sp.Rational(v)
        if v.is_real:
            return float(v)
    return v


# --------------------------------------------------------------------------- #
# Statistics result
# --------------------------------------------------------------------------- #
@dataclass
class Regression:
    slope: float
    intercept: float
    r: float

    def predict(self, x: float) -> float:
        return self.slope * x + self.intercept


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class MathEngine:
    """Exact symbolic / logical / statistical computation."""

    # ---------------- symbolic (SymPy) ---------------- #
    def _sym(self, name: str):
        return _sp.Symbol(name)

    def evaluate(self, expr: str) -> Any:
        """Evaluate an arithmetic/symbolic expression to an exact value."""
        if _HAS_SYMPY:
            try:
                return _pyify(_sp.sympify(expr))
            except Exception as e:
                raise MathError(f"cannot evaluate {expr!r}: {e}")
        try:
            tree = ast.parse(expr, mode="eval")
        except (SyntaxError, ValueError) as e:
            # keyless / no-sympy fallback must fail as data, not crash the loop
            raise MathError(f"cannot evaluate {expr!r}: {e}")
        return _safe_arith(tree)

    def simplify(self, expr: str) -> str:
        self._require_sympy()
        return str(_sp.simplify(_sp.sympify(expr)))

    def expand(self, expr: str) -> str:
        self._require_sympy()
        return str(_sp.expand(_sp.sympify(expr)))

    def factor(self, expr: str) -> str:
        self._require_sympy()
        return str(_sp.factor(_sp.sympify(expr)))

    def solve(self, equation: str, var: str = "x") -> List[Any]:
        """Solve ``equation`` (an expression assumed == 0, or 'lhs = rhs') for ``var``."""
        self._require_sympy()
        x = self._sym(var)
        if "=" in equation:
            lhs, rhs = equation.split("=", 1)
            eq = _sp.Eq(_sp.sympify(lhs), _sp.sympify(rhs))
        else:
            eq = _sp.sympify(equation)
        sols = _sp.solve(eq, x)
        return sorted((_pyify(s) for s in sols),
                      key=lambda v: (isinstance(v, str), _sortkey(v)))

    def differentiate(self, expr: str, var: str = "x", order: int = 1) -> str:
        self._require_sympy()
        x = self._sym(var)
        return str(_sp.diff(_sp.sympify(expr), x, order))

    def integrate(self, expr: str, var: str = "x",
                  bounds: Optional[Tuple[float, float]] = None) -> Any:
        self._require_sympy()
        x = self._sym(var)
        e = _sp.sympify(expr)
        if bounds is not None:
            return _pyify(_sp.integrate(e, (x, bounds[0], bounds[1])))
        return str(_sp.integrate(e, x))

    def limit(self, expr: str, var: str, point: Any, direction: str = "+") -> Any:
        self._require_sympy()
        x = self._sym(var)
        pt = _sp.oo if point in ("oo", "inf", float("inf")) else _sp.sympify(point)
        return _pyify(_sp.limit(_sp.sympify(expr), x, pt, dir=direction))

    # ---------------- constraints & proofs (Z3) ---------------- #
    @staticmethod
    def real(name: str):
        MathEngine._require_z3_static()
        return _z3.Real(name)

    @staticmethod
    def int_(name: str):
        MathEngine._require_z3_static()
        return _z3.Int(name)

    @staticmethod
    def bool_(name: str):
        MathEngine._require_z3_static()
        return _z3.Bool(name)

    def satisfiable(self, constraints: Sequence[Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check satisfiability of Z3 constraints; return (sat, model-as-dict)."""
        self._require_z3()
        s = _z3.Solver()
        for c in constraints:
            s.add(c)
        if s.check() == _z3.sat:
            m = s.model()
            model = {str(d.name()): _z3_value(m[d]) for d in m.decls()}
            return True, model
        return False, None

    def prove(self, claim: Any, assumptions: Sequence[Any] = ()) -> bool:
        """Prove ``claim`` valid under ``assumptions`` (unsat of assumptions ∧ ¬claim)."""
        self._require_z3()
        s = _z3.Solver()
        for a in assumptions:
            s.add(a)
        s.add(_z3.Not(claim))
        return s.check() == _z3.unsat

    # ---------------- statistics ---------------- #
    @staticmethod
    def mean(data: Sequence[float]) -> float:
        if not data:
            raise MathError("mean of empty data")
        return sum(data) / len(data)

    @staticmethod
    def median(data: Sequence[float]) -> float:
        if not data:
            raise MathError("median of empty data")
        s = sorted(data)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

    @classmethod
    def variance(cls, data: Sequence[float], *, sample: bool = True) -> float:
        n = len(data)
        if n < (2 if sample else 1):
            raise MathError("not enough data for variance")
        m = cls.mean(data)
        ss = sum((x - m) ** 2 for x in data)
        return ss / (n - 1 if sample else n)

    @classmethod
    def stdev(cls, data: Sequence[float], *, sample: bool = True) -> float:
        return math.sqrt(cls.variance(data, sample=sample))

    @classmethod
    def correlation(cls, xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) != len(ys) or len(xs) < 2:
            raise MathError("need equal-length series of length >= 2")
        mx, my = cls.mean(xs), cls.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if sx == 0 or sy == 0:
            return 0.0
        return cov / (sx * sy)

    @classmethod
    def linregress(cls, xs: Sequence[float], ys: Sequence[float]) -> Regression:
        if len(xs) != len(ys) or len(xs) < 2:
            raise MathError("need equal-length series of length >= 2")
        mx, my = cls.mean(xs), cls.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx else 0.0
        intercept = my - slope * mx
        return Regression(slope=slope, intercept=intercept, r=cls.correlation(xs, ys))

    # ---------------- optimization ---------------- #
    @staticmethod
    def minimize_scalar(f: Callable[[float], float], a: float, b: float,
                        *, tol: float = 1e-7, max_iter: int = 200) -> Tuple[float, float]:
        """Golden-section minimisation of a unimodal ``f`` on [a, b]."""
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
            if f(c) < f(d):
                b, d = d, c
                c = b - gr * (b - a)
            else:
                a, c = c, d
                d = a + gr * (b - a)
        x = (a + b) / 2
        return x, f(x)

    def critical_points(self, expr: str, var: str = "x") -> List[Dict[str, Any]]:
        """Symbolic critical points with min/max/saddle classification."""
        self._require_sympy()
        x = self._sym(var)
        e = _sp.sympify(expr)
        d1, d2 = _sp.diff(e, x), _sp.diff(e, x, 2)
        out = []
        for s in _sp.solve(d1, x):
            if not getattr(s, "is_real", False):
                continue
            second = d2.subs(x, s)
            kind = ("minimum" if second > 0 else "maximum" if second < 0 else "saddle")
            out.append({"x": _pyify(s), "value": _pyify(e.subs(x, s)), "kind": kind})
        return sorted(out, key=lambda r: _sortkey(r["x"]))

    # ---------------- game theory ---------------- #
    @staticmethod
    def pure_nash(payoff_row: Sequence[Sequence[float]],
                  payoff_col: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
        """Pure-strategy Nash equilibria of a 2-player normal-form game."""
        n, m = len(payoff_row), len(payoff_row[0])
        eqs = []
        for i in range(n):
            for j in range(m):
                row_best = payoff_row[i][j] == max(payoff_row[k][j] for k in range(n))
                col_best = payoff_col[i][j] == max(payoff_col[i][k] for k in range(m))
                if row_best and col_best:
                    eqs.append((i, j))
        return eqs

    @staticmethod
    def dominant_strategy(payoff: Sequence[Sequence[float]], *, player: str = "row"
                          ) -> Optional[int]:
        """Return a strictly dominant strategy index for ``player`` if one exists."""
        if player == "col":
            payoff = [list(col) for col in zip(*payoff)]
        n = len(payoff)
        for i in range(n):
            if all(i == k or all(payoff[i][j] > payoff[k][j] for j in range(len(payoff[i])))
                   for k in range(n)):
                return i
        return None

    @staticmethod
    def zero_sum_value_2x2(matrix: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Value & optimal mixed strategy of a 2×2 zero-sum game (row maximiser)."""
        (a, b), (c, d) = matrix[0], matrix[1]
        # check for a pure saddle point first
        for i in range(2):
            for j in range(2):
                if (matrix[i][j] == min(matrix[i]) and
                        matrix[i][j] == max(matrix[k][j] for k in range(2))):
                    return {"value": matrix[i][j], "pure": (i, j),
                            "row_strategy": [1.0 if k == i else 0.0 for k in range(2)]}
        denom = (a + d - b - c)
        if denom == 0:
            return {"value": (a + d) / 2, "pure": None, "row_strategy": [0.5, 0.5]}
        p = (d - c) / denom        # P(row plays strategy 0)
        value = (a * d - b * c) / denom
        return {"value": value, "pure": None, "row_strategy": [p, 1 - p]}

    # ---------------- guards ---------------- #
    @staticmethod
    def _require_sympy() -> None:
        if not _HAS_SYMPY:
            raise MathError("SymPy is required for symbolic operations")

    def _require_z3(self) -> None:
        self._require_z3_static()

    @staticmethod
    def _require_z3_static() -> None:
        if not _HAS_Z3:
            raise MathError("Z3 is required for constraint solving / proofs")


def _sortkey(v: Any):
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def _z3_value(v: Any) -> Any:
    try:
        if v.is_int():
            return v.as_long()
        if v.is_real():
            return float(v.as_fraction())
    except Exception:
        pass
    return str(v)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA math-engine self-test")
    print("=" * 70)
    print(f"sympy={has_sympy()}  z3={has_z3()}")

    m = MathEngine()

    # symbolic
    assert m.evaluate("2 + 3*4") == 14
    print(f"\nevaluate 2+3*4      : {m.evaluate('2 + 3*4')}")
    print(f"evaluate 1/3        : {m.evaluate('1/3')} (exact)")
    assert m.solve("x**2 - 4", "x") == [-2, 2]
    print(f"solve x^2-4=0       : {m.solve('x**2 - 4', 'x')}")
    print(f"d/dx x^3            : {m.differentiate('x**3', 'x')}")
    assert m.integrate("2*x", "x", (0, 3)) == 9
    print(f"∫₀³ 2x dx           : {m.integrate('2*x', 'x', (0, 3))}")
    print(f"lim sin(x)/x x->0   : {m.limit('sin(x)/x', 'x', 0)}")
    assert m.limit("sin(x)/x", "x", 0) == 1

    # proofs (z3)
    x = MathEngine.real("x")
    assert m.prove(x + 1 > 1, assumptions=[x > 0])
    print("\nprove x>0 ⊢ x+1>1   : True")
    sat, model = m.satisfiable([x > 5, x < 3])
    assert not sat
    print(f"sat(x>5 ∧ x<3)      : {sat}")
    y = MathEngine.int_("y")
    sat2, model2 = m.satisfiable([y > 2, y < 6, y % 2 == 0])
    print(f"sat(2<y<6, even)    : {sat2} model={model2}")
    assert sat2

    # statistics
    xs, ys = [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]
    reg = m.linregress(xs, ys)
    print(f"\nlinregress y=2x     : slope={reg.slope:.2f} r={reg.r:.2f}")
    assert abs(reg.slope - 2.0) < 1e-9 and abs(reg.r - 1.0) < 1e-9
    print(f"mean/stdev          : {m.mean(ys):.1f} / {m.stdev(ys):.2f}")

    # optimization
    xmin, fmin = m.minimize_scalar(lambda t: (t - 2) ** 2, 0, 5)
    print(f"\nmin (x-2)^2 on[0,5] : x={xmin:.4f}")
    assert abs(xmin - 2.0) < 1e-3
    cps = m.critical_points("x**3 - 3*x", "x")
    print(f"crit pts x^3-3x     : {cps}")
    assert {cp["x"] for cp in cps} == {-1, 1}

    # game theory — prisoner's dilemma (payoffs; higher better)
    #   rows/cols: 0=cooperate, 1=defect
    A = [[3, 0], [5, 1]]   # row payoffs
    B = [[3, 5], [0, 1]]   # col payoffs
    ne = m.pure_nash(A, B)
    print(f"\nprisoner's dilemma  : pure NE = {ne}")
    assert ne == [(1, 1)]                       # mutual defection
    assert m.dominant_strategy(A, player="row") == 1

    # matching pennies — zero-sum, no pure equilibrium, value 0
    mp = m.zero_sum_value_2x2([[1, -1], [-1, 1]])
    print(f"matching pennies    : value={mp['value']} strategy={mp['row_strategy']}")
    assert mp["value"] == 0 and mp["row_strategy"] == [0.5, 0.5]

    print("\nALL SELF-TESTS PASSED ✓")
