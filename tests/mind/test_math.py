"""Tests for nyxara.mind.math."""

from __future__ import annotations

import pytest

from nyxara.mind.math import MathEngine, MathError, Regression, has_sympy, has_z3


@pytest.fixture
def m():
    return MathEngine()


# -------------------- evaluate / arithmetic -------------------- #
def test_evaluate_arithmetic(m):
    assert m.evaluate("2 + 3*4") == 14
    assert m.evaluate("(1+2)*3") == 9


def test_evaluate_exact_rational(m):
    import sympy
    assert m.evaluate("1/3") == sympy.Rational(1, 3)


def test_evaluate_bad_expr(m):
    with pytest.raises(MathError):
        m.evaluate("this is not math @@@")


# -------------------- symbolic (sympy) -------------------- #
@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_solve_quadratic(m):
    assert m.solve("x**2 - 4", "x") == [-2, 2]


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_solve_with_equals(m):
    assert m.solve("2*x = 10", "x") == [5]


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_simplify_expand_factor(m):
    assert m.expand("(x+1)**2") == "x**2 + 2*x + 1"
    assert m.factor("x**2 - 1") == "(x - 1)*(x + 1)"


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_differentiate(m):
    assert m.differentiate("x**3", "x") == "3*x**2"
    assert m.differentiate("x**3", "x", order=2) == "6*x"


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_integrate_definite_and_indefinite(m):
    assert m.integrate("2*x", "x", (0, 3)) == 9
    assert m.integrate("2*x", "x") == "x**2"


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_limit(m):
    assert m.limit("sin(x)/x", "x", 0) == 1
    assert m.limit("1/x", "x", "oo") == 0


# -------------------- z3 proofs -------------------- #
@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_prove_valid(m):
    x = MathEngine.real("x")
    assert m.prove(x + 1 > 1, assumptions=[x > 0])


@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_prove_invalid(m):
    x = MathEngine.real("x")
    # x>0 does NOT imply x>5
    assert not m.prove(x > 5, assumptions=[x > 0])


@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_satisfiable_unsat(m):
    x = MathEngine.real("x")
    sat, model = m.satisfiable([x > 5, x < 3])
    assert sat is False and model is None


@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_satisfiable_sat_with_model(m):
    y = MathEngine.int_("y")
    sat, model = m.satisfiable([y > 2, y < 6, y % 2 == 0])
    assert sat is True
    assert model["y"] == 4


@pytest.mark.skipif(not has_z3(), reason="z3 not installed")
def test_prove_de_morgan(m):
    p, q = MathEngine.bool_("p"), MathEngine.bool_("q")
    import z3
    claim = z3.Not(z3.And(p, q)) == z3.Or(z3.Not(p), z3.Not(q))
    assert m.prove(claim)


# -------------------- statistics -------------------- #
def test_mean_median(m):
    assert m.mean([1, 2, 3, 4]) == 2.5
    assert m.median([3, 1, 2]) == 2
    assert m.median([1, 2, 3, 4]) == 2.5


def test_variance_stdev(m):
    assert m.variance([2, 4, 4, 4, 5, 5, 7, 9], sample=False) == 4.0
    assert m.stdev([2, 4, 4, 4, 5, 5, 7, 9], sample=False) == 2.0


def test_variance_needs_data(m):
    with pytest.raises(MathError):
        m.variance([1], sample=True)


def test_correlation_perfect(m):
    assert abs(m.correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert abs(m.correlation([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-9


def test_linregress(m):
    reg = m.linregress([1, 2, 3, 4, 5], [3, 5, 7, 9, 11])  # y = 2x + 1
    assert abs(reg.slope - 2.0) < 1e-9
    assert abs(reg.intercept - 1.0) < 1e-9
    assert abs(reg.r - 1.0) < 1e-9
    assert isinstance(reg, Regression)
    assert reg.predict(10) == pytest.approx(21.0)


def test_mean_empty(m):
    with pytest.raises(MathError):
        m.mean([])


# -------------------- optimization -------------------- #
def test_minimize_scalar(m):
    x, fx = m.minimize_scalar(lambda t: (t - 3) ** 2 + 1, 0, 10)
    assert abs(x - 3.0) < 1e-3
    assert abs(fx - 1.0) < 1e-3


@pytest.mark.skipif(not has_sympy(), reason="sympy not installed")
def test_critical_points(m):
    cps = m.critical_points("x**3 - 3*x", "x")
    xs = {cp["x"] for cp in cps}
    assert xs == {-1, 1}
    kinds = {cp["x"]: cp["kind"] for cp in cps}
    assert kinds[-1] == "maximum" and kinds[1] == "minimum"


# -------------------- game theory -------------------- #
def test_pure_nash_prisoners_dilemma(m):
    A = [[3, 0], [5, 1]]
    B = [[3, 5], [0, 1]]
    assert m.pure_nash(A, B) == [(1, 1)]


def test_dominant_strategy(m):
    A = [[3, 0], [5, 1]]  # defect (row 1) strictly dominates cooperate
    assert m.dominant_strategy(A, player="row") == 1


def test_dominant_strategy_none(m):
    A = [[1, 0], [0, 1]]  # no strict dominance
    assert m.dominant_strategy(A) is None


def test_zero_sum_matching_pennies(m):
    res = m.zero_sum_value_2x2([[1, -1], [-1, 1]])
    assert res["value"] == 0
    assert res["row_strategy"] == [0.5, 0.5]
    assert res["pure"] is None


def test_zero_sum_with_saddle(m):
    # a game with a pure saddle point
    res = m.zero_sum_value_2x2([[4, 3], [2, 1]])
    assert res["pure"] is not None


def test_pure_nash_coordination(m):
    # coordination game: two pure equilibria
    A = [[2, 0], [0, 1]]
    B = [[2, 0], [0, 1]]
    ne = m.pure_nash(A, B)
    assert (0, 0) in ne and (1, 1) in ne


# -------------------- faculty integration -------------------- #
def test_math_beats_llm_as_verifiable_faculty():
    # the math engine wired as a verifiable faculty should win arithmetic over the LLM
    from nyxara.mind.faculties import (CallableFaculty, FacultyRegistry, FacultySelector,
                                       Task, TaskType)
    eng = MathEngine()
    math_fac = CallableFaculty("math", [TaskType.ARITHMETIC],
                               lambda t: (eng.evaluate(t.payload), 1.0), verifiable=True)
    reg = FacultyRegistry()
    reg.register(math_fac)
    sel = FacultySelector(reg)
    prop = sel.dispatch(Task(TaskType.ARITHMETIC, payload="6*7"))
    assert prop.content == 42
    assert prop.confidence == 1.0
