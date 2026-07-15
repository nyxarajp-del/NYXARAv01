"""Tests for nyxara.mind.compute_reasoner — the general reduce→run→verify reasoner.

The point under test is the honest contract: NYXARA answers a problem by *computing and verifying*
it herself (symbolic CAS or a program she runs), the truth-bearing step is never the proposer's
word, and an answer that cannot be certified is *refused* (None), never bluffed.
"""

from __future__ import annotations

import pytest

from nyxara.mind.compute_reasoner import ComputeReasoner, ComputeResult, solve_by_compute

sympy = pytest.importorskip("sympy")


# --------------------------------------------------------------------------- #
# Symbolic reducer — general, self-verifying, needs no model
# --------------------------------------------------------------------------- #
def test_solves_linear_system_beyond_old_faculties():
    """A two-unknown system — a class the old single-variable AlgebraFaculty could not do."""
    r = ComputeReasoner()
    res = r.solve("solve 3*x + 2*y = 12 and x - y = 1 for x, y")
    assert res is not None and res.verified
    assert res.detail["unknowns"] == ["x", "y"]
    # x = 14/5, y = 9/5 — check exactly against sympy's own solution
    assert "14/5" in res.answer and "9/5" in res.answer


def test_solves_cubic_and_verifies_by_substitution():
    r = ComputeReasoner()
    res = r.solve("solve x**3 - 6*x**2 + 11*x - 6 = 0")
    assert res is not None and res.verified
    for root in ("1", "2", "3"):
        assert root in res.answer


def test_proves_true_identity_and_refutes_false_one():
    r = ComputeReasoner()
    ok = r.solve("prove (a+b)**2 = a**2 + 2*a*b + b**2")
    assert ok is not None and ok.verified and ok.answer.startswith("PROVEN")
    bad = r.solve("prove (a+b)**2 = a**2 + b**2")
    assert bad is not None and bad.answer.startswith("REFUTED")


def test_calculus_derivative_and_integral():
    r = ComputeReasoner()
    d = r.solve("differentiate x**3 + sin(x)")
    assert d is not None and d.verified and "3*x**2" in d.answer and "cos(x)" in d.answer
    i = r.solve("integrate 2*x")
    assert i is not None and i.verified and i.answer.startswith("x**2")


def test_number_theory_factor_primality_gcd():
    r = ComputeReasoner()
    assert r.solve("is 561 prime").answer == "composite"
    assert r.solve("is 17 prime").answer == "prime"
    f = r.solve("factor 5040")
    assert f is not None and f.verified          # product of factors re-checked == 5040
    assert r.solve("gcd of 48 and 36").answer == "12"


def test_evaluates_closed_expression_but_not_prose():
    r = ComputeReasoner()
    assert r.solve("what is 2**10 + 24").answer == "1048"
    # prose with no computable expression must abstain, not hallucinate
    assert r.solve("what is the meaning of the story") is None


# --------------------------------------------------------------------------- #
# Program-of-thought — the model proposes a PROGRAM, NYXARA runs & verifies it
# --------------------------------------------------------------------------- #
def _proposer(programs):
    """Build a proposer that always returns the given list of candidate programs."""
    return lambda problem, n: list(programs)


def test_program_is_executed_and_agreement_certifies():
    """Two independently-written programs that agree → self-consistency certificate."""
    # a problem the symbolic reducer does NOT handle: trailing zeros of 100!
    # two genuinely-computing, independently-written programs (Legendre's formula & direct count)
    prog1 = ("z=0\nn=100\np=5\nwhile p<=n:\n    z+=n//p\n    p*=5\nresult=z\n")
    prog2 = ("f=1\nfor k in range(1,101):\n    f*=k\ns=str(f)\nc=0\n"
             "for ch in reversed(s):\n    if ch=='0':\n        c+=1\n    else:\n        break\n"
             "result=c\n")
    r = ComputeReasoner(proposer=_proposer([prog1, prog2]))
    res = r.solve("how many trailing zeros are in 100 factorial")
    assert res is not None
    assert res.answer == "24"
    assert res.method in ("self-consistency", "cross-verified")
    assert res.detail.get("samples") == 2


def test_wrong_proposer_code_is_refused_not_emitted():
    """The verifier gates the model: two DIFFERENT wrong outputs never reach agreement → abstain."""
    prog1 = "result = 41\n"
    prog2 = "result = 43\n"
    r = ComputeReasoner(proposer=_proposer([prog1, prog2]))
    # nothing the symbolic reducer can decide, and the programs disagree -> honest None
    assert r.solve("give me the answer to the ultimate question") is None


def test_lone_uncrosscheckable_program_abstains():
    """A single program output that nothing corroborates is refused (never bluff)."""
    r = ComputeReasoner(proposer=_proposer(["result = 12345\n"]))
    assert r.solve("some opaque question with no checkable structure") is None


def test_program_crosschecks_symbolic_and_boosts_confidence():
    """When a program reproduces the symbolic answer, confidence is upgraded to cross-verified."""
    prog = "result = 12\n"
    r = ComputeReasoner(proposer=_proposer([prog]))
    res = r.solve("gcd of 48 and 36")
    assert res is not None and res.answer == "12"
    assert res.method == "cross-verified"
    assert res.confidence >= 0.98


def test_proposer_that_only_asserts_answer_without_result_is_ignored():
    """A snippet that doesn't set `result` (the sandbox contract) contributes nothing."""
    r = ComputeReasoner(proposer=_proposer(["print('the answer is 42')\n"]))
    # no `result` var -> no program answer -> abstain on an otherwise-undecidable prompt
    assert r.solve("what is your favourite colour and why") is None


# --------------------------------------------------------------------------- #
# Contract / API
# --------------------------------------------------------------------------- #
def test_answer_tuple_shape_matches_faculty_slot():
    got = solve_by_compute("gcd of 100 and 60")
    assert isinstance(got, tuple) and got[0] == "20" and 0.0 < got[1] <= 1.0


def test_empty_and_garbage_input_abstains():
    r = ComputeReasoner()
    assert r.solve("") is None
    assert r.solve("   ") is None
    assert isinstance(r.solve("solve x**2 = 4"), ComputeResult)
