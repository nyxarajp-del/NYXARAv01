"""Tests for nyxara.growth.prover — proof-carrying answers (Pillar F · Edge 2)."""

from __future__ import annotations

from nyxara.growth.prover import ProofClaim, ProofVerdict, Prover


def _p() -> Prover:
    return Prover(seed=1)


# --------------------------------------------------------------------------- #
# arithmetic — exact over the rationals
# --------------------------------------------------------------------------- #
def test_arithmetic_true_equation_is_proven():
    r = _p().prove(ProofClaim("arithmetic", "2 + 2 = 4"))
    assert r.verdict is ProofVerdict.PROVEN and r.proven

def test_arithmetic_false_equation_is_refuted():
    r = _p().prove(ProofClaim("arithmetic", "2 + 2 = 5"))
    assert r.verdict is ProofVerdict.REFUTED

def test_arithmetic_is_exact_not_floating():
    # 1/3 + 1/3 + 1/3 == 1 exactly under rationals (float would drift)
    r = _p().prove(ProofClaim("arithmetic", "1/3 + 1/3 + 1/3 = 1"))
    assert r.verdict is ProofVerdict.PROVEN

def test_arithmetic_candidate_answer_checked():
    r = _p().check_answer("arithmetic", "6 * 7", "42")
    assert r.verdict is ProofVerdict.PROVEN
    bad = _p().check_answer("arithmetic", "6 * 7", "41")
    assert bad.verdict is ProofVerdict.REFUTED


# --------------------------------------------------------------------------- #
# algebra — identity (sympy / PIT) + candidate substitution
# --------------------------------------------------------------------------- #
def test_algebra_identity_is_proven():
    r = _p().prove(ProofClaim("algebra", "(x+1)^2 = x^2 + 2*x + 1"))
    assert r.verdict is ProofVerdict.PROVEN

def test_algebra_false_identity_is_refuted():
    r = _p().prove(ProofClaim("algebra", "(x+1)^2 = x^2 + 1"))
    assert r.verdict is ProofVerdict.REFUTED

def test_algebra_identity_via_pit_fallback(monkeypatch):
    # Force the stdlib polynomial-identity-testing path (no sympy).
    p = _p()
    monkeypatch.setattr(p, "_sympy_identity", lambda *a, **k: None)
    r = p.prove(ProofClaim("algebra", "(x+1)^2 = x^2 + 2*x + 1"))
    assert r.verdict is ProofVerdict.PROVEN
    assert r.method == "polynomial identity testing"
    bad = p.prove(ProofClaim("algebra", "(x+1)^2 = x^2 + 1"))
    assert bad.verdict is ProofVerdict.REFUTED

def test_algebra_candidate_solution_verified():
    r = _p().check_answer("algebra", "2*x + 3 = 7", "x = 2")
    assert r.verdict is ProofVerdict.PROVEN
    wrong = _p().check_answer("algebra", "2*x + 3 = 7", "x = 3")
    assert wrong.verdict is ProofVerdict.REFUTED


# --------------------------------------------------------------------------- #
# logic — truth-table enumeration
# --------------------------------------------------------------------------- #
def test_logic_tautology_is_proven():
    r = _p().prove(ProofClaim("logic", "A or not A"))
    assert r.verdict is ProofVerdict.PROVEN

def test_logic_contradiction_is_refuted():
    r = _p().prove(ProofClaim("logic", "A and not A"))
    assert r.verdict is ProofVerdict.REFUTED

def test_logic_implication_tautology():
    # modus-ponens shape: ((A -> B) and A) -> B  is valid
    r = _p().prove(ProofClaim("logic", "((A -> B) and A) -> B"))
    assert r.verdict is ProofVerdict.PROVEN

def test_logic_contingent_is_not_valid():
    r = _p().prove(ProofClaim("logic", "A and B"))
    assert r.verdict is ProofVerdict.REFUTED        # not valid; a falsifier exists
    assert "falsified" in r.certificate


# --------------------------------------------------------------------------- #
# inequality
# --------------------------------------------------------------------------- #
def test_inequality_with_candidate():
    r = _p().check_answer("inequality", "x < 10", "x = 3")
    assert r.verdict is ProofVerdict.PROVEN
    bad = _p().check_answer("inequality", "x < 10", "x = 20")
    assert bad.verdict is ProofVerdict.REFUTED


# --------------------------------------------------------------------------- #
# number theory — exact stdlib
# --------------------------------------------------------------------------- #
def test_primality():
    assert _p().prove(ProofClaim("number_theory", "is 17 prime")).verdict is ProofVerdict.PROVEN
    assert _p().prove(ProofClaim("number_theory", "is 21 prime")).verdict is ProofVerdict.REFUTED

def test_gcd_and_divides():
    assert _p().prove(ProofClaim("number_theory", "gcd(12, 18) = 6")).proven
    assert not _p().prove(ProofClaim("number_theory", "gcd(12, 18) = 4")).proven
    assert _p().prove(ProofClaim("number_theory", "3 divides 12")).proven
    assert not _p().prove(ProofClaim("number_theory", "5 divides 12")).proven


# --------------------------------------------------------------------------- #
# honesty — abstain rather than bluff
# --------------------------------------------------------------------------- #
def test_unknown_kind_abstains():
    r = _p().prove(ProofClaim("astrology", "the stars say yes"))
    assert r.verdict is ProofVerdict.UNPROVABLE and r.confidence == 0.0

def test_non_arithmetic_statement_abstains():
    r = _p().prove(ProofClaim("arithmetic", "the meaning of life = happiness"))
    assert r.verdict is ProofVerdict.UNPROVABLE

def test_evaluator_rejects_dangerous_input():
    # a function call is not pure arithmetic -> abstain, never execute
    r = _p().prove(ProofClaim("arithmetic", "__import__('os').system('x') = 0"))
    assert r.verdict is ProofVerdict.UNPROVABLE

def test_checker_never_raises():
    for kind in ("arithmetic", "algebra", "logic", "inequality", "number_theory"):
        r = _p().prove(ProofClaim(kind, ""))         # empty / garbage input
        assert r.verdict in (ProofVerdict.UNPROVABLE, ProofVerdict.REFUTED, ProofVerdict.PROVEN)
