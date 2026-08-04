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

def test_logic_contingent_is_abstained_not_refuted():
    """Satisfiable-but-not-valid is NOT a refutation, and the difference is load-bearing.

    `A and B` is true whenever both hold — calling that REFUTED conflates "I could not prove
    this" with "this is false". godel_loop.ReflectionTower.detect_contradictions treats every
    REFUTED belief as a contradiction in her own logic and retracts it, so the old verdict
    silently deleted every contingent propositional belief she held.
    """
    r = _p().prove(ProofClaim("logic", "A and B"))
    assert r.verdict is ProofVerdict.UNPROVABLE
    assert "contingent" in r.certificate


def test_a_real_contradiction_is_still_refuted():
    """The other half: an unsatisfiable formula must stay REFUTED, so retraction still works."""
    r = _p().prove(ProofClaim("logic", "A and not A"))
    assert r.verdict is ProofVerdict.REFUTED
    assert "contradiction" in r.certificate


def test_a_bare_expression_asserts_nothing_and_proves_nothing():
    """`2+2` makes no claim, so PROVEN was certifying a claim that was never made."""
    r = _p().prove(ProofClaim("arithmetic", "2+2"))
    assert r.verdict is ProofVerdict.UNPROVABLE
    assert _p().prove(ProofClaim("arithmetic", "2+2 = 4")).verdict is ProofVerdict.PROVEN
    assert _p().prove(ProofClaim("arithmetic", "2+2 = 5")).verdict is ProofVerdict.REFUTED


def test_sampled_agreement_is_not_certifiable_even_though_it_is_proven():
    """Schwartz–Zippel is overwhelming evidence, not a decision procedure.

    It keeps the PROVEN verdict (it IS strong), but must not be embeddable in training data as
    a certificate — `exact` is the flag that separates the two.
    """
    from nyxara.growth.prover import Prover

    prover = Prover()
    r = prover._pit_identity("(x+1)*(x+1)", "x*x + 2*x + 1", "x",
                             ProofClaim("algebra", "(x+1)*(x+1) = x*x + 2*x + 1"))
    assert r.verdict is ProofVerdict.PROVEN
    assert r.exact is False
    assert r.certifiable is False


def test_an_exact_symbolic_proof_is_certifiable():
    r = _p().prove(ProofClaim("algebra", "(x+1)^2 = x^2+2*x+1"))
    assert r.verdict is ProofVerdict.PROVEN
    if "sympy" in r.method:                 # the exact path; PIT fallback is checked above
        assert r.exact is True and r.certifiable is True


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


# --------------------------------------------------------------------------- #
# General decision procedure — kind auto-detected from the statement's structure
# --------------------------------------------------------------------------- #
def test_certify_autodetects_arithmetic():
    assert _p().certify("2 + 2 = 4").verdict is ProofVerdict.PROVEN
    assert _p().certify("2 + 2 = 5").verdict is ProofVerdict.REFUTED


def test_certify_autodetects_algebra_identity_and_candidate():
    assert _p().certify("(x+1)^2 = x^2 + 2*x + 1").verdict is ProofVerdict.PROVEN
    assert _p().certify("2*x + 3 = 7", "x = 2").verdict is ProofVerdict.PROVEN
    assert _p().certify("2*x + 3 = 7", "x = 5").verdict is ProofVerdict.REFUTED


def test_certify_autodetects_logic_and_number_theory():
    assert _p().certify("A or not A").verdict is ProofVerdict.PROVEN
    assert _p().certify("A and not A").verdict is ProofVerdict.REFUTED
    assert _p().certify("is 17 prime").verdict is ProofVerdict.PROVEN


def test_unknown_kind_falls_through_to_detection():
    # a caller that does not know the kind still gets a decision, not a blanket abstain
    r = Prover(seed=1).prove(ProofClaim("mystery", "2 + 2 = 4"))
    assert r.verdict is ProofVerdict.PROVEN


def test_certify_abstains_on_undecidable_prose():
    assert _p().certify("the meaning of life is happiness").verdict is ProofVerdict.UNPROVABLE
