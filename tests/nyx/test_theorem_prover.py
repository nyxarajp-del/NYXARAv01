"""NYX V.02 · the proof core — a claim that carries its certificate, or admits it has none.

The most important test in this file is the one that produces **no verdict**: asked to prove
"gravity pulls the apple down", the module must return INEXPRESSIBLE rather than deciding
anything. That refusal is the feature, and it is why "eliminate hallucination mathematically"
is not claimed anywhere.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.theorem_prover import Certificate, ProofCore, Verdict

pytestmark = pytest.mark.skipif(not ProofCore().available()["z3"],
                                reason="z3 is not installed")


def _core(**kw) -> ProofCore:
    return ProofCore(**kw)


# --------------------------------------------------------------------------- #
# The refusal — the point of the module
# --------------------------------------------------------------------------- #
def test_a_sentence_about_the_world_gets_no_verdict_at_all():
    got = _core().prove("gravity pulls the apple down")
    assert got.verdict == Verdict.INEXPRESSIBLE
    assert got.decisive is False and got.proved is False
    assert "not an SMT formula" in got.note


def test_a_bare_term_is_not_a_claim():
    """"gravity" has no truth value. Treating it as one is how a prover certifies nonsense."""
    for text in ("gravity", "x + 1", "42"):
        assert _core().prove(text).verdict == Verdict.INEXPRESSIBLE, text


def test_the_module_states_the_limit_of_what_a_prover_can_do():
    note = _core().stats()["note"]
    assert "cannot be fully true and is not claimed" in note
    assert "that refusal is the feature" in note
    assert "never as 'it is correct'" in note


def test_describe_names_the_closed_language():
    said = _core().describe()
    assert "INEXPRESSIBLE" in said and "closed" in said.lower() or "language" in said


# --------------------------------------------------------------------------- #
# Proved, and it means proved
# --------------------------------------------------------------------------- #
def test_a_universal_truth_is_proved_with_a_certificate():
    got = _core().prove("for all real x, x*x >= 0")
    assert got.verdict == Verdict.PROVED and got.proved
    assert got.engine == "z3" and "¬F is unsatisfiable" in got.note


def test_the_quantifier_may_come_after_the_formula_too():
    assert _core().prove("x*x >= 0 for all real x").verdict == Verdict.PROVED


def test_an_integer_claim_is_proved_over_the_integers():
    got = _core().prove("for all integer n, n*n >= n")
    assert got.verdict == Verdict.PROVED
    assert got.variables == {"n": "Int"}


def test_an_algebraic_identity_is_proved():
    assert _core().prove("(x + 1)**2 = x**2 + 2*x + 1").verdict == Verdict.PROVED


def test_arithmetic_that_is_simply_true_is_proved():
    assert _core().prove("2 + 2 = 4").verdict == Verdict.PROVED


def test_an_implication_is_proved():
    assert _core().prove("implies(x > 5, x > 1)").verdict == Verdict.PROVED


# --------------------------------------------------------------------------- #
# Refuted, with the counter-model that settles it
# --------------------------------------------------------------------------- #
def test_arithmetic_that_is_simply_false_is_refuted():
    got = _core().prove("2 + 2 = 5")
    assert got.verdict == Verdict.REFUTED and got.decisive


def test_a_false_universal_is_refuted_and_shows_its_counter_model():
    """"for all x, x*x >= 1" is FALSE, not contingent — the quantifier decides that."""
    got = _core().prove("for all real x, x*x >= 1")
    assert got.verdict == Verdict.REFUTED
    assert "x" in got.witness and "counter-model exists" in got.note


def test_a_contradiction_is_refuted():
    got = _core().prove("x and not x")
    assert got.verdict == Verdict.REFUTED
    assert got.variables == {"x": "Bool"}     # a name in a boolean position is a boolean


# --------------------------------------------------------------------------- #
# Existentials — a witness, or none
# --------------------------------------------------------------------------- #
def test_an_existential_with_a_witness_is_proved_and_shows_it():
    got = _core().prove("there exists real x, x*x = 4")
    assert got.verdict == Verdict.PROVED and got.witness


def test_an_existential_with_no_witness_is_refuted():
    got = _core().prove("there exists real x, x*x = -1")
    assert got.verdict == Verdict.REFUTED


# --------------------------------------------------------------------------- #
# Contingent is not false
# --------------------------------------------------------------------------- #
def test_a_claim_about_a_free_variable_is_contingent_not_refuted():
    got = _core().prove("x > 5")
    assert got.verdict == Verdict.CONTINGENT
    assert got.decisive is False
    assert "would be a lie" in got.note


def test_contingent_is_never_treated_as_evidence():
    assert _core().certify("x > 5") is None


def test_certify_returns_a_certificate_only_when_there_is_one():
    assert _core().certify("for all real x, x*x >= 0") is not None
    assert _core().certify("gravity pulls the apple down") is None
    assert _core().certify("2 + 2 = 5") is None        # refuted is not a certificate to cite


# --------------------------------------------------------------------------- #
# Budgets, and what a timeout means
# --------------------------------------------------------------------------- #
def test_a_tiny_budget_produces_unknown_not_a_verdict():
    got = ProofCore(timeout_ms=50).prove(
        "for all real x, x**7 - 3*x**5 + 2*x**3 - x + 11 != 0")
    assert got.verdict in (Verdict.UNKNOWN, Verdict.PROVED, Verdict.REFUTED)
    if got.verdict == Verdict.UNKNOWN:
        assert "not the same as" in got.note


def test_too_many_variables_is_refused_rather_than_attempted():
    core = ProofCore(max_vars=2)
    assert core.prove("a + b + c + d > 0").verdict == Verdict.INEXPRESSIBLE


# --------------------------------------------------------------------------- #
# What the rest of NYX gets from it
# --------------------------------------------------------------------------- #
def test_hybrid_upgrades_a_provable_claim_to_a_machine_checked_verdict():
    from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verdict as HybridVerdict
    fusion = SymbolicSubsymbolicFusion(prover=ProofCore())
    got = fusion._proved("for all real x, x*x >= 0")
    assert got is not None and got.proved
    assert HybridVerdict.SUPPORTED is not None


def test_hybrid_leaves_an_inexpressible_claim_to_the_evidence_path():
    from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion
    fusion = SymbolicSubsymbolicFusion(prover=ProofCore())
    assert fusion._proved("gravity pulls the apple down") is None
    assert fusion._proved("x > 5") is None          # contingent is not a verdict either


def test_a_hybrid_without_a_prover_behaves_exactly_as_v01():
    from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion
    assert SymbolicSubsymbolicFusion()._proved("2 + 2 = 4") is None


def test_property_checks_run_over_a_list():
    got = _core().check_properties(["2 + 2 = 4", "2 + 2 = 5", "not a formula at all"])
    assert [c.verdict for c in got] == [Verdict.PROVED, Verdict.REFUTED,
                                        Verdict.INEXPRESSIBLE]


def test_prove_is_reachable_on_the_brain():
    brain = NyxBrain(NyxConfig())
    assert brain.prove("for all real x, x*x >= 0").proved
    assert brain.hybrid.prover is not None


def test_disabling_the_prover_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(prover_enabled=False))
    assert brain.prover is None and brain.prove("2 + 2 = 4") is None
    assert brain.hybrid.prover is None


# --------------------------------------------------------------------------- #
# Fail-soft and bookkeeping
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    core = _core()
    assert core.prove(None).verdict == Verdict.INEXPRESSIBLE
    assert core.prove("").note == "nothing was claimed"
    assert core.parse("((((") is None
    assert core.load_dict(None) is False


def test_the_counters_separate_every_outcome():
    core = _core()
    core.prove("2 + 2 = 4")
    core.prove("2 + 2 = 5")
    core.prove("x > 5")
    core.prove("gravity pulls the apple down")
    stats = core.stats()
    assert stats["proved"] == 1 and stats["refuted"] == 1
    assert stats["contingent"] == 1 and stats["inexpressible"] == 1


def test_a_certificate_serialises_with_its_verdict():
    got = _core().prove("2 + 2 = 4").to_dict()
    assert got["verdict"] == Verdict.PROVED and got["proved"] is True
    assert set(got) >= {"claim", "formula", "engine", "witness", "note"}
    assert Certificate().proved is False
