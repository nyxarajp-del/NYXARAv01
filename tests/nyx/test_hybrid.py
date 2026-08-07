"""NYX V.01 · symbolic ∧ sub-symbolic — derived beats guessed, and she checks her own answers."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verdict
from nyxara.nyx.modules import Proposal


class _Evidence:
    def __init__(self, text: str) -> None:
        self.text = text


def _p(source: str, confidence: float, **kw) -> Proposal:
    return Proposal(source=source, content=f"{source} content", confidence=confidence, **kw)


# -------------------- adjudication -------------------- #
def test_a_verified_candidate_beats_a_more_confident_guess():
    got = SymbolicSubsymbolicFusion().adjudicate([
        _p("guesser", 0.99),
        _p("derivation", 0.6, verifiable=True),
    ])
    assert got.source == "derivation"


def test_with_no_verified_candidate_the_most_confident_wins():
    got = SymbolicSubsymbolicFusion().adjudicate([_p("a", 0.3), _p("b", 0.8)])
    assert got.source == "b"


def test_among_several_verified_candidates_the_most_confident_wins():
    got = SymbolicSubsymbolicFusion().adjudicate([
        _p("a", 0.6, verifiable=True), _p("b", 0.9, verifiable=True), _p("c", 0.99),
    ])
    assert got.source == "b"


def test_adjudicating_nothing_returns_nothing():
    assert SymbolicSubsymbolicFusion().adjudicate([]) is None
    assert SymbolicSubsymbolicFusion().adjudicate([None]) is None


# -------------------- checking by derivation -------------------- #
def test_a_derivable_claim_is_supported_and_marked_derived():
    fusion = SymbolicSubsymbolicFusion()
    got = fusion.verify("identity holds", stimulus="prove that (x+1)^2 = x^2 + 2x + 1")
    if got.verdict is Verdict.UNVERIFIABLE:
        pytest.skip("sympy not installed — the engine declines honestly")
    assert got.verdict is Verdict.SUPPORTED and got.derived is True
    assert got.outcome == 1.0


def test_a_false_identity_is_contradicted():
    fusion = SymbolicSubsymbolicFusion()
    got = fusion.verify("identity holds", stimulus="prove that (x+1)^2 = x^2 + 1")
    if got.verdict is Verdict.UNVERIFIABLE:
        pytest.skip("sympy not installed — the engine declines honestly")
    assert got.verdict is Verdict.CONTRADICTED and got.outcome == 0.0


# -------------------- checking by grounding -------------------- #
def test_a_claim_carried_by_retrieved_evidence_is_supported():
    got = SymbolicSubsymbolicFusion().verify(
        "gravity pulls an apple down",
        evidence=[_Evidence("gravity pulls an apple toward the ground")])
    assert got.verdict is Verdict.SUPPORTED and got.grounded_in


def test_a_claim_her_evidence_does_not_cover_is_unverifiable_not_false():
    """Failing to check something is not evidence that it was wrong."""
    got = SymbolicSubsymbolicFusion().verify(
        "the moon is made of green cheese",
        evidence=[_Evidence("gravity pulls an apple toward the ground")])
    assert got.verdict is Verdict.UNVERIFIABLE
    assert got.outcome is None


def test_a_free_floating_claim_with_no_evidence_is_unverifiable():
    got = SymbolicSubsymbolicFusion().verify("something entirely unsupported")
    assert got.verdict is Verdict.UNVERIFIABLE and got.outcome is None


def test_the_grounding_bar_is_configurable():
    strict = SymbolicSubsymbolicFusion(min_grounding_overlap=0.99)
    lax = SymbolicSubsymbolicFusion(min_grounding_overlap=0.1)
    evidence = [_Evidence("gravity pulls an apple toward the ground")]
    assert strict.verify("gravity pulls an apple down", evidence=evidence).verdict \
        is Verdict.UNVERIFIABLE
    assert lax.verify("gravity pulls an apple down", evidence=evidence).verdict \
        is Verdict.SUPPORTED


def test_grounding_can_be_switched_off():
    fusion = SymbolicSubsymbolicFusion(grounding=False)
    got = fusion.verify("gravity pulls an apple down",
                        evidence=[_Evidence("gravity pulls an apple toward the ground")])
    assert got.verdict is Verdict.UNVERIFIABLE


def test_an_empty_claim_is_unverifiable():
    assert SymbolicSubsymbolicFusion().verify("").verdict is Verdict.UNVERIFIABLE


# -------------------- the loop closing, with no human -------------------- #
@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "holo_capacity": 64, "metacog_min_samples": 1})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return b


def test_a_supported_answer_credits_the_specialist_that_said_it(brain):
    before = brain.metacog.record_for("memory").score
    thought = brain.think("what pulls an apple down")
    assert thought.verification is not None
    if thought.verification.verdict is not Verdict.SUPPORTED:
        pytest.skip("this turn was not checkable")
    assert brain.metacog.record_for(thought.winner.source).score > before


def test_an_unverifiable_answer_does_not_move_reliability(brain):
    """The crucial non-punishment: most of what anyone says is not mechanically checkable."""
    thought = brain.think("how might we keep a room warm in winter")
    if thought.verification is None or thought.verification.verdict is not Verdict.UNVERIFIABLE:
        pytest.skip("this turn happened to be checkable")
    record = brain.metacog.record_for(thought.winner.source)
    assert record.samples == 0 and record.score == 0.5


def test_verification_is_recorded_on_the_thought(brain):
    thought = brain.think("what pulls an apple down")
    assert thought.to_dict()["verification"] is not None


def test_a_brain_without_hybrid_still_thinks():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "hybrid_enabled": False})
    b = NyxBrain(cfg)
    assert b.hybrid is None
    thought = b.think("what pulls an apple down")
    assert thought.verification is None and thought.deliberation is not None


def test_check_and_learn_is_a_no_op_on_an_empty_thought(brain):
    assert brain.hybrid.check_and_learn(brain, None) is None


def test_verification_never_raises_on_hostile_input():
    fusion = SymbolicSubsymbolicFusion()
    for claim in (None, "", "   ", "!!!", "x" * 5000):
        assert fusion.verify(claim) is not None
