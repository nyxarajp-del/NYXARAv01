"""Tests for the grounded verifier / hallucination trap — growth/verified_distill.py (Part M)."""
from __future__ import annotations

from nyxara.growth.verified_distill import GroundedVerifier


def test_decidable_correct_is_proven_and_durable():
    r = GroundedVerifier().verify("what is 6*7?", "42")
    assert r.verdict == "proven" and r.durable is True
    assert r.confidence == 1.0 and "42" in r.certificate


def test_decidable_wrong_is_refuted_not_durable():
    r = GroundedVerifier().verify("what is 6*7?", "41")
    assert r.verdict == "refuted" and r.durable is False and r.confidence == 0.0
    assert "≠" in r.certificate  # a machine-checkable counter-example


def test_open_domain_high_conf_is_graded_durable():
    r = GroundedVerifier().verify("Who painted the Mona Lisa?", "Leonardo da Vinci",
                                  prior_confidence=0.9)
    assert r.verdict == "graded" and r.durable is True  # >= bake_threshold, but NOT "proven"
    assert not r.proven


def test_hedged_answer_is_downgraded_and_tentative():
    r = GroundedVerifier(bake_threshold=0.75).verify("capital of X?", "maybe it's Y, not sure",
                                                     prior_confidence=0.9)
    assert r.verdict == "graded" and r.durable is False and r.confidence < 0.75


def test_empty_answer_is_zero_confidence():
    r = GroundedVerifier().verify("anything?", "   ")
    assert r.confidence == 0.0 and r.durable is False


def test_self_correct_fixes_a_refuted_answer():
    v = GroundedVerifier()

    def generate(prompt: str) -> str:
        # the counter-example is fed in; the "model" returns the right answer
        assert "wrong" in prompt.lower()
        return "42"

    answer, result = v.self_correct("what is 6*7?", "41", generate=generate)
    assert answer == "42" and result.proven and result.durable


def test_self_correct_noop_without_model():
    v = GroundedVerifier()
    answer, result = v.self_correct("what is 6*7?", "41")  # no generate -> stays refuted
    assert answer == "41" and result.refuted
