"""Tests for nyxara.causal.axiom_synth."""

from __future__ import annotations

from nyxara.causal.axiom_synth import AxiomSynthesizer, Expr


def _samples(fn, lo=0, hi=3):
    return [({"a": a, "b": b}, fn(a, b)) for a in range(lo, hi) for b in range(lo, hi)]


def test_discovers_a_rule_that_fits_all_samples():
    syn = AxiomSynthesizer(seed=7)
    samples = _samples(lambda a, b: a + b)
    expr = syn.discover(samples, ["a", "b"])
    assert expr is not None
    assert all(expr.eval(env) == t for env, t in samples)


def test_adopts_only_after_an_exhaustive_proof():
    syn = AxiomSynthesizer(seed=11)
    samples = _samples(lambda a, b: a + b)
    ax = syn.synthesize("addition", samples, ["a", "b"],
                        reference_fn=lambda e: e["a"] + e["b"],
                        domain={"a": range(-3, 4), "b": range(-3, 4)})
    assert ax is not None
    assert ax.proof.proved is True
    assert ax.proof.method == "exhaustive"
    assert ax.adopted is True
    assert ax in syn.adopted()


def test_rejects_a_rule_that_only_interpolates():
    # Fit samples for a+b, but hand a WRONG reference (a*b): the proof must fail and reject it.
    syn = AxiomSynthesizer(seed=3)
    samples = _samples(lambda a, b: a + b)
    ax = syn.synthesize("bogus", samples, ["a", "b"],
                        reference_fn=lambda e: e["a"] * e["b"],
                        domain={"a": range(-2, 3), "b": range(-2, 3)})
    assert ax is not None
    assert ax.proof.proved is False
    assert ax.adopted is False
    assert ax not in syn.adopted()


def test_exhaustive_proof_reports_a_counterexample():
    syn = AxiomSynthesizer()
    wrong = Expr("op", "*", Expr("var", name="a"), Expr("var", name="b"))
    proof = syn.prove_exhaustive(wrong, lambda e: e["a"] + e["b"],
                                 {"a": range(0, 3), "b": range(0, 3)})
    assert proof.proved is False
    assert "counterexample" in proof.detail
