"""Tests for digital-genomic recombination — growth/genomic_recombination.py (Part D)."""
from __future__ import annotations

from nyxara.growth.genomic_recombination import GenomicRecombiner, evaluate

VAR = ("var",)


def test_crossover_breeds_verified_child():
    # parent A = x*x ; parent B = x + 3 ; target child = x*x + 3 (a subtree splice neither parent is)
    a = ("mul", VAR, VAR)
    b = ("add", VAR, ("const", 3))
    spec = [(x, x * x + 3) for x in range(6)]

    res = GenomicRecombiner(seed=1).breed(a, b, spec)
    assert res.verified and res.child is not None
    assert all(evaluate(res.child, x) == x * x + 3 for x in range(-3, 8))  # generalises beyond the spec
    assert "matches" in res.certificate


def test_unreachable_spec_is_not_adopted():
    # exponential is outside add/sub/mul over these parents -> fail-closed, nothing adopted
    a = ("mul", VAR, VAR)
    b = ("add", VAR, ("const", 1))
    spec = [(x, 2 ** x) for x in range(1, 6)]
    res = GenomicRecombiner(seed=1, mutation_rounds=50).breed(a, b, spec)
    assert res.verified is False and res.child is None


def test_disabled_recombiner_is_noop():
    res = GenomicRecombiner(enabled=False).breed(("var",), ("const", 1), [(1, 1)])
    assert res.verified is False and "disabled" in res.reason


def test_deterministic_under_seed():
    a, b = ("mul", VAR, VAR), ("add", VAR, ("const", 2))
    spec = [(x, x * x + 2) for x in range(6)]
    r1 = GenomicRecombiner(seed=7).breed(a, b, spec)
    r2 = GenomicRecombiner(seed=7).breed(a, b, spec)
    assert r1.child == r2.child and r1.verified and r2.verified
