"""Tests for nyxara.causal.causal_knots — the Knot Mutation Failure hallucination gate."""

from __future__ import annotations

import pytest

from nyxara.causal.causal_knots import (CONCORDANT, DISCORDANT, KnotLattice,
                                        KnotMutationFailure)


def test_consistent_chain_ties_cleanly():
    lat = KnotLattice()
    lat.tie("rain", "wet_ground", CONCORDANT)
    lat.tie("wet_ground", "slippery", CONCORDANT)
    lat.tie("rain", "slippery", CONCORDANT)          # consistent with the chain
    assert len(lat) == 3
    assert lat.status()["components"] == 1


def test_contradiction_raises_knot_mutation_failure():
    lat = KnotLattice()
    lat.tie("rain", "wet_ground", CONCORDANT)
    lat.tie("wet_ground", "slippery", CONCORDANT)
    with pytest.raises(KnotMutationFailure) as exc:
        # rain ⇒ slippery via the chain, so claiming rain PREVENTS slippery cannot tie
        lat.tie("rain", "slippery", DISCORDANT)
    failure = exc.value
    assert "rain" in failure.cycle and "slippery" in failure.cycle
    assert failure.expected == CONCORDANT and failure.got == DISCORDANT


def test_assert_a_proposition_and_its_negation_fails_to_link():
    lat = KnotLattice()
    lat.assert_fact("sky_is_blue", True)
    with pytest.raises(KnotMutationFailure):
        lat.assert_fact("sky_is_blue", False)


def test_check_is_nonmutating_and_reports_first_contradiction():
    lat = KnotLattice()
    lat.tie("a", "b", CONCORDANT)
    before = len(lat)
    verdict = lat.check([("b", "c", CONCORDANT), ("a", "c", DISCORDANT)])
    assert verdict.consistent is False
    assert verdict.tied == 1                          # b~c tied, a~c contradicts
    assert verdict.failure is not None
    assert len(lat) == before                         # the live lattice is untouched


def test_commit_only_writes_a_fully_consistent_batch():
    lat = KnotLattice()
    ok = lat.commit([("x", "y", CONCORDANT), ("y", "z", CONCORDANT)])
    assert ok.consistent and len(lat) == 2
    bad = lat.commit([("z", "w", CONCORDANT), ("x", "w", DISCORDANT)])
    assert bad.consistent is False
    assert len(lat) == 2                              # nothing from the bad batch was written


def test_double_negation_is_consistent():
    lat = KnotLattice()
    lat.tie("p", "q", DISCORDANT)                     # p opposes q
    lat.tie("q", "r", DISCORDANT)                     # q opposes r
    lat.tie("p", "r", CONCORDANT)                     # ⇒ p agrees with r (two flips)
    assert lat.status()["components"] == 1
