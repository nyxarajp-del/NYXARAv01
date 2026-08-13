"""The hyper-dimensional latent manifold: many variables as one object, and foresight."""

from __future__ import annotations

import pytest

from nyxara.njp.manifold import Manifold

np = pytest.importorskip("numpy")


def test_a_snapshot_is_deterministic_and_distinct():
    m = Manifold(dim=2048, seed=7)
    a, b = m.encode([1, 2, 3]), m.encode([1, 2, 3])
    other = m.encode([90, 91, 92])
    assert m.similarity(a.vector, b.vector) == 1.0
    assert abs(m.similarity(a.vector, other.vector)) < 0.2


def test_a_thousand_variables_ride_in_one_object():
    """The actual content of 'sees a thousand variables at once', held to the measured truth.

    At dim=32768 a 1000-cell configuration is one vector and EVERY member is still recoverable
    from it — that is the honest version of the claim, and the width it actually costs.
    """
    m = Manifold(dim=32768, seed=7)
    ids = list(range(1000))
    snap = m.encode(ids)
    assert len(snap.vector) == 32768
    assert snap.width == 1000
    assert all(m.similarity(snap.vector, m.atom(c)) > 0.0 for c in ids)


def test_past_capacity_it_degrades_gracefully_rather_than_collapsing():
    """Overfilling a manifold loses members slowly, and the probe is what says where the line is.

    Pinned because the failure mode matters: 991/1000 recoverable is a usable, honestly-degraded
    snapshot; a cliff would make the whole structure worthless the moment it was crossed.
    """
    m = Manifold(dim=10000, seed=7)
    assert m.capacity_probe(stop=4096) == 512          # the clean ceiling on this width
    ids = list(range(1000))                            # nearly double it
    snap = m.encode(ids)
    recovered = sum(1 for c in ids if m.similarity(snap.vector, m.atom(c)) > 0.0)
    assert recovered > 950


def test_capacity_is_measured_not_quoted():
    """A probe that reports where cleanup ACTUALLY breaks on this machine."""
    small, large = Manifold(dim=512, seed=7), Manifold(dim=8192, seed=7)
    assert large.capacity_probe() > small.capacity_probe()


def test_binding_is_invertible_so_order_survives():
    """'A turns B' and 'B turns A' must not be the same object."""
    m = Manifold(dim=4096, seed=7)
    role, filler = m.atom(1), m.atom(2)
    bound = m.bind(role, filler)
    assert m.similarity(m.bind(bound, role), filler) == 1.0
    assert abs(m.similarity(bound, filler)) < 0.2


def test_it_predicts_a_learned_transition_without_settling():
    """Pre-cognition: the result is known before the event is run."""
    m = Manifold(dim=4096, seed=7, min_samples=4)
    before, after = m.encode([1, 2, 3]), m.encode([4, 5, 6])
    for _ in range(12):
        m.learn_transition(before, after)
    pool = [1, 2, 3, 4, 5, 6, 7, 8]
    got = m.precognition(before, k=3, candidates=pool)
    assert got.trusted is True
    assert set(got.cells) == {4, 5, 6}
    assert m.score_prediction(got.cells, [4, 5, 6]) == 1.0


def test_an_unearned_prediction_says_so_rather_than_guessing():
    """Foresight that has not been earned is reported as absent, with the reason."""
    m = Manifold(dim=4096, seed=7, min_samples=8)
    got = m.precognition(m.encode([1, 2]), candidates=[1, 2, 3])
    assert got.trusted is False
    assert got.reason


def test_the_transition_map_stays_bounded():
    """The low-rank form is what keeps this from being a 400 MB matrix per brain."""
    m = Manifold(dim=1024, seed=7, max_pairs=16)
    for i in range(100):
        m.learn_transition(m.encode([i, i + 1]), m.encode([i + 2, i + 3]))
    assert m.stats()["pairs"] <= 16
    assert m.samples == 100
