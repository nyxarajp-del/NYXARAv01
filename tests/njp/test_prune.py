"""Growth without forgetting is hoarding. Pruning and compression are load-bearing."""

from __future__ import annotations

from nyxara.njp.fabric import Fabric


def test_weak_synapses_are_dropped():
    f = Fabric(seed=5, manifold_dim=512, prune_threshold=0.5)
    f.connect(1, 2, 0.01)             # far below the floor
    f.connect(2, 3, 0.9)
    f.stimulate([1, 2])
    rep = f.expand(result=f.settle())
    assert rep.pruned >= 1
    assert 2 not in f.out.get(1, {})


def test_disconnect_clears_both_directions():
    """A stale reverse index is how a pruned synapse comes back to life."""
    f = Fabric(seed=5, manifold_dim=512)
    f.connect(1, 2, 0.5)
    assert f.disconnect(1, 2) is True
    assert 2 not in f.out.get(1, {})
    assert 1 not in f.inn.get(2, set())


def test_consolidate_compresses_the_weak_tail_and_keeps_the_strong():
    """Crossing a ceiling must degrade what is barely used, never the load-bearing structure."""
    f = Fabric(seed=5, manifold_dim=512)
    for i in range(2, 40):
        f.connect(1, i, 0.001 * i)     # a spread of weights, weakest first
    f.connect(1, 500, 3.5)             # the strong one
    before = f.n_synapses
    f.consolidate(fraction=0.25)
    assert f.n_synapses < before
    assert f.out[1].get(500) == 3.5, "consolidation must not touch load-bearing structure"


def test_weights_stay_bounded():
    """Unbounded weights make the automaton diverge rather than learn."""
    f = Fabric(seed=5, manifold_dim=512, hebbian_rate=5.0)
    f.connect(1, 2, 1.0)
    for _ in range(20):
        f.stimulate([1, 2])
        f.expand(result=f.settle())
    for targets in f.out.values():
        for w in targets.values():
            assert -4.0 <= w <= 4.0
