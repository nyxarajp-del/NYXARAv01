"""The claim this whole package has to earn: she physically expands after a task."""

from __future__ import annotations

from nyxara.njp.fabric import Fabric


def _turn(f: Fabric, cells, outcome: float = 1.0):
    f.stimulate(cells)
    return f.expand(outcome=outcome, result=f.settle())


def test_a_cold_fabric_grows_from_its_first_turn():
    """The bootstrap case: no synapses at all, and one turn has to produce some.

    This is the regression that matters most — plasticity reads the firing trace, the trace comes
    from propagation, and propagation needs synapses. If the drive is not recorded as firing, the
    fabric mints cells and stays at zero synapses forever.
    """
    f = Fabric(seed=11, manifold_dim=512)
    assert f.n_synapses == 0
    rep = _turn(f, [1, 2, 3, 4])
    assert rep.grown > 0
    assert rep.expanded is True
    assert f.n_synapses > 0


def test_synapse_count_rises_across_turns():
    f = Fabric(seed=11, manifold_dim=512)
    counts = []
    for cells in ([1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]):
        _turn(f, cells)
        counts.append(f.n_synapses)
    assert counts == sorted(counts)
    assert counts[-1] > counts[0]
    assert f.stats()["growing"] is True


def test_a_repeat_potentiates_rather_than_growing():
    """Growth is for what is new. Seeing the same thing again strengthens what is there."""
    f = Fabric(seed=11, manifold_dim=512)
    _turn(f, [1, 2, 3])
    rep = _turn(f, [1, 2, 3])
    assert rep.potentiated > 0
    assert rep.grown == 0


def test_growth_budget_is_respected():
    """One strange turn must not be able to rewire the whole fabric."""
    f = Fabric(seed=11, manifold_dim=512, growth_budget=2)
    rep = _turn(f, list(range(1, 30)))
    assert rep.grown <= 2


def test_a_bad_outcome_does_not_grow():
    """Negative outcome weakens what fired; it never rewards it with new wiring."""
    f = Fabric(seed=11, manifold_dim=512)
    rep = _turn(f, [1, 2, 3], outcome=-1.0)
    assert rep.grown == 0


def test_expand_without_a_settle_changes_nothing():
    f = Fabric(seed=11, manifold_dim=512)
    rep = f.expand(outcome=1.0)
    assert rep.expanded is False
    assert rep.net_synapses == 0
