"""Kal se aaj: the brain that wakes up must be the brain that went to sleep."""

from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.fabric import Fabric


def test_a_grown_fabric_survives_a_round_trip():
    f = Fabric(seed=9, manifold_dim=512)
    for cells in ([1, 2, 3], [3, 4, 5], [5, 6, 7]):
        f.stimulate(cells)
        f.expand(result=f.settle())
    payload = f.to_dict()

    reborn = Fabric(seed=9, manifold_dim=512)
    reborn.load_dict(payload)
    assert reborn.n_cells == f.n_cells
    assert reborn.n_synapses == f.n_synapses
    assert reborn.out == f.out


def test_a_reloaded_brain_keeps_growing_from_where_it_was():
    """The point of persistence: tomorrow starts from today's size, not from zero."""
    brain = NJPBrain()
    for text in ("gravity pulls the apple", "the apple falls down"):
        brain.think(text)
    payload = brain.to_dict()
    grown = brain.fabric.n_synapses
    assert grown > 0

    reborn = NJPBrain()
    reborn.load_dict(payload)
    assert reborn.fabric.n_synapses == grown
    reborn.think("gravity pulls the moon")
    assert reborn.fabric.n_synapses > grown


def test_the_same_word_maps_to_the_same_cell_across_brains():
    """Without stable ids a reloaded fabric is wired for concepts it can no longer name."""
    a, b = NJPBrain(), NJPBrain()
    assert a.encode("gravity apple") == b.encode("gravity apple")


def test_a_corrupt_sidecar_leaves_a_working_brain():
    """A bad file must never block boot."""
    brain = NJPBrain()
    brain.load_dict({"fabric": {"cells": "not a dict"}, "turns": "nonsense"})
    assert brain.think("still works").percept is not None
