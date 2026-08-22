"""The substrate can name what it predicts — brain.lexicon and Readout.predict_symbols.

**The blocker this removes.** :meth:`~nyxara.njp.learn.Readout.predict` returns gradient-trained
probabilities over *cell ids*, and :func:`~nyxara.njp.brain._cell_id` is a blake2b digest — one-way
by construction. So the one learner in the package that uses real gradients could be trained, could
be measured, and could not contribute a word to any conversation about what is true. Not because
the learning was weak: the loss falls, the head trains, and every number it produces is about
integers nobody can read.

The map back was never hard to have. She computes it on every turn in
:meth:`~nyxara.njp.brain.NJPBrain.encode` and was discarding it. Keeping it is the whole fix.

**What is deliberately not claimed here.** The head's objective is
``(what fired last turn) → (what fires this turn)``, so its symbols are *what tends to come up
next*, not *what is true of this subject*. The tests below pin the naming, the exclusions and the
persistence — the mechanics that make it honest — and the docstring on ``predict_symbols`` carries
the measured strength, which is weak. Nothing here asserts the proposals are good.
"""
from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.learn import available


pytestmark = pytest.mark.skipif(not available(), reason="gradients need numpy")


def _taught(epochs: int = 6) -> NJPBrain:
    brain = NJPBrain()
    for _ in range(epochs):
        for sentence in ("a sparrow is a bird", "birds need water",
                         "a crow is a bird", "crow needs grain",
                         "Aag se garmi hoti hai"):
            brain.think(sentence)
    return brain


# --------------------------------------------------------------------------- #
# The inverse map
# --------------------------------------------------------------------------- #
def test_encoding_a_turn_records_what_each_cell_is_called():
    brain = NJPBrain()
    cells = brain.encode("sparrow needs water")
    assert cells
    assert all(brain.name_cell(cell) for cell in cells)


def test_a_cell_she_never_encoded_has_no_name():
    """Empty rather than a placeholder — a stand-in is a symbol she never learned."""
    assert NJPBrain().name_cell(1234567890) == ""


def test_the_first_spelling_of_a_cell_wins():
    """A later word must not silently rename an established cell."""
    brain = NJPBrain()
    cell = brain.encode("sparrow")[0]
    first = brain.name_cell(cell)
    brain.encode("sparrow")
    assert brain.name_cell(cell) == first


def test_the_lexicon_survives_a_reload():
    """A reloaded brain wired for cells it cannot name is worse than starting fresh."""
    saved = NJPBrain()
    saved.think("a sparrow is a bird")
    saved.think("birds need water")
    cell = saved.encode("sparrow")[0]

    reloaded = NJPBrain()
    reloaded.load_dict(saved.to_dict())
    assert reloaded.name_cell(cell) == "sparrow"
    assert len(reloaded.lexicon) == len(saved.lexicon)


# --------------------------------------------------------------------------- #
# What the head is allowed to say
# --------------------------------------------------------------------------- #
def test_the_substrate_returns_concepts_rather_than_cell_ids():
    proposals = _taught().fabric_proposes("sparrow", k=4)
    assert proposals
    for name, probability in proposals:
        assert isinstance(name, str) and name
        assert 0.0 <= probability <= 1.0


def test_an_unnamed_cell_is_dropped_never_invented():
    brain = _taught()
    brain.lexicon.clear()
    assert brain.fabric_proposes("sparrow", k=4) == []


def test_it_does_not_propose_the_concepts_it_was_given():
    """Handing back the query is not a prediction, and reads as agreement with whatever asked."""
    brain = _taught()
    names = {n for n, _ in brain.fabric_proposes("sparrow water", k=6)}
    assert "sparrow" not in names
    assert "water" not in names


def test_a_floor_silences_a_head_that_has_learned_nothing():
    assert NJPBrain().fabric_proposes("sparrow", k=4, floor=0.99) == []


def test_it_degrades_to_nothing_without_a_readout():
    brain = _taught()
    brain.readout = None
    assert brain.fabric_proposes("sparrow", k=4) == []


def test_the_proposals_are_ranked_most_confident_first():
    proposals = _taught().fabric_proposes("sparrow", k=5)
    scores = [p for _, p in proposals]
    assert scores == sorted(scores, reverse=True)


def test_k_is_honoured():
    assert len(_taught().fabric_proposes("sparrow", k=2)) <= 2
