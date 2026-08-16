"""Nine cells sharing one input are one cell, however long the head trains.

The one gradient learner in :mod:`nyxara.njp` hashed cells into a fixed 512-wide input. Measured
after 1,320 turns: **4,700 cells mapped, 4,188 collisions** — 89% of cells sharing a slot with
roughly eight others, and an accuracy of 0.0 while the loss fell from 75 to 66. A representation
in which nine different cells are literally the same input cannot distinguish them by training.

The fixed width was defended on the grounds that resizing on every birth would throw away
everything learned. That is true, and it is not the only option: widening in steps and copying
the learned weights into the larger matrices preserves every slot that already existed.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nyxara.njp.brain import NJPBrain
from nyxara.njp.learn import Readout, available


pytestmark = pytest.mark.skipif(not available(), reason="gradients need numpy")


# --------------------------------------------------------------------------- #
# Every cell gets its own input
# --------------------------------------------------------------------------- #
def test_distinct_cells_get_distinct_slots():
    readout = Readout(width=64, hidden=8)
    slots = [readout.slot(cell) for cell in range(50)]
    assert len(set(slots)) == len(slots)


def test_more_cells_than_the_initial_width_still_collide_with_nothing():
    readout = Readout(width=64, hidden=8)
    for cell in range(400):
        readout.slot(cell)
    assert readout.stats()["slot_collisions"] == 0
    assert readout.width > 64


def test_cells_that_hash_to_one_slot_are_probed_apart():
    """`cell % width` collides at the birthday rate long before the table is full."""
    readout = Readout(width=32, hidden=8)
    assert readout.slot(1) != readout.slot(33)      # both hash to 1 at width 32


# --------------------------------------------------------------------------- #
# Growing is not forgetting
# --------------------------------------------------------------------------- #
def test_growth_carries_the_learned_weights_across():
    readout = Readout(width=32, hidden=8)
    for cell in range(20):
        readout.slot(cell)
    # Make the rows distinguishable, so a lost one is visible.
    for cell, slot in readout._slots.items():
        readout.W1.data[slot] = float(cell + 1)
    remembered = {cell: readout.W1.data[slot].copy()
                  for cell, slot in readout._slots.items()}

    for cell in range(20, 200):
        readout.slot(cell)
    assert readout.growths > 0

    for cell, row in remembered.items():
        assert np.allclose(readout.W1.data[readout.slot(cell)], row), cell


def test_growth_carries_the_optimiser_moments_across():
    """Resetting them makes the step after a growth behave as though training just started."""
    readout = Readout(width=32, hidden=8)
    for cell in range(20):
        readout.slot(cell)
    readout._m[0][readout.slot(3)] = 7.0
    for cell in range(20, 200):
        readout.slot(cell)
    assert readout._m[0][readout.slot(3)] == pytest.approx(7.0)


def test_a_slot_moving_is_not_a_slot_forgetting():
    readout = Readout(width=32, hidden=8)
    before = readout.slot(5)
    readout.W1.data[before] = 3.0
    for cell in range(200):
        readout.slot(cell)
    after = readout.slot(5)
    assert np.allclose(readout.W1.data[after], 3.0)


# --------------------------------------------------------------------------- #
# Growth is bounded
# --------------------------------------------------------------------------- #
def test_the_width_stops_at_its_ceiling():
    """Growth must not become the memory leak the fixed width was avoiding."""
    readout = Readout(width=16, hidden=4, max_width=64)
    for cell in range(500):
        readout.slot(cell)
    assert readout.width <= 64


def test_a_full_table_collides_and_says_so():
    readout = Readout(width=16, hidden=4, max_width=16)
    for cell in range(200):
        readout.slot(cell)
    assert readout.stats()["slot_collisions"] > 0, "a genuinely full table must report it"


# --------------------------------------------------------------------------- #
# It still learns
# --------------------------------------------------------------------------- #
def test_the_head_still_trains_after_growing():
    readout = Readout(width=32, hidden=8, lr=0.05)
    for cell in range(120):
        readout.slot(cell)
    step = readout.train_step([([1, 2, 3], [4, 5, 6])])
    assert step is not None
    assert step.loss_after <= step.loss_before


def test_a_whole_brain_maps_every_cell_it_grows():
    brain = NJPBrain()
    for i in range(40):
        brain.think(f"cell {i} se anokha nateeja {i} hota hai")
    stats = brain.stats()["readout"]
    if stats.get("cells_mapped"):
        assert stats["collision_rate"] == 0.0
