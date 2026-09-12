"""Earning the moves a law is made of.

The test that matters is not that 252 moves were generated. It is that a generated move buys a
separation the supplied seven cannot make — and, just as hard, that it is **not** credited for the
ones they can.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.substrate import (
    LIFTS, SCALES, SUPPLIED, WIDTH, Earned, Shape, earns, every_move, reach, separations,
)
from nyxara.njp.substrateschool import (
    PAIRS, examine, stride, strides_are_already_covered, swap,
)


# --------------------------------------------------------------------------------------------- #
#  the substrate contains the vocabulary it replaces
# --------------------------------------------------------------------------------------------- #
def test_every_supplied_move_is_a_special_case_of_four_numbers():
    """Checked against V.90's own implementations, not against the arithmetic here."""
    from nyxara.njp.regularity import MOVES

    theirs = {m.name: m for m in MOVES}
    row = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
    for shape in SUPPLIED:
        assert shape.name in theirs, shape.name
        assert shape.of(row) == pytest.approx(list(theirs[shape.name](row)), abs=1e-12), shape.name


def test_a_move_is_four_numbers_and_a_lookup_table_is_not_expressible():
    """The guard is structural. A substrate that could fit anything would fit noise."""
    assert Shape().size == 4
    assert len(SCALES) <= 4 and len(LIFTS) <= 3, "widening these turns a substrate into a fitter"


def test_a_shape_carries_no_name_of_its_own():
    assert Shape(-1, 6).name == "(-1, 6, 1, 0)"
    assert Shape(-1, 6, called="reversed").name == "reversed"


def test_strides_that_lose_information_are_not_moves():
    """A stride sharing a factor with the width folds inputs together, and a law built on one is
    comparing against something that threw the row away."""
    every = every_move()
    assert all(m.stride % WIDTH != 0 for m in every)
    seen = {(m.stride, m.offset, m.scale, m.lift) for m in every}
    assert len(seen) == len(every), "deduplicated by behaviour, so no two act alike"


def test_the_substrate_is_much_larger_than_what_was_supplied():
    every = every_move()
    assert len(every) > 100 and len(SUPPLIED) == 7


# --------------------------------------------------------------------------------------------- #
#  what a vocabulary can see, computed rather than felt
# --------------------------------------------------------------------------------------------- #
def test_a_law_reads_a_pair_of_positions_not_one():
    """Moves apply to each side independently, so the reachable set is a product.

    This is the number that turned *the supplied moves feel expressive* into *they reach sixteen of
    forty-nine*, and it is what a generated move has to buy.
    """
    assert len(reach(SUPPLIED)) == 16
    assert len(reach(every_move())) == WIDTH * WIDTH == 49
    assert reach(SUPPLIED) < reach(every_move())


def test_the_unreadable_region_is_where_the_fixture_lives():
    offsets = {m.offset % WIDTH for m in SUPPLIED}
    assert offsets == {0, 1, 2, 6}
    assert 3 not in offsets and 4 not in offsets and 5 not in offsets


# --------------------------------------------------------------------------------------------- #
#  a move earns its place by the law it makes possible
# --------------------------------------------------------------------------------------------- #
def test_the_supplied_vocabulary_cannot_separate_the_fixture_pair():
    got = separations(swap(3, 4), swap(3, 5), list(SUPPLIED), tries=30, rng=random.Random(91))
    assert got == [], f"the supplied seven were supposed to be blind here, and found {len(got)}"


def test_a_generated_move_buys_that_separation():
    """Some generated move must make the difference statable. 144 of the 252 do."""
    works = Shape(1, 3, -1.0, 0.0)
    got = separations(swap(3, 4), swap(3, 5), list(SUPPLIED) + [works], tries=30,
                      rng=random.Random(91))
    assert got, "no generated move made the difference statable"


def test_reaching_a_new_position_is_not_by_itself_enough():
    """My first guess, wrong, and the reason it is wrong is worth keeping.

    A `first` law reads a **pair** of offsets, so one new offset only yields pairs with the four
    already there — and `(3, 4)` is not among them. Plain slide-by-three therefore buys nothing.
    Slide-by-three **negated** does, and the negation is what does the work: V.90 dropped `<=` as
    "`>=` with the sides swapped", which holds only when sides swap freely, and they do not — a
    law may not put `x` on both sides. `scale = -1` puts the missing direction back.
    """
    plain = Shape(1, 3, 1.0, 0.0)
    flipped = Shape(1, 3, -1.0, 0.0)
    assert not separations(swap(3, 4), swap(3, 5), list(SUPPLIED) + [plain], tries=30,
                           rng=random.Random(91))
    assert separations(swap(3, 4), swap(3, 5), list(SUPPLIED) + [flipped], tries=30,
                       rng=random.Random(91))


def test_a_move_that_is_a_supplied_one_in_other_numbers_earns_nothing():
    """`(1, 0, 1, 0)` is `as it is`. Crediting it is reporting an input back as a discovery."""
    got = earns(Shape(1, 0, 1.0, 0.0),
                [("a", swap(3, 4), "b", swap(3, 5))], tries=20, rng=random.Random(91))
    assert got.supplied_already and not got.earns


def test_nothing_is_bought_where_the_supplied_vocabulary_already_works():
    """Stride two against stride three. The supplied seven separate them, so no move may be paid."""
    got = earns(Shape(1, 3, 1.0, 0.0),
                [("a", stride(2), "b", stride(3))], tries=20, rng=random.Random(91))
    assert got.bought == [] and not got.earns


def test_an_empty_earning_claims_nothing():
    assert not Earned().earns
    assert Earned(move=Shape()).to_dict()["bought"] == []


# --------------------------------------------------------------------------------------------- #
#  and where generation buys nothing, worked out rather than asserted
# --------------------------------------------------------------------------------------------- #
def test_stride_permutations_are_already_covered_and_that_is_provable():
    """A version that only showed where generation wins would be advertising, not measuring."""
    got = strides_are_already_covered()
    assert got["supplied_offsets"] == [0, 1, 2, 6]
    assert len(got["strides_that_could_hide"]) < 2, got["strides_that_could_hide"]
    assert got["enough"]


def test_the_exam_has_a_pair_for_each_answer():
    assert sum(1 for p in PAIRS if p[4]) == 1, "one pair needs a new move"
    assert sum(1 for p in PAIRS if not p[4]) == 3, "three must not credit one"


def test_the_substrate_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 4
    assert got["flattered"] == 0 and got["missed"] == 0
    assert got["supplied_reach"] < got["whole_reach"]
    assert got["passes"]
