"""What a vocabulary costs and what it buys.

A measure of vocabularies has two opposite ways to be worthless — preferring the biggest and
preferring the smallest — and a test suite that guards only one of them certifies the other. Both
are pinned here, and so is the reason the exam's pairs are hard rather than random.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.worth import (
    SEED, VOCABULARIES, WIDTH, Vocabulary, Weighed, by_hand, compare, hard_pairs, separates, weigh,
)
from nyxara.njp.worthschool import easy_pairs, examine


def _named(name):
    return next(v for v in VOCABULARIES if v.name == name)


# --------------------------------------------------------------------------------------------- #
#  the two opposite failures
# --------------------------------------------------------------------------------------------- #
def test_the_biggest_vocabulary_does_not_win():
    """The failure a search told to invent a representation makes: invent an enormous one."""
    ranked = compare()
    biggest = max(VOCABULARIES, key=lambda v: len(v.distinct))
    assert ranked[0].vocabulary is not None
    assert ranked[0].vocabulary.name != biggest.name
    assert len(ranked[0].vocabulary.distinct) < len(biggest.distinct) / 4


def test_the_smallest_vocabulary_does_not_win_either():
    """The mirror failure, and the one my first measure actually committed.

    `worth / cost` put a one-move vocabulary first: 41 of 105 pairs for one bit. Charging for the
    leftovers is what fixes it, and the ratio is kept only as the diagnostic that shows the hole.
    """
    ranked = compare()
    one = weigh(_named("one move"), hard_pairs()[1])
    assert ranked[0].vocabulary.name != "one move"
    assert one.value > ranked[0].value, "on the ratio alone it still wins — that is the point"
    assert one.total > ranked[0].total * 50, "and on the whole description it is crushed"


def test_an_incomplete_vocabulary_pays_for_what_it_leaves_behind():
    held = hard_pairs()[1]
    seven = weigh(_named("the supplied seven"), held)
    assert not seven.complete and seven.leftover > 0
    assert seven.total == pytest.approx(seven.cost + seven.leftover * by_hand(seven.of), abs=1e-3)
    assert by_hand(105) > 6.0


# --------------------------------------------------------------------------------------------- #
#  the demotion
# --------------------------------------------------------------------------------------------- #
def test_seven_slides_describe_everything_more_cheaply_than_v91s_substrate():
    """The scale's first use, and it demotes this repository's own previous version."""
    held = hard_pairs()[1]
    slides = weigh(_named("slides only"), held)
    mine = weigh(_named("V.91's substrate"), held)
    assert slides.complete and mine.complete, "both explain everything"
    assert slides.total < mine.total / 2, "and one of them does it in less than half the bits"
    assert len(_named("slides only").distinct) * 30 < len(_named("V.91's substrate").distinct)


def test_the_supplied_seven_spend_their_budget_badly():
    """Three of their moves sit at offset zero, so they reach 16 pairs where slides reach 49."""
    seven, slides = _named("the supplied seven"), _named("slides only")
    assert seven.offsets == [0, 1, 2, WIDTH - 1]
    assert slides.offsets == list(range(WIDTH))
    assert seven.reach == 16 and slides.reach == 49
    assert len(seven.distinct) < len(slides.distinct), "and it is not even bigger"


# --------------------------------------------------------------------------------------------- #
#  the measure only bites where the work is
# --------------------------------------------------------------------------------------------- #
def test_random_pairs_saturate_and_would_certify_anything():
    """The evidence that the hard pairs are not an arbitrary choice."""
    easy = easy_pairs()
    full = [v.name for v in VOCABULARIES if weigh(v, easy).worth == len(easy)]
    assert len(full) >= len(VOCABULARIES) - 1
    assert "one move" not in full or len(full) == len(VOCABULARIES)


def test_hard_pairs_are_held_out_from_the_half_that_is_looked_at():
    seen, held = hard_pairs()
    assert seen and held
    assert not (set(seen) & set(held))
    assert len(seen) + len(held) == (WIDTH * (WIDTH - 1) // 2) * ((WIDTH * (WIDTH - 1) // 2) - 1) // 2


# --------------------------------------------------------------------------------------------- #
#  bookkeeping
# --------------------------------------------------------------------------------------------- #
def test_a_one_move_vocabulary_still_costs_a_bit():
    """Without the `+1` it costs nothing and wins every ratio by dividing by zero."""
    assert Vocabulary("x", ((1, 0, 1.0),)).cost == 1.0
    assert Vocabulary("y", ()).cost == 1.0


def test_separation_is_symmetric_and_a_thing_never_separates_from_itself():
    seen, _ = hard_pairs()
    one, two = seen[0]
    for vocabulary in VOCABULARIES:
        assert separates(one, two, vocabulary) == separates(two, one, vocabulary)
        assert not separates(one, one, vocabulary)


def test_an_empty_weighing_claims_nothing():
    assert not Weighed().complete
    assert Weighed().leftover == 0
    assert Weighed().to_dict()["worth"] == 0


def test_the_scale_passes_its_retrodiction():
    got = examine()
    assert got["steady"], got["orders"]
    assert got["winner"] == "slides only"
    assert got["winner"] not in (got["biggest"], got["smallest"])
    assert got["rows"][0]["complete"]
    assert got["passes"]
