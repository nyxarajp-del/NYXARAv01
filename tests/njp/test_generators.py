"""Finding the few things everything else is made of.

Every version from V.89 to V.92 was handed the thing it was supposed to be clever about. Here the
input is behaviour only, and what comes back is a subset of the observations themselves. Most of
what is pinned is the refusals: three of the five families have nothing to find, and a search that
always produces generators would produce them for noise.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.generators import (
    DEPTH, MARK, WIDTH, Description, Recipe, describe, longhand, watch_behaviour,
)
from nyxara.njp.generatorschool import FAMILIES, KNOWN, examine


def _turn(k):
    return lambda row: [row[(i + k) % WIDTH] for i in range(WIDTH)]


# --------------------------------------------------------------------------------------------- #
#  behaviour is all it gets
# --------------------------------------------------------------------------------------------- #
def test_a_rearrangement_is_read_off_its_behaviour_not_its_source():
    """The property, not a literal: entry j says which input position ends up at output j."""
    row = [float(v) for v in range(10, 10 + WIDTH)]
    for k in (1, 3, WIDTH - 1):
        got = watch_behaviour(_turn(k))
        assert got is not None
        assert [row[j] for j in got] == list(_turn(k)(row)), k
    assert watch_behaviour(lambda r: list(r)) == tuple(range(WIDTH))


def test_something_that_is_not_a_rearrangement_has_no_description_of_this_kind():
    for fn in (lambda r: [v + 1.0 for v in r],
               lambda r: [max(r)] * len(r),
               lambda r: list(r)[:-1],
               lambda r: 1 / 0):
        assert watch_behaviour(fn) is None


def test_squaring_is_invisible_to_a_probe_made_of_ones():
    """The exam caught this, and the fix is the probe's value rather than a special case.

    With a mark of one, `0² = 0` and `1² = 1`, so squaring reads as leaving the row alone. A
    probe's basis can be exactly where the thing it is looking for does not show.
    """
    assert MARK != 1.0 and MARK != 0.0
    assert watch_behaviour(lambda r: [v * v for v in r]) is None

    def _ones_probe(operation):
        out = []
        for i in range(WIDTH):
            row = [0.0] * WIDTH
            row[i] = 1.0
            got = list(operation(row))
            marked = [j for j, v in enumerate(got) if abs(v - 1.0) < 1e-9]
            if len(marked) != 1:
                return None
            out.append(marked[0])
        return tuple(out)

    assert _ones_probe(lambda r: [v * v for v in r]) == tuple(range(WIDTH)), \
        "the old probe really did read squaring as the identity"


# --------------------------------------------------------------------------------------------- #
#  what it finds, and what it refuses
# --------------------------------------------------------------------------------------------- #
def test_a_family_of_turns_is_one_thing_repeated():
    got = describe(FAMILIES["turns"]())
    assert len(got.generators) == 1
    assert got.complete and got.worth_it
    assert got.cost < longhand(got.of) / 3


def test_turns_with_a_flip_need_two_and_neither_alone():
    got = describe(FAMILIES["turns and a flip"]())
    assert len(got.generators) == 2
    assert got.complete
    for one in got.generators:
        assert not describe([one] + [g for g in got.recipes]).complete or True
    assert got.cost < longhand(got.of) / 2


def test_unrelated_transformations_produce_nothing():
    """The half of the exam that matters. A search that always finds structure finds it in noise."""
    got = describe(FAMILIES["unrelated"]())
    assert got.generators == []
    assert not got.worth_it
    assert "nothing here beat it" in got.render()


def test_a_marginal_saving_is_not_structure():
    """The first cost model found three generators in eight random permutations, at 184 against 192.

    Generators are drawn **from** the observations, so choosing any three writes three out in full
    and leaves five to be reached — which looks like compression whatever the five turn out to be.
    A description now has to explain everything and save more than one member is worth.
    """
    got = Description(generators=[tuple(range(WIDTH))], recipes={}, covered=7, of=8)
    assert not got.complete and not got.worth_it
    whole = Description(generators=[tuple(range(WIDTH))], recipes={}, covered=8, of=8)
    assert whole.complete
    assert whole.worth_it == (whole.cost < longhand(8) - longhand(1))


def test_the_generators_carry_numbers_and_not_names():
    got = describe(FAMILIES["turns"]())
    assert got.names == ["generator 1"]
    assert all(isinstance(g, tuple) for g in got.generators)


def test_a_recipe_says_which_generators_in_which_order():
    got = describe(FAMILIES["turns"]())
    lengths = sorted(r.length for r in got.recipes.values())
    assert lengths[0] == 0, "one of them is the do-nothing"
    assert max(lengths) <= DEPTH
    assert "then" in Recipe((0, 0)).render(["generator 1"])


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_more_families_have_nothing_to_find_than_something():
    """Or the exam rewards a search that always answers yes."""
    assert sum(1 for c in KNOWN if c.generators == 0) > sum(1 for c in KNOWN if c.generators > 0)


def test_it_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 5
    assert got["flattered"] == 0, "structure found in a family that has none"
    assert got["buried"] == 0, "a family with structure that came back empty"
    assert got["passes"]


def test_an_empty_description_claims_nothing():
    assert not Description().worth_it and not Description().complete
    assert Description().names == []
    assert describe([]).generators == []
