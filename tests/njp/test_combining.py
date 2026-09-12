"""The haystack is charged to the needle.

V.94 found *what* to combine and was told *how*. Here the how is searched too — and the search pays
for its own size, which is the only thing that stops "invent a way of combining" being an
instruction a search can satisfy by widening itself.
"""

from __future__ import annotations

import pytest

from nyxara.njp.combining import LETTERS, LONGEST, WAYS, Found, Way, charged, search, ways
from nyxara.njp.combiningschool import FAMILIES, KNOWN, examine, widening
from nyxara.njp.generators import WIDTH, longhand


# --------------------------------------------------------------------------------------------- #
#  the claim the version rests on
# --------------------------------------------------------------------------------------------- #
def test_a_wider_search_cannot_buy_a_cheaper_answer():
    """The same answer costs more as the list of candidates grows."""
    steps = [s for s in widening() if s["worth_it"]]
    assert len(steps) >= 3
    assert {s["way"] for s in steps} == {"ab"}, "the same answer throughout"
    for before, after in zip(steps, steps[1:]):
        assert after["cost"] > before["cost"], (before, after)
        assert after["charged"] > before["charged"]


def test_the_charge_is_the_logarithm_of_how_many_were_looked_at():
    assert charged(1) == 0.0
    assert charged(2) == 1.0
    assert charged(1024) == 10.0
    assert charged(0) == 0.0, "nothing considered is nothing owed, not an error"


def test_a_million_candidates_would_owe_twenty_bits_before_saying_anything():
    assert charged(2 ** 20) == 20.0
    assert charged(len(WAYS)) == pytest.approx(3.807, abs=0.01)


# --------------------------------------------------------------------------------------------- #
#  the candidate list is not a hint
# --------------------------------------------------------------------------------------------- #
def test_the_useless_ways_are_left_in_on_purpose():
    """A list containing only sensible candidates has had its answer chosen for it."""
    names = {w.name for w in WAYS}
    assert {"a", "b", "aa", "bb"} <= names, "the degenerate ones are in the list"
    assert "ab" in names and "ba" in names


def test_a_way_that_ignores_an_argument_reaches_almost_nothing():
    turns = FAMILIES["turns"]()
    ignoring = Way("a")
    assert ignoring.of(turns[1], turns[2]) == turns[1], "it really does ignore the second"
    got = search(turns, longest=1)
    assert not got.worth_it, "with only degenerate ways available, nothing may be found"


def test_no_degenerate_way_ever_wins():
    for case in KNOWN:
        got = search(FAMILIES[case.name]())
        if got.worth_it and got.way is not None:
            assert len(set(got.way.word)) > 1, f"{case.name} won with {got.way.word}"


def test_composition_is_just_one_member_of_the_list():
    """Nothing marks `ab` as special; it wins by costing less, having paid to be chosen."""
    one, two = FAMILIES["turns"]()[1], FAMILIES["turns"]()[3]
    assert Way("ab").of(one, two) == tuple(one[two[i]] for i in range(WIDTH))
    assert Way("ba").of(one, two) == tuple(two[one[i]] for i in range(WIDTH))
    assert len(LETTERS) == 2 and LONGEST >= 2


def test_a_word_that_does_not_give_a_rearrangement_gives_nothing():
    assert Way("ab").of((0, 1), (0, 0)) is None
    assert Way("ab").of((0, 1, 2), (0, 1)) is None
    assert Way("zz").of((0, 1), (1, 0)) is None


# --------------------------------------------------------------------------------------------- #
#  finding, and refusing
# --------------------------------------------------------------------------------------------- #
def test_turns_are_one_seed_under_one_way():
    got = search(FAMILIES["turns"]())
    assert got.worth_it and len(got.seeds) == 1 and got.complete
    assert got.cost < longhand(got.of) / 3
    assert got.cost > got.bare, "the way it was chosen is part of the bill"


def test_unrelated_transformations_produce_nothing():
    got = search(FAMILIES["unrelated"]())
    assert not got.worth_it and got.way is None
    assert "no way of combining pays for itself" in got.render()


def test_the_number_of_ways_is_counted_and_not_estimated():
    """Or the charge is a guess, and a guessed charge is not a charge."""
    assert len(ways(1)) == 2 and len(ways(2)) == 6 and len(ways(3)) == 14
    assert len({w.name for w in ways(3)}) == 14


def test_an_empty_search_claims_nothing():
    assert not Found().worth_it and not Found().complete
    assert Found().to_dict()["way"] == ""
    assert not search([]).worth_it


def test_it_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 4
    assert got["flattered"] == 0 and got["buried"] == 0
    assert got["degenerate"] == 0
    assert got["cost_climbs_with_the_search"]
    assert got["nothing_from_degenerate_ways_alone"]
    assert got["passes"]
