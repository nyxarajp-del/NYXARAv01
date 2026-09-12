"""Where the tower stops, in bits.

V.95 charged for picking a way out of a language and never for the language existing. Adding that
line opens a regress — a language is chosen from a family, which is chosen from — and it closes by
arithmetic rather than by assertion. Most of what is pinned here is that the arithmetic has more
than one answer.
"""

from __future__ import annotations

import pytest

from nyxara.njp.generators import WIDTH, longhand
from nyxara.njp.tower import LANGUAGES, Climb, Language, Storey, climb, price
from nyxara.njp.towerschool import KNOWN, WORLDS, examine


def _turns():
    return WORLDS["a sensible default"][0]()


# --------------------------------------------------------------------------------------------- #
#  the regress closes by arithmetic
# --------------------------------------------------------------------------------------------- #
def test_a_sensible_default_stops_at_one():
    """Naming a language costs more than naming well inside one saves."""
    got = climb(_turns(), default=Language(3))
    assert got.stops_at == 1
    assert got.worth_climbing == [0, 1], "storey two does not pay here"


def test_an_absurd_default_makes_the_climb_worth_it():
    """The proof the measure is not a preference. Same arithmetic, different answer."""
    got = climb(_turns(), default=Language(8))
    assert got.stops_at == 2
    assert 2 in got.worth_climbing
    by_height = {s.height: s for s in got.storeys}
    assert by_height[2].toll < by_height[1].toll, "a smaller language, named, costs less"
    assert by_height[2].description == by_height[1].description, "and explains exactly as much"


def test_nothing_to_explain_stops_at_the_ground():
    got = climb(WORLDS["nothing to explain"][0](), default=Language(3))
    assert got.stops_at == 0
    assert got.worth_climbing == [0]


def test_three_worlds_give_three_different_heights():
    """A module that always answers the same number is not measuring anything."""
    heights = {climb(make(), default=lang).stops_at for make, lang in WORLDS.values()}
    assert heights == {0, 1, 2}


# --------------------------------------------------------------------------------------------- #
#  why there is a top at all
# --------------------------------------------------------------------------------------------- #
def test_a_languages_toll_never_falls_as_it_widens():
    """The toll does not shrink as you climb. The savings do. That is why there is a top."""
    tolls = [lang.toll for lang in sorted(LANGUAGES, key=lambda l: l.longest)]
    assert tolls == sorted(tolls)
    assert tolls[0] < tolls[-1]


def test_a_language_is_one_number_because_only_one_is_read():
    """It carried two at first, and the search never varied the second.

    Size was computed over an alphabet nothing used, which prices a language for expressiveness it
    does not have. A parameter nothing reads is not a parameter.
    """
    assert Language(3).size == 2 + 4 + 8 == 14
    assert Language(1).size == 2
    assert Language(3).name == "words to 3"
    assert not hasattr(Language(3), "letters")


def test_the_toll_and_the_description_are_charged_separately():
    """They are different debts — one for having chosen, one for what was said."""
    got = climb(_turns(), default=Language(3))
    for storey in got.storeys:
        assert storey.total == pytest.approx(storey.toll + storey.description, abs=0.01)
    ground = next(s for s in got.storeys if s.height == 0)
    assert ground.toll == 0.0, "writing everything out chooses nothing"


def test_the_ground_floor_is_always_longhand():
    got = climb(_turns())
    ground = next(s for s in got.storeys if s.height == 0)
    assert ground.description == longhand(len(set(_turns())), WIDTH)


def test_a_language_that_explains_nothing_falls_back_to_longhand():
    bare, worked, _ = price(WORLDS["nothing to explain"][0](), Language(3))
    assert not worked
    assert bare == longhand(WIDTH, WIDTH)


def test_an_empty_climb_claims_nothing():
    assert Climb().stops_at == 0 and Climb().worth_climbing == []
    assert Storey().total == 0.0


def test_it_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 3
    assert got["heights_seen"] == [0, 1, 2]
    assert got["toll_never_falls"]
    assert got["passes"]


def test_every_world_has_a_different_answer_written_down():
    assert sorted(c.stops_at for c in KNOWN) == [0, 1, 2]
