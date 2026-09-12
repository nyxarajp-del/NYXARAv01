"""The bill charges for choosing, not for knowing.

Eight versions of accounting undone by one line — `charged(1) == 0` — and closed by counting the
source as well as the run. Most of what is pinned here is the hole itself, because a defect nothing
asserts is a defect that comes back.
"""

from __future__ import annotations

import pytest

from nyxara.njp.combining import charged
from nyxara.njp.supply import Across, Route, across, carried, free_lunch


# --------------------------------------------------------------------------------------------- #
#  the hole
# --------------------------------------------------------------------------------------------- #
def test_a_choice_among_one_option_costs_nothing():
    """The line that undoes V.92 through V.98, asserted so it cannot quietly come back."""
    assert free_lunch() == 0.0
    assert charged(1) == 0.0


def test_the_cheapest_tower_is_the_one_that_considered_nothing():
    """Which is the tower with the most handed to it — the opposite of what was being measured."""
    searching = charged(3)          # name the answer out of three possibilities
    handed = charged(1)             # name nothing; somebody wrote it down
    assert handed < searching
    assert handed == 0.0


def test_hard_coding_beats_the_v98_baseline_on_one_world():
    """Measured, not argued: 55.58 against 57.17 on the eight observations V.98 was built around."""
    from nyxara.njp.bedrock import direct, reach_needed
    from nyxara.njp.generators import WIDTH, watch_behaviour
    from nyxara.njp.tower import Language, price

    turn = lambda k: (lambda r: [r[(i + k) % WIDTH] for i in range(WIDTH)])
    obs = sorted({watch_behaviour(turn(k)) for k in range(WIDTH)})
    straight = direct(obs)
    lang = Language(2)
    bare, ok, _ = price(obs, lang)
    told = charged(1) + charged(1) + lang.toll + bare
    assert ok
    assert told < straight.total
    assert straight.total - told == pytest.approx(charged(reach_needed(obs)), abs=0.01), \
        "the gap is exactly what naming the answer costs — the bill has no column for being told"


# --------------------------------------------------------------------------------------------- #
#  the closure
# --------------------------------------------------------------------------------------------- #
def test_a_hard_code_carries_in_its_source_what_a_search_pays_at_run_time():
    for bound in (2, 3, 8, 64):
        assert carried(bound) == charged(bound)
    assert carried(1) == 0.0, "one possibility is no information, wherever it sits"


def test_over_several_worlds_the_two_routes_cost_the_same():
    """The theorem. The bits move from one column to the other; a two-part code sees them either way."""
    for n in (1, 2, 3, 5):
        got = across([3, 4, 5, 6, 7][:n], [53.0, 120.0, 290.0, 480.0, 700.0][:n])
        assert got.worlds == n
        assert got.same, got.render()
        assert got.searching.chosen > 0 and got.told.told > 0
        assert got.searching.told == 0.0 and got.told.chosen == 0.0


def test_the_saving_was_an_artefact_of_measuring_one_world():
    """On one world the hard-code looks free. It is free **for that world**, and only that one."""
    one = across([3], [53.0])
    assert one.same, "even at one world, once the source is counted"
    assert one.told.told == pytest.approx(charged(3), abs=1e-9)


def test_a_gap_either_way_would_be_a_defect():
    """If the columns stopped balancing, something in the accounting would be wrong again."""
    got = across([4, 4], [10.0, 10.0])
    assert got.searching.total == pytest.approx(got.told.total, abs=0.01)
    assert "the same, to the digit" in got.render()


def test_an_empty_comparison_claims_nothing():
    assert not Across().same
    assert Route().total == 0.0
    assert across([], []).worlds == 0
    assert not across([], []).same, "nothing compared is not agreement"
