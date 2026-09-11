"""V.78 — how far it reaches, and the four ways a ladder has no edge.

The failure mode worth testing is not *misses the cliff*. It is **finds a cliff in every ladder** —
a number that always arrives looks like a result, and a roadmap built on one sends work at noise.
So most of what is pinned here is refusal: no edge where competence never starts, none where it
never stops, none where it skips a rung, and the difference between stopping and fading.
"""

from __future__ import annotations

import random

from nyxara.njp.measurement import Benchmark
from nyxara.njp.reach import CLEAR, CLIFF, WOBBLE, Ladder, Rung, climb
from nyxara.njp.reachschool import DIAL, ITEMS, KNOWN, examine, retrodict


def _at(shape, *, skew: float = 0.5, wobbles: bool = False, leaks: bool = False):
    def _make(at):
        rng = random.Random(78)
        gold = ["A" if rng.random() < skew else "B" for _ in range(ITEMS)]
        share = max(0.0, min(1.0, shape(at)))
        hit = random.Random(78 + at + 1000)
        said = [g if hit.random() < share else ("B" if g == "A" else "A") for g in gold]
        # Unstable **and good**, which is the only way to test instability on its own. A coin
        # looks unstable and is also at its floor, so a rung built from one fails for two reasons
        # at once and proves neither. This one scores 0.95 on odd runs and 0.75 on even ones: well
        # clear of its floor either way, and a fifth of the score apart between identical runs.
        run = {"n": 0}

        def _flips(i: int) -> str:
            if i == 0:
                run["n"] += 1
            if run["n"] % 2 == 0 and i % 5 == 0:
                return "B" if said[i] == "A" else "A"
            return said[i]

        predict = _flips if wobbles else (lambda i: said[i])
        return Benchmark(name=str(at), items=list(range(ITEMS)), gold=gold, predict=predict,
                         train=list(range(ITEMS)) if leaks else (),
                         key=(lambda i: i) if leaks else None)
    return _make


# --------------------------------------------------------------------------------------------- #
#  a rung holds only when it has earned it
# --------------------------------------------------------------------------------------------- #
def test_a_rung_is_measured_against_its_own_floor_not_a_shared_one():
    """Difficulty usually moves the floor, and 0.88 on a rung whose majority is 0.90 is nothing."""
    ladder = climb(_at(lambda _n: 0.88, skew=0.9), (1,), called="d")
    assert ladder.rungs[0].score > 0.85
    assert not ladder.rungs[0].holds
    assert ladder.barren


def test_a_rung_whose_score_will_not_hold_still_does_not_count_as_held():
    """It scores well clear of its floor and still does not count, because it will not repeat."""
    ladder = climb(_at(lambda _n: 0.95, wobbles=True), (1, 2, 3), called="d")
    assert all(r.above >= CLEAR for r in ladder.rungs), "it is not failing on the floor"
    assert all(r.spread > WOBBLE for r in ladder.rungs)
    assert ladder.barren


def test_a_rung_measured_on_what_it_was_taught_does_not_count_as_held():
    ladder = climb(_at(lambda _n: 0.95, leaks=True), (1, 2, 3), called="d")
    assert all(r.leaks for r in ladder.rungs)
    assert ladder.barren


def test_a_rung_that_could_not_be_built_is_recorded_rather_than_skipped():
    def _boom(at):
        if at == 2:
            raise RuntimeError("no")
        return _at(lambda _n: 0.95)(at)
    ladder = climb(_boom, (1, 2, 3), called="d")
    assert len(ladder.rungs) == 3
    assert "could not be built" in ladder.rungs[1].note
    assert not ladder.rungs[1].holds


# --------------------------------------------------------------------------------------------- #
#  the four ways there is no edge
# --------------------------------------------------------------------------------------------- #
def test_no_edge_where_competence_never_starts():
    ladder = climb(_at(lambda _n: 0.5), DIAL, called="d")
    assert ladder.barren and ladder.edge is None


def test_no_edge_where_competence_never_stops():
    """`no boundary within what was tried` is not `no boundary`, and the flag says which."""
    ladder = climb(_at(lambda _n: 0.95), DIAL, called="d")
    assert ladder.exhausted and ladder.edge is None
    assert "not turned far enough" in ladder.verdict


def test_no_edge_where_it_holds_above_a_rung_it_fails():
    ladder = climb(_at(lambda n: 0.51 if n in (3, 5) else 0.92), DIAL, called="d")
    assert ladder.patchy and ladder.edge is None
    assert "not a boundary" in ladder.verdict


def test_an_empty_ladder_claims_nothing():
    assert Ladder().edge is None and Ladder().verdict == "nothing measured"
    assert not Ladder().barren and not Ladder().exhausted


# --------------------------------------------------------------------------------------------- #
#  stopping and fading are different facts
# --------------------------------------------------------------------------------------------- #
def test_a_collapse_is_reported_as_a_cliff_with_a_place_to_attack():
    ladder = climb(_at(lambda n: 0.92 if n <= 4 else 0.51), DIAL, called="d")
    assert ladder.edge == 4
    assert ladder.cliff is not None and ladder.cliff[0] == 5
    assert ladder.cliff[1] >= CLIFF


def test_a_fade_has_an_edge_like_any_other_and_no_cliff():
    """Reporting the fade *instead of* the edge threw away the more useful of the two facts."""
    ladder = climb(_at(lambda n: 0.82 - 0.05 * (n - 1)), DIAL, called="d")
    assert ladder.edge is not None
    assert ladder.cliff is None and ladder.graceful
    assert "no one place to attack" in ladder.verdict


def test_the_first_rung_past_the_edge_is_named_because_its_neighbour_is_known_to_work():
    ladder = climb(_at(lambda n: 0.92 if n <= 4 else 0.51), DIAL, called="d")
    assert ladder.next_to_build == 5


def test_nothing_is_named_to_build_where_there_is_no_edge():
    for shape in (lambda _n: 0.5, lambda _n: 0.95):
        assert climb(_at(shape), DIAL, called="d").next_to_build is None


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_it_gets_all_five_ladders_right():
    got = retrodict()
    assert got["right"] == got["of"], [(r["case"], r["want"], r["got"]) for r in got["rows"]
                                       if not r["ok"]]


def test_it_invents_no_edges():
    """A boundary reported where there is none becomes the next thing somebody builds."""
    assert retrodict()["invented"] == 0


def test_the_exam_states_its_own_pass_condition():
    got = examine()
    assert got["passes"] is True
    assert set(got) >= {"right", "invented", "missed", "of"}


def test_only_one_of_the_five_has_an_edge_to_find():
    """Four of the fixtures are refusals. An organ that always answers scores one in five."""
    assert sum(1 for c in KNOWN if c.edge is not None) == 2


def test_the_bars_are_stated_numbers_rather_than_taste():
    assert 0.0 < CLEAR < CLIFF < 1.0 and 0.0 < WOBBLE < 1.0
    assert Rung().holds is False
