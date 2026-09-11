"""V.80 — how often does the analogy finder see a shape that is not there?

`njp/fusion.py` names its own danger exactly — *"matching on four of five edges is exactly the
false analogy that makes this kind of system"* worthless — and guards against it with exact
isomorphism, a minimum edge count and a bounded radius. **Every one of those guards was an
argument.** Nothing had ever shown it two domains with no relationship and asked what it said.

What is pinned here is that number, from both sides: it must still find the shapes that are there,
and it must stop claiming ones that are not. A guard tight enough never to fire is a refusal, not
a guard, so recall is asserted alongside every false-alarm figure.
"""

from __future__ import annotations

import random

from nyxara.njp.fusion import DRAWS, LUCK, MIN_EDGES, Fusion
from nyxara.njp.fusionschool import (
    SHAPES, Explainer, Report, Store, _fuser, density, examine, planted, sweep, unrelated,
)


# --------------------------------------------------------------------------------------------- #
#  it finds what is there
# --------------------------------------------------------------------------------------------- #
def test_it_finds_a_planted_shape_in_two_domains_that_share_no_word():
    got = examine(40)
    assert got.recall == 1.0


def test_the_guard_does_not_cost_a_single_real_analogy():
    """A guard that fires on the findings is a refusal. Recall has to survive it untouched."""
    assert examine(40, luck=0.0).recall == examine(40, luck=LUCK).recall == 1.0


# --------------------------------------------------------------------------------------------- #
#  and stops claiming what is not
# --------------------------------------------------------------------------------------------- #
def test_it_used_to_invent_analogies_between_small_dense_unrelated_domains():
    """The finding this version exists for: a fifth of the time, with the guard off."""
    without = examine(80, nodes=3, edges=4, luck=0.0)
    assert without.false_alarms > 0.10, without.to_dict()


def test_and_does_not_with_the_guard_on():
    with_guard = examine(80, nodes=3, edges=4, luck=LUCK)
    assert with_guard.false_alarms == 0.0, with_guard.to_dict()
    assert with_guard.recall == 1.0


def test_the_guard_changes_nothing_where_there_was_nothing_to_fix():
    """On larger sparser domains exact isomorphism was already enough, and stays enough."""
    off = examine(60, nodes=8, edges=10, luck=0.0)
    on = examine(60, nodes=8, edges=10, luck=LUCK)
    assert off.false_alarms == on.false_alarms == 0.0


def test_the_invented_end_is_the_small_dense_end_and_the_sweep_shows_it():
    rows = density(60, luck=0.0)
    big = next(r for r in rows if r.nodes >= 8)
    small = next(r for r in rows if r.nodes == 3)
    assert small.false_alarms > big.false_alarms


# --------------------------------------------------------------------------------------------- #
#  the guard is a comparison against chance, and says so
# --------------------------------------------------------------------------------------------- #
def test_surprise_is_high_for_a_shape_two_random_graphs_would_also_make():
    rng = random.Random(80)
    rows, seeds = unrelated(rng, nodes=3, edges=4)
    fuser = _fuser(rows)
    left = fuser.pattern(seeds["left"][0], domain="left")
    right = fuser.pattern(seeds["right"][0], domain="right")
    assert fuser.surprise(left, right) > LUCK


def test_and_low_for_one_they_would_not():
    rng = random.Random(80)
    rows, seeds = planted(rng, size=4)
    fuser = _fuser(rows)
    left = fuser.pattern(seeds["left"][0], domain="left")
    right = fuser.pattern(seeds["right"][0], domain="right")
    assert fuser.surprise(left, right) <= LUCK


def test_a_shape_too_small_to_compare_is_not_called_surprising():
    fuser = _fuser([])
    from nyxara.njp.fusion import Pattern
    lone = Pattern(seed="a", nodes=("a",), edges=frozenset())
    assert fuser.surprise(lone, lone) == 0.0


def test_the_draws_are_enough_to_tell_the_bar_from_zero():
    assert DRAWS >= 20 and 0.0 < LUCK < 0.5


# --------------------------------------------------------------------------------------------- #
#  the negative control is as easy to match as the positive
# --------------------------------------------------------------------------------------------- #
def test_unrelated_domains_use_one_relation_like_the_planted_ones_do():
    """Drawing each edge from nine relations made a false match nearly impossible, and reported
    a flawless 0.000 at every bar — a floor measured on a softer problem than the finding."""
    rng = random.Random(1)
    rows, _seeds = unrelated(rng, nodes=5, edges=6)
    assert len({relation for _a, relation, _b in rows}) == 1


def test_unrelated_domains_are_the_size_they_are_asked_for():
    rng = random.Random(1)
    rows, seeds = unrelated(rng, nodes=6, edges=7)
    assert len(rows) == 14 and len(seeds) == 2


def test_a_planted_pair_shares_no_word():
    rng = random.Random(1)
    rows, _ = planted(rng, size=4)
    left = {a for a, _r, _b in rows if a.startswith("a")}
    right = {a for a, _r, _b in rows if a.startswith("b")}
    assert left and right and not (left & right)


# --------------------------------------------------------------------------------------------- #
#  the bar that was argued is now measured
# --------------------------------------------------------------------------------------------- #
def test_the_minimum_edge_count_is_swept_rather_than_asserted():
    rows = sweep(40, bars=(1, 2, 3, 4, 5))
    assert [r.min_edges for r in rows] == [1, 2, 3, 4, 5]
    assert all(0.0 <= r.recall <= 1.0 for r in rows)


def test_the_set_bar_still_finds_every_planted_shape():
    assert examine(40, min_edges=MIN_EDGES).recall == 1.0


def test_a_bar_above_the_shape_finds_nothing_which_is_why_it_cannot_be_raised():
    """`MIN_EDGES` cannot close the invented end: past four it rejects the feedback loop too."""
    assert examine(40, min_edges=6).recall == 0.0


def test_worth_is_recall_less_false_alarms_so_firing_on_everything_scores_zero():
    rep = Report(found=10, planted_pairs=10, claimed=10, unrelated_pairs=10)
    assert rep.recall == 1.0 and rep.false_alarms == 1.0 and rep.worth == 0.0


def test_an_empty_report_claims_nothing():
    assert Report().recall == 0.0 and Report().false_alarms == 0.0


def test_the_shapes_swept_are_a_stated_list():
    assert len(SHAPES) >= 5 and (3, 4) in SHAPES


def test_the_store_is_the_smallest_thing_the_finder_will_read():
    fuser = Fusion(Explainer(Store([("a", "causes", "b")])))
    assert fuser.pattern("a").size == 1
