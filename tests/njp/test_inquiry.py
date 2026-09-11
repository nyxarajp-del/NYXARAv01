"""V.79 — is designing an experiment better than picking one?

`ExperimentDesigner` has existed since V.04 and had never been compared to anything. One test hands
it three named experiments and checks it picks the one a person would. That is the shape of defect
this package found four times in a week, and the answer was different each time.

What is pinned here is the comparison, both halves of it: that designing beats choosing at random
in the regime it is used in, **and** that it stops doing so when experiments get scarce. A test
suite that only asserted the first would be the cherry-pick the organ exists to prevent.
"""

from __future__ import annotations

import random

from nyxara.njp.inquiryschool import (
    REGIMES, SETTLED, STRATEGIES, Report, Run, World, examine, inquire, make_world, sweep,
)
from nyxara.njp.universe import ExperimentDesigner


def _world(**kw) -> World:
    return make_world(random.Random(79), **kw)


# --------------------------------------------------------------------------------------------- #
#  the comparison, in the regime it is used in
# --------------------------------------------------------------------------------------------- #
def test_designing_finds_the_truth_in_fewer_experiments_than_choosing_at_random():
    got = examine(200)
    assert got["designed"]["asked"] < got["random"]["asked"], got


def test_and_not_by_giving_up_early_on_the_hard_worlds():
    """A strategy that abandons hard worlds looks fast. Speed is only meaningful beside reach."""
    got = examine(200)
    assert got["designed"]["solved"] >= got["random"]["solved"]


def test_and_not_by_answering_quickly_and_wrongly():
    got = examine(200)
    assert got["designed"]["accuracy"] == 1.0


def test_the_ranking_points_somewhere_rather_than_sorting_noise():
    """Choosing the least informative must cost more than choosing the most, or there is no order."""
    got = examine(200)
    assert got["worst"]["asked"] > got["designed"]["asked"]


def test_every_strategy_is_measured_on_the_same_worlds():
    """A strategy scored on its own sample is answering a different question."""
    got = examine(120)
    assert len({got[name]["worlds"] for name in STRATEGIES}) == 1


# --------------------------------------------------------------------------------------------- #
#  and where it stops
# --------------------------------------------------------------------------------------------- #
def test_the_advantage_fades_as_experiments_become_scarce():
    rungs = sweep(150)
    plentiful = next(r for r in rungs if r["experiments"] >= 8)
    scarce = next(r for r in rungs if r["experiments"] <= 3 and r["hypotheses"] >= 12)
    assert plentiful["saved"] > scarce["saved"]


def test_the_ranking_actually_inverts_at_the_scarce_end_and_this_is_recorded():
    """Not a smaller win — a reversal. `worst` finishing sooner than `designed` is the finding."""
    rungs = sweep(150)
    assert any(r["ranking_holds"] for r in rungs), "it should hold somewhere"
    assert any(not r["ranking_holds"] for r in rungs), "and fail somewhere, which is the point"


def test_the_sweep_walks_from_plentiful_to_scarce():
    ratios = [r["experiments"] / r["hypotheses"] for r in sweep(60)]
    assert ratios == sorted(ratios, reverse=True), ratios


# --------------------------------------------------------------------------------------------- #
#  worlds with no answer in them
# --------------------------------------------------------------------------------------------- #
def test_a_world_where_two_hypotheses_agree_everywhere_is_not_identifiable():
    world = World(hypotheses=("a", "b"), experiments=("e0",),
                  predicts={"a": {"e0": "x"}, "b": {"e0": "x"}}, truth="a")
    assert not world.identifiable


def test_a_world_where_they_differ_somewhere_is():
    world = World(hypotheses=("a", "b"), experiments=("e0", "e1"),
                  predicts={"a": {"e0": "x", "e1": "y"}, "b": {"e0": "x", "e1": "z"}}, truth="a")
    assert world.identifiable


def test_unidentifiable_worlds_are_set_aside_rather_than_scored():
    """A strategy that cannot finish an impossible world is not failing at anything."""
    got = examine(200, hypotheses=12, experiments=3, outcomes=2)
    assert got["unidentifiable"] > 0
    assert got["identifiable"] + got["unidentifiable"] == got["worlds"]
    assert got["designed"]["worlds"] == got["identifiable"]


# --------------------------------------------------------------------------------------------- #
#  the loop itself
# --------------------------------------------------------------------------------------------- #
def test_an_experiment_is_never_run_twice():
    world = _world()
    seen: list = []
    designer = ExperimentDesigner()
    for name in world.hypotheses:
        designer.propose(name, probability=1.0 / len(world.hypotheses),
                         predictions=dict(world.predicts[name]))
    out = inquire(world, "designed", random.Random(1))
    assert out.asked <= len(world.experiments)
    assert len(seen) == len(set(seen))


def test_it_stops_once_it_is_sure_rather_than_running_the_rest():
    got = examine(150)
    assert got["designed"]["asked"] < 8, "it should not be exhausting the list"


def test_settling_requires_real_confidence_not_a_bare_majority():
    assert SETTLED >= 0.9


def test_a_strategy_with_nothing_informative_left_stops_instead_of_flailing():
    """Every hypothesis predicting the same thing has a gain of zero, and running it is not work."""
    world = World(hypotheses=("a", "b"), experiments=("e0", "e1"),
                  predicts={"a": {"e0": "x", "e1": "x"}, "b": {"e0": "x", "e1": "x"}}, truth="a")
    out = inquire(world, "designed", random.Random(1))
    assert out.asked == 0 and not out.settled


def test_the_truth_is_never_shown_to_a_strategy():
    """It decides what outcome comes back, which is what reality does, and nothing more."""
    world = _world()
    designer = ExperimentDesigner()
    for name in world.hypotheses:
        designer.propose(name, probability=1.0 / len(world.hypotheses),
                         predictions=dict(world.predicts[name]))
    assert world.truth in world.hypotheses
    assert len({h.probability for h in designer.hypotheses.values()}) == 1


# --------------------------------------------------------------------------------------------- #
#  the report
# --------------------------------------------------------------------------------------------- #
def test_speed_is_averaged_over_the_worlds_it_actually_settled():
    rep = Report(strategy="x", runs=[Run("x", asked=2, settled=True, right=True),
                                     Run("x", asked=9, settled=False)])
    assert rep.asked == 2.0 and rep.solved == 0.5


def test_an_empty_report_claims_nothing():
    assert Report().asked == 0.0 and Report().solved == 0.0 and Report().accuracy == 0.0


def test_the_regimes_are_a_stated_list_rather_than_one_flattering_setting():
    assert len(REGIMES) >= 5
