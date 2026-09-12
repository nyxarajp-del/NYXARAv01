"""Phase 3 ⑤ — a goal reached by searching imagined futures, and what would falsify it.

The plan's rule for the phase is that each mechanism ships with the measurement that would take it
back out. For a planner there are two such measurements, and they fail in different ways:

* **A planner that never searches.** If the chosen intervention is the same whatever the target
  is, this is not planning — it is one hardcoded move with a report attached.
  :func:`test_the_target_decides_the_plan` is that test.
* **A plan that is never scored.** ``brain.py:2469``: *"113 predictions registered, 0 scored. A
  prediction that cannot in principle be observed is not a prediction; it is a counter going
  up."* So :attr:`~nyxara.njp.rollout.RolloutPlanner.settled` is the number this module lives or
  dies by, and :func:`test_a_plan_is_settled_by_a_real_reading` asserts it moves.

The rest guard the properties that make the ranking mean something: that reaching beats
approaching, that overshooting does not beat arriving, that the model's own confidence in an
extrapolation is charged for, and that "nothing I can set reaches this" is a conclusion the search
can actually arrive at.
"""

from __future__ import annotations

import pytest

from nyxara.njp.predict import PredictionEngine
from nyxara.njp.rollout import Candidate, RolloutPlanner, Target
from nyxara.njp.universe import InternalUniverse


def _world() -> InternalUniverse:
    """A two-chain universe fitted from readings: water → growth, and heat → rate → yield.

    Variables are dotted — ``plant.water``, not ``water`` — because that is what the pipeline
    actually produces. ``field._sync_world`` names every reading ``f"{kind}.{key}"`` so that six
    watered plants are six *samples* of one relation rather than six universes, and
    :meth:`~nyxara.njp.doing.Goal.named` requires the same shape. A bare variable can be planned
    over directly but can never arrive through a :class:`~nyxara.njp.doing.Goal`, which is worth
    the fixture matching rather than papering over.
    """
    universe = InternalUniverse()
    universe.declare("plant.water", "plant.growth", sign=1)
    universe.declare("oven.heat", "oven.rate", sign=1)
    universe.declare("oven.rate", "oven.yield", sign=1)
    for i in range(8):
        water = float(i)
        universe.observe({"plant.water": water, "plant.growth": 2.0 + 1.5 * water},
                         order=["plant.water", "plant.growth"])
    for i in range(8):
        heat = 10.0 + 5.0 * i
        rate = 1.0 + 0.4 * heat
        universe.observe({"oven.heat": heat, "oven.rate": rate, "oven.yield": 3.0 + 2.0 * rate},
                         order=["oven.heat", "oven.rate", "oven.yield"])
    return universe


def _planner(universe: InternalUniverse, **kw) -> RolloutPlanner:
    return RolloutPlanner(None, universe=universe, predictor=PredictionEngine(), **kw)


# --------------------------------------------------------------------------- #
# The two falsifiers
# --------------------------------------------------------------------------- #
def test_the_target_decides_the_plan():
    """A different goal must produce a different intervention, or nothing is being searched."""
    universe = _world()
    planner = _planner(universe)

    up = planner.search(Target(variable="plant.growth", value=16.0))
    down = planner.search(Target(variable="plant.growth", value=3.0, at_most=True))
    assert up.chosen is not None and down.chosen is not None
    assert up.chosen.setting > down.chosen.setting, (up.why, down.why)

    # And a goal on a different variable reaches for a different lever entirely.
    other = planner.search(Target(variable="oven.yield", value=60.0))
    assert other.chosen is not None
    assert other.chosen.lever != up.chosen.lever, (other.why, up.why)


def test_a_plan_is_settled_by_a_real_reading():
    """The number this module is judged on is `settled`, and only a reading can raise it."""
    universe = _world()
    planner = _planner(universe)
    plan = planner.pursue(Target(variable="plant.growth", value=17.0))
    assert plan.committed
    assert planner.settled == 0, "committing must not score anything by itself"

    settlement = planner.settle({"plant.growth": 16.1})
    assert settlement is not None
    assert settlement.scored
    assert planner.settled == 1
    assert settlement.error is not None and settlement.error > 0.0
    # And it went through the prediction engine too, so the error reaches the shared surprise
    # channel rather than sitting in a counter here.
    assert settlement.surprise is not None
    assert planner.outstanding() is None


def test_a_reading_that_says_nothing_settles_nothing():
    """A claim closed by a reading that could not address it is the counter going up again."""
    universe = _world()
    planner = _planner(universe)
    planner.pursue(Target(variable="plant.growth", value=17.0))
    assert planner.settle({"oven.heat": 30.0}) is None
    assert planner.settled == 0
    assert planner.outstanding() is not None, "the claim must stay open, not be marked scored"


# --------------------------------------------------------------------------- #
# What makes the ranking mean something
# --------------------------------------------------------------------------- #
def test_reaching_beats_approaching():
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.growth", value=17.0))
    assert plan.chosen is not None and plan.chosen.reached
    assert plan.chosen.progress == pytest.approx(1.0)


def test_overshooting_is_worth_no_more_than_arriving():
    """Progress is clipped at the target, or the search selects the most violent move available."""
    target = Target(variable="plant.growth", value=10.0)
    arrives = Candidate(predicted=10.0)
    overshoots = Candidate(predicted=1000.0)
    before, need = 4.0, target.need(4.0)
    for candidate in (arrives, overshoots):
        moved = target.gain(before, candidate.predicted)
        candidate.progress = max(0.0, min(1.0, moved / need))
    assert arrives.progress == overshoots.progress == pytest.approx(1.0)


def test_extrapolation_is_allowed_and_charged_for():
    """A plan past the observed range must be *offered*, and must carry a lower confidence.

    Clipping the sweep at the observed range was the first version of this and it was wrong:
    ``growth >= 12`` needs more water than was ever measured, so the whole search came back with
    three percent of the way as the best plan available while the model, asked directly, answers
    the question. The honesty is paid for by the confidence, not by refusing to look.
    """
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.growth", value=16.0))
    assert plan.chosen is not None
    assert plan.chosen.setting > 7.0, "water was observed over 0…7; reaching 16 needs more"
    inside = [c for c in plan.candidates if c.setting <= 7.0 and c.confidence > 0.0]
    assert inside, "in-range settings must still be considered"
    assert plan.chosen.confidence < max(c.confidence for c in inside)


def test_a_lever_is_never_pushed_through_a_sign_it_has_never_taken():
    """Water was only ever observed at or above zero, so negative water is not a plan."""
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.growth", value=1.0, at_most=True))
    assert plan.candidates
    assert all(c.setting >= 0.0 for c in plan.candidates), \
        [c.setting for c in plan.candidates if c.setting < 0.0]


def test_the_smallest_move_that_works_wins():
    """Among plans that all arrive at the same confidence, the least change is the better plan."""
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.growth", value=8.0, at_most=True))
    assert plan.chosen is not None
    ties = [c for c in plan.candidates
            if c.score == pytest.approx(plan.chosen.score) and c.reached]
    assert plan.chosen.effort == pytest.approx(min(c.effort for c in ties))


def test_a_chain_is_a_plan():
    """A lever two hops upstream counts, or the only lever there is stays off the table."""
    universe = _world()
    planner = _planner(universe)
    target = Target(variable="oven.yield", value=60.0)
    levers = planner.levers(target)
    assert "oven.rate" in levers and "oven.heat" in levers
    assert levers.index("oven.rate") < levers.index("oven.heat"), "the shorter route ranks first"
    assert target.variable not in levers, "setting the target is asserting the answer, not a plan"


def test_no_lever_is_a_finding():
    """"Nothing I can set reaches this" has to be a conclusion the search can arrive at."""
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.water", value=99.0))
    assert plan.chosen is None
    assert planner.no_lever == 1
    assert "that is a finding, not a plan" in plan.why


def test_a_goal_already_met_is_not_planned_for():
    universe = _world()
    planner = _planner(universe)
    plan = planner.search(Target(variable="plant.growth", value=0.0))
    assert plan.chosen is None and plan.considered == 0
    assert "already at" in plan.why
    assert not planner.commit(plan)


# --------------------------------------------------------------------------- #
# The bridge from a goal, and the guard on it
# --------------------------------------------------------------------------- #
def test_a_goal_becomes_a_target_only_when_the_universe_knows_the_variable():
    from nyxara.njp.doing import Goal

    universe = _world()
    assert Target.from_goal(Goal(metric="plant.growth", target=17.0), universe) is not None
    assert Target.from_goal(Goal(metric="concepts.observations", target=12.0), universe) is None
    assert Target.from_goal(Goal(metric="", target=1.0), universe) is None


def test_at_most_survives_the_bridge():
    from nyxara.njp.doing import Goal

    universe = _world()
    target = Target.from_goal(Goal(metric="plant.growth", target=3.0, at_most=True), universe)
    assert target is not None and target.at_most
    assert target.met_by(2.0) and not target.met_by(4.0)


def test_the_agency_plans_for_a_world_goal_instead_of_running_affordances():
    """The affordance table cannot move a number out in the world; the causal model can name how.

    Without this branch such a goal runs all eight affordances, moves nothing eight times, and is
    recorded as impossible — a true statement about her cognitive actions and a false one about
    her.
    """
    from nyxara.njp.doing import CognitiveAgency, Goal

    class _Brain:
        def __init__(self, universe):
            self.universe = universe
            self.rollout = RolloutPlanner(None, universe=universe,
                                          predictor=PredictionEngine())

        def stats(self):
            return {"plant.growth": {}}

    universe = _world()
    brain = _Brain(universe)
    agency = CognitiveAgency()
    attempt = agency.act(brain, Goal(metric="plant.growth", target=17.0, why="probe"))
    assert attempt.action == "plan"
    assert agency.planned_world == 1 and agency.committed_world == 1
    assert brain.rollout.committed == 1
    # And it was not credited into the action table, which records what happened between two
    # readings a moment apart — a shape a plan does not have.
    assert agency.values == {}


# --------------------------------------------------------------------------- #
# Grading, and the reason it is not `reconcile`
# --------------------------------------------------------------------------- #
def test_grade_scores_without_observing():
    """Settling a plan must not fold its reading into the fits a second time."""
    universe = _world()
    before = universe.relations[("plant.water", "plant.growth")].n
    roll = universe.imagine("set plant.water", {"plant.water": 5.0}, steps=1)
    error = universe.grade(roll, {"plant.growth": 9.0})
    assert error is not None and roll.scored
    assert universe.relations[("plant.water", "plant.growth")].n == before, \
        "grade must not observe — reconcile is the method that does both"
    assert universe.reconciled == 1


def test_reconcile_still_observes_and_grades():
    """The refactor must leave the field's existing path exactly as it was."""
    universe = _world()
    before = universe.relations[("plant.water", "plant.growth")].n
    universe.imagine("continue", {"plant.water": 5.0}, steps=1)
    universe.reconcile({"plant.water": 5.0, "plant.growth": 9.0},
                       order=["plant.water", "plant.growth"])
    assert universe.relations[("plant.water", "plant.growth")].n == before + 1
    assert universe.reconciled == 1


def test_grade_returns_none_when_the_reading_addresses_nothing():
    universe = _world()
    roll = universe.imagine("set plant.water", {"plant.water": 5.0}, steps=1)
    assert universe.grade(roll, {"nothing_it_predicted": 1.0}) is None
    assert universe.reconciled == 0


def test_a_planner_with_no_universe_reports_rather_than_raises():
    planner = RolloutPlanner(None, universe=None, predictor=None)
    plan = planner.search(Target(variable="plant.growth", value=1.0))
    assert plan.chosen is None
    assert "no causal model" in plan.why
    assert planner.settle({"plant.growth": 1.0}) is None


def test_counters_survive_a_round_trip():
    universe = _world()
    planner = _planner(universe)
    planner.pursue(Target(variable="plant.growth", value=17.0))
    planner.settle({"plant.growth": 17.0})
    revived = _planner(_world())
    revived.load_dict(planner.to_dict())
    assert revived.settled == planner.settled
    assert revived.committed == planner.committed
