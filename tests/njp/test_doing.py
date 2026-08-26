"""Stage G: goal → plan → action → outcome, on actions she can actually take.

`agency.success_rate` read zero for the life of the project, and `NJPBrain.pursue` explains why —
correctly::

    "No plan without actions, no actions without a plan. Nothing else in NJP names an action:
     there is no actuator, no action vocabulary, no affordance list — she models what follows
     what, and never what she could *do*."

The answer is not a body. It is that she already performs eight or nine cognitive actions on fixed
cadences — ask, attack, mine, test, consolidate, discover, tune, restructure, reflect — and
nothing ever asked *which of them would move the number that is stuck*. A cadence fires
`consolidate` every eight turns whether or not consolidation is what she needs; that is a
schedule, not agency.

The goal is not invented either: the curriculum already computes it — "only 10 of 12 required
concepts.observations" is a metric, a target and a reason, and it was a sentence in a report.

What the tests guard is mostly the ways this could be faked: `goals_reached` counting actions
instead of outcomes, an untried action never getting a turn, and — the one that matters —
reporting a plan when the honest answer is that nothing she can do moves this.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp import NJPBrain
from nyxara.njp.doing import AFFORDANCES, Affordance, CognitiveAgency, Goal

SESSION = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
           "sparrows need water", "aag lagi", "garmi hui", "pasina aaya",
           "a crow is a bird", "crows need water"]


def _lived(turns: int = 120) -> NJPBrain:
    brain = NJPBrain()
    for turn in range(turns):
        brain.think(SESSION[turn % len(SESSION)])
    return brain


def _agency(*names: str, moves: float = 0.0) -> CognitiveAgency:
    """An agency whose actions are stubs with a known effect, so the *choosing* is what is tested."""
    box = {"value": 0.0}

    def _make(name: str) -> Affordance:
        def _run(_brain, _name=name):
            if _name in names:
                box["value"] += moves
            return True
        return Affordance(_name := name, "stub", _run)

    agency = CognitiveAgency(tuple(_make(n) for n in ("alpha", "beta", "gamma")))
    agency._box = box                                   # type: ignore[attr-defined]
    return agency


class _Stub:
    """A brain that is one number."""

    def __init__(self, box, start: float = 0.0) -> None:
        self._box = box
        self._box["value"] = start

    def stats(self):
        return {"probe": {"value": self._box["value"]}}


# --------------------------------------------------------------------------- #
# the goal comes from the measurement that already existed
# --------------------------------------------------------------------------- #

def test_the_goal_is_the_rung_the_curriculum_says_is_blocking():
    brain = _lived(40)
    goal = CognitiveAgency.goal_from(brain.curriculum.assess(brain))
    assert goal is not None and goal.named
    assert goal.target > 0
    assert goal.why


def test_evidence_before_threshold():
    """A rung that has not been exercised enough cannot be judged on its metric, and driving the
    metric before the samples exist is optimising a number computed from nothing."""
    stage = SimpleNamespace(letter="A", name="prediction", organ="predictive", metric="accuracy",
                            threshold=0.55, sample_organ="predictive", sample_metric="scored",
                            min_samples=20)
    thin = SimpleNamespace(current=SimpleNamespace(stage=stage, enough_evidence=False, why="x"))
    assert CognitiveAgency.goal_from(thin).metric == "predictive.scored"
    ready = SimpleNamespace(current=SimpleNamespace(stage=stage, enough_evidence=True, why="x"))
    assert CognitiveAgency.goal_from(ready).metric == "predictive.accuracy"


def test_a_ladder_with_nothing_left_to_climb_yields_no_goal():
    assert CognitiveAgency.goal_from(SimpleNamespace(current=None)) is None
    agency = CognitiveAgency()
    attempt = agency.act(NJPBrain(), goal=None, report=SimpleNamespace(current=None))
    assert not attempt.ran and agency.no_goal == 1


# --------------------------------------------------------------------------- #
# choosing
# --------------------------------------------------------------------------- #

def test_every_action_gets_a_turn_before_any_is_preferred():
    """`MetaLearner` had to learn this the hard way: taking the first option in insertion order
    lets whichever action was declared first monopolise the metric."""
    agency = _agency("beta", moves=1.0)
    brain = _Stub(agency._box)
    goal = Goal(metric="probe.value", target=99.0)
    chosen = [agency.act(brain, goal=goal).action for _ in range(3)]
    assert sorted(chosen) == ["alpha", "beta", "gamma"]


def test_after_that_she_goes_with_the_record():
    agency = _agency("beta", moves=1.0)
    brain = _Stub(agency._box)
    goal = Goal(metric="probe.value", target=99.0)
    for _ in range(3):
        agency.act(brain, goal=goal)
    assert agency.act(brain, goal=goal).action == "beta"
    assert agency.best_for("probe.value").action == "beta"


def test_nothing_that_works_is_a_finding_not_a_plan():
    """The honest answer, and the one a scheduled no-op would hide. Reporting a plan here is how a
    counter moves off zero without anything having happened."""
    agency = _agency(moves=0.0)                       # nothing moves anything
    brain = _Stub(agency._box)
    goal = Goal(metric="probe.value", target=99.0)
    for _ in range(3):
        agency.act(brain, goal=goal)
    attempt = agency.act(brain, goal=goal)
    assert attempt.action == ""
    assert not attempt.ran
    assert "finding" in attempt.why
    assert agency.acted == 3, "a refused plan must not count as acting"


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #

def test_a_goal_is_reached_by_the_metric_and_never_by_acting():
    agency = _agency("beta", moves=1.0)
    brain = _Stub(agency._box)
    goal = Goal(metric="probe.value", target=2.0)
    reached = [agency.act(brain, goal=goal).reached for _ in range(4)]
    assert agency.acted >= 2
    assert any(reached)
    assert agency.goals_reached == sum(1 for r in reached if r)


def test_an_action_that_raises_is_an_action_that_changed_nothing():
    def _boom(_brain):
        raise RuntimeError("no")

    box = {"value": 0.0}
    agency = CognitiveAgency((Affordance("boom", "stub", _boom),))
    attempt = agency.act(_Stub(box), goal=Goal(metric="probe.value", target=1.0))
    assert attempt.ran and not attempt.reached
    assert agency.acted == 1


def test_a_reduction_is_a_goal_too():
    """Half of what she can achieve on her own is a *reduction*, and a framework that could only
    ask for numbers to go up would have to pretend those were something else."""
    box = {"value": 5.0}

    def _drop(_brain):
        box["value"] -= 1.0
        return True

    agency = CognitiveAgency((Affordance("drop", "stub", _drop),))
    brain = _Stub(box, start=5.0)
    goal = Goal(metric="probe.value", target=3.0, at_most=True)
    for _ in range(3):
        agency.act(brain, goal=goal)
    assert agency.goals_reached >= 1
    assert agency.values[("probe.value", "drop")].mean_delta > 0, "a drop must credit as progress"


# --------------------------------------------------------------------------- #
# on the real brain
# --------------------------------------------------------------------------- #

def test_she_closes_a_deficit_she_can_actually_close():
    """Untested assumptions: §11 says they are the kind that kills a model, and `test` closes
    them. Chosen because it is what measurement supports — most affordances only move their own
    counters, and crediting those would be actions congratulating themselves."""
    brain = _lived(120)
    untested = brain.stats().get("assumptions", {}).get("unknown_unknowns")
    if not untested:
        brain.assumptions.mine(brain.world)
        untested = brain.stats().get("assumptions", {}).get("unknown_unknowns")
    assert untested, "no untested assumptions to close"

    agency = CognitiveAgency()
    goal = agency.deficit_goal(brain)
    assert goal is not None and goal.at_most and goal.target == 0.0
    for _ in range(len(AFFORDANCES) + 2):
        if agency.act(brain, goal=goal).reached:
            break
    assert agency.goals_reached >= 1, agency.stats()


def test_the_loop_pursues_without_being_asked():
    brain = _lived(240)
    assert brain.loop.totals["actions_taken"] > 0
    assert brain.stats()["agency"]["acted"] == brain.doing.acted
    assert brain.pipeline_report()["goal→action"]["state"] == "closed"


def test_stage_g_reads_the_organ_that_can_be_exercised():
    """The curriculum merged `doing` under `agency` rather than over it: the planner over the
    dynamics model is still real and still reported, and what is added is the half that moves."""
    brain = _lived(240)
    block = brain.stats()["agency"]
    assert "known_actions" in block          # the planner is still there
    assert block["acted"] == brain.doing.acted
    assert block["success_rate"] is not None


def test_an_action_may_not_start_another():
    """`reflect` takes a turn and a turn schedules an action. Without the guard that is a loop."""
    brain = _lived(30)
    brain.doing._acting = True
    attempt = brain.doing.act(brain, goal=Goal(metric="predictive.scored", target=999.0))
    assert not attempt.ran
    assert "already acting" in attempt.why
    brain.doing._acting = False


def test_agency_gated_off_is_absent_not_broken():
    brain = NJPBrain(config=SimpleNamespace(doing_enabled=False))
    for line in SESSION[:5]:
        brain.think(line)
    assert brain.doing is None
    assert "doing" not in brain.stats()
    assert "goal→action" not in brain.pipeline_report()


# --------------------------------------------------------------------------- #
# a repeated question is not a less confident one
# --------------------------------------------------------------------------- #

def test_asking_the_same_question_twice_does_not_lose_its_confidence():
    """Pre-existing, and present at the base commit: the second ask read `believed 0.00` while
    the first read `believed 0.448`. That is not a hedge, it is a contradiction — and it feeds
    calibration, the black box and the router, so it is not cosmetic either.

    `_answer_as_taught` returns before anything attaches evidence. The fix prices it by a *check*
    rather than a floor: the same walk over the same facts, and only an answer the walk still
    reaches gets its confidence back.
    """
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    first = brain.think("what does a sparrow need?")
    assert first.answer.strip() == "water" and first.epistemic_confidence > 0.0
    for _ in range(2):
        again = brain.think("what does a sparrow need?")
        assert again.answer.strip() == "water"
        assert again.epistemic_confidence == first.epistemic_confidence


def test_a_memory_the_store_no_longer_supports_is_not_made_confident():
    """The other half. A recalled claim that no longer follows keeps today's behaviour, because
    a floor here would hand confidence to exactly the memories that have gone stale."""
    brain = NJPBrain()
    brain.think("the sky is blue")
    brain.think("what colour is the sky?")
    again = brain.think("what colour is the sky?")
    assert again.epistemic_confidence == 0.0 or again.answer.strip() == ""
