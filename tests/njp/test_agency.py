"""Stage G — goal → plan → action → outcome, and the nine-rung ladder that gates it.

The tests that matter are the ones a plausible-looking planner would fail: that a plan may only
walk transitions she has actually observed, that a brain which has learned nothing produces no
plan rather than a confident one, and that the curriculum cannot be climbed out of order.
"""

from __future__ import annotations

from nyxara.njp.agency import Agent, Plan
from nyxara.njp.brain import NJPBrain
from nyxara.njp.curriculum import STAGES, Curriculum
from nyxara.njp.predictive import PredictiveWorldModel, WorldState


def _door_world(repeats: int = 6) -> PredictiveWorldModel:
    """closed --open--> open --walk--> outside;  closed --walk--> closed (a real dead end)."""
    model = PredictiveWorldModel()
    for _ in range(repeats):
        model.observe(WorldState.of(["door closed"]), "open",
                      next_state=WorldState.of(["door open"]))
        model.observe(WorldState.of(["door open"]), "walk",
                      next_state=WorldState.of(["outside"]))
        model.observe(WorldState.of(["door closed"]), "walk",
                      next_state=WorldState.of(["door closed"]))
    model.observe(WorldState.of(["door closed"]))
    return model


class _World:
    """A tiny real world for the actuator to move through."""

    def __init__(self) -> None:
        self.state = "door closed"

    def __call__(self, action: str) -> WorldState:
        if self.state == "door closed" and action == "open":
            self.state = "door open"
        elif self.state == "door open" and action == "walk":
            self.state = "outside"
        return WorldState.of([self.state])


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def test_the_action_set_comes_from_experience_not_from_a_list():
    """An agent that can plan with actions it has never seen the consequences of writes fiction."""
    assert Agent(_door_world()).actions() == ["open", "walk"]
    assert Agent(PredictiveWorldModel()).actions() == []


def test_a_brain_that_has_learned_nothing_makes_no_plan():
    plan = Agent(PredictiveWorldModel()).plan(WorldState.of(["outside"]))
    assert not plan.found
    assert "no action has ever been observed" in plan.reason


def test_it_finds_the_real_multi_step_route():
    plan = Agent(_door_world()).plan(WorldState.of(["outside"]))
    assert plan.found
    assert [(s.action, s.expected) for s in plan.steps] == [
        ("open", "door open"), ("walk", "outside")]


def test_a_plan_may_only_walk_transitions_actually_observed():
    """The bug this pins: `predict` smooths, so every outcome has a little invented mass.

    A planner given that distribution routed straight from ``door closed`` to ``outside`` on
    0.077 of smoothing — a one-step plan for a thing that has never once happened. Imagination
    may consider the unobserved; a plan may not.
    """
    model = _door_world()
    smoothed = model.predict(WorldState.of(["door closed"]), "open")
    assert smoothed.probability_of("outside") > 0.0, "fixture: smoothing should offer the edge"
    observed = model.successors(WorldState.of(["door closed"]), "open")
    assert "outside" not in observed
    assert Agent(model).plan(WorldState.of(["outside"])).depth == 2


def test_an_unreachable_goal_is_refused_rather_than_approximated():
    plan = Agent(_door_world()).plan(WorldState.of(["the moon"]))
    assert not plan.found
    assert "no route" in plan.reason


def test_a_long_or_thin_plan_says_it_is_speculative():
    plan = Agent(_door_world()).plan(WorldState.of(["outside"]))
    assert plan.found
    assert plan.speculative == (plan.depth > 2 or plan.confidence < 0.25)


# --------------------------------------------------------------------------- #
# Acting
# --------------------------------------------------------------------------- #
def test_without_hands_a_plan_stays_a_proposal():
    agent = Agent(_door_world())
    outcome = agent.act(agent.plan(WorldState.of(["outside"])))
    assert not outcome.acted
    assert "proposal" in outcome.reason
    assert agent.acted == 0


def test_pursuing_a_goal_reaches_it_and_learns_on_the_way():
    agent = Agent(_door_world())
    outcomes = agent.pursue(WorldState.of(["outside"]), actuator=_World(), steps=4)
    assert outcomes and outcomes[-1].reached_goal
    assert all(o.as_expected for o in outcomes)
    assert agent.stats()["goals_reached"] == 1


def test_an_action_that_raises_is_an_outcome_not_a_crash():
    agent = Agent(_door_world())

    def broken(_action: str) -> WorldState:
        raise RuntimeError("actuator is on fire")

    outcome = agent.act(agent.plan(WorldState.of(["outside"])), actuator=broken)
    assert not outcome.acted and "raised" in outcome.reason
    assert agent.values["open"].tried == 1 and agent.values["open"].worked == 0


def test_credit_lands_on_the_action_that_worked():
    agent = Agent(_door_world())
    agent.pursue(WorldState.of(["outside"]), actuator=_World(), steps=4)
    assert agent.values["open"].worked == 1
    assert agent.values["open"].reliability > 0.5


def test_state_survives_a_round_trip():
    agent = Agent(_door_world())
    agent.pursue(WorldState.of(["outside"]), actuator=_World(), steps=4)
    revived = Agent(_door_world())
    revived.load_dict(agent.to_dict())
    assert revived.goals_reached == agent.goals_reached
    assert revived.values["open"].worked == agent.values["open"].worked


# --------------------------------------------------------------------------- #
# The curriculum
# --------------------------------------------------------------------------- #
def test_every_stage_names_an_organ_metric_that_exists():
    """A stage measured by a number nobody computes is a stage that can never be reached."""
    brain = NJPBrain()
    brain.think("aag se garmi milti hai")
    stats = brain.stats()
    stats["agency"] = brain.agent.stats() if brain.agent else {}
    for stage in STAGES:
        assert stage.organ in stats, f"stage {stage.letter} reads a missing organ {stage.organ}"
        assert stage.metric in stats[stage.organ], \
            f"stage {stage.letter} reads a missing metric {stage.organ}.{stage.metric}"


def test_a_newborn_brain_is_honestly_at_the_bottom():
    report = Curriculum().assess(NJPBrain())
    assert report.depth == 0
    assert report.current is not None and report.current.stage.letter == "A"


def test_depth_counts_only_consecutive_mastery():
    """Mastering G on an absent floor is not depth 7 — it is a number resting on nothing."""
    brain = NJPBrain()
    for line in ("dog is a animal", "cat is a animal", "dog needs food",
                 "cat needs food", "aag se garmi milti hai") * 3:
        brain.think(line)
    report = Curriculum().assess(brain)
    assert report.depth <= len(report.mastered)
    if report.mastered and report.mastered[0] != "A":
        assert report.depth == 0


def test_built_but_silent_is_not_reported_as_switched_off():
    """The distinction this module claims to keep, and got wrong first time."""
    brain = NJPBrain()
    brain.think("aag se garmi milti hai")
    report = Curriculum().assess(brain)
    reasoning = next(r for r in report.results if r.stage.letter == "F")
    assert reasoning.present, "a live MetaReasoner was reported as not built"
    assert reasoning.status != "absent"


def test_the_blocker_is_named_with_its_number():
    brain = NJPBrain()
    brain.think("aag se garmi milti hai")
    report = Curriculum().assess(brain)
    assert report.blocked_by
    # It names the metric standing in the way. A *number* is not always available and saying so
    # is the honest case: after one turn `predictive.accuracy` genuinely has no reading, and
    # inventing a zero there would report a working organ as a failing one.
    current = report.current
    assert current is not None
    assert current.stage.organ in report.blocked_by
    assert current.stage.metric in report.blocked_by


def test_advice_says_what_to_go_and_do():
    brain = NJPBrain()
    advice = Curriculum().advice(brain)
    assert advice.startswith("stage A")
    assert "prediction" in advice


def test_the_brain_exposes_its_own_report_card():
    brain = NJPBrain()
    brain.think("aag se garmi milti hai")
    card = brain.report_card()
    assert card and "depth" in card and len(card["stages"]) == len(STAGES)
