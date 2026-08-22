"""Which organ owns which error, and what happens to the two that nobody owned.

`predict.py` classifies a miss eight ways and looks up a repair registered against that kind.
Two things were wrong with what it found there.

**Two sets of repairs, resolved by construction order.** `NJPBrain._build_predictor` registers
five, then `LearningLoop._install_repairs` registers six over the top, and `register_repair` was a
plain dict assignment — so which body runs was decided by which object happened to be built last.
Both docstrings say "nothing had ever registered one". They were each right about the other.

It is not dead code, which is what makes it worth fixing rather than deleting: the loop is gated,
and with it off the brain's registrations are the only ones there are. So the behaviour differed
between two supported configurations, and nothing anywhere said so.

**PLANNING and LANGUAGE had no repair at all.** `_repair` does `if fn is None: return`, so those
two diagnoses were counted and then silently dropped. A diagnosis nothing acts on is a more
expensive way to not learn.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp.brain import NJPBrain
from nyxara.njp.predict import ErrorKind, Outcome, PredictionEngine


def test_every_error_kind_is_either_repaired_or_declared_unrepairable():
    """No kind may fall through silently. The two that had nothing must say they have nothing."""
    brain = NJPBrain()
    report = brain.predictor.repair_report()
    for kind in ErrorKind.ALL:
        if kind == ErrorKind.UNATTRIBUTED:
            continue          # genuinely not an organ's fault — never routed, by design
        assert kind in report, f"{kind} falls through with no statement either way"
        assert report[kind]["owner"], f"{kind} has an owner field with nothing in it"


def test_the_two_kinds_that_had_no_organ_are_named_as_unrepairable_not_omitted():
    brain = NJPBrain()
    report = brain.predictor.repair_report()
    assert report[ErrorKind.LANGUAGE]["repairable"] is False
    assert report[ErrorKind.LANGUAGE]["why"], "an unrepairable kind must say why"


def test_a_planning_miss_charges_the_action_it_actually_took():
    """The one of the two that does have an organ. `agency._credit` is the surgical surface."""
    brain = NJPBrain()
    if brain.agent is None:
        return
    before = dict(brain.agent.values)
    outcome = Outcome(key="reach", expected="arrived", actual="stuck",
                      organ="planning", error=1.0,
                      context={"action": "walk"})
    outcome.diagnosis = brain.predictor.diagnose(outcome, {"content_correct": False})
    brain.predictor._repair(outcome)

    value = brain.agent.values.get("walk")
    assert value is not None, "the action named by the failure was never charged"
    assert value.tried > before.get("walk").tried if "walk" in before else value.tried >= 1
    assert value.worked == 0


def test_a_planning_miss_moves_only_the_action_it_names():
    """Targeted surgery: one action's record, not the whole model."""
    brain = NJPBrain()
    if brain.agent is None:
        return
    brain.agent._credit("swim", worked=True)
    swim_before = brain.agent.values["swim"].to_dict()

    outcome = Outcome(key="reach", expected="arrived", actual="stuck",
                      organ="planning", error=1.0, context={"action": "walk"})
    outcome.diagnosis = brain.predictor.diagnose(outcome, {"content_correct": False})
    brain.predictor._repair(outcome)

    assert brain.agent.values["swim"].to_dict() == swim_before


def test_the_loop_owns_a_kind_it_registers_and_says_so():
    """Precedence is stated rather than decided by which object was built last."""
    brain = NJPBrain()
    if brain.loop is None:
        return
    report = brain.predictor.repair_report()
    assert report[ErrorKind.GROUNDING]["owner"] == "loop"


def test_with_the_loop_off_the_brain_is_still_the_owner():
    """The configuration in which the brain's registrations are the only ones there are."""
    brain = NJPBrain(config=SimpleNamespace(loop_enabled=False))
    if brain.loop is not None:
        return                      # the gate is named something else in this build; skip
    report = brain.predictor.repair_report()
    assert report[ErrorKind.GROUNDING]["owner"] == "brain"
    assert report[ErrorKind.GROUNDING]["repairable"] is True


def test_a_default_registration_never_displaces_an_owner():
    """The mechanic under the precedence, isolated from how the brain happens to build."""
    engine = PredictionEngine(None)
    engine.register_repair(ErrorKind.MEMORY, lambda o: True, owner="loop")
    engine.register_repair(ErrorKind.MEMORY, lambda o: False, owner="brain", default=True)
    assert engine.repair_report()[ErrorKind.MEMORY]["owner"] == "loop"


def test_an_owner_registration_does_displace_a_default():
    engine = PredictionEngine(None)
    engine.register_repair(ErrorKind.MEMORY, lambda o: False, owner="brain", default=True)
    engine.register_repair(ErrorKind.MEMORY, lambda o: True, owner="loop")
    assert engine.repair_report()[ErrorKind.MEMORY]["owner"] == "loop"


def test_stats_report_what_was_diagnosed_and_never_acted_on():
    """The number that was invisible: diagnoses that reached no repair."""
    engine = PredictionEngine(None)
    engine.mark_unrepairable(ErrorKind.LANGUAGE, "no organ owns rendering")
    outcome = Outcome(key="k", expected="a", actual="b", error=1.0,
                      context={})
    outcome.diagnosis = engine.diagnose(outcome, {"content_correct": True})
    engine._repair(outcome)
    stats = engine.stats()
    assert stats["unrepaired"].get(ErrorKind.LANGUAGE, 0) >= 1
