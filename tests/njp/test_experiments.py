"""The Active Scientist chain, and the four places it was broken end to end.

``experiments_run`` and ``bits_gained`` were zero over every session ever run, and the reason was
not that experimentation was unimplemented — ``ExperimentDesigner`` designed one on *every turn*.
The chain from a sentence to a settled hypothesis has five links and four of them were cut:

    a sentence reporting a happening   → grounded to nothing (intransitive assertions were
                                          refused for having no object)
    → an event on the timeline         → so ``world.events`` stood at 1 over a session full of them
    → ``world.counterfactual``         → so every effect "has only happened 0 times, too few to
                                          say what it depends on"
    → a settled hypothesis             → so nothing was ever eliminated
    → information gained               → carried per turn and never accumulated, so a session
                                          that gained 0.8652 bits reported 0

Each test below pins one link. The last pins the whole chain, because four separately-correct
links still deliver nothing if the fifth is cut — which is the failure this file exists for.
"""

from __future__ import annotations

import pytest

from nyxara.njp import NJPBrain
from nyxara.njp.grounding import Grounder


def _fire_session(brain: NJPBrain) -> None:
    brain.think("aag se garmi hoti hai")
    brain.think("garmi se pasina hota hai")
    for _ in range(5):
        brain.think("aag lagi")
        brain.think("garmi hui")
        brain.think("pasina aaya")


# --------------------------------------------------------------------------- #
# link 1 — an intransitive assertion is a complete claim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sentence,subject,relation", [
    ("aag lagi", "aag", "lagi"),
    ("garmi hui", "garmi", "hui"),
    ("the fire spread", "fire", "spread"),
    ("the glass broke", "glass", "broke"),
])
def test_a_happening_with_no_object_still_grounds(sentence, subject, relation):
    triples = Grounder().ground(sentence).triples
    assert triples, sentence
    assert (triples[0].subject, triples[0].predicate) == (subject, relation)
    assert triples[0].object == ""


def test_a_transitive_reading_that_lost_its_object_is_still_refused():
    """The intransitive allowance is for one frame, not a general relaxation."""
    from nyxara.njp.semantics import compile_meaning
    assert not compile_meaning("cricket ke baare mein mujhe zyada nahi pata").complete


# --------------------------------------------------------------------------- #
# link 2 — an intransitive relation is a happening by grammar, not by lexicon
# --------------------------------------------------------------------------- #

def test_a_happening_reaches_the_timeline_without_being_in_a_list():
    """`_EVENT_PREDICATES` decides the ambiguous case. It must not decide the unambiguous one.

    "lagi", "hui" and "aaya" are in no list anywhere, and a timeline that only records the
    happenings somebody enumerated is a timeline of one language's verbs.
    """
    brain = NJPBrain()
    _fire_session(brain)
    stats = brain.world.stats()
    assert stats["events"] >= 12, stats
    assert stats["event_kinds"] >= 3, stats


def test_a_stated_fact_does_not_land_on_the_timeline():
    """A relation that *holds* is a fact; only one that *happened* is an event."""
    brain = NJPBrain()
    for _ in range(6):
        brain.think("plants need water")
    assert brain.world.stats()["events"] == 0, brain.world.stats()


# --------------------------------------------------------------------------- #
# link 3 — both ends of a hypothesis are named in the law's vocabulary
# --------------------------------------------------------------------------- #

def test_the_law_and_the_timeline_are_reconciled_at_both_ends():
    """A stated law names things ("aag"); an intransitive event files under its action ("lagi").

    No stem is shared between those two words in any language, so the existing stem rule could
    never bridge them — it is not a near miss, it is a different vocabulary. And mapping only the
    effect leaves the pair half-translated, which is indistinguishable from an unanswerable
    question and was being reported as one.
    """
    brain = NJPBrain()
    _fire_session(brain)
    loop, world = brain.loop, brain.world
    assert loop._as_recorded(world, "aag") == "lagi"
    assert loop._as_recorded(world, "garmi") == "hui"
    verdict = world.counterfactual(loop._as_recorded(world, "aag"),
                                   loop._as_recorded(world, "garmi"))
    assert verdict.answerable, verdict


# --------------------------------------------------------------------------- #
# links 4 and 5 — the hypothesis is settled, and the gain is a session's gain
# --------------------------------------------------------------------------- #

def test_a_designed_experiment_is_actually_run():
    """"Curiosity that computes the informative experiment and never performs it is not
    curiosity; it is a report about curiosity." — the method's own docstring, measured at 0."""
    brain = NJPBrain()
    _fire_session(brain)
    totals = brain.loop.stats()["totals"]
    assert totals["experiments_run"] > 0, totals


def test_information_gained_survives_the_turn_it_was_gained_on():
    """Carried on the report since `_run_experiments` was written, never accumulated.

    A session could gain real bits and ``stats()`` would still answer 0 to "how much did she
    learn by experimenting". A measurement that exists for one turn is not a measurement.
    """
    brain = NJPBrain()
    _fire_session(brain)
    totals = brain.loop.stats()["totals"]
    assert totals["bits_gained"] > 0.0, totals


def test_an_experiment_is_settled_against_the_record_not_against_the_model():
    """The outcome comes from her own event history, never from the fitted universe.

    Grading a hypothesis with the model the hypothesis is about moves every probability while
    nothing is learned.
    """
    brain = NJPBrain()
    _fire_session(brain)
    verdict = brain.world.counterfactual("lagi", "hui")
    # "the effect never happened without the cause" — which supports the arrow.
    assert verdict.answerable and verdict.still_happens is False, verdict


# --------------------------------------------------------------------------- #
# a law she was told, overturned by what she saw
# --------------------------------------------------------------------------- #

def test_a_stated_law_the_record_contradicts_is_refuted():
    """`world.refute_law` had no caller anywhere, so `refuted_laws` was structurally zero.

    A law she was *told* could never be contradicted by anything she *saw* — which is the whole
    of what a world model is for. The verdict comes from occurrences over her own event record,
    never from the fitted model: grading a law with the model the law is about moves every
    probability while nothing is learned.
    """
    brain = NJPBrain()
    brain.think("aag se garmi hoti hai")
    session = (["aag lagi", "garmi hui"] + ["pasina aaya"] * 3 + ["garmi hui"] * 8
               + ["pasina aaya"] * 2 + ["garmi hui"] * 4)
    for line in session:
        brain.think(line)
    verdict = brain.world.counterfactual(brain.loop._as_recorded(brain.world, "aag"),
                                         brain.loop._as_recorded(brain.world, "garmi"))
    assert verdict.still_happens is True, verdict
    assert brain.world.stats()["refuted_laws"] >= 1, brain.world.stats()


def test_a_law_the_record_supports_is_not_refuted():
    """Refutation needs a counterexample, not merely an experiment."""
    brain = NJPBrain()
    brain.think("aag se garmi hoti hai")
    for _ in range(5):
        brain.think("aag lagi")
        brain.think("garmi hui")
    assert brain.world.stats()["refuted_laws"] == 0, brain.world.stats()
