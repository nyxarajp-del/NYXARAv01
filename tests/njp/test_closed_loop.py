"""Does the loop actually close?

The tests that matter here are the ones a system with all the right organs and none of the wiring
would fail. Every organ these exercise was already written, tested and reachable *in isolation*
before this file existed — what was missing was the caller and the return edge. So none of these
assert that a mechanism exists; they assert that asking a question reaches it, and that what
happened afterwards got back to it.

The numbers are the point. Each assertion is on a counter that was measured at exactly zero over a
real multi-turn session while every underlying algorithm worked perfectly when called by hand.
"""

from __future__ import annotations

import pytest

from nyxara.njp import NJPBrain
from nyxara.njp.grounding import _measurements

# The session every test here teaches from: one entity, two variables, an exact 2× relation and
# no noise. Exactness is deliberate — a fitted slope of 2.0 is a claim that can be checked to
# 1e-6, where "roughly correlated" would pass on a coincidence.
# Ordered so the last reading is the round one: an intervention is resolved against what was
# most recently observed, so the tail of this tuple is what "halve the water" halves.
_TAUGHT = (
    "the plant got 2 litres of water and grew 4 cm",
    "the plant got 5 litres of water and grew 10 cm",
    "the plant got 8 litres of water and grew 16 cm",
    "the plant got 3 litres of water and grew 6 cm",
    "the plant got 10 litres of water and grew 20 cm",
)


def _taught_brain() -> NJPBrain:
    brain = NJPBrain()
    for line in _TAUGHT:
        brain.think(line)
    return brain


# --------------------------------------------------------------------------- #
# Ingestion — the causal engine has to be fed before anything can reach it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentence,expected", [
    ("the plant got 10 litres of water and grew 20 cm", [("water", 10.0), ("growth", 20.0)]),
    ("the plant got 5 litres of water and grew 10 cm", [("water", 5.0), ("growth", 10.0)]),
])
def test_a_measurement_sentence_yields_every_quantity_not_just_the_first(sentence, expected):
    # The whole point of the extractor: two variables observed together are what a coefficient is
    # fitted to, and reading one of the two destroys the pairing rather than halving it.
    assert _measurements(sentence) == expected


def test_a_standing_law_is_not_read_as_a_measurement():
    # "water boils at 100" is a claim about water in general, already read by the threshold
    # pattern. Treating it as an observation of some particular thing would put a law on the
    # simulator's evidence pile.
    brain = NJPBrain()
    triples = brain.think("water boils at 100 degrees").percept.grounding.triples
    assert not any(t.source == "measurement" for t in triples), [t.to_dict() for t in triples]


def test_an_unnameable_quantity_is_dropped_rather_than_stored_under_its_unit():
    # A bare count with no noun, no governing quantity-verb and no unit names no variable. Storing
    # it anyway is how a simulator ends up fitting arrows between things that were never measured.
    assert _measurements("the plant has 10 leaves") == []


def test_numeric_statements_reach_the_causal_engine():
    # Measured before this wiring existed: universe.state was {} and usable_relations() was []
    # after exactly this session, so what_if answered at confidence 0.0 for want of any data.
    brain = _taught_brain()
    relations = brain.universe.usable_relations()
    assert relations, brain.universe.stats()

    arrow = next((r for r in relations
                  if r.cause == "plant.water" and r.effect == "plant.growth"), None)
    assert arrow is not None, [(r.cause, r.effect) for r in relations]
    assert abs(arrow.slope - 2.0) < 1e-6, arrow.slope
    assert arrow.r2 > 0.99, arrow.r2


def test_the_do_operator_answers_from_what_a_conversation_taught():
    # The engine could always do this when handed numbers directly. The claim under test is that
    # a conversation is now a way of handing it numbers.
    brain = _taught_brain()
    out = brain.universe.what_if("plant.water", 1.0)
    assert abs(out.predicted["plant.growth"] - 2.0) < 1e-6, out.predicted
    assert out.confidence > 0.5, out.confidence
    assert not out.reason, out.reason


def test_a_counterfactual_far_outside_the_observed_range_loses_confidence():
    # The honesty property, and the one a plausible-looking simulator fails: reaching further than
    # the data has to cost something, or every extrapolation is stated as firmly as an observation.
    brain = _taught_brain()
    near = brain.universe.what_if("plant.water", 4.0)
    far = brain.universe.what_if("plant.water", 500.0)
    assert far.confidence < near.confidence, (near.confidence, far.confidence)


# --------------------------------------------------------------------------- #
# Routing — the question has to reach the organ that can answer it
# --------------------------------------------------------------------------- #
def test_a_counterfactual_reads_as_a_causal_query_not_a_lookup():
    # `CAUSAL_QUERY` is the only act whose pathways include SIMULATE, and nothing used to produce
    # it for "what if": the `why` cue does not match the phrase and the `what is` cue requires a
    # copula. So the one act that could reach the do-operator was unreachable by construction.
    from nyxara.njp.relevance import SpeechAct, SpeechActReader

    reader = SpeechActReader()
    for question in ("what if I halve the water", "what happens if I remove the water",
                     "agar main paani aadha kar doon"):
        assert reader.read(question).kind == SpeechAct.CAUSAL_QUERY, question


def test_an_intervention_resolves_to_the_variable_the_universe_actually_named():
    # The universe names variables `entity.attribute`; a question says "the water". Reconciling
    # them against observed variables rather than by guessing an entity is what stops her
    # simulating confidently over a subject the Master never mentioned.
    brain = _taught_brain()
    assert brain._counterfactual_context("what if I halve the water") == {
        "variable": "plant.water", "value": 5.0,          # half of the 10 last observed
    }
    assert brain._counterfactual_context("what if I halve the sunlight") == {}


def test_an_intervention_whose_size_is_not_stated_is_declined():
    # "reduce the water" names no magnitude. A factor she invented would come back through the
    # do-operator carrying the same confidence as one the Master gave, and nothing downstream
    # could tell them apart afterwards.
    brain = _taught_brain()
    assert brain._counterfactual_context("what if I reduce the water") == {}


def test_what_if_halving_the_water_gets_a_real_number():
    # The whole slice, end to end. Measured before it: the empty string, from the
    # `if is_question: return ""` in _compose, with a working do-operator two calls away.
    brain = _taught_brain()
    answer = brain.think("what if I halve the water").answer
    assert answer, "a counterfactual over an observed variable answered with nothing"
    assert "plant.growth" in answer, answer
    assert "10" in answer, answer                         # 20 → 10, the halved growth
    assert "confidence" in answer.lower(), answer


def test_a_counterfactual_that_changes_nothing_is_not_an_answer():
    # Setting a variable to the value it already holds entails no consequence, and reporting the
    # premise back as though it were a prediction is the counterfactual form of an echo.
    brain = _taught_brain()
    assert brain.think("what if the water were 10").answer == ""


def test_the_answer_states_the_confidence_the_do_operator_earned():
    # Not that a number is present, but that it *moves the right way*: a question inside the
    # observed range must come back more confident than one reaching well past it. This is the
    # assertion a template with a hard-coded confidence would fail.
    brain = _taught_brain()
    inside = brain.think("what if the water were 4").answer          # within the 2..10 observed
    outside = brain.think("what if the water were 90").answer        # far beyond it
    assert inside and outside, (inside, outside)

    def _confidence(text: str) -> float:
        return float(text.rsplit("confidence", 1)[1].strip(" )"))

    assert _confidence(inside) > _confidence(outside), (inside, outside)


def test_a_counterfactual_over_a_variable_never_observed_is_declined():
    # The honest failure. She has never measured sunlight, so there is no arrow to push on, and
    # inventing one is the single worst thing a simulator can do.
    brain = _taught_brain()
    assert brain.think("what if I halve the sunlight").answer == ""


def test_a_strategy_that_produces_nothing_yields_to_one_that_can():
    # `causal` (explanation) and `simulate` (intervention) are both eligible for a causal problem,
    # and `causal` has the higher prior. It produces nothing on an intervention — explanation is
    # not intervention — and abstaining there meant the organ that owns the question was never
    # asked. The retry is what makes the registry's second-best reachable.
    brain = _taught_brain()
    context = dict(brain._counterfactual_context("what if I halve the water"),
                   grounded=False, subject="", about_self=False)
    solution = brain.metareason.solve("what if I halve the water", context=context)
    assert solution.answered, solution.to_dict()
    assert solution.strategy == "simulate", solution.to_dict()
    assert "causal" in solution.attempts, solution.attempts


def test_an_ordinary_empirical_question_is_not_dragged_into_the_causal_path():
    # The regression guard for the classifier change. The causal boost is scored off the parsed
    # intervention, not off another keyword, so a question that names no intervention she can
    # perform is classified exactly as it was before.
    from nyxara.njp.metareason import ProblemKind, ProblemClassifier

    classifier = ProblemClassifier()
    plain = classifier.classify("what is the boiling point of mercury", context={"grounded": False})
    assert plain.kind == ProblemKind.EMPIRICAL, plain.to_dict()


# --------------------------------------------------------------------------- #
# The return edge — an outcome has to reach what staked a claim on it
# --------------------------------------------------------------------------- #
def _graded_brain(names=("Ravi", "Sita", "Amit", "Neha")) -> NJPBrain:
    """Ask before teaching, so the answer is deliberated and a later fact can grade it.

    The order is the whole point: she answers on one turn and the Master states the fact on a
    later one, so what does the grading is independent of what is being graded.
    """
    brain = NJPBrain()
    for name in names:
        brain.think(f"where does {name} live")
        brain.think(f"{name} lives in Delhi")
    return brain


def test_a_strategy_is_graded_by_reality_and_not_only_by_its_own_critic():
    # MetaReasoner.outcome exists to override the critic's provisional credit — its docstring
    # says so — and nothing had ever called it, so strategy selection was trained entirely on
    # the opinion of the critic that had just approved the answer.
    brain = _graded_brain()
    assert brain.loop.totals["strategies_graded"] > 0, brain.loop.totals


def test_an_answer_she_asserts_is_staked_as_a_belief_that_can_be_found_wrong():
    # The ledger only ever held the Master's testimony, so the one class of claim that could be
    # checked against an independent later fact was the class never entered.
    brain = NJPBrain()
    brain.think("Ravi lives in Pune")
    brain.think("where does Ravi live")
    held = [b for b in brain.beliefs.known() if "pune" in b.claim.lower()]
    assert held, [b.claim for b in brain.beliefs.known()]
    assert held[0].falsifier, held[0].to_dict()


def test_reliability_becomes_measurable_so_temper_stops_being_a_no_op():
    # The sharpest cascade in the audit: settle/retract were never called, so `_outcomes` stayed
    # empty, `reliability()` always reported zero samples, and `temper()` returned its input
    # unchanged for ever. An entire calibration path, written and tested and inert.
    brain = NJPBrain()
    for name in ("Ravi", "Sita", "Amit", "Neha", "Vikram", "Priya", "Arjun"):
        brain.think(f"{name} lives in Pune")
        brain.think(f"where does {name} live")
        brain.think(f"{name} lives in Delhi")       # the Master says otherwise

    reliability = brain.beliefs.reliability("located_in")
    assert reliability.samples >= 5, reliability.to_dict()
    assert brain.beliefs.temper(0.9, "located_in") < 0.9, reliability.to_dict()


def test_a_belief_is_retracted_rather_than_deleted():
    # Quarantine, not deletion. Driving a belief to zero and keeping it is what lets "this has
    # failed before" stay answerable; dropping the record makes the same mistake available again.
    brain = NJPBrain()
    brain.beliefs.hold("the sky is green", confidence=0.8, domain="colour",
                       falsifier="the sky is observed to be another colour")
    assert brain.beliefs.retract("the sky is green", why="contradicted by observation")
    assert brain.beliefs.stats()["retracted"] == 1

    # The tombstone: still answerable, still carrying its history, at zero confidence.
    tombstone = brain.beliefs.why("the sky is green")
    assert tombstone, "a retracted belief was deleted rather than quarantined"
    assert tombstone["confidence"] == 0.0, tombstone
    assert tombstone["history"], tombstone


def test_a_greeting_still_cannot_reach_physics():
    # The failure `relevance.py` was written for, re-checked after making a new pathway live: a
    # greeting must not acquire a route to the world model just because one now exists.
    brain = _taught_brain()
    answer = brain.think("hello NYXARA").answer
    assert "plant" not in answer.lower(), answer
    assert "confidence" not in answer.lower(), answer
