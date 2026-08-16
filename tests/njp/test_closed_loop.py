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
_TAUGHT = (
    "the plant got 10 litres of water and grew 20 cm",
    "the plant got 5 litres of water and grew 10 cm",
    "the plant got 2 litres of water and grew 4 cm",
    "the plant got 8 litres of water and grew 16 cm",
    "the plant got 3 litres of water and grew 6 cm",
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
