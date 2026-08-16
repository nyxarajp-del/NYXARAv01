"""The three questions she could be told the answer to and could never be asked.

All three were reproduced live on a fresh brain before being fixed, and all three had the same
shape: a mechanism written, tested and reachable, with nothing able to reach it from the sentence
the Master actually types. These tests are the reproductions, kept.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder


# --------------------------------------------------------------------------- #
# 1 · a relation she can be told and could never be asked
# --------------------------------------------------------------------------- #

def _taught():
    brain = NJPBrain()
    for sentence in ("aag se garmi hoti hai", "paani se pyaas bujhti hai",
                     "sardi se paani jamta hai"):
        brain.think(sentence)
    return brain


@pytest.mark.parametrize("question,expected", [
    ("aag se kya hota hai?", "garmi"),
    ("paani se kya hota hai?", "pyaas"),
    ("sardi se kya hota hai?", "paani"),
    ("aag kya karti hai?", "garmi"),
    ("what does aag cause?", "garmi"),
    ("what is the effect of aag?", "garmi"),
    ("what happens because of aag?", "garmi"),
])
def test_a_causal_question_reaches_the_causal_fact(question, expected):
    """Measured: the store had five ways to WRITE `causes` and the question table had none."""
    assert _taught().think(question).answer == expected


@pytest.mark.parametrize("question,expected", [
    ("garmi kyun hoti hai?", "aag"),
    ("pyaas kyun bujhti hai?", "paani"),
    ("what causes garmi?", "aag"),
    ("what is the cause of garmi?", "aag"),
    ("why does garmi happen?", "aag"),
])
def test_the_same_edge_reads_backwards(question, expected):
    """One edge, two ends. Storing every causal fact twice would make contradictions lie."""
    assert _taught().think(question).answer == expected


def test_an_effect_with_no_known_cause_says_so():
    """`aag` causes things and nothing is known to cause it. Silence is the correct answer."""
    assert _taught().think("aag ka karan kya hai?").answer == ""


def test_the_read_and_write_verb_lists_cannot_drift_apart():
    """They are one constant. When they were two, a verb she could be told in was not one she
    could be asked in — and the inflected plural was the half that went missing."""
    grounder = Grounder()
    for sentence in ("dhoop se kapde sukhte hai", "garmi se paani ubalta hai"):
        assert grounder.ground(sentence).triples, f"{sentence!r} did not extract"


# --------------------------------------------------------------------------- #
# 2 · arithmetic — covered in full by test_calculate.py; the reproduction is kept here
# --------------------------------------------------------------------------- #

def test_two_plus_two():
    """The original transcript line. It returned the empty string."""
    assert NJPBrain().think("2+2 kitna hai?").answer == "4"


# --------------------------------------------------------------------------- #
# 3 · a counterfactual over a variable she was told about in Hinglish
# --------------------------------------------------------------------------- #

def _measured():
    brain = NJPBrain()
    for i in range(1, 4):
        brain.think(f"plant got {i * 2} litres of water")
        brain.think(f"the plant grew {i * 10} cm")
    return brain


@pytest.mark.parametrize("question", [
    "What if the water were 3?",
    "what if the water is halved?",
    "what if we halve the water?",
    "what if the water is doubled?",
])
def test_a_counterfactual_gets_a_real_number(question):
    """Not the regex — an empty universe. `_MEASURE_SUBJECT` was English-only, so in Hinglish no
    sentence could report a measurement, `universe.state` stayed empty, and every counterfactual
    answered "no usable arrow leaves the intervened variables"."""
    answer = _measured().think(question).answer
    assert answer, f"{question!r} answered with nothing"
    assert any(character.isdigit() for character in answer)
    assert "confidence" in answer


def test_extrapolating_past_the_observed_range_costs_confidence():
    """The part that makes it honest rather than merely responsive."""
    brain = _measured()
    inside = brain.think("What if the water were 3?").answer
    outside = brain.think("what if the water is doubled?").answer

    def _confidence(text):
        return float(text.rsplit("confidence", 1)[1].strip(" )"))

    assert _confidence(outside) < _confidence(inside)


@pytest.mark.parametrize("sentence,subject", [
    ("paudhe ko 2 litre paani mila", "paudhe"),
    ("pouda 20 cm badha", "pouda"),
    ("paani 3 litre hai", "paani"),
    ("paudhe ne 5 litre paani liya", "paudhe"),
])
def test_a_measurement_in_hinglish_names_an_entity_not_a_sentence(sentence, subject):
    """Hindi is verb-final, so the SVO rule swallowed the quantity into the subject —
    `"paudhe ko 2 litre paani"`, which will never match the same plant twice and quietly makes
    every Hinglish measurement its own single-observation variable."""
    triples = Grounder().ground(sentence).triples
    assert triples, f"{sentence!r} reported no measurement"
    assert triples[0].subject == subject
    assert not any(character.isdigit() for character in triples[0].subject)


def test_a_standing_law_is_not_mistaken_for_a_reading():
    """"pani 100 degree par ubalta hai" is a claim about water, not a measurement of some
    particular water. One word — the locative `par` — separates them."""
    triples = Grounder().ground("pani 100 degree par ubalta hai").triples
    assert triples
    assert triples[0].subject == "pani"
    assert triples[0].predicate == "boils"
