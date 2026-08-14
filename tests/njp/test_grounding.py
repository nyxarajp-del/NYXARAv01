"""She understands what she is told, and can be asked about it afterwards.

The transcript this file exists for, as it behaved before grounding:

    "my name is Jay"   -> I do not have anything to say about that yet.
    "what is my name"  -> I do not have anything to say about that yet.

Every word was a blake2b hash of itself. The fabric learned which hashes co-fired, which is real
and was the only thing being learned — nothing survived a turn that could be asked a question.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import SELF_ENTITY, Epistemic, Grounder


@pytest.fixture()
def grounder() -> Grounder:
    return Grounder()


@pytest.fixture()
def taught(grounder: Grounder) -> Grounder:
    for line in ("my name is Jay", "I live in Mumbai",
                 "Ravi lives in Delhi", "Ravi works at Infosys"):
        grounder.ground(line)
    return grounder


# ---- extraction ------------------------------------------------------------ #
def test_a_statement_becomes_a_triple(grounder: Grounder):
    got = grounder.ground("my name is Jay")
    assert got.grounded
    assert got.triples[0].as_tuple() == (SELF_ENTITY, "has_name", "Jay")


def test_first_person_resolves_to_one_entity(grounder: Grounder):
    """"my name" and "I live" must reach the same subject or she stores what she cannot retrieve."""
    grounder.ground("my name is Jay")
    grounder.ground("I live in Mumbai")
    assert {s for s, _p in grounder.facts} == {SELF_ENTITY.lower()}


def test_case_is_preserved(grounder: Grounder):
    """Matching is case-insensitive; storing is not. The Master's name is spelled his way."""
    assert grounder.ground("my name is Jay").triples[0].object == "Jay"


def test_a_third_party_keeps_its_own_subject(taught: Grounder):
    assert taught.answer("where does Ravi live").text == "Delhi"
    assert taught.answer("where do I live").text == "Mumbai"


# ---- answering ------------------------------------------------------------- #
@pytest.mark.parametrize("question,expected", [
    ("what is my name", "Jay"),
    ("who am I", "Jay"),
    ("where do I live", "Mumbai"),
    ("where does Ravi live", "Delhi"),
])
def test_she_answers_from_structure(taught: Grounder, question: str, expected: str):
    got = taught.answer(question)
    assert got.answered and got.text == expected


def test_the_writer_and_the_reader_agree_on_the_edge_name(taught: Grounder):
    """"my name is X" writes has_name; "what is my name" reads name. They must fold together."""
    assert taught.answer("what is my name").answered


def test_what_she_does_not_know_comes_back_unknown(taught: Grounder):
    got = taught.answer("what is my pin code")
    assert got.state == Epistemic.UNKNOWN
    assert not got.answered and got.text == ""


def test_unknown_is_not_a_low_confidence_guess(taught: Grounder):
    """The distinction the whole tri-state exists for."""
    known = taught.answer("what is my name")
    unknown = taught.answer("what is my blood group")
    assert known.state == Epistemic.BELIEVED and known.confidence > 0.0
    assert unknown.state == Epistemic.UNKNOWN and unknown.confidence == 0.0


def test_one_uncorroborated_source_is_believed_never_known(taught: Grounder):
    """Confidence is not corroboration. A single source that could be wrong is a belief."""
    assert taught.answer("what is my name").state == Epistemic.BELIEVED


# ---- contradiction and revision -------------------------------------------- #
def test_a_clashing_fact_is_detected(taught: Grounder):
    got = taught.ground("my name is Raj")
    assert got.contradictions
    prior, now = got.contradictions[0]
    assert (prior.object, now.object) == ("Jay", "Raj")


def test_the_belief_is_actually_revised(taught: Grounder):
    """Detecting a clash and still answering with the superseded fact is the worst outcome."""
    taught.ground("my name is Raj")
    assert taught.answer("what is my name").text == "Raj"


def test_a_contested_answer_loses_confidence(taught: Grounder):
    before = taught.answer("what is my name").confidence
    taught.ground("my name is Raj")
    assert taught.answer("what is my name").confidence < before


def test_a_contested_claim_is_never_known(taught: Grounder):
    taught.ground("my name is Raj")
    assert taught.answer("what is my name").state != Epistemic.KNOWN


def test_the_superseded_belief_is_kept_on_record(taught: Grounder):
    """"he said Jay, then Raj" is a different state from "he said Raj"."""
    taught.ground("my name is Raj")
    history = [(t.object, t.superseded) for t in taught.history(SELF_ENTITY, "name")]
    assert history == [("Jay", True), ("Raj", False)]


def test_a_non_functional_relation_accumulates_rather_than_clashing(grounder: Grounder):
    """Liking two things is not a contradiction. Treating it as one would make growth look like conflict."""
    grounder.ground("Jay likes tea")
    got = grounder.ground("Jay likes coffee")
    assert not got.contradictions


def test_repeating_the_same_fact_is_not_a_contradiction(grounder: Grounder):
    grounder.ground("my name is Jay")
    assert not grounder.ground("my name is Jay").contradictions


# ---- concepts are discovered, not declared --------------------------------- #
def test_types_are_discovered_from_relations():
    """Nothing anywhere says "Jay is a person". The ladder has to find it from what they share."""
    from nyxara.cognition.concept_formation import AbstractionLadder

    ladder = AbstractionLadder()
    grounder = Grounder(ladder=ladder)
    for line in ("my name is Jay", "I live in Mumbai",
                 "Ravi lives in Delhi", "Ravi works at Infosys",
                 "Amit lives in Pune", "Amit works at TCS",
                 "Sara lives in Chennai", "Sara works at Wipro"):
        grounder.ground(line)
    ladder.ascend()

    groups = [set(c.members) for c in ladder._concepts.values()]
    assert any({"Ravi", "Amit", "Sara"} <= g for g in groups), "the people were never grouped"
    assert any({"Delhi", "Pune", "Chennai"} <= g for g in groups), "the places were never grouped"


# ---- learned patterns ------------------------------------------------------ #
def test_a_sentence_the_core_cannot_parse_grounds_to_nothing():
    """The precondition for learning: there has to be a real gap to close."""
    assert not Grounder().ground("cricket ke baare mein mujhe zyada nahi pata").grounded


def test_induction_turns_one_deep_parse_into_a_reusable_pattern():
    """The parser after a thousand turns must not be the parser she shipped with.

    Induction is driven directly here rather than through the fluent surface: that call is
    seconds against a 2.4 GB on-device model, which does not belong in a unit test. What is under
    test is the induction and reuse, not the model.
    """
    from nyxara.njp.grounding import GroundedTriple

    grounder = Grounder()
    sentence = "my lucky number is seven"
    assert not grounder.ground(sentence).grounded, "the core should not already parse this"

    learned = grounder._learn_from(sentence, GroundedTriple(
        subject=SELF_ENTITY, predicate="lucky_number", object="seven", confidence=0.65))
    assert learned, "a successful deep parse must induce a pattern"

    got = grounder.ground("my lucky number is nine")
    assert got.grounded, "the induced pattern was not reused"
    assert got.triples[0].as_tuple() == (SELF_ENTITY, "lucky_number", "nine")
    assert got.triples[0].source == "learned-pattern"


def test_an_induced_pattern_answers_questions_like_any_other():
    from nyxara.njp.grounding import GroundedTriple

    grounder = Grounder()
    grounder._learn_from("my lucky number is seven", GroundedTriple(
        subject=SELF_ENTITY, predicate="lucky_number", object="seven", confidence=0.65))
    grounder.ground("my lucky number is seven")
    assert grounder.answer("what is my lucky_number").text == "seven"


def test_induction_refuses_a_pattern_with_no_context():
    """A regex induced from nothing but the object matches everything. Never learn one."""
    from nyxara.njp.grounding import GroundedTriple

    grounder = Grounder()
    assert not grounder._learn_from("seven", GroundedTriple(
        subject=SELF_ENTITY, predicate="lucky_number", object="seven"))


def test_a_learned_pattern_that_stops_paying_is_pruned():
    """A bad induced regex that matched everything would poison every later turn."""
    grounder = Grounder(pattern_min_trials=2, pattern_floor=0.9)
    bad = type(grounder.patterns[0])(
        regex=r"^(?P<o>.+)$", predicate="", learned=True, hits=0, misses=5)
    grounder.patterns.append(bad)
    grounder.ground("some unrelated sentence")
    assert bad not in grounder.patterns


def test_seed_patterns_are_never_pruned():
    grounder = Grounder(pattern_min_trials=1, pattern_floor=1.0)
    seeds = [p for p in grounder.patterns if not p.learned]
    for _ in range(5):
        grounder.ground("something that matches nothing at all here")
    assert all(p in grounder.patterns for p in seeds)


# ---- persistence ----------------------------------------------------------- #
def test_what_she_learned_survives_a_reload(taught: Grounder):
    fresh = Grounder()
    fresh.load_dict(taught.to_dict())
    assert fresh.answer("what is my name").text == "Jay"


# ---- wired into the brain -------------------------------------------------- #
def test_the_brain_answers_the_transcript_that_used_to_fail():
    brain = NJPBrain()
    brain.think("my name is Jay")
    brain.think("I live in Mumbai")
    assert "Jay" in brain.think("what is my name").answer
    assert "Mumbai" in brain.think("where do I live").answer


def test_the_brain_says_nothing_rather_than_inventing():
    brain = NJPBrain()
    brain.think("my name is Jay")
    thought = brain.think("what is my pin code")
    assert thought.answer == ""
    assert thought.epistemic == Epistemic.UNKNOWN


def test_the_brain_carries_the_epistemic_state_beside_the_claim():
    """The hedge must not be inside the answer — the voice's faithfulness check reads that."""
    brain = NJPBrain()
    brain.think("my name is Jay")
    thought = brain.think("what is my name")
    assert thought.answer == "Jay", "the claim must be bare"
    assert thought.epistemic == Epistemic.BELIEVED
    assert thought.epistemic_confidence > 0.0


def test_grounding_does_not_stop_the_fabric_growing():
    """Structure is added alongside co-activation, never instead of it."""
    brain = NJPBrain()
    before = brain.fabric.n_synapses
    brain.think("my name is Jay")
    assert brain.fabric.n_synapses > before
