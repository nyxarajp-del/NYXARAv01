"""She holds the entity the question named, under a relation it did not ask for.

``_lookup`` is one dict lookup on ``(subject, predicate)``, and that exactness was the second
measured bottleneck after extraction. "What is X?" always reads as ``is_a``, while the sentence
that taught her about X may have stored ``means``, ``part_of``, ``involves`` or ``occurs_when``.
After 1,200 corpus pairs the store held 521 facts and questions about them answered UNKNOWN.

The danger this path runs toward is named in :mod:`nyxara.njp.relevance`: asked *"How are you?"*
she once returned a verified pendulum law, because it was the most *verified* thing in the store
and nothing asked whether a true thing had anything to do with the question. So the tests here are
mostly about what retrieval must **refuse** — the discipline is that a fact is a candidate only if
the question names its subject, and that a retrieved fact is never stated as known.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Epistemic, Grounder


@pytest.fixture()
def taught() -> Grounder:
    grounder = Grounder()
    for sentence in (
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Overfitting occurs when a model is trained too much on the training data.",
        "Clustering involves grouping similar data points.",
        "The blue whale is the largest animal on Earth.",
        "The two main categories of Artificial Intelligence are Narrow AI and General AI.",
    ):
        grounder.ground(sentence)
    return grounder


# --------------------------------------------------------------------------- #
# What it must find
# --------------------------------------------------------------------------- #
def test_a_neighbouring_relation_answers_the_question(taught):
    """`part_of` answers "what is X" — the fact was there and the predicate key missed it."""
    answer = taught.answer_by_recall("What is deep learning?")
    assert answer.answered
    assert "machine learning" in answer.text


def test_a_definition_by_circumstance_answers_what_is(taught):
    answer = taught.answer_by_recall("What is overfitting?")
    assert answer.answered
    assert "trained too much" in answer.text


def test_an_unusual_phrasing_still_reaches_the_fact(taught):
    """No question pattern reads "tell me about X", and it still names its entity."""
    answer = taught.answer_by_recall("Tell me about deep learning")
    assert answer.answered


# --------------------------------------------------------------------------- #
# What it must refuse — the safety property
# --------------------------------------------------------------------------- #
def test_an_entity_she_was_never_told_about_stays_unknown(taught):
    assert not taught.answer_by_recall("What is quantum chromodynamics?").answered


def test_a_question_naming_nothing_she_knows_stays_unknown(taught):
    """The store is full and none of it is about a pin code."""
    assert not taught.answer_by_recall("What is my pin code?").answered


def test_a_social_turn_reaches_no_fact_however_full_the_store_is(taught):
    """The reproduced failure: a greeting must not be answered from world knowledge."""
    assert not taught.answer_by_recall("How are you?").answered


def test_relatedness_never_substitutes_for_aboutness(taught):
    """"learning" appears in two stored subjects; a bare mention is not naming one."""
    got = taught.recall("Is learning hard?")
    assert all(r.subject_score >= 0.6 for r in got), "every candidate must be squarely named"


def test_the_arrow_is_not_answered_backwards():
    """Asked what causes X, a `causes` edge FROM X answers the opposite question."""
    grounder = Grounder()
    grounder.ground("aag se garmi hoti hai")
    assert not grounder.answer_by_recall("aag ka karan kya hai?").answered


# --------------------------------------------------------------------------- #
# Which relations may answer a question that named no relation
# --------------------------------------------------------------------------- #
# Measured on 200 held-out corpus questions: of 60 retrieved answers, 29 came from `owns` and
# `increases` and 0 of those were right — `owns` scored a mean F1 of 0.003 across 17 answers. They
# were immune to `affinity_floor` because they never reached the affinity table: the question's
# own predicate could not be read, and the uniform-prior branch awarded full weight to a subject
# known by exactly one relation. Restricting that branch to relations that *define* took 58
# answers to 27 and left all 4 correct ones standing.
def test_a_relation_that_does_not_define_cannot_answer_what_is():
    """`Ravi owns a red car` is a fine fact and a useless answer to "what is Ravi"."""
    grounder = Grounder()
    grounder.ground("Ravi has a red car")
    assert not grounder.answer_by_recall("What is Ravi?").answered


def test_the_same_relation_answers_when_it_is_asked_for_by_name():
    """The restriction is on answering an unread question, not on the relation itself."""
    grounder = Grounder()
    grounder.ground("Ravi has a red car")
    answer = grounder.answer_by_recall("What does Ravi own?")
    assert answer.answered
    assert "red car" in answer.text


def test_a_defining_relation_still_answers_an_unread_question(taught):
    """"tell me about X" names no relation, and a definition is what it wants."""
    assert taught.answer_by_recall("Tell me about deep learning").answered


def test_the_retrieval_floor_is_separate_from_the_fact_floor():
    """They gate different decisions: worth keeping, versus answers what was asked."""
    grounder = Grounder()
    assert grounder.recall_floor != grounder.min_confidence
    assert grounder.recall_floor >= 0.6, "set by measurement — see the module comment"


def test_raising_the_floor_removes_answers_and_never_adds_them(taught):
    strict = Grounder()
    strict.facts = taught.facts
    strict.recall_floor = 0.95
    loose = Grounder()
    loose.facts = taught.facts
    loose.recall_floor = 0.2
    questions = ["What is deep learning?", "What is overfitting?", "What is clustering?"]
    answered_strict = {q for q in questions if strict.answer_by_recall(q).answered}
    answered_loose = {q for q in questions if loose.answer_by_recall(q).answered}
    assert answered_strict <= answered_loose


# --------------------------------------------------------------------------- #
# What it must never claim
# --------------------------------------------------------------------------- #
def test_a_retrieved_fact_is_believed_and_never_known(taught):
    """Having found something is not corroboration of it."""
    answer = taught.answer_by_recall("What is deep learning?")
    assert answer.state == Epistemic.BELIEVED
    assert not answer.known


def test_retrieval_can_only_lower_confidence(taught):
    """The score is the fact's own confidence, discounted twice. It cannot exceed it."""
    answer = taught.answer_by_recall("What is deep learning?")
    stored = max(t.confidence for t in taught._facts_of("deep learning"))
    assert 0.0 < answer.confidence <= stored


def test_a_hop_costs_more_than_a_direct_statement(taught):
    got = taught.recall("What is deep learning?", limit=8)
    direct = [r for r in got if r.hops == 0]
    hopped = [r for r in got if r.hops == 1]
    if direct and hopped:
        assert max(r.score for r in direct) > max(r.score for r in hopped)


def test_the_answer_says_it_was_retrieved(taught):
    """A retrieved answer must not be indistinguishable from one stored under the right key."""
    assert "recalled" in taught.answer_by_recall("What is deep learning?").why


def test_a_conditional_fact_carries_its_condition_into_the_reason():
    """Answering a conditional claim without its condition states something false."""
    grounder = Grounder()
    grounder.ground("If the temperature falls below 0 degrees, water freezes.")
    answer = grounder.answer_by_recall("Tell me about water")
    if answer.answered:
        assert "holds when" in answer.why


# --------------------------------------------------------------------------- #
# Where it sits in the turn
# --------------------------------------------------------------------------- #
def test_retrieval_does_not_pre_empt_the_reasoner():
    """A derived answer beats a retrieved one, and must therefore be tried first.

    `sparrow needs water` is inherited through `sparrow is_a bird`. Retrieval would answer the
    flat `sparrow is_a bird` — about the right entity, and a strictly worse answer.
    """
    brain = NJPBrain()
    for sentence in ("a sparrow is a bird", "a bird needs water", "a crow is a bird"):
        brain.think(sentence)
    assert "water" in brain.think("what does sparrow need?").answer


def test_an_exact_relation_still_wins_over_retrieval():
    brain = NJPBrain()
    brain.think("my name is Jay")
    thought = brain.think("what is my name")
    assert thought.answer == "Jay"
    assert not thought.recalled, "an exact hit must not go through retrieval at all"


def test_the_brain_still_says_nothing_rather_than_inventing():
    brain = NJPBrain()
    brain.think("my name is Jay")
    thought = brain.think("what is my pin code")
    assert thought.answer == ""
    assert thought.epistemic == Epistemic.UNKNOWN


# --------------------------------------------------------------------------- #
# The bug retrieval exposed: a statement misread as a question never gets extracted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentence", [
    "Overfitting occurs when a model is trained too much on the training data.",
    "If the temperature falls below 0 degrees, water freezes.",
    "When a model memorises noise, it generalises poorly.",
])
def test_a_conditional_statement_is_not_read_as_a_question(sentence):
    """"when" and "if" join clauses here. The intent reader called all three questions."""
    brain = NJPBrain()
    assert not brain.think(sentence).percept.grounding.is_question, sentence


@pytest.mark.parametrize("sentence", [
    "Overfitting occurs when a model is trained too much on the training data.",
    "If the temperature falls below 0 degrees, water freezes.",
])
def test_a_conditional_statement_reaches_the_extractor(sentence):
    """Separate from the classification above, and deliberately a smaller claim.

    Being read as a statement is necessary for extraction and does not guarantee it: "when a
    model memorises noise, **it generalises poorly**" is classified correctly and still yields
    nothing, because its main clause has a pronoun subject and a verb no pattern lists. That is a
    coverage limit, not a misclassification, and folding the two into one assertion would let a
    verb-list gap read as a regression in the question reader.
    """
    brain = NJPBrain()
    assert brain.think(sentence).percept.grounding.triples, sentence


def test_a_real_question_is_still_a_question():
    brain = NJPBrain()
    for text in ("what is my name", "mera naam kya hai", "tell me my name", "When does rain fall?"):
        assert brain.think(text).percept.grounding.is_question, text
