"""Two members of a kind have to be able to meet, and a thing belongs to many kinds at once.

Measured over 1,200 corpus pairs: 1,226 facts across 831 subjects, and **12 objects out of 1,226
were also subjects**. For ``is_a`` it was 3 of 443. The fact graph was a star — every subject
pointing at a phrase that nothing else ever mentioned — and objects ran to ten and seventeen
words, because a whole clause was being stored where a kind belonged.

Two organs were starved by that and both looked broken instead. :mod:`nyxara.njp.core` composes
by walking from the object of one edge to the subject of the next, which a star has none of;
after 1,320 turns it had formed 9 schemas and answered **0** questions.
:mod:`nyxara.njp.concepts` needs members that share a property, and no two of those phrases were
ever equal.

The second half is older and worse: ``is_a`` was listed as a *functional* relation, one that holds
a single value. It is not. Sparrow is_a bird and sparrow is_a animal are both true at once — that
is what a taxonomy is — and treating the second as a contradiction of the first meant every
hierarchy retracted itself as fast as it was built.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder, _kind_head


# --------------------------------------------------------------------------- #
# The bare kind, alongside the kind phrase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("phrase, head", [
    ("largest animal on Earth", "animal"),
    ("land animal", "animal"),
    ("species of whale and also the largest animal", "species"),
    ("AI-powered software application designed to simulate humans", "application"),
    ("machine learning technique where a model is reused", "technique"),
])
def test_the_head_of_a_kind_phrase_is_found(phrase, head):
    assert _kind_head(phrase) == head


@pytest.mark.parametrize("phrase", ["of the thing", "the", "in", ""])
def test_a_phrase_with_no_head_yields_nothing(phrase):
    """Better to add no edge than to assert membership of a preposition."""
    assert _kind_head(phrase) == ""


def test_members_of_one_kind_reach_the_same_node():
    """The whole point: three sentences that share no object phrase share a kind."""
    grounder = Grounder()
    for sentence in ("The blue whale is the largest animal on Earth.",
                     "The African elephant is the largest land animal.",
                     "The cheetah is the fastest land animal."):
        grounder.ground(sentence)
    animals = {t.subject.lower()
               for (subject, predicate), triples in grounder.facts.items()
               if predicate == "is_a"
               for t in triples if t.object == "animal" and not t.superseded}
    assert animals == {"blue whale", "african elephant", "cheetah"}


def test_the_full_phrase_is_kept_as_well():
    """The phrase is the more informative claim; the head is the one that connects."""
    grounder = Grounder()
    grounder.ground("The blue whale is the largest animal on Earth.")
    objects = {t.object for triples in grounder.facts.values() for t in triples}
    assert "animal on Earth" in objects
    assert "animal" in objects


def test_a_derived_kind_is_worth_less_than_the_phrase_it_came_from():
    """A parse of a parse, produced by heuristics about English rather than by a statement."""
    grounder = Grounder()
    grounder.ground("The blue whale is the largest animal on Earth.")
    triples = [t for v in grounder.facts.values() for t in v if t.predicate == "is_a"]
    phrase = next(t for t in triples if t.object == "animal on Earth")
    head = next(t for t in triples if t.object == "animal")
    assert head.confidence < phrase.confidence
    assert head.source == "kind-head"


def test_a_single_word_kind_grows_no_extra_edge():
    grounder = Grounder()
    grounder.ground("a sparrow is a bird")
    assert not [t for v in grounder.facts.values() for t in v if t.source == "kind-head"]


def test_only_is_a_grows_a_head():
    """A "head" of a `means` paraphrase is not a kind — asserting one invents a taxonomy."""
    grounder = Grounder()
    grounder.ground("Machine learning refers to algorithms that learn from data")
    assert not [t for v in grounder.facts.values() for t in v if t.source == "kind-head"]


# --------------------------------------------------------------------------- #
# A thing belongs to many kinds at once
# --------------------------------------------------------------------------- #
def test_a_second_kind_does_not_retract_the_first():
    grounder = Grounder()
    grounder.ground("a sparrow is a bird")
    grounder.ground("a sparrow is a animal")
    live = [t.object for t in grounder._facts_of("sparrow") if not t.superseded]
    assert "bird" in live and "animal" in live


def test_a_genuinely_functional_relation_still_revises():
    """The fix is about `is_a` only. One subject still has one name."""
    grounder = Grounder()
    grounder.ground("my name is Jay")
    grounder.ground("my name is Raj")
    assert grounder.answer("what is my name").text == "Raj"


# --------------------------------------------------------------------------- #
# What it was all for: inheritance answers
# --------------------------------------------------------------------------- #
def test_a_property_is_inherited_through_a_derived_kind():
    """`blue whale is_a animal` exists only because the head was extracted from a phrase."""
    brain = NJPBrain()
    for sentence in ("The blue whale is the largest animal on Earth.",
                     "The cheetah is the fastest land animal.",
                     "An animal needs food."):
        brain.think(sentence)
    assert "food" in brain.think("What does blue whale need?").answer
    assert "food" in brain.think("What does cheetah need?").answer


def test_the_core_records_that_it_composed():
    brain = NJPBrain()
    for sentence in ("The blue whale is the largest animal on Earth.", "An animal needs food."):
        brain.think(sentence)
    brain.think("What does blue whale need?")
    stats = brain.learner.stats()
    assert stats["answers_given"] >= 1
    assert stats["derived_composed"] >= 1


def test_an_inherited_answer_is_weaker_than_a_stated_one():
    """It is defeasible, and the confidence has to say so."""
    brain = NJPBrain()
    for sentence in ("The blue whale is the largest animal on Earth.", "An animal needs food."):
        brain.think(sentence)
    inherited = brain.learner.predict("blue whale", "requires")
    stated = brain.learner.predict("animal", "requires")
    assert inherited.ok and stated.ok
    assert inherited.confidence < stated.confidence
