"""A corpus of instances cannot teach the properties of kinds.

Measured on the bundled corpus, twice and independently. Of the 92 distinct kinds she extracts,
**not one** has a single property stated about it. And across every kind with two or more members,
there is **not one** ``(predicate, object)`` pair that two members agree on — 54 species, 31 of
them carrying some superlative, and no two the same superlative.

So the corpus can teach that a blue whale is an animal and can never teach that animals need food.
Without the second, :meth:`nyxara.njp.core.CognitiveLearningCore._inherit` has nothing to inherit
and ``answers_given`` is structurally zero however correct the machinery is — which is exactly
where it sat after every other repair.

The seed is shipped for the same reason ``eval/data/holdout_realworld.jsonl`` is: some evidence
has to come from outside the generator. It enters as testimony, below the confidence a parsed
statement earns, tagged so provenance can always separate it from anything she worked out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.study import (
    DEFAULT_KINDS,
    Corpus,
    DEFAULT_CORPUS,
    Tutor,
    _content,
    _f1,
    seed_kinds,
)


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in Path(DEFAULT_KINDS).read_text().splitlines() if line]


# --------------------------------------------------------------------------- #
# The data itself
# --------------------------------------------------------------------------- #
def test_every_row_is_a_complete_claim(rows):
    assert rows
    for row in rows:
        assert row.get("subject") and row.get("predicate") and row.get("object"), row


def test_no_row_is_stated_twice(rows):
    keys = [(r["subject"], r["predicate"], r["object"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_it_is_about_kinds_rather_than_individuals(rows):
    """A named individual in here would be a fact about the world smuggled in as a taxonomy."""
    subjects = {r["subject"] for r in rows}
    assert all(s == s.lower() for s in subjects), "a capital is the signature of a proper noun"


def test_it_cannot_answer_a_held_out_question(rows):
    """Shipped data that answers the exam is not a seed, it is leakage.

    Two things about how this is measured, both of which were wrong before and both of which
    change what the test actually proves.

    **The exam sample must span the corpus.** This used to read ``limit=3000`` when a limited load
    meant "the first 3000 rows", so it examined the head of a topic-grouped file and called that
    the held-out set. Now that a limited load *samples*, this scans a subset drawn from all 35,693
    pairs — and immediately found a collision the head-only version could never have reached.

    **A predicate is structure, not claim content.** That collision was
    ``language is_used_for communication`` scoring 0.500 against *"The most commonly used type of
    language is English."* The seed does not contain "English" and cannot answer the question; the
    score came from ``language`` plus ``used``, and ``used`` is in the seed only because it is a
    fragment of the relation *name*. Scoring a relation name as though it were content measures
    the grammar of the triple format rather than what the triple claims. So the comparison is
    against subject and object — the claim — which is what "does this seed state the answer"
    actually means.
    """
    held = Corpus.split(Corpus.load(DEFAULT_CORPUS, limit=3000, seed=0))[1][:200]
    worst, worst_pair = 0.0, ("", "")
    for row in rows:
        claim = _content(f"{row['subject']} {row['object']}")
        for pair in held:
            score = _f1(_content(pair.answer), claim)
            if score > worst:
                worst, worst_pair = score, (f"{row['subject']} {row['object']}", pair.answer)
    assert worst < 0.4, (f"a seed fact scores {worst:.3f} against a held-out answer\n"
                         f"  seed:   {worst_pair[0]}\n  answer: {worst_pair[1][:160]}")


# --------------------------------------------------------------------------- #
# How it enters
# --------------------------------------------------------------------------- #
def test_seeding_puts_the_kinds_in_the_fact_store():
    brain = NJPBrain()
    report = seed_kinds(brain)
    assert report.asserted == report.read > 0
    assert report.kinds > 20
    assert brain.grounder._facts_of("animal")


def test_a_seeded_fact_says_where_it_came_from():
    """Provenance is what stops it being mistaken for something she worked out."""
    brain = NJPBrain()
    seed_kinds(brain)
    assert all(t.source == "seed" for t in brain.grounder._facts_of("animal"))


def test_a_kind_claim_is_worth_less_than_a_parsed_statement():
    """Most birds fly, and a penguin is still a bird."""
    brain = NJPBrain()
    seed_kinds(brain)
    brain.think("Ravi lives in Mumbai")
    seeded = max(t.confidence for t in brain.grounder._facts_of("animal"))
    parsed = max(t.confidence for t in brain.grounder._facts_of("ravi"))
    assert seeded < parsed


def test_seeding_writes_no_turns():
    """These are not things anyone said to her; running them as conversation would log a lie."""
    brain = NJPBrain()
    turns = brain.turns
    seed_kinds(brain)
    assert brain.turns == turns


def test_the_concept_layer_sees_the_kinds():
    brain = NJPBrain()
    before = len(brain.genesis.observations)
    seed_kinds(brain)
    assert len(brain.genesis.observations) > before


def test_a_missing_file_changes_nothing():
    brain = NJPBrain()
    report = seed_kinds(brain, path="/nonexistent/kinds.jsonl")
    assert report.asserted == 0


# --------------------------------------------------------------------------- #
# What it is for
# --------------------------------------------------------------------------- #
def test_a_corpus_entity_inherits_from_a_seeded_kind():
    """The whole point: the corpus supplies the member, the seed supplies the property."""
    brain = NJPBrain()
    seed_kinds(brain)
    brain.think("The blue whale is the largest animal on Earth.")
    assert "food" in brain.think("What does the blue whale need?").answer
    assert brain.learner.stats()["derived_composed"] >= 1


def test_without_the_seed_there_is_nothing_to_inherit():
    """Not a hypothetical — this is the state every other repair left it in."""
    brain = NJPBrain()
    brain.think("The blue whale is the largest animal on Earth.")
    assert brain.think("What does the blue whale need?").answer == ""


def test_an_inherited_seed_property_is_defeasible():
    brain = NJPBrain()
    seed_kinds(brain)
    brain.think("The blue whale is the largest animal on Earth.")
    inherited = brain.learner.predict("blue whale", "requires")
    assert inherited.ok
    assert inherited.confidence < 0.8


# --------------------------------------------------------------------------- #
# It has a caller
# --------------------------------------------------------------------------- #
def test_the_tutor_seeds_before_teaching():
    brain = NJPBrain()
    tutor = Tutor(brain)
    pairs = Corpus.load(DEFAULT_CORPUS, limit=40)
    tutor.study(Corpus.split(pairs)[0][:5])
    assert tutor.seeded is not None
    assert tutor.seeded.asserted > 0


def test_seeding_can_be_switched_off_to_measure_the_corpus_alone():
    brain = NJPBrain()
    tutor = Tutor(brain, seed=False)
    pairs = Corpus.load(DEFAULT_CORPUS, limit=40)
    tutor.study(Corpus.split(pairs)[0][:5])
    assert tutor.seeded is None


def test_seeding_happens_once():
    brain = NJPBrain()
    tutor = Tutor(brain)
    pairs = Corpus.split(Corpus.load(DEFAULT_CORPUS, limit=80))[0]
    tutor.study(pairs[:5])
    first = tutor.seeded
    tutor.study(pairs[5:10])
    assert tutor.seeded is first
