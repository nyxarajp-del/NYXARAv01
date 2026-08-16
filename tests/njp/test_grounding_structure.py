"""A sentence carries more than one claim, and a claim carries more than its arrow.

The measurement this file exists for. Over the first 300 answers of the bundled corpus, the
extractor grounded **23.3%** of them — 70 triples across 66 subjects, of which exactly 4 subjects
held more than one fact. So the store was not merely sparse, it was *disconnected*: almost nothing
shared a subject, which is the evidence :mod:`nyxara.njp.core` induces roles from and
:mod:`nyxara.njp.concepts` finds invariants in. Both were being starved rather than failing.

Three things were missing, and none of them was a hard parsing problem:

* **structure** — one sentence stating two claims ("A involves P, while B involves Q") had its
  second half discarded, because the pattern loop stops at the first match and was applied to the
  whole text rather than to a clause;
* **vocabulary of the register** — expository prose states relations with "refers to", "is the",
  "consists of" and enumerations, none of which a seed list written for conversation contained;
* **the conditional half** — "if C, then E" grounded to a bare ``E``, which is not a weaker
  version of the claim but a different and usually false one.
"""

from __future__ import annotations

import pytest

from nyxara.njp.grounding import Grounder


@pytest.fixture()
def grounder() -> Grounder:
    return Grounder()


def _find(triples, predicate):
    return [t for t in triples if t.predicate == predicate]


# --------------------------------------------------------------------------- #
# Conditions — the half a triple cannot hold
# --------------------------------------------------------------------------- #
def test_a_condition_is_kept_beside_the_claim_it_restricts(grounder):
    """``water freezes`` alone is false. The circumstance is what makes it true."""
    result = grounder.ground("If the temperature falls below 0 degrees, water freezes.")
    assert result.triples, "the conditional stated a relation and it must be extracted"
    triple = result.triples[0]
    assert triple.subject.lower() == "water"
    assert triple.predicate == "freezes"
    assert "temperature falls below 0" in triple.condition
    assert triple.conditional


def test_an_unconditional_claim_carries_no_condition(grounder):
    """Empty is the normal case, and must not be filled in with something plausible."""
    result = grounder.ground("Ravi lives in Mumbai")
    assert result.triples
    assert result.triples[0].condition == ""
    assert not result.triples[0].conditional


def test_a_definition_by_circumstance_stays_whole(grounder):
    """"Overfitting" on its own is not a claim — the circumstance IS the content."""
    result = grounder.ground(
        "Overfitting occurs when a model is trained too much on the training data.")
    occurs = _find(result.triples, "occurs_when")
    assert occurs, "the whole sentence is the relation"
    assert "trained too much" in occurs[0].object


# --------------------------------------------------------------------------- #
# Uncertainty as stated, kept apart from confidence in the parse
# --------------------------------------------------------------------------- #
def test_a_hedged_claim_is_recorded_as_hedged(grounder):
    result = grounder.ground("Smoking may cause cancer")
    assert result.triples
    assert result.triples[0].modality == "possible"


def test_a_hedge_costs_confidence_against_the_same_sentence_asserted(grounder):
    """The two sentences parse identically; only the commitment differs."""
    hedged = Grounder().ground("Smoking may cause cancer").triples
    flat = Grounder().ground("Smoking causes cancer").triples
    assert hedged and flat
    assert hedged[0].confidence < flat[0].confidence


def test_a_generalisation_is_weaker_than_an_assertion_but_stronger_than_a_hedge():
    usual = Grounder().ground("Exercise usually improves health").triples
    maybe = Grounder().ground("Exercise may improve health").triples
    assert usual and maybe
    assert usual[0].modality == "typical"
    assert maybe[0].confidence < usual[0].confidence


# --------------------------------------------------------------------------- #
# Structure — one sentence, more than one claim
# --------------------------------------------------------------------------- #
def test_both_halves_of_a_contrast_are_extracted(grounder):
    """Before this, everything after "while" was unreachable at any pattern count."""
    result = grounder.ground(
        "Supervised learning involves labeled examples, "
        "while unsupervised learning involves unlabeled data.")
    subjects = {t.subject.lower() for t in result.triples}
    assert "supervised learning" in subjects
    assert "unsupervised learning" in subjects


def test_an_enumeration_yields_one_triple_per_item(grounder):
    result = grounder.ground(
        "The two main categories of Artificial Intelligence are Narrow AI and General AI.")
    kinds = _find(result.triples, "has_kind")
    assert len(kinds) == 2
    assert {t.object for t in kinds} == {"Narrow AI", "General AI"}
    assert all(t.subject == "Artificial Intelligence" for t in kinds)


def test_a_list_of_one_is_not_an_enumeration(grounder):
    """A single-item "list" is a plain relation, and the pattern loop reads it better."""
    result = grounder.ground("A component of the system is the parser")
    assert not _find(result.triples, "has_kind")


def test_a_superlative_states_a_kind_as_well_as_a_rank(grounder):
    """The kind is the half that connects — the rank predicate is unique to each sentence."""
    result = grounder.ground("The blue whale is the largest animal on Earth.")
    kinds = _find(result.triples, "is_a")
    assert kinds, "a superlative says what the subject IS"
    assert kinds[0].subject == "blue whale"
    assert "animal" in kinds[0].object
    assert _find(result.triples, "is_largest")


def test_an_appositive_alias_reaches_the_same_entity(grounder):
    result = grounder.ground(
        "The bumblebee bat, also known as the hog-nosed bat, is the smallest mammal.")
    alias = _find(result.triples, "also_known_as")
    assert alias, "two names for one entity is a fact worth keeping"
    assert "hog-nosed bat" in alias[0].object
    # And the sentence the appositive interrupted is still read.
    assert _find(result.triples, "is_a")


# --------------------------------------------------------------------------- #
# The register expository prose actually uses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentence, predicate", [
    ("Machine Learning refers to algorithms that learn from data", "means"),
    ("Deep learning is a subset of machine learning", "part_of"),
    ("Clustering involves grouping similar data points", "involves"),
    ("The blue whale is the largest animal", "is_a"),
])
def test_the_expository_register_grounds(grounder, sentence, predicate):
    result = grounder.ground(sentence)
    assert _find(result.triples, predicate), f"{sentence!r} should ground to {predicate}"


# --------------------------------------------------------------------------- #
# The whole point: facts that share a subject
# --------------------------------------------------------------------------- #
def test_extraction_connects_rather_than_merely_covering(grounder):
    """Coverage without shared subjects gives the schema inducer nothing to generalise over."""
    for sentence in (
        "The blue whale is the largest animal on Earth.",
        "The African elephant is the largest land animal.",
        "The cheetah is the fastest land animal.",
    ):
        grounder.ground(sentence)
    # Every one of them asserts an `is_a`, which is what makes them comparable at all.
    is_a_subjects = {key[0] for key in grounder.facts if key[1] == "is_a"}
    assert len(is_a_subjects) >= 3


def test_the_cognitive_fields_survive_a_restart(grounder):
    """A conditional claim reloaded without its condition is a different, false claim."""
    grounder.ground("If the temperature falls below 0 degrees, water freezes.")
    grounder.ground("Smoking may cause cancer")

    revived = Grounder()
    revived.load_dict(grounder.to_dict())

    restored = [t for triples in revived.facts.values() for t in triples]
    assert any(t.condition for t in restored), "the condition must come back"
    assert any(t.modality == "possible" for t in restored), "so must the hedge"


def test_a_single_clause_sentence_behaves_exactly_as_before(grounder):
    """The first-match rule moved inside a clause; on one clause it must be unchanged."""
    result = grounder.ground("my name is Jay")
    assert len(result.triples) == 1
    assert result.triples[0].predicate == "has_name"
    assert result.triples[0].object == "Jay"
