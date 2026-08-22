"""One entity, one key — njp/canon.py, and the three defects it closed.

The audit that motivated this file found the reasoning core in good order and the connections into
it broken. Every test here pins one of those connections, and each one is written as the *measured*
failure rather than as a property, because the properties all looked fine while the failures were
live:

* ``birds`` and ``bird`` were two entities, so a correct inheritance walked to a kind holding
  nothing — and the same inference scored 1.00 when the sentence happened to use the singular;
* a lookup miss was answered with a neighbouring fact, so ``"what does a sparrow need?"`` returned
  ``bird`` — confident, sourced, and about the wrong question;
* two equally-supported answers had no state of their own and collapsed into silence.

The last group is the guard that keeps the fix from being worse than the defect: canonicalisation
must not touch Devanagari, must not merge multi-word entities, and must not take away the
*definitional* recall that the retrieval path was built for.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.canon import canonical_entity, canonical_relation, singular
from nyxara.njp.grounding import Epistemic, Grounder


# --------------------------------------------------------------------------- #
# The rule itself
# --------------------------------------------------------------------------- #
def test_a_plural_and_its_singular_are_one_key():
    assert canonical_entity("birds") == canonical_entity("bird") == "bird"


def test_articles_and_case_carry_no_identity():
    for surface in ("a bird", "the Bird", "these birds", "The BIRDS"):
        assert canonical_entity(surface) == "bird", surface


def test_only_the_head_is_folded_so_multiword_entities_stay_distinct():
    """The failure this avoids is worse than the one it fixes: `_kind_head` would collide these."""
    assert canonical_entity("deep learning") == "deep learning"
    assert canonical_entity("machine learning") != canonical_entity("deep learning")


def test_a_singular_word_ending_in_s_is_not_stripped():
    for word in ("physics", "species", "status", "analysis", "news", "gas", "class"):
        assert singular(word) == word, word


def test_irregular_plurals_the_suffix_rules_would_get_wrong():
    assert singular("children") == "child"
    assert singular("leaves") == "leaf"
    assert singular("glasses") == "glass"
    assert singular("boxes") == "box"


def test_short_words_are_never_shortened_into_something_else():
    """"as" is not the plural of "a". A rule that produced it would shred real short names."""
    for word in ("as", "is", "us", "gas"):
        assert len(singular(word)) >= min(2, len(word))


def test_non_latin_is_returned_exactly_as_lowercasing_would_have():
    """The guarantee that made this safe to put on the write path."""
    for surface in ("गरमी", "चिड़िया", "आग", "सपना"):
        assert canonical_entity(surface) == surface.strip().lower(), surface


def test_the_directional_causal_words_are_left_alone():
    """`_PREDICATE_ALIASES` omits them on purpose: folding them forward reverses the question."""
    for word in ("cause", "karan", "wajah"):
        assert canonical_relation(word) == word


def test_participles_reach_the_relation_the_alias_table_already_names():
    assert canonical_relation("needed") == "requires"
    assert canonical_relation("required") == "requires"


def test_an_unknown_relation_keeps_its_own_name():
    assert canonical_relation("zorbulates") == "zorbulates"


# --------------------------------------------------------------------------- #
# Defect 1 — the inheritance chain
# --------------------------------------------------------------------------- #
def test_inheritance_survives_the_plural_the_kind_was_taught_in():
    """The audit's exact reproduction. Before canonicalisation this answered "bird"."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    assert brain.think("what does a sparrow need?").answer == "water"


def test_the_singular_phrasing_that_always_worked_still_does():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("bird needs water")
    assert brain.think("what does a sparrow need?").answer == "water"


def test_the_store_files_both_spellings_under_one_key():
    grounder = Grounder()
    grounder.ground("birds need water")
    assert grounder.answer("what does a bird need?").text == "water"


# --------------------------------------------------------------------------- #
# Defect 2 — a miss is a miss
# --------------------------------------------------------------------------- #
def test_a_question_that_named_its_relation_is_not_answered_from_another_one():
    """The measured defect: `is_a` was returned as the answer to a `requires` question."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    thought = brain.think("what does a sparrow need?")
    assert thought.answer == ""
    assert thought.epistemic == Epistemic.UNKNOWN


def test_the_refusal_is_counted_so_the_gap_is_visible():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("what does a sparrow need?")
    assert brain.grounder.stats()["recall_refusals"] >= 1


def test_a_definitional_question_still_reaches_a_definitional_relation():
    """The case retrieval was built for, and the one the strict rule must not take away."""
    grounder = Grounder()
    grounder.ground("Deep learning is a subset of machine learning that uses neural networks.")
    answer = grounder.answer_by_recall("What is deep learning?")
    assert answer.answered
    assert "machine learning" in answer.text


# --------------------------------------------------------------------------- #
# Defect 3 — a tie is its own state
# --------------------------------------------------------------------------- #
def test_two_equally_supported_causes_are_conflicting_not_silent():
    brain = NJPBrain()
    brain.think("Aag se garmi hoti hai")
    brain.think("Dhoop se garmi hoti hai")
    thought = brain.think("garmi ka karan kya hai?")
    assert thought.epistemic == Epistemic.CONFLICTING
    assert thought.answer == "", "a conflict is not an answer"


def test_a_conflict_carries_both_readings_so_an_experiment_can_be_designed():
    grounder = Grounder()
    grounder.ground("Aag se garmi hoti hai")
    grounder.ground("Dhoop se garmi hoti hai")
    answer = grounder.answer("garmi ka karan kya hai?")
    assert answer.conflicting
    assert len(answer.triples) >= 2


def test_a_contest_that_was_resolved_is_believed_rather_than_conflicting():
    """`_revise` supersedes the loser. That is settled, and refusing to answer it would regress."""
    brain = NJPBrain()
    brain.think("mera naam Jay hai")
    brain.think("mera naam Rahul hai")
    thought = brain.think("mera naam kya hai?")
    assert thought.answer == "Rahul"
    assert thought.epistemic == Epistemic.BELIEVED


# --------------------------------------------------------------------------- #
# The guard: Hinglish must be untouched
# --------------------------------------------------------------------------- #
def test_hinglish_causal_turns_behave_exactly_as_before():
    brain = NJPBrain()
    brain.think("Aag se garmi hoti hai")
    assert brain.think("aag se kya hota hai?").answer == "garmi"


def test_the_masters_name_still_round_trips():
    brain = NJPBrain()
    brain.think("mera naam Jay hai")
    assert brain.think("mera naam kya hai?").answer == "Jay"
