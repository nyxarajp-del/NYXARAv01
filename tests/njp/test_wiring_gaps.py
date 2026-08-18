"""Organs that were built, reachable, and never actually connected to anything.

Each of these was found by measurement rather than by reading, and each has the same shape: a
handle assigned and never read, a relation she could be told and never asked for, a rung that
answered with the very value the question ruled out. None of them looked broken from `stats()`,
which is why they lasted.

They are collected in one file because they are one class of defect, and because the way to keep
them fixed is the same: assert the *connection*, not the organ.
"""

from __future__ import annotations

import pytest

from nyxara.knowledge.base import KnowledgeBase
from nyxara.njp.brain import NJPBrain
from nyxara.njp.core import _exclusions, _strip_exclusions
from nyxara.njp.grounding import GroundedTriple, Grounder


# --------------------------------------------------------------------------- #
# The knowledge handle: assigned by `attach_kernel`, read nowhere
# --------------------------------------------------------------------------- #

@pytest.fixture()
def with_kb():
    brain = NJPBrain()
    kb = KnowledgeBase()
    kb.ingest_text(
        "The Wittig reaction forms an alkene from an aldehyde and a phosphonium ylide.",
        source="chem-notes")
    brain.attach_kernel(knowledge=kb)
    return brain


def test_the_knowledge_base_is_reachable_from_a_turn(with_kb):
    """She foraged the internet and her brain never heard about it.

    `growth/foraging.py` distils verified web findings into the kernel's stores, and
    `attach_kernel` hands the brain a `KnowledgeBase` handle. Across all of `nyxara/njp/` that
    handle was never read, so none of it could reach a turn.
    """
    thought = with_kb.think("what is the wittig reaction?")
    assert "alkene" in (thought.answer or "").lower()
    assert "chem-notes" in (thought.recalled or "")


def test_a_retrieved_passage_never_becomes_a_fact(with_kb):
    """Evidence of relevance is not evidence of truth, and prose she has not parsed is neither."""
    before = dict(with_kb.grounder.facts)
    with_kb.think("what is the wittig reaction?")
    assert dict(with_kb.grounder.facts) == before


def test_the_fact_store_still_outranks_the_knowledge_base(with_kb):
    """It is a last resort. A stated relation must never lose to a retrieved paragraph."""
    with_kb.grounder._assert(GroundedTriple(
        subject="wittig reaction", predicate="is_a", object="olefination",
        confidence=0.9, source="master"))
    thought = with_kb.think("what is the wittig reaction?")
    assert "olefination" in (thought.answer or "").lower()


def test_a_brain_with_no_knowledge_base_is_unaffected():
    brain = NJPBrain()
    assert brain.knowledge is None
    brain.think("what is the wittig reaction?")   # must not raise


# --------------------------------------------------------------------------- #
# Relations she could be told and never asked for
# --------------------------------------------------------------------------- #

@pytest.fixture()
def told():
    grounder = Grounder()
    for subject, predicate, obj in (("sparrow", "capable_of", "fly"),
                                    ("sparrow", "has_property", "feathered"),
                                    ("sparrow", "is_a", "bird")):
        grounder._assert(GroundedTriple(subject=subject, predicate=predicate, object=obj,
                                        confidence=0.7, source="conceptnet"))
    return grounder


@pytest.mark.parametrize("question, predicate, expected", [
    ("what is sparrow capable of?", "capable_of", "fly"),
    ("what can a sparrow do?", "capable_of", "fly"),
    ("sparrow kya kar sakta hai?", "capable_of", "fly"),
    ("what are the properties of sparrow?", "has_property", "feathered"),
])
def test_the_two_relations_a_corpus_supplies_most_of_are_now_askable(
        told, question, predicate, expected):
    """Measured before the fix: `_read_question("what is sparrow capable of?")` returned
    `('sparrow capable of', 'is_a')` — the generic "what is X" swallowed the whole tail, so the
    fact was stored, `_lookup`-able, reasoned over by the Core, and unaskable in English."""
    assert told._read_question(question.lower()) == ("sparrow", predicate)
    assert told.answer(question).text == expected


def test_the_generic_what_is_x_still_works(told):
    """The new patterns sit above it, so the thing to check is that they did not shadow it."""
    assert told.answer("what is sparrow?").text == "bird"


# --------------------------------------------------------------------------- #
# The rung that answered with the value the question ruled out
# --------------------------------------------------------------------------- #

@pytest.fixture()
def chained():
    brain = NJPBrain()
    for sentence in ("aag causes garmi", "garmi causes pasina"):
        brain.think(sentence)
    return brain


def test_a_question_that_excludes_the_stated_answer_gets_the_composed_one(chained):
    """The defect, exactly as it was measured::

        answer("what does aag cause besides garmi?")  ->  'garmi'   direct, 0.9
        reach("aag", "causes")                        ->  'pasina'  composed

    The ladder was right and the reading of the question was wrong.
    """
    got = chained.learner.answer("what does aag cause besides garmi?")
    assert got.answer == "pasina"
    assert got.kind == "composed"


@pytest.mark.parametrize("phrasing", [
    "what does aag cause besides garmi?",
    "what does aag cause other than garmi?",
    "what does aag cause apart from garmi?",
    "what does aag cause except garmi?",
])
def test_every_way_of_saying_not_that_one_is_read(chained, phrasing):
    assert chained.learner.answer(phrasing).answer == "pasina"


def test_without_an_exclusion_the_stated_fact_still_wins(chained):
    """A derivation must never override an observation. The exclusion is the only thing that
    may push past a direct hit, and only because the question said so."""
    got = chained.learner.answer("what does aag cause?")
    assert (got.answer, got.kind) == ("garmi", "direct")


def test_an_excluded_answer_is_not_offered_by_any_rung(chained):
    from nyxara.njp.core import CognitiveLearningCore
    core: CognitiveLearningCore = chained.learner
    assert core.predict("aag", "causes", exclude={"garmi"}).answer != "garmi"


def test_the_exclusion_grammar_reads_the_value_and_not_the_subject():
    assert _exclusions("what does aag cause besides garmi?") == {"garmi"}
    assert _exclusions("what does aag cause?") == set()
    # "instead of" is deliberately not an exclusion: it usually replaces the subject, and reading
    # it as one would silently drop the wrong half of the question.
    assert _exclusions("what does aag cause instead of garmi?") == set()
    assert "besides" not in _strip_exclusions("what does aag cause besides garmi?")


# --------------------------------------------------------------------------- #
# The program library the index could not read
# --------------------------------------------------------------------------- #

def test_the_program_library_is_attached_to_the_brain():
    """`R` was permanently absent from the index with the note "no noesis engine attached",
    while the engine sat fully built one package over."""
    assert NJPBrain().noesis is not None


def test_the_index_reads_the_program_library():
    from nyxara.njp.index import measure

    brain = NJPBrain()
    assert measure(brain, benchmarks=False).value("R") == 0.0
    brain.noesis.run(3)
    after = measure(brain, benchmarks=False)
    assert after.value("R") > 0.0
    assert "abstraction" in after.get("R").detail


def test_the_library_is_held_and_not_driven_by_a_turn():
    """Attached so the index can read it; the WAKE/SLEEP loop stays something run deliberately,
    not something every turn pays for."""
    brain = NJPBrain()
    before = brain.noesis.cycles
    brain.think("aag causes garmi")
    assert brain.noesis.cycles == before
