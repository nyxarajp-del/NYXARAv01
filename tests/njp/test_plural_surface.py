"""She is told things in the plural, and that has to reach the same engine the singular does.

The inference machinery was never the problem. Multi-hop inheritance over invented vocabulary
works, and its confidence decays correctly by hop distance::

    zorble is a flimter / flimter is a grex / grex needs vaskin
    what does grex need     -> vaskin  0.50
    what does flimter need  -> vaskin  0.45
    what does zorble need   -> vaskin  0.22

Say the identical three facts in the plural and every one of them extracts nothing, so the same
question has nothing to walk. Five surfaces, measured before this file was written:

    dogs are mammals       -> []          (the is_a backbone, in the plural)
    birds have feathers    -> []          (`has` was covered, `have` was not)
    sparrows can fly       -> []          (capable_of: askable, unwritable)
    elephants are large    -> []          (has_property: askable, unwritable)
    what do sparrows need  -> 'bird'      (not silence — a confident WRONG answer)

The last one is the one that matters, and it is worth being precise about why. It is the failure
`nyxara.njp.canon` was written to eliminate, quoted from its own docstring::

    "a sparrow is a bird" + "birds need water" -> "what does a sparrow need?" -> bird   BEFORE
                                                                             -> water  AFTER

Canon fixed the *store key*, and it works: the singular question answers `water` today. Nothing
fixed the *question surface*. `_QUESTION_PATTERNS` matched `what does` and never `what do`, so the
plural form fell through to the generic `what is X` reading, took the `is_a` edge, and answered
with the kind instead of the kind's property. Same wrong word, same mechanism, one door along.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.canon import canonical_entity
from nyxara.njp.grounding import Grounder


@pytest.fixture()
def grounder() -> Grounder:
    return Grounder()


def _triples(grounder: Grounder, text: str):
    return [t.as_tuple() for t in grounder.ground(text).triples]


# ---- extraction: the plural register ---------------------------------------- #
@pytest.mark.parametrize("plural,singular_form", [
    ("dogs are mammals", "a dog is a mammal"),
    ("sparrows are birds", "a sparrow is a bird"),
])
def test_plural_copula_states_a_kind(grounder: Grounder, plural: str, singular_form: str):
    """`Xs are Ys` is the same claim as `an X is a Y`, and must reach the same edge."""
    got = _triples(grounder, plural)
    assert got, f"{plural!r} extracted nothing"
    subject, predicate, obj = got[0]
    assert predicate == "is_a"
    # The head is folded to its singular so the kind meets the facts held about it. That is the
    # whole point of the fix: a plural that stores its own separate entity has learned nothing.
    assert subject in ("dog", "sparrow")
    assert obj in ("mammal", "bird")


def test_plural_have_is_the_same_relation_as_has(grounder: Grounder):
    """`has` was in the seed table and `have` was not — one token's worth of coverage.

    Asserted on the canonical key rather than the spelling. `canon` is explicit that it rewrites
    the lookup key and leaves the surface as written, so "birds" staying "birds" on the triple is
    the contract working, not a leak; what has to hold is that it *finds* what "a bird has" wrote.
    """
    subject, predicate, obj = _triples(grounder, "birds have feathers")[0]
    assert (canonical_entity(subject), predicate, obj) == ("bird", "owns", "feathers")


@pytest.mark.parametrize("text", ["sparrows can fly", "a sparrow can fly",
                                  "a sparrow is capable of flight"])
def test_capable_of_can_be_written_and_not_only_asked(grounder: Grounder, text: str):
    """The read/write asymmetry, in the direction this file had not met it before.

    `_QUESTION_PATTERNS` has had `capable_of` since the commonsense-corpus work, and reasons in a
    comment about "`sparrow capable_of fly` in the store". No pattern anywhere could put it there.
    A relation she can be asked for and can never be told is the mirror of the gap that comment
    was written to close.
    """
    got = _triples(grounder, text)
    assert got, f"{text!r} extracted nothing"
    assert got[0][1] == "capable_of"
    assert canonical_entity(got[0][0]) == "sparrow"


@pytest.mark.parametrize("text,subject,prop", [
    ("elephants are large", "elephant", "large"),
    ("a bird is small", "bird", "small"),
])
def test_adjectival_predication_is_a_property(grounder: Grounder, text: str,
                                              subject: str, prop: str):
    """`has_property` is askable (`what are the properties of X`) and was unwritable."""
    assert _triples(grounder, text) == [(subject, "has_property", prop)]


def test_a_kind_and_a_property_are_told_apart(grounder: Grounder):
    """`Xs are Ys` and `Xs are <adj>` share a surface and state different things.

    Decided on the object's own plurality through `canon.singular`, not on a word list: a plural
    object names a kind she can inherit through, a bare one names a property she cannot.
    """
    assert _triples(grounder, "dogs are mammals")[0][1] == "is_a"
    assert _triples(grounder, "dogs are loyal")[0][1] == "has_property"


# ---- what the existing extractors already claim, still claimed ---------------- #
@pytest.mark.parametrize("text,predicate", [
    # `_ENUMERATION` and `_COMPOSITION` run before the pattern loop and return early. A new `are`
    # rule that swallowed these would trade one silent failure for a louder one.
    ("the parts of a car are wheels and doors", "has_part"),
    ("the types of mammals are dogs and cats", "has_kind"),
])
def test_enumerations_are_not_swallowed_by_the_new_copula(grounder: Grounder,
                                                          text: str, predicate: str):
    got = _triples(grounder, text)
    assert len(got) == 2
    assert {t[1] for t in got} == {predicate}


def test_the_definite_copula_still_reads_as_a_kind(grounder: Grounder):
    """`X is the Y` had its own rule before this change and keeps it."""
    assert any(t[1] == "is_a" for t in _triples(grounder, "Paris is the capital of France"))


# ---- the question surface, and the wrong answer it produced ------------------- #
def test_the_plural_question_reaches_the_relation_it_names():
    """`what do Xs need` read as `('Master', '')` and answered from the wrong edge entirely."""
    grounder = Grounder()
    subject, predicate = grounder._read_question("what do sparrows need")
    assert predicate == "requires", "read no relation at all — the `does`-only auxiliary"
    assert canonical_entity(subject) == "sparrow"


def test_a_kind_is_never_returned_as_its_own_property():
    """The canon docstring's failure, reproduced through the plural question surface.

    Asserting on `!= "bird"` as well as `== "water"` is deliberate. "It answered nothing" and "it
    answered the kind instead of the property" are different failures, and only one of them is a
    wrong answer being stated to the Master.
    """
    brain = NJPBrain()
    brain.think("sparrows are birds")
    brain.think("birds need water")

    answer = brain.think("what do sparrows need").answer
    assert answer != "bird", "answered with the kind instead of what the kind needs"
    assert answer == "water"


def test_the_singular_question_was_already_right_and_stays_right():
    """The control. If this ever fails the fix has damaged what canon already repaired."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    assert brain.think("what does a sparrow need").answer == "water"


@pytest.mark.parametrize("question", ["what do birds have", "what do sparrows like"])
def test_the_other_plural_question_forms_read_a_relation(question: str):
    """`need` was the one that produced a wrong answer; the whole `do|does` family was broken."""
    subject, predicate = Grounder()._read_question(question)
    assert predicate, f"{question!r} read no relation at all"


# ---- end to end: the inference the plural register could not reach ------------ #
def test_multi_hop_inheritance_works_in_the_plural():
    """The singular form of exactly this scores 1.00 in the seven-stage benchmark.

    Invented vocabulary, so nothing here can be answered from anything but these three sentences.
    """
    brain = NJPBrain()
    for fact in ("zorbles are flimters", "flimters are grexes", "grexes need vaskin"):
        brain.think(fact)
    assert brain.think("what do zorbles need").answer == "vaskin"
