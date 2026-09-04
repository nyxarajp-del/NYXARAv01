"""The question form, induced rather than listed (NJP V.38).

The claim is that this module holds no phrasings. Most of these tests are ways of trying to catch
it holding one.
"""
from __future__ import annotations

import pytest

from nyxara.njp.asking import LESSON, Asking, install, teach, uninstall


@pytest.fixture
def taught():
    return teach()


# --------------------------------------------------------------------------- #
# Empty until shown something
# --------------------------------------------------------------------------- #
def test_a_new_learner_reads_nothing():
    blank = Asking()
    for question in ("what brings about thunder?", "why does rain happen?",
                     "how do you bake bread?", "what is a mammal?"):
        assert blank.read(question) is None
    assert blank.to_dict()["settled"] == 0


def test_one_demonstration_settles_nothing():
    """One demonstration cannot separate the cue from the topic."""
    one = Asking()
    one.show("what brings about thunder?", topic="thunder", walk="because")
    assert one.read("what brings about rain?") is None
    one.show("what brings about rain?", topic="rain", walk="because")
    assert one.read("what brings about hail?") == ("because", ("hail",))


def test_the_same_question_twice_is_still_one_filler():
    two = Asking()
    for _ in range(4):
        two.show("what brings about thunder?", topic="thunder", walk="because")
    assert two.read("what brings about rain?") is None


# --------------------------------------------------------------------------- #
# What it learned, and what it refuses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,walk", [
    ("what brings about thunder?", "because"),
    ("what gives rise to rain?", "because"),
    ("what leads to erosion?", "because"),
    ("what is responsible for inflation?", "because"),
    ("how come rain happens?", "because"),
    ("what sets off an alarm?", "because"),
    ("what is a fuse there for?", "for"),
    ("by what means does a pump operate?", "mechanism"),
    ("what is the recipe for bread?", "procedure"),
])
def test_a_taught_form_reads_on_a_topic_nobody_demonstrated(taught, question, walk):
    got = taught.read(question)
    assert got is not None and got[0] == walk


@pytest.mark.parametrize("question", [
    "what is a mammal?", "who discovered penicillin?", "where is paris?",
    "when did the war happen?", "is copper a metal?", "how many bones are there?",
    "who wrote hamlet?", "which planet is largest?", "what is the capital of france?",
])
def test_a_question_that_is_not_one_of_these_reads_as_none(taught, question):
    assert taught.read(question) is None


@pytest.mark.parametrize("question", [
    "what accounts for inflation?", "what lies behind erosion?",
    "on account of what does rain occur?", "through what process does rain arise?",
])
def test_a_cue_nobody_demonstrated_is_not_guessed(taught, question):
    """The honest ceiling. She can learn these from two demonstrations and not before."""
    assert taught.read(question) is None


def test_a_cue_can_be_learned_at_any_time(taught):
    assert taught.read("what accounts for inflation?") is None
    taught.show("what accounts for zorbit?", topic="zorbit", walk="because")
    taught.show("what accounts for glimf?", topic="glimf", walk="because")
    assert taught.read("what accounts for inflation?") == ("because", ("inflation",))


# --------------------------------------------------------------------------- #
# The two defects the measurement found
# --------------------------------------------------------------------------- #
def test_a_cue_seen_in_one_surface_is_not_read_at_the_general_level():
    """`what is {x} there for?` leaves the cue `there`, which read "how many bones are there?"."""
    one = Asking()
    one.show("what is glorp there for?", topic="glorp", walk="for")
    one.show("what is vokith there for?", topic="vokith", walk="for")
    general = [c for c in one.cues.values() if c.level == "cue" and c.key == ("there",)]
    assert general and general[0].generalises is False and general[0].settled is None
    # And the specific levels still carry the form it was actually taught in.
    assert one.read("what is a fuse there for?") == ("for", ("fuse",))


def test_the_shape_of_a_form_is_kept_not_merely_which_sides_had_words():
    """`2 ▫ 2` and `1 ▫ 1` are different forms; with booleans alone they matched."""
    one = Asking()
    one.show("what is glorp there for?", topic="glorp", walk="for")
    one.show("what is vokith there for?", topic="vokith", walk="for")
    assert one.read("how many bones are there?") is None


def test_a_contested_cue_is_never_read(taught):
    """Demonstrated with two answers is demonstrated with neither."""
    contested = {c.key for c in taught.contested()}
    assert ("job",) in contested
    for cue in taught.contested():
        assert cue.settled is None


def test_a_negative_contests_a_cue_it_shares_with_a_positive():
    one = Asking()
    one.show("what job does glorp do?", topic="glorp", walk="for")
    one.show("what job does vokith do?", topic="vokith", walk="for")
    before = [c for c in one.cues.values() if c.level == "cue" and c.key == ("job",)]
    assert before and before[0].contested is False
    one.show("whose job is glorp?", topic="glorp", walk="")
    one.show("whose job is vokith?", topic="vokith", walk="")
    after = [c for c in one.cues.values() if c.level == "cue" and c.key == ("job",)]
    assert after[0].contested is True and after[0].settled is None


def test_the_negatives_in_the_shipped_lesson_change_no_reading():
    """Measured rather than claimed: this lesson's negatives are a guard that has not fired."""
    positives = tuple(row for row in LESSON if row[1])
    with_neg, without = teach(), teach(lesson=positives)
    probes = ["what is a mammal?", "who discovered penicillin?", "how many bones are there?",
              "what brings about thunder?", "what job does a fuse do?", "where is paris?"]
    assert [with_neg.read(q) for q in probes] == [without.read(q) for q in probes]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_installing_extends_the_reader_and_never_overrides_it():
    from nyxara.njp.explainread import read_explanation_question as read

    uninstall()
    try:
        assert read("what brings about thunder?") is None
        table = read("what is a stethoscope for?")
        install()
        # what the table knew, it still answers, identically
        assert read("what is a stethoscope for?") == table
        # and what it could not reach is now reachable
        assert read("what brings about thunder?") == ("because", ("thunder",))
        # and a question that is neither stays neither
        assert read("what is a mammal?") is None
    finally:
        uninstall()


def test_a_brain_teaches_the_forms_when_it_is_built():
    from nyxara.njp import explainread
    from nyxara.njp.brain import NJPBrain

    uninstall()
    try:
        NJPBrain()
        assert explainread.LEARNED is not None
        assert explainread.read_explanation_question("what leads to erosion?") is not None
    finally:
        uninstall()


def test_the_lesson_does_not_cover_the_exam():
    """A lesson that grew to cover the gauntlet would make teaching look like reasoning."""
    from nyxara.njp.explaingauntlet import WHY_FORMS, Gauntlet

    covered, untaught = Gauntlet._covered()
    assert untaught, "the lesson has grown to cover every phrasing the gauntlet examines"
    assert set(covered) | set(untaught) == set(WHY_FORMS)
    assert not set(untaught) & {form for form, walk in LESSON if walk}
