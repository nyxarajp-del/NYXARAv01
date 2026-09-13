"""The wire from reading to answering, and the module it destroyed on the way.

Most of what is pinned here is a failure — the live one V.99 ended on, the confabulation the first
draft produced, the transfer result that is negative, and the precondition that would have caught
an hour of measuring a broken package. A defect nothing asserts is a defect that comes back.
"""

from __future__ import annotations

import pytest

from nyxara.njp.askable import (UNKNOWN, Answer, World, always_unknown, answer_type, ask,
                                overlap_span, read_world, supplied_price)

KOLA = ("The Kola Superdeep Borehole is located on the Kola Peninsula in Russia. "
        "Drilling began in 1970 and reached a depth of 12,262 metres in 1989. "
        "The project stopped because the rock temperature reached 180 degrees Celsius.")


# --------------------------------------------------------------------------------------------- #
#  the failure V.99 ended on
# --------------------------------------------------------------------------------------------- #
def test_the_passage_that_returned_nothing_now_answers():
    """Three questions that returned '' with the answer sitting in the passage."""
    world = read_world(KOLA)
    assert "12,262" in ask(world, "How deep did the Kola Superdeep Borehole reach?").text
    assert "180 degrees" in ask(world, "Why did the project stop?").text
    assert ask(world, "In which year did drilling begin?").text == "1970"


def test_reading_produces_structure_where_there_was_none():
    """`relations=()` was the V.99 symptom: information entered in a form nothing could query."""
    world = read_world(KOLA)
    assert world.facts, "a passage read into no facts is a passage nothing can be asked about"
    kinds = {span.kind for span in world.spans}
    assert {"date", "measure", "cause"} <= kinds


def test_a_question_the_passage_cannot_answer_returns_unknown_not_a_guess():
    world = read_world(KOLA)
    got = ask(world, "Who discovered penicillin?")
    assert got.unknown and got.text == UNKNOWN
    assert got.why, "an abstention without a reason is not better than a guess with one"


def test_unknown_is_a_word_and_not_an_empty_string():
    """An empty answer and a refusal look identical to a caller. V.99 spent a version on that."""
    assert Answer().text == UNKNOWN
    assert Answer().unknown
    assert not Answer(text="1970").unknown


# --------------------------------------------------------------------------------------------- #
#  the confabulation the first draft produced
# --------------------------------------------------------------------------------------------- #
def test_a_place_question_is_not_answered_with_its_own_subject():
    """"Where is the borehole located?" answered "Kola Superdeep Borehole" — the question's subject.

    The `wanted` tuple is a preference order and the first draft read the span list in its own
    order instead. A wrong answer, not an abstention, which is what this exam punishes hardest.
    """
    got = ask(read_world(KOLA), "Where is the borehole located?")
    assert "Peninsula" in got.text or "Russia" in got.text
    assert got.text != "Kola Superdeep Borehole"


def test_the_null_got_the_same_correction():
    """Withholding a fix from the control would make the control weak on purpose — V.85."""
    got = overlap_span(read_world(KOLA), "Where is the borehole located?")
    assert got.text != "Kola Superdeep Borehole"


# --------------------------------------------------------------------------------------------- #
#  the nulls the protocol requires
# --------------------------------------------------------------------------------------------- #
def test_always_unknown_never_answers_anything():
    assert always_unknown(read_world(KOLA), "How deep did it reach?").unknown


def test_the_overlap_null_really_does_answer():
    """A null that abstains everywhere is not a null, it is the other null."""
    assert not overlap_span(read_world(KOLA), "In which year did drilling begin?").unknown


def test_answer_type_reads_the_wh_word():
    assert answer_type("When did it start?") == "date"
    assert answer_type("How deep is it?") == "measure"
    assert answer_type("How many people?") == "number"
    assert answer_type("Why did it stop?") == "cause"
    assert answer_type("What is it?") == "thing", "most of the corpus is untyped, and that is honest"


def test_an_empty_world_answers_nothing_and_says_why():
    got = ask(World(), "anything at all?")
    assert got.unknown and got.why == "nothing was read"


# --------------------------------------------------------------------------------------------- #
#  the accounting, per V.99
# --------------------------------------------------------------------------------------------- #
def test_the_supplied_grammar_is_charged_rather_than_free():
    """V.99: an accounting that prices only run-time choices reports supplied structure as free."""
    assert supplied_price() > 0.0
    assert supplied_price(rules=1, types=1) == 0.0, "one option is no choice, wherever it sits"


# --------------------------------------------------------------------------------------------- #
#  the precondition that was missing for an hour
# --------------------------------------------------------------------------------------------- #
def test_the_laboratory_is_intact():
    from nyxara.njp.askableschool import intact

    assert intact() == (), "a number measured against a package with a hole in it means nothing"


def test_the_precondition_notices_a_symbol_that_vanished():
    """The exact destruction of V.100: `njp.grounding` overwritten and `Grounder` gone.

    A path fingerprint would have seen one tracked file change, which is what an ordinary edit
    looks like. What vanished was a name, so the guard asks for names.
    """
    import nyxara.njp.grounding as grounding
    from nyxara.njp.askableschool import intact

    keep = grounding.Grounder
    try:
        del grounding.Grounder
        assert any("Grounder" in line for line in intact())
    finally:
        grounding.Grounder = keep
    assert intact() == ()


def test_the_grounder_this_module_destroyed_is_still_there():
    """Restored byte-clean from git. Pinned so the third time is also the last."""
    from nyxara.njp.grounding import Grounder

    assert Grounder is not None


# --------------------------------------------------------------------------------------------- #
#  V.101: extent, discovered rather than supplied
# --------------------------------------------------------------------------------------------- #
def test_extent_follows_the_question_instead_of_being_a_constant():
    """The V.100 transfer defect: 3-word spans whatever the corpus wanted.

    A question that quotes most of its sentence must leave a short remainder, and one that shares
    only a topic word a long one. Same organ, same passage, two extents.
    """
    from nyxara.njp.askable import _by_extent

    sentence = ("Marie Curie won the Nobel Prize in Physics in 1903 for her research on "
                "radiation phenomena discovered by Henri Becquerel")
    narrow = _by_extent(sentence, "What did Marie Curie win the Nobel Prize in Physics in 1903 for?")
    wide = _by_extent(sentence, "Tell me about Curie")
    assert len(wide.split()) > len(narrow.split()), (narrow, wide)


def test_the_remainder_is_what_the_question_did_not_already_say():
    from nyxara.njp.askable import _runs_absent_from

    runs = _runs_absent_from("Drilling began in 1970 and reached 12,262 metres",
                             "When did drilling begin?")
    words = {w for _, _, run in runs for w in run}
    assert "Drilling" not in words, "the question already said it"
    assert "1970" in words


def test_a_tie_is_broken_by_evidence_and_not_by_position():
    """Two sentences sharing the same words: the one holding the demanded type wins.

    On the develop split 48 of 51 tie-abstentions had real overlap and were answerable. Position
    order would have been a coin-flip; the answer's own type is evidence about where it lives.
    """
    world = read_world("The programme was praised widely. The programme cost 4.2 million dollars.")
    got = ask(world, "How much did the programme cost?")
    assert "4.2" in got.text


def test_the_answer_type_table_has_exactly_one_copy():
    """Three drifting copies of the same table is its own defect; the chooser and both answerers
    read `SATISFIES`."""
    from nyxara.njp.askable import SATISFIES

    assert SATISFIES["date"] == ("date",)
    assert "measure" in SATISFIES["number"]


def test_the_gain_did_not_come_from_the_split_it_was_tuned_on():
    """Recorded, not asserted live: develop 0.243, held 0.240, sealed 0.246.

    `sealed` was carved from rows no other split touches and scored once, after every change. Three
    numbers within 0.006 of each other is what "not overfitted to 300 rows" looks like; had they
    diverged, the develop figure would have been the one to disbelieve.
    """
    from nyxara.njp.askableschool import SEALED, splits

    made = splits()
    seen = {(r.passage, r.question) for r in made["develop"]} | {
        (r.passage, r.question) for r in made["held"]}
    fresh = {(r.passage, r.question) for r in made["sealed"]}
    assert len(made["sealed"]) == SEALED
    assert not (fresh & seen), "a sealed split that overlaps another split is not sealed"


def test_the_transfer_gate_is_still_failed_and_that_is_recorded():
    """Pinned as a negative. transfer 0.054 against the same null's 0.072.

    Extent discovery narrowed the gap from roughly threefold to roughly a third and did not close
    it. If someone later makes this pass, that is a result worth noticing rather than a test that
    quietly started succeeding.
    """
    from nyxara.njp.askableschool import grade, splits

    rows = splits()["transfer"][:120]
    mine = grade("world model", "transfer", rows, ask)
    null = grade("word overlap", "transfer", rows, overlap_span)
    assert mine.accuracy <= null.accuracy + 0.05, (
        "the transfer gate may have been passed — re-measure on the full split and update the "
        "documented result rather than deleting this test")
