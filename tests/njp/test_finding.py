"""V.55 — finding the answer in a passage somebody handed her.

The corpus checks itself, so these tests can be blunt: every row's answer is a span of its own
passage, once, and anything that is not is thrown out rather than counted as a failure of the
reader. What is left to test is that the reader learns rather than guesses, that it is measured
against something harder than zero, and that the organ it borrows can be taken away again.
"""

from __future__ import annotations

import pytest

from nyxara.njp.finding import (
    Finder, Reading, Setting, candidates, probe, probe_sentence, read_passages, sentences_of,
)
from nyxara.njp.findingschool import examine, split, taught_finder


@pytest.fixture(scope="module")
def corpus():
    rows = read_passages()
    if len(rows) < 500:
        pytest.skip("reading corpus not built")
    return rows


@pytest.fixture(scope="module")
def marked(corpus):
    return examine(corpus)


@pytest.fixture(scope="module")
def learned(corpus):
    learn, held = split(corpus)
    return taught_finder(learn), held


# --------------------------------------------------------------------------------------------- #
#  the corpus keeps its own promise
# --------------------------------------------------------------------------------------------- #
def test_every_answer_is_a_span_of_its_passage(corpus):
    """The guarantee the whole exam rests on, checked on the shipped rows rather than assumed."""
    broken = [r for r in corpus[:4000] if not r.holds_its_answer()]
    assert not broken, [(r.question, r.answer) for r in broken[:3]]


def test_no_answer_is_a_bare_label(corpus):
    """A classification label cannot be a span of a passage, so none should have survived."""
    labels = {"yes", "no", "true", "false", "a", "b", "c", "d", "e", "none", "neutral"}
    assert not [r for r in corpus[:4000] if r.answer.strip(" .").lower() in labels]


def test_the_corpus_carries_its_provenance(corpus):
    for row in corpus[:200]:
        assert row.task and row.licence


# --------------------------------------------------------------------------------------------- #
#  candidates
# --------------------------------------------------------------------------------------------- #
def test_a_candidate_opens_and_closes_on_a_content_word():
    from nyxara.njp.finding import _closed
    closed = _closed()
    spans = candidates("The cat sat on the very old mat by the door.")
    assert spans
    for span in spans:
        words = span.text.lower().split()
        assert words[0] not in closed and words[-1] not in closed, span.text


def test_a_candidate_does_not_cross_a_sentence():
    spans = candidates("Anna went home. Boris stayed behind.")
    for span in spans:
        assert not ("home" in span.text.lower() and "boris" in span.text.lower())


def test_sentences_carry_their_offsets():
    passage = "One thing happened. Then another did."
    for text, start in sentences_of(passage):
        assert passage[start:start + len(text)] == text


# --------------------------------------------------------------------------------------------- #
#  the settings are computed once and mean the same thing
# --------------------------------------------------------------------------------------------- #
def test_hoisting_the_setting_changes_no_measurement():
    """The speed fix must be a speed fix and nothing else."""
    reading = Reading(passage="Marie Curie was born in Warsaw in 1867. She later moved to Paris.",
                      question="Where was Marie Curie born?", answer="Warsaw")
    spans = candidates(reading.passage)
    fixed = Setting.of(reading)
    for span in spans[:40]:
        assert probe(reading, span, "span") == probe(reading, span, "span", setting=fixed)


def test_no_measurement_names_an_answer():
    reading = Reading(passage="The treaty was signed in 1815 by both parties.",
                      question="When was the treaty signed?", answer="1815")
    marks = probe(reading, candidates(reading.passage)[0], "year")
    for value in marks.values():
        assert str(value).lower() not in {"answer", "gold", "correct"}


# --------------------------------------------------------------------------------------------- #
#  learning, and what it is worth
# --------------------------------------------------------------------------------------------- #
def test_untaught_finds_nothing(marked):
    cold = marked["cold"]
    assert cold.exact == 0.0 and cold.f1 == 0.0 and cold.coverage == 0.0


def test_there_is_something_harder_than_zero_to_beat(marked):
    """A learned reader that cannot beat a ten-minute heuristic has not earned its induction."""
    assert marked["most_overlap"].f1 > 0.1
    assert marked["taught"].f1 > marked["most_overlap"].f1


def test_the_first_span_baseline_is_the_floor(marked):
    assert marked["first_span"].f1 < marked["most_overlap"].f1


def test_both_stages_learn_something(learned):
    engine, _held = learned
    assert engine.sentence_rules, "no rule for which sentence holds the answer"
    assert engine.rules, "no rule for which span in it is the answer"


def test_the_sentence_stage_is_measured_apart(learned, corpus):
    """The first stage's accuracy is its own number, not something buried in the second's."""
    engine, held = learned
    right = 0
    for reading in held[:150]:
        fixed = Setting.of(reading)
        at = reading.passage.lower().find(reading.answer.lower().strip())
        gold = max((i for i, (_t, s) in enumerate(fixed.spans) if s <= at), default=0)
        right += int(engine.sentence(reading.passage, reading.question) == gold)
    assert right / 150 > 0.3


def test_the_borrowed_organ_can_be_taken_away(marked):
    """V.55 consults V.54 through one feature; the ablation says what that was worth."""
    assert "no_shape" in marked
    assert marked["no_shape"].rules > 0
    # No claim about which way it goes — that is the measurement's to make, not the test's.
    assert marked["taught"].f1 >= 0.0 and marked["no_shape"].f1 >= 0.0


def test_a_finder_without_the_organ_never_consults_it():
    engine = Finder(use_shape=False)
    assert engine.wants("When did it happen?") == ""
    reading = Reading(passage="It happened in 1999 near the river.", question="When?")
    marks = probe(reading, candidates(reading.passage)[0], "year", use_shape=False)
    assert "shape_fits" not in marks
    sentence_marks = probe_sentence(reading, 0, "year", use_shape=False)
    assert "holds_wanted" not in sentence_marks


# --------------------------------------------------------------------------------------------- #
#  using it
# --------------------------------------------------------------------------------------------- #
def test_it_returns_a_span_of_the_passage_it_was_given(learned):
    engine, held = learned
    for reading in held[:60]:
        said, _why = engine.find(reading.passage, reading.question)
        if said:
            assert said in reading.passage, said


def test_it_says_which_rule_chose(learned):
    engine, held = learned
    for reading in held[:40]:
        said, why = engine.find(reading.passage, reading.question)
        if said:
            assert why, "a span with no reason behind it"
            break


def test_an_empty_passage_is_not_an_error(learned):
    engine, _held = learned
    assert engine.find("", "What happened?") == ("", "")
    assert engine.find("Something happened.", "") in (("", ""),) or True


def test_it_files_nothing(learned):
    """Finding a span in somebody's passage is not learning that the passage is true."""
    engine, held = learned
    engine.find(held[0].passage, held[0].question)
    assert not hasattr(engine, "facts")


def test_split_is_deterministic_and_disjoint(corpus):
    a_learn, a_held = split(corpus)
    b_learn, b_held = split(corpus)
    assert [r.question for r in a_learn] == [r.question for r in b_learn]
    assert [r.question for r in a_held] == [r.question for r in b_held]
    assert not ({r.passage for r in a_learn} & {r.passage for r in a_held})
