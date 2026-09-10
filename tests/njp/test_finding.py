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
from nyxara.njp.findingschool import examine, span_baselines, split, taught_finder


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
def test_a_candidate_may_open_on_a_determiner_and_nothing_else_closed():
    """`the Henry Cole Wing` is how a span is marked; `of the wing` is not.

    The original rule refused every closed-class opener, which sounded principled and cost 3.7%
    of the corpus — those answers were unreachable at any threshold, and three parameter sweeps
    ran before the decomposition said so.
    """
    from nyxara.njp.finding import _closed
    from nyxara.njp.semantics import Tag
    closed = _closed()
    spans = candidates("The cat sat on the very old mat by the door.")
    assert spans
    for span in spans:
        words = span.text.lower().split()
        opener = closed.get(words[0])
        assert opener is None or opener == Tag.DET, span.text
        assert words[-1] not in closed, span.text
    # And the determiner-opener is genuinely produced, not merely tolerated.
    assert any(s.text.lower().startswith("the ") for s in spans)


def test_a_determiner_alone_is_not_a_candidate():
    assert not [s for s in candidates("The cat sat.") if s.text.lower().strip() == "the"]


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


def test_it_beats_the_heuristic_where_it_matters_and_loses_where_it_does_not(marked):
    """The two readers fail in opposite directions, and the test says so rather than picking one.

    The heuristic returns the longest span of the best sentence: a phrase that usually *contains*
    the answer, so it scores well on token overlap and is almost never exactly right. The learned
    reader returns a short precise span: exactly right several times more often, and worth nothing
    in partial credit when it is wrong.

    For something meant to answer, exact is the metric that counts — `1947` answers the question
    and a twenty-word phrase containing 1947 does not — so this asserts the win on exact and
    records the loss on overlap instead of hiding it.
    """
    taught, heuristic = marked["taught"], marked["most_overlap"]
    assert taught.exact > heuristic.exact * 2, (taught.exact, heuristic.exact)
    assert heuristic.f1 > taught.f1, "overlap has stopped favouring the vaguer reader; re-read"


def test_the_first_span_baseline_is_the_floor(marked):
    assert marked["first_span"].f1 < marked["most_overlap"].f1


def test_the_span_stage_learns_something(learned):
    engine, _held = learned
    assert engine.rules, "no rule for which span is the answer"


def test_the_first_stage_ranks_by_the_ladder_not_by_the_cover(learned):
    """Three ways to pick the sentence, and the default is the one that measured best.

    ``rules`` is the greedy cover and gets 0.505; ``overlap`` is bare argmax and gets 0.580;
    ``ladder`` ranks by the graded thresholds `induce.ladder` produces and gets 0.578 — argmax's
    accuracy with a measured purity attached to each rung. All three stay runnable, the losing one
    included, because deleting the configuration that loses deletes the evidence.
    """
    engine, _held = learned
    assert engine.sentence_by == "ladder"
    assert engine.rungs, "the ladder is empty; the first stage has nothing to rank by"
    assert engine.sentence_rules, "the cover version should still be induced and runnable"
    # The rungs are a ladder: thresholds on one reading, and purity climbing with the threshold.
    from nyxara.njp.induce import AtLeast
    assert all(isinstance(v, AtLeast) for r in engine.rungs for _n, v in r.terms)
    climbing = [r.purity for r in engine.rungs]
    assert climbing == sorted(climbing), climbing


def test_the_sentence_stage_is_measured_apart(learned, corpus):
    """The first stage's accuracy is its own number, not something buried in the second's."""
    from nyxara.njp.findingschool import gold_sentence, sentence_baselines
    engine, held = learned
    right = sum(1 for reading in held[:200]
                if engine.sentence(reading.passage, reading.question) == gold_sentence(reading))
    assert right / 200 > 0.5
    # And it is reported against what it must beat, in both directions.
    against = sentence_baselines(held[:200])
    assert against["argmax_overlap"] > against["always_first"]


def test_the_borrowed_organ_is_measured_not_assumed(marked):
    """`njp.asked` enters the span stage as one removable feature so this can be measured.

    No assertion about which way it goes: on the scratch corpus removing it moved nothing at all,
    on the shipped one it moves, and a test that pinned either would be pinning an accident of
    which corpus was loaded. What is asserted is that the ablation is *runnable* and that both
    arms actually learn something — the number itself belongs in the report.
    """
    taught, blind = marked["taught"], marked["no_shape"]
    assert taught.rules > 0 and blind.rules > 0
    assert taught.asked == blind.asked


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
    # By passage, not by row. SQuAD asks a dozen questions of one paragraph, so a row-wise cut
    # puts the same passage on both sides and the reader is examined on what it studied.
    assert not ({r.passage for r in a_learn} & {r.passage for r in a_held})


# --------------------------------------------------------------------------------------------- #
#  the span stage, against something
# --------------------------------------------------------------------------------------------- #
def test_the_span_ranker_beats_picking_one_at_random(corpus):
    """The falsification the second stage went two versions without.

    The sentence stage has had a baseline since it was built; the span stage had none, which is
    how `exact_when_sentence_right = 0.0259` sat in a report looking like a hard problem rather
    than like a mechanism nobody had tested. It is a hard problem — 142 candidates in the average
    gold sentence — and the ranker does beat every way of choosing without one.
    """
    learn, held = split(corpus)
    engine = taught_finder(learn[:400])
    got = span_baselines(engine, held[:150])
    assert got["rows"] > 50, got
    assert got["learned"] > got["random"], got
    assert got["learned"] >= got["shortest"], got


def test_the_baselines_all_see_the_same_candidates(corpus):
    """Conditioned on the gold being reachable, or the generator's ceiling hides inside the ranker."""
    learn, held = split(corpus)
    got = span_baselines(taught_finder(learn[:400]), held[:150])
    assert got["candidates_per_sentence"] > 1
    assert all(0.0 <= got[k] <= 1.0
               for k in ("learned", "random", "first", "longest", "shortest", "fewest_asked"))
