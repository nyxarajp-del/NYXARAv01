"""V.54 — what kind of thing an answer has to be, and the veto that follows from it.

The organ predicts a *shape*, never a fact, and its only use is to refuse a candidate that cannot
be what was asked for. So the tests come in two halves: that the shape is learned rather than
typed, and that the refusal is safe — a veto that suppresses correct answers has traded one
failure for another and is worse than nothing.
"""

from __future__ import annotations

import pytest

from nyxara.njp.asked import (
    KINDS, Asked, probe, read_questions, satisfies, shape_of,
)
from nyxara.njp.askedschool import examine, false_vetoes, split, veto_bar


@pytest.fixture(scope="module")
def corpus():
    rows = read_questions()
    if len(rows) < 1000:
        pytest.skip("question corpus not built")
    return rows


@pytest.fixture(scope="module")
def learned(corpus):
    learn, held = split(corpus)
    engine = Asked()
    engine.learn_from(learn)
    return engine, held


@pytest.fixture(scope="module")
def marked(corpus):
    return examine(corpus)


# --------------------------------------------------------------------------------------------- #
#  the shape of an answer is about the answer
# --------------------------------------------------------------------------------------------- #
def test_shape_is_read_off_the_surface():
    assert shape_of("yes") == "polar"
    assert shape_of("No.") == "polar"
    assert shape_of("1942") == "year"
    assert shape_of("May 1942") == "year"
    assert shape_of("190") == "count"
    assert shape_of("3,500") == "count"
    assert shape_of("chemical energy") == "span"
    assert shape_of("because Old Persian cuneiform had been found there") == "phrase"
    assert shape_of("") == ""


def test_a_year_is_a_year_before_it_is_a_count():
    """Ordered on purpose: 1994 satisfies both tests and only one of them is informative."""
    assert shape_of("1994") == "year"
    assert KINDS.index("year") < KINDS.index("count")


def test_a_question_reading_names_no_answer_kind():
    """Every measurement is something you could take off any sentence, knowing nothing."""
    reading = probe("How many field goals were made?")
    assert set(reading) == {"opens", "opens_two", "second_class", "third_class",
                            "has_number", "length", "ends_word", "closed_share"}
    for value in reading.values():
        assert str(value).lower() not in {k.lower() for k in KINDS}


def test_no_question_word_table_in_the_module():
    """`how many -> a number` has to be a finding. It is not allowed to be a line of code."""
    import io
    import pathlib
    import re
    import tokenize
    source = pathlib.Path("nyxara/njp/asked.py")
    kept = [tok.string for tok in
            tokenize.generate_tokens(io.StringIO(source.read_text()).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING)]
    code = " ".join(kept).lower()
    # Whole words. A substring test fails on `whole`, which contains `who` and is a parameter
    # name in `_share` — a false alarm that says nothing about whether a table was written.
    for phrase in ("how many", "when", "who", "where", "what year", "how much"):
        assert not re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", code), phrase


# --------------------------------------------------------------------------------------------- #
#  what teaching is worth
# --------------------------------------------------------------------------------------------- #
def test_untaught_predicts_nothing_and_vetoes_nothing():
    engine = Asked()
    assert engine.expects("When was the Battle of the Coral Sea fought?") == ("", "")
    rejected, why = engine.contradicts("How many were made?", "coastal protection")
    assert not rejected and why == ""


def test_induction_off_is_the_floor(marked):
    blind = marked["no_rules"]
    assert blind.rules == 0
    assert blind.coverage == 0.0
    assert blind.false_veto == 0.0


def test_taught_beats_the_base_rate(marked):
    taught, base = marked["taught"], marked["base_rate"]
    assert taught.when_fires > base.when_fires
    assert taught.coverage > 0.2


def test_it_worked_out_that_counting_questions_want_numbers(learned):
    """Not asserted as a string — asserted as a rule the cover found and labelled `count`."""
    engine, _held = learned
    counting = [r for r in engine.rules if r.label == "count"]
    assert counting, "no rule for `count` at all"
    assert any("many" in r.render() for r in counting)


# --------------------------------------------------------------------------------------------- #
#  the veto, and whether it is safe
# --------------------------------------------------------------------------------------------- #
def test_the_veto_does_not_catch_the_case_it_was_built_for(learned):
    """The uncomfortable one, asserted so it cannot quietly stop being true.

    A counting question offered a phrase is the failure that started this module, and at the
    shipped bar the veto says **nothing** about it: `opens_two is how many` reaches 0.816 and the
    bar is 0.90. Dropping the bar to 0.80 catches it and costs one correct answer in thirty-four.
    Both halves are asserted here — that the shipped configuration is silent, and that the
    mechanism does work when it is allowed to.
    """
    engine, _held = learned
    rejected, _why = engine.contradicts("How many field goals were made?", "coastal protection")
    assert not rejected, "the safe bar has started catching this; re-read the cost table"

    import dataclasses
    louder = dataclasses.replace(engine, veto_purity=0.80)
    rejected, why = louder.contradicts("How many field goals were made?", "coastal protection")
    assert rejected and "count" in why


def test_what_the_veto_can_say_at_the_shipped_bar(learned):
    """Two rules clear it, and this is what they buy."""
    engine, _held = learned
    qualifying = [r for r in engine.rules if r.purity >= engine.veto_purity]
    assert qualifying, "no rule clears the bar — the veto is inert, see veto_bar()"
    rejected, why = engine.contradicts("Which food was eaten at dinner?",
                                       "because the guests had already eaten before arriving")
    assert not rejected or "span" in why


def test_a_refinement_is_not_a_contradiction():
    """`1947` is a `year` only because the year test runs before the span test."""
    assert satisfies("year", "span")
    assert satisfies("count", "span")
    assert satisfies("year", "count")
    assert satisfies("span", "phrase") and satisfies("phrase", "span")
    # And the divides that are real, in both directions.
    assert not satisfies("polar", "span")
    assert not satisfies("span", "polar")
    assert not satisfies("span", "count")


def test_a_year_answers_a_what_year_question(learned):
    """It did not, until the itemised mistakes were read: `opens is what` expects a span, and a
    year is a span with the year test having got there first."""
    engine, _held = learned
    rejected, why = engine.contradicts("what year was the film That Hagen Girl released?", "1947")
    assert not rejected, why


def test_the_veto_passes_a_right_shaped_answer(learned):
    engine, _held = learned
    for question, answer in (("How many field goals were made?", "17"),
                             ("Is the sky blue?", "yes"),
                             ("Which country is it from?", "Indonesia")):
        rejected, why = engine.contradicts(question, answer)
        assert not rejected, (question, answer, why)


def test_the_veto_never_argues_about_wordiness(learned):
    """`span` and `phrase` differ only in length, and length is not a reason to refuse."""
    engine, _held = learned
    rejected, _why = engine.contradicts(
        "Who did the lineup instability start with?",
        "Turner was replaced by bassist and vocalist John Wetton")
    assert not rejected


def test_the_veto_rarely_rejects_a_correct_answer(marked):
    """The number that decides whether this may be wired into anything.

    Every held-out question is handed its own correct answer. Every veto is a mistake, and one
    that would have suppressed a right answer — the failure this organ exists to *reduce*.
    Measured at the shipped bar: 20 of 5,000.
    """
    assert marked["taught"].false_veto <= 0.01


def test_a_silent_veto_is_not_a_perfect_one(corpus):
    """The trap this module fell into: at a bar no rule can clear, the cost column reads 0.0000.

    So the table is required to show the veto *firing* wherever it claims to be safe. A row with
    a zero cost and a zero reach is a mechanism doing nothing, not a mechanism getting it right.
    """
    rows = veto_bar(bars=(0.90, 0.99), questions=corpus)
    shipped = next(r for r in rows if r[0] == 0.90)
    silent = next(r for r in rows if r[0] == 0.99)
    # The shipped bar must earn its cheap cost by actually firing.
    assert shipped[1] > 0.05, "the shipped veto fires on nothing; its zero cost means nothing"
    assert shipped[2] < 0.01
    # And a bar nothing can clear is the counter-example: a flawless record and no mechanism.
    assert silent[1] == 0.0 and silent[2] == 0.0


def test_the_damage_is_itemised_not_summarised(corpus):
    """Whatever the rate, the individual mistakes are inspectable rather than a percentage."""
    bad = false_vetoes(questions=corpus, limit=5)
    for question, answer, reason in bad:
        assert question and answer and reason


def test_it_abstains_where_it_has_no_rule(learned):
    engine, _held = learned
    kind, why = engine.expects("Zorbin flimwaxes the queep?")
    if not kind:
        assert why == ""
        assert engine.contradicts("Zorbin flimwaxes the queep?", "1994") == (False, "")


def test_a_question_it_cannot_read_is_not_an_error(learned):
    engine, _held = learned
    assert engine.expects("") == ("", "")
    assert engine.contradicts("", "anything") == (False, "")


# --------------------------------------------------------------------------------------------- #
#  the corpus
# --------------------------------------------------------------------------------------------- #
def test_the_corpus_carries_its_provenance(corpus):
    for row in corpus[:200]:
        assert row.task and row.source and row.licence


def test_deliberately_wrong_answers_are_not_in_the_corpus(corpus):
    """FLAN has 14,485 rows whose source task exists to produce a false answer."""
    import re
    bad = re.compile(r"incorrect|wrong|implausible|distractor", re.I)
    assert not [q for q in corpus if bad.search(q.task)]


def test_answers_that_name_a_category_are_not_in_the_corpus(corpus):
    """`drop_answer_type_generation` answers "How many...?" with the word `number`."""
    named = {"number", "date", "span", "person", "entity", "other", "location"}
    offenders = [q for q in corpus if q.answer.strip(" .").lower() in named]
    assert not offenders


def test_split_is_deterministic_and_disjoint(corpus):
    learn_a, held_a = split(corpus)
    learn_b, held_b = split(corpus)
    assert [q.question for q in learn_a] == [q.question for q in learn_b]
    assert [q.question for q in held_a] == [q.question for q in held_b]
    assert not ({q.question for q in learn_a} & {q.question for q in held_a})


# --------------------------------------------------------------------------------------------- #
#  the brain
# --------------------------------------------------------------------------------------------- #
def test_a_brain_expects_nothing_until_it_has_been_shown_questions():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    assert brain.asked is not None
    assert brain.answer_kind("When was the Battle of the Coral Sea fought?")["kind"] == ""
    assert brain.check_answer("How many were made?", "coastal protection")["wrong_kind"] is False


def test_the_brain_learns_kinds_from_a_shuffle_never_a_prefix(corpus):
    """A prefix of the fold is one part of one submix — the bias that broke the first curve."""
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    got = brain.learn_answer_kinds(limit=6000)
    if not got["questions"]:
        pytest.skip("corpus not built")
    assert got["rules"] > 0
    assert brain.answer_kind("How many were made?")["kind"] == "count"


def test_check_answer_is_a_veto_and_supplies_nothing():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    if not brain.learn_answer_kinds(limit=6000)["questions"]:
        pytest.skip("corpus not built")
    verdict = brain.check_answer("Which food was eaten?", "salmon")
    assert set(verdict) == {"wrong_kind", "why", "expected"}
    assert "answer" not in str(verdict.get("why", "")).lower() or not verdict["wrong_kind"]


def test_exports_do_not_shadow():
    import nyxara.njp as package
    from nyxara.njp import asked as module
    assert package.Asked is module.Asked
    assert package.AskedQuestion is module.Question
    assert package.satisfies is module.satisfies
