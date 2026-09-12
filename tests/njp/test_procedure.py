"""V.53 — the procedural organ. What a task definition says, read out of it.

Every number quoted here comes from :mod:`nyxara.njp.procedureschool` on the shipped corpus of 698
distinct task definitions and the thirty-six of them marked by hand. The tests that assert a score
assert a *floor*, not the exact figure, so an improvement is not a failure; the tests that assert a
mechanism assert that removing it costs something, which is the only evidence that it is a
mechanism rather than a decoration.
"""

from __future__ import annotations

import pytest

from nyxara.njp.procedure import (
    LESSONS, Procedure, ProcedureReader, taught_procedures,
)
from nyxara.njp.procedureschool import HELD_OUT, SEALED, examine, read_corpus, sample


@pytest.fixture(scope="module")
def corpus():
    rows = read_corpus()
    if not rows:
        pytest.skip("task-definition corpus not built")
    return rows


@pytest.fixture(scope="module")
def reader():
    return taught_procedures()


@pytest.fixture(scope="module")
def marked(corpus):
    return examine(corpus)


# --------------------------------------------------------------------------------------------- #
#  the demonstrations
# --------------------------------------------------------------------------------------------- #
def test_every_mark_is_found_in_its_own_lesson(reader):
    """A mark the text does not contain teaches nothing. V.48 lost a whole lesson this way."""
    assert reader.unlearnt() == []


def test_lessons_are_verbatim_corpus_rows(corpus):
    """No demonstration is written for the reader's convenience; each is a real definition."""
    texts = {" ".join(str(r.get("instruction") or "").split()) for r in corpus}
    for lesson in LESSONS:
        assert " ".join(lesson.text.split()) in texts, lesson.task


def test_no_lesson_is_an_audited_definition():
    """Teaching on an audited item would make its score a memory rather than a reading."""
    audited = {mark.task for mark in HELD_OUT + SEALED}
    assert not (audited & {lesson.task for lesson in LESSONS})


def test_the_audit_is_the_sample_it_claims_to_be(corpus):
    """The thirty-six are the fixed shuffle's first thirty-six, not a hand-picked set."""
    drawn = {str(row.get("task") or "") for row in sample(corpus)}
    assert drawn == {mark.task for mark in HELD_OUT + SEALED}


# --------------------------------------------------------------------------------------------- #
#  cold, and what teaching is worth
# --------------------------------------------------------------------------------------------- #
def test_cold_reads_nothing(marked):
    """The floor. A reader with no demonstrations returns a procedure with the text and no more."""
    cold = marked["cold"]
    assert cold.overall == 0.0
    assert cold.given.recall == 0.0
    assert cold.action == 0.0
    # And it does not invent an answer space either, on any of the thirteen that name none.
    assert cold.silent_right == cold.silent_asked


def test_teaching_beats_cold_on_held_out_and_on_sealed(marked):
    assert marked["taught"].overall >= 0.80
    assert marked["sealed"].overall >= 0.80
    assert marked["taught"].overall > marked["cold"].overall


def test_one_lesson_is_not_fourteen_lessons(marked):
    """If a single demonstration got most of the way there, the other thirteen are decoration."""
    assert marked["one_lesson"].overall < marked["taught"].overall / 3


def test_removing_the_abstracted_shapes_costs_reach(corpus):
    """The falsification: strip the cued level and the corpus goes unread, not merely misread.

    On the thirty-six audited items the two levels are within a point of each other, which on its
    own would say the abstraction is decoration. Over all 698 it is not close: the frames read a
    prerequisite in three definitions in five and the full reader in seven in eight.
    """
    from nyxara.njp.procedureschool import coverage
    full = coverage(corpus)
    frames = coverage(corpus, _frames_only())
    assert full["with_given"] > frames["with_given"] + 0.15
    assert full["with_goal"] >= frames["with_goal"]


def _frames_only() -> ProcedureReader:
    engine = taught_procedures()
    engine.shapes = {k: v for k, v in engine.shapes.items() if v.level == "frame"}
    return engine


def test_a_cued_shape_of_nothing_but_holes_is_refused(reader):
    for shape in reader.shapes.values():
        if shape.level == "cued":
            assert shape.anchors >= 1, shape.render()


def test_a_cued_shape_needs_two_witnesses_or_a_role_word(reader):
    for shape in reader.shapes.values():
        if shape.level == "cued" and shape.trusted and not shape.role_word:
            assert len(shape.witnesses) >= 2, shape.render()


def test_shapes_generalise(reader, corpus):
    """A shape that has only ever read the definition it was induced from has shown nothing."""
    for row in corpus[:200]:
        reader.read(str(row.get("instruction") or ""), task=str(row.get("task") or ""))
    assert any(shape.generalises for shape in reader.shapes.values())


# --------------------------------------------------------------------------------------------- #
#  what was counted rather than typed
# --------------------------------------------------------------------------------------------- #
def test_boundaries_are_counted_not_declared(reader):
    learned = reader.learned()
    # Stops and opens are tags, not the words that happened to sit next to a demonstration.
    assert learned["stops"]["goal"] == ["END", "PUNCT"]
    assert learned["opens"]["given"] == ["WORD"]
    # Joiners likewise: `and` and `,` were never written down and both are read.
    assert set(learned["joiners"]) == {"CONJ", "PUNCT"}
    assert learned["triggers"] == ["if"]
    assert learned["standalone_triggers"] == ["otherwise"]


def test_no_instruction_phrase_is_written_into_the_module():
    """The two phrases that would have been easiest to type, and are not typed."""
    import io
    import pathlib
    import tokenize
    source = pathlib.Path(ProcedureReader.__module__.replace(".", "/") + ".py")
    body = source.read_text().split("LESSONS: Tuple[Lesson, ...] = (")[0] + "\n"
    # Comments and docstrings out — both talk *about* these phrases, at length, saying that the
    # module does not contain them. What is left is the executable text.
    kept = [tok.string for tok in tokenize.generate_tokens(io.StringIO(body).readline)
            if tok.type not in (tokenize.COMMENT, tokenize.STRING)]
    code = " ".join(kept).lower()
    assert "you are given" not in code
    assert "your task is to" not in code
    assert "you're given" not in code


# --------------------------------------------------------------------------------------------- #
#  reading one
# --------------------------------------------------------------------------------------------- #
def test_two_prerequisites_are_two(reader):
    got = reader.read("In this task, you are given a question and a context passage. You have to "
                      "answer the question based on the given passage.")
    assert [p.lower() for p in got.given] == ["question", "context passage"]
    assert got.action == "answer"


def test_a_conjunction_inside_a_phrase_is_not_a_coordination(reader):
    """"a sentence in the English and Hindi language" is one prerequisite, not two."""
    got = reader.read("In this task, you are given a sentence in the English and Hindi language. "
                      "Your task is check if the Hindi sentence is translation of English.")
    assert len(got.given) == 1
    assert "language" in got.given[0].lower()


def test_a_quoted_example_is_not_an_answer_space(reader):
    """The parenthesised vowels are an example of the input, and five of them are not answers."""
    got = reader.read("In this task, you need to count the number of vowels (letters 'a', 'e', "
                      "'i', 'o', 'u') / consonants (all letters other than vowels) in the given "
                      "sentence.")
    assert got.outputs == []
    assert got.action == "count"


def test_an_answer_space_stated_as_a_choice_is_read(reader):
    got = reader.read('In this task, you are given a sentence in the English and Hindi language. '
                      'Your task is check if the Hindi sentence is translation of English. if the '
                      'translation is correct than generate label "Yes", otherwise generate label '
                      '"No".')
    assert {v.lower() for v in got.outputs} == {"yes", "no"}
    assert got.decides
    assert len(got.conditions) >= 1


def test_a_condition_keeps_its_trigger_and_its_consequence(reader):
    got = reader.read("In this task, you will be given a list of integers. You should remove all "
                      "of the odd integers from the list. If every integer in the input list is "
                      'odd then an empty list ("[]") should be returned. Otherwise, answer with '
                      "the list of even numbers separated by comma inside brackets.")
    assert got.conditions
    for condition in got.conditions:
        assert condition.trigger and condition.consequence


def test_the_definition_itself_is_kept(reader):
    text = ("In this task, you are given a context tweet and an answer. Your job is to generate a "
            "question. Note that your question should be answerable based on the given tweet.")
    got = reader.read(text)
    assert got.text == text
    assert any("Note that" in note for note in got.caveats)


def test_an_apostrophe_is_not_a_quote(reader):
    """"the reviewer's sentiment into: ..." opened a quote and lost five real answers to it."""
    got = reader.read('In this task, you will be given a movie review and a question about the '
                      "reviewer's sentiment. Classify the reviewer's sentiment into: "
                      '"no sentiment expressed", "negative", "neutral", "positive", and "mixed".')
    assert len(got.outputs) == 5
    assert "no sentiment expressed" in [v.lower() for v in got.outputs]


def test_prerequisites_met_says_what_is_missing(reader):
    got = reader.read("In this task, you are given a question and a context passage. You have to "
                      "answer the question based on the given passage.")
    ready, missing = got.prerequisites_met(["a question"])
    assert not ready
    assert any("passage" in m for m in missing)
    ready, missing = got.prerequisites_met(["a question", "the context passage"])
    assert ready and missing == []


def test_reading_files_nothing(reader):
    got = reader.read("In this task, you are given a sentence. You must judge whether it is long.")
    assert isinstance(got, Procedure)
    assert got.to_dict()["text"]


# --------------------------------------------------------------------------------------------- #
#  the corpus, and the brain
# --------------------------------------------------------------------------------------------- #
def test_the_corpus_is_the_698_it_claims(corpus):
    assert len(corpus) >= 690
    assert len({str(r.get("instruction") or "") for r in corpus}) == len(corpus)
    for row in corpus[:50]:
        assert row.get("licence") and row.get("source") and row.get("task")


def test_the_taxonomy_comes_out_of_the_reading(corpus, reader):
    from nyxara.njp.procedureschool import actions
    tally = dict(actions(corpus, reader))
    for verb in ("generate", "classify", "translate", "convert"):
        assert tally.get(verb, 0) > 5, verb
    # And the unread share is reported rather than hidden.
    assert "—" in tally


def test_the_brain_reads_a_procedure_and_can_say_what_it_needs():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    got = brain.read_procedure("In this task, you are given a question and a context passage. You "
                               "have to answer the question based on the given passage.",
                               task="t")
    assert got is not None and got.action == "answer"
    assert brain.can_do("t")["known"] is False       # reading is not filing


def test_the_answer_space_is_reachable_from_english():
    """V.49's lesson: a predicate that no question can reach is a fact stored and unreachable."""
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    filed = brain.learn_procedures(limit=120, file=True)
    if not filed["read"]:
        pytest.skip("corpus not built")
    named = [n for n, p in brain._procedures_read.items() if p.decides]
    assert named, "no procedure in 120 named its answer space"
    name = named[0]
    space = ", ".join(brain._procedures_read[name].outputs)
    for question in (f"what are the answers for {name}?",
                     f"what are the options for {name}?",
                     f"what can {name} answer?",
                     f"what does {name} answer with?"):
        answer = brain.perceive(question).grounding.answer
        assert answer.text == space, (question, answer.state, answer.text)


def test_requires_and_produces_are_reachable_and_distinct():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    if not brain.learn_procedures(limit=120, file=True)["read"]:
        pytest.skip("corpus not built")
    name = next(n for n, p in brain._procedures_read.items() if p.given and p.goal)
    produced = brain.perceive(f"what does {name} produce?").grounding.answer
    assert produced.text == brain._procedures_read[name].goal
    # `requires` finds the prerequisites; with two of them it says there are two rather than
    # picking one, which is the repo's rule and not a failure to answer.
    required = brain.perceive(f"what does {name} require?").grounding.answer
    heads = {t.object for t in required.triples}
    assert heads and heads <= set(brain._procedures_read[name].given)


def test_can_do_answers_from_a_read_procedure():
    from nyxara.njp.brain import NJPBrain
    brain = NJPBrain()
    if not brain.learn_procedures(limit=120, file=False)["read"]:
        pytest.skip("corpus not built")
    name = next(n for n, p in brain._procedures_read.items() if p.given)
    verdict = brain.can_do(name, [])
    assert verdict["known"] and not verdict["ready"] and verdict["missing"]
    held = brain._procedures_read[name].given
    assert brain.can_do(name, held)["ready"]


def test_exports_do_not_shadow(reader):
    import nyxara.njp as package
    from nyxara.njp import procedure as module
    assert package.Procedure is module.Procedure
    assert package.ProcedureLesson is module.Lesson
    assert package.Branch is module.Condition
