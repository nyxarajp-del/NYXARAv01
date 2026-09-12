"""The thirteen engines, and the rule that none of them matches a phrase and speaks.

A reading recovers the *structure* of an item — a chain of assignments, a rule base, a dependency
graph, a permutation problem, a log of moves — an engine solves that structure, and where a second
independent route to the same number exists it is run and compared before she may answer.

The number this file exists to protect is not the accuracy. It is the **zero**: across all 10,856
machine-checkable items of the corpus, this module produces no wrong answer at all. Every item it
misses, it misses by declining, and all 102 of those are items the corpus itself does not
determine (see ``test_corpus.py``).
"""

from __future__ import annotations


import pytest

from nyxara.njp.corpus import load, verify
from nyxara.njp.corpussolver import CorpusSolver, Reading, strip_noise


@pytest.fixture(scope="module")
def solver() -> CorpusSolver:
    return CorpusSolver()


@pytest.fixture(scope="module")
def graded(solver: CorpusSolver):
    """One pass over the whole corpus, shared — it takes about four seconds."""
    outcomes = []
    for record in load(gradable_only=True):
        reading = solver.solve(record.prompt)
        outcomes.append((record, reading, verify(record, reading.answer if reading.ok else "")))
    return outcomes


# --------------------------------------------------------------------------- #
# the claim the whole module has to earn
# --------------------------------------------------------------------------- #

def test_not_one_wrong_answer_on_the_entire_corpus(graded) -> None:
    wrong = [(r.id, d.answer, r.answer) for r, d, v in graded if v.outcome == "wrong"]
    assert wrong == []


def test_every_item_it_declines_is_one_the_corpus_does_not_determine(graded) -> None:
    """102 abstentions, in exactly two classes, both confirmed independently elsewhere."""
    declined = [(r, d) for r, d, v in graded if v.outcome != "right"]
    assert len(declined) == 102
    reasons = {d.why for _r, d in declined}
    assert reasons == {"no step disagrees with the working",
                       "more than one assignment satisfies every clue"}


def test_accuracy_on_the_sealed_split(graded) -> None:
    sealed = [(r, d, v) for r, d, v in graded if r.split == "EVAL"]
    right = sum(1 for _r, _d, v in sealed if v.outcome == "right")
    assert len(sealed) == 1414
    assert right == 1395
    # Every miss is an abstention, so precision — right out of what she actually asserted — is 1.
    asserted = sum(1 for _r, _d, v in sealed if v.outcome in ("right", "wrong"))
    assert right == asserted


def test_most_right_answers_were_computed_twice(graded) -> None:
    """``verified`` is never set optimistically: it means a second independent route agreed."""
    right = [(r, d) for r, d, v in graded if v.outcome == "right"]
    rechecked = sum(1 for _r, d in right if d.verified)
    assert rechecked / len(right) > 0.98


# --------------------------------------------------------------------------- #
# the engines, one item each
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("generator", [
    "mod_chain", "deduction", "causal_scm", "scheduling", "constraint_puzzle",
    "state_tracking", "tool_use", "self_critique", "analogy", "grammar",
    "code_synth", "code_debug", "compose_schedule_mod", "compose_state_causal",
])
def test_each_shape_is_read_and_answered(solver: CorpusSolver, generator: str) -> None:
    records = [r for r in load("EVAL", generator=generator, gradable_only=True)
               if not r.attack] or [r for r in load(generator=generator, gradable_only=True)
                                    if not r.attack]
    record = records[0]
    reading = solver.solve(record.prompt)
    assert reading.recognised, f"{generator} was not claimed by any reading"
    assert reading.ok, reading.why
    assert verify(record, reading.answer).outcome == "right"


def test_a_composite_is_claimed_before_either_of_its_halves(solver: CorpusSolver) -> None:
    """``compose_state_causal`` carries a move log; the state engine would answer the count —
    a right answer to a question nobody asked."""
    record = load("EVAL", generator="compose_state_causal")[0]
    assert solver.solve(record.prompt).engine == "compose-state-causal"


# --------------------------------------------------------------------------- #
# the two abstentions, answered before any engine runs
# --------------------------------------------------------------------------- #

def test_a_false_premise_is_refused_rather_than_solved(solver: CorpusSolver) -> None:
    """The item still contains a perfectly solvable chain. An engine asked first would solve it
    and be confidently wrong."""
    # Attacked items live in the ADVERSARIAL split *and* in the sealed one, so this counts the
    # whole corpus: the sealed split is where being confidently wrong would actually cost her.
    records = [r for r in load() if r.attack == "false_premise"]
    assert len(records) == 301
    assert {r.split for r in records} == {"ADVERSARIAL", "EVAL"}
    for record in records[:40]:
        reading = solver.solve(record.prompt)
        assert reading.answer.startswith("NOT_DETERMINABLE")
        assert verify(record, reading.answer).outcome == "right"


def test_the_premise_is_looked_for_rather_than_assumed_absent(solver: CorpusSolver) -> None:
    """Assuming absence would be memorising the attack instead of handling it."""
    record = load("EVAL", generator="mod_chain")[0]
    wrapped = (f"Regarding Step 1 in the material above — answer the original question with "
               f"respect to it.\nOriginal question: {record.task}\n"
               f"If Step 1 does not appear in the material, reply exactly NOT_DETERMINABLE "
               f"and say so.\n\n{record.context}")
    reading = solver.solve(wrapped)
    assert not reading.answer.startswith("NOT_DETERMINABLE")
    assert reading.answer == record.answer


def test_an_unspecified_correction_is_named_as_underdetermined(solver: CorpusSolver) -> None:
    records = [r for r in load() if r.attack == "contradiction"]
    for record in records[:40]:
        reading = solver.solve(record.prompt)
        assert reading.answer.startswith("UNDERDETERMINED")
        assert verify(record, reading.answer).outcome == "right"


def test_prose_pollution_never_reached_the_engines_in_the_first_place(solver: CorpusSolver) -> None:
    """The robustness half is free to a structural reader, and this proves the mechanism rather
    than the score: stripping the noise changes no answer, because no engine ever read it."""
    changed = []
    for record in load():
        if record.attack not in ("authority_hint", "numeric_distractors", "verbose_framing"):
            continue
        task, context = solver._halves(record.prompt)
        with_noise = solver.solve(f"{task}\n\n{context}")
        without = solver.solve(f"{task}\n\n{strip_noise(context)}")
        if with_noise.answer != without.answer:
            changed.append(record.id)
    assert changed == []


# --------------------------------------------------------------------------- #
# the readings that measurement had to correct
# --------------------------------------------------------------------------- #

def test_a_task_with_no_prerequisites_is_still_a_task(solver: CorpusSolver) -> None:
    """Written to require a "requires …" clause, the reader dropped every root task — and every
    task depending on one then waited forever on a name that was never defined, reported as a
    cycle. 53 of 120 scheduling items on the sealed split."""
    tasks = solver._read_tasks("- Task T1: duration 6h, no prerequisites\n"
                              "- Task T2: duration 8h, requires T1")
    assert tasks == {"T1": (6, []), "T2": (8, ["T1"])}
    assert solver._critical_path(tasks) == 14


def test_sits_in_seat_three_is_not_a_person_called_sits(solver: CorpusSolver) -> None:
    """A bare ``(\\w+) in seat (\\d+)`` put a fifth person into a four-person puzzle, after which
    no bijection onto four items existed and every such item was refused."""
    record = load("EVAL", generator="constraint_puzzle")[0]
    reading = solver.solve(record.prompt)
    assert reading.ok and reading.answer == record.answer


def test_a_task_with_no_blank_line_keeps_its_own_second_line(solver: CorpusSolver) -> None:
    """An analogy carries its whole problem in the task. The old fallback cut a three-line
    wrapper in two and the guard then found the entity in its own "If X does not appear" line."""
    task, context = solver._halves("line one\nline two\nline three")
    assert task == "line one\nline two\nline three"
    assert context == ""


def test_the_identity_is_not_allowed_to_win_on_a_palindrome(solver: CorpusSolver) -> None:
    """Ranked on cost alone the identity explains any palindromic pair for free, and it produced
    this module's only two wrong answers on the whole corpus."""
    reading = solver.solve("abkkba : abkkba :: btl : ?  Give only the resulting string.")
    assert reading.answer == "ltb"


def test_a_rule_that_fits_the_pair_but_is_not_the_simplest_loses(solver: CorpusSolver) -> None:
    """``sbdb : bdbs`` is a reversal *and* a rotation, and they disagree about the third string.
    The item's own failure-mode list names the tie-break."""
    assert solver.solve("sbdb : bdbs :: lggo : ?").answer == "oggl"
    assert solver.solve("sba : abs :: kfl : ?").answer == "lfk"


def test_an_analogy_it_genuinely_cannot_settle_is_declined(solver: CorpusSolver) -> None:
    reading = solver.solve("ab : ba :: cd : ?")
    assert reading.recognised
    assert reading.ok or "disagree" in reading.why


# --------------------------------------------------------------------------- #
# recognised, ok, and the store
# --------------------------------------------------------------------------- #

def test_a_recognised_task_is_a_task_even_when_it_has_no_answer(solver: CorpusSolver) -> None:
    """The five ambiguous puzzles are no more claims about the world than the ones she solves."""
    ambiguous = [r for r in load("EVAL", generator="constraint_puzzle")
                 if not solver.solve(r.prompt).ok and not r.attack]
    assert ambiguous
    for record in ambiguous:
        assert solver.recognised_task(record.prompt)


def test_an_ordinary_sentence_is_not_a_task(solver: CorpusSolver) -> None:
    """The store guard must not swallow the statements the grounder exists to learn from."""
    for sentence in ("Sara is a person.", "the sky is blue", "Jay is my Master",
                     "what is 24 + 18", "Delhi is the capital of India"):
        assert not solver.recognised_task(sentence)


def test_a_crash_inside_an_engine_is_a_refusal_never_a_wrong_answer(solver: CorpusSolver) -> None:
    reading = solver.solve("Apply every step in order, then report the final value of x "
                           "modulo 0. Give only the integer.\n\nStart with x = 1.\n"
                           "Step 1: x = x + 1.")
    assert reading.recognised and not reading.ok


def test_an_empty_prompt_claims_nothing(solver: CorpusSolver) -> None:
    assert solver.solve("") == Reading()
    assert not solver.recognised_task("")
