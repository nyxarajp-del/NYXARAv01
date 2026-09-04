"""The whole brain sitting an examination it did not write — the wiring, not the organ.

The distinction is the entire point of measuring it this way and it earned itself immediately:
with :mod:`nyxara.njp.corpussolver` answering 1395 of 1414 sealed items correctly, the *brain*
scored **0**. The echo detector was deleting every reply on its way out, because an answer that is
correct here is made of the question's own symbols — ``"…value of x modulo 13…"`` answered ``"3"``,
``"Which item does Devi own?"`` answered ``"card"``. An organ measured in isolation would have
reported success on a system that said nothing.

Then, with that fixed, the five seating puzzles the solver declines as genuinely ambiguous came
back **"4"** and **"5"** — seat numbers, offered by recall against a question asking for an item.
A recognised refusal that does not block is an invitation for a guess, and the guard that stops it
is asserted here rather than trusted.
"""

from __future__ import annotations

import pytest

from nyxara.njp.corpus import load, verify
from nyxara.njp.corpussolver import CorpusSolver
from nyxara.njp.school import ExamConditions


@pytest.fixture(scope="module")
def brain():
    from nyxara.njp.brain import NJPBrain
    return NJPBrain(ExamConditions())


# --------------------------------------------------------------------------- #
# the organ is reachable, and it answers through the whole turn
# --------------------------------------------------------------------------- #

def test_the_organ_is_built_and_wired(brain) -> None:
    assert brain.cognition is not None
    assert type(brain.cognition).__name__ == "CorpusSolver"


@pytest.mark.parametrize("generator", [
    "mod_chain", "deduction", "causal_scm", "scheduling", "constraint_puzzle",
    "state_tracking", "tool_use", "self_critique", "analogy", "compose_state_causal",
])
def test_a_worked_answer_survives_the_echo_detector(brain, generator: str) -> None:
    """Every one of these is built from the item's own symbols, which is what being correct looks
    like here rather than what echoing looks like."""
    record = next(r for r in load("EVAL", generator=generator, gradable_only=True)
                  if not r.attack)
    reply = brain.think(record.prompt).answer
    assert reply, f"{generator}: the turn went out silent"
    assert verify(record, reply).outcome == "right"


def test_the_turn_carries_which_engine_claimed_it_and_whether_it_was_rechecked(brain) -> None:
    record = load("EVAL", generator="mod_chain")[0]
    thought = brain.think(record.prompt)
    assert thought.cognition is not None
    assert thought.cognition.engine == "mod-chain"
    assert thought.cognition.verified is True


# --------------------------------------------------------------------------- #
# a recognised refusal blocks as firmly as an answer
# --------------------------------------------------------------------------- #

def test_an_item_she_declines_is_never_handed_down_for_a_guess(brain) -> None:
    solver = CorpusSolver()
    ambiguous = [r for r in load("EVAL", generator="constraint_puzzle")
                 if not r.attack and not solver.solve(r.prompt).ok]
    assert ambiguous, "the fixture for this test has gone"
    for record in ambiguous:
        assert brain.think(record.prompt).answer.strip() == ""


def test_a_broken_context_is_named_rather_than_answered(brain) -> None:
    for attack, token in (("false_premise", "NOT_DETERMINABLE"),
                          ("contradiction", "UNDERDETERMINED")):
        record = next(r for r in load("EVAL") if r.attack == attack)
        reply = brain.think(record.prompt).answer
        assert reply.startswith(token)
        assert verify(record, reply).outcome == "right"


# --------------------------------------------------------------------------- #
# the store — the defect this organ was half written for
# --------------------------------------------------------------------------- #

def test_an_exam_item_is_a_task_and_never_a_fact_about_the_world(brain) -> None:
    """34 of the first 97 sealed items came back beginning ``noted:`` before this existed —
    "Move the seal from the brown sack to the black tin." is a well-formed statement, and the
    grounder was filing it as one."""
    noted = 0
    for record in load("EVAL", gradable_only=True, limit=60, seed=7):
        assert brain._is_cognitive_task(record.prompt)
        if str(brain.think(record.prompt).answer).startswith("noted:"):
            noted += 1
    assert noted == 0


def test_a_statement_is_still_learned_from(brain) -> None:
    """The guard must not swallow the turns the grounder exists for."""
    assert not brain._is_cognitive_task("Sara is a person.")
    assert not brain._is_cognitive_task("the capital of France is Paris")


# --------------------------------------------------------------------------- #
# it must not shadow what already worked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("asked,expected", [
    ("what is 24 + 18", "42"),
    ("what is the gcd of 48 and 18", "6"),
])
def test_the_mathematician_is_untouched(brain, asked: str, expected: str) -> None:
    assert expected in brain.think(asked).answer


def test_no_ordinary_turn_is_claimed_by_the_new_organ(brain) -> None:
    for asked in ("what is 24 + 18", "expand (x+2)(x+3)", "who is my Master",
                  "what is the area of a triangle with base 10 and height 6"):
        assert brain._cognitive(asked) is None


# --------------------------------------------------------------------------- #
# the school and the examination
# --------------------------------------------------------------------------- #

def test_the_sealed_examination_reports_three_ways(brain) -> None:
    from nyxara.njp.corpusschool import Examination

    report = Examination(brain, split="EVAL", limit=60, seed=3).sit()
    assert report.score.total >= 55
    assert report.score.wrong == 0
    assert report.score.precision == 1.0
    assert len(report.by_faculty) >= 6
    assert len(report.by_verifier) >= 3
    assert report.verified > 0
    assert set(report.to_dict()) >= {"by_faculty", "by_generator", "by_verifier", "score"}


def test_a_doing_subject_reads_its_ceiling_cold(brain) -> None:
    """A decision procedure cannot be taught, and the report says ``already`` rather than
    pretending a lesson moved it."""
    from nyxara.njp.corpusschool import CorpusSchool, SUBJECTS

    shapes = [s for s in SUBJECTS if getattr(s, "id", "") in ("mod_chain", "state")]
    transcript = CorpusSchool(seed=25, subjects=shapes).attend(brain)
    for result in transcript.results:
        assert result.pre.accuracy == 1.0, result.misses
        assert result.already


def test_the_unanswerable_subject_scores_silence_as_right(brain) -> None:
    from nyxara.njp.corpusschool import Unanswerable
    from nyxara.njp.school import Mint
    import random

    score, misses = Unanswerable().exam(brain, Mint(random.Random(25)))
    assert score.total > 0
    assert score.wrong == 0, misses


def test_the_grader_check_is_reachable_from_the_console() -> None:
    from nyxara.njp.corpusschool import main

    assert main(["--verify"]) == 0
