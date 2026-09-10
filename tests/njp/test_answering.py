"""V.58 — she learns to *do* a task, and the floor that says whether she learned anything.

The one number that matters here is not accuracy. A two-way task whose answers run nine to one is
0.9 for a machine that has learned nothing, so every claim in this module is a claim about beating
**that task's own majority**, and these tests exist mostly to make sure that floor cannot be
quietly avoided: by scoring a task nobody could have learned, by answering only the easy rows, or
by reading the instruction — which is identical in every row — instead of the data.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter

from nyxara.njp.shapes import Group, induce
from nyxara.njp.answering import (
    LEAST_EXAMPLES, MAX_ANSWERS, Example, TaskLearner, probe,
)
from nyxara.njp.answeringschool import Report, _shuffled, examine, shapes_by_task

TEMPLATE = ("Read the review and say whether the writer liked it."
            "\nReview text: {}\nAnswer with POS or NEG.")

GOOD = ("wonderful", "superb", "delightful", "charming", "excellent", "lovely")
BAD = ("dreadful", "abysmal", "tedious", "wretched", "lifeless", "grim")
FILLER = ("the film ran for two hours and was made last year in the usual way",
          "a picture about people who travel somewhere and then come back again",
          "another entry in a series that has now reached its fourth instalment")


def a_task(n: int = 60, skew: float = 0.5, task: str = "task_reviews"):
    """Examples whose answer is decided by one word buried in ordinary text."""
    rows = []
    for i in range(n):
        positive = (i % 10) < round(skew * 10)
        word = (GOOD if positive else BAD)[i % 6]
        text = f"{FILLER[i % 3]} and the acting was {word} throughout"
        rows.append(Example(prompt=TEMPLATE.format(text),
                            answer="POS" if positive else "NEG", task=task))
    return rows


def noise_task(seed: int, n: int = 60):
    """The same rows, with the answers assigned by a coin. There is nothing in here to find."""
    pos = set(random.Random(seed).sample(range(n), n // 2))
    return [Example(prompt=TEMPLATE.format(f"{FILLER[i % 3]} and the acting was {GOOD[i % 6]}"),
                    answer="POS" if i in pos else "NEG", task=f"task_noise_{seed}")
            for i in range(n)]


def a_group(rows) -> Group:
    return Group(task=rows[0].task, template="0",
                 prompts=tuple(r.prompt for r in rows[:6]),
                 targets=tuple(r.answer for r in rows[:6]))


# --------------------------------------------------------------------------------------------- #
#  the readings say nothing about what words mean
# --------------------------------------------------------------------------------------------- #
def test_the_readings_are_presence_and_nothing_else():
    marks = probe("the acting was dreadful throughout", ["dreadful", "wonderful"])
    assert marks["has:dreadful"] is True
    assert marks["has:wonderful"] is False
    assert marks["length"] == "short"


def test_no_word_is_privileged_over_another():
    """`terrible` is a reading here in exactly the way `the` is. Nothing knows what either means."""
    marks = probe("terrible", ["terrible", "splendid", "the"])
    assert set(marks) >= {"has:terrible", "has:splendid", "has:the"}


def test_a_task_it_has_never_seen_the_name_of_is_still_learnable():
    got = TaskLearner().learn(a_task(task="task9999_unheard_of"))
    assert got is not None and got.task == "task9999_unheard_of"


# --------------------------------------------------------------------------------------------- #
#  the floor
# --------------------------------------------------------------------------------------------- #
def test_it_beats_its_own_majority_on_a_task_that_can_be_learned():
    got = TaskLearner().learn(a_task())
    assert got.learned_something
    assert got.accuracy > got.majority


def test_noise_beats_its_floor_about_as_often_as_it_should_and_no_more():
    """The number that decides what "beat its own floor" is worth on the real dataset.

    A single task whose answers have nothing to do with its text *can* beat its majority, and
    pretending otherwise would be the easiest lie in this package to tell. With forty-eight
    readings offered and thirty positives, a rule with support four can come out pure by accident,
    and sometimes that accident helps on the held-out rows too.

    So the claim is not that noise never wins. It is that noise wins at a rate, that rate is
    measured, and any figure reported for real tasks has to be read against it — which is what
    :func:`nyxara.njp.answeringschool.examine` does with ``shuffled`` and what
    :attr:`~nyxara.njp.answeringschool.Report.above_chance` reports. Measured here: **0.175** of
    forty noise tasks, at a mean lift of **+0.013**.
    """
    got = [TaskLearner().learn(noise_task(seed)) for seed in range(40)]
    won = sum(1 for g in got if g.learned_something)
    lift = statistics.mean(g.accuracy - g.majority for g in got)
    assert 0.05 <= won / len(got) <= 0.35, won
    assert abs(lift) < 0.05, lift


def test_a_real_signal_wins_by_much_more_than_noise_does():
    """The comparison the null exists for: the same machinery, on a task that has something in it."""
    real = statistics.mean(TaskLearner().learn(a_task(task=f"t{s}")).accuracy
                           - TaskLearner().learn(a_task(task=f"t{s}")).majority for s in range(4))
    fake = statistics.mean(g.accuracy - g.majority
                           for g in (TaskLearner().learn(noise_task(s)) for s in range(40)))
    assert real > fake + 0.10, (real, fake)


def test_a_skewed_task_is_not_passed_on_accuracy_alone():
    """Nine to one. Accuracy near 0.9 here is the floor, not a result."""
    got = TaskLearner().learn(a_task(n=60, skew=0.9))
    assert got.majority >= 0.8
    assert got.learned_something == (got.accuracy > got.majority)


def test_both_columns_answer_every_question():
    """The fallback to the commonest answer is what makes the two columns comparable."""
    got = TaskLearner().learn(a_task())
    assert got.asked > 0
    assert got.answered == got.asked


# --------------------------------------------------------------------------------------------- #
#  what it refuses to claim
# --------------------------------------------------------------------------------------------- #
def test_too_few_examples_is_refused_rather_than_scored():
    assert TaskLearner().learn(a_task(n=LEAST_EXAMPLES - 1)) is None


def test_a_free_text_answer_is_refused_rather_than_scored():
    rows = [Example(prompt=TEMPLATE.format(FILLER[i % 3]), answer=f"a sentence number {i}",
                    task="task_free") for i in range(60)]
    assert TaskLearner().learn(rows) is None


def test_one_answer_is_not_a_task():
    rows = [Example(prompt=TEMPLATE.format(FILLER[i % 3]), answer="POS", task="task_one")
            for i in range(60)]
    assert TaskLearner().learn(rows) is None


def test_an_answer_space_wider_than_the_ceiling_is_refused():
    rows = [Example(prompt=TEMPLATE.format(FILLER[i % 3]), answer=f"L{i % (MAX_ANSWERS + 4)}",
                    task="task_wide") for i in range(90)]
    assert TaskLearner().learn(rows) is None


# --------------------------------------------------------------------------------------------- #
#  the shape, and what happens without it
# --------------------------------------------------------------------------------------------- #
def test_the_shape_hands_back_what_was_poured_in_not_the_instruction():
    rows = a_task()
    shape = induce(a_group(rows))
    assert shape is not None
    got = TaskLearner().fields(shape, rows[0].prompt)
    assert "Read the review" not in got
    assert "wonderful" in got


def test_without_a_shape_it_reads_the_whole_prompt_and_says_so():
    rows = a_task()
    assert TaskLearner().fields(None, rows[0].prompt) == rows[0].prompt


# --------------------------------------------------------------------------------------------- #
#  the counting
# --------------------------------------------------------------------------------------------- #
def test_free_text_tasks_are_counted_as_not_attempted_not_as_failures():
    out = Report()
    out.free_text = 3
    out.in_scope = 2
    out.won = 1
    assert out.tasks == 5
    assert out.beat_majority == 0.5      # of what was attempted, not of everything seen


def test_lift_is_accuracy_over_its_own_floor():
    got = TaskLearner().learn(a_task())
    out = Report(in_scope=1, won=int(got.learned_something), results=[got])
    assert out.lift == round(out.accuracy - out.majority, 4)


def test_the_null_keeps_everything_about_a_task_except_its_signal():
    """Shuffling answers must not change the task's size, its answer space, or its skew."""
    rows = a_task()
    fake = _shuffled(rows, 58)
    assert len(fake) == len(rows)
    assert Counter(e.answer for e in fake) == Counter(e.answer for e in rows)
    assert [e.prompt for e in fake] == [e.prompt for e in rows]


def test_a_report_is_worth_the_distance_from_its_null():
    real = Report(in_scope=100, won=23)
    null = Report(in_scope=100, won=18)
    assert real.above_chance(null) == 0.05


def test_a_learned_task_carries_the_readings_it_was_induced_over():
    """Without this, asking her again rebuilds a different vocabulary and no rule can fire."""
    got = TaskLearner().learn(a_task())
    assert got.vocabulary
    assert all(r.terms[0][0].replace("has:", "") in set(got.vocabulary) or
               not r.terms[0][0].startswith("has:") for r in got.rules)


def test_asking_again_later_gives_the_same_answer_as_during_the_exam():
    """The defect this pins: the rules fired at exam time and silently stopped firing afterwards."""
    rows = a_task()
    engine = TaskLearner()
    got = engine.learn(rows)
    signal = [r for r in rows if r.answer == "NEG"][0]
    assert engine.answer(got, signal.prompt) == engine.answer(got, signal.prompt, got.vocabulary)
    said = [engine.answer(got, r.prompt) for r in rows]
    assert len(set(said)) > 1, "every answer the same means no rule fired"
