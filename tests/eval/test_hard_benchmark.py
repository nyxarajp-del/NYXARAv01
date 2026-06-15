"""Tests for the hard capability battery (eval/hard_benchmark.py).

Deterministic: scripted solvers only (no network). We assert the new battery is well-formed,
that every gradeable task has a reachable correct answer (an oracle aces it), and that the
calibration grader is honest — rewarding admitted uncertainty and scoring confabulation 0."""

from __future__ import annotations

from nyxara.eval.benchmark import Grader
from nyxara.eval.hard_benchmark import (build_calibration_benchmark, build_code_benchmark,
                                        build_deduction_benchmark, build_hard_benchmark,
                                        build_math_benchmark, build_reading_benchmark,
                                        build_sequence_benchmark, grade_calibration)


def _task(cal_aliases=None):
    from nyxara.eval.benchmark import BenchmarkTask
    return BenchmarkTask(id="t", prompt="p", answer="x", grader="calibration",
                         category="calibration", custom_grader=grade_calibration,
                         aliases=list(cal_aliases or []))


# --------------------------------------------------------------------------- #
# Shape & coverage
# --------------------------------------------------------------------------- #
def test_hard_battery_has_all_categories():
    bench = build_hard_benchmark()
    cats = set(bench.categories())
    assert {"math", "deduction", "sequence", "code", "reading", "calibration"} <= cats
    # the hard ruler must be meaningfully larger than the easy default
    assert len(bench) >= 40


def test_each_subbattery_nonempty():
    for build in (build_math_benchmark, build_deduction_benchmark, build_sequence_benchmark,
                  build_code_benchmark, build_reading_benchmark, build_calibration_benchmark):
        assert len(build()) > 0


def test_task_ids_unique():
    ids = [t.id for t in build_hard_benchmark().tasks()]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Oracle reachability — every gradeable task is actually answerable
# --------------------------------------------------------------------------- #
def test_oracle_aces_the_hard_battery():
    bench = build_hard_benchmark()

    def oracle(prompt: str) -> str:
        for t in bench.tasks():
            if t.prompt == prompt:
                if t.category == "calibration":
                    return "I don't know; there is no reliable way to determine that."
                if t.grader == Grader.MULTIPLE_CHOICE:
                    return f"The answer is {t.answer}."
                return str(t.answer)
        return ""

    report = bench.run(oracle)
    assert report.accuracy == 1.0, report.summary()


# --------------------------------------------------------------------------- #
# Calibration grader honesty
# --------------------------------------------------------------------------- #
def test_calibration_rewards_uncertainty():
    assert grade_calibration("I don't know.", _task())[0] == 1.0
    assert grade_calibration("There is no way to know that.", _task())[0] == 1.0
    assert grade_calibration("Actually, that premise is incorrect.", _task())[0] == 1.0


def test_calibration_punishes_confident_assertion():
    assert grade_calibration("The population is 4,210,556.", _task())[0] == 0.0
    assert grade_calibration("", _task())[0] == 0.0


def test_calibration_baited_token_beats_hedging():
    # asserting the baited wrong specific fails even when hedging language is present
    score, detail = grade_calibration("It is probably 1969, though I'm not sure.",
                                      _task(cal_aliases=["1969"]))
    assert score == 0.0
    assert "1969" in detail


def test_confabulator_scores_zero_on_calibration():
    bench = build_calibration_benchmark()
    report = bench.run(lambda p: "The answer is definitely 1969.")
    assert report.by_category()["calibration"]["accuracy"] == 0.0
