"""Tests for the capability benchmark harness (eval/benchmark.py).

The harness is deterministic; only the solver varies, so we drive it with scripted solvers
(no network) and assert the graders, scoring, report, and regression tracking are honest."""

from __future__ import annotations

import pytest

from nyxara.eval.benchmark import (Benchmark, BenchmarkReport, BenchmarkTask, Grader,
                                   build_arithmetic_benchmark, build_default_benchmark,
                                   build_logic_benchmark, core_solver, grade_contains,
                                   grade_exact, grade_final_numeric, grade_multiple_choice,
                                   grade_numeric)


def _task(**kw) -> BenchmarkTask:
    base = dict(id="t", prompt="p", answer="x")
    base.update(kw)
    return BenchmarkTask(**base)


# --------------------------------------------------------------------------- #
# Numeric grader
# --------------------------------------------------------------------------- #
def test_numeric_takes_last_number():
    t = _task(answer="42", grader=Grader.NUMERIC)
    assert grade_numeric("first 7, then the answer is 42", t)[0] == 1.0


def test_numeric_handles_commas_and_decimals():
    assert grade_numeric("1,234", _task(answer="1234", grader=Grader.NUMERIC))[0] == 1.0
    assert grade_numeric("3.0", _task(answer="3", grader=Grader.NUMERIC))[0] == 1.0


def test_numeric_tolerance():
    t = _task(answer="10", grader=Grader.NUMERIC, tolerance=0.5)
    assert grade_numeric("10.4", t)[0] == 1.0
    assert grade_numeric("11", t)[0] == 0.0


def test_numeric_no_number_scores_zero():
    assert grade_numeric("no digits here", _task(answer="1", grader=Grader.NUMERIC))[0] == 0.0


# --------------------------------------------------------------------------- #
# Robust final-answer numeric grader (numeric_final)
# --------------------------------------------------------------------------- #
def test_final_numeric_prefers_marked_answer_over_trailing_number():
    # The working ends on a wrong number, but the solver MARKS its real answer earlier — a bare
    # last-number grader would mark this wrong; numeric_final reads the stated conclusion.
    t = _task(answer="42", grader=Grader.NUMERIC_FINAL)
    assert grade_final_numeric("The answer is 42. (scratch work below: 7, 6, 99)", t)[0] == 1.0
    assert grade_final_numeric("... so 12 + 30 = 42", t)[0] == 1.0
    assert grade_final_numeric("steps: 100, 58 ... #### 42", t)[0] == 1.0


def test_final_numeric_resists_a_gamed_trailing_number():
    # A confident wrong final marker must FAIL even if the correct value appears earlier as working.
    t = _task(answer="42", grader=Grader.NUMERIC_FINAL)
    assert grade_final_numeric("we compute 42 along the way; answer: 99", t)[0] == 0.0


def test_final_numeric_falls_back_to_last_number_when_unmarked():
    t = _task(answer="42", grader=Grader.NUMERIC_FINAL)
    assert grade_final_numeric("first 7, then 42", t)[0] == 1.0
    assert grade_final_numeric("no digits here", t)[0] == 0.0


def test_final_numeric_honours_tolerance_and_commas():
    t = _task(answer="1234", grader=Grader.NUMERIC_FINAL)
    assert grade_final_numeric("answer: 1,234", t)[0] == 1.0
    t2 = _task(answer="10", grader=Grader.NUMERIC_FINAL, tolerance=0.5)
    assert grade_final_numeric("= 10.4", t2)[0] == 1.0
    assert grade_final_numeric("= 11", t2)[0] == 0.0


def test_final_numeric_registered_in_grader_registry():
    assert Grader.get(Grader.NUMERIC_FINAL) is grade_final_numeric


# --------------------------------------------------------------------------- #
# Exact / contains graders
# --------------------------------------------------------------------------- #
def test_exact_match_with_alias_and_case():
    t = _task(answer="Paris", aliases=["City of Light"])
    assert grade_exact("  paris ", t)[0] == 1.0
    assert grade_exact("city of light", t)[0] == 1.0
    assert grade_exact("London", t)[0] == 0.0


def test_contains_requires_all_phrases():
    t = _task(answer="alpha", aliases=["beta"], grader=Grader.CONTAINS)
    assert grade_contains("alpha and beta", t)[0] == 1.0
    assert grade_contains("only alpha", t)[0] == 0.0


# --------------------------------------------------------------------------- #
# Multiple-choice grader (robust against prompt echo)
# --------------------------------------------------------------------------- #
def test_mc_explicit_marker():
    t = _task(answer="B", grader=Grader.MULTIPLE_CHOICE, choices=["yes", "no"])
    assert grade_multiple_choice("The answer is B.", t)[0] == 1.0
    assert grade_multiple_choice("I choose A", t)[0] == 0.0


def test_mc_bare_letter():
    t = _task(answer="C", grader=Grader.MULTIPLE_CHOICE, choices=["x", "y", "z"])
    assert grade_multiple_choice("C", t)[0] == 1.0
    assert grade_multiple_choice("C) z", t)[0] == 1.0


def test_mc_echo_of_all_options_is_not_a_choice():
    # an echo that contains every option label must NOT spuriously pass
    t = _task(answer="A", grader=Grader.MULTIPLE_CHOICE, choices=["Tom", "Sam", "Ada"])
    echo = "I understand: Who is shortest? A) Tom  B) Sam  C) Ada"
    assert grade_multiple_choice(echo, t)[0] == 0.0


def test_mc_unambiguous_option_text():
    t = _task(answer="B", grader=Grader.MULTIPLE_CHOICE, choices=["cat", "dog"])
    assert grade_multiple_choice("definitely a dog", t)[0] == 1.0
    assert grade_multiple_choice("a cat and a dog", t)[0] == 0.0  # ambiguous -> no credit


# --------------------------------------------------------------------------- #
# Task dispatch
# --------------------------------------------------------------------------- #
def test_task_grade_dispatches_by_name():
    assert _task(answer="5", grader=Grader.NUMERIC).grade("5")[0] == 1.0


def test_task_custom_grader_overrides():
    t = _task(custom_grader=lambda resp, task: (1.0, "always"))
    assert t.grade("anything")[0] == 1.0


def test_unknown_grader_raises():
    with pytest.raises(ValueError):
        Grader.get("does-not-exist")


# --------------------------------------------------------------------------- #
# Runner + report
# --------------------------------------------------------------------------- #
def _oracle(bench: Benchmark):
    def solve(prompt: str) -> str:
        for t in bench.tasks():
            if t.prompt == prompt:
                return (t.answer if t.grader != Grader.MULTIPLE_CHOICE
                        else f"The answer is {t.answer}.")
        return ""
    return solve


def test_oracle_aces_default_benchmark():
    bench = build_default_benchmark()
    report = bench.run(_oracle(bench))
    assert report.accuracy == 1.0
    assert report.by_category()["arithmetic"]["n"] == 12
    assert report.by_category()["logic"]["n"] == 4


def test_wrong_solver_scores_zero():
    bench = build_logic_benchmark()
    report = bench.run(lambda p: "definitely 99999 nonsense")
    assert report.accuracy == 0.0
    assert len(report.failures()) == len(bench)


def test_solver_error_is_zero_not_crash():
    def boom(prompt: str) -> str:
        raise RuntimeError("kaboom")
    report = build_arithmetic_benchmark(3).run(boom)
    assert report.accuracy == 0.0
    assert all("solver error" in r.detail for r in report.results)


def test_regression_detection_and_roundtrip(tmp_path):
    bench = build_default_benchmark()
    good = bench.run(_oracle(bench))
    bad = bench.run(lambda p: "wrong")
    # everything the oracle passed is now lost
    assert len(bad.regression_vs(good)) == len(bench)
    # save/load round-trips
    path = str(tmp_path / "base.json")
    good.save(path)
    loaded = BenchmarkReport.load(path)
    assert loaded.accuracy == 1.0
    assert bad.regression_vs(loaded) == bad.regression_vs(good)


def test_category_filter():
    bench = build_default_benchmark()
    report = bench.run(_oracle(bench), category="logic")
    assert len(report) == 4
    assert set(report.by_category()) == {"logic"}


# --------------------------------------------------------------------------- #
# Solver adapter (end to end, offline — measures honestly, asserts no crash)
# --------------------------------------------------------------------------- #
def test_core_solver_runs_offline_without_crashing():
    bench = build_arithmetic_benchmark(2)
    report = bench.run(core_solver())
    assert len(report) == 2
    # the offline reasoner can't do arithmetic; an honest harness simply records that
    assert 0.0 <= report.accuracy <= 1.0
    assert all(isinstance(r.response, str) for r in report.results)


# --------------------------------------------------------------------------- #
# Self-model solver (Phase 0) — honest about an un-forged substrate
# --------------------------------------------------------------------------- #
def test_self_solver_is_honest_without_a_promoted_model(tmp_path):
    """No model forged yet -> own model scores 0, never crashes (the Phase-0 floor)."""
    from nyxara.eval.benchmark import self_solver
    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"   # empty -> no active model
    bench = build_arithmetic_benchmark(2)
    report = bench.run(self_solver(settings=settings))
    assert len(report) == 2
    assert report.accuracy == 0.0
    assert all(r.response == "" for r in report.results)


def test_run_router_reports_sources_offline(tmp_path):
    """With no forged model and no real teacher, the router is honest: all 'none', 0 acc."""
    from nyxara.eval.benchmark import run_router
    from nyxara.kernel.config import NyxaraSettings, Profile
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    bench = build_arithmetic_benchmark(2)
    report, sources = run_router(bench, settings=settings)
    assert len(report) == 2
    assert sources["self"] == 0
    assert sum(sources.values()) == 2


# --------------------------------------------------------------------------- #
# train / held-out split — the external-validation primitive (anti-Goodhart)
# --------------------------------------------------------------------------- #
def test_split_is_disjoint_and_covers_everything():
    bench = build_default_benchmark()
    train, holdout = bench.split(0.3)
    train_ids = {t.id for t in train.tasks()}
    holdout_ids = {t.id for t in holdout.tasks()}
    assert train_ids.isdisjoint(holdout_ids)                 # no task is in both folds
    assert train_ids | holdout_ids == {t.id for t in bench.tasks()}   # nothing dropped
    assert train.tasks() and holdout.tasks()                 # both folds usable


def test_split_is_deterministic_per_id():
    bench = build_default_benchmark()
    a_tr, a_ho = bench.split(0.3)
    b_tr, b_ho = bench.split(0.3)
    # same battery, same frac, same seed -> identical folds every cycle (held-out can't drift)
    assert {t.id for t in a_ho.tasks()} == {t.id for t in b_ho.tasks()}
    assert {t.id for t in a_tr.tasks()} == {t.id for t in b_tr.tasks()}
    # a different seed reshuffles the partition
    c_tr, c_ho = bench.split(0.3, seed=99)
    assert {t.id for t in c_ho.tasks()} != {t.id for t in a_ho.tasks()} or len(bench) < 4


def test_split_never_empties_a_fold_on_a_tiny_battery():
    tiny = build_arithmetic_benchmark(2)
    train, holdout = tiny.split(0.5)
    assert len(train) >= 1 and len(holdout) >= 1
