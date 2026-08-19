"""The measurement that decides whether any of the distillation worked.

Its two failure modes are opposite and both fatal: reporting ``distilled`` from an experiment
where the teacher was never actually removed, and filing a clean dependency result under
"battery too small". Most of what follows is about those two.
"""

from __future__ import annotations

from nyxara.eval.teacher_removal import (
    TeacherRemoval, run_teacher_removal,
)

TASKS = list(range(40))


def _score(_task, answer):
    return answer == "right"


def _solver(hit_rate_out_of_ten):
    def _solve(task):
        return "right" if task % 10 < hit_rate_out_of_ten else f"wrong{task}"
    return _solve


class _RemovedCore:
    class njp:
        cortex = None
        shadow = None
        fabric = None


class _TeacherStillLoaded:
    class njp:
        class cortex:
            @staticmethod
            def available():
                return True
        shadow = None
        fabric = None


def _run(a, b, c, core=_RemovedCore(), tasks=TASKS):
    return run_teacher_removal(
        tasks, battery="unit", solve_a=_solver(a), solve_b=_solver(b), solve_c=_solver(c),
        score=_score, core_c=core)


# --------------------------------------------------------------------------- #
# The verdicts
# --------------------------------------------------------------------------- #
def test_capability_that_survives_removal_is_distilled():
    result = _run(3, 9, 9)
    assert result.verdict == "distilled"
    assert result.c_vs_a["delta"] > 0


def test_capability_that_vanishes_with_the_teacher_is_dependent():
    """C tracked A exactly while the teacher demonstrably moved 24 tasks. That is the sharpest
    possible dependency finding, and it must not be filed as 'underpowered'."""
    result = _run(3, 9, 3)
    assert result.verdict == "dependent"
    assert "dependency, not an ability" in result.explain()


def test_a_genuinely_small_battery_still_reports_underpowered():
    assert _run(3, 4, 3, tasks=list(range(6))).verdict == "underpowered"


def test_partial_transfer_is_named_rather_than_rounded_to_success():
    assert _run(2, 9, 6).verdict == "partial"


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #
def test_a_removal_that_did_not_remove_can_never_report_distilled():
    result = _run(3, 9, 9, core=_TeacherStillLoaded())
    assert result.verdict == "broken"
    assert "still available" in result.removal_detail


def test_a_lingering_graft_counts_as_the_teacher_still_present():
    class _StillGrafted:
        class njp:
            cortex = None
            shadow = None

            class fabric:
                grafts = ["blk.0.ffn_gate"]

    assert _run(3, 9, 9, core=_StillGrafted()).verdict == "broken"


def test_a_lingering_shadow_teacher_counts_too():
    class _StillShadowed:
        class njp:
            cortex = None

            class shadow_cognition:
                teacher = object()
            fabric = None

    assert _run(3, 9, 9, core=_StillShadowed()).verdict == "broken"


def test_no_core_at_all_is_unverified_not_assumed_clean():
    result = _run(3, 9, 9, core=None)
    assert result.verdict == "broken"
    assert "could not be verified" in result.removal_detail


def test_identical_arms_are_inert_a_finding_about_the_battery():
    result = run_teacher_removal(
        TASKS, battery="unit", solve_a=lambda t: "right", solve_b=lambda t: "right",
        solve_c=lambda t: "right", score=_score, core_c=_RemovedCore())
    assert result.verdict == "inert"
    assert "about the battery" in result.explain()


def test_a_crashing_solver_fails_its_tasks_without_losing_the_experiment():
    def _boom(_task):
        raise RuntimeError("solver died")

    result = run_teacher_removal(
        TASKS, battery="unit", solve_a=_boom, solve_b=_solver(9), solve_c=_solver(9),
        score=_score, core_c=_RemovedCore())
    assert result.a.n == len(TASKS)
    assert result.a.score == 0.0
    assert result.verdict in ("distilled", "partial")


def test_check_removed_reports_a_clean_core_as_verified():
    ok, detail = TeacherRemoval.check_removed(_RemovedCore())
    assert ok is True
    assert "no cortex" in detail


# --------------------------------------------------------------------------- #
# The scoring adapter — where a misread contract silently perfects every arm
# --------------------------------------------------------------------------- #
def test_an_empty_answer_does_not_pass_a_benchmark_task():
    """The bug this exists to prevent, caught on the first live run: `Benchmark.grade` returns a
    `(score, reason)` TUPLE, and every non-empty tuple is truthy. `bool(task.grade(answer))`
    therefore passes every task including the failures — arm A scored 20/20 on a battery where
    nineteen of twenty answers were the empty string, and the verdict would have been a confident
    `inert` built on nothing."""
    from nyxara.eval.general_novel import build_general_novel_benchmark
    from nyxara.eval.teacher_removal import score_benchmark_task

    task = build_general_novel_benchmark().tasks()[0]
    assert bool(task.grade("")), "the naive read is truthy — that is the trap"
    assert score_benchmark_task(task, "") is False
    assert score_benchmark_task(task, "nonsense") is False


def test_the_adapter_accepts_a_correct_answer():
    from nyxara.eval.general_novel import build_general_novel_benchmark
    from nyxara.eval.teacher_removal import score_benchmark_task

    task = build_general_novel_benchmark().tasks()[0]   # x + y = 7, x - y = 3
    assert score_benchmark_task(task, "x=5, y=2") is True


def test_the_adapter_handles_the_other_grade_shapes():
    from nyxara.eval.teacher_removal import score_benchmark_task

    class _Tuple:
        def grade(self, a):
            return (1.0 if a == "ok" else 0.0, "reason")

    class _Object:
        def grade(self, a):
            return type("V", (), {"passed": a == "ok"})()

    class _Scalar:
        def grade(self, a):
            return 1.0 if a == "ok" else 0.0

    class _Raises:
        def grade(self, a):
            raise RuntimeError("grader exploded")

    for cls in (_Tuple, _Object, _Scalar):
        assert score_benchmark_task(cls(), "ok") is True
        assert score_benchmark_task(cls(), "no") is False
    assert score_benchmark_task(_Raises(), "ok") is False
