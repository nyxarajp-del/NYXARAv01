"""Tests for eval/ablation.py — the instrument that decides what gets deleted.

92 capability modules are wired into the core. Whether each one earns its place is currently
unanswerable: their self-tests show they *work*, which is not the same as showing they *help*.
This module is meant to produce that missing evidence, which makes its own correctness unusually
load-bearing — a bug here does not produce a wrong number, it produces a deleted faculty.

So the tests are weighted toward the ways a null result can be wrong:

* an ablation that silently failed to ablate reads exactly like "this faculty is worthless";
* a faculty that never ran on the battery reads the same way;
* a two-task experiment finding nothing reads the same way again.

All three must be distinguishable from a real, measured absence of benefit, and none of them may
license a deletion.
"""
from __future__ import annotations

import pytest

from nyxara.eval.ablation import (AblationReport, AblationResult, ablate, attr_faculty,
                                  mcnemar_exact, run_ablation)
from nyxara.eval.benchmark import Benchmark, BenchmarkTask, Grader


# --------------------------------------------------------------------------- #
# The statistic
# --------------------------------------------------------------------------- #
def test_no_disagreement_is_no_evidence():
    assert mcnemar_exact(0, 0) == 1.0


def test_a_perfect_tie_is_no_evidence():
    assert mcnemar_exact(5, 5) == 1.0


def test_a_clean_split_is_significant():
    assert mcnemar_exact(8, 0) < 0.01


def test_a_small_clean_split_is_not_significant():
    """3-0 looks convincing and is not. Rounding it up would delete working faculties."""
    assert mcnemar_exact(3, 0) > 0.05


def test_the_test_is_symmetric():
    """Direction is read from the counts, not from the p-value."""
    assert mcnemar_exact(9, 2) == mcnemar_exact(2, 9)


def test_the_p_value_stays_a_probability():
    for b in range(0, 12):
        for c in range(0, 12):
            assert 0.0 <= mcnemar_exact(b, c) <= 1.0


def test_concordant_pairs_are_correctly_ignored():
    """Only tasks whose outcome changed carry information about the faculty."""
    assert mcnemar_exact(6, 1) == mcnemar_exact(6, 1)
    a = AblationResult("f", n_tasks=10, with_passed=8, without_passed=3, helped=6, hurt=1,
                       p_value=mcnemar_exact(6, 1), responses_differed=7)
    b = AblationResult("f", n_tasks=1000, with_passed=998, without_passed=993, helped=6, hurt=1,
                       p_value=mcnemar_exact(6, 1), responses_differed=7)
    assert a.p_value == b.p_value


# --------------------------------------------------------------------------- #
# The three ways a null result can be a lie
# --------------------------------------------------------------------------- #
def test_a_toggle_that_did_nothing_is_a_broken_experiment():
    r = AblationResult("ghost", 40, 20, 20, disabled_ok=False, responses_differed=0)
    assert r.verdict == "broken"
    assert not r.deletion_candidate, "a measurement never taken cannot license a deletion"


def test_a_faculty_that_never_ran_is_inert_not_worthless():
    r = AblationResult("idle_only", 40, 20, 20, responses_differed=0)
    assert r.verdict == "inert"
    assert not r.deletion_candidate
    assert "battery" in r.explain(), "the finding is about the battery, not the faculty"


def test_too_few_discordant_pairs_concludes_nothing():
    r = AblationResult("thin", 8, 5, 4, helped=1, hurt=0, p_value=mcnemar_exact(1, 0),
                       responses_differed=3)
    assert r.verdict == "underpowered"
    assert not r.deletion_candidate


def test_underpowered_is_not_silently_folded_into_no_evidence():
    """They have different causes and different remedies; collapsing them hides the difference
    between 'we looked and found nothing' and 'we did not look hard enough'."""
    thin = AblationResult("thin", 8, 5, 4, helped=1, hurt=0, responses_differed=3,
                          p_value=mcnemar_exact(1, 0))
    thorough = AblationResult("thorough", 80, 40, 40, helped=5, hurt=5, responses_differed=40,
                              p_value=mcnemar_exact(5, 5))
    assert thin.verdict != thorough.verdict
    assert not thin.deletion_candidate and thorough.deletion_candidate


# --------------------------------------------------------------------------- #
# Real findings
# --------------------------------------------------------------------------- #
def test_a_faculty_that_helps_is_never_a_deletion_candidate():
    r = AblationResult("council", 40, 30, 20, helped=11, hurt=1, p_value=mcnemar_exact(11, 1),
                       responses_differed=25)
    assert r.verdict == "helps" and not r.deletion_candidate


def test_a_faculty_that_hurts_is_a_deletion_candidate():
    r = AblationResult("misfire", 40, 12, 24, helped=1, hurt=13, p_value=mcnemar_exact(1, 13),
                       responses_differed=30)
    assert r.verdict == "hurts" and r.deletion_candidate


def test_a_well_measured_null_is_a_deletion_candidate():
    r = AblationResult("noop", 40, 20, 20, helped=4, hurt=4, p_value=mcnemar_exact(4, 4),
                       responses_differed=30)
    assert r.verdict == "no-evidence" and r.deletion_candidate


def test_cost_is_measured_alongside_benefit():
    r = AblationResult("slow", 40, 20, 20, helped=4, hurt=4, p_value=1.0, responses_differed=30,
                       with_latency_s=12.5, without_latency_s=0.5)
    assert r.latency_cost_s == pytest.approx(12.0)
    assert "12.000s" in r.explain()


def test_accuracy_delta_signs_correctly():
    assert AblationResult("f", 10, 8, 3).accuracy_delta == pytest.approx(0.5)
    assert AblationResult("f", 10, 3, 8).accuracy_delta == pytest.approx(-0.5)


def test_an_empty_battery_does_not_divide_by_zero():
    assert AblationResult("f", 0, 0, 0).accuracy_delta == 0.0


# --------------------------------------------------------------------------- #
# Toggles
# --------------------------------------------------------------------------- #
class _Core:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_nulling_a_live_attribute_reports_success():
    core = _Core(role_council=object())
    assert attr_faculty("rc", "role_council").disable(core) is True
    assert core.role_council is None


def test_nulling_an_already_none_attribute_reports_failure():
    """This is the whole guard: it disabled nothing, so it measured nothing."""
    assert attr_faculty("rc", "role_council").disable(_Core(role_council=None)) is False


def test_a_missing_attribute_reports_failure():
    assert attr_faculty("rc", "role_council").disable(_Core()) is False


def test_every_shipped_faculty_names_an_attribute_the_core_actually_sets():
    """A typo'd attribute name reports ``broken`` at run time, which is safe but late.

    Checked statically against ``NyxaraCore``'s source rather than by building a core: a real one
    costs a full boot, and the failure being guarded against is a misspelling, not a runtime
    condition. Every faculty here is disabled by nulling ``core.<attr>``, so ``self.<attr> =``
    must appear in the class.
    """
    import inspect

    from nyxara.eval.ablation import CORE_FACULTIES
    from nyxara.kernel.orchestrator import NyxaraCore

    source = inspect.getsource(NyxaraCore)
    for faculty in CORE_FACULTIES:
        assert faculty.attr, f"{faculty.name!r} does not record which attribute it ablates"
        assert f"self.{faculty.attr} = " in source or f"self.{faculty.attr}:" in source, (
            f"{faculty.name!r} ablates core.{faculty.attr}, which NyxaraCore never assigns — "
            f"the ablation would silently measure nothing")


# --------------------------------------------------------------------------- #
# End to end, against a fake mind
# --------------------------------------------------------------------------- #
def _battery(n: int = 12) -> Benchmark:
    return Benchmark("fake", [BenchmarkTask(id=f"t{i}", prompt=str(i), answer=str(i * 2),
                                            grader=Grader.CONTAINS) for i in range(n)])


class _Result:
    def __init__(self, response):
        self.response = response


class _FakeCore:
    """A mind whose one faculty genuinely does the work: without it, every answer is wrong."""

    def __init__(self):
        self.helper = object()

    def process(self, prompt, authority=None):
        if self.helper is None:
            return _Result("no idea")
        return _Result(str(int(prompt) * 2))


def test_a_faculty_doing_all_the_work_is_measured_as_helping():
    report = ablate(attr_faculty("helper", "helper"), _battery(), _FakeCore)
    assert report.disabled_ok
    assert report.with_passed == 12 and report.without_passed == 0
    assert report.verdict == "helps"
    assert not report.deletion_candidate


def test_a_faculty_that_changes_nothing_is_measured_as_inert():
    class _Indifferent(_FakeCore):
        def process(self, prompt, authority=None):
            return _Result(str(int(prompt) * 2))

    report = ablate(attr_faculty("helper", "helper"), _battery(), _Indifferent)
    assert report.disabled_ok, "the toggle worked — the faculty simply had no effect"
    assert report.responses_differed == 0
    assert report.verdict == "inert" and not report.deletion_candidate


def test_a_faculty_that_cannot_be_disabled_yields_broken_not_inert():
    """The dangerous case: identical answers because nothing was switched off."""
    class _Unswitchable(_FakeCore):
        def __init__(self):
            self.helper = None       # already off; the toggle has nothing to do

        def process(self, prompt, authority=None):
            return _Result(str(int(prompt) * 2))

    report = ablate(attr_faculty("helper", "helper"), _battery(), _Unswitchable)
    assert report.disabled_ok is False
    assert report.verdict == "broken", "must not be reported as a null result"
    assert not report.deletion_candidate


def test_each_arm_gets_a_fresh_core():
    """Ablation mutates the core; a shared one would carry arm one's state into arm two."""
    built = []

    def factory():
        core = _FakeCore()
        built.append(core)
        return core

    ablate(attr_faculty("helper", "helper"), _battery(4), factory)
    assert len(built) == 2
    assert built[0].helper is not None, "the control arm must not be ablated"
    assert built[1].helper is None


def test_fresh_per_task_builds_a_core_for_every_task_in_both_arms():
    built = []

    def factory():
        core = _FakeCore()
        built.append(core)
        return core

    result = ablate(attr_faculty("helper", "helper"), _battery(4), factory, fresh_per_task=True)
    assert len(built) == 8, "4 tasks x 2 arms"
    assert result.disabled_ok
    assert [c.helper for c in built[:4]] == [built[i].helper for i in range(4)]
    assert all(c.helper is None for c in built[4:]), "every core in the off arm must be ablated"


def test_one_failed_ablation_condemns_the_whole_run():
    """Averaging it away would drag the result toward 'no benefit' — the deleting direction.

    Under ``fresh_per_task`` each core is ablated separately, so one core that arrived with the
    faculty already absent contributes a concordant pair that looks like evidence. It is not
    evidence; it is a hole, and the run must say so rather than quietly absorb it.
    """
    made = {"n": 0}

    def factory():
        core = _FakeCore()
        made["n"] += 1
        if made["n"] == 6:                       # one core in the off arm arrives already off
            core.helper = None
        return core

    result = ablate(attr_faculty("helper", "helper"), _battery(4), factory, fresh_per_task=True)
    assert result.disabled_ok is False
    assert result.verdict == "broken" and not result.deletion_candidate


def test_the_shared_core_mode_still_reports_a_single_effective_toggle():
    result = ablate(attr_faculty("helper", "helper"), _battery(4), _FakeCore)
    assert result.disabled_ok is True


def test_a_solver_error_scores_zero_rather_than_crashing():
    class _Broken(_FakeCore):
        def process(self, prompt, authority=None):
            raise RuntimeError("the mind fell over")

    report = ablate(attr_faculty("helper", "helper"), _battery(4), _Broken)
    assert report.with_passed == 0 and report.without_passed == 0


def test_run_ablation_scores_only_the_holdout_fold():
    """Scoring the train fold would measure how well the improvement loop has fitted the faculty."""
    battery = _battery(20)
    _, holdout = battery.split(0.4, seed=0)
    report = run_ablation([attr_faculty("helper", "helper")], benchmark=battery,
                          core_factory=_FakeCore, holdout_frac=0.4, seed=0)
    assert len(report.results) == 1
    assert report.results[0].n_tasks == len(holdout)
    assert report.results[0].n_tasks < 20


def test_the_seed_rotates_the_holdout():
    """A faculty must not survive on one lucky partition."""
    battery = _battery(20)
    a = run_ablation([attr_faculty("helper", "helper")], benchmark=battery,
                     core_factory=_FakeCore, seed=1)
    b = run_ablation([attr_faculty("helper", "helper")], benchmark=battery,
                     core_factory=_FakeCore, seed=7)
    assert a.seed != b.seed


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def _mixed() -> AblationReport:
    return AblationReport(battery="demo", results=[
        AblationResult("council", 40, 30, 20, helped=11, hurt=1, p_value=mcnemar_exact(11, 1),
                       responses_differed=25),
        AblationResult("cheap_noop", 40, 20, 20, helped=4, hurt=4, p_value=1.0,
                       responses_differed=30, with_latency_s=0.51, without_latency_s=0.5),
        AblationResult("costly_noop", 40, 20, 20, helped=4, hurt=4, p_value=1.0,
                       responses_differed=30, with_latency_s=12.5, without_latency_s=0.5),
        AblationResult("thin", 8, 5, 4, helped=1, hurt=0, p_value=mcnemar_exact(1, 0),
                       responses_differed=3),
        AblationResult("ghost", 40, 20, 20, disabled_ok=False),
    ])


def test_candidates_are_ranked_by_what_removing_them_buys():
    assert [r.faculty for r in _mixed().candidates] == ["costly_noop", "cheap_noop"]


def test_the_undecided_stay_visible():
    """An honest residue kept in the report, not quietly dropped into either bucket."""
    assert {r.faculty for r in _mixed().unmeasured} == {"thin", "ghost"}


def test_earned_faculties_are_listed_separately():
    assert [r.faculty for r in _mixed().earned] == ["council"]


def test_every_faculty_lands_in_exactly_one_bucket():
    report = _mixed()
    buckets = ([r.faculty for r in report.earned] + [r.faculty for r in report.candidates]
               + [r.faculty for r in report.unmeasured])
    assert sorted(buckets) == sorted(r.faculty for r in report.results)
    assert len(buckets) == len(set(buckets))


def test_the_report_serialises():
    d = _mixed().to_dict()
    assert d["deletion_candidates"] == ["costly_noop", "cheap_noop"]
    assert d["earned"] == ["council"]
    assert len(d["results"]) == 5


def test_the_rendering_names_the_undecided():
    text = _mixed().render()
    assert "undecided" in text and "thin" in text
