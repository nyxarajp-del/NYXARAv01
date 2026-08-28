"""The acceptance test's own acceptance test.

Two things have to hold for `eval/unseen` to be worth running, and both were false in its first
version:

* the world must be **invented at run time**, so no corpus can contain it and no amount of data
  can move the score;
* a stage the run could not put must report ``measurable=False`` and stay out of the denominator,
  never score as a pass. The first version compared "two readings" against "two readings" and read
  transfer 5/5 — a stage passing for a reason that had nothing to do with transfer.

`test_transfer_is_honest_about_the_floor` pins the finding: a line through two points always has a
sign, so readings-to-correct-sign floors at two and the transfer question usually cannot be asked
at all. When a mechanism arrives that makes it askable, this test is what says so.
"""

from __future__ import annotations

import random

from nyxara.eval.unseen import Stage, UnseenReport, World, _world, run, run_once


def test_the_world_is_invented_and_seed_stable():
    first = _world(random.Random(7), tag="un")
    again = _world(random.Random(7), tag="un")
    assert (first.cause, first.effect) == (again.cause, again.effect)
    other = _world(random.Random(8), tag="un")
    assert other.cause != first.cause


def test_the_two_worlds_of_one_run_share_no_vocabulary():
    rng = random.Random(3)
    first, second = _world(rng, tag="un"), _world(rng, tag="tr")
    assert {first.cause, first.effect}.isdisjoint({second.cause, second.effect})


def test_the_prior_disagrees_with_the_world_by_construction():
    """A prediction that is already right attributes nothing and corrects nothing."""
    for seed in range(6):
        world = _world(random.Random(seed), tag="un")
        assert world.belief_slope == -world.slope
        assert (world.belief(4.0) - world.belief(1.0)) * (world.slope) < 0


def test_an_unmeasurable_stage_stays_out_of_the_denominator():
    report = UnseenReport(stages=[Stage("a", True), Stage("b", False, measurable=False)])
    assert report.score == 1.0, "a question that could not be asked is not a failure"


def test_a_failed_stage_does_count():
    report = UnseenReport(stages=[Stage("a", True), Stage("b", False)])
    assert report.score == 0.5


def test_failing_is_a_scored_stage():
    """Recovery is the capability under test, so a lucky first-time hit scores lower."""
    report = run_once(seed=0)
    by_name = {s.name: s for s in report.stages}
    assert by_name["fail"].achieved, "the prior is wrong on purpose; the miss is the point"


def test_the_recovery_loop_runs_end_to_end():
    report = run_once(seed=1)
    by_name = {s.name: s.achieved for s in report.stages}
    for stage in ("model", "predict", "fail", "diagnose", "revise"):
        assert by_name[stage], f"{stage} did not happen"
    assert report.recovered


def test_transfer_is_honest_about_the_floor():
    """A line through two points always has a sign, so the count floors and cannot be asked.

    This is the architectural finding, not a tuning failure — and it is what mechanisms ③–⑩ have
    to change. If transfer ever becomes measurable across several seeds, this assertion is the one
    that should be revisited rather than deleted.
    """
    results = [run_once(seed=s) for s in range(4)]
    transfers = [next(x for x in r.stages if x.name == "transfer") for r in results]
    assert all(t.achieved is False for t in transfers)
    assert any(not t.measurable for t in transfers), (
        "if every run can now ask the transfer question, something real has changed")


def test_run_reports_every_stage():
    payload = run(seeds=3)
    assert payload["runs"] == 3
    assert set(payload["by_stage"]) == {"model", "predict", "fail", "diagnose",
                                        "revise", "transfer"}
    assert payload["recovered"] == 3
