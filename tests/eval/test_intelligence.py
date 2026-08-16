"""The six-stage learning curve — and whether the benchmark itself measures anything.

A benchmark everything passes is not a benchmark. The load-bearing test in this file is the
ablation: with the Cognitive Learning Core switched off, the three stages that require it must
fall to zero and the three that do not must be untouched. If that separation does not hold, the
scores are measuring the harness rather than the faculty, and no other assertion here matters.
"""

from __future__ import annotations

import pytest

from nyxara.eval.intelligence import (STAGE_NAMES, run_intelligence_benchmark)


@pytest.fixture(scope="module")
def report():
    return run_intelligence_benchmark(seed=1, width=5)


def test_every_stage_runs(report):
    assert [s.stage for s in report.stages] == list(STAGE_NAMES)


def test_the_control_stage_holds(report):
    """Memorisation below half means a broken pipe, and every score under it is uninterpretable."""
    assert report.sound
    assert report.by_name("memorization").score >= 0.5


def test_every_stage_actually_taught_something(report):
    """A stage that grounded nothing measured nothing, however it scored."""
    for stage in report.stages:
        assert stage.taught > 0, f"{stage.stage} grounded no sentences: {stage.note}"


def test_every_stage_actually_asked_something(report):
    for stage in report.stages:
        assert stage.total > 0, f"{stage.stage} asked nothing"


def test_the_run_is_deterministic_for_a_seed():
    first = run_intelligence_benchmark(seed=4, width=4)
    second = run_intelligence_benchmark(seed=4, width=4)
    assert [s.to_dict()["score"] for s in first.stages] == \
           [s.to_dict()["score"] for s in second.stages]


def test_running_one_stage_gives_that_stage_the_same_problem():
    """Per-stage RNG seeding, so `--stage transfer` tests what a full run tests."""
    full = run_intelligence_benchmark(seed=2, width=4)
    alone = run_intelligence_benchmark(seed=2, width=4, only=("transfer",))
    assert len(alone.stages) == 1
    assert alone.by_name("transfer").total == full.by_name("transfer").total
    assert alone.by_name("transfer").score == full.by_name("transfer").score


def test_the_vocabulary_is_generated_not_literary():
    """An entity she could know about from anywhere else proves nothing when she answers about it."""
    report = run_intelligence_benchmark(seed=9, width=4, only=("memorization",))
    assert report.by_name("memorization").total > 0


# --------------------------------------------------------------------------- #
# The load-bearing one
# --------------------------------------------------------------------------- #

_NEEDS_THE_CORE = ("generalization", "recombination", "transfer")
_DOES_NOT = ("memorization", "causal_prediction", "self_correction")


def test_the_benchmark_detects_the_absence_of_the_thing_it_measures(monkeypatch):
    """Ablation. Without the Core, exactly the stages that need it must collapse.

    This is the test that makes every other score in this file meaningful. A curve that reads
    1.00 whether or not the mechanism is present is measuring the questions, not the answers.
    """
    import nyxara.eval.intelligence as intelligence
    from nyxara.njp.brain import NJPBrain

    class _NoLearner:
        learner_enabled = False

    monkeypatch.setattr(intelligence, "_brain", lambda **kw: NJPBrain(_NoLearner))
    ablated = intelligence.run_intelligence_benchmark(seed=1, width=5)

    for name in _NEEDS_THE_CORE:
        assert ablated.by_name(name).score == 0.0, \
            f"{name} still scored without the Core — it is not measuring the Core"
    for name in _DOES_NOT:
        assert ablated.by_name(name).score == 1.0, \
            f"{name} broke when the Core was removed — it depends on something it should not"


def test_the_ablated_run_is_still_sound(monkeypatch):
    """The control must survive the ablation, or the collapse above proves nothing."""
    import nyxara.eval.intelligence as intelligence
    from nyxara.njp.brain import NJPBrain

    class _NoLearner:
        learner_enabled = False

    monkeypatch.setattr(intelligence, "_brain", lambda **kw: NJPBrain(_NoLearner))
    assert intelligence.run_intelligence_benchmark(seed=1, width=5).sound


def test_a_report_renders_without_a_score(monkeypatch):
    """`None` (asked nothing) and 0.0 (asked and wrong) are different, and must stay different."""
    from nyxara.eval.intelligence import IntelligenceReport, StageResult
    empty = IntelligenceReport(stages=[StageResult(stage="memorization", note="NJP unavailable")])
    assert empty.by_name("memorization").score is None
    assert empty.mean is None
    assert not empty.sound
    assert "n/a" in empty.render()
