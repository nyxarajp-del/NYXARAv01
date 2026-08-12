"""Tests for nyxara.mind.metacontrol — first-class metacognitive compute allocation.

The controller must (a) estimate difficulty deterministically from NYXARA's own signals — no
LLM anywhere, (b) order easy before hard, (c) map difficulty to a monotone compute budget where
an easy low-stakes turn is genuinely ONE forward pass and a hard turn earns the deep search,
(d) never fast-path a risky turn, (e) close the loop — lived outcomes correct an over-confident
estimator and raise its confidence bar, (f) count an escalated-but-successful turn as
under-allocation, and (g) persist calibration across restarts.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.metacontrol import (RUNG_FAST, ComputeBudget, DifficultyEstimate,
                                     DifficultySignals, MetacognitiveController)

EASY = "hi there"
HARD = ("prove that every finite division ring is commutative, deriving the theorem "
        "from first principles with a rigorous argument")


def _controller(**kwargs) -> MetacognitiveController:
    s = NyxaraSettings.for_profile(Profile.TEST)
    kwargs.setdefault("path", None)
    kwargs.setdefault("effort_memory", False)  # falsy but not None: no shared store in units
    mc = MetacognitiveController(settings=s, **kwargs)
    mc.effort_memory = None
    return mc


def test_estimate_is_deterministic():
    mc = _controller()
    a = mc.estimate(EASY, novelty=0.4, recall_strength=0.6)
    b = mc.estimate(EASY, novelty=0.4, recall_strength=0.6)
    assert a.to_dict() == b.to_dict()


def test_estimate_orders_easy_before_hard():
    mc = _controller()
    easy = mc.estimate(EASY)
    hard = mc.estimate(HARD, novelty=0.9, recall_strength=0.0)
    assert easy.difficulty < hard.difficulty


def _sweep(mc):
    """One budget per difficulty step, 0.0 … 1.0, at fixed moderate stakes."""
    for i in range(11):
        d = i / 10.0
        yield mc.allocate(DifficultyEstimate(difficulty=d, calibrated=d,
                                             signature="question/short/normal",
                                             signals=DifficultySignals(stakes=0.3)))


def test_budget_mapping_is_monotone():
    mc = _controller()
    prev = None
    for b in _sweep(mc):
        if prev is not None:
            assert b.entry_rung >= prev.entry_rung
            assert b.max_rung >= prev.max_rung
            assert b.samples >= prev.samples
            assert b.max_seconds >= prev.max_seconds
        prev = b


class _Ceiling:
    """A hardware-scaled ceiling, as growth/effective_scale.scaled_budget would return one."""

    def __init__(self, max_rung, samples, max_seconds):
        self.max_rung, self.samples, self.max_seconds = max_rung, samples, max_seconds


def test_budget_mapping_stays_monotone_under_a_scaled_down_ceiling():
    """The regression: a modest machine scales the ceiling BELOW the static band table.

    The moderate/hard rows are written off the static config (``samples`` and ``samples + 2``)
    while the extreme row takes the scaled ceiling — so a ceiling under ``samples + 2`` used to
    make the hard band buy MORE sampling than the extreme band above it.
    """
    for ceil_samples in range(1, 9):
        mc = _controller(ceiling=_Ceiling(max_rung=4, samples=ceil_samples, max_seconds=60.0))
        prev = None
        for b in _sweep(mc):
            if prev is not None:
                assert b.samples >= prev.samples, f"inverted at ceiling={ceil_samples}"
            prev = b


def test_no_band_outspends_the_machine_ceiling():
    """A ceiling that is not enforced is not a ceiling."""
    for ceil_samples, ceil_rung in ((1, 1), (2, 2), (3, 1), (6, 4)):
        mc = _controller(ceiling=_Ceiling(max_rung=ceil_rung, samples=ceil_samples,
                                          max_seconds=60.0))
        for b in _sweep(mc):
            assert b.samples <= ceil_samples
            assert b.max_rung <= ceil_rung


def test_entry_rung_never_exceeds_the_max_rung():
    """Entering above the ceiling leaves the ladder nowhere to start — a contradiction, not a budget."""
    for ceil_rung in range(1, 5):
        mc = _controller(ceiling=_Ceiling(max_rung=ceil_rung, samples=4, max_seconds=60.0))
        for b in _sweep(mc):
            assert b.entry_rung <= b.max_rung, f"entry above max at ceiling={ceil_rung}"


def test_easy_gets_one_forward_pass():
    mc = _controller()
    plan = mc.plan(EASY)
    assert plan.entry_rung == RUNG_FAST
    assert plan.max_rung == 1 and plan.samples == 1


def test_high_stakes_never_fast_path():
    mc = _controller()
    plan = mc.plan("delete the production database")
    assert plan.entry_rung >= 1


def test_hard_gets_deep_budget():
    """A hard turn buys more of every resource than an easy one — relative to the ceiling in force.

    This asserted ``max_seconds > 60.0`` and could never pass: ``tests/conftest.py`` pins
    ``NYXARA_METACONTROL__MAX_SECONDS_CEILING=6`` for the whole suite on purpose, so no plan
    anywhere in these tests may exceed six seconds. The test was asserting against an absolute
    its own harness forbids. What "deep budget" actually means is *deeper than the easy path* and
    *up against whatever ceiling applies*, which is the same relative form the module's own
    self-test at ``metacontrol.py`` uses.
    """
    mc = _controller()
    easy = mc.plan(EASY)
    plan = mc.plan(HARD, novelty=1.0, recall_strength=0.0, competence=0.0)
    assert plan.entry_rung >= 2
    assert plan.entry_rung > easy.entry_rung
    assert plan.max_rung >= easy.max_rung
    assert plan.samples >= easy.samples
    # it spends the whole allowance it is permitted, whatever that allowance has been set to
    ceiling = float(mc._cfgv("max_seconds_ceiling", 600.0))
    assert abs(plan.max_seconds - min(ceiling, 300.0)) < 1e-6, \
        f"hard did not reach its ceiling: {plan.max_seconds} of {ceiling}"


def test_calibration_loop_corrects_overconfident_easiness():
    mc = _controller()
    raw = mc.estimate(EASY).difficulty
    base_target = mc.plan(EASY).confidence_target
    # she keeps calling this easy — and keeps failing it
    for _ in range(30):
        mc.record_outcome(mc.plan(EASY), success=False)
    corrected = mc.estimate(EASY).calibrated
    assert corrected > raw                       # the estimate is pushed harder
    assert mc.calibrator.overconfidence() > 0    # over-confidence was measured...
    assert mc.plan(EASY).confidence_target >= base_target  # ...and raises the bar


def test_escalation_records_underallocation():
    mc = _controller()
    for _ in range(30):
        # the turn "succeeded" but only because the ladder climbed past its entry rung —
        # that is an under-allocation, and must count against predicted easiness.
        mc.record_outcome(mc.plan(EASY), success=True, verified_score=0.9, escalated=True)
    assert mc.calibrator.accuracy() == 0.0
    assert mc.calibrator.overconfidence() > 0


def test_persistence_round_trip(tmp_path: Path):
    mc = _controller()
    for _ in range(7):
        mc.record_outcome(mc.plan(EASY), success=True, verified_score=0.9)
    p = tmp_path / "metacontrol.json"
    mc.save(p)
    s = NyxaraSettings.for_profile(Profile.TEST)
    mc2 = MetacognitiveController(settings=s, path=p, effort_memory=False)
    assert mc2.calibrator.samples == mc.calibrator.samples
    assert abs(mc2._band["easy"].mean - mc._band["easy"].mean) < 1e-9


def test_corrupt_state_never_breaks_construction(tmp_path: Path):
    p = tmp_path / "metacontrol.json"
    p.write_text("{not valid json", encoding="utf-8")
    s = NyxaraSettings.for_profile(Profile.TEST)
    mc = MetacognitiveController(settings=s, path=p, effort_memory=False)
    assert mc.calibrator.samples == 0  # fresh state, no crash


def test_disabled_controller_is_inert():
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.metacontrol.enabled = False
    mc = MetacognitiveController(settings=s, path=None, effort_memory=False)
    assert mc.enabled() is False


def test_should_stop_respects_confidence_target():
    b = ComputeBudget(max_rung=3, samples=3, max_seconds=30.0, entry_rung=1,
                      confidence_target=0.7)
    assert MetacognitiveController.should_stop(b, 1, 0.8)
    assert not MetacognitiveController.should_stop(b, 1, 0.6)
    no_target = ComputeBudget(max_rung=3, samples=3, max_seconds=30.0)
    assert not MetacognitiveController.should_stop(no_target, 1, 0.99)


def test_effort_history_nudges_difficulty():
    class _EM:
        def mean_for(self, sig, rung):
            return 0.9 if rung in (1, 2) else None

        def suggest(self, sig):
            return 1

    class _HardEM:
        def mean_for(self, sig, rung):
            return None

        def suggest(self, sig):
            return 3

    s = NyxaraSettings.for_profile(Profile.TEST)
    known_easy = MetacognitiveController(settings=s, path=None, effort_memory=_EM())
    known_hard = MetacognitiveController(settings=s, path=None, effort_memory=_HardEM())
    q = "what is the tallest mountain on each continent of the world today"
    assert known_easy.estimate(q).difficulty < known_hard.estimate(q).difficulty
