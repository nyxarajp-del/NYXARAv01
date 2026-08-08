"""NYX V.01 · L-CHRONOS — deciding by simulating outcomes, and admitting when it cannot see."""

from __future__ import annotations

import random

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.chronos import TemporalCausalMatrix
from nyxara.nyx.modules import Proposal


def _opts(*names) -> list:
    return [Proposal(source=n, content=f"{n} content", confidence=0.5) for n in names]


def _teach(chronos, *, better: str, worse: str, n: int = 40) -> None:
    """Give the world model real dynamics: one option genuinely pays off more than the other."""
    rng = random.Random(0)
    for _ in range(n):
        state = [rng.uniform(0.0, 1.0)]
        for action, gain in ((better, 0.6), (worse, 0.05)):
            nxt = [min(1.0, state[0] + gain * rng.uniform(0.2, 1.0))]
            chronos.learn(state, action, nxt, reward=nxt[0] - state[0])


@pytest.fixture
def chronos():
    return TemporalCausalMatrix(min_coverage=8, max_branches=120, horizon=3)


# -------------------- it only applies to decisions -------------------- #
def test_a_question_of_fact_has_no_futures(chronos):
    assert chronos.applies("what pulls an apple down") is False
    assert chronos.applies("gravity pulls an apple toward the ground") is False


def test_a_decision_does(chronos):
    assert chronos.applies("should I invest or hold") is True
    assert chronos.applies("which option is better to pick") is True


def test_empty_input_is_not_a_decision(chronos):
    for junk in ("", "   ", None):
        assert chronos.applies(junk) is False


# -------------------- blindness is reported, not faked -------------------- #
def test_an_unlearned_world_model_reports_itself_blind(chronos):
    """The simulator returns zeros for every option — passing that off as foresight is a lie."""
    got = chronos.explore(_opts("invest", "hold"), state=[0.5])
    assert got.blind is True and got.ran is False
    assert "learned 0 transitions" in got.reason


def test_a_blind_simulation_contributes_no_evidence(chronos):
    got = chronos.explore(_opts("invest", "hold"), state=[0.5])
    assert chronos.evidence(got) == {}


def test_availability_is_reported_honestly(chronos):
    assert chronos.available() is False
    _teach(chronos, better="invest", worse="hold")
    assert chronos.available() is True


def test_a_single_option_is_not_a_comparison(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest"), state=[0.5])
    assert got.ran is False and "fewer than two" in got.reason


# -------------------- the simulation itself -------------------- #
def test_the_option_that_actually_pays_off_wins(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest", "hold"), state=[0.3])
    assert got.ran is True and got.best == "invest"


def test_every_option_is_rolled_through_futures(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest", "hold"), state=[0.3])
    assert len(got.options) == 2
    assert all(o.branches > 0 for o in got.options)
    assert sum(o.branches for o in got.options) <= got.branches


def test_the_branch_budget_is_respected(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest", "hold"), state=[0.3], branches=10_000)
    assert got.branches <= chronos.max_branches


def test_the_bad_tail_is_measured_not_only_the_average(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest", "hold"), state=[0.3])
    assert any(o.tail_risk != 0.0 for o in got.options)


def test_the_report_carries_the_models_coverage_and_confidence(chronos):
    _teach(chronos, better="invest", worse="hold")
    got = chronos.explore(_opts("invest", "hold"), state=[0.3])
    assert got.coverage >= chronos.min_coverage
    assert 0.0 <= got.confidence <= 1.0


# -------------------- feeding the decision -------------------- #
def test_evidence_favours_the_better_option(chronos):
    _teach(chronos, better="invest", worse="hold")
    evidence = chronos.evidence(chronos.explore(_opts("invest", "hold"), state=[0.3]))
    assert evidence["invest"] > evidence["hold"]


def test_evidence_influence_is_bounded(chronos):
    """Foresight informs the decision; it does not get to override everything in one step."""
    _teach(chronos, better="invest", worse="hold")
    evidence = chronos.evidence(chronos.explore(_opts("invest", "hold"), state=[0.3]))
    assert all(0.5 <= v <= 2.0 for v in evidence.values())


def test_options_that_fared_identically_are_not_evidence(chronos):
    class _Flat:
        ran, blind, options = True, False, [
            type("O", (), {"option": "a", "score": 1.0})(),
            type("O", (), {"option": "b", "score": 1.0})()]

    assert TemporalCausalMatrix.evidence(_Flat()) == {}


# -------------------- robustness -------------------- #
def test_a_failed_simulation_sees_nothing_rather_than_crashing(chronos):
    _teach(chronos, better="invest", worse="hold")
    chronos._world_model = object()          # a model with no usable surface
    got = chronos.explore(_opts("invest", "hold"), state=[0.3])
    assert got.ran is False


def test_learning_into_a_missing_model_fails_quietly():
    c = TemporalCausalMatrix(world_model=None)
    c._tried = True                          # pretend construction failed
    assert c.learn([0.0], "a", [1.0]) is False
    assert c.coverage() == 0


# -------------------- wired into the cycle -------------------- #
@pytest.fixture
def brain():
    return NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "chronos_min_coverage": 8, "chronos_max_branches": 64}))


def test_a_factual_turn_does_not_simulate_futures(brain):
    assert brain.think("what pulls an apple down").futures is None


def test_a_decision_turn_simulates_once_the_model_has_learned(brain):
    if brain.chronos is None:
        pytest.skip("chronos disabled")
    _teach(brain.chronos, better="memory", worse="graph")
    got = brain.think("should I invest or should I hold")
    assert got.futures is not None
    assert got.futures.ran or got.futures.reason


def test_a_brain_without_chronos_still_thinks():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "chronos_enabled": False}))
    assert b.chronos is None
    assert b.think("should I invest or hold").futures is None


def test_stats_report_whether_she_can_see_ahead(brain):
    if brain.chronos is None:
        pytest.skip("chronos disabled")
    stats = brain.stats()["chronos"]
    assert {"available", "coverage", "min_coverage"} <= set(stats)
