"""She can say which strategy fails *here*, not only which fails in general.

Before this organ the finest resolution she had was `(kind, strategy)` — "simulation beats
retrieval on causal problems". A kind is not a condition. Two causal questions, one where
grounding found the entity and one where it found nothing, are the same kind and routinely
different problems; averaging them produces a number true of neither, and the reliable case
subsidises the shaky one.

Three things recorded fragments and none recorded the episode: `ledger.ErrorRecord` holds what was
believed and what was true with no trace of how it was reached, `observe/turn_ledger.py` records
which language rung spoke, and `metareason.Strategy` holds the average this file exists to
disaggregate.

The rule that keeps it honest is the one most of these tests are about: below `min_samples` the
answer is `None`, never a number. "This strategy fails here" and "I have never tried this strategy
here" are different claims, and only one of them is evidence.
"""

from __future__ import annotations

import pytest

from nyxara.njp.blackbox import BlackBox, CognitiveEpisode, Conditions, band
from nyxara.njp.brain import NJPBrain


def _episode(strategy: str, conditions: Conditions, *, correct: bool) -> CognitiveEpisode:
    return CognitiveEpisode(stimulus="q", strategy=strategy, conditions=conditions,
                            correct=correct, error=0.0 if correct else 1.0)


@pytest.fixture()
def here() -> Conditions:
    return Conditions(kind="causal", act="ask", epistemic="believed",
                      grounded=False, novelty="high", confidence="low")


@pytest.fixture()
def elsewhere() -> Conditions:
    return Conditions(kind="causal", act="ask", epistemic="known",
                      grounded=True, novelty="low", confidence="high")


# ---- the honesty rule ------------------------------------------------------- #
def test_a_thin_record_answers_nothing(here: Conditions):
    """Four episodes is not a rate. The answer is `None`, not a low number."""
    box = BlackBox(min_samples=5)
    for _ in range(4):
        box.record(_episode("simulate", here, correct=False))
    assert box.similar(here, "simulate") is None
    assert box.penalty(here, "simulate") == 0.0


def test_the_record_speaks_once_there_is_enough_of_it(here: Conditions):
    box = BlackBox(min_samples=5)
    for _ in range(5):
        box.record(_episode("simulate", here, correct=False))
    verdict = box.similar(here, "simulate")
    assert verdict is not None
    assert verdict.trials == 5
    assert verdict.rate == 0.0


def test_a_condition_never_tried_is_not_a_condition_failed(here: Conditions,
                                                           elsewhere: Conditions):
    """The distinction the whole organ exists to preserve."""
    box = BlackBox(min_samples=3)
    for _ in range(6):
        box.record(_episode("simulate", here, correct=False))
    assert box.similar(here, "simulate") is not None      # tried, and it went badly
    assert box.similar(elsewhere, "simulate") is None     # never tried here at all


def test_an_episode_with_no_strategy_is_not_filed(here: Conditions):
    """It could not answer the question the record exists for, and would dilute every rate."""
    box = BlackBox()
    assert box.record(_episode("", here, correct=True)) is False
    assert box.stats()["episodes"] == 0


def test_bands_keep_an_absent_measurement_distinct_from_a_low_one():
    """Folding `None` into `low` would let unmeasured turns look like unconfident ones."""
    assert band(None) == "unknown"
    assert band(float("nan")) == "unknown"
    assert (band(0.1), band(0.5), band(0.9)) == ("low", "mid", "high")


# ---- the penalty may only ever lower ---------------------------------------- #
def test_a_strategy_doing_well_here_is_never_promoted(here: Conditions):
    """A record of her own past behaviour is not independent evidence about the world."""
    box = BlackBox(min_samples=3)
    for _ in range(8):
        box.record(_episode("simulate", here, correct=True))
    assert box.penalty(here, "simulate") == 0.0


def test_the_penalty_is_charged_where_it_is_earned_and_nowhere_else(here: Conditions,
                                                                    elsewhere: Conditions):
    """A failure under C must not follow the strategy into D."""
    box = BlackBox(min_samples=3)
    for _ in range(8):
        box.record(_episode("simulate", here, correct=False))
    for _ in range(8):
        box.record(_episode("simulate", elsewhere, correct=True))
    assert box.penalty(here, "simulate") > 0.0
    assert box.penalty(elsewhere, "simulate") == 0.0


def test_a_strategy_mediocre_everywhere_is_not_punished_twice(here: Conditions,
                                                              elsewhere: Conditions):
    """The shortfall is measured against the strategy's own baseline, not against perfection."""
    box = BlackBox(min_samples=3)
    for correct in (True, False) * 4:
        box.record(_episode("ladder", here, correct=correct))
        box.record(_episode("ladder", elsewhere, correct=correct))
    # Half right here, half right in general — no shortfall, so nothing to charge.
    assert box.penalty(here, "ladder") == 0.0


def test_failing_names_the_conditions_she_is_worst_under(here: Conditions,
                                                         elsewhere: Conditions):
    box = BlackBox(min_samples=3)
    for _ in range(8):
        box.record(_episode("simulate", elsewhere, correct=True))
    for _ in range(8):
        box.record(_episode("simulate", here, correct=False))
    failing = box.failing()
    assert failing and failing[0].strategy == "simulate"
    assert failing[0].underperforming


# ---- bounds and persistence -------------------------------------------------- #
def test_the_record_is_bounded(here: Conditions):
    box = BlackBox(capacity=16, min_samples=2)
    for _ in range(200):
        box.record(_episode("simulate", here, correct=True))
    assert len(box.episodes) == 16
    assert box.recorded == 200


def test_a_record_survives_a_round_trip(here: Conditions):
    box = BlackBox(min_samples=3)
    for _ in range(6):
        box.record(_episode("simulate", here, correct=False))
    restored = BlackBox()
    restored.load_dict(box.to_dict())
    verdict = restored.similar(here, "simulate")
    assert verdict is not None and verdict.trials == 6 and verdict.rate == 0.0


def test_a_corrupt_snapshot_leaves_an_empty_record_not_an_exception():
    box = BlackBox()
    box.load_dict({"episodes": "not a list", "capacity": "nonsense"})
    assert box.episodes == []


# ---- it changes a decision, or it is only logging ---------------------------- #
def test_a_measured_failure_here_demotes_the_strategy_here(here: Conditions):
    """The consumer. Without this the organ is a log nobody reads."""
    from nyxara.njp.metareason import MetaReasoner, ProblemKind

    def _solve(problem, ctx):
        return "answer"

    box = BlackBox(min_samples=3)
    meta = MetaReasoner(blackbox=box)
    meta.register("good", (ProblemKind.CAUSAL,), _solve, prior=0.5)
    meta.register("bad", (ProblemKind.CAUSAL,), _solve, prior=0.5)

    # Same prior, same kind: with no record the choice is the bandit's alone and either may win.
    for _ in range(8):
        box.record(_episode("bad", here, correct=False))
        box.record(_episode("bad", Conditions(kind="factual"), correct=True))

    chosen = meta.choose(ProblemKind.CAUSAL, conditions=here)
    assert chosen is not None and chosen.name == "good"


def test_without_conditions_the_choice_is_exactly_what_it_was(here: Conditions):
    """An untried condition, and an absent black box, must both change nothing."""
    from nyxara.njp.metareason import MetaReasoner, ProblemKind

    def _solve(problem, ctx):
        return "answer"

    box = BlackBox(min_samples=3)
    for _ in range(8):
        box.record(_episode("bad", here, correct=False))

    with_box = MetaReasoner(blackbox=box)
    without = MetaReasoner()
    for meta in (with_box, without):
        meta.register("good", (ProblemKind.CAUSAL,), _solve, prior=0.5)
        meta.register("bad", (ProblemKind.CAUSAL,), _solve, prior=0.9)

    # `bad` has the higher prior and a bad record *elsewhere*; no conditions are supplied, so the
    # record is not consulted and the prior stands in both.
    assert with_box.choose(ProblemKind.CAUSAL).name == without.choose(ProblemKind.CAUSAL).name


# ---- the brain wiring --------------------------------------------------------- #
def test_the_brain_files_a_graded_turn():
    """`resolve` is the only place an episode can be written: the outcome is what makes it one."""
    brain = NJPBrain()
    assert brain.blackbox is not None
    thought = brain.think("what causes rain")
    thought.solution = type("S", (), {"strategy": "causal", "kind": "causal"})()
    brain.resolve(thought, correct=0.0, actual="evaporation")
    assert brain.blackbox.stats()["episodes"] == 1


def test_a_turn_with_no_strategy_files_nothing():
    brain = NJPBrain()
    thought = brain.think("hello")
    brain.resolve(thought, correct=1.0)
    assert brain.blackbox.stats()["episodes"] == 0


def test_the_organ_can_be_switched_off():
    """Off must be indistinguishable from a record with nothing in it yet.

    The package's standing rule, as `test_wired.py` puts it: a disabled organ is absent, not
    zeroed. Selection must fall back to the kind-level bandit with nothing else changed.
    """
    class _Off:
        blackbox_enabled = False

    brain = NJPBrain(_Off())
    assert brain.blackbox is None
    brain.resolve(brain.think("what causes rain"), correct=1.0)  # must not raise


def test_the_gate_is_a_real_setting():
    """Not an ad-hoc attribute: it lives in the settings model beside every other organ's."""
    from nyxara.kernel.config import get_settings
    settings = get_settings()
    assert settings.njp.blackbox_enabled is True
    assert settings.njp.blackbox_min_samples >= 2
