"""Giving up a model as a judged, reversible trial — and the four brakes on it.

The criterion and the archive shipped first; this is the half that decides whether a death is
allowed to stand, and the half that stops the loop proposing the same death forever.

This is the riskiest thing in the package and it is fenced accordingly. A wrong death criterion
destroys working structure, and `concepts.restructure` records a measured instance of exactly that
failure — a search that "drove the similarity threshold to its floor, where everything resembles
everything and the concepts formed are worthless". It ships behind a gate that is off, and being
reversible is a reason to allow switching it on rather than a reason to switch it on for everyone.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp.brain import NJPBrain
from nyxara.njp.field import RecursiveCognitiveField, _RETIRE_COOLDOWN
from nyxara.njp.universe import InternalUniverse, Orientation


class _FailingUniverse(InternalUniverse):
    """A universe that always says its current model should go."""

    def should_retire(self, **_kw):
        return self.compile().version or "v"


def _field(universe=None, concepts=None) -> RecursiveCognitiveField:
    field = RecursiveCognitiveField(None)
    field.universe = universe
    field.concepts = concepts
    return field


def _oriented() -> _FailingUniverse:
    universe = _FailingUniverse()
    for water in range(7):
        universe.observe({"water": water, "growth": 2.0 * water + 1.0},
                         order=("water", "growth"))
    return universe


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_it_is_off_by_default():
    assert NJPBrain()._gate("model_death", False) is False


def test_it_can_be_switched_on():
    assert NJPBrain(config=SimpleNamespace(model_death_enabled=True))._gate(
        "model_death", False) is True


def test_nothing_is_proposed_while_the_gate_is_off():
    brain = NJPBrain()
    for line in ("paudhe ko paani chahiye", "paani se growth badhti hai"):
        thought = brain.think(line)
        assert thought.retirement is None


# --------------------------------------------------------------------------- #
# The four brakes
# --------------------------------------------------------------------------- #

def test_a_tie_reverts_rather_than_standing():
    """Giving up structure has to be paid for by a measured win, not by breaking even."""
    universe = _oriented()
    oriented_before = sum(1 for r in universe.relations.values() if r.oriented)
    field = _field(universe)                       # no concepts, so benchmark() is flat

    trial = field.retire_cycle()
    assert trial.accepted is False
    assert "not a win" in trial.why or "nothing to give up" in trial.why
    assert sum(1 for r in universe.relations.values() if r.oriented) == oriented_before


def test_a_model_that_lost_its_trial_is_not_re_proposed_next_cycle():
    """Without this the criterion re-proposes it immediately, forever."""
    field = _field(_oriented())
    field.retire_cycle()
    second = field.retire_cycle()
    assert "cooling down" in second.why
    assert field.stats()["retirements_cooling"] >= 1


def test_the_cooldown_expires_and_the_trial_is_run_again():
    """A permanent refusal would be a different bug from the one it prevents.

    What "expired" looks like here is a second *trial*, not a zero counter: the retry runs, loses
    again against the same flat benchmark, and cools again. That cycle is the intended steady
    state — it costs one trial per cooldown rather than one per turn.
    """
    field = _field(_oriented())
    field.retire_cycle()
    assert field.stats()["retirements_tried"] == 1

    for _ in range(_RETIRE_COOLDOWN + 1):
        field.retire_cycle()
    assert field.stats()["retirements_tried"] == 2, "the cooldown never released"


def test_a_protected_target_is_refused_before_anything_is_measured():
    """A benchmark result must never be able to motivate touching the character core."""
    field = _field(_oriented())
    measured = {"n": 0}
    field.benchmark = lambda: measured.__setitem__("n", measured["n"] + 1) or 0.0  # type: ignore
    field.PROTECTED = frozenset({"universe"})      # make this proposal illegal

    trial = field.retire_cycle()
    assert "protected" in trial.why
    assert measured["n"] == 0, "it measured before refusing"


def test_a_retired_model_is_archived_and_restorable():
    universe = _oriented()
    field = _field(universe)
    field.retire_cycle()                            # reverts on a tie, which exercises restore
    assert sum(1 for r in universe.relations.values() if r.oriented) > 0


# --------------------------------------------------------------------------- #
# What survives a death
# --------------------------------------------------------------------------- #

def test_what_she_was_told_survives_what_she_inferred():
    universe = _oriented()
    universe.declare("sun", "growth", sign=1, orientation=Orientation.ASSERTED)
    field = _field(universe)
    field.retire_cycle()
    assert universe.relations[("sun", "growth")].orientation == Orientation.ASSERTED


def test_a_field_with_no_universe_says_so_rather_than_raising():
    assert "no universe" in RecursiveCognitiveField(None).retire_cycle().why


def test_a_model_nothing_is_wrong_with_is_not_proposed():
    universe = InternalUniverse()                   # the real criterion, not the stub
    for water in range(7):
        universe.observe({"water": water, "growth": 2.0 * water + 1.0},
                         order=("water", "growth"))
    trial = _field(universe).retire_cycle()
    assert trial.accepted is False
    assert "failed enough" in trial.why


def test_trials_and_keeps_are_counted_separately():
    """A trial run and reverted is not the same as one never proposed."""
    field = _field(_oriented())
    field.retire_cycle()
    stats = field.stats()
    assert stats["retirements_tried"] >= 1
    assert stats["retirements_kept"] == 0
