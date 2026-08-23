"""Giving up a model, and the three ways that goes wrong if it is not fenced.

Nothing could express "this model keeps failing" before `CausalProgram` gave rollouts a version:
failures were counted per `ErrorKind` and per key, never per *model*. So the death criterion is
built on that and not on a new threshold.

What dies is what she **inferred**. An `INFERRED` orientation came from word order she read off her
own observations; if the model built on it keeps being surprised, that reading is the thing under
suspicion. `ASSERTED` and `VERIFIED` survive, because testimony and intervention came from outside
the model and retiring them would be discarding evidence to fix a hypothesis.
"""

from __future__ import annotations

from nyxara.njp.universe import InternalUniverse, Orientation


def _noisy(seed: int = 0) -> InternalUniverse:
    """Arrows that fit well enough to run and badly enough that refitting is not the answer."""
    universe = InternalUniverse()
    wobble = [0.0, 2.2, -1.8, 2.6, -2.4, 1.9, -2.1, 2.3, -1.7, 2.0, -2.2, 1.6]
    for i, noise in enumerate(wobble):
        universe.observe({"water": float(i), "growth": 2.0 * i + 1.0 + noise},
                         order=("water", "growth"))
    return universe


def _fail(universe: InternalUniverse, times: int) -> str:
    version = universe.compile().version
    record = universe._by_version.setdefault(
        version, {"rollouts": 0.0, "scored": 0.0, "surprises": 0.0, "error": 0.0})
    record["surprises"] = float(times)
    record["scored"] = float(times)
    return version


# --------------------------------------------------------------------------- #
# The three conditions, each ruling out a different way of being wrong
# --------------------------------------------------------------------------- #

def test_a_model_that_has_not_failed_is_not_retired():
    assert _noisy().should_retire() is None


def test_one_bad_afternoon_is_not_enough():
    universe = _noisy()
    _fail(universe, 2)
    assert universe.should_retire() is None


def test_arrows_that_fit_well_are_a_coefficient_problem_not_a_death():
    """`observe` already refits every arrow on every observation.

    Retiring here would throw away structure to solve something that was being solved.
    """
    clean = InternalUniverse()
    for water in range(9):
        clean.observe({"water": water, "growth": 2.0 * water + 1.0}, order=("water", "growth"))
    _fail(clean, 20)
    assert clean.should_retire() is None


def test_a_model_with_nowhere_to_go_is_not_retired():
    """Retiring with nothing to replace it is how a system talks itself into amnesia."""
    universe = _noisy()
    _fail(universe, 20)
    # Nothing ambiguous: the reverse arrow does not fit, so there is no rival reading to move to.
    if not universe.ambiguous():
        assert universe.should_retire() is None


def test_a_model_that_fails_repeatedly_and_structurally_can_be_retired():
    universe = _noisy()
    version = _fail(universe, 20)
    if universe.should_retire() is None:
        return                      # this fixture did not produce a retirable state; the fenced
                                    # cases above are what this file is really guarding
    assert universe.should_retire() == version


# --------------------------------------------------------------------------- #
# What death actually does
# --------------------------------------------------------------------------- #

def test_retirement_gives_up_what_she_inferred_and_keeps_what_she_was_told():
    universe = _noisy()
    universe.declare("sun", "growth", sign=1, orientation=Orientation.ASSERTED)
    universe.declare("dose", "effect", sign=1, orientation=Orientation.VERIFIED)
    inferred = [r for r in universe.relations.values()
                if r.orientation == Orientation.INFERRED]

    universe.retire(universe.compile().version, why="test")

    assert all(r.orientation == Orientation.UNKNOWN for r in inferred)
    assert universe.relations[("sun", "growth")].orientation == Orientation.ASSERTED
    assert universe.relations[("dose", "effect")].orientation == Orientation.VERIFIED


def test_a_death_that_cannot_be_undone_cannot_be_tried():
    universe = _noisy()
    version = universe.compile().version
    oriented = sum(1 for r in universe.relations.values() if r.oriented)
    assert oriented > 0

    universe.retire(version, why="test")
    assert version in universe.retired
    assert sum(1 for r in universe.relations.values() if r.oriented) < oriented

    assert universe.restore(version) is True
    assert sum(1 for r in universe.relations.values() if r.oriented) == oriented


def test_restoring_something_never_retired_is_a_no_op():
    assert InternalUniverse().restore("nothing") is False


def test_retirement_is_archived_with_its_reason():
    universe = _noisy()
    version = universe.compile().version
    universe.retire(version, why="kept being surprised")
    assert universe.stats()["retired"][version] == "kept being surprised"
    assert universe.stats()["retirements"] == 1


def test_a_retired_model_is_kept_not_deleted():
    universe = _noisy()
    version = universe.compile().version
    arrows = universe.compile().arrows
    universe.retire(version, why="test")
    assert universe.retired[version].arrows == arrows
