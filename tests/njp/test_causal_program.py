"""Naming the causal model, and why that is the whole of "compile it".

The obvious reading of "compile the world model to something executable" is a second simulator,
and building one would be wrong. `intervene` already severs, propagates, prices confidence along
the path and records the route; `imagine` already registers a rollout so `reconcile` can grade it.
Compiling adds nothing to *execution*, and writing a second engine beside a working one is the
duplication this package refuses. There is no `run` method here and there should not be one.

What was genuinely missing is identity. A prediction could not say which model made it, so "the
model was wrong" could never become "this model has been wrong six times" — and that sentence is
the only form in which a retirement decision can be made at all. This is the prerequisite the
death criterion could not be built without.
"""

from __future__ import annotations

from nyxara.njp.universe import CausalProgram, InternalUniverse, Orientation


def _linear(order=("water", "growth")) -> InternalUniverse:
    universe = InternalUniverse()
    for water in range(7):
        universe.observe({"water": water, "growth": 2.0 * water + 1.0}, order=order)
    return universe


def test_a_model_can_name_itself():
    program = _linear().compile()
    assert program.version
    assert ("water", "growth", Orientation.INFERRED) in program.arrows


def test_two_universes_that_fit_the_same_graph_get_the_same_name():
    """A hash over the arrows, so the name means the same thing on any machine."""
    assert _linear().compile().version == _linear().compile().version


def test_declaring_one_arrow_changes_the_model_version():
    universe = _linear()
    before = universe.compile().version
    universe.declare("growth", "sugar", sign=1, orientation=Orientation.ASSERTED)
    universe.observe({"growth": 3.0, "sugar": 6.0}, order=("growth", "sugar"))
    assert universe.compile().version != before


def test_an_unoriented_arrow_is_not_part_of_the_model():
    """It fits and it does not run, so it is not something the model is doing."""
    bare = InternalUniverse()
    for water in range(7):
        bare.observe({"water": water, "growth": 2.0 * water + 1.0})    # no stated order
    program = bare.compile()
    assert program.arrows == ()
    assert program.abstained, "a pair it refuses to orient is part of its description"


def test_what_it_refuses_is_part_of_the_description():
    """A model with no opinion and a model that declined to guess are different objects."""
    empty = InternalUniverse().compile()
    refusing = InternalUniverse()
    for water in range(7):
        refusing.observe({"water": water, "growth": 2.0 * water + 1.0})
    assert empty.abstained == ()
    assert refusing.compile().abstained != ()


def test_a_rollout_names_the_model_that_made_it():
    universe = _linear()
    roll = universe.imagine("halve", {"water": 1.0})
    assert roll.model_version == universe.compile().version


def test_failures_are_charged_to_the_model_that_claimed_them():
    """The sentence a retirement decision needs, and nothing could say it before."""
    universe = _linear()
    roll = universe.imagine("halve", {"water": 1.0})
    universe.reconcile({"water": 1.0, "growth": 9.0})

    # Charged to the model that MADE the claim, not the one that exists when it is graded.
    # `reconcile` observes reality before scoring — that is the point of it — so the fit has
    # already moved by the time the verdict lands, and the version has moved with it. Booking
    # the failure against the later model would blame whichever model happened to be current.
    record = universe.error_by_model()
    assert roll.model_version in record
    assert record[roll.model_version]["scored"] == 1
    assert record[roll.model_version]["error"] > 0.0


def test_a_changed_model_starts_its_own_record():
    """Otherwise a model's failures would be inherited by the one that replaced it."""
    universe = _linear()
    first = universe.imagine("a", {"water": 1.0}).model_version
    universe.reconcile({"water": 1.0, "growth": 9.0})

    universe.declare("growth", "sugar", sign=1, orientation=Orientation.ASSERTED)
    universe.observe({"growth": 3.0, "sugar": 6.0}, order=("growth", "sugar"))
    second = universe.imagine("b", {"water": 1.0}).model_version

    assert first != second
    assert set(universe.error_by_model()) >= {first, second}


def test_the_program_round_trips():
    program = _linear().compile()
    revived = CausalProgram.from_dict(program.to_dict())
    assert revived.version == program.version
    assert revived.arrows == program.arrows


def test_compiling_never_raises_on_an_empty_model():
    assert InternalUniverse().compile().version is not None


def test_there_is_no_second_simulator():
    """`intervene` executes. This names. Adding a `run` here would be the duplication refused."""
    assert not hasattr(CausalProgram, "run")
    assert not hasattr(CausalProgram, "predict")
