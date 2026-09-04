"""What cannot work, and exactly why (NJP V.46)."""
from __future__ import annotations

import pytest

from nyxara.njp.boundary import MAX_CORE, Boundary, Constraint as C, Impossible


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #
def test_a_derived_condition_carries_the_chain_that_produced_it():
    got = Boundary([C("asserts", "A"), C("requires", "A", "D"), C("requires", "D", "E")])
    by_name = {n.condition: n for n in got.necessary()}
    assert by_name["E"].derived and len(by_name["E"].because) == 3
    assert by_name["A"].derived is False, "a stated condition is not a finding"


def test_a_closure_that_returns_its_own_inputs_is_marked_as_such():
    got = Boundary([C("asserts", "A"), C("asserts", "B")])
    assert all(not n.derived for n in got.necessary())


def test_nothing_is_concluded_from_an_absence():
    """Unstated is unknown, not false. A monotone closure, not a database."""
    got = Boundary([C("requires", "A", "D"), C("excludes", "D", "C"), C("asserts", "C")])
    assert got.impossible() is None, "A was never asserted, so D was never forced"


def test_a_cyclic_rule_set_terminates():
    got = Boundary([C("asserts", "A"), C("requires", "A", "B"), C("requires", "B", "A")])
    assert {n.condition for n in got.necessary()} == {"A", "B"}


# --------------------------------------------------------------------------- #
# Impossibility, with a core
# --------------------------------------------------------------------------- #
@pytest.fixture
def blocked():
    return Boundary([
        C("asserts", "A", label="A"), C("asserts", "B", label="B"),
        C("requires", "A", "D", label="A->D"), C("requires", "B", "E", label="B->E"),
        C("requires", "E", "F", label="E->F"), C("asserts", "C", label="C"),
        C("excludes", "D", "C", label="D-x-C"), C("requires", "X", "Y", label="unused"),
    ])


def test_no_solution_names_the_conflict(blocked):
    got = blocked.impossible()
    assert isinstance(got, Impossible)
    assert set(got.conflict) == {"D", "C"}
    assert "NO SOLUTION" in got.render()


def test_the_core_is_minimised_to_what_is_responsible(blocked):
    """"Unsatisfiable" is not actionable. A core of four out of eight is."""
    got = blocked.impossible()
    labels = {c.label for c in got.core}
    assert labels == {"A", "A->D", "C", "D-x-C"}
    assert "B->E" not in labels and "unused" not in labels
    assert len(got.core) <= MAX_CORE


def test_the_derivation_that_reached_the_conflict_is_shown(blocked):
    got = blocked.impossible()
    assert any(n.condition == "D" and n.derived for n in got.derivation)
    assert "reached by" in got.render()


def test_a_satisfiable_set_reports_what_must_hold():
    got = Boundary([C("asserts", "A"), C("requires", "A", "D"), C("excludes", "D", "Z")])
    assert got.impossible() is None
    assert "SATISFIABLE" in got.render() and "D" in got.render()


def test_an_unreadable_constraint_is_returned_not_dropped():
    got = Boundary([C("asserts", "A"), C("nonsense", "x"), "not a constraint"])
    assert len(got.ignored) == 2 and len(got.constraints) == 1
    assert "ignored 2" in got.render()


# --------------------------------------------------------------------------- #
# The funnel is the product
# --------------------------------------------------------------------------- #
def test_the_funnel_names_the_stage_that_did_the_work():
    got = Boundary().prune(list(range(1000)), [
        ("even", lambda n: n % 2 == 0),
        ("div by 5", lambda n: n % 5 == 0),
        ("> 900", lambda n: n > 900),
    ])
    assert got.start == 1000 and got.end == 9
    assert [removed for _name, removed in got.stages] == [500, 400, 91]
    assert "1000 candidates" in got.render()


def test_a_constraint_that_never_binds_is_named():
    got = Boundary().prune(list(range(10)), [
        ("real", lambda n: n < 5), ("vacuous", lambda n: True)])
    assert got.idle == ["vacuous"]
    assert "never binding: vacuous" in got.render()
