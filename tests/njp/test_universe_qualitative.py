"""A stated law says which way, never how much — and that is still an answer.

Measured over 1,200 corpus pairs: `universe.relations` 40, `usable_relations` **0**,
`observations` 0, `rollouts` 0. Every arrow stated by the causal skeleton, not one of them ever
fitted, and every counterfactual answered "no usable arrow leaves the intervened variables".

The cause is structural, not a bug in the fitting. `Relation.usable` requires three samples and
an R² — it is a test for whether a *number* can be predicted — and a text corpus states laws
almost without ever stating quantities. So the simulator held a causal skeleton it could never
use for anything, and waiting for numbers was waiting forever.

"Fire causes heat" cannot say how much heat. It does say that removing the fire removes the heat,
and that is the counterfactual anyone actually asks.
"""

from __future__ import annotations

import pytest

from nyxara.njp.universe import InternalUniverse


@pytest.fixture()
def told() -> InternalUniverse:
    """Two stated arrows, a chain, and no quantities anywhere — the corpus's situation."""
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    universe.declare("garmi", "pasina", sign=1)
    universe.observe({"aag": 1.0, "garmi": 1.0, "pasina": 1.0})
    return universe


# --------------------------------------------------------------------------- #
# It answers at all
# --------------------------------------------------------------------------- #
def test_a_stated_law_answers_a_counterfactual(told):
    result = told.what_if("aag", 0.0)
    assert result.answerable, "this is the answer the skeleton was always able to give"
    assert {d.variable for d in result.deltas if d.qualitative} == {"garmi", "pasina"}


def test_the_direction_follows_the_intervention(told):
    """Removing a positively-signed cause pushes its effects down, not merely 'changes' them."""
    down = {d.variable: d.direction for d in told.what_if("aag", 0.0).deltas if d.qualitative}
    assert down == {"garmi": -1, "pasina": -1}
    up = {d.variable: d.direction for d in told.what_if("aag", 5.0).deltas if d.qualitative}
    assert up == {"garmi": 1, "pasina": 1}


def test_a_negative_law_reverses_the_direction():
    universe = InternalUniverse()
    universe.declare("rain", "dust", sign=-1)
    universe.observe({"rain": 1.0, "dust": 1.0})
    delta = [d for d in universe.what_if("rain", 5.0).deltas if d.qualitative]
    assert delta and delta[0].direction == -1


def test_it_travels_more_than_one_hop(told):
    """`pasina` is two arrows from `aag`, and a chain is the whole point of a skeleton."""
    via = {d.variable: d.via for d in told.what_if("aag", 0.0).deltas if d.qualitative}
    assert "pasina" in via
    assert "garmi→pasina" in via["pasina"][0]


# --------------------------------------------------------------------------- #
# It does not pretend to be a measurement
# --------------------------------------------------------------------------- #
def test_no_magnitude_is_invented(told):
    """`after` stays None. A number there would be a fabrication, not an estimate."""
    for delta in told.what_if("aag", 0.0).deltas:
        if delta.qualitative:
            assert delta.after is None
            assert delta.change is None


def test_a_direction_is_worth_less_than_a_fit(told):
    for delta in told.what_if("aag", 0.0).deltas:
        if delta.qualitative:
            assert delta.confidence <= 0.35


def test_the_answer_says_it_is_direction_only(told):
    assert "direction only" in told.what_if("aag", 0.0).reason


def test_an_unmeasured_arrow_is_not_counted_as_usable(told):
    assert told.usable_relations() == []
    assert len(told.directional_relations()) == 2
    assert told.stats()["directional_relations"] == 2


# --------------------------------------------------------------------------- #
# It never displaces a real fit
# --------------------------------------------------------------------------- #
def test_a_fitted_arrow_wins_over_a_stated_one():
    """Where numbers exist, the numeric pass answers and the qualitative one stands aside."""
    universe = InternalUniverse()
    universe.declare("water", "growth", sign=1)
    for x in (1.0, 2.0, 3.0, 4.0):
        universe.observe({"water": x, "growth": 2.0 * x})
    result = universe.what_if("water", 10.0)
    growth = [d for d in result.deltas if d.variable == "growth"]
    assert growth and not growth[0].qualitative
    assert growth[0].after is not None


def test_an_unsigned_arrow_stays_unusable():
    """`declare` without a sign says an arrow may exist, not which way it runs."""
    universe = InternalUniverse()
    universe.declare("a", "b")
    universe.observe({"a": 1.0, "b": 1.0})
    assert not universe.directional_relations()
    assert not universe.what_if("a", 0.0).answerable


def test_nothing_at_all_is_still_refused():
    universe = InternalUniverse()
    universe.observe({"lonely": 1.0})
    result = universe.what_if("lonely", 0.0)
    assert not result.answerable
    assert "no arrow of any kind" in result.reason


def test_an_intervention_with_no_prior_value_claims_no_direction():
    """Set to 3 from nothing — is that up or down? Unknown, and reported as unknown."""
    universe = InternalUniverse()
    universe.declare("x", "y", sign=1)
    result = universe.intervene({"x": 3.0})
    qualitative = [d for d in result.deltas if d.qualitative]
    assert qualitative and qualitative[0].direction == 0
