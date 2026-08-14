"""The Internal Universe and the Counterfactual Engine, plus curiosity as information gain.

The tests that matter here are the ones a plausible-looking simulator would fail: that the
do-operator actually *severs* incoming arrows rather than conditioning, that extrapolation is
reported as less trustworthy instead of borrowing the fit's credibility, and that an experiment
which separates no hypothesis scores exactly zero however interesting it looks.
"""

from __future__ import annotations

import math

from nyxara.njp.universe import ExperimentDesigner, InternalUniverse


def _plant_universe() -> InternalUniverse:
    """growth = 2·water + 1, with light held constant so only one arrow can explain anything."""
    universe = InternalUniverse()
    for water in range(7):
        universe.observe({"water": water, "light": 5, "growth": 2.0 * water + 1.0})
    return universe


def test_a_relation_is_a_real_least_squares_fit():
    universe = _plant_universe()
    relation = universe.relations[("water", "growth")]
    assert abs(relation.slope - 2.0) < 1e-6
    assert abs(relation.intercept - 1.0) < 1e-6
    assert relation.r2 > 0.999
    assert relation.usable


def test_r2_falls_when_the_arrow_explains_nothing():
    universe = InternalUniverse()
    for i, noise in enumerate([3.0, -1.0, 4.0, -2.0, 0.5, 5.0, -3.0]):
        universe.observe({"x": float(i), "y": noise})
    assert universe.relations[("x", "y")].r2 < 0.5
    assert not universe.relations[("x", "y")].usable


def test_the_do_operator_predicts_a_value_never_observed():
    universe = _plant_universe()
    out = universe.what_if("water", 3.0)
    assert out.answerable
    assert abs(out.predicted["growth"] - 7.0) < 1e-6      # 2·3 + 1


def test_removing_a_cause_entirely_is_the_zero_intervention():
    universe = _plant_universe()
    out = universe.without("water")
    assert abs(out.predicted["growth"] - 1.0) < 1e-6      # the intercept alone


def test_a_graded_intervention_differs_from_a_total_one():
    """"Plant with half water" and "plant with no water" are different questions."""
    universe = _plant_universe()
    half = universe.what_if("water", 3.0).predicted["growth"]
    none = universe.without("water").predicted["growth"]
    full = universe.what_if("water", 6.0).predicted["growth"]
    assert none < half < full


def test_the_intervened_variable_is_severed_not_conditioned():
    """``do(X)`` cuts X's own incoming arrows; conditioning on X would not.

    Direction has to come from the skeleton. Without one the fit is symmetric — that is Markov
    equivalence, not a defect — so the test supplies the direction the way the brain does.
    """
    class _Link:
        def __init__(self, ok):
            self.causal = self.stated = ok
            self.refuted = self.together = 0

    class _World:
        def link(self, cause, effect):
            return _Link(cause == "water" and effect == "growth")

        def links(self, causal_only=True):
            return []

    universe = InternalUniverse(world=_World())
    for water in range(7):
        universe.observe({"water": water, "growth": 2.0 * water + 1.0})

    out = universe.intervene({"growth": 99.0})
    water = next((d for d in out.deltas if d.variable == "water"), None)
    assert water is None or water.change in (None, 0.0)
    # And the arrow that does exist still works in its own direction.
    assert abs(universe.what_if("water", 3.0).predicted["growth"] - 7.0) < 1e-6


def test_a_direction_observational_data_cannot_settle_is_reported_not_hidden():
    universe = _plant_universe()
    pairs = universe.ambiguous()
    assert ("growth", "water") in pairs or ("water", "growth") in pairs
    assert universe.stats()["ambiguous_pairs"] >= 1


def test_extrapolation_is_reported_as_less_trustworthy():
    universe = _plant_universe()
    inside = universe.what_if("water", 3.0)
    outside = universe.what_if("water", 500.0)
    assert outside.confidence < inside.confidence / 2, (inside.confidence, outside.confidence)
    # It still answers — refusing would be its own kind of dishonesty — but says how far out it is.
    assert outside.predicted["growth"] > inside.predicted["growth"]


def test_a_rollout_is_registered_as_outstanding_and_then_scored():
    universe = _plant_universe()
    roll = universe.imagine("water it more", {"water": 8.0}, steps=2)
    assert roll.states and not roll.scored
    error = universe.reconcile({"water": 8.0, "light": 5.0, "growth": 17.0})
    assert roll.scored
    assert error < 1e-6                                    # 2·8 + 1 = 17


def test_being_wrong_is_counted_as_a_surprise():
    universe = _plant_universe()
    universe.imagine("water it more", {"water": 8.0}, steps=1)
    universe.reconcile({"water": 8.0, "light": 5.0, "growth": -400.0})
    assert universe.surprises >= 1


def test_a_stated_arrow_is_watched_before_any_number_arrives():
    universe = InternalUniverse()
    relation = universe.declare("aag", "garmi")
    assert relation.stated and not relation.usable
    assert universe.stats()["stated_arrows"] == 1


def test_attribute_variables_consult_the_skeleton_by_their_entity():
    """``pani.boils`` is about ``pani``; a skeleton keyed on entities must still be readable."""
    class _Link:
        causal = stated = True
        refuted = together = 0

    class _World:
        def link(self, cause, effect):
            return _Link()

        def links(self, causal_only=True):
            return []

    universe = InternalUniverse(world=_World())
    assert universe._permitted("pani.boils", "bhaap.forms")
    assert universe._entity("pani.boils") == "pani"


# --------------------------------------------------------------------------- #
# Curiosity as information gain
# --------------------------------------------------------------------------- #
def _three_rivals() -> ExperimentDesigner:
    designer = ExperimentDesigner()
    designer.propose("water_is_cause",
                     predictions={"remove_water": "dies", "remove_light": "lives"})
    designer.propose("light_is_cause",
                     predictions={"remove_water": "lives", "remove_light": "dies"})
    designer.propose("both_needed",
                     predictions={"remove_water": "dies", "remove_light": "dies"})
    return designer


def test_entropy_is_real_shannon_bits():
    designer = _three_rivals()
    # Three roughly-equal rivals sit near log2(3) ≈ 1.585 bits.
    assert 1.3 < designer.prior_entropy() <= math.log2(3) + 1e-9


def test_an_experiment_that_separates_nothing_gains_exactly_zero():
    """The whole point: 'interesting' is not a property of an experiment; informative is."""
    designer = _three_rivals()
    assert designer.evaluate("play_music").gain == 0.0


def test_the_chosen_experiment_is_the_most_informative_one():
    designer = _three_rivals()
    chosen = designer.design(["remove_water", "remove_light", "play_music"])
    assert chosen is not None
    assert chosen.gain >= designer.evaluate("remove_water").gain
    assert chosen.gain > 0.0


def test_a_refuted_hypothesis_is_ruled_out_not_merely_discounted():
    designer = _three_rivals()
    designer.observe_result("remove_light", "dies")
    survivors = {h.name: h.probability for h in designer.hypotheses.values()}
    assert survivors["water_is_cause"] == 0.0            # predicted "lives", was contradicted
    assert survivors["light_is_cause"] > 0.0
    assert designer.bits_gained > 0.0


def test_two_experiments_can_resolve_the_question_completely():
    designer = _three_rivals()
    designer.observe_result("remove_light", "dies")
    designer.observe_result("remove_water", "lives")
    live = [h for h in designer.hypotheses.values() if h.probability > 0]
    assert len(live) == 1 and live[0].name == "light_is_cause"
    assert designer.prior_entropy() < 1e-9              # no uncertainty left


def test_state_survives_a_round_trip():
    universe = _plant_universe()
    revived = InternalUniverse()
    revived.load_dict(universe.to_dict())
    assert abs(revived.relations[("water", "growth")].slope - 2.0) < 1e-6
    assert abs(revived.what_if("water", 3.0).predicted["growth"] - 7.0) < 1e-6
