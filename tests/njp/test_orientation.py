"""Fit is not direction, and the propagation used to be unable to tell them apart.

The reproduced failure: forty observations of ``growth = 2·water + 3·light``, then
``do(water=2)``. The answer moved ``light`` from 5.0 to 3.79, along the path
``['do', 'water→growth', 'growth→light']``. Light is exogenous — nothing upstream of it exists —
and the model had no way to know that, because the only gate on propagation was ``usable``, which
is ``n >= 3 and r2 >= 0.25``: a goodness-of-fit test with no causal content whatsoever.

The two fields that *do* carry orientation, ``stated`` and ``sign``, were consulted only by
``directional``, which requires ``not usable``. So the moment an arrow had enough data to be used,
everything known about which way it runs became unreachable. The system detected the resulting
symmetry correctly — ``ambiguous()`` has been right the whole time — and then did nothing with the
detection, because nothing read it.

What is asserted here is the decision that followed: an edge whose direction nothing has
established does not propagate, the counterfactual says so instead of returning a number, and the
pair becomes an experiment rather than a dead end.
"""

from __future__ import annotations

import random


from nyxara.njp.universe import InternalUniverse, Orientation


def _linear_universe() -> InternalUniverse:
    """``growth = 2·water + 3·light``, with water and light independent of each other."""
    universe = InternalUniverse()
    rng = random.Random(0)
    for _ in range(40):
        water, light = rng.uniform(1, 10), rng.uniform(1, 10)
        universe.observe({"water": water, "light": light,
                          "growth": 2.0 * water + 3.0 * light})
    return universe


# --------------------------------------------------------------------------- #
# The reproduced failure
# --------------------------------------------------------------------------- #

def test_an_intervention_does_not_run_backwards_up_an_unoriented_arrow():
    """The bug, exactly as it was measured: do(water) moved light."""
    universe = _linear_universe()
    out = universe.intervene({"water": 2.0}, base={"water": 8.0, "light": 5.0})
    moved = {d.variable for d in out.changed()}
    assert "light" not in moved, (
        "light is exogenous; the only thing that reached it was growth→light fitting well")


def test_the_reverse_arrow_fits_perfectly_well_and_that_is_the_point():
    """`usable` cannot be the gate, and this is why: the nonsense arrow is a good fit."""
    universe = _linear_universe()
    back = universe.relations.get(("growth", "light"))
    assert back is not None and back.usable, "the backwards arrow is a fine statistical fit"
    assert not back.oriented, "and nothing has established that it runs that way"


def test_a_markov_equivalent_pair_is_answered_with_a_refusal_not_a_number():
    universe = _linear_universe()
    out = universe.intervene({"growth": 40.0}, base={"water": 8.0, "light": 5.0})
    assert not out.answerable
    assert "direction" in out.reason.lower() and "unverified" in out.reason.lower()


# --------------------------------------------------------------------------- #
# The four states, and what each buys
# --------------------------------------------------------------------------- #

def test_an_arrow_starts_unoriented_however_well_it_fits():
    universe = _linear_universe()
    forward = universe.relations[("water", "growth")]
    assert forward.usable and forward.r2 > 0.3
    assert forward.orientation == Orientation.UNKNOWN


def test_a_stated_law_orients_the_arrow_and_the_intervention_then_runs():
    universe = _linear_universe()
    universe.declare("water", "growth", sign=1, orientation=Orientation.ASSERTED)
    out = universe.intervene({"water": 2.0}, base={"water": 8.0, "light": 5.0})
    moved = {d.variable for d in out.changed()}
    assert "growth" in moved, "a declared arrow must still propagate"
    assert "light" not in moved, "and must not licence the arrow that was never declared"


def test_orientation_does_not_weaken_when_a_later_statement_says_less():
    """A verified direction is not un-verified by someone mentioning the arrow again."""
    universe = InternalUniverse()
    universe.declare("a", "b", sign=1, orientation=Orientation.VERIFIED)
    universe.declare("a", "b", sign=1, orientation=Orientation.ASSERTED)
    assert universe.relations[("a", "b")].orientation == Orientation.VERIFIED


def test_exogenous_severs_every_arrow_into_a_variable():
    """The negation `declare` never had. There was no way to say a variable has no parents."""
    universe = _linear_universe()
    universe.declare("water", "growth", sign=1, orientation=Orientation.ASSERTED)
    universe.declare("growth", "light", sign=1, orientation=Orientation.ASSERTED)  # a lie
    universe.exogenous("light")
    out = universe.intervene({"water": 2.0}, base={"water": 8.0, "light": 5.0})
    assert "light" not in {d.variable for d in out.changed()}


# --------------------------------------------------------------------------- #
# Persistence — the latent bug beside the reported one
# --------------------------------------------------------------------------- #

def test_a_declared_direction_survives_a_round_trip():
    """`sign` was written by `declare`, read by `directional`, and serialised by nothing."""
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1, orientation=Orientation.ASSERTED)
    revived = InternalUniverse()
    revived.load_dict(universe.to_dict())
    relation = revived.relations[("aag", "garmi")]
    assert relation.sign == 1
    assert relation.orientation == Orientation.ASSERTED


def test_a_snapshot_written_before_orientation_existed_still_loads():
    """Fail-soft: an old brain on disk must not be a broken brain."""
    universe = InternalUniverse()
    universe.declare("a", "b", sign=1, orientation=Orientation.ASSERTED)
    snapshot = universe.to_dict()
    for row in snapshot["relations"]:
        row.pop("orientation", None)
        row.pop("sign", None)
    revived = InternalUniverse()
    revived.load_dict(snapshot)
    assert revived.relations[("a", "b")].orientation == Orientation.UNKNOWN


# --------------------------------------------------------------------------- #
# Ambiguity becomes an agenda item
# --------------------------------------------------------------------------- #

def test_an_ambiguous_pair_is_seeded_as_an_experiment_rather_than_only_counted():
    """`ambiguous()` was right all along and was read by a demo and a test.

    Two rival hypotheses that predict differently under intervention is the one experiment that
    can break Markov equivalence, and the designer already picks by information gain.
    """
    from nyxara.njp.universe import ExperimentDesigner

    universe = _linear_universe()
    designer = ExperimentDesigner()
    seeded = universe.seed_ambiguities(designer)
    assert seeded > 0
    names = set(designer.hypotheses)
    assert any("water" in n and "growth" in n for n in names)
    # Both directions must be present, or there is nothing for evidence to decide between.
    assert len(names) >= 2


def test_an_experiment_that_separates_nothing_is_still_refused():
    """The seeding must not smuggle a zero-information experiment past the gain test."""
    from nyxara.njp.universe import ExperimentDesigner

    designer = ExperimentDesigner()
    InternalUniverse().seed_ambiguities(designer)      # nothing observed, nothing ambiguous
    assert not designer.hypotheses


# --------------------------------------------------------------------------- #
# Honesty about the cost
# --------------------------------------------------------------------------- #

def test_the_refusals_are_counted_separately_from_the_failures():
    """"Fewer answers" and "wrong answers" must not look the same in the report."""
    universe = _linear_universe()
    universe.intervene({"growth": 40.0}, base={"water": 8.0, "light": 5.0})
    stats = universe.stats()
    assert stats["abstained_for_direction"] >= 1
    assert stats["oriented_relations"] == 0
