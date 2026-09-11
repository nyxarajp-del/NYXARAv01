"""Inventing a quantity nobody named — and the four gates that make that a claim rather than a hope.

The important tests here are not that something is discovered. They are that the *obvious* quantity
is refused as a duplicate, that a quantity which does not survive being jiggled is thrown out, that
nothing survives a world where nothing should, and that a name cannot be attached to anything that
has not earned one.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.genesis import (
    DEPTH, ENDS, JIGGLE, LUCK, SAME, SEPARATES, STEADY, STEPS, Measurement, Register, Trace,
    _recipes, admit, agree, criticise, mine,
)
from nyxara.njp.genesisschool import KNOWN, WORLDS, doses, fog, gains, seeded, spans


# --------------------------------------------------------------------------------------------- #
#  the search is arithmetic, not vocabulary
# --------------------------------------------------------------------------------------------- #
def test_the_search_is_composed_not_listed():
    """Four steps and six endings compose into hundreds of recipes; none of them is an answer."""
    got = _recipes()
    assert len(got) > 100, "a handful of hand-written quantities is a supplied vocabulary"
    assert len(STEPS) <= 6 and len(ENDS) <= 8, "the primitives must stay few to stay honest"
    assert DEPTH >= 2


def test_a_recipe_is_written_as_arithmetic():
    assert Measurement(steps=(), end="mean").recipe == "mean(x)"
    # Steps are stored outermost first, so ("d", "norm") is d applied to norm applied to x.
    assert Measurement(steps=("d", "norm"), end="spread").recipe == "spread(d(norm(x)))"
    assert Measurement(steps=("norm", "d"), end="spread").recipe == "spread(norm(d(x)))"


def test_nothing_in_the_worlds_names_a_property():
    """The exam's worlds emit probes and readings. A trait name anywhere would be the old game."""
    import inspect

    import nyxara.njp.genesisschool as school

    source = inspect.getsource(school)
    for word in ("sharpness", "difficulty", "confidence", "boundary_", "weight", "salience"):
        assert word not in source, f"the world names {word}, which is a supplied vocabulary"


def test_a_measurement_carries_a_number_until_somebody_defends_a_name():
    got = Measurement(steps=("d",), end="spread", number=7)
    assert got.name == "measurement 7" and not got.called
    with pytest.raises(ValueError, match="earned"):
        got.christen("boundary sharpness")


def test_a_name_is_allowed_only_after_something_stands():
    from nyxara.njp.latent import Standing

    got = Measurement(steps=("d",), end="spread", number=7)
    got.held.append(Standing(kind="observational"))
    assert got.christen("how fast it falls away").name == "how fast it falls away"


# --------------------------------------------------------------------------------------------- #
#  the gates
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_worlds():
    return spans(), gains()


def test_the_obvious_quantity_is_refused_as_a_duplicate(two_worlds):
    """`mean(x)` is the score. Rediscovering it and calling it an invention is the failure mode."""
    a, b = two_worlds
    plain = Measurement(steps=(), end="mean", axis=0, number=1)
    got = criticise(plain, a, b, seeded(), rng=random.Random(87))
    assert got.refused and "different arithmetic" in got.refused
    assert not got.held


def test_a_quantity_that_moves_when_the_reading_is_jiggled_is_thrown_out(two_worlds):
    """It is measuring the observation rather than the item."""
    a, b = two_worlds
    every = [criticise(m, a, b, (), rng=random.Random(87)) for m in mine(a)]
    shaky = [m for m in every if m.refused and "jiggling" in m.refused]
    assert shaky, "with noise in the readings, something must fail the steadiness gate"
    assert all(m.steady < STEADY for m in shaky)


def test_a_flat_quantity_is_refused_before_anything_else(two_worlds):
    """And an axis the probes do not have reads nothing rather than raising."""
    a, b = two_worlds
    assert a[0].along(9) == [] and a[0].along(-1) == []
    flat = Measurement(steps=(), end="sum", axis=9, number=1)
    got = criticise(flat, a, b, (), rng=random.Random(87))
    assert got.refused and "measures nothing" in got.refused


def test_nowhere_to_check_is_reported_rather_than_passed():
    got = criticise(Measurement(steps=(), end="mean"), spans(), [], ())
    assert got.refused and "nowhere to check" in got.refused


def test_a_survivor_holds_observational_evidence_and_nothing_above_it(two_worlds):
    a, b = two_worlds
    kept = [m for m in (criticise(m, a, b, seeded(), rng=random.Random(87)) for m in mine(a))
            if m.held]
    assert kept, "something must survive on a world where the structure is real"
    for got in kept:
        assert got.claims("observational")
        assert not got.causal, "no intervention happened, so nothing may claim to cause"
        assert "predicts" in got.held[-1].says


# --------------------------------------------------------------------------------------------- #
#  three worlds, three jobs
# --------------------------------------------------------------------------------------------- #
def test_nothing_survives_a_world_with_no_structure_in_it():
    """The half of the exam that matters. Finding structure is easy; finding *this* structure is not."""
    got = admit(fog(), fog(), fog(), seeded=seeded(), rng=random.Random(87))
    assert got.kept == [], f"invented {[m.recipe for m in got.kept]} out of fog"


def test_checking_in_a_meaningless_world_keeps_nothing():
    got = admit(spans(), fog(), doses(), seeded=seeded(), rng=random.Random(87))
    assert got.kept == []


def test_transfer_is_not_optional():
    got = admit(spans(), gains(), fog(), seeded=seeded(), rng=random.Random(87))
    assert got.kept == []
    assert any("third world" in m.refused for m in got.refused)


def test_agreement_is_zero_when_one_column_does_not_move():
    assert agree([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert agree([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0
    assert agree([1.0], [1.0]) == 0.0
    assert 0.5 < SAME <= 1.0 and 0.0 < STEADY < 1.0 and SEPARATES > 0 and JIGGLE > 0


def test_a_trace_reads_only_along_one_axis_at_a_time():
    """A neighbourhood that moved two coordinates gives a number that is about neither."""
    t = Trace(probes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
              readings=(0.1, 0.2, 0.3, 0.9))
    assert t.axes == 2
    assert t.along(0) == [(0.0, 0.1), (1.0, 0.2)]
    assert t.along(1) == [(0.0, 0.1), (1.0, 0.3)]


def test_an_empty_register_says_so():
    assert "nothing survived" in Register().render()
    assert Register().names == []
    assert mine([]) == []


def test_the_plain_reading_height_does_not_separate_the_groups(two_worlds):
    """The fixture's honesty check. If height worked, the discovery would be trivial.

    The worlds draw the centre reading from the same distribution on both sides, so every quantity
    computed from the middle alone is matched. Measured: `max(x)` separates them by 0.2091 pooled
    spreads, under the 0.30 bar. What is left to find is the *shape* around it, and nothing in the
    supplied primitives is a word for that.
    """
    a, b = two_worlds
    height = criticise(Measurement(steps=(), end="max", axis=0, number=1), a, b, (),
                       rng=random.Random(87))
    assert height.refused and "separates" in height.refused
    assert height.separates < SEPARATES


def test_normalising_is_what_turns_a_height_into_a_shape(two_worlds):
    """`mean(norm(x))` survives where `mean(x)` and `max(x)` do not, and that is the composition.

    Dividing through by the item's own top throws away how high the neighbourhood is and leaves how
    broad it is. Nobody wrote that quantity down; it fell out of one step composed with one ending.
    """
    a, b = two_worlds
    shaped = criticise(Measurement(steps=("norm",), end="mean", axis=0, number=1), a, b, seeded(),
                       rng=random.Random(87))
    assert shaped.held and shaped.separates > 1.0
    assert shaped.recipe == "mean(norm(x))"
