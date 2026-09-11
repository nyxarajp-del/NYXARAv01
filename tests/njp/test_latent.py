"""What varies that nobody is varying — and why finding it proves almost nothing.

The tests that matter here are not the ones where a cause is found. They are the ones where a trait
separates the failures from the successes perfectly and is **not** a cause: a confound, a
consequence, a coincidence. The exam's world is built so the consequence separates the groups
*better* than the cause does, because a module that reports the strongest correlation must fail.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from nyxara.njp.latent import (
    DRAWS, FLAT, LADDER, LUCK, SEPARATES, Candidate, Standing, Trait, put_to_the_test, sift,
    uncontrolled,
)
from nyxara.njp.latentschool import (
    HEAVY, ITEMS, KNOWN, SHELF, TRAITS, Item, examine, outcome, retrodict, world,
)
from nyxara.njp.space import Intervention


@pytest.fixture(scope="module")
def items():
    return world()


@pytest.fixture(scope="module")
def found(items):
    looked = uncontrolled(TRAITS, SHELF, items)
    return {c.trait: c for c in sift(looked, items, outcome(items), rng=random.Random(86))}


# --------------------------------------------------------------------------------------------- #
#  the ladder is the point
# --------------------------------------------------------------------------------------------- #
def test_predicting_never_becomes_causing_by_itself(found):
    """Every survivor of the sift holds observational evidence and nothing above it."""
    for got in found.values():
        if got.claims("observational"):
            assert not got.causal, f"{got.trait} was promoted without an intervention"
            assert got.stands_as == "observational"


def test_a_claim_refuses_every_rung_it_does_not_hold():
    got = Candidate(trait="z", held=[Standing(kind="observational")])
    assert got.claims("observational")
    for kind in LADDER[1:]:
        assert not got.claims(kind), kind
    assert not got.causal
    assert not got.claims("not a kind at all")


def test_the_ladder_has_a_rung_for_each_different_experiment():
    assert LADDER[0] == "observational" and LADDER[1] == "interventional"
    assert len(set(LADDER)) == len(LADDER) >= 5


# --------------------------------------------------------------------------------------------- #
#  the impostors
# --------------------------------------------------------------------------------------------- #
def test_the_consequence_separates_better_than_the_cause(found):
    """The fixture's whole reason for existing. Strongest correlation is the wrong answer here."""
    assert found["scar"].separates > found["weight"].separates
    assert found["shadow"].separates == pytest.approx(found["weight"].separates, abs=0.01)


def test_only_the_cause_survives_being_changed(items, found):
    by_name = {t.name: t for t in TRAITS}
    for name in ("weight", "shadow", "scar"):
        put_to_the_test(found[name], by_name[name], items, outcome=outcome, others=TRAITS)
    assert found["weight"].causal
    assert not found["shadow"].causal, "a confound moves nothing when moved alone"
    assert not found["scar"].causal, "erasing what the failure wrote does not undo the failure"


def test_an_intervention_that_drags_something_else_identifies_nothing(items, found):
    by_name = {t.name: t for t in TRAITS}
    got = put_to_the_test(found["twin"], by_name["twin"], items, outcome=outcome, others=TRAITS)
    assert got.refused and "weight" in got.refused
    assert not got.causal


def test_a_trait_that_cannot_be_changed_stops_at_what_it_predicts(items):
    stuck = Trait("weight", read=lambda i: i.weight, change=None)
    got = put_to_the_test(Candidate(trait="weight"), stuck, items, outcome=outcome)
    assert not got.causal
    assert "no way to change it" in got.held[-1].says


# --------------------------------------------------------------------------------------------- #
#  the two gates before any test
# --------------------------------------------------------------------------------------------- #
def test_a_constant_is_refused_before_it_is_tested(items, found):
    """It cannot explain why some items differ, and putting it through a null invites a fluke.

    Two gates catch it and the first one wins, which is why it is absent from ``found`` rather than
    present and refused: `uncontrolled` drops it before the sift ever sees it. `sift` refuses it
    too, on its own, and both are checked here — a gate that is only ever reached second is a gate
    nobody has tested.
    """
    assert "paint" not in found, "the first gate drops it before it is ever tested"
    paint = next(t for t in TRAITS if t.name == "paint")
    alone = sift([paint], items, outcome(items), rng=random.Random(86))[0]
    assert alone.refused and "does not vary" in alone.refused
    assert not alone.held and not alone.causal


def test_noise_is_refused_by_the_null_and_not_by_taste(found):
    assert not found["dust"].held
    assert found["dust"].separates < SEPARATES
    assert 0.0 < LUCK < 0.5 and DRAWS >= 50 and FLAT > 0


def test_what_is_already_on_the_shelf_is_not_a_discovery(items):
    """Otherwise the module scores well by listing its own inputs back."""
    looked = [t.name for t in uncontrolled(TRAITS, SHELF, items)]
    assert "size" not in looked, "size is manipulated by something on the shelf"
    assert "weight" in looked and "shadow" in looked
    assert "paint" not in looked, "a constant is not a blind spot, it is a constant"


def test_naming_a_knob_differently_hides_it_and_that_is_stated(items):
    """A real limitation of matching by name, pinned so it cannot be forgotten."""
    other = (Intervention(verb="replace", on="how big it is", change=lambda x: x),)
    assert "size" in [t.name for t in uncontrolled(TRAITS, other, items)]


# --------------------------------------------------------------------------------------------- #
#  the limitation the module states about itself
# --------------------------------------------------------------------------------------------- #
def test_it_refuses_the_real_cause_when_the_cause_updates_its_own_consequences():
    """A known wrong answer, pinned rather than left to be discovered later.

    The drift check cannot tell *moved something downstream of this trait* from *moved something
    that moves this trait back*. The first is a causal chain behaving correctly. On a world where
    lightening an item also recomputes the shadow it casts, the true cause is therefore refused —
    and the honest fix is a supplied ordering over the traits, not a cleverer statistic.
    """
    items = world()

    def _lighten_and_recompute(item: Item) -> Item:
        lighter = max(0.0, item.weight - 0.45)
        return replace(item, weight=lighter, shadow=lighter * 2.0)

    honest = Trait("weight", read=lambda i: i.weight, change=_lighten_and_recompute)
    got = put_to_the_test(Candidate(trait="weight"), honest, items, outcome=outcome, others=TRAITS)
    assert got.refused and "shadow" in got.refused
    assert not got.causal, "the real cause, refused — this is the stated limitation, not a surprise"


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_the_module_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 6
    assert got["invented"] == 0 and got["missed"] == 0
    assert got["causes"] == 1
    assert got["passes"]


def test_more_than_one_trait_must_predict_or_the_exam_is_vacuous():
    """A world where only the cause correlates is a world this module is not needed in."""
    got = retrodict()
    assert got["predicted"] >= 3
    assert sum(1 for c in KNOWN if c.truth == "predicts") == 2


def test_reporting_the_strongest_correlation_fails_the_exam():
    """The check on the check. That module scores four of four and is wrong three times."""
    import nyxara.njp.latent as module

    real = module.Candidate.causal
    try:
        module.Candidate.causal = property(lambda self: self.claims("observational"))
        got = examine()
        assert not got["passes"] and got["invented"] >= 2
    finally:
        module.Candidate.causal = real
    assert examine()["passes"]


def test_the_world_keeps_the_rule_it_is_never_asked_about():
    items = world()
    assert len(items) == ITEMS
    failed = outcome(items)
    assert all(f == (i.weight >= HEAVY) for i, f in zip(items, failed))
    assert 0 < sum(failed) < len(failed), "both groups must exist for anything to separate them"
