"""What two changes do together — and the three ways that number is nothing.

``I = ΔAB − ΔA − ΔB`` is a subtraction of four noisy means, and subtractions like that are very
good at being large. So most of what is pinned here is the controls: the bootstrap, the order test,
and the do-nothing change. Two of these tests exist because the exam failed first and the fix is
what they pin.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.interact import (
    DRAWS, LUCK, MECHANISMS, ORDER, PLACEBO, TOGETHER, Interaction, Reading, compose, interaction,
    mechanisms, search,
)
from nyxara.njp.interactschool import (
    FLIP_A, FLIP_B, ITEMS, KNOWN, NOTHING, Box, _additive, _leaky, _measure, _ordered, _synergy,
    examine,
)
from nyxara.njp.space import Held, Intervention


def _look(rule, **kw):
    return interaction(FLIP_A, FLIP_B, Box(rule=rule), measure=_measure, nothing=NOTHING,
                       rng=random.Random(85), **kw)


# --------------------------------------------------------------------------------------------- #
#  the thing itself
# --------------------------------------------------------------------------------------------- #
def test_it_finds_what_no_single_variable_search_can():
    """Neither switch alone does anything. Nine single-variable experiments would find nine nulls."""
    got = _look(_synergy)
    assert got.found and got.verdict == "interacting"
    assert got.alone_a == pytest.approx(got.base, abs=0.01)
    assert got.alone_b == pytest.approx(got.base, abs=0.01)
    assert got.extra > TOGETHER


def test_additivity_is_refused_however_large_the_parts():
    """Both switches work. Together they are worth exactly the sum, so there is nothing to find."""
    got = _look(_additive)
    assert not got.found and got.verdict == "additive"
    assert abs(got.extra) < TOGETHER
    assert got.together > got.base, "the parts really do work — that is what makes this a test"


def test_a_negative_interaction_is_a_finding_too():
    from nyxara.njp.interactschool import _antagonism

    got = _look(_antagonism)
    assert got.found and got.extra < -TOGETHER


# --------------------------------------------------------------------------------------------- #
#  the three controls
# --------------------------------------------------------------------------------------------- #
def test_order_dependence_is_not_reported_as_an_interaction():
    """A consumes what B needs, so `A then B` and `B then A` are different worlds."""
    got = _look(_ordered)
    assert got.verdict == "order-dependent"
    assert abs(got.order) >= ORDER
    assert not got.found, "a mutation is not an interaction, whatever the subtraction says"


def test_a_linear_leak_is_caught_directly_because_the_subtraction_cancels_it():
    """The defect the exam found in the organ, pinned.

    The first apparatus control was written only as `I(A, nothing)`. A leak growing linearly with
    how often the harness has been touched cancels out of that subtraction **exactly** — measured
    at +0.0000 on a fixture whose every reading was inflated by +0.1982. A control written in the
    same shape as the thing it guards inherits its blind spots.
    """
    got = _look(_leaky)
    assert got.verdict == "apparatus"
    assert got.placebo >= PLACEBO
    assert abs(got.extra) < TOGETHER, \
        "the interaction form sees nothing here — that is the whole point"


def test_without_a_do_nothing_change_the_apparatus_is_not_checked():
    """Unsupplied is reported, never passed."""
    got = interaction(FLIP_A, FLIP_B, Box(rule=_leaky), measure=_measure, nothing=None,
                      rng=random.Random(85))
    assert "was not checked" in got.says
    assert got.placebo == 0.0 and got.verdict != "apparatus"


def test_the_bootstrap_is_what_makes_the_sign_a_claim():
    got = _look(_synergy, draws=DRAWS)
    assert 0.0 < got.luck <= LUCK
    assert 0.0 < LUCK < 0.5


# --------------------------------------------------------------------------------------------- #
#  what it refuses to do
# --------------------------------------------------------------------------------------------- #
def test_an_interaction_is_not_a_mechanism():
    """Found, and consistent with every mechanism on the list. Narrowing here would be inventing."""
    assert mechanisms(_look(_synergy)) == list(MECHANISMS)
    assert len(MECHANISMS) >= 5
    assert mechanisms(_look(_additive)) == []


def test_runs_of_different_lengths_are_not_subtracted():
    """Five runs reading different items are five different questions."""
    def _shrinking(box):
        return _measure(box)[: ITEMS - (10 if box.a else 0)]

    got = interaction(FLIP_A, FLIP_B, Box(rule=_synergy), measure=_shrinking, nothing=NOTHING)
    assert not got.ran
    assert "not the same question" in got.says


def test_an_uncomposable_intervention_says_so_rather_than_being_skipped():
    opaque = Intervention(verb="add", on="x", run=lambda: None)
    got = interaction(opaque, FLIP_B, Box(rule=_synergy), measure=_measure)
    assert not got.ran and "not composable" in got.says
    with pytest.raises(ValueError, match="composable"):
        compose(opaque, FLIP_B)


def test_a_broken_promise_makes_the_number_about_nothing():
    """V.83's rule, carried into composition: the promises of both halves must both hold."""
    strict = Held(name="A is never on", read=lambda box: float(box.a), slack=0.5)
    fussy = Intervention(verb="add", on="switch A", change=FLIP_A.change, holds=(strict,))
    got = interaction(fussy, FLIP_B, Box(rule=_synergy), measure=_measure, nothing=NOTHING)
    assert got.verdict == "spoiled" and got.broke == ["A is never on"]
    assert not got.found


def test_a_bare_interaction_claims_nothing():
    blank = Interaction()
    assert blank.verdict == "not run" and not blank.found
    assert Reading().score == 0.0 and not Reading().sound
    assert Reading(per_item=(1.0,), broke=("x",)).sound is False
    assert 0.0 < TOGETHER < 0.5 and 0.0 < ORDER < 0.5 and 0.0 < PLACEBO < 0.5


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
def test_the_search_passes_its_retrodiction():
    got = examine()
    assert got["right"] == got["of"] == 6
    assert got["invented"] == 0 and got["missed"] == 0
    assert got["passes"]


def test_half_the_worlds_are_ones_where_finding_a_pair_is_wrong():
    """Without them the exam is passed by a function that answers `they interact` to everything."""
    assert sum(1 for c in KNOWN if c.verdict != "interacting") == 4


def test_calling_everything_an_interaction_fails_the_exam():
    """The check on the check. An exam nothing can fail is not an exam."""
    import nyxara.njp.interact as module

    real = module.Interaction.verdict
    try:
        module.Interaction.verdict = property(lambda self: "interacting")
        got = examine()
        assert not got["passes"] and got["invented"] > 0
    finally:
        module.Interaction.verdict = real
    assert examine()["passes"]


def test_a_pairwise_search_returns_findings_first():
    got = search([FLIP_A, FLIP_B, NOTHING], Box(rule=_synergy), measure=_measure,
                 nothing=NOTHING, rng=random.Random(85))
    assert len(got) == 3, "three interventions make three pairs"
    assert got[0].found, "the real one sorts to the top"
