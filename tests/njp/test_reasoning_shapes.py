"""EPISODE → SEMANTIC → SKILL → PROGRAM — grading the *form* of an inference, and using it.

**What was already built, and why nothing here duplicates it.** :mod:`nyxara.njp.genome` already
mines the shape of every derivation, counts recurrence, prices each shape by the MDL saving naming
it would earn, and separates :meth:`~nyxara.njp.genome.ReasoningGenome.candidates` from
:meth:`~nyxara.njp.genome.ReasoningGenome.liabilities`. It is recorded on the turn path. A second
shape-miner was started for this work and deleted once that was found: a worse duplicate of a
working organ is the failure this package refuses everywhere else.

:mod:`nyxara.growth.noesis` is likewise not the home for this. It is a real DreamCoder-style
library learner, and its 26 base primitives are ``add sub mul idiv mod eq lt … reverse sort_ cons
filter_gt count_gt`` — all integer and list arithmetic. Nothing there can touch a triple, a
relation or a causal edge, so it cannot *represent* ``CAUSAL_TRANSITIVITY``, let alone learn it.
Wiring its cycle into the turn loop would move ``cycles`` and ``compression_gain`` and make the
intelligence index report term R improving, while what improved was integer-list puzzles.

**The two gaps that were real.** The genome's grading hung off
:meth:`~nyxara.njp.brain.NJPBrain.resolve`, an external callback a conversation never makes, so on
a ``think()``-only session it recorded shapes for ever and graded none — measured ``graded: 0`` with
``success: None`` on every shape, which makes ``liabilities()`` structurally unable to fire and lets
``candidates()`` propose a shape for promotion on recurrence alone. And nothing anywhere read either
list, so a form with a measured record of failure went on answering at full confidence.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain


def _wrong_three_times() -> NJPBrain:
    """Three inheritances through `is_a → requires`, each corrected by the Master."""
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "birds need water",
                 "what does a sparrow need?", "sparrow needs food",
                 "a crow is a bird", "what does a crow need?", "crow needs grain",
                 "a robin is a bird", "what does a robin need?", "robin needs worms"):
        brain.think(turn)
    return brain


# --------------------------------------------------------------------------- #
# The shape is mined, and it is the same shape through different entities
# --------------------------------------------------------------------------- #
def test_two_inferences_through_different_entities_are_one_shape():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    brain.think("a crow is a bird")
    brain.think("what does a crow need?")
    assert brain.genome.stats()["shapes"] == 1
    assert brain.genome.shapes[("is_a", "requires")].seen >= 2


# --------------------------------------------------------------------------- #
# It is graded by the outcome the Master supplies, on the ordinary turn path
# --------------------------------------------------------------------------- #
def test_the_masters_later_statement_grades_the_form_not_only_the_answer():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    brain.think("sparrow needs food")
    assert brain.genome.stats()["graded"] >= 1


def test_a_shape_that_keeps_being_wrong_becomes_a_liability():
    brain = _wrong_three_times()
    liabilities = brain.genome.liabilities()
    assert liabilities, "a form used three times and wrong three times must be visible"
    assert liabilities[0].shape == ("is_a", "requires")
    assert liabilities[0].success == 0.0


def test_a_discredited_shape_is_not_proposed_for_promotion():
    """It recurs and it saves — and it is 0-for-3. Naming it would teach her to be wrong faster."""
    brain = _wrong_three_times()
    shape = brain.genome.shapes[("is_a", "requires")]
    assert shape.saving > 0.0, "it does recur often enough to be worth compressing on size alone"
    assert shape not in brain.genome.candidates()


def test_the_loop_reports_what_it_graded():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    report = brain.think("sparrow needs food").loop
    assert report is not None and report.shapes_graded >= 1


# --------------------------------------------------------------------------- #
# A derived answer reports what its own chain was worth
# --------------------------------------------------------------------------- #
def test_a_composed_answer_no_longer_reports_zero():
    """`core` prices the chain — link confidences times transitivity, capped below KNOWN.

    Starting the epistemic prior from `thought.confidence` discarded it: that field is the
    gauntlet's verdict and is correctly 0.0 whenever it abstains, which it does for every composed
    claim. So a two-hop chain and a five-hop one both read 0.0 and were indistinguishable.
    """
    brain = NJPBrain()
    brain.think("a finch is a bird")
    brain.think("birds need water")
    thought = brain.think("what does a finch need?")
    assert thought.answer == "water"
    assert thought.epistemic_confidence > 0.0
    assert thought.epistemic_confidence <= thought.derivation.confidence


# --------------------------------------------------------------------------- #
# The record is used, and only ever downward
# --------------------------------------------------------------------------- #
def test_an_untested_shape_discounts_nothing():
    """A new inference must not be penalised for being new."""
    brain = NJPBrain()
    brain.think("a finch is a bird")
    brain.think("birds need water")
    thought = brain.think("what does a finch need?")
    assert brain.genome.reliability(thought.trace) is None
    assert brain._temper_by_shape(0.8, thought) == 0.8


def test_a_measured_failure_rate_lowers_what_the_answer_is_worth():
    brain = NJPBrain()
    brain.think("a finch is a bird")
    brain.think("birds need water")
    thought = brain.think("what does a finch need?")
    base = thought.epistemic_confidence

    shape = brain.genome.shapes[("is_a", "requires")]
    shape.graded, shape.correct = 4, 1          # measured 1-for-4

    assert brain.genome.reliability(thought.trace) == 0.25
    assert brain._temper_by_shape(base, thought) < base


def test_the_shape_discount_can_only_lower():
    brain = NJPBrain()
    brain.think("a finch is a bird")
    brain.think("birds need water")
    thought = brain.think("what does a finch need?")
    shape = brain.genome.shapes[("is_a", "requires")]
    shape.graded, shape.correct = 4, 4          # a perfect record
    for base in (0.1, 0.5, 0.9):
        assert brain._temper_by_shape(base, thought) <= base


def test_reliability_separates_untested_from_failing():
    """`None` and `0.0` are different answers and the caller must tell them apart."""
    brain = NJPBrain()
    brain.think("a finch is a bird")
    brain.think("birds need water")
    thought = brain.think("what does a finch need?")
    assert brain.genome.reliability(thought.trace) is None

    shape = brain.genome.shapes[("is_a", "requires")]
    shape.graded, shape.correct = 3, 0
    assert brain.genome.reliability(thought.trace) == 0.0


def test_a_thought_with_no_trace_discounts_nothing():
    brain = NJPBrain()

    class _Bare:
        trace = None

    assert brain._temper_by_shape(0.7, _Bare()) == 0.7
