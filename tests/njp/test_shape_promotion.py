"""A reasoning form she keeps re-deriving becomes a strategy she can choose.

`njp/genome.py` has computed this for a long time. It strips the entities out of a derivation and
keeps the form — `aag —causes→ garmi —causes→ pasina` becomes `('causes', 'causes')` — counts how
often each form recurs, tracks how often it was right, and computes a real MDL saving: `n·k` steps
to keep re-deriving against `k + n` to name it once and apply it. `candidates()` returns the forms
that clear all three conditions.

Nothing called it. `grep -rn '\\.candidates()'` finds `genome.stats()` and the genome's own tests.
The producer was finished and the consumer was never built, so a shape she had re-derived forty
times was reported in a stats dict and re-derived a forty-first.

`metareason.register` takes arms at runtime with no freeze, and a fresh arm is explored first
because `ucb` returns `prior + 1.0` at zero trials. So the bridge is short. What it needs is a
walk that follows a *mixed* predicate chain — `_compose` follows one predicate repeatedly, which
is a different shape of thing — and a promotion step that refuses the forms the genome already
knows are bad.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain


def _teach(brain: NJPBrain, lines) -> None:
    for line in lines:
        brain.think(line)


# --------------------------------------------------------------------------- #
# The walk
# --------------------------------------------------------------------------- #

def test_a_shape_walk_follows_a_mixed_predicate_chain():
    """`_compose` chains one predicate with itself. A shape is a sequence of different ones."""
    brain = NJPBrain()
    _teach(brain, ("a sparrow is a bird", "a bird needs water"))

    out = brain.learner.walk_shape("sparrow", ("is_a", "requires"))
    assert out.ok
    assert out.answer == "water"
    assert len(out.support) == 2
    assert out.support[0][1] == "is_a" and out.support[1][1] == "requires"


def test_a_shape_walk_that_does_not_complete_returns_nothing():
    brain = NJPBrain()
    _teach(brain, ("a sparrow is a bird",))
    assert not brain.learner.walk_shape("sparrow", ("is_a", "requires")).ok


def test_a_shape_answer_is_priced_below_a_stated_fact():
    """A chain is defeasible by construction and its confidence has to say so."""
    brain = NJPBrain()
    _teach(brain, ("a sparrow is a bird", "a bird needs water"))
    out = brain.learner.walk_shape("sparrow", ("is_a", "requires"))
    assert 0.0 < out.confidence < 1.0
    assert out.kind == "composed"


def test_a_longer_chain_is_worth_less_than_a_shorter_one():
    """Depth is priced, not merely bounded."""
    brain = NJPBrain()
    _teach(brain, ("a sparrow is a bird", "a bird needs water",
                   "water causes growth"))
    short = brain.learner.walk_shape("sparrow", ("is_a", "requires"))
    longer = brain.learner.walk_shape("sparrow", ("is_a", "requires", "causes"))
    assert short.ok and longer.ok
    assert longer.confidence < short.confidence


def test_the_walk_never_returns_the_subject_it_started_from():
    """`a causes b` and `b causes a` are both sayable; deriving that a causes itself is not."""
    brain = NJPBrain()
    _teach(brain, ("a causes b", "b causes a"))
    out = brain.learner.walk_shape("a", ("causes", "causes"))
    assert out.answer != "a"


# --------------------------------------------------------------------------- #
# The promotion
# --------------------------------------------------------------------------- #

def test_a_recurring_successful_shape_becomes_a_registered_strategy():
    brain = NJPBrain()
    if brain.genome is None or brain.metareason is None:
        pytest.skip("genome or metareason gated off in this build")

    # Built directly: what is under test is promotion, not extraction.
    from nyxara.njp.genome import Shape
    shape = ("is_a", "requires")
    brain.genome.shapes[shape] = Shape(shape=shape, seen=8, graded=8, correct=8,
                                       total_depth=24)

    promoted = brain._promote_shapes()
    assert promoted >= 1
    assert "shape:is_a>requires" in brain.metareason.strategies


def test_a_shape_the_genome_calls_a_liability_is_never_promoted():
    """Recurring and usually wrong is a habit worth breaking, not a primitive worth naming."""
    brain = NJPBrain()
    if brain.genome is None or brain.metareason is None:
        pytest.skip("genome or metareason gated off in this build")

    from nyxara.njp.genome import Shape
    brain.genome.shapes[("is_a", "causes")] = Shape(shape=("is_a", "causes"), seen=20,
                                                    graded=20, correct=2, total_depth=60)
    brain._promote_shapes()
    assert "shape:is_a>causes" not in brain.metareason.strategies


def test_promotion_is_idempotent():
    brain = NJPBrain()
    if brain.genome is None or brain.metareason is None:
        pytest.skip("genome or metareason gated off in this build")

    from nyxara.njp.genome import Shape
    brain.genome.shapes[("is_a", "requires")] = Shape(shape=("is_a", "requires"), seen=8, graded=8,
                                                   correct=8, total_depth=24)
    first = brain._promote_shapes()
    second = brain._promote_shapes()
    assert first >= 1 and second == 0


def test_a_promoted_shape_can_actually_answer():
    """The arm is not decorative: it walks the chain and returns the answer."""
    brain = NJPBrain()
    if brain.genome is None or brain.metareason is None:
        pytest.skip("genome or metareason gated off in this build")

    _teach(brain, ("a sparrow is a bird", "a bird needs water"))
    from nyxara.njp.genome import Shape
    brain.genome.shapes[("is_a", "requires")] = Shape(shape=("is_a", "requires"), seen=8, graded=8,
                                                   correct=8, total_depth=24)
    brain._promote_shapes()

    strategy = brain.metareason.strategies["shape:is_a>requires"]
    answer = strategy.solve("what does sparrow need", {"subject": "sparrow"})
    assert answer and "water" in str(answer).lower()


def test_a_promoted_arm_keeps_its_record_across_a_restart():
    """The prerequisite fixed earlier: `load_dict` used to drop arms not registered yet."""
    brain = NJPBrain()
    if brain.genome is None or brain.metareason is None:
        pytest.skip("genome or metareason gated off in this build")

    from nyxara.njp.genome import Shape
    brain.genome.shapes[("is_a", "requires")] = Shape(shape=("is_a", "requires"), seen=8, graded=8,
                                                   correct=8, total_depth=24)
    brain._promote_shapes()
    arm = brain.metareason.strategies["shape:is_a>requires"]
    arm.wins, arm.trials = 6.0, 7
    snapshot = brain.metareason.to_dict()

    revived = NJPBrain()
    revived.metareason.load_dict(snapshot)
    revived.genome.shapes[("is_a", "requires")] = Shape(shape=("is_a", "requires"), seen=8, graded=8,
                                                     correct=8, total_depth=24)
    revived._promote_shapes()
    assert revived.metareason.strategies["shape:is_a>requires"].trials == 7
