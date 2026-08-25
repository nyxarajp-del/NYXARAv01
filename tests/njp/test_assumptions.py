"""Phase 3's missing half: the assumptions her model rests on that nothing has examined.

She could already say what she knows (`beliefs`) and what she does not (`curiosity`). Neither can
answer the question that actually kills a model — *which assumptions have I never tested?* — and
that third knowledge state was the whole of what Phase 3 was missing: the hypothesis engine, the
experiment engine and the counterfactual engine were all present and firing.

An arrow is not one claim. ``aag → garmi`` asserts four separable things, and she was holding all
four on the evidence for one:

    sole_cause     nothing *else* produces the effect
    unconditional  it holds outright, not only under some circumstance nobody named
    directed       it runs this way and not the other
    unconfounded   nothing produces *both* ends

Every test below is a query over her own event record, never a guess — which is what makes this an
organ rather than a checklist. And the milestone the phase is graded on runs through it end to
end: ``unknown → hypothesis → experiment → result → discovery``.
"""

from __future__ import annotations

import pytest

from nyxara.njp import NJPBrain
from nyxara.njp.assume import AssumptionKind, AssumptionMiner, AssumptionStatus


def _confounded() -> NJPBrain:
    """`lagi` precedes both `hui` and `aaya`; `hui` also occurs on its own."""
    brain = NJPBrain()
    for line in (["aag lagi", "garmi hui", "pasina aaya"] * 5 + ["garmi hui"] * 4) * 2:
        brain.think(line)
    return brain


# --------------------------------------------------------------------------- #
# the third knowledge state
# --------------------------------------------------------------------------- #

def test_every_arrow_surfaces_four_untested_claims():
    """Believing an arrow creates four claims nothing has examined. Silence is not the record."""
    brain = _confounded()
    miner = AssumptionMiner()
    mined = miner.mine(brain.world)
    assert mined, brain.world.stats()
    assert len(mined) % len(AssumptionKind.ALL) == 0
    assert all(a.status == AssumptionStatus.UNTESTED for a in mined)
    assert miner.stats()["unknown_unknowns"] == len(mined)


def test_a_self_loop_is_not_an_arrow():
    """It arises whenever one kind of event repeats inside the window, and every test here is
    degenerate on it — its reverse is itself, so `directed` always "refutes" a claim nobody made."""
    brain = _confounded()
    miner = AssumptionMiner()
    miner.mine(brain.world)
    assert not [a for a in miner.assumptions.values() if a.cause == a.effect]


def test_the_open_surface_is_reported_as_a_level_not_a_sum():
    """`assumptions_open` is what Phase 3 exists to report, and it read zero except on one turn
    in ten when it was assigned from every report rather than from the passes that ran."""
    brain = _confounded()
    totals = brain.loop.stats()["totals"]
    assert totals["assumptions_open"] == brain.assumptions.stats()["unknown_unknowns"]
    assert totals["assumptions_open"] > 0, totals


# --------------------------------------------------------------------------- #
# the four tests, each against the record
# --------------------------------------------------------------------------- #

def test_a_rival_cause_refutes_sole_cause_and_names_itself():
    brain = _confounded()
    miner = AssumptionMiner()
    miner.mine(brain.world)
    miner.test(brain.world, limit=64)
    found = [a for a in miner.discoveries() if a.kind == AssumptionKind.SOLE_CAUSE]
    assert found, miner.stats()
    assert all(a.found for a in found), [a.to_dict() for a in found]


def test_a_shared_upstream_cause_refutes_unconfounded():
    """The one threat that makes a genuine correlation a false arrow, and the only one of the
    four that cannot be seen by looking at the pair alone."""
    brain = _confounded()
    miner = AssumptionMiner()
    miner.mine(brain.world)
    miner.test(brain.world, limit=64)
    found = [a for a in miner.discoveries() if a.kind == AssumptionKind.UNCONFOUNDED]
    assert found, miner.stats()
    assert all(a.found for a in found)


def test_a_cause_that_often_fails_to_produce_its_effect_is_conditional():
    """The record answers *whether* there is a condition, never *which* — and that is the value.

    A conditional probability well below 1 says the arrow depends on something the model does not
    have, without needing to know what it is, which is exactly the shape of an unknown-unknown.
    """
    brain = NJPBrain()
    brain.think("aag se garmi hoti hai")
    # Fire sometimes produces heat and often does not.
    for _ in range(4):
        brain.think("aag lagi")
        brain.think("pasina aaya")
    for _ in range(4):
        brain.think("aag lagi")
        brain.think("garmi hui")
    miner = AssumptionMiner()
    miner.mine(brain.world)
    miner.test(brain.world, limit=64)
    conditional = [a for a in miner.assumptions.values()
                   if a.kind == AssumptionKind.UNCONDITIONAL
                   and a.status != AssumptionStatus.UNTESTED]
    assert conditional, miner.stats()


def test_too_few_occurrences_is_undecidable_not_untested():
    """"I looked and could not tell" and "nobody looked" call for different next actions."""
    brain = NJPBrain()
    brain.think("aag se garmi hoti hai")
    brain.think("aag lagi")
    brain.think("garmi hui")
    miner = AssumptionMiner()
    miner.mine(brain.world)
    decided = miner.test(brain.world, limit=64)
    assert decided, brain.world.stats()
    assert any(a.status == AssumptionStatus.UNDECIDABLE for a in decided), \
        [a.to_dict() for a in decided]


# --------------------------------------------------------------------------- #
# the milestone, end to end
# --------------------------------------------------------------------------- #

def test_unknown_becomes_hypothesis_becomes_experiment_becomes_discovery():
    """Phase 3's milestone chain, driven through the brain rather than the miner.

        unknown  ->  hypothesis  ->  experiment  ->  result  ->  discovery
    """
    brain = _confounded()
    totals = brain.loop.stats()["totals"]
    assert totals["assumptions_mined"] > 0, totals          # unknown
    assert totals["assumptions_tested"] > 0, totals          # experiment
    assert totals["assumptions_refuted"] > 0, totals         # result
    discoveries = brain.assumptions.discoveries()
    assert discoveries, brain.assumptions.stats()
    # A discovery names what was found — the rival cause, the reversal, the confounder.
    assert all(d.found for d in discoveries), [d.to_dict() for d in discoveries]


def test_an_untested_assumption_becomes_one_of_her_own_questions():
    """An unknown-unknown has to become a *known* one before anything can act on it."""
    brain = _confounded()
    gaps = brain.curiosity.stats().get("by_gap", {})
    assert gaps.get("untested_assumption", 0) > 0, gaps


def test_a_discovery_is_written_back_into_the_model_it_was_an_assumption_of():
    """Finding that C also causes B and not recording C -> B makes the discovery a remark."""
    brain = NJPBrain()
    before = brain.world.stats()["stated_laws"]
    for line in (["aag lagi", "garmi hui", "pasina aaya"] * 5 + ["garmi hui"] * 4) * 2:
        brain.think(line)
    assert brain.world.stats()["stated_laws"] > before, brain.world.stats()


def test_a_condition_and_a_confounder_are_not_invented_as_arrows():
    """Both are real findings and neither is an edge: "something unmodelled decides when" names
    no variable, and a shared upstream cause is already two arrows the record has."""
    from nyxara.njp.integrate import LearningLoop
    import inspect
    source = inspect.getsource(LearningLoop._record_discovery)
    assert "UNCONDITIONAL" not in source
    assert "UNCONFOUNDED" not in source


# --------------------------------------------------------------------------- #
# honesty
# --------------------------------------------------------------------------- #

def test_surviving_is_recorded_as_holding_never_as_proven():
    """Absence of a counterexample in a short record is not a demonstration."""
    brain = _confounded()
    miner = AssumptionMiner()
    miner.mine(brain.world)
    miner.test(brain.world, limit=64)
    held = [a for a in miner.assumptions.values() if a.status == AssumptionStatus.HOLDS]
    assert held
    assert AssumptionStatus.HOLDS == "holds"
    assert not any(getattr(a, "proven", False) for a in held)


def test_the_miner_never_raises():
    miner = AssumptionMiner()
    for junk in (None, object(), "not a world"):
        assert miner.mine(junk) == []
        assert miner.test(junk) == []
    assert isinstance(miner.stats(), dict)
