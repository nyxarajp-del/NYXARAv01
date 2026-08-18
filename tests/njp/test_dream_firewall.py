"""A dream must never become one of the independent supports.

`truth.py` has argued this in its docstring since it was written — *"REAL DATA → belief;
SIMULATION → hypothesis → test request → real evidence → belief"* — and `judge()` implements it
correctly: a reading marked `simulated` is recorded, is reported, may hold a claim at CONJECTURE,
and can never count toward `min_sources`.

Nothing ever set the mark. `Evidence.simulated` was a field no `Source` assigned, so the rule was
a gate with nothing routed through it.

The second half of this file is the hole that mattered more, and it was opened by the ingest work
rather than found in it: `brain.observations` is built from **every triple in the fact store**,
and `ObservationSource` counted matching rows. A quarter-million ConceptNet rows are not a
quarter-million witnesses — they are one source read a quarter of a million times, which is the
failure `EvidenceKind.CONSENSUS` already names and the one a bulk loader makes cheap.
"""

from __future__ import annotations

import pytest

from nyxara.njp.beliefs import BeliefLedger, EvidenceKind
from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import GroundedTriple
from nyxara.njp.truth import (
    SIMULATED_ORIGINS, Evidence, ObservationSource, Source, TruthGauntlet, Verdict,
)

CLAIM = "aag causes garmi in the kitchen"


class _Hard(Source):
    """A stand-in for a proof or a surviving prediction, so the hard requirement can be met."""

    name = "formal"
    hard = True

    def check(self, claim, *, context=None):
        return Evidence(source=self.name, supports=True, weight=1.0, hard=True, detail="proved")


class _Ctx:
    def __init__(self, observations):
        self.observations = observations


# --------------------------------------------------------------------------- #
# The mark nobody set
# --------------------------------------------------------------------------- #

def test_a_simulated_observation_is_marked_rather_than_merely_weak():
    got = ObservationSource([(CLAIM, "dream:turn-9")]).check(CLAIM)
    assert got is not None and got.supports
    assert got.simulated is True
    assert "simulated" in got.detail


def test_a_real_observation_is_not_marked():
    got = ObservationSource([(CLAIM, "master")]).check(CLAIM)
    assert got is not None and got.simulated is False


def test_the_origin_tag_is_read_from_its_leading_segment():
    """`dream:turn-9`, `simulation/rollout-2` and bare `dream` are all the same origin."""
    for tag in ("dream", "dream:turn-9", "simulation/rollout-2", "SELFPLAY:game-3"):
        got = ObservationSource([(CLAIM, tag)]).check(CLAIM)
        assert got is not None and got.simulated is True, tag


def test_a_bare_string_observation_still_works():
    """Backwards compatible: an observation with no stated origin is one anonymous source."""
    got = ObservationSource([CLAIM]).check(CLAIM)
    assert got is not None and got.simulated is False


def test_a_dream_can_never_be_one_of_the_independent_supports():
    """The rule the whole firewall exists for, end to end through the gauntlet."""
    gauntlet = TruthGauntlet(sources=[_Hard(), ObservationSource()],
                             min_sources=2, require_hard=True)
    real = gauntlet.judge(CLAIM, context=_Ctx([(CLAIM, "master")]))
    assert real.verdict == Verdict.ESTABLISHED

    dreamt = gauntlet.judge(CLAIM, context=_Ctx([(CLAIM, "dream")]))
    assert dreamt.verdict != Verdict.ESTABLISHED
    assert dreamt.simulated_supports == 1
    assert dreamt.supports == 1, "the hard source still counts; the dream does not"


def test_a_dream_is_still_recorded_and_reported():
    """Excluded from the count, not from the record — a hypothesis is a real thing to hold."""
    gauntlet = TruthGauntlet(sources=[ObservationSource()], min_sources=2)
    got = gauntlet.judge(CLAIM, context=_Ctx([(CLAIM, "simulation")]))
    assert got.verdict == Verdict.CONJECTURE
    assert got.evidence and any(e.simulated for e in got.evidence)


def test_no_amount_of_dreaming_establishes_anything():
    """The loop this prevents: imagine, store, corroborate the next imagining."""
    gauntlet = TruthGauntlet(sources=[ObservationSource()], min_sources=2, require_hard=False)
    many = [(CLAIM, f"dream:{i}") for i in range(50)]
    assert gauntlet.judge(CLAIM, context=_Ctx(many)).verdict != Verdict.ESTABLISHED


# --------------------------------------------------------------------------- #
# Independence is by origin, not by row
# --------------------------------------------------------------------------- #

def test_one_corpus_read_many_times_is_one_source():
    """A bulk loader makes this failure cheap, so the count has to be over origins."""
    rows = [(CLAIM, "conceptnet")] * 500
    got = ObservationSource(rows).check(CLAIM)
    assert got is not None
    assert "1 independent source" in got.detail


def test_two_different_origins_are_two_sources():
    got = ObservationSource([(CLAIM, "conceptnet"), (CLAIM, "master")]).check(CLAIM)
    assert got is not None
    assert "2 independent source" in got.detail


def test_a_real_origin_beside_a_dream_is_not_simulated():
    """One genuine witness is enough to make the reading a reading of the world."""
    got = ObservationSource([(CLAIM, "master"), (CLAIM, "dream")]).check(CLAIM)
    assert got is not None and got.simulated is False
    assert "simulated: dream" in got.detail


def test_the_brain_carries_provenance_into_its_observations():
    """The wire: what the fact store knows about where a triple came from must survive to here."""
    brain = NJPBrain()
    brain.grounder._assert(GroundedTriple(
        subject="aag", predicate="causes", object="garmi",
        confidence=0.8, source="conceptnet", text="aag causes garmi"))
    brain.think("what does aag cause?")
    assert brain.observations, "the turn recorded no observations at all"
    assert all(isinstance(o, tuple) and len(o) == 2 for o in brain.observations)
    assert any(origin == "conceptnet" for _text, origin in brain.observations)


# --------------------------------------------------------------------------- #
# The belief ledger's half of the same rule
# --------------------------------------------------------------------------- #

def test_a_simulated_reason_earns_nothing():
    ledger = BeliefLedger()
    ledger.hold(CLAIM, confidence=0.9)
    belief = ledger.support(CLAIM, EvidenceKind.SIMULATED, "imagined it")
    assert belief is not None and belief.earned == 0.0


def test_a_simulated_reason_does_not_dilute_a_real_one():
    """Dropped rather than weighted at zero, so it cannot take a slot in the discount ladder."""
    ledger = BeliefLedger()
    ledger.hold(CLAIM, confidence=0.9)
    alone = ledger.support(CLAIM, EvidenceKind.TESTIMONY, "the Master said so").earned
    both = ledger.support(CLAIM, EvidenceKind.SIMULATED, "and she dreamt it").earned
    assert both == pytest.approx(alone)


def test_believing_a_dream_reads_as_unsupported():
    """The 'may lower' half: resting on simulation alone makes the gap visible to `audit`."""
    ledger = BeliefLedger()
    ledger.hold(CLAIM, confidence=0.9)
    belief = ledger.support(CLAIM, EvidenceKind.SIMULATED, "imagined it")
    assert belief is not None and belief.unsupported is True


def test_simulation_is_never_hard():
    assert EvidenceKind.SIMULATED not in EvidenceKind.HARD
    assert not EvidenceKind.supporting(EvidenceKind.SIMULATED)
    for kind in (EvidenceKind.PROOF, EvidenceKind.OBSERVATION, EvidenceKind.PREDICTION,
                 EvidenceKind.TESTIMONY, EvidenceKind.INFERENCE, EvidenceKind.CONSENSUS):
        assert EvidenceKind.supporting(kind), kind


def test_the_simulated_origins_are_the_organs_that_generate_their_own_material():
    """Named rather than inferred, because a missing name is a silent hole in the firewall."""
    assert {"dream", "simulation", "counterfactual", "selfplay"} <= SIMULATED_ORIGINS
