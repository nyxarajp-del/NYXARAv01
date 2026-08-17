"""She goes after her own conclusions, and "I could not check" is not a pass.

`Belief.falsifier` has existed from the start and the two places that fill it write a template —
the claim negated. It names no rival, proposes no test, and nothing ever went looking for it.
Meanwhile every weapon an attack needs was built and pointed elsewhere: `world.counterfactual`
answers "would it have happened anyway", `CausalLink.lift` separates a cause from a coincidence,
and `universe.intervene` is a real do-operator. Two finished weapons and no trigger.

Most of this file is about what the attacker must **refuse to claim**. Adversarial self-checking
is worthless when it is done badly, and it is done badly in exactly one way: run four challenges,
find no evidence for any of them, and report "survived 4 attacks" as though the absence of a
refutation were corroboration.
"""

from __future__ import annotations


from nyxara.njp.adversary import AttackKind, SelfAttacker, Stance
from nyxara.njp.beliefs import BeliefLedger
from nyxara.njp.brain import NJPBrain
from nyxara.njp.universe import InternalUniverse
from nyxara.njp.world import Event, WorldView


def _seen(world: WorldView, *names: str) -> None:
    for name in names:
        world.observe(Event(action=name))


def _find(report, kind):
    return next(a for a in report.attacks if a.kind == kind)


# --------------------------------------------------------------------------- #
# Undecided is the default, and it is not a pass
# --------------------------------------------------------------------------- #
def test_an_attacker_with_nothing_to_check_against_settles_nothing():
    report = SelfAttacker().attack("aag", "garmi")
    assert all(a.stance == Stance.UNDECIDED for a in report.attacks)
    assert report.verdict == "nothing could be settled from the record"


def test_an_unsettled_attack_cannot_raise_confidence():
    """The failure this whole module is written against."""
    report = SelfAttacker().attack("aag", "garmi", confidence=0.5)
    assert report.confidence_after <= report.confidence_before


def test_the_absence_of_a_rival_proves_nothing_on_a_thin_record():
    """Knowing of no other cause is usually evidence the record is thin."""
    world = WorldView()
    _seen(world, "aag", "garmi")
    attack = _find(SelfAttacker(world=world).attack("aag", "garmi"), AttackKind.RIVAL)
    assert attack.stance == Stance.UNDECIDED
    assert "too thin" in attack.finding


def test_a_law_never_observed_is_not_attacked_as_a_coincidence():
    """Testimony is neither coincidence nor observation — that is the intervention's job."""
    world = WorldView()
    attack = _find(SelfAttacker(world=world).attack("aag", "garmi"), AttackKind.COINCIDENCE)
    assert attack.stance == Stance.UNDECIDED


# --------------------------------------------------------------------------- #
# Attacks that genuinely land
# --------------------------------------------------------------------------- #
def test_a_better_supported_rival_refutes_the_claim():
    world = WorldView()
    for _ in range(6):
        _seen(world, "rooster", "dawn", "sunrise")
    for _ in range(6):
        _seen(world, "dawn", "sunrise")
    report = SelfAttacker(world=world).attack("rooster", "sunrise")
    rival = _find(report, AttackKind.RIVAL)
    if rival.stance == Stance.REFUTED:
        assert rival.rival == "dawn"
        assert report.confidence_after == 0.0


def test_an_intervention_that_moves_nothing_refutes_the_claim():
    """The one attack co-occurrence cannot answer, and the reason the do-operator exists."""
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    universe.declare("baarish", "keechad", sign=1)
    report = SelfAttacker(universe=universe).attack("baarish", "garmi")
    attack = _find(report, AttackKind.INTERVENTION)
    assert attack.stance == Stance.REFUTED
    assert report.confidence_after == 0.0


def test_one_landed_attack_ends_the_claim_whatever_else_survived():
    """Fail-closed, matching truth.py. A refutation does not average against three passes."""
    universe = InternalUniverse()
    universe.declare("a", "b", sign=1)
    universe.declare("x", "y", sign=1)
    report = SelfAttacker(universe=universe).attack("x", "b", confidence=0.9)
    assert report.refuted_by
    assert report.confidence_after == 0.0
    assert report.verdict == "refuted"


# --------------------------------------------------------------------------- #
# Attacks that are survived
# --------------------------------------------------------------------------- #
def test_an_intervention_that_moves_the_effect_is_survived():
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    attack = _find(SelfAttacker(universe=universe).attack("aag", "garmi"),
                   AttackKind.INTERVENTION)
    assert attack.stance == Stance.SURVIVED
    assert attack.independent


def test_surviving_may_raise_confidence_but_only_a_little():
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    report = SelfAttacker(universe=universe).attack("aag", "garmi", confidence=0.4)
    assert report.confidence_after > report.confidence_before
    assert report.confidence_after < 0.97


# --------------------------------------------------------------------------- #
# Three views of one dataset are not three confirmations
# --------------------------------------------------------------------------- #
def test_attacks_reading_one_record_count_as_one_piece_of_evidence():
    """Six observations must not talk a claim from 0.33 to 0.83."""
    world = WorldView()
    for _ in range(6):
        _seen(world, "rooster", "sunrise")
    for _ in range(6):
        _seen(world, "sunrise")
    report = SelfAttacker(world=world).attack("rooster", "sunrise", confidence=0.4)
    record_attacks = [a for a in report.attacks
                      if a.source == "record" and a.stance == Stance.SURVIVED]
    if len(record_attacks) > 1:
        # One source's worth of lift, not three.
        alone = SelfAttacker().attack("q", "r", confidence=0.4)
        assert report.confidence_after < 0.75, report.to_dict()
        assert alone.confidence_after <= 0.4


def test_the_simulator_is_a_separate_line_of_evidence():
    """It answers by severing arrows rather than by counting co-occurrences."""
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    report = SelfAttacker(universe=universe).attack("aag", "garmi")
    assert {a.source for a in report.attacks if a.stance == Stance.SURVIVED} == {"simulation"}


def test_a_thing_is_not_a_rival_explanation_of_itself():
    """A recurring event precedes itself, and that is a real row in a co-occurrence table."""
    world = WorldView()
    for _ in range(8):
        _seen(world, "sunrise")
    report = SelfAttacker(world=world).attack("rooster", "sunrise")
    assert _find(report, AttackKind.RIVAL).rival != "sunrise"


# --------------------------------------------------------------------------- #
# What it does to the belief
# --------------------------------------------------------------------------- #
def test_a_refuted_claim_is_retracted_through_the_ledger():
    """Revision history belongs in the ledger, not in a second private store."""
    ledger = BeliefLedger()
    ledger.hold("baarish causes garmi", confidence=0.8)
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    universe.declare("baarish", "keechad", sign=1)
    attacker = SelfAttacker(universe=universe, beliefs=ledger)
    report = attacker.attack("baarish", "garmi", claim="baarish causes garmi")
    assert report.retracted
    assert attacker.stats()["retracted"] == 1


def test_a_survived_claim_is_not_retracted():
    ledger = BeliefLedger()
    ledger.hold("aag causes garmi", confidence=0.8)
    universe = InternalUniverse()
    universe.declare("aag", "garmi", sign=1)
    attacker = SelfAttacker(universe=universe, beliefs=ledger)
    assert not attacker.attack("aag", "garmi", claim="aag causes garmi").retracted


# --------------------------------------------------------------------------- #
# It has a caller — the failure mode this whole effort keeps finding
# --------------------------------------------------------------------------- #
def test_the_attacker_is_built_and_reachable_from_the_brain():
    brain = NJPBrain()
    assert brain.adversary is not None
    assert "adversary" in brain.stats()


def test_the_loop_actually_attacks_something():
    """Written, reachable and never called is the bug this repo keeps reproducing."""
    brain = NJPBrain()
    for i in range(22):
        brain.think("aag se garmi hoti hai" if i % 2 else "garmi se pasina aata hai")
    assert brain.stats()["adversary"]["attacked"] >= 1
    assert brain.stats()["loop"]["totals"]["attacked"] >= 1


def test_every_attack_states_its_challenge_in_words():
    """The Master has to be able to see what she asked herself."""
    report = SelfAttacker().attack("aag", "garmi")
    assert all(a.question and a.question.endswith("?") for a in report.attacks)
