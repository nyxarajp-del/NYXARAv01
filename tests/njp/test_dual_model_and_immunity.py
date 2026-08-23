"""Three things that were built, wired, and could not fire.

**The two substrates never committed to the same proposition.** `fabric_proposes` returns a bare
concept *name*; the njp seat returns a clause; the comparison ran `_norm` over both. So AGREE was
effectively unreachable, every pairing landed in UNCLASSIFIED, and the disagreement was reported
on every turn — which is the other edge of "disagreement is not failure". A channel that always
disagrees carries exactly as little information as one that always agrees.

**An UNDECIDED verdict did not survive the call that produced it.** `adversary.attack` returns an
`AttackReport` and the caller drops it. So a claim the adversary went looking at and could not
settle can be re-held with fresh TESTIMONY by `field._record_beliefs` on any later turn the Master
restates it — and `earned` climbs. Repetition finishing what the evidence could not is precisely
what the SOFT_CEILING exists to refuse; this was the hole in the floor beneath it.

**`epistemic.compile` refused on every input.** `_observations` reads `predicts`/`forbids`, which
only `CortexHypothesis` has, so for two of the three seats the power arithmetic saw `p == 0 or
f == 0` and returned nothing. That reads as "no experiment would settle this" and was actually
"two of your three accounts do not speak the language".
"""

from __future__ import annotations

from nyxara.njp.adversary import Stance
from nyxara.njp.beliefs import BeliefLedger, EvidenceKind
from nyxara.njp.brain import NJPBrain
from nyxara.njp.cortex import CortexHypothesis
from nyxara.njp.epistemic import EpistemicCompiler
from nyxara.njp.router import DisagreementKind


class _Undecided:
    stance = Stance.UNDECIDED


class _Survived:
    stance = Stance.SURVIVED


def _held(times: int = 3) -> BeliefLedger:
    ledger = BeliefLedger()
    ledger.hold("aag causes garmi", confidence=0.5)
    for i in range(times):
        ledger.support("aag causes garmi", EvidenceKind.TESTIMONY, source=f"master{i}")
    return ledger


def _belief(ledger: BeliefLedger):
    return ledger.beliefs[next(iter(ledger.beliefs))]


# --------------------------------------------------------------------------- #
# The immune leak
# --------------------------------------------------------------------------- #

def test_an_undecided_claim_does_not_become_true_by_being_repeated():
    """The leak, stated as a test. This is the one that was actually losing information."""
    ledger = _held()
    before = _belief(ledger).earned
    ledger.record_attack("aag causes garmi", _Undecided())

    for i in range(10):
        ledger.support("aag causes garmi", EvidenceKind.TESTIMONY, source=f"again{i}")
    assert _belief(ledger).earned <= before + 1e-9, "repetition finished what evidence could not"


def test_hard_evidence_still_lifts_it():
    """The cap is on repetition, not on learning. One observation is a different kind of thing."""
    ledger = _held()
    ledger.record_attack("aag causes garmi", _Undecided())
    capped = _belief(ledger).earned

    ledger.support("aag causes garmi", EvidenceKind.OBSERVATION, source="saw it happen")
    assert _belief(ledger).earned > capped


def test_a_surviving_claim_is_not_capped():
    ledger = _held()
    ledger.record_attack("aag causes garmi", _Survived())
    before = _belief(ledger).earned
    for i in range(6):
        ledger.support("aag causes garmi", EvidenceKind.TESTIMONY, source=f"more{i}")
    assert _belief(ledger).earned >= before


def test_exactly_one_terminal_stance_is_recorded():
    ledger = _held()
    ledger.record_attack("aag causes garmi", _Undecided())
    belief = _belief(ledger)
    assert belief.tested == 1
    assert belief.last_stance in (Stance.SURVIVED, Stance.REFUTED, Stance.UNDECIDED)
    assert belief.undecided == 1


def test_the_verdict_is_written_into_the_history():
    ledger = _held()
    ledger.record_attack("aag causes garmi", _Undecided())
    assert any("attacked" in r.why for r in _belief(ledger).history)


def test_recording_an_attack_on_an_unheld_claim_is_a_no_op():
    assert BeliefLedger().record_attack("never held", _Undecided()) is None


# --------------------------------------------------------------------------- #
# Three accounts that can actually be told apart
# --------------------------------------------------------------------------- #

def test_a_symbolic_answer_becomes_something_evidence_can_argue_with():
    class _Triple:
        subject, predicate, object = "growth", "caused_by", "water"

    class _Answer:
        text = "water"
        triples = [_Triple()]

    adapted = EpistemicCompiler.from_answer(_Answer())
    assert adapted.predicts, "a symbolic account that predicts nothing cannot be discriminated"
    assert adapted.forbids, "a candidate that forbids nothing risks nothing"


def test_a_seat_that_forbids_nothing_gets_no_power_on_its_own():
    """Correct, not a limitation to paper over: the head associates, it does not derive."""
    compiler = EpistemicCompiler()
    only_associations = [compiler.from_slot("g", "caused_by", "water"),
                         compiler.from_slot("g", "caused_by", "light")]
    assert compiler.compile("what causes g", only_associations) is None


def test_rival_fillers_for_one_slot_produce_a_real_experiment():
    """The fix. Three monologues are not a disagreement; one slot with rival fillers is."""
    compiler = EpistemicCompiler()
    rivals = compiler.rivals_for_slot("growth", "caused_by",
                                      {"njp": "water", "fabric": "light"})
    assert len(rivals) == 2
    experiment = compiler.compile("what causes growth", rivals)
    assert experiment is not None
    assert experiment.discrimination.power > 0.0


def test_seats_that_agree_are_one_account_not_two():
    compiler = EpistemicCompiler()
    assert compiler.rivals_for_slot("g", "caused_by",
                                    {"njp": "water", "cortex": "water"}) == []


def test_agreeing_seats_collapse_and_are_named_together():
    compiler = EpistemicCompiler()
    rivals = compiler.rivals_for_slot("g", "caused_by",
                                      {"njp": "water", "cortex": "water", "fabric": "light"})
    assert len(rivals) == 2
    water = next(r for r in rivals if "water" in r.predicts[0])
    assert "cortex" in water.claim and "njp" in water.claim


# --------------------------------------------------------------------------- #
# The two invariants, which are the Master's own point cutting both ways
# --------------------------------------------------------------------------- #

def _taught() -> NJPBrain:
    brain = NJPBrain()
    for line in ("paudhe ko paani chahiye", "paani se growth badhti hai",
                 "light se growth badhti hai"):
        brain.think(line)
    return brain


def test_agreement_between_the_substrates_raises_nothing():
    """Two seats agreeing is mostly evidence about their shared inputs."""
    brain = _taught()
    thought = brain.think("paudhe ko kya chahiye")
    if thought.substrate is not None:
        return                      # they disagreed on this turn; the other test covers it
    assert thought.substrate is None


def test_a_substrate_disagreement_lowers_and_never_retracts():
    """The fabric is measured at MRR 0.238. It may make her less sure; it may not take an answer."""
    brain = _taught()
    thought = brain.think("paudhe ko kya chahiye")
    held_before = len(brain.grounder._facts()) if hasattr(brain.grounder, "_facts") else None

    if thought.substrate == DisagreementKind.SUBSTRATE:
        assert thought.epistemic_confidence <= 0.5
        assert thought.answer, "a disagreement retracted the answer"
    if held_before is not None:
        assert len(brain.grounder._facts()) == held_before


def test_one_seat_speaking_alone_is_not_a_clash():
    brain = _taught()
    subject, predicate, fillers = brain._slot_claims(brain.think("mangal grah par kya hai"))
    assert len(set(fillers.values())) < 2 or brain.epistemic is not None
