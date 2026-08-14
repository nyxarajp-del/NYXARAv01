"""The belief ledger: every claim carries its own case, and her confidence is auditable.

The tests below are mostly about the ways a belief system flatters itself — confidence that grows
by repetition, contradictions resolved by deleting the inconvenient side, and calibration
measured against the confidence held *after* the answer was known.
"""

from __future__ import annotations

from nyxara.njp.beliefs import BeliefLedger, EvidenceKind


def test_a_belief_carries_its_whole_case():
    ledger = BeliefLedger()
    ledger.hold("water boils at 100", confidence=0.6, domain="physics",
                produced_by="grounding", falsifier="water that does not boil at 100 at 1 atm",
                explanation="vapour pressure equals atmospheric pressure")
    ledger.support("water boils at 100", EvidenceKind.OBSERVATION, "kettle", "senses")

    case = ledger.why("water boils at 100")
    assert case["known"] and case["hard_support"] == 1
    assert case["falsifiable"] and case["produced_by"] == "grounding"
    assert case["explanation"]
    assert case["history"], "no record of how the belief came to be held"


def test_repeating_a_claim_does_not_make_it_truer():
    """Soft evidence saturates: being told a thing ten times is not being shown it once."""
    ledger = BeliefLedger()
    ledger.hold("the sky is green", confidence=0.3, domain="general")
    for i in range(10):
        ledger.support("the sky is green", EvidenceKind.TESTIMONY, f"telling {i}")
    belief = ledger.beliefs["the sky is green"]
    assert belief.confidence < 0.8, belief.confidence
    assert belief.hard_support == 0


def test_hard_evidence_outranks_any_amount_of_soft():
    ledger = BeliefLedger()
    ledger.hold("a", confidence=0.1)
    ledger.hold("b", confidence=0.1)
    for i in range(6):
        ledger.support("a", EvidenceKind.CONSENSUS, f"source {i}")
    ledger.support("b", EvidenceKind.PROOF, "derived")
    assert ledger.beliefs["b"].earned > ledger.beliefs["a"].earned


def test_audit_finds_confidence_that_was_not_earned():
    ledger = BeliefLedger()
    ledger.hold("the moon is cheese", confidence=0.85, domain="astronomy")
    flags = ledger.audit()
    assert flags and flags[0]["claim"].startswith("the moon")
    problems = flags[0]["problems"]
    assert "no hard evidence" in problems and "no falsifier" in problems


def test_a_contradiction_is_won_by_evidence_not_by_recency():
    ledger = BeliefLedger()
    ledger.hold("water boils at 100", confidence=0.6, domain="physics")
    ledger.support("water boils at 100", EvidenceKind.OBSERVATION, "kettle")
    ledger.hold("water boils at 90", confidence=0.7, domain="physics")   # asserted harder

    strong, weak = ledger.contradict("water boils at 100", "water boils at 90")
    assert strong.confidence > weak.confidence
    # Neither is deleted: what she used to think is part of what she knows about herself.
    assert "water boils at 90" in ledger.beliefs
    assert strong.contested and weak.contested


def test_two_bare_assertions_both_collapse():
    ledger = BeliefLedger()
    ledger.hold("x", confidence=0.8)
    ledger.hold("not x", confidence=0.8)
    a, b = ledger.contradict("x", "not x")
    # With no reason to prefer either, being confident about one of them is the error.
    assert a.confidence == 0.8 and b.confidence == 0.8 or (a.confidence < 0.9 and b.confidence < 0.9)
    assert a.contested and b.contested


def test_calibration_is_measured_against_the_confidence_held_beforehand():
    ledger = BeliefLedger()
    for i in range(6):
        claim = f"guess {i}"
        ledger.hold(claim, confidence=0.9, domain="guessing")
        ledger.settle(claim, true=(i == 0))          # confident, and wrong five times out of six

    reliability = ledger.reliability("guessing")
    assert reliability.samples == 6
    # Settling drives confidence to 1.0 or 0.0. If the score were taken afterwards it would read
    # as perfect calibration forever — the classic way this measurement gets quietly broken.
    assert reliability.bias is not None and reliability.bias > 0.5, reliability.to_dict()
    assert reliability.brier > 0.5
    assert not reliability.calibrated


def test_temper_corrects_a_stated_confidence_by_measured_bias():
    ledger = BeliefLedger()
    for i in range(6):
        claim = f"guess {i}"
        ledger.hold(claim, confidence=0.9, domain="guessing")
        ledger.settle(claim, true=(i == 0))
    assert ledger.temper(0.9, "guessing") < 0.5


def test_temper_ignores_a_history_too_short_to_mean_anything():
    """Correcting on three samples is correcting on noise, which makes her worse-calibrated."""
    ledger = BeliefLedger()
    for i in range(3):
        claim = f"guess {i}"
        ledger.hold(claim, confidence=0.9, domain="thin")
        ledger.settle(claim, true=False)
    assert ledger.temper(0.9, "thin") == 0.9


def test_unknown_separates_the_three_kinds_of_not_knowing():
    ledger = BeliefLedger()
    ledger.ask("what is the boiling point of mercury?")
    ledger.hold("p", confidence=0.9)                    # asserted, unsupported
    ledger.hold("q", confidence=0.6)
    ledger.hold("not q", confidence=0.6)
    ledger.contradict("q", "not q")

    kinds = {row["kind"] for row in ledger.unknown()}
    assert {"open_question", "unsupported", "contested"} <= kinds


def test_known_requires_support_not_just_confidence():
    ledger = BeliefLedger()
    ledger.hold("bare assertion", confidence=0.95)
    ledger.hold("evidenced", confidence=0.5)
    ledger.support("evidenced", EvidenceKind.OBSERVATION, "seen it")
    names = {b.claim for b in ledger.known()}
    assert "evidenced" in names
    assert "bare assertion" not in names


def test_retraction_keeps_the_belief_and_records_the_outcome():
    ledger = BeliefLedger()
    ledger.hold("it will rain", confidence=0.8, domain="weather")
    ledger.retract("it will rain", why="it did not rain")
    belief = ledger.beliefs["it will rain"]
    assert belief.confidence == 0.0 and belief.outcome is False
    assert any("did not rain" in r.why for r in belief.history)
    assert ledger.reliability("weather").samples == 1


def test_state_survives_a_round_trip():
    ledger = BeliefLedger()
    ledger.hold("water boils at 100", confidence=0.6, domain="physics",
                falsifier="water that does not boil")
    ledger.support("water boils at 100", EvidenceKind.OBSERVATION, "kettle")
    ledger.ask("what about mercury?")

    revived = BeliefLedger()
    revived.load_dict(ledger.to_dict())
    case = revived.why("water boils at 100")
    assert case["known"] and case["hard_support"] == 1
    assert "what about mercury?" in revived.open_questions
