"""Tests for nyxara.social.intent_repair."""

from __future__ import annotations

from nyxara.social.intent_repair import (
    RepairAction,
    RepairDetector,
    RepairManager,
    RepairStrategist,
    RepairType,
    TroubleSignal,
    TroubleType,
)


def _det():
    return RepairDetector()


def _strat():
    return RepairStrategist(addressee="JP")


# -------------------- detection -------------------- #
def test_detect_confusion():
    sig = _det().detect("Huh? I don't follow.")
    assert any(s.type is TroubleType.NON_UNDERSTANDING for s in sig)


def test_detect_correction():
    sig = _det().detect("No, that's not what I meant.")
    assert any(s.type is TroubleType.MISUNDERSTANDING for s in sig)


def test_detect_correction_i_meant():
    sig = _det().detect("I meant the other server.")
    assert any(s.type is TroubleType.MISUNDERSTANDING for s in sig)


def test_detect_ambiguity():
    sig = _det().detect("Run it.", candidate_interpretations=["a", "b"])
    assert any(s.type is TroubleType.AMBIGUITY for s in sig)


def test_no_ambiguity_single_interpretation():
    sig = _det().detect("Run it.", candidate_interpretations=["a"])
    assert not any(s.type is TroubleType.AMBIGUITY for s in sig)


def test_detect_low_confidence():
    sig = _det().detect("Do the thing.", interpretation_confidence=0.3)
    assert any(s.type is TroubleType.LOW_CONFIDENCE for s in sig)


def test_high_confidence_no_signal():
    sig = _det().detect("Do the thing.", interpretation_confidence=0.95)
    assert not any(s.type is TroubleType.LOW_CONFIDENCE for s in sig)


def test_detect_contradiction():
    sig = _det().detect("It's off.", contradicts_ground=True)
    assert any(s.type is TroubleType.CONTRADICTION for s in sig)


def test_no_trouble_clear_input():
    sig = _det().detect("Yes, exactly right.", interpretation_confidence=0.9)
    assert sig == []


def test_from_other_false_ignores_confusion_markers():
    # if NYXARA itself produced the text, confusion markers aren't other-trouble
    sig = _det().detect("huh", from_other=False, interpretation_confidence=0.9)
    assert not any(s.type is TroubleType.NON_UNDERSTANDING for s in sig)


# -------------------- strategy -------------------- #
def test_repair_paraphrase_for_non_understanding():
    sig = _det().detect("I don't understand.")
    rep = _strat().choose(sig, interpretation="harden the firewall")
    assert rep.type is RepairType.PARAPHRASE


def test_repair_apology_for_misunderstanding():
    sig = _det().detect("No, I meant something else.")
    rep = _strat().choose(sig, topic="deployment")
    assert rep.type is RepairType.APOLOGY_RETRY
    assert "deployment" in rep.utterance


def test_repair_specification_for_ambiguity():
    sig = _det().detect("Run it.", candidate_interpretations=["backup", "scan"])
    rep = _strat().choose(sig, candidate_interpretations=["backup", "scan"])
    assert rep.type is RepairType.SPECIFICATION
    assert "backup" in rep.utterance and "scan" in rep.utterance
    assert rep.targets == ["backup", "scan"]


def test_repair_confirmation_for_low_confidence():
    sig = _det().detect("Do it.", interpretation_confidence=0.3)
    rep = _strat().choose(sig, interpretation="restart the service")
    assert rep.type is RepairType.CONFIRMATION
    assert "restart the service" in rep.utterance


def test_repair_none_when_no_signals():
    rep = _strat().choose([])
    assert rep.type is RepairType.NONE
    assert not rep.needed


def test_priority_misunderstanding_over_low_confidence():
    sigs = [
        TroubleSignal(TroubleType.LOW_CONFIDENCE, "x", 0.5),
        TroubleSignal(TroubleType.MISUNDERSTANDING, "y", 0.8),
    ]
    rep = _strat().choose(sigs)
    assert rep.type is RepairType.APOLOGY_RETRY


def test_deferential_apology_addresses_master():
    rep = RepairStrategist(deferential=True).choose(
        [TroubleSignal(TroubleType.MISUNDERSTANDING, "x")])
    assert "forgive me" in rep.utterance.lower()


def test_non_deferential_no_apology_lead():
    rep = RepairStrategist(deferential=False).choose(
        [TroubleSignal(TroubleType.MISUNDERSTANDING, "x")])
    assert not rep.utterance.lower().startswith("forgive")


def test_contradiction_repair():
    rep = _strat().choose([TroubleSignal(TroubleType.CONTRADICTION, "x")],
                          ground_conflict="firewall was active")
    assert rep.type is RepairType.CONFIRMATION
    assert "firewall was active" in rep.utterance


# -------------------- manager / loop guard -------------------- #
def test_manager_handles_trouble():
    mgr = RepairManager(addressee="JP")
    action = mgr.handle("Huh?")
    assert action.needed
    assert mgr.attempts == 1


def test_manager_clear_input_no_repair():
    mgr = RepairManager()
    action = mgr.handle("Perfect, thank you.")
    assert action.type is RepairType.NONE


def test_manager_escalates_after_max_attempts():
    mgr = RepairManager(max_attempts=2)
    mgr.handle("Huh?")
    mgr.handle("What?")
    action = mgr.handle("Still confused.")
    assert action.type is RepairType.ESCALATE


def test_manager_resets_on_clear_turn():
    mgr = RepairManager(max_attempts=3)
    mgr.handle("Huh?")
    assert mgr.attempts == 1
    mgr.handle("Got it now.")  # clear -> reset
    assert mgr.attempts == 0


def test_manager_resolved_resets():
    mgr = RepairManager()
    mgr.handle("Huh?")
    mgr.resolved()
    assert mgr.attempts == 0


def test_manager_history():
    mgr = RepairManager()
    mgr.handle("Huh?")
    mgr.handle("What?")
    assert len(mgr.history) == 2


def test_manager_status():
    mgr = RepairManager()
    mgr.handle("Huh?")
    s = mgr.status()
    assert "attempts" in s and "repairs" in s and s["last"] is not None


# -------------------- shapes -------------------- #
def test_repair_action_to_dict():
    a = RepairAction(RepairType.CONFIRMATION, "confirm?", addresses=TroubleType.LOW_CONFIDENCE)
    d = a.to_dict()
    assert d["type"] == "confirmation" and d["addresses"] == "low_confidence"


def test_trouble_signal_to_dict():
    s = TroubleSignal(TroubleType.AMBIGUITY, "two readings", 0.6)
    d = s.to_dict()
    assert d["type"] == "ambiguity"
