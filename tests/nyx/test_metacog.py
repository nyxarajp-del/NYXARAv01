"""NYX V.01 · recursive meta-cognition — learning from her own mistakes, no human in the loop."""

from __future__ import annotations

import pytest

from nyxara.nyx.metacog import RecursiveMetaCognition
from nyxara.nyx.modules import Proposal


def _prop(source: str, confidence: float = 0.8, **kw) -> Proposal:
    return Proposal(source=source, content=f"{source} says something",
                    confidence=confidence, **kw)


def _mc(**kw) -> RecursiveMetaCognition:
    kw.setdefault("min_samples", 3)
    return RecursiveMetaCognition(**kw)


def _resolve(mc, source: str, *, claimed: float, correct: float, n: int = 1) -> None:
    for i in range(n):
        cid = f"{source}-{i}-{correct}"
        mc.observe_cycle(cycle_id=cid, winner=_prop(source, claimed))
        mc.observe_outcome(cycle_id=cid, correct=correct)


# -------------------- "why did I think that?" -------------------- #
def test_observe_cycle_records_the_winner_and_what_it_beat():
    mc = _mc()
    winner = Proposal(source="derivation", content="proven", confidence=1.0,
                      rationale="maths derivation, verified", evidence=["maths"])
    losers = [_prop("creative"), _prop("graph")]
    got = mc.observe_cycle(cycle_id="c1", winner=winner, considered=[winner] + losers)

    assert got.winner == "derivation"
    assert "maths derivation, verified" in got.why
    assert set(got.beaten) == {"creative", "graph"}


def test_a_cycle_with_no_winner_is_recorded_without_crashing():
    assert _mc().observe_cycle(cycle_id="c1", winner=None).winner == ""


# -------------------- "how right was I?" -------------------- #
def test_reliability_starts_agnostic():
    rec = _mc().record_for("creative")
    assert rec.score == 0.5 and rec.samples == 0 and rec.trusted is False


def test_being_repeatedly_wrong_lowers_reliability():
    """The whole point: she demotes a faculty on measured outcomes, with no human feedback."""
    mc = _mc()
    _resolve(mc, "creative", claimed=0.9, correct=0.0, n=6)
    rec = mc.record_for("creative")
    assert rec.score < 0.3
    assert rec.trusted is True
    assert rec.weight < 1.0                       # actually demoted in the competition


def test_being_repeatedly_right_raises_reliability():
    mc = _mc()
    _resolve(mc, "derivation", claimed=0.9, correct=1.0, n=6)
    rec = mc.record_for("derivation")
    assert rec.score > 0.7 and rec.weight > 1.0


def test_untrusted_records_stay_near_neutral():
    """One bad turn must not condemn a faculty — thin evidence is treated as thin."""
    mc = _mc(min_samples=10)
    _resolve(mc, "creative", claimed=0.9, correct=0.0, n=1)
    assert abs(mc.record_for("creative").weight - 1.0) < 0.15


def test_overconfidence_is_detected():
    mc = _mc()
    mc.observe_cycle(cycle_id="c1", winner=_prop("creative", 0.95))
    got = mc.observe_outcome(cycle_id="c1", correct=0.0)
    assert got is not None and got.overconfident is True
    assert got.calibration_gap > 0.9


def test_well_calibrated_confidence_is_not_flagged():
    mc = _mc()
    mc.observe_cycle(cycle_id="c1", winner=_prop("memory", 0.5))
    got = mc.observe_outcome(cycle_id="c1", correct=0.6)
    assert got is not None and got.overconfident is False


def test_calibration_error_accumulates():
    mc = _mc()
    _resolve(mc, "creative", claimed=1.0, correct=0.0, n=5)
    assert mc.record_for("creative").calibration_error > 0.5


def test_outcome_for_an_unknown_cycle_is_ignored():
    assert _mc().observe_outcome(cycle_id="never-seen", correct=1.0) is None


# -------------------- feeding attention -------------------- #
def test_weight_for_an_unseen_specialist_is_neutral():
    assert _mc().weight_for("nobody") == 1.0


def test_weakest_reports_only_trusted_records():
    mc = _mc(min_samples=4)
    _resolve(mc, "creative", claimed=0.9, correct=0.0, n=6)   # trusted, bad
    _resolve(mc, "graph", claimed=0.5, correct=0.0, n=1)      # untrusted, bad
    assert mc.weakest() == ["creative"]


# -------------------- bounds and persistence -------------------- #
def test_history_is_bounded():
    mc = _mc(calibration_window=8)
    for i in range(40):
        mc.observe_cycle(cycle_id=f"c{i}", winner=_prop("graph"))
    assert len(mc.history) <= 8


def test_round_trip_preserves_learned_reliability():
    mc = _mc()
    _resolve(mc, "creative", claimed=0.9, correct=0.0, n=6)
    _resolve(mc, "derivation", claimed=0.9, correct=1.0, n=6)
    data = mc.to_dict()

    restored = _mc()
    assert restored.load_dict(data) is True
    # scores are stored rounded to 4dp to keep the sidecar readable — the only loss
    assert restored.record_for("creative").score == pytest.approx(
        mc.record_for("creative").score, abs=1e-4)
    assert restored.weight_for("derivation") == pytest.approx(mc.weight_for("derivation"),
                                                              abs=1e-4)
    assert restored.record_for("creative").samples == mc.record_for("creative").samples


def test_corrupt_sidecar_leaves_an_empty_record_not_a_crash():
    mc = _mc()
    assert mc.load_dict(None) is False
    assert mc.load_dict({"reliability": "nonsense"}) is True
    assert mc.reliability == {}


def test_stats_reports_pending_and_resolved_honestly():
    mc = _mc()
    mc.observe_cycle(cycle_id="open", winner=_prop("graph"))
    _resolve(mc, "memory", claimed=0.5, correct=1.0, n=2)
    stats = mc.stats()
    assert stats["pending"] == 1 and stats["resolved"] == 2
