"""Tests for nyxara.memory.competence — competence that is measured, not declared (Rule 4).

The CompetenceLedger keeps a Beta posterior per capability, seeded from the boot prior and moved
by real outcomes, and writes the posterior back to the SelfModel so routing tracks measured
performance. These tests assert priors are anchored yet yield to evidence, continuous scores count
as fractional success, and the write-back is fail-open.
"""

from __future__ import annotations

from nyxara.memory.competence import CapabilityStat, CompetenceLedger
from nyxara.memory.self_model import SelfModel


def _sm():
    sm = SelfModel()
    sm.set_capability("reasoning", level=0.82, confidence=0.7)
    sm.set_capability("transfer", level=0.4, confidence=0.2)
    return sm


def test_strong_prior_resists_single_failure_but_moves():
    sm = _sm()
    led = CompetenceLedger(self_model=sm)
    before = sm.capability("reasoning").level
    led.record("reasoning", False)
    after = sm.capability("reasoning").level
    assert after < before          # a miss nudges it down
    assert after > 0.6             # but does not collapse a well-established prior


def test_sustained_success_lifts_weak_prior():
    sm = _sm()
    led = CompetenceLedger(self_model=sm)
    for _ in range(30):
        led.record("transfer", True)
    cap = sm.capability("transfer")
    assert cap.level > 0.8
    assert cap.confidence > 0.7    # evidence raises confidence


def test_continuous_score_tracks_measured_accuracy():
    led = CompetenceLedger()
    led.seed("open_world", level=0.5, confidence=0.2)
    for _ in range(25):
        led.record("open_world", 0.75)     # a held-out accuracy, not a win/lose
    stat = led.stat("open_world")
    assert 0.68 <= stat.level <= 0.82


def test_unknown_capability_created_on_first_record():
    led = CompetenceLedger()
    stat = led.record("brand_new", True)
    assert stat is not None
    assert led.stat("brand_new") is not None


def test_confidence_grows_with_evidence_and_saturates_below_one():
    stat = CapabilityStat("x", alpha=1.0, beta=1.0)
    c0 = stat.confidence
    for _ in range(200):
        stat.observe(1.0)
    assert stat.confidence > c0
    assert stat.confidence < 1.0


def test_record_is_fail_open_on_bad_input():
    led = CompetenceLedger(self_model=_sm())
    # a non-numeric, non-bool score must not raise
    assert led.record("reasoning", object()) is None


def test_seed_from_self_model_is_idempotent():
    sm = _sm()
    led = CompetenceLedger(self_model=sm)
    names_before = set(led.names())
    led.seed_from_self_model(sm)           # calling again must not duplicate/overwrite
    assert set(led.names()) == names_before


def test_writeback_updates_self_model_level():
    sm = _sm()
    led = CompetenceLedger(self_model=sm)
    for _ in range(10):
        led.record("reasoning", True)
    # the self-model now reflects the ledger's posterior, not the frozen boot prior
    assert abs(sm.capability("reasoning").level - led.stat("reasoning").level) < 1e-6
