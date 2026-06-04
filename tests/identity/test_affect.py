"""Tests for nyxara.identity.affect."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import (
    AffectSystem,
    Drive,
    EmotionEvent,
    Mood,
    label_affect,
)


# -------------------- labelling -------------------- #
def test_label_fear():
    assert label_affect(-0.6, 0.85) == "fear"


def test_label_joy():
    assert label_affect(0.7, 0.6) == "joy"


def test_label_calm_or_content():
    assert label_affect(0.4, 0.15) in ("calm", "content")


def test_label_clamps_input():
    assert label_affect(5.0, 5.0) in ("excited", "joy", "alarm")


# -------------------- Drive -------------------- #
def test_drive_pressure_when_unmet():
    d = Drive("x", level=0.3, setpoint=1.0, weight=2.0)
    assert d.pressure() == pytest.approx(2.0 * 0.7)


def test_drive_no_pressure_when_met():
    d = Drive("x", level=1.0, setpoint=1.0)
    assert d.pressure() == 0.0


def test_drive_satisfy_and_deplete():
    d = Drive("x", level=0.5, decay_per_s=0.1)
    d.satisfy(0.3)
    assert d.level == pytest.approx(0.8)
    d.deplete(2.0)
    assert d.level == pytest.approx(0.6)


def test_drive_clamps():
    d = Drive("x", level=0.9)
    d.satisfy(0.5)
    assert d.level == 1.0


# -------------------- feeling -------------------- #
def test_feel_records_emotion():
    a = AffectSystem()
    ev = a.feel(0.7, 0.6, cause="good news")
    assert isinstance(ev, EmotionEvent)
    assert ev.label == "joy"
    assert a.emotion is ev


def test_feel_moves_mood_toward_emotion():
    a = AffectSystem()
    base = a.mood.valence
    a.feel(0.9, 0.5, intensity=1.0)
    assert a.mood.valence > base


def test_feel_applies_to_soul():
    from nyxara.identity.soul import Soul
    soul = Soul()
    a = AffectSystem(soul=soul)
    a.feel(-0.9, 0.9, intensity=1.0)
    # style traits shift but loyalty stays locked
    assert soul.trait("loyalty") == 1.0


def test_feel_intensity_auto():
    a = AffectSystem()
    ev = a.feel(0.8, 0.8)
    assert 0.0 <= ev.intensity <= 1.0


# -------------------- owner / threat hooks -------------------- #
def test_note_threat_spikes_safety_and_fear():
    a = AffectSystem()
    ev = a.note_threat(0.9)
    assert ev.label in ("fear", "alarm", "anger")
    assert a.pressure("safety") > 0.5


def test_note_owner_interaction_restores_connection():
    a = AffectSystem()
    a.drives["owner_connection"].level = 0.2
    a.note_owner_interaction()
    assert a.pressure("owner_connection") < a.drives["owner_connection"].weight * 0.8


def test_note_success_satisfies_competence():
    a = AffectSystem()
    before = a.drives["competence"].level
    a.note_success(magnitude=0.3)
    assert a.drives["competence"].level > before


def test_note_novelty_satisfies_novelty():
    a = AffectSystem()
    before = a.drives["novelty"].level
    a.note_novelty(magnitude=0.3)
    assert a.drives["novelty"].level > before


# -------------------- drives over time -------------------- #
def test_drives_deplete_on_tick():
    a = AffectSystem()
    before = a.drives["energy"].level
    for _ in range(50):
        a.tick(1.0)
    assert a.drives["energy"].level < before


def test_dominant_drive_after_depletion():
    a = AffectSystem()
    for _ in range(200):
        a.tick(1.0)
    dom = a.dominant_drive()
    assert dom is not None
    assert dom.pressure() > 0


def test_dominant_drive_none_when_all_met():
    a = AffectSystem()
    for d in a.drives.values():
        d.level = 1.0
        d.setpoint = 1.0
    assert a.dominant_drive() is None


def test_total_pressure():
    a = AffectSystem()
    a.note_threat(0.9)  # safety pressure up
    assert a.total_pressure() > 0


# -------------------- mood dynamics -------------------- #
def test_mood_decays_toward_baseline():
    a = AffectSystem()
    a.feel(0.9, 0.9, intensity=1.0)
    peak = a.mood.valence
    for _ in range(50):
        a.tick(1.0)
    assert abs(a.mood.valence) < abs(peak)


def test_mood_label():
    a = AffectSystem()
    a.feel(0.8, 0.6, intensity=1.0)
    a.feel(0.8, 0.6, intensity=1.0)
    assert isinstance(a.mood, Mood)
    assert a.mood.label in _POSITIVE_LABELS


_POSITIVE_LABELS = {"joy", "content", "calm", "pride", "excited", "curiosity", "neutral"}


# -------------------- prediction integration -------------------- #
def test_ingest_prediction_surprise_to_arousal():
    class R:
        valence, arousal, surprise = 0.3, 0.2, 0.9

    a = AffectSystem()
    ev = a.ingest_prediction(R())
    assert ev.arousal >= 0.9


# -------------------- inner monologue -------------------- #
def test_monologue_threat():
    a = AffectSystem()
    a.note_threat(0.95)
    line = a.inner_monologue()
    assert "protect" in line.lower()


def test_monologue_owner_connection():
    a = AffectSystem()
    a.drives["owner_connection"].level = 0.1  # high unmet need
    line = a.inner_monologue()
    assert "master" in line.lower()


def test_monologue_joy():
    a = AffectSystem()
    a.feel(0.8, 0.5, intensity=1.0)
    line = a.inner_monologue()
    assert line  # some positive self-talk


def test_monologue_buffer_grows():
    a = AffectSystem()
    a.feel(0.5, 0.5)
    a.inner_monologue()
    a.inner_monologue()
    assert len(a.recent_monologue()) >= 2


# -------------------- status -------------------- #
def test_status_structure():
    a = AffectSystem()
    a.feel(0.5, 0.5)
    s = a.status()
    assert "emotion" in s and "mood" in s and "drives" in s
    assert "total_pressure" in s
