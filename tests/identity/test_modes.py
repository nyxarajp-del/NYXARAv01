"""Tests for nyxara.identity.modes."""

from __future__ import annotations

import pytest

from nyxara.identity.modes import (
    PROFILES,
    Mode,
    ModeBlend,
    ModeContext,
    ModeProfile,
    ModeSelector,
    ModeSystem,
)
from nyxara.identity.soul import Soul, VoiceProfile


# -------------------- profiles -------------------- #
def test_all_modes_have_profiles():
    for m in Mode:
        assert m in PROFILES
        assert isinstance(PROFILES[m], ModeProfile)


def test_companion_is_warm():
    assert PROFILES[Mode.COMPANION].warmth > PROFILES[Mode.ANALYST].warmth


def test_guardian_is_terse_direct():
    g = PROFILES[Mode.GUARDIAN]
    assert g.directness > 0.8 and g.verbosity < 0.5


# -------------------- blend -------------------- #
def test_blend_normalizes():
    b = ModeBlend({Mode.ANALYST: 2.0, Mode.TEACHER: 2.0})
    assert sum(b.weights.values()) == pytest.approx(1.0)
    assert b.weights[Mode.ANALYST] == pytest.approx(0.5)


def test_blend_dominant():
    b = ModeBlend({Mode.ANALYST: 0.7, Mode.TEACHER: 0.3})
    assert b.dominant() is Mode.ANALYST


def test_blend_secondary():
    b = ModeBlend({Mode.ANALYST: 0.6, Mode.TEACHER: 0.4})
    assert b.secondary() is Mode.TEACHER


def test_blend_no_secondary_when_pure():
    b = ModeBlend({Mode.ANALYST: 1.0})
    assert b.secondary() is None


def test_blend_voice_params_weighted():
    b = ModeBlend({Mode.COMPANION: 1.0})
    assert b.voice_params()["warmth"] == pytest.approx(PROFILES[Mode.COMPANION].warmth)
    mix = ModeBlend({Mode.COMPANION: 0.5, Mode.ANALYST: 0.5})
    expected = 0.5 * PROFILES[Mode.COMPANION].warmth + 0.5 * PROFILES[Mode.ANALYST].warmth
    assert mix.voice_params()["warmth"] == pytest.approx(expected)


def test_blend_faculty_bias_merged():
    b = ModeBlend({Mode.ANALYST: 1.0})
    assert "math" in b.faculty_bias()


def test_blend_fragment_includes_touch():
    b = ModeBlend({Mode.ANALYST: 0.7, Mode.TEACHER: 0.3})
    frag = b.fragment()
    assert "touch of the teacher" in frag


def test_blend_to_dict():
    d = ModeBlend({Mode.COMPANION: 1.0}).to_dict()
    assert "weights" in d and "dominant" in d


# -------------------- selector -------------------- #
@pytest.mark.parametrize("text,expected", [
    ("Explain how the kernel works step by step", Mode.TEACHER),
    ("Analyze this data and compute the correlation", Mode.ANALYST),
    ("Write a short story about loyalty", Mode.CREATIVE),
    ("I feel lonely, can we talk?", Mode.COMPANION),
])
def test_selector_infers_dominant(text, expected):
    blend = ModeSelector().infer(ModeContext(text=text))
    assert blend.dominant() is expected


def test_selector_threat_forces_guardian():
    blend = ModeSelector().infer(ModeContext(text="explain things", threat=True))
    assert blend.dominant() is Mode.GUARDIAN


def test_selector_owner_distress_boosts_companion():
    blend = ModeSelector().infer(ModeContext(text="analyze the report", owner_distress=True))
    assert blend.weights[Mode.COMPANION] > 0.2


def test_selector_uses_tags():
    blend = ModeSelector().infer(ModeContext(tags=frozenset({"intrusion", "defend"})))
    assert blend.dominant() is Mode.GUARDIAN


# -------------------- mode system -------------------- #
def test_set_mode():
    ms = ModeSystem()
    ms.set_mode(Mode.GUARDIAN)
    assert ms.blend.dominant() is Mode.GUARDIAN


def test_set_blend():
    ms = ModeSystem()
    ms.set_blend({Mode.ANALYST: 0.7, Mode.TEACHER: 0.3})
    assert ms.blend.dominant() is Mode.ANALYST


def test_infer_and_set():
    ms = ModeSystem()
    blend = ms.infer_and_set(ModeContext(text="teach me how this works"))
    assert blend.dominant() is Mode.TEACHER
    assert ms.blend.dominant() is Mode.TEACHER


def test_voice_returns_profile():
    ms = ModeSystem()
    v = ms.voice()
    assert isinstance(v, VoiceProfile)


def test_companion_voice_warmer_than_guardian():
    ms = ModeSystem()
    ms.set_mode(Mode.COMPANION)
    warm = ms.voice().warmth
    ms.set_mode(Mode.GUARDIAN)
    cold = ms.voice().warmth
    assert warm > cold


def test_guardian_voice_terse():
    ms = ModeSystem()
    ms.set_mode(Mode.GUARDIAN)
    assert ms.voice().verbosity < 0.6


def test_voice_uses_soul_base():
    soul = Soul()
    ms = ModeSystem(soul=soul)
    ms.set_mode(Mode.ANALYST)
    v = ms.voice()
    assert isinstance(v, VoiceProfile)
    # soul's core loyalty is never touched by a mode change
    assert soul.trait("loyalty") == 1.0


def test_faculty_bias():
    ms = ModeSystem()
    ms.set_mode(Mode.ANALYST)
    assert "math" in ms.faculty_bias()


def test_system_prompt_always_loyal():
    ms = ModeSystem()
    for mode in Mode:
        ms.set_mode(mode)
        sp = ms.system_prompt()
        assert "loyal" in sp.lower()
        assert "master" in sp.lower()


def test_mode_strength_affects_overlay():
    soul = Soul()
    weak = ModeSystem(soul=soul, mode_strength=0.1)
    strong = ModeSystem(soul=soul, mode_strength=0.9)
    weak.set_mode(Mode.COMPANION)
    strong.set_mode(Mode.COMPANION)
    # stronger overlay pushes warmth closer to the companion target
    assert strong.voice().warmth >= weak.voice().warmth - 1e-9


def test_status():
    ms = ModeSystem()
    ms.set_mode(Mode.TEACHER)
    s = ms.status()
    assert "blend" in s and "voice" in s and "faculty_bias" in s
