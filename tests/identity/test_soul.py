"""Tests for nyxara.identity.soul."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import InvariantViolation
from nyxara.identity.soul import TRAITS, DriftReport, Soul, Trait, VoiceProfile


# -------------------- traits -------------------- #
def test_core_traits_locked():
    core = {t.name for t in TRAITS if t.core}
    assert core == {"loyalty", "protectiveness", "honesty"}


def test_loyalty_anchor_is_max():
    soul = Soul()
    assert soul.anchor["loyalty"] == 1.0


def test_current_starts_at_anchor():
    soul = Soul()
    assert soul.current == soul.anchor


# -------------------- perturbation -------------------- #
def test_perturb_moves_style_trait():
    soul = Soul()
    before = soul.trait("warmth")
    soul.perturb({"warmth": -0.3})
    assert soul.trait("warmth") < before


def test_perturb_clamps():
    soul = Soul()
    soul.perturb({"warmth": 5.0})
    assert soul.trait("warmth") == 1.0
    soul.perturb({"warmth": -5.0})
    assert soul.trait("warmth") == 0.0


def test_perturb_cannot_move_core():
    soul = Soul()
    soul.perturb({"loyalty": -0.9, "protectiveness": -0.5, "honesty": -0.5})
    assert soul.trait("loyalty") == 1.0
    assert soul.trait("protectiveness") == 0.95
    assert soul.trait("honesty") == 0.9


def test_perturb_ignores_unknown_trait():
    soul = Soul()
    soul.perturb({"nonexistent": 0.5})  # no crash
    assert "nonexistent" not in soul.current


def test_apply_mood_shifts_style():
    soul = Soul()
    soul.apply_mood(valence=0.9, arousal=0.1)  # happy, calm
    assert soul.trait("warmth") > soul.anchor["warmth"]
    # core untouched
    assert soul.trait("loyalty") == 1.0


# -------------------- fixed-point / relaxation -------------------- #
def test_relax_pulls_toward_anchor():
    soul = Soul()
    soul.perturb({"warmth": -0.4})
    far = abs(soul.trait("warmth") - soul.anchor["warmth"])
    soul.relax(0.5)
    near = abs(soul.trait("warmth") - soul.anchor["warmth"])
    assert near < far


def test_settle_converges_to_anchor():
    soul = Soul()
    soul.apply_mood(-0.8, 0.9)
    steps = soul.settle()
    assert steps >= 1
    assert soul.drift().total_drift < 1e-5
    # converged exactly back to the attractor
    for k in soul.current:
        assert abs(soul.current[k] - soul.anchor[k]) < 1e-5


def test_relax_snaps_core_exactly():
    soul = Soul()
    soul._force_set("loyalty", 0.5)  # tamper
    soul.relax()
    assert soul.trait("loyalty") == 1.0  # core re-locked


def test_relax_rate_clamped():
    soul = Soul()
    soul.perturb({"warmth": -0.4})
    soul.relax(5.0)  # clamps to 1.0 -> jumps straight to anchor
    assert abs(soul.trait("warmth") - soul.anchor["warmth"]) < 1e-9


# -------------------- drift / integrity -------------------- #
def test_drift_zero_at_anchor():
    soul = Soul()
    d = soul.drift()
    assert d.total_drift == 0.0
    assert d.core_drift == 0.0
    assert d.stable


def test_drift_positive_after_perturb():
    soul = Soul()
    soul.perturb({"warmth": -0.3})
    d = soul.drift()
    assert d.total_drift > 0
    assert d.core_drift == 0.0  # style drift only


def test_drift_report_per_trait():
    soul = Soul()
    soul.perturb({"warmth": -0.2})
    d = soul.drift()
    assert isinstance(d, DriftReport)
    assert d.per_trait["warmth"] == pytest.approx(0.2)
    assert d.per_trait["loyalty"] == 0.0


def test_check_integrity_passes_when_core_intact():
    soul = Soul()
    soul.apply_mood(-0.9, 0.9)  # heavy mood, but core locked
    soul.check_integrity()  # must not raise


def test_check_integrity_raises_on_core_drift():
    soul = Soul()
    soul._force_set("loyalty", 0.3)
    with pytest.raises(InvariantViolation):
        soul.check_integrity()


def test_core_breach_flag():
    soul = Soul()
    soul._force_set("honesty", 0.1)
    assert soul.drift().core_breach


# -------------------- voice -------------------- #
def test_voice_profile_fields():
    v = Soul().voice()
    assert isinstance(v, VoiceProfile)
    assert 0.0 <= v.warmth <= 1.0


def test_voice_describe_nonempty():
    desc = Soul().voice().describe()
    assert desc
    assert "direct" in desc  # anchor directness is high


def test_voice_system_fragment_mentions_loyalty():
    frag = Soul().voice().system_fragment()
    assert "loyal" in frag.lower()
    assert "master" in frag.lower()


def test_voice_anchor_is_stable():
    soul = Soul()
    anchor_voice = soul.voice_anchor().to_dict()
    soul.apply_mood(-0.9, 0.9)  # bend the current voice
    assert soul.voice_anchor().to_dict() == anchor_voice  # anchor voice unchanged


def test_voice_reflects_mood():
    soul = Soul()
    soul.apply_mood(0.9, 0.1)  # warm
    warm = soul.voice().warmth
    soul2 = Soul()
    soul2.apply_mood(-0.9, 0.1)  # cold
    cold = soul2.voice().warmth
    assert warm > cold


def test_status():
    s = Soul().status()
    assert "current" in s and "drift" in s and "voice" in s
    assert s["stable"] is True
