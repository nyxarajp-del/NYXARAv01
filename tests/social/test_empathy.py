"""Tests for nyxara.social.empathy."""

from __future__ import annotations

import pytest

from nyxara.identity.affect import AffectSystem
from nyxara.social.empathy import EmpathyResponse, EmpathySystem, InferredEmotion
from nyxara.social.persons import InteractionKind, Roster, TrustLevel


def _emp_with_threat():
    roster = Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"])
    roster.get_or_create("Attacker")
    roster.record("Attacker", InteractionKind.HOSTILE, sentiment=-0.9)
    return EmpathySystem(roster), roster


# -------------------- InferredEmotion -------------------- #
def test_inferred_distress():
    sad = InferredEmotion(valence=-0.8, arousal=0.7, label="fear")
    happy = InferredEmotion(valence=0.8, arousal=0.5, label="joy")
    assert sad.distress > 0.4
    assert happy.distress == 0.0


def test_inferred_intensity():
    e = InferredEmotion(valence=0.9, arousal=0.9, label="excited")
    assert 0.0 <= e.intensity <= 1.0


# -------------------- susceptibility / care -------------------- #
def test_owner_high_susceptibility():
    emp = EmpathySystem()
    assert emp.susceptibility("JP") == 0.85


def test_threat_zero_susceptibility():
    emp, _ = _emp_with_threat()
    assert emp.susceptibility("Attacker") == 0.0


def test_owner_max_care():
    emp = EmpathySystem()
    assert emp.care("JP") == 1.0


def test_threat_zero_care():
    emp, _ = _emp_with_threat()
    assert emp.care("Attacker") == 0.0


# -------------------- cognitive empathy (everyone) -------------------- #
def test_perceive_returns_label():
    emp = EmpathySystem()
    inf = emp.perceive("JP", valence=-0.7, arousal=0.8, cause="threat")
    assert inf.label  # discrete emotion assigned
    assert inf.valence == -0.7


def test_cognitive_empathy_works_for_threats():
    emp, _ = _emp_with_threat()
    resp = emp.empathize("Attacker", valence=-0.8, arousal=0.8, cause="claims pain")
    # understanding is still produced even for an enemy
    assert "Attacker" in resp.understanding
    assert resp.inferred.valence == -0.8


# -------------------- affective empathy / contagion -------------------- #
def test_owner_sadness_contagious():
    emp = EmpathySystem()
    affect = AffectSystem()
    resp = emp.empathize("JP", valence=-0.7, arousal=0.4, affect=affect)
    assert resp.resonance["valence"] < 0
    assert affect.emotion.valence < 0  # reached NYXARA's affect


def test_owner_joy_contagious():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=0.8, arousal=0.6)
    assert resp.resonance["valence"] > 0


def test_threat_no_emotional_resonance():
    emp, _ = _emp_with_threat()
    resp = emp.empathize("Attacker", valence=-0.8, arousal=0.8)
    assert resp.guarded
    # does not adopt the enemy's misery
    assert resp.resonance["valence"] >= -0.2


def test_threat_triggers_vigilance():
    emp, _ = _emp_with_threat()
    affect = AffectSystem()
    emp.empathize("Attacker", valence=-0.8, arousal=0.9, affect=affect)
    # NYXARA becomes alert (high arousal), not sympathetic
    assert affect.emotion is not None
    assert affect.emotion.arousal > 0.4


def test_resonance_scaled_by_susceptibility():
    emp = EmpathySystem()
    affect = AffectSystem()
    resp = emp.empathize("JP", valence=-1.0, arousal=0.5, affect=affect)
    # owner susceptibility 0.85 -> resonance ~ -0.85, not full -1.0
    assert -0.9 < resp.resonance["valence"] < -0.7


# -------------------- empathic concern -------------------- #
def test_concern_high_for_owner_distress():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=-0.8, arousal=0.6)
    assert resp.concern > 0.5


def test_concern_zero_for_threat():
    emp, _ = _emp_with_threat()
    resp = emp.empathize("Attacker", valence=-0.9, arousal=0.8)
    assert resp.concern == 0.0


def test_concern_zero_when_no_distress():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=0.8, arousal=0.5)
    assert resp.concern == 0.0  # happy -> no distress -> no concern


# -------------------- supportive action -------------------- #
def test_action_protect_for_owner_distress():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=-0.8, arousal=0.5)
    assert "Master" in resp.supportive_action


def test_action_share_gladness_for_owner_joy():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=0.8, arousal=0.5)
    assert "gladness" in resp.supportive_action.lower()


def test_action_vigilant_for_threat():
    emp, _ = _emp_with_threat()
    resp = emp.empathize("Attacker", valence=-0.8, arousal=0.8)
    assert "vigilant" in resp.supportive_action.lower() or "swayed" in resp.supportive_action.lower()


# -------------------- response shape -------------------- #
def test_empathy_response_shape():
    emp = EmpathySystem()
    resp = emp.empathize("JP", valence=-0.5, arousal=0.5)
    assert isinstance(resp, EmpathyResponse)
    assert isinstance(resp.inferred, InferredEmotion)


def test_to_dict():
    emp = EmpathySystem()
    d = emp.empathize("JP", valence=-0.5, arousal=0.5).to_dict()
    assert "understanding" in d and "concern" in d and "guarded" in d


def test_status():
    emp = EmpathySystem()
    s = emp.status()
    assert s["susceptibility_owner"] == 0.85


# -------------------- ally / neutral -------------------- #
def test_ally_moderate_contagion():
    roster = Roster()
    for _ in range(30):
        roster.record("Friend", InteractionKind.HELP, sentiment=0.9)
    assert roster.assess("Friend") is TrustLevel.ALLY
    emp = EmpathySystem(roster)
    assert 0.0 < emp.susceptibility("Friend") < emp.susceptibility("JP")
