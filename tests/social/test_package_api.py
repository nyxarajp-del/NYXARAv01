"""The nyxara.social package now exposes a curated public surface.

Previously social/__init__.py was empty, so the (real, tested) social faculties could only be
reached by full module path and were never composed as a unit. These tests pin the package API.
"""

from __future__ import annotations

import nyxara.social as social


def test_public_surface_is_exported():
    expected = {
        "TheoryOfMind", "EmpathySystem", "DialogueManager", "Conversation",
        "ImplicatureEngine", "Roster", "Person", "CommonGround", "CultureSystem",
        "RepairManager", "detect_speech_act",
    }
    assert expected.issubset(set(social.__all__))
    for name in expected:
        assert hasattr(social, name), f"{name} missing from package namespace"


def test_imports_are_the_real_classes():
    from nyxara.social import EmpathySystem, Roster, CultureSystem, RepairManager
    roster = Roster()
    emp = EmpathySystem(roster=roster)
    # a real empathic read, not a stub
    resp = emp.empathize(roster.owner.name, valence=0.6, arousal=0.4, cause="good news")
    assert resp.understanding
    assert CultureSystem(roster=roster).adapt(text="hello bhai, kaise ho").language in {
        "hi", "en", "hinglish"}
    assert RepairManager().handle("Huh? I don't follow.").type.value != "none"
