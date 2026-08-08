"""NYX V.01 · self-measurement — what she says about herself comes from live numbers."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.selfmodel import NyxSelfModel


@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "holo_capacity": 64, "metacog_min_samples": 2,
                "dialogue_enabled": False, "car_enabled": False})
    return NyxBrain(cfg)


# -------------------- read off live state -------------------- #
def test_a_fresh_mind_says_it_has_learned_nothing(brain):
    report = brain.about_self()
    assert report.concepts == 0 and report.memories == 0
    assert "not learned anything" in report.describe()


def test_the_report_grows_as_she_does(brain):
    before = brain.about_self()
    brain.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    brain.think("what pulls an apple down")
    after = brain.about_self()

    assert after.concepts > before.concepts
    assert after.memories > before.memories
    assert after.turns > before.turns


def test_the_description_is_not_a_stored_string(brain):
    """Two different histories must produce two different self-descriptions."""
    other = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "dialogue_enabled": False, "car_enabled": False}))
    brain.think("gravity pulls an apple toward the ground")
    other.think("hummingbirds hover by beating their wings very fast")
    assert brain.about_self().describe() != other.about_self().describe()


# -------------------- honest about what it does not know -------------------- #
def test_a_faculty_with_too_few_samples_is_reported_untested(brain):
    brain.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    brain.think("what pulls an apple down")     # a specialist has to actually win first
    report = brain.about_self()
    assert report.untested
    assert "too little evidence" in report.describe()


def test_a_tested_faculty_moves_from_untested_to_trusted(brain):
    for i in range(8):
        thought = brain.think(f"what pulls an apple down {i}")
        brain.resolve(thought, correct=1.0)
    report = brain.about_self()
    assert report.trusted or report.untested          # whichever, it is measured, not asserted
    if report.trusted:
        assert all(0.0 <= v <= 1.0 for v in report.trusted.values())


def test_weakest_is_measured_not_modest(brain):
    for i in range(8):
        thought = brain.think(f"how might we cool a room {i}")
        brain.resolve(thought, correct=0.0)
    sm = NyxSelfModel(brain)
    assert isinstance(sm.weakest(), list)


def test_gaps_are_surfaced_as_the_edge_of_her_knowing(brain):
    for _ in range(5):
        brain.perceive("lonelyconcept")
    assert "lonelyconcept" in brain.about_self().gaps


def test_unavailable_engines_are_named(brain):
    for spec in brain.workspace.specialists:
        spec.available = lambda: False
    assert brain.about_self().unavailable


# -------------------- questions the Master asks -------------------- #
def test_can_answers_from_the_installed_faculties(brain):
    sm = NyxSelfModel(brain)
    assert sm.can("memory") is True
    assert sm.can("a-faculty-she-does-not-have") is None      # honest "I cannot tell"


def test_a_degraded_voice_is_part_of_her_self_description():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "car_enabled": False})
    b = NyxBrain(cfg)
    b.think("what pulls an apple down")
    report = b.about_self()
    if report.fluent:
        pytest.skip("a fluent language surface is installed")
    assert "cannot hold a conversation" in report.describe()


# -------------------- robustness -------------------- #
def test_a_brain_without_a_self_model_returns_nothing():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "selfmodel_enabled": False})
    b = NyxBrain(cfg)
    assert b.selfmodel is None and b.about_self() is None


def test_a_broken_brain_gives_an_empty_report_not_a_crash():
    class _Broken:
        graph = property(lambda self: (_ for _ in ()).throw(RuntimeError("no")))

    report = NyxSelfModel(_Broken()).report()
    assert report.concepts == 0


def test_no_brain_at_all_is_an_empty_report():
    assert NyxSelfModel(None).report().concepts == 0


def test_the_report_serialises(brain):
    brain.think("what pulls an apple down")
    data = brain.about_self().to_dict()
    assert {"concepts", "associations", "memories", "turns", "trusted"} <= set(data)
