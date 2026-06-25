"""Tests for nyxara.cognition.language_grounding — learning dynamics from language.

These prove the headline requirement: NYXARA can read a natural-language passage and
*build a world model from it* — forming concepts and predicting dynamics — WITHOUT ever
being hand-fed a numeric (state, action, next_state, reward) tuple.
"""

from __future__ import annotations

from nyxara.cognition.language_grounding import (
    LanguageGrounder,
    VariableRegistry,
    extract_transitions_deterministic,
)
from nyxara.mind.world_model import build_world_model


# -------------------- VariableRegistry: stable, fixed-width -------------------- #
def test_registry_assigns_stable_dimensions():
    reg = VariableRegistry()
    i_temp = reg.index_of("temperature")
    i_speed = reg.index_of("speed")
    # asking again never moves a variable
    assert reg.index_of("temperature") == i_temp
    assert reg.index_of("Temperature") == i_temp  # case-insensitive
    assert i_temp != i_speed


def test_registry_vectors_are_fixed_width_and_float():
    reg = VariableRegistry(capacity=8)
    v = reg.vectorize({"temperature": 20})
    assert len(v) == 8
    assert all(isinstance(x, float) for x in v)
    # the named value lands at its stable dimension; everything else is 0.0
    assert v[reg.index_of("temperature")] == 20.0
    assert sum(1 for x in v if x != 0.0) == 1


# -------------------- deterministic extraction -------------------- #
def test_numeric_range_extraction():
    reg = VariableRegistry()
    trs = extract_transitions_deterministic(
        "Heating the water raises its temperature from 20 to 80 degrees.", reg)
    assert len(trs) == 1
    t = trs[0]
    dim = reg.index_of("temperature")
    assert t.state[dim] == 20.0 and t.next_state[dim] == 80.0
    assert t.action == "heat"


def test_comparative_without_numbers():
    reg = VariableRegistry()
    trs = extract_transitions_deterministic("After a while the room felt warmer.", reg)
    assert trs, "a comparative adjective alone should still yield a transition"
    t = trs[0]
    dim = reg.index_of("temperature")  # 'warmer' maps to the temperature variable
    assert t.next_state[dim] > t.state[dim]  # warmer = positive delta


def test_negation_flips_reward_sign():
    reg = VariableRegistry()
    good = extract_transitions_deterministic(
        "Heating it from 10 to 20 is good.", reg)
    bad = extract_transitions_deterministic(
        "Heating it from 10 to 20 is not good.", reg)
    assert good[0].reward > 0
    assert bad[0].reward < good[0].reward  # negation pushes reward down


def test_direction_verb_on_named_variable():
    reg = VariableRegistry()
    trs = extract_transitions_deterministic(
        "Pressing the pedal increases the speed.", reg)
    assert trs
    dim = reg.index_of("speed")
    assert trs[0].next_state[dim] > trs[0].state[dim]


def test_unknown_verb_still_yields_a_transition():
    reg = VariableRegistry()
    # 'frobnicating' is unknown, but the numeric range carries the dynamics
    trs = extract_transitions_deterministic(
        "Frobnicating the gizmo changes the charge from 5 to 9.", reg)
    assert len(trs) == 1
    dim = reg.index_of("charge")
    assert trs[0].state[dim] == 5.0 and trs[0].next_state[dim] == 9.0


def test_empty_text_is_safe():
    reg = VariableRegistry()
    assert extract_transitions_deterministic("", reg) == []
    assert extract_transitions_deterministic("Hello there, how are you?", reg) == []


# -------------------- the headline: learn dynamics from prose -------------------- #
def test_textbook_trains_world_model_and_predicts():
    """Read a textbook paragraph; the world model then predicts the described dynamics —
    with NO manually-fed numeric tuples."""
    grounder = LanguageGrounder()
    wm = build_world_model("auto")
    textbook = (
        "Heating the water raises its temperature from 20 to 80 degrees, which is good. "
        "Heating the metal raises its temperature from 10 to 60 degrees. "
        "Heating the air raises its temperature from 15 to 45 degrees."
    )
    report = grounder.learn(textbook, wm)
    assert report["transitions"] >= 3
    assert "temperature" in report["variables"]
    assert len(wm) >= 3

    dim = grounder.registry.index_of("temperature")
    state = grounder.registry.vectorize({"temperature": 20.0})
    pred = wm.predict(state, "heat")
    # learned purely from language: heating raises temperature
    assert pred.next_state[dim] > state[dim]


def test_concepts_form_from_language():
    """Two same-effect descriptions in different domains should land in one concept —
    concept formation driven entirely by what the text says actions *do*."""
    grounder = LanguageGrounder()
    wm = build_world_model("auto")
    if not hasattr(wm, "concept_report"):
        import pytest
        pytest.skip("kNN floor (numpy absent) does not form a concept hierarchy")
    # heating and warming both raise temperature; many samples so the cluster is firm
    text = " ".join(
        f"Heating the water raises its temperature from {b} to {b + 40} degrees."
        for b in range(0, 60, 10)
    ) + " " + " ".join(
        f"Warming the room raises its temperature from {b} to {b + 40} degrees."
        for b in range(0, 60, 10)
    )
    grounder.learn(text, wm)
    assert wm.concepts.concept_of("heat") is not None
    assert wm.concepts.concept_of("heat") == wm.concepts.concept_of("warm")


def test_llm_is_gated_off_under_mock():
    """With only the mock provider, the LLM ceiling must NOT engage — the deterministic
    floor is used (the report says so)."""
    from nyxara.mind.llm import LLM
    from nyxara.kernel.config import NyxaraSettings, Profile

    llm = LLM(settings=NyxaraSettings.for_profile(Profile.TEST))
    assert llm.chosen_provider().name == "mock"
    grounder = LanguageGrounder(llm=llm)
    assert grounder._use_llm() is False
    wm = build_world_model("auto")
    report = grounder.learn(
        "Heating the water raises its temperature from 20 to 80 degrees.", wm)
    assert report["via"] == "deterministic"
    assert report["transitions"] == 1


# -------------------- integration through NyxaraCore -------------------- #
def test_core_learn_from_text():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore()
    if core.world_model is None:
        import pytest
        pytest.skip("growth disabled — no world model in this build")
    before = len(core.world_model)
    report = core.learn_from_text(
        "Cooling the water lowers its temperature from 80 to 30 degrees.")
    assert report.get("transitions", 0) >= 1
    assert len(core.world_model) > before
    # empty input is handled gracefully
    assert "error" in core.learn_from_text("   ")
