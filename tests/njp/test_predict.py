"""Prediction error is the central learning signal — and it says *where* she was wrong.

"She was wrong" is almost useless as a training signal: it is the same string whether she misheard
the question, retrieved the wrong memory, reasoned badly from correct facts, or knew the answer
and phrased it terribly. Four different repairs, one undifferentiated error.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.predict import ErrorKind, Outcome, PredictionEngine


@pytest.fixture()
def engine() -> PredictionEngine:
    return PredictionEngine()


# ---- the loop --------------------------------------------------------------- #
def test_a_correct_prediction_scores_as_correct(engine: PredictionEngine):
    engine.predict("k", "Mumbai", confidence=0.9)
    got = engine.observe("k", "Mumbai")
    assert got is not None and got.correct and got.error == 0.0


def test_a_wrong_prediction_scores_as_wrong(engine: PredictionEngine):
    engine.predict("k", "Mumbai", confidence=0.9)
    got = engine.observe("k", "Delhi")
    assert got is not None and not got.correct and got.error > 0.0


def test_an_unpredicted_observation_is_not_an_error(engine: PredictionEngine):
    """Having had no expectation is not the same as having had a wrong one."""
    assert engine.observe("never-predicted", "anything") is None
    assert engine.scored == 0


def test_being_confidently_wrong_costs_more_than_being_unsure(engine: PredictionEngine):
    engine.predict("sure", "a", confidence=1.0)
    engine.predict("unsure", "a", confidence=0.0)
    confident = engine.observe("sure", "z")
    hesitant = engine.observe("unsure", "z")
    assert confident.surprise > hesitant.surprise


# ---- the taxonomy ----------------------------------------------------------- #
def test_nothing_perceived_is_a_perception_error(engine: PredictionEngine):
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={"stimulus": "a real turn", "concepts": []})
    assert got.diagnosis.kind == ErrorKind.PERCEPTION


def test_a_fact_that_grounded_to_nothing_is_a_grounding_error(engine: PredictionEngine):
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "my name is Jay", "concepts": ["name", "jay"],
        "expected_triples": True, "triples": []})
    assert got.diagnosis.kind == ErrorKind.GROUNDING


def test_an_answer_on_record_that_was_not_retrieved_is_a_memory_error(engine: PredictionEngine):
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "s", "concepts": ["s"], "in_memory": True, "recalled": False})
    assert got.diagnosis.kind == ErrorKind.MEMORY


def test_a_missed_next_state_belongs_to_the_world_model(engine: PredictionEngine):
    engine.predict("k", "wet", confidence=0.8, organ="world")
    got = engine.observe("k", "dry", evidence={"stimulus": "s", "concepts": ["s"]})
    assert got.diagnosis.kind == ErrorKind.WORLD_MODEL


def test_right_facts_and_a_wrong_conclusion_is_a_reasoning_error(engine: PredictionEngine):
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "s", "concepts": ["s"], "triples": [("a", "b", "c")], "facts_correct": True})
    assert got.diagnosis.kind == ErrorKind.REASONING


def test_right_content_said_wrongly_is_a_language_error(engine: PredictionEngine):
    engine.predict("k", "the answer is Mumbai", confidence=0.5)
    got = engine.observe("k", "erm", evidence={
        "stimulus": "s", "concepts": ["s"], "content_correct": True})
    assert got.diagnosis.kind == ErrorKind.LANGUAGE


def test_an_unclear_error_is_never_guessed_at(engine: PredictionEngine):
    """A wrong attribution actively trains the wrong organ. Declining is strictly better."""
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={"stimulus": "s", "concepts": ["s"]})
    assert got.diagnosis.kind == ErrorKind.UNATTRIBUTED
    assert not got.diagnosis.attributed


def test_perception_is_diagnosed_before_grounding(engine: PredictionEngine):
    """An organ can only be blamed once the ones feeding it are cleared."""
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "s", "concepts": [], "expected_triples": True, "triples": []})
    assert got.diagnosis.kind == ErrorKind.PERCEPTION


# ---- repair routing --------------------------------------------------------- #
def test_the_diagnosis_routes_to_the_owning_organ(engine: PredictionEngine):
    called: list[str] = []
    engine.register_repair(ErrorKind.GROUNDING, lambda o: called.append("grounding") or True)
    engine.register_repair(ErrorKind.MEMORY, lambda o: called.append("memory") or True)

    engine.predict("k", "x", confidence=0.5)
    engine.observe("k", "y", evidence={
        "stimulus": "my name is Jay", "concepts": ["jay"],
        "expected_triples": True, "triples": []})
    assert called == ["grounding"], "the wrong organ was repaired"


def test_an_unattributed_error_repairs_nothing(engine: PredictionEngine):
    called: list[str] = []
    for kind in ErrorKind.ALL:
        engine.register_repair(kind, lambda o, k=kind: called.append(k) or True)
    engine.predict("k", "x", confidence=0.5)
    engine.observe("k", "y", evidence={"stimulus": "s", "concepts": ["s"]})
    assert called == []


def test_a_repair_that_fails_is_recorded_as_unrepaired(engine: PredictionEngine):
    engine.register_repair(ErrorKind.REASONING, lambda o: False)
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "s", "concepts": ["s"], "triples": [1], "facts_correct": True})
    assert got.diagnosis.attributed and not got.diagnosis.repaired


def test_a_repair_that_raises_never_breaks_the_turn(engine: PredictionEngine):
    def _boom(_outcome: Outcome) -> bool:
        raise RuntimeError("repair exploded")

    engine.register_repair(ErrorKind.REASONING, _boom)
    engine.predict("k", "x", confidence=0.5)
    got = engine.observe("k", "y", evidence={
        "stimulus": "s", "concepts": ["s"], "triples": [1], "facts_correct": True})
    assert got is not None and not got.diagnosis.repaired


# ---- reading the trend ------------------------------------------------------ #
def test_worst_kind_names_what_is_costing_her_most(engine: PredictionEngine):
    for i in range(4):
        engine.predict(f"g{i}", "x", confidence=0.5)
        engine.observe(f"g{i}", "y", evidence={
            "stimulus": "s", "concepts": ["s"], "expected_triples": True, "triples": []})
    engine.predict("w", "x", confidence=0.5, organ="world")
    engine.observe("w", "y", evidence={"stimulus": "s", "concepts": ["s"]})
    assert engine.worst_kind() == ErrorKind.GROUNDING


def test_improvement_is_measured_not_asserted(engine: PredictionEngine):
    for i in range(20):
        engine.predict(f"k{i}", "target", confidence=0.5)
        engine.observe(f"k{i}", "wrong" if i < 10 else "target")
    assert engine.improving() is True


def test_improvement_is_unknown_before_there_is_evidence(engine: PredictionEngine):
    engine.predict("k", "a", confidence=0.5)
    engine.observe("k", "a")
    assert engine.improving() is None


def test_calibration_catches_confident_wrongness(engine: PredictionEngine):
    """A mind that is 90% confident should be right nine times in ten."""
    for i in range(30):
        engine.predict(f"k{i}", "target", confidence=0.9)
        engine.observe(f"k{i}", "wrong")
    got = engine.calibration()
    assert got.samples == 30
    assert got.gap > 0.5 and not got.calibrated


def test_calibration_is_not_claimed_on_thin_evidence(engine: PredictionEngine):
    engine.predict("k", "a", confidence=0.5)
    engine.observe("k", "a")
    assert not engine.calibration().calibrated


# ---- wired ------------------------------------------------------------------- #
def test_the_brain_scores_and_diagnoses_a_turn():
    brain = NJPBrain()
    thought = brain.think("my name is Jay")
    brain.resolve(thought, correct=1.0, actual=thought.answer)
    assert brain.stats()["predict"]["scored"] == 1


def test_the_brain_registers_a_prediction_even_when_it_does_not_know():
    """"I don't know" is a real prediction about the world and a diagnosable miss."""
    brain = NJPBrain()
    brain.think("what is my pin code")
    assert brain.predictor.predictions == 1


def test_an_organ_that_is_off_is_absent():
    class _Off:
        predict_enabled = False

    brain = NJPBrain(_Off())
    assert brain.predictor is None
    assert "predict" not in brain.stats()
