"""Tests for the Autonomous Scientist's self-directed question generation (Gap A).

She must pose her OWN novel questions — driven by her curiosity engine and weighted by
measured learning progress (competence) — instead of only replaying a fixed template
battery. With no faculties wired she still falls back deterministically, so the offline
path never regresses. All hermetic, no LLM, no network.
"""

from __future__ import annotations

from nyxara.growth.autonomous_scientist import AutonomousScientist, QuestionOrigin
from nyxara.memory.competence import CompetenceLedger


class _Q:
    """A stand-in for a curiosity-engine Question (text + value/novelty)."""

    def __init__(self, text: str, value: float = 0.9, novelty: float = 1.0) -> None:
        self.text = text
        self.value = value
        self.novelty = novelty


def test_curiosity_question_is_chosen_over_the_template_battery():
    auto = AutonomousScientist(
        curiosity_source=lambda: _Q("Why does a dropped object accelerate?"),
        competence=CompetenceLedger())
    question, origin, _domain = auto._observe()
    assert origin is QuestionOrigin.CURIOSITY
    assert question == "Why does a dropped object accelerate?"


def test_bare_offline_path_still_falls_back_to_a_seed_deterministically():
    # no curiosity, no gap source, no competence → the deterministic template battery
    a = AutonomousScientist()
    b = AutonomousScientist()
    qa, oa, _da = a._observe()
    qb, ob, _db = b._observe()
    assert oa is QuestionOrigin.SEED and ob is QuestionOrigin.SEED
    assert qa == qb  # deterministic


def test_curiosity_source_accepts_a_plain_string():
    auto = AutonomousScientist(curiosity_source=lambda: "Is entropy always increasing?")
    question, origin, _domain = auto._observe()
    assert origin is QuestionOrigin.CURIOSITY
    assert "entropy" in question


def test_curiosity_pass_like_object_uses_its_chosen_question():
    class _Pass:
        chosen = _Q("What governs orbital period?", value=0.8)

    auto = AutonomousScientist(curiosity_source=lambda: _Pass())
    question, origin, _domain = auto._observe()
    assert origin is QuestionOrigin.CURIOSITY
    assert "orbital" in question


def test_learning_progress_prefers_the_least_mastered_topic():
    led = CompetenceLedger()
    # she is already expert at "gravity", a novice at "plasma"
    for _ in range(20):
        led.record("gravity", 1.0)
    gain_mastered = auto_gain(led, "explain gravity please")
    gain_novel = auto_gain(led, "explain plasma please")
    assert gain_novel > gain_mastered  # pulled toward what she has NOT yet mastered


def auto_gain(led: CompetenceLedger, question: str) -> float:
    return AutonomousScientist(competence=led)._learn_gain(question)


def test_competence_is_updated_after_a_cycle_so_curriculum_advances():
    led = CompetenceLedger()
    auto = AutonomousScientist(
        curiosity_source=lambda: _Q("Is 13 a prime number?"), competence=led)
    before = led.stat("prime")
    auto.step()
    after = led.stat("prime")
    # a topic token from the question is now tracked in the ledger (learning progress recorded)
    assert after is not None
    assert before is None or after.observations >= before.observations
