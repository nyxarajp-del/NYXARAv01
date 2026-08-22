"""Why she was wrong, not just that she was — predict.ErrorKind and the repair it selects.

**The measurement this file exists for.** `ErrorKind` has named eight distinct faults since it was
written, each with its own repair. Run over a 48-turn session and counted::

    by_kind: {'world_model': 43, perception: 0, grounding: 0, memory: 0,
              reasoning: 0, planning: 0, language: 0, unattributed: 0}
    evidence keys ever supplied: ['concepts', 'stimulus', 'triples']

An eight-way discrimination decided before it runs. Most branches test evidence keys no caller
supplied, so they could not fire at all, and the one path that registers predictions every turn —
the manifold's next-state anticipation — is caught by the organ check above them.

Three defects behind that, each reproduced before it was touched:

* **She lost what she said.** `_open_deferred` read `grounding.answer`, but every *derived* answer
  — the interesting kind — arrives on `thought.answer` instead. She answered "water" through a
  correct two-step inheritance and the record stored `<unknown>`, so the grade compared reality
  against nothing and `corrections`, belief settlement and retraction were all gated shut.
* **`in_memory` did not mean what it says.** It was supplied as `bool(pending.answered)` — "did
  she speak" — so every wrong answer became a retrieval fault, including values the record had
  never held.
* **REASONING was a catch-all.** `facts_correct` defaults True and nothing supplies it, so an
  honest abstention was booked as a reasoning failure and the self-model was told her reasoning
  had failed for not knowing something.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.predict import ErrorKind, Outcome, PredictionEngine


def _deferred(brain: NJPBrain):
    return [o for o in brain.predictor.outcomes if o.key.startswith("deferred")]


# --------------------------------------------------------------------------- #
# She keeps hold of what she actually said
# --------------------------------------------------------------------------- #
def test_a_derived_answer_is_recorded_as_what_she_said():
    """It arrives on `thought.answer`, never on `grounding.answer`. Before, it read `<unknown>`."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    assert brain.think("what does a sparrow need?").answer == "water"
    brain.think("sparrow needs food")

    outcomes = _deferred(brain)
    assert outcomes, "the Master's statement must grade the open question"
    assert outcomes[0].expected == "water"
    assert outcomes[0].actual == "food"


def test_being_corrected_on_a_derived_answer_counts_as_a_correction():
    """`rep.corrections` is gated on `pending.answered`, which a lost answer left empty."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    report = brain.think("sparrow needs food").loop
    assert report is not None
    assert report.corrections >= 1


# --------------------------------------------------------------------------- #
# The discrimination itself
# --------------------------------------------------------------------------- #
def test_a_composed_value_the_world_contradicts_is_a_relation_error():
    """Every premise true, the defeasible step too general. Not memory, not perception."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    brain.think("sparrow needs food")
    assert _deferred(brain)[0].diagnosis.kind == ErrorKind.RELATION


def test_an_abstention_blames_no_organ():
    """Not knowing is not a defect in the reasoner, and training it as one teaches her to guess."""
    brain = NJPBrain()
    brain.think("a crow is a bird")
    brain.think("what does a crow need?")
    brain.think("crow needs grain")
    outcomes = _deferred(brain)
    assert outcomes
    assert outcomes[0].expected == "<unknown>"
    assert outcomes[0].diagnosis.kind == ErrorKind.UNATTRIBUTED
    assert not outcomes[0].diagnosis.attributed


def test_memory_is_claimed_only_when_the_record_actually_held_the_answer():
    """The evidence key means "the true value was on record", and must keep meaning that."""
    engine = PredictionEngine()
    outcome = Outcome(key="k", expected="water", actual="food", organ="grounding")

    on_record = engine.diagnose(outcome, {"stimulus": "s", "concepts": ["c"],
                                          "expected_triples": True, "triples": [1],
                                          "in_memory": True, "said": "water"})
    assert on_record.kind == ErrorKind.MEMORY

    never_held = engine.diagnose(outcome, {"stimulus": "s", "concepts": ["c"],
                                           "expected_triples": True, "triples": [1],
                                           "in_memory": False, "derived": True,
                                           "said": "water", "predicate": "requires"})
    assert never_held.kind == ErrorKind.RELATION


def test_the_held_set_is_snapshotted_when_she_is_asked_not_when_she_is_graded():
    """The fact that grades her is written by the same turn that grades her.

    Looked up at resolution time it is always present, so `in_memory` reports she had it all
    along — about a value the store first saw one sentence earlier.
    """
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    brain.think("sparrow needs food")
    # `food` was on record by the time the grade ran, and must not be read as a retrieval miss.
    assert _deferred(brain)[0].diagnosis.kind != ErrorKind.MEMORY


# --------------------------------------------------------------------------- #
# The repair the diagnosis selects
# --------------------------------------------------------------------------- #
def test_a_relation_error_lowers_exactly_one_predicate():
    """The point of diagnosing: one number moves, and it is the one the error named."""
    brain = NJPBrain()
    for predicate in ("requires", "is_a", "causes"):
        brain.learner._transitivity(predicate)
    before = {p: t.value for p, t in brain.learner.transitivity.items()}

    class _Outcome:
        context = {"predicate": "requires"}

    assert brain._repair_relation(_Outcome()) is True
    after = {p: t.value for p, t in brain.learner.transitivity.items()}

    assert after["requires"] < before["requires"]
    assert after["is_a"] == before["is_a"]
    assert after["causes"] == before["causes"]


def test_one_exception_barely_moves_a_well_evidenced_relation():
    """A Beta posterior, so a hundred correct chains are not undone by one sparrow."""
    brain = NJPBrain()
    seasoned = brain.learner._transitivity("causes")
    seasoned.confirm(30.0)
    untested = brain.learner._transitivity("zorbulates")

    before_seasoned, before_untested = seasoned.value, untested.value
    seasoned.refute(1.0)
    untested.refute(1.0)

    assert abs(seasoned.value - before_seasoned) < abs(untested.value - before_untested)


def test_the_repair_declines_when_no_predicate_is_named():
    brain = NJPBrain()

    class _Outcome:
        context = {}

    assert brain._repair_relation(_Outcome()) is False


def test_a_relation_error_is_repaired_end_to_end():
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    brain.think("what does a sparrow need?")
    brain.think("sparrow needs food")
    outcome = _deferred(brain)[0]
    assert outcome.diagnosis.kind == ErrorKind.RELATION
    assert outcome.diagnosis.repaired
    assert brain.learner.transitivity["requires"].value < 0.55, "the prior must have moved down"


def test_the_session_no_longer_reports_one_kind_for_everything():
    """The measured defect: 43 of 43 diagnoses were `world_model` before the evidence existed."""
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "birds need water", "what does a sparrow need?",
                 "sparrow needs food", "a crow is a bird", "what does a crow need?",
                 "crow needs grain"):
        brain.think(turn)
    fired = {k for k, v in brain.predictor.by_kind.items() if v}
    assert ErrorKind.RELATION in fired
    assert len(fired) >= 2
