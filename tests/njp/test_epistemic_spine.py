"""UNKNOWN → experiment → evidence — epistemic.py as the loop's spine, not a bystander.

**The measurement.** ``epistemic.py`` implements discriminating-experiment selection, rival
synthesis under the do-operator, and EVPI pricing through ``planning/voi``. Counted over a 30-turn
session containing two genuine causal ties::

    epistemic stats: {'compiled': 0, 'experiments': 0, 'refused': 0}

Zero, because its only caller sat inside ``_consult_cortex`` behind a cortex *disagreement*, and a
session with no language model attached never reaches it. Alongside that,
:meth:`~nyxara.njp.epistemic.EpistemicCompiler.from_attack` — documented as *"the join the
adversary never had"* — had no caller anywhere in the package, so the join it names did not exist
either: the adversary produced ``UNDECIDED`` verdicts and nothing consumed them, which is the
accumulating end state the module was written against, reached through its own front door.

Two joins fix it, and neither needs a cortex. A live tie in the record is already a rival
hypothesis set; an undecided attack is already an open question.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.epistemic import EpistemicCompiler, ObservationKind


class _Attack:
    """The shape `from_attack` reads: a cause, an effect, and what refuted it (nothing)."""

    def __init__(self, cause: str, effect: str, refuted_by=None) -> None:
        self.cause = cause
        self.effect = effect
        self.refuted_by = refuted_by


# --------------------------------------------------------------------------- #
# Rivals for a tie between stated causes
# --------------------------------------------------------------------------- #
def test_two_stated_causes_of_one_effect_become_testable_rivals():
    """Two claims that only assert themselves forbid nothing, and discriminate to zero."""
    rivals = EpistemicCompiler.rivals_for_effect("garmi", ["aag", "dhoop"])
    assert len(rivals) == 3, "only-A, only-B, and both-independently"
    claims = [r.claim for r in rivals]
    assert "aag causes garmi" in claims
    assert "dhoop causes garmi" in claims


def test_the_rivals_actually_disagree_about_something():
    """`discriminate` scores zero unless one candidate predicts what another forbids."""
    compiler = EpistemicCompiler()
    experiment = compiler.compile("q", compiler.rivals_for_effect("garmi", ["aag", "dhoop"]))
    assert experiment is not None
    assert experiment.discrimination.discriminating
    assert experiment.discrimination.power > 0.0


def test_the_separating_observation_is_an_intervention():
    """No amount of further observing separates two stated causes. Only removal does."""
    compiler = EpistemicCompiler()
    experiment = compiler.compile("q", compiler.rivals_for_effect("garmi", ["aag", "dhoop"]))
    assert experiment.observation.kind == ObservationKind.INTERVENE
    assert "do(not" in experiment.observation.text


def test_the_question_says_what_each_outcome_rules_out():
    compiler = EpistemicCompiler()
    experiment = compiler.compile("garmi ka karan?",
                                  compiler.rivals_for_effect("garmi", ["aag", "dhoop"]))
    assert "rules out" in experiment.question


def test_one_cause_is_not_a_tie():
    assert EpistemicCompiler.rivals_for_effect("garmi", ["aag"]) == []


def test_the_same_cause_stated_twice_is_one_hypothesis():
    """Two copies of a candidate would score as agreement between rivals."""
    assert EpistemicCompiler.rivals_for_effect("garmi", ["aag", "aag"]) == []


def test_an_effect_nobody_named_yields_no_rivals():
    assert EpistemicCompiler.rivals_for_effect("", ["aag", "dhoop"]) == []


# --------------------------------------------------------------------------- #
# Join 1 — a live tie compiles an experiment, on the ordinary turn path
# --------------------------------------------------------------------------- #
def test_a_conflicting_turn_compiles_an_experiment_without_any_cortex():
    brain = NJPBrain()
    brain.think("Aag se garmi hoti hai")
    brain.think("Dhoop se garmi hoti hai")
    thought = brain.think("garmi ka karan kya hai?")

    assert thought.epistemic == "conflicting"
    assert brain.epistemic.stats()["experiments"] >= 1
    assert thought.experiment is not None
    assert thought.experiment.discrimination.discriminating


def test_the_experiment_is_filed_where_questions_get_retired():
    """Curiosity already resolves questions, so an experiment landing there cannot leak."""
    brain = NJPBrain()
    brain.think("Aag se garmi hoti hai")
    brain.think("Dhoop se garmi hoti hai")
    brain.think("garmi ka karan kya hai?")
    filed = [q for q in brain.curiosity.questions.values() if "do(not" in q.text]
    assert filed


def test_a_tie_between_two_VALUES_is_not_an_intervention():
    """`do(not Jay)` is nonsense wearing the shape of an experiment.

    Two names for one person is a contradiction in the record, and what settles it is the Master
    saying which — not removing a person and seeing what happens.
    """
    brain = NJPBrain()
    brain.think("mera naam Jay hai")
    brain.think("mera naam Rahul hai")
    brain.think("mera naam kya hai?")
    assert brain.epistemic.stats()["compiled"] == 0


def test_an_ordinary_turn_compiles_nothing():
    brain = NJPBrain()
    brain.think("Aag se garmi hoti hai")
    brain.think("aag se kya hota hai?")
    assert brain.epistemic.stats()["compiled"] == 0


# --------------------------------------------------------------------------- #
# Join 2 — an undecided attack becomes the experiment that would settle it
# --------------------------------------------------------------------------- #
def test_an_undecided_attack_compiles_an_experiment():
    brain = NJPBrain()
    experiment = brain.epistemic.from_attack(_Attack("aag", "garmi"))
    assert experiment is not None
    assert experiment.discrimination.discriminating


def test_a_refuted_attack_compiles_nothing():
    """Refuted means the record settled it. Testing a question that has an answer is confirmation."""
    brain = NJPBrain()
    assert brain.epistemic.from_attack(_Attack("aag", "garmi", refuted_by=["observed"])) is None


def test_the_attack_join_is_reachable_from_the_loop():
    """The method existed and had no caller anywhere, which is the same as not existing."""
    from nyxara.njp.integrate import LoopReport

    brain = NJPBrain()
    report = LoopReport()
    before = brain.epistemic.stats()["experiments"]

    brain.loop._experiment_from_attack(_Attack("aag", "garmi"), report)

    assert brain.epistemic.stats()["experiments"] > before
    assert report.experiments_run == 1
    assert report.bits_gained > 0.0


# --------------------------------------------------------------------------- #
# What it must still refuse
# --------------------------------------------------------------------------- #
def test_an_observation_every_candidate_agrees_on_is_refused():
    """Confirmation feels like inquiry and costs the same. Only the split is inquiry."""
    class _Agreeing:
        def __init__(self, claim):
            self.claim = claim
            self.predicts = ["the sun rises"]
            self.forbids = []

    compiler = EpistemicCompiler()
    assert compiler.compile("q", [_Agreeing("a"), _Agreeing("b")]) is None
    assert compiler.stats()["refused"] == 1


def test_a_lone_candidate_is_refused():
    compiler = EpistemicCompiler()
    assert compiler.compile("q", EpistemicCompiler.rivals_for_effect("x", ["a"])) is None
