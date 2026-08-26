"""Phase 4: what a teacher leaves behind, and how little of it is allowed to be its answers.

The phase ends with one test and the plan calls it the point of the whole thing —
*"NJP + teacher vs NJP after distillation vs NJP alone. Teacher OFF ke baad performance kitni
retain hui? Yahi real acquisition evidence hai."* Everything here exists to make that number mean
something, which mostly means making it hard to get.

The cortex already put the strong model through NJP's gates, and what survived became a **fact**.
Switch the teacher off and she keeps what it told her and none of what it knew how to do. §17 is
explicit about the difference — *"Weights ko symbolic brain mein copy nahi karna. Behavior ko
structured knowledge/programs mein convert karna."* — so the only durable effect of a lesson here
is a **property of the relation**: the demonstrated conclusion is thrown away and *"this relation
chains"* is kept, which applies to entities the teacher never mentioned.

The enabling bug is worth its own note. `core.reach` and `core.connects` compose any predicate,
priced by that predicate's transitivity — and the only two calls to either, anywhere in the
package, passed the literal string ``"causes"`` from inside ``core.py`` itself. So a causal chain
composed through `think` and no other relation did, while `compile_meaning` had been returning
``subject``/``relation``/``object`` for those very questions the whole time. Three finished parts,
no wire between them.
"""

from __future__ import annotations

from nyxara.njp import NJPBrain
from nyxara.njp.teacher import (
    Distiller,
    Lesson,
    RecordedTeacher,
    Step,
    TeacherCouncil,
    Verdict,
)

RELATION = "kizzle"


def _chain(brain: NJPBrain, *names: str, relation: str = RELATION) -> None:
    for left, right in zip(names, names[1:]):
        brain.think(f"{left} {relation}s {right}")


def _lesson(*names: str, relation: str = RELATION, source: str = "recorded") -> Lesson:
    return Lesson(task=f"does {names[0]} {relation} {names[-1]}?", answer="yes",
                  steps=tuple(Step(a, relation, b) for a, b in zip(names, names[1:])),
                  source=source, confidence=0.7)


# --------------------------------------------------------------------------- #
# the wire that was missing
# --------------------------------------------------------------------------- #

def test_a_relation_that_is_not_causes_composes_through_think():
    """Regression: `core.reach`/`.connects` had zero production callers and were reached only
    with the literal predicate "causes"."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    said = brain.think("does zorbo kizzle pluron?")
    assert said.answer.strip().lower().startswith("yes")
    assert said.solution.strategy == "compose"
    assert brain.think("what does zorbo ultimately kizzle?").answer.strip() == "pluron"


def test_composition_does_not_invent_a_link_that_is_not_there():
    """The half that makes the other half worth having."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    _chain(brain, "morvo", "brenth")
    assert not brain.think("does zorbo kizzle brenth?").answer.strip().lower().startswith("yes")


def test_a_polar_question_is_about_the_object_it_names():
    """`core.answer` returns the store's best object for the subject whatever was asked, so
    "does zorbo kizzle pluron?" came back "vanth" — non-empty, criticised clean, accepted, and the
    turn was over before `compose` could be tried. Answering a different question is not a defect
    the critic can see."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    said = brain.think("does zorbo kizzle pluron?").answer.strip().lower()
    assert said != "vanth"
    assert said.startswith("yes")


def test_a_composed_answer_is_priced_off_the_chain_that_produced_it():
    """`believed` with 0.00 confidence is not a hedge, it is a contradiction. `solve` copies its
    context, so the walk `compose` made had no route back to the turn that called it."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    thought = brain.think("does zorbo kizzle pluron?")
    assert thought.epistemic == "believed"
    assert thought.epistemic_confidence > 0.0
    assert thought.derivation is not None


# --------------------------------------------------------------------------- #
# verification: a demonstration is a claim like any other
# --------------------------------------------------------------------------- #

def test_an_answer_with_no_working_teaches_nothing():
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    got = Distiller().distil(Lesson(task="does zorbo kizzle pluron?", answer="yes"), brain)
    assert got.verification.verdict == Verdict.UNDECIDED
    assert got.moved == 0.0


def test_a_demonstration_she_cannot_check_is_undecided_not_probably_right():
    brain = NJPBrain()
    distiller = Distiller()
    got = distiller.distil(_lesson("zorbo", "vanth", "pluron"), brain)
    assert got.verification.verdict == Verdict.UNDECIDED
    assert got.verification.corroborated == 0
    assert distiller.distilled == 0
    assert got.moved == 0.0


def test_a_demonstration_her_facts_contradict_moves_the_posterior_down():
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    before = brain.learner._transitivity(RELATION).value
    distiller = Distiller()
    got = distiller.distil(_lesson("zorbo", "quilon", "pluron"), brain)
    assert got.verification.verdict == Verdict.REFUTED
    assert distiller.refuted == 1
    assert brain.learner._transitivity(RELATION).value < before


def test_the_conclusion_is_never_stored_as_a_fact():
    """The entire design in one assertion. What is kept is that the relation chains; what she was
    shown — that *these* entities chain — is thrown away."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    before = {(s, p, o) for s, p, o, _c in brain.learner._edges()}
    Distiller().distil(_lesson("zorbo", "vanth", "pluron"), brain)
    after = {(s, p, o) for s, p, o, _c in brain.learner._edges()}
    assert after == before
    assert ("zorbo", RELATION, "pluron") not in after


def test_a_verified_chain_confirms_that_the_relation_chains():
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    distiller = Distiller()
    got = distiller.distil(_lesson("zorbo", "vanth", "pluron"), brain)
    assert got.verification.verdict == Verdict.SURVIVED
    assert got.relation == RELATION
    assert got.after > got.before
    assert distiller.distilled == 1
    assert distiller.relations[RELATION] == 1


def test_a_mixed_chain_is_a_shape_and_not_a_transitivity_claim():
    """`sparrow is_a bird`, `bird needs water` working once is not evidence that `is_a` chains."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    brain.think("birds need water")
    edges = {(s, p) for s, p, _o, _c in brain.learner._edges()}
    relations = {p for _s, p in edges}
    assert len(relations) >= 2, edges
    steps = tuple(Step(s, p, o) for s, p, o, _c in brain.learner._edges())
    lesson = Lesson(task="what does a sparrow need?", answer="water", steps=steps,
                    source="recorded", confidence=0.7)
    distiller = Distiller()
    got = distiller.distil(lesson, brain)
    assert got.relation == ""
    assert got.moved == 0.0
    assert distiller.distilled == 0


def test_one_demonstration_is_one_observation():
    """A teacher whose word could move the posterior further would be a teacher whose word is
    evidence, which is what the gates exist to refuse."""
    brain = NJPBrain()
    _chain(brain, "zorbo", "vanth", "pluron")
    posterior = brain.learner._transitivity(RELATION)
    before = posterior.alpha
    Distiller().distil(_lesson("zorbo", "vanth", "pluron"), brain)
    assert posterior.alpha - before <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# the council
# --------------------------------------------------------------------------- #

def test_a_teacher_that_cannot_answer_contributes_nothing():
    council = TeacherCouncil([RecordedTeacher({})])
    assert council.available is False
    assert council.ask("does zorbo kizzle pluron?") == []


def test_a_recorded_teacher_answers_only_what_it_was_given():
    teacher = RecordedTeacher({"does zorbo kizzle pluron?": _lesson("zorbo", "vanth", "pluron")})
    assert teacher.teach("does zorbo kizzle pluron?") is not None
    assert teacher.teach("does morvo kizzle quilon?") is None
    assert teacher.asked == 2 and teacher.answered == 1


def test_disagreement_is_recorded_rather_than_voted_away():
    """§19 is explicit that specialisation is not majority rule. Which teacher is right is a
    question for verification, and it answers it per lesson; a majority would answer it by
    counting teachers, which is a fact about the council and not about the world."""
    task = "does zorbo kizzle pluron?"
    yes = RecordedTeacher({task: _lesson("zorbo", "vanth", "pluron")}, name="a")
    no = RecordedTeacher({task: Lesson(task=task, answer="no", steps=(), source="b")}, name="b")
    council = TeacherCouncil([yes, no])
    lessons = council.ask(task)
    assert len(lessons) == 2
    assert council.disagreements == 1
    assert {l.source for l in lessons} == {"a", "b"}


# --------------------------------------------------------------------------- #
# acquisition: it has to reach entities the teacher never mentioned
# --------------------------------------------------------------------------- #

def test_what_is_distilled_reaches_a_chain_the_teacher_never_saw():
    """Retention in one test. The demonstration is on one chain; the question is about another."""
    brain = NJPBrain()
    _chain(brain, "aa0", "aa1", "aa2", "aa3", "aa4")          # demonstrated
    _chain(brain, "bb0", "bb1", "bb2", "bb3", "bb4")          # never mentioned
    question = "does bb0 kizzle bb4?"
    assert not brain.think(question).answer.strip().lower().startswith("yes")

    distiller = Distiller()
    got = distiller.distil(_lesson("aa0", "aa1", "aa2", "aa3", "aa4"), brain)
    assert got.verification.verdict == Verdict.SURVIVED
    assert brain.think(question).answer.strip().lower().startswith("yes")


def test_the_three_arms_separate_and_the_control_holds():
    from nyxara.eval.acquisition import run_acquisition_benchmark

    report = run_acquisition_benchmark(seed=42, chains=4)
    alone, taught, distilled = (report.arm("alone"), report.arm("taught"),
                                report.arm("distilled"))
    assert alone.score < taught.score, report.render()
    assert distilled.score > alone.score, report.render()
    assert report.retention is not None and report.retention > 0.5
    # "always yes" scores 0.50 and cannot win: the distilled arm earns its score on both halves.
    assert distilled.false_positive_rate == 0.0
    assert report.transitivity_after > report.transitivity_before
    assert report.lessons_verified > 0 and report.lessons_refuted == 0


def test_retention_is_refused_when_there_was_no_advantage_to_keep():
    """A ratio over a denominator of about zero is noise with a percent sign, and reporting it
    would make the least informative run look like the most decisive one."""
    from nyxara.eval.acquisition import run_acquisition_benchmark

    report = run_acquisition_benchmark(seed=42, chains=3, hops=3)
    assert report.arm("alone").score == report.arm("taught").score == 1.0
    assert report.retention is None
    assert "n/a" in report.render()
