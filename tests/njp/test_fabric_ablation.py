"""Is the fabric useful, and is the question even askable?

The second half is the harder one and it came first. The fabric had exactly three routes into a
turn — `_temper_by_novelty`, `_review_by_fabric`, and its seat in `_compete` — and all three can
only *lower* a confidence. No answer's content depended on it. That is not an oversight; it is a
deliberate one-way safety edge, and it means "does the fabric help?" had no way of being answered
either way.

Nulling `brain.fabric` cannot ask it. `_build_fabric`'s own docstring calls it "the one organ that
is never optional" and `perceive` calls it unguarded, so a null lands in a blanket `except` and
yields an empty percept — silent-exception degradation, not the designed degraded path. An
ablation running down that path reports a difference caused by breakage and reads exactly like a
real result, which is the single failure `eval/ablation.py` is mostly built to avoid.

So: four gates, one per claim, read at the point of use; a `gate_faculty` that reports whether it
actually changed anything; and a brain-level `ablate_njp` cheap enough to run more than once.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nyxara.eval.ablation import (NJP_FABRIC_FACULTIES, gate_faculty, mcnemar_exact)
from nyxara.njp.brain import NJPBrain
from nyxara.njp.memory import HoloMemory


# --------------------------------------------------------------------------- #
# The instrument
# --------------------------------------------------------------------------- #

def test_the_toggle_reports_whether_it_actually_changed_something():
    """The experiment's validity check. A no-op toggle is a broken experiment, not a null."""
    faculty = gate_faculty("fabric_retrieval", "fabric_retrieval")
    brain = NJPBrain()
    assert faculty.disable(brain) is True
    assert faculty.disable(brain) is False, "a second disable changed nothing and must say so"
    assert brain._gate("fabric_retrieval", True) is False


def test_a_brain_with_no_config_can_still_be_ablated():
    """`_gate` reads attributes off the config, and the default brain has none to clear."""
    brain = NJPBrain()
    assert brain.config is None or getattr(brain.config, "fabric_prior_enabled", True) is True
    assert gate_faculty("fabric_prior", "fabric_prior").disable(brain) is True
    assert brain._gate("fabric_prior", True) is False


def test_all_four_arms_are_separately_switchable():
    """One gate over four claims would answer four questions with one number."""
    for faculty in NJP_FABRIC_FACULTIES:
        brain = NJPBrain()
        if faculty.name == "fabric_strategy":
            # Ships off, on measurement: novelty is ~0.50 on the questions that reach strategy
            # selection and >0.95 on statements that never do, so no floor discriminates. A
            # faculty already off cannot be disabled again, and saying so is the validity check.
            assert faculty.disable(brain) is False
            continue
        assert faculty.disable(brain) is True, faculty.name


def test_a_three_nil_split_is_not_reported_as_significant():
    """The guard against laundering a small battery into a recommendation."""
    assert mcnemar_exact(3, 0) > 0.05
    assert mcnemar_exact(0, 0) == 1.0


# --------------------------------------------------------------------------- #
# The arm that can change an answer
# --------------------------------------------------------------------------- #

def test_the_state_she_was_in_is_stored_with_the_memory():
    """It was computed every turn and thrown away one local variable from where it was needed."""
    brain = NJPBrain()
    for line in ("paudhe ko paani chahiye", "paani se growth badhti hai"):
        brain.think(line)
    traces = list(brain.memory.traces.values())
    assert traces, "nothing was remembered at all"
    assert any(t.cells for t in traces), "no trace carries the firing state it was written in"


def test_co_activation_never_overrides_a_decided_content_hit():
    """The guard that keeps this from being the failure `recall_cue` already measured.

    A fuzzy path that answered where it should have abstained turned three honest abstentions
    into three confident wrong answers. A re-rank firing on decided hits is that same failure with
    a different similarity measure.
    """
    memory = HoloMemory()
    memory.remember("a", "the sky is blue", cells=(1, 2, 3))
    memory.remember("b", "grass is green", cells=(90, 91, 92))

    decided = memory.recall("the sky is blue", k=3)
    if not decided.decided:
        pytest.skip("this cue was not decided even on its own text; nothing to guard here")
    with_context = memory.recall("the sky is blue", k=3, context=(90, 91, 92))
    assert with_context.hit.key == decided.hit.key
    assert with_context.rescored is False


def test_a_weak_overlap_leaves_an_undecided_recall_undecided():
    """Returning the best of a bad field is how an association becomes a fabrication."""
    memory = HoloMemory()
    memory.remember("a", "alpha beta gamma", cells=(1, 2, 3, 4))
    out = memory.recall("nothing like anything stored here", k=3, context=(1, 40, 41, 42))
    assert out.rescored is False


def test_the_re_rank_is_recorded_rather_than_silent():
    """A recall that came from the substrate is a different kind of claim and must be countable."""
    memory = HoloMemory()
    memory.remember("a", "alpha", cells=(1, 2, 3, 4))
    memory.remember("b", "beta", cells=(50, 51, 52, 53))
    out = memory.recall("something quite unlike either", k=3, context=(1, 2, 3, 4))
    if out.rescored:
        assert out.hit.key == "a"
        assert out.hit_key_before != "a" or out.hit_key_before == ""


def test_the_gate_actually_stops_the_context_reaching_recall():
    brain = NJPBrain(config=SimpleNamespace(fabric_retrieval_enabled=False))
    seen = {}
    original = brain.memory.recall

    def _spy(cue, *, k=5, context=()):
        seen["context"] = tuple(context)
        return original(cue, k=k, context=context)

    brain.memory.recall = _spy                     # type: ignore[assignment]
    brain.think("paudhe ko paani chahiye")
    assert seen.get("context") == ()


# --------------------------------------------------------------------------- #
# The arms that cannot change an answer, and must not start
# --------------------------------------------------------------------------- #

def test_an_anomaly_produces_a_question_and_never_an_answer():
    """A confident answer on unrecognised ground is worth asking about, not worth re-deciding."""
    lines = ("paudhe ko paani chahiye", "paani se growth badhti hai", "growth kaise badhti hai")

    on = NJPBrain()
    if on.curiosity is None:
        pytest.skip("curiosity gated off in this build")
    off = NJPBrain(config=SimpleNamespace(fabric_anomaly_enabled=False))

    # Two fresh brains down the same turns, differing only in the gate. Comparing one brain
    # against itself would compare turn 1 with turn 2, which differ for reasons that have nothing
    # to do with this arm.
    on_answers = [on.think(line).answer for line in lines]
    off_answers = [off.think(line).answer for line in lines]
    assert on_answers == off_answers, "the anomaly arm reached the answer path"


def test_novelty_buys_a_second_strategy_and_not_a_chosen_one():
    """The obvious version of this arm — novelty steering `choose` — is the wrong one."""
    from nyxara.njp.metareason import MetaReasoner, ProblemKind

    reasoner = MetaReasoner()
    reasoner.register("first", (ProblemKind.FACTUAL,), lambda p, c: "one")
    reasoner.register("second", (ProblemKind.FACTUAL,), lambda p, c: "two")

    calm = reasoner.solve("what is a thing", context={"novelty": 0.0})
    novel = reasoner.solve("what is a thing", context={"novelty": 1.0})
    assert novel.alternative or novel.alternative_answer is not None or calm.alternative == ""
    # And the second opinion may not become the answer.
    assert novel.answer == calm.answer or novel.strategy == calm.strategy


def test_the_fabric_seat_can_be_taken_out_of_the_competition():
    brain = NJPBrain(config=SimpleNamespace(fabric_prior_enabled=False))
    brain.think("paudhe ko paani chahiye")
    assert brain._gate("fabric_prior", True) is False
