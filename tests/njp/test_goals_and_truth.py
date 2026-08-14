"""The last two organs that were installed and never spoke.

Both reported a flat zero on every session, and for the same underlying reason as the rest: the
algorithm was written and nothing fed it.

``goals.nodes = 0`` — :class:`~nyxara.njp.intent.IntentReader` already pulled ``goal``,
``actions``, ``ordering`` and ``polarity`` out of every command, and no caller consumed any of it.
The ordering constraint is the one that mattered: "pehle test chala phir code fix kar" states a
**dependency**, and dropping it is the failure mode a tool-using agent must not have.

``truth.establish_rate = 0.0`` — this one is subtler and easy to misread as a virtue. The gauntlet
is fail-closed by design, so a zero looks like strictness. It was not strictness: the brain
supplied neither ``observations`` nor ``holdout`` and wired no predictor, so **no source could
ever support a claim** — only refute one. A gate that cannot pass anything has not been tested.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.truth import Verdict


@pytest.fixture()
def brain() -> NJPBrain:
    return NJPBrain()


# =============================== GOALS ====================================== #
def test_a_command_becomes_a_goal_with_tasks_under_it(brain: NJPBrain):
    brain.think("pehle test chala phir code fix kar")
    stats = brain.stats()["goals"]
    assert stats["nodes"] > 0
    kinds = {n["kind"] for n in brain.goals.tree()}
    assert "goal" in kinds and "task" in kinds


def test_an_ordering_constraint_becomes_a_real_dependency(brain: NJPBrain):
    """"Run the tests first" is a dependency, not a second unrelated job."""
    brain.think("pehle test chala phir code fix kar")
    blocked = [n for n in brain.goals.tree() if n["state"] == "blocked"]
    assert blocked, "the ordering constraint was dropped"
    assert blocked[0]["dependencies"], "blocked with nothing to be blocked on"


def test_the_ordering_endpoints_are_matched_to_their_tasks(brain: NJPBrain):
    """The two halves of an ordering pair are phrases; the actions are the verbs inside them.

    Matching them by string equality never matched anything, so every constraint was silently
    dropped while the tasks themselves looked fine.
    """
    loop = brain.loop
    tasks = {"test": _Node("n1"), "fix": _Node("n2")}
    assert loop._task_for("test chala", tasks) is tasks["test"]
    assert loop._task_for("code fix kar", tasks) is tasks["fix"]
    assert loop._task_for("something else entirely", tasks) is None


def test_a_statement_creates_no_goal(brain: NJPBrain):
    """Only being asked to do something is being asked to do something."""
    brain.think("my name is Jay")
    assert not [n for n in brain.goals.tree() if n["kind"] == "goal"]


def test_an_action_she_was_told_not_to_do_is_not_recorded(brain: NJPBrain):
    """Recording a prohibition as a task is the worst possible reading of an instruction."""
    thought = brain.think("code fix kar lekin test mat chala")
    polarity = getattr(thought.intent, "polarity", None) or {}
    forbidden = {a for a, do in polarity.items() if not do}
    if not forbidden:
        pytest.skip("the intent reader found no negated action in this phrasing")
    names = {n["name"] for n in brain.goals.tree()}
    assert not (names & forbidden)


def test_a_question_she_chose_to_investigate_becomes_a_task(brain: NJPBrain):
    for i in range(20):
        brain.think(f"person{i} is a person")
        brain.think(f"person{i} lives in Mumbai")
    tasks = [n for n in brain.goals.tree() if n["kind"] == "task"]
    assert tasks, "curiosity raised questions and committed to none of them"
    assert any(n["kind"] == "mission" for n in brain.goals.tree())


def test_the_mission_is_created_once_and_only_once(brain: NJPBrain):
    for i in range(24):
        brain.think(f"person{i} is a person")
        brain.think(f"person{i} lives in Mumbai")
    missions = [n for n in brain.goals.tree() if n["kind"] == "mission"]
    assert len(missions) == 1


def test_progress_rolls_up_from_real_children(brain: NJPBrain):
    brain.think("pehle test chala phir code fix kar")
    goal = next(n for n in brain.goals.tree() if n["kind"] == "goal")
    children = [n for n in brain.goals.tree() if n["parent"] == goal["id"]]
    assert children
    brain.goals.complete(children[0]["id"])
    assert 0.0 < brain.goals.progress(goal["id"]) < 1.0


def test_the_loop_reports_what_it_added(brain: NJPBrain):
    thought = brain.think("pehle test chala phir code fix kar")
    goals = thought.loop.to_dict()["goals"]
    assert goals["added"] > 0 and goals["open"] > 0


# =============================== TRUTH ====================================== #
def test_a_fact_stated_once_is_never_established(brain: NJPBrain):
    """One assertion is one source. Establishment needs something outside it to agree."""
    brain.think("Ravi lives in Mumbai")
    thought = brain.think("where does Ravi live")
    assert thought.answer == "Mumbai"
    assert not thought.verified and not thought.assertable


def test_an_independently_restated_fact_is_established(brain: NJPBrain):
    brain.think("my name is Jay")
    brain.think("my name is Jay")               # said again, a separate episode
    thought = brain.think("what is my name")
    assert thought.judgement.verdict == Verdict.ESTABLISHED
    assert thought.verified and thought.assertable


def test_establishment_needs_a_hard_source(brain: NJPBrain):
    """Soft agreement alone never establishes anything — that is how consensus becomes bias."""
    brain.think("my name is Jay")
    brain.think("my name is Jay")
    thought = brain.think("what is my name")
    assert thought.judgement.hard_supports >= 1
    assert thought.judgement.supports >= brain.truth.min_sources


def test_the_hard_source_can_actually_be_surprised(brain: NJPBrain):
    """The whole reason corroboration-by-restatement counts as hard: it can come out wrong."""
    brain.think("my name is Jay")
    brain.think("my name is Jay")
    assert brain.think("what is my name").assertable

    brain.think("my name is Raj")               # the Master says otherwise
    thought = brain.think("what is my name")
    assert not thought.assertable, "a contested fact was still assertable"


def test_a_claim_with_no_evidence_is_not_established(brain: NJPBrain):
    thought = brain.think("my name is Jay")     # nothing corroborates a first assertion
    assert thought.judgement.verdict in (Verdict.ABSTAINED, Verdict.CONJECTURE)
    assert not thought.assertable


def test_a_turn_cannot_corroborate_itself(brain: NJPBrain):
    """She would ground a fact, find it in the record, and report that a source agreed with her."""
    thought = brain.think("Sara lives in Delhi")
    assert not thought.verified
    # The turn's own triple must not be in the pool the gauntlet was given.
    texts = " | ".join(brain.observations)
    assert "Sara lives in Delhi" not in texts


def test_the_evidence_pools_are_cleared_between_turns(brain: NJPBrain):
    """Stale evidence would establish a claim against facts that have nothing to do with it."""
    brain.think("my name is Jay")
    brain.think("my name is Jay")
    brain.think("what is my name")
    assert brain.holdout, "the question should have had held-out support"
    brain.think("hello")                        # grounds nothing
    assert brain.holdout == []


def test_the_holdout_only_contains_other_assertions_of_the_same_fact(brain: NJPBrain):
    brain.think("my name is Jay")
    brain.think("Ravi lives in Mumbai")
    brain.think("my name is Jay")
    brain.think("what is my name")
    assert brain.holdout
    assert all(t.predicate == "has_name" for t in brain.holdout)


def test_the_predictor_agrees_and_disagrees(brain: NJPBrain):
    class _T:
        object = "Jay"

    assert NJPBrain._claim_survives("Jay", _T()) is True
    assert NJPBrain._claim_survives("Raj", _T()) is False


def test_the_establish_rate_is_no_longer_structurally_zero(brain: NJPBrain):
    for _ in range(2):
        brain.think("my name is Jay")
        brain.think("Ravi lives in Mumbai")
    brain.think("what is my name")
    brain.think("where does Ravi live")
    stats = brain.stats()["truth"]
    assert stats["judged"] > 0
    assert stats["established"] > 0
    assert stats["establish_rate"] > 0.0


def test_being_able_to_pass_is_not_the_same_as_passing_everything(brain: NJPBrain):
    """The gate had never been tested because nothing could get through it. It still holds."""
    for _ in range(2):
        brain.think("my name is Jay")
    brain.think("Ravi lives in Mumbai")          # said exactly once
    brain.think("what is my name")
    brain.think("where does Ravi live")
    stats = brain.stats()["truth"]
    assert 0.0 < stats["establish_rate"] < 1.0


# --- a stand-in for a goal node, for the pure ordering-match test --- #
class _Node:
    def __init__(self, nid: str) -> None:
        self.nid = nid
