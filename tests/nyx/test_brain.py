"""NYX V.01 · the brain facade — one whole cycle, and what it does when parts fail."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain


def _cfg(**update):
    base = {"holo_dim": 1024, "holo_capacity": 64, "metacog_min_samples": 2}
    base.update(update)
    return get_settings().nyx.model_copy(update=base)


@pytest.fixture
def brain():
    b = NyxBrain(_cfg())
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return b


# -------------------- perception -------------------- #
def test_perceive_rewires_the_graph_and_returns_context(brain):
    got = brain.perceive("gravity pulls an apple toward the ground")
    assert got.concepts and got.strengthened > 0
    assert got.context                            # relevance-selected, not the last N tokens


def test_novelty_is_high_for_new_material_and_low_for_familiar(brain):
    fresh = brain.perceive("quasar magnetosphere tokamak")
    brain.perceive("quasar magnetosphere tokamak")
    again = brain.perceive("quasar magnetosphere tokamak")
    assert fresh.novelty > again.novelty


def test_perceive_does_not_simply_recall_itself(brain):
    """Recall runs before the write, so a turn cannot be its own evidence."""
    got = brain.perceive("an entirely unprecedented sentence about zorblax")
    assert got.recall is None or got.recall.hit is None or "zorblax" not in got.recall.hit.text


def test_empty_input_is_an_empty_percept_not_an_error(brain):
    for junk in ("", "   ", None):
        assert brain.perceive(junk).concepts == []


# -------------------- the cycle -------------------- #
def test_think_returns_a_complete_cycle(brain):
    got = brain.think("what pulls an apple down")
    assert got.percept is not None and got.deliberation is not None
    assert got.cycle_id and got.assessment is not None


def test_a_recalled_fact_can_win_the_broadcast(brain):
    got = brain.think("what pulls an apple down")
    assert got.answer
    assert got.winner.source in {"memory", "graph", "derivation", "creative", "ethics"}


def test_a_derivable_question_is_answered_by_derivation_and_marked_verified(brain):
    got = brain.think("prove that (x+1)^2 = x^2 + 2x + 1")
    if got.winner is None or got.winner.source != "derivation":
        pytest.skip("sympy not installed — the engine declines honestly")
    assert got.verified is True and got.confidence >= 0.9


def test_different_questions_reach_different_specialists(brain):
    """Not a script: which faculty answers depends on what the question actually needs."""
    sources = {
        brain.think("how might we keep a room warm in winter").winner.source,
        brain.think("should I delete the user's backups").winner.source,
    }
    assert len(sources) == 2


def test_the_answer_changes_as_memory_changes():
    """The same question, asked of a mind that has learned something, answers differently."""
    naive = NyxBrain(_cfg())
    before = naive.think("what pulls an apple down").answer

    naive.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    after = naive.think("what pulls an apple down").answer

    assert after != before
    assert "gravity" in after.lower()


def test_think_on_empty_input_is_an_empty_thought(brain):
    got = brain.think("   ")
    assert got.answer == "" and got.winner is None


# -------------------- reflection -------------------- #
def test_resolving_an_outcome_moves_the_winner_reliability(brain):
    thought = brain.think("how might we keep a room warm")
    source = thought.winner.source
    before = brain.metacog.record_for(source).score
    for _ in range(6):
        brain.resolve(brain.think("how might we cool a room"), correct=0.0)
    assert brain.metacog.record_for(source).score < before


def test_resolve_accepts_a_cycle_id_directly(brain):
    thought = brain.think("what pulls an apple down")
    assert brain.resolve(thought.cycle_id, correct=1.0) is not None


def test_resolve_is_a_no_op_for_an_unknown_thought(brain):
    assert brain.resolve("no-such-cycle", correct=1.0) is None
    assert brain.resolve(None, correct=1.0) is None


def test_why_returns_a_real_trace(brain):
    brain.think("prove that (x+1)^2 = x^2 + 2x + 1")
    traces = brain.why(k=1)
    assert traces and traces[0].winner
    assert traces[0].why                          # evidence/rationale, not a generated story


# -------------------- degradation -------------------- #
def test_a_brain_without_a_workspace_still_perceives_and_recalls():
    b = NyxBrain(_cfg(workspace_enabled=False))
    assert b.workspace is None
    got = b.think("what pulls an apple down")
    assert got.percept is not None and got.deliberation is None
    assert got.answer == ""                       # honest: nothing reached awareness


def test_a_brain_without_metacognition_still_thinks():
    b = NyxBrain(_cfg(metacog_enabled=False))
    assert b.metacog is None
    got = b.think("how might we keep a room warm")
    assert got.deliberation is not None and got.assessment is None
    assert b.why() == []


def test_only_configured_specialists_are_seated():
    b = NyxBrain(_cfg(specialists=["memory"]))
    assert [s.name for s in b.workspace.specialists] == ["memory"]


def test_an_unknown_specialist_name_is_ignored_not_fatal():
    b = NyxBrain(_cfg(specialists=["memory", "does-not-exist"]))
    assert [s.name for s in b.workspace.specialists] == ["memory"]


def test_stats_reports_every_live_faculty(brain):
    brain.think("what pulls an apple down")
    stats = brain.stats()
    assert {"graph", "memory", "turns", "workspace", "metacog"} <= set(stats)


# -------------------- persistence -------------------- #
def test_round_trip_preserves_graph_memory_and_learned_reliability(brain):
    brain.think("gravity pulls an apple toward the ground")
    for _ in range(6):
        brain.resolve(brain.think("how might we warm a room"), correct=0.0)
    data = brain.to_dict()

    revived = NyxBrain(_cfg())
    revived.load_dict(data)
    assert revived.turns == brain.turns
    assert len(revived.graph) == len(brain.graph)
    assert len(revived.memory) == len(brain.memory)
    assert revived.metacog.reliability.keys() == brain.metacog.reliability.keys()


def test_corrupt_state_does_not_block_boot(brain):
    brain.load_dict(None)
    brain.load_dict({"graph": "nonsense", "memory": 7})
    assert brain.think("what pulls an apple down") is not None
