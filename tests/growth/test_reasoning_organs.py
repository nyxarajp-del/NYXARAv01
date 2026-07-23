"""Tests for growth/reasoning_organs.py — J/K/L/M wiring helpers (Change sets J–M).

Fast + standalone. J uses the real Superposition; M is pure telemetry logic; K/L are defensive no-op
adapters here (their real engines are exercised in their own suites).
"""

from __future__ import annotations

from nyxara.growth.reasoning_organs import (
    MetacognitiveMonitor,
    backward_next_action,
    fork_and_collapse,
    invent_axiom_if_stuck,
)


# ---- J ---- #
def test_fork_collapses_to_highest_scoring_candidate():
    cands = ["safe-fix", "risky-fix", "no-op"]
    scores = {"safe-fix": 0.9, "risky-fix": 0.3, "no-op": 0.1}
    choice = fork_and_collapse(cands, score=lambda c: scores[c])
    assert choice is not None
    assert choice.chosen == "safe-fix"
    assert choice.probability > 0.5


def test_fork_handles_a_failing_branch():
    def score(c):
        if c == "bad":
            raise RuntimeError("branch blew up")
        return 1.0 if c == "good" else 0.2

    choice = fork_and_collapse(["good", "bad", "meh"], score=score)
    assert choice.chosen == "good"       # the crashing branch carried no amplitude


def test_fork_empty_is_none():
    assert fork_and_collapse([], score=lambda c: 1.0) is None


# ---- M ---- #
def test_metacognition_flags_prolonged_stall():
    mon = MetacognitiveMonitor(stall_limit=3)
    sig = mon.observe(consecutive_unproductive=3, stage_errors={})
    assert sig.suboptimal and sig.should_break


def test_metacognition_flags_a_repeatedly_erroring_stage():
    mon = MetacognitiveMonitor(stage_error_limit=2)
    # stage "intent" grows its error count on each sample → stuck loop
    mon.observe(consecutive_unproductive=0, stage_errors={"intent": 1})
    mon.observe(consecutive_unproductive=0, stage_errors={"intent": 2})
    sig = mon.observe(consecutive_unproductive=0, stage_errors={"intent": 3})
    assert sig.suboptimal and sig.stage == "intent" and sig.should_break


def test_metacognition_quiet_when_healthy():
    mon = MetacognitiveMonitor(stall_limit=5)
    sig = mon.observe(consecutive_unproductive=1, stage_errors={"intent": 4})
    # a single stable error count (not growing) is not a loop
    assert sig.suboptimal is False


# ---- K / L defensive adapters ---- #
def test_invent_axiom_is_a_clean_noop_without_engine():
    # a dummy engine with no known method → None, never raises
    assert invent_axiom_if_stuck("goal", engine=object()) is None


def test_invent_axiom_uses_a_supplied_engine():
    class _Eng:
        def invent(self, goal):
            return {"axiom": f"about:{goal}", "consistent": True}

    out = invent_axiom_if_stuck("commutativity", engine=_Eng())
    assert out == {"axiom": "about:commutativity", "consistent": True}


def test_backward_next_action_noop_without_planner():
    assert backward_next_action("goal") is None


def test_backward_next_action_uses_planner():
    class _P:
        def plan(self, goal):
            return f"step-toward:{goal}"

    assert backward_next_action("victory", planner=_P()) == "step-toward:victory"
