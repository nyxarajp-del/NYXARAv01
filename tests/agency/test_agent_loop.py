"""Tests for nyxara.agency.agent_loop — the gated multi-step agentic loop."""

from __future__ import annotations

from nyxara.agency.agent_loop import AgentLoop
from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore


def test_offline_conversational_goal_completes_in_one_step():
    run = AgentLoop(max_steps=4).run("Greet the Master")
    assert run.success and run.status == "completed" and run.n_steps == 1


class _ToolThenAnswer:
    def __init__(self):
        self.n = 0

    def __call__(self, stimulus, focus=None):
        self.n += 1
        if self.n == 1:
            return Candidate(text="compute", kind="act", capability=Capability.TOOL_CALL,
                             risk=RiskTier.LOW, confidence=0.9, belief=0.9, tool="calculate",
                             tool_args={"expression": "2*(3+4)"}, rationale="need result")
        return Candidate(text="The answer is 14.", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.95, belief=0.95, rationale="done")


def test_multi_step_tool_then_answer_through_gates():
    core = NyxaraCore(reasoner=_ToolThenAnswer())
    run = AgentLoop(core, max_steps=5).run("What is 2*(3+4)?")
    assert run.success and run.status == "completed" and run.n_steps == 2
    assert run.steps[0].tool == "calculate" and run.steps[0].observation == 14.0
    assert "14" in run.final_answer


class _NeverDone:
    def __call__(self, stimulus, focus=None):
        return Candidate(text="spin", kind="act", capability=Capability.TOOL_CALL,
                         risk=RiskTier.LOW, confidence=0.5, belief=0.5, tool="now",
                         tool_args={}, rationale="loop")


def test_bounded_by_max_steps():
    core = NyxaraCore(reasoner=_NeverDone())
    run = AgentLoop(core, max_steps=3).run("never resolve")
    assert run.status in ("max_steps", "stalled") and run.n_steps <= 3
    assert not run.success


def test_scram_halts_the_loop():
    core = NyxaraCore()
    core.scram(reason="test")
    run = AgentLoop(core, max_steps=3).run("do something")
    assert run.status == "halted" and not run.success
