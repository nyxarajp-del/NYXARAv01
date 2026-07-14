"""Tests for AgentLoop × self-correction — a stuck loop is BROKEN, not silently abandoned.

Before, when the agent loop caught NYXARA repeating herself it simply stopped with
``status="stalled"`` — she gave up. With the :class:`SelfCorrectionLoop` controller wired in,
the loop instead recognises the epistemic gap, runs a recovery (experiment / decompose /
reframe), injects the finding as a fresh observation, and keeps going — only stopping once the
recovery budget is honestly spent. These tests pin that behaviour, and that a genuinely
low-confidence final answer is gated to an honest abstention.
"""
from __future__ import annotations

import pytest

from nyxara.agency.agent_loop import AgentLoop
from nyxara.agency.permissions import Capability, RiskTier
from nyxara.growth.self_correction import SelfCorrectionLoop
from nyxara.kernel.orchestrator import Candidate, NyxaraCore
from nyxara.mind.metacognition import HONEST_ABSTENTION


class _Repeater:
    """A reasoner that always calls the same tool with the same args → a constant result."""

    def __call__(self, stimulus, focus=None):
        return Candidate(text="same again", kind="act",
                         capability=Capability.TOOL_CALL, risk=RiskTier.LOW,
                         confidence=0.8, belief=0.8, tool="calculate",
                         tool_args={"expression": "2*(3+4)"}, rationale="always the same")


def test_stuck_loop_triggers_recovery_instead_of_silently_stalling(tmp_path):
    core = NyxaraCore(reasoner=_Repeater())
    ctrl = SelfCorrectionLoop(path=tmp_path / "sc.json", seed=7)
    loop = AgentLoop(core, max_steps=8, self_correction=ctrl, max_recoveries=2)
    run = loop.run("keep hitting the same wall")
    recovered = [s for s in run.steps if s.kind == "self_correction"]
    assert recovered, "a repeating loop must trigger at least one self-correction recovery"
    assert recovered[0].observation, "the recovery must inject a finding as a new observation"


def test_recovery_budget_is_bounded(tmp_path):
    """Recovery is finite: once the budget is spent she stops honestly rather than spinning forever."""
    core = NyxaraCore(reasoner=_Repeater())
    ctrl = SelfCorrectionLoop(path=tmp_path / "sc.json", seed=7)
    loop = AgentLoop(core, max_steps=12, self_correction=ctrl, max_recoveries=2)
    run = loop.run("an unbreakable wall")
    recoveries = [s for s in run.steps if s.kind == "self_correction"]
    assert len(recoveries) <= 2, "recoveries must not exceed the configured budget"
    assert run.status in ("stalled", "max_steps", "escalated", "completed", "abstained")


def test_loop_without_controller_behaves_as_before(tmp_path):
    """No controller ⇒ the classic stall behaviour is preserved (backward compatible).

    Passing ``self_correction=None`` falls back to the core's controller (the production
    default), so to exercise the no-controller path we disable it on the core itself."""
    core = NyxaraCore(reasoner=_Repeater())
    core.self_correction = None
    loop = AgentLoop(core, max_steps=6)
    assert loop.self_correction is None
    run = loop.run("same wall, no controller")
    assert not any(s.kind == "self_correction" for s in run.steps)
    assert run.status in ("stalled", "max_steps")


class _LowConfidence:
    """A reasoner that answers immediately but with very low confidence."""

    def __call__(self, stimulus, focus=None):
        return Candidate(text="I think maybe 42, but I'm really not sure at all.",
                         kind="respond", capability=Capability.MESSAGE_SEND,
                         risk=RiskTier.LOW, confidence=0.15, belief=0.15,
                         rationale="a guess")


def test_low_confidence_answer_is_gated_to_honest_abstention(tmp_path):
    core = NyxaraCore(reasoner=_LowConfidence())
    ctrl = SelfCorrectionLoop(path=tmp_path / "sc.json", seed=7, answer_min_confidence=0.55)
    loop = AgentLoop(core, max_steps=3, self_correction=ctrl)
    run = loop.run("what is the 500th digit of an unknown constant?")
    assert run.status == "abstained"
    assert run.final_answer == HONEST_ABSTENTION
    assert not run.success


def test_confident_answer_still_completes(tmp_path):
    """The gate must not over-abstain: a normal confident reply completes untouched."""
    core = NyxaraCore(reasoner=None)   # default offline reasoner
    ctrl = SelfCorrectionLoop(path=tmp_path / "sc.json", seed=7)
    loop = AgentLoop(core, max_steps=4, self_correction=ctrl)
    run = loop.run("Say hello to the Master")
    assert run.status == "completed" and run.success
