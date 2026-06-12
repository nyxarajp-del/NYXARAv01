"""Tests for nyxara.growth.skill_memory — experiential skill learning."""

from __future__ import annotations

from nyxara.agency.agent_loop import AgentLoop
from nyxara.agency.permissions import Capability, RiskTier
from nyxara.growth.skill_memory import Skill, SkillMemory
from nyxara.kernel.orchestrator import Candidate, NyxaraCore
from nyxara.memory.store import MemoryStore


class _ToolThenAnswer:
    def __init__(self):
        self.n = 0

    def __call__(self, stimulus, focus=None):
        self.n += 1
        if self.n == 1:
            return Candidate(text="compute", kind="act", capability=Capability.TOOL_CALL,
                             risk=RiskTier.LOW, confidence=0.9, belief=0.9, tool="calculate",
                             tool_args={"expression": "6*7"}, rationale="need product")
        return Candidate(text="The product is 42.", kind="respond",
                         capability=Capability.MESSAGE_SEND, risk=RiskTier.LOW,
                         confidence=0.95, belief=0.95, rationale="done")


def _run(goal: str):
    return AgentLoop(NyxaraCore(reasoner=_ToolThenAnswer()), max_steps=4).run(goal)


def test_capture_successful_run():
    skills = SkillMemory()
    sk = skills.capture(_run("What is 6 times 7?"))
    assert sk is not None and "calculate" in sk.tools and len(skills) == 1


def test_as_prompt_surfaces_learned_skill():
    skills = SkillMemory()
    skills.capture(_run("What is 6 times 7?"))
    block = skills.as_prompt("times", k=3)
    assert "calculate" in block and block.strip()


def test_dedup_reinforces_same_goal():
    skills = SkillMemory()
    skills.capture(_run("What is 6 times 7?"))
    again = skills.capture(_run("What is 6 times 7?"))
    assert again is not None and again.uses >= 2 and len(skills) == 1


def test_unsuccessful_run_is_not_learned():
    class _NeverDone:
        def __call__(self, stimulus, focus=None):
            return Candidate(text="spin", kind="act", capability=Capability.TOOL_CALL,
                             risk=RiskTier.LOW, confidence=0.4, belief=0.4, tool="now",
                             tool_args={}, rationale="loop")
    bad = AgentLoop(NyxaraCore(reasoner=_NeverDone()), max_steps=2).run("never")
    assert SkillMemory().capture(bad) is None


def test_persisted_through_memory_store():
    store = MemoryStore()
    skills = SkillMemory(store=store)
    skills.capture(_run("multiply 6 and 7"))
    recalled = skills.recall("multiply", k=3)
    assert recalled and any("multiply" in s.goal for s in recalled)


def test_direct_teaching():
    skills = SkillMemory()
    sk = skills.learn("greet warmly", procedure="reply with a warm greeting", tools=[])
    assert sk.uses >= 1 and len(skills) == 1


def test_skill_serialisation_round_trip():
    sk = Skill(goal="g", tools=["calculate"], procedure="do it")
    again = Skill.from_dict(sk.to_dict())
    assert again.goal == "g" and again.tools == ["calculate"]


def test_agent_loop_captures_into_skill_memory():
    skills = SkillMemory()
    AgentLoop(NyxaraCore(reasoner=_ToolThenAnswer()), max_steps=4,
              skill_memory=skills).run("What is 6 times 7?")
    assert len(skills) == 1
