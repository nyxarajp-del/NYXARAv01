"""Tests for nyxara.agency.autonomous_tool_forge — self-correcting permanent tool forging."""

from __future__ import annotations

from nyxara.agency.autonomous_tool_forge import AutonomousToolForge
from nyxara.agency.permissions import Authority
from nyxara.agency.tools import ToolParam, ToolRegistry
from nyxara.growth.skill_memory import SkillMemory


class _BuggyThenFixedLLM:
    """First emits a buggy reverse(), then ships the correct one after reading the failure."""

    def __init__(self):
        self.n = 0

    def generate(self, prompt, *, system=None, **kw):
        self.n += 1
        if self.n == 1:                      # bug: reverses by word, not character
            return "def handle(text):\n    return ' '.join(text.split()[::-1])\n"
        return "def handle(text):\n    return text[::-1]\n"


def _reverse_examples():
    return [{"args": {"text": "abc"}, "expect": "cba"},
            {"args": {"text": "nyxara"}, "expect": "araxyn"}]


def test_forge_self_fixes_then_deploys_and_remembers():
    reg = ToolRegistry()
    skills = SkillMemory()
    forge = AutonomousToolForge(registry=reg, llm=_BuggyThenFixedLLM(), skill_memory=skills)

    out = forge.forge("reverse the characters of a string",
                      params=[ToolParam("text", type="str", required=True)],
                      examples=_reverse_examples())

    assert out.ok and out.deployed
    assert out.origin == "llm"
    assert out.attempts >= 2                 # the first buggy draft was debugged
    assert out.fixes                         # a real sandbox failure was read and fixed
    assert out.skill_id                      # the winning procedure was remembered


def test_forged_tool_is_invokable_in_the_registry():
    reg = ToolRegistry()
    forge = AutonomousToolForge(registry=reg, llm=_BuggyThenFixedLLM(),
                                skill_memory=SkillMemory())
    out = forge.forge("reverse a string",
                      params=[ToolParam("text", type="str", required=True)],
                      examples=_reverse_examples())
    assert out.deployed
    res = reg.invoke(out.tool_name, {"text": "hello"}, authority=Authority.OWNER)
    assert res.ok and res.value == "olleh"


def test_skill_recorded_references_the_forged_tool():
    reg = ToolRegistry()
    skills = SkillMemory()
    forge = AutonomousToolForge(registry=reg, llm=_BuggyThenFixedLLM(), skill_memory=skills)
    out = forge.forge("reverse a string",
                      params=[ToolParam("text", type="str", required=True)],
                      examples=_reverse_examples())
    recalled = skills.recall("reverse", k=3)
    assert recalled and any(out.tool_name in s.tools for s in recalled)


def test_offline_falls_back_to_deterministic_recipe():
    reg = ToolRegistry()
    forge = AutonomousToolForge(registry=reg, llm=None, skill_memory=SkillMemory())
    out = forge.forge("compute the length of a string",
                      params=[ToolParam("text", type="str", required=True)])
    assert out.ok and out.deployed              # still produces a working tool with no LLM
    assert out.origin in ("template", "llm")


def test_forge_never_raises_on_a_broken_llm():
    class _Broken:
        def generate(self, *a, **k):
            raise RuntimeError("boom")
    reg = ToolRegistry()
    forge = AutonomousToolForge(registry=reg, llm=_Broken(), skill_memory=SkillMemory())
    out = forge.forge("do something", params=[ToolParam("x", type="str", required=True)])
    # falls through to the deterministic recipe path; the call is data, never an exception
    assert out is not None and isinstance(out.ok, bool)
