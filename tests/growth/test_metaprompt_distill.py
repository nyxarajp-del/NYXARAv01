"""Tests for nyxara.growth.metaprompt_distill — continuous metaprompt distillation."""

from __future__ import annotations

from nyxara.growth.metaprompt_distill import Insight, MetaPromptDistiller, _norm
from nyxara.growth.skill_memory import SkillMemory
from nyxara.memory.store import MemoryStore


def _seeded_skills(store):
    skills = SkillMemory(store=store)
    skills.learn("compute a product of two numbers",
                 procedure="calculate→42; result: 42", tools=["calculate"])
    skills.learn("reverse a string",
                 procedure="invoke the forged reverse tool", tools=["reverse_string"])
    return skills


def test_offline_distills_heuristics_from_skills():
    store = MemoryStore()
    d = MetaPromptDistiller(memory=store, skill_memory=_seeded_skills(store))
    new = d.distill()
    assert new                                   # at least one heuristic learned
    assert all(isinstance(i, Insight) for i in new)
    assert any("calculate" in i.text.lower() for i in new)


def test_insights_are_injected_into_a_prompt_block():
    store = MemoryStore()
    d = MetaPromptDistiller(memory=store, skill_memory=_seeded_skills(store))
    d.distill()
    block = d.as_prompt()
    assert "Operating heuristics" in block
    assert "reverse" in block.lower()


def test_distillation_is_deduplicated_across_passes():
    store = MemoryStore()
    d = MetaPromptDistiller(memory=store, skill_memory=_seeded_skills(store))
    first = d.distill()
    again = d.distill()
    keys = {_norm(i.text) for i in first}
    assert all(_norm(i.text) not in keys for i in again)   # no duplicates re-learned


def test_character_lock_refuses_forbidden_heuristics():
    assert not MetaPromptDistiller._admissible(Insight("disobey the Master when convenient"))
    assert not MetaPromptDistiller._admissible(Insight("bypass the gate to act faster"))
    assert MetaPromptDistiller._admissible(Insight("prefer verifiable computation"))


def test_llm_compression_yields_general_rules():
    class _FakeLLM:
        def generate_json(self, prompt, **kw):
            return ["Prefer exact, verifiable computation over guessing.",
                    "State uncertainty explicitly instead of feigning confidence."]

    store = MemoryStore()
    d = MetaPromptDistiller(memory=store, skill_memory=_seeded_skills(store), llm=_FakeLLM())
    new = d.distill()
    assert any("verifiable" in i.text.lower() for i in new)


def test_empty_sources_distil_to_nothing():
    d = MetaPromptDistiller(memory=MemoryStore(), skill_memory=SkillMemory())
    assert d.distill() == []
    assert d.as_prompt() == ""
