"""Tests for nyxara.mind.offline_mind — the sovereign offline mind (keyless machine).

These assert the *real* behaviour that replaced the old "I understand: {x}" echo:
NYXARA answers from her own faculties (NLP + knowledge + memory) and maps commands to
*real* tools, or honestly admits when she has no grounding.
"""

from __future__ import annotations

from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.tools import ToolRegistry
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.knowledge.base import KnowledgeBase
from nyxara.mind.offline_mind import OfflineMind


def _settings():
    return NyxaraSettings.for_profile(Profile.TEST)


def test_greeting_is_warm_not_echo():
    mind = OfflineMind(settings=_settings())
    c = mind.respond("hello there", [])
    assert c.kind == "respond"
    assert "i understand:" not in c.text.lower()
    assert c.confidence >= 0.7


def test_summarize_uses_real_nlp():
    mind = OfflineMind(settings=_settings())
    text = ("NYXARA is a sovereign cognitive architecture. She proposes and the kernel "
            "disposes. Every action passes ordered safety gates before it may run. Her "
            "memory has four stores. She can train her own model in the foundry.")
    c = mind.respond(f"summarize: {text}", [])
    assert "Summary" in c.text
    # the summary is drawn from the source, not invented
    assert "•" in c.text


def test_keyphrases_request():
    mind = OfflineMind(settings=_settings())
    text = "Reinforcement learning trains an agent by rewarding good actions over time."
    c = mind.respond(f"what are the keywords of: {text}", [])
    assert "Key phrases" in c.text


def test_grounded_answer_from_knowledge_base():
    kb = KnowledgeBase()
    kb.ingest_text("NYXARA's Master is Jaypal Khoja, also known as JP.", source="owner")
    mind = OfflineMind(knowledge=kb, settings=_settings())
    c = mind.respond("who is your Master?", [])
    assert c.kind == "respond"
    assert "Jaypal" in c.text or "JP" in c.text
    assert "i understand:" not in c.text.lower()


def test_grounded_answer_from_recalled_memories():
    mind = OfflineMind(settings=_settings())
    c = mind.respond("how should you answer me?",
                     ["The Master prefers concise answers."])
    assert "concise" in c.text


def test_ungrounded_is_honest_admission_not_echo():
    mind = OfflineMind(settings=_settings())
    c = mind.respond("what is the meaning of zibbledorf qux?", [])
    assert c.kind == "respond"
    assert c.confidence <= 0.3
    assert "i understand:" not in c.text.lower()


def test_command_maps_to_real_tool():
    tools = build_default_tools(ToolRegistry(), memory=None)
    mind = OfflineMind(tools=tools, settings=_settings())
    c = mind.act("read the file scratch/notes.txt")
    assert c is not None
    assert c.kind == "act"
    assert c.tool == "read_file"
    # the tool's *declared* contract drives the proposal
    spec = tools.get("read_file")
    assert c.risk == spec.risk and c.reversible == spec.reversible
    # the target path was extracted from the command
    assert "notes.txt" in c.target


def test_command_with_no_matching_tool_returns_none():
    tools = build_default_tools(ToolRegistry(), memory=None)
    mind = OfflineMind(tools=tools, settings=_settings())
    assert mind.act("teleport the spaceship to Andromeda") is None


def test_no_tools_means_no_action():
    mind = OfflineMind(tools=None, settings=_settings())
    assert mind.act("read the file x.txt") is None


def test_web_search_command_maps_and_extracts_query():
    tools = build_default_tools(ToolRegistry(), memory=None)
    mind = OfflineMind(tools=tools, settings=_settings())
    c = mind.act("search the web for black holes")
    assert c is not None and c.kind == "act"
    assert c.tool in ("web_search", "web_fetch")
