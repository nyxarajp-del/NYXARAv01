"""Tests for nyxara.agency.llm_tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.agency.llm_tool import register_foundry_tools, register_llm_tool
from nyxara.agency.permissions import Authority, Capability
from nyxara.agency.tools import ToolRegistry
from nyxara.growth.foundry import Foundry
from nyxara.growth.learn import Experience, ReplayBuffer
from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.mind.llm import LLM


def _llm() -> LLM:
    return LLM(settings=NyxaraSettings.for_profile(Profile.TEST))   # mock provider


def _foundry(tmp_path) -> Foundry:
    settings = NyxaraSettings.for_profile(Profile.TEST)
    settings.llm.self_model_dir = tmp_path / "foundry"
    settings.foundry.backend = "ngram"   # fast, torch-independent
    rb = ReplayBuffer(capacity=50)
    for _ in range(20):
        rb.add(Experience(action="serve jp", features={}, reward=1.0,
                          context="nyxara serves the master jp loyally"))
    return Foundry(settings=settings, replay=rb)


# -------------------- the LLM as a governed tool -------------------- #
def test_llm_tool_registered_with_tool_call_capability():
    reg = ToolRegistry()
    spec = register_llm_tool(reg, _llm())
    assert spec.name == "llm.complete"
    assert spec.capability is Capability.TOOL_CALL   # the LLM is a tool, not the driver
    assert spec.rate_resource == "llm"


def test_llm_tool_generation_is_governed_and_auditable():
    reg = ToolRegistry()
    register_llm_tool(reg, _llm())
    r = reg.invoke("llm.complete", {"prompt": "who is your master?", "system": "You are NYXARA."})
    assert r.ok
    assert "who is your master?" in r.value["text"]   # mock echoes the prompt
    assert r.value["provider"] == "mock"
    # auditable: the result serialises with provider/model/usage
    d = r.to_dict()
    assert d["tool"] == "llm.complete" and d["ok"]


def test_llm_tool_charges_spend_budget():
    from nyxara.agency.governor import Governor
    from nyxara.kernel.config import ResourceLimits
    reg = ToolRegistry(governor=Governor(ResourceLimits(daily_spend_budget=1.5)))
    register_llm_tool(reg, _llm(), cost=1.0)
    assert reg.invoke("llm.complete", {"prompt": "a"}).ok          # 1.0 charged
    drained = reg.invoke("llm.complete", {"prompt": "b"})          # would exceed 1.5
    assert not drained.ok and "spend" in (drained.error or "")


# -------------------- foundry tools are fail-closed (SELF_MODIFY) -------------------- #
def test_foundry_tools_registered_as_self_modify(tmp_path):
    reg = ToolRegistry()
    specs = register_foundry_tools(reg, _foundry(tmp_path))
    assert {s.name for s in specs} == {
        "foundry.train", "foundry.evaluate", "foundry.promote", "foundry.rollback"}
    assert all(s.capability is Capability.SELF_MODIFY for s in specs)


def test_autonomous_promote_escalates_to_master(tmp_path):
    reg = ToolRegistry()
    register_foundry_tools(reg, _foundry(tmp_path))
    r = reg.invoke("foundry.promote", {"version": 1}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner    # SELF_MODIFY caps autonomous risk -> escalate


def test_owner_confirmed_train_runs(tmp_path):
    reg = ToolRegistry()
    register_foundry_tools(reg, _foundry(tmp_path))
    r = reg.invoke("foundry.train", {"spec_kind": "ngram"},
                   authority=Authority.AUTONOMOUS, owner_confirmed=True)
    assert r.ok and r.value["param_count"] > 0 and r.value["version"] == 1
