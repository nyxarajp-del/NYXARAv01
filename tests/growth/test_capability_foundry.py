"""Tests for growth/capability_foundry.py — the Capability Foundry (Level 15).

All deterministic and fully offline (no LLM injected → template path), so the whole
plan → write → test → benchmark → deploy pipeline runs in CI without an API key. Every
test uses a ``tmp_path`` root so nothing touches the real ``~/.nyxara``.
"""

from __future__ import annotations

import hashlib

import pytest

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolRegistry
from nyxara.growth.capability_foundry import (CapabilityFoundry, ForgeStage)
from nyxara.kernel.errors import AuthorizationError


def _foundry(tmp_path) -> CapabilityFoundry:
    return CapabilityFoundry(registry=ToolRegistry(), root=str(tmp_path))


# --------------------------------------------------------------------------- #
# plan / classify
# --------------------------------------------------------------------------- #
def test_plan_classifies_known_shape(tmp_path):
    plan = _foundry(tmp_path).plan("please add a tool to reverse a string")
    assert plan.shape == "text_reverse"
    assert plan.capability is Capability.TOOL_CALL
    assert plan.risk is RiskTier.LOW
    assert plan.tool_name == "text_reverse"
    assert plan.examples  # examples drive both test + benchmark


def test_plan_unknown_falls_back_to_generic(tmp_path):
    plan = _foundry(tmp_path).plan("generate a photorealistic image of a cat")
    assert plan.shape == "generic"
    assert plan.tool_name.startswith("cap_")


def test_plan_names_are_collision_free(tmp_path):
    f = _foundry(tmp_path)
    f.forge("reverse a string", authority=Authority.OWNER)   # registers "text_reverse"
    plan2 = f.plan("reverse a string")
    assert plan2.tool_name not in f.registry.names()         # must not collide


# --------------------------------------------------------------------------- #
# synthesize / static safety scan
# --------------------------------------------------------------------------- #
def test_synthesize_template_defines_handle(tmp_path):
    f = _foundry(tmp_path)
    plan = f.plan("sha256 hash")
    source, origin = f.synthesize(plan)
    assert origin == "template"
    assert "def handle" in source
    ok, _ = f._scan_source(source)
    assert ok


def test_scan_rejects_os_reach(tmp_path):
    ok, reason = _foundry(tmp_path)._scan_source(
        "import os\ndef handle(x):\n    return os.listdir('.')\n")
    assert not ok and "os" in reason


def test_scan_rejects_eval(tmp_path):
    ok, _ = _foundry(tmp_path)._scan_source(
        "def handle(x):\n    return eval(x)\n")
    assert not ok


def test_scan_rejects_dunder_escape(tmp_path):
    ok, _ = _foundry(tmp_path)._scan_source(
        "def handle(x):\n    return ().__class__.__bases__\n")
    assert not ok


# --------------------------------------------------------------------------- #
# test / benchmark stages
# --------------------------------------------------------------------------- #
def test_test_stage_passes_for_correct_source(tmp_path):
    f = _foundry(tmp_path)
    plan = f.plan("reverse a string")
    source, _ = f.synthesize(plan)
    report = f.test(plan, source)
    assert report.ok and report.passed == report.total


def test_test_stage_fails_for_wrong_expectation(tmp_path):
    f = _foundry(tmp_path)
    plan = f.plan("reverse a string")
    plan.examples = [{"args": {"text": "abc"}, "expect": "WRONG"}]
    report = f.test(plan, "def handle(text):\n    return text[::-1]\n")
    assert not report.ok and report.failures


def test_benchmark_reports_pass_rate_and_latency(tmp_path):
    f = _foundry(tmp_path)
    plan = f.plan("factorial")
    source, _ = f.synthesize(plan)
    bench = f.benchmark(plan, source)
    assert bench.pass_rate == 1.0
    assert bench.runs >= 1
    assert bench.mean_latency_ms >= 0.0


# --------------------------------------------------------------------------- #
# gauntlet — sovereign-core refusal
# --------------------------------------------------------------------------- #
def test_gauntlet_refuses_sovereign_core_under_autonomous(tmp_path):
    f = _foundry(tmp_path)
    plan = f.plan("reverse a string")
    plan.capability = Capability.MODIFY_RULES
    ok, reason = f.gauntlet(plan, "def handle(text):\n    return text\n",
                            authority=Authority.AUTONOMOUS)
    assert not ok and "sovereign core" in reason


def test_deploy_refuses_privileged_under_autonomous(tmp_path):
    from nyxara.growth.capability_foundry import ForgedTool
    f = _foundry(tmp_path)
    plan = f.plan("reverse a string")
    plan.capability = Capability.SELF_MODIFY
    forged = ForgedTool(version=1, name=plan.tool_name,
                        source="def handle(text):\n    return text\n",
                        plan=plan.to_dict(), capability=plan.capability.value)
    with pytest.raises(AuthorizationError):
        f.deploy(plan, forged, authority=Authority.AUTONOMOUS)


# --------------------------------------------------------------------------- #
# forge end-to-end + invoke through the registry
# --------------------------------------------------------------------------- #
def test_forge_end_to_end_and_invoke(tmp_path):
    f = _foundry(tmp_path)
    res = f.forge("please add a sha256 hash skill", authority=Authority.OWNER)
    assert res.deployed and res.stage is ForgeStage.DONE
    assert res.test_passed and res.benchmark_score == 1.0
    assert res.tool_name in f.registry.names()

    r = f.registry.invoke(res.tool_name, {"text": "abc"}, authority=Authority.OWNER)
    assert r.ok and r.value == hashlib.sha256(b"abc").hexdigest()


def test_forge_autonomous_safe_tier_deploys(tmp_path):
    # the user's choice: a safe-tier gap forges and deploys autonomously (no Master).
    f = _foundry(tmp_path)
    res = f.forge("base64 encode", authority=Authority.AUTONOMOUS)
    assert res.deployed
    r = f.registry.invoke(res.tool_name, {"text": "hi"}, authority=Authority.AUTONOMOUS)
    assert r.ok and r.value == "aGk="


# --------------------------------------------------------------------------- #
# persistence / restore / rollback
# --------------------------------------------------------------------------- #
def test_persistence_reload_and_restore(tmp_path):
    f1 = CapabilityFoundry(registry=ToolRegistry(), root=str(tmp_path))
    res = f1.forge("reverse a string", authority=Authority.OWNER)
    assert (tmp_path / f"v{res.version}" / "handler.py").exists()

    # a fresh foundry on the same root reloads the manifest AND restores the tool.
    reg2 = ToolRegistry()
    f2 = CapabilityFoundry(registry=reg2, root=str(tmp_path))
    assert len(f2.forged) >= 1
    assert res.tool_name in reg2.names()                 # survived the "restart"
    r = reg2.invoke(res.tool_name, {"text": "xyz"}, authority=Authority.OWNER)
    assert r.ok and r.value == "zyx"


def test_rollback_unregisters(tmp_path):
    f = _foundry(tmp_path)
    res = f.forge("reverse a string", authority=Authority.OWNER)
    assert res.tool_name in f.registry.names()
    removed = f.rollback(res.tool_name, authority=Authority.OWNER)
    assert removed and res.tool_name not in f.registry.names()
    assert all((not t.deployed) for t in f.forged if t.name == res.tool_name)


def test_rollback_requires_owner(tmp_path):
    f = _foundry(tmp_path)
    res = f.forge("reverse a string", authority=Authority.OWNER)
    with pytest.raises(AuthorizationError):
        f.rollback(res.tool_name, authority=Authority.AUTONOMOUS)


# --------------------------------------------------------------------------- #
# orchestrator wiring
# --------------------------------------------------------------------------- #
def test_orchestrator_wires_capability_foundry():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert core.capability_foundry is not None
    rep = core.report()
    assert "capabilities_forged" in rep
    out = core.forge_capability("reverse a string", authority=Authority.OWNER)
    assert out["deployed"] is True
    assert out["tool_name"] in core.tools.names()


def test_forge_capability_tool_registered_and_owner_gated():
    from nyxara.agency.default_tools import build_default_tools
    reg = build_default_tools(ToolRegistry())
    assert "forge_capability" in reg.names()
    # invoked autonomously it escalates to the Master (SELF_MODIFY / HIGH).
    r = reg.invoke("forge_capability", {"need": "reverse a string"},
                   authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner
