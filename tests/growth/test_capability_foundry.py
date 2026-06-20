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


# --------------------------------------------------------------------------- #
# Expanded recipe library — real, tested tools (not echo scaffolds)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("need, args, expect", [
    ("count the vowels in a string", {"text": "hello"}, 2),
    ("check if a string is a palindrome", {"text": "Racecar"}, True),
    ("apply rot13 to text", {"text": "Hello"}, "Uryyb"),
    ("compute the standard deviation", {"numbers": [2, 4, 4, 4, 5, 5, 7, 9]}, 2.0),
    ("sort a list of numbers", {"numbers": [3, 1, 2]}, [1, 2, 3]),
    ("decode base64", {"text": "aGk="}, "hi"),
    ("the day of the week for a date", {"date": "2024-01-01"}, "Monday"),
    ("sum of digits of a number", {"n": 123}, 6),
])
def test_expanded_recipes_forge_and_compute(tmp_path, need, args, expect):
    f = _foundry(tmp_path)
    res = f.forge(need, authority=Authority.OWNER)
    assert res.deployed and res.origin == "template"
    assert res.tool_name != "generic" and not res.tool_name.startswith("cap_")
    r = f.registry.invoke(res.tool_name, args, authority=Authority.OWNER)
    assert r.ok and r.value == expect


def test_validate_email_is_real_not_echo(tmp_path):
    f = _foundry(tmp_path)
    res = f.forge("validate an email address", authority=Authority.OWNER)
    assert res.deployed and res.tool_name == "is_email"
    good = f.registry.invoke("is_email", {"text": "a@b.com"}, authority=Authority.OWNER)
    bad = f.registry.invoke("is_email", {"text": "nope"}, authority=Authority.OWNER)
    assert good.value is True and bad.value is False


# --------------------------------------------------------------------------- #
# Parametric synthesis — real code whose body depends on a constant in the gap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("need, args, expect", [
    ("convert celsius to fahrenheit", {"value": 100.0}, 212.0),
    ("convert fahrenheit to celsius", {"value": 32.0}, 0.0),
    ("convert km to miles", {"value": 1.0}, 0.621371),
    ("convert an integer to base 16", {"n": 255}, "ff"),
    ("convert a number to base 2", {"n": 5}, "101"),
    ("caesar cipher with a shift of 3", {"text": "abc"}, "def"),
    ("round to 2 decimal places", {"x": 3.14159}, 3.14),
    ("multiply by 7", {"x": 6}, 42),
])
def test_parametric_synthesis_forges_real_tools(tmp_path, need, args, expect):
    f = _foundry(tmp_path)
    res = f.forge(need, authority=Authority.OWNER)
    assert res.deployed and res.stage is ForgeStage.DONE
    # parametric tools are concrete, not the generic echo scaffold
    assert not res.tool_name.startswith("cap_")
    r = f.registry.invoke(res.tool_name, args, authority=Authority.OWNER)
    assert r.ok
    if isinstance(expect, float):
        assert abs(r.value - expect) <= 1e-6
    else:
        assert r.value == expect


def test_truly_unknown_gap_still_falls_back_to_generic(tmp_path):
    # an open-ended creative gap with no recipe/parametric match stays the honest scaffold
    plan = _foundry(tmp_path).plan("generate a photorealistic image of a cat")
    assert plan.shape == "generic"


def test_generic_scaffold_is_honestly_labelled(tmp_path):
    # when the echo scaffold IS used, it must self-identify (never over-claim a capability)
    f = _foundry(tmp_path)
    res = f.forge("generate a photorealistic image of a cat", authority=Authority.OWNER)
    assert res.deployed
    r = f.registry.invoke(res.tool_name, {"payload": "x"}, authority=Authority.OWNER)
    assert r.ok and r.value == {"echo": "x", "scaffold": True}


# --------------------------------------------------------------------------- #
# learned recipe selection (RecipeLearner)
# --------------------------------------------------------------------------- #
from nyxara.growth.capability_foundry import RecipeLearner  # noqa: E402


def test_recipe_learner_suggests_similar_past_success():
    rl = RecipeLearner(min_samples=2, min_similarity=0.1)
    rl.observe("frobnicate the wobble gadget", "text_reverse")
    rl.observe("frobnicate the wobble gadget", "text_reverse")
    assert rl.suggest("please frobnicate this wobble gadget") == "text_reverse"
    assert rl.suggest("alpha beta gamma unrelated query") is None     # no overlap


def test_recipe_learner_needs_min_samples():
    rl = RecipeLearner(min_samples=3, min_similarity=0.0)
    rl.observe("count the vowels here", "count_vowels")
    assert rl.suggest("count the vowels here") is None                 # below threshold


def test_recipe_learner_ignores_generic():
    rl = RecipeLearner(min_samples=1, min_similarity=0.0)
    rl.observe("anything at all", "generic")
    assert rl._needs == []                                             # never learns generic


def test_recipe_learner_round_trip():
    rl = RecipeLearner(min_samples=2, min_similarity=0.4)
    rl.observe("a b c", "k1")
    rl.observe("d e f", "k2")
    back = RecipeLearner.from_dict(rl.to_dict())
    assert back._needs == rl._needs and back._keys == rl._keys
    assert back.min_samples == 2 and back.min_similarity == 0.4


def test_resolve_recipe_learned_uses_learner(tmp_path):
    f = _foundry(tmp_path)
    f.recipe_learner = RecipeLearner(min_samples=1, min_similarity=0.05)
    f.recipe_learner.observe("frobnicate the wobble gadget", "text_reverse")
    assert f._resolve_recipe_learned("please frobnicate this wobble gadget").key == "text_reverse"
    assert f._resolve_recipe_learned("alpha beta gamma delta epsilon").key == "generic"


def test_recipe_learner_persists_through_forge(tmp_path):
    f = _foundry(tmp_path)
    f.forge("reverse a string", authority=Authority.OWNER)             # logs gap -> text_reverse
    reloaded = _foundry(tmp_path)                                       # same root -> manifest
    assert reloaded.recipe_learner._needs                              # learned mapping survived
    assert "text_reverse" in reloaded.recipe_learner._keys
