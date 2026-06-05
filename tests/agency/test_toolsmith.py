"""Tests for nyxara.agency.toolsmith."""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.agency.toolsmith import (ArgBinding, CompositeTool, ToolStep, Toolsmith,
                                     ValidationReport)
from nyxara.kernel.errors import AuthorizationError, ToolError, ValidationError


def _smith():
    reg = ToolRegistry()

    @reg.tool("fetch", params=[ToolParam("url", "str")], capability=Capability.NET_OUT)
    def fetch(url):
        return {"title": "hello", "status": 200}

    @reg.tool("get_field", params=[ToolParam("data", "dict"), ToolParam("key", "str")])
    def get_field(data, key):
        return data[key]

    @reg.tool("shout", params=[ToolParam("text", "str")])
    def shout(text):
        return text.upper()

    return Toolsmith(reg), reg


def _pipeline():
    return [
        ToolStep("s1", "fetch", {"url": ArgBinding.inp("url")}),
        ToolStep("s2", "get_field", {"data": ArgBinding.from_step("s1"),
                                     "key": ArgBinding.literal("title")}),
        ToolStep("s3", "shout", {"text": ArgBinding.from_step("s2")}),
    ]


# -------------------- ArgBinding -------------------- #
def test_binding_literal():
    assert ArgBinding.literal(5).resolve({}, {}) == 5


def test_binding_input():
    assert ArgBinding.inp("x").resolve({"x": 9}, {}) == 9


def test_binding_input_missing():
    with pytest.raises(ValidationError):
        ArgBinding.inp("x").resolve({}, {})


def test_binding_step():
    assert ArgBinding.from_step("s1").resolve({}, {"s1": "v"}) == "v"


def test_binding_step_key():
    assert ArgBinding.from_step("s1", "k").resolve({}, {"s1": {"k": 7}}) == 7


def test_binding_step_missing():
    with pytest.raises(ValidationError):
        ArgBinding.from_step("s9").resolve({}, {})


def test_binding_step_bad_key():
    with pytest.raises(ValidationError):
        ArgBinding.from_step("s1", "nope").resolve({}, {"s1": {"k": 1}})


def test_binding_unknown_kind():
    with pytest.raises(ValidationError):
        ArgBinding("weird").resolve({}, {})


# -------------------- compose: static validation -------------------- #
def test_compose_infers_metadata():
    smith, _ = _smith()
    comp = smith.compose("p", params=[ToolParam("url", "str")], steps=_pipeline(),
                         output_step="s3")
    assert comp.capability is Capability.NET_OUT  # most-privileged step (LOW tie -> fetch)
    assert comp.risk is RiskTier.LOW
    assert comp.reversible is True
    assert not comp.owner_exclusive


def test_compose_requires_steps():
    smith, _ = _smith()
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=[], output_step="x")


def test_compose_duplicate_name():
    smith, reg = _smith()
    with pytest.raises(ToolError):
        smith.compose("fetch", params=[ToolParam("url", "str")], steps=_pipeline(),
                      output_step="s3")


def test_compose_unknown_tool():
    smith, _ = _smith()
    with pytest.raises(ToolError):
        smith.compose("p", params=[], steps=[ToolStep("s1", "nope", {})], output_step="s1")


def test_compose_duplicate_step_id():
    smith, _ = _smith()
    steps = [ToolStep("s1", "shout", {"text": ArgBinding.literal("a")}),
             ToolStep("s1", "shout", {"text": ArgBinding.literal("b")})]
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=steps, output_step="s1")


def test_compose_unknown_arg_binding():
    smith, _ = _smith()
    steps = [ToolStep("s1", "shout", {"nope": ArgBinding.literal("a")})]
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=steps, output_step="s1")


def test_compose_unknown_input_ref():
    smith, _ = _smith()
    steps = [ToolStep("s1", "shout", {"text": ArgBinding.inp("ghost")})]
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=steps, output_step="s1")


def test_compose_forward_step_ref():
    smith, _ = _smith()
    steps = [ToolStep("s1", "shout", {"text": ArgBinding.from_step("s2")}),
             ToolStep("s2", "shout", {"text": ArgBinding.literal("x")})]
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=steps, output_step="s1")


def test_compose_unbound_required_arg():
    smith, _ = _smith()
    steps = [ToolStep("s1", "get_field", {"data": ArgBinding.literal({})})]  # missing key
    with pytest.raises(ValidationError):
        smith.compose("p", params=[], steps=steps, output_step="s1")


def test_compose_bad_output_step():
    smith, _ = _smith()
    with pytest.raises(ValidationError):
        smith.compose("p", params=[ToolParam("url", "str")], steps=_pipeline(),
                      output_step="s99")


# -------------------- privilege inference -------------------- #
def test_compose_inherits_highest_risk():
    smith, reg = _smith()
    reg.register(ToolSpec("rm", handler=lambda path: "gone",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    comp = smith.compose("p", params=[ToolParam("path", "str")],
                         steps=[
                             ToolStep("s1", "fetch", {"url": ArgBinding.literal("u")}),
                             ToolStep("s2", "rm", {"path": ArgBinding.inp("path")}),
                         ], output_step="s2")
    assert comp.risk is RiskTier.HIGH
    assert comp.capability is Capability.FS_DELETE
    assert not comp.reversible


def test_compose_owner_exclusive_detected():
    smith, reg = _smith()
    reg.register(ToolSpec("edit", handler=lambda: "x",
                          capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL))
    comp = smith.compose("p", params=[], steps=[ToolStep("s1", "edit", {})],
                         output_step="s1")
    assert comp.owner_exclusive
    assert comp.capability is Capability.MODIFY_RULES
    assert comp.risk is RiskTier.CRITICAL


def test_compose_cost_summed():
    smith, reg = _smith()
    reg.register(ToolSpec("a", handler=lambda: 1, cost=0.5))
    reg.register(ToolSpec("b", handler=lambda: 2, cost=0.3))
    comp = smith.compose("p", params=[], steps=[ToolStep("s1", "a", {}),
                                                ToolStep("s2", "b", {})],
                         output_step="s2")
    assert abs(comp.cost - 0.8) < 1e-9


# -------------------- validate -------------------- #
def test_validate_runs_pipeline():
    smith, _ = _smith()
    comp = smith.compose("p", params=[ToolParam("url", "str")], steps=_pipeline(),
                         output_step="s3")
    report = smith.validate(comp, {"url": "http://x"})
    assert isinstance(report, ValidationReport)
    assert report.ok and report.output == "HELLO"
    assert report.steps_run == 3


def test_validate_with_output_key():
    smith, _ = _smith()
    comp = smith.compose("p", params=[ToolParam("url", "str")],
                         steps=[ToolStep("s1", "fetch", {"url": ArgBinding.inp("url")})],
                         output_step="s1", output_key="status")
    report = smith.validate(comp, {"url": "http://x"})
    assert report.ok and report.output == 200


def test_validate_catches_runtime_error():
    smith, reg = _smith()

    @reg.tool("boom", params=[])
    def boom():
        raise RuntimeError("nope")

    comp = smith.compose("p", params=[], steps=[ToolStep("s1", "boom", {})],
                         output_step="s1")
    report = smith.validate(comp, {})
    assert not report.ok and "RuntimeError" in (report.error or "")


def test_validate_collects_sandbox_effects():
    smith, reg = _smith()
    reg.register(ToolSpec("w", handler=lambda path: "real",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_WRITE, target_param="path",
                          dry_run=lambda ctx, path: ctx.write_file(path, "x") or "prev"))
    comp = smith.compose("p", params=[ToolParam("path", "str")],
                         steps=[ToolStep("s1", "w", {"path": ArgBinding.inp("path")})],
                         output_step="s1")
    report = smith.validate(comp, {"path": "/tmp/a"})
    assert report.ok and len(report.effects) == 1


# -------------------- install -------------------- #
def test_install_registers_callable_tool():
    smith, reg = _smith()
    comp = smith.compose("pipe", params=[ToolParam("url", "str")], steps=_pipeline(),
                         output_step="s3")
    smith.install(comp)
    assert "pipe" in reg.names()
    r = reg.invoke("pipe", {"url": "http://x"})
    assert r.ok and r.value == "HELLO"


def test_install_validates_when_sample_given():
    smith, reg = _smith()

    @reg.tool("boom", params=[])
    def boom():
        raise RuntimeError("nope")

    comp = smith.compose("p", params=[], steps=[ToolStep("s1", "boom", {})],
                         output_step="s1")
    with pytest.raises(ToolError):
        smith.install(comp, sample_args={})


def test_install_owner_exclusive_refused_autonomously():
    smith, reg = _smith()
    reg.register(ToolSpec("edit", handler=lambda: "x",
                          capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL))
    comp = smith.compose("backdoor", params=[], steps=[ToolStep("s1", "edit", {})],
                         output_step="s1")
    with pytest.raises(AuthorizationError):
        smith.install(comp, authority=Authority.AUTONOMOUS)


def test_install_owner_exclusive_allowed_for_owner():
    smith, reg = _smith()
    reg.register(ToolSpec("edit", handler=lambda: "edited",
                          capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL))
    comp = smith.compose("backdoor", params=[], steps=[ToolStep("s1", "edit", {})],
                         output_step="s1")
    spec = smith.install(comp, authority=Authority.OWNER)
    assert spec.name == "backdoor"


def test_installed_composite_escalates_when_dangerous():
    smith, reg = _smith()
    reg.register(ToolSpec("rm", handler=lambda path: "gone",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    comp = smith.compose("nuke", params=[ToolParam("path", "str")],
                         steps=[ToolStep("s1", "rm", {"path": ArgBinding.inp("path")})],
                         output_step="s1")
    smith.install(comp, target_param="path")
    r = reg.invoke("nuke", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner


# -------------------- serialization -------------------- #
def test_composite_to_dict():
    smith, _ = _smith()
    comp = smith.compose("p", params=[ToolParam("url", "str")], steps=_pipeline(),
                         output_step="s3", description="d")
    d = comp.to_dict()
    assert d["name"] == "p" and len(d["steps"]) == 3 and d["capability"] == "net.out"
