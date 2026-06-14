"""Tests for nyxara.agency.tools."""

from __future__ import annotations

import pytest

from nyxara.agency.governor import Governor
from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolResult, ToolSpec
from nyxara.kernel.config import ResourceLimits
from nyxara.kernel.errors import ToolError, ValidationError


def _registry(**lim):
    base = dict(max_tool_calls_per_min=100, daily_spend_budget=100.0,
                max_concurrent_tasks=4)
    base.update(lim)
    return ToolRegistry(governor=Governor(ResourceLimits(**base)))


# -------------------- ToolParam coercion -------------------- #
def test_param_str_ok_and_bad():
    p = ToolParam("x", "str")
    assert p.coerce("a") == "a"
    with pytest.raises(ValidationError):
        p.coerce(1)


def test_param_int_rejects_bool():
    p = ToolParam("x", "int")
    assert p.coerce(5) == 5
    with pytest.raises(ValidationError):
        p.coerce(True)


def test_param_float_accepts_int():
    p = ToolParam("x", "float")
    assert p.coerce(3) == 3.0
    assert p.coerce(2.5) == 2.5


def test_param_float_rejects_bool():
    with pytest.raises(ValidationError):
        ToolParam("x", "float").coerce(True)


def test_param_bool():
    p = ToolParam("x", "bool")
    assert p.coerce(True) is True
    with pytest.raises(ValidationError):
        p.coerce(1)


def test_param_list_and_dict():
    assert ToolParam("x", "list").coerce([1, 2]) == [1, 2]
    assert ToolParam("x", "dict").coerce({"a": 1}) == {"a": 1}


def test_param_any_passthrough():
    assert ToolParam("x", "any").coerce(object) is object


def test_param_unknown_type():
    with pytest.raises(ValidationError):
        ToolParam("x", "frobnicate").coerce("v")


# -------------------- ToolSpec.validate_args -------------------- #
def test_validate_args_fills_defaults():
    spec = ToolSpec("t", handler=lambda a, b: (a, b),
                    params=[ToolParam("a", "str"), ToolParam("b", "int", required=False, default=7)])
    out = spec.validate_args({"a": "x"})
    assert out == {"a": "x", "b": 7}


def test_validate_args_missing_required():
    spec = ToolSpec("t", handler=lambda a: a, params=[ToolParam("a", "str")])
    with pytest.raises(ValidationError):
        spec.validate_args({})


def test_validate_args_unknown_arg():
    spec = ToolSpec("t", handler=lambda a: a, params=[ToolParam("a", "str")])
    with pytest.raises(ValidationError):
        spec.validate_args({"a": "x", "z": 1})


def test_spec_schema():
    spec = ToolSpec("t", handler=lambda a: a, params=[ToolParam("a", "str")],
                    capability=Capability.FS_READ, risk=RiskTier.MODERATE)
    s = spec.schema()
    assert s["name"] == "t" and s["capability"] == "fs.read" and s["risk"] == "moderate"


# -------------------- registration -------------------- #
def test_register_and_get():
    reg = _registry()
    spec = reg.register(ToolSpec("t", handler=lambda: 1))
    assert reg.get("t") is spec
    assert "t" in reg.names()


def test_register_duplicate_refused():
    reg = _registry()
    reg.register(ToolSpec("t", handler=lambda: 1))
    with pytest.raises(ToolError):
        reg.register(ToolSpec("t", handler=lambda: 2))


def test_tool_decorator():
    reg = _registry()

    @reg.tool("greet", params=[ToolParam("name", "str")])
    def greet(name: str) -> str:
        return f"hi {name}"

    r = reg.invoke("greet", {"name": "JP"})
    assert r.ok and r.value == "hi JP"


def test_schemas_listing():
    reg = _registry()
    reg.register(ToolSpec("a", handler=lambda: 1))
    reg.register(ToolSpec("b", handler=lambda: 2))
    assert [s["name"] for s in reg.schemas()] == ["a", "b"]


# -------------------- invoke: happy path -------------------- #
def test_invoke_basic():
    reg = _registry()
    reg.register(ToolSpec("up", handler=lambda text: text.upper(),
                          params=[ToolParam("text", "str")]))
    r = reg.invoke("up", {"text": "hi"})
    assert isinstance(r, ToolResult)
    assert r.ok and r.value == "HI"
    assert r.duration_s >= 0.0


def test_invoke_unknown_tool():
    reg = _registry()
    with pytest.raises(ToolError):
        reg.invoke("nope")


def test_invoke_validation_error_propagates():
    reg = _registry()
    reg.register(ToolSpec("t", handler=lambda a: a, params=[ToolParam("a", "int")]))
    with pytest.raises(ValidationError):
        reg.invoke("t", {"a": "not-int"})


def test_invoke_handler_exception_is_data():
    reg = _registry()

    def boom():
        raise RuntimeError("kaboom")

    reg.register(ToolSpec("boom", handler=boom))
    r = reg.invoke("boom")
    assert not r.ok and "RuntimeError" in r.error


# -------------------- invoke: authorization -------------------- #
def test_invoke_denied_owner_exclusive():
    reg = _registry()
    reg.register(ToolSpec("editrules", handler=lambda: "done",
                          capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL))
    r = reg.invoke("editrules", authority=Authority.AUTONOMOUS)
    assert not r.ok and not r.requires_owner  # hard denied (Rule 8)
    assert r.decision["rule_basis"] == "Rule 8"


def test_invoke_escalates_irreversible():
    reg = _registry()
    reg.register(ToolSpec("del", handler=lambda path: "gone",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    r = reg.invoke("del", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner


def test_invoke_owner_confirmed_runs_escalated():
    reg = _registry()
    reg.register(ToolSpec("del", handler=lambda path: f"gone:{path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    r = reg.invoke("del", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS,
                   owner_confirmed=True)
    assert r.ok and r.value == "gone:/etc/x"


def test_invoke_owner_authority_runs():
    reg = _registry()
    reg.register(ToolSpec("del", handler=lambda path: "gone",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    r = reg.invoke("del", {"path": "/x"}, authority=Authority.OWNER)
    assert r.ok


def test_invoke_target_scope_respected_by_grant():
    reg = _registry()
    reg.policy.grant("tmp", Capability.FS_DELETE, max_risk=RiskTier.HIGH,
                     scopes=["/tmp/*"], reversible_only=False)
    reg.register(ToolSpec("del", handler=lambda path: f"gone:{path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    inside = reg.invoke("del", {"path": "/tmp/cache"}, authority=Authority.AUTONOMOUS)
    outside = reg.invoke("del", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS)
    assert inside.ok
    assert not outside.ok and outside.requires_owner


# -------------------- invoke: governance -------------------- #
def test_invoke_rate_limited():
    reg = _registry(max_tool_calls_per_min=2)
    reg.register(ToolSpec("ping", handler=lambda: "pong"))
    assert reg.invoke("ping").ok
    assert reg.invoke("ping").ok
    r = reg.invoke("ping")  # 3rd exceeds bucket of 2
    assert not r.ok and "rate" in r.error
    assert r.retry_after > 0


def test_invoke_spend_budget():
    reg = _registry(daily_spend_budget=1.0)
    reg.register(ToolSpec("pricey", handler=lambda: "ok", cost=0.6))
    assert reg.invoke("pricey").ok
    r = reg.invoke("pricey")  # would exceed 1.0
    assert not r.ok and "budget" in r.error


# -------------------- invoke: sandbox -------------------- #
def test_invoke_dry_run_does_not_execute():
    reg = _registry()
    calls = {"n": 0}

    def handler(path):
        calls["n"] += 1
        return "real"

    def dry(ctx, path):
        ctx.write_file(path, "x")
        return "preview"

    reg.register(ToolSpec("w", handler=handler, params=[ToolParam("path", "str")],
                          capability=Capability.FS_WRITE, target_param="path",
                          sandbox_first=True, dry_run=dry))
    r = reg.invoke("w", {"path": "/tmp/a"}, dry_run=True)
    assert r.simulated and r.value == "preview"
    assert len(r.effects) == 1
    assert calls["n"] == 0  # real handler never ran


def test_invoke_dry_run_no_budget_spent():
    reg = _registry(max_tool_calls_per_min=1, daily_spend_budget=0.0)
    reg.register(ToolSpec("w", handler=lambda: "real", cost=5.0,
                          capability=Capability.FS_WRITE,
                          dry_run=lambda ctx: "preview"))
    # dry-run must not be blocked by an empty budget or consume rate
    r = reg.invoke("w", dry_run=True)
    assert r.simulated and r.value == "preview"
    # the single rate token is still available for a real call
    assert reg.governor.allow("tool").allowed


def test_invoke_sandbox_first_then_real():
    reg = _registry()

    def dry(ctx, path):
        ctx.write_file(path, "x")
        return "preview"

    reg.register(ToolSpec("w", handler=lambda path: f"wrote:{path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_WRITE, target_param="path",
                          sandbox_first=True, dry_run=dry))
    r = reg.invoke("w", {"path": "/tmp/a"})
    assert r.ok and r.value == "wrote:/tmp/a"
    assert len(r.effects) == 1  # rehearsed before the real write


def test_invoke_dry_run_without_dry_handler():
    reg = _registry()
    reg.register(ToolSpec("noop", handler=lambda: "real"))
    r = reg.invoke("noop", dry_run=True)
    assert r.simulated and r.value is None


# -------------------- result shape -------------------- #
def test_result_to_dict():
    reg = _registry()
    reg.register(ToolSpec("t", handler=lambda: 1))
    d = reg.invoke("t").to_dict()
    assert d["ok"] is True and "decision" in d and "effects" in d
