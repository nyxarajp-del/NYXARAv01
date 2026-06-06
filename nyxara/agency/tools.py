"""NYXARA · agency/tools.py — the governed tool registry & execution spine (⚙, fail-closed).

A mind that can only think is harmless and useless in equal measure. Tools are how
NYXARA *reaches into the world* — read a file, fetch a URL, query a service, run a
check. This module is the spine that makes that reach **safe by construction**: every
tool is typed, capability-bound, governed, and (optionally) rehearsed in the sandbox
before it ever touches reality.

A :class:`ToolSpec` declares a tool's typed parameters, the single :class:`Capability`
it needs (least privilege), its risk tier, reversibility, spend cost and rate-limit
class. :class:`ToolRegistry.invoke` runs the **full safety pipeline** on every call:

1. **Validate & coerce** arguments against the typed schema (reject on mismatch).
2. **Authorise** through :class:`~nyxara.agency.permissions.PermissionPolicy` — denied
   calls never run; escalated calls run only with the Master's explicit confirmation.
3. **Govern** through :class:`~nyxara.agency.governor.Governor` — rate-limit the call
   and charge its cost against the spend budget; throttle rather than overrun.
4. **Rehearse** in :class:`~nyxara.sim.sandbox.Sandbox` first (when configured) so the
   intended effects — especially irreversible ones — are seen before they are taken.
5. **Execute** the real handler under a wall-clock deadline, capturing the result,
   timing, and any error as data.

The result is a uniform :class:`ToolResult` — NYXARA always knows whether a call ran,
was simulated, was throttled, was refused, or needs the Master.

Depends on :mod:`agency.permissions`, :mod:`agency.governor`, :mod:`sim.sandbox`,
:mod:`kernel.errors`. Pure standard library otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from nyxara.agency.governor import Governor
from nyxara.agency.permissions import (Authority, Capability, PermissionDecision,
                                       PermissionPolicy, PermissionRequest, RiskTier,
                                       build_default_policy)
from nyxara.kernel.errors import ToolError, ValidationError
from nyxara.sim.sandbox import Sandbox, SandboxContext

__all__ = [
    "ToolParam",
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
]


# --------------------------------------------------------------------------- #
# Typed parameters
# --------------------------------------------------------------------------- #
_TYPE_TOKENS: Dict[str, Tuple[type, ...]] = {
    "str": (str,), "int": (int,), "float": (float, int), "bool": (bool,),
    "list": (list, tuple), "dict": (dict,), "any": (object,),
}


@dataclass
class ToolParam:
    name: str
    type: str = "str"                      # one of _TYPE_TOKENS
    required: bool = True
    default: Any = None
    description: str = ""

    def coerce(self, value: Any) -> Any:
        """Validate/coerce a value to this parameter's type (raise on mismatch)."""
        token = self.type
        allowed = _TYPE_TOKENS.get(token)
        if allowed is None:
            raise ValidationError(f"unknown parameter type {token!r}",
                                  context={"param": self.name})
        if token == "any":
            return value
        # bool must be checked before int (bool is a subclass of int)
        if token == "bool":
            if isinstance(value, bool):
                return value
            raise ValidationError(f"parameter {self.name!r} must be bool",
                                  context={"got": type(value).__name__})
        if token == "float":
            if isinstance(value, bool):
                raise ValidationError(f"parameter {self.name!r} must be a number")
            if isinstance(value, (int, float)):
                return float(value)
            raise ValidationError(f"parameter {self.name!r} must be a number",
                                  context={"got": type(value).__name__})
        if token == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"parameter {self.name!r} must be int",
                                      context={"got": type(value).__name__})
            return value
        if isinstance(value, allowed):
            return value
        raise ValidationError(f"parameter {self.name!r} must be {token}",
                              context={"got": type(value).__name__})

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type, "required": self.required,
                "default": self.default, "description": self.description}


# --------------------------------------------------------------------------- #
# Tool specification
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    handler: Callable[..., Any]
    description: str = ""
    params: List[ToolParam] = field(default_factory=list)
    capability: Capability = Capability.TOOL_CALL
    risk: RiskTier = RiskTier.LOW
    reversible: bool = True
    cost: float = 0.0                      # spend units charged on use
    rate_resource: str = "tool"            # governor bucket
    target_param: Optional[str] = None     # which arg names the target (for scope checks)
    sandbox_first: bool = False
    dry_run: Optional[Callable[[SandboxContext], Any]] = None  # (ctx, **args) -> Any
    timeout_s: Optional[float] = None

    def validate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce/fill arguments against the declared schema."""
        declared = {p.name: p for p in self.params}
        unknown = set(args) - set(declared)
        if unknown:
            raise ValidationError(f"unknown arguments for tool {self.name!r}",
                                  context={"unknown": sorted(unknown)})
        out: Dict[str, Any] = {}
        for p in self.params:
            if p.name in args and args[p.name] is not None:
                out[p.name] = p.coerce(args[p.name])
            elif p.required:
                raise ValidationError(f"missing required argument {p.name!r}",
                                      context={"tool": self.name})
            else:
                out[p.name] = p.default
        return out

    def schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "capability": self.capability.value, "risk": self.risk.label,
                "reversible": self.reversible, "cost": self.cost,
                "params": [p.to_dict() for p in self.params]}


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    tool: str
    ok: bool
    value: Any = None
    error: Optional[str] = None
    simulated: bool = False
    requires_owner: bool = False
    retry_after: float = 0.0
    duration_s: float = 0.0
    timed_out: bool = False
    effects: List[Dict[str, Any]] = field(default_factory=list)
    decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "value": self.value,
                "error": self.error, "simulated": self.simulated,
                "requires_owner": self.requires_owner,
                "retry_after": round(self.retry_after, 4),
                "duration_s": round(self.duration_s, 5), "timed_out": self.timed_out,
                "effects": self.effects, "decision": self.decision}


# --------------------------------------------------------------------------- #
# Registry / executor
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """A registry of typed tools, executed through the full safety pipeline."""

    def __init__(self, *, policy: Optional[PermissionPolicy] = None,
                 governor: Optional[Governor] = None,
                 sandbox: Optional[Sandbox] = None) -> None:
        self.policy = policy or build_default_policy()
        self.governor = governor or Governor()
        self.sandbox = sandbox or Sandbox()
        self._tools: Dict[str, ToolSpec] = {}

    # ---- registration ---- #
    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ToolError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec
        return spec

    def tool(self, name: str, **spec_kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register a function as a tool."""
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(ToolSpec(name=name, handler=fn,
                                   description=spec_kw.pop("description", fn.__doc__ or ""),
                                   **spec_kw))
            return fn
        return deco

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[n].schema() for n in self.names()]

    # ---- the safety pipeline ---- #
    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None, *,
               authority: Authority = Authority.AUTONOMOUS,
               owner_confirmed: bool = False, dry_run: bool = False) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"no such tool {name!r}", context={"known": self.names()})

        # 1. validate & coerce arguments
        clean = spec.validate_args(args or {})

        # 2. authorise
        target = str(clean.get(spec.target_param, "")) if spec.target_param else ""
        decision = self.policy.check(PermissionRequest(
            capability=spec.capability, target=target, risk=spec.risk,
            reversible=spec.reversible, authority=authority,
            reason=f"tool:{name}"))
        ddict = decision.to_dict()
        if decision.denied:
            return ToolResult(name, ok=False, error=f"denied: {decision.reason}",
                              decision=ddict)
        if decision.escalated and not owner_confirmed:
            return ToolResult(name, ok=False, requires_owner=True,
                              error=f"requires the Master: {decision.reason}",
                              decision=ddict)

        # 3. pure dry-run: rehearse only — never touch reality, never burn real budget
        effects: List[Dict[str, Any]] = []
        if dry_run:
            if spec.dry_run is not None:
                sim = self.sandbox.run(lambda ctx: spec.dry_run(ctx, **clean),
                                       rollback_after=True)
                return ToolResult(name, ok=sim.success, value=sim.value,
                                  simulated=True, error=sim.error,
                                  effects=[e.to_dict() for e in sim.effects],
                                  decision=ddict)
            return ToolResult(name, ok=True, simulated=True, value=None,
                              effects=effects, decision=ddict)

        # 4. govern — rate then spend
        rate = self.governor.allow(spec.rate_resource)
        if not rate.allowed:
            return ToolResult(name, ok=False, error="rate limited",
                              retry_after=rate.retry_after, decision=ddict)
        if spec.cost > 0:
            spend = self.governor.charge(spec.cost)
            if not spend.allowed:
                return ToolResult(name, ok=False, error="spend budget exhausted",
                                  decision=ddict)

        # 5. rehearse in the sandbox (preview effects before reality)
        if spec.sandbox_first and spec.dry_run is not None:
            sim = self.sandbox.run(lambda ctx: spec.dry_run(ctx, **clean),
                                   rollback_after=True)
            effects = [e.to_dict() for e in sim.effects]

        # 6. execute for real, under a wall-clock deadline
        dl = self.governor.deadline(spec.timeout_s, label=f"tool:{name}")
        start = time.monotonic()
        try:
            value = spec.handler(**clean)
        except Exception as exc:  # noqa: BLE001 — failures are returned as data
            return ToolResult(name, ok=False, error=f"{type(exc).__name__}: {exc}",
                              duration_s=time.monotonic() - start, effects=effects,
                              decision=ddict)
        duration = time.monotonic() - start
        return ToolResult(name, ok=True, value=value, duration_s=duration,
                          timed_out=dl.expired, effects=effects, decision=ddict)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.governor import Governor
    from nyxara.kernel.config import ResourceLimits

    print("=" * 70)
    print("NYXARA tool-registry self-test")
    print("=" * 70)

    reg = ToolRegistry(governor=Governor(ResourceLimits(max_tool_calls_per_min=3,
                                                        daily_spend_budget=5.0)))

    # a simple, reversible, low-risk read tool
    @reg.tool("echo", params=[ToolParam("text", "str")], description="echo text back")
    def _echo(text: str) -> str:
        return text.upper()

    r = reg.invoke("echo", {"text": "hello"})
    print(f"\necho                : ok={r.ok} value={r.value!r}")
    assert r.ok and r.value == "HELLO"

    # argument validation
    bad = None
    try:
        reg.invoke("echo", {"text": 123})
    except ValidationError as e:
        bad = e
    print(f"type validation     : rejected bad arg ✓ ({bad.message})")
    assert bad is not None

    # a HIGH-risk irreversible tool invoked autonomously -> escalates to the Master
    reg.register(ToolSpec("delete", handler=lambda path: f"deleted {path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    r = reg.invoke("delete", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS)
    print(f"\nautonomous delete   : ok={r.ok} requires_owner={r.requires_owner}")
    assert not r.ok and r.requires_owner

    # the Master confirms -> it runs
    r = reg.invoke("delete", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS,
                   owner_confirmed=True)
    assert r.ok and r.value == "deleted /etc/x"
    print(f"owner-confirmed     : ok={r.ok} value={r.value!r}")

    # a sandbox-first tool: rehearse effects before committing
    def _write_dry(ctx, path, content):
        ctx.write_file(path, content)
        return "previewed"
    reg.register(ToolSpec("write", handler=lambda path, content: f"wrote {len(content)}b",
                          params=[ToolParam("path", "str"), ToolParam("content", "str")],
                          capability=Capability.FS_WRITE, risk=RiskTier.LOW,
                          target_param="path", sandbox_first=True, dry_run=_write_dry))
    r = reg.invoke("write", {"path": "/tmp/a", "content": "hi"})
    print(f"\nsandbox-first write : ok={r.ok} previewed_effects={len(r.effects)} value={r.value!r}")
    assert r.ok and len(r.effects) == 1   # the write was rehearsed first

    # pure dry-run: never touches reality
    r = reg.invoke("write", {"path": "/tmp/a", "content": "hi"}, dry_run=True)
    assert r.simulated and r.value == "previewed"
    print(f"dry-run only        : simulated={r.simulated} ✓")

    # rate limiting: drain the 3/min tool bucket (fresh budget for a clean demo)
    reg.governor.reset()
    reg.invoke("echo", {"text": "a"}); reg.invoke("echo", {"text": "b"})
    reg.invoke("echo", {"text": "c"})
    drained = reg.invoke("echo", {"text": "d"})  # 4th call exceeds the 3/min bucket
    print(f"\nrate limit          : ok={drained.ok} error={drained.error!r} "
          f"retry_after={drained.retry_after:.1f}s")
    assert not drained.ok and "rate" in (drained.error or "")

    print(f"\nschemas             : {reg.names()}")
    print("\nALL SELF-TESTS PASSED ✓")
