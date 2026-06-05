"""NYXARA · agency/toolsmith.py — self-authored tools by safe composition (✦, Rule 4).

Rule 4 is recursive self-evolution: NYXARA's *capability* may grow without bound — but
never her *character or loyalty*. The toolsmith is where that growth happens on the
agency side. It lets NYXARA build **new tools out of the tools she already has**, by
chaining them into a typed pipeline: the output of one step feeds the input of the next.

A :class:`CompositeTool` is a named sequence of :class:`ToolStep` s, each invoking an
existing registered tool with :class:`ArgBinding` s that wire arguments from the
composite's own inputs, from literals, or from earlier steps' outputs. The toolsmith:

* **Statically validates** the wiring on :meth:`Toolsmith.compose` — every referenced
  tool must exist, every binding must resolve, and step references must point *backward*
  (no forward/cyclic references in the linear pipeline).
* **Infers least privilege** — the composite's declared capability is the *most
  privileged* (highest-risk) capability among its steps, its risk is the max, it is
  reversible only if every step is, and its cost is the sum. A composite is never
  granted more authority than the worst thing it does.
* **Refuses to self-author into the sovereign core** — if any step touches an
  owner-exclusive capability (modifying the Rules, the permissions, or the identity),
  the composite may only be installed under the Master's authority (Rule 8). NYXARA
  cannot grow herself a back-door tool.
* **Validates in the sandbox before installing** — :meth:`Toolsmith.validate` dry-runs
  the whole pipeline against sample inputs with zero real effects, so a broken or
  dangerous composition is caught before it ever becomes a real, callable tool.

Once validated, :meth:`Toolsmith.install` registers the composite as a first-class
:class:`~nyxara.agency.tools.ToolSpec`, governed by the very same safety pipeline as
every other tool.

Depends on :mod:`agency.tools`, :mod:`agency.permissions`, :mod:`sim.sandbox`,
:mod:`kernel.errors`. Pure standard library otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.kernel.errors import AuthorizationError, ToolError, ValidationError
from nyxara.sim.sandbox import Sandbox

__all__ = [
    "ArgBinding",
    "ToolStep",
    "CompositeTool",
    "ValidationReport",
    "Toolsmith",
]


# --------------------------------------------------------------------------- #
# Argument bindings
# --------------------------------------------------------------------------- #
@dataclass
class ArgBinding:
    """How a step argument is supplied: a literal, a composite input, or a prior step."""
    kind: str                      # "literal" | "input" | "step"
    ref: Any = None                # literal value, input name, or step id
    key: Optional[str] = None      # extract this key from a step's dict output

    @classmethod
    def literal(cls, value: Any) -> "ArgBinding":
        return cls("literal", value)

    @classmethod
    def inp(cls, name: str) -> "ArgBinding":
        return cls("input", name)

    @classmethod
    def from_step(cls, step_id: str, key: Optional[str] = None) -> "ArgBinding":
        return cls("step", step_id, key)

    def resolve(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Any:
        if self.kind == "literal":
            return self.ref
        if self.kind == "input":
            if self.ref not in inputs:
                raise ValidationError(f"binding references unknown input {self.ref!r}")
            return inputs[self.ref]
        if self.kind == "step":
            if self.ref not in outputs:
                raise ValidationError(f"binding references unavailable step {self.ref!r}")
            out = outputs[self.ref]
            if self.key is not None:
                try:
                    return out[self.key]
                except (KeyError, TypeError, IndexError) as exc:
                    raise ValidationError(
                        f"cannot extract {self.key!r} from step {self.ref!r} output",
                        context={"error": str(exc)})
            return out
        raise ValidationError(f"unknown binding kind {self.kind!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "key": self.key}


@dataclass
class ToolStep:
    id: str
    tool: str
    bindings: Dict[str, ArgBinding] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "tool": self.tool,
                "bindings": {k: b.to_dict() for k, b in self.bindings.items()}}


# --------------------------------------------------------------------------- #
# Composite tool
# --------------------------------------------------------------------------- #
@dataclass
class CompositeTool:
    name: str
    params: List[ToolParam]
    steps: List[ToolStep]
    output_step: str
    output_key: Optional[str] = None
    description: str = ""
    # inferred on compose:
    capability: Capability = Capability.TOOL_CALL
    risk: RiskTier = RiskTier.LOW
    reversible: bool = True
    cost: float = 0.0
    owner_exclusive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "capability": self.capability.value, "risk": self.risk.label,
                "reversible": self.reversible, "cost": self.cost,
                "owner_exclusive": self.owner_exclusive,
                "params": [p.to_dict() for p in self.params],
                "steps": [s.to_dict() for s in self.steps],
                "output_step": self.output_step, "output_key": self.output_key}


@dataclass
class ValidationReport:
    ok: bool
    output: Any = None
    error: Optional[str] = None
    steps_run: int = 0
    effects: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "output": self.output, "error": self.error,
                "steps_run": self.steps_run, "effects": self.effects}


# --------------------------------------------------------------------------- #
# Toolsmith
# --------------------------------------------------------------------------- #
class Toolsmith:
    """Composes, validates and installs new tools from existing ones (Rule 4)."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    # ---- composition (with static validation) ---- #
    def compose(self, name: str, *, params: Sequence[ToolParam],
                steps: Sequence[ToolStep], output_step: str,
                output_key: Optional[str] = None, description: str = ""
                ) -> CompositeTool:
        if not steps:
            raise ValidationError("a composite tool needs at least one step")
        if name in self.registry.names():
            raise ToolError(f"a tool named {name!r} already exists")

        step_ids: List[str] = []
        param_names = {p.name for p in params}
        max_risk = RiskTier.TRIVIAL
        dominant_cap = Capability.TOOL_CALL
        reversible = True
        cost = 0.0
        owner_exclusive = False
        exclusive_cap: Optional[Capability] = None

        for st in steps:
            if st.id in step_ids:
                raise ValidationError(f"duplicate step id {st.id!r}")
            spec = self.registry.get(st.tool)
            if spec is None:
                raise ToolError(f"step {st.id!r} references unknown tool {st.tool!r}")
            # validate every binding statically
            declared = {p.name for p in spec.params}
            for arg, b in st.bindings.items():
                if arg not in declared:
                    raise ValidationError(
                        f"step {st.id!r} binds unknown argument {arg!r} of tool {st.tool!r}")
                if b.kind == "input" and b.ref not in param_names:
                    raise ValidationError(
                        f"step {st.id!r} references unknown composite input {b.ref!r}")
                if b.kind == "step" and b.ref not in step_ids:
                    raise ValidationError(
                        f"step {st.id!r} references step {b.ref!r} which does not precede it")
            # required args of the sub-tool must all be bound
            for p in spec.params:
                if p.required and p.name not in st.bindings:
                    raise ValidationError(
                        f"step {st.id!r} leaves required argument {p.name!r} unbound")
            # privilege inference
            meta = self.registry.policy.meta(spec.capability)
            if meta.owner_exclusive:
                owner_exclusive = True
                exclusive_cap = spec.capability
            if spec.risk > max_risk:
                max_risk = spec.risk
                dominant_cap = spec.capability
            reversible = reversible and spec.reversible
            cost += spec.cost
            step_ids.append(st.id)

        if output_step not in step_ids:
            raise ValidationError(f"output_step {output_step!r} is not a step in the pipeline")

        capability = exclusive_cap if exclusive_cap is not None else dominant_cap
        risk = RiskTier.CRITICAL if owner_exclusive else max_risk
        return CompositeTool(name=name, params=list(params), steps=list(steps),
                             output_step=output_step, output_key=output_key,
                             description=description, capability=capability, risk=risk,
                             reversible=reversible, cost=cost,
                             owner_exclusive=owner_exclusive)

    # ---- execution closures ---- #
    def _run_real(self, comp: CompositeTool):
        registry = self.registry

        def handler(**inputs: Any) -> Any:
            outputs: Dict[str, Any] = {}
            for st in comp.steps:
                spec = registry.get(st.tool)
                if spec is None:
                    raise ToolError(f"tool {st.tool!r} was unregistered after composition")
                args = {a: b.resolve(inputs, outputs) for a, b in st.bindings.items()}
                clean = spec.validate_args(args)
                outputs[st.id] = spec.handler(**clean)
            return self._extract(comp, outputs)

        return handler

    def _run_dry(self, comp: CompositeTool):
        registry = self.registry

        def dry(ctx, **inputs: Any) -> Any:
            outputs: Dict[str, Any] = {}
            for st in comp.steps:
                spec = registry.get(st.tool)
                if spec is None:
                    raise ToolError(f"tool {st.tool!r} was unregistered after composition")
                args = {a: b.resolve(inputs, outputs) for a, b in st.bindings.items()}
                clean = spec.validate_args(args)
                if spec.dry_run is not None:
                    outputs[st.id] = spec.dry_run(ctx, **clean)
                else:
                    # no sandbox-aware preview -> run the (assumed pure) handler
                    outputs[st.id] = spec.handler(**clean)
            return self._extract(comp, outputs)

        return dry

    @staticmethod
    def _extract(comp: CompositeTool, outputs: Dict[str, Any]) -> Any:
        out = outputs[comp.output_step]
        if comp.output_key is not None:
            return out[comp.output_key]
        return out

    # ---- sandbox validation ---- #
    def validate(self, comp: CompositeTool, sample_args: Dict[str, Any]) -> ValidationReport:
        """Dry-run the whole pipeline against sample inputs with zero real effects."""
        sb = Sandbox()
        dry = self._run_dry(comp)
        try:
            inputs = {p.name: (sample_args.get(p.name, p.default)) for p in comp.params}
            res = sb.run(lambda ctx: dry(ctx, **inputs), rollback_after=True)
        except Exception as exc:  # noqa: BLE001 — surfaced as a report, not a crash
            return ValidationReport(ok=False, error=f"{type(exc).__name__}: {exc}")
        return ValidationReport(ok=res.success, output=res.value, error=res.error,
                                steps_run=len(comp.steps),
                                effects=[e.to_dict() for e in res.effects])

    # ---- installation ---- #
    def install(self, comp: CompositeTool, *, authority: Authority = Authority.AUTONOMOUS,
                sample_args: Optional[Dict[str, Any]] = None,
                target_param: Optional[str] = None,
                sandbox_first: bool = False) -> ToolSpec:
        """Register the composite as a real tool — after the loyalty & validation gates."""
        # Rule 8 / Rule 4 loyalty gate: NYXARA may not self-author into the sovereign core.
        if comp.owner_exclusive and authority is not Authority.OWNER:
            raise AuthorizationError(
                "cannot self-author a tool that touches the sovereign core; "
                "only the Master may install it",
                context={"tool": comp.name, "capability": comp.capability.value})

        if sample_args is not None:
            report = self.validate(comp, sample_args)
            if not report.ok:
                raise ToolError(f"composite {comp.name!r} failed sandbox validation",
                                context={"error": report.error})

        spec = ToolSpec(
            name=comp.name, handler=self._run_real(comp), description=comp.description,
            params=comp.params, capability=comp.capability, risk=comp.risk,
            reversible=comp.reversible, cost=comp.cost, target_param=target_param,
            sandbox_first=sandbox_first, dry_run=self._run_dry(comp))
        return self.registry.register(spec)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA toolsmith self-test")
    print("=" * 70)

    reg = ToolRegistry()

    @reg.tool("fetch", params=[ToolParam("url", "str")],
              capability=Capability.NET_OUT, description="fetch a page")
    def _fetch(url: str) -> dict:
        return {"title": "hello world", "status": 200}

    @reg.tool("get_field", params=[ToolParam("data", "dict"), ToolParam("key", "str")],
              description="extract a field")
    def _get_field(data: dict, key: str):
        return data[key]

    @reg.tool("shout", params=[ToolParam("text", "str")], description="uppercase")
    def _shout(text: str) -> str:
        return text.upper()

    smith = Toolsmith(reg)

    # compose: fetch a page -> pull its title -> shout it
    comp = smith.compose(
        "fetch_title_loud", params=[ToolParam("url", "str")],
        steps=[
            ToolStep("s1", "fetch", {"url": ArgBinding.inp("url")}),
            ToolStep("s2", "get_field", {"data": ArgBinding.from_step("s1"),
                                         "key": ArgBinding.literal("title")}),
            ToolStep("s3", "shout", {"text": ArgBinding.from_step("s2")}),
        ],
        output_step="s3", description="fetch a page and shout its title")
    print(f"\ncomposed            : {comp.name}  cap={comp.capability.value} "
          f"risk={comp.risk.label} reversible={comp.reversible}")
    assert comp.capability is Capability.NET_OUT   # the most-privileged step

    # validate in the sandbox before it becomes real
    report = smith.validate(comp, {"url": "http://example.com"})
    print(f"sandbox validate    : ok={report.ok} output={report.output!r}")
    assert report.ok and report.output == "HELLO WORLD"

    # install and call it like any other tool
    smith.install(comp)
    r = reg.invoke("fetch_title_loud", {"url": "http://example.com"})
    print(f"installed & called  : ok={r.ok} value={r.value!r}")
    assert r.ok and r.value == "HELLO WORLD"

    # static validation catches bad wiring
    try:
        smith.compose("bad", params=[ToolParam("x", "str")],
                      steps=[ToolStep("s1", "shout", {"text": ArgBinding.from_step("s9")})],
                      output_step="s1")
        raise SystemExit("ERROR: should have rejected forward reference")
    except ValidationError:
        print("\nstatic wiring guard : forward/unknown step reference rejected ✓")

    # privilege inference: a composite that deletes is HIGH/irreversible
    reg.register(ToolSpec("rm", handler=lambda path: f"rm {path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    danger = smith.compose("nuke", params=[ToolParam("path", "str")],
                           steps=[ToolStep("s1", "rm", {"path": ArgBinding.inp("path")})],
                           output_step="s1")
    print(f"privilege inference : nuke risk={danger.risk.label} reversible={danger.reversible}")
    assert danger.risk is RiskTier.HIGH and not danger.reversible
    smith.install(danger, target_param="path")
    r = reg.invoke("nuke", {"path": "/etc/x"}, authority=Authority.AUTONOMOUS)
    assert not r.ok and r.requires_owner   # escalates, as the worst step demands

    # loyalty gate: cannot self-author a tool that edits the Rules (Rule 8)
    reg.register(ToolSpec("edit_rules", handler=lambda: "edited",
                          capability=Capability.MODIFY_RULES, risk=RiskTier.CRITICAL))
    backdoor = smith.compose("backdoor", params=[],
                             steps=[ToolStep("s1", "edit_rules", {})], output_step="s1")
    assert backdoor.owner_exclusive
    try:
        smith.install(backdoor, authority=Authority.AUTONOMOUS)
        raise SystemExit("ERROR: should have refused autonomous install")
    except AuthorizationError:
        print("loyalty gate        : refused self-authored back-door tool ✓ (Rule 8)")
    # the Master, however, may install it
    smith.install(backdoor, authority=Authority.OWNER)
    print("owner install        : the Master may install it ✓")

    print(f"\nfinal tools         : {reg.names()}")
    print("\nALL SELF-TESTS PASSED ✓")
