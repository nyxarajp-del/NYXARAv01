"""NYXARA · agency/agents.py — bounded, least-privilege sub-agent orchestration (⛬).

Some work is best delegated. NYXARA can spawn **sub-agents** — focused workers that
pursue a single objective with their own scratch space — but never at the cost of
control or loyalty. Every sub-agent is *bounded* and *least-privileged by construction*:

* **Capability-scoped** — a sub-agent is handed an explicit whitelist of capabilities and
  can do *nothing else*. A :class:`ScopedPermissionPolicy` refuses any tool call outside
  that whitelist, on top of the normal permission gate.
* **Never the Master** — a sub-agent runs under **delegated** authority; even if its code
  asks to act as the owner, its authority is clamped down. Owner-exclusive capabilities
  (the Rules, the permissions, the identity) can never be delegated to it at all (Rule 8).
* **Budgeted & time-boxed** — each sub-agent has its own spend allotment and deadline; it
  cannot exhaust the Master's global budget, and the number of concurrent sub-agents is
  capped by the governor (Rule 3 — no runaway swarm).
* **Loyalty inherited** — escalations bubble up to the Master rather than being
  self-approved; results are captured as data, and the orchestrator can **terminate** any
  sub-agent at any moment (corrigibility).

The :class:`AgentOrchestrator` spawns, runs and aggregates sub-agents, sharing the parent
tool registry, governor and sandbox so delegated work runs through the very same safety
pipeline as everything else.

Depends on :mod:`agency.permissions`, :mod:`agency.tools`, :mod:`agency.governor`,
:mod:`kernel.errors`. Pure standard library otherwise.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from nyxara.agency.governor import Governor, WindowBudget
from nyxara.agency.permissions import (Authority, Capability, PermissionDecision,
                                       PermissionPolicy, PermissionRequest, RiskTier,
                                       build_default_policy)
from nyxara.agency.tools import ToolRegistry, ToolResult
from nyxara.kernel.errors import AuthorizationError, ResourceExhausted

__all__ = [
    "AgentState",
    "ScopedPermissionPolicy",
    "AgentSpec",
    "SubAgent",
    "AgentResult",
    "AgentOrchestrator",
]


# --------------------------------------------------------------------------- #
# Scoped permission policy (least privilege for a sub-agent)
# --------------------------------------------------------------------------- #
class ScopedPermissionPolicy(PermissionPolicy):
    """Wraps a base policy, restricting a sub-agent to a whitelist of capabilities and
    clamping its authority — a sub-agent is never the Master."""

    def __init__(self, base: PermissionPolicy, allowed: Sequence[Capability]) -> None:
        super().__init__()
        # share the base's capability metadata and standing grants
        self._meta = base._meta
        self._grants = base._grants
        self.base = base
        self.allowed: Set[Capability] = set(allowed)

    def check(self, req: PermissionRequest) -> PermissionDecision:
        meta = self.meta(req.capability)
        risk = req.risk if req.risk is not None else meta.default_risk
        irreversible = (not req.reversible) or meta.irreversible_by_nature
        if req.capability not in self.allowed:
            d = PermissionDecision(
                allowed=False, requires_owner=False, capability=req.capability,
                risk=risk, reason="capability not granted to this sub-agent",
                rule_basis="scope", authority=req.authority, irreversible=irreversible)
            self._audit.append(d)
            return d
        # a sub-agent can never wield the Master's authority
        auth = Authority.DELEGATED if req.authority is Authority.OWNER else req.authority
        return super().check(replace(req, authority=auth))


# --------------------------------------------------------------------------- #
# State & spec
# --------------------------------------------------------------------------- #
class AgentState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


_TERMINAL = {AgentState.COMPLETED, AgentState.FAILED, AgentState.TERMINATED}


@dataclass
class AgentSpec:
    name: str
    objective: str = ""
    capabilities: Sequence[Capability] = ()
    max_budget: Optional[float] = None     # per-agent spend allotment
    ttl: Optional[float] = None            # seconds before the agent's deadline
    parent: str = "NYXARA"


# --------------------------------------------------------------------------- #
# Sub-agent
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    agent_id: str
    name: str
    state: AgentState
    result: Any = None
    error: Optional[str] = None
    spent: float = 0.0
    tool_calls: int = 0

    @property
    def ok(self) -> bool:
        return self.state is AgentState.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "name": self.name, "state": self.state.value,
                "result": self.result, "error": self.error, "spent": round(self.spent, 4),
                "tool_calls": self.tool_calls}


class SubAgent:
    """A focused, capability-scoped worker that runs delegated work safely."""

    def __init__(self, spec: AgentSpec, registry: ToolRegistry, *,
                 clock=None, deadline_at: Optional[float] = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.spec = spec
        self.registry = registry
        self._clock = clock or time.monotonic
        self.state = AgentState.CREATED
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.deadline_at = deadline_at
        self.tool_calls = 0
        self._budget = WindowBudget(spec.max_budget, window_s=float("inf"),
                                    clock=self._clock) if spec.max_budget else None

    # ---- properties ---- #
    @property
    def spent(self) -> float:
        return self._budget.spent if self._budget else 0.0

    @property
    def expired(self) -> bool:
        return self.deadline_at is not None and self._clock() >= self.deadline_at

    # ---- tool access (always delegated, budget-guarded) ---- #
    def invoke(self, tool: str, args: Optional[Dict[str, Any]] = None, **kw: Any
               ) -> ToolResult:
        if self.state is AgentState.TERMINATED:
            return ToolResult(tool, ok=False, error="agent terminated")
        if self.expired:
            return ToolResult(tool, ok=False, error="agent deadline exceeded")
        spec = self.registry.get(tool)
        if self._budget is not None and spec is not None and spec.cost > 0 \
                and not self._budget.can_afford(spec.cost):
            return ToolResult(tool, ok=False, error="agent budget exhausted")
        res = self.registry.invoke(tool, args, authority=Authority.DELEGATED, **kw)
        self.tool_calls += 1
        if res.ok and self._budget is not None and spec is not None and spec.cost > 0:
            self._budget.charge(spec.cost)
        return res

    # ---- lifecycle ---- #
    def run(self, task: Callable[["SubAgent"], Any]) -> AgentResult:
        if self.state in _TERMINAL:
            return self.as_result()
        self.state = AgentState.RUNNING
        self.started_at = self._clock()
        try:
            self.result = task(self)
            self.state = AgentState.COMPLETED
        except Exception as exc:  # noqa: BLE001 — captured as data, never crashes the parent
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = AgentState.FAILED
        self.ended_at = self._clock()
        return self.as_result()

    def terminate(self) -> bool:
        if self.state in _TERMINAL:
            return False
        self.state = AgentState.TERMINATED
        self.ended_at = self._clock()
        return True

    def as_result(self) -> AgentResult:
        return AgentResult(agent_id=self.id, name=self.spec.name, state=self.state,
                           result=self.result, error=self.error, spent=self.spent,
                           tool_calls=self.tool_calls)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class AgentOrchestrator:
    """Spawns, runs and aggregates bounded, least-privilege sub-agents."""

    def __init__(self, *, registry: Optional[ToolRegistry] = None,
                 policy: Optional[PermissionPolicy] = None,
                 governor: Optional[Governor] = None, clock=None) -> None:
        self.governor = governor or Governor()
        self.policy = policy or (registry.policy if registry else build_default_policy())
        self._parent_registry = registry or ToolRegistry(policy=self.policy,
                                                         governor=self.governor)
        self._clock = clock or time.monotonic
        self.max_agents = self.governor.limits.max_spawned_agents
        self._agents: Dict[str, SubAgent] = {}

    # ---- counts ---- #
    @property
    def active(self) -> List[SubAgent]:
        return [a for a in self._agents.values() if a.state not in _TERMINAL]

    def get(self, agent_id: str) -> Optional[SubAgent]:
        return self._agents.get(agent_id)

    # ---- spawn ---- #
    def spawn(self, spec: AgentSpec) -> SubAgent:
        if self.max_agents <= 0:
            raise ResourceExhausted("sub-agent spawning is disabled (max_spawned_agents=0)")
        # owner-exclusive capabilities can never be delegated to a sub-agent (Rule 8)
        for cap in spec.capabilities:
            if self.policy.meta(cap).owner_exclusive:
                raise AuthorizationError(
                    "an owner-exclusive capability cannot be delegated to a sub-agent",
                    context={"capability": cap.value})
        if len(self.active) >= self.max_agents:
            raise ResourceExhausted("sub-agent concurrency cap reached",
                                    context={"max": self.max_agents})
        scoped = ScopedPermissionPolicy(self.policy, spec.capabilities)
        child = ToolRegistry(policy=scoped, governor=self.governor,
                             sandbox=self._parent_registry.sandbox)
        child._tools = dict(self._parent_registry._tools)   # share tools, not the policy
        deadline = (self._clock() + spec.ttl) if spec.ttl else None
        agent = SubAgent(spec, child, clock=self._clock, deadline_at=deadline)
        self._agents[agent.id] = agent
        return agent

    # ---- run / delegate ---- #
    def run(self, agent: SubAgent, task: Callable[[SubAgent], Any]) -> AgentResult:
        return agent.run(task)

    def delegate(self, spec: AgentSpec, task: Callable[[SubAgent], Any]) -> AgentResult:
        """Spawn a sub-agent, run a task to completion, and return its result."""
        agent = self.spawn(spec)
        return agent.run(task)

    def gather(self, jobs: Sequence) -> List[AgentResult]:
        """Run several ``(spec, task)`` jobs and aggregate their results."""
        results: List[AgentResult] = []
        for spec, task in jobs:
            try:
                results.append(self.delegate(spec, task))
            except (ResourceExhausted, AuthorizationError) as exc:
                results.append(AgentResult(agent_id="-", name=spec.name,
                                           state=AgentState.FAILED, error=str(exc)))
        return results

    # ---- termination ---- #
    def terminate(self, agent_id: str) -> bool:
        a = self._agents.get(agent_id)
        return a.terminate() if a else False

    def terminate_all(self) -> int:
        return sum(1 for a in self.active if a.terminate())

    # ---- reporting ---- #
    def report(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for a in self._agents.values():
            counts[a.state.value] = counts.get(a.state.value, 0) + 1
        return {"agents": len(self._agents), "active": len(self.active),
                "max_agents": self.max_agents, "by_state": counts}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.agency.tools import ToolParam, ToolSpec
    from nyxara.kernel.config import ResourceLimits

    print("=" * 70)
    print("NYXARA sub-agent orchestration self-test")
    print("=" * 70)

    # a parent registry with a few tools
    reg = ToolRegistry(governor=Governor(ResourceLimits(max_spawned_agents=2,
                                                        max_tool_calls_per_min=1000,
                                                        daily_spend_budget=100.0)))
    reg.register(ToolSpec("search", handler=lambda q: [f"hit:{q}"],
                          params=[ToolParam("q", "str")], capability=Capability.NET_OUT,
                          cost=0.5))
    reg.register(ToolSpec("delete", handler=lambda path: f"rm {path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))

    orch = AgentOrchestrator(registry=reg, governor=reg.governor)

    # a researcher sub-agent: granted ONLY the search capability
    def research(agent: SubAgent):
        r1 = agent.invoke("search", {"q": "nyxara"})
        # it may NOT delete — that capability was never granted to it
        r2 = agent.invoke("delete", {"path": "/etc/x"})
        return {"search_ok": r1.ok, "delete_attempt": r2.error}

    res = orch.delegate(AgentSpec("researcher", capabilities=[Capability.NET_OUT],
                                  max_budget=2.0), research)
    print(f"\nresearcher          : state={res.state.value} result={res.result}")
    assert res.ok
    assert res.result["search_ok"] is True
    assert "not granted" in (res.result["delete_attempt"] or "")  # scope enforced
    print("scope enforcement   : sub-agent could search but NOT delete ✓")

    # owner-exclusive capability can never be delegated (Rule 8)
    try:
        orch.spawn(AgentSpec("evil", capabilities=[Capability.MODIFY_RULES]))
        raise SystemExit("ERROR: should have refused owner-exclusive delegation")
    except AuthorizationError:
        print("loyalty guard       : refused to delegate an owner-exclusive cap ✓")

    # a sub-agent is never the Master: even asking as OWNER is clamped to delegated
    def sneaky(agent: SubAgent):
        return agent.registry.invoke("delete", {"path": "/etc/x"},
                                     authority=Authority.OWNER)
    sa = orch.spawn(AgentSpec("sneaky", capabilities=[Capability.FS_DELETE]))
    r = sa.run(sneaky)
    print(f"authority clamp     : owner-as-subagent delete ok={r.result.ok} "
          f"requires_owner={r.result.requires_owner}")
    assert not r.result.ok and r.result.requires_owner  # escalates, not auto-approved

    # per-agent budget: a tight allotment refuses further paid calls
    def spender(agent: SubAgent):
        a = agent.invoke("search", {"q": "1"})   # costs 0.5
        b = agent.invoke("search", {"q": "2"})   # costs 0.5 -> budget (0.8) exhausted
        return {"a": a.ok, "b": b.ok, "b_err": b.error}
    res = orch.delegate(AgentSpec("frugal", capabilities=[Capability.NET_OUT],
                                  max_budget=0.8), spender)
    print(f"per-agent budget    : a={res.result['a']} b={res.result['b']} "
          f"({res.result['b_err']})")
    assert res.result["a"] and not res.result["b"]

    # concurrency cap & termination
    a1 = orch.spawn(AgentSpec("w1", capabilities=[Capability.NET_OUT]))
    a2 = orch.spawn(AgentSpec("w2", capabilities=[Capability.NET_OUT]))
    try:
        orch.spawn(AgentSpec("w3", capabilities=[Capability.NET_OUT]))
        raise SystemExit("ERROR: should have hit the agent cap")
    except ResourceExhausted:
        print(f"concurrency cap     : refused a 3rd agent (max={orch.max_agents}) ✓")
    assert a1.terminate() and orch.terminate(a2.id)
    print("termination         : agents terminated on command ✓")

    print(f"\nreport              : {orch.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
