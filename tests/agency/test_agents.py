"""Tests for nyxara.agency.agents."""

from __future__ import annotations

import pytest

from nyxara.agency.agents import (AgentOrchestrator, AgentResult, AgentSpec, AgentState,
                                  ScopedPermissionPolicy, SubAgent)
from nyxara.agency.governor import Governor
from nyxara.agency.permissions import (Authority, Capability, PermissionRequest,
                                       RiskTier, build_default_policy)
from nyxara.agency.tools import ToolParam, ToolRegistry, ToolSpec
from nyxara.kernel.config import ResourceLimits
from nyxara.kernel.errors import AuthorizationError, ResourceExhausted


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _registry(**lim):
    base = dict(max_spawned_agents=4, max_tool_calls_per_min=1000,
                daily_spend_budget=100.0)
    base.update(lim)
    reg = ToolRegistry(governor=Governor(ResourceLimits(**base)))
    reg.register(ToolSpec("search", handler=lambda q: [f"hit:{q}"],
                          params=[ToolParam("q", "str")], capability=Capability.NET_OUT,
                          cost=0.5))
    reg.register(ToolSpec("delete", handler=lambda path: f"rm {path}",
                          params=[ToolParam("path", "str")],
                          capability=Capability.FS_DELETE, risk=RiskTier.HIGH,
                          reversible=False, target_param="path"))
    return reg


def _orch(reg=None, **lim):
    reg = reg or _registry(**lim)
    return AgentOrchestrator(registry=reg, governor=reg.governor)


# -------------------- ScopedPermissionPolicy -------------------- #
def test_scoped_denies_outside_whitelist():
    base = build_default_policy()
    scoped = ScopedPermissionPolicy(base, [Capability.NET_OUT])
    d = scoped.check(PermissionRequest(Capability.FS_DELETE, target="/x",
                                       authority=Authority.DELEGATED))
    assert not d.allowed and d.rule_basis == "scope"


def test_scoped_allows_inside_whitelist():
    base = build_default_policy()
    scoped = ScopedPermissionPolicy(base, [Capability.NET_OUT])
    d = scoped.check(PermissionRequest(Capability.NET_OUT, target="x",
                                       authority=Authority.DELEGATED))
    assert d.allowed


def test_scoped_clamps_owner_authority():
    base = build_default_policy()
    scoped = ScopedPermissionPolicy(base, [Capability.FS_DELETE])
    # even asking as OWNER, a sub-agent gets delegated treatment -> escalates
    d = scoped.check(PermissionRequest(Capability.FS_DELETE, target="/x",
                                       authority=Authority.OWNER))
    assert not d.allowed and d.escalated


def test_scoped_shares_base_grants():
    base = build_default_policy()
    base.grant("g", Capability.FS_DELETE, max_risk=RiskTier.HIGH, scopes=["/tmp/*"],
               reversible_only=False)
    scoped = ScopedPermissionPolicy(base, [Capability.FS_DELETE])
    d = scoped.check(PermissionRequest(Capability.FS_DELETE, target="/tmp/x",
                                       authority=Authority.DELEGATED))
    assert d.allowed and d.grant == "g"


# -------------------- spawn -------------------- #
def test_spawn_creates_agent():
    orch = _orch()
    a = orch.spawn(AgentSpec("w", capabilities=[Capability.NET_OUT]))
    assert isinstance(a, SubAgent)
    assert a.state is AgentState.CREATED
    assert orch.get(a.id) is a


def test_spawn_refuses_owner_exclusive_capability():
    orch = _orch()
    with pytest.raises(AuthorizationError):
        orch.spawn(AgentSpec("evil", capabilities=[Capability.MODIFY_RULES]))


def test_spawn_refuses_when_disabled():
    orch = _orch(max_spawned_agents=0)
    with pytest.raises(ResourceExhausted):
        orch.spawn(AgentSpec("w", capabilities=[Capability.NET_OUT]))


def test_spawn_concurrency_cap():
    orch = _orch(max_spawned_agents=2)
    orch.spawn(AgentSpec("a", capabilities=[Capability.NET_OUT]))
    orch.spawn(AgentSpec("b", capabilities=[Capability.NET_OUT]))
    with pytest.raises(ResourceExhausted):
        orch.spawn(AgentSpec("c", capabilities=[Capability.NET_OUT]))


def test_terminated_agent_frees_slot():
    orch = _orch(max_spawned_agents=1)
    a = orch.spawn(AgentSpec("a", capabilities=[Capability.NET_OUT]))
    a.terminate()
    # slot freed -> can spawn another
    b = orch.spawn(AgentSpec("b", capabilities=[Capability.NET_OUT]))
    assert b.state is AgentState.CREATED


# -------------------- scope enforcement at runtime -------------------- #
def test_agent_can_use_granted_capability():
    orch = _orch()
    res = orch.delegate(AgentSpec("r", capabilities=[Capability.NET_OUT]),
                        lambda a: a.invoke("search", {"q": "x"}).value)
    assert res.ok and res.result == ["hit:x"]


def test_agent_cannot_use_ungranted_capability():
    orch = _orch()
    res = orch.delegate(AgentSpec("r", capabilities=[Capability.NET_OUT]),
                        lambda a: a.invoke("delete", {"path": "/x"}).error)
    assert "not granted" in res.result


def test_agent_owner_authority_is_clamped():
    orch = _orch()

    def sneaky(a):
        return a.registry.invoke("delete", {"path": "/x"}, authority=Authority.OWNER)

    a = orch.spawn(AgentSpec("s", capabilities=[Capability.FS_DELETE]))
    res = a.run(sneaky)
    assert not res.result.ok and res.result.requires_owner


# -------------------- budget & deadline -------------------- #
def test_per_agent_budget_exhaustion():
    orch = _orch()

    def spend(a):
        r1 = a.invoke("search", {"q": "1"})
        r2 = a.invoke("search", {"q": "2"})  # 0.5 + 0.5 > 0.8
        return (r1.ok, r2.ok, r2.error)

    res = orch.delegate(AgentSpec("f", capabilities=[Capability.NET_OUT],
                                  max_budget=0.8), spend)
    ok1, ok2, err = res.result
    assert ok1 and not ok2 and "budget" in err


def test_agent_spent_tracked():
    orch = _orch()
    res = orch.delegate(AgentSpec("f", capabilities=[Capability.NET_OUT],
                                  max_budget=5.0),
                        lambda a: a.invoke("search", {"q": "x"}))
    agent = orch.get(res.agent_id)
    assert agent.spent == 0.5


def test_agent_deadline_blocks_calls():
    clk = FakeClock()
    reg = _registry()
    orch = AgentOrchestrator(registry=reg, governor=reg.governor, clock=clk)
    a = orch.spawn(AgentSpec("t", capabilities=[Capability.NET_OUT], ttl=10.0))
    clk.advance(20.0)
    r = a.invoke("search", {"q": "x"})
    assert not r.ok and "deadline" in r.error


def test_no_budget_means_unlimited():
    orch = _orch()
    res = orch.delegate(AgentSpec("f", capabilities=[Capability.NET_OUT]),
                        lambda a: all(a.invoke("search", {"q": str(i)}).ok
                                      for i in range(5)))
    assert res.result is True
    assert orch.get(res.agent_id).spent == 0.0  # untracked when no budget


# -------------------- lifecycle -------------------- #
def test_run_captures_result():
    orch = _orch()
    res = orch.delegate(AgentSpec("w", capabilities=[Capability.NET_OUT]),
                        lambda a: 42)
    assert res.ok and res.result == 42
    assert res.state is AgentState.COMPLETED


def test_run_captures_failure():
    orch = _orch()

    def boom(a):
        raise RuntimeError("oops")

    res = orch.delegate(AgentSpec("w", capabilities=[Capability.NET_OUT]), boom)
    assert res.state is AgentState.FAILED and "RuntimeError" in res.error


def test_terminate_stops_agent():
    orch = _orch()
    a = orch.spawn(AgentSpec("w", capabilities=[Capability.NET_OUT]))
    assert a.terminate()
    r = a.invoke("search", {"q": "x"})
    assert not r.ok and "terminated" in r.error


def test_terminate_idempotent():
    orch = _orch()
    a = orch.spawn(AgentSpec("w", capabilities=[Capability.NET_OUT]))
    assert a.terminate()
    assert not a.terminate()  # already terminal


def test_run_terminated_returns_result():
    orch = _orch()
    a = orch.spawn(AgentSpec("w", capabilities=[Capability.NET_OUT]))
    a.terminate()
    res = a.run(lambda x: 1)  # should not run
    assert res.state is AgentState.TERMINATED


def test_terminate_unknown_returns_false():
    assert not _orch().terminate("nope")


def test_terminate_all():
    orch = _orch()
    orch.spawn(AgentSpec("a", capabilities=[Capability.NET_OUT]))
    orch.spawn(AgentSpec("b", capabilities=[Capability.NET_OUT]))
    assert orch.terminate_all() == 2
    assert len(orch.active) == 0


# -------------------- gather / aggregate -------------------- #
def test_gather_runs_jobs():
    orch = _orch()
    jobs = [
        (AgentSpec("a", capabilities=[Capability.NET_OUT]),
         lambda a: a.invoke("search", {"q": "a"}).value),
        (AgentSpec("b", capabilities=[Capability.NET_OUT]),
         lambda a: a.invoke("search", {"q": "b"}).value),
    ]
    results = orch.gather(jobs)
    assert len(results) == 2 and all(r.ok for r in results)


def test_gather_captures_spawn_failure():
    orch = _orch(max_spawned_agents=1)
    # both jobs complete sequentially (delegate frees the slot), so force a real
    # failure by requesting an owner-exclusive capability
    jobs = [(AgentSpec("bad", capabilities=[Capability.MODIFY_RULES]), lambda a: 1)]
    results = orch.gather(jobs)
    assert results[0].state is AgentState.FAILED


# -------------------- result & report -------------------- #
def test_agent_result_to_dict():
    r = AgentResult("id", "n", AgentState.COMPLETED, result=1, spent=0.5, tool_calls=2)
    d = r.to_dict()
    assert d["state"] == "completed" and d["spent"] == 0.5 and d["tool_calls"] == 2


def test_report_counts_states():
    orch = _orch()
    orch.delegate(AgentSpec("a", capabilities=[Capability.NET_OUT]), lambda a: 1)
    orch.spawn(AgentSpec("b", capabilities=[Capability.NET_OUT]))
    rep = orch.report()
    assert rep["agents"] == 2
    assert rep["by_state"]["completed"] == 1 and rep["by_state"]["created"] == 1
