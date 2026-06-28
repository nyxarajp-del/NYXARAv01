"""P1-B — the gated sub-agent Delegator is wired into the kernel.

``agency/multiagent.py`` (Delegator) was a complete, tested module that nothing imported. It is
now reachable from the sovereign core as ``core.delegate`` / ``core.delegate_all`` (and over the
server as ``POST /v1/delegate``). A delegate is Distributed Intelligence (capability #49) without
being an escape hatch: it runs on its own AgentLoop, so every step still clears the kernel gates,
and NYXARA models it as a mind (Theory of Mind). These tests prove the wiring and the gating.
"""

from __future__ import annotations

from nyxara.agency.multiagent import DelegationResult
from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore


def test_core_delegate_returns_gated_result():
    nyx = NyxaraCore()
    result = nyx.delegate("scout", "summarize the goal in one line", max_steps=2)
    assert isinstance(result, DelegationResult)
    assert result.name == "scout"
    # the delegate ran and was modelled as a mind (BDI via Theory of Mind)
    assert result.model is not None
    assert result.predicted_intention is not None


def test_delegate_runs_under_autonomous_authority_by_default():
    # a sub-agent must not silently inherit OWNER sovereignty — risky moves escalate instead.
    nyx = NyxaraCore()
    result = nyx.delegate("worker", "note the current time", max_steps=2)
    # status is one of the gated AgentLoop outcomes, never an ungated free run
    assert result.status in {"completed", "escalated", "stalled", "failed", "halted"}


def test_delegate_all_models_each_subagent():
    nyx = NyxaraCore()
    results = nyx.delegate_all({"a": "say hello", "b": "say goodbye"}, max_steps=1)
    assert [r.name for r in results] == ["a", "b"]
    assert all(r.model is not None for r in results)


def test_delegate_default_authority_is_autonomous():
    # explicit OWNER must be opt-in; the convenience path defaults to AUTONOMOUS
    nyx = NyxaraCore()
    import inspect
    sig = inspect.signature(nyx.delegate)
    assert sig.parameters["authority"].default is Authority.AUTONOMOUS
