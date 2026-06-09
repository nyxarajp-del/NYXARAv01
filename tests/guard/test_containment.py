"""Tests for nyxara.guard.containment."""

from __future__ import annotations

import pytest

from nyxara.guard.containment import (Component, ComponentState, Containment,
                                      ContainmentAction, ContainmentLevel)
from nyxara.kernel.errors import AuthorizationError, InvariantViolation


def _cm():
    cm = Containment()
    cm.register("kernel", kind="core", capabilities={"read", "write", "modify"})
    cm.register("planner", kind="subsystem",
                capabilities={"read", "write_plan", "exec_plan"}, depends_on={"kernel"})
    cm.register("tool", kind="tool", capabilities={"net_out", "read"},
                connects_to={"planner"})
    cm.register("agent", kind="agent", capabilities={"net_out", "write", "spawn"},
                connects_to={"tool"})
    return cm


# -------------------- levels & state -------------------- #
def test_level_ordering():
    assert ContainmentLevel.MONITOR < ContainmentLevel.QUARANTINE < ContainmentLevel.TERMINATE
    assert ContainmentLevel.RESTRICT.label == "restrict"


# -------------------- registration -------------------- #
def test_register():
    cm = Containment()
    c = cm.register("x", capabilities={"read"})
    assert c.healthy and c.capabilities == {"read"}


def test_get_unknown():
    assert Containment().get("nope") is None


def test_require_unknown_raises():
    with pytest.raises(InvariantViolation):
        Containment().contain("nope", ContainmentLevel.RESTRICT)


# -------------------- capability sandboxing -------------------- #
def test_restrict_strips_risky_keeps_safe():
    cm = _cm()
    a = cm.restrict("planner", reason="anomaly")
    assert "write_plan" in a.stripped and "exec_plan" in a.stripped
    assert cm.effective_capabilities("planner") == {"read"}
    assert cm.get("planner").degraded


def test_monitor_does_not_strip():
    cm = _cm()
    cm.monitor("planner")
    assert cm.effective_capabilities("planner") == {"read", "write_plan", "exec_plan"}
    assert cm.get("planner").state is ComponentState.SUSPECT


def test_isolate_strips_network():
    cm = _cm()
    cm.isolate("tool", radius=0)
    assert "net_out" not in cm.effective_capabilities("tool")
    assert cm.get("tool").isolated


def test_quarantine_strips_all():
    cm = _cm()
    cm.quarantine("agent", zone="z")
    assert cm.effective_capabilities("agent") == set()
    assert cm.get("agent").state is ComponentState.QUARANTINED


def test_terminate_strips_all():
    cm = _cm()
    cm.terminate("agent")
    assert cm.effective_capabilities("agent") == set()
    assert cm.get("agent").state is ComponentState.TERMINATED


# -------------------- level is monotonic -------------------- #
def test_level_only_increases():
    cm = _cm()
    cm.quarantine("agent", zone="z")
    cm.contain("agent", ContainmentLevel.RESTRICT)  # weaker request
    assert cm.get("agent").level is ContainmentLevel.QUARANTINE  # not downgraded


# -------------------- blast radius -------------------- #
def test_blast_radius_finds_neighbours():
    cm = _cm()
    radius = cm.blast_radius("tool", 1)
    assert "tool" in radius
    assert "planner" in radius or "agent" in radius


def test_isolate_contains_neighbours():
    cm = _cm()
    a = cm.isolate("tool", radius=1)
    assert a.affected  # neighbours got restricted
    for nid in a.affected:
        assert cm.get(nid).level >= ContainmentLevel.RESTRICT


def test_no_radius_no_neighbours():
    cm = _cm()
    a = cm.restrict("tool", radius=0)
    assert a.affected == []


def test_blast_radius_respects_distance():
    cm = Containment()
    cm.register("a", connects_to={"b"})
    cm.register("b", connects_to={"c"})
    cm.register("c")
    assert "c" not in cm.blast_radius("a", 1)
    assert "c" in cm.blast_radius("a", 2)


# -------------------- zones -------------------- #
def test_quarantine_zone_membership():
    cm = _cm()
    cm.quarantine("agent", zone="rogue")
    assert "agent" in cm.zone("rogue")


def test_zone_empty_default():
    assert Containment().zone("nope") == []


# -------------------- release (sticky, owner-only) -------------------- #
def test_release_requires_owner():
    cm = _cm()
    cm.quarantine("agent", zone="z")
    with pytest.raises(AuthorizationError):
        cm.release("agent", owner=False)


def test_release_restores_caps():
    cm = _cm()
    cm.quarantine("agent", zone="z")
    cm.release("agent", owner=True)
    c = cm.get("agent")
    assert c.healthy and c.capabilities == {"net_out", "write", "spawn"}
    assert c.zone is None


def test_release_clears_zone():
    cm = _cm()
    cm.quarantine("agent", zone="z")
    cm.release("agent", owner=True)
    assert "agent" not in cm.zone("z")


def test_contained_stays_until_release():
    cm = _cm()
    cm.restrict("planner")
    assert cm.is_contained("planner")
    cm.release("planner", owner=True)
    assert not cm.is_contained("planner")


# -------------------- snapshot / rollback -------------------- #
def test_snapshot_rollback_restores():
    cm = _cm()
    safe = cm.snapshot()
    cm.terminate("kernel")
    assert cm.get("kernel").state is ComponentState.TERMINATED
    cm.rollback(safe)
    assert cm.get("kernel").healthy


def test_rollback_restores_capabilities():
    cm = _cm()
    safe = cm.snapshot()
    cm.quarantine("agent", zone="z")
    cm.rollback(safe)
    assert cm.effective_capabilities("agent") == {"net_out", "write", "spawn"}


def test_rollback_unknown_snapshot():
    with pytest.raises(InvariantViolation):
        Containment().rollback("nope")


def test_snapshot_independent():
    cm = _cm()
    s1 = cm.snapshot()
    cm.restrict("planner")
    s2 = cm.snapshot()
    cm.rollback(s1)
    assert cm.effective_capabilities("planner") == {"read", "write_plan", "exec_plan"}
    cm.rollback(s2)
    assert cm.effective_capabilities("planner") == {"read"}


# -------------------- defensive-only invariant -------------------- #
def test_terminate_targets_internal_component():
    cm = _cm()
    a = cm.terminate("planner")
    assert a.component == "planner"  # always one of OUR components


def test_no_offensive_verbs():
    # the public API contains only containment verbs, never an attack verb
    api = set(dir(Containment))
    for word in ("attack", "retaliate", "strike", "exploit", "counterattack"):
        assert word not in api


# -------------------- tamper-evident log -------------------- #
def test_log_verifies():
    cm = _cm()
    cm.restrict("planner")
    assert cm.verify_log()


def test_log_tamper_detected():
    cm = _cm()
    cm.restrict("planner")
    cm._log[0]["rec"]["kind"] = "nothing"
    with pytest.raises(InvariantViolation):
        cm.verify_log()


def test_empty_log_verifies():
    assert Containment().verify_log()


def test_log_records_actions():
    cm = _cm()
    cm.restrict("planner")
    cm.quarantine("agent", zone="z")
    kinds = [e["rec"]["kind"] for e in cm._log]
    assert "contain" in kinds and "quarantine" in kinds


# -------------------- shapes / reporting -------------------- #
def test_action_to_dict():
    cm = _cm()
    d = cm.restrict("planner").to_dict()
    assert d["level"] == "restrict" and "stripped" in d


def test_component_to_dict():
    cm = _cm()
    d = cm.get("planner").to_dict()
    assert d["id"] == "planner" and "capabilities" in d


def test_report():
    cm = _cm()
    cm.terminate("agent")
    r = cm.report()
    assert r["components"] == 4 and r["by_state"]["terminated"] == 1
