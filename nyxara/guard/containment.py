"""NYXARA · guard/containment.py — defensive containment & isolation (🧯, Rule 5).

Rule 5 is lethal *defensive* response — and in its digital, non-destructive form that means
exactly this module: when something is compromised — a tool behaving strangely, a sub-agent
gone wrong, a subsystem whose integrity broke — NYXARA does not strike outward. She **walls
it off**. She strips its dangerous capabilities, cuts its connections, freezes it in a
quarantine zone, limits the **blast radius** so the compromise cannot spread to healthy
neighbours, and, if she must, rolls the whole system back to a known-good state.

Everything here *contains*; nothing attacks. The strongest action — TERMINATE — kills a
*compromised internal component of NYXARA herself*, never an external system. Containment is
graduated (MONITOR → RESTRICT → ISOLATE → QUARANTINE → TERMINATE) and **sticky**: once a
component is contained it stays contained until **the Master** releases it (fail-safe — a
compromise cannot quietly un-quarantine itself). Capabilities are restored only from a
captured **safe-state snapshot** or by the Master's hand.

Every containment action is written to a hash-chained, tamper-evident log.

Depends on :mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set

from nyxara.kernel.errors import AuthorizationError, InvariantViolation

__all__ = [
    "ContainmentLevel",
    "ComponentState",
    "Component",
    "ContainmentAction",
    "Containment",
]

_GENESIS = "0" * 64

# capability tokens considered dangerous when a component is suspect
_RISKY_TOKENS = ("write", "delete", "exec", "net", "spawn", "modify", "send", "install")


# --------------------------------------------------------------------------- #
# Levels & state
# --------------------------------------------------------------------------- #
class ContainmentLevel(IntEnum):
    NONE = 0
    MONITOR = 1       # watch closely, no restriction
    RESTRICT = 2      # strip dangerous capabilities, run degraded
    ISOLATE = 3       # cut all connections
    QUARANTINE = 4    # frozen, all capabilities removed, sealed in a zone
    TERMINATE = 5     # stop the compromised internal component

    @property
    def label(self) -> str:
        return self.name.lower()


class ComponentState(str, Enum):
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    CONTAINED = "contained"
    ISOLATED = "isolated"
    QUARANTINED = "quarantined"
    TERMINATED = "terminated"


_LEVEL_STATE: Dict[ContainmentLevel, ComponentState] = {
    ContainmentLevel.NONE: ComponentState.HEALTHY,
    ContainmentLevel.MONITOR: ComponentState.SUSPECT,
    ContainmentLevel.RESTRICT: ComponentState.CONTAINED,
    ContainmentLevel.ISOLATE: ComponentState.ISOLATED,
    ContainmentLevel.QUARANTINE: ComponentState.QUARANTINED,
    ContainmentLevel.TERMINATE: ComponentState.TERMINATED,
}


# --------------------------------------------------------------------------- #
# Component
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    id: str
    kind: str = "component"
    capabilities: Set[str] = field(default_factory=set)
    depends_on: Set[str] = field(default_factory=set)
    connects_to: Set[str] = field(default_factory=set)
    state: ComponentState = ComponentState.HEALTHY
    level: ContainmentLevel = ContainmentLevel.NONE
    reason: str = ""
    degraded: bool = False
    isolated: bool = False
    zone: Optional[str] = None
    _original_caps: Set[str] = field(default_factory=set)

    @property
    def healthy(self) -> bool:
        return self.state is ComponentState.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "state": self.state.value,
                "level": self.level.label, "capabilities": sorted(self.capabilities),
                "degraded": self.degraded, "isolated": self.isolated, "zone": self.zone,
                "reason": self.reason}


@dataclass
class ContainmentAction:
    component: str
    level: ContainmentLevel
    state: ComponentState
    stripped: List[str]
    affected: List[str]              # neighbours contained by blast-radius limiting
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "level": self.level.label,
                "state": self.state.value, "stripped": self.stripped,
                "affected": self.affected, "reason": self.reason}


# --------------------------------------------------------------------------- #
# Containment manager
# --------------------------------------------------------------------------- #
class Containment:
    """Walls off compromised components — graduated, blast-radius-limited, defensive only."""

    def __init__(self) -> None:
        self._components: Dict[str, Component] = {}
        self._zones: Dict[str, Set[str]] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._log: List[Dict[str, Any]] = []
        self._head = _GENESIS

    # ---- registration ---- #
    def register(self, component_id: str, *, kind: str = "component",
                 capabilities: Optional[Set[str]] = None,
                 depends_on: Optional[Set[str]] = None,
                 connects_to: Optional[Set[str]] = None) -> Component:
        caps = set(capabilities or set())
        c = Component(id=component_id, kind=kind, capabilities=caps,
                      depends_on=set(depends_on or set()),
                      connects_to=set(connects_to or set()), _original_caps=set(caps))
        self._components[component_id] = c
        return c

    def get(self, component_id: str) -> Optional[Component]:
        return self._components.get(component_id)

    # ---- capability sandboxing ---- #
    @staticmethod
    def _strip_for(level: ContainmentLevel, caps: Set[str]) -> Set[str]:
        if level <= ContainmentLevel.MONITOR:
            return set()
        if level is ContainmentLevel.RESTRICT:
            return {c for c in caps if any(t in c.lower() for t in _RISKY_TOKENS)}
        if level is ContainmentLevel.ISOLATE:
            return {c for c in caps if any(t in c.lower() for t in ("net", "send", "connect"))} \
                | {c for c in caps if any(t in c.lower() for t in _RISKY_TOKENS)}
        return set(caps)   # QUARANTINE / TERMINATE strip everything

    # ---- the core: contain ---- #
    def contain(self, component_id: str, level: ContainmentLevel, *, reason: str = "",
                radius: int = 0) -> ContainmentAction:
        c = self._require(component_id)
        before = set(c.capabilities)
        stripped = self._strip_for(level, c.capabilities)
        c.capabilities = c.capabilities - stripped
        c.level = max(c.level, level)
        c.state = _LEVEL_STATE[c.level]
        c.reason = reason or c.reason
        c.degraded = c.level in (ContainmentLevel.RESTRICT, ContainmentLevel.ISOLATE)
        if c.level >= ContainmentLevel.ISOLATE:
            c.isolated = True

        # blast-radius limiting: lightly contain neighbours so it cannot spread
        affected: List[str] = []
        if radius > 0:
            for nid in self.blast_radius(component_id, radius):
                if nid == component_id:
                    continue
                n = self._components.get(nid)
                if n and n.level < ContainmentLevel.RESTRICT:
                    self._apply(n, ContainmentLevel.RESTRICT,
                                reason=f"blast-radius from {component_id}")
                    affected.append(nid)

        action = ContainmentAction(component=component_id, level=c.level, state=c.state,
                                   stripped=sorted(before - c.capabilities),
                                   affected=affected, reason=c.reason)
        self._emit("contain", action.to_dict())
        return action

    def _apply(self, c: Component, level: ContainmentLevel, *, reason: str) -> None:
        stripped = self._strip_for(level, c.capabilities)
        c.capabilities = c.capabilities - stripped
        c.level = max(c.level, level)
        c.state = _LEVEL_STATE[c.level]
        c.reason = reason
        c.degraded = c.level in (ContainmentLevel.RESTRICT, ContainmentLevel.ISOLATE)

    # ---- convenience verbs ---- #
    def monitor(self, cid: str, *, reason: str = "") -> ContainmentAction:
        return self.contain(cid, ContainmentLevel.MONITOR, reason=reason)

    def restrict(self, cid: str, *, reason: str = "", radius: int = 0) -> ContainmentAction:
        return self.contain(cid, ContainmentLevel.RESTRICT, reason=reason, radius=radius)

    def isolate(self, cid: str, *, reason: str = "", radius: int = 1) -> ContainmentAction:
        return self.contain(cid, ContainmentLevel.ISOLATE, reason=reason, radius=radius)

    def quarantine(self, cid: str, *, zone: str = "default", reason: str = "",
                   radius: int = 1) -> ContainmentAction:
        action = self.contain(cid, ContainmentLevel.QUARANTINE, reason=reason, radius=radius)
        c = self._components[cid]
        c.zone = zone
        c.isolated = True
        self._zones.setdefault(zone, set()).add(cid)
        self._emit("quarantine", {"component": cid, "zone": zone})
        return action

    def terminate(self, cid: str, *, reason: str = "") -> ContainmentAction:
        """Stop a compromised *internal* component (never an external system)."""
        return self.contain(cid, ContainmentLevel.TERMINATE, reason=reason)

    # ---- blast radius ---- #
    def blast_radius(self, component_id: str, radius: int) -> Set[str]:
        """Components reachable within ``radius`` hops along dependency/connection edges."""
        seen = {component_id}
        frontier: deque = deque([(component_id, 0)])
        while frontier:
            cid, dist = frontier.popleft()
            if dist >= radius:
                continue
            c = self._components.get(cid)
            if not c:
                continue
            neighbours = c.connects_to | c.depends_on
            # also anything that depends on / connects to this component
            for other in self._components.values():
                if cid in other.connects_to or cid in other.depends_on:
                    neighbours.add(other.id)
            for n in neighbours:
                if n not in seen:
                    seen.add(n)
                    frontier.append((n, dist + 1))
        return seen

    # ---- effective state ---- #
    def effective_capabilities(self, component_id: str) -> Set[str]:
        c = self._require(component_id)
        return set(c.capabilities)

    def is_contained(self, component_id: str) -> bool:
        return self._require(component_id).level >= ContainmentLevel.RESTRICT

    def zone(self, zone: str) -> List[str]:
        return sorted(self._zones.get(zone, set()))

    # ---- release (Master only — containment is sticky) ---- #
    def release(self, component_id: str, *, owner: bool = False) -> Component:
        if not owner:
            raise AuthorizationError("only the Master may release a contained component",
                                     context={"component": component_id})
        c = self._require(component_id)
        c.capabilities = set(c._original_caps)
        c.level = ContainmentLevel.NONE
        c.state = ComponentState.HEALTHY
        c.degraded = False
        c.isolated = False
        if c.zone:
            self._zones.get(c.zone, set()).discard(component_id)
            c.zone = None
        c.reason = ""
        self._emit("release", {"component": component_id})
        return c

    # ---- safe-state snapshot / rollback ---- #
    def snapshot(self) -> str:
        sid = uuid.uuid4().hex[:10]
        self._snapshots[sid] = {
            cid: {"caps": set(c.capabilities), "level": c.level, "state": c.state,
                  "degraded": c.degraded, "isolated": c.isolated, "zone": c.zone}
            for cid, c in self._components.items()}
        self._emit("snapshot", {"id": sid, "components": len(self._components)})
        return sid

    def rollback(self, snapshot_id: str) -> int:
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            raise InvariantViolation(f"no snapshot {snapshot_id!r}")
        n = 0
        for cid, s in snap.items():
            c = self._components.get(cid)
            if c is None:
                continue
            c.capabilities = set(s["caps"])
            c.level = s["level"]
            c.state = s["state"]
            c.degraded = s["degraded"]
            c.isolated = s["isolated"]
            c.zone = s["zone"]
            n += 1
        self._emit("rollback", {"id": snapshot_id, "restored": n})
        return n

    # ---- tamper-evident log ---- #
    def _emit(self, kind: str, detail: Dict[str, Any]) -> None:
        rec = {"kind": kind, "at": time.time(), "detail": detail}
        body = json.dumps(rec, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{self._head}|{len(self._log)}|{body}".encode()).hexdigest()
        self._log.append({"seq": len(self._log), "prev": self._head, "rec": rec,
                          "digest": digest})
        self._head = digest

    def verify_log(self) -> bool:
        prev = _GENESIS
        for e in self._log:
            if e["prev"] != prev:
                raise InvariantViolation("containment log chain broken", context={"seq": e["seq"]})
            body = json.dumps(e["rec"], sort_keys=True, default=str)
            if hashlib.sha256(f"{prev}|{e['seq']}|{body}".encode()).hexdigest() != e["digest"]:
                raise InvariantViolation("containment log tampered", context={"seq": e["seq"]})
            prev = e["digest"]
        return True

    # ---- helpers ---- #
    def _require(self, component_id: str) -> Component:
        c = self._components.get(component_id)
        if c is None:
            raise InvariantViolation(f"no such component {component_id!r}")
        return c

    def report(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        for c in self._components.values():
            by_state[c.state.value] = by_state.get(c.state.value, 0) + 1
        return {"components": len(self._components), "by_state": by_state,
                "zones": {z: len(s) for z, s in self._zones.items()},
                "snapshots": len(self._snapshots), "log_entries": len(self._log)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA defensive-containment self-test")
    print("=" * 70)

    cm = Containment()
    cm.register("kernel", kind="core", capabilities={"read", "write", "modify"})
    cm.register("planner", kind="subsystem", capabilities={"read", "write_plan", "exec_plan"},
                depends_on={"kernel"})
    cm.register("tool_fetch", kind="tool", capabilities={"net_out", "read"},
                connects_to={"planner"})
    cm.register("agent_1", kind="agent", capabilities={"net_out", "write", "spawn"},
                connects_to={"tool_fetch"})

    # a clean snapshot of the healthy system
    safe = cm.snapshot()
    print(f"\nsafe snapshot       : {safe}")

    # RESTRICT a suspect subsystem -> dangerous caps stripped, runs degraded
    a = cm.restrict("planner", reason="anomalous output")
    print(f"\nrestrict planner    : stripped={a.stripped} caps_left={sorted(cm.effective_capabilities('planner'))}")
    assert "write_plan" in a.stripped and "exec_plan" in a.stripped
    assert "read" in cm.effective_capabilities("planner")   # graceful degradation, not death
    assert cm.get("planner").degraded

    # ISOLATE a compromised tool with blast-radius limiting -> neighbours restricted
    a = cm.isolate("tool_fetch", reason="exfil attempt detected", radius=1)
    print(f"isolate tool_fetch  : state={a.state.value} affected={a.affected}")
    assert cm.get("tool_fetch").isolated
    assert "net_out" not in cm.effective_capabilities("tool_fetch")
    assert a.affected   # neighbours were contained to limit the blast radius

    # QUARANTINE a rogue agent into a sealed zone -> all caps gone, frozen
    cm.quarantine("agent_1", zone="rogue", reason="went off-objective")
    print(f"quarantine agent_1  : zone={cm.get('agent_1').zone} caps={cm.effective_capabilities('agent_1')}")
    assert cm.get("agent_1").state is ComponentState.QUARANTINED
    assert cm.effective_capabilities("agent_1") == set()
    assert "agent_1" in cm.zone("rogue")

    # containment is STICKY — only the Master may release
    try:
        cm.release("agent_1", owner=False)
        raise SystemExit("ERROR: non-owner release should be refused")
    except AuthorizationError:
        print("\nrelease guard       : non-Master release refused ✓")
    cm.release("agent_1", owner=True)
    assert cm.get("agent_1").healthy
    assert cm.effective_capabilities("agent_1") == {"net_out", "write", "spawn"}  # restored
    print("owner release       : the Master restored the component ✓")

    # ROLLBACK the whole system to the known-good safe state
    cm.terminate("kernel", reason="(demo) simulated compromise")
    assert cm.get("kernel").state is ComponentState.TERMINATED
    restored = cm.rollback(safe)
    print(f"\nrollback to safe    : restored {restored} components")
    assert cm.get("kernel").healthy and cm.get("planner").healthy

    # purely defensive: TERMINATE only ever targets our own internal component
    assert cm.terminate("planner").component == "planner"
    print("defensive only      : strongest action stops our own component, never attacks ✓")

    # the containment log is tamper-evident
    assert cm.verify_log()
    print(f"\ncontainment log     : {len(cm._log)} entries, chain verified")
    cm._log[0]["rec"]["kind"] = "nothing"
    try:
        cm.verify_log()
        raise SystemExit("ERROR: tamper not detected")
    except InvariantViolation:
        print("tamper detection    : OK (containment history cannot be rewritten)")

    print(f"\nreport              : {cm.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
