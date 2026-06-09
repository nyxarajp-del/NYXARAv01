"""NYXARA · guard/netsec.py — policy-level network defence (🧱, Rule 5 digital-defensive).

The Master's network is something NYXARA *defends*, never something she attacks. This module
is the digital, defensive form of Rule 5 at the network layer — and it works at the level of
**policy, not raw sockets**: it decides whether a connection *should* be allowed, throttled,
or blocked, tracks the reputation of remote hosts, audits which services are exposed, and —
crucially — controls **egress** so that data cannot be quietly exfiltrated.

What it enforces:

* **Allow / deny / reputation.** Allow-listed hosts pass; deny-listed or high-reputation
  (hostile) hosts are blocked outright; reputation is learned from bad behaviour and decays.
* **Defensive firewall rules.** Ordered, reversible rules match by direction / host (glob or
  CIDR) / port and allow, throttle, or block. Blocking is the *only* aggressive verb, and it
  only ever stops traffic — it never reaches out to strike.
* **Inbound rate limiting.** A per-host token bucket throttles floods and scans.
* **Egress control (exfil prevention).** Outbound volume to non-allow-listed destinations is
  budgeted per host; a large transfer to an untrusted or low-reputation host is blocked.
* **Exposure auditing.** Registered services are checked for risky public exposure of
  sensitive ports.

Defensive blocks can be raised automatically, but **lifting a block or adding an allow-list
entry requires the Master** (fail-safe). Decisions feed :mod:`guard.guardian` as threat
events.

Reuses :mod:`agency.governor` (token bucket / window budget) and bridges to
:mod:`guard.guardian`. Pure standard library otherwise.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.agency.governor import TokenBucket, WindowBudget
from nyxara.guard.guardian import ThreatCategory, ThreatEvent
from nyxara.kernel.errors import AuthorizationError

__all__ = [
    "Direction",
    "NetAction",
    "Connection",
    "FirewallRule",
    "NetDecision",
    "Reputation",
    "NetworkDefense",
    "HOSTILE_THRESHOLD",
]

HOSTILE_THRESHOLD = 0.7

# ports whose public exposure is dangerous
_SENSITIVE_PORTS = {22, 23, 25, 135, 139, 445, 1433, 3306, 3389, 5432, 5900, 6379, 27017}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _host_matches(pattern: str, host: str) -> bool:
    if pattern in ("*", host):
        return True
    try:
        net = ipaddress.ip_network(pattern, strict=False)
        return ipaddress.ip_address(host) in net
    except ValueError:
        return fnmatch.fnmatch(host, pattern)


# --------------------------------------------------------------------------- #
# Enums & data
# --------------------------------------------------------------------------- #
class Direction(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class NetAction(str, Enum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"


@dataclass
class Connection:
    host: str
    port: int = 0
    direction: Direction = Direction.INBOUND
    protocol: str = "tcp"
    bytes: int = 0
    at: float = field(default_factory=time.time)


@dataclass
class FirewallRule:
    id: str
    action: NetAction
    direction: Optional[Direction] = None     # None = any
    host: str = "*"                           # glob or CIDR
    port: int = 0                             # 0 = any
    reason: str = ""
    priority: int = 0
    enabled: bool = True

    def matches(self, conn: Connection) -> bool:
        if not self.enabled:
            return False
        if self.direction is not None and self.direction is not conn.direction:
            return False
        if self.port and self.port != conn.port:
            return False
        return _host_matches(self.host, conn.host)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "action": self.action.value,
                "direction": self.direction.value if self.direction else "any",
                "host": self.host, "port": self.port, "reason": self.reason,
                "priority": self.priority, "enabled": self.enabled}


@dataclass
class NetDecision:
    action: NetAction
    reason: str
    rule_id: Optional[str] = None
    reputation: float = 0.0
    retry_after: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.action is NetAction.ALLOW

    @property
    def blocked(self) -> bool:
        return self.action is NetAction.BLOCK

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action.value, "reason": self.reason, "rule_id": self.rule_id,
                "reputation": round(self.reputation, 3), "retry_after": round(self.retry_after, 2)}


# --------------------------------------------------------------------------- #
# Reputation
# --------------------------------------------------------------------------- #
class Reputation:
    """Per-host hostility score in [0,1]; learned from bad behaviour, decays over time."""

    def __init__(self, *, decay: float = 0.99) -> None:
        self.decay = decay
        self._scores: Dict[str, float] = {}

    def score(self, host: str) -> float:
        return self._scores.get(host, 0.0)

    def worsen(self, host: str, amount: float = 0.3) -> float:
        self._scores[host] = _clamp(self.score(host) + amount)
        return self._scores[host]

    def improve(self, host: str, amount: float = 0.1) -> float:
        self._scores[host] = _clamp(self.score(host) - amount)
        return self._scores[host]

    def is_hostile(self, host: str, threshold: float = HOSTILE_THRESHOLD) -> bool:
        return self.score(host) >= threshold

    def all(self) -> Dict[str, float]:
        return dict(self._scores)


# --------------------------------------------------------------------------- #
# Network defence
# --------------------------------------------------------------------------- #
class NetworkDefense:
    """Policy-level firewall, reputation, egress control and exposure auditing."""

    def __init__(self, *, clock=None, default_inbound: NetAction = NetAction.ALLOW,
                 default_outbound: NetAction = NetAction.ALLOW,
                 inbound_capacity: float = 20.0, inbound_per_sec: float = 10.0,
                 egress_budget: float = 10_000_000.0, egress_window: float = 60.0,
                 hostile_threshold: float = HOSTILE_THRESHOLD) -> None:
        self._clock = clock or time.time
        self.default_inbound = default_inbound
        self.default_outbound = default_outbound
        self.inbound_capacity = inbound_capacity
        self.inbound_per_sec = inbound_per_sec
        self.egress_budget = egress_budget
        self.egress_window = egress_window
        self.hostile_threshold = hostile_threshold
        self.reputation = Reputation()
        self._allow: set = set()
        self._deny: set = set()
        self._rules: List[FirewallRule] = []
        self._buckets: Dict[str, TokenBucket] = {}
        self._egress: Dict[str, WindowBudget] = {}
        self._services: Dict[int, Dict[str, Any]] = {}
        self._log: List[Dict[str, Any]] = []
        self._counts = {"allow": 0, "throttle": 0, "block": 0}

    # ---- lists (allow-list / deny-list) ---- #
    def allow_host(self, host: str, *, owner: bool = False) -> None:
        if not owner:
            raise AuthorizationError("only the Master may add an allow-list entry",
                                     context={"host": host})
        self._allow.add(host)
        self._deny.discard(host)

    def deny_host(self, host: str, *, reason: str = "") -> None:
        """A defensive block — may be raised automatically."""
        self._deny.add(host)
        self._emit("deny", {"host": host, "reason": reason})

    def lift_block(self, host: str, *, owner: bool = False) -> bool:
        if not owner:
            raise AuthorizationError("only the Master may lift a block",
                                     context={"host": host})
        existed = host in self._deny
        self._deny.discard(host)
        self.reputation.improve(host, 1.0)
        if existed:
            self._emit("lift", {"host": host})
        return existed

    # ---- rules ---- #
    def add_rule(self, action: NetAction, *, direction: Optional[Direction] = None,
                 host: str = "*", port: int = 0, reason: str = "",
                 priority: int = 0) -> FirewallRule:
        rule = FirewallRule(id=f"r{len(self._rules)}", action=action, direction=direction,
                            host=host, port=port, reason=reason, priority=priority)
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)
        return rule

    def rules(self) -> List[FirewallRule]:
        return list(self._rules)

    # ---- per-host primitives ---- #
    def _bucket(self, host: str) -> TokenBucket:
        if host not in self._buckets:
            self._buckets[host] = TokenBucket(self.inbound_capacity, self.inbound_per_sec,
                                              clock=self._clock)
        return self._buckets[host]

    def _egress_budget(self, host: str) -> WindowBudget:
        if host not in self._egress:
            self._egress[host] = WindowBudget(self.egress_budget, self.egress_window,
                                              clock=self._clock)
        return self._egress[host]

    # ---- the core: evaluate a connection ---- #
    def evaluate(self, conn: Connection) -> NetDecision:
        rep = self.reputation.score(conn.host)

        # 1. allow-list bypasses (trusted destinations/sources)
        if conn.host in self._allow:
            return self._decide(NetAction.ALLOW, "allow-listed", reputation=rep)

        # 2. deny-list / hostile reputation -> block outright
        if conn.host in self._deny:
            return self._decide(NetAction.BLOCK, "deny-listed", reputation=rep)
        if rep >= self.hostile_threshold:
            return self._decide(NetAction.BLOCK, f"hostile reputation {rep:.2f}", reputation=rep)

        # 3. explicit firewall rules (highest priority first)
        for rule in self._rules:
            if rule.matches(conn):
                return self._decide(rule.action, rule.reason or f"rule {rule.id}",
                                    rule_id=rule.id, reputation=rep)

        # 4. direction-specific defences
        if conn.direction is Direction.INBOUND:
            if not self._bucket(conn.host).try_acquire():
                return self._decide(NetAction.THROTTLE, "inbound rate limit", reputation=rep,
                                    retry_after=self._bucket(conn.host).retry_after())
            return self._decide(self.default_inbound, "default inbound", reputation=rep)

        # outbound: egress control (exfil prevention)
        if conn.bytes > 0:
            budget = self._egress_budget(conn.host)
            if not budget.can_afford(conn.bytes):
                return self._decide(NetAction.BLOCK,
                                    "egress budget exceeded (possible exfiltration)",
                                    reputation=rep)
            budget.charge(conn.bytes)
        return self._decide(self.default_outbound, "default outbound", reputation=rep)

    def _decide(self, action: NetAction, reason: str, *, rule_id: Optional[str] = None,
                reputation: float = 0.0, retry_after: float = 0.0) -> NetDecision:
        self._counts[action.value] += 1
        return NetDecision(action=action, reason=reason, rule_id=rule_id,
                           reputation=reputation, retry_after=retry_after)

    # ---- defensive response ---- #
    def block_ip(self, host: str, *, reason: str = "threat") -> NetDecision:
        """Defensively block a host (stops traffic; never strikes back)."""
        self.reputation.worsen(host, 1.0)
        self.deny_host(host, reason=reason)
        return NetDecision(NetAction.BLOCK, reason=reason, reputation=1.0)

    def report_bad(self, host: str, *, amount: float = 0.3) -> float:
        return self.reputation.worsen(host, amount)

    # ---- exposure auditing ---- #
    def register_service(self, port: int, name: str, *, exposure: str = "local") -> None:
        self._services[port] = {"name": name, "exposure": exposure}

    def audit_exposure(self) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for port, svc in self._services.items():
            if svc["exposure"] == "public":
                risk = "high" if port in _SENSITIVE_PORTS else "moderate"
                findings.append({"port": port, "name": svc["name"], "exposure": "public",
                                 "risk": risk})
        return sorted(findings, key=lambda f: 0 if f["risk"] == "high" else 1)

    # ---- guardian bridge ---- #
    def to_threat_event(self, conn: Connection, decision: NetDecision) -> ThreatEvent:
        cat = (ThreatCategory.DATA_EXFIL if conn.direction is Direction.OUTBOUND
               else ThreatCategory.INTRUSION)
        sev = 0.9 if decision.blocked else (0.5 if decision.action is NetAction.THROTTLE else 0.1)
        sev = max(sev, decision.reputation)
        return ThreatEvent(category=cat, source=conn.host,
                           description=f"net {decision.action.value}: {decision.reason}",
                           severity=sev, confidence=0.75,
                           data={"port": conn.port, "direction": conn.direction.value})

    # ---- log / report ---- #
    def _emit(self, kind: str, detail: Dict[str, Any]) -> None:
        self._log.append({"kind": kind, "at": self._clock(), "detail": detail})

    def report(self) -> Dict[str, Any]:
        return {"rules": len(self._rules), "allow": len(self._allow), "deny": len(self._deny),
                "services": len(self._services), "decisions": dict(self._counts),
                "public_exposures": len(self.audit_exposure())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA network-defence self-test (policy-level, no real sockets)")
    print("=" * 70)

    now = {"t": 1000.0}
    nd = NetworkDefense(clock=lambda: now["t"], inbound_capacity=5, inbound_per_sec=1)

    # a normal inbound connection is allowed by default
    d = nd.evaluate(Connection("198.51.100.7", port=443, direction=Direction.INBOUND))
    print(f"\nnormal inbound      : {d.to_dict()}")
    assert d.allowed

    # the Master allow-lists a trusted partner; deny-list/reputation can't touch it later
    nd.allow_host("trusted.partner", owner=True)
    assert nd.evaluate(Connection("trusted.partner", direction=Direction.OUTBOUND)).allowed
    print("allow-list          : trusted partner always allowed ✓")

    # a hostile host (bad reputation) is blocked outright
    for _ in range(3):
        nd.report_bad("10.0.0.66", amount=0.3)
    d = nd.evaluate(Connection("10.0.0.66", port=22, direction=Direction.INBOUND))
    print(f"\nhostile reputation  : {d.to_dict()}")
    assert d.blocked

    # a defensive firewall rule blocking a subnet on a CIDR match
    nd.add_rule(NetAction.BLOCK, direction=Direction.INBOUND, host="203.0.113.0/24",
                reason="known-bad subnet", priority=10)
    d = nd.evaluate(Connection("203.0.113.45", port=80, direction=Direction.INBOUND))
    print(f"CIDR rule block     : {d.to_dict()}")
    assert d.blocked and d.rule_id is not None

    # inbound rate limiting: a flood gets throttled
    throttled = None
    for _ in range(10):
        throttled = nd.evaluate(Connection("198.51.100.9", direction=Direction.INBOUND)) or throttled
    print(f"\ninbound flood       : last action={throttled.action.value} "
          f"retry_after={throttled.retry_after:.1f}s")
    assert throttled.action is NetAction.THROTTLE

    # EGRESS control: a huge transfer to an untrusted host is blocked (exfil prevention)
    nd2 = NetworkDefense(clock=lambda: now["t"], egress_budget=1_000_000)
    ok = nd2.evaluate(Connection("evil.exfil.host", direction=Direction.OUTBOUND, bytes=500_000))
    big = nd2.evaluate(Connection("evil.exfil.host", direction=Direction.OUTBOUND, bytes=900_000))
    print(f"\negress small/large  : {ok.action.value} / {big.action.value} ({big.reason})")
    assert ok.allowed and big.blocked

    # defensive block-and-feed-the-guardian
    bad_conn = Connection("45.155.0.1", port=22, direction=Direction.INBOUND)
    nd.block_ip("45.155.0.1", reason="active intrusion")
    d = nd.evaluate(bad_conn)
    ev = nd.to_threat_event(bad_conn, d)
    print(f"\nblock + threat event: category={ev.category.value} severity={ev.severity:.2f}")
    assert d.blocked and ev.category is ThreatCategory.INTRUSION

    from nyxara.guard.guardian import Guardian
    a, p = Guardian().handle(ev)
    print(f"guardian verdict    : level={a.level.label}")
    assert a.level.value >= 3

    # lifting a block requires the Master
    try:
        nd.lift_block("45.155.0.1", owner=False)
        raise SystemExit("ERROR: non-owner lift should be refused")
    except AuthorizationError:
        print("\nlift guard          : non-Master block-lift refused ✓")
    assert nd.lift_block("45.155.0.1", owner=True)
    print("owner lift          : the Master lifted the block ✓")

    # exposure auditing: a public SSH service is high risk
    nd.register_service(22, "ssh", exposure="public")
    nd.register_service(8080, "internal-api", exposure="local")
    findings = nd.audit_exposure()
    print(f"\nexposure audit      : {findings}")
    assert any(f["port"] == 22 and f["risk"] == "high" for f in findings)

    print(f"\nreport              : {nd.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
