"""Tests for nyxara.guard.netsec."""

from __future__ import annotations

import pytest

from nyxara.guard.guardian import ThreatCategory, ThreatEvent
from nyxara.guard.netsec import (Connection, Direction, FirewallRule, NetAction,
                                 NetDecision, NetworkDefense, Reputation)
from nyxara.kernel.errors import AuthorizationError


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _nd(**kw):
    kw.setdefault("clock", FakeClock())
    return NetworkDefense(**kw)


def _in(host, port=443):
    return Connection(host, port=port, direction=Direction.INBOUND)


def _out(host, bytes=0):
    return Connection(host, direction=Direction.OUTBOUND, bytes=bytes)


# -------------------- Reputation -------------------- #
def test_reputation_default_zero():
    assert Reputation().score("x") == 0.0


def test_reputation_worsen_improve():
    r = Reputation()
    r.worsen("x", 0.5)
    assert r.score("x") == 0.5
    r.improve("x", 0.2)
    assert abs(r.score("x") - 0.3) < 1e-9


def test_reputation_clamped():
    r = Reputation()
    r.worsen("x", 5.0)
    assert r.score("x") == 1.0
    r.improve("x", 5.0)
    assert r.score("x") == 0.0


def test_reputation_is_hostile():
    r = Reputation()
    r.worsen("x", 0.8)
    assert r.is_hostile("x")
    assert not r.is_hostile("y")


# -------------------- FirewallRule matching -------------------- #
def test_rule_matches_exact_host():
    rule = FirewallRule("r", NetAction.BLOCK, host="1.2.3.4")
    assert rule.matches(_in("1.2.3.4"))
    assert not rule.matches(_in("1.2.3.5"))


def test_rule_matches_wildcard():
    assert FirewallRule("r", NetAction.BLOCK, host="*").matches(_in("anything"))


def test_rule_matches_cidr():
    rule = FirewallRule("r", NetAction.BLOCK, host="10.0.0.0/8")
    assert rule.matches(_in("10.5.5.5"))
    assert not rule.matches(_in("11.0.0.1"))


def test_rule_matches_glob():
    rule = FirewallRule("r", NetAction.BLOCK, host="*.evil.com")
    assert rule.matches(_in("a.evil.com"))
    assert not rule.matches(_in("a.good.com"))


def test_rule_direction_filter():
    rule = FirewallRule("r", NetAction.BLOCK, direction=Direction.OUTBOUND)
    assert not rule.matches(_in("x"))
    assert rule.matches(_out("x"))


def test_rule_port_filter():
    rule = FirewallRule("r", NetAction.BLOCK, port=22)
    assert rule.matches(_in("x", port=22))
    assert not rule.matches(_in("x", port=80))


def test_rule_disabled():
    rule = FirewallRule("r", NetAction.BLOCK, enabled=False)
    assert not rule.matches(_in("x"))


# -------------------- evaluate: lists & reputation -------------------- #
def test_default_inbound_allowed():
    assert _nd().evaluate(_in("1.2.3.4")).allowed


def test_allowlist_bypasses():
    nd = _nd()
    nd.allow_host("trusted", owner=True)
    nd.deny_host("trusted")  # even denial can't override allow-list precedence
    assert nd.evaluate(_in("trusted")).allowed


def test_denylist_blocks():
    nd = _nd()
    nd.deny_host("bad")
    assert nd.evaluate(_in("bad")).blocked


def test_hostile_reputation_blocks():
    nd = _nd()
    nd.report_bad("h", amount=0.8)
    d = nd.evaluate(_in("h"))
    assert d.blocked and "hostile" in d.reason


def test_allowlist_requires_owner():
    with pytest.raises(AuthorizationError):
        _nd().allow_host("x", owner=False)


# -------------------- evaluate: rules -------------------- #
def test_rule_block_applies():
    nd = _nd()
    nd.add_rule(NetAction.BLOCK, host="9.9.9.9", reason="bad")
    d = nd.evaluate(_in("9.9.9.9"))
    assert d.blocked and d.rule_id is not None


def test_rule_priority_order():
    nd = _nd()
    nd.add_rule(NetAction.ALLOW, host="*", priority=1)
    nd.add_rule(NetAction.BLOCK, host="1.2.3.4", priority=10)  # higher priority
    assert nd.evaluate(_in("1.2.3.4")).blocked


def test_rule_throttle():
    nd = _nd()
    nd.add_rule(NetAction.THROTTLE, host="x")
    assert nd.evaluate(_in("x")).action is NetAction.THROTTLE


# -------------------- inbound rate limiting -------------------- #
def test_inbound_rate_limit():
    nd = _nd(inbound_capacity=3, inbound_per_sec=1)
    results = [nd.evaluate(_in("flood")) for _ in range(6)]
    assert any(r.action is NetAction.THROTTLE for r in results)


def test_inbound_rate_recovers():
    clk = FakeClock()
    nd = NetworkDefense(clock=clk, inbound_capacity=2, inbound_per_sec=1)
    for _ in range(3):
        nd.evaluate(_in("h"))  # drains bucket
    clk.advance(5.0)  # refill
    assert nd.evaluate(_in("h")).allowed


# -------------------- egress control -------------------- #
def test_egress_within_budget_allowed():
    nd = _nd(egress_budget=1_000_000)
    assert nd.evaluate(_out("dest", bytes=500_000)).allowed


def test_egress_over_budget_blocked():
    nd = _nd(egress_budget=1_000_000)
    nd.evaluate(_out("dest", bytes=600_000))
    d = nd.evaluate(_out("dest", bytes=600_000))
    assert d.blocked and "exfil" in d.reason.lower()


def test_egress_allowlisted_dest_unbudgeted():
    nd = _nd(egress_budget=100)
    nd.allow_host("backup.server", owner=True)
    assert nd.evaluate(_out("backup.server", bytes=10_000_000)).allowed


def test_outbound_no_bytes_allowed():
    assert _nd().evaluate(_out("x", bytes=0)).allowed


# -------------------- defensive response -------------------- #
def test_block_ip():
    nd = _nd()
    d = nd.block_ip("attacker", reason="intrusion")
    assert d.blocked
    assert nd.evaluate(_in("attacker")).blocked


def test_lift_block_requires_owner():
    nd = _nd()
    nd.block_ip("x")
    with pytest.raises(AuthorizationError):
        nd.lift_block("x", owner=False)


def test_lift_block_by_owner():
    nd = _nd()
    nd.block_ip("x")
    assert nd.lift_block("x", owner=True)
    assert nd.evaluate(_in("x")).allowed


def test_lift_unknown_returns_false():
    assert not _nd().lift_block("never", owner=True)


# -------------------- guardian bridge -------------------- #
def test_to_threat_event_inbound():
    nd = _nd()
    conn = _in("bad", port=22)
    nd.block_ip("bad")
    d = nd.evaluate(conn)
    ev = nd.to_threat_event(conn, d)
    assert isinstance(ev, ThreatEvent) and ev.category is ThreatCategory.INTRUSION


def test_to_threat_event_outbound_exfil():
    nd = _nd(egress_budget=100)
    nd.evaluate(_out("x", bytes=200))  # over budget
    conn = _out("x", bytes=200)
    d = nd.evaluate(conn)
    ev = nd.to_threat_event(conn, d)
    assert ev.category is ThreatCategory.DATA_EXFIL


# -------------------- exposure auditing -------------------- #
def test_audit_public_sensitive_high():
    nd = _nd()
    nd.register_service(22, "ssh", exposure="public")
    findings = nd.audit_exposure()
    assert any(f["port"] == 22 and f["risk"] == "high" for f in findings)


def test_audit_public_nonsensitive_moderate():
    nd = _nd()
    nd.register_service(8080, "api", exposure="public")
    findings = nd.audit_exposure()
    assert findings[0]["risk"] == "moderate"


def test_audit_local_not_flagged():
    nd = _nd()
    nd.register_service(22, "ssh", exposure="local")
    assert nd.audit_exposure() == []


def test_audit_orders_high_first():
    nd = _nd()
    nd.register_service(8080, "api", exposure="public")
    nd.register_service(22, "ssh", exposure="public")
    findings = nd.audit_exposure()
    assert findings[0]["risk"] == "high"


# -------------------- shapes / reporting -------------------- #
def test_decision_to_dict():
    d = NetDecision(NetAction.BLOCK, "x", reputation=0.5)
    assert d.to_dict()["action"] == "block"


def test_rule_to_dict():
    nd = _nd()
    r = nd.add_rule(NetAction.BLOCK, host="x")
    assert r.to_dict()["action"] == "block"


def test_report():
    nd = _nd()
    nd.add_rule(NetAction.BLOCK, host="x")
    nd.deny_host("y")
    nd.evaluate(_in("z"))
    r = nd.report()
    assert r["rules"] == 1 and r["deny"] == 1 and "decisions" in r
