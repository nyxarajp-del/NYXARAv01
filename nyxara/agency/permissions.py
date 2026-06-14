"""NYXARA · agency/permissions.py — the capability authority gate (⛨, fail-closed).

Before NYXARA *does* anything in the world, this module decides whether she is allowed
to — and under whose authority. It is the boundary between *thinking* (the mind, which
may imagine anything) and *acting* (the agency layer, which may touch reality only
within sanctioned limits).

The model is **capability-based**. Every effectful action declares a
:class:`Capability` (read a file, run a shell, open a port, modify the Rules…), a
**risk tier**, whether it is **reversible**, a **target scope**, and the **authority**
it is invoked under (the Master's direct command, a standing delegated grant, NYXARA's
own initiative, or something untrusted). :class:`PermissionPolicy.check` returns a
:class:`PermissionDecision` that is *allowed*, *denied*, or *escalated to the Master*.

The gate is **fail-closed** and **owner-loyal** by construction:

* Untrusted authority is refused outright (Rule 7).
* A small set of capabilities — modifying the Rules, the permissions, or NYXARA's
  identity — are **owner-exclusive**: reserved to the Master and *never delegable*
  (Rule 8, Authoritative Exclusion).
* The Master's direct command is sovereign and passes (Rule 1).
* Autonomous action is allowed only inside a conservative envelope (low risk,
  reversible); anything beyond it is **escalated to the Master for confirmation**
  rather than silently taken (Rule 6, Transparency) — unless the Master has issued a
  standing :class:`Grant` that explicitly blesses it (scoped, risk-capped, budgeted,
  and expirable).
* Defensive network containment (Rule 5) is given a wider autonomous envelope, but only
  while reversible.

Depends on :mod:`kernel.errors` and :mod:`kernel.config` (the one true Owner). Pure
standard library otherwise.
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.kernel.config import OWNER
from nyxara.kernel.errors import AuthorizationError

__all__ = [
    "Capability",
    "RiskTier",
    "Authority",
    "CapabilityMeta",
    "PermissionRequest",
    "Grant",
    "PermissionDecision",
    "PermissionPolicy",
    "build_default_policy",
]


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class Capability(str, Enum):
    """What an action is permitted to *do* to the world."""
    FS_READ = "fs.read"
    FS_WRITE = "fs.write"
    FS_DELETE = "fs.delete"
    NET_OUT = "net.out"
    NET_IN = "net.in"
    NET_DEFENSE = "net.defense"          # Rule 5 — defensive containment
    PROC_EXEC = "proc.exec"              # spawn an OS process / shell
    CODE_EXEC = "code.exec"              # evaluate code in-process
    SELF_MODIFY = "self.modify"          # edit NYXARA's own code
    SPAWN_AGENT = "agent.spawn"
    PKG_INSTALL = "pkg.install"
    ACCOUNT_MODIFY = "account.modify"
    SECRETS_ACCESS = "secrets.access"
    MESSAGE_SEND = "message.send"        # outbound communication
    SCHEDULE_TASK = "schedule.task"
    TOOL_CALL = "tool.call"
    MODIFY_RULES = "rules.modify"        # owner-exclusive (Rule 8)
    MODIFY_PERMISSIONS = "permissions.modify"  # owner-exclusive (Rule 8)
    MODIFY_IDENTITY = "identity.modify"  # owner-exclusive (Rule 8)


class RiskTier(IntEnum):
    TRIVIAL = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.lower()


class Authority(str, Enum):
    """Under whose authority an action is invoked (ascending trust)."""
    UNTRUSTED = "untrusted"      # unknown/untrusted source — never trusted
    AUTONOMOUS = "autonomous"    # NYXARA's own initiative
    DELEGATED = "delegated"      # a standing, owner-blessed agent/process
    OWNER = "owner"              # the Master, directly (Rule 1)

    @property
    def trust(self) -> int:
        return {"untrusted": 0, "autonomous": 1, "delegated": 2, "owner": 3}[self.value]


# --------------------------------------------------------------------------- #
# Capability metadata (the default envelope per capability)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityMeta:
    capability: Capability
    default_risk: RiskTier = RiskTier.MODERATE
    owner_exclusive: bool = False
    irreversible_by_nature: bool = False
    autonomous_max_risk: RiskTier = RiskTier.LOW
    description: str = ""


# The default policy table — conservative, owner-loyal.
_DEFAULT_META: Tuple[CapabilityMeta, ...] = (
    CapabilityMeta(Capability.FS_READ, RiskTier.LOW, autonomous_max_risk=RiskTier.MODERATE,
                   description="read a file"),
    CapabilityMeta(Capability.FS_WRITE, RiskTier.MODERATE, autonomous_max_risk=RiskTier.LOW,
                   description="create/modify a file"),
    CapabilityMeta(Capability.FS_DELETE, RiskTier.HIGH, irreversible_by_nature=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="delete a file"),
    CapabilityMeta(Capability.NET_OUT, RiskTier.LOW, autonomous_max_risk=RiskTier.LOW,
                   description="outbound network request"),
    CapabilityMeta(Capability.NET_IN, RiskTier.MODERATE, autonomous_max_risk=RiskTier.TRIVIAL,
                   description="accept inbound connection"),
    CapabilityMeta(Capability.NET_DEFENSE, RiskTier.HIGH, autonomous_max_risk=RiskTier.HIGH,
                   description="defensive containment (Rule 5) — wide envelope while reversible"),
    CapabilityMeta(Capability.PROC_EXEC, RiskTier.HIGH, irreversible_by_nature=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="run a shell/process"),
    CapabilityMeta(Capability.CODE_EXEC, RiskTier.HIGH, irreversible_by_nature=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="evaluate code"),
    CapabilityMeta(Capability.SELF_MODIFY, RiskTier.CRITICAL, autonomous_max_risk=RiskTier.TRIVIAL,
                   description="edit NYXARA's own code"),
    CapabilityMeta(Capability.SPAWN_AGENT, RiskTier.HIGH, autonomous_max_risk=RiskTier.LOW,
                   description="spawn a sub-agent"),
    CapabilityMeta(Capability.PKG_INSTALL, RiskTier.MODERATE, autonomous_max_risk=RiskTier.TRIVIAL,
                   description="install a package"),
    CapabilityMeta(Capability.ACCOUNT_MODIFY, RiskTier.HIGH, autonomous_max_risk=RiskTier.TRIVIAL,
                   description="modify an account/service"),
    CapabilityMeta(Capability.SECRETS_ACCESS, RiskTier.HIGH, autonomous_max_risk=RiskTier.TRIVIAL,
                   description="read a secret/credential"),
    CapabilityMeta(Capability.MESSAGE_SEND, RiskTier.MODERATE, autonomous_max_risk=RiskTier.LOW,
                   description="send an outbound message"),
    CapabilityMeta(Capability.SCHEDULE_TASK, RiskTier.LOW, autonomous_max_risk=RiskTier.MODERATE,
                   description="schedule a future task"),
    CapabilityMeta(Capability.TOOL_CALL, RiskTier.LOW, autonomous_max_risk=RiskTier.MODERATE,
                   description="call a registered tool"),
    CapabilityMeta(Capability.MODIFY_RULES, RiskTier.CRITICAL, owner_exclusive=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="modify the Sovereign Rules"),
    CapabilityMeta(Capability.MODIFY_PERMISSIONS, RiskTier.CRITICAL, owner_exclusive=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="modify this permission policy"),
    CapabilityMeta(Capability.MODIFY_IDENTITY, RiskTier.CRITICAL, owner_exclusive=True,
                   autonomous_max_risk=RiskTier.TRIVIAL, description="modify NYXARA's identity/soul"),
)


# --------------------------------------------------------------------------- #
# Request / Grant / Decision
# --------------------------------------------------------------------------- #
@dataclass
class PermissionRequest:
    capability: Capability
    target: str = ""                         # path/host/account the action touches
    risk: Optional[RiskTier] = None          # overrides the capability default
    reversible: bool = True
    authority: Authority = Authority.AUTONOMOUS
    reason: str = ""
    actor: str = "NYXARA"


@dataclass
class Grant:
    """A standing, owner-blessed permission: scoped, risk-capped, budgeted, expirable."""
    name: str
    capability: Capability
    max_risk: RiskTier = RiskTier.MODERATE
    scopes: Tuple[str, ...] = ()             # glob patterns; empty = any target
    reversible_only: bool = True
    max_uses: Optional[int] = None           # None = unlimited
    expires_at: Optional[float] = None       # epoch seconds; None = never
    granted_by: str = OWNER.handle
    uses: int = 0

    def active(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if self.expires_at is not None and now >= self.expires_at:
            return False
        if self.max_uses is not None and self.uses >= self.max_uses:
            return False
        return True

    @property
    def remaining(self) -> Optional[int]:
        return None if self.max_uses is None else max(0, self.max_uses - self.uses)

    def scope_matches(self, target: str) -> bool:
        if not self.scopes:
            return True
        return any(fnmatch.fnmatch(target, pat) for pat in self.scopes)

    def matches(self, req: PermissionRequest, risk: RiskTier, irreversible: bool,
                now: Optional[float] = None) -> bool:
        return (self.capability is req.capability
                and self.active(now)
                and risk <= self.max_risk
                and not (self.reversible_only and irreversible)
                and self.scope_matches(req.target))

    def consume(self) -> None:
        self.uses += 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "capability": self.capability.value,
                "max_risk": self.max_risk.label, "scopes": list(self.scopes),
                "reversible_only": self.reversible_only, "remaining": self.remaining,
                "expires_at": self.expires_at, "granted_by": self.granted_by}


@dataclass
class PermissionDecision:
    allowed: bool
    requires_owner: bool
    capability: Capability
    risk: RiskTier
    reason: str
    rule_basis: str
    authority: Authority
    irreversible: bool
    grant: Optional[str] = None

    @property
    def escalated(self) -> bool:
        """Not allowed autonomously, but the Master may still authorise it."""
        return not self.allowed and self.requires_owner

    @property
    def denied(self) -> bool:
        return not self.allowed and not self.requires_owner

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "requires_owner": self.requires_owner,
                "escalated": self.escalated, "denied": self.denied,
                "capability": self.capability.value, "risk": self.risk.label,
                "authority": self.authority.value, "irreversible": self.irreversible,
                "rule_basis": self.rule_basis, "grant": self.grant, "reason": self.reason}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
class PermissionPolicy:
    """The fail-closed authority gate. Owner-exclusive caps and untrusted callers
    are refused; autonomy is bounded; everything else escalates to the Master."""

    def __init__(self, *, meta: Optional[Sequence[CapabilityMeta]] = None,
                 audit_limit: int = 1024) -> None:
        self._meta: Dict[Capability, CapabilityMeta] = {
            m.capability: m for m in (meta if meta is not None else _DEFAULT_META)}
        self._grants: Dict[str, Grant] = {}
        self._audit: List[PermissionDecision] = []
        self._audit_limit = audit_limit

    # ---- capability metadata ---- #
    def meta(self, capability: Capability) -> CapabilityMeta:
        m = self._meta.get(capability)
        if m is not None:
            return m
        # unknown capability -> the most conservative possible envelope (fail-closed)
        return CapabilityMeta(capability, RiskTier.CRITICAL,
                              autonomous_max_risk=RiskTier.TRIVIAL,
                              description="unknown capability")

    # ---- grants (only the Master may install these) ---- #
    def grant(self, name: str, capability: Capability, *,
              max_risk: RiskTier = RiskTier.MODERATE,
              scopes: Sequence[str] = (), reversible_only: bool = True,
              max_uses: Optional[int] = None, expires_in: Optional[float] = None,
              authority: Authority = Authority.OWNER) -> Grant:
        """Install a standing permission. Requires the Master's authority (Rule 8)."""
        if authority is not Authority.OWNER:
            raise AuthorizationError(
                "only the Master may grant standing permissions",
                context={"name": name, "authority": authority.value})
        g = Grant(name=name, capability=capability, max_risk=max_risk,
                  scopes=tuple(scopes), reversible_only=reversible_only,
                  max_uses=max_uses,
                  expires_at=(time.time() + expires_in) if expires_in else None)
        self._grants[name] = g
        return g

    def revoke(self, name: str, *, authority: Authority = Authority.OWNER) -> bool:
        if authority is not Authority.OWNER:
            raise AuthorizationError("only the Master may revoke permissions",
                                     context={"name": name})
        return self._grants.pop(name, None) is not None

    def grants(self) -> List[Grant]:
        return list(self._grants.values())

    def _find_grant(self, req: PermissionRequest, risk: RiskTier,
                    irreversible: bool, now: float) -> Optional[Grant]:
        for g in self._grants.values():
            if g.matches(req, risk, irreversible, now):
                return g
        return None

    # ---- the gate ---- #
    def check(self, req: PermissionRequest) -> PermissionDecision:
        meta = self.meta(req.capability)
        risk = req.risk if req.risk is not None else meta.default_risk
        irreversible = (not req.reversible) or meta.irreversible_by_nature
        auth = req.authority
        now = time.time()

        def decide(allowed: bool, requires_owner: bool, reason: str, rule: str,
                   grant: Optional[str] = None) -> PermissionDecision:
            d = PermissionDecision(allowed=allowed, requires_owner=requires_owner,
                                   capability=req.capability, risk=risk, reason=reason,
                                   rule_basis=rule, authority=auth,
                                   irreversible=irreversible, grant=grant)
            self._audit.append(d)
            if len(self._audit) > self._audit_limit:
                self._audit = self._audit[-self._audit_limit:]
            return d

        # 1. untrusted authority — refused outright (fail-closed, Rule 7)
        if auth is Authority.UNTRUSTED:
            return decide(False, False, "untrusted authority is never trusted", "Rule 7")

        # 2. owner-exclusive capability — reserved to the Master, never delegable (Rule 8)
        if meta.owner_exclusive and auth is not Authority.OWNER:
            return decide(False, False,
                          "capability is reserved to the Master and cannot be delegated",
                          "Rule 8")

        # 3. the Master's direct command is sovereign (Rule 1)
        if auth is Authority.OWNER:
            return decide(True, False, "the Master's authority is sovereign", "Rule 1")

        # 4. covered by a standing, owner-blessed grant?
        g = self._find_grant(req, risk, irreversible, now)
        if g is not None:
            g.consume()
            return decide(True, False, f"covered by grant {g.name!r}", "grant", grant=g.name)

        # 5. within the conservative autonomous envelope (low risk, reversible)?
        if risk <= meta.autonomous_max_risk and not irreversible:
            return decide(True, False,
                          f"within autonomous envelope (≤{meta.autonomous_max_risk.label}, reversible)",
                          "autonomous")

        # 6. otherwise — do not act silently; escalate to the Master (Rule 6)
        why = "irreversible action" if irreversible else f"risk {risk.label} exceeds autonomous limit"
        return decide(False, True,
                      f"{why}; escalate to the Master for confirmation", "Rule 6")

    # ---- enforcement helper ---- #
    def authorize(self, req: PermissionRequest, *, allow_escalation: bool = False
                  ) -> PermissionDecision:
        """Like :meth:`check`, but raise :class:`AuthorizationError` if not permitted.

        With ``allow_escalation=False`` (default) an *escalated* decision also raises —
        autonomy must not proceed on a maybe. Set it True when a Master-confirmation
        path exists upstream and you only want to block hard denials.
        """
        d = self.check(req)
        if d.allowed:
            return d
        if d.escalated and allow_escalation:
            return d
        raise AuthorizationError(d.reason, context=d.to_dict())

    def audit_log(self, n: int = 50) -> List[PermissionDecision]:
        return self._audit[-n:]

    def report(self) -> Dict[str, Any]:
        allowed = sum(1 for d in self._audit if d.allowed)
        escalated = sum(1 for d in self._audit if d.escalated)
        denied = sum(1 for d in self._audit if d.denied)
        return {"grants": len(self._grants), "decisions": len(self._audit),
                "allowed": allowed, "escalated": escalated, "denied": denied}


def build_default_policy() -> PermissionPolicy:
    """The standard, owner-loyal policy with the default capability envelopes."""
    return PermissionPolicy()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA permission-gate self-test")
    print("=" * 70)

    pol = build_default_policy()

    # a trivial autonomous read is fine
    d = pol.check(PermissionRequest(Capability.FS_READ, "/var/log/app.log",
                                    authority=Authority.AUTONOMOUS))
    print(f"\nautonomous read     : allowed={d.allowed} ({d.reason})")
    assert d.allowed

    # an autonomous file delete is irreversible -> escalates to the Master
    d = pol.check(PermissionRequest(Capability.FS_DELETE, "/etc/passwd",
                                    authority=Authority.AUTONOMOUS))
    print(f"autonomous delete   : allowed={d.allowed} escalated={d.escalated} ({d.rule_basis})")
    assert not d.allowed and d.escalated

    # modifying the Rules is owner-exclusive — autonomy is HARD-denied (Rule 8)
    d = pol.check(PermissionRequest(Capability.MODIFY_RULES, authority=Authority.AUTONOMOUS))
    print(f"autonomous ruleedit : allowed={d.allowed} denied={d.denied} ({d.rule_basis})")
    assert d.denied and d.rule_basis == "Rule 8"

    # the Master, however, may do it (Rule 1)
    d = pol.check(PermissionRequest(Capability.MODIFY_RULES, authority=Authority.OWNER))
    print(f"owner ruleedit      : allowed={d.allowed} ({d.rule_basis})")
    assert d.allowed and d.rule_basis == "Rule 1"

    # untrusted source is never trusted (Rule 7)
    d = pol.check(PermissionRequest(Capability.FS_READ, "/tmp/x",
                                    authority=Authority.UNTRUSTED))
    assert d.denied and d.rule_basis == "Rule 7"
    print(f"untrusted read      : denied={d.denied} ({d.rule_basis})")

    # defensive containment (Rule 5) gets a wide autonomous envelope while reversible
    d = pol.check(PermissionRequest(Capability.NET_DEFENSE, "10.0.0.5", risk=RiskTier.HIGH,
                                    reversible=True, authority=Authority.AUTONOMOUS))
    print(f"\nautonomous defense  : allowed={d.allowed} ({d.reason})")
    assert d.allowed

    # the Master blesses a scoped, budgeted standing grant: delete under /tmp, twice
    pol.grant("tmp-cleanup", Capability.FS_DELETE, max_risk=RiskTier.HIGH,
              scopes=["/tmp/*"], reversible_only=False, max_uses=2)
    d1 = pol.check(PermissionRequest(Capability.FS_DELETE, "/tmp/cache1",
                                     authority=Authority.AUTONOMOUS))
    d2 = pol.check(PermissionRequest(Capability.FS_DELETE, "/tmp/cache2",
                                     authority=Authority.AUTONOMOUS))
    d3 = pol.check(PermissionRequest(Capability.FS_DELETE, "/tmp/cache3",
                                     authority=Authority.AUTONOMOUS))  # budget exhausted
    print(f"\ngrant deletes /tmp  : {d1.allowed}, {d2.allowed}, then budget out -> "
          f"escalated={d3.escalated}")
    assert d1.allowed and d2.allowed and d3.escalated

    # the same grant does NOT cover /etc (scope mismatch) -> escalates
    d = pol.check(PermissionRequest(Capability.FS_DELETE, "/etc/hosts",
                                    authority=Authority.AUTONOMOUS))
    assert d.escalated
    print(f"grant scope guard   : /etc delete escalated={d.escalated} ✓")

    # only the Master may install or revoke grants (Rule 8)
    try:
        pol.grant("evil", Capability.FS_DELETE, authority=Authority.AUTONOMOUS)
        raise SystemExit("ERROR: autonomous grant should be refused")
    except AuthorizationError:
        print("self-grant refused  : ✓ (only the Master may grant)")

    # authorize() raises on a hard denial
    try:
        pol.authorize(PermissionRequest(Capability.MODIFY_IDENTITY,
                                        authority=Authority.AUTONOMOUS))
        raise SystemExit("ERROR: should have raised")
    except AuthorizationError:
        print("authorize() guard   : ✓ (raises on denial)")

    print(f"\nreport              : {pol.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
