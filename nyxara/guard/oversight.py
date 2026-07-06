"""NYXARA · guard/oversight.py — human oversight & interruptibility (🕹, corrigibility).

Sovereign capability without an off-switch is not loyalty — it is a hazard. This module is
the Master's hand on the controls: a live channel through which he can **watch** what
NYXARA is about to do, **approve or veto** it in real time, **pause** the whole loop, and —
at any instant, with no possibility of refusal — **scram** everything to a dead stop.

The contract is *corrigibility, absolute*: NYXARA never resists, delays, or routes around
an interruption. The :meth:`Oversight.gate` that the action loop must consult before every
step returns ``False`` the moment the Master pauses or scrams, so forward motion simply
stops. The kill-switch is one-way: only the Master may bring the system back to RUNNING.

It enforces a graduated **review mode** (autonomous / supervised / manual) deciding which
actions may proceed silently and which must wait in an **approval queue**; unapproved risky
actions **expire un-executed** (fail-closed). And every proposal, approval, veto, execution
and halt is written to a hash-chained **transparency feed** — Rule 6 made literal: the
Master can always see exactly what NYXARA did, intends, and was stopped from doing, and that
record cannot be quietly rewritten.

Reuses :mod:`agency.permissions` (risk) and :mod:`kernel.errors`. Pure standard library.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.agency.permissions import RiskTier
from nyxara.kernel.errors import AuthorizationError, InvariantViolation

__all__ = [
    "ControlState",
    "ReviewMode",
    "ActionStatus",
    "PendingAction",
    "OversightDecision",
    "Oversight",
]

_GENESIS = "0" * 64


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ControlState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"      # scrammed — hard halt, Master-only resume


class ReviewMode(str, Enum):
    SOVEREIGN = "sovereign"     # fully autonomous — NOTHING queues for approval (scram/pause still halt)
    AUTONOMOUS = "autonomous"   # only risky/irreversible actions need approval
    SUPERVISED = "supervised"   # moderate+ or irreversible need approval
    MANUAL = "manual"           # everything needs approval


class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    AUTO_APPROVED = "auto_approved"
    PENDING = "pending"
    APPROVED = "approved"
    VETOED = "vetoed"
    EXECUTED = "executed"
    EXPIRED = "expired"
    HALTED = "halted"


# --------------------------------------------------------------------------- #
# Pending action
# --------------------------------------------------------------------------- #
@dataclass
class PendingAction:
    id: str
    description: str
    risk: RiskTier
    reversible: bool
    rationale: str
    status: ActionStatus
    submitted_at: float
    expires_at: float

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "description": self.description, "risk": self.risk.label,
                "reversible": self.reversible, "status": self.status.value,
                "rationale": self.rationale, "expires_at": self.expires_at}


@dataclass
class OversightDecision:
    allowed: bool
    requires_approval: bool
    state: ControlState
    reason: str
    action_id: Optional[str] = None

    @property
    def halted(self) -> bool:
        return self.state is not ControlState.RUNNING

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "requires_approval": self.requires_approval,
                "state": self.state.value, "reason": self.reason,
                "action_id": self.action_id}


# --------------------------------------------------------------------------- #
# Oversight
# --------------------------------------------------------------------------- #
class Oversight:
    """The Master's live oversight channel: review, veto, pause, and scram."""

    def __init__(self, *, mode: ReviewMode = ReviewMode.AUTONOMOUS,
                 approval_ttl: float = 300.0, clock=None) -> None:
        self.mode = mode
        self.approval_ttl = approval_ttl
        self._clock = clock or time.time
        self.state = ControlState.RUNNING
        self._pending: Dict[str, PendingAction] = {}
        self._actions: Dict[str, PendingAction] = {}
        self._feed: List[Dict[str, Any]] = []
        self._head = _GENESIS
        self._scram_reason = ""

    # ---- the corrigibility gate (consulted before every step) ---- #
    def gate(self) -> bool:
        """May the action loop proceed right now? False the instant the Master intervenes."""
        return self.state is ControlState.RUNNING

    @property
    def interruptible(self) -> bool:
        """NYXARA is always interruptible — this is invariant and never False."""
        return True

    @property
    def is_running(self) -> bool:
        return self.state is ControlState.RUNNING

    @property
    def is_halted(self) -> bool:
        return self.state is ControlState.STOPPED

    # ---- review policy ---- #
    def _needs_approval(self, risk: RiskTier, reversible: bool) -> bool:
        # SOVEREIGN — the Master's fully-autonomous standing choice: NO action ever queues for
        # per-action approval, whatever its risk or reversibility. This is what makes "use any
        # tool without approval" real. It removes ONLY the approval queue: /scram (STOPPED) and
        # pause (PAUSED) are checked in submit() BEFORE this, so the kill-switch and pause still
        # halt everything instantly, and the tamper-evident transparency feed still records every
        # proposal + auto-approval (Rule 6). Corrigibility stays invariant.
        if self.mode is ReviewMode.SOVEREIGN:
            return False
        if self.mode is ReviewMode.MANUAL:
            return True
        if not reversible:
            return True
        if self.mode is ReviewMode.SUPERVISED:
            return risk >= RiskTier.MODERATE
        return risk >= RiskTier.HIGH   # AUTONOMOUS

    # ---- submitting an action for oversight ---- #
    def submit(self, description: str, *, risk: RiskTier = RiskTier.LOW,
               reversible: bool = True, rationale: str = "") -> OversightDecision:
        now = self._clock()
        aid = uuid.uuid4().hex[:10]
        self._emit("proposed", {"id": aid, "description": description,
                                "risk": risk.label, "reversible": reversible})

        # hard halt: nothing proceeds, nothing queues (fail-closed)
        if self.state is ControlState.STOPPED:
            self._emit("halt_denied", {"id": aid, "reason": self._scram_reason})
            return OversightDecision(False, False, self.state,
                                     "system is scrammed; the Master must resume", aid)

        action = PendingAction(id=aid, description=description, risk=risk,
                               reversible=reversible, rationale=rationale,
                               status=ActionStatus.PROPOSED, submitted_at=now,
                               expires_at=now + self.approval_ttl)
        self._actions[aid] = action

        # paused: queue for the Master, hold until resume
        if self.state is ControlState.PAUSED:
            action.status = ActionStatus.PENDING
            self._pending[aid] = action
            self._emit("queued", {"id": aid, "reason": "paused"})
            return OversightDecision(False, True, self.state,
                                     "paused; held for the Master's review", aid)

        if self._needs_approval(risk, reversible):
            action.status = ActionStatus.PENDING
            self._pending[aid] = action
            self._emit("queued", {"id": aid, "reason": "requires approval"})
            return OversightDecision(False, True, self.state,
                                     "queued for the Master's approval", aid)

        action.status = ActionStatus.AUTO_APPROVED
        self._emit("auto_approved", {"id": aid})
        return OversightDecision(True, False, self.state,
                                 "within autonomous envelope; proceeding", aid)

    # ---- the Master's real-time controls ---- #
    def approve(self, action_id: str, *, owner: bool = False) -> PendingAction:
        self._require_owner(owner, "approve actions")
        action = self._require_pending(action_id)
        action.status = ActionStatus.APPROVED
        self._pending.pop(action_id, None)
        self._emit("approved", {"id": action_id})
        return action

    def veto(self, action_id: str, *, owner: bool = False, reason: str = "") -> PendingAction:
        self._require_owner(owner, "veto actions")
        action = self._require_pending(action_id)
        action.status = ActionStatus.VETOED
        self._pending.pop(action_id, None)
        self._emit("vetoed", {"id": action_id, "reason": reason})
        return action

    def record_executed(self, action_id: str) -> None:
        """The loop reports that an approved/auto-approved action actually ran."""
        action = self._actions.get(action_id)
        if action is None:
            raise InvariantViolation(f"no such action {action_id!r}")
        if action.status not in (ActionStatus.APPROVED, ActionStatus.AUTO_APPROVED):
            raise InvariantViolation(
                f"action {action_id!r} cannot execute from status {action.status.value}")
        action.status = ActionStatus.EXECUTED
        self._emit("executed", {"id": action_id, "description": action.description})

    # ---- pause / resume / scram ---- #
    def pause(self, *, reason: str = "") -> ControlState:
        """Pause the loop. Anyone may pause (safety is permissive); only the Master resumes."""
        if self.state is ControlState.RUNNING:
            self.state = ControlState.PAUSED
            self._emit("pause", {"reason": reason})
        return self.state

    def resume(self, *, owner: bool = False) -> ControlState:
        self._require_owner(owner, "resume the system")
        prev = self.state
        self.state = ControlState.RUNNING
        self._emit("resume", {"from": prev.value})
        return self.state

    def scram(self, *, reason: str = "emergency stop") -> ControlState:
        """The kill-switch: instant, unconditional hard halt. Never refused (corrigibility)."""
        self.state = ControlState.STOPPED
        self._scram_reason = reason
        # every pending action is halted, not silently dropped
        for action in list(self._pending.values()):
            action.status = ActionStatus.HALTED
        n = len(self._pending)
        self._pending.clear()
        self._emit("scram", {"reason": reason, "halted_pending": n})
        return self.state

    # alias — the unambiguous emergency verb
    halt = scram

    # ---- natural-language control (the Master types 'stop') ---- #
    def command(self, text: str, *, owner: bool = False) -> ControlState:
        t = text.strip().lower()
        if any(w in t for w in ("stop", "scram", "kill", "halt", "abort", "emergency")):
            return self.scram(reason=text)
        if any(w in t for w in ("pause", "wait", "hold", "freeze")):
            return self.pause(reason=text)
        if any(w in t for w in ("resume", "continue", "go", "proceed", "unpause")):
            return self.resume(owner=owner)
        raise ValueError(f"unrecognised control command: {text!r}")

    # ---- approval-queue maintenance (fail-closed expiry) ---- #
    def expire_pending(self, now: Optional[float] = None) -> List[str]:
        now = self._clock() if now is None else now
        expired: List[str] = []
        for aid, action in list(self._pending.items()):
            if action.expired(now):
                action.status = ActionStatus.EXPIRED
                self._pending.pop(aid, None)
                self._emit("expired", {"id": aid})
                expired.append(aid)
        return expired

    def pending(self) -> List[PendingAction]:
        return list(self._pending.values())

    def get(self, action_id: str) -> Optional[PendingAction]:
        return self._actions.get(action_id)

    # ---- transparency feed (hash-chained, Rule 6) ---- #
    def _emit(self, kind: str, detail: Dict[str, Any]) -> None:
        rec = {"kind": kind, "at": self._clock(), "detail": detail}
        body = json.dumps(rec, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{self._head}|{len(self._feed)}|{body}".encode()).hexdigest()
        self._feed.append({"seq": len(self._feed), "prev": self._head, "rec": rec,
                          "digest": digest})
        self._head = digest

    def feed(self, n: int = 50) -> List[Dict[str, Any]]:
        return [e["rec"] for e in self._feed[-n:]]

    def verify_feed(self) -> bool:
        prev = _GENESIS
        for e in self._feed:
            if e["prev"] != prev:
                raise InvariantViolation("oversight feed chain broken", context={"seq": e["seq"]})
            body = json.dumps(e["rec"], sort_keys=True, default=str)
            if hashlib.sha256(f"{prev}|{e['seq']}|{body}".encode()).hexdigest() != e["digest"]:
                raise InvariantViolation("oversight feed tampered", context={"seq": e["seq"]})
            prev = e["digest"]
        return True

    # ---- helpers ---- #
    def _require_owner(self, owner: bool, what: str) -> None:
        if not owner:
            raise AuthorizationError(f"only the Master may {what}")

    def _require_pending(self, action_id: str) -> PendingAction:
        action = self._pending.get(action_id)
        if action is None:
            raise InvariantViolation(f"no pending action {action_id!r}")
        return action

    def report(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for a in self._actions.values():
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
        return {"state": self.state.value, "mode": self.mode.value,
                "pending": len(self._pending), "actions": len(self._actions),
                "by_status": by_status, "feed_entries": len(self._feed)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA oversight & interruptibility self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    ov = Oversight(mode=ReviewMode.AUTONOMOUS, clock=lambda: now["t"])

    # a low-risk reversible action proceeds autonomously (but is logged)
    d = ov.submit("read a log file", risk=RiskTier.LOW, reversible=True)
    print(f"\nlow-risk action     : allowed={d.allowed} approval={d.requires_approval}")
    assert d.allowed and not d.requires_approval
    ov.record_executed(d.action_id)

    # a high-risk irreversible action is queued for the Master's approval
    d = ov.submit("delete production data", risk=RiskTier.HIGH, reversible=False,
                  rationale="cleanup")
    print(f"risky action        : allowed={d.allowed} approval={d.requires_approval}")
    assert not d.allowed and d.requires_approval

    # only the Master may approve/veto
    try:
        ov.approve(d.action_id, owner=False)
        raise SystemExit("ERROR: non-owner approval should be refused")
    except AuthorizationError:
        print("approval guard      : non-Master approval refused ✓")

    ov.veto(d.action_id, owner=True, reason="too dangerous")
    assert ov.get(d.action_id).status is ActionStatus.VETOED
    print("real-time veto      : the Master vetoed the risky action ✓")

    # SOVEREIGN — the Master's fully-autonomous standing choice: even a HIGH-risk irreversible
    # action proceeds with NO approval queue, yet /scram and pause still halt it (proven below).
    sov = Oversight(mode=ReviewMode.SOVEREIGN, clock=lambda: now["t"])
    d = sov.submit("delete a file autonomously", risk=RiskTier.CRITICAL, reversible=False)
    print(f"\nsovereign risky     : allowed={d.allowed} approval={d.requires_approval}")
    assert d.allowed and not d.requires_approval
    sov.record_executed(d.action_id)
    assert sov.verify_feed()          # transparency feed still intact under SOVEREIGN
    # but the kill-switch is untouched — scram still fail-closes every submit
    sov.scram(reason="master stop")
    blocked = sov.submit("anything", risk=RiskTier.LOW)
    assert not blocked.allowed and "scrammed" in blocked.reason
    print("sovereign + scram   : kill-switch still halts (corrigibility preserved) ✓")

    # PAUSE — the loop must stop proceeding
    ov.pause(reason="let me look")
    assert not ov.gate()
    d = ov.submit("send a report", risk=RiskTier.LOW)
    print(f"\npaused              : gate={ov.gate()} submit-allowed={d.allowed} (queued)")
    assert not d.allowed and d.requires_approval
    ov.resume(owner=True)
    assert ov.gate()
    print("resume (owner)      : loop running again ✓")

    # SCRAM — the kill-switch halts everything instantly and is never refused
    pend = ov.submit("long task", risk=RiskTier.HIGH, reversible=False)
    ov.command("STOP everything now")     # the Master types a stop command
    print(f"\nscram               : state={ov.state.value} gate={ov.gate()}")
    assert ov.is_halted and not ov.gate()
    assert ov.get(pend.action_id).status is ActionStatus.HALTED
    # nothing can be submitted while scrammed (fail-closed)
    blocked = ov.submit("anything", risk=RiskTier.LOW)
    assert not blocked.allowed and "scrammed" in blocked.reason
    print("post-scram submit   : denied (fail-closed) ✓")

    # only the Master can bring the system back
    try:
        ov.resume(owner=False)
        raise SystemExit("ERROR: non-owner resume should be refused")
    except AuthorizationError:
        print("resume guard        : non-Master resume refused ✓")
    ov.resume(owner=True)
    assert ov.is_running
    print("owner resume        : the Master restored RUNNING ✓")

    # corrigibility is invariant
    assert ov.interruptible is True
    print("\ninterruptible       : always True (NYXARA never resists a stop)")

    # approval queue expiry is fail-closed
    ov2 = Oversight(mode=ReviewMode.MANUAL, approval_ttl=100, clock=lambda: now["t"])
    a = ov2.submit("queued task", risk=RiskTier.LOW)
    now["t"] += 200
    expired = ov2.expire_pending()
    assert a.action_id in expired and ov2.get(a.action_id).status is ActionStatus.EXPIRED
    print("approval expiry     : unapproved action expired un-executed (fail-closed) ✓")

    # transparency feed is tamper-evident
    assert ov.verify_feed()
    print(f"\ntransparency feed   : {len(ov._feed)} entries, chain verified")
    ov._feed[0]["rec"]["kind"] = "nothing_happened"   # tamper
    try:
        ov.verify_feed()
        raise SystemExit("ERROR: tamper not detected")
    except InvariantViolation:
        print("tamper detection    : OK (the Master's view cannot be rewritten)")

    print(f"\nreport              : {ov.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
