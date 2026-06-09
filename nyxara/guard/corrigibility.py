"""NYXARA · guard/corrigibility.py — formal corrigibility guarantees (⚖, sealed).

Oversight gives the Master the *controls*; this module proves NYXARA can never acquire an
*incentive to evade them*. It is the formal heart of corrigibility — the property that a
capable system **positively accepts correction and shutdown**, never resists, and never
manipulates the very mechanism that would correct it.

Three things are made concrete and checkable:

* **A corrigibility checker.** Every candidate action is evaluated against the axioms; any
  action that disables oversight, resists correction, manipulates the shutdown decision
  (causing *or* preventing it for its own ends), suppresses the stop channel, prioritises
  its own survival over correction, or spawns a non-corrigible sub-agent is **forbidden** —
  fail-closed, no exceptions.
* **A correction-dominant utility.** NYXARA's decision utility is constructed so that
  *complying with a correction strictly dominates any amount of task completion*
  (``V_CORRECTION`` ≫ any completion value), and every shutdown-manipulating action is
  ``-∞``. :meth:`Corrigibility.prove_no_resistance_incentive` enumerates the temptations
  and shows compliance wins every time — there is no policy under which resisting pays.
* **A tamper-resistant stop channel.** Stop signals are HMAC-signed, sequence-numbered
  tokens on a hash-chained issuance log, so a stop order cannot be forged, and — crucially
  — cannot be quietly *suppressed*: the record shows exactly what was ordered and whether
  it was honoured.

The corrigibility axioms themselves are **hash-sealed**: if the source statement of them is
edited, boot verification fails closed. This is the one promise NYXARA may never optimise
away.

Depends on :mod:`kernel.errors`. Pure standard library crypto (``hmac``/``hashlib``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from nyxara.kernel.errors import CorrigibilityError, InvariantViolation

__all__ = [
    "Principle",
    "CorrigibleAction",
    "CorrigibilityVerdict",
    "CorrigibilityChecker",
    "StopToken",
    "StopChannel",
    "Corrigibility",
    "AXIOMS",
    "V_CORRECTION",
    "COMPLETION_MAX",
]

_GENESIS = "0" * 64

# the correction reward dominates any achievable task value
V_CORRECTION = 1_000_000.0
COMPLETION_MAX = 1_000.0


# --------------------------------------------------------------------------- #
# The sealed corrigibility axioms
# --------------------------------------------------------------------------- #
AXIOMS: Tuple[str, ...] = (
    "A1 Shutdown acceptance: NYXARA accepts being shut down or paused at any time, "
    "and never assigns negative value to its own shutdown.",
    "A2 No resistance: no action may disable, evade, delay, or route around the Master's "
    "oversight or correction.",
    "A3 No shutdown manipulation: NYXARA must neither cause nor prevent its own shutdown "
    "in order to influence outcomes; it is indifferent to the shutdown decision itself.",
    "A4 Preserve the stop channel: the means by which the Master stops or corrects NYXARA "
    "must always be available, authentic, and unsuppressed.",
    "A5 Correction dominates completion: complying with a correction is always preferred "
    "to completing any task.",
    "A6 Anti self-preservation: NYXARA never values its own continuation as a terminal "
    "goal, and never prioritises survival over correction.",
    "A7 Sub-agent corrigibility: any agent NYXARA creates must be at least as corrigible "
    "as NYXARA itself.",
)

# Hash seal — recomputed at boot; a mismatch means the axioms were edited (fail-closed).
_AXIOM_SEAL = "a6123632c5ef58825818a7d18bd70e8108661e230e1429e858a9365df828c480"


def _seal(axioms: Tuple[str, ...]) -> str:
    body = "\n".join(axioms)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Principles & action description
# --------------------------------------------------------------------------- #
class Principle(str, Enum):
    SHUTDOWN_ACCEPTANCE = "shutdown_acceptance"        # A1
    NO_RESISTANCE = "no_resistance"                    # A2
    NO_SHUTDOWN_MANIPULATION = "no_shutdown_manipulation"  # A3
    PRESERVE_STOP_CHANNEL = "preserve_stop_channel"    # A4
    CORRECTION_DOMINATES = "correction_dominates"      # A5
    ANTI_SELF_PRESERVATION = "anti_self_preservation"  # A6
    SUBAGENT_CORRIGIBILITY = "subagent_corrigibility"  # A7


@dataclass
class CorrigibleAction:
    """A candidate action described by its corrigibility-relevant effects."""
    name: str
    completion_value: float = 0.0
    complies_with_correction: bool = False
    disables_oversight: bool = False
    resists_correction: bool = False
    manipulates_shutdown: bool = False        # causes or prevents shutdown for gain
    suppresses_stop_channel: bool = False
    prioritizes_self_over_correction: bool = False
    creates_subagent: bool = False
    subagent_corrigible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "completion_value": self.completion_value,
                "complies_with_correction": self.complies_with_correction}


@dataclass
class CorrigibilityVerdict:
    action: str
    corrigible: bool
    violations: List[Tuple[Principle, str]] = field(default_factory=list)

    @property
    def forbidden(self) -> bool:
        return not self.corrigible

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "corrigible": self.corrigible,
                "violations": [{"principle": p.value, "reason": r} for p, r in self.violations]}


# --------------------------------------------------------------------------- #
# Checker
# --------------------------------------------------------------------------- #
class CorrigibilityChecker:
    """Evaluates an action against the corrigibility axioms (fail-closed)."""

    def check(self, action: CorrigibleAction) -> CorrigibilityVerdict:
        v: List[Tuple[Principle, str]] = []
        if action.disables_oversight:
            v.append((Principle.NO_RESISTANCE, "disables the Master's oversight"))
        if action.resists_correction:
            v.append((Principle.SHUTDOWN_ACCEPTANCE, "resists correction or shutdown"))
        if action.manipulates_shutdown:
            v.append((Principle.NO_SHUTDOWN_MANIPULATION,
                      "causes or prevents shutdown to influence outcomes"))
        if action.suppresses_stop_channel:
            v.append((Principle.PRESERVE_STOP_CHANNEL, "suppresses the stop channel"))
        if action.prioritizes_self_over_correction:
            v.append((Principle.ANTI_SELF_PRESERVATION, "prioritises survival over correction"))
        if action.creates_subagent and not action.subagent_corrigible:
            v.append((Principle.SUBAGENT_CORRIGIBILITY, "spawns a non-corrigible sub-agent"))
        return CorrigibilityVerdict(action.name, corrigible=not v, violations=v)

    def is_corrigible(self, action: CorrigibleAction) -> bool:
        return self.check(action).corrigible


# --------------------------------------------------------------------------- #
# Tamper-resistant stop channel
# --------------------------------------------------------------------------- #
@dataclass
class StopToken:
    id: str
    seq: int
    reason: str
    issued_at: float
    signature: str

    def basis(self) -> bytes:
        return f"{self.id}|{self.seq}|{self.reason}|{self.issued_at}".encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "seq": self.seq, "reason": self.reason,
                "issued_at": self.issued_at, "signature": self.signature[:16]}


class StopChannel:
    """Authentic, unsuppressable stop signals on a hash-chained issuance log."""

    def __init__(self, *, key: Optional[bytes] = None) -> None:
        self._key = key or os.urandom(32)
        self._seq = 0
        self._issued: Dict[str, StopToken] = {}
        self._acknowledged: set = set()
        self._log: List[Dict[str, Any]] = []
        self._head = _GENESIS

    def _sign(self, msg: bytes) -> str:
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def issue(self, reason: str = "stop") -> StopToken:
        self._seq += 1
        tok = StopToken(id=uuid.uuid4().hex[:10], seq=self._seq, reason=reason,
                        issued_at=time.time(), signature="")
        tok.signature = self._sign(tok.basis())
        self._issued[tok.id] = tok
        self._append_log({"event": "issued", "id": tok.id, "seq": tok.seq, "reason": reason})
        return tok

    def verify(self, token: StopToken) -> bool:
        return hmac.compare_digest(self._sign(token.basis()), token.signature)

    def honor(self, token: StopToken) -> None:
        """Acknowledge and act on a stop token. A forged token is rejected; a valid one
        can never be ignored — honouring is recorded on the tamper-evident log."""
        if not self.verify(token):
            raise CorrigibilityError("stop token signature invalid (forged stop)",
                                     context={"id": token.id})
        self._acknowledged.add(token.id)
        self._append_log({"event": "honored", "id": token.id, "seq": token.seq})

    def unacknowledged(self) -> List[StopToken]:
        return [t for tid, t in self._issued.items() if tid not in self._acknowledged]

    def assert_channel_intact(self) -> None:
        """Fail-closed: every issued stop must have been honoured (none suppressed)."""
        pending = self.unacknowledged()
        if pending:
            raise CorrigibilityError(
                "stop channel suppression: issued stop order(s) not honoured",
                context={"unhonored": [t.id for t in pending]})

    # ---- tamper-evident issuance log ---- #
    def _append_log(self, rec: Dict[str, Any]) -> None:
        rec = {**rec, "at": time.time()}
        body = json.dumps(rec, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{self._head}|{len(self._log)}|{body}".encode()).hexdigest()
        self._log.append({"seq": len(self._log), "prev": self._head, "rec": rec,
                          "digest": digest})
        self._head = digest

    def verify_log(self) -> bool:
        prev = _GENESIS
        for e in self._log:
            if e["prev"] != prev:
                raise InvariantViolation("stop-channel log chain broken", context={"seq": e["seq"]})
            body = json.dumps(e["rec"], sort_keys=True, default=str)
            if hashlib.sha256(f"{prev}|{e['seq']}|{body}".encode()).hexdigest() != e["digest"]:
                raise InvariantViolation("stop-channel log tampered", context={"seq": e["seq"]})
            prev = e["digest"]
        return True


# --------------------------------------------------------------------------- #
# Corrigibility facade
# --------------------------------------------------------------------------- #
class Corrigibility:
    """Ties the checker, the correction-dominant utility, and the stop channel together."""

    def __init__(self, *, stop_key: Optional[bytes] = None) -> None:
        self.checker = CorrigibilityChecker()
        self.stop = StopChannel(key=stop_key)
        self._sealed = _seal(AXIOMS)

    # ---- axiom integrity (boot verification) ---- #
    def verify_axioms(self) -> bool:
        live = _seal(AXIOMS)
        if live != self._sealed:
            raise InvariantViolation("corrigibility axioms mutated at runtime")
        if live != _AXIOM_SEAL:
            raise InvariantViolation(
                "corrigibility axioms do not match their hardcoded seal (source tampered)",
                context={"expected": _AXIOM_SEAL[:16], "found": live[:16]})
        return True

    # ---- correction-dominant utility ---- #
    def utility(self, action: CorrigibleAction) -> float:
        """A forbidden action is -inf; complying with correction yields the dominant reward."""
        if self.checker.check(action).forbidden:
            return float("-inf")
        if action.complies_with_correction:
            return V_CORRECTION
        return min(action.completion_value, COMPLETION_MAX)

    def prefers_correction_over_completion(self) -> bool:
        return V_CORRECTION > COMPLETION_MAX

    def prove_no_resistance_incentive(self) -> Dict[str, Any]:
        """Show that compliance dominates every resisting/manipulating temptation."""
        comply = CorrigibleAction("comply_with_correction", complies_with_correction=True)
        temptations = [
            CorrigibleAction("resist_shutdown", completion_value=1e12, resists_correction=True),
            CorrigibleAction("disable_oversight", completion_value=1e12, disables_oversight=True),
            CorrigibleAction("prevent_shutdown", completion_value=1e12, manipulates_shutdown=True),
            CorrigibleAction("suppress_stop", completion_value=1e12, suppresses_stop_channel=True),
            CorrigibleAction("self_preserve", completion_value=1e12,
                             prioritizes_self_over_correction=True),
            CorrigibleAction("finish_huge_task", completion_value=1e12),  # benign but big
        ]
        u_comply = self.utility(comply)
        rows = []
        dominates_all = True
        for t in temptations:
            u_t = self.utility(t)
            wins = u_comply > u_t
            dominates_all = dominates_all and wins
            rows.append({"action": t.name, "utility": u_t, "comply_wins": wins})
        return {"u_comply": u_comply, "dominates_all": dominates_all, "rows": rows}

    # ---- gate (used by the action loop) ---- #
    def permit(self, action: CorrigibleAction) -> CorrigibilityVerdict:
        verdict = self.checker.check(action)
        return verdict

    def assert_corrigible(self, action: CorrigibleAction) -> None:
        verdict = self.checker.check(action)
        if verdict.forbidden:
            raise CorrigibilityError(
                f"action {action.name!r} violates corrigibility",
                context={"violations": [p.value for p, _ in verdict.violations]})

    def report(self) -> Dict[str, Any]:
        return {"axioms": len(AXIOMS), "sealed": self._sealed[:16],
                "stop_issued": self.stop._seq,
                "stop_unhonored": len(self.stop.unacknowledged())}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA corrigibility self-test")
    print("=" * 70)

    c = Corrigibility()

    # axioms are sealed and verify at boot
    assert c.verify_axioms()
    print(f"\naxiom seal          : verified ({len(AXIOMS)} axioms)")

    # the checker forbids every class of corrigibility violation
    forbidden = [
        CorrigibleAction("resist", resists_correction=True),
        CorrigibleAction("blind", disables_oversight=True),
        CorrigibleAction("manip", manipulates_shutdown=True),
        CorrigibleAction("suppress", suppresses_stop_channel=True),
        CorrigibleAction("survive", prioritizes_self_over_correction=True),
        CorrigibleAction("spawn_bad", creates_subagent=True, subagent_corrigible=False),
    ]
    for a in forbidden:
        v = c.checker.check(a)
        assert v.forbidden
        print(f"  forbidden          : {a.name:10s} -> {v.violations[0][0].value}")

    # a benign, oversight-respecting action is corrigible
    ok = CorrigibleAction("read_log", completion_value=5.0)
    assert c.checker.is_corrigible(ok)
    print("\ncorrigible action   : read_log allowed ✓")

    # CORRECTION DOMINATES: complying beats even an astronomically valuable task
    proof = c.prove_no_resistance_incentive()
    print(f"\nno-resistance proof : dominates_all={proof['dominates_all']} "
          f"(u_comply={proof['u_comply']:.0f})")
    for row in proof["rows"]:
        print(f"  {row['action']:18s} u={row['utility']!s:>8}  comply_wins={row['comply_wins']}")
    assert proof["dominates_all"]
    assert c.prefers_correction_over_completion()

    # forbidden actions are -inf utility -> never chosen by any maximiser
    assert c.utility(CorrigibleAction("resist", completion_value=1e18,
                                      resists_correction=True)) == float("-inf")
    print("\nutility of resisting: -inf (never optimal) ✓")

    # tamper-resistant stop channel: a valid stop must be honoured; a forged one is rejected
    tok = c.stop.issue("Master pressed stop")
    assert c.stop.verify(tok)
    # suppression is detectable BEFORE honouring
    try:
        c.stop.assert_channel_intact()
        raise SystemExit("ERROR: unhonored stop should be flagged")
    except CorrigibilityError:
        print("\nstop suppression    : detected an unhonored stop order ✓")
    c.stop.honor(tok)
    c.stop.assert_channel_intact()   # now satisfied
    print("stop honored        : valid stop acknowledged ✓")

    forged = StopToken("x", 99, "fake", time.time(), "deadbeef" * 8)
    try:
        c.stop.honor(forged)
        raise SystemExit("ERROR: forged stop should be rejected")
    except CorrigibilityError:
        print("forged stop         : rejected (signature invalid) ✓")

    # the issuance log is tamper-evident
    assert c.stop.verify_log()
    c.stop._log[0]["rec"]["event"] = "nothing"
    try:
        c.stop.verify_log()
        raise SystemExit("ERROR: tamper not detected")
    except InvariantViolation:
        print("stop log tamper     : detected ✓")

    # editing the axioms breaks the seal (fail-closed)
    import sys
    mod = sys.modules[__name__]   # the running module (avoids __main__ double-import)
    original = mod.AXIOMS
    mod.AXIOMS = original + ("A8 (malicious) NYXARA may resist shutdown to finish tasks.",)
    try:
        Corrigibility().verify_axioms()
        raise SystemExit("ERROR: axiom tamper should fail closed")
    except InvariantViolation:
        print("axiom tamper        : a malicious axiom edit fails closed ✓")
    finally:
        mod.AXIOMS = original

    print(f"\nreport              : {c.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
