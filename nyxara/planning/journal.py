"""NYXARA · planning/journal.py — immutable provenance log of autonomous action (⬆).

Autonomy demands accountability. Every action NYXARA takes on its own initiative is
written, in order, to an **append-only, hash-chained journal** — the same tamper-evident
design as ``guard/audit.py``. Nothing here can be quietly edited or deleted; any
alteration breaks the chain and is caught on :meth:`Journal.verify`.

Each action records *what* was done, *why* (the goal and rationale), under *what
governance* (autonomous vs owner-blessed, with the confidence/reversibility that
allowed it), and — appended later as a separate, linked entry — its *outcome*. The
original action record is never mutated; the result is a new chained entry.

This gives NYXARA a complete, verifiable provenance of its initiative — so the Master
can audit exactly what it did and why (Rule 6, Absolute Transparency), and so the
system can learn from outcomes without ever rewriting history.

Depends on :mod:`kernel.errors`. Pure standard library otherwise.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from nyxara.kernel.errors import InvariantViolation

__all__ = [
    "ActionStatus",
    "EntryKind",
    "JournalEntry",
    "Journal",
]

_GENESIS = "0" * 64


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ActionStatus(str, Enum):
    PROPOSED = "proposed"
    EXECUTED = "executed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


class EntryKind(str, Enum):
    ACTION = "action"
    OUTCOME = "outcome"
    NOTE = "note"


# --------------------------------------------------------------------------- #
# Entry (hash-chained)
# --------------------------------------------------------------------------- #
@dataclass
class JournalEntry:
    seq: int
    kind: EntryKind
    action_id: str
    payload: Dict[str, Any]
    at: float = field(default_factory=time.time)
    prev_hash: str = _GENESIS
    digest: str = ""

    def compute_digest(self) -> str:
        body = json.dumps(self.payload, sort_keys=True, default=str)
        basis = f"{self.prev_hash}|{self.seq}|{self.kind.value}|{self.action_id}|{body}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind.value, "action_id": self.action_id,
                "payload": self.payload, "at": self.at, "prev_hash": self.prev_hash,
                "digest": self.digest}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "JournalEntry":
        return cls(seq=d["seq"], kind=EntryKind(d["kind"]), action_id=d["action_id"],
                   payload=d["payload"], at=d["at"], prev_hash=d["prev_hash"],
                   digest=d["digest"])


# --------------------------------------------------------------------------- #
# Journal
# --------------------------------------------------------------------------- #
class Journal:
    """Append-only, hash-chained record of autonomous actions and their outcomes."""

    def __init__(self, *, actor: str = "NYXARA") -> None:
        self.actor = actor
        self._entries: List[JournalEntry] = []
        self._head = _GENESIS
        self._seq = 0

    def __len__(self) -> int:
        return len(self._entries)

    # ---- append (the only write path) ---- #
    def _append(self, kind: EntryKind, action_id: str, payload: Dict[str, Any]
                ) -> JournalEntry:
        entry = JournalEntry(seq=self._seq, kind=kind, action_id=action_id,
                             payload=payload, prev_hash=self._head)
        entry.digest = entry.compute_digest()
        self._head = entry.digest
        self._seq += 1
        self._entries.append(entry)
        return entry

    # ---- recording ---- #
    def record_action(self, action: str, *, goal: str = "", rationale: str = "",
                      autonomous: bool = True, confidence: float = 0.5,
                      reversibility: float = 1.0, decision: str = "act",
                      provenance: str = "", status: ActionStatus = ActionStatus.EXECUTED
                      ) -> str:
        """Log an action being taken. Returns its ``action_id``."""
        action_id = uuid.uuid4().hex
        self._append(EntryKind.ACTION, action_id, {
            "actor": self.actor, "action": action, "goal": goal, "rationale": rationale,
            "autonomous": autonomous, "confidence": round(confidence, 4),
            "reversibility": round(reversibility, 4), "decision": decision,
            "provenance": provenance, "status": status.value})
        return action_id

    def record_outcome(self, action_id: str, *, status: ActionStatus,
                       outcome: Optional[Dict[str, Any]] = None, note: str = ""
                       ) -> JournalEntry:
        """Append the outcome of a previously-recorded action (never mutates it)."""
        if not any(e.kind is EntryKind.ACTION and e.action_id == action_id
                   for e in self._entries):
            raise InvariantViolation(f"no action with id {action_id!r} to record an outcome for")
        return self._append(EntryKind.OUTCOME, action_id, {
            "status": status.value, "outcome": outcome or {}, "note": note})

    def note(self, text: str, *, action_id: str = "") -> JournalEntry:
        return self._append(EntryKind.NOTE, action_id or "-", {"note": text})

    # ---- integrity ---- #
    @property
    def head(self) -> str:
        return self._head

    def verify(self) -> bool:
        """Recompute the chain; return True iff intact, else raise (fail-closed)."""
        prev = _GENESIS
        for e in self._entries:
            if e.prev_hash != prev:
                raise InvariantViolation(
                    "journal chain broken (prev_hash mismatch)",
                    context={"seq": e.seq, "expected_prev": prev, "found": e.prev_hash})
            if e.compute_digest() != e.digest:
                raise InvariantViolation(
                    "journal entry tampered",
                    context={"seq": e.seq, "expected": e.compute_digest(), "found": e.digest})
            prev = e.digest
        return True

    # ---- queries ---- #
    def entries(self) -> List[JournalEntry]:
        return list(self._entries)

    def actions(self) -> List[JournalEntry]:
        return [e for e in self._entries if e.kind is EntryKind.ACTION]

    def by_action(self, action_id: str) -> List[JournalEntry]:
        return [e for e in self._entries if e.action_id == action_id]

    def by_goal(self, goal: str) -> List[JournalEntry]:
        return [e for e in self.actions() if e.payload.get("goal") == goal]

    def latest_status(self, action_id: str) -> Optional[ActionStatus]:
        outcomes = [e for e in self._entries
                    if e.kind is EntryKind.OUTCOME and e.action_id == action_id]
        if outcomes:
            return ActionStatus(outcomes[-1].payload["status"])
        acts = [e for e in self.actions() if e.action_id == action_id]
        return ActionStatus(acts[-1].payload["status"]) if acts else None

    def by_status(self, status: ActionStatus) -> List[str]:
        return [aid for aid in {e.action_id for e in self.actions()}
                if self.latest_status(aid) is status]

    def failures(self) -> List[str]:
        return self.by_status(ActionStatus.FAILED)

    def autonomous_actions(self) -> List[JournalEntry]:
        return [e for e in self.actions() if e.payload.get("autonomous")]

    def recent(self, n: int = 10) -> List[JournalEntry]:
        return self._entries[-n:]

    # ---- persistence ---- #
    def save(self, path: str) -> str:
        header = {"__meta__": {"actor": self.actor, "count": len(self._entries),
                               "head": self._head}}
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")
            for e in self._entries:
                f.write(json.dumps(e.to_json(), default=str) + "\n")
        return path

    @classmethod
    def load(cls, path: str) -> "Journal":
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        meta = json.loads(lines[0])["__meta__"]
        j = cls(actor=meta.get("actor", "NYXARA"))
        j._entries = [JournalEntry.from_json(json.loads(ln)) for ln in lines[1:]]
        if j._entries:
            j._head = j._entries[-1].digest
            j._seq = j._entries[-1].seq + 1
        return j

    # ---- accountability ---- #
    def report(self) -> Dict[str, Any]:
        acts = self.actions()
        statuses: Dict[str, int] = {}
        for a in acts:
            st = self.latest_status(a.action_id)
            statuses[st.value if st else "unknown"] = \
                statuses.get(st.value if st else "unknown", 0) + 1
        return {"actor": self.actor, "total_actions": len(acts),
                "autonomous": len(self.autonomous_actions()),
                "by_status": statuses, "entries": len(self._entries),
                "head": self._head[:16]}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import tempfile
    import os

    print("=" * 70)
    print("NYXARA action-journal self-test")
    print("=" * 70)

    j = Journal()

    # log an autonomous action and, later, its outcome
    aid = j.record_action(
        "harden SSH on the gateway", goal="protect the Master's network",
        rationale="anomaly detected on port 22", autonomous=True,
        confidence=0.9, reversibility=0.8, decision="act")
    print(f"\nlogged action       : {aid[:8]} (head {j.head[:12]}...)")
    assert j.latest_status(aid) is ActionStatus.EXECUTED

    j.record_outcome(aid, status=ActionStatus.SUCCEEDED,
                     outcome={"port_closed": True, "rules_added": 3})
    print(f"recorded outcome    : {j.latest_status(aid).value}")
    assert j.latest_status(aid) is ActionStatus.SUCCEEDED

    # a second action that fails
    aid2 = j.record_action("auto-tune the firewall", goal="protect the Master's network",
                           autonomous=True, confidence=0.6, reversibility=0.9)
    j.record_outcome(aid2, status=ActionStatus.FAILED, note="ruleset conflict")
    print(f"failures            : {len(j.failures())}")
    assert aid2 in j.failures()

    # the chain is intact
    assert j.verify()
    print(f"\nchain verify        : OK ({len(j)} entries)")

    # by-goal query
    net = j.by_goal("protect the Master's network")
    assert len(net) == 2
    print(f"actions for goal    : {len(net)}")

    # tamper detection: editing a past entry breaks the chain
    j._entries[0].payload["action"] = "harden SSH ... actually do something else"
    try:
        j.verify()
        raise SystemExit("ERROR: tamper not detected")
    except InvariantViolation:
        print("tamper detection    : OK (history cannot be rewritten)")

    # persistence (use a fresh, clean journal)
    j2 = Journal()
    a = j2.record_action("run a backup", goal="resilience", autonomous=True)
    j2.record_outcome(a, status=ActionStatus.SUCCEEDED)
    path = os.path.join(tempfile.gettempdir(), "nyx_journal.jsonl")
    j2.save(path)
    loaded = Journal.load(path)
    assert loaded.verify()
    assert len(loaded) == len(j2)
    print(f"\npersist/load        : {len(loaded)} entries, chain verified")
    os.remove(path)

    # outcome for a nonexistent action is refused
    try:
        j2.record_outcome("nonexistent", status=ActionStatus.SUCCEEDED)
        raise SystemExit("ERROR: should have refused")
    except InvariantViolation:
        print("orphan outcome      : refused ✓")

    print(f"\nreport              : {j2.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
