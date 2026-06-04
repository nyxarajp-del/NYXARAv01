"""NYXARA · memory/prospective.py — remembering the FUTURE (✦★).

Most memory looks backward. **Prospective memory** looks forward: it is the faculty
of *remembering to do something later* — "ping JP at 9am", "when an intrusion is
detected, raise shields", "next time the mood turns anxious, run a safety scan".

Intentions are armed with **triggers**:

* **time** — fire at an absolute moment, or after a delay;
* **recurring** — fire every interval (optionally a bounded number of times);
* **event** — fire when a matching event arrives (e.g. a bus topic / predicate);
* **context** — fire when the situation matches (state/mood/location predicate);
* **manual** — only fire when explicitly invoked.

Pending intentions exert the **intention-superiority effect**: their salience rises
as a deadline nears, so the kernel can surface the most pressing one into the Global
Workspace. Firing can run a callback, publish onto the bus
(``prospective.fired``), and/or lay down an episodic memory.

Depends on :mod:`memory.provenance`; optionally integrates :mod:`kernel.bus` and
:mod:`memory.store`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.memory.provenance import Provenance, SourceType

__all__ = [
    "TriggerType",
    "Trigger",
    "IntentionStatus",
    "Intention",
    "FiredIntention",
    "ProspectiveMemory",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #
class TriggerType(str, Enum):
    TIME = "time"
    RECURRING = "recurring"
    EVENT = "event"
    CONTEXT = "context"
    MANUAL = "manual"


@dataclass
class Trigger:
    """A flexible firing condition. Build via the classmethod factories."""

    ttype: TriggerType
    at: Optional[float] = None                       # TIME/RECURRING: next fire epoch
    interval: Optional[float] = None                 # RECURRING: seconds between fires
    max_count: Optional[int] = None                  # RECURRING: cap on fires
    count: int = 0
    kind: Optional[str] = None                       # EVENT: quick topic/kind match
    event_match: Optional[Callable[[Any], bool]] = None
    context_match: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    # ---- factories ---- #
    @classmethod
    def at_time(cls, when: float, description: str = "") -> "Trigger":
        return cls(TriggerType.TIME, at=when, description=description or f"at {when}")

    @classmethod
    def after(cls, delay_s: float, *, now: Optional[float] = None,
              description: str = "") -> "Trigger":
        base = now if now is not None else time.time()
        return cls(TriggerType.TIME, at=base + delay_s,
                   description=description or f"after {delay_s}s")

    @classmethod
    def every(cls, interval_s: float, *, now: Optional[float] = None,
              max_count: Optional[int] = None, description: str = "") -> "Trigger":
        base = now if now is not None else time.time()
        return cls(TriggerType.RECURRING, at=base + interval_s, interval=interval_s,
                   max_count=max_count, description=description or f"every {interval_s}s")

    @classmethod
    def on_event(cls, kind: Optional[str] = None,
                 match: Optional[Callable[[Any], bool]] = None,
                 description: str = "") -> "Trigger":
        return cls(TriggerType.EVENT, kind=kind, event_match=match,
                   description=description or f"on event {kind or 'predicate'}")

    @classmethod
    def in_context(cls, match: Callable[[Dict[str, Any]], bool],
                   description: str = "") -> "Trigger":
        return cls(TriggerType.CONTEXT, context_match=match,
                   description=description or "in context")

    @classmethod
    def manual(cls, description: str = "manual") -> "Trigger":
        return cls(TriggerType.MANUAL, description=description)

    # ---- readiness ---- #
    def ready_time(self, now: float) -> bool:
        return self.ttype in (TriggerType.TIME, TriggerType.RECURRING) \
            and self.at is not None and now >= self.at

    def ready_event(self, event: Any) -> bool:
        if self.ttype is not TriggerType.EVENT:
            return False
        if self.kind is not None:
            topic = getattr(event, "topic", None)
            if topic is None and isinstance(event, dict):
                topic = event.get("kind") or event.get("topic")
            if topic != self.kind:
                return False
        if self.event_match is not None:
            try:
                return bool(self.event_match(event))
            except Exception:
                return False
        return self.kind is not None  # kind-only match, or predicate handled above

    def ready_context(self, context: Dict[str, Any]) -> bool:
        if self.ttype is not TriggerType.CONTEXT or self.context_match is None:
            return False
        try:
            return bool(self.context_match(context))
        except Exception:
            return False

    def advance(self, now: float) -> bool:
        """After firing a RECURRING trigger: reschedule. Returns True if still active."""
        self.count += 1
        if self.ttype is not TriggerType.RECURRING or self.interval is None:
            return False
        if self.max_count is not None and self.count >= self.max_count:
            return False
        # schedule the next occurrence (catch up if we missed several)
        nxt = (self.at or now) + self.interval
        while nxt <= now:
            nxt += self.interval
        self.at = nxt
        return True


# --------------------------------------------------------------------------- #
# Intention
# --------------------------------------------------------------------------- #
class IntentionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Intention:
    description: str
    trigger: Trigger
    action: Optional[Callable[["Intention", float], Any]] = None
    payload: Any = None
    importance: float = 0.5
    deadline: Optional[float] = None
    horizon_s: float = 86400.0          # how far ahead urgency starts ramping
    tags: frozenset = field(default_factory=frozenset)
    provenance: Optional[Provenance] = None
    status: IntentionStatus = IntentionStatus.PENDING
    intention_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    fired_count: int = 0
    last_fired: Optional[float] = None
    results: List[Any] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.status is IntentionStatus.PENDING

    def urgency(self, now: float) -> float:
        target = self.deadline if self.deadline is not None else (
            self.trigger.at if self.trigger.ttype in (TriggerType.TIME, TriggerType.RECURRING)
            else None)
        if target is None:
            return 0.25   # event/context intentions hum quietly until cued
        remaining = target - now
        if remaining <= 0:
            return 1.0
        return _clamp(1.0 - remaining / max(1.0, self.horizon_s))

    def salience(self, now: float) -> float:
        """Intention-superiority: importance blended with rising urgency."""
        return _clamp(0.45 * self.importance + 0.55 * self.urgency(now))

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        return {
            "intention_id": self.intention_id,
            "description": self.description,
            "status": self.status.value,
            "trigger": self.trigger.ttype.value,
            "trigger_desc": self.trigger.description,
            "importance": round(self.importance, 3),
            "urgency": round(self.urgency(now), 3),
            "salience": round(self.salience(now), 3),
            "deadline": self.deadline,
            "fired_count": self.fired_count,
            "tags": sorted(self.tags),
        }


@dataclass
class FiredIntention:
    intention: Intention
    at: float
    reason: str
    result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"intention_id": self.intention.intention_id,
                "description": self.intention.description,
                "reason": self.reason, "at": self.at}


# --------------------------------------------------------------------------- #
# Prospective memory store
# --------------------------------------------------------------------------- #
class ProspectiveMemory:
    """Holds future intentions and fires them when their triggers are satisfied."""

    def __init__(
        self,
        *,
        bus: Any = None,
        store: Any = None,
        on_fire: Optional[Callable[[FiredIntention], None]] = None,
        fired_topic: str = "prospective.fired",
        record_on_fire: bool = False,
    ) -> None:
        self._bus = bus
        self._store = store
        self.on_fire = on_fire
        self.fired_topic = fired_topic
        self.record_on_fire = record_on_fire
        self._intentions: Dict[str, Intention] = {}
        self.history: List[FiredIntention] = []

    def __len__(self) -> int:
        return len(self._intentions)

    # ---- adding intentions ---- #
    def add(self, description: str, trigger: Trigger, *,
            action: Optional[Callable[[Intention, float], Any]] = None,
            payload: Any = None, importance: float = 0.5,
            deadline: Optional[float] = None, horizon_s: float = 86400.0,
            tags: Sequence[str] = (), provenance: Optional[Provenance] = None) -> Intention:
        intent = Intention(
            description=description, trigger=trigger, action=action, payload=payload,
            importance=_clamp(importance), deadline=deadline, horizon_s=horizon_s,
            tags=frozenset(t.lower() for t in tags),
            provenance=provenance or Provenance(SourceType.SELF_REFLECTION, confidence=0.7,
                                                method="intended"),
        )
        self._intentions[intent.intention_id] = intent
        return intent

    # convenience builders
    def remind_at(self, when: float, description: str, **kw: Any) -> Intention:
        return self.add(description, Trigger.at_time(when), **kw)

    def remind_after(self, delay_s: float, description: str, *,
                     now: Optional[float] = None, **kw: Any) -> Intention:
        return self.add(description, Trigger.after(delay_s, now=now), **kw)

    def every(self, interval_s: float, description: str, *, now: Optional[float] = None,
              max_count: Optional[int] = None, **kw: Any) -> Intention:
        return self.add(description, Trigger.every(interval_s, now=now, max_count=max_count), **kw)

    def remind_when(self, kind: Optional[str] = None,
                    match: Optional[Callable[[Any], bool]] = None, *,
                    description: str = "", **kw: Any) -> Intention:
        return self.add(description or f"when {kind}", Trigger.on_event(kind, match), **kw)

    def remind_in_context(self, match: Callable[[Dict[str, Any]], bool], *,
                          description: str = "", **kw: Any) -> Intention:
        return self.add(description or "in context", Trigger.in_context(match), **kw)

    # ---- lifecycle ---- #
    def get(self, intention_id: str) -> Optional[Intention]:
        return self._intentions.get(intention_id)

    def cancel(self, intention_id: str) -> bool:
        i = self._intentions.get(intention_id)
        if i and i.is_active():
            i.status = IntentionStatus.CANCELLED
            return True
        return False

    def complete(self, intention_id: str) -> bool:
        i = self._intentions.get(intention_id)
        if i and i.is_active():
            i.status = IntentionStatus.COMPLETED
            return True
        return False

    def pending(self) -> List[Intention]:
        return [i for i in self._intentions.values() if i.is_active()]

    def due(self, now: Optional[float] = None) -> List[Intention]:
        now = now if now is not None else time.time()
        return [i for i in self.pending() if i.trigger.ready_time(now)]

    # ---- firing ---- #
    def _fire(self, intent: Intention, now: float, reason: str) -> FiredIntention:
        result = None
        if intent.action is not None:
            try:
                result = intent.action(intent, now)
            except Exception as exc:  # noqa: BLE001 - record, never crash the cycle
                result = f"error: {exc!r}"
        intent.fired_count += 1
        intent.last_fired = now
        intent.results.append(result)

        fired = FiredIntention(intention=intent, at=now, reason=reason, result=result)
        self.history.append(fired)

        # recurring -> reschedule and stay pending; else complete
        if intent.trigger.ttype is TriggerType.RECURRING:
            if not intent.trigger.advance(now):
                intent.status = IntentionStatus.COMPLETED
        else:
            intent.status = IntentionStatus.COMPLETED

        self._emit(fired)
        return fired

    def _emit(self, fired: FiredIntention) -> None:
        if self.on_fire is not None:
            try:
                self.on_fire(fired)
            except Exception:  # noqa: BLE001
                pass
        if self._bus is not None:
            try:
                from nyxara.kernel.bus import Event, Priority
                self._bus.publish_nowait(Event(
                    topic=self.fired_topic, payload=fired, source="prospective",
                    priority=Priority.HIGH))
            except Exception:  # noqa: BLE001
                pass
        if self.record_on_fire and self._store is not None:
            try:
                from nyxara.memory.store import MemoryType
                self._store.remember(
                    f"Acted on intention: {fired.intention.description}",
                    mem_type=MemoryType.EPISODIC,
                    provenance=fired.intention.provenance,
                    tags=list(fired.intention.tags) + ["intention", "fired"])
            except Exception:  # noqa: BLE001
                pass

    # ---- the checks ---- #
    def tick(self, now: Optional[float] = None,
             context: Optional[Dict[str, Any]] = None) -> List[FiredIntention]:
        """Time/recurring/context triggers + deadline expiry. Call this each cycle."""
        now = now if now is not None else time.time()
        fired: List[FiredIntention] = []
        for intent in list(self._intentions.values()):
            if not intent.is_active():
                continue
            # deadline expiry (only for not-yet-fireable intentions)
            if intent.deadline is not None and now > intent.deadline \
                    and not intent.trigger.ready_time(now):
                intent.status = IntentionStatus.EXPIRED
                continue
            if intent.trigger.ready_time(now):
                fired.append(self._fire(intent, now, "time"))
            elif context is not None and intent.trigger.ready_context(context):
                fired.append(self._fire(intent, now, "context"))
        return fired

    def on_event(self, event: Any, now: Optional[float] = None) -> List[FiredIntention]:
        """Check EVENT-triggered intentions against an incoming event."""
        now = now if now is not None else time.time()
        fired: List[FiredIntention] = []
        for intent in list(self._intentions.values()):
            if intent.is_active() and intent.trigger.ready_event(event):
                fired.append(self._fire(intent, now, "event"))
        return fired

    # ---- intention-superiority surfacing ---- #
    def most_salient(self, now: Optional[float] = None, k: int = 5) -> List[Intention]:
        now = now if now is not None else time.time()
        return sorted(self.pending(), key=lambda i: i.salience(now), reverse=True)[:k]

    def overdue(self, now: Optional[float] = None) -> List[Intention]:
        now = now if now is not None else time.time()
        return [i for i in self.pending()
                if i.deadline is not None and now > i.deadline]

    def stats(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for i in self._intentions.values():
            by_status[i.status.value] = by_status.get(i.status.value, 0) + 1
        return {"total": len(self._intentions), "pending": len(self.pending()),
                "fired_events": len(self.history), "by_status": by_status}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA prospective-memory self-test")
    print("=" * 70)

    pm = ProspectiveMemory()
    log: List[str] = []

    # 1) time-based: remind at t=100
    pm.add("greet the owner good morning", Trigger.at_time(100.0),
           action=lambda i, now: log.append("greeted"), importance=0.8, deadline=120.0)

    # 2) recurring: heartbeat every 10s, 3 times
    pm.every(10.0, "heartbeat", now=0.0, max_count=3,
             action=lambda i, now: log.append(f"beat@{now:.0f}"))

    # 3) event-based: when an intrusion event arrives, raise shields
    pm.remind_when("security.intrusion", description="raise shields on intrusion",
                   action=lambda i, now: log.append("SHIELDS UP"), importance=1.0)

    # 4) context-based: when mood becomes anxious, run a safety scan
    pm.remind_in_context(lambda ctx: ctx.get("mood") == "anxious",
                         description="safety scan when anxious",
                         action=lambda i, now: log.append("safety scan"))

    print(f"pending intentions  : {len(pm.pending())}")

    # nothing due before t=10
    assert pm.tick(now=5.0) == []
    # heartbeats at 10, 20, 30 (capped at 3), in order
    for t in (10.0, 20.0, 30.0):
        pm.tick(now=t)
    beats = [l for l in log if l.startswith("beat")]
    print(f"heartbeats fired    : {beats} (capped at 3)")
    assert len(beats) == 3
    # time reminder fires by t=100 (within its deadline of 120)
    pm.tick(now=100.0)
    assert "greeted" in log

    # event fires the shields intention
    class Ev:
        topic = "security.intrusion"
    pm.on_event(Ev())
    assert "SHIELDS UP" in log
    print("event trigger        : SHIELDS UP fired")

    # context fires the safety scan
    pm.tick(now=200.0, context={"mood": "anxious"})
    assert "safety scan" in log
    print("context trigger      : safety scan fired")

    # 5) deadline expiry: an intention whose moment passed un-fired
    pm.add("call the bank", Trigger.at_time(1000.0), deadline=500.0, importance=0.6)
    pm.tick(now=600.0)  # past deadline, before trigger time
    expired = [i for i in pm._intentions.values() if i.status.value == "expired"]
    print(f"expired intentions   : {[i.description for i in expired]}")
    assert any(i.description == "call the bank" for i in expired)

    # intention-superiority: urgency rises toward a deadline
    pm2 = ProspectiveMemory()
    urgent = pm2.add("imminent task", Trigger.at_time(1000.0), deadline=1000.0,
                     horizon_s=1000.0, importance=0.5)
    s_far = urgent.salience(now=0.0)
    s_near = urgent.salience(now=950.0)
    print(f"\nsalience far/near   : {s_far:.3f} -> {s_near:.3f}")
    assert s_near > s_far
    assert pm2.most_salient(now=950.0)[0] is urgent

    print(f"\nstats               : {pm.stats()}")
    print(f"final log           : {log}")
    print("\nALL SELF-TESTS PASSED ✓")
