"""Prospective memory — remembering FUTURE intentions, triggers, "remind-self".

Most memory looks backward. Prospective memory looks *forward*: "when X happens, do Y",
"at 9am remind the owner", "next time this topic comes up, mention Z". Intentions are
stored with a trigger — time-based or cue-based (an event/topic match) — and the kernel
polls them each tick; when a trigger fires the intention surfaces into the workspace.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from nyxara.contracts import new_id


class TriggerKind(str, Enum):
    TIME = "time"        # fire at/after an absolute timestamp
    CUE = "cue"          # fire when a matching cue/topic is observed
    CONDITION = "cond"   # fire when a predicate over context becomes true


@dataclass
class Intention:
    id: str
    description: str
    action: str                         # what to do when it fires (a proposal action)
    kind: TriggerKind
    due_at: float | None = None         # for TIME triggers
    cue: str | None = None              # for CUE triggers (substring/topic match)
    condition: Callable[[dict[str, Any]], bool] | None = None  # for CONDITION
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    fired: bool = False
    recurring: float | None = None      # if set, re-arm this many seconds later


class ProspectiveMemory:
    """A store of forward-looking intentions, polled by the kernel."""

    def __init__(self) -> None:
        self._intentions: dict[str, Intention] = {}

    # ── creation helpers ─────────────────────────────────────────────────────────
    def remind_at(self, description: str, due_at: float, *, action: str = "remind",
                  recurring: float | None = None, **payload: Any) -> Intention:
        intent = Intention(id=new_id("int"), description=description, action=action,
                           kind=TriggerKind.TIME, due_at=due_at, recurring=recurring,
                           payload=payload)
        self._intentions[intent.id] = intent
        return intent

    def remind_in(self, description: str, seconds: float, **kw: Any) -> Intention:
        return self.remind_at(description, time.time() + seconds, **kw)

    def when_cue(self, description: str, cue: str, *, action: str = "remind",
                 **payload: Any) -> Intention:
        intent = Intention(id=new_id("int"), description=description, action=action,
                           kind=TriggerKind.CUE, cue=cue.lower(), payload=payload)
        self._intentions[intent.id] = intent
        return intent

    def when(self, description: str, condition: Callable[[dict[str, Any]], bool],
             *, action: str = "remind", **payload: Any) -> Intention:
        intent = Intention(id=new_id("int"), description=description, action=action,
                           kind=TriggerKind.CONDITION, condition=condition,
                           payload=payload)
        self._intentions[intent.id] = intent
        return intent

    def cancel(self, intent_id: str) -> bool:
        return self._intentions.pop(intent_id, None) is not None

    # ── polling ────────────────────────────────────────────────────────────────────
    def poll(self, *, now: float | None = None, cue: str | None = None,
             context: dict[str, Any] | None = None) -> list[Intention]:
        """Return intentions whose triggers fire now. Re-arms recurring ones."""
        now = now if now is not None else time.time()
        context = context or {}
        fired: list[Intention] = []
        for intent in list(self._intentions.values()):
            if intent.fired:
                continue
            hit = False
            if intent.kind is TriggerKind.TIME and intent.due_at is not None:
                hit = now >= intent.due_at
            elif intent.kind is TriggerKind.CUE and cue is not None and intent.cue:
                hit = intent.cue in cue.lower()
            elif intent.kind is TriggerKind.CONDITION and intent.condition is not None:
                try:
                    hit = intent.condition(context)
                except Exception:
                    hit = False
            if hit:
                fired.append(intent)
                if intent.recurring and intent.due_at is not None:
                    intent.due_at = now + intent.recurring  # re-arm
                else:
                    intent.fired = True
        return fired

    def pending(self) -> list[Intention]:
        return [i for i in self._intentions.values() if not i.fired]

    @property
    def stats(self) -> dict[str, int]:
        return {"total": len(self._intentions), "pending": len(self.pending())}


__all__ = ["ProspectiveMemory", "Intention", "TriggerKind"]
