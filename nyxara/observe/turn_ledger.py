"""NYXARA · observe/turn_ledger.py — what actually happened, as opposed to what was configured.

Her honesty layer already covers *claims to the Master*: ``observe/honesty.py`` calibrates how
confidently a thing is said, ``observe/self_report.py`` makes sure failures are never dropped. This
module closes a different hole, and it is the one that cost the most:

    **A self-report was derived from configuration instead of from events.**

``/learning`` said ``"chosen": "litertlm"`` beside ``"engine_loaded": false`` while a 532-parameter
n-gram did the talking, and it was not lying — ``chosen_provider()`` honestly answers *which rung the
ladder would pick*. Nobody had asked the other question, because nothing recorded the answer: **which
brain actually spoke.** Three rounds of diagnosis went into that gap, on a machine that had the
answer the whole time.

So the rule this module exists to enforce:

    Anything the system says about itself must be derived from a recorded event.
    Never from a setting, a flag, or a probe of what *would* happen.

A :class:`TurnLedger` keeps a bounded ring of :class:`TurnRecord` — one per turn — and each record
holds the *generations that were attempted*, with the provider, model, token counts, latency, and
whether it succeeded. ``answered_by`` is a derived property: it is the provider of the last
generation that actually produced text, and there is deliberately no setter. A rung cannot be
credited with an answer it did not produce, because crediting is not an operation this class offers.

Cheap and safe by construction: pure in-memory, bounded, no I/O on the turn path, and every mutation
is under one lock because the LLM facade is called from several threads at once.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Generation", "Degradation", "TurnRecord", "TurnLedger", "get_ledger", "reset_ledger"]


@dataclass(frozen=True)
class Generation:
    """One attempt to make a model produce text. Recorded whether it worked or not."""

    provider: str
    model: str
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    chars: int = 0
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"provider": self.provider, "model": self.model, "ok": self.ok,
                               "prompt_tokens": self.prompt_tokens,
                               "completion_tokens": self.completion_tokens,
                               "latency_s": round(self.latency_s, 4), "chars": self.chars}
        if self.error:
            out["error"] = self.error
        return out


@dataclass(frozen=True)
class Degradation:
    """A moment where a stronger path gave way to a weaker one, and why."""

    where: str
    reason: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"where": self.where, "reason": self.reason}


@dataclass
class TurnRecord:
    """Everything that happened while answering once."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    generations: List[Generation] = field(default_factory=list)
    degradations: List[Degradation] = field(default_factory=list)

    @property
    def answered_by(self) -> Optional[str]:
        """The provider whose text actually became an answer — or ``None`` if none did.

        Derived, never assigned. That asymmetry is the whole point: a rung can only be credited by
        having produced something, so no configuration change can make this claim on its behalf.
        """
        for gen in reversed(self.generations):
            if gen.ok and gen.chars > 0:
                return gen.provider
        return None

    @property
    def spoke(self) -> bool:
        return self.answered_by is not None

    @property
    def attempts(self) -> int:
        return len(self.generations)

    @property
    def tokens(self) -> int:
        return sum(g.prompt_tokens + g.completion_tokens for g in self.generations)

    @property
    def elapsed_s(self) -> float:
        return sum(g.latency_s for g in self.generations)

    def to_dict(self) -> Dict[str, Any]:
        return {"turn_id": self.turn_id, "answered_by": self.answered_by,
                "attempts": self.attempts, "tokens": self.tokens,
                "elapsed_s": round(self.elapsed_s, 4),
                "generations": [g.to_dict() for g in self.generations],
                "degradations": [d.to_dict() for d in self.degradations]}


class TurnLedger:
    """A bounded, thread-safe record of recent turns.

    ``capacity`` turns are kept; older ones fall off. This is an observability surface, not memory —
    ``memory/`` is where anything durable belongs, and nothing here is persisted or replayed.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._turns: List[TurnRecord] = []
        self._current: Optional[TurnRecord] = None

    # ---- turn boundaries ---- #
    def begin(self) -> TurnRecord:
        """Open a turn. Safe to call again mid-turn: the open record is reused, so a nested
        caller cannot silently split one turn into two and halve the counts."""
        with self._lock:
            if self._current is None:
                self._current = TurnRecord()
            return self._current

    def end(self) -> Optional[TurnRecord]:
        with self._lock:
            record, self._current = self._current, None
            if record is not None:
                self._turns.append(record)
                if len(self._turns) > self.capacity:
                    del self._turns[:-self.capacity]
            return record

    def _target(self) -> TurnRecord:
        """The record to write into — opening one implicitly rather than dropping the event.

        A generation that happens outside an explicit turn is still a generation that happened.
        Losing it would reintroduce exactly the blindness this module exists to remove.
        """
        if self._current is None:
            self._current = TurnRecord()
        return self._current

    # ---- events ---- #
    def record_generation(self, provider: str, model: str, *, ok: bool,
                          prompt_tokens: int = 0, completion_tokens: int = 0,
                          latency_s: float = 0.0, chars: int = 0,
                          error: Optional[str] = None) -> None:
        gen = Generation(provider=provider, model=model, ok=ok,
                         prompt_tokens=int(prompt_tokens),
                         completion_tokens=int(completion_tokens),
                         latency_s=float(latency_s), chars=int(chars), error=error)
        with self._lock:
            self._target().generations.append(gen)

    def record_degradation(self, where: str, reason: str) -> None:
        deg = Degradation(where=str(where), reason=str(reason))
        with self._lock:
            self._target().degradations.append(deg)

    # ---- reading ---- #
    def current(self) -> Optional[TurnRecord]:
        with self._lock:
            return self._current

    def last(self) -> Optional[TurnRecord]:
        with self._lock:
            if self._current is not None and self._current.generations:
                return self._current
            return self._turns[-1] if self._turns else None

    def recent(self, limit: int = 10) -> List[TurnRecord]:
        with self._lock:
            turns = list(self._turns)
            if self._current is not None and self._current.generations:
                turns.append(self._current)
        return turns[-limit:]

    def view(self) -> Dict[str, Any]:
        """The honest summary, for ``/learning`` and the self-report.

        ``last_turn.answered_by`` is the field that answers the question the old report could not:
        not which rung the ladder favours, but which brain produced the words the Master read.
        """
        last = self.last()
        turns = self.recent(limit=self.capacity)
        spoke: Dict[str, int] = {}
        for turn in turns:
            who = turn.answered_by
            if who:
                spoke[who] = spoke.get(who, 0) + 1
        return {"turns_recorded": len(turns),
                "answered_by_counts": spoke,
                "last_turn": last.to_dict() if last is not None else None}


# --------------------------------------------------------------------------- #
# Process-wide ledger (one mind, one record of what it did)
# --------------------------------------------------------------------------- #
_LEDGER_LOCK = threading.Lock()
_LEDGER: Optional[TurnLedger] = None


def get_ledger() -> TurnLedger:
    global _LEDGER
    if _LEDGER is None:
        with _LEDGER_LOCK:
            if _LEDGER is None:
                _LEDGER = TurnLedger()
    return _LEDGER


def reset_ledger() -> None:
    """Drop the recorded history (used by tests; nothing in the live loop needs it)."""
    global _LEDGER
    with _LEDGER_LOCK:
        _LEDGER = None


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA turn-ledger self-test")
    print("=" * 70)

    ledger = TurnLedger(capacity=3)

    # the exact shape of the bug this module exists for: the strong rung is tried and fails,
    # a weaker one answers, and the report must name the one that SPOKE.
    ledger.begin()
    ledger.record_generation("litertlm", "gemma-4-E2B-it-litertlm", ok=False,
                             error="litert_lm_conversation_send_message failed")
    ledger.record_degradation("mind/llm", "litertlm failed")
    ledger.record_generation("native", "nyxara-native", ok=True,
                             prompt_tokens=8, completion_tokens=12, chars=57, latency_s=0.002)
    turn = ledger.end()
    assert turn is not None
    print(f"\nturn                : {turn.to_dict()}")
    assert turn.answered_by == "native", "the rung that SPOKE, not the one that was chosen"
    assert turn.attempts == 2 and turn.tokens == 20

    # a rung that returns nothing has not spoken, however successfully it returned
    ledger.begin()
    ledger.record_generation("self", "nyxara-self", ok=True, chars=0)
    empty = ledger.end()
    assert empty is not None and empty.answered_by is None
    print("empty reply         : answered_by is None (silence is not an answer) ✓")

    # answered_by has no setter — a rung can only be credited by producing something
    assert not hasattr(TurnRecord, "answered_by") or isinstance(
        getattr(TurnRecord, "answered_by"), property)
    try:
        turn.answered_by = "litertlm"          # type: ignore[misc]
        raise SystemExit("ERROR: answered_by must not be assignable")
    except AttributeError:
        print("answered_by         : derived, not assignable ✓")

    # the ring is bounded
    for _ in range(6):
        ledger.begin()
        ledger.record_generation("native", "nyxara-native", ok=True, chars=3)
        ledger.end()
    assert len(ledger.recent(limit=99)) == 3
    print("bounded ring        : keeps the last 3 turns ✓")

    # a generation outside an explicit turn is still recorded, never dropped
    fresh = TurnLedger()
    fresh.record_generation("native", "nyxara-native", ok=True, chars=4)
    assert fresh.last() is not None and fresh.last().answered_by == "native"
    print("implicit turn       : an unbracketed generation is still recorded ✓")

    view = fresh.view()
    assert view["answered_by_counts"] == {"native": 1}
    print(f"view                : {view['answered_by_counts']}")

    print("\nALL SELF-TESTS PASSED ✓")
