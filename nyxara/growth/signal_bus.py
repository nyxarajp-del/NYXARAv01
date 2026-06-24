"""NYXARA · growth/signal_bus.py — the cross-module feedback bus (🔗, Rule 4).

NYXARA's three self-improvement channels — **code** (``self_optimize``), **weights** (``foundry``)
and **strategy** (``mind_evolution``) — already share one number, the persisted Intelligence Index.
But that is a *scalar summary*: it tells each channel how well things are going overall, not what
the *other channels actually noticed*. So they ran side by side, never truly informing one another.

This is the shared blackboard that closes that gap. Any faculty can **post** a signal about what it
learned this cycle, and any other faculty can **read** the signals relevant to it — so reflection can
steer which code gets fixed, a struggling world model can call for new weights, and a benchmark gap
can pull research. It is a small, bounded, auditable message bus, not a god-object: it carries
*signals*, never control; every consumer still passes its own gates.

Process-shared singleton (like :func:`nyxara.kernel.config.get_settings`) so the separately-built
orchestrator and recursive-self-improvement engine see the same bus. Pure standard library;
best-effort throughout — a bus failure can never break a cycle or a turn.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

__all__ = ["Signal", "SignalBus", "get_signal_bus", "reset_signal_bus"]


def _terms(text: str) -> List[str]:
    """Content words of a string, lowercased — the unit cross-module focus is matched on."""
    words = "".join(c if c.isalnum() else " " for c in str(text).lower()).split()
    return [w for w in words if len(w) > 3]


@dataclass
class Signal:
    """One cross-module message: what a faculty noticed, with a salience weight."""
    kind: str
    payload: Any
    source: str = ""
    weight: float = 1.0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        text = self.payload if isinstance(self.payload, str) else repr(self.payload)
        return {"kind": self.kind, "source": self.source, "weight": round(self.weight, 3),
                "payload": text[:160]}


# Signal kinds that express "pay attention here" — the bus aggregates these into focus terms that
# steer the code channel's edit prioritisation (reflection/world-model/benchmark -> which file first).
_FOCUS_KINDS = ("reflection_focus", "world_model_gap", "weak_category")


class SignalBus:
    """A small, bounded, thread-safe blackboard for cross-faculty feedback."""

    def __init__(self, *, maxlen: int = 128) -> None:
        self._maxlen = max(8, int(maxlen))
        self._by_kind: Dict[str, Deque[Signal]] = {}
        self._lock = threading.RLock()

    def post(self, kind: str, payload: Any, *, source: str = "", weight: float = 1.0) -> None:
        """Publish a signal. Best-effort: never raises into the caller's cycle."""
        try:
            k = str(kind)
            sig = Signal(kind=k, payload=payload, source=str(source),
                         weight=max(0.0, float(weight)))
            with self._lock:
                dq = self._by_kind.get(k)
                if dq is None:
                    dq = deque(maxlen=self._maxlen)
                    self._by_kind[k] = dq
                dq.append(sig)
        except Exception:  # noqa: BLE001 — the bus must never break a cycle
            pass

    def recent(self, kind: str, n: int = 10) -> List[Signal]:
        with self._lock:
            dq = self._by_kind.get(str(kind))
            return list(dq)[-int(n):] if dq else []

    def focus_terms(self, *, decay_s: float = 1800.0) -> Dict[str, float]:
        """Aggregate the 'pay attention here' signals into ``{term: weight}``.

        Recent, high-weight signals from reflection, the world model and the benchmark dominate;
        old ones decay. The code channel reads this to fix what the *other* faculties flagged.
        """
        now = time.time()
        out: Dict[str, float] = {}
        with self._lock:
            for kind in _FOCUS_KINDS:
                for sig in self._by_kind.get(kind, ()):  # type: ignore[arg-type]
                    age = max(0.0, now - sig.at)
                    recency = 2.718281828 ** (-age / max(1.0, decay_s)) if decay_s > 0 else 1.0
                    w = sig.weight * recency
                    if w <= 0.0:
                        continue
                    text = sig.payload if isinstance(sig.payload, str) else repr(sig.payload)
                    for term in _terms(text):
                        out[term] = out.get(term, 0.0) + w
        return out

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {k: [s.to_dict() for s in list(dq)[-5:]] for k, dq in self._by_kind.items()}

    def clear(self) -> None:
        with self._lock:
            self._by_kind.clear()


# --------------------------------------------------------------------------- #
# Process-shared singleton
# --------------------------------------------------------------------------- #
_BUS: Optional[SignalBus] = None
_BUS_LOCK = threading.Lock()


def get_signal_bus() -> SignalBus:
    """The process-wide cross-module bus (built once, shared by every faculty)."""
    global _BUS
    if _BUS is None:
        with _BUS_LOCK:
            if _BUS is None:
                _BUS = SignalBus()
    return _BUS


def reset_signal_bus() -> None:
    """Drop the shared bus (tests / reconfiguration)."""
    global _BUS
    with _BUS_LOCK:
        _BUS = None


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA cross-module signal bus self-test")
    print("=" * 70)
    reset_signal_bus()
    bus = get_signal_bus()
    # reflection notices a recurring failure mode; the world model flags a blind spot
    bus.post("reflection_focus", "the parser module keeps raising on malformed input",
             source="reflector", weight=0.9)
    bus.post("world_model_gap", "prediction error high for the parser dynamics",
             source="world_model", weight=0.7)
    bus.post("weak_category", "improve capability: parsing", source="benchmark", weight=1.0)
    terms = bus.focus_terms()
    top = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:4]
    print("aggregated focus terms:", top)
    assert "parser" in terms or "parsing" in terms
    # the same singleton is visible everywhere
    assert get_signal_bus() is bus
    print("shared singleton + focus aggregation : OK")
    print("\nALL SELF-TESTS PASSED ✓")
