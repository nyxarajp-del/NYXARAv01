"""NYXARA · njp/pulse.py — the beat she grows on (💓, NJP V.01).

A mind that only changes when spoken to is not learning; it is responding. This is the other half
of NJP's life: a beat that keeps running between turns, folding in what the last turns produced.

Each pulse does three things, on three different cadences, because they cost three different
amounts:

* **expand** (every pulse, ~1s) — drain the experience queue and run
  :meth:`~nyxara.njp.fabric.Fabric.expand` over it. This is the cheap one: it is the plasticity
  pass, and it is what makes growth continuous rather than only per-conversation.
* **consolidate** (slow) — compress the weakest structure so growth has room to continue, and
  close a generation in :mod:`nyxara.njp.ledger` so today has a row to be compared against
  tomorrow.
* **evolve** (slowest) — one self-rewrite attempt. A forge is not a thought and does not get to
  cost one.

**Oversight is checked first, every time.** A paused or scrammed mind does not grow, does not
consolidate and does not rewrite itself; an oversight gate that cannot be read is treated as STOP.
Autonomy is not extra permission.

**No thread is started here.** The pulse is driven by whatever clock the kernel already runs — the
autonomic loop or the heartbeat — because a background thread that mutates the brain underneath a
turn is a race, not a feature. :meth:`PulseEngine.beat` is safe to call as often as the host likes:
it does its own rate limiting, so a caller ticking at 10 Hz and one ticking at 0.5 Hz both get the
cadences configured here.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

__all__ = ["PulseReport", "PulseEngine"]


def _permitted(oversight: Any) -> bool:
    """An unreadable oversight gate means *do not act*, never *carry on*."""
    if oversight is None:
        return True
    try:
        check = getattr(oversight, "gate", None)
        if callable(check):
            return bool(check())
        return bool(check) if check is not None else True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class PulseReport:
    """What one beat actually did. Empty is a real and common answer."""

    beat: int = 0
    expanded: int = 0                 # experiences folded in this beat
    grown: int = 0                    # NEW synapses this beat
    born: int = 0                     # NEW cells this beat
    consolidated: bool = False
    generation: Optional[int] = None
    evolved: Optional[Dict[str, Any]] = None
    blocked: str = ""                 # non-empty when oversight refused the beat
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"beat": self.beat, "expanded": self.expanded, "grown": self.grown,
                "born": self.born, "consolidated": self.consolidated,
                "generation": self.generation, "evolved": self.evolved,
                "blocked": self.blocked, "ms": round(self.ms, 3)}


class PulseEngine:
    """The continuous beat: expand often, consolidate sometimes, rewrite herself rarely."""

    def __init__(self, brain: Any, *, every_s: float = 1.0,
                 consolidate_every_s: float = 60.0, evolve_every_s: float = 300.0,
                 queue_capacity: int = 4096, enabled: bool = True) -> None:
        self.brain = brain
        self.every_s = max(0.05, float(every_s))
        self.consolidate_every_s = max(1.0, float(consolidate_every_s))
        self.evolve_every_s = max(1.0, float(evolve_every_s))
        self.enabled = bool(enabled)

        # Bounded on purpose: if the host stops beating, experience must not accumulate without
        # limit. The oldest is dropped, and `dropped` says so rather than hiding it.
        self.queue: Deque[Any] = deque(maxlen=max(16, int(queue_capacity)))
        self.dropped = 0

        self.beats = 0
        self.expansions = 0
        self.blocked_beats = 0
        self._marks: Dict[str, float] = {}
        self.last: Optional[PulseReport] = None

    # ---- feeding ---------------------------------------------------------- #
    def submit(self, result: Any, *, outcome: float = 1.0) -> None:
        """Hand one lived turn to the beat. Called by the brain at the end of every think."""
        try:
            if len(self.queue) == self.queue.maxlen:
                self.dropped += 1
            self.queue.append((result, float(outcome)))
        except Exception:  # noqa: BLE001
            pass

    def _due(self, name: str, every_s: float) -> bool:
        """One clock for every cadence, so the beats cannot drift apart."""
        try:
            now = time.monotonic()
            last = self._marks.get(name, 0.0)
            if last and (now - last) < float(every_s):
                return False
            self._marks[name] = now
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the beat --------------------------------------------------------- #
    def beat(self, *, oversight: Any = None, force: bool = False) -> PulseReport:
        """One pulse. Rate-limited internally, so any caller cadence is safe."""
        rep = PulseReport(beat=self.beats)
        t0 = time.perf_counter()
        if not self.enabled:
            return rep
        if not _permitted(oversight):
            self.blocked_beats += 1
            rep.blocked = "oversight is paused or scrammed"
            rep.ms = (time.perf_counter() - t0) * 1000.0
            self.last = rep
            return rep
        if not force and not self._due("pulse", self.every_s):
            return rep
        try:
            self.beats += 1
            rep.beat = self.beats
            fabric = getattr(self.brain, "fabric", None)

            # 1. expand — the plasticity pass over everything lived since the last beat
            if fabric is not None:
                while self.queue:
                    result, outcome = self.queue.popleft()
                    growth = fabric.expand(outcome=outcome, result=result)
                    rep.expanded += 1
                    rep.grown += growth.grown
                    rep.born += growth.born
                    self.expansions += 1

            # 2. consolidate — make room, and close a generation
            if self._due("consolidate", self.consolidate_every_s):
                if fabric is not None:
                    fabric.consolidate()
                rep.consolidated = True
                ledger = getattr(self.brain, "ledger", None)
                if ledger is not None and fabric is not None:
                    evolver = getattr(self.brain, "evolver", None)
                    gen = ledger.record(
                        fabric_stats=fabric.stats(),
                        edits_kept=getattr(evolver, "kept", 0),
                        edits_rolled_back=getattr(evolver, "rolled_back", 0),
                        note="pulse")
                    rep.generation = gen.n

            # 3. evolve — one self-rewrite attempt, on the slowest cadence
            if self._due("evolve", self.evolve_every_s):
                evolver = getattr(self.brain, "evolver", None)
                if evolver is not None:
                    step = evolver.beat(oversight=oversight)
                    if step is not None:
                        rep.evolved = step.to_dict()
                    evolver.accelerate(oversight=oversight)

            rep.ms = (time.perf_counter() - t0) * 1000.0
            self.last = rep
            return rep
        except Exception:  # noqa: BLE001 — a background beat never raises into the host loop
            rep.ms = (time.perf_counter() - t0) * 1000.0
            self.last = rep
            return rep

    def stats(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "beats": self.beats, "expansions": self.expansions,
                "queued": len(self.queue), "dropped": self.dropped,
                "blocked_beats": self.blocked_beats,
                "every_s": self.every_s,
                "consolidate_every_s": self.consolidate_every_s,
                "evolve_every_s": self.evolve_every_s,
                "last": self.last.to_dict() if self.last is not None else None}
