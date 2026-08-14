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
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
    # The slower timescales. Each is a real pass over a real organ, and each is None/0 when its
    # cadence was not due — an absent number here means "not this beat", never "nothing happened".
    memory_promoted: int = 0
    memory_forgotten: int = 0
    questions_raised: int = 0
    abstractions_confirmed: int = 0
    abstractions_refuted: int = 0
    dreamed: bool = False
    # What the dream actually did. `dreamed` alone was a boolean that could be True while nothing
    # changed; these are the numbers that say whether the pass was worth running.
    replayed: int = 0
    dream_loss_before: float = 0.0
    dream_loss_after: float = 0.0
    generation: Optional[int] = None
    evolved: Optional[Dict[str, Any]] = None
    blocked: str = ""                 # non-empty when oversight refused the beat
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"beat": self.beat, "expanded": self.expanded, "grown": self.grown,
                "born": self.born, "consolidated": self.consolidated,
                "memory_promoted": self.memory_promoted,
                "memory_forgotten": self.memory_forgotten,
                "questions_raised": self.questions_raised,
                "abstractions_confirmed": self.abstractions_confirmed,
                "abstractions_refuted": self.abstractions_refuted,
                "dreamed": self.dreamed, "replayed": self.replayed,
                "dream_loss_before": round(self.dream_loss_before, 6),
                "dream_loss_after": round(self.dream_loss_after, 6),
                "dream_improved": self.dream_loss_after < self.dream_loss_before,
                "generation": self.generation, "evolved": self.evolved,
                "blocked": self.blocked, "ms": round(self.ms, 3)}


class PulseEngine:
    """The continuous beat: expand often, consolidate sometimes, rewrite herself rarely."""

    def __init__(self, brain: Any, *, every_s: float = 1.0,
                 wonder_every_s: float = 30.0, discover_every_s: float = 45.0,
                 memory_every_s: float = 120.0, dream_every_s: float = 600.0,
                 dream_batch: int = 16, dream_epochs: int = 4,
                 consolidate_every_s: float = 60.0, evolve_every_s: float = 300.0,
                 queue_capacity: int = 4096, enabled: bool = True) -> None:
        self.brain = brain
        self.every_s = max(0.05, float(every_s))
        self.consolidate_every_s = max(1.0, float(consolidate_every_s))
        # The slower timescales, fastest-first. Each is slower than the work it summarises:
        # wondering about gaps is pointless more often than gaps appear, and dreaming over the
        # same sixteen episodes every second would be replay without any new experience in it.
        self.wonder_every_s = max(1.0, float(wonder_every_s))
        self.discover_every_s = max(1.0, float(discover_every_s))
        self.memory_every_s = max(1.0, float(memory_every_s))
        self.dream_every_s = max(1.0, float(dream_every_s))
        self.dream_batch = max(1, int(dream_batch))
        self.dream_epochs = max(1, int(dream_epochs))
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
        if not self._due("pulse", self.every_s) and not force:
            return rep
        # A forced beat IS a beat, so it restarts the cadence. Without stamping the mark here the
        # next unforced call sees no previous beat and runs immediately, letting one extra pass
        # slip through right after every forced one.
        self._marks["pulse"] = time.monotonic()
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

            # 3. the slower timescales. Cognition does not run at one rate: perception and
            # response are immediate, working reasoning is slower, memory consolidation slower
            # still, and learning-how-to-learn slowest of all. Running these every beat would
            # make them noise; running them never is what left them decorative.
            if self._due("wonder", self.wonder_every_s):
                curiosity = getattr(self.brain, "curiosity", None)
                if curiosity is not None:
                    rep.questions_raised = len(curiosity.wonder())

            if self._due("discover", self.discover_every_s):
                discoverer = getattr(self.brain, "discoverer", None)
                if discoverer is not None:
                    found = discoverer.discover()
                    rep.abstractions_confirmed = found.confirmed
                    rep.abstractions_refuted = found.refuted

            if self._due("memory", self.memory_every_s):
                levels = getattr(self.brain, "levels", None)
                if levels is not None:
                    moved = levels.consolidate()
                    rep.memory_promoted = moved.promoted
                    rep.memory_forgotten = moved.forgotten

            # 4. dream — offline replay on the slowest cadence but one. Recent experience is
            # re-run so predictions can be tested and contradictions surface without a live turn
            # riding on the answer. Computational offline simulation; not conscious dreaming.
            if self._due("dream", self.dream_every_s):
                self._dream(rep)

            # 5. evolve — one self-rewrite attempt, on the slowest cadence
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

    def _dream(self, rep: PulseReport) -> None:
        """Offline replay that actually **trains**, and reports whether it helped.

        This used to touch timestamps and re-run discovery — a pass that could report
        ``dreamed=True`` while changing nothing that would ever affect an answer. Now it does the
        thing replay is for: re-run what happened, ask the readout what it would predict *now*,
        and backpropagate the difference.

        The loss is measured before and after on the **same** batch, so a dream that did not help
        says so rather than being credited for having run. Replay is still offline simulation —
        that is what replay is — but it now changes weights instead of timestamps.
        """
        try:
            levels = getattr(self.brain, "levels", None)
            discoverer = getattr(self.brain, "discoverer", None)
            readout = getattr(self.brain, "readout", None)
            fabric = getattr(self.brain, "fabric", None)
            if levels is None:
                return
            from nyxara.njp.levels import Level

            # Both experiential levels. Autobiographical entries are episodes too — they are
            # protected from *forgetting*, which is not a reason to exclude them from replay, and
            # reading only EPISODIC meant a session that talked mostly about the Master replayed
            # almost nothing.
            episodes = (levels.at(Level.EPISODIC)
                        + levels.at(Level.AUTOBIOGRAPHICAL))[-self.dream_batch:]
            batch: List[Tuple[Sequence[int], Sequence[int]]] = []
            for entry in episodes:
                levels.touch(entry.key)      # replay IS rehearsal — that is what it is for
                rep.replayed += 1
                # (what fired, what fired next) straight off the trace the settle recorded.
                trace = self._trace_for(entry, fabric)
                batch.extend(trace)

            if readout is not None and batch:
                rep.dream_loss_before = readout.loss_on(batch)
                for _ in range(self.dream_epochs):
                    readout.train_step(batch)
                rep.dream_loss_after = readout.loss_on(batch)

            if discoverer is not None and rep.replayed:
                discoverer.discover()
            rep.dreamed = bool(rep.replayed)
        except Exception:  # noqa: BLE001 — a failed dream changes nothing
            pass

    def _trace_for(self, entry: Any, fabric: Any) -> List[Tuple[Sequence[int], Sequence[int]]]:
        """Consecutive firing steps from a replayed episode, as (before, after) pairs.

        Re-settling the fabric would be the obvious approach and is the wrong one: it mutates the
        live substrate during what is supposed to be an *offline* pass, so the dream would change
        the very thing it is meant to be learning about.
        """
        out: List[Tuple[Sequence[int], Sequence[int]]] = []
        try:
            brain = self.brain
            store = getattr(getattr(brain, "levels", None), "store", None)
            trace_obj = store.recall_key(entry.key) if store is not None else None
            text = str(getattr(trace_obj, "text", "") or "")
            if not text or brain is None or not hasattr(brain, "encode"):
                return out
            cells = brain.encode(text)
            for i in range(len(cells) - 1):
                out.append(([cells[i]], [cells[i + 1]]))
            return out
        except Exception:  # noqa: BLE001
            return out

    def stats(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "beats": self.beats, "expansions": self.expansions,
                "queued": len(self.queue), "dropped": self.dropped,
                "blocked_beats": self.blocked_beats,
                "every_s": self.every_s,
                "consolidate_every_s": self.consolidate_every_s,
                "evolve_every_s": self.evolve_every_s,
                "timescales": {"pulse": self.every_s, "wonder": self.wonder_every_s,
                               "discover": self.discover_every_s,
                               "consolidate": self.consolidate_every_s,
                               "memory": self.memory_every_s, "dream": self.dream_every_s,
                               "evolve": self.evolve_every_s},
                "last": self.last.to_dict() if self.last is not None else None}
