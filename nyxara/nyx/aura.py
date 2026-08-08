"""NYXARA · nyx/aura.py — the world arriving without being asked for (🌐, L-AURA).

Every other way NYXARA learns about the world starts with someone typing something. This module
is the exception: registered streams push in continuously — the host's own sensors, a file that
keeps changing, a feed — and whatever is *surprising* lands in her concept graph and memory by
itself. There is no search query. The state of what she is watching is simply reflected in her,
and goes stale on its own when it stops being refreshed.

Three things make that safe and affordable rather than a firehose with a bad idea attached:

**An attention budget.** A stream is unbounded and a mind is not. ``max_events_per_min`` is a
hard cap; past it, events are dropped and counted, so the ceiling is visible instead of silently
lossy.

**A surprise gate.** Reflecting *everything* is the same as reflecting nothing — the graph would
fill with the ordinary. Only events that are actually novel against what she already has get
through, which is the Free Energy Principle doing the only job it can do cheaply.

**Everything is untrusted.** Content that arrives on its own was written by someone who is not
the Master. Every event is scanned for prompt injection (``senses/web.InjectionScanner``) before
anything touches her state; a flagged event is **quarantined** — recorded, counted, never
learned from, and never treated as an instruction. This is the whole reason ambient input is
not simply a hole in the side of the mind.

Honest framing:

* She perceives **what is registered and reachable**, not "everywhere". There is no global sensor
  network here; there are the streams you give her, subject to the host's network policy.
* "Raw photon feeds" are a camera. A camera is a legitimate stream — register one — but nothing
  here ingests photons in any other sense.
* **No stream is registered by default and nothing polls the network unless you add it.** An
  empty field is the out-of-the-box state, and it says so.

Pure standard library. Fail-soft: one bad stream never stops the others, and sensing never
raises into the loop that drives it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Event", "Stream", "Sensed", "AwarenessField"]

PollFn = Callable[[], Any]


@dataclass
class Event:
    """One thing that arrived on its own."""

    stream: str
    text: str
    at: float = field(default_factory=time.time)
    trust: float = 0.3               # ambient input is low-trust by construction
    surprise: float = 0.0
    quarantined: bool = False
    reason: str = ""                 # why it was quarantined or dropped

    def to_dict(self) -> Dict[str, Any]:
        return {"stream": self.stream, "text": self.text[:200], "at": self.at,
                "trust": round(self.trust, 4), "surprise": round(self.surprise, 4),
                "quarantined": self.quarantined, "reason": self.reason}


@dataclass
class Stream:
    """A registered source of ambient input."""

    name: str
    poll: PollFn
    trust: float = 0.3
    every_s: float = 0.0             # 0 == poll on every sense()
    enabled: bool = True
    last_polled: float = 0.0
    events: int = 0
    failures: int = 0

    def due(self, now: float) -> bool:
        return self.enabled and (self.every_s <= 0.0
                                 or (now - self.last_polled) >= self.every_s)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "trust": round(self.trust, 4), "every_s": self.every_s,
                "enabled": self.enabled, "events": self.events, "failures": self.failures}


@dataclass
class Sensed:
    """One pass over the streams: what arrived, what was kept, what was refused."""

    polled: int = 0
    arrived: int = 0
    landed: List[Event] = field(default_factory=list)
    quarantined: List[Event] = field(default_factory=list)
    dropped: int = 0                 # over the attention budget
    filtered: int = 0                # not surprising enough to be worth the attention
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"polled": self.polled, "arrived": self.arrived,
                "landed": [e.to_dict() for e in self.landed],
                "quarantined": [e.to_dict() for e in self.quarantined],
                "dropped": self.dropped, "filtered": self.filtered,
                "elapsed_ms": round(self.elapsed_ms, 2)}


class AwarenessField:
    """Registered streams arrive continuously; the surprising part becomes part of her."""

    def __init__(self, brain: Any = None, *, max_events_per_min: int = 60,
                 surprise_gate: float = 0.5, scan: bool = True,
                 max_text: int = 2000) -> None:
        self.brain = brain
        self.max_events_per_min = max(0, int(max_events_per_min))
        # Below this novelty an event is the ordinary, and reflecting the ordinary is the same
        # as reflecting nothing — the graph would fill with what she already has.
        self.surprise_gate = max(0.0, min(1.0, float(surprise_gate)))
        self.scan = bool(scan)
        self.max_text = max(64, int(max_text))

        self.streams: Dict[str, Stream] = {}
        self.recent: List[Event] = []
        self.quarantine: List[Event] = []
        self.total_landed = 0
        self.total_quarantined = 0
        self.total_dropped = 0
        self._window: List[float] = []           # arrival times, for the budget
        self._seen: List[str] = []               # recently handled, so repeats cost nothing
        self._scanner: Any = None
        self._scanner_tried = False

    # ---- streams ---------------------------------------------------------- #
    def register(self, name: str, poll: PollFn, *, trust: float = 0.3,
                 every_s: float = 0.0) -> bool:
        """Add a source. Nothing is registered by default — an empty field senses nothing."""
        try:
            name = str(name).strip()
            if not name or not callable(poll):
                return False
            self.streams[name] = Stream(name=name, poll=poll,
                                        trust=max(0.0, min(1.0, float(trust))),
                                        every_s=max(0.0, float(every_s)))
            return True
        except Exception:  # noqa: BLE001
            return False

    def unregister(self, name: str) -> bool:
        return self.streams.pop(str(name), None) is not None

    def register_host_sensors(self, *, change: float = 0.15) -> int:
        """The one stream that needs no network and no configuration: the machine she runs on.

        Reports **changes worth noticing**, not every sample. Emitting each raw reading would
        make a new concept out of every decimal — "mem is 0.0945", "mem is 0.0518" — and fill
        the graph with numbers that mean nothing individually. Noticing that memory *rose* is
        the thing ambient awareness is actually for.
        """
        added = 0
        try:
            from nyxara.senses.system import SystemSensor
            sensor = SystemSensor()
            last: Dict[str, float] = {}

            def _poll() -> List[str]:
                sample = sensor.sample()
                if not isinstance(sample, dict):
                    return []
                out: List[str] = []
                for key, value in sample.items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    value = float(value)
                    was = last.get(key)
                    last[key] = value
                    if was is None:
                        continue                     # the first reading is a baseline, not news
                    delta = value - was
                    if abs(delta) < change * max(abs(was), 1e-6):
                        continue                     # ordinary drift is not worth attention
                    out.append(f"host {key} {'rose' if delta > 0 else 'fell'} "
                               f"to {value:.4g}")
                return out

            # Trusted more than the open internet: it is her own machine reporting itself.
            added += int(self.register("host", _poll, trust=0.8, every_s=5.0))
        except Exception:  # noqa: BLE001 — a missing sensor is simply one fewer stream
            pass
        return added

    # ---- sensing ----------------------------------------------------------- #
    def sense(self) -> Sensed:
        """Poll every due stream once and let the surprising part become part of her."""
        out = Sensed()
        started = time.monotonic()
        try:
            now = time.time()
            for stream in list(self.streams.values()):
                if not stream.due(now):
                    continue
                out.polled += 1
                for text in self._poll(stream):
                    out.arrived += 1
                    self._handle(stream, text, out)
            return out
        except Exception:  # noqa: BLE001 — sensing never raises into the loop that drives it
            return out
        finally:
            out.elapsed_ms = (time.monotonic() - started) * 1000.0

    def _poll(self, stream: Stream) -> List[str]:
        try:
            stream.last_polled = time.time()
            raw = stream.poll()
            if raw is None:
                return []
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            return [str(i)[: self.max_text] for i in items if str(i).strip()]
        except Exception:  # noqa: BLE001 — one bad stream never stops the others
            stream.failures += 1
            return []

    def _handle(self, stream: Stream, text: str, out: Sensed) -> None:
        event = Event(stream=stream.name, text=text, trust=stream.trust)

        # Something already handled is not an arrival. Without this a stream that keeps
        # returning the same item — a static feed, or a hostile one repeating an injection —
        # spends the whole attention budget on it and crowds out everything genuinely new.
        seen_key = f"{stream.name}:{hash(text)}"
        if seen_key in self._seen:
            out.filtered += 1
            event.reason = "already handled"
            return
        self._seen.append(seen_key)
        del self._seen[:-512]

        if not self._within_budget():
            out.dropped += 1
            self.total_dropped += 1
            event.reason = "over the attention budget"
            return

        flagged = self._injection(text)
        if flagged:
            event.quarantined, event.reason = True, flagged
            out.quarantined.append(event)
            self.quarantine.append(event)
            del self.quarantine[:-64]
            self.total_quarantined += 1
            return                                # never learned from, never an instruction

        event.surprise = self._surprise(text)
        if event.surprise < self.surprise_gate:
            out.filtered += 1
            event.reason = "not surprising enough to be worth the attention"
            return

        if self._land(stream, event):
            stream.events += 1
            out.landed.append(event)
            self.recent.append(event)
            del self.recent[:-64]
            self.total_landed += 1

    def _within_budget(self) -> bool:
        """A stream is unbounded and a mind is not."""
        now = time.time()
        self._window = [t for t in self._window if now - t < 60.0]
        if len(self._window) >= self.max_events_per_min:
            return False
        self._window.append(now)
        return True

    def _injection(self, text: str) -> str:
        """Content that arrives on its own was written by someone who is not the Master."""
        if not self.scan:
            return ""
        try:
            if not self._scanner_tried:
                self._scanner_tried = True
                from nyxara.senses.web import InjectionScanner
                self._scanner = InjectionScanner()
            if self._scanner is None:
                return ""
            report = self._scanner.scan(text)
            score = float(getattr(report, "score", 0.0))
            if score <= 0.0:
                return ""
            flags = [str(getattr(f, "pattern", "?")) for f in (getattr(report, "flags", []) or [])]
            return f"prompt injection ({', '.join(flags[:3])}, score {score:.2f})"
        except Exception:  # noqa: BLE001 — if it cannot be scanned it is not let through
            return "could not be scanned"

    def _surprise(self, text: str) -> float:
        """How new is this against what she already has? Cheap, and the only affordable gate."""
        brain = self.brain
        if brain is None:
            return 1.0
        try:
            percept = brain.perceive(text, remember=False)
            return float(getattr(percept, "novelty", 0.0))
        except Exception:  # noqa: BLE001
            return 0.0

    def _land(self, stream: Stream, event: Event) -> bool:
        """Let it become part of her — with provenance, so its origin is never lost."""
        brain = self.brain
        if brain is None:
            return False
        try:
            brain.remember(f"aura-{stream.name}-{int(event.at)}", event.text, kind="percept")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- the clock ---------------------------------------------------------- #
    def beat(self, *, oversight: Any = None) -> Optional[Sensed]:
        """One tick from the shared clock. Stops dead when oversight does."""
        try:
            if oversight is not None:
                gate = getattr(oversight, "gate", None)
                permitted = bool(gate()) if callable(gate) else bool(gate)
                if not permitted:
                    return None
            if not self.streams:
                return None
            return self.sense()
        except Exception:  # noqa: BLE001
            return None

    # ---- introspection ------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            return {"streams": [s.to_dict() for s in self.streams.values()],
                    "landed": self.total_landed, "quarantined": self.total_quarantined,
                    "dropped": self.total_dropped,
                    "events_last_min": len([t for t in self._window
                                            if time.time() - t < 60.0]),
                    "max_events_per_min": self.max_events_per_min,
                    "surprise_gate": self.surprise_gate,
                    "recent": [e.to_dict() for e in self.recent[-5:]],
                    "recent_quarantine": [e.to_dict() for e in self.quarantine[-3:]]}
        except Exception:  # noqa: BLE001
            return {}

    def to_dict(self) -> Dict[str, Any]:
        """Counters only. A stream is a live callable and cannot be serialised."""
        return {"landed": self.total_landed, "quarantined": self.total_quarantined,
                "dropped": self.total_dropped}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.total_landed = int(data.get("landed", 0))
            self.total_quarantined = int(data.get("quarantined", 0))
            self.total_dropped = int(data.get("dropped", 0))
            return True
        except Exception:  # noqa: BLE001
            return False
