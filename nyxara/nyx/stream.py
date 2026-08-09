"""NYXARA · nyx/stream.py — what happened while you were away (🌊, NYX V.02).

:mod:`nyxara.nyx.aura` already takes events in continuously, and :mod:`nyxara.nyx.car` already
thinks a self-directed thought between prompts. What neither does is **synthesise**: a thousand
events arrive, are absorbed one at a time, and nothing ever compresses them into something a
person could read on returning.

:class:`PerpetualStream` is that synthesis, at several time-scales at once:

* **Multi-scale digests** — seconds, minutes, hours, days. Each scale keeps its own window and
  its own summary, because "what just happened" and "what has been happening all week" are
  different questions and one buffer cannot answer both.
* **Compression, not accumulation.** A digest keeps what was *new*, what *repeated*, and what
  *changed* — not the raw stream. That is what makes it readable rather than a log.
* **A briefing on return.** New concepts, recurring themes, what her own agenda moved, and what
  her omega loop changed — in one paragraph.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **"Never sleeping" is true only while the process is running.** ``nyxara-daemon`` runs; a CLI
  session that exits does not. Ticks are counted so the claim stays checkable, and ``/scram``
  stops it dead.
* It rides the **existing** heartbeat — no second thread, which is the same rule V.01 kept.
* It digests **what actually arrived**: registered streams, her own turns and her own beats.
  Nothing here reaches out on its own; that is :mod:`nyxara.nyx.aura`'s job and its own
  network policy.
* A digest reports **what she saw**, not what it means. Where a scale has seen nothing, it says
  so rather than producing an empty-sounding summary.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

__all__ = ["Moment", "Digest", "PerpetualStream", "SCALES"]

#: The scales she keeps, and the window each one covers, in seconds.
SCALES: Tuple[Tuple[str, float], ...] = (
    ("seconds", 60.0),
    ("minutes", 3600.0),
    ("hours", 86400.0),
    ("days", 7 * 86400.0),
)


@dataclass
class Moment:
    """One thing that arrived. Kept only long enough to be compressed."""

    at: float = field(default_factory=time.time)
    kind: str = ""              # turn | event | beat | change
    concepts: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"at": round(self.at, 3), "kind": self.kind,
                "concepts": list(self.concepts), "detail": self.detail}


@dataclass
class Digest:
    """One scale, compressed: what was new, what repeated, what changed."""

    scale: str = ""
    window_s: float = 0.0
    moments: int = 0
    novel: List[str] = field(default_factory=list)
    recurring: List[Tuple[str, int]] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    @property
    def empty(self) -> bool:
        return self.moments == 0

    def render(self) -> str:
        if self.empty:
            return f"{self.scale:8}: nothing arrived in the last {self.window_s:g}s"
        parts = [f"{self.scale:8}: {self.moments} moment(s)"]
        if self.novel:
            parts.append("new: " + ", ".join(self.novel[:6]))
        if self.recurring:
            parts.append("recurring: " + ", ".join(f"{c}×{n}" for c, n in self.recurring[:4]))
        if self.changes:
            parts.append("changed: " + "; ".join(self.changes[:3]))
        return "  |  ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"scale": self.scale, "window_s": self.window_s, "moments": self.moments,
                "novel": list(self.novel),
                "recurring": [[c, n] for c, n in self.recurring],
                "changes": list(self.changes), "empty": self.empty,
                "at": round(self.at, 3)}


class PerpetualStream:
    """Absorb continuously, compress at several scales, and have a briefing ready."""

    def __init__(self, brain: Any = None, *, capacity: int = 2048,
                 every_s: float = 15.0, novel_cap: int = 12) -> None:
        self.brain = brain
        self.every_s = max(0.0, float(every_s))
        self.novel_cap = max(1, int(novel_cap))
        self._moments: Deque[Moment] = deque(maxlen=max(16, int(capacity)))
        self._seen: Dict[str, float] = {}          # concept → when she first met it

        self.ticks = 0
        self.absorbed = 0
        self.digests: Dict[str, Digest] = {}
        self._last_tick = 0.0
        self._last_briefed = time.time()

    # ---- absorbing ----------------------------------------------------------- #
    def absorb(self, kind: str, *, concepts: Optional[List[str]] = None,
               detail: str = "") -> bool:
        """Take one thing in. Compression happens on the beat, not here."""
        try:
            moment = Moment(kind=str(kind or "event"),
                            concepts=[str(c).strip().lower()
                                      for c in (concepts or []) if str(c).strip()][:24],
                            detail=str(detail or "").strip()[:240])
            if not moment.concepts and not moment.detail:
                return False
            self._moments.append(moment)
            self.absorbed += 1
            for concept in moment.concepts:
                self._seen.setdefault(concept, moment.at)
            return True
        except Exception:  # noqa: BLE001
            return False

    def absorb_turn(self, text: str) -> bool:
        """A turn of conversation — the commonest thing that arrives."""
        try:
            from nyxara.nyx.graph import concepts_in
            return self.absorb("turn", concepts=concepts_in(str(text or ""), cap=16),
                               detail=str(text or "")[:160])
        except Exception:  # noqa: BLE001
            return False

    # ---- the beat ------------------------------------------------------------- #
    def due(self, *, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        return self.every_s <= 0.0 or (now - self._last_tick) >= self.every_s

    def tick(self, *, oversight: Any = None, force: bool = False) -> Dict[str, Digest]:
        """Compress every scale. Rides the existing heartbeat — there is no second thread."""
        try:
            if self._stopped(oversight):
                return {}
            if not force and not self.due():
                return {}
            self._last_tick = time.time()
            self.ticks += 1
            self._absorb_own_state()
            for scale, window in SCALES:
                self.digests[scale] = self._compress(scale, window)
            return dict(self.digests)
        except Exception:  # noqa: BLE001 — a failed tick digests nothing; it does not crash
            return {}

    def _absorb_own_state(self) -> None:
        """Her own beats count as things that happened — that is most of what she does."""
        try:
            agenda = getattr(self.brain, "agenda", None)
            if agenda is not None:
                for goal in agenda.open_goals()[:3]:
                    if goal.findings:
                        last = goal.findings[-1]
                        if last.ok:
                            self.absorb("beat", concepts=[goal.subject],
                                        detail=f"{goal.kind} {goal.subject}: {last.finding}")
            omega = getattr(self.brain, "omega", None)
            if omega is not None and omega.ledger:
                step = omega.ledger[-1]
                if step.status == "promoted" and step.knob:
                    self.absorb("change", concepts=[step.knob],
                                detail=f"tuned {step.target}.{step.knob}: "
                                       f"{step.was} → {step.now}")
        except Exception:  # noqa: BLE001 — her own state is a source, never a requirement
            return

    def _compress(self, scale: str, window: float) -> Digest:
        """Keep what was new, what repeated, and what changed. Not the stream itself."""
        out = Digest(scale=scale, window_s=window)
        try:
            cutoff = time.time() - window
            inside = [m for m in self._moments if m.at >= cutoff]
            out.moments = len(inside)
            if not inside:
                return out
            counts: Counter = Counter()
            for moment in inside:
                counts.update(moment.concepts)
                if moment.kind == "change":
                    out.changes.append(moment.detail)
            # "New" means first met inside this window — which is what makes it new *at this
            # scale*, rather than new to her forever.
            out.novel = [c for c in counts
                         if self._seen.get(c, 0.0) >= cutoff][: self.novel_cap]
            out.recurring = [(c, n) for c, n in counts.most_common(8) if n > 1]
            out.changes = out.changes[:6]
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- the briefing ------------------------------------------------------------ #
    def briefing(self) -> str:
        """One paragraph for someone who has just come back."""
        try:
            if not self.digests:
                self.tick(force=True)
            away = max(0.0, time.time() - self._last_briefed)
            self._last_briefed = time.time()
            live = [d for d in self.digests.values() if not d.empty]
            if not live:
                return (f"Nothing arrived while you were away ({away:.0f}s). "
                        f"I absorb what is registered and what I do myself; I do not go "
                        f"looking on my own.")
            lines = [f"While you were away ({away:.0f}s), across {self.ticks} digest tick(s):"]
            for digest in live:
                lines.append("  " + digest.render())
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _stopped(oversight: Any) -> bool:
        if oversight is None:
            return False
        try:
            for attr in ("scrammed", "is_scrammed", "paused", "is_paused"):
                probe = getattr(oversight, attr, None)
                if probe is None:
                    continue
                if bool(probe() if callable(probe) else probe):
                    return True
            return False
        except Exception:  # noqa: BLE001 — an unreadable oversight is treated as STOP
            return True

    # ---- introspection --------------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        try:
            return {
                "ticks": self.ticks, "absorbed": self.absorbed,
                "held": len(self._moments), "concepts_seen": len(self._seen),
                "every_s": self.every_s,
                "scales": {s: self.digests[s].to_dict() for s in self.digests},
                "note": ("'never sleeping' is true only while the process is running — "
                         "nyxara-daemon runs, a CLI session that exits does not, and the tick "
                         "count keeps the claim checkable. It rides the existing heartbeat, "
                         "with no second thread. It digests what ACTUALLY ARRIVED — registered "
                         "streams, her turns, her own beats — and never reaches out on its "
                         "own. /scram stops it"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        try:
            if not self.digests:
                return ("no digest yet — I compress on the beat, and no beat has run.")
            return "\n".join(d.render() for d in self.digests.values())
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        # The raw stream is deliberately not persisted — compressing it is the whole point, so
        # storing it back would undo the compression.
        return {"ticks": self.ticks, "absorbed": self.absorbed,
                "last_briefed": self._last_briefed,
                "seen": {k: round(v, 2) for k, v in list(self._seen.items())[-512:]},
                "digests": [d.to_dict() for d in self.digests.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.ticks = int(data.get("ticks", 0))
            self.absorbed = int(data.get("absorbed", 0))
            self._last_briefed = float(data.get("last_briefed", time.time()))
            self._seen = {str(k): float(v) for k, v in (data.get("seen") or {}).items()}
            self.digests = {}
            for raw in data.get("digests", []) or []:
                if not isinstance(raw, dict) or not raw.get("scale"):
                    continue
                self.digests[str(raw["scale"])] = Digest(
                    scale=str(raw["scale"]), window_s=float(raw.get("window_s", 0.0)),
                    moments=int(raw.get("moments", 0)),
                    novel=[str(c) for c in (raw.get("novel") or [])],
                    recurring=[(str(p[0]), int(p[1])) for p in (raw.get("recurring") or [])
                               if isinstance(p, (list, tuple)) and len(p) == 2],
                    changes=[str(c) for c in (raw.get("changes") or [])])
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
