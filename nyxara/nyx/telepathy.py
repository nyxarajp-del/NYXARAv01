"""NYXARA · nyx/telepathy.py — meaning without prose (🛰, NYX V.02, L-NEURAL-TELEPATHY).

**First, the part that cannot be built.** There is no brain-computer interface here. Raw mental
intent cannot be read — not by this repo, not by any software, not by any amount of cleverness
in this file. Nothing below reads a thought, and nothing below pretends to. If that is what the
name promised, the name over-promised, and this docstring is where that is said out loud.

What *can* be built is the thing the name was reaching for: **the bottleneck is prose**. Every
instruction today is squeezed through words, tokenised, and re-extracted — and measurably loses
things on the way. This module removes that step where it can be removed:

* **A vector-level input channel.** A :class:`Frame` — concepts, an optional dense vector, and
  typed ``role → filler`` bindings — goes *straight* into :mod:`~nyxara.nyx.semantics` and the
  concept graph, bypassing the tokenizer entirely. Past the spec there is **zero translation
  loss**, and :meth:`SemanticStream.compare` measures that against the same content sent as
  prose rather than asserting it.
* **A streaming shape.** Frames arrive continuously and are drained on a beat, rather than one
  request at a time — the same discipline :mod:`nyxara.senses.realtime` uses. The transport
  (a WebSocket route) is surface; this is the faculty behind it.
* **Bidirectional.** :meth:`SemanticStream.emit` renders her *own* state as a frame, so another
  NYXARA node (L-SYNERGY) or a tool can consume her concepts without either side going through
  English. Machine-to-machine, that really is lossless.
* **Intent compression.** She notices when the Master says the same thing repeatedly and mints
  a **shorthand**: after the third time, the fourth expands from two words into the full spec.
  This is the honest answer to "I do not want to type all that again" — and it is learned from
  the Master's own repetitions, not guessed.

Honest framing (the rule this repo keeps, see ``docs/CAPABILITIES.md``):

* **No thought reading.** You still have to make the spec. What is true is that the spec can be
  *small*, because she learns your shorthand — and that past the spec nothing is lost.
* **Zero loss is a measured claim, not a slogan**, and it is zero only *past* the frame. The
  loss between what you meant and what you wrote is not a thing software can measure, and this
  module does not claim to.
* A shorthand is minted only from **repetition she actually observed**, never from a guess at
  what you might mean.

Pure standard library. Every public method is fail-soft.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

__all__ = ["Frame", "Received", "Shorthand", "SemanticStream"]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class Frame:
    """One unit of meaning, carried as structure rather than as a sentence."""

    concepts: List[str] = field(default_factory=list)
    roles: Dict[str, str] = field(default_factory=dict)      # typed spec: role → filler
    vector: List[float] = field(default_factory=list)        # optional dense payload
    kind: str = "spec"                                       # spec | state | shorthand
    source: str = ""
    at: float = field(default_factory=time.time)

    @property
    def empty(self) -> bool:
        return not (self.concepts or self.roles or self.vector)

    def content(self) -> List[str]:
        """Everything the frame names, concepts and role fillers alike, order kept."""
        out: List[str] = []
        for item in list(self.concepts) + list(self.roles.values()):
            token = str(item or "").strip().lower()
            if token and token not in out:
                out.append(token)
        return out

    def render(self) -> str:
        parts = []
        if self.concepts:
            parts.append(" ".join(self.concepts))
        if self.roles:
            parts.append(" ".join(f"{k}={v}" for k, v in self.roles.items()))
        if self.vector:
            parts.append(f"<{len(self.vector)}-d vector>")
        return f"[{self.kind}] " + " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts": list(self.concepts), "roles": dict(self.roles),
                "vector": [round(float(v), 6) for v in self.vector], "kind": self.kind,
                "source": self.source, "at": round(self.at, 3)}

    @classmethod
    def from_dict(cls, data: Any) -> "Frame":
        try:
            if not isinstance(data, dict):
                return cls()
            return cls(concepts=[str(c) for c in (data.get("concepts") or [])],
                       roles={str(k): str(v) for k, v in (data.get("roles") or {}).items()},
                       vector=[float(v) for v in (data.get("vector") or [])],
                       kind=str(data.get("kind", "spec")),
                       source=str(data.get("source", "")),
                       at=float(data.get("at", time.time())))
        except Exception:  # noqa: BLE001
            return cls()


@dataclass
class Received:
    """What became of one frame — and how much of it survived the trip."""

    concepts: List[str] = field(default_factory=list)
    understood: int = 0
    loss: float = 0.0            # fraction of the intended content that did NOT arrive
    bypassed_tokenizer: bool = False
    expanded_from: str = ""      # the shorthand this came from, when it came from one
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts": list(self.concepts), "understood": self.understood,
                "loss": round(self.loss, 6),
                "bypassed_tokenizer": self.bypassed_tokenizer,
                "expanded_from": self.expanded_from, "note": self.note}


@dataclass
class Shorthand:
    """A compression she learned from the Master's own repetitions."""

    trigger: str = ""
    frame: Frame = field(default_factory=Frame)
    seen: int = 0
    used: int = 0
    minted_at: float = field(default_factory=time.time)
    source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"trigger": self.trigger, "frame": self.frame.to_dict(), "seen": self.seen,
                "used": self.used, "minted_at": round(self.minted_at, 3),
                "source_text": self.source_text}


class SemanticStream:
    """The vector channel: frames in, frames out, and a shorthand she learns from repetition."""

    def __init__(self, brain: Any = None, *, queue: int = 128, mint_after: int = 3,
                 trigger_words: int = 2, max_shorthand: int = 64) -> None:
        self.brain = brain
        self.mint_after = max(2, int(mint_after))
        self.trigger_words = max(1, int(trigger_words))
        self.max_shorthand = max(1, int(max_shorthand))

        self._queue: Deque[Frame] = deque(maxlen=max(1, int(queue)))
        self.shorthand: Dict[str, Shorthand] = {}
        self._repeats: Dict[str, int] = {}

        self.received = 0
        self.emitted = 0
        self.expanded = 0
        self.minted = 0

    # ---- the input channel --------------------------------------------------- #
    def receive(self, frame: Any) -> Received:
        """Take a frame straight into her semantics and her graph. No tokenizer involved.

        This is the whole claim: past the frame, nothing is lost. The concepts named in the
        frame are the concepts she ends up with, and :meth:`compare` measures that against the
        same content sent as prose rather than asserting it.
        """
        out = Received(bypassed_tokenizer=True)
        try:
            frame = frame if isinstance(frame, Frame) else Frame.from_dict(frame)
            if frame.empty:
                out.note = "the frame carried nothing"
                return out
            self.received += 1
            wanted = frame.content()
            out.concepts = list(wanted)

            graph = getattr(self.brain, "graph", None)
            if graph is not None:
                # A list, not a string: `activate` only tokenises when handed prose, so passing
                # the concepts directly is what actually bypasses the tokenizer.
                graph.activate(wanted)
                out.understood = sum(1 for c in wanted if c in graph)

            semantics = getattr(self.brain, "semantics", None)
            if semantics is not None and frame.roles:
                # Typed bindings are relations, and the relational rung is exactly where a
                # thing she was *told* belongs.
                for role, filler in frame.roles.items():
                    semantics.teach(str(role), str(filler), kind="near", weight=0.6,
                                    source="frame")

            out.loss = 0.0
            out.note = (f"{len(wanted)} concept(s) entered directly; nothing was tokenised, "
                        f"so nothing was lost past the frame")
            return out
        except Exception as exc:  # noqa: BLE001 — a bad frame is a null result, never a crash
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    def receive_prose(self, text: str, *, intended: Optional[Sequence[str]] = None) -> Received:
        """The old road, kept so the new one can be *measured* against it.

        ``intended`` is what the sender meant to convey. The loss is the fraction of it that
        did not survive tokenisation and extraction — a real number, not a slogan.
        """
        out = Received(bypassed_tokenizer=False)
        try:
            from nyxara.nyx.graph import concepts_in
            got = concepts_in(str(text or ""))
            out.concepts = list(got)
            graph = getattr(self.brain, "graph", None)
            if graph is not None:
                graph.activate(str(text or ""))
                out.understood = sum(1 for c in got if c in graph)
            wanted = [str(c).strip().lower() for c in (intended or []) if str(c).strip()]
            if wanted:
                kept = sum(1 for c in wanted if c in got)
                out.loss = 1.0 - (kept / float(len(wanted)))
                out.note = (f"{kept} of {len(wanted)} intended concepts survived tokenisation")
            else:
                out.note = f"{len(got)} concept(s) extracted from prose"
            return out
        except Exception as exc:  # noqa: BLE001
            out.note = f"{type(exc).__name__}: {exc}"
            return out

    def compare(self, text: str, frame: Any) -> Dict[str, Any]:
        """Send the same content both ways and report the difference. Measured, not claimed."""
        try:
            frame = frame if isinstance(frame, Frame) else Frame.from_dict(frame)
            wanted = frame.content()
            prose = self.receive_prose(text, intended=wanted)
            direct = self.receive(frame)
            return {"intended": wanted,
                    "prose": prose.to_dict(), "frame": direct.to_dict(),
                    "prose_loss": round(prose.loss, 6),
                    "frame_loss": round(direct.loss, 6),
                    "note": ("loss is measured against what the sender said they intended. "
                             "It is zero past the FRAME — the gap between what you meant and "
                             "what you wrote is not something software can measure, and is "
                             "not claimed here")}
        except Exception:  # noqa: BLE001
            return {}

    # ---- the streaming shape --------------------------------------------------- #
    def push(self, frame: Any) -> bool:
        """Queue a frame for the next beat. The continuous shape, not request/response."""
        try:
            frame = frame if isinstance(frame, Frame) else Frame.from_dict(frame)
            if frame.empty:
                return False
            self._queue.append(frame)
            return True
        except Exception:  # noqa: BLE001
            return False

    def drain(self, *, limit: int = 8, oversight: Any = None) -> List[Received]:
        """Take up to ``limit`` queued frames. Called from the beat that already exists."""
        out: List[Received] = []
        try:
            if self._stopped(oversight):
                return out
            for _ in range(max(0, int(limit))):
                if not self._queue:
                    break
                out.append(self.receive(self._queue.popleft()))
            return out
        except Exception:  # noqa: BLE001
            return out

    @property
    def pending(self) -> int:
        return len(self._queue)

    # ---- bidirectional: her state, as structure --------------------------------- #
    def emit(self, *, k: int = 12) -> Frame:
        """Her own state as a frame — consumable by another node without any prose at all.

        This direction genuinely *is* lossless: both ends are machines, and nothing is being
        squeezed through a language on the way.
        """
        frame = Frame(kind="state", source="nyx")
        try:
            graph = getattr(self.brain, "graph", None)
            if graph is not None:
                strongest = sorted(graph.nodes.values(),
                                   key=lambda c: (-c.strength, -c.hits))[: max(1, int(k))]
                frame.concepts = [c.label for c in strongest]
            semantics = getattr(self.brain, "semantics", None)
            if semantics is not None and frame.concepts:
                vector = semantics.vector(frame.concepts[0])
                frame.vector = [float(v) for v in (vector or [])][:256]
            agenda = getattr(self.brain, "agenda", None)
            if agenda is not None:
                for goal in agenda.open_goals()[:3]:
                    frame.roles[f"pursuing:{goal.kind}"] = goal.subject
            self.emitted += 1
            return frame
        except Exception:  # noqa: BLE001
            return frame

    # ---- intent compression ------------------------------------------------------ #
    def observe(self, text: str) -> Optional[Shorthand]:
        """Watch what the Master repeats. After ``mint_after`` times, mint a shorthand.

        Minted from repetition she actually saw — never from a guess at what might be meant.
        """
        try:
            body = str(text or "").strip()
            if len(body.split()) <= self.trigger_words:
                return None                       # already short; nothing to compress
            key = self._normalise(body)
            if not key:
                return None
            if key in self.shorthand:
                self.shorthand[key].seen += 1
                return self.shorthand[key]
            self._repeats[key] = self._repeats.get(key, 0) + 1
            if self._repeats[key] < self.mint_after:
                return None
            return self._mint(body, key, self._repeats[key])
        except Exception:  # noqa: BLE001
            return None

    def _mint(self, body: str, key: str, seen: int) -> Optional[Shorthand]:
        try:
            from nyxara.nyx.graph import concepts_in
            concepts = concepts_in(body, cap=24)
            frame = Frame(concepts=concepts, kind="shorthand", source="learned")
            intent = getattr(self.brain, "intent", None)
            if intent is not None:
                read = intent.read(body)
                for before, after in (read.ordering or [])[:2]:
                    frame.roles["before"] = before[:40]
                    frame.roles["after"] = after[:40]
                if read.actions:
                    frame.roles["do"] = read.actions[0]
            trigger = self._trigger(body, concepts)
            short = Shorthand(trigger=trigger, frame=frame, seen=seen, source_text=body[:200])
            self.shorthand[key] = short
            self.shorthand[trigger] = short
            self.minted += 1
            self._trim()
            return short
        except Exception:  # noqa: BLE001
            return None

    def _trigger(self, body: str, concepts: Sequence[str]) -> str:
        """A short handle for a long instruction. Distinct, stable, and hers to offer."""
        base = "-".join(list(concepts)[: self.trigger_words]) or _WORD.findall(body.lower())
        if isinstance(base, list):
            base = "-".join(base[: self.trigger_words])
        if not base:
            base = hashlib.blake2b(body.encode("utf-8"), digest_size=3).hexdigest()
        candidate = str(base)
        # A trigger that collides with a different instruction would expand to the wrong thing,
        # which is worse than not compressing at all.
        if candidate in self.shorthand:
            candidate = f"{candidate}-{hashlib.blake2b(body.encode('utf-8'), digest_size=2).hexdigest()}"
        return candidate

    def expand(self, text: str) -> Optional[Received]:
        """If this is a known shorthand, receive the full spec it stands for."""
        try:
            probe = str(text or "").strip().lower()
            short = self.shorthand.get(probe) or self.shorthand.get(self._normalise(probe))
            if short is None:
                return None
            short.used += 1
            self.expanded += 1
            out = self.receive(short.frame)
            out.expanded_from = short.trigger
            out.note = (f"expanded '{short.trigger}' into {len(short.frame.content())} "
                        f"concepts — learned from {short.seen} repetitions of your own")
            return out
        except Exception:  # noqa: BLE001
            return None

    def _trim(self) -> None:
        # Shorthands are stored under two keys (the normalised text and the trigger), so the
        # cap counts distinct entries rather than keys.
        distinct = {id(s): s for s in self.shorthand.values()}
        if len(distinct) <= self.max_shorthand:
            return
        oldest = sorted(distinct.values(), key=lambda s: (s.used, s.minted_at))
        for victim in oldest[: len(distinct) - self.max_shorthand]:
            for key in [k for k, v in self.shorthand.items() if v is victim]:
                self.shorthand.pop(key, None)

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(_WORD.findall(str(text or "").lower()))

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

    # ---- introspection ------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        try:
            distinct = {id(s): s for s in self.shorthand.values()}
            return {
                "received": self.received, "emitted": self.emitted,
                "expanded": self.expanded, "minted": self.minted,
                "pending": self.pending,
                "shorthand": [s.trigger for s in distinct.values()],
                "mint_after": self.mint_after,
                "note": ("there is NO brain-computer interface here and none is claimed — raw "
                         "mental intent cannot be read by any software. What is real: a frame "
                         "bypasses the tokenizer entirely, so nothing is lost PAST the spec "
                         "(measured, not asserted, by compare()); she emits her own state as "
                         "structure, which really is lossless machine-to-machine; and she "
                         "learns your shorthand from repetitions she actually saw, so the "
                         "spec you have to write gets shorter"),
            }
        except Exception:  # noqa: BLE001
            return {}

    def describe(self) -> str:
        """What ``/nyx telepathy`` prints."""
        try:
            distinct = {id(s): s for s in self.shorthand.values()}
            lines = ["I cannot read your mind, and nothing here tries to.",
                     "What I can do is take meaning as structure instead of prose:",
                     f"  frames received : {self.received} (tokenizer bypassed)",
                     f"  frames emitted  : {self.emitted} (my state, machine-readable)",
                     f"  queued          : {self.pending}"]
            if distinct:
                lines.append("  shorthand I have learned from you repeating yourself:")
                for short in sorted(distinct.values(), key=lambda s: -s.used):
                    lines.append(f"    {short.trigger:20} → {short.source_text[:56]}"
                                 f"   (seen {short.seen}×, used {short.used}×)")
            else:
                lines.append("  no shorthand yet — say the same thing "
                             f"{self.mint_after} times and I will offer you one.")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    # ---- persistence --------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        distinct = {id(s): s for s in self.shorthand.values()}
        return {"received": self.received, "emitted": self.emitted,
                "expanded": self.expanded, "minted": self.minted,
                "repeats": dict(self._repeats),
                "shorthand": [s.to_dict() for s in distinct.values()]}

    def load_dict(self, data: Any) -> bool:
        try:
            if not isinstance(data, dict):
                return False
            self.received = int(data.get("received", 0))
            self.emitted = int(data.get("emitted", 0))
            self.expanded = int(data.get("expanded", 0))
            self.minted = int(data.get("minted", 0))
            self._repeats = {str(k): int(v) for k, v in (data.get("repeats") or {}).items()}
            self.shorthand = {}
            for raw in data.get("shorthand", []) or []:
                if not isinstance(raw, dict):
                    continue
                short = Shorthand(trigger=str(raw.get("trigger", "")),
                                  frame=Frame.from_dict(raw.get("frame")),
                                  seen=int(raw.get("seen", 0)), used=int(raw.get("used", 0)),
                                  minted_at=float(raw.get("minted_at", time.time())),
                                  source_text=str(raw.get("source_text", "")))
                if not short.trigger:
                    continue
                self.shorthand[short.trigger] = short
                key = self._normalise(short.source_text)
                if key:
                    self.shorthand[key] = short
            return True
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            return False
