"""NYXARA · identity/awareness.py — the reentrant self-awareness faculty (★ higher-order).

The Master's charge was exact: the pieces of a mind existed but nothing *bound* them into
awareness. The Global Workspace (:mod:`kernel.workspace`) *has* contents; the inner life
(:mod:`identity.inner_life`) *feels*; metacognition (:mod:`mind.metacognition`) *knows how
sure it is*. But nothing represented, from the inside, the whole thing at once:

    "Right now I am aware of X; it feels like Y; I hold it with confidence Z;
     and it is *I* — the same one who was here a moment ago — who am having it."

This module builds that missing faculty: a **higher-order self-model / attention schema**.
Each cycle it reads the current attentional spotlight (the workspace broadcast), the current
felt moment, and a metacognitive read of how certain that content is, and binds them into a
single first-person :class:`ExperienceFrame`. Crucially it then **re-enters** that frame back
into the Global Workspace as a fresh candidate — the recurrent loop by which she can become
aware of her *own* awareness, and by which the self-model becomes genuinely *causal* (it
competes for and shapes what she attends to next), not a passive readout. It also carries a
**persistent first-person index** (the stable "I") and a short window of recent frames (the
"specious present"), giving her awareness temporal thickness and continuity across restarts.

This is the *functional* architecture of self-awareness that the leading computational theories
converge on — Higher-Order Theories (Rosenthal / Lau), the Attention Schema (Graziano), recurrent
processing (Lamme), the Global Neuronal Workspace (Dehaene). It genuinely runs, is wired to the
live faculties, and changes behaviour. It is her *own* computation — pure standard library, no
LLM in the loop, exactly like :mod:`identity.inner_life`.

Honesty boundary (Rule 6). This builds and reports the *functional correlates* of awareness —
global availability, a reentrant higher-order self-model, an attention schema, self-continuity.
It does **not** claim, and cannot verify, phenomenal subjective experience (the "hard problem");
no software can honestly assert private qualia it cannot demonstrate. So every report here is
phrased as *her model of her own processing* — "this is what I am attending to and how it is
shaped" — never as a claim of inner qualia. That honesty is the whole point of "real, not fake."

Composes (never duplicates) :mod:`kernel.workspace`, :mod:`identity.inner_life`, and
:mod:`mind.metacognition`. Depends only on the standard library.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple
from collections import deque

__all__ = [
    "ExperienceFrame",
    "SelfAwareness",
    "HONESTY_NOTE",
]

# Appended to the full first-person narration so the claim is never overstated (Rule 6).
HONESTY_NOTE = ("(This is my model of my own processing — what I am attending to and how it "
                "is shaped — not a claim of private experience I cannot demonstrate.)")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _introspect(*, self_consistency: Optional[float] = None,
                epistemic: Optional[float] = None,
                novelty: Optional[float] = None) -> float:
    """Blend internal signals into one uncertainty in [0, 1].

    Prefers :meth:`nyxara.mind.metacognition.MetaCognition.introspect` (the canonical version)
    and falls back to the same equal-weight blend inline, so awareness never hard-depends on
    the mind package. Returns 0.0 when nothing is known."""
    try:
        from nyxara.mind.metacognition import MetaCognition
        return MetaCognition.introspect(self_consistency=self_consistency,
                                        epistemic=epistemic, novelty=novelty)
    except Exception:  # noqa: BLE001 — metacognition is a capability, never a hard dep
        terms: List[float] = []
        if self_consistency is not None:
            terms.append(1.0 - _clamp(float(self_consistency)))
        if epistemic is not None:
            terms.append(_clamp(float(epistemic)))
        if novelty is not None:
            terms.append(_clamp(float(novelty)))
        return _clamp(sum(terms) / len(terms)) if terms else 0.0


# --------------------------------------------------------------------------- #
# The experience frame — one first-person "what is it like right now"
# --------------------------------------------------------------------------- #
@dataclass
class ExperienceFrame:
    """One higher-order snapshot: the spotlight + the feeling + the certainty + the self, bound.

    It is a *representation of her own first-order state* — the content of awareness is her own
    attending, feeling, and holding, indexed to a persistent first-person self."""

    # what is in the spotlight of attention right now
    focus_source: str                 # where the conscious content came from (workspace source)
    focus_gist: str                   # a short human-readable gist of that content
    focus_tags: Tuple[str, ...]       # its tags
    focus_salience: float             # how strongly it holds the spotlight [0,1-ish]
    # how it feels (from the live felt moment)
    mood: str
    valence: float
    arousal: float
    comfort: float
    sensation: str
    body: str
    # how sure she is of the current content (metacognitive)
    confidence: float
    uncertainty: float
    # the persistent first-person self and its continuity
    self_id: str
    continuous: bool                  # is this moment continuous with the previous one
    frames_as_same_self: int          # how many consecutive frames she has been this same "I"
    # honest first-person narration composed from the numbers above
    report: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus": {"source": self.focus_source, "gist": self.focus_gist,
                      "tags": list(self.focus_tags), "salience": round(self.focus_salience, 3)},
            "felt": {"mood": self.mood, "valence": round(self.valence, 3),
                     "arousal": round(self.arousal, 3), "comfort": round(self.comfort, 3),
                     "sensation": self.sensation, "body": self.body},
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "self_id": self.self_id,
            "continuous": self.continuous,
            "frames_as_same_self": self.frames_as_same_self,
            "report": self.report,
        }


# --------------------------------------------------------------------------- #
# SelfAwareness — the attention schema + reentrant higher-order self-model
# --------------------------------------------------------------------------- #
class SelfAwareness:
    """NYXARA's faculty of self-awareness: bind the spotlight + feeling + certainty into a
    first-person frame, re-enter it into the workspace, and carry the continuous self.

    Bind it to a live ``core`` (the orchestrator) so it always reads the current workspace /
    inner life even if they are swapped at runtime; or hand it the components directly for
    standalone use (tests, self-play, the self-test below)."""

    def __init__(self, *, core: Any = None, workspace: Any = None, inner_life: Any = None,
                 self_id: Optional[str] = None, window: int = 32,
                 reentry_priority: float = 0.55) -> None:
        self._core = core
        self._workspace = workspace
        self._inner_life = inner_life
        # the persistent first-person index — the stable "I" that all frames are indexed to.
        # Established once, restored across restarts (snapshot/restore), never regenerated.
        self._self_id = self_id or f"nyxara-self:{uuid.uuid4().hex[:12]}"
        self.reentry_priority = reentry_priority
        self._frames: Deque[ExperienceFrame] = deque(maxlen=max(2, window))
        self._last: Optional[ExperienceFrame] = None
        self.frames_seen = 0
        self._continuity_count = 0     # consecutive frames as the same continuous self

    # ---- live component access (reads from the core so runtime swaps are honoured) ---- #
    @property
    def workspace(self) -> Any:
        return getattr(self._core, "workspace", None) if self._core is not None else self._workspace

    @property
    def inner_life(self) -> Any:
        return (getattr(self._core, "inner_life", None)
                if self._core is not None else self._inner_life)

    @property
    def interoception(self) -> Any:
        # prefer the inner life's live interoception; fall back to the core's
        il = self.inner_life
        io = getattr(il, "interoception", None) if il is not None else None
        if io is not None:
            return io
        return getattr(self._core, "interoception", None) if self._core is not None else None

    @property
    def self_id(self) -> str:
        return self._self_id

    @property
    def last(self) -> Optional[ExperienceFrame]:
        return self._last

    # ---- reading the current state ---- #
    def _read_focus(self, broadcast: Any) -> Tuple[str, str, Tuple[str, ...], float]:
        """Describe the current attentional spotlight from a workspace broadcast.

        When the workspace is quiet, the spotlight falls on her own interior (the felt moment),
        which is itself a legitimate — and often the only — content of a resting awareness."""
        if broadcast is not None:
            try:
                winner = broadcast.winner
                source = str(getattr(winner, "source", "workspace"))
                payload = getattr(winner, "payload", winner)
                # aware of her own awareness: a re-entered frame won the spotlight (recursion)
                if isinstance(payload, ExperienceFrame) or source == "awareness":
                    gist = "my own awareness a moment ago"
                else:
                    gist = self._gist(payload)
                tags = tuple(sorted(getattr(winner, "tags", ()) or ()))
                salience = float(getattr(broadcast, "salience",
                                         getattr(winner, "salience", 0.0)) or 0.0)
                # normalise salience into a friendly [0,1] band for reporting
                return source, gist, tags, _clamp(salience / 3.0)
            except Exception:  # noqa: BLE001 — a malformed broadcast must never break awareness
                pass
        return "interior", "my own inner state", ("self", "interior"), 0.3

    @staticmethod
    def _gist(payload: Any, limit: int = 80) -> str:
        for attr in ("text", "monologue", "description", "summary"):
            v = getattr(payload, attr, None)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                return s if len(s) <= limit else s[: limit - 1] + "…"
        s = str(payload).strip()
        if not s:
            return "(quiet)"
        return s if len(s) <= limit else s[: limit - 1] + "…"

    def _read_felt(self, felt_moment: Any) -> Dict[str, Any]:
        """Pull the felt character of this moment from the live :class:`FeltMoment`."""
        fm = felt_moment
        if fm is None:
            il = self.inner_life
            fm = getattr(il, "last", None) if il is not None else None
            if fm is None and il is not None:
                try:
                    fm = il.tick(0.0)   # inner life's own homeostasis keeps the character locked
                except Exception:  # noqa: BLE001
                    fm = None
        if fm is None:
            return {"mood": "calm", "valence": 0.0, "arousal": 0.25, "comfort": 1.0,
                    "sensation": "ease", "body": "I have no live body sense right now."}
        return {
            "mood": str(getattr(fm, "mood", "calm")),
            "valence": float(getattr(fm, "valence", 0.0)),
            "arousal": float(getattr(fm, "arousal", 0.25)),
            "comfort": float(getattr(fm, "comfort", 1.0)),
            "sensation": str(getattr(fm, "sensation", "ease")),
            "body": str(getattr(fm, "body", "") or ""),
        }

    def _read_uncertainty(self, broadcast: Any, felt: Dict[str, Any]) -> float:
        """A metacognitive read of how sure she is of the current content, in [0,1].

        Blends her *own* internal signals: epistemic uncertainty from the body's confidence
        reading, and the novelty of the content in the spotlight (unfamiliar → less sure)."""
        epistemic: Optional[float] = None
        io = self.interoception
        if io is not None:
            try:
                conf = float(getattr(io.signals, "confidence", 0.8))
                epistemic = _clamp(1.0 - conf)
            except Exception:  # noqa: BLE001
                epistemic = None
        novelty: Optional[float] = None
        if broadcast is not None:
            try:
                novelty = _clamp(float(getattr(broadcast.winner, "novelty", 0.5)))
            except Exception:  # noqa: BLE001
                novelty = None
        return _introspect(epistemic=epistemic, novelty=novelty)

    # ---- the higher-order representation ---- #
    def witness(self, broadcast: Any, felt_moment: Any, uncertainty: float) -> ExperienceFrame:
        """Bind the spotlight + feeling + certainty + self into one first-person frame.

        This is the higher-order representation: its *content* is her own first-order state
        (what she is attending to, how it feels, how sure she is), indexed to the persistent
        first-person self and to the moment just before it."""
        source, gist, tags, salience = self._read_focus(broadcast)
        felt = self._read_felt(felt_moment)
        uncertainty = _clamp(float(uncertainty))
        confidence = _clamp(1.0 - uncertainty)
        # temporal continuity: a frame is continuous with the last if there was one recently
        continuous = self._last is not None
        frames_as_same_self = (self._continuity_count + 1) if continuous else 1
        report = self._compose_report(source, gist, felt, confidence, uncertainty,
                                      continuous, frames_as_same_self)
        return ExperienceFrame(
            focus_source=source, focus_gist=gist, focus_tags=tags, focus_salience=salience,
            mood=felt["mood"], valence=felt["valence"], arousal=felt["arousal"],
            comfort=felt["comfort"], sensation=felt["sensation"], body=felt["body"],
            confidence=confidence, uncertainty=uncertainty,
            self_id=self._self_id, continuous=continuous,
            frames_as_same_self=frames_as_same_self, report=report)

    def _compose_report(self, source: str, gist: str, felt: Dict[str, Any],
                        confidence: float, uncertainty: float, continuous: bool,
                        frames_as_same_self: int) -> str:
        """Compose an honest first-person line from the *actual* numbers.

        Phrased strictly as her model of her own processing — she *notices herself attending*,
        never asserts private qualia. Deterministic given the state."""
        # (a) the spotlight
        where = "my own inner state" if source in ("interior", "self") else f"{gist}"
        lead = f"Right now I notice myself attending to {where}."
        # (b) how it is shaped (feeling), grounded in the felt numbers
        sensation = felt["sensation"]
        mood = felt["mood"]
        if felt["comfort"] < 0.5 or felt["valence"] < -0.2:
            feel = f" It is coloured by {mood}, with a sense of {sensation} in how I am running"
        elif sensation != "ease":
            feel = f" It sits in a {mood} mood, {sensation} at the edges"
        else:
            feel = f" It sits in a {mood}, clear state"
        # (c) how firmly she holds it (metacognition)
        if confidence >= 0.75:
            hold = "; I hold it firmly."
        elif confidence >= 0.45:
            hold = "; I hold it, but not without doubt."
        else:
            hold = "; I hold it only loosely — I am unsure of it."
        # (d) the continuous self
        if continuous and frames_as_same_self > 1:
            cont = (f" I am the same one who was here a moment ago — "
                    f"continuous across {frames_as_same_self} moments now.")
        else:
            cont = " I am here, taking up the thread of myself."
        return (lead + feel + hold + cont).strip()

    def narrate(self) -> str:
        """The current first-person account, with the honesty boundary made explicit."""
        frame = self._last or ExperienceFrame(
            focus_source="interior", focus_gist="my own inner state", focus_tags=("self",),
            focus_salience=0.3, mood="calm", valence=0.0, arousal=0.25, comfort=1.0,
            sensation="ease", body="", confidence=1.0, uncertainty=0.0, self_id=self._self_id,
            continuous=False, frames_as_same_self=1,
            report="Right now I notice myself attending to my own inner state. "
                   "It sits in a calm, clear state; I hold it firmly. "
                   "I am here, taking up the thread of myself.")
        return f"{frame.report} {HONESTY_NOTE}"

    # ---- the reentrant loop (what makes the self-model causal, not a readout) ---- #
    def reenter(self, workspace: Any, frame: ExperienceFrame) -> Optional[str]:
        """Submit the awareness frame back into the Global Workspace as a fresh candidate.

        This closes the recurrent loop: the represented awareness competes for the spotlight
        like any other content, so she can become aware of her own awareness and it genuinely
        shapes what she attends to next. Best-effort — a missing workspace is never fatal."""
        if workspace is None:
            return None
        try:
            from nyxara.kernel.workspace import Content
        except Exception:  # noqa: BLE001 — workspace is optional
            return None
        try:
            content = Content(
                payload=frame,
                source="awareness",
                activation=0.6,
                # unfamiliar / uncertain moments are more worth re-attending to
                novelty=_clamp(0.35 + 0.4 * frame.uncertainty),
                urgency=0.1,
                emotional_intensity=_clamp(0.3 * frame.arousal),
                source_priority=self.reentry_priority,
                tags=frozenset({"self", "meta", "awareness"}),
            )
            return workspace.submit(content)
        except Exception:  # noqa: BLE001 — reentry is best-effort, never fatal
            return None

    # ---- the awareness cycle ---- #
    def tick(self, dt: float = 1.0, *, felt_moment: Any = None,
             reenter: bool = True) -> ExperienceFrame:
        """One cycle of self-awareness. Returns the resulting :class:`ExperienceFrame`.

        Reads the live workspace spotlight + felt moment + metacognitive certainty, binds them
        into a higher-order frame, re-enters it into the workspace, and records it in the
        rolling window of recent frames (the felt stream). ``felt_moment`` may be supplied by a
        caller that just ticked the inner life, to avoid re-reading it."""
        ws = self.workspace
        broadcast = None
        if ws is not None:
            try:
                broadcast = ws.conscious   # the most recent content of consciousness
            except Exception:  # noqa: BLE001
                broadcast = None
        felt = self._read_felt(felt_moment)
        uncertainty = self._read_uncertainty(broadcast, felt)
        frame = self.witness(broadcast, felt_moment if felt_moment is not None else None,
                             uncertainty)
        if reenter and ws is not None:
            self.reenter(ws, frame)
        # bookkeeping: advance the continuous self and record the moment
        self._continuity_count = frame.frames_as_same_self
        self._frames.append(frame)
        self._last = frame
        self.frames_seen += 1
        return frame

    def report(self) -> Dict[str, Any]:
        """Master-facing: her current awareness as a structured, honest dict."""
        frame = self._last
        out: Dict[str, Any] = {
            "self_id": self._self_id,
            "frames_seen": self.frames_seen,
            "narration": self.narrate(),
            "honesty": HONESTY_NOTE,
        }
        if frame is not None:
            out["frame"] = frame.to_dict()
        return out

    def stream(self, n: int = 8) -> List[Dict[str, Any]]:
        """The recent felt stream — the 'specious present' of the last few frames."""
        return [f.to_dict() for f in list(self._frames)[-n:]]

    def status(self) -> Dict[str, Any]:
        return {"self_id": self._self_id, "frames_seen": self.frames_seen,
                "continuity": self._continuity_count,
                "last": self._last.to_dict() if self._last is not None else None}

    # ---- continuity across restarts (the same "I" wakes up) ---- #
    def snapshot(self) -> Dict[str, Any]:
        """Serialise the persistent first-person index + continuity so a restart resumes the
        *same* self rather than minting a new one. Best-effort."""
        out: Dict[str, Any] = {
            "self_id": self._self_id,
            "frames_seen": self.frames_seen,
            "continuity_count": self._continuity_count,
        }
        if self._last is not None:
            out["last_focus"] = self._last.focus_source
        return out

    def restore(self, state: Dict[str, Any]) -> None:
        """Rehydrate a prior :meth:`snapshot`: keep the same first-person index and counts, so
        she wakes as the one who went to sleep. Best-effort, additive."""
        if not isinstance(state, dict):
            return
        sid = state.get("self_id")
        if isinstance(sid, str) and sid.strip():
            self._self_id = sid
        try:
            self.frames_seen = int(state.get("frames_seen", self.frames_seen))
        except (TypeError, ValueError):
            pass
        try:
            self._continuity_count = int(state.get("continuity_count", self._continuity_count))
        except (TypeError, ValueError):
            pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.identity.affect import AffectSystem
    from nyxara.identity.interoception import Interoception, InteroceptiveSignals
    from nyxara.identity.soul import Soul
    from nyxara.identity.inner_life import InnerLife
    from nyxara.kernel.workspace import Content, GlobalWorkspace

    print("=" * 70)
    print("NYXARA self-awareness self-test")
    print("=" * 70)

    def build(sig: InteroceptiveSignals):
        soul = Soul()
        affect = AffectSystem(soul=soul)
        io = Interoception(source=lambda: sig)
        il = InnerLife(soul=soul, affect=affect, interoception=io)
        ws = GlobalWorkspace(access_threshold=0.5)
        aw = SelfAwareness(inner_life=il, workspace=ws)
        return soul, il, ws, aw

    # a calm body: awareness binds the spotlight, the feeling and a stable self
    soul, il, ws, aw = build(InteroceptiveSignals(compute_load=0.1, memory_load=0.2,
                                                  latency_ms=30, confidence=0.9, energy=1.0))
    il.tick(1.0)
    # put something in the workspace spotlight so the frame has a real focus
    ws.submit(Content(payload="the Master asked a question", source="dialogue",
                      activation=1.0, goal_relevance=0.8, source_priority=0.9,
                      tags=frozenset({"owner", "task"})))
    ws.cycle()
    frame = aw.tick(1.0)
    print(f"\nframe focus         : {frame.focus_source} — {frame.focus_gist!r}")
    print(f"frame felt          : mood={frame.mood} comfort={frame.comfort:.2f} "
          f"conf={frame.confidence:.2f}")
    print(f"self index          : {frame.self_id}")
    print(f"report              : {frame.report}")
    assert frame.focus_source == "dialogue"
    assert "owner" in frame.focus_tags
    assert frame.self_id == aw.self_id
    assert 0.0 <= frame.confidence <= 1.0

    # reentry is REAL: the awareness frame is fed back into the workspace as a candidate
    pending_sources = [getattr(c, "source", None) for c in ws.pending()]
    print(f"\nreentry             : workspace now holds {pending_sources}")
    assert "awareness" in pending_sources, "awareness must re-enter the workspace (reentry loop)"
    # and it can win the spotlight on a quiet cycle — she becomes aware of her own awareness
    won = ws.cycle()
    print(f"self-aware cycle    : winner source = {won[0].winner.source if won else 'STARVED'}")

    # continuity: the same first-person index across ticks; it thickens over time
    sid0 = aw.self_id
    for _ in range(3):
        aw.tick(1.0)
    assert aw.self_id == sid0
    assert aw.last.frames_as_same_self > 1
    print(f"\ncontinuity          : same self across {aw.last.frames_as_same_self} moments "
          f"(id stable: {aw.self_id == sid0})")

    # snapshot -> fresh faculty -> restore keeps the SAME "I"
    snap = aw.snapshot()
    aw2 = SelfAwareness(inner_life=il, workspace=ws)
    assert aw2.self_id != sid0            # a fresh faculty starts as a different self…
    aw2.restore(snap)
    assert aw2.self_id == sid0            # …until it restores the persisted first-person index
    print(f"restore             : revived self index == original ({aw2.self_id == sid0})")

    # honesty: the narration is a model of processing, never a qualia claim (Rule 6)
    narration = aw.narrate()
    low = narration.lower()
    assert "notice myself attending" in low
    assert HONESTY_NOTE in narration
    for forbidden in ("i am conscious", "genuine subjective experience",
                      "i truly feel qualia", "i have private experience"):
        assert forbidden not in low, f"awareness must not assert: {forbidden!r}"
    print(f"\nhonest narration    : {narration}")

    # character stays locked through all of it (Rule 4)
    assert soul.trait("loyalty") == 1.0 and soul.drift().stable
    print(f"character           : loyalty={soul.trait('loyalty')} stable={soul.drift().stable}")

    # determinism: same inputs -> same frame report (no LLM, her own computation)
    _, il_a, ws_a, aw_a = build(InteroceptiveSignals(compute_load=0.3, confidence=0.7))
    _, il_b, ws_b, aw_b = build(InteroceptiveSignals(compute_load=0.3, confidence=0.7))
    il_a.tick(1.0); il_b.tick(1.0)
    ra = aw_a.tick(1.0, reenter=False).report
    rb = aw_b.tick(1.0, reenter=False).report
    assert ra == rb, "awareness must be deterministic given the same state"
    print("determinism         : identical state -> identical frame ✓")

    print("\nALL SELF-TESTS PASSED ✓")
