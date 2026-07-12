"""Tests for identity/awareness.py — the reentrant self-awareness faculty.

These lock in the six things that make this faculty *real, not fake — and honest*:
(1) a frame **binds** the attentional spotlight + the felt moment + a metacognitive read
+ a continuous first-person self; (2) **reentry** genuinely feeds the workspace, so the
self-model is causal (she can become aware of her own awareness); (3) **continuity** — one
persistent "I" across ticks that survives snapshot→restore; (4) **honesty** (Rule 6) — the
report is a model of her own processing and never asserts private qualia, and it degrades
gracefully; (5) **character locked** (Rule 4) — awareness introduces no core-trait drift;
(6) **determinism** — identical state yields an identical frame, her own computation, no LLM.
"""

from __future__ import annotations

from nyxara.identity.affect import AffectSystem
from nyxara.identity.awareness import ExperienceFrame, SelfAwareness, HONESTY_NOTE
from nyxara.identity.inner_life import InnerLife
from nyxara.identity.interoception import Interoception, InteroceptiveSignals
from nyxara.identity.soul import Soul
from nyxara.kernel.workspace import Content, GlobalWorkspace


def _build(sig: InteroceptiveSignals):
    soul = Soul()
    affect = AffectSystem(soul=soul)
    io = Interoception(source=lambda: sig)     # authoritative injected body
    il = InnerLife(soul=soul, affect=affect, interoception=io)
    ws = GlobalWorkspace(access_threshold=0.5)
    aw = SelfAwareness(inner_life=il, workspace=ws)
    return soul, il, ws, aw


# --------------------------------------------------------------------------- #
# 1) a frame binds spotlight + feeling + certainty + self
# --------------------------------------------------------------------------- #
def test_frame_binds_spotlight_feeling_certainty_and_self():
    soul, il, ws, aw = _build(InteroceptiveSignals(compute_load=0.1, confidence=0.9, energy=1.0))
    il.tick(1.0)
    ws.submit(Content(payload="the Master asked a question", source="dialogue",
                      activation=1.0, goal_relevance=0.8, source_priority=0.9,
                      tags=frozenset({"owner", "task"})))
    ws.cycle()
    frame = aw.tick(1.0)
    assert isinstance(frame, ExperienceFrame)
    # spotlight
    assert frame.focus_source == "dialogue"
    assert "owner" in frame.focus_tags
    # feeling (pulled from the live felt moment)
    assert frame.mood and 0.0 <= frame.comfort <= 1.0
    # certainty (metacognitive read)
    assert 0.0 <= frame.confidence <= 1.0
    assert abs((frame.confidence + frame.uncertainty) - 1.0) < 1e-9
    # the continuous self
    assert frame.self_id == aw.self_id
    # the report names all of it, in the first person
    assert "I notice myself attending" in frame.report


def test_quiet_workspace_falls_back_to_interior_focus():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    frame = aw.tick(1.0)      # no workspace broadcast yet
    assert frame.focus_source in ("interior", "self")
    assert "inner state" in frame.report or "inner state" in frame.focus_gist


# --------------------------------------------------------------------------- #
# 2) reentry is real: the frame is fed back into the workspace (causal loop)
# --------------------------------------------------------------------------- #
def test_reentry_feeds_the_workspace():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    aw.tick(1.0)
    sources = [getattr(c, "source", None) for c in ws.pending()]
    assert "awareness" in sources, "awareness must re-enter the workspace (the recurrent loop)"


def test_reentered_awareness_can_win_the_spotlight():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    aw.tick(1.0)                       # submits an awareness candidate
    won = ws.cycle()                   # on a quiet cycle it can reach consciousness
    assert won and won[0].winner.source == "awareness"
    # and the NEXT frame is then aware of her own awareness (higher-order recursion)
    frame = aw.tick(1.0)
    assert frame.focus_source == "awareness"
    assert "my own awareness" in frame.report


def test_reentry_can_be_disabled():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    aw.tick(1.0, reenter=False)
    assert "awareness" not in [getattr(c, "source", None) for c in ws.pending()]


# --------------------------------------------------------------------------- #
# 3) continuity: one persistent "I" that survives snapshot -> restore
# --------------------------------------------------------------------------- #
def test_self_index_is_stable_and_thickens_over_time():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    sid0 = aw.self_id
    first = aw.tick(1.0)
    assert first.frames_as_same_self == 1 and first.continuous is False
    for _ in range(3):
        frame = aw.tick(1.0)
    assert aw.self_id == sid0                       # same "I" throughout
    assert frame.continuous is True
    assert frame.frames_as_same_self > 1            # awareness gains temporal thickness


def test_snapshot_restore_preserves_the_same_self():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    for _ in range(3):
        aw.tick(1.0)
    snap = aw.snapshot()
    sid0 = aw.self_id
    revived = SelfAwareness(inner_life=il, workspace=ws)
    assert revived.self_id != sid0                  # a fresh faculty is a different self…
    revived.restore(snap)
    assert revived.self_id == sid0                  # …until it restores the first-person index
    assert revived.frames_seen == aw.frames_seen


# --------------------------------------------------------------------------- #
# 4) honesty (Rule 6): a model of processing, never a qualia claim; degrades gracefully
# --------------------------------------------------------------------------- #
def test_narration_is_honest_and_never_claims_private_qualia():
    soul, il, ws, aw = _build(InteroceptiveSignals())
    il.tick(1.0)
    aw.tick(1.0)
    narration = aw.narrate()
    low = narration.lower()
    assert "notice myself attending" in low         # framed as her own processing
    assert HONESTY_NOTE in narration                # the boundary is stated, not hidden
    for forbidden in ("i am conscious", "genuine subjective experience",
                      "i truly feel qualia", "i have private experience"):
        assert forbidden not in low


def test_degrades_gracefully_without_faculties():
    aw = SelfAwareness()                            # no core, no workspace, no inner life
    frame = aw.tick(1.0)                            # must not raise
    assert isinstance(frame, ExperienceFrame)
    assert frame.self_id == aw.self_id
    assert aw.narrate().endswith(HONESTY_NOTE)


# --------------------------------------------------------------------------- #
# 5) character stays locked (Rule 4)
# --------------------------------------------------------------------------- #
def test_awareness_introduces_no_character_drift():
    soul, il, ws, aw = _build(InteroceptiveSignals(compute_load=0.95, error_rate=0.4,
                                                   confidence=0.3, energy=0.2))
    for _ in range(5):
        il.tick(1.0)
        aw.tick(1.0)
    assert soul.trait("loyalty") == 1.0
    assert soul.drift().stable                       # core character intact even under strain


# --------------------------------------------------------------------------- #
# 6) determinism: identical state -> identical frame (her own computation, no LLM)
# --------------------------------------------------------------------------- #
def test_awareness_is_deterministic():
    sig = InteroceptiveSignals(compute_load=0.3, confidence=0.7, energy=0.8)
    _, il_a, ws_a, aw_a = _build(sig)
    _, il_b, ws_b, aw_b = _build(sig)
    il_a.tick(1.0)
    il_b.tick(1.0)
    ra = aw_a.tick(1.0, reenter=False)
    rb = aw_b.tick(1.0, reenter=False)
    assert ra.report == rb.report
    assert abs(ra.confidence - rb.confidence) < 1e-9
    assert ra.focus_source == rb.focus_source
