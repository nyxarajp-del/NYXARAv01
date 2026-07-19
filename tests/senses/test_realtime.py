"""Tests for nyxara.senses.realtime — the always-on perception loop.

Hermetic: no real devices, no threads except the lifecycle test, no Whisper/OCR. Every
capture is real bytes (stdlib-encoded PNG / WAV) delivered through a stub LiveSensor, so
the loop's detectors run on genuine media exactly as they would live.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from nyxara.senses.binding import Percept
from nyxara.senses.generate import encode_png
from nyxara.senses.live import LiveSensor
from nyxara.senses.realtime import (
    ModalityChannel,
    PerceptEvent,
    RealtimePerception,
    _floats_to_wav,
)

RATE = 16000


# --------------------------------------------------------------------------- #
# Real media builders (pure stdlib)
# --------------------------------------------------------------------------- #
def checker_png(v: int, size: int = 64) -> bytes:
    """A real PNG checkerboard whose contrast depends on ``v`` (different v ⇒ new image)."""
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            px = v if (x // 8 + y // 8) % 2 else 255 - v
            row.extend((px, px, px))
        rows.append(bytes(row))
    return encode_png(size, size, rows)


def half_moved_png(v: int, size: int = 64) -> bytes:
    """Like checker_png but with the bottom half inverted — regional (motion-like) change."""
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            px = v if (x // 8 + y // 8) % 2 else 255 - v
            if y >= size // 2:
                px = 255 - px
            row.extend((px, px, px))
        rows.append(bytes(row))
    return encode_png(size, size, rows)


def tone_wav(amp: float, secs: float = 1.0) -> bytes:
    sig = [amp * math.sin(2 * math.pi * 440 * i / RATE) for i in range(int(secs * RATE))]
    return _floats_to_wav(sig, rate=RATE)


# --------------------------------------------------------------------------- #
# Stub sensors — scripted real bytes, honest (None, note) when the script runs out
# --------------------------------------------------------------------------- #
class StubSensor(LiveSensor):
    def __init__(self, frames: List[bytes] = (), clips: List[bytes] = (),
                 cam_frames: List[bytes] = ()) -> None:
        super().__init__()
        self.frames = list(frames)
        self.clips = list(clips)
        self.cam_frames = list(cam_frames)

    def available(self):
        return {"camera": bool(self.cam_frames), "screen": bool(self.frames),
                "mic": bool(self.clips)}

    def capture_screen(self, monitor: int = 0):
        return (self.frames.pop(0), "screen: stub") if self.frames \
            else (None, "screen: script exhausted")

    def capture_camera(self, index: int = 0):
        return (self.cam_frames.pop(0), "camera: stub") if self.cam_frames \
            else (None, "camera: script exhausted")

    def capture_microphone(self, seconds: Optional[float] = None,
                           rate: Optional[int] = None):
        return (self.clips.pop(0), "mic: stub") if self.clips \
            else (None, "mic: script exhausted")


class DeadSensor(LiveSensor):
    """A box with no devices at all — every capture is an honest miss."""

    def available(self):
        return {"camera": False, "screen": False, "mic": False}

    def capture_camera(self, index: int = 0):
        return None, "camera: no device"

    def capture_screen(self, monitor: int = 0):
        return None, "screen: no device"

    def capture_microphone(self, seconds=None, rate=None):
        return None, "mic: no device"


def make_loop(sensor: LiveSensor, **kw) -> RealtimePerception:
    """A loop with all cadence delays zeroed so tests drive it synchronously."""
    rp = RealtimePerception(sensor, settings=object(), journal_path=None, **kw)
    rp.interval_s = rp.burst_interval_s = rp.idle_interval_s = 0.0
    rp.min_escalation_interval_s = 0.0
    for ch in rp.channels.values():
        ch.interval = ch.base_interval = 0.0
        ch.next_due = 0.0
    return rp


def drive(rp: RealtimePerception, n: int) -> List[PerceptEvent]:
    out: List[PerceptEvent] = []
    for _ in range(n):
        out.extend(rp.tick_once())
        for ch in rp.channels.values():
            ch.next_due = 0.0
    return out


# --------------------------------------------------------------------------- #
# Hearing: VAD + self-calibration + endpointing
# --------------------------------------------------------------------------- #
def test_silence_produces_no_event_and_teaches_the_noise_floor():
    rp = make_loop(StubSensor(clips=[tone_wav(0.0), tone_wav(0.001)]))
    events = drive(rp, 2)
    assert [e for e in events if e.modality == "mic"] == []
    assert rp._vad_threshold() >= rp.vad_rms  # floor learned, threshold real


def test_loud_tone_is_heard_as_one_event_with_honest_transcript():
    rp = make_loop(StubSensor(clips=[tone_wav(0.5), tone_wav(0.0)]))
    events = drive(rp, 1)
    mic = [e for e in events if e.modality == "mic"]
    assert len(mic) == 1
    assert mic[0].kind in ("speech", "sound")
    # no Whisper in CI ⇒ honestly no transcript, with the reason carried in the note
    assert mic[0].transcript is None
    assert mic[0].percept.data.get("duration_s", 0) > 0


def test_endpointing_collects_until_silence_not_fixed_length():
    # voiced, voiced, silent, silent → one utterance spanning the voiced chunks
    rp = make_loop(StubSensor(clips=[tone_wav(0.5), tone_wav(0.5), tone_wav(0.0),
                                     tone_wav(0.0), tone_wav(0.0)]))
    rp.endpoint_silence_s = 1.0  # one silent chunk ends the utterance
    events = drive(rp, 1)
    mic = [e for e in events if e.modality == "mic"]
    assert len(mic) == 1
    # utterance = trigger + follow-on voiced + the closing silence probe ⇒ ≥ 2 s
    assert mic[0].percept.data["duration_s"] >= 2.0


def test_wake_word_gets_maximum_salience_and_bypasses_rate_limit():
    calls: List[str] = []
    rp = make_loop(StubSensor(), escalate=lambda s, m: calls.append(s) or True)
    rp.min_escalation_interval_s = 3600.0     # everything else would be rate-limited
    rp._last_escalation_at = time.time()      # limiter engaged right now
    ev = rp._score("wake_word", "mic",
                   Percept.from_text("nyxara are you there", source="live:mic"))
    ev.kind = "wake_word"
    ev.salience = 1.0
    ev.transcript = "Nyxara, are you there?"
    rp._flush_escalations([ev])
    assert calls and "name" in calls[0]
    assert rp._has_wake_word("hey NYXARA wake up")
    assert rp._has_wake_word("nixara please")      # fuzzy: close mishearing still lands
    assert not rp._has_wake_word("nothing relevant here")


# --------------------------------------------------------------------------- #
# Sight: change, dedup, motion
# --------------------------------------------------------------------------- #
def test_identical_frames_produce_no_change_event():
    png = checker_png(0)
    rp = make_loop(StubSensor(frames=[png, png, png]))
    events = drive(rp, 3)
    assert [e for e in events if e.kind in ("visual_change", "motion")] == []


def test_changed_frame_is_seen():
    rp = make_loop(StubSensor(frames=[checker_png(0), checker_png(0), checker_png(200)]))
    events = drive(rp, 3)
    changes = [e for e in events if e.kind == "visual_change"]
    assert len(changes) == 1
    assert changes[0].percept.data["change_bits"] > rp.change_hamming


def test_camera_regional_change_is_motion_with_region_map():
    rp = make_loop(StubSensor(cam_frames=[checker_png(0), half_moved_png(0)]))
    rp.motion_regions = 2
    events = drive(rp, 2)
    motion = [e for e in events if e.kind == "motion"]
    assert len(motion) == 1
    regions = motion[0].percept.data["motion_regions"]
    assert regions and any("bottom" in r for r in regions)


# --------------------------------------------------------------------------- #
# Attention dynamics: habituation + orienting burst
# --------------------------------------------------------------------------- #
def test_quiet_habituates_and_events_trigger_orienting_burst():
    rp = RealtimePerception(DeadSensor(), settings=object(), journal_path=None)
    ch = ModalityChannel(kind="screen", name="screen", interval=rp.interval_s,
                         base_interval=rp.interval_s)
    now = time.time()
    # sustained quiet: interval grows toward the ceiling
    for _ in range(10):
        rp._reschedule(ch, had_event=False, now=now)
    assert ch.interval == rp.max_interval_s
    # an event: orienting reflex snaps to the fast burst cadence
    rp._reschedule(ch, had_event=True, now=now)
    assert ch.interval == rp.burst_interval_s and ch.burst_until > now
    # after the burst expires, quiet resumes habituation from the base
    rp._reschedule(ch, had_event=False, now=ch.burst_until + 1.0)
    assert ch.interval >= rp.interval_s


# --------------------------------------------------------------------------- #
# Escalation discipline
# --------------------------------------------------------------------------- #
def test_salient_event_escalates_once_and_is_rate_limited():
    calls: List[Tuple[str, list]] = []
    rp = make_loop(StubSensor(frames=[checker_png(0), checker_png(0), checker_png(200),
                                      checker_png(200), checker_png(30)]),
                   escalate=lambda s, m: calls.append((s, m)) or True)
    rp.min_escalation_interval_s = 3600.0     # only the first one may pass
    drive(rp, 5)
    assert len(calls) == 1
    stimulus, media = calls[0]
    assert "change" in stimulus and isinstance(media[0], Percept)


def test_gate_false_blocks_all_escalation_fail_closed():
    calls: List[str] = []
    rp = make_loop(StubSensor(frames=[checker_png(0), checker_png(0), checker_png(200)]),
                   escalate=lambda s, m: calls.append(s) or True,
                   gate=lambda: False)
    events = drive(rp, 3)
    assert any(e.kind == "visual_change" for e in events)  # she still SAW it
    assert calls == [] and rp.escalations == 0             # but took no action


def test_engaged_core_defers_and_retries():
    state = {"engaged": True, "runs": 0}

    def escalate(s, m):
        if state["engaged"]:
            return None            # the Master's turn is running
        state["runs"] += 1
        return object()

    rp = make_loop(StubSensor(frames=[checker_png(0), checker_png(0), checker_png(200)]),
                   escalate=escalate)
    drive(rp, 3)
    assert rp.deferred_escalations == 1 and state["runs"] == 0
    state["engaged"] = False       # turn finished; next tick flushes the queue
    drive(rp, 1)
    assert state["runs"] == 1 and rp.pending_count() == 0


# --------------------------------------------------------------------------- #
# Honest degradation + lifecycle
# --------------------------------------------------------------------------- #
def test_headless_box_runs_quietly_with_zero_events_and_zero_errors():
    rp = make_loop(DeadSensor())
    rp.idle_interval_s = 30.0
    events = drive(rp, 5)
    assert events == [] and rp.errors == 0
    assert all(ch.interval == 30.0 for ch in rp.channels.values())
    st = rp.status()
    assert st["available"] == {"camera": False, "screen": False, "mic": False}
    assert st["events"] == 0


def test_journal_records_escalated_events(tmp_path):
    path = tmp_path / "perception.jsonl"
    rp = make_loop(StubSensor(frames=[checker_png(0), checker_png(0), checker_png(200)]),
                   escalate=lambda s, m: True)
    rp._journal_path = path
    rp.journal_enabled = True
    drive(rp, 3)
    assert path.exists() and '"visual_change"' in path.read_text(encoding="utf-8")


def test_thread_lifecycle_start_is_idempotent_and_stop_joins():
    rp = RealtimePerception(DeadSensor(), settings=object(), journal_path=None)
    assert rp.start() and rp.running
    assert rp.start()              # idempotent
    rp.stop(timeout=3.0)
    assert not rp.running


def test_status_reports_measured_reality():
    rp = make_loop(StubSensor(clips=[tone_wav(0.5), tone_wav(0.0)]),
                   escalate=lambda s, m: True)
    drive(rp, 1)
    st = rp.status()
    assert st["mic_mode"] in ("chunked", "streaming")
    assert st["events"] >= 1 and st["event_counts"]
    assert st["recent_events"] and st["escalations"] >= 1
    assert "mic" in st["channels"]
