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


# --------------------------------------------------------------------------- #
# Extended senses — interoception, files, network, devices (deterministic stubs)
# --------------------------------------------------------------------------- #
from types import SimpleNamespace


def host_settings(**over) -> SimpleNamespace:
    """A real-config-shaped namespace with host senses ON (tests own the sensors)."""
    base = dict(system=True, fswatch=True, network=True, devices=True,
                window_watch=False, sound_events=True, faces=True, bodies=True,
                watch_paths=[], reachability=False)
    base.update(over)
    return SimpleNamespace(**base)


class VitalsStub:
    def __init__(self, samples):
        self.samples = list(samples)
        self.fresh: List[List[str]] = []

    def sample(self):
        return dict(self.samples.pop(0)) if self.samples \
            else {"cpu": 0.1, "mem": 0.3, "load1": 0.2, "note": ""}

    def new_processes(self):
        return self.fresh.pop(0) if self.fresh else []


class NetStub:
    def __init__(self, runs):
        self.runs = list(runs)
        self._last_state = {"online": True}

    def transitions(self):
        return self.runs.pop(0) if self.runs else []


class WatchStub:
    roots = ["stub"]
    truncated = False

    def __init__(self, diffs=()):
        self.diffs = list(diffs)

    def diff(self):
        return self.diffs.pop(0) if self.diffs else ([], [], [])


class DeviceStub:
    def __init__(self, diffs):
        self.diffs = list(diffs)
        self._last = {}

    def diff(self):
        return self.diffs.pop(0) if self.diffs else ({}, {})


def make_host_loop(**kw) -> RealtimePerception:
    rp = RealtimePerception(DeadSensor(), settings=host_settings(), journal_path=None,
                            **kw)
    rp.interval_s = rp.burst_interval_s = rp.idle_interval_s = 0.0
    rp.min_escalation_interval_s = 0.0
    rp.system_sensor = VitalsStub([])
    rp.network_sensor = NetStub([])
    rp.device_sensor = DeviceStub([])
    rp.watcher = WatchStub()             # never scan the real repo in a unit test
    for ch in rp.channels.values():
        ch.interval = ch.base_interval = 0.0
        ch.next_due = 0.0
    return rp


def test_host_channels_exist_only_when_configured():
    rp = make_host_loop()
    kinds = {c.kind for c in rp.channels.values()}
    assert {"system", "files", "network", "devices"} <= kinds
    # bare-object settings (hermetic default) build no host channels
    bare = RealtimePerception(DeadSensor(), settings=object(), journal_path=None)
    assert {c.kind for c in bare.channels.values()} == {"camera", "screen", "mic"}
    # and a real-shaped config with everything off builds none either
    off = RealtimePerception(DeadSensor(), journal_path=None,
                             settings=host_settings(system=False, fswatch=False,
                                                    network=False, devices=False))
    assert {c.kind for c in off.channels.values()} == {"camera", "screen", "mic"}


def test_cpu_spike_raises_one_alert_then_stays_quiet_until_rearmed():
    got: List[str] = []
    rp = make_host_loop(escalate=lambda s, m: got.append(s) or True)
    rp.system_sensor = VitalsStub([
        {"cpu": 0.20, "mem": 0.3, "load1": 0.5, "note": ""},
        {"cpu": 0.97, "mem": 0.3, "load1": 4.0, "note": "", "top_process": "stress"},
        {"cpu": 0.96, "mem": 0.3, "load1": 4.0, "note": ""},   # still hot: no re-alert
        {"cpu": 0.10, "mem": 0.3, "load1": 0.5, "note": ""},   # cooled: re-armed
        {"cpu": 0.95, "mem": 0.3, "load1": 4.0, "note": ""},   # fires again
    ])
    events = drive(rp, 5)
    alerts = [e for e in events if e.kind == "system_alert"]
    assert len(alerts) == 2
    assert alerts[0].salience >= 0.85
    assert "97%" in alerts[0].percept.content and "stress" in alerts[0].percept.content
    assert any("CPU" in s for s in got)


def test_vitals_keepalive_feeds_the_mind_but_never_escalates():
    got: List[str] = []
    rp = make_host_loop(escalate=lambda s, m: got.append(s) or True)
    rp.scene_interval_s = 0.0            # emit the keep-alive on every tick
    events = drive(rp, 2)
    vitals = [e for e in events if e.kind == "vitals"]
    assert vitals and all(v.salience <= 0.2 for v in vitals)
    assert got == []                     # calm vitals never become a cognitive cycle
    assert rp.status()["last_vitals"]    # but the measured reading is introspectable


def test_new_process_is_announced_once():
    rp = make_host_loop()
    rp.system_sensor.fresh = [[], ["suspicious-tool"]]
    events = drive(rp, 2)
    procs = [e for e in events if e.kind == "process_new"]
    assert len(procs) == 1
    assert "suspicious-tool" in procs[0].percept.content


def test_file_changes_are_perceived_from_a_real_directory(tmp_path):
    got: List[str] = []
    rp = RealtimePerception(DeadSensor(), journal_path=None,
                            settings=host_settings(watch_paths=[str(tmp_path)],
                                                   system=False, network=False,
                                                   devices=False),
                            escalate=lambda s, m: got.append(s) or True)
    rp.min_escalation_interval_s = 0.0
    for ch in rp.channels.values():
        ch.interval = ch.base_interval = 0.0
        ch.next_due = 0.0
    drive(rp, 1)                          # builds the watcher, primes the snapshot
    (tmp_path / "report.txt").write_text("hello")
    events = drive(rp, 1)
    files = [e for e in events if e.kind == "file_change"]
    assert len(files) == 1
    assert "report.txt" in files[0].percept.content
    assert any("report.txt" in s for s in got)


def test_network_transition_becomes_a_first_person_stimulus():
    got: List[str] = []
    rp = make_host_loop(escalate=lambda s, m: got.append(s) or True)
    rp.network_sensor = NetStub([
        [], [("network_change", "The network connection just went down.", 0.8)]])
    events = drive(rp, 2)
    net = [e for e in events if e.kind == "network_change"]
    assert len(net) == 1 and net[0].salience >= 0.8
    assert any("network connection just went down" in s for s in got)


def test_plugged_camera_becomes_a_live_channel_and_unplug_drops_it():
    rp = make_host_loop()
    assert "camera1" not in rp.channels
    rp.device_sensor = DeviceStub([
        ({"/dev/video1": "camera (/dev/video1)"}, {}),
        ({}, {"/dev/video1": "camera (/dev/video1)"}),
    ])
    events = drive(rp, 1)
    assert [e.kind for e in events if e.modality == "devices"] == ["device_plugged"]
    assert "camera1" in rp.channels      # she gained an eye without a restart
    events = drive(rp, 1)
    assert [e.kind for e in events if e.modality == "devices"] == ["device_removed"]
    assert "camera1" not in rp.channels


# --------------------------------------------------------------------------- #
# Vision extras — faces, lights, motion direction, window tracking
# --------------------------------------------------------------------------- #
def flat_png(v: int, size: int = 64) -> bytes:
    from nyxara.senses.generate import encode_png
    row = bytes([v, v, v] * size)
    return encode_png(size, size, [row] * size)


def test_face_appearance_is_a_high_salience_event():
    rp = make_loop(StubSensor(cam_frames=[checker_png(0), checker_png(0),
                                          checker_png(200), checker_png(0)]))
    scripts = [[(10, 10, 40, 40)], []]   # one face, then nobody
    rp.detect = lambda data, cascade: scripts.pop(0) if scripts else []
    events = drive(rp, 4)
    faces = [e for e in events if e.kind == "face"]
    assert len(faces) == 2
    assert faces[0].salience >= 0.9 and "appeared" in faces[0].percept.content
    assert "dropped" in faces[1].percept.content


def test_absent_detector_never_reports_absent_people():
    rp = make_loop(StubSensor(cam_frames=[checker_png(0), checker_png(0),
                                          checker_png(200)]))
    rp.detect = lambda data, cascade: None      # cv2 not installed
    events = drive(rp, 3)
    assert [e for e in events if e.kind in ("face", "person")] == []


def test_body_without_face_is_a_person_event():
    rp = make_loop(StubSensor(cam_frames=[checker_png(0), checker_png(0),
                                          checker_png(200)]))
    from nyxara.senses.realtime import _BODY_CASCADE
    rp.detect = lambda data, cascade: ([(5, 5, 30, 60)] if cascade == _BODY_CASCADE
                                       else [])
    events = drive(rp, 3)
    people = [e for e in events if e.kind == "person"]
    assert len(people) == 1 and people[0].salience >= 0.8


def test_brightness_step_is_an_explicit_lights_event():
    rp = make_loop(StubSensor(frames=[flat_png(20), flat_png(20), flat_png(220)]))
    events = drive(rp, 3)
    lights = [e for e in events if e.kind == "lights"]
    assert len(lights) == 1
    assert "on" in lights[0].percept.content and lights[0].salience >= 0.6


def test_window_change_carries_attention_word_salience():
    rp = make_loop(StubSensor(frames=[flat_png(20)] * 4))
    rp.window_watch = True
    rp.attention_words = ["error"]
    titles = ["editor", "editor", "build ERROR — terminal", None]
    rp.window_title = lambda: titles.pop(0) if titles else None
    events = drive(rp, 3)
    wins = [e for e in events if e.kind == "window_change"]
    assert len(wins) == 1                # first sighting primes, only the change fires
    assert wins[0].salience >= 0.8       # "error" in the title is worth waking up for


# --------------------------------------------------------------------------- #
# Hearing extras — alarm classification bypasses the escalation rate limit
# --------------------------------------------------------------------------- #
def beeps_wav(secs: float = 3.0) -> bytes:
    half = RATE // 4
    sig = [0.6 * math.sin(2 * math.pi * 1000 * i / RATE)
           if (i // half) % 2 == 0 else 0.0 for i in range(int(secs * RATE))]
    return _floats_to_wav(sig, rate=RATE)


def test_periodic_beeping_is_an_urgent_alarm():
    got: List[str] = []
    rp = make_loop(StubSensor(clips=[beeps_wav(), tone_wav(0.0)]),
                   escalate=lambda s, m: got.append(s) or True)
    rp.sound_spike_snr = 1.0
    rp.min_escalation_interval_s = 3600.0        # an ordinary event would wait…
    rp._last_escalation_at = time.time()
    events = drive(rp, 1)
    alarms = [e for e in events if e.kind == "alarm"]
    assert len(alarms) == 1 and alarms[0].salience >= 0.9
    assert got and "alarm-like beeping" in got[0]   # …but an alarm never waits


# --------------------------------------------------------------------------- #
# Self-health — a sense that keeps failing becomes its own percept
# --------------------------------------------------------------------------- #
def test_repeated_sense_failures_raise_one_degraded_event():
    rp = make_host_loop()
    rp.system_sensor = object()          # no .sample(): every tick raises inside
    events = drive(rp, 5)
    degraded = [e for e in events if e.kind == "sense_degraded"]
    assert len(degraded) == 1            # rate-limited: one cry, not a siren
    assert "system" in degraded[0].percept.content
    assert rp.channels["system"].error_streak >= 3
