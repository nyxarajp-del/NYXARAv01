"""NYXARA · senses/realtime.py — continuous real-time perception (👁👂🫀, always on).

Every other sense answers only when asked: :mod:`senses.live` grabs one frame or one clip
per call, the file senses decode stored media, and nothing perceives *between* prompts.
This module closes that gap — it is the loop that keeps NYXARA's eyes and ears open. A
background thread continuously captures camera frames, screen frames and microphone audio
through the existing :class:`~nyxara.senses.live.LiveSensor`, detects salient moments
**natively** — energy voice-activity detection with a self-calibrating noise floor,
full-utterance endpointing, wake-word spotting, sound-event classification (bang /
alarm-beep / sustained hum), perceptual-hash visual change, region-grid motion with
direction tracking, face and body detection (Haar cascades when OpenCV is present),
lights-on/off, active-window tracking, OCR screen reading, and predictive surprise —
and escalates what matters into a real autonomous cognitive cycle, so she reacts to the
world unprompted.

Beyond eyes and ears, the same loop carries her **interoception and proprioception**
(:mod:`senses.system`, :mod:`senses.watch`): host vitals (CPU, memory, disk, heat,
battery) with rising-edge alerts, genuinely new processes, filesystem change watching,
network transitions (offline/online, new listening ports, Wi-Fi), and device hot-plug —
a camera plugged in mid-flight becomes a live channel without a restart. The loop also
watches *itself*: a sense that keeps failing raises its own ``sense_degraded`` percept.

The LLM is **never** in this path: every detector below is signal processing and
statistics over the existing pure-python senses. The model only ever sees the *result*,
delivered as an ordinary stimulus through the sovereign ``process()`` gates.

Design, matching the rest of NYXARA:

* **Real, never fake.** Every percept originates in genuine device I/O; a headless box
  produces *no* events — the loop keeps running at a slow probe cadence, honestly
  reporting "no device", and never fabricates a frame, a sample or a transcript.
* **Attention, not firehose.** Channels habituate (quiet doubles the capture interval),
  orient (an event snaps the channel to a fast burst cadence), deduplicate (the private
  :class:`~nyxara.senses.binding.Binder` collapses near-identical percepts) and are
  surprise-gated by a :class:`~nyxara.senses.predictive.PerceptualPredictor` — the
  unexpected rises to the mind, the routine stays in the senses.
* **Stdlib-first, degrade honestly, never raise.** Streaming microphone (``sounddevice``)
  is optional; without it, chunked polling through the sensor works everywhere. Whisper,
  OCR and every heavy backend are import-guarded; their absence is a note, never a crash.
* **Policy-aware.** Per-modality opt-outs are enforced twice — here and inside
  ``LiveSensor.enabled`` — and every escalation passes the oversight gate *and* the full
  sovereign cycle. A paused mind sees detections recorded but takes no action.
"""

from __future__ import annotations

import array
import difflib
import glob
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from nyxara.senses.audio import (
    Audio,
    classify_sound,
    decode_pcm,
    rms,
)
from nyxara.senses.binding import Binder, Percept
from nyxara.senses.live import LiveSensor, pcm16_to_wav_bytes
from nyxara.senses.predictive import PerceptualPredictor, SensoryPredictor
from nyxara.senses.vision import (
    Vision,
    average_hash,
    decode_image,
    gray_from_pixels,
    hamming,
    resize_gray,
)

__all__ = ["PerceptEvent", "ModalityChannel", "RealtimePerception", "detect_rects"]

# Channels that sense the host itself — always "available": the box is always there.
_HOST_KINDS = frozenset({"system", "files", "network", "devices"})
# Event kinds that must never wait out the escalation rate limit.
_URGENT_KINDS = frozenset({"wake_word", "alarm", "face"})
# Event kinds whose percept content is already a first-person sentence.
_SELF_WORDED_KINDS = frozenset({
    "system_alert", "process_new", "file_change", "network_change", "port_change",
    "device_plugged", "device_removed", "lights", "window_change", "person",
    "sense_degraded",
})
# Region labels for the 4x4 motion grid (row-major), so a motion event can say WHERE.
_REGION_NAMES = [
    "top-left", "top", "top", "top-right",
    "left", "center", "center", "right",
    "left", "center", "center", "right",
    "bottom-left", "bottom", "bottom", "bottom-right",
]
_GRID = 4  # motion grid is _GRID x _GRID regions
# How many pending (deferred) escalations we keep while the Master's turn runs.
_MAX_PENDING = 8
# Journal rotation: past this many lines, rewrite keeping the newest half.
_JOURNAL_MAX_LINES = 2000


def _floats_to_wav(sig: Sequence[float], *, rate: int) -> bytes:
    """Mono float signal in [-1, 1] → int16 WAV bytes (for re-analysis/transcription)."""
    ints = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in sig))
    if sys.byteorder != "little":
        ints.byteswap()
    return pcm16_to_wav_bytes(ints.tobytes(), rate=rate, channels=1)


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# --------------------------------------------------------------------------- #
# Haar-cascade detection (OpenCV, optional) — real faces/bodies, never a guess
# --------------------------------------------------------------------------- #
_FACE_CASCADE = "haarcascade_frontalface_default.xml"
_BODY_CASCADE = "haarcascade_upperbody.xml"
_CASCADE_CACHE: Dict[str, Any] = {}


def detect_rects(data: bytes, cascade: str) -> Optional[List[Tuple[int, int, int, int]]]:
    """Detect faces/bodies in raw image bytes with an OpenCV Haar cascade.

    Returns ``(x, y, w, h)`` boxes, ``[]`` when the frame genuinely contains none, and
    ``None`` when detection is *unavailable* (no cv2/numpy, undecodable frame) — absence
    of the detector is never reported as absence of people."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    try:
        clf = _CASCADE_CACHE.get(cascade)
        if clf is None:
            path = Path(cv2.data.haarcascades) / cascade
            if not path.exists():
                return None
            clf = cv2.CascadeClassifier(str(path))
            if clf.empty():
                return None
            _CASCADE_CACHE[cascade] = clf
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        rects = clf.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(30, 30))
        return [tuple(int(v) for v in r) for r in rects]
    except Exception:  # noqa: BLE001 — a broken detector is an absent detector
        return None


def _active_window_title() -> Optional[str]:
    """The focused window's title via ``xdotool`` (X11 only; honest ``None`` otherwise)."""
    import subprocess
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    try:
        res = subprocess.run(["xdotool", "getactivewindow", "getwindowname"],
                             capture_output=True, text=True, timeout=1.0)
        title = res.stdout.strip()
        return title or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
@dataclass
class PerceptEvent:
    """One salient moment the loop noticed — what, where, how sure, and the percept."""

    kind: str                      # speech | wake_word | sound | alarm | visual_change
    #                              # | motion | screen_text | face | person | lights
    #                              # | window_change | system_alert | process_new | vitals
    #                              # | file_change | network_change | port_change
    #                              # | device_plugged | device_removed | sense_degraded
    #                              # | surprise | anomaly | scene
    modality: str                  # camera | screen | mic | system | files | network | devices
    percept: Percept
    salience: float = 0.0
    surprise: float = 0.0
    transcript: Optional[str] = None   # real Whisper text, or honestly None
    text: Optional[str] = None         # OCR'd screen text, or honestly None
    note: str = ""                     # provenance / honest degradation note
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "modality": self.modality,
                "percept": self.percept.to_dict(), "salience": round(self.salience, 3),
                "surprise": round(self.surprise, 4), "transcript": self.transcript,
                "text": (self.text or "")[:200] or None, "note": self.note,
                "at": round(self.at, 3)}


# --------------------------------------------------------------------------- #
# Channel — one modality's cadence + change memory
# --------------------------------------------------------------------------- #
@dataclass
class ModalityChannel:
    """Per-channel adaptive cadence (habituation / orienting) and last-seen state."""

    kind: str                       # "camera" | "screen" | "mic" | "system" | "files"
    #                               # | "network" | "devices"
    name: str                       # "camera0", "screen", "mic", "system", ...
    index: int = 0                  # camera device index
    interval: float = 2.0           # current cadence (seconds)
    base_interval: float = 2.0
    next_due: float = 0.0
    burst_until: float = 0.0
    quiet_streak: int = 0
    ticks: int = 0
    captures: int = 0
    misses: int = 0
    last_note: str = ""
    last_hash: Optional[int] = None
    last_change_bits: Optional[int] = None
    last_regions: Optional[List[int]] = None
    last_ocr_text: str = ""
    last_scene_at: float = 0.0
    # vision extras — faces/bodies, lights, motion direction, active window
    last_faces: Optional[int] = None
    last_face_cx: Optional[float] = None
    last_bodies: int = 0
    last_brightness: Optional[float] = None
    last_motion_cx: Optional[float] = None
    last_motion_cy: Optional[float] = None
    last_window: str = ""
    # self-health — a sense that keeps failing becomes its own percept
    error_streak: int = 0
    last_degraded_at: float = 0.0

    def status(self) -> Dict[str, Any]:
        return {"interval_s": round(self.interval, 2), "burst": time.time() < self.burst_until,
                "ticks": self.ticks, "captures": self.captures, "misses": self.misses,
                "last_note": self.last_note, "last_change_bits": self.last_change_bits}


# --------------------------------------------------------------------------- #
# Optional streaming microphone (sounddevice) — continuous listening with pre-roll
# --------------------------------------------------------------------------- #
class _MicStream:
    """A persistent input stream feeding a bounded ring buffer of int16 blocks.

    Real continuous listening: audio flows even between ticks, so an utterance can be
    reconstructed including the moments *before* the voice trigger (pre-roll — no clipped
    word-starts). Entirely optional; construction fails honestly to ``None`` without the
    backend or a device.
    """

    BLOCK_S = 0.1

    def __init__(self, rate: int) -> None:
        import sounddevice as sd  # type: ignore  # ImportError → caller degrades

        self.rate = int(rate)
        self._lock = threading.Lock()
        blocks = int(30.0 / self.BLOCK_S)             # ~30 s of audio
        self._ring: Deque[bytes] = deque(maxlen=blocks)
        self._stream = sd.InputStream(
            samplerate=self.rate, channels=1, dtype="int16",
            blocksize=int(self.rate * self.BLOCK_S), callback=self._on_block)
        self._stream.start()

    def _on_block(self, indata, frames, t, status) -> None:  # noqa: ANN001 — sd callback
        try:
            with self._lock:
                self._ring.append(bytes(indata.tobytes()))
        except Exception:  # noqa: BLE001 — a callback must never raise
            pass

    def read_last(self, seconds: float) -> bytes:
        """The most recent ``seconds`` of raw int16 PCM (may be shorter while filling)."""
        n = max(1, int(seconds / self.BLOCK_S))
        with self._lock:
            return b"".join(list(self._ring)[-n:])

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class RealtimePerception:
    """NYXARA's always-on senses: capture → detect → bind → surprise → escalate."""

    def __init__(self, sensor: Optional[LiveSensor] = None, *,
                 escalate: Optional[Callable[[str, List[Percept]], Any]] = None,
                 presence: Any = None, mind: Any = None,
                 gate: Optional[Callable[[], bool]] = None,
                 remember: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                 settings: Any = None,
                 journal_path: Optional[Path] = None) -> None:
        cfg = settings if settings is not None else self._default_settings()
        self.sensor = sensor or LiveSensor(
            mic_seconds=float(getattr(cfg, "mic_chunk_s", 1.0)),
            enabled={"camera": bool(getattr(cfg, "camera", True)),
                     "screen": bool(getattr(cfg, "screen", True)),
                     "mic": bool(getattr(cfg, "mic", True))})
        self.escalate = escalate
        self.presence = presence
        self.mind = mind
        self.gate = gate
        self.remember = remember

        # ---- config snapshot (plain floats/ints so tests can tweak freely) ---- #
        g = lambda name, dflt: getattr(cfg, name, dflt)  # noqa: E731
        self.interval_s = float(g("interval_s", 2.0))
        self.max_interval_s = float(g("max_interval_s", 30.0))
        self.idle_interval_s = float(g("idle_interval_s", 30.0))
        self.burst_interval_s = float(g("burst_interval_s", 0.5))
        self.burst_duration_s = float(g("burst_duration_s", 10.0))
        self.mic_chunk_s = float(g("mic_chunk_s", 1.0))
        self.max_utterance_s = float(g("max_utterance_s", 15.0))
        self.endpoint_silence_s = float(g("endpoint_silence_s", 0.8))
        self.vad_rms = float(g("vad_rms", 0.02))
        self.vad_snr = float(g("vad_snr", 3.0))
        self.sound_spike_snr = float(g("sound_spike_snr", 6.0))
        self.wake_words = [str(w).casefold() for w in (g("wake_words", None) or ["nyxara"])]
        self.transcribe = bool(g("transcribe", True))
        self.whisper_model = str(g("whisper_model", "base"))
        self.change_hamming = int(g("change_hamming", 10))
        self.motion_regions = int(g("motion_regions", 3))
        self.scene_interval_s = float(g("scene_interval_s", 60.0))
        self.ocr = bool(g("ocr", True))
        self.max_cameras = int(g("max_cameras", 2))
        self.comodal_window_s = float(g("comodal_window_s", 2.0))
        self.surprise_threshold = float(g("surprise_threshold", 0.6))
        self.min_escalation_interval_s = float(g("min_escalation_interval_s", 20.0))
        self.max_events = int(g("max_events", 256))
        self.journal_enabled = bool(g("journal", True))

        # ---- extended senses. NOTE: host senses and the window watcher fall back to
        # OFF when no real config is passed (bare-object settings ⇒ hermetic tests and
        # deterministic self-tests); PerceptionConfig turns them ON by default. ---- #
        self.sound_events = bool(g("sound_events", True))
        self.faces = bool(g("faces", True))
        self.bodies = bool(g("bodies", True))
        self.lights_delta = float(g("lights_delta", 60.0))
        self.window_watch = bool(g("window_watch", False))
        self.attention_words = [str(w).casefold()
                                for w in (g("attention_words", None) or [])]
        self.system_enabled = bool(g("system", False))
        self.system_interval_s = float(g("system_interval_s", 10.0))
        self.cpu_alert = float(g("cpu_alert", 0.90))
        self.mem_alert = float(g("mem_alert", 0.90))
        self.disk_alert = float(g("disk_alert", 0.95))
        self.battery_alert = float(g("battery_alert", 0.15))
        self.temp_alert_c = float(g("temp_alert_c", 85.0))
        self.process_watch = bool(g("process_watch", True))
        self.fswatch_enabled = bool(g("fswatch", False))
        self.fswatch_interval_s = float(g("fswatch_interval_s", 5.0))
        self.watch_paths = [str(p) for p in (g("watch_paths", None) or [])]
        self.fswatch_depth = int(g("fswatch_depth", 3))
        self.fswatch_max_entries = int(g("fswatch_max_entries", 2000))
        self.network_enabled = bool(g("network", False))
        self.network_interval_s = float(g("network_interval_s", 15.0))
        self.reachability = bool(g("reachability", True))
        self.port_watch = bool(g("port_watch", True))
        self.devices_enabled = bool(g("devices", False))
        self.devices_interval_s = float(g("devices_interval_s", 10.0))

        # ---- perception state (loop-private; the core binds escalated percepts itself) #
        self.binder = Binder(window=120.0)
        self.predictor = PerceptualPredictor(warmup=4)
        self._stream_predictors: Dict[str, SensoryPredictor] = {}
        self.vision = Vision()
        self.audio = Audio(whisper_model=self.whisper_model)
        self.events: Deque[PerceptEvent] = deque(maxlen=self.max_events)
        self.event_counts: Dict[str, int] = {}
        self.escalations = 0
        self.deferred_escalations = 0
        self.errors = 0
        self.stage_errors: Dict[str, int] = {}
        self.last_transcript: Optional[str] = None
        self._noise_floor: Deque[float] = deque(maxlen=64)
        self._pending: List[Tuple[str, List[Percept], float]] = []
        self._last_escalation_at = 0.0
        self._journal_path = journal_path if journal_path is not None else self._default_journal()
        self._journal_lines = 0
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

        # ---- extended-sense state: host sensors, file watcher, detector hooks ---- #
        # (plain attributes so tests can inject deterministic stand-ins)
        self.system_sensor: Any = None
        self.network_sensor: Any = None
        self.device_sensor: Any = None
        self.watcher: Any = None
        self.detect = detect_rects            # face/body detector hook
        self.window_title = _active_window_title
        self.last_vitals: Dict[str, Any] = {}
        self._active_alerts: set = set()      # rising-edge alert keys (cpu, mem, ...)
        if self.system_enabled or self.network_enabled or self.devices_enabled:
            try:
                from nyxara.senses.system import DeviceSensor, NetworkSensor, SystemSensor
                if self.system_enabled:
                    self.system_sensor = SystemSensor(data_dir=self._data_dir())
                if self.network_enabled:
                    self.network_sensor = NetworkSensor(probe=self.reachability,
                                                        ports=self.port_watch)
                if self.devices_enabled:
                    self.device_sensor = DeviceSensor()
            except Exception:  # noqa: BLE001 — host senses are a capability, never a dep
                self._note_error("host_sensors")

        self.channels: Dict[str, ModalityChannel] = self._build_channels()

        # streaming mic — best-effort; honest fallback to chunked polling via the sensor
        self._mic_stream: Optional[_MicStream] = None
        self.mic_mode = "chunked"
        self._mic_rate = int(getattr(self.sensor, "mic_rate", 16000))

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_settings() -> Any:
        try:
            from nyxara.kernel.config import get_settings
            return get_settings().perception
        except Exception:  # noqa: BLE001 — config unavailable ⇒ built-in defaults
            return object()

    @staticmethod
    def _default_journal() -> Optional[Path]:
        try:
            from nyxara.kernel.config import get_settings
            d = get_settings().paths.data_dir
            return Path(d) / "perception.jsonl" if d is not None else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _data_dir() -> Optional[Path]:
        try:
            from nyxara.kernel.config import get_settings
            d = get_settings().paths.data_dir
            return Path(d) if d is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _build_channels(self) -> Dict[str, ModalityChannel]:
        chans: Dict[str, ModalityChannel] = {}

        def add(kind: str, name: str, index: int = 0,
                base: Optional[float] = None) -> None:
            b = self.interval_s if base is None else base
            chans[name] = ModalityChannel(kind=kind, name=name, index=index,
                                          interval=b, base_interval=b)

        if self.sensor.enabled.get("camera", True):
            nodes = sorted(glob.glob("/dev/video*"))[: self.max_cameras]
            for i, _ in enumerate(nodes or [None]):     # at least camera0 everywhere
                add("camera", f"camera{i}", i)
        if self.sensor.enabled.get("screen", True):
            add("screen", "screen")
        if self.sensor.enabled.get("mic", True):
            add("mic", "mic")
        # host senses — interoception ticks on its own, slower cadence
        if self.system_sensor is not None:
            add("system", "system", base=self.system_interval_s)
        if self.fswatch_enabled:
            add("files", "files", base=self.fswatch_interval_s)
        if self.network_sensor is not None:
            add("network", "network", base=self.network_interval_s)
        if self.device_sensor is not None:
            add("devices", "devices", base=self.devices_interval_s)
        return chans

    def _ensure_watcher(self) -> Any:
        """Build the file watcher on first use (lazy: roots may come from config)."""
        if self.watcher is not None or not self.fswatch_enabled:
            return self.watcher
        try:
            from nyxara.senses.watch import FileWatcher
            roots = [Path(p) for p in self.watch_paths]
            ignore: List[Path] = []
            data_dir = self._data_dir()
            if not roots:
                roots = [Path.cwd()]
                if data_dir is not None:
                    roots.append(data_dir)
            if data_dir is not None:
                # her own journals/memory churn is not the world changing
                ignore = [data_dir / n for n in
                          ("perception.jsonl", "memory", "replay", "audit")]
            self.watcher = FileWatcher(roots, depth=self.fswatch_depth,
                                       max_entries=self.fswatch_max_entries,
                                       ignore_paths=ignore)
        except Exception:  # noqa: BLE001
            self._note_error("fswatch")
            self.fswatch_enabled = False
        return self.watcher

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Open the senses (idempotent). Returns True if the loop is now running."""
        if self.running:
            return True
        self._stop.clear()
        self._maybe_open_mic_stream()
        self._thread = threading.Thread(target=self._run, name="nyxara-perception",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        """Close the senses. Joins the thread and releases the mic stream."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        if self._mic_stream is not None:
            self._mic_stream.close()
            self._mic_stream = None
            self.mic_mode = "chunked"

    def _maybe_open_mic_stream(self) -> None:
        if not self.sensor.enabled.get("mic", True) or self._mic_stream is not None:
            return
        try:
            self._mic_stream = _MicStream(self._mic_rate)
            self.mic_mode = "streaming"
        except Exception:  # noqa: BLE001 — no backend/device ⇒ chunked polling
            self._mic_stream = None
            self.mic_mode = "chunked"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick_once()
            except Exception:  # noqa: BLE001 — the loop itself must never die
                self._note_error("tick")
            # sleep until the soonest channel is due (bounded, so stop() is responsive)
            now = time.time()
            due = [c.next_due for c in self.channels.values()]
            delay = max(0.05, min((min(due) - now) if due else 1.0, 1.0))
            self._stop.wait(delay)

    # ------------------------------------------------------------------ #
    # One sweep
    # ------------------------------------------------------------------ #
    def tick_once(self) -> List[PerceptEvent]:
        """One synchronous sweep over all due channels. Returns the new events."""
        now = time.time()
        avail = self._available()
        events: List[PerceptEvent] = []
        # snapshot the channel list — device hot-plug may add/drop channels mid-sweep
        for ch in list(self.channels.values()):
            if now < ch.next_due:
                continue
            ch.ticks += 1
            if not avail.get(ch.kind, ch.kind in _HOST_KINDS):
                ch.misses += 1
                ch.interval = self.idle_interval_s
                ch.next_due = now + ch.interval
                continue
            try:
                if ch.kind == "mic":
                    events.extend(self._sense_mic(ch, now))
                elif ch.kind == "system":
                    events.extend(self._sense_system(ch, now))
                elif ch.kind == "files":
                    events.extend(self._sense_files(ch, now))
                elif ch.kind == "network":
                    events.extend(self._sense_network(ch, now))
                elif ch.kind == "devices":
                    events.extend(self._sense_devices(ch, now))
                else:
                    events.extend(self._sense_visual(ch, now))
                ch.error_streak = 0
            except Exception:  # noqa: BLE001
                self._note_error(ch.name)
                ch.error_streak += 1
                events.extend(self._maybe_degraded(ch, now))
            self._reschedule(ch, had_event=any(e.modality == ch.kind for e in events),
                             now=time.time())
        for e in events:
            self._register(e)
        self._flush_escalations(events)
        self._maybe_reset_frame()
        return events

    def run_for(self, n: int) -> List[PerceptEvent]:
        out: List[PerceptEvent] = []
        for _ in range(max(1, int(n))):
            out.extend(self.tick_once())
        return out

    def _available(self) -> Dict[str, bool]:
        try:
            return self.sensor.available()
        except Exception:  # noqa: BLE001
            self._note_error("probe")
            return {}

    def _reschedule(self, ch: ModalityChannel, *, had_event: bool, now: float) -> None:
        if had_event:
            # orienting response: something happened — look/listen harder for a while
            ch.quiet_streak = 0
            ch.burst_until = now + self.burst_duration_s
            ch.interval = self.burst_interval_s
        elif now < ch.burst_until:
            ch.interval = self.burst_interval_s
        else:
            # habituation: sustained quiet slows the channel down toward the ceiling
            ch.quiet_streak += 1
            base = self._presence_scaled_base(ch)
            ch.interval = min(base * (2 ** min(ch.quiet_streak, 6)), self.max_interval_s) \
                if ch.quiet_streak > 1 else base
        ch.next_due = now + ch.interval

    def _presence_scaled_base(self, ch: ModalityChannel) -> float:
        base = ch.base_interval
        try:
            state = getattr(self.presence, "state", None)
            arousal = getattr(state, "arousal", None)
            if arousal is not None:
                if arousal <= 1:      # asleep / dreaming
                    base *= 4.0
                elif arousal == 2:    # idle
                    base *= 2.0
        except Exception:  # noqa: BLE001
            pass
        return base

    # ------------------------------------------------------------------ #
    # Hearing
    # ------------------------------------------------------------------ #
    def _read_mic_chunk(self, seconds: float) -> Tuple[List[float], str]:
        """One chunk of live audio as a mono float signal (streaming ring or sensor)."""
        if self._mic_stream is not None:
            pcm = self._mic_stream.read_last(seconds)
            if pcm:
                sig, _ = decode_pcm(pcm16_to_wav_bytes(pcm, rate=self._mic_rate))
                return sig, "mic: streaming"
            return [], "mic: stream filling"
        data, note = self.sensor.capture_microphone(seconds=seconds)
        if not data:
            return [], note
        sig, _ = decode_pcm(data)
        return sig, note

    def _vad_threshold(self) -> float:
        floor = _median(list(self._noise_floor))
        return max(self.vad_rms, floor * self.vad_snr)

    def _sense_mic(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        sig, note = self._read_mic_chunk(self.mic_chunk_s)
        ch.last_note = note
        if not sig:
            ch.misses += 1
            return []
        ch.captures += 1
        level = rms(sig)
        threshold = self._vad_threshold()
        self._observe_stream(ch.name, [level])

        if level < threshold:
            self._noise_floor.append(level)      # quiet chunk teaches the noise floor
            return []

        # Voice/sound onset → collect the whole utterance (endpointing). Streaming mode
        # re-reads the trigger WITH 0.5 s of pre-roll from the ring (no clipped word
        # starts) and must let real time pass between reads so each chunk is new audio;
        # chunked mode blocks inside the sensor, so the trigger chunk itself is the start.
        if self._mic_stream is not None:
            pre, _ = self._read_mic_chunk(self.mic_chunk_s + 0.5)
            utter = list(pre or sig)
        else:
            utter = list(sig)
        voiced_s = self.mic_chunk_s
        silent_s = 0.0
        while voiced_s + silent_s < self.max_utterance_s:
            if self._mic_stream is not None and self._stop.wait(self.mic_chunk_s):
                break               # shutting down mid-utterance — keep what we heard
            nxt, _ = self._read_mic_chunk(self.mic_chunk_s)
            if not nxt:
                break
            utter.extend(nxt)
            if rms(nxt) >= threshold:
                voiced_s += self.mic_chunk_s
                silent_s = 0.0
            else:
                silent_s += self.mic_chunk_s
                if silent_s >= self.endpoint_silence_s:
                    break

        wav = _floats_to_wav(utter, rate=self._mic_rate)
        analysis = self.audio.analyze_bytes(wav, transcribe=self.transcribe)
        if analysis.transcript:
            self.last_transcript = analysis.transcript

        spike = level >= max(self.vad_rms, _median(list(self._noise_floor) or [self.vad_rms])
                             * self.sound_spike_snr)
        kind = "speech" if voiced_s >= 0.35 else ("sound" if spike else "speech")
        if analysis.transcript and self._has_wake_word(analysis.transcript):
            kind = "wake_word"

        # non-speech sounds get a real class from their envelope + spectrum: a bang,
        # a periodic alarm-beep, or a sustained hum — pure statistics, never a guess.
        # An untranscribed "speech" trigger is also checked: real alarms are continuous
        # energy (they pass VAD) but their envelope repeats like no voice does.
        profile = None
        if self.sound_events and (kind == "sound"
                                  or (kind == "speech" and not analysis.transcript)):
            profile = classify_sound(utter, self._mic_rate)
            if profile.label == "alarm":
                kind = "alarm"
            elif kind == "speech":
                profile = None      # a voice stays a voice; classes annotate only sounds

        p = Percept.from_audio(analysis, source="live:mic", at=now)
        p.tags.append(kind)
        p.data.update({"rms": round(level, 5), "threshold": round(threshold, 5),
                       "duration_s": round(len(utter) / max(1, self._mic_rate), 2)})
        if profile is not None:
            p.tags.append(profile.label)
            p.data["sound_class"] = profile.to_dict()
        ev = self._score(kind, "mic", p, note=analysis.note or note,
                         transcript=analysis.transcript)
        if kind == "wake_word":
            ev.salience = 1.0
            p.salience = 1.0
        elif kind == "alarm":
            ev.salience = max(ev.salience, 0.9)     # alarms are urgent by nature
        return [ev]

    def _has_wake_word(self, transcript: str) -> bool:
        toks = [t.strip(".,!?;:'\"").casefold() for t in transcript.split()]
        for w in self.wake_words:
            for t in toks:
                if t and difflib.SequenceMatcher(None, t, w).ratio() >= 0.75:
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Sight
    # ------------------------------------------------------------------ #
    def _sense_visual(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        if ch.kind == "camera":
            data, note = self.sensor.capture_camera(index=ch.index)
        else:
            data, note = self.sensor.capture_screen()
        ch.last_note = note
        if not data:
            ch.misses += 1
            return []
        ch.captures += 1

        analysis = self.vision.analyze_bytes(data)
        h = analysis.average_hash if analysis.average_hash is not None \
            else analysis.perceptual_hash
        gray = self._gray(data)
        brightness = (sum(sum(r) for r in gray) / max(1, sum(len(r) for r in gray))
                      if gray else 0.0)
        self._observe_stream(ch.name, [brightness / 255.0])

        events: List[PerceptEvent] = []
        bits: Optional[int] = None
        if h is not None and ch.last_hash is not None:
            bits = hamming(h, ch.last_hash)
            ch.last_change_bits = bits
        changed = bits is not None and bits > self.change_hamming

        # lights on/off: a step in mean brightness is its own explicit event, not
        # just an anonymous hash change
        if gray and ch.last_brightness is not None \
                and abs(brightness - ch.last_brightness) >= self.lights_delta:
            went_on = brightness > ch.last_brightness
            where = "the room" if ch.kind == "camera" else "the screen"
            p = Percept.from_text(
                f"The lights just went {'on' if went_on else 'off'} in {where} "
                f"(brightness {ch.last_brightness:.0f} → {brightness:.0f}).",
                source=f"live:{ch.name}", tags=["lights"], at=now)
            p.data.update({"brightness": round(brightness, 1),
                           "previous": round(ch.last_brightness, 1)})
            ev = self._score("lights", ch.kind, p, note=note)
            ev.salience = max(ev.salience, 0.6)
            events.append(ev)
        if gray:
            ch.last_brightness = brightness

        regions = self._region_hashes(gray) if gray else None
        moved_idx: List[int] = []
        if regions and ch.last_regions and len(regions) == len(ch.last_regions):
            moved_idx = [i for i, (a, b) in enumerate(zip(regions, ch.last_regions))
                         if hamming(a, b) > 3]
        moved = sorted({_REGION_NAMES[i] for i in moved_idx})

        if changed:
            p = Percept.from_image(analysis, source=f"live:{ch.name}", at=now)
            p.tags.append("change")
            p.data["change_bits"] = bits
            if ch.kind == "camera" and len(moved) >= self.motion_regions:
                p.tags.append("motion")
                p.data["motion_regions"] = moved
                direction = self._motion_direction(ch, moved_idx)
                if direction:
                    p.data["direction"] = direction
                events.append(self._score("motion", ch.kind, p, note=note))
            else:
                events.append(self._score("visual_change", ch.kind, p, note=note))
            if ch.kind == "camera" and self.faces:
                events.extend(self._sense_faces(ch, data, now, note))
            if ch.kind == "screen" and self.ocr:
                events.extend(self._read_screen(ch, data, now))
        elif now - ch.last_scene_at >= self.scene_interval_s:
            # scene keep-alive: the world model stays fed even when nothing changes
            ch.last_scene_at = now
            p = Percept.from_image(analysis, source=f"live:{ch.name}", at=now)
            p.tags.append("scene")
            p.data["brightness"] = round(brightness, 1)
            ev = self._score("scene", ch.kind, p, note=note)
            ev.salience = min(ev.salience, 0.2)     # never escalated on its own
            events.append(ev)

        if ch.kind == "screen" and self.window_watch:
            events.extend(self._sense_window(ch, now))

        if h is not None:
            ch.last_hash = h
        if regions:
            ch.last_regions = regions
        return events

    def _motion_direction(self, ch: ModalityChannel, moved_idx: List[int]
                          ) -> Optional[str]:
        """Direction of travel from the moving-region centroid across frames."""
        if not moved_idx:
            ch.last_motion_cx = ch.last_motion_cy = None
            return None
        cx = sum(i % _GRID for i in moved_idx) / len(moved_idx)
        cy = sum(i // _GRID for i in moved_idx) / len(moved_idx)
        last_cx, last_cy = ch.last_motion_cx, ch.last_motion_cy
        ch.last_motion_cx, ch.last_motion_cy = cx, cy
        if last_cx is None or last_cy is None:
            return None
        dx, dy = cx - last_cx, cy - last_cy
        parts: List[str] = []
        if abs(dx) >= 0.7:
            parts.append("rightward" if dx > 0 else "leftward")
        if abs(dy) >= 0.7:
            parts.append("downward" if dy > 0 else "upward")
        return " and ".join(parts) or None

    def _sense_faces(self, ch: ModalityChannel, data: bytes, now: float,
                     note: str) -> List[PerceptEvent]:
        """Faces (and bodies without a face) in a camera frame — Haar cascades, run
        only on frames that already changed (cost guard). ``None`` from the detector
        means *unavailable*, never 'nobody there'."""
        rects = self.detect(data, _FACE_CASCADE)
        if rects is None:
            return []
        events: List[PerceptEvent] = []
        n = len(rects)
        prev = ch.last_faces
        cx = (sum(x + w / 2.0 for x, y, w, hh in rects) / n) if n else None
        if (prev is None and n > 0) or (prev is not None and n != prev):
            appeared = n > (prev or 0)
            if appeared:
                msg = (f"{n} face{'s' if n != 1 else ''} appeared in {ch.name}"
                       f" (was {prev if prev is not None else 'none'}).")
            else:
                msg = (f"Face count in {ch.name} dropped to {n} (was {prev}).")
            p = Percept.from_text(msg, source=f"live:{ch.name}", tags=["face"], at=now)
            p.data.update({"faces": n, "previous": prev,
                           "boxes": [list(r) for r in rects[:8]]})
            ev = self._score("face", "camera", p, note=note)
            if appeared:
                ev.salience = max(ev.salience, 0.9)
            events.append(ev)
        elif n and prev and cx is not None and ch.last_face_cx is not None:
            span = max((x + w for x, y, w, hh in rects), default=1)
            if abs(cx - ch.last_face_cx) >= 0.15 * max(1, span):
                direction = "rightward" if cx > ch.last_face_cx else "leftward"
                p = Percept.from_text(
                    f"The face in {ch.name} moved {direction}.",
                    source=f"live:{ch.name}", tags=["face"], at=now)
                p.data.update({"faces": n, "direction": direction})
                events.append(self._score("face", "camera", p, note=note))
        ch.last_faces = n
        ch.last_face_cx = cx if cx is not None else ch.last_face_cx

        if n == 0 and self.bodies:
            bodies = self.detect(data, _BODY_CASCADE)
            nb = len(bodies) if bodies is not None else 0
            if nb and not ch.last_bodies:
                p = Percept.from_text(
                    f"You see a person in {ch.name} (body detected, face not visible).",
                    source=f"live:{ch.name}", tags=["person"], at=now)
                p.data["bodies"] = nb
                ev = self._score("person", "camera", p, note=note)
                ev.salience = max(ev.salience, 0.8)
                events.append(ev)
            ch.last_bodies = nb
        else:
            ch.last_bodies = 0
        return events

    def _sense_window(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        """Focused-window tracking (X11): a title change is a real attention shift."""
        title = self.window_title()
        if not title or title == ch.last_window:
            return []
        prev, ch.last_window = ch.last_window, title
        if not prev:
            return []                       # first sighting primes, never startles
        lowered = title.casefold()
        hot = any(w in lowered for w in self.attention_words)
        p = Percept.from_text(
            f'The active window changed to "{title[:120]}".',
            source="live:screen", tags=["window"], at=now)
        p.data.update({"title": title[:200], "previous": prev[:200]})
        ev = self._score("window_change", "screen", p)
        ev.salience = max(ev.salience, 0.8 if hot else 0.3)
        return [ev]

    @staticmethod
    def _gray(data: bytes) -> Optional[List[List[int]]]:
        dec = decode_image(data)
        if dec is None:
            return None
        w, h, px = dec
        gray = gray_from_pixels(w, h, px)
        if max(w, h) > 64 and w and h:
            scale = 64 / max(w, h)
            gray = resize_gray(gray, max(1, int(w * scale)), max(1, int(h * scale)))
        return gray

    @staticmethod
    def _region_hashes(gray: List[List[int]]) -> Optional[List[int]]:
        h, w = len(gray), len(gray[0]) if gray else 0
        if h < _GRID or w < _GRID:
            return None
        rh, rw = h // _GRID, w // _GRID
        out: List[int] = []
        for gy in range(_GRID):
            for gx in range(_GRID):
                block = [row[gx * rw:(gx + 1) * rw]
                         for row in gray[gy * rh:(gy + 1) * rh]]
                out.append(average_hash(block, size=4))
        return out

    def _read_screen(self, ch: ModalityChannel, data: bytes, now: float
                     ) -> List[PerceptEvent]:
        text, note = self.vision._ocr_bytes(data)
        if not text:
            return []
        sim = difflib.SequenceMatcher(None, text, ch.last_ocr_text).ratio() \
            if ch.last_ocr_text else 0.0
        ch.last_ocr_text = text
        if sim >= 0.7:
            return []
        p = Percept.from_text(text[:300], source=f"live:{ch.name}",
                              tags=["screen_text"], at=now)
        ev = self._score("screen_text", "screen", p, note=note)
        ev.text = text[:500]
        return [ev]

    # ------------------------------------------------------------------ #
    # Interoception — host vitals, processes (senses/system.py)
    # ------------------------------------------------------------------ #
    def _sense_system(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        sample = self.system_sensor.sample()
        self.last_vitals = sample
        ch.captures += 1
        ch.last_note = sample.get("note", "") or "system: vitals read"
        events: List[PerceptEvent] = []

        cpus = os.cpu_count() or 1
        vec = [float(sample.get("cpu", 0.0)), float(sample.get("mem", 0.0)),
               min(1.0, float(sample.get("load1", 0.0)) / cpus)]
        self._observe_stream("system", vec)

        # rising-edge alerts with hysteresis: a sustained condition alerts once, not
        # every tick; it re-arms after dropping 5 points below its threshold
        checks: List[Tuple[str, bool, bool, str]] = []
        cpu = sample.get("cpu")
        if cpu is not None:
            top = sample.get("top_process")
            checks.append(("cpu", cpu >= self.cpu_alert, cpu < self.cpu_alert - 0.05,
                           f"Your CPU is at {cpu:.0%}"
                           + (f" — '{top}' is the top consumer." if top else ".")))
        mem = sample.get("mem")
        if mem is not None:
            checks.append(("mem", mem >= self.mem_alert, mem < self.mem_alert - 0.05,
                           f"Your memory is {mem:.0%} full."))
        disk = sample.get("disk")
        if disk is not None:
            checks.append(("disk", disk >= self.disk_alert, disk < self.disk_alert - 0.05,
                           f"Your disk is {disk:.0%} full."))
        batt, charging = sample.get("battery"), sample.get("battery_charging")
        if batt is not None:
            low = batt <= self.battery_alert and charging is not True
            checks.append(("battery", low, batt > self.battery_alert + 0.05,
                           f"Your battery is down to {batt:.0%} and discharging."))
        temp = sample.get("temp_c")
        if temp is not None:
            checks.append(("temp", temp >= self.temp_alert_c,
                           temp < self.temp_alert_c - 5.0,
                           f"Your temperature reached {temp:.0f}°C."))

        alerts: List[str] = []
        for key, firing, cleared, message in checks:
            if firing and key not in self._active_alerts:
                self._active_alerts.add(key)
                alerts.append(message)
            elif cleared:
                self._active_alerts.discard(key)
        if alerts:
            p = Percept.from_text(" ".join(alerts), source="live:system",
                                  tags=["system", "alert"], at=now)
            p.data.update({k: sample[k] for k in
                           ("cpu", "mem", "disk", "battery", "temp_c") if k in sample})
            ev = self._score("system_alert", "system", p,
                             note=sample.get("note", ""))
            ev.salience = max(ev.salience, 0.85)
            events.append(ev)
        elif now - ch.last_scene_at >= self.scene_interval_s:
            # vitals keep-alive: the world model stays fed even when the body is calm
            ch.last_scene_at = now
            p = Percept.from_text(
                "vitals: " + ", ".join(f"{k}={sample[k]}" for k in
                                       ("cpu", "mem", "disk", "load1") if k in sample),
                source="live:system", tags=["vitals"], at=now)
            p.data.update(sample)
            ev = self._score("vitals", "system", p, note=sample.get("note", ""))
            ev.salience = min(ev.salience, 0.2)     # never escalated on its own
            events.append(ev)

        if self.process_watch:
            fresh = self.system_sensor.new_processes()
            if fresh:
                names = ", ".join(f"'{n}'" for n in fresh[:5])
                more = f" (+{len(fresh) - 5} more)" if len(fresh) > 5 else ""
                p = Percept.from_text(
                    f"A new program started running: {names}{more}.",
                    source="live:system", tags=["process"], at=now)
                p.data["new_processes"] = fresh[:16]
                events.append(self._score("process_new", "system", p))
        return events

    # ------------------------------------------------------------------ #
    # Filesystem watching (senses/watch.py)
    # ------------------------------------------------------------------ #
    def _sense_files(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        watcher = self._ensure_watcher()
        if watcher is None:
            ch.misses += 1
            ch.last_note = "files: watcher unavailable"
            return []
        created, modified, deleted = watcher.diff()
        ch.captures += 1
        ch.last_note = f"files: watching {len(watcher.roots)} root(s)" \
                       + (" [truncated]" if watcher.truncated else "")
        total = len(created) + len(modified) + len(deleted)
        if not total:
            return []
        parts: List[str] = []
        for verb, paths in (("created", created), ("modified", modified),
                            ("deleted", deleted)):
            if paths:
                shown = ", ".join(Path(p).name for p in paths[:4])
                more = f" (+{len(paths) - 4} more)" if len(paths) > 4 else ""
                parts.append(f"{verb}: {shown}{more}")
        p = Percept.from_text(
            f"Files changed under your watched folders — {'; '.join(parts)}.",
            source="live:files", tags=["files"], at=now)
        p.data.update({"created": created[:8], "modified": modified[:8],
                       "deleted": deleted[:8], "total": total})
        ev = self._score("file_change", "files", p)
        ev.salience = max(ev.salience, min(0.9, 0.3 + 0.05 * total))
        return [ev]

    # ------------------------------------------------------------------ #
    # Network sensing (senses/system.py NetworkSensor)
    # ------------------------------------------------------------------ #
    def _sense_network(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        transitions = self.network_sensor.transitions()
        ch.captures += 1
        state = getattr(self.network_sensor, "_last_state", None) or {}
        ch.last_note = "network: " + ("online" if state.get("online") else
                                      "offline" if state.get("online") is False
                                      else "state unknown")
        if "rx_bps" in state and "tx_bps" in state:
            # traffic as a sensory stream: a statistical break (sudden saturation,
            # sudden dead air) surfaces through the same anomaly path as the eyes
            self._observe_stream("network", [
                min(1.0, state["rx_bps"] / 1e8), min(1.0, state["tx_bps"] / 1e8)])
        events: List[PerceptEvent] = []
        for kind, message, salience in transitions:
            p = Percept.from_text(message, source="live:network",
                                  tags=["network"], at=now)
            p.data.update({k: state[k] for k in ("online", "reachable", "ssid")
                           if k in state})
            ev = self._score(kind, "network", p)
            ev.salience = max(ev.salience, salience)
            events.append(ev)
        return events

    # ------------------------------------------------------------------ #
    # Device hot-plug (senses/system.py DeviceSensor) — she gains/loses organs live
    # ------------------------------------------------------------------ #
    def _sense_devices(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        added, removed = self.device_sensor.diff()
        ch.captures += 1
        ch.last_note = f"devices: {len(getattr(self.device_sensor, '_last', None) or {})} present"
        events: List[PerceptEvent] = []
        for dev_id, label in added.items():
            message = f"A {label} was just plugged in."
            if dev_id.startswith("/dev/video"):
                message += " You can see through it now."
                self._adopt_camera(dev_id)
            p = Percept.from_text(message, source="live:devices",
                                  tags=["device"], at=now)
            p.data["device"] = dev_id
            ev = self._score("device_plugged", "devices", p)
            ev.salience = max(ev.salience, 0.7)
            events.append(ev)
        for dev_id, label in removed.items():
            if dev_id.startswith("/dev/video"):
                self._release_camera(dev_id)
            p = Percept.from_text(f"The {label} was just removed.",
                                  source="live:devices", tags=["device"], at=now)
            p.data["device"] = dev_id
            ev = self._score("device_removed", "devices", p)
            ev.salience = max(ev.salience, 0.6)
            events.append(ev)
        return events

    def _adopt_camera(self, node: str) -> None:
        """A freshly plugged ``/dev/videoN`` becomes a live channel, no restart."""
        if not self.sensor.enabled.get("camera", True):
            return
        suffix = node.removeprefix("/dev/video")
        index = int(suffix) if suffix.isdigit() else 0
        name = f"camera{index}"
        cameras = sum(1 for c in self.channels.values() if c.kind == "camera")
        if name in self.channels or cameras >= self.max_cameras:
            return
        self.channels[name] = ModalityChannel(kind="camera", name=name, index=index,
                                              interval=self.interval_s,
                                              base_interval=self.interval_s)

    def _release_camera(self, node: str) -> None:
        suffix = node.removeprefix("/dev/video")
        name = f"camera{suffix}" if suffix.isdigit() else ""
        self.channels.pop(name, None)

    # ------------------------------------------------------------------ #
    # Self-health — a sense that keeps failing is itself a percept
    # ------------------------------------------------------------------ #
    def _maybe_degraded(self, ch: ModalityChannel, now: float) -> List[PerceptEvent]:
        if ch.error_streak < 3 or now - ch.last_degraded_at < 600.0:
            return []
        ch.last_degraded_at = now
        p = Percept.from_text(
            f"Your {ch.name} sense keeps failing — {ch.error_streak} consecutive "
            f"errors (last note: {ch.last_note or 'none'}).",
            source=f"live:{ch.name}", tags=["anomaly"], at=now)
        ev = self._score("sense_degraded", ch.kind, p)
        ev.salience = max(ev.salience, 0.6)
        return [ev]

    # ------------------------------------------------------------------ #
    # Scoring / registration
    # ------------------------------------------------------------------ #
    def _score(self, kind: str, modality: str, p: Percept, *, note: str = "",
               transcript: Optional[str] = None) -> PerceptEvent:
        bound, _ = self.binder.perceive(p)
        surprise = self.predictor.observe(bound, boost_salience=True)
        ev = PerceptEvent(kind=kind, modality=modality, percept=bound,
                          salience=bound.salience, surprise=surprise.surprise,
                          transcript=transcript, note=note)
        if surprise.anomaly:
            ev.salience = max(ev.salience, 0.8)
        return ev

    def _observe_stream(self, name: str, vec: List[float]) -> None:
        sp = self._stream_predictors.get(name)
        if sp is None:
            sp = self._stream_predictors[name] = SensoryPredictor(warmup=8)
        try:
            res = sp.observe(vec)
            if res.anomaly:
                # a statistical break (lights off, sustained noise) even without a
                # hash/VAD trigger — recorded as its own event next registration pass
                p = Percept.from_text(
                    f"sensory anomaly on {name}: observed {vec}, expected "
                    f"{[round(x, 3) for x in (res.predicted or vec)]}",
                    source=f"live:{name}", tags=["anomaly"])
                self._register(self._score("anomaly", name.rstrip("0123456789"), p))
        except Exception:  # noqa: BLE001 — dim mismatch etc. must not break sensing
            self._note_error("stream")

    def add_listener(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to live events (each delivered as ``PerceptEvent.to_dict()``).
        Callbacks run on the perception thread and must be fast and thread-safe."""
        if cb not in self._listeners:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        try:
            self._listeners.remove(cb)
        except ValueError:
            pass

    def _register(self, ev: PerceptEvent) -> None:
        self.events.append(ev)
        self.event_counts[ev.kind] = self.event_counts.get(ev.kind, 0) + 1
        if self._listeners:
            d = ev.to_dict()
            for cb in list(self._listeners):
                try:
                    cb(d)
                except Exception:  # noqa: BLE001 — a slow/broken listener never breaks sensing
                    self._note_error("listener")
        try:
            if self.presence is not None:
                self.presence.on_stimulus(ev.salience, reason=f"live {ev.kind}")
        except Exception:  # noqa: BLE001
            self._note_error("presence")
        try:
            if self.mind is not None:
                from nyxara.observe.mindscope import ThoughtKind
                self.mind.record(ThoughtKind.PERCEPTION,
                                 f"[live/{ev.modality}] {ev.kind}: {ev.percept.content[:120]}",
                                 salience=ev.salience)
        except Exception:  # noqa: BLE001
            self._note_error("mind")
        try:
            from nyxara.growth.signal_bus import get_signal_bus
            get_signal_bus().post("live_percept", ev.to_dict(), source="realtime")
        except Exception:  # noqa: BLE001
            self._note_error("signal_bus")

    # ------------------------------------------------------------------ #
    # Escalation into cognition
    # ------------------------------------------------------------------ #
    def _flush_escalations(self, fresh: Sequence[PerceptEvent]) -> None:
        if self.escalate is None:
            return
        # retry anything deferred while the Master's turn was running
        still: List[Tuple[str, List[Percept], float]] = []
        for stim, media, at in self._pending:
            if not self._try_escalate(stim, media):
                still.append((stim, media, at))
        self._pending = still[-_MAX_PENDING:]

        worthy = [e for e in fresh if self._worth_escalating(e)]
        if not worthy:
            return
        top = max(worthy, key=lambda e: e.salience)
        now = time.time()
        if top.kind not in _URGENT_KINDS \
                and now - self._last_escalation_at < self.min_escalation_interval_s:
            return
        # cross-modal merge: simultaneous events across modalities travel together
        group = [e for e in worthy
                 if abs(e.at - top.at) <= self.comodal_window_s] or [top]
        if len({e.modality for e in group}) > 1:
            for e in group:
                e.salience = min(1.0, e.salience * 1.25)   # simultaneity is evidence
        stim = self._compose_stimulus(group)
        media = [e.percept for e in group]
        self._last_escalation_at = now
        if not self._try_escalate(stim, media):
            self._pending.append((stim, media, now))
            self._pending = self._pending[-_MAX_PENDING:]
            self.deferred_escalations += 1
        else:
            for e in group:
                self._journal(e)
                self._remember(e)

    def _worth_escalating(self, e: PerceptEvent) -> bool:
        if e.kind in ("scene", "vitals"):
            return False                    # keep-alives feed the world model only
        if e.kind in ("wake_word", "speech", "sound", "alarm", "visual_change",
                      "motion", "screen_text", "face", "person", "lights",
                      "system_alert", "process_new", "file_change", "network_change",
                      "port_change", "device_plugged", "device_removed"):
            return True
        # window_change, sense_degraded, anomaly, surprise: only when truly salient
        return e.salience >= self.surprise_threshold

    def _try_escalate(self, stimulus: str, media: List[Percept]) -> bool:
        try:
            if self.gate is not None and not self.gate():
                return True     # oversight says halt — drop, don't queue forever
            result = self.escalate(stimulus, media) if self.escalate else None
            if result is None:
                return False    # the core is engaged; requeue for the next tick
            self.escalations += 1
            return True
        except Exception:  # noqa: BLE001
            self._note_error("escalate")
            return True         # never let a broken escalation grow the queue unbounded

    def _compose_stimulus(self, group: Sequence[PerceptEvent]) -> str:
        parts: List[str] = []
        for e in group:
            if e.kind == "wake_word":
                parts.append(f'You heard your name: "{e.transcript}"')
            elif e.kind == "speech":
                if e.transcript:
                    parts.append(f'You heard speech just now: "{e.transcript}"')
                else:
                    parts.append("You heard a voice on the microphone "
                                 f"(no transcription available: {e.note or 'whisper absent'}).")
            elif e.kind == "sound":
                parts.append("You heard a sudden loud sound on the microphone.")
            elif e.kind == "motion":
                regions = ", ".join(e.percept.data.get("motion_regions", [])) or "the frame"
                parts.append(f"You saw motion in the camera ({regions}).")
            elif e.kind == "visual_change":
                bits = e.percept.data.get("change_bits")
                where = "the camera view" if e.modality == "camera" else "the screen"
                parts.append(f"You saw {where} change substantially"
                             + (f" (hash distance {bits} bits)." if bits is not None else "."))
            elif e.kind == "screen_text":
                parts.append(f'Text appeared on screen: "{(e.text or "")[:200]}"')
            elif e.kind == "alarm":
                beep = (e.percept.data.get("sound_class") or {}).get("beep_hz")
                parts.append("You hear a periodic alarm-like beeping"
                             + (f" (~{beep} beeps/s)." if beep else "."))
            elif e.kind == "face":
                parts.append(f"You see it with your own eyes: {e.percept.content}")
            elif e.kind in _SELF_WORDED_KINDS:
                parts.append(e.percept.content[:300])   # written in first person at birth
            elif e.kind == "anomaly":
                parts.append(f"Your senses flagged an anomaly: {e.percept.content[:160]}")
            else:
                parts.append(f"You perceived something surprising: {e.percept.content[:120]}")
        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Memory + journal
    # ------------------------------------------------------------------ #
    def _remember(self, ev: PerceptEvent) -> None:
        try:
            if self.remember is not None:
                self.remember(f"[perceived/{ev.kind}] {ev.percept.content[:160]}",
                              ev.to_dict())
        except Exception:  # noqa: BLE001
            self._note_error("memory")

    def _journal(self, ev: PerceptEvent) -> None:
        if not self.journal_enabled or self._journal_path is None:
            return
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
            self._journal_lines += 1
            if self._journal_lines >= _JOURNAL_MAX_LINES:
                self._rotate_journal()
        except Exception:  # noqa: BLE001
            self._note_error("journal")

    def _rotate_journal(self) -> None:
        try:
            lines = self._journal_path.read_text(encoding="utf-8").splitlines()
            keep = lines[-_JOURNAL_MAX_LINES // 2:]
            self._journal_path.write_text("\n".join(keep) + "\n", encoding="utf-8")
            self._journal_lines = len(keep)
        except Exception:  # noqa: BLE001
            self._note_error("journal")

    # ------------------------------------------------------------------ #
    # Housekeeping / introspection
    # ------------------------------------------------------------------ #
    def _maybe_reset_frame(self) -> None:
        if len(self.binder.frame) > 512:
            self.binder.new_frame()

    def _note_error(self, stage: str) -> None:
        self.errors += 1
        self.stage_errors[stage] = self.stage_errors.get(stage, 0) + 1

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in list(self.events)[-max(1, n):]]

    def pending_count(self) -> int:
        return len(self._pending)

    def status(self) -> Dict[str, Any]:
        """Honest, measured state of the senses — what she can perceive right now."""
        return {
            "running": self.running,
            "mic_mode": self.mic_mode,
            "available": self._available(),
            "noise_floor": round(_median(list(self._noise_floor)), 5),
            "vad_threshold": round(self._vad_threshold(), 5),
            "channels": {name: ch.status() for name, ch in self.channels.items()},
            "events": sum(self.event_counts.values()),
            "event_counts": dict(self.event_counts),
            "recent_events": self.recent(5),
            "escalations": self.escalations,
            "deferred_escalations": self.deferred_escalations,
            "pending": len(self._pending),
            "last_transcript": self.last_transcript,
            "last_vitals": dict(self.last_vitals),
            "active_alerts": sorted(self._active_alerts),
            "journal_lines": self._journal_lines,
            "errors": self.errors,
            "stage_errors": dict(self.stage_errors),
        }


# --------------------------------------------------------------------------- #
# Self-test / demo — honest about what this box can actually perceive
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA realtime-perception self-test (real devices; honest when absent)")
    print("=" * 70)

    # 1) On THIS box: run one real sweep and report honestly.
    live = RealtimePerception(settings=object())
    avail = live._available()
    print(f"\nlive availability    : {avail}")
    evs = live.tick_once()
    print(f"real sweep events    : {len(evs)} "
          f"(channels: {list(live.channels)}, mic mode: {live.mic_mode})")

    # 2) Deterministic proof with a stub sensor: VAD, change detection, escalation.
    from nyxara.senses.generate import encode_png

    def png_of(v: int) -> bytes:
        rows = []
        for y in range(64):
            row = bytearray()
            for x in range(64):
                px = v if (x // 8 + y // 8) % 2 else 255 - v
                row.extend((px, px, px))
            rows.append(bytes(row))
        return encode_png(64, 64, rows)

    def tone_wav(amp: float, secs: float = 1.0, rate: int = 16000) -> bytes:
        sig = [amp * math.sin(2 * math.pi * 440 * i / rate)
               for i in range(int(secs * rate))]
        return _floats_to_wav(sig, rate=rate)

    class StubSensor(LiveSensor):
        def __init__(self, frames, clips):
            super().__init__()
            self.frames, self.clips = list(frames), list(clips)
        def available(self):
            return {"camera": False, "screen": bool(self.frames), "mic": bool(self.clips)}
        def capture_screen(self, monitor: int = 0):
            return (self.frames.pop(0), "screen: stub") if self.frames else (None, "done")
        def capture_microphone(self, seconds=None, rate=None):
            return (self.clips.pop(0), "mic: stub") if self.clips else (None, "done")

    got: List[str] = []
    stub = StubSensor(frames=[png_of(0), png_of(0), png_of(200)],
                      clips=[tone_wav(0.0), tone_wav(0.5), tone_wav(0.0)])
    rp = RealtimePerception(stub, escalate=lambda s, m: got.append(s) or True,
                            settings=object(), journal_path=None)
    rp.interval_s = rp.burst_interval_s = 0.0
    for ch in rp.channels.values():
        ch.interval = ch.base_interval = 0.0
    for _ in range(4):
        rp.tick_once()
        for ch in rp.channels.values():
            ch.next_due = 0.0
    kinds = [e.kind for e in rp.events]
    print(f"\nstub events          : {kinds}")
    assert "speech" in kinds or "sound" in kinds, "loud clip must be heard"
    assert "visual_change" in kinds, "changed frame must be seen"
    assert got, "salient events must escalate into cognition"
    print(f"escalated stimuli    : {got}")
    print(f"status               : events={rp.status()['events']} "
          f"escalations={rp.status()['escalations']}")

    # 3) Host senses with deterministic stand-ins: vitals alert, files, network.
    class VitalsStub:
        def __init__(self):
            self.samples = [{"cpu": 0.20, "mem": 0.4, "load1": 0.5, "note": ""},
                            {"cpu": 0.97, "mem": 0.4, "load1": 3.0, "note": "",
                             "top_process": "stress"}]
        def sample(self):
            return self.samples.pop(0) if self.samples \
                else {"cpu": 0.10, "mem": 0.4, "load1": 0.5, "note": ""}
        def new_processes(self):
            return []

    class WatchStub:
        roots = ["stub"]
        truncated = False
        def __init__(self):
            self.diffs = [([], [], []), (["/tmp/new.txt"], [], [])]
        def diff(self):
            return self.diffs.pop(0) if self.diffs else ([], [], [])

    class NetStub:
        _last_state = {"online": True}
        def __init__(self):
            self.runs = [[], [("network_change", "The network connection just went down.",
                              0.8)]]
        def transitions(self):
            return self.runs.pop(0) if self.runs else []

    got2: List[str] = []
    rp2 = RealtimePerception(StubSensor([], []), settings=object(),
                             escalate=lambda s, m: got2.append(s) or True,
                             journal_path=None)
    rp2.min_escalation_interval_s = 0.0
    rp2.system_sensor, rp2.watcher = VitalsStub(), WatchStub()
    rp2.network_sensor, rp2.fswatch_enabled = NetStub(), True
    for name, kind_ in (("system", "system"), ("files", "files"), ("network", "network")):
        rp2.channels[name] = ModalityChannel(kind=kind_, name=name,
                                             interval=0.0, base_interval=0.0)
    for _ in range(3):
        rp2.tick_once()
        for ch in rp2.channels.values():
            ch.next_due = 0.0
    kinds2 = [e.kind for e in rp2.events]
    print(f"\nhost-sense events    : {kinds2}")
    assert "system_alert" in kinds2, "a 97% CPU spike must raise a system alert"
    assert "file_change" in kinds2, "a created file must be noticed"
    assert "network_change" in kinds2, "an offline transition must be noticed"
    assert got2, "host events must escalate into cognition"
    print(f"host stimuli         : {got2}")

    # 4) Sound classification is live in the mic path: periodic beeps become "alarm".
    def beeps_wav(secs: float = 3.0, rate: int = 16000) -> bytes:
        sig = [(0.6 * math.sin(2 * math.pi * 1000 * i / rate)
                if (i // (rate // 4)) % 2 == 0 else 0.0) for i in range(int(secs * rate))]
        return _floats_to_wav(sig, rate=rate)

    rp3 = RealtimePerception(StubSensor(frames=[], clips=[beeps_wav()]),
                             settings=object(), journal_path=None)
    rp3.transcribe = False
    rp3.sound_spike_snr = 1.0           # ensure the spike gate opens for the stub clip
    for ch in rp3.channels.values():
        ch.next_due = 0.0
    rp3.tick_once()
    kinds3 = [e.kind for e in rp3.events]
    print(f"\nalarm-clip events    : {kinds3}")
    assert "alarm" in kinds3, "periodic beeping must classify as an alarm"

    print("\nNOTE: on a headless box the real sweep above reports no devices — that is the")
    print("correct behaviour: she never fabricates a percept. On a machine with a webcam,")
    print("display or microphone the same loop perceives the live world continuously.")
    print("\nALL SELF-TESTS PASSED ✓")
