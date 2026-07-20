"""NYXARA · senses/realtime.py — continuous real-time perception (👁👂, always on).

Every other sense answers only when asked: :mod:`senses.live` grabs one frame or one clip
per call, the file senses decode stored media, and nothing perceives *between* prompts.
This module closes that gap — it is the loop that keeps NYXARA's eyes and ears open. A
background thread continuously captures camera frames, screen frames and microphone audio
through the existing :class:`~nyxara.senses.live.LiveSensor`, detects salient moments
**natively** — energy voice-activity detection with a self-calibrating noise floor,
full-utterance endpointing, wake-word spotting, perceptual-hash visual change, region-grid
motion, OCR screen reading, and predictive surprise — and escalates what matters into a
real autonomous cognitive cycle, so she reacts to the world unprompted.

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
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from nyxara.senses.audio import (
    Audio,
    AudioAnalysis,
    audio_fingerprint,
    decode_pcm,
    parse_header,
    rms,
)
from nyxara.senses.binding import Binder, Modality, Percept
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

__all__ = ["PerceptEvent", "ModalityChannel", "RealtimePerception"]

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
# Event
# --------------------------------------------------------------------------- #
@dataclass
class PerceptEvent:
    """One salient moment the loop noticed — what, where, how sure, and the percept."""

    kind: str                      # speech | wake_word | sound | visual_change | motion
    #                              # | screen_text | surprise | anomaly | scene
    modality: str                  # camera | screen | mic
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

    kind: str                       # "camera" | "screen" | "mic"
    name: str                       # "camera0", "screen", "mic", ...
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

    def _build_channels(self) -> Dict[str, ModalityChannel]:
        chans: Dict[str, ModalityChannel] = {}

        def add(kind: str, name: str, index: int = 0) -> None:
            chans[name] = ModalityChannel(kind=kind, name=name, index=index,
                                          interval=self.interval_s,
                                          base_interval=self.interval_s)

        if self.sensor.enabled.get("camera", True):
            nodes = sorted(glob.glob("/dev/video*"))[: self.max_cameras]
            for i, _ in enumerate(nodes or [None]):     # at least camera0 everywhere
                add("camera", f"camera{i}", i)
        if self.sensor.enabled.get("screen", True):
            add("screen", "screen")
        if self.sensor.enabled.get("mic", True):
            add("mic", "mic")
        return chans

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
        for ch in self.channels.values():
            if now < ch.next_due:
                continue
            ch.ticks += 1
            if not avail.get(ch.kind, False):
                ch.misses += 1
                ch.interval = self.idle_interval_s
                ch.next_due = now + ch.interval
                continue
            try:
                if ch.kind == "mic":
                    events.extend(self._sense_mic(ch, now))
                else:
                    events.extend(self._sense_visual(ch, now))
            except Exception:  # noqa: BLE001
                self._note_error(ch.name)
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

        p = Percept.from_audio(analysis, source="live:mic", at=now)
        p.tags.append(kind)
        p.data.update({"rms": round(level, 5), "threshold": round(threshold, 5),
                       "duration_s": round(len(utter) / max(1, self._mic_rate), 2)})
        ev = self._score(kind, "mic", p, note=analysis.note or note,
                         transcript=analysis.transcript)
        if kind == "wake_word":
            ev.salience = 1.0
            p.salience = 1.0
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

        regions = self._region_hashes(gray) if gray else None
        moved: List[str] = []
        if regions and ch.last_regions and len(regions) == len(ch.last_regions):
            moved = sorted({_REGION_NAMES[i] for i, (a, b)
                            in enumerate(zip(regions, ch.last_regions))
                            if hamming(a, b) > 3})

        if changed:
            p = Percept.from_image(analysis, source=f"live:{ch.name}", at=now)
            p.tags.append("change")
            p.data["change_bits"] = bits
            if ch.kind == "camera" and len(moved) >= self.motion_regions:
                p.tags.append("motion")
                p.data["motion_regions"] = moved
                events.append(self._score("motion", ch.kind, p, note=note))
            else:
                events.append(self._score("visual_change", ch.kind, p, note=note))
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

        if h is not None:
            ch.last_hash = h
        if regions:
            ch.last_regions = regions
        return events

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
        if top.kind != "wake_word" \
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
        if e.kind in ("scene",):
            return False
        if e.kind in ("wake_word", "speech", "sound", "visual_change", "motion",
                      "screen_text"):
            return True
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

    print("\nNOTE: on a headless box the real sweep above reports no devices — that is the")
    print("correct behaviour: she never fabricates a percept. On a machine with a webcam,")
    print("display or microphone the same loop perceives the live world continuously.")
    print("\nALL SELF-TESTS PASSED ✓")
