"""NYXARA · senses/live.py — real-time perception from real devices (📷🎙🖥).

Every other sense in :mod:`nyxara.senses` takes in *stored* media: :class:`~nyxara.senses.vision.Vision`
decodes image **files**, :class:`~nyxara.senses.audio.Audio` decodes audio **files**, the web
senses fetch **pages**. Nothing here reached out and grabbed the *live* world. This module closes
that gap: it captures a real **camera frame**, a real **screen frame**, and a real **microphone
clip** straight from the hardware, and returns them as bytes in formats the existing senses already
decode — **PNG** for sight, **WAV** for hearing — so live intake flows through the very same
``analyze → Percept → Binder → predictive-surprise`` pipeline, with no new decoders.

Design, matching the rest of NYXARA:

* **Real, never fake.** Every path performs genuine device I/O (OpenCV / v4l2, ``mss`` / Pillow /
  X11, ``sounddevice`` / ``pyaudio`` / ALSA). When no device or backend is present the sensor
  returns ``(None, note)`` — an *honest* absence. It never fabricates a frame or a sample.
* **Stdlib-first, degrade honestly, never raise.** Heavy backends are optional and import-guarded;
  the WAV writer and PNG encoder are pure standard library (the PNG path reuses
  :func:`nyxara.senses.generate.encode_png`). Any backend failure is swallowed into a note.
* **Privacy-aware.** Capturing a camera or microphone is sensitive. This module is a passive
  instrument — it captures only when explicitly asked. The *policy* of when NYXARA may look or
  listen lives one layer up (the embodied loop's oversight gate + opt-in flags), never here.

Nothing in this module is required for import; on a bare, headless box every capture simply reports
"no device", which is the correct real behaviour.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
import wave
from io import BytesIO
from typing import Dict, List, Optional, Tuple

__all__ = ["Capture", "LiveSensor", "pcm16_to_wav_bytes"]

# A capture returns (bytes_or_None, human_note). Bytes are PNG (image) or WAV (audio).
Capture = Tuple[Optional[bytes], str]

# How long we let an external CLI (ffmpeg/arecord/…) run before giving up. Capture is meant to be a
# quick snapshot, so a stuck device becomes an honest "no capture" instead of a hang.
_CLI_TIMEOUT = 8.0
# Availability probes are cheap but not free; cache the verdict briefly so a decide→act loop that
# calls available() every step does not re-stat /dev or re-import a backend each time.
_PROBE_TTL = 5.0


# --------------------------------------------------------------------------- #
# WAV helper (pure stdlib) — turn raw 16-bit PCM into a real, playable WAV blob
# --------------------------------------------------------------------------- #
def pcm16_to_wav_bytes(pcm: bytes, *, rate: int, channels: int = 1) -> bytes:
    """Wrap little-endian 16-bit PCM in a RIFF/WAV container (what :mod:`audio` decodes)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(max(1, channels))
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm)
    return buf.getvalue()


def _run_cli(argv: List[str]) -> Optional[bytes]:
    """Run an external capture command and return its stdout bytes, or ``None`` (never raises)."""
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=_CLI_TIMEOUT, check=False)
    except Exception:  # noqa: BLE001 — missing binary / timeout / permission ⇒ honest miss
        return None
    data = proc.stdout or b""
    return data if data else None


# --------------------------------------------------------------------------- #
# The live sensor
# --------------------------------------------------------------------------- #
class LiveSensor:
    """Grabs real frames/clips from real devices, as PNG/WAV bytes the senses already decode."""

    _MODALITIES = ("camera", "screen", "mic")

    def __init__(self, *, mic_seconds: float = 1.0, mic_rate: int = 16000,
                 enabled: Optional[Dict[str, bool]] = None) -> None:
        self.mic_seconds = max(0.05, float(mic_seconds))
        self.mic_rate = int(mic_rate)
        # Per-modality opt-in policy. Absent ⇒ all modalities permitted; a modality set to False
        # is treated as "never look/listen here" — available() masks it and capture() refuses.
        self.enabled = {m: True for m in self._MODALITIES}
        if enabled:
            self.enabled.update({k: bool(v) for k, v in enabled.items()})
        self._probe_cache: Dict[str, Tuple[float, bool]] = {}

    # ------------------------------------------------------------------ #
    # Availability — a cheap, cached, best-effort "is this device reachable?"
    # ------------------------------------------------------------------ #
    def available(self) -> Dict[str, bool]:
        """Which live modalities are reachable *and* permitted right now (camera / screen / mic)."""
        probes = {"camera": self._camera_available, "screen": self._screen_available,
                  "mic": self._mic_available}
        return {m: self.enabled.get(m, True) and self._cached(m, probes[m])
                for m in self._MODALITIES}

    def _cached(self, key: str, probe) -> bool:
        now = time.time()
        hit = self._probe_cache.get(key)
        if hit is not None and now - hit[0] < _PROBE_TTL:
            return hit[1]
        try:
            val = bool(probe())
        except Exception:  # noqa: BLE001 — a probe must never break the loop
            val = False
        self._probe_cache[key] = (now, val)
        return val

    @staticmethod
    def _has(mod: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:  # noqa: BLE001
            return False

    def _camera_available(self) -> bool:
        # A real capture device (v4l2 node) plus a way to read it, or OpenCV which manages its own.
        has_node = bool(glob.glob("/dev/video*"))
        if has_node and (self._has("cv2") or shutil.which("ffmpeg")):
            return True
        # macOS/Windows expose cameras through OpenCV without a /dev node.
        return self._has("cv2") and sys.platform in ("darwin", "win32")

    def _screen_available(self) -> bool:
        if sys.platform in ("darwin", "win32"):
            return self._has("mss") or self._has("PIL")
        # X11/Wayland: need a display and a grabber.
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
        return (self._has("mss") or self._has("PIL")
                or shutil.which("ffmpeg") is not None or shutil.which("import") is not None)

    def _mic_available(self) -> bool:
        if self._has("sounddevice"):
            try:  # confirm at least one *input* channel exists (a real mic), not just the lib
                import sounddevice as sd  # type: ignore
                if any(d.get("max_input_channels", 0) > 0 for d in sd.query_devices()):
                    return True
            except Exception:  # noqa: BLE001
                pass
        if self._has("pyaudio"):
            return True
        return shutil.which("arecord") is not None or shutil.which("ffmpeg") is not None

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #
    def capture(self, kind: str, **kw) -> Capture:
        if kind in self.enabled and not self.enabled[kind]:
            return None, f"{kind}: disabled by policy"
        if kind == "camera":
            return self.capture_camera(**kw)
        if kind == "screen":
            return self.capture_screen(**kw)
        if kind == "mic":
            return self.capture_microphone(**kw)
        return None, f"unknown live modality: {kind!r}"

    # ------------------------------------------------------------------ #
    # Camera → PNG bytes
    # ------------------------------------------------------------------ #
    def capture_camera(self, index: int = 0) -> Capture:
        """Grab one real frame from camera ``index`` as PNG bytes (OpenCV, then ffmpeg/v4l2)."""
        # Backend 1: OpenCV — opens the device and encodes the frame to PNG itself.
        if self._has("cv2"):
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(index)
                try:
                    ok, frame = cap.read()
                finally:
                    cap.release()
                if ok and frame is not None:
                    ok2, buf = cv2.imencode(".png", frame)
                    if ok2:
                        return bytes(buf.tobytes()), "camera: opencv"
            except Exception:  # noqa: BLE001
                pass
        # Backend 2: ffmpeg reading the v4l2 node directly, one PNG frame to stdout.
        if shutil.which("ffmpeg") and glob.glob("/dev/video*"):
            data = _run_cli(["ffmpeg", "-hide_banner", "-loglevel", "error",
                             "-f", "v4l2", "-i", f"/dev/video{index}",
                             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
            if data:
                return data, "camera: ffmpeg/v4l2"
        return None, "camera: no device or backend"

    # ------------------------------------------------------------------ #
    # Screen → PNG bytes
    # ------------------------------------------------------------------ #
    def capture_screen(self, monitor: int = 0) -> Capture:
        """Grab one real screenshot as PNG bytes (mss, then Pillow, then an X11 CLI)."""
        # Backend 1: mss — fast cross-platform grab; encode with our stdlib PNG encoder.
        if self._has("mss"):
            try:
                import mss  # type: ignore
                from nyxara.senses.generate import encode_png
                with mss.mss() as sct:
                    mons = sct.monitors
                    mon = mons[monitor] if 0 <= monitor < len(mons) else mons[0]
                    shot = sct.grab(mon)
                    w, h = shot.width, shot.height
                    rgb = shot.rgb  # already RGB, row-major, w*h*3 bytes
                    stride = w * 3
                    rows = [rgb[y * stride:(y + 1) * stride] for y in range(h)]
                    return encode_png(w, h, rows), "screen: mss"
            except Exception:  # noqa: BLE001
                pass
        # Backend 2: Pillow ImageGrab (macOS/Windows, and X11 with the xcb backend).
        if self._has("PIL"):
            try:
                from PIL import ImageGrab  # type: ignore
                img = ImageGrab.grab()
                out = BytesIO()
                img.convert("RGB").save(out, format="PNG")
                data = out.getvalue()
                if data:
                    return data, "screen: pillow"
            except Exception:  # noqa: BLE001
                pass
        # Backend 3: an X11 CLI grabber to stdout.
        if os.environ.get("DISPLAY"):
            if shutil.which("import"):  # ImageMagick
                data = _run_cli(["import", "-silent", "-window", "root", "png:-"])
                if data:
                    return data, "screen: imagemagick"
            if shutil.which("ffmpeg"):
                data = _run_cli(["ffmpeg", "-hide_banner", "-loglevel", "error",
                                 "-f", "x11grab", "-i", os.environ["DISPLAY"],
                                 "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
                if data:
                    return data, "screen: ffmpeg/x11grab"
        return None, "screen: no display or backend"

    # ------------------------------------------------------------------ #
    # Microphone → WAV bytes
    # ------------------------------------------------------------------ #
    def capture_microphone(self, seconds: Optional[float] = None,
                           rate: Optional[int] = None) -> Capture:
        """Record a real mic clip as WAV bytes (sounddevice, then pyaudio, then ALSA/ffmpeg)."""
        secs = self.mic_seconds if seconds is None else max(0.05, float(seconds))
        rate = self.mic_rate if rate is None else int(rate)
        # Backend 1: sounddevice — records int16 PCM; we wrap it in a stdlib WAV container.
        if self._has("sounddevice"):
            try:
                import sounddevice as sd  # type: ignore
                frames = int(secs * rate)
                rec = sd.rec(frames, samplerate=rate, channels=1, dtype="int16")
                sd.wait()
                pcm = bytes(rec.tobytes())
                if pcm:
                    return pcm16_to_wav_bytes(pcm, rate=rate, channels=1), "mic: sounddevice"
            except Exception:  # noqa: BLE001
                pass
        # Backend 2: pyaudio — read a stream of int16 frames, wrap as WAV.
        if self._has("pyaudio"):
            try:
                import pyaudio  # type: ignore
                pa = pyaudio.PyAudio()
                chunk = 1024
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                                 input=True, frames_per_buffer=chunk)
                buf = bytearray()
                try:
                    for _ in range(max(1, int(rate / chunk * secs))):
                        buf.extend(stream.read(chunk, exception_on_overflow=False))
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()
                if buf:
                    return pcm16_to_wav_bytes(bytes(buf), rate=rate, channels=1), "mic: pyaudio"
            except Exception:  # noqa: BLE001
                pass
        # Backend 3: ALSA arecord / ffmpeg to a WAV on stdout.
        if shutil.which("arecord"):
            data = _run_cli(["arecord", "-q", "-d", str(int(max(1, round(secs)))),
                             "-f", "S16_LE", "-r", str(rate), "-c", "1", "-t", "wav", "-"])
            if data:
                return data, "mic: arecord"
        if shutil.which("ffmpeg"):
            data = _run_cli(["ffmpeg", "-hide_banner", "-loglevel", "error",
                             "-f", "alsa", "-i", "default", "-t", str(secs),
                             "-ar", str(rate), "-ac", "1", "-f", "wav", "-"])
            if data:
                return data, "mic: ffmpeg/alsa"
        return None, "mic: no input device or backend"


# --------------------------------------------------------------------------- #
# Self-test / demo — honest about what this box can actually perceive
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA live-sensor self-test (real devices; honest when absent)")
    print("=" * 70)

    sensor = LiveSensor()
    avail = sensor.available()
    print(f"\nlive availability    : {avail}")

    for kind in ("camera", "screen", "mic"):
        data, note = sensor.capture(kind)
        if data:
            print(f"{kind:<8} captured    : {len(data)} bytes  ({note})")
        else:
            print(f"{kind:<8} unavailable : {note}")

    # The WAV wrapper is pure stdlib and always works — prove it round-trips through the audio sense.
    from nyxara.senses.audio import Audio
    silence = pcm16_to_wav_bytes(b"\x00\x00" * 16000, rate=16000, channels=1)
    an = Audio().analyze_bytes(silence)
    print(f"\nstdlib WAV wrapper   : {an.info.to_dict() if an.info else None}")
    assert an.info is not None and abs(an.info.duration - 1.0) < 1e-3, "WAV wrapper is broken"

    print("\nNOTE: on a headless box camera/screen/mic report 'unavailable' — that is the correct")
    print("real behaviour: NYXARA never fabricates a frame. Run on a machine with a webcam,")
    print("display, or microphone and the same calls return genuine live perception.")
    print("\nALL SELF-TESTS PASSED ✓")
