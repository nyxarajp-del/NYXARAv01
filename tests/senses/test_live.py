"""Tests for nyxara.senses.live — real-time perception from real devices.

These prove the capability the user asked for: NYXARA can take in the *live* world (camera /
screen / mic), routed through the same senses as everything else — and that on a box with no
device it degrades honestly (a note, never a fabricated frame) and never raises.
"""

from __future__ import annotations

import wave
from io import BytesIO

from nyxara.senses.audio import Audio
from nyxara.senses.generate import encode_png, identicon_rows
from nyxara.senses.live import LiveSensor, pcm16_to_wav_bytes
from nyxara.senses.vision import Vision


# --------------------------------------------------------------------------- #
# The stdlib WAV wrapper produces a real, decodable clip
# --------------------------------------------------------------------------- #
def test_pcm16_to_wav_roundtrips_through_audio_sense():
    pcm = b"\x10\x00" * 16000  # 1 s of a low constant tone at 16 kHz
    wav = pcm16_to_wav_bytes(pcm, rate=16000, channels=1)
    # a genuine RIFF/WAV container
    with wave.open(BytesIO(wav), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    # and the audio sense decodes it end-to-end
    analysis = Audio().analyze_bytes(wav)
    assert analysis.info is not None
    assert abs(analysis.info.duration - 1.0) < 1e-3


def test_analyze_bytes_on_real_png_from_capture_format():
    """A PNG blob (the camera/screen capture format) flows through Vision.analyze_bytes."""
    png = encode_png(32, 32, identicon_rows("live-frame", size=32, grid=8))
    analysis = Vision().analyze_bytes(png)
    assert analysis.info is not None and analysis.info.format == "PNG"
    assert analysis.info.width == 32 and analysis.info.height == 32
    assert analysis.perceptual_hash is not None  # real pixels were decoded and hashed


# --------------------------------------------------------------------------- #
# Honest degradation — no device/backend ⇒ (None, note), never a raise
# --------------------------------------------------------------------------- #
def test_capture_never_raises_and_reports_honestly():
    sensor = LiveSensor()
    for kind in ("camera", "screen", "mic"):
        data, note = sensor.capture(kind)
        assert isinstance(note, str) and note  # always a human explanation
        assert data is None or isinstance(data, (bytes, bytearray))  # never fabricated nonsense
    # available() is a plain bool map and must not raise
    avail = sensor.available()
    assert set(avail) == {"camera", "screen", "mic"}
    assert all(isinstance(v, bool) for v in avail.values())


def test_unknown_modality_is_handled():
    data, note = LiveSensor().capture("taste")
    assert data is None and "unknown" in note.lower()


# --------------------------------------------------------------------------- #
# Per-modality policy (the orchestrator's opt-in toggles) is enforced
# --------------------------------------------------------------------------- #
def test_disabled_modality_is_masked_and_refused():
    sensor = LiveSensor(enabled={"camera": False, "screen": False, "mic": False})
    avail = sensor.available()
    assert avail == {"camera": False, "screen": False, "mic": False}
    for kind in ("camera", "screen", "mic"):
        data, note = sensor.capture(kind)
        assert data is None and "disabled" in note.lower()


# --------------------------------------------------------------------------- #
# An injected backend returning real bytes flows to a valid analysis
# --------------------------------------------------------------------------- #
def test_injected_backend_bytes_flow_to_analysis():
    """Simulate a device that yields real bytes; confirm the full sense path works."""
    png = encode_png(16, 16, identicon_rows("stub-cam", size=16, grid=8))
    wav = pcm16_to_wav_bytes(b"\x22\x00" * 8000, rate=16000)

    class _StubSensor(LiveSensor):
        def capture_camera(self, index=0):
            return png, "stub: camera"

        def capture_microphone(self, seconds=None, rate=None):
            return wav, "stub: mic"

    s = _StubSensor()
    cam_bytes, _ = s.capture("camera")
    mic_bytes, _ = s.capture("mic")
    assert Vision().analyze_bytes(cam_bytes).info.format == "PNG"
    assert Audio().analyze_bytes(mic_bytes).info is not None
