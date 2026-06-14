"""NYXARA · senses/audio.py — audio perception, header-deep & dependency-light (👂).

Hearing, like sight, must work on a bare box and sharpen when better tools exist. The
*core* here is pure standard library: it parses **WAV (RIFF)** and **AIFF (FORM)** headers
straight from raw bytes, decodes their PCM samples (8/16/24/32-bit, any channel count) to
a mono floating-point signal, and runs real signal analysis on it — with no audio stack
required and nothing that raises because a library is missing.

What it perceives:

* **Structure** — :func:`parse_header` yields format, channels, sample rate, bit depth,
  frame count and duration.
* **Signal features** — RMS loudness, peak, zero-crossing rate, and **silence detection**
  (per-frame energy thresholding → silent segments and a silence ratio).
* **Fingerprinting** — :func:`audio_fingerprint` builds a compact energy-envelope hash so
  near-duplicate clips have a small Hamming distance (dedup, like the vision hashes).

The :class:`Audio` facade bridges *real* files: it decodes WAV/AIFF itself, and for other
formats (mp3, flac, …) uses ``soundfile``/``librosa`` when present, degrading honestly to
header-only with a note when not. Transcription (Whisper / SpeechRecognition) is an
optional hook that returns ``None`` plus a note when its library is absent.

Pure standard library at the core; ``soundfile`` / ``librosa`` / ``whisper`` optional.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.senses.vision import hamming, hex_hash

__all__ = [
    "AudioInfo",
    "AudioAnalysis",
    "parse_header",
    "decode_pcm",
    "rms",
    "peak",
    "zero_crossing_rate",
    "silence_segments",
    "silence_ratio",
    "audio_fingerprint",
    "Audio",
]


# --------------------------------------------------------------------------- #
# Audio info
# --------------------------------------------------------------------------- #
@dataclass
class AudioInfo:
    format: str
    channels: int
    sample_rate: int
    sample_width: int          # bytes per sample
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def bit_depth(self) -> int:
        return self.sample_width * 8

    @property
    def bitrate(self) -> int:
        return self.sample_rate * self.channels * self.bit_depth

    def to_dict(self) -> Dict[str, Any]:
        return {"format": self.format, "channels": self.channels,
                "sample_rate": self.sample_rate, "bit_depth": self.bit_depth,
                "n_frames": self.n_frames, "duration": round(self.duration, 4),
                "bitrate": self.bitrate}


# --------------------------------------------------------------------------- #
# Header parsing (pure python)
# --------------------------------------------------------------------------- #
def _extended80(b: bytes) -> float:
    """Decode an 80-bit IEEE-754 extended float (AIFF sample rate)."""
    sign = b[0] >> 7
    exp = ((b[0] & 0x7F) << 8) | b[1]
    mant = int.from_bytes(b[2:10], "big")
    if exp == 0 and mant == 0:
        return 0.0
    f = mant * 2.0 ** (exp - 16383 - 63)
    return -f if sign else f


def _parse_wav(data: bytes) -> Optional[AudioInfo]:
    if data[8:12] != b"WAVE":
        return None
    channels = sr = bits = 0
    n_frames = 0
    i = 12
    n = len(data)
    while i + 8 <= n:
        cid = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        body = data[i + 8:i + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            channels = int.from_bytes(body[2:4], "little")
            sr = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
        elif cid == b"data":
            frame_bytes = max(1, channels * (bits // 8))
            n_frames = size // frame_bytes if frame_bytes else 0
        i += 8 + size + (size & 1)
    if channels and sr and bits:
        return AudioInfo("WAV", channels, sr, bits // 8, n_frames)
    return None


def _parse_aiff(data: bytes) -> Optional[AudioInfo]:
    if data[8:12] not in (b"AIFF", b"AIFC"):
        return None
    i = 12
    n = len(data)
    channels = sr = bits = n_frames = 0
    while i + 8 <= n:
        cid = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "big")
        body = data[i + 8:i + 8 + size]
        if cid == b"COMM" and len(body) >= 18:
            channels = int.from_bytes(body[0:2], "big")
            n_frames = int.from_bytes(body[2:6], "big")
            bits = int.from_bytes(body[6:8], "big")
            sr = int(_extended80(body[8:18]))
        i += 8 + size + (size & 1)
    if channels and sr and bits:
        return AudioInfo("AIFF", channels, sr, bits // 8, n_frames)
    return None


def parse_header(data: bytes) -> Optional[AudioInfo]:
    """Decode format/channels/rate/depth/frames from raw audio bytes (WAV/AIFF)."""
    if len(data) < 16:
        return None
    if data[:4] == b"RIFF":
        return _parse_wav(data)
    if data[:4] == b"FORM":
        return _parse_aiff(data)
    return None


# --------------------------------------------------------------------------- #
# PCM decoding -> mono float signal in [-1, 1]
# --------------------------------------------------------------------------- #
def _data_chunk(data: bytes, info: AudioInfo) -> bytes:
    if info.format == "WAV":
        i, n = 12, len(data)
        while i + 8 <= n:
            cid = data[i:i + 4]
            size = int.from_bytes(data[i + 4:i + 8], "little")
            if cid == b"data":
                return data[i + 8:i + 8 + size]
            i += 8 + size + (size & 1)
    else:  # AIFF SSND chunk (skip 8-byte offset/blocksize preamble)
        i, n = 12, len(data)
        while i + 8 <= n:
            cid = data[i:i + 4]
            size = int.from_bytes(data[i + 4:i + 8], "big")
            if cid == b"SSND":
                return data[i + 16:i + 8 + size]
            i += 8 + size + (size & 1)
    return b""


def _ints_from_bytes(raw: bytes, width: int, big_endian: bool) -> List[int]:
    if width == 1:  # 8-bit: WAV unsigned, AIFF signed
        if big_endian:
            return [b - (256 if b >= 128 else 0) for b in raw]  # AIFF signed
        return [b - 128 for b in raw]                            # WAV unsigned-centred
    if width == 2:
        a = array.array("h")
        a.frombytes(raw[: len(raw) // 2 * 2])
        if (sys.byteorder == "little") == big_endian:
            a.byteswap()
        return list(a)
    if width == 4:
        a = array.array("i")
        a.frombytes(raw[: len(raw) // 4 * 4])
        if (sys.byteorder == "little") == big_endian:
            a.byteswap()
        return list(a)
    if width == 3:  # 24-bit manual
        out: List[int] = []
        order = "big" if big_endian else "little"
        for k in range(0, len(raw) - 2, 3):
            out.append(int.from_bytes(raw[k:k + 3], order, signed=True))
        return out
    return []


def decode_pcm(data: bytes) -> Tuple[List[float], Optional[AudioInfo]]:
    """Decode WAV/AIFF PCM to a mono float signal in [-1, 1]."""
    info = parse_header(data)
    if info is None:
        return [], None
    raw = _data_chunk(data, info)
    if not raw:
        return [], info
    big = info.format == "AIFF"
    ints = _ints_from_bytes(raw, info.sample_width, big)
    if not ints:
        return [], info
    full_scale = float(1 << (info.bit_depth - 1))
    ch = max(1, info.channels)
    if ch == 1:
        mono = ints
    else:
        mono = [sum(ints[k:k + ch]) / ch for k in range(0, len(ints) - ch + 1, ch)]
    return [s / full_scale for s in mono], info


# --------------------------------------------------------------------------- #
# Signal features
# --------------------------------------------------------------------------- #
def rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def peak(samples: Sequence[float]) -> float:
    return max((abs(s) for s in samples), default=0.0)


def zero_crossing_rate(samples: Sequence[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = sum(1 for i in range(1, len(samples))
                    if (samples[i - 1] >= 0) != (samples[i] >= 0))
    return crossings / (len(samples) - 1)


def _frames(samples: Sequence[float], frame: int):
    for start in range(0, len(samples), frame):
        yield start, samples[start:start + frame]


def silence_segments(samples: Sequence[float], *, frame: int = 1024,
                     threshold: float = 0.02) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` sample ranges whose energy is below ``threshold``."""
    segs: List[Tuple[int, int]] = []
    cur_start: Optional[int] = None
    for start, chunk in _frames(samples, frame):
        if rms(chunk) < threshold:
            if cur_start is None:
                cur_start = start
        else:
            if cur_start is not None:
                segs.append((cur_start, start))
                cur_start = None
    if cur_start is not None:
        segs.append((cur_start, len(samples)))
    return segs


def silence_ratio(samples: Sequence[float], *, frame: int = 1024,
                  threshold: float = 0.02) -> float:
    if not samples:
        return 1.0
    silent = sum(e - s for s, e in silence_segments(samples, frame=frame, threshold=threshold))
    return silent / len(samples)


def audio_fingerprint(samples: Sequence[float], *, bands: int = 64) -> int:
    """A compact energy-envelope hash; near-duplicate clips share a small distance."""
    if not samples:
        return 0
    step = max(1, len(samples) // (bands + 1))
    env = [rms(samples[i:i + step]) for i in range(0, step * (bands + 1), step)][:bands + 1]
    while len(env) < bands + 1:
        env.append(0.0)
    h = 0
    for i in range(bands):
        h = (h << 1) | (1 if env[i] > env[i + 1] else 0)
    return h


# --------------------------------------------------------------------------- #
# Analysis result
# --------------------------------------------------------------------------- #
@dataclass
class AudioAnalysis:
    info: Optional[AudioInfo]
    rms: float = 0.0
    peak: float = 0.0
    zero_crossing_rate: float = 0.0
    silence_ratio: float = 0.0
    fingerprint: Optional[int] = None
    transcript: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"info": self.info.to_dict() if self.info else None,
                "rms": round(self.rms, 5), "peak": round(self.peak, 5),
                "zero_crossing_rate": round(self.zero_crossing_rate, 5),
                "silence_ratio": round(self.silence_ratio, 4),
                "fingerprint": hex_hash(self.fingerprint) if self.fingerprint is not None else None,
                "transcript": self.transcript, "note": self.note}


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class Audio:
    """Bridges real audio files to the pure-python perception algorithms."""

    def __init__(self, *, dup_threshold: int = 6, silence_threshold: float = 0.02) -> None:
        self.dup_threshold = dup_threshold
        self.silence_threshold = silence_threshold

    # ---- header / decode ---- #
    @staticmethod
    def inspect_bytes(data: bytes) -> Optional[AudioInfo]:
        return parse_header(data)

    def inspect(self, path: str) -> Optional[AudioInfo]:
        with open(path, "rb") as f:
            head = f.read(4096)
        info = parse_header(head)
        return info or self._soundfile_info(path)

    def _soundfile_info(self, path: str) -> Optional[AudioInfo]:
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            return None
        try:  # pragma: no cover - exercised only with the lib
            f = sf.info(path)
            width = {"PCM_16": 2, "PCM_24": 3, "PCM_32": 4}.get(f.subtype, 2)
            return AudioInfo(f.format, f.channels, f.samplerate, width, f.frames)
        except Exception:  # noqa: BLE001
            return None

    def samples(self, path: str) -> Tuple[List[float], Optional[AudioInfo]]:
        with open(path, "rb") as f:
            data = f.read()
        sig, info = decode_pcm(data)
        if sig:
            return sig, info
        return self._soundfile_samples(path)

    def _soundfile_samples(self, path: str) -> Tuple[List[float], Optional[AudioInfo]]:
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            return [], self.inspect(path)
        try:  # pragma: no cover - exercised only with the lib
            data, sr = sf.read(path, always_2d=True)
            mono = [float(sum(row) / len(row)) for row in data]
            info = AudioInfo("?", data.shape[1], sr, 2, data.shape[0])
            return mono, info
        except Exception:  # noqa: BLE001
            return [], self.inspect(path)

    # ---- transcription hook ---- #
    def transcribe(self, path: str) -> Tuple[Optional[str], str]:
        try:
            import whisper  # type: ignore
        except ImportError:
            return None, "whisper not installed; transcription unavailable"
        try:  # pragma: no cover - exercised only with the lib
            model = whisper.load_model("base")
            return model.transcribe(path).get("text", "").strip(), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"transcription failed: {exc}"

    # ---- full analysis ---- #
    def analyze(self, path: str, *, transcribe: bool = False) -> AudioAnalysis:
        sig, info = self.samples(path)
        if not sig:
            return AudioAnalysis(info=info,
                                 note="no PCM decoded (install soundfile for non-WAV/AIFF)")
        transcript = None
        note = ""
        if transcribe:
            transcript, note = self.transcribe(path)
        return AudioAnalysis(
            info=info, rms=rms(sig), peak=peak(sig),
            zero_crossing_rate=zero_crossing_rate(sig),
            silence_ratio=silence_ratio(sig, threshold=self.silence_threshold),
            fingerprint=audio_fingerprint(sig), transcript=transcript, note=note)

    # ---- deduplication ---- #
    def is_duplicate(self, fp_a: int, fp_b: int, *, threshold: Optional[int] = None) -> bool:
        thr = self.dup_threshold if threshold is None else threshold
        return hamming(fp_a, fp_b) <= thr


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import io
    import wave

    print("=" * 70)
    print("NYXARA audio self-test")
    print("=" * 70)

    def make_wav(freq: float, secs: float, sr: int = 8000, amp: float = 0.5,
                 silence_tail: float = 0.0) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            frames = bytearray()
            n = int(sr * secs)
            for i in range(n):
                v = int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr))
                frames += struct.pack("<h", v)
            for _ in range(int(sr * silence_tail)):
                frames += struct.pack("<h", 0)
            w.writeframes(bytes(frames))
        return buf.getvalue()

    tone = make_wav(440, 0.5, silence_tail=0.5)
    info = parse_header(tone)
    print(f"\nWAV header          : {info.to_dict()}")
    assert info.format == "WAV" and info.channels == 1 and info.sample_rate == 8000
    assert abs(info.duration - 1.0) < 0.01   # 0.5s tone + 0.5s silence

    sig, info2 = decode_pcm(tone)
    print(f"decoded samples     : {len(sig)} (rms={rms(sig):.3f}, peak={peak(sig):.3f})")
    assert len(sig) == info.n_frames
    assert 0.2 < rms(sig) < 0.45              # ~0.5 amplitude sine over half-silence

    # silence detection: the second half is silent
    sr = silence_ratio(sig, frame=512)
    print(f"silence ratio       : {sr:.2f}")
    assert 0.4 < sr < 0.6

    # zero-crossing rate is higher for a higher-frequency tone
    low = decode_pcm(make_wav(200, 0.3))[0]
    high = decode_pcm(make_wav(1500, 0.3))[0]
    print(f"ZCR low/high        : {zero_crossing_rate(low):.3f} / {zero_crossing_rate(high):.3f}")
    assert zero_crossing_rate(high) > zero_crossing_rate(low)

    # fingerprint dedup: same tone (slightly quieter) is a near-duplicate; noise is not
    a = audio_fingerprint(decode_pcm(make_wav(440, 0.5, amp=0.5))[0])
    b = audio_fingerprint(decode_pcm(make_wav(440, 0.5, amp=0.4))[0])
    c = audio_fingerprint(decode_pcm(make_wav(440, 0.5, silence_tail=1.5))[0])
    print(f"\nfingerprint a       : {hex_hash(a)}")
    print(f"hamming near        : {hamming(a, b)} (similar)")
    print(f"hamming different   : {hamming(a, c)}")
    aud = Audio()
    assert aud.is_duplicate(a, b)
    print("dedup verdict       : amplitude-scaled clip = duplicate ✓")

    print("\nALL SELF-TESTS PASSED ✓")
