"""Tests for nyxara.senses.audio."""

from __future__ import annotations

import io
import math
import struct
import wave

from nyxara.senses.audio import (Audio, AudioAnalysis, AudioInfo, audio_fingerprint,
                                 decode_pcm, parse_header, peak, rms, silence_ratio,
                                 silence_segments, zero_crossing_rate)
from nyxara.senses.vision import hamming


def make_wav(freq=440.0, secs=0.2, sr=8000, amp=0.5, channels=1, width=2,
             silence_tail=0.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        frames = bytearray()
        n = int(sr * secs)
        full = (1 << (width * 8 - 1)) - 1
        for i in range(n):
            v = int(amp * full * math.sin(2 * math.pi * freq * i / sr))
            for _ in range(channels):
                if width == 1:
                    frames += struct.pack("<B", (v + 128) & 0xFF)
                elif width == 2:
                    frames += struct.pack("<h", v)
                elif width == 4:
                    frames += struct.pack("<i", v)
        for _ in range(int(sr * silence_tail)):
            for _ in range(channels):
                frames += (b"\x80" if width == 1 else b"\x00" * width)
        w.writeframes(bytes(frames))
    return buf.getvalue()


# -------------------- header parsing -------------------- #
def test_parse_wav_header():
    info = parse_header(make_wav(secs=0.5, sr=8000))
    assert info.format == "WAV" and info.channels == 1
    assert info.sample_rate == 8000 and info.bit_depth == 16
    assert abs(info.duration - 0.5) < 0.01


def test_parse_wav_stereo():
    info = parse_header(make_wav(channels=2))
    assert info.channels == 2


def test_parse_wav_8bit():
    info = parse_header(make_wav(width=1))
    assert info.bit_depth == 8 and info.sample_width == 1


def test_parse_wav_32bit():
    info = parse_header(make_wav(width=4))
    assert info.bit_depth == 32


def test_parse_aiff_header():
    # hand-crafted minimal AIFF: FORM + COMM (44100 Hz, mono, 16-bit, 100 frames)
    sr_ext = b"\x40\x0e\xac\x44\x00\x00\x00\x00\x00\x00"  # 80-bit float for 44100
    comm = b"COMM" + (18).to_bytes(4, "big") + (1).to_bytes(2, "big") \
        + (100).to_bytes(4, "big") + (16).to_bytes(2, "big") + sr_ext
    body = b"AIFF" + comm
    aiff = b"FORM" + len(body).to_bytes(4, "big") + body
    info = parse_header(aiff)
    assert info.format == "AIFF" and info.channels == 1
    assert info.sample_rate == 44100 and info.n_frames == 100


def test_parse_unknown():
    assert parse_header(b"\x00" * 64) is None


def test_parse_too_short():
    assert parse_header(b"RIFF") is None


# -------------------- AudioInfo -------------------- #
def test_audio_info_duration_bitrate():
    i = AudioInfo("WAV", 2, 44100, 2, 88200)
    assert i.duration == 2.0
    assert i.bitrate == 44100 * 2 * 16


def test_audio_info_zero_rate():
    assert AudioInfo("WAV", 1, 0, 2, 100).duration == 0.0


def test_audio_info_to_dict():
    d = AudioInfo("WAV", 1, 8000, 2, 8000).to_dict()
    assert d["format"] == "WAV" and d["duration"] == 1.0


# -------------------- decoding -------------------- #
def test_decode_pcm_length():
    sig, info = decode_pcm(make_wav(secs=0.25, sr=8000))
    assert len(sig) == info.n_frames == 2000


def test_decode_pcm_range():
    sig, _ = decode_pcm(make_wav(amp=0.9))
    assert all(-1.0 <= s <= 1.0 for s in sig)
    assert peak(sig) > 0.8


def test_decode_stereo_mixes_to_mono():
    sig, info = decode_pcm(make_wav(channels=2, secs=0.2, sr=8000))
    assert info.channels == 2
    assert len(sig) == info.n_frames  # mono after mixing


def test_decode_8bit():
    sig, info = decode_pcm(make_wav(width=1, amp=0.5))
    assert info.bit_depth == 8 and peak(sig) > 0.3


def test_decode_unknown_returns_empty():
    sig, info = decode_pcm(b"\x00" * 64)
    assert sig == [] and info is None


# -------------------- features -------------------- #
def test_rms_silence_zero():
    assert rms([0.0] * 100) == 0.0


def test_rms_known():
    assert abs(rms([0.5, -0.5, 0.5, -0.5]) - 0.5) < 1e-9


def test_rms_empty():
    assert rms([]) == 0.0


def test_peak():
    assert peak([0.1, -0.9, 0.3]) == 0.9
    assert peak([]) == 0.0


def test_zero_crossing_rate():
    # alternating sign every sample -> ZCR = 1.0
    assert zero_crossing_rate([1.0, -1.0, 1.0, -1.0]) == 1.0
    # constant -> 0
    assert zero_crossing_rate([1.0, 1.0, 1.0]) == 0.0


def test_zero_crossing_short():
    assert zero_crossing_rate([1.0]) == 0.0


def test_zcr_higher_for_higher_freq():
    low, _ = decode_pcm(make_wav(150, 0.3))
    high, _ = decode_pcm(make_wav(2000, 0.3))
    assert zero_crossing_rate(high) > zero_crossing_rate(low)


# -------------------- silence -------------------- #
def test_silence_segments_detected():
    sig, _ = decode_pcm(make_wav(440, 0.3, silence_tail=0.3))
    segs = silence_segments(sig, frame=256)
    assert segs  # the trailing silence is found
    # the last silent segment reaches the end
    assert segs[-1][1] == len(sig)


def test_silence_ratio_half():
    sig, _ = decode_pcm(make_wav(440, 0.4, silence_tail=0.4))
    r = silence_ratio(sig, frame=256)
    assert 0.4 < r < 0.6


def test_silence_ratio_full_silence():
    assert silence_ratio([0.0] * 4096, frame=256) == 1.0


def test_silence_ratio_empty():
    assert silence_ratio([]) == 1.0


def test_silence_ratio_loud_low():
    sig, _ = decode_pcm(make_wav(440, 0.3, amp=0.8))
    assert silence_ratio(sig, frame=256) < 0.2


# -------------------- fingerprint -------------------- #
def test_fingerprint_deterministic():
    sig, _ = decode_pcm(make_wav(440, 0.4))
    assert audio_fingerprint(sig) == audio_fingerprint(sig)


def test_fingerprint_amplitude_invariant():
    a = audio_fingerprint(decode_pcm(make_wav(440, 0.4, amp=0.6))[0])
    b = audio_fingerprint(decode_pcm(make_wav(440, 0.4, amp=0.3))[0])
    assert hamming(a, b) <= 4  # scaling amplitude keeps the envelope shape


def test_fingerprint_distinguishes_structure():
    tone = audio_fingerprint(decode_pcm(make_wav(440, 0.5))[0])
    half = audio_fingerprint(decode_pcm(make_wav(440, 0.5, silence_tail=1.5))[0])
    assert hamming(tone, half) > 4


def test_fingerprint_empty():
    assert audio_fingerprint([]) == 0


# -------------------- Audio facade -------------------- #
def test_inspect_bytes():
    assert Audio.inspect_bytes(make_wav()).format == "WAV"


def test_inspect_file(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(sr=8000, secs=0.5))
    info = Audio().inspect(str(p))
    assert info.format == "WAV" and info.sample_rate == 8000


def test_samples_from_file(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(secs=0.2))
    sig, info = Audio().samples(str(p))
    assert len(sig) == info.n_frames


def test_analyze(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(440, 0.4, amp=0.6, silence_tail=0.4))
    res = Audio().analyze(str(p))
    assert isinstance(res, AudioAnalysis)
    assert res.rms > 0 and res.fingerprint is not None
    assert 0.3 < res.silence_ratio < 0.7


def test_analyze_to_dict(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(secs=0.2))
    d = Audio().analyze(str(p)).to_dict()
    assert "rms" in d and "fingerprint" in d and "info" in d


def test_is_duplicate():
    aud = Audio(dup_threshold=4)
    a = audio_fingerprint(decode_pcm(make_wav(440, 0.4, amp=0.6))[0])
    b = audio_fingerprint(decode_pcm(make_wav(440, 0.4, amp=0.3))[0])
    c = audio_fingerprint(decode_pcm(make_wav(440, 0.5, silence_tail=1.5))[0])
    assert aud.is_duplicate(a, b)
    assert not aud.is_duplicate(a, c)


def test_transcribe_graceful(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(make_wav(secs=0.2))
    text, note = Audio().transcribe(str(p))
    # whisper almost certainly absent here -> honest note, no crash
    if text is None:
        assert "whisper" in note or "transcription" in note
