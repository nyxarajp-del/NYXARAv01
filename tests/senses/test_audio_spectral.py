"""Tests for nyxara.senses.audio sound-event classification (impulse/alarm/sustained).

All signals are generated in-test with stdlib math — real waveforms, no devices.
"""

from __future__ import annotations

import math
import random
from typing import List

from nyxara.senses.audio import classify_sound

SR = 8000


def burst(amp: float = 0.9) -> List[float]:
    """1 s clip with one 60 ms decaying bang in the middle."""
    sig = [0.0] * (SR // 4)
    sig += [amp * math.exp(-i / 120) * math.sin(2 * math.pi * 250 * i / SR)
            for i in range(SR // 16)]
    return sig + [0.0] * (SR - len(sig))


def hum(secs: float = 2.0) -> List[float]:
    return [0.4 * math.sin(2 * math.pi * 440 * i / SR) for i in range(int(secs * SR))]


def beeps(secs: float = 3.0, period_s: float = 0.5) -> List[float]:
    half = int(SR * period_s / 2)
    return [0.6 * math.sin(2 * math.pi * 1000 * i / SR)
            if (i // half) % 2 == 0 else 0.0 for i in range(int(secs * SR))]


def test_bang_is_impulse():
    prof = classify_sound(burst(), SR)
    assert prof.label == "impulse"
    assert prof.peak_to_mean >= 3.0 and prof.active_s <= 0.35


def test_steady_tone_is_sustained_not_alarm():
    prof = classify_sound(hum(), SR)
    assert prof.label == "sustained"
    assert prof.beep_hz is None          # a flat envelope cannot beep


def test_periodic_beeping_is_alarm_with_measured_rate():
    prof = classify_sound(beeps(), SR)
    assert prof.label == "alarm"
    assert prof.beep_hz is not None and abs(prof.beep_hz - 2.0) < 0.5


def test_single_on_off_is_not_an_alarm():
    sig = hum(1.0) + [0.0] * SR          # one tone then silence: no repetition
    assert classify_sound(sig, SR).label != "alarm"


def test_irregular_bursts_are_not_an_alarm():
    rng = random.Random(7)
    sig: List[float] = []
    for _ in range(5):                   # bursts at chaotic intervals
        sig += [0.0] * int(SR * rng.uniform(0.1, 0.9))
        sig += [0.6 * math.sin(2 * math.pi * 900 * i / SR) for i in range(SR // 10)]
    assert classify_sound(sig, SR).label != "alarm"


def test_silence_yields_honest_residual_class():
    prof = classify_sound([0.0] * SR, SR)
    assert prof.label == "sound" and prof.beep_hz is None


def test_band_energies_sum_to_one_when_present():
    prof = classify_sound(beeps(), SR)
    total = prof.band_low + prof.band_mid + prof.band_high
    assert abs(total - 1.0) < 1e-6 or total == 0.0
