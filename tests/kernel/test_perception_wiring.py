"""Tests for continuous real-time perception wired into the kernel.

The senses loop (:mod:`nyxara.senses.realtime`) is built on the core and its salient
events escalate through :meth:`NyxaraCore._perception_escalate` into full sovereign
AUTONOMOUS cycles — the live percept lands in the core's own binder via the ``media=``
intake, presence wakes, and a paused mind takes no action. Under pytest the loop is
built but never auto-started, so the suite stays hermetic and thread-free.
"""

from __future__ import annotations

import math
from typing import List, Optional

from nyxara.kernel.config import NyxaraSettings, Profile
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.senses.binding import Modality
from nyxara.senses.generate import encode_png
from nyxara.senses.live import LiveSensor
from nyxara.senses.realtime import RealtimePerception, _floats_to_wav

RATE = 16000


def checker_png(v: int, size: int = 64) -> bytes:
    rows = []
    for y in range(size):
        row = bytearray()
        for x in range(size):
            px = v if (x // 8 + y // 8) % 2 else 255 - v
            row.extend((px, px, px))
        rows.append(bytes(row))
    return encode_png(size, size, rows)


def tone_wav(amp: float, secs: float = 1.0) -> bytes:
    sig = [amp * math.sin(2 * math.pi * 440 * i / RATE) for i in range(int(secs * RATE))]
    return _floats_to_wav(sig, rate=RATE)


class StubSensor(LiveSensor):
    def __init__(self, frames: List[bytes] = (), clips: List[bytes] = ()) -> None:
        super().__init__()
        self.frames = list(frames)
        self.clips = list(clips)

    def available(self):
        return {"camera": False, "screen": bool(self.frames), "mic": bool(self.clips)}

    def capture_screen(self, monitor: int = 0):
        return (self.frames.pop(0), "screen: stub") if self.frames \
            else (None, "screen: script exhausted")

    def capture_microphone(self, seconds: Optional[float] = None,
                           rate: Optional[int] = None):
        return (self.clips.pop(0), "mic: stub") if self.clips \
            else (None, "mic: script exhausted")


def wire_stub(core: NyxaraCore, sensor: StubSensor) -> RealtimePerception:
    """A zero-delay loop escalating into the real core, exactly as the builder wires it."""
    rp = RealtimePerception(sensor, escalate=core._perception_escalate,
                            presence=core.presence, mind=core.mind,
                            gate=core._embodied_gate,
                            remember=core._perception_remember,
                            settings=object(), journal_path=None)
    rp.min_escalation_interval_s = 0.0
    for ch in rp.channels.values():
        ch.interval = ch.base_interval = 0.0
        ch.next_due = 0.0
    return rp


def drive(rp: RealtimePerception, n: int) -> None:
    for _ in range(n):
        rp.tick_once()
        for ch in rp.channels.values():
            ch.next_due = 0.0


# --------------------------------------------------------------------------- #
# Construction / config posture
# --------------------------------------------------------------------------- #
def test_core_builds_perception_but_never_autostarts_under_pytest():
    core = NyxaraCore()
    # under the TEST profile perception is disabled by config ⇒ honest None; under a
    # DEV-config run it is built but must NOT be running (pytest guard).
    if core.perception is not None:
        assert not core.perception.running


def test_profiles_gate_perception_correctly():
    assert NyxaraSettings().perception.enabled is True                       # max-power default
    assert NyxaraSettings.for_profile(Profile.TEST).perception.enabled is False  # hermetic
    m = NyxaraSettings.for_profile(Profile.MAX)
    assert m.perception.enabled is True
    assert m.perception.min_escalation_interval_s == 10.0


# --------------------------------------------------------------------------- #
# Escalation runs a REAL sovereign cycle and binds the percept
# --------------------------------------------------------------------------- #
def test_live_percept_escalates_into_a_real_autonomous_cycle():
    core = NyxaraCore()
    rp = wire_stub(core, StubSensor(frames=[checker_png(0), checker_png(0),
                                            checker_png(200)]))
    before = len(core.binder.frame.by_modality(Modality.IMAGE))
    drive(rp, 3)
    assert rp.escalations == 1
    # the live image percept landed in the CORE's binder through the media= intake
    assert len(core.binder.frame.by_modality(Modality.IMAGE)) > before
    # and the cycle left a real journal entry (a full turn ran, not a side channel)
    assert len(core.journal) > 0


def test_heard_speech_becomes_an_audio_percept_in_the_core():
    core = NyxaraCore()
    rp = wire_stub(core, StubSensor(clips=[tone_wav(0.5), tone_wav(0.0)]))
    before = len(core.binder.frame.by_modality(Modality.AUDIO))
    drive(rp, 1)
    assert rp.escalations == 1
    assert len(core.binder.frame.by_modality(Modality.AUDIO)) > before


def test_paused_mind_perceives_but_takes_no_action():
    core = NyxaraCore()
    core.pause()
    rp = wire_stub(core, StubSensor(frames=[checker_png(0), checker_png(0),
                                            checker_png(200)]))
    journal_before = len(core.journal)
    drive(rp, 3)
    # she still SAW the change (detection is passive) …
    assert any(e.kind == "visual_change" for e in rp.events)
    # … but the oversight gate held: no escalation, no cycle ran
    assert rp.escalations == 0
    assert len(core.journal) == journal_before


def test_engaged_core_is_never_interrupted():
    core = NyxaraCore()
    core._engaged = True     # a foreground Master turn is running
    rp = wire_stub(core, StubSensor(frames=[checker_png(0), checker_png(0),
                                            checker_png(200)]))
    drive(rp, 3)
    assert rp.escalations == 0 and rp.deferred_escalations == 1
    core._engaged = False    # turn over — the deferred percept now gets its cycle
    drive(rp, 1)
    assert rp.escalations == 1


# --------------------------------------------------------------------------- #
# Reporting surfaces
# --------------------------------------------------------------------------- #
def test_reports_carry_the_perception_surface():
    core = NyxaraCore()
    pr = core.power_report()
    assert "realtime_perception" in pr["faculties"]
    if core.perception is not None:
        assert "perception" in core.report()
        st = core.report()["perception"]
        assert {"running", "available", "channels"} <= set(st)
