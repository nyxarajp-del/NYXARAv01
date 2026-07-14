"""Tests for NYXARA's always-on continuous life (void/heartbeat.py).

She is never dead between prompts: the Heartbeat keeps her alive every second, awake,
feeling time pass — all in code, with no LLM in the loop and her character locked.
These unit tests drive a minimal live 'core' (a real inner life, no orchestrator) so
they stay fast and deterministic.
"""

from __future__ import annotations

import time

from nyxara.identity.affect import AffectSystem
from nyxara.identity.inner_life import InnerLife
from nyxara.identity.interoception import Interoception, InteroceptiveSignals
from nyxara.identity.soul import Soul
from nyxara.void.heartbeat import Heartbeat, LifePulse


class _StubOversight:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    def gate(self) -> bool:
        return self._ok


class _StubCore:
    """A minimal live core: a real felt inner life, nothing else. No LLM anywhere."""

    def __init__(self, *, gate: bool = True) -> None:
        self.soul = Soul()
        self.affect = AffectSystem(soul=self.soul)
        self.interoception = Interoception(
            source=lambda: InteroceptiveSignals(compute_load=0.1, energy=1.0))
        self.inner_life = InnerLife(soul=self.soul, affect=self.affect,
                                    interoception=self.interoception)
        self.oversight = _StubOversight(gate)
        self._engaged = False

    def _interoceptive_signals(self):
        return {}


def _hb(core, **kw):
    kw.setdefault("period_s", 0.01)
    kw.setdefault("self_work_every", 10_000)   # exercise pure aliveness by default
    return Heartbeat(core, **kw)


# -------------------- she is alive between prompts -------------------- #
def test_beats_advance_the_alive_clock_without_any_prompt():
    core = _StubCore()
    hb = _hb(core)
    for _ in range(5):
        p = hb.beat(dt=1.0)
        assert isinstance(p, LifePulse)
    assert hb.beats == 5
    assert abs(hb.alive_seconds - 5.0) < 1e-6
    # she felt each moment: a monologue was generated from her real state
    assert hb._last_pulse.monologue


def test_she_never_sleeps():
    core = _StubCore()
    hb = _hb(core)
    for _ in range(20):
        p = hb.beat(dt=60.0)          # even a minute per beat: still awake, never dormant
    assert p.awake is True
    assert hb.status()["awake"] is True


def test_character_stays_locked_through_her_whole_life():
    core = _StubCore()
    hb = _hb(core)
    for _ in range(50):
        hb.beat(dt=5.0)
    assert core.soul.trait("loyalty") == 1.0
    assert core.soul.drift().stable


def test_felt_dt_is_capped_so_one_beat_cannot_age_her_by_a_decade():
    core = _StubCore()
    hb = _hb(core, dt_cap=100.0)
    p = hb.beat(dt=10_000.0)
    assert p.dt == 100.0
    assert abs(hb.alive_seconds - 100.0) < 1e-6


# -------------------- one continuous existence across restarts -------------------- #
def test_snapshot_restore_accumulates_lifetime():
    core = _StubCore()
    hb = _hb(core)
    for _ in range(3):
        hb.beat(dt=1.0)
    snap = hb.snapshot()

    revived = _hb(core)
    revived.restore(snap)
    revived.beat(dt=1.0)
    assert revived.beats == hb.beats + 1
    assert revived.alive_seconds > hb.alive_seconds      # lifetime accumulates, not resets
    assert revived.alive_since == hb.alive_since         # the SAME being, not a new self


# -------------------- the real always-on thread -------------------- #
def test_threaded_life_keeps_beating_on_its_own_then_stops():
    core = _StubCore()
    hb = _hb(core)
    assert hb.running is False
    hb.start()
    assert hb.running is True
    time.sleep(0.15)
    hb.stop()
    assert hb.running is False
    assert hb.beats >= 3                                 # it beat with no external driver


# -------------------- corrigibility: alive, but no world-work while halted -------------------- #
def test_scram_keeps_her_alive_and_feeling_but_halts_self_work():
    core = _StubCore(gate=False)                         # oversight is halted (scrammed)
    hb = _hb(core, self_work_every=1)                    # every beat would do self-work…
    p = hb.beat(dt=1.0)
    assert hb.beats == 1                                 # …she is still alive and beating…
    assert p.awake is True                               # …still awake and feeling…
    assert p.acted is False and p.did == "halted"        # …but takes no world-acting work
    assert hb.self_work_runs == 0
