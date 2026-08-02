"""Tests for F12 — hardware-level self-sensing (senses/hardware.py).

She must sense real telemetry (honest about what's exposed), throttle her compute budget under thermal
load, NOT fabricate a throttle she cannot sense, run event-driven idle, and keep the honest boundary
(low-power idle, not literally 0 W / not neuromorphic). No LLM anywhere.
"""

from __future__ import annotations

from nyxara.senses.hardware import ComputeGovernor, EventDrivenIdle, HardwareSense, Telemetry


def test_read_reports_availability_honestly():
    t = HardwareSense().read()
    avail = t.available()
    # every advertised metric is either a real number or honestly marked unavailable
    for key, present in avail.items():
        assert present == (getattr(t, key) is not None)


def test_governor_full_budget_when_cool_and_idle():
    assert ComputeGovernor().budget(Telemetry(cpu_percent=5, temperature_c=35)) == 1.0


def test_governor_throttles_toward_floor_when_hot():
    gov = ComputeGovernor(floor=0.2)
    assert gov.budget(Telemetry(temperature_c=90)) == 0.2           # at/above hot → floor
    warm = gov.budget(Telemetry(temperature_c=72))
    assert 0.2 < warm < 1.0                                          # in between → partial throttle


def test_governor_is_monotonic_in_temperature():
    gov = ComputeGovernor()
    b = [gov.budget(Telemetry(temperature_c=c)) for c in (40, 65, 75, 88)]
    assert all(b[i] >= b[i + 1] for i in range(len(b) - 1))          # hotter ⇒ never more budget


def test_governor_does_not_fabricate_a_throttle_it_cannot_sense():
    # no telemetry at all → full budget (honesty over theatre)
    assert ComputeGovernor().budget(Telemetry()) == 1.0


def test_event_driven_idle_wakes_on_salience():
    idle = EventDrivenIdle(threshold=0.5)
    assert idle.should_wake(0.9) is True
    assert idle.should_wake(0.1) is False
    assert 0.0 <= idle.duty_cycle() <= 1.0


def test_idle_posture_is_honest_about_power():
    posture = EventDrivenIdle().posture(False)
    assert "not zero" in posture and "neuromorphic" in posture     # the honest boundary, pinned
