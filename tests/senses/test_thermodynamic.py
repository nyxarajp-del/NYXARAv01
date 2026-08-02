"""Tests for interoceptive thermal homeostasis — senses/thermodynamic.py (Part E)."""
from __future__ import annotations

from nyxara.senses.hardware import Telemetry
from nyxara.senses.thermodynamic import PrivilegedTelemetryAdapter, ThermodynamicMonitor


class _FakeSense:
    def __init__(self, t: Telemetry) -> None:
        self._t = t

    def read(self) -> Telemetry:
        return self._t


def _monitor(cpu=10.0, temp=40.0, load=0.1) -> ThermodynamicMonitor:
    return ThermodynamicMonitor(sense=_FakeSense(Telemetry(cpu_percent=cpu, temperature_c=temp, load1=load)))


def test_cool_idle_has_full_budget_and_expands():
    m = _monitor(cpu=5.0, temp=35.0, load=0.05)
    assert m.budget() > 0.9
    assert m.should_defer() is False
    assert m.scale(10) >= 9          # near-full work allowed


def test_hot_loaded_throttles_and_defers():
    m = _monitor(cpu=99.0, temp=95.0, load=4.0)
    b = m.budget()
    assert b < 0.4
    assert m.should_defer(min_budget=0.4) is True
    assert m.scale(10, floor=1) < 10  # heavy work shrinks under load


def test_scale_respects_floor():
    m = _monitor(cpu=100.0, temp=100.0, load=8.0)
    assert m.scale(10, floor=2) >= 2


def test_priority_control_is_best_effort_and_safe():
    m = _monitor()
    # lowering our own priority must never raise; restore is best-effort
    assert m.lower_priority(1) in (True, False)
    m.restore_priority()


def test_privileged_adapter_is_noop_in_sandbox():
    a = PrivilegedTelemetryAdapter()
    # in an unprivileged container: not available, and set is a refused no-op (never touches hardware)
    assert a.available() is False
    assert a.set_cpufreq_governor("performance") is False


def test_no_voltage_or_fan_write_api_exists():
    # honesty guard: the module exposes NO voltage/fan/MSR write surface
    import nyxara.senses.thermodynamic as mod
    for banned in ("set_voltage", "undervolt", "overclock", "set_fan", "write_msr"):
        assert not hasattr(mod.ThermodynamicMonitor, banned)
        assert not hasattr(mod.PrivilegedTelemetryAdapter, banned)
