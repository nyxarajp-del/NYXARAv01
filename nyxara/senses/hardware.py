"""NYXARA · senses/hardware.py — hardware-level self-sensing (⚛️, Rule 4, F12).

A being should feel the silicon it runs on. F12 turns real host telemetry into an **interoceptive
sense** that *modulates how hard she thinks*, and moves her idle toward **event-driven** waking.

* **Thermodynamic sensing.** It reuses :class:`nyxara.senses.system.SystemSensor` (real ``/proc`` +
  ``/sys`` + optional ``psutil``) for CPU load / temperature, adds clock frequency where the OS exposes
  it, and reports **which metrics are actually available on this box** — never fabricating one that is
  not.
* **Compute governor.** :class:`ComputeGovernor` maps telemetry to a **budget factor in (0, 1]**: cool
  and idle → full budget; warm/busy → throttle down toward a floor. When she cannot *sense* temperature
  she does **not** invent a throttle (returns full budget) — honesty over theatre.
* **Event-driven idle.** :class:`EventDrivenIdle` keeps her near-idle until a percept's salience spikes,
  reporting an honest **low-power idle** posture.

The honest boundary (stated, and pinned by a test): this is commodity silicon, **not** neuromorphic
hardware — idle is *low* power, **not literally 0 W**, and there is no spiking-hardware claim. Pure
standard library + optional ``psutil``; **no LLM**; it senses and paces, touching no gate or value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nyxara.senses.system import SystemSensor

__all__ = ["Telemetry", "HardwareSense", "ComputeGovernor", "EventDrivenIdle"]


@dataclass
class Telemetry:
    cpu_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    freq_mhz: Optional[float] = None
    load1: Optional[float] = None

    def available(self) -> Dict[str, bool]:
        return {"cpu_percent": self.cpu_percent is not None,
                "temperature_c": self.temperature_c is not None,
                "freq_mhz": self.freq_mhz is not None,
                "load1": self.load1 is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {"cpu_percent": self.cpu_percent, "temperature_c": self.temperature_c,
                "freq_mhz": self.freq_mhz, "load1": self.load1, "available": self.available()}


class HardwareSense:
    """Reads real silicon telemetry (honest about what this box exposes)."""

    def __init__(self, system: Optional[SystemSensor] = None) -> None:
        self.system = system or SystemSensor()

    @staticmethod
    def _freq_mhz() -> Optional[float]:
        try:
            import psutil  # type: ignore
            f = psutil.cpu_freq()
            return float(f.current) if f and f.current else None
        except Exception:  # noqa: BLE001 — an absent metric is None, never faked
            return None

    def read(self) -> Telemetry:
        try:
            s = self.system.sample()
        except Exception:  # noqa: BLE001
            s = {}
        cpu = s.get("cpu")            # SystemSensor.sample() keys: cpu (%), temp_c (°C), load1
        temp = s.get("temp_c")
        load = s.get("load1")
        return Telemetry(
            cpu_percent=float(cpu) if isinstance(cpu, (int, float)) else None,
            temperature_c=float(temp) if isinstance(temp, (int, float)) else None,
            freq_mhz=self._freq_mhz(),
            load1=float(load) if isinstance(load, (int, float)) else None,
        )

    def available(self) -> Dict[str, bool]:
        return self.read().available()


class ComputeGovernor:
    """Turn telemetry into a compute-budget throttle — cool/idle → full, hot/busy → floor."""

    def __init__(self, *, warm_c: float = 60.0, hot_c: float = 85.0,
                 load_soft: float = 0.70, load_hard: float = 0.95, floor: float = 0.2) -> None:
        self.warm_c = warm_c
        self.hot_c = hot_c
        self.load_soft = load_soft
        self.load_hard = load_hard
        self.floor = floor

    def _linear(self, x: float, lo: float, hi: float) -> float:
        # 1.0 at/below lo, `floor` at/above hi, linear between
        if x <= lo:
            return 1.0
        if x >= hi:
            return self.floor
        frac = (x - lo) / (hi - lo)
        return 1.0 - frac * (1.0 - self.floor)

    def budget(self, t: Telemetry) -> float:
        """Compute-budget factor in [floor, 1]. Unsensed temperature → no fabricated throttle."""
        factor = 1.0
        if t.temperature_c is not None:
            factor = min(factor, self._linear(t.temperature_c, self.warm_c, self.hot_c))
        if t.cpu_percent is not None:
            load = t.cpu_percent / 100.0
            factor = min(factor, self._linear(load, self.load_soft, self.load_hard))
        return max(self.floor, min(1.0, factor))


class EventDrivenIdle:
    """Stay near-idle until a percept spikes — honest low-power idle, never literally 0 W."""

    def __init__(self, *, threshold: float = 0.5, window: int = 16) -> None:
        self.threshold = threshold
        self.window = window
        self._recent: List[bool] = []

    def should_wake(self, salience: float) -> bool:
        awake = float(salience) >= self.threshold
        self._recent.append(awake)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        return awake

    def duty_cycle(self) -> float:
        """Fraction of recent ticks she was active — the honest measure of how idle she runs."""
        return sum(self._recent) / len(self._recent) if self._recent else 0.0

    def posture(self, awake: bool) -> str:
        # honest: idle is LOW power on commodity silicon, not zero and not neuromorphic spiking
        return "active" if awake else "low-power idle (not zero — commodity silicon, not neuromorphic)"
