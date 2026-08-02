"""NYXARA · senses/thermodynamic.py — interoceptive thermal/compute homeostasis (Part E).

A living organism *feels* its body and adjusts. NYXARA does the honest, achievable version of that:
she reads the silicon (CPU load, temperature, clock, load average — via the existing ``HardwareSense``
over stdlib ``/proc``/``/sys`` and optional psutil) and adapts **her own software** to it — before a heavy
pass she checks the thermal/compute budget and, when hot or loaded, defers work, shrinks reasoning depth,
and lowers her process priority; when cool she expands. "Adjust thought-speed to the physical compute
budget," implemented as ``budget()`` → software throttling.

⚠️ Honest boundary (stated, not faked): a userspace process **cannot** set CPU voltage/frequency, undervolt/
overclock, or restructure cache — that needs root + kernel drivers + vendor MSR access and can damage
hardware. This module does the real, safe knobs: **read-only** telemetry, **process priority** (``os.nice``
/ ``psutil``), and **cooperative** deferral. CPU-frequency-governor / fan control is exposed only through a
host-only, permission-gated :class:`PrivilegedTelemetryAdapter` that is a **read-only no-op** without root
(as in this container). No voltage/cache writes, no kernel modules, ever.

Reuses ``senses/hardware.py`` (``HardwareSense``, ``Telemetry``, ``ComputeGovernor``).
"""
from __future__ import annotations

import os
from typing import Any, Optional

from nyxara.senses.hardware import ComputeGovernor, HardwareSense, Telemetry

__all__ = ["ThermodynamicMonitor", "PrivilegedTelemetryAdapter"]


class ThermodynamicMonitor:
    """Feel the hardware; adapt NYXARA's own software to the live compute budget (Part E)."""

    def __init__(self, *, sense: Any = None, governor: Optional[ComputeGovernor] = None) -> None:
        self.sense = sense if sense is not None else HardwareSense()
        self.governor = governor if governor is not None else ComputeGovernor()
        self._raised_nice: int = 0            # how much we've lowered our own priority (for restore)

    # ---- sensing ---- #
    def read(self) -> Telemetry:
        return self.sense.read()

    def budget(self, telemetry: Optional[Telemetry] = None) -> float:
        """Live compute-budget factor in [floor, 1] — 1.0 = plenty, low = hot/loaded, throttle."""
        t = telemetry if telemetry is not None else self.read()
        return float(self.governor.budget(t))

    # ---- software self-regulation ---- #
    def should_defer(self, *, min_budget: float = 0.4,
                     telemetry: Optional[Telemetry] = None) -> bool:
        """True when the machine is too hot/loaded to start a heavy pass now — defer it."""
        return self.budget(telemetry) < float(min_budget)

    def scale(self, amount: float, *, floor: int = 1,
              telemetry: Optional[Telemetry] = None) -> int:
        """Scale a work amount (reasoning depth, agent count, pool size) by the live budget."""
        return max(int(floor), int(round(float(amount) * self.budget(telemetry))))

    def lower_priority(self, delta: int = 5) -> bool:
        """Lower our OWN process priority (be a good citizen under load). Reversible via psutil.

        ``os.nice`` can only *increase* niceness portably; we track how far so :meth:`restore_priority`
        can undo it when psutil is available (plain os.nice cannot lower niceness without privilege).
        """
        delta = max(0, int(delta))
        try:
            proc = self._psutil_proc()
            if proc is not None:
                cur = proc.nice()
                proc.nice(cur + delta)
            else:
                os.nice(delta)
            self._raised_nice += delta
            return True
        except Exception:  # noqa: BLE001 — priority control is best-effort
            return False

    def restore_priority(self) -> bool:
        """Undo any priority lowering (needs psutil; a plain-os.nice environment cannot lower niceness)."""
        if self._raised_nice <= 0:
            return True
        try:
            proc = self._psutil_proc()
            if proc is None:
                return False
            proc.nice(proc.nice() - self._raised_nice)
            self._raised_nice = 0
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _psutil_proc() -> Any:
        try:
            import psutil
            return psutil.Process()
        except Exception:  # noqa: BLE001 — psutil is an optional enhancer
            return None


class PrivilegedTelemetryAdapter:
    """Host-only, permission-gated deep-telemetry/cpufreq bonus — a READ-ONLY no-op without privilege.

    On a real Linux host running as root it *could* deepen sensing (perf/eBPF) and adjust the documented,
    reversible cpufreq **governor** via ``/sys`` — the one legitimate OS knob. Everywhere else (this
    container included) it does nothing. It NEVER writes MSRs / voltage / cache and NEVER loads a kernel
    module. This is the honest ceiling of "hardware breathing" from software.
    """

    def available(self) -> bool:
        """True only with the privilege to touch the (read-mostly) OS knobs — false in a sandbox."""
        try:
            if os.name != "posix" or not hasattr(os, "geteuid"):
                return False
            if os.geteuid() != 0:
                return False
            return os.path.isdir("/sys/devices/system/cpu/cpu0/cpufreq")
        except Exception:  # noqa: BLE001
            return False

    def read_cpufreq_governor(self) -> Optional[str]:
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:  # noqa: BLE001 — unreadable → honestly None
            return None

    def set_cpufreq_governor(self, name: str) -> bool:
        """Set the cpufreq governor (reversible OS policy) — no-op unless privileged. Never voltage."""
        if not self.available():
            return False
        ok = False
        try:
            import glob
            for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(name)
                    ok = True
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            return False
        return ok
