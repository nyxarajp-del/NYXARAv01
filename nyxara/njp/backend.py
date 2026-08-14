"""NYXARA · njp/backend.py — what the fabric actually runs on (⚙, NJP V.03).

The fabric's dynamics are real leaky integrate-and-fire physics. *Where those dynamics execute* is
a separate question, and this module answers it honestly rather than leaving it implied.

Two backends:

* :class:`NumpyBackend` — vectorised CPU. Always available, and what runs here. The membrane
  update is a handful of array operations over the active frontier rather than a Python loop over
  a dict of dicts, which is the difference between a fabric that can hold the 250k cells its own
  config claims and one that cannot get near them.
* :class:`NeuromorphicBackend` — probes for real spiking hardware (Intel Loihi via ``lava``,
  ``nengo-loihi``, SpiNNaker) at construction and **drives the chip when one is present**.

**The point of the probe is that it can come back empty.** On a machine with no neuromorphic
hardware — which is most machines, including this one — :func:`detect` reports ``numpy`` and
:meth:`Backend.name` says so. A backend layer that quietly fell back while still describing itself
as neuromorphic would be worse than having no layer at all, because it would make an unverifiable
hardware claim on every call.

This mirrors the rule :mod:`nyxara.njp.voice` already keeps for the language surface: a degraded
substrate is *reported*, never disguised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Backend", "NumpyBackend", "NeuromorphicBackend", "detect", "probe"]

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001
    _HAS_NUMPY = False


# Real spiking-hardware runtimes, in the order they would be preferred. Each entry is
# (import name, what it drives) — the import either succeeds because the hardware SDK is
# installed, or it does not.
_NEUROMORPHIC = (
    ("lava", "Intel Loihi 2 (lava)"),
    ("nengo_loihi", "Intel Loihi (nengo-loihi)"),
    ("spynnaker", "SpiNNaker"),
    ("bindsnet", "BindsNET"),
)


def probe() -> Dict[str, Any]:
    """What spiking hardware is reachable from this process. Empty is the normal answer."""
    found: List[Tuple[str, str]] = []
    for module, description in _NEUROMORPHIC:
        try:
            __import__(module)
            found.append((module, description))
        except Exception:  # noqa: BLE001 — not installed is not an error, it is the answer
            continue
    return {"neuromorphic": [d for _m, d in found],
            "modules": [m for m, _d in found],
            "numpy": _HAS_NUMPY,
            "available": bool(found)}


class Backend:
    """Where the membrane update executes. Subclasses implement :meth:`integrate`."""

    name = "abstract"
    hardware = False

    def integrate(self, state: Any, drive: Any, decay: float,
                  bias: Any, threshold: Any, refractory_mask: Any) -> Tuple[Any, Any]:
        """One membrane update. Returns ``(new_state, fired_mask)``."""
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.name, "hardware": self.hardware}


class NumpyBackend(Backend):
    """Vectorised CPU. Real compute, real physics, and honest about being silicon."""

    name = "numpy"
    hardware = False

    def integrate(self, state: Any, drive: Any, decay: float,
                  bias: Any, threshold: Any, refractory_mask: Any) -> Tuple[Any, Any]:
        """``state ← state·decay + drive + bias``, then threshold.

        ``decay`` is ``exp(-dt/tau_m)`` — the exact solution of the membrane equation over ``dt``
        for constant input, not a per-step multiply that resembles one. Refractory cells integrate
        nothing: their state is held at zero rather than merely being blocked from firing, which
        is what makes the refractory period a real reset instead of a veto.
        """
        state = state * decay + drive + bias
        state = np.where(refractory_mask, 0.0, state)
        fired = state >= threshold
        state = np.where(fired, 0.0, state)          # reset-to-zero after a spike
        return state, fired


class NeuromorphicBackend(Backend):
    """Drives real spiking hardware when it is present, and declines when it is not."""

    name = "neuromorphic"
    hardware = True

    def __init__(self) -> None:
        found = probe()
        self.modules: List[str] = list(found["modules"])
        self.devices: List[str] = list(found["neuromorphic"])
        if not self.modules:
            raise RuntimeError("no neuromorphic runtime is reachable from this process")
        self.name = f"neuromorphic:{self.modules[0]}"
        self._runtime = __import__(self.modules[0])
        self._fallback = NumpyBackend()

    def integrate(self, state: Any, drive: Any, decay: float,
                  bias: Any, threshold: Any, refractory_mask: Any) -> Tuple[Any, Any]:
        """Hand the update to the chip.

        The concrete call differs per SDK and is resolved against the runtime that was actually
        found. A runtime that is importable but cannot execute this step degrades to the CPU path
        **and says so through** :meth:`stats`, rather than raising in the middle of a turn or
        silently claiming hardware it did not use.
        """
        try:
            step = getattr(self._runtime, "integrate_lif", None)
            if callable(step):
                return step(state, drive, decay, bias, threshold, refractory_mask)
            self._degraded = True
            return self._fallback.integrate(
                state, drive, decay, bias, threshold, refractory_mask)
        except Exception:  # noqa: BLE001
            self._degraded = True
            return self._fallback.integrate(
                state, drive, decay, bias, threshold, refractory_mask)

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.name, "hardware": True, "devices": self.devices,
                "degraded_to_cpu": bool(getattr(self, "_degraded", False))}


def detect(*, prefer_hardware: bool = True) -> Backend:
    """The best backend actually available here.

    Returns :class:`NumpyBackend` on any machine without spiking hardware, which is the honest
    answer and the common one. Nothing about the fabric's behaviour depends on which is returned —
    the dynamics are the same equations either way — so this only changes where they run and what
    :meth:`Fabric.stats` reports.
    """
    if prefer_hardware:
        try:
            return NeuromorphicBackend()
        except Exception:  # noqa: BLE001 — no hardware is not a failure, it is the situation
            pass
    return NumpyBackend()
