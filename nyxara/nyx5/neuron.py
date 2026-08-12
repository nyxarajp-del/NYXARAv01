"""NYXARA · nyx5/neuron.py — the leaky integrate-and-fire neuron (⚡, pillar 1).

The atom of NYX-5's spiking substrate. A biological neuron integrates incoming charge on its
membrane, leaks it away continuously, and **fires a spike** the instant the membrane crosses a
threshold — then falls silent for a short refractory window. This module is that, honestly:

    v(t) = v(t₀)·exp(-(t-t₀)/τ) + Σ incoming charge

leak is applied lazily (only when the neuron is touched), so nothing is clock-stepped — the whole
network advances on **events**, not on a dense tensor sweep. This is a *simulation* on commodity
silicon; there are no literal GHz spikes and **no backpropagation** — a neuron never sees a global
error signal, only the charge that actually arrives.

Pure standard library, deterministic. Depends on nothing else in NYX-5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

__all__ = ["NeuronState", "SpikeEvent", "LIFNeuron"]


class NeuronState(str, Enum):
    """A neuron is either at rest, actively integrating charge, or in its refractory dead-time."""

    RESTING = "resting"
    INTEGRATING = "integrating"
    REFRACTORY = "refractory"


@dataclass(order=False)
class SpikeEvent:
    """One spike travelling through the network: *when* it is delivered, *who* fired it, *how much*
    charge it carries (the source synapse's weight). Logical time (ms), never wall-clock."""

    time: float
    neuron_id: int
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"time": round(self.time, 6), "neuron_id": self.neuron_id,
                "weight": round(self.weight, 6)}


@dataclass
class LIFNeuron:
    """A leaky integrate-and-fire neuron with an explicit refractory window.

    ``tau`` is the membrane leak time-constant (ms); larger ⇒ charge lingers longer. ``threshold``
    is the fire level; on firing the membrane resets to ``v_reset`` and the neuron is refractory
    until ``refractory`` ms have passed. All dynamics are computed lazily from the last-touched
    time, so an untouched neuron costs nothing.
    """

    neuron_id: int
    tau: float = 20.0
    threshold: float = 1.0
    v_reset: float = 0.0
    refractory: float = 2.0
    v: float = 0.0
    last_update: float = 0.0
    last_spike: float = -1e9
    spike_count: int = 0

    def _leak_to(self, t: float) -> None:
        """Advance the membrane to time ``t`` by applying exponential leak (lazy, event-driven)."""
        dt = t - self.last_update
        if dt > 0.0 and self.v != 0.0:
            self.v *= math.exp(-dt / self.tau)
        self.last_update = t

    def state(self, t: float) -> NeuronState:
        if t - self.last_spike < self.refractory:
            return NeuronState.REFRACTORY
        return NeuronState.INTEGRATING if self.v > 0.0 else NeuronState.RESTING

    def integrate(self, t: float, charge: float) -> bool:
        """Deliver ``charge`` at time ``t``. Returns True iff the neuron fires this instant.

        During the refractory window incoming charge is ignored (the membrane is clamped at reset),
        exactly as a real neuron cannot be re-excited immediately after a spike.
        """
        if t - self.last_spike < self.refractory:
            # refractory: hold at reset, absorb nothing
            self.last_update = t
            self.v = self.v_reset
            return False
        self._leak_to(t)
        self.v += charge
        if self.v >= self.threshold:
            self.v = self.v_reset
            self.last_spike = t
            self.spike_count += 1
            return True
        return False

    def reset(self) -> None:
        self.v = self.v_reset
        self.last_update = 0.0
        self.last_spike = -1e9
        self.spike_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"neuron_id": self.neuron_id, "tau": self.tau, "threshold": self.threshold,
                "v_reset": self.v_reset, "refractory": self.refractory}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LIFNeuron":
        return cls(neuron_id=int(d["neuron_id"]), tau=float(d.get("tau", 20.0)),
                   threshold=float(d.get("threshold", 1.0)), v_reset=float(d.get("v_reset", 0.0)),
                   refractory=float(d.get("refractory", 2.0)))
