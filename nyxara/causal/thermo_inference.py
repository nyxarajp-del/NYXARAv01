"""NYXARA · causal/thermo_inference.py — Thermodynamic Active-Inference Heartbeat (CAUSAL · 3).

*The Master's ask:* a continuous, non-equilibrium loop that measures NYXARA's own **surprise /
entropy** every beat and, when something spikes (a network hiccup, a code bug, unexpected data),
**pre-emptively heals** her internal state *before* the next prompt — never dead between turns.

What that means here, honestly. :class:`ThermodynamicMonitor` consumes one scalar (or vector)
**surprise** signal per beat — a prediction error the rest of the mind already produces (free
energy, health, prediction-engine residuals). It maintains an online mean/variance (Welford) and
an EWMA baseline, and computes three real quantities each beat:

    * **surprise** ``S`` — how far this observation fell from expectation;
    * **entropy** ``H`` — Shannon entropy of the recent-surprise distribution (how disordered);
    * **free energy** ``F = S − T·H`` — a Helmholtz analogue; ``T`` is an internal "temperature".

When the *standardised* surprise ``z = (S − μ)/σ`` exceeds a threshold, the beat is a **spike** and
the monitor asks for a pre-emptive heal (a registered callback, or a wired ``core``'s self-repair /
health pass). Between spikes it slowly cools ``T`` toward equilibrium.

The honest boundary. This is a real online-statistics controller over a signal NYXARA already
emits; it genuinely runs every heartbeat and genuinely fires repair before a prompt. It is **not**
microsecond hardware thermodynamics, and "pre-cognitive healing" means *acting on a rising
prediction-error trend*, not seeing the future.

Bridges: fed by :mod:`nyxara.mind.free_energy`, :mod:`nyxara.mind.active_inference_loop`; heals via
:mod:`nyxara.growth.self_repair` / :mod:`nyxara.agency.health`; driven from
:mod:`nyxara.void.heartbeat`.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Sequence

__all__ = ["ThermoReading", "ThermodynamicMonitor"]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class ThermoReading:
    """One beat of the thermodynamic loop."""

    beat: int
    at: float
    surprise: float          # raw prediction error this beat
    z: float                 # standardised surprise (σ-units from the running mean)
    entropy: float           # Shannon entropy of the recent-surprise distribution (bits)
    temperature: float       # internal "temperature" (drives exploration/dissipation)
    free_energy: float       # F = surprise − T·entropy
    spike: bool              # did this beat exceed the surprise threshold?
    healed: bool             # was a pre-emptive heal fired this beat?
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"beat": self.beat, "at": round(self.at, 3),
                "surprise": round(self.surprise, 5), "z": round(self.z, 3),
                "entropy": round(self.entropy, 4), "temperature": round(self.temperature, 4),
                "free_energy": round(self.free_energy, 5), "spike": self.spike,
                "healed": self.healed, "note": self.note}


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
class ThermodynamicMonitor:
    """An online non-equilibrium surprise/entropy controller with pre-emptive healing."""

    def __init__(self, *, window: int = 64, spike_z: float = 2.5, warmup: int = 8,
                 temperature: float = 1.0, cool: float = 0.98, heat: float = 0.5,
                 heal: Optional[Callable[[ThermoReading], bool]] = None) -> None:
        if window < 4:
            raise ValueError("window must be >= 4")
        self.window = int(window)
        self.spike_z = float(spike_z)
        self.warmup = int(warmup)
        self.temperature = float(temperature)
        self._t0 = float(temperature)
        self.cool = float(cool)          # multiplicative cooling toward equilibrium each calm beat
        self.heat = float(heat)          # additive heating on a spike (more exploration/repair)
        self._heal = heal
        self._recent: Deque[float] = deque(maxlen=self.window)
        self._beats = 0
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0                   # Welford's aggregate for variance
        self._spikes = 0
        self._heals = 0
        self._last: Optional[ThermoReading] = None

    # -- online statistics -------------------------------------------------- #
    def _update_stats(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (x - self._mean)

    @property
    def _std(self) -> float:
        if self._n < 2:
            return 0.0
        return math.sqrt(max(0.0, self._m2 / (self._n - 1)))

    def _entropy(self) -> float:
        """Shannon entropy (bits) of the recent-surprise magnitudes, histogrammed."""
        if len(self._recent) < 2:
            return 0.0
        vals = [abs(v) for v in self._recent]
        total = sum(vals)
        if total <= 0.0:
            return 0.0
        h = 0.0
        for v in vals:
            p = v / total
            if p > 0.0:
                h -= p * math.log2(p)
        return h

    # -- one beat ----------------------------------------------------------- #
    def observe(self, surprise: Any, *, note: str = "") -> ThermoReading:
        """Feed one surprise signal (scalar or vector) and get the beat's reading."""
        s = _scalarise(surprise)
        self._beats += 1
        mean = self._mean
        # floor the std by the data scale so a perfectly-flat baseline that suddenly jumps still
        # registers as surprising (zero-variance would otherwise divide-guard every spike to z=0).
        eff_std = max(self._std, 1e-9 + 1e-6 * abs(mean))
        z = (s - mean) / eff_std
        self._recent.append(s)
        self._update_stats(s)
        entropy = self._entropy()
        spike = self._beats > self.warmup and z >= self.spike_z
        # temperature dynamics: heat on a spike, cool toward baseline otherwise
        if spike:
            self.temperature += self.heat
        else:
            self.temperature = self._t0 + (self.temperature - self._t0) * self.cool
        free_energy = s - self.temperature * entropy
        healed = False
        if spike:
            self._spikes += 1
            healed = self._fire_heal(
                ThermoReading(self._beats, time.time(), s, z, entropy, self.temperature,
                              free_energy, True, False, note))
            if healed:
                self._heals += 1
        reading = ThermoReading(beat=self._beats, at=time.time(), surprise=s, z=z,
                                entropy=entropy, temperature=self.temperature,
                                free_energy=free_energy, spike=spike, healed=healed, note=note)
        self._last = reading
        return reading

    def beat(self, core: Any = None, *, signal: Any = None, note: str = "") -> ThermoReading:
        """Heartbeat entry point: pull a surprise signal from ``core`` (or use ``signal``)."""
        s = signal if signal is not None else self._read_core_surprise(core)
        return self.observe(s if s is not None else 0.0, note=note)

    # -- healing ------------------------------------------------------------ #
    def _fire_heal(self, reading: ThermoReading) -> bool:
        if self._heal is None:
            return False
        try:
            return bool(self._heal(reading))
        except Exception:  # noqa: BLE001 — a failing healer must never crash the heartbeat
            return False

    def set_healer(self, heal: Callable[[ThermoReading], bool]) -> None:
        self._heal = heal

    @staticmethod
    def _read_core_surprise(core: Any) -> Optional[float]:
        """Best-effort: read a live surprise scalar from whatever the core exposes."""
        if core is None:
            return None
        for path in ("free_energy", "predictive", "interoception", "health"):
            obj = getattr(core, path, None)
            if obj is None:
                continue
            for attr in ("surprise", "last_surprise", "prediction_error", "variational_free_energy"):
                val = getattr(obj, attr, None)
                if callable(val):
                    try:
                        val = val()
                    except Exception:  # noqa: BLE001
                        val = None
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    # -- introspection ------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        return {"beats": self._beats, "spikes": self._spikes, "heals": self._heals,
                "mean_surprise": round(self._mean, 5), "std_surprise": round(self._std, 5),
                "temperature": round(self.temperature, 4),
                "last": None if self._last is None else self._last.to_dict()}


def _scalarise(x: Any) -> float:
    """Reduce a scalar or vector surprise to a single magnitude (L2 norm)."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, Sequence):
        return math.sqrt(sum(float(v) * float(v) for v in x))
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
