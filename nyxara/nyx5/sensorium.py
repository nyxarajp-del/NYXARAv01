"""NYXARA · nyx5/sensorium.py — the polymorphic sensorium (👁️, pillar 5).

A human perceives only through the senses evolution handed them. NYX-5 is not so limited: any raw
numeric or byte stream that is *actually available on the host* can become a sense. This module turns an
arbitrary channel — host CPU/temperature/memory vitals, loop telemetry, any caller-supplied numeric
stream — into **spike trains** the spiking substrate can feel, and it *discovers* new channels as they
appear (bounded by a cap).

Honest boundary: it perceives only streams that are genuinely wired in. It does not conjure Wi-Fi or
packet senses out of nothing — feed it a real data source and it will encode it; feed it nothing and it
is a silent no-op. Encoding is **rate coding**: a channel's normalised value sets how many, and which,
neurons it excites, so louder signals spike more.

Pure standard library. Depends on nothing else in NYX-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

__all__ = ["SensedSpikes", "PolymorphicSensorium"]


@dataclass
class SensedSpikes:
    """The spikes a set of channel readings produced, plus which channels were seen."""

    spikes: List[Tuple[int, float, float]] = field(default_factory=list)  # (neuron_id, t, charge)
    channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_spikes": len(self.spikes), "channels": self.channels}


class PolymorphicSensorium:
    """Encode arbitrary named numeric channels into spikes, auto-registering new channels (capped)."""

    def __init__(self, n_neurons: int, *, max_channels: int = 32, neurons_per_channel: int = 4) -> None:
        self.n_neurons = int(n_neurons)
        self.max_channels = int(max_channels)
        self.neurons_per_channel = int(neurons_per_channel)
        self._channels: Dict[str, int] = {}          # channel name -> base neuron index
        self._range: Dict[str, Tuple[float, float]] = {}  # running min/max per channel (normalisation)

    def channels(self) -> List[str]:
        return list(self._channels)

    def _register(self, name: str) -> bool:
        if name in self._channels:
            return True
        if len(self._channels) >= self.max_channels:
            return False                              # cap reached — new senses politely refused
        base = (len(self._channels) * self.neurons_per_channel) % self.n_neurons
        self._channels[name] = base
        self._range[name] = (float("inf"), float("-inf"))
        return True

    def _normalise(self, name: str, value: float) -> float:
        lo, hi = self._range[name]
        lo, hi = min(lo, value), max(hi, value)
        self._range[name] = (lo, hi)
        if hi <= lo:
            return 0.5                                # not enough spread yet — mid intensity
        return (value - lo) / (hi - lo)

    def sense(self, readings: Mapping[str, float], *, t: float = 0.0) -> SensedSpikes:
        """Turn a batch of ``{channel: value}`` readings into spikes (rate-coded, normalised, bounded)."""
        out = SensedSpikes()
        for name, value in readings.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue                              # non-numeric channel: skip, never crash
            if not self._register(name):
                continue
            out.channels.append(name)
            intensity = self._normalise(name, v)
            base = self._channels[name]
            fire = max(1, round(intensity * self.neurons_per_channel))
            for k in range(fire):
                nid = (base + k) % self.n_neurons
                out.spikes.append((nid, t, 0.5 + 0.5 * intensity))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"n_neurons": self.n_neurons, "max_channels": self.max_channels,
                "neurons_per_channel": self.neurons_per_channel,
                "channels": self._channels, "range": {k: list(v) for k, v in self._range.items()}}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self._channels = {k: int(v) for k, v in d.get("channels", {}).items()}
        self._range = {k: (float(a), float(b)) for k, (a, b) in d.get("range", {}).items()}
