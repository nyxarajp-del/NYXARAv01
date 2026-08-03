"""NYXARA · mind/active_inference_loop.py — the continuous inference tick (⚛️, Stage B).

The free-energy machinery (``mind/free_energy.py`` + ``mind/active_inference.py``) is real, but until
now it ran **only inside a turn** as advisory annotation — nothing drove it on the background cadence.
A living mind does not wait for a prompt to notice that its world drifted: it **predicts every tick,
measures the gap, and acts to close it before the gap becomes a problem**.

This module is the missing continuous step. On each autonomic/heartbeat tick it:

* reads whatever numeric channels are available (interoception host vitals, loop telemetry — her own
  internal state) and folds them into a :class:`~nyxara.mind.active_inference.GenerativeModel`, so the
  **prediction error is her surprise** (the free-energy signal);
* computes a real **Shannon entropy** over the belief state — the honest reading of "measure the
  entropy of internal/world states" (this is *active-inference* variational free energy, **not** literal
  thermodynamics; named truthfully);
* when surprise or entropy crosses a threshold, names the **most-informative channel to probe** (an
  epistemic, uncertainty-reducing action) so the loop can act **pre-emptively** — through the same
  sovereign gauntlet as every other autonomic act, never ungoverned.

Pure standard library; **no LLM**; degrades to a silent no-op when no channel is readable. It changes
only *what she attends to and when she probes*, never what she values or what she is allowed to do.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from nyxara.mind.active_inference import GenerativeModel

__all__ = ["InferenceReading", "ContinuousActiveInference"]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class InferenceReading:
    """One continuous-inference tick: per-channel surprise + aggregate free energy + belief entropy."""

    surprise: Dict[str, float] = field(default_factory=dict)
    free_energy: float = 0.0            # mean surprise across channels — the quantity she reduces
    entropy: float = 0.0               # normalised Shannon entropy of the surprise distribution [0,1]
    preemptive: bool = False           # did surprise/entropy cross the threshold this tick?
    probe: Optional[str] = None        # the most-informative channel to attend to next (if preemptive)
    probe_gain: float = 0.0            # expected information gain of probing ``probe``
    channels: int = 0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "free_energy": round(self.free_energy, 4),
            "entropy": round(self.entropy, 4),
            "preemptive": self.preemptive,
            "probe": self.probe,
            "probe_gain": round(self.probe_gain, 4),
            "channels": self.channels,
            "surprise": {k: round(v, 4) for k, v in self.surprise.items()},
        }


class ContinuousActiveInference:
    """A continuously-ticking generative model that measures surprise/entropy and flags pre-emption.

    Bounded and honest: it holds one online predictor per channel, never grows without limit (the
    channel set is fixed by the caller), and produces a *suggestion* (which channel to probe) — the
    kernel's gauntlet still disposes whether any action actually runs.
    """

    def __init__(self, *, surprise_threshold: float = 0.55, entropy_threshold: float = 0.85,
                 free_energy_engine: Any = None) -> None:
        self.model = GenerativeModel()
        self.surprise_threshold = _clamp01(surprise_threshold)
        self.entropy_threshold = _clamp01(entropy_threshold)
        # optional: a shared FreeEnergyEngine, used only for a richer pre-emptive plan when present.
        self.free_energy_engine = free_energy_engine
        self.ticks = 0
        self.preemptions = 0
        self.last: Optional[InferenceReading] = None

    # --------------------------------------------------------------------- #
    def observe(self, channels: Dict[str, float]) -> InferenceReading:
        """Fold one tick of observations, returning the surprise/entropy reading.

        ``channels`` maps a stable channel name → its current numeric value. Absent or non-numeric
        values are skipped. With zero usable channels this is a no-op reading (free_energy 0)."""
        per: Dict[str, float] = {}
        for key, val in channels.items():
            try:
                per[key] = self.model.observe(key, float(val))
            except (TypeError, ValueError):
                continue
        fe = self.model.free_energy()
        entropy = self._belief_entropy(per)
        probe, gain = self._most_informative(per.keys())
        # pre-empt when the world surprised her (high mean error) OR her belief is diffuse (high entropy)
        preemptive = bool(per) and (fe >= self.surprise_threshold or entropy >= self.entropy_threshold)
        self.ticks += 1
        if preemptive:
            self.preemptions += 1
        reading = InferenceReading(
            surprise=per, free_energy=fe, entropy=entropy, preemptive=preemptive,
            probe=probe if preemptive else None, probe_gain=gain if preemptive else 0.0,
            channels=len(per),
        )
        self.last = reading
        return reading

    # --------------------------------------------------------------------- #
    @staticmethod
    def _belief_entropy(per: Dict[str, float]) -> float:
        """Normalised Shannon entropy of the surprise distribution, in ``[0, 1]``.

        A single dominant surprise → low entropy (she knows *where* the problem is); surprise spread
        evenly across every channel → entropy near 1 (diffuse uncertainty, nothing localised). This is
        the honest, computable "entropy of her internal state" — not a thermodynamic quantity."""
        vals = [max(0.0, float(v)) for v in per.values()]
        total = sum(vals)
        n = len(vals)
        if n <= 1 or total <= 0.0:
            return 0.0
        h = 0.0
        for v in vals:
            if v <= 0.0:
                continue
            p = v / total
            h -= p * math.log(p)
        return _clamp01(h / math.log(n))

    def _most_informative(self, keys: Sequence[str]) -> tuple[Optional[str], float]:
        keys = list(keys)
        if not keys:
            return None, 0.0
        gains = {k: self.model.expected_information_gain(k) for k in keys}
        best = max(keys, key=lambda k: gains[k])
        return best, gains[best]

    # --------------------------------------------------------------------- #
    def report(self) -> Dict[str, Any]:
        """Observable status — safe to surface on the console / API / loop status."""
        d: Dict[str, Any] = {
            "ticks": self.ticks,
            "preemptions": self.preemptions,
            "surprise_threshold": self.surprise_threshold,
            "entropy_threshold": self.entropy_threshold,
        }
        if self.last is not None:
            d["last"] = self.last.to_dict()
        return d
