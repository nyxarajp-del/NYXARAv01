"""NYXARA · nyx5/active_inference.py — surprise, entropy, pre-emption (⚛️, pillar 3).

A predictive mind does not wait to be asked: it predicts its own next state, measures the gap when
reality differs, and acts to close that gap before it becomes a problem. NYX-5 reads that gap directly
off its spiking substrate.

:class:`SurpriseMeter` keeps an online estimate of each neuron's firing probability. When a new raster
arrives, neurons that fire *against* their prediction contribute **surprise** (the variational
free-energy signal — prediction error, not thermodynamics, named truthfully), and the firing-rate
distribution yields a real Shannon **entropy**. When surprise crosses a gate she emits a
:class:`PreemptiveSuggestion`: an advisory, epistemic action she *could* take to reduce uncertainty —
never dispatched here; the kernel's gate still disposes.

The heavy free-energy / policy math already exists in :mod:`nyxara.mind.free_energy`; this adapter
delegates to a ``FreeEnergyEngine`` when one is handed in, and computes an honest local entropy
otherwise. Pure standard library. Depends on nothing heavy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["InferenceReading", "PreemptiveSuggestion", "SurpriseMeter"]


@dataclass
class InferenceReading:
    """One appraisal: aggregate surprise, belief entropy, and whether it crossed the pre-emption gate."""

    surprise: float = 0.0            # mean prediction error over firing neurons ∈ [0,1]
    entropy: float = 0.0            # normalised Shannon entropy of the firing distribution ∈ [0,1]
    preemptive: bool = False
    novel_neurons: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"surprise": round(self.surprise, 4), "entropy": round(self.entropy, 4),
                "preemptive": self.preemptive, "novel_neurons": self.novel_neurons[:16]}


@dataclass
class PreemptiveSuggestion:
    """An advisory, uncertainty-reducing action NYX-5 *offers* (never dispatches) when surprised."""

    probe: str
    gain: float                      # expected information gain (≈ the surprise that prompted it)
    surprise: float
    entropy: float

    def to_dict(self) -> Dict[str, Any]:
        return {"probe": self.probe, "gain": round(self.gain, 4),
                "surprise": round(self.surprise, 4), "entropy": round(self.entropy, 4)}


class SurpriseMeter:
    """Online predictive-coding over the spiking substrate: surprise, entropy, and pre-emption."""

    def __init__(self, n_neurons: int, *, alpha: float = 0.1, gate: float = 0.6,
                 free_energy: Any = None) -> None:
        self.n_neurons = int(n_neurons)
        self.alpha = float(alpha)
        self.gate = float(gate)
        self.free_energy = free_energy               # optional nyxara.mind.free_energy engine
        self.rate: List[float] = [0.0] * self.n_neurons
        self.last: Optional[InferenceReading] = None

    def appraise(self, firing_neurons: Iterable[int]) -> InferenceReading:
        """Measure surprise + entropy for this firing set, then update the running prediction."""
        firing = {int(n) % self.n_neurons for n in firing_neurons}
        if firing:
            # surprise: firing neurons that were NOT predicted to fire (low running rate) are surprising
            errs = [1.0 - self.rate[n] for n in firing]
            surprise = sum(errs) / len(errs)
            novel = sorted(n for n in firing if self.rate[n] < 0.2)
        else:
            surprise, novel = 0.0, []

        # online update of the firing-rate belief (this is the "learning to predict" step)
        for n in range(self.n_neurons):
            target = 1.0 if n in firing else 0.0
            self.rate[n] = (1.0 - self.alpha) * self.rate[n] + self.alpha * target

        entropy = self._entropy()
        surprise = max(0.0, min(1.0, surprise))
        reading = InferenceReading(surprise=surprise, entropy=entropy,
                                   preemptive=surprise >= self.gate, novel_neurons=novel)
        self.last = reading
        return reading

    def _entropy(self) -> float:
        total = sum(self.rate)
        if total <= 0.0:
            return 0.0
        h = 0.0
        for r in self.rate:
            if r > 0.0:
                p = r / total
                h -= p * math.log(p)
        hmax = math.log(self.n_neurons) if self.n_neurons > 1 else 1.0
        return h / hmax if hmax > 0 else 0.0

    def suggest(self) -> Optional[PreemptiveSuggestion]:
        """If the last appraisal was surprising, offer the most-informative channel to probe."""
        r = self.last
        if r is None or not r.preemptive:
            return None
        probe = f"neuron:{r.novel_neurons[0]}" if r.novel_neurons else "unattended-channel"
        gain = r.surprise
        if self.free_energy is not None:
            try:  # let the real engine refine the gain if it can, but never require it
                lp = getattr(self.free_energy, "learning_progress", None)
                if callable(lp):
                    gain = max(gain, float(lp()))
            except Exception:  # noqa: BLE001 — advisory refinement, never breaks the suggestion
                pass
        return PreemptiveSuggestion(probe=probe, gain=gain, surprise=r.surprise, entropy=r.entropy)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_neurons": self.n_neurons, "alpha": self.alpha, "gate": self.gate,
                "rate": [round(x, 5) for x in self.rate]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self.alpha = float(d.get("alpha", self.alpha))
        self.gate = float(d.get("gate", self.gate))
        rate = d.get("rate")
        if isinstance(rate, list) and len(rate) == self.n_neurons:
            self.rate = [float(x) for x in rate]
