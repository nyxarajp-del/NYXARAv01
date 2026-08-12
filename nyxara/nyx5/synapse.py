"""NYXARA · nyx5/synapse.py — the synapse and its STDP learning rule (⚡, pillars 1-2).

A synapse carries charge from a pre-neuron to a post-neuron with a weight and a conduction delay.
Its weight is **not** trained by backpropagation — it changes by a *local* rule that only ever sees
the relative timing of the two neurons it connects: **Spike-Timing-Dependent Plasticity (STDP)**.

    pre fires just BEFORE post  ⇒  the pre helped cause the post  ⇒  potentiate  Δw = +A⁺·exp(-Δt/τ⁺)
    pre fires just AFTER  post  ⇒  the pre could not have caused it ⇒  depress    Δw = -A⁻·exp(-Δt/τ⁻)

This is Hebb's "fire together, wire together" made causal and quantitative. Weights are clamped to
``[w_min, w_max]``; learning is online, unsupervised, and needs no target labels — the network learns
from the mere order of its own spikes. Pure standard library, deterministic.

Depends on nothing else in NYX-5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

__all__ = ["Synapse", "STDPRule"]


@dataclass
class Synapse:
    """A directed connection ``pre -> post`` with a weight, a conduction delay, and spike memory.

    ``last_pre_t`` / ``last_post_t`` remember the most recent spike times of the endpoints so the
    STDP rule can compute Δt locally.
    """

    pre: int
    post: int
    weight: float = 0.5
    delay: float = 1.0
    last_pre_t: float = -1e9
    last_post_t: float = -1e9

    @property
    def stale_since(self) -> float:
        return max(self.last_pre_t, self.last_post_t)

    def to_dict(self) -> Dict[str, Any]:
        return {"pre": self.pre, "post": self.post, "weight": round(self.weight, 6),
                "delay": self.delay}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Synapse":
        return cls(pre=int(d["pre"]), post=int(d["post"]), weight=float(d.get("weight", 0.5)),
                   delay=float(d.get("delay", 1.0)))


@dataclass
class STDPRule:
    """Exponential Spike-Timing-Dependent Plasticity, clamped and local.

    ``a_plus`` / ``a_minus`` are the potentiation / depression amplitudes; ``tau_plus`` / ``tau_minus``
    the timing windows (ms). ``w_min`` / ``w_max`` clamp the weight. The rule is applied at spike
    events, never as a batch gradient.
    """

    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.0
    w_max: float = 1.0

    def _clamp(self, w: float) -> float:
        return max(self.w_min, min(self.w_max, w))

    def on_post_spike(self, syn: Synapse, t_post: float) -> float:
        """The post-neuron just fired at ``t_post``: potentiate by how recently the pre fired."""
        syn.last_post_t = t_post
        dt = t_post - syn.last_pre_t
        if dt >= 0.0 and dt < 1e8:
            syn.weight = self._clamp(syn.weight + self.a_plus * math.exp(-dt / self.tau_plus))
        return syn.weight

    def on_pre_spike(self, syn: Synapse, t_pre: float) -> float:
        """The pre-neuron just fired at ``t_pre``: depress by how recently the post fired (pre-after-
        post is anti-causal), then record the pre time for the next post spike."""
        dt = t_pre - syn.last_post_t
        if dt >= 0.0 and dt < 1e8:
            syn.weight = self._clamp(syn.weight - self.a_minus * math.exp(-dt / self.tau_minus))
        syn.last_pre_t = t_pre
        return syn.weight

    def to_dict(self) -> Dict[str, Any]:
        return {"a_plus": self.a_plus, "a_minus": self.a_minus, "tau_plus": self.tau_plus,
                "tau_minus": self.tau_minus, "w_min": self.w_min, "w_max": self.w_max}
