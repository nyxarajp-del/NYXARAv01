"""NYXARA · nyx5/snn.py — the spiking network (⚡, pillars 1-2 composed).

Composes the four leaf faculties into a working brain-in-miniature: leaky integrate-and-fire neurons
(:mod:`nyx5.neuron`), a live rewiring graph (:mod:`nyx5.topology`), STDP local learning
(:mod:`nyx5.synapse`), all driven by the event queue (:mod:`nyx5.event_queue`).

One call to :meth:`stimulate` injects input spikes, drains the resulting cascade in logical time, and —
crucially — the network **learns from the same turn it is thinking through**: every pre→post delivery
carries an STDP weight update, and periodically the topology prunes dead synapses and grows new ones
between co-active neurons. There is **no train/infer split** and **no backpropagation**: the only signal
is local spike timing. Bounded by an event budget so a turn can never run away.

Pure standard library, deterministic under a seed. This is a *simulation* on commodity silicon.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from nyxara.nyx5.event_queue import EventDrivenScheduler
from nyxara.nyx5.neuron import LIFNeuron, SpikeEvent
from nyxara.nyx5.synapse import STDPRule
from nyxara.nyx5.topology import LiveTopology

__all__ = ["SpikeResponse", "SpikingNetwork"]

InputSpike = Union[int, Tuple[int, float], Tuple[int, float, float]]


@dataclass
class SpikeResponse:
    """The result of one :meth:`SpikingNetwork.stimulate` — an honest, bounded read of the cascade."""

    raster: List[Tuple[float, int]] = field(default_factory=list)  # (time, neuron_id) output spikes
    n_spikes: int = 0
    mean_rate: float = 0.0            # spikes per neuron over the window
    energy_proxy: float = 0.0        # spike count — the honest "how much did it cost" signal
    duration: float = 0.0
    events: int = 0                  # scheduler deliveries this call
    grew: int = 0
    pruned: int = 0

    def firing_neurons(self) -> List[int]:
        return sorted({nid for _, nid in self.raster})

    def to_dict(self) -> Dict[str, Any]:
        return {"n_spikes": self.n_spikes, "mean_rate": round(self.mean_rate, 6),
                "energy_proxy": self.energy_proxy, "duration": round(self.duration, 4),
                "events": self.events, "grew": self.grew, "pruned": self.pruned,
                "firing_neurons": self.firing_neurons()}


class SpikingNetwork:
    """An event-driven spiking neural network that learns online by STDP and structural plasticity."""

    def __init__(self, n_neurons: int = 256, *, tau: float = 20.0, threshold: float = 1.0,
                 refractory: float = 2.0, stdp: Optional[STDPRule] = None,
                 max_synapses: int = 8192, prune_threshold: float = 0.05,
                 max_events: int = 20000, coactive_window: float = 5.0,
                 initial_connectivity: float = 0.02, seed: int = 42) -> None:
        self.n_neurons = int(n_neurons)
        self.max_events = int(max_events)
        self.coactive_window = float(coactive_window)
        self._rng = random.Random(seed)
        self.neurons: List[LIFNeuron] = [
            LIFNeuron(neuron_id=i, tau=tau, threshold=threshold, refractory=refractory)
            for i in range(self.n_neurons)
        ]
        self.stdp = stdp or STDPRule()
        self.topology = LiveTopology(max_synapses=max_synapses, prune_threshold=prune_threshold)
        self.scheduler = EventDrivenScheduler()
        self._clock = 0.0                         # persistent logical time across stimulations
        self._recent_fire: Deque[Tuple[float, int]] = deque(maxlen=256)
        self.total_spikes = 0
        self.ticks = 0
        self._seed_topology(initial_connectivity)

    # ---- construction ---- #
    def _seed_topology(self, density: float) -> None:
        """Seed a sparse random graph so the very first stimulation already propagates."""
        target = int(density * self.n_neurons * self.n_neurons)
        target = min(target, self.topology.max_synapses)
        for _ in range(target):
            pre = self._rng.randrange(self.n_neurons)
            post = self._rng.randrange(self.n_neurons)
            if pre != post:
                self.topology.connect(pre, post, weight=self._rng.uniform(0.3, 0.7),
                                      delay=self._rng.uniform(1.0, 3.0))

    # ---- the cognitive tick ---- #
    def stimulate(self, inputs: Iterable[InputSpike], *, duration: float = 50.0) -> SpikeResponse:
        """Inject ``inputs``, drain the cascade, learn by STDP, and rewire. Returns a bounded read."""
        t0 = self._clock
        incoming = self._build_incoming()
        raster: List[Tuple[float, int]] = []
        coactive: List[Tuple[int, int]] = []

        for spec in inputs:
            nid, at, charge = self._normalise(spec)
            self.scheduler.schedule(SpikeEvent(time=t0 + at, neuron_id=nid, weight=charge))

        def handle(ev: SpikeEvent) -> None:
            neuron = self.neurons[ev.neuron_id]
            fired = neuron.integrate(ev.time, ev.weight)
            if not fired:
                return
            raster.append((ev.time, ev.neuron_id))
            self.total_spikes += 1
            # potentiate synapses that just helped cause this spike (pre-before-post)
            for syn in incoming.get(ev.neuron_id, ()):  # type: ignore[arg-type]
                self.stdp.on_post_spike(syn, ev.time)
            # depress anti-causal synapses + propagate the spike downstream
            for syn in list(self.topology.outgoing(ev.neuron_id)):
                self.stdp.on_pre_spike(syn, ev.time)
                self.scheduler.schedule(SpikeEvent(time=ev.time + syn.delay,
                                                   neuron_id=syn.post, weight=syn.weight))
            # remember for structural co-activation growth
            for pt, pid in self._recent_fire:
                if 0.0 < (ev.time - pt) <= self.coactive_window:
                    coactive.append((pid, ev.neuron_id))
            self._recent_fire.append((ev.time, ev.neuron_id))

        delivered = self.scheduler.run_until(t0 + duration, max_events=self.max_events,
                                             handler=handle)
        self._clock = max(self._clock + duration, self.scheduler.now)
        self.ticks += 1

        grew = self.topology.grow(coactive) if coactive else 0
        pruned = self.topology.prune(self._clock)

        n = len(raster)
        return SpikeResponse(raster=raster, n_spikes=n,
                             mean_rate=n / self.n_neurons if self.n_neurons else 0.0,
                             energy_proxy=float(n), duration=duration, events=delivered,
                             grew=grew, pruned=pruned)

    def _build_incoming(self) -> Dict[int, List[Any]]:
        incoming: Dict[int, List[Any]] = {}
        for syn in self.topology.all_synapses():
            incoming.setdefault(syn.post, []).append(syn)
        return incoming

    def _normalise(self, spec: InputSpike) -> Tuple[int, float, float]:
        if isinstance(spec, int):
            return spec % self.n_neurons, 0.0, 1.0
        if len(spec) == 2:
            nid, at = spec  # type: ignore[misc]
            return int(nid) % self.n_neurons, float(at), 1.0
        nid, at, charge = spec  # type: ignore[misc]
        return int(nid) % self.n_neurons, float(at), float(charge)

    # ---- observability / persistence ---- #
    def stats(self) -> Dict[str, Any]:
        d = self.topology.snapshot_stats()
        d.update({"neurons": self.n_neurons, "total_spikes": self.total_spikes, "ticks": self.ticks,
                  "clock": round(self._clock, 3)})
        return d

    def to_dict(self) -> Dict[str, Any]:
        return {"n_neurons": self.n_neurons, "clock": self._clock, "ticks": self.ticks,
                "total_spikes": self.total_spikes, "stdp": self.stdp.to_dict(),
                "topology": self.topology.to_dict(),
                "neurons": [nrn.to_dict() for nrn in self.neurons]}

    def load_dict(self, d: Dict[str, Any]) -> None:
        self._clock = float(d.get("clock", 0.0))
        self.ticks = int(d.get("ticks", 0))
        self.total_spikes = int(d.get("total_spikes", 0))
        if "topology" in d:
            self.topology.load_dict(d["topology"])
        stdp = d.get("stdp")
        if stdp:
            self.stdp = STDPRule(**{k: stdp[k] for k in stdp if hasattr(STDPRule, k)})
