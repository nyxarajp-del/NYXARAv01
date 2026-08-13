"""NYXARA · nyx5/nyx5_brain.py — the NYX-5 brain facade (⚡, the single object the kernel holds).

`Nyx5Brain` is the one object `NyxaraCore` builds and holds as ``self.nyx5``. It owns the spiking
substrate, the hyperdimensional long-term memory, the surprise/active-inference meter, and — gated by
config — the advanced faculties (chrono, sensorium, immune, intent, concept-collapse, axiom-forge,
negentropy, conduit, holographic swarm, mesh, ontogenesis, retarget, phagocytosis, and the
conversational/persona/advisory layers). Every method is **fail-soft**: on any error it degrades to a null
result, never breaking a turn.

Honest, as everywhere in NYX-5: this is a spiking-network *simulation* on commodity silicon (not
neuromorphic hardware, no backpropagation). The brain *proposes* colour and, when installed in the
reason-seat, candidates — but the kernel's gate always disposes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.nyx5.active_inference import PreemptiveSuggestion, SurpriseMeter
from nyxara.nyx5.anticipation import ProactiveAnticipator
from nyxara.nyx5.autopoiesis import AutopoieticRewriter
from nyxara.nyx5.axiom_forge import SubAxiomaticEngine
from nyxara.nyx5.chrono import ChronoDilator
from nyxara.nyx5.concept_space import ConceptCollapser
from nyxara.nyx5.conduit import SymbioticConduit
from nyxara.nyx5.continuum import NarrativeContinuum
from nyxara.nyx5.dialectic import SovereignDialectic
from nyxara.nyx5.hd_memory import HDMemory
from nyxara.nyx5.holo_swarm import HolographicShard
from nyxara.nyx5.immune import ImmuneGuillotine
from nyxara.nyx5.intent import IntentSymbiote
from nyxara.nyx5.mesh import EntangledMesh
from nyxara.nyx5.mirroring import EpistemicMirror
from nyxara.nyx5.negentropy import NegentropyMaintainer
from nyxara.nyx5.omni_forge import OmniForge
from nyxara.nyx5.ontogenesis import OntologicalBytecodeGenesis
from nyxara.nyx5.phagocytosis import DigitalPhagocyte
from nyxara.nyx5.retarget import OntologicalCompiler
from nyxara.nyx5.sensorium import PolymorphicSensorium
from nyxara.nyx5.snn import SpikeResponse, SpikingNetwork
from nyxara.nyx5.threading import AnticipatoryThreads
from nyxara.nyx5.voice import WitMatrix

__all__ = ["Nyx5Tick", "Nyx5Brain"]


@dataclass
class Nyx5Tick:
    """One perceive-tick: the network's response plus the active-inference appraisal."""

    n_spikes: int = 0
    firing_rate: float = 0.0
    surprise: float = 0.0
    entropy: float = 0.0
    preemptive: bool = False
    suggestion: Optional[PreemptiveSuggestion] = None
    firing_neurons: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {"n_spikes": self.n_spikes, "firing_rate": round(self.firing_rate, 4),
             "surprise": round(self.surprise, 4), "entropy": round(self.entropy, 4),
             "preemptive": self.preemptive}
        if self.suggestion is not None:
            d["suggestion"] = self.suggestion.to_dict()
        return d


class Nyx5Brain:
    """The composed NYX-5 brain: spiking substrate + HDC memory + active inference + faculties."""

    def __init__(self, config: Any) -> None:
        self.config = config
        c = config
        self.net = SpikingNetwork(
            n_neurons=c.n_neurons, tau=c.tau_membrane_ms, threshold=c.v_threshold,
            refractory=c.refractory_ms, max_synapses=c.max_synapses,
            prune_threshold=c.prune_threshold, max_events=c.max_events_per_tick, seed=c.seed)
        self.memory = HDMemory(dim=c.hd_dim, seed=c.seed)
        self.meter = SurpriseMeter(c.n_neurons, gate=c.surprise_gate)

        # advanced faculties — built when their flag is on (all fail-soft, all config-gated)
        self.chrono = ChronoDilator(deadline_ms=c.chrono_deadline_ms, max_depth=c.chrono_max_depth,
                                    trigger=c.chrono_trigger) if c.chrono_enabled else None
        self.sensorium = PolymorphicSensorium(
            c.n_neurons, max_channels=c.sensorium_max_channels) if c.sensorium_enabled else None
        self.immune = ImmuneGuillotine(
            max_amputations_per_turn=c.immune_max_amputations_per_turn) if c.immune_enabled else None
        self.intent = IntentSymbiote(cache_size=c.intent_cache_size,
                                     speculate=c.intent_speculate) if c.intent_enabled else None
        self.concept = ConceptCollapser(dim=c.hd_dim, seed=c.seed) if c.concept_collapse_enabled else None
        self.axioms = SubAxiomaticEngine(max_systems=c.axiom_max_systems) if c.axiom_forge_enabled else None
        self.negentropy = NegentropyMaintainer(
            interval_ticks=c.negentropy_interval_ticks) if c.negentropy_enabled else None
        self.conduit = SymbioticConduit(ambiguity_gate=c.conduit_ambiguity_gate,
                                        dim=c.hd_dim, seed=c.seed) if c.conduit_enabled else None
        self.mirror = EpistemicMirror() if c.mirroring_enabled else None
        self.continuum = NarrativeContinuum(dim=c.hd_dim, seed=c.seed) if c.continuum_enabled else None
        self.threads = AnticipatoryThreads(
            directions=c.threading_directions) if c.threading_enabled else None
        self.voice = WitMatrix(safety_immutable=c.voice_safety_immutable) if c.voice_enabled else None
        self.dialectic = SovereignDialectic(
            advisory_only=c.dialectic_advisory_only) if c.dialectic_enabled else None
        self.anticipator = ProactiveAnticipator(
            lookahead=c.anticipation_lookahead) if c.anticipation_enabled else None

        # Pillars 6, 9, 14, 15, 16, 17 and 18. The docstring above has always claimed the brain
        # owns these; until now it did not build them, so their config flags
        # (holo_swarm/omni_forge/autopoiesis/mesh/ontogenesis/retarget/phagocytosis_enabled) were
        # decorative — nothing read them and flipping one changed nothing. Same fail-soft,
        # config-gated construction as every faculty above.
        self.swarm = HolographicShard(c.node_id, dim=c.hd_dim,
                                      seed=c.seed) if c.holo_swarm_enabled else None
        # The forge keeps its own default-deny permission gate: the kernel supplies the real one.
        self.omni_forge = OmniForge(
            max_forges_per_turn=c.omni_forge_max_forges_per_turn) if c.omni_forge_enabled else None
        self.autopoiesis = AutopoieticRewriter(
            require_gauntlet=c.autopoiesis_require_gauntlet,
            max_rewrites_per_cycle=c.autopoiesis_max_rewrites_per_cycle,
        ) if c.autopoiesis_enabled else None
        self.mesh = EntangledMesh(c.node_id or "node0",
                                  max_delta_bytes=c.mesh_max_delta_bytes) if c.mesh_enabled else None
        self.ontogenesis = OntologicalBytecodeGenesis(
            max_vm_steps=c.ontogenesis_max_vm_steps) if c.ontogenesis_enabled else None
        self.retarget = OntologicalCompiler(
            require_emulation=c.retarget_require_emulation,
            max_steps=c.ontogenesis_max_vm_steps) if c.retarget_enabled else None
        self.phagocyte = DigitalPhagocyte(
            static_only=c.phagocytosis_static_only) if c.phagocytosis_enabled else None

    # ---- text -> neurons / spikes (deterministic hashing) ---- #
    def _neurons(self, text: str) -> List[int]:
        toks = [t for t in (text or "").lower().split() if t]
        return [int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.config.n_neurons
                for tok in toks[:64]]

    def _encode(self, text: str) -> List[tuple]:
        """Spike-injection form for the SNN: (neuron_id, time, charge)."""
        return [(nid, float(i), 1.5) for i, nid in enumerate(self._neurons(text))]

    # ---- the perceive tick ---- #
    def perceive(self, text: str) -> Nyx5Tick:
        """Encode + stimulate + learn (STDP) + appraise surprise. Fail-soft to an empty tick."""
        try:
            resp: SpikeResponse = self.net.stimulate(self._encode(text))
            firing = resp.firing_neurons()
            reading = self.meter.appraise(firing)
            return Nyx5Tick(n_spikes=resp.n_spikes, firing_rate=resp.mean_rate,
                            surprise=reading.surprise, entropy=reading.entropy,
                            preemptive=reading.preemptive, suggestion=self.meter.suggest(),
                            firing_neurons=firing[:16])
        except Exception:  # noqa: BLE001 — a perceive failure never breaks a turn
            return Nyx5Tick()

    def remember(self, key: str, text: str) -> None:
        try:
            self.memory.remember(key, self._neurons(text))     # bag-of-token-neurons (order-free)
        except Exception:  # noqa: BLE001
            pass

    def recall(self, text: str, *, threshold: float = 0.0):
        try:
            return self.memory.recall(self._neurons(text), threshold=threshold)
        except Exception:  # noqa: BLE001
            return None

    def suggest(self) -> Optional[PreemptiveSuggestion]:
        try:
            return self.meter.suggest()
        except Exception:  # noqa: BLE001
            return None

    # Every optional faculty this facade owns, in pillar order. The roster is the single place
    # that says what the brain is made of — stats() reports from it, so a faculty that is added
    # here shows up in the report, and one that is off is absent rather than reporting zeros.
    _FACULTIES = ("chrono", "sensorium", "immune", "intent", "concept", "axioms", "negentropy",
                  "conduit", "mirror", "continuum", "threads", "voice", "dialectic",
                  "anticipator", "swarm", "omni_forge", "autopoiesis", "mesh", "ontogenesis",
                  "retarget", "phagocyte")

    def faculties(self) -> List[str]:
        """The faculties actually built on this brain — what is live, not what was configured."""
        return [name for name in self._FACULTIES if getattr(self, name, None) is not None]

    def stats(self) -> Dict[str, Any]:
        try:
            s = self.net.stats()
            s["memories"] = len(self.memory)
            s["as_reasoner"] = bool(getattr(self.config, "as_reasoner", False))
            s["faculties"] = self.faculties()
            return s
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"net": self.net.to_dict(), "memory": self.memory.to_dict(),
                "meter": self.meter.to_dict(),
                "conduit": self.conduit.to_dict() if self.conduit else None,
                "intent": self.intent.to_dict() if self.intent else None,
                "continuum": self.continuum.to_dict() if self.continuum else None}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            if d.get("net"):
                self.net.load_dict(d["net"])
            if d.get("memory"):
                self.memory.load_dict(d["memory"])
            if d.get("meter"):
                self.meter.load_dict(d["meter"])
            if d.get("conduit") and self.conduit:
                self.conduit.load_dict(d["conduit"])
            if d.get("intent") and self.intent:
                self.intent.load_dict(d["intent"])
            if d.get("continuum") and self.continuum:
                self.continuum.load_dict(d["continuum"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            pass
