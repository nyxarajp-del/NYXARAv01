"""NYXARA · nyx5/ — the NYX-5 neuromorphic brain (⚡).

NYX-5 is an **event-driven Spiking Neural Network *simulation*** running on **commodity silicon** —
leaky integrate-and-fire neurons stepped by an in-process heapq event queue in pure Python/NumPy. It
is **not neuromorphic** hardware: there are no literal GHz spikes, no microsecond wall-clock cognition,
and no 0.1 W power envelope. Those are the *architectural analogy*, not measured facts. Learning is
**local** (STDP): there is **no backpropagation** and no global training phase. Long-term memory reuses
the audited 10,000-D hyperdimensional algebra (``cognition/hyper_dimensional_vectors.py``), so old
memory does not degrade catastrophically.

NYX-5 is a *capability*, never a hard dependency and never a sovereign: every faculty is fail-soft, and
even when NYX-5 occupies the reason-seat its proposals still pass through the kernel's unchanged,
fail-closed gate. The mind proposes; the kernel disposes; the Master is sovereign.
"""

from __future__ import annotations

from nyxara.nyx5.active_inference import PreemptiveSuggestion, SurpriseMeter
from nyxara.nyx5.anticipation import ProactiveAnticipator
from nyxara.nyx5.autopoiesis import AutopoieticRewriter, RewriteProposal
from nyxara.nyx5.axiom_forge import SubAxiomaticEngine, Truth
from nyxara.nyx5.chrono import ChronoDilator
from nyxara.nyx5.concept_space import ConceptCollapser
from nyxara.nyx5.conduit import SymbioticConduit
from nyxara.nyx5.continuum import NarrativeContinuum
from nyxara.nyx5.dialectic import SovereignDialectic
from nyxara.nyx5.event_queue import EventDrivenScheduler
from nyxara.nyx5.hd_memory import HDMemory, SpikeEncoder
from nyxara.nyx5.holo_swarm import HolographicShard
from nyxara.nyx5.immune import ImmuneGuillotine
from nyxara.nyx5.intent import IntentSymbiote
from nyxara.nyx5.mesh import EntangledMesh
from nyxara.nyx5.mirroring import EpistemicMirror
from nyxara.nyx5.negentropy import NegentropyMaintainer
from nyxara.nyx5.neuron import LIFNeuron, NeuronState, SpikeEvent
from nyxara.nyx5.nyx5_brain import Nyx5Brain, Nyx5Tick
from nyxara.nyx5.omni_forge import OmniForge
from nyxara.nyx5.ontogenesis import OntologicalBytecodeGenesis, StackVM
from nyxara.nyx5.phagocytosis import DigitalPhagocyte
from nyxara.nyx5.reasoner import Nyx5Reasoner
from nyxara.nyx5.retarget import OntologicalCompiler
from nyxara.nyx5.sensorium import PolymorphicSensorium
from nyxara.nyx5.snn import SpikeResponse, SpikingNetwork
from nyxara.nyx5.synapse import STDPRule, Synapse
from nyxara.nyx5.threading import AnticipatoryThreads
from nyxara.nyx5.topology import LiveTopology
from nyxara.nyx5.voice import WitMatrix

__all__ = [
    # substrate (pillars 1-3)
    "LIFNeuron", "NeuronState", "SpikeEvent", "STDPRule", "Synapse", "LiveTopology",
    "EventDrivenScheduler", "SpikingNetwork", "SpikeResponse", "HDMemory", "SpikeEncoder",
    "SurpriseMeter", "PreemptiveSuggestion",
    # advanced (pillars 4-7)
    "ChronoDilator", "PolymorphicSensorium", "HolographicShard", "ImmuneGuillotine",
    # interface / synthesis (pillars 8-10)
    "IntentSymbiote", "OmniForge", "ConceptCollapser",
    # frontier (pillars 11-13)
    "SubAxiomaticEngine", "Truth", "NegentropyMaintainer", "SymbioticConduit",
    # self-genesis / mesh / language (pillars 14-17)
    "AutopoieticRewriter", "RewriteProposal", "EntangledMesh", "OntologicalBytecodeGenesis",
    "StackVM", "OntologicalCompiler",
    # defensive / persona / advisory / proactive (pillars 18-24)
    "DigitalPhagocyte", "EpistemicMirror", "NarrativeContinuum", "AnticipatoryThreads",
    "WitMatrix", "SovereignDialectic", "ProactiveAnticipator",
    # facade + reason-seat
    "Nyx5Brain", "Nyx5Tick", "Nyx5Reasoner",
]
