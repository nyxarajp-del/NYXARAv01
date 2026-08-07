"""NYXARA · nyx/ — NYX V.01, the unified brain (🧠).

NYXARA already had the parts: a Global Workspace (``kernel/workspace.py``), a holographic
field (``memory/holographic_field.py``), free-energy/active inference (``mind/free_energy.py``),
superposition (``quantum/superposition_states.py``), a spiking substrate (``nyx5/``), an
always-on heartbeat (``void/heartbeat.py``). What it did not have was **one place where those
become a single thought**. That is what this package is.

NYX V.01 composes what exists rather than replacing it. New code lives here only where there
was a genuine gap: a concept graph that rewires itself, memory that is addressed by content
instead of by a token window, specialists that *compete* for a conscious broadcast, and a
meta-cognition that measures its own reliability and feeds that back into attention.

One cycle of :meth:`NyxBrain.think` is: **perceive → ground → deliberate → reflect**. The
specialists are not new reasoning engines — each is a thin adapter over a faculty NYXARA
already has (recall, her concept graph, first-principles derivation, creativity, the three
ethical frameworks). What is new is that they compete on one stage, and that the winner is
chosen partly by each faculty's *measured* track record rather than a hand-tuned priority.

Honest, in this repo's tradition (``docs/CAPABILITIES.md``, ``observe/honesty.py``):

* the graph is **bounded** Hebbian bookkeeping — it forgets, on purpose, and it is not
  biological synaptogenesis;
* memory has **no token window**, which is true; it is **not infinite**, which is not claimed;
* the Global Workspace is a **simulation of the architecture** of consciousness (competition →
  bottleneck → broadcast), not a claim about phenomenal experience;
* reliability is an **exponential moving average of observed outcomes** — ordinary online
  statistics, reported with its sample count so a thin record is visible as thin.

The one control law enforced rather than merely weighted: **verifiable beats probabilistic**.
A checked derivation takes the broadcast over a louder guess.

NYX proposes; the kernel disposes; the Master is sovereign. Every faculty here is fail-soft and
config-gated: a failure degrades to a null result, never a broken turn.
"""

from __future__ import annotations

from nyxara.nyx.brain import NyxBrain, NyxPercept, NyxThought
from nyxara.nyx.graph import (
    Activation,
    Concept,
    DynamicNeuralGraph,
    GraphStats,
    concepts_in,
)
from nyxara.nyx.ground import Grounded, Understanding, WorldGrounding
from nyxara.nyx.holomem import HoloMemory, Recall, Trace
from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verdict, Verification
from nyxara.nyx.metacog import Assessment, RecursiveMetaCognition, Reliability
from nyxara.nyx.modules import (
    CreativeSpecialist,
    DerivationSpecialist,
    EthicsSpecialist,
    GraphSpecialist,
    MemorySpecialist,
    Proposal,
    Situation,
    SpecialistModule,
    default_specialists,
)
from nyxara.nyx.superpose import Collapsed, SolutionSuperposition
from nyxara.nyx.workspace import Deliberation, NyxWorkspace

__all__ = [
    # pillar 1 — the dynamic neural graph + content-addressed memory
    "DynamicNeuralGraph", "Concept", "Activation", "GraphStats", "concepts_in",
    "HoloMemory", "Trace", "Recall",
    # pillar 3 — the specialists, the conscious bottleneck, and measured self-trust
    "SpecialistModule", "Proposal", "Situation", "default_specialists",
    "MemorySpecialist", "GraphSpecialist", "DerivationSpecialist",
    "CreativeSpecialist", "EthicsSpecialist",
    "NyxWorkspace", "Deliberation",
    "RecursiveMetaCognition", "Reliability", "Assessment",
    # pillar 4 — candidates held together, collapsed only when a decision is needed
    "SolutionSuperposition", "Collapsed",
    # pillar 2 — derived beats guessed, and she checks what she is about to say
    "SymbolicSubsymbolicFusion", "Verification", "Verdict",
    # symbol grounding — what her words are actually about
    "WorldGrounding", "Understanding", "Grounded",
    # the facade the kernel holds
    "NyxBrain", "NyxPercept", "NyxThought",
]
