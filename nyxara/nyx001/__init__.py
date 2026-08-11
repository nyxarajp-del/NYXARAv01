"""NYXARA · nyx001/ — NYX V.001, the primary brain (🜂).

NYX V.001 is the merge of everything NYXARA had into one mind, plus the substrate none of it had:

* **NYX V.01 / V.02 / V.03** (:mod:`nyxara.nyx`) — the concept graph, holographic memory, global
  workspace, specialists, superposition, meta-cognition, and the V.02/V.03 faculties.
* **NYX-5** (:mod:`nyxara.nyx5`) — the event-driven spiking substrate, HDC long-term memory and
  active-inference surprise meter.
* **Layers 0–17** (:mod:`nyxara.nyx001.layers`) — a from-scratch cognitive stack that inherits no
  pretrained intelligence at all: random initialisation, a learned world model, and everything
  above it built out of experience.

The three are not alternatives. V.01–.03 already depended on NYX-5 (``nyx/omega.py`` reaches for
``nyx5.autopoiesis``, ``nyx/synergy.py`` for ``nyx5.mesh``), and the Layer stack supplies what
neither had: a substrate that starts from nothing and learns.

**What the Layer stack refuses.** Its charter forbids inheriting intelligence, and that is enforced
rather than asserted — no pretrained weights, no LLM call, no human-written knowledge rule, no
hand-coded reasoning chain, no copied Transformer recipe. Every parameter is born from a seeded
generator in :mod:`nyxara.nyx001.substrate.init`, and the sequence mechanism is a selective
state-space scan, not attention. NYXARA's existing language ladder is untouched and stays a
separate, optional modality: the Layer stack does not read from it.

**What that costs, stated plainly.** A system that inherits nothing knows nothing at birth. Layer
1's encoder is a hash sketch, so early perception is lexical rather than semantic; concepts that
track meaning only emerge after the world model has learned what predicts what. This is the
charter's own ordering — Environment → Perception → Concepts → World Model → Language — and it is
slow on purpose. NYX V.001's Layer stack is a working cognitive system at medium scale. It is not
a language model and does not compete with one.

The one number everything else rests on is :meth:`~nyxara.nyx001.layers.stack.CognitiveStack.
is_learning` — whether the world model's error is actually falling. It returns ``None``, not
``False``, when there is not yet enough evidence to say, which is the rule the whole package
keeps: a quantity that cannot yet be measured is reported as absent, never as zero.

The mind proposes; the kernel disposes; the Master is sovereign.
"""

from __future__ import annotations

from nyxara.nyx001.layers import (
    ActiveLearning,
    CognitiveCompression,
    CognitiveStack,
    ContradictionEngine,
    CounterfactualEngine,
    CuriosityEngine,
    CycleResult,
    DynamicAttention,
    EpisodicMemory,
    MetaLearner,
    Observation,
    Perception,
    Planner,
    ReasoningEngine,
    ResourceAwareCognition,
    SelfCritique,
    SelfImprovement,
    SelfModel,
    SemanticMemory,
    Substrate,
    WorldModel,
)
from nyxara.nyx001.substrate import (
    AdaptiveOptimizer,
    CognitiveState,
    CompositeObjective,
    HAS_NUMPY,
)

__all__ = [
    "CognitiveStack", "CycleResult", "Observation",
    "Substrate", "Perception", "WorldModel", "EpisodicMemory", "SemanticMemory",
    "DynamicAttention", "ReasoningEngine", "CounterfactualEngine", "Planner",
    "CuriosityEngine", "ContradictionEngine", "SelfCritique", "SelfModel", "MetaLearner",
    "CognitiveCompression", "ResourceAwareCognition", "ActiveLearning", "SelfImprovement",
    "AdaptiveOptimizer", "CognitiveState", "CompositeObjective", "HAS_NUMPY",
]
