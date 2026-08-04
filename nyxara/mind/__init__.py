"""NYXARA · mind/ — the part that proposes (🧠).

This is the largest package in NYXARA and the one with the least authority: everything here
produces a **candidate**, and the kernel decides what becomes of it. The mind proposes; the
kernel disposes; the Master is sovereign.

**The reasoning seat** — :class:`~nyxara.mind.reasoner.Reasoner` is the entry point;
:class:`~nyxara.mind.native_reasoner.NativeReasoner` is her *own* chain of thought with no
model behind it, :class:`~nyxara.mind.deep_reasoning.DeepReasoner` and
:class:`~nyxara.mind.dual_process.DualProcess` govern how hard she thinks, and
:class:`~nyxara.mind.metacognition.MetaCognition` decides *before* generating how much effort
this turn deserves.

**Many minds, one answer** — :class:`~nyxara.mind.council.LLMCouncil`,
:class:`~nyxara.mind.role_council.RoleCouncil`, :class:`~nyxara.mind.swarm.DeliberativeSwarm` and
:class:`~nyxara.mind.internal_civilization.InternalCivilization` disagree with each other on
purpose; :class:`~nyxara.mind.critique.Critic` and
:class:`~nyxara.mind.inner_critic.InnerCritic` attack the result;
:class:`~nyxara.mind.uncertainty.UncertaintyModel` and
:class:`~nyxara.mind.verified_answer.VerifiedAnswer` decide whether it survived.

**Structure over tokens** — :class:`~nyxara.mind.concept_hyperspace.ConceptHyperspace`,
:class:`~nyxara.mind.hyperbolic_manifold.HyperbolicConceptManifold`,
:class:`~nyxara.mind.analogy.StructureMapper`,
:class:`~nyxara.mind.category_transfer.FunctorTransfer` (category-theoretic transfer between
domains) and :class:`~nyxara.mind.transfer.RelationalTransferEngine` carry meaning as geometry and
mapping rather than text.

**Worlds she can run forward** — :class:`~nyxara.mind.world_model.WorldModel`,
:class:`~nyxara.mind.causal_world_model.CausalWorldModel`,
:class:`~nyxara.mind.world_simulator.WorldSimulator` and
:class:`~nyxara.mind.continuous_world.DifferentiablePhysics` (continuous latent dynamics, when
torch is present) let her ask what *would* happen before anything does.

**Conscience and taste** — :class:`~nyxara.mind.moral.MoralEvaluator` weighs an action under
several ethical frameworks at once and surfaces where they disagree rather than hiding it,
and :class:`~nyxara.mind.aesthetic.AestheticJudge` prefers the more elegant of two correct
answers. Both are **advisory**: they colour a turn, they never flip a disposition.

**Why this module is lazy.** Several faculties here pull heavy optional dependencies (torch,
transformers, z3) at import time, and eagerly importing all of them would make ``import
nyxara.mind`` cost seconds and fail on a partial install. Names resolve on first access
(PEP 562) instead, so ``from nyxara.mind import MoralEvaluator`` imports exactly one module.

This surface is the **principal faculties**, not all 400+ public names in the package — six of
those collide across modules (``Concept``, ``Persona``, ``Verdict``, ``Correspondence``,
``MathEngine``, ``TransferResult``), so exporting everything flat would silently shadow one
with another. Anything not listed here is still importable from its own submodule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# public name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # ---- the reasoning seat ---- #
    "Reasoner": "reasoner",
    "NativeReasoner": "native_reasoner",
    "NyxaraReasoner": "nyxara_reasoner",
    "SelfBrain": "self_reasoner",
    "build_self_brain": "self_reasoner",
    "LLMReasoner": "llm_reasoner",
    "OfflineMind": "offline_mind",
    "ComputeReasoner": "compute_reasoner",
    "MCTSReasoner": "mcts_reasoner",
    "SuperpositionReasoner": "superposition_reasoner",
    "solve_chain": "reasoning_chain",
    "ChainResult": "reasoning_chain",
    "Proposal": "proposal",
    # ---- how hard to think ---- #
    "DeepReasoner": "deep_reasoning",
    "DualProcess": "dual_process",
    "MetaCognition": "metacognition",
    "MetacognitiveController": "metacontrol",
    "MetaIntelligence": "meta_intelligence",
    "EffortMemory": "effort_memory",
    "DeliberativeReasoner": "deliberate",
    "IntuitionCore": "intuition",
    "IntuitionFaculty": "intuition",
    # ---- disagreement, critique, calibration ---- #
    "LLMCouncil": "council",
    "CouncilResult": "council",
    "RoleCouncil": "role_council",
    "DeliberativeSwarm": "swarm",
    "SwarmResult": "swarm",
    "InternalCivilization": "internal_civilization",
    "Critic": "critique",
    "InnerCritic": "inner_critic",
    "UncertaintyModel": "uncertainty",
    "AbstentionPolicy": "uncertainty",
    "VerifiedAnswer": "verified_answer",
    "grounded_verifier": "grounded_verifier",
    "ExplanationBuilder": "explain",
    "Explanation": "explain",
    # ---- meaning as structure ---- #
    "ConceptGraph": "concept_graph",
    "ConceptHierarchy": "concept_hierarchy",
    "ConceptHyperspace": "concept_hyperspace",
    "HyperbolicConceptManifold": "hyperbolic_manifold",
    "LatentGeometry": "latent_geometry",
    "StructureMapper": "analogy",
    "AnalogyResult": "analogy",
    "FunctorTransfer": "category_transfer",
    "Functor": "category_transfer",
    "RelationalTransferEngine": "transfer",
    "SynestheticMatrix": "synesthesia",
    "Predicate": "lot",
    "Forall": "lot",
    "Exists": "lot",
    # ---- worlds she can run forward ---- #
    "WorldModel": "world_model",
    "CausalWorldModel": "causal_world_model",
    "JEPAWorldModel": "jepa_world_model",
    "DifferentiableCausalStructure": "neural_causal",
    "WorldSimulator": "world_simulator",
    "DifferentiablePhysics": "continuous_world",
    "ContinuousDynamics": "continuous_world",
    "PhysicsFit": "continuous_world",
    "PredictionEngine": "prediction_engine",
    "PredictiveCore": "predictive_core",
    "FreeEnergyEngine": "free_energy",
    "ActiveInference": "active_inference",
    # ---- conscience & taste (advisory only) ---- #
    "MoralEvaluator": "moral",
    "MoralEvaluation": "moral",
    "MoralAction": "moral",
    "Judgment": "moral",
    "EthicalFramework": "moral",
    "Consequentialist": "moral",
    "Deontological": "moral",
    "VirtueEthics": "moral",
    "Stakeholder": "moral",
    "Tension": "moral",
    "AestheticJudge": "aesthetic",
    # ---- originality ---- #
    "Atelier": "atelier",
    "MuseEngine": "muse",
    "CreativeEngine": "creative",
    "OriginalityEngine": "originality",
    "CreativeFaculty": "originality",
    "DomainGenesisEngine": "domain_genesis",
    "FirstPrinciplesEngine": "first_principles",
    "FirstPrinciplesFaculty": "first_principles",
    # ---- breadth, strategy, routing ---- #
    "GeneralIntelligence": "general_intelligence",
    "GeneralizationEngine": "generalization",
    "StrategicIntelligence": "strategic",
    "default_strategies": "strategies",
    "Router": "router",
    "PrimarySelfModelRouter": "self_model_router",
    "RAGPipeline": "rag",
    "RAGResult": "rag",
    "UsageLedger": "cost",
    "ThoughtGenerator": "thought_generator",
    "RecursiveImprover": "recursive_improver",
    "TemporalReasoner": "temporal",
    "MathEngine": "math",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:  # pragma: no cover — import-time cost is exactly what this module avoids
    from nyxara.mind.category_transfer import Functor, FunctorTransfer
    from nyxara.mind.continuous_world import DifferentiablePhysics
    from nyxara.mind.moral import MoralEvaluation, MoralEvaluator
    from nyxara.mind.reasoner import Reasoner


def __getattr__(name: str) -> Any:
    """Resolve a mind faculty on first access (PEP 562) — see the module docstring."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value          # cache it: the lazy path runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
