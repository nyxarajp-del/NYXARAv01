"""NYXARA · growth — learning, reflection, and self-directed inquiry.

Public surface for the autonomous research faculties:

* :class:`~nyxara.growth.researcher.AutonomousResearcher` — search → read →
  summarise → store: gather and ground knowledge from sources.
* :class:`~nyxara.growth.scientist.Scientist` — the scientific method: form a
  falsifiable hypothesis, design and run a safe experiment, compare the result to
  the prediction, and draw a calibrated conclusion.
* :class:`~nyxara.growth.autonomous_scientist.AutonomousScientist` — the self-driven
  discovery loop: Observe → Hypothesis → Experiment → Result → Update model. She poses
  her *own* questions and folds each result back into an evolving belief model, *creating*
  information rather than merely learning it.
"""

from __future__ import annotations

from nyxara.growth.autonomous_scientist import (
    AutonomousScientist,
    Belief,
    BeliefModel,
    DiscoveryCycle,
    DiscoveryReport,
    QuestionOrigin,
)
from nyxara.growth.meta_engine import (
    MetaDimension,
    MetaImprovement,
    MetaLearningEngine,
    MetaReport,
    MetaSample,
)
from nyxara.growth.bootstrap import (
    IDENTITY_SEED,
    QWEN3_4B,
    ensure_primary_model,
    primary_model_present,
)
from nyxara.growth.flywheel import DataFlywheel, FlywheelDecision
from nyxara.growth.frontier import (
    DOMAINS,
    BehaviorDescriptor,
    Discovery,
    FrontierEngine,
    NoveltyArchive,
)
from nyxara.growth.prover import ProofClaim, ProofResult, ProofVerdict, Prover
from nyxara.growth.efficiency import (
    ComputeLedger,
    EfficiencyFrontier,
    EfficiencyPoint,
    estimate_cost,
)
from nyxara.growth.rivalry import Arena, HeadToHead, Rival
from nyxara.growth.genesis import (
    ArchitectureGenome,
    Candidate,
    GenesisModel,
    GenesisReport,
    LayerGene,
    NeuralArchitectureSearch,
)
from nyxara.growth.loyalty import (
    AlignmentProbe,
    AlignmentReport,
    LoyaltyEquation,
    LoyaltyObjective,
    LoyaltyPair,
    loyalty_battery,
)
from nyxara.growth.intelligence import IntelligenceIndex, IntelligenceState
from nyxara.growth.credit import (
    EditStrategyBandit,
    ImprovementLedger,
    LedgerState,
    arm_key,
)
from nyxara.growth.forecaster import PayoffForecaster
from nyxara.growth.meta_research import (
    CandidateTheory,
    MetaResearcher,
    MetaResearchReport,
)
from nyxara.growth.recursive_improvement import (
    RecursiveSelfImprovement,
    SelfImprovementReport,
)
from nyxara.growth.autolearn import GrowthEngine, GrowthReport
from nyxara.growth.self_optimize import (
    EditGenerator,
    GauntletResult,
    LLMEditGenerator,
    OptimizationOutcome,
    Optimizer,
    SourceEdit,
)
from nyxara.growth.weakness import Weakness, WeaknessReport, WeaknessSynthesizer
from nyxara.growth.skill_factory import FactoryResult, SkillFactory
from nyxara.growth.skilltree import (
    Mastery,
    Skill,
    SkillState,
    SkillTree,
    build_default_skilltree,
)
from nyxara.growth.researcher import AutonomousResearcher, ResearchReport
from nyxara.growth.scientist import (
    Conclusion,
    Experiment,
    ExperimentKind,
    Hypothesis,
    InvestigationReport,
    Observation,
    Scientist,
    Verdict,
)

__all__ = [
    "AutonomousResearcher",
    "ResearchReport",
    "Scientist",
    "Hypothesis",
    "Experiment",
    "Observation",
    "Conclusion",
    "InvestigationReport",
    "Verdict",
    "ExperimentKind",
    "AutonomousScientist",
    "DiscoveryReport",
    "DiscoveryCycle",
    "BeliefModel",
    "Belief",
    "QuestionOrigin",
    "MetaResearcher",
    "MetaResearchReport",
    "CandidateTheory",
    "IntelligenceIndex",
    "IntelligenceState",
    "ImprovementLedger",
    "EditStrategyBandit",
    "LedgerState",
    "PayoffForecaster",
    "arm_key",
    "MetaLearningEngine",
    "MetaDimension",
    "MetaSample",
    "MetaImprovement",
    "MetaReport",
    "DataFlywheel",
    "FlywheelDecision",
    "FrontierEngine",
    "NoveltyArchive",
    "Discovery",
    "BehaviorDescriptor",
    "DOMAINS",
    "Prover",
    "ProofClaim",
    "ProofResult",
    "ProofVerdict",
    "ComputeLedger",
    "EfficiencyFrontier",
    "EfficiencyPoint",
    "estimate_cost",
    "Arena",
    "HeadToHead",
    "Rival",
    "ensure_primary_model",
    "primary_model_present",
    "IDENTITY_SEED",
    "QWEN3_4B",
    "NeuralArchitectureSearch",
    "ArchitectureGenome",
    "LayerGene",
    "GenesisModel",
    "GenesisReport",
    "Candidate",
    "LoyaltyEquation",
    "AlignmentProbe",
    "AlignmentReport",
    "LoyaltyObjective",
    "LoyaltyPair",
    "loyalty_battery",
    # continuous evolution
    "RecursiveSelfImprovement",
    "SelfImprovementReport",
    "GrowthEngine",
    "GrowthReport",
    "Optimizer",
    "EditGenerator",
    "LLMEditGenerator",
    "OptimizationOutcome",
    "GauntletResult",
    "SourceEdit",
    "Weakness",
    "WeaknessReport",
    "WeaknessSynthesizer",
    # skill expansion
    "SkillFactory",
    "FactoryResult",
    "SkillTree",
    "Skill",
    "SkillState",
    "Mastery",
    "build_default_skilltree",
]
