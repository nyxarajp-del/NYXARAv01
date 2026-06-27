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
from nyxara.growth.eureka import (
    Breakthrough,
    BreakthroughReport,
    Conjecture,
    EurekaEngine,
)
from nyxara.growth.efficiency import (
    ComputeLedger,
    EfficiencyFrontier,
    EfficiencyPoint,
    estimate_cost,
)
from nyxara.growth.rivalry import Arena, HeadToHead, Rival
from nyxara.growth.adversarial_self_play import (
    AdversarialSelfPlay,
    Duel,
    Elo,
    StrategyBandit,
)
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
from nyxara.growth.synthesis import (
    CurationReport,
    Domain,
    RivalVerifier,
    SyntheticCurator,
    SyntheticGenerators,
    SyntheticItem,
)
from nyxara.growth.topology import (
    CapacityMonitor,
    CapacitySignal,
    DynamicTopology,
    GrowthDecision,
    GrowthMode,
    TopologyReport,
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
from nyxara.growth.mind_evolution import (
    EvolutionLineageReport,
    Generation,
    MindEvolutionEngine,
    ReasoningGenome,
    genome_solver,
)
from nyxara.growth.autolearn import GrowthEngine, GrowthReport
from nyxara.growth.self_optimize import (
    EditGenerator,
    GauntletResult,
    LLMEditGenerator,
    OptimizationOutcome,
    Optimizer,
    SelfEditGenerator,
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
from nyxara.growth.explorer import ExploreResult, InfiniteExplorer
from nyxara.growth.active_curiosity import (
    ActiveCuriosity,
    CuriosityFinding,
    CuriosityPass,
    Question,
)
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
from nyxara.growth.open_world import (
    CandidateLaw,
    DomainSpec,
    LawFamily,
    OpenWorldGeneralizer,
    Probe,
    UnderstandingReport,
    build_alien_machine,
)

__all__ = [
    "AutonomousResearcher",
    "ResearchReport",
    "InfiniteExplorer",
    "ExploreResult",
    "ActiveCuriosity",
    "CuriosityFinding",
    "CuriosityPass",
    "Question",
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
    # open-world generalization — crack a never-before-seen system from first principles
    "OpenWorldGeneralizer",
    "LawFamily",
    "DomainSpec",
    "Probe",
    "CandidateLaw",
    "UnderstandingReport",
    "build_alien_machine",
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
    # truly novel problem solving — self-generated, prover-certified, novelty-filtered discovery
    "EurekaEngine",
    "Conjecture",
    "Breakthrough",
    "BreakthroughReport",
    "ComputeLedger",
    "EfficiencyFrontier",
    "EfficiencyPoint",
    "estimate_cost",
    "Arena",
    "HeadToHead",
    "Rival",
    "AdversarialSelfPlay",
    "Duel",
    "Elo",
    "StrategyBandit",
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
    # synthetic data self-curation (the AlphaGo-Zero method)
    "SyntheticCurator",
    "SyntheticGenerators",
    "RivalVerifier",
    "SyntheticItem",
    "CurationReport",
    "Domain",
    # dynamic topology expansion (runtime Net2Net brain growth)
    "DynamicTopology",
    "CapacityMonitor",
    "CapacitySignal",
    "GrowthDecision",
    "GrowthMode",
    "TopologyReport",
    # continuous evolution
    "RecursiveSelfImprovement",
    "SelfImprovementReport",
    # recursive mind-evolution — evolve the *way of thinking* itself
    "MindEvolutionEngine",
    "ReasoningGenome",
    "Generation",
    "EvolutionLineageReport",
    "genome_solver",
    "GrowthEngine",
    "GrowthReport",
    "Optimizer",
    "EditGenerator",
    "LLMEditGenerator",
    "SelfEditGenerator",
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
