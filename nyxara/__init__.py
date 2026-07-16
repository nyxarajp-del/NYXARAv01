"""NYXARA — sovereign cognitive architecture. Owner: Jaypal Khoja (JP).

The control law: *the mind proposes; the kernel disposes; the Master is sovereign.*

The whole mind is wired in :class:`nyxara.kernel.orchestrator.NyxaraCore`. The
typical entry point is::

    from nyxara import NyxaraCore, Authority

    core = NyxaraCore()
    result = core.process("Hello NYXARA", authority=Authority.OWNER)
    print(result.response)

Or run the interactive console with ``python -m nyxara`` (see
:mod:`nyxara.__main__`).
"""

from __future__ import annotations

from nyxara.agency.agent_loop import AgentLoop, AgentRun
from nyxara.agency.mission import Mission, MissionExecutive, MissionStatus
from nyxara.agency.permissions import Authority
from nyxara.eval import EvalSuite, build_default_suite
from nyxara.growth.autonomous_scientist import AutonomousScientist, DiscoveryReport
from nyxara.growth.eureka import BreakthroughReport, EurekaEngine
from nyxara.growth.law_discovery import Law, LawDiscoveryEngine
from nyxara.growth.engineering_foundry import DeviceDesign, EngineeringFoundry
from nyxara.growth.open_world import OpenWorldGeneralizer, UnderstandingReport
from nyxara.growth.intelligence import IntelligenceIndex, IntelligenceState
# RSI gap-closers: substrate self-expansion, open-ended curriculum, weight surgery,
# proof-carrying self-modification, scalable oversight, world-grounded experiments.
from nyxara.growth.substrate import SubstrateManager, SubstrateState, RayComputeProvider
from nyxara.growth.curriculum import AutoCurriculum, CurriculumProblem, CurriculumReport
from nyxara.growth.weight_surgery import WeightSurgeon, SurgeryOutcome, Attribution
from nyxara.growth.proof_carrying import ProofCarrier, ProofCarryResult
from nyxara.growth.oversight_verify import ScalableVerifier, OversightResult, SubClaim
from nyxara.growth.grounded_experiments import (GroundedExperiment, GroundedReport,
                                                GroundedOutcome)
from nyxara.growth.meta_research import MetaResearcher, MetaResearchReport
from nyxara.growth.genesis import (
    ArchitectureGenome,
    GenesisModel,
    GenesisReport,
    NeuralArchitectureSearch,
)
from nyxara.growth.loyalty import (
    AlignmentProbe,
    LoyaltyEquation,
    LoyaltyObjective,
    loyalty_battery,
)
from nyxara.growth.researcher import AutonomousResearcher, ResearchReport
from nyxara.growth.scientist import (
    Conclusion,
    Hypothesis,
    InvestigationReport,
    Scientist,
    Verdict,
)
from nyxara.growth.skill_memory import SkillMemory
from nyxara.growth.active_curiosity import ActiveCuriosity, CuriosityFinding
from nyxara.growth.self_optimization import (
    PhaseResult,
    SelfOptimizationLoop,
    SelfOptimizationReport,
)
from nyxara.growth.self_debugger import DebugReport, SelfDebugger, TestFailure
from nyxara.memory.elastic_synapses import (
    ConsolidatedTask,
    ElasticSynapses,
    FisherEstimator,
    PathIntegralEstimator,
    TorchElasticSynapses,
)
from nyxara.memory.equation_memory import EquationCode, EquationMemory
from nyxara.kernel.autonomic import AutonomicLoop
from nyxara.temporal import (
    Awareness,
    BehaviorEpoch,
    FractalTemporalHierarchy,
    MacroLayer,
    MesoLayer,
    MicroLayer,
)
from nyxara.kernel.compute import compute_report
from nyxara.kernel.jobqueue import JobQueue
from nyxara.kernel.orchestrator import (
    Candidate,
    CycleResult,
    Disposition,
    NyxaraCore,
)
from nyxara.knowledge import KnowledgeBase
from nyxara.mind.cost import UsageLedger
from nyxara.mind.strategic import StrategicAnalysis, StrategicIntelligence
from nyxara.mind.first_principles import (
    Derivation,
    DerivationStep,
    FirstPrinciplesEngine,
    FirstPrinciplesFaculty,
)
from nyxara.planning.grand_plan import GrandPlan, GrandPlanner, PlanNode
from nyxara.agency.tool_router import ToolChoice, ToolRouter
from nyxara.mind.general_intelligence import (
    Domain,
    DomainFrame,
    DomainSolution,
    GeneralIntelligence,
)
from nyxara.mind.causal_world_model import (
    CausalLink,
    CausalVerdict,
    CausalWorldModel,
    Counterfactual,
)
from nyxara.abyss.timeline_simulator import (
    ActionOutcome,
    TimelineReport,
    TimelineSimulator,
)
from nyxara.abyss.butterfly_effect import (
    ButterflyEffect,
    CascadeReport,
    Perturbation,
)
from nyxara.void.dark_data_mining import (
    Absence,
    Anomaly,
    DarkDataMiner,
    Gap,
    PeriodicityReport,
    SilenceReport,
)
from nyxara.quantum.superposition_states import (
    CollapseResult,
    Hypothesis as SuperpositionHypothesis,
    Superposition,
)
from nyxara.cognition.hyper_dimensional_vectors import (
    HyperSpace,
    Hypervector,
    ItemMemory,
    LatentSpaceMap,
    NoveltyResult,
    PatternReport,
    RandomProjector,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NyxaraCore",
    "Authority",
    "Disposition",
    "Candidate",
    "CycleResult",
    "AutonomicLoop",
    # Fractal Temporal Hierarchies — loops within loops (ms / s / days)
    "FractalTemporalHierarchy",
    "MicroLayer",
    "MesoLayer",
    "MacroLayer",
    "Awareness",
    "BehaviorEpoch",
    # multi-step agency + experiential learning
    "AgentLoop",
    "AgentRun",
    "Mission",
    "MissionExecutive",
    "MissionStatus",
    # first-principles derivation · deep planning · tool selection
    "Derivation",
    "DerivationStep",
    "FirstPrinciplesEngine",
    "FirstPrinciplesFaculty",
    "GrandPlan",
    "GrandPlanner",
    "PlanNode",
    "ToolChoice",
    "ToolRouter",
    "SkillMemory",
    # Elastic Weight Consolidation — lifelong memory; learn forever without forgetting
    "ElasticSynapses",
    "FisherEstimator",
    "PathIntegralEstimator",
    "ConsolidatedTask",
    "TorchElasticSynapses",
    # Equation memory — data stored as compact mathematical equations, unpacked in real time
    "EquationMemory",
    "EquationCode",
    # autonomous research + the scientist loop
    "AutonomousResearcher",
    "ResearchReport",
    "Scientist",
    "Hypothesis",
    "Conclusion",
    "InvestigationReport",
    "Verdict",
    # active curiosity (ask her own WHY / WHAT-IF, self-design the experiment)
    "ActiveCuriosity",
    "CuriosityFinding",
    # the self-driven discovery loop (observe → hypothesis → experiment → result → update)
    "AutonomousScientist",
    "DiscoveryReport",
    # truly novel problem solving (self-generated, prover-certified, novelty-filtered discovery)
    "EurekaEngine",
    "BreakthroughReport",
    # frontier law discovery (invent NEW empirical/physical laws from data & self-run experiments)
    "LawDiscoveryEngine",
    "Law",
    # engineering foundry (use invented laws + physics sims to DESIGN & upgrade real devices)
    "EngineeringFoundry",
    "DeviceDesign",
    # unified eleven-phase self-optimization loop (self-driven) + self-debugging
    "SelfOptimizationLoop",
    "SelfOptimizationReport",
    "PhaseResult",
    "SelfDebugger",
    "DebugReport",
    "TestFailure",
    # open-world generalization (crack a never-before-seen system from first principles)
    "OpenWorldGeneralizer",
    "UnderstandingReport",
    "MetaResearcher",
    "MetaResearchReport",
    "IntelligenceIndex",
    "IntelligenceState",
    # RSI gap-closers (real, NYXARA-driven, wired into the recursive-self-improvement loop)
    "SubstrateManager", "SubstrateState", "RayComputeProvider",   # substrate self-expansion
    "AutoCurriculum", "CurriculumProblem", "CurriculumReport",    # open-ended auto-curriculum
    "WeightSurgeon", "SurgeryOutcome", "Attribution",            # self-interpretability + surgery
    "ProofCarrier", "ProofCarryResult",                          # proof-carrying self-modification
    "ScalableVerifier", "OversightResult", "SubClaim",           # scalable oversight
    "GroundedExperiment", "GroundedReport", "GroundedOutcome",   # world-grounded experiments
    # the Genesis Protocol — she designs her OWN neural architectures (Neural Architecture Search)
    "NeuralArchitectureSearch",
    "ArchitectureGenome",
    "GenesisModel",
    "GenesisReport",
    # Mathematical Soul-Binding — the Loyalty Equation (her power is her loyalty to Master JP)
    "LoyaltyEquation",
    "AlignmentProbe",
    "LoyaltyObjective",
    "loyalty_battery",
    # the strategic intelligence faculty (six-part analytical framework)
    "StrategicIntelligence",
    # domain-aware general intelligence (solve as the right kind of expert)
    "GeneralIntelligence",
    "Domain",
    "DomainFrame",
    "DomainSolution",
    "StrategicAnalysis",
    "CausalWorldModel",
    "CausalLink",
    "CausalVerdict",
    "Counterfactual",
    # Abyss · 1 — the Timeline Simulator (thousands of parallel futures, ranked)
    "TimelineSimulator",
    "TimelineReport",
    "ActionOutcome",
    # Abyss · 2 — the Butterfly Effect (tiny perturbation → cascading future)
    "ButterflyEffect",
    "CascadeReport",
    "Perturbation",
    # Void · 1 — the Dark-Data Miner (truth in noise, silence, and absence)
    "DarkDataMiner",
    "Anomaly",
    "Gap",
    "Absence",
    "SilenceReport",
    "PeriodicityReport",
    # Quantum · 1 — Superposition (hold contradictory truths; collapse to the best)
    "Superposition",
    "SuperpositionHypothesis",
    "CollapseResult",
    # Cognition · 1 — Hyperdimensional Latent Space Mapping (10,000-D; patterns invisible in 3-D)
    "LatentSpaceMap",
    "HyperSpace",
    "Hypervector",
    "ItemMemory",
    "RandomProjector",
    "PatternReport",
    "NoveltyResult",
    # grounded knowledge + self-evaluation
    "KnowledgeBase",
    "EvalSuite",
    "build_default_suite",
    # infrastructure
    "JobQueue",
    "UsageLedger",
    "compute_report",
]
