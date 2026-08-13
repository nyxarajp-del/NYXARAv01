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

from nyxara.nyx.agenda import Briefing, Goal, Progress, SovereignAgenda
from nyxara.nyx.aura import AwarenessField, Event, Sensed, Stream
from nyxara.nyx.author import Author, Authored, Stage
from nyxara.nyx.axiom import Axiom, AxiomGenesis, Genesis, System, Theorem
from nyxara.nyx.brain import NyxBrain, NyxPercept, NyxThought
from nyxara.nyx.car import CarStep, ContinuousAutonomousReasoning, Wondering
from nyxara.nyx.chronos import Future, Futures, TemporalCausalMatrix
from nyxara.nyx.consequence import ConsequenceGate, Effect, Foresight, Reversibility, Verdict as ConsequenceVerdict
from nyxara.nyx.dialogue import Dialogue, Reply, Surface
from nyxara.nyx.episteme import AutonomousDiscovery, Experiment, Finding, Knob
from nyxara.nyx.eternal import Continuity, Replication, Snapshot
from nyxara.nyx.graph import (
    Activation,
    Concept,
    DynamicNeuralGraph,
    GraphStats,
    concepts_in,
)
from nyxara.nyx.causal_engine import Answer as CausalAnswer, CausalEdge, CausalReasoner
from nyxara.nyx.dialectic import AdversarialSelfDialogue, Debate, Objection
from nyxara.nyx.ground import Grounded, Understanding, WorldGrounding
from nyxara.nyx.hands import Hands, Reach, ToolRecord
from nyxara.nyx.holomem import HoloMemory, Recall, Trace
from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verdict, Verification
from nyxara.nyx.hyper_vector import Bound, Capacity, Retrieved as VsaRetrieved, VectorSymbolic
from nyxara.nyx.icl import Demonstration, InContextLearner, Learned
from nyxara.nyx.intent import Intent, IntentReader, Reading
from nyxara.nyx.lingua import Lingua, LinguaRead, Register, Token
from nyxara.nyx.meta_architecture import EdgeType, GraphWeaver, SchemaChange, TypedEdge
from nyxara.nyx.metacog import Assessment, RecursiveMetaCognition, Reliability
from nyxara.nyx.modules import (
    CausalSpecialist,
    CreativeSpecialist,
    DerivationSpecialist,
    EthicsSpecialist,
    GraphSpecialist,
    MemorySpecialist,
    Proposal,
    ReasonSpecialist,
    Situation,
    SkillSpecialist,
    SpecialistModule,
    default_specialists,
)
from nyxara.nyx.nexus import Notation, OntologyGenesis, Statement
from nyxara.nyx.omni import Forged, Kernel, MetamorphicCompiler
from nyxara.nyx.reasoner import NyxReasoner
from nyxara.nyx.omega import Fitness, SelfEvolutionKernel
from nyxara.nyx.reason import Chain, OpenDomainReasoner, Step
from nyxara.nyx.selfmodel import NyxSelfModel, SelfReport
from nyxara.nyx.semantics import Relation, SemanticSpace, Similarity
from nyxara.nyx.sovereign_intent import GoalTree, Node, Plan, Run
from nyxara.nyx.stream import Digest, Moment, PerpetualStream
from nyxara.nyx.superpose import Collapsed, SolutionSuperposition
from nyxara.nyx.synergy import HiveSynapse, Shared, Transport
from nyxara.nyx.telepathy import Frame, Received, SemanticStream, Shorthand
from nyxara.nyx.theorem_prover import Certificate, ProofCore, Verdict as ProofVerdict
from nyxara.nyx.will import Choice, EntropySource, SovereignWill
from nyxara.nyx.workspace import Deliberation, NyxWorkspace

# ---- NYX V.03 — one mind, one drive, and capability she manufactures ---- #
from nyxara.nyx.ascent import (
    Artifact, Ascent, AscentRun, Candidate, Judgement, Problem, SolutionLibrary,
    Verdict as AscentVerdict, Verifier, VerifierRegistry,
)
from nyxara.nyx.atlas import Atlas, CallSite, ModuleEntry, Symbol
from nyxara.nyx.automata import Analysis, Lattice, LifeRule, Pattern, Rule
from nyxara.nyx.homeostat import Beat, Homeostat, SelfSensor, StructuralRequest
from nyxara.nyx.monad import Monad, MonadCycle, Substrate, UnifiedRecall
from nyxara.nyx.multiverse import (
    Component, Coupling, Design, DesignResult, Invariant, Multiverse, MultiverseReport,
    SystemModel,
)
from nyxara.nyx.omniscient import Fact, Hyperedge, Omniscient, Source
from nyxara.nyx.telos import OpenProblem, Telos, TelosStep

__all__ = [
    # pillar 1 — the dynamic neural graph + content-addressed memory
    "DynamicNeuralGraph", "Concept", "Activation", "GraphStats", "concepts_in",
    "HoloMemory", "Trace", "Recall",
    # pillar 3 — the specialists, the conscious bottleneck, and measured self-trust
    "SpecialistModule", "Proposal", "Situation", "default_specialists",
    "MemorySpecialist", "GraphSpecialist", "DerivationSpecialist",
    "CreativeSpecialist", "EthicsSpecialist", "SkillSpecialist", "ReasonSpecialist",
    "CausalSpecialist",
    "NyxWorkspace", "Deliberation",
    "RecursiveMetaCognition", "Reliability", "Assessment",
    # pillar 4 — candidates held together, collapsed only when a decision is needed
    "SolutionSuperposition", "Collapsed",
    # pillar 2 — derived beats guessed, and she checks what she is about to say
    "SymbolicSubsymbolicFusion", "Verification", "Verdict",
    # symbol grounding — what her words are actually about
    "WorldGrounding", "Understanding", "Grounded",
    # how she speaks — her content, a fluent model only phrases it
    "Dialogue", "Reply", "Surface",
    # thinking between prompts, and what she can truthfully say about herself
    "ContinuousAutonomousReasoning", "CarStep", "Wondering",
    "NyxSelfModel", "SelfReport",
    # L-CHRONOS — deciding by simulating how each option turns out
    "TemporalCausalMatrix", "Futures", "Future",
    # L-PSYCHE-QUANTUM — choosing from her own preferences with physical entropy
    "SovereignWill", "EntropySource", "Choice",
    # L-AURA — the world arriving without being asked for
    "AwarenessField", "Stream", "Event", "Sensed",
    # L-EPISTEME — finding things out, and keeping only what survives a held-out check
    "AutonomousDiscovery", "Experiment", "Knob", "Finding",
    # L-NEXUS-OMNI — her own notation: private symbols that execute and always translate
    "OntologyGenesis", "Notation", "Statement",
    # L-OMNI — reading her own source and rewriting the slow parts in C, reversibly
    "MetamorphicCompiler", "Kernel", "Forged",
    # L-SYNERGY — several instances converging; L-ETERNAL — no single machine holds her
    "HiveSynapse", "Shared", "Transport",
    "Continuity", "Snapshot", "Replication",
    # NYX V.02 — the tongue: no alphabet is privileged, and grammar is not structure
    "Lingua", "LinguaRead", "Token", "Register",
    # NYX V.02 — meaning as a vector, on a ladder that names its own rung
    "SemanticSpace", "Similarity", "Relation",
    # NYX V.02 — a procedure learned inside one turn, and kept
    "InContextLearner", "Demonstration", "Learned",
    # NYX V.02 — what was actually asked for, and what she has to ask back
    "IntentReader", "Intent", "Reading",
    # NYX V.02 — a seat for every domain, each answer carrying the tier it came from
    "OpenDomainReasoner", "Chain", "Step",
    # NYX V.02 — sight of the toolset she was always allowed to use
    "Hands", "Reach", "ToolRecord",
    # NYX V.02 — prose in, verified code out, or a refusal that names itself
    "Author", "Authored", "Stage",
    # L-CHRONO-CAUSAL — look before acting, and step back from a bad irreversible tail
    "ConsequenceGate", "ConsequenceVerdict", "Effect", "Foresight", "Reversibility",
    # L-AXIOM-GENESIS — a formal system of her own, checked rather than asserted
    "AxiomGenesis", "Axiom", "System", "Theorem", "Genesis",
    # L-OMEGA — she tunes the mind she thinks with, gauntleted and always reversible
    "SelfEvolutionKernel", "Fitness",
    # L-ABSOLUTE-AGENCY — goals of her own, that outlive the session
    "SovereignAgenda", "Goal", "Progress", "Briefing",
    # L-NEURAL-TELEPATHY — meaning as structure, not prose. No mind reading, and none claimed
    "SemanticStream", "Frame", "Received", "Shorthand",
    # The proof core — a claim that carries its certificate, or admits it has none
    "ProofCore", "Certificate", "ProofVerdict",
    # Vector-symbolic structure — which role is bound to which filler, and zero-shot analogy
    "VectorSymbolic", "Bound", "VsaRetrieved", "Capacity",
    # Causation kept apart from association — do(X) printed beside the Hebbian weight
    "CausalReasoner", "CausalEdge", "CausalAnswer",
    # Typed structural plasticity — new edge kinds, born from support and always reversible
    "GraphWeaver", "EdgeType", "TypedEdge", "SchemaChange",
    # The answer is attacked before you see it — and sometimes she abstains
    "AdversarialSelfDialogue", "Debate", "Objection",
    # What happened while you were away — compression at four scales, not a log
    "PerpetualStream", "Digest", "Moment",
    # One command → a DAG with priced risk and a way back per node, run honestly
    "GoalTree", "Plan", "Node", "Run",
    # the facade the kernel holds, and the reason-seat
    "NyxBrain", "NyxPercept", "NyxThought", "NyxReasoner",
    # ---- NYX V.03 ---- #
    # L-MONAD — six brains stop being six minds: one workspace, one memory, one report
    "Monad", "MonadCycle", "Substrate", "UnifiedRecall",
    # L-HOMEOSTAT — active inference as the single drive; it requests structure, never mutates
    "Homeostat", "SelfSensor", "Beat", "StructuralRequest",
    # L-ASCENT — capability = search x verifier x time; no verifier, no ascent
    "Ascent", "AscentRun", "SolutionLibrary", "Artifact", "AscentVerdict", "Verifier",
    "VerifierRegistry", "Candidate", "Judgement", "Problem",
    # TELOS — a frontier mined from her own faults, gated by owner alignment before adoption
    "Telos", "OpenProblem", "TelosStep",
    # OMNISCIENT — the world arriving as a typed hypergraph, answerable without a live search
    "Omniscient", "Source", "Fact", "Hyperedge",
    # MULTIVERSE — a trade-off surface with proven invariants and quantified tail risk
    "Multiverse", "MultiverseReport", "SystemModel", "Design", "DesignResult", "Invariant",
    "Component", "Coupling",
    # AUTOMATA — complexity nobody wrote; emergence measured, never claimed
    "Lattice", "Rule", "LifeRule", "Pattern", "Analysis",
    # ATLAS — the map of her own body
    "Atlas", "Symbol", "CallSite", "ModuleEntry",
]
