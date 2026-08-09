"""NYXARA · nyx/brain.py — the NYX V.01 brain facade (🧠, the single object the kernel holds).

:class:`NyxBrain` is the one object ``NyxaraCore`` builds and holds as ``self.nyx``. It owns the
self-rewiring concept graph and the content-addressed associative memory, and — as later phases
land — the workspace, the specialists, the superposition and the meta-cognition that turn those
into one cognitive cycle.

Every method is **fail-soft**: on any error it degrades to a null result, never breaking a turn.
Every faculty is **config-gated**, so a deployment can run exactly as much brain as it wants.

Honest, as everywhere in NYX V.01: the graph is bounded Hebbian bookkeeping (it forgets, on
purpose); memory has **no token context window** — which is true — but it is **not infinite**,
which is not claimed. The mind proposes; the kernel disposes; the Master is sovereign.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.nyx.agenda import Briefing, Goal, SovereignAgenda
from nyxara.nyx.aura import AwarenessField
from nyxara.nyx.author import Author, Authored
from nyxara.nyx.axiom import AxiomGenesis, Genesis
from nyxara.nyx.car import CarStep, ContinuousAutonomousReasoning
from nyxara.nyx.causal_engine import Answer as CausalAnswer, CausalReasoner
from nyxara.nyx.chronos import Futures, TemporalCausalMatrix
from nyxara.nyx.consequence import ConsequenceGate, Verdict
from nyxara.nyx.dialectic import AdversarialSelfDialogue, Debate
from nyxara.nyx.dialogue import Dialogue, Reply
from nyxara.nyx.episteme import AutonomousDiscovery
from nyxara.nyx.eternal import Continuity
from nyxara.nyx.graph import Activation, DynamicNeuralGraph
from nyxara.nyx.hands import Hands, Reach
from nyxara.nyx.ground import Grounded, WorldGrounding
from nyxara.nyx.holomem import HoloMemory, Recall, Trace
from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verification
from nyxara.nyx.hyper_vector import Retrieved as VsaRetrieved, VectorSymbolic
from nyxara.nyx.icl import InContextLearner, Learned
from nyxara.nyx.intent import Intent, IntentReader
from nyxara.nyx.lingua import Lingua, LinguaRead
from nyxara.nyx.meta_architecture import GraphWeaver, TypedEdge
from nyxara.nyx.metacog import RecursiveMetaCognition
from nyxara.nyx.nexus import OntologyGenesis
from nyxara.nyx.omega import SelfEvolutionKernel, Step as OmegaStep
from nyxara.nyx.omni import MetamorphicCompiler
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
)
from nyxara.nyx.reason import Chain, OpenDomainReasoner
from nyxara.nyx.selfmodel import NyxSelfModel, SelfReport
from nyxara.nyx.sovereign_intent import GoalTree, Plan, Run
from nyxara.nyx.semantics import SemanticSpace
from nyxara.nyx.stream import Digest, PerpetualStream
from nyxara.nyx.superpose import Collapsed, SolutionSuperposition
from nyxara.nyx.telepathy import Frame, Received, SemanticStream
from nyxara.nyx.theorem_prover import Certificate as ProofCertificate, ProofCore
from nyxara.nyx.synergy import HiveSynapse
from nyxara.nyx.will import Choice, SovereignWill
from nyxara.nyx.workspace import Deliberation, NyxWorkspace

__all__ = ["NyxPercept", "NyxThought", "NyxBrain"]

# Being *told* something and being *asked* something are different events, and conflating them
# is why a stored question can come back as its own answer. A question is not knowledge; a
# statement the Master makes is.
_QUESTION = re.compile(
    r"^\s*(what|why|how|who|whom|whose|when|where|which|is|are|was|were|do|does|did|can|could|"
    r"will|would|should|shall|may|might|have|has|had|am|tell me|explain)\b", re.I)


class _Latency:
    """Time one step of a real thought and attribute it to the module that did the work.

    This is L-OMNI's only input, and it is why the layer is not a hand-written list of functions
    to optimise: the module she is measurably slowest in is the module she goes and reads.
    Absent a compiler the timer is inert, so measuring costs nothing when nothing uses it.
    """

    __slots__ = ("omni", "_module", "_t0")

    def __init__(self, omni: Any) -> None:
        self.omni = omni
        self._module = ""
        self._t0 = 0.0

    def at(self, module: str) -> "_Latency":
        self._module = module
        return self

    def __enter__(self) -> "_Latency":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        try:
            if self.omni is not None:
                self.omni.observe(self._module, (time.perf_counter() - self._t0) * 1000.0)
        except Exception:  # noqa: BLE001 — a stopwatch never breaks a thought
            pass
        return False


def is_question(text: str) -> bool:
    """Was this asked, rather than asserted?

    NYX V.02: this now asks :mod:`nyxara.nyx.intent`, which decides from *mood* across English,
    Hinglish and Devanagari rather than from a leading wh-word — the regex below could not see
    "kya ye kaam karta hai" as a question at all, and read
    "mera code fix kar do lekin pehle test chala" as a statement. The regex stays as the floor
    under it, so a failure in the reader degrades to V.01 behaviour rather than to nothing.
    """
    try:
        from nyxara.nyx.intent import is_question as _read_mood
        return _read_mood(text)
    except Exception:  # noqa: BLE001 — the reader is a capability, never a dependency
        pass
    try:
        text = str(text or "").strip()
        return bool(text) and (text.endswith("?") or _QUESTION.match(text) is not None)
    except Exception:  # noqa: BLE001
        return False


# Every specialist NYX can seat, by the name used in ``NyxConfig.specialists``.
_SPECIALISTS = {
    "memory": MemorySpecialist,
    "graph": GraphSpecialist,
    "derivation": DerivationSpecialist,
    "creative": CreativeSpecialist,
    "ethics": EthicsSpecialist,
    "skill": SkillSpecialist,
    "reason": ReasonSpecialist,
    "causal": CausalSpecialist,
}


@dataclass
class NyxPercept:
    """One perceive-tick: what the graph did, and what memory brought back."""

    concepts: List[str] = field(default_factory=list)
    born: List[str] = field(default_factory=list)
    strengthened: int = 0
    pruned: int = 0
    spread: List[str] = field(default_factory=list)
    context: List[Trace] = field(default_factory=list)
    recall: Optional[Recall] = None
    grounded: Optional[Grounded] = None

    @property
    def novelty(self) -> float:
        """How much of this turn was new to her — 0.0 (all familiar) … 1.0 (all new)."""
        if not self.concepts:
            return 0.0
        return min(1.0, len(self.born) / float(len(self.concepts)))

    def to_dict(self) -> Dict[str, Any]:
        return {"concepts": self.concepts, "born": self.born,
                "strengthened": self.strengthened, "pruned": self.pruned,
                "spread": self.spread, "novelty": round(self.novelty, 4),
                "context": [t.key for t in self.context],
                "recall": self.recall.to_dict() if self.recall is not None else None,
                "grounded": self.grounded.to_dict() if self.grounded is not None else None}


@dataclass
class NyxThought:
    """One complete cycle: perceive → ground in memory → deliberate → reflect."""

    stimulus: str = ""
    percept: Optional[NyxPercept] = None
    deliberation: Optional[Deliberation] = None
    collapsed: Optional[Collapsed] = None
    futures: Optional[Futures] = None
    choice: Optional[Choice] = None
    verification: Optional[Verification] = None
    assessment: Any = None                       # nyx.metacog.Assessment
    induced: Any = None                          # nyx.icl.Learned, when the turn taught her one
    intent: Any = None                           # nyx.intent.Intent — what was actually asked
    debate: Any = None                           # nyx.dialectic.Debate — the winner, attacked
    cycle_id: str = ""

    @property
    def winner(self) -> Optional[Proposal]:
        return self.deliberation.winner if self.deliberation is not None else None

    @property
    def answer(self) -> str:
        """What reached awareness. Empty when nothing did — which is a real, honest outcome."""
        winner = self.winner
        return winner.content if winner is not None else ""

    @property
    def verified(self) -> bool:
        """Whether the answer was *derived and checked*, not merely the most confident guess."""
        winner = self.winner
        return bool(winner is not None and winner.verifiable)

    @property
    def confidence(self) -> float:
        winner = self.winner
        return float(winner.confidence) if winner is not None else 0.0

    @property
    def decided(self) -> bool:
        """Whether one answer actually dominated. ``False`` means she is genuinely unsure."""
        return bool(self.collapsed.decided) if self.collapsed is not None else bool(self.winner)

    @property
    def entropy(self) -> float:
        """How spread out her belief still is, in bits. High = several answers are live."""
        return float(self.collapsed.entropy) if self.collapsed is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus, "cycle_id": self.cycle_id,
                "answer": self.answer, "verified": self.verified,
                "confidence": round(self.confidence, 4),
                "decided": self.decided, "entropy": round(self.entropy, 4),
                "source": self.winner.source if self.winner else None,
                "percept": self.percept.to_dict() if self.percept else None,
                "deliberation": (self.deliberation.to_dict()
                                 if self.deliberation is not None else None),
                "collapsed": self.collapsed.to_dict() if self.collapsed is not None else None,
                "futures": self.futures.to_dict() if self.futures is not None else None,
                "choice": self.choice.to_dict() if self.choice is not None else None,
                "verification": (self.verification.to_dict()
                                 if self.verification is not None else None),
                "induced": (self.induced.to_dict() if self.induced is not None else None),
                "intent": (self.intent.to_dict() if self.intent is not None else None),
                "debate": (self.debate.to_dict() if self.debate is not None else None),
                "assessment": (self.assessment.to_dict()
                               if self.assessment is not None else None)}


class NyxBrain:
    """The composed NYX V.01 brain: graph + memory + a workspace of competing specialists."""

    def __init__(self, config: Any) -> None:
        self.config = c = config
        self.graph = DynamicNeuralGraph(
            max_nodes=c.graph_max_nodes, max_edges=c.graph_max_edges,
            hebbian_rate=c.hebbian_rate, decay_rate=c.decay_rate,
            prune_threshold=c.graph_prune_threshold,
            rewire_budget=c.rewire_budget_per_tick,
            spread_depth=c.spread_depth, spread_falloff=c.spread_falloff)
        self.memory = HoloMemory(
            dim=c.holo_dim, capacity=c.holo_capacity, seed=c.seed,
            recall_threshold=c.holo_recall_threshold, link_rate=c.link_rate,
            max_links_per_trace=c.max_links_per_trace)
        self.lingua = self._build_lingua(c)
        self.semantics = self._build_semantics(c)
        self._wire_semantics(c)
        self.intent = self._build_intent(c)
        self.icl = self._build_icl(c)
        self.reason = self._build_reason(c)
        self.axiom = self._build_axiom(c)
        self.omega = self._build_omega(c)
        self.agenda = self._build_agenda(c)
        self.telepathy = self._build_telepathy(c)
        self.prover = self._build_prover(c)
        self.vsa = self._build_vsa(c)
        self.consequence = self._build_consequence(c)
        self.hands = self._build_hands(c)
        self.author = self._build_author(c)
        self.causal = self._build_causal(c)
        self.weaver = self._build_weaver(c)
        self.dialectic = self._build_dialectic(c)
        self.stream = self._build_stream(c)
        self.goals = self._build_goals(c)
        self.metacog = self._build_metacog(c)
        self.workspace = self._build_workspace(c)
        self.prover = self.prover if getattr(self, 'prover', None) is not None \
            else self._build_prover(c)
        self.hybrid = self._build_hybrid(c)
        self.ground = self._build_ground(c)
        self.dialogue = self._build_dialogue(c)
        self.chronos = self._build_chronos(c)
        self.will = self._build_will(c)
        self.aura = self._build_aura(c)
        self.nexus = self._build_nexus(c)
        self.synergy = self._build_synergy(c)
        self.eternal = self._build_eternal(c)
        self.episteme = self._build_episteme(c)
        self.omni = self._build_omni(c)
        self.car = self._build_car(c)
        self.selfmodel = NyxSelfModel(self) if getattr(c, "selfmodel_enabled", True) else None
        self.tools: Any = None
        self.knowledge: Any = None
        self.core: Any = None
        self.turns = 0
        self._last_investigation = 0.0
        self._last_forge = 0.0

    @staticmethod
    def _build_lingua(c: Any) -> Optional[Lingua]:
        if not getattr(c, "lingua_enabled", True):
            return None
        try:
            return Lingua(max_tokens=getattr(c, "lingua_max_tokens", 512),
                          min_concept_len=getattr(c, "lingua_min_concept_len", 2),
                          transliterate_bridge=getattr(c, "lingua_transliterate", True),
                          use_nlp=getattr(c, "lingua_use_nlp", True))
        except Exception:  # noqa: BLE001 — without a tongue she falls back to the ASCII floor
            return None

    def _build_causal(self, c: Any) -> Optional[CausalReasoner]:
        if not getattr(c, "causal_enabled", True):
            return None
        try:
            return CausalReasoner(self,
                                  min_confidence=getattr(c, "causal_min_confidence", 0.3),
                                  max_edges=getattr(c, "causal_max_edges", 2048),
                                  wire_to_graph=getattr(c, "causal_wire_to_graph", True))
        except Exception:  # noqa: BLE001 — without it every arrow is back to being association
            return None

    def _build_weaver(self, c: Any) -> Optional[GraphWeaver]:
        if not getattr(c, "weaver_enabled", True):
            return None
        try:
            return GraphWeaver(self, min_support=getattr(c, "weaver_min_support", 3),
                               max_types=getattr(c, "weaver_max_types", 32),
                               rewire_budget=getattr(c, "weaver_rewire_budget", 16),
                               max_edges=getattr(c, "weaver_max_edges", 8192))
        except Exception:  # noqa: BLE001 — without it the graph keeps one untyped edge kind
            return None

    def _build_dialectic(self, c: Any) -> Optional[AdversarialSelfDialogue]:
        if not getattr(c, "dialectic_enabled", True):
            return None
        try:
            return AdversarialSelfDialogue(
                self, abstain_above=getattr(c, "dialectic_abstain_above", 1.2),
                weaken_above=getattr(c, "dialectic_weaken_above", 0.4),
                budget_ms=getattr(c, "dialectic_budget_ms", 250.0),
                use_prover=getattr(c, "dialectic_use_prover", True))
        except Exception:  # noqa: BLE001 — without it the winner ships unattacked, as in V.01
            return None

    def _build_stream(self, c: Any) -> Optional[PerpetualStream]:
        if not getattr(c, "stream_enabled", True):
            return None
        try:
            return PerpetualStream(self, capacity=getattr(c, "stream_capacity", 2048),
                                   every_s=getattr(c, "stream_every_s", 15.0),
                                   novel_cap=getattr(c, "stream_novel_cap", 12))
        except Exception:  # noqa: BLE001 — without it events accumulate and never compress
            return None

    def _build_goals(self, c: Any) -> Optional[GoalTree]:
        if not getattr(c, "goals_enabled", True):
            return None
        try:
            return GoalTree(self, max_nodes=getattr(c, "goals_max_nodes", 24),
                            max_steps=getattr(c, "goals_max_steps", 12),
                            max_repairs=getattr(c, "goals_max_repairs", 3),
                            execute=getattr(c, "goals_execute", True))
        except Exception:  # noqa: BLE001 — without it a command is one action, never a plan
            return None

    def _build_vsa(self, c: Any) -> Optional[VectorSymbolic]:
        if not getattr(c, "vsa_enabled", True):
            return None
        try:
            return VectorSymbolic(self, dim=getattr(c, "vsa_dim", 10000),
                                  seed=getattr(c, "seed", 42),
                                  min_margin=getattr(c, "vsa_min_margin", 0.05))
        except Exception:  # noqa: BLE001 — without it her concepts stay a bag of words
            return None

    @staticmethod
    def _build_prover(c: Any) -> Optional[ProofCore]:
        if not getattr(c, "prover_enabled", True):
            return None
        try:
            return ProofCore(timeout_ms=getattr(c, "prover_timeout_ms", 3000),
                             max_vars=getattr(c, "prover_max_vars", 8))
        except Exception:  # noqa: BLE001 — without it "verifiable" means what it meant in V.01
            return None

    def _build_telepathy(self, c: Any) -> Optional[SemanticStream]:
        if not getattr(c, "telepathy_enabled", True):
            return None
        try:
            return SemanticStream(self, queue=getattr(c, "telepathy_queue", 128),
                                  mint_after=getattr(c, "telepathy_mint_after", 3),
                                  trigger_words=getattr(c, "telepathy_trigger_words", 2),
                                  max_shorthand=getattr(c, "telepathy_max_shorthand", 64))
        except Exception:  # noqa: BLE001 — without it every instruction goes through prose
            return None

    def _build_agenda(self, c: Any) -> Optional[SovereignAgenda]:
        if not getattr(c, "agenda_enabled", True):
            return None
        try:
            return SovereignAgenda(self, max_goals=getattr(c, "agenda_max_goals", 24),
                                   max_open=getattr(c, "agenda_max_open", 6),
                                   stall_after=getattr(c, "agenda_stall_after", 3),
                                   every_s=getattr(c, "agenda_every_s", 30.0),
                                   budget_ms=getattr(c, "agenda_budget_ms", 400.0))
        except Exception:  # noqa: BLE001 — without it she thinks one thought and forgets it
            return None

    def _build_omega(self, c: Any) -> Optional[SelfEvolutionKernel]:
        if not getattr(c, "omega_enabled", True):
            return None
        try:
            return SelfEvolutionKernel(
                self, every_s=getattr(c, "omega_every_s", 120.0),
                step=getattr(c, "omega_step", 0.2),
                min_samples=getattr(c, "omega_min_samples", 12),
                stall_after=getattr(c, "omega_stall_after", 4),
                rule_synth_on_stall=getattr(c, "omega_rule_synth_on_stall", True),
                rule_population=getattr(c, "omega_rule_population", 8),
                rule_generations=getattr(c, "omega_rule_generations", 4),
                rule_budget_s=getattr(c, "omega_rule_budget_s", 4.0),
                min_edges_for_health=getattr(c, "omega_min_edges_for_health", 24),
                replay_k=getattr(c, "omega_replay_k", 24),
                seed=getattr(c, "seed", 42))
        except Exception:  # noqa: BLE001 — without it her constants stay whatever was typed
            return None

    def _build_axiom(self, c: Any) -> Optional[AxiomGenesis]:
        if not getattr(c, "axiom_enabled", True):
            return None
        try:
            return AxiomGenesis(self, bound=getattr(c, "axiom_bound", 4),
                                timeout_ms=getattr(c, "axiom_timeout_ms", 2000),
                                max_systems=getattr(c, "axiom_max_systems", 16))
        except Exception:  # noqa: BLE001 — without it every premise she has was given to her
            return None

    def _build_consequence(self, c: Any) -> Optional[ConsequenceGate]:
        if not getattr(c, "consequence_enabled", True):
            return None
        try:
            return ConsequenceGate(
                self, workspace=getattr(c, "consequence_workspace", ""),
                tail_risk_ceiling=getattr(c, "consequence_tail_risk_ceiling", 0.5),
                branches=getattr(c, "consequence_branches", 64),
                require_foresight_for_irreversible=getattr(
                    c, "consequence_fail_closed_irreversible", True),
                prefer_reversible_variant=getattr(c, "consequence_prefer_variant", True))
        except Exception:  # noqa: BLE001 — without it she acts without looking, as V.01 did
            return None

    def _build_author(self, c: Any) -> Optional[Author]:
        if not getattr(c, "author_enabled", True):
            return None
        try:
            return Author(self,
                          sandbox_timeout_s=getattr(c, "author_sandbox_timeout_s", 5.0),
                          load=getattr(c, "author_load", True),
                          lineage=getattr(c, "author_lineage", True),
                          max_source_bytes=getattr(c, "author_max_source_bytes", 20_000))
        except Exception:  # noqa: BLE001 — without it she can only lower code, never write it
            return None

    def _build_hands(self, c: Any) -> Optional[Hands]:
        if not getattr(c, "hands_enabled", True):
            return None
        try:
            return Hands(self, registry=getattr(self, "tools", None),
                         min_score=getattr(c, "hands_min_score", 0.5),
                         max_per_beat=getattr(c, "hands_max_per_beat", 1),
                         autonomous=getattr(c, "hands_autonomous", True))
        except Exception:  # noqa: BLE001 — without it her brain is blind to tools, as in V.01
            return None

    def _build_reason(self, c: Any) -> Optional[OpenDomainReasoner]:
        if not getattr(c, "reason_enabled", True):
            return None
        try:
            return OpenDomainReasoner(
                self, min_confidence=getattr(c, "reason_min_confidence", 0.3),
                use_generalization=getattr(c, "reason_use_generalization", True),
                use_associative=getattr(c, "reason_use_associative", True))
        except Exception:  # noqa: BLE001 — without it, outside four domains she is silent again
            return None

    def _build_intent(self, c: Any) -> Optional[IntentReader]:
        if not getattr(c, "intent_enabled", True):
            return None
        try:
            return IntentReader(
                lingua=self.lingua,
                min_reading_gap=getattr(c, "intent_min_reading_gap", 0.15),
                max_open_questions=getattr(c, "intent_max_open_questions", 3))
        except Exception:  # noqa: BLE001 — without it she is back to the opener regex
            return None

    def _build_icl(self, c: Any) -> Optional[InContextLearner]:
        if not getattr(c, "icl_enabled", True):
            return None
        try:
            return InContextLearner(self, min_demos=getattr(c, "icl_min_demos", 2),
                                    max_demos=getattr(c, "icl_max_demos", 24),
                                    remember=getattr(c, "icl_remember", True))
        except Exception:  # noqa: BLE001 — without it a demonstration is only a thing she saw
            return None

    @staticmethod
    def _build_semantics(c: Any) -> Optional[SemanticSpace]:
        if not getattr(c, "semantics_enabled", True):
            return None
        try:
            return SemanticSpace(
                dim=getattr(c, "semantics_dim", 64),
                min_count=getattr(c, "semantics_min_count", 3),
                grade_floor=getattr(c, "semantics_grade_floor", 0.25),
                train_budget_s=getattr(c, "semantics_train_budget_s", 0.5),
                train_every=getattr(c, "semantics_train_every", 32),
                max_vocab=getattr(c, "semantics_max_vocab", 8192),
                max_relations=getattr(c, "semantics_max_relations", 8192),
                seed=getattr(c, "seed", 42))
        except Exception:  # noqa: BLE001 — without it her concepts are labels again, as in V.01
            return None

    def _wire_semantics(self, c: Any) -> None:
        """Give the graph its semantic prior. Silent no-op when the space is absent."""
        try:
            if self.semantics is None:
                return
            self.graph.attach_semantics(
                self.semantics,
                birth_links=getattr(c, "semantics_birth_links", 2),
                birth_weight=getattr(c, "semantics_birth_weight", 0.1),
                synonym_bridge=getattr(c, "semantics_synonym_bridge", 0.5),
                merge_min_weight=getattr(c, "semantics_merge_min_weight", 0.9))
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _build_metacog(c: Any) -> Optional[RecursiveMetaCognition]:
        if not getattr(c, "metacog_enabled", True):
            return None
        try:
            return RecursiveMetaCognition(
                reliability_lr=c.reliability_lr, calibration_window=c.calibration_window,
                min_samples=c.metacog_min_samples, overconfidence_gap=c.overconfidence_gap)
        except Exception:  # noqa: BLE001 — meta-cognition is a capability, never required
            return None

    def _build_workspace(self, c: Any) -> Optional[NyxWorkspace]:
        if not getattr(c, "workspace_enabled", True):
            return None
        try:
            wanted = list(getattr(c, "specialists", []) or [])
            seated = [_SPECIALISTS[name]() for name in wanted if name in _SPECIALISTS]
            return NyxWorkspace(specialists=seated, metacog=self.metacog,
                                capacity=c.workspace_capacity,
                                access_threshold=c.access_threshold)
        except Exception:  # noqa: BLE001 — without a workspace she still perceives and recalls
            return None

    def _build_hybrid(self, c: Any) -> Optional[SymbolicSubsymbolicFusion]:
        if not getattr(c, "hybrid_enabled", True):
            return None
        try:
            return SymbolicSubsymbolicFusion(
                grounding=getattr(c, "grounding_check", True),
                min_grounding_overlap=getattr(c, "min_grounding_overlap", 0.5),
                semantics=self.semantics, prover=getattr(self, "prover", None))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _build_ground(c: Any) -> Optional[WorldGrounding]:
        if not getattr(c, "ground_enabled", True):
            return None
        try:
            return WorldGrounding(
                read_unknown=getattr(c, "ground_read_unknown", True),
                max_new_per_turn=getattr(c, "ground_max_new_per_turn", 4),
                bind_to_graph=getattr(c, "ground_bind_to_graph", True))
        except Exception:  # noqa: BLE001 — without grounding her words are honestly unanchored
            return None

    @staticmethod
    def _build_dialogue(c: Any) -> Optional[Dialogue]:
        if not getattr(c, "dialogue_enabled", True):
            return None
        try:
            soul = None
            try:
                from nyxara.identity.soul import Soul
                soul = Soul()
            except Exception:  # noqa: BLE001 — she can speak without an identity fragment
                soul = None
            return Dialogue(require_fluent_surface=getattr(c, "require_fluent_surface", True),
                            soul=soul, max_tokens=getattr(c, "reply_max_tokens", 220))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _build_will(c: Any) -> Optional[SovereignWill]:
        if not getattr(c, "will_enabled", True):
            return None
        try:
            from nyxara.nyx.will import EntropySource
            return SovereignWill(
                entropy=EntropySource(prefer=getattr(c, "entropy_source", "auto")),
                temperature=getattr(c, "will_temperature", 0.6),
                may_decline=getattr(c, "will_may_decline", True),
                record=getattr(c, "will_record", True))
        except Exception:  # noqa: BLE001 — without it she computes the maximum, as code does
            return None

    @staticmethod
    def _build_chronos(c: Any) -> Optional[TemporalCausalMatrix]:
        if not getattr(c, "chronos_enabled", True):
            return None
        try:
            return TemporalCausalMatrix(
                max_branches=c.chronos_max_branches, horizon=c.chronos_horizon,
                budget_ms=c.chronos_budget_ms, risk_aversion=c.chronos_risk_aversion,
                min_coverage=c.chronos_min_coverage, seed=c.seed)
        except Exception:  # noqa: BLE001 — without it she simply cannot see ahead
            return None

    def _build_aura(self, c: Any) -> Optional[AwarenessField]:
        if not getattr(c, "aura_enabled", True):
            return None
        try:
            field_ = AwarenessField(
                self, max_events_per_min=getattr(c, "aura_max_events_per_min", 60),
                surprise_gate=getattr(c, "aura_surprise_gate", 0.5),
                scan=getattr(c, "aura_scan", True),
                max_text=getattr(c, "aura_max_text", 2000))
            if getattr(c, "aura_host_sensors", True):
                field_.register_host_sensors()
            return field_
        except Exception:  # noqa: BLE001 — without it nothing arrives on its own
            return None

    def _build_synergy(self, c: Any) -> Optional[HiveSynapse]:
        if not getattr(c, "synergy_enabled", False):
            return None
        try:
            return HiveSynapse(self, node_id=getattr(c, "node_id", "") or "node",
                               dim=c.holo_dim, seed=c.seed)
        except Exception:  # noqa: BLE001 — without it she is simply one instance
            return None

    def _build_eternal(self, c: Any) -> Optional[Continuity]:
        if not getattr(c, "eternal_enabled", False):
            return None
        try:
            return Continuity(self, node_id=getattr(c, "node_id", "") or "node",
                              nodes=list(getattr(c, "eternal_nodes", []) or []),
                              snapshot_every_s=getattr(c, "eternal_snapshot_every_s", 60.0))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _build_nexus(c: Any) -> Optional[OntologyGenesis]:
        if not getattr(c, "nexus_enabled", True):
            return None
        try:
            return OntologyGenesis(
                translate_always=getattr(c, "nexus_translate_always", True),
                max_notations=getattr(c, "nexus_max_notations", 16),
                max_vm_steps=getattr(c, "nexus_max_vm_steps", 100_000))
        except Exception:  # noqa: BLE001 — without it she only ever uses borrowed symbols
            return None

    def _build_episteme(self, c: Any) -> Optional[AutonomousDiscovery]:
        if not getattr(c, "episteme_enabled", True):
            return None
        try:
            return AutonomousDiscovery(
                self, trials=getattr(c, "episteme_trials", 20),
                holdout_fraction=getattr(c, "episteme_holdout_fraction", 0.3),
                max_holdout_error=getattr(c, "episteme_max_holdout_error", 0.05),
                budget_ms=getattr(c, "episteme_budget_ms", 4000.0), seed=c.seed)
        except Exception:  # noqa: BLE001 — without it she only knows what she was told
            return None

    def _build_omni(self, c: Any) -> Optional[MetamorphicCompiler]:
        if not getattr(c, "omni_enabled", True):
            return None
        try:
            return MetamorphicCompiler(
                self, hot_swap=getattr(c, "omni_hot_swap", True),
                min_speedup=getattr(c, "omni_min_speedup", 1.2),
                max_forges_per_hour=getattr(c, "omni_max_forges_per_hour", 2),
                cases=getattr(c, "omni_cases", 24),
                scan_per_beat=getattr(c, "omni_scan_per_beat", 6), seed=c.seed)
        except Exception:  # noqa: BLE001 — without it she stays exactly as fast as she was
            return None

    def _build_car(self, c: Any) -> Optional[ContinuousAutonomousReasoning]:
        if not getattr(c, "car_enabled", True):
            return None
        try:
            return ContinuousAutonomousReasoning(
                self, budget_ms=getattr(c, "car_budget_ms", 250.0),
                interval_s=getattr(c, "car_interval_s", 30.0),
                max_questions=getattr(c, "car_max_questions", 3))
        except Exception:  # noqa: BLE001 — without it she simply does not think between prompts
            return None

    # ---- perception ------------------------------------------------------ #
    def perceive(self, text: str, *, remember: bool = True) -> NyxPercept:
        """Rewire the graph from ``text``, lay the turn down in memory, and bring back context.

        This is the whole of pillar 1 in one call: co-activation reshapes the network, and the
        associative context that comes back is selected by *relevance*, not by recency in a
        buffer. Fail-soft to an empty percept.
        """
        out = NyxPercept()
        try:
            text = str(text or "")
            if not text.strip():
                return out
            self.turns += 1
            # Read the turn into the semantic space *before* activating, so a concept born
            # this tick is placed next to what it means rather than born isolated. Training
            # itself is not done here: gradient descent is not a thought, and does not get to
            # cost one — it happens on a maintenance beat, in :meth:`consolidate_meaning`.
            if self.semantics is not None:
                self.semantics.train_on(text, source="turn", train=False)
            act: Activation = self.graph.activate(text)
            out.concepts = list(act.direct)
            out.born = list(act.born)
            out.strengthened = act.strengthened
            out.pruned = act.pruned
            out.spread = [n for n, _w in act.spread]

            # Ground the words before reasoning with them: an unknown word met in a sentence
            # that describes it can be learned here and now, and a word she has no referent for
            # is reported ungrounded rather than quietly treated as understood.
            if self.ground is not None:
                out.grounded = self.ground.understand(out.concepts, text=text, graph=self.graph)

            # Recall *before* writing, so the turn does not simply recall itself.
            rec = self.memory.recall(text, k=self.config.recall_k)
            out.recall = rec
            out.context = self.memory.context(text, k=self.config.recall_k)
            # A question is not knowledge — storing it only makes it its own best match when
            # asked again. A statement is something she was *told*, which is real evidence.
            if remember and not is_question(text):
                self.memory.remember(f"turn-{self.turns}", text, kind="told")
            return out
        except Exception:  # noqa: BLE001 — a perceive failure never breaks a turn
            return out

    # ---- the cognitive cycle --------------------------------------------- #
    def think(self, stimulus: str, *, remember: bool = True,
              goals: Optional[Dict[str, float]] = None) -> NyxThought:
        """One whole thought: perceive → ground → deliberate → reflect.

        This is the cycle the rest of NYX hangs off. The specialists each get the same grounded
        situation; exactly one bid reaches the conscious bottleneck; and meta-cognition records
        *why* that one won, so ``/nyx why`` can answer from a real trace rather than a story.

        The outcome — whether the winner was actually right — is fed back separately through
        :meth:`resolve`, because it is usually not knowable in the same instant.
        """
        out = NyxThought(stimulus=str(stimulus or ""))
        try:
            if not out.stimulus.strip():
                return out
            # Perceive without writing: the episode is laid down *after* deliberating, so what
            # gets remembered is "I was asked X and concluded Y" rather than the bare question.
            # Storing a naked question would make it its own best match when asked again.
            # L-OMNI's one input: where the time in a real thought of hers actually goes. No
            # workload is synthesised and no function is nominated by hand — she optimises the
            # module she is measurably slowest in.
            clock = _Latency(self.omni)
            with clock.at("nyxara.nyx.graph"):
                out.percept = self.perceive(out.stimulus, remember=False)
            out.cycle_id = f"cycle-{self.turns}"

            # Three layers that only *watch* a turn go past: the causal engine notes which
            # concepts were seen together (co-occurrence is its evidence, never its answer), the
            # weaver reads any typed relation the sentence states outright, and the stream takes
            # the turn in so a briefing has something to compress. None of them can change what
            # she concludes here.
            if self.causal is not None:
                self.causal.observe(out.percept.concepts)
            if self.weaver is not None:
                self.weaver.read(out.stimulus)
            if self.stream is not None:
                self.stream.absorb_turn(out.stimulus)

            # Read what was actually asked for before deciding anything. Everything downstream
            # — the ordering constraint, whether a tool is even wanted, whether she should be
            # answering at all or asking — hangs off this.
            if self.intent is not None:
                with clock.at("nyxara.nyx.intent"):
                    out.intent = self.intent.read(out.stimulus)

            # In-context learning happens *before* deliberation, because a turn that carries
            # demonstrations is teaching her a procedure, and the specialists should get to
            # compete with that procedure already in hand. When the block names its own probe,
            # the induced answer is the turn's answer and there is nothing to deliberate about.
            if self.icl is not None:
                with clock.at("nyxara.nyx.icl"):
                    out.induced = self.icl.learn(out.stimulus)
                if out.induced is not None and out.induced.answer:
                    out.deliberation = self._induced_only(out.induced)
                    if remember:
                        self.memory.remember(f"turn-{self.turns}", out.answer,
                                             kind="conclusion", cue=out.stimulus)
                    return out

            if self.workspace is None:
                return out
            situation = Situation(
                stimulus=out.stimulus, concepts=out.percept.concepts,
                context=out.percept.context, recall=out.percept.recall,
                novelty=out.percept.novelty, brain=self)
            with clock.at("nyxara.nyx.workspace"):
                out.deliberation = self.workspace.deliberate(situation, goals=goals)

            # Measure how sure she actually is across *every* candidate, not just the winner.
            # The workspace decides what reaches awareness; this says whether anything really
            # dominated, and keeps the runners-up alive with real probabilities.
            if out.deliberation is not None and out.deliberation.proposals \
                    and getattr(self.config, "superposition_enabled", True):
                state = SolutionSuperposition.from_proposals(
                    out.deliberation.proposals,
                    collapse_threshold=self.config.collapse_threshold,
                    max_candidates=self.config.max_candidates)
                # L-CHRONOS: on a *decision*, simulate how each option turns out and fold the
                # ranking in as evidence. It contributes nothing when the turn is a question of
                # fact, or when the world model has learned too little to see ahead — which is
                # the point: no foresight is better than fabricated foresight.
                if self.chronos is not None and self.chronos.applies(out.stimulus):
                    with clock.at("nyxara.nyx.chronos"):
                        out.futures = self.chronos.explore(out.deliberation.proposals)
                    evidence = self.chronos.evidence(out.futures)
                    if evidence:
                        state.observe(evidence)
                with clock.at("nyxara.nyx.superpose"):
                    out.collapsed = state.collapse()

                # L-PSYCHE-QUANTUM: where the decision is genuinely open — several answers
                # still live — she *chooses* rather than taking the maximum, sampling from her
                # own preferences with physically-sourced entropy. Deliberately not applied to
                # a settled or verified answer: truth is not a preference, and a whim must not
                # displace something she checked.
                if self.will is not None and not out.collapsed.decided and not out.verified:
                    out.choice = self.will.choose(out.deliberation.proposals)
                    if out.choice.picked and out.choice.picked != out.deliberation.winner.source:
                        chosen = next((p for p in out.deliberation.proposals
                                       if p.source == out.choice.picked), None)
                        if chosen is not None:
                            out.deliberation.winner = chosen

            if self.metacog is not None and out.deliberation is not None:
                out.assessment = self.metacog.observe_cycle(
                    cycle_id=out.cycle_id, winner=out.deliberation.winner,
                    considered=out.deliberation.proposals)

            # Check what she is about to say against her own engines, and let the verdict
            # credit or debit the specialist that said it — the loop closes with no human.
            if self.hybrid is not None and out.answer:
                with clock.at("nyxara.nyx.hybrid"):
                    out.verification = self.hybrid.check_and_learn(self, out)

            # The winner has won a competition for salience. That is not the same as being
            # right, and until now nothing attacked it between winning and shipping. A
            # refutation makes her abstain outright; a lesser objection cuts the confidence she
            # reports — which is the whole point, not a failure of the debate.
            if self.dialectic is not None and out.answer \
                    and getattr(self.config, "dialectic_review_turns", True):
                with clock.at("nyxara.nyx.dialectic"):
                    out.debate = self.dialectic.review(out)
                self._apply_debate(out)

            # What is worth remembering from a turn is what she *concluded*; the question is
            # provenance. A turn she had no answer to writes nothing here — the concepts are
            # already in the graph, and storing the bare question would make it its own best
            # match, so re-asking would echo the question back as the answer.
            if remember and out.answer:
                self.memory.remember(
                    f"turn-{self.turns}", out.answer,
                    kind=("conclusion" if out.verified else "episode"),
                    cue=out.stimulus)
            return out
        except Exception:  # noqa: BLE001 — a thought that fails is empty, never fatal
            return out

    @staticmethod
    def _apply_debate(out: NyxThought) -> None:
        """Let the debate's verdict actually reach the answer. A review nothing obeys is theatre."""
        try:
            debate, winner = out.debate, out.winner
            if debate is None or winner is None or not debate.debated:
                return
            if debate.abstained:
                # She says nothing rather than something she cannot defend.
                winner.content = ""
                winner.confidence = 0.0
            elif debate.confidence_after < debate.confidence_before:
                winner.confidence = float(debate.confidence_after)
        except Exception:  # noqa: BLE001 — a debate that cannot be applied leaves the answer be
            return

    def converse(self, text: str, *, goals: Optional[Dict[str, float]] = None) -> Reply:
        """Think, then say it. The content is hers; a fluent model only phrases it.

        This is the path the Master actually talks to. When no fluent language surface is
        installed she still answers — in her own words, with a note saying so — rather than
        letting a fallback n-gram babble in her name.
        """
        try:
            thought = self.think(text, goals=goals)
            if self.dialogue is None:
                return Reply(text=thought.answer, source=(thought.winner.source
                                                          if thought.winner else ""),
                             confidence=thought.confidence, verified=thought.verified,
                             decided=thought.decided)
            return self.dialogue.respond(thought, brain=self)
        except Exception:  # noqa: BLE001 — she always says something, never crashes mid-sentence
            return Reply()

    def resolve(self, thought: Any, *, correct: float) -> Any:
        """Tell meta-cognition how a thought actually turned out (0.0 … 1.0).

        No human is required in this loop: a derivation that checked out, a claim that survived
        a grounding check, or an answer that was acted on are all measurable signals.
        """
        try:
            if self.metacog is None or thought is None:
                return None
            cycle_id = thought if isinstance(thought, str) else getattr(thought, "cycle_id", "")
            if not cycle_id:
                return None
            return self.metacog.observe_outcome(cycle_id=cycle_id, correct=correct)
        except Exception:  # noqa: BLE001
            return None

    def why(self, *, k: int = 1) -> List[Any]:
        """The most recent reasoning traces — what won, on what evidence, over what."""
        try:
            return self.metacog.why(k=k) if self.metacog is not None else []
        except Exception:  # noqa: BLE001
            return []

    def remember(self, key: str, text: str, *, kind: str = "episode") -> None:
        """Lay something down deliberately — a conclusion, a fact, a grounded concept."""
        try:
            self.memory.remember(key, text, kind=kind)
            self.graph.activate(text)
        except Exception:  # noqa: BLE001
            pass

    def recall(self, cue: str, *, k: Optional[int] = None) -> Optional[Recall]:
        """Content-addressed recall — no token window is consulted."""
        try:
            return self.memory.recall(cue, k=int(k or self.config.recall_k))
        except Exception:  # noqa: BLE001
            return None

    def related(self, concept: str, *, k: int = 8) -> List[Any]:
        """What the graph currently associates with a concept, strongest first."""
        try:
            return self.graph.neighbours(concept, k=k)
        except Exception:  # noqa: BLE001
            return []

    def gaps(self, *, k: int = 5) -> List[str]:
        """Concepts she keeps meeting but has never connected — the honest edge of her knowing."""
        try:
            return self.graph.gaps(k=k)
        except Exception:  # noqa: BLE001
            return []

    # ---- thinking between prompts ---------------------------------------- #
    def tick(self, *, oversight: Any = None) -> Optional[CarStep]:
        """One beat of the shared clock: the world arrives, and she thinks on her own.

        Called from the heartbeat and the autonomic loop. No new thread; oversight is honoured
        by both halves, so a paused or scrammed mind neither senses nor wonders. Sensing is
        capped per minute and thinking at most once per ``car_interval_s``.
        """
        try:
            if self.aura is not None:
                self.aura.beat(oversight=oversight)   # the world arrives on the same clock
            self._episteme_beat(oversight)
            self._omni_beat(oversight)
            if self.synergy is not None:
                self.synergy.beat(oversight=oversight)
            if self.eternal is not None:
                self.eternal.beat(oversight=oversight)
            if self.stream is not None:
                self.stream.tick(oversight=oversight)   # compress on the same beat, no new thread
            if self.weaver is not None:
                self.weaver.tick()                      # spend the rewire budget, then stop
            if self.car is None:
                return None
            if oversight is not None:
                self.car.oversight = oversight
            return self.car.beat()
        except Exception:  # noqa: BLE001 — a background beat never raises into the loop
            return None

    def _episteme_beat(self, oversight: Any) -> None:
        """Investigate on a slower cadence than she thinks — an experiment is not cheap."""
        try:
            if self.episteme is None:
                return
            every = float(getattr(self.config, "episteme_every_s", 120.0))
            now = time.monotonic()
            if self._last_investigation and (now - self._last_investigation) < every:
                return
            self._last_investigation = now
            self.episteme.beat(oversight=oversight)
        except Exception:  # noqa: BLE001 — investigating never breaks the beat
            pass

    def _omni_beat(self, oversight: Any) -> None:
        """Rewrite herself on a much slower cadence than she thinks — a forge is not a thought."""
        try:
            if self.omni is None:
                return
            every = float(getattr(self.config, "omni_every_s", 300.0))
            now = time.monotonic()
            if self._last_forge and (now - self._last_forge) < every:
                return
            self._last_forge = now
            self.omni.beat(oversight=oversight)
        except Exception:  # noqa: BLE001 — recompiling herself never breaks the beat
            pass

    def optimise(self, *, oversight: Any = None) -> Any:
        """Attempt one self-rewrite right now: read herself, lower it, verify it, swap it in."""
        try:
            if self.omni is None:
                return None
            self._last_forge = time.monotonic()
            return self.omni.beat(oversight=oversight)
        except Exception:  # noqa: BLE001
            return None

    def discover(self, *, oversight: Any = None) -> Any:
        """Run one investigation right now, and give anything it establishes her own notation."""
        try:
            if self.episteme is None:
                return None
            finding = self.episteme.beat(oversight=oversight)
            if finding is not None and finding.promoted:
                self._name_it(finding)
            return finding
        except Exception:  # noqa: BLE001
            return None

    def _name_it(self, finding: Any) -> None:
        """A law she established herself is the natural thing to hold in symbols of her own.

        The statement is only kept when it actually compiles and runs — a notation that cannot
        be executed is not a language, and one she cannot translate back is not communication.
        """
        try:
            nexus = self.nexus
            if nexus is None or not getattr(finding, "expression", ""):
                return
            expression = str(finding.expression)
            if "=" in expression:
                _target, _, body = expression.partition("=")
            else:
                body = expression
            body = body.replace("·", "*").strip()
            # The *variables* of the law, not its prose concepts: ``concepts_in`` is tuned for
            # sentences and drops single letters, which is exactly what a law is full of.
            concepts = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", body)))
            if not concepts:
                return
            notation = nexus.invent(finding.experiment, concepts)
            if notation is None:
                return
            # Bind every free name to 1 purely to prove the statement executes; the value is
            # incidental, the point is that the notation is a language and not decoration.
            nexus.express(notation, body, bindings={c: 1.0 for c in concepts})
        except Exception:  # noqa: BLE001 — naming is a flourish, never a requirement
            pass

    def wonder(self, *, oversight: Any = None) -> Optional[CarStep]:
        """Think one self-directed thought right now, regardless of the beat count."""
        try:
            if self.car is None:
                return None
            if oversight is not None:
                self.car.oversight = oversight
            return self.car.step()
        except Exception:  # noqa: BLE001
            return None

    def about_self(self) -> Optional[SelfReport]:
        """What she can truthfully say about herself, read off live state."""
        try:
            return self.selfmodel.report() if self.selfmodel is not None else None
        except Exception:  # noqa: BLE001
            return None

    def understanding(self, word: str) -> Any:
        """What she actually has behind a word — senses, neighbours, and whether it is grounded."""
        try:
            return self.ground.understanding(word) if self.ground is not None else None
        except Exception:  # noqa: BLE001
            return None

    def understand(self, text: str) -> Optional[LinguaRead]:
        """How the turn was *written*: scripts, languages, code-mixing, register, concepts.

        Distinct from :meth:`understanding`, which asks what a single word is *about*. This
        one is form, that one is meaning — and NYX V.01 had neither for anything outside
        ASCII. Returns ``None`` only when the tongue is disabled.
        """
        try:
            return self.lingua.read(text) if self.lingua is not None else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _induced_only(induced: Learned) -> Deliberation:
        """A one-bidder deliberation carrying the induced answer.

        The turn asked a question and supplied the rule to answer it with; there is nothing for
        the specialists to disagree about. ``verifiable`` is set only when the program predicted
        a demonstration held out of the induction — fit alone is not transfer.
        """
        proposal = Proposal(
            source="skill", content=induced.answer,
            confidence=float(induced.confidence), verifiable=bool(induced.generalized),
            novelty=0.5, urgency=0.2, tags=frozenset({"induction", "procedure"}),
            evidence=[induced.program] if induced.program else [],
            rationale=("induced from the demonstrations in this turn"
                       + (" and confirmed on a held-out one" if induced.generalized else "")))
        return Deliberation(winner=proposal, proposals=[proposal],
                            salience=float(induced.confidence), coalition=["skill"])

    def attach_kernel(self, *, tools: Any = None, knowledge: Any = None,
                      core: Any = None) -> None:
        """Give her the live toolset and knowledge base the kernel holds.

        Called once by :class:`~nyxara.kernel.orchestrator.NyxaraCore` after construction. Until
        this happens her brain is *blind* to tools — which is exactly the V.01 state, and is a
        wiring gap, never a permission one: the tools were always registered and always allowed.
        """
        try:
            self.tools = tools
            self.knowledge = knowledge
            self.core = core
            if self.hands is not None and tools is not None:
                self.hands._registry = tools
                self.hands._tried_router = False
        except Exception:  # noqa: BLE001
            return

    def bind_structure(self, name: str, roles: Optional[Dict[str, str]] = None, *,
                       sequence: Optional[List[str]] = None) -> Any:
        """Hold a whole role-filler structure in one vector — not a bag of words.

        "The engine turns the flywheel" and "the flywheel turns the engine" have identical
        concepts and opposite meanings; this is what tells them apart.
        """
        try:
            return self.vsa.encode(name, roles, sequence=sequence) \
                if self.vsa is not None else None
        except Exception:  # noqa: BLE001
            return None

    def analogy(self, a: str, b: str, c: str) -> Optional[VsaRetrieved]:
        """``a : b :: c : ?`` over structures she holds — no training, no examples."""
        try:
            return self.vsa.analogy(a, b, c) if self.vsa is not None else None
        except Exception:  # noqa: BLE001
            return None

    def prove(self, claim: str) -> Optional[ProofCertificate]:
        """A machine-checkable verdict, or an honest report that there is none to be had.

        Most of what she says is not formally expressible, and for those this returns
        ``INEXPRESSIBLE`` rather than inventing a verdict — that refusal is the point.
        """
        try:
            return self.prover.prove(claim) if self.prover is not None else None
        except Exception:  # noqa: BLE001
            return None

    def why_causal(self, cause: str, effect: str, *, question: str = "") -> Optional[CausalAnswer]:
        """Would the effect still happen if she intervened? Not "do these two co-occur".

        The answer prints ``do(X)`` beside the Hebbian weight, so association and causation are
        never blurred into one number.
        """
        try:
            return self.causal.ask(cause, effect, question=question) \
                if self.causal is not None else None
        except Exception:  # noqa: BLE001
            return None

    def root_cause(self, *, k: int = 3) -> List[CausalAnswer]:
        """The upstream causes she can actually identify — never a guess dressed as a cause."""
        try:
            return self.causal.root_cause(k=k) if self.causal is not None else []
        except Exception:  # noqa: BLE001
            return []

    def weave(self, a: str = "", b: str = "", *, type: str = "co-occurs") -> Optional[TypedEdge]:
        """Lay a typed, directed edge over the graph without touching a Hebbian weight."""
        try:
            if self.weaver is None:
                return None
            if not a and not b:
                self.weaver.tick()
                return None
            return self.weaver.relate(a, b, type=type)
        except Exception:  # noqa: BLE001
            return None

    def debate(self, claim: str, *, confidence: float = 0.5,
               verified: bool = False) -> Optional[Debate]:
        """Attack a claim before it ships. Three outcomes, and one of them is silence."""
        try:
            return self.dialectic.debate(claim, confidence=confidence, verified=verified) \
                if self.dialectic is not None else None
        except Exception:  # noqa: BLE001
            return None

    def digest(self, *, force: bool = False) -> Dict[str, Digest]:
        """Compress what arrived, at every scale at once. Rides the beat; no second thread."""
        try:
            return self.stream.tick(force=force) if self.stream is not None else {}
        except Exception:  # noqa: BLE001
            return {}

    def catch_up(self) -> str:
        """One paragraph for someone who has just come back."""
        try:
            return self.stream.briefing() if self.stream is not None else ""
        except Exception:  # noqa: BLE001
            return ""

    def decompose(self, request: str) -> Optional[Plan]:
        """One instruction → a DAG, with a risk class and a way back priced per node.

        Nothing runs here. This is the plan the Master gets to look at first.
        """
        try:
            return self.goals.plan(request) if self.goals is not None else None
        except Exception:  # noqa: BLE001
            return None

    def carry_out(self, request: str = "", *, oversight: Any = None,
                  dry_run: bool = False) -> Optional[Run]:
        """Plan it and run it — or, with no request, continue the newest unfinished plan.

        A node she cannot finish is reported, never silently skipped, and the run says
        *incomplete* rather than *done*.
        """
        try:
            if self.goals is None:
                return None
            if not str(request or "").strip():
                return self.goals.resume(oversight=oversight)
            return self.goals.pursue(request, oversight=oversight, dry_run=dry_run)[1]
        except Exception:  # noqa: BLE001
            return None

    def receive_frame(self, frame: Any) -> Optional[Received]:
        """Take meaning as structure rather than as prose — the tokenizer is not involved.

        There is no mind reading here and none is claimed. What is real is that past the
        frame, nothing is lost.
        """
        try:
            return self.telepathy.receive(frame) if self.telepathy is not None else None
        except Exception:  # noqa: BLE001
            return None

    def emit_frame(self) -> Optional[Frame]:
        """Her own state as structure, for another node or tool to consume without prose."""
        try:
            return self.telepathy.emit() if self.telepathy is not None else None
        except Exception:  # noqa: BLE001
            return None

    def pursue(self, *, oversight: Any = None, force: bool = False) -> Any:
        """One budgeted step on the goal she most wants to advance. ``None`` when she does not.

        Unlike :meth:`wonder`, which thinks one thought and forgets it, this advances something
        she set herself and that survives the session.
        """
        try:
            return self.agenda.beat(oversight=oversight, force=force) \
                if self.agenda is not None else None
        except Exception:  # noqa: BLE001
            return None

    def briefing(self) -> Optional[Briefing]:
        """What she would tell the Master unprompted: chose, found, stuck, declined."""
        try:
            return self.agenda.brief() if self.agenda is not None else None
        except Exception:  # noqa: BLE001
            return None

    def set_goal(self, subject: str, *, kind: str = "understand") -> Optional[Goal]:
        """A goal the Master hands her. The Master is sovereign over the agenda."""
        try:
            return self.agenda.add(subject, kind=kind) if self.agenda is not None else None
        except Exception:  # noqa: BLE001
            return None

    def evolve(self, *, oversight: Any = None, force: bool = False,
               ignore_sample_floor: bool = False) -> Optional[OmegaStep]:
        """One budgeted change to the constants she thinks with — gauntleted and reversible.

        ``force`` means *run now* — it waives the cadence, not the sample floor. Below the
        floor she still declines, because hill-climbing on a handful of turns is fitting
        noise, and waiving that needs its own explicit ``ignore_sample_floor``. She tunes her
        knobs, never her constitution.
        """
        try:
            if self.omega is None:
                return None
            return self.omega.evolve(oversight=oversight, force=force,
                                     ignore_sample_floor=ignore_sample_floor)
        except Exception:  # noqa: BLE001
            return None

    def invent_axioms(self, problem: str, *, name: str = "") -> Optional[Genesis]:
        """Write down a new axiom system for a problem no domain covers, and check it.

        Consistency is a **bounded** model search: a model found proves consistency at that
        size; none found proves nothing, and is reported as exactly that. She invents formal
        systems, not physical truths.
        """
        try:
            return self.axiom.invent(problem, name=name) if self.axiom is not None else None
        except Exception:  # noqa: BLE001
            return None

    def foresee(self, action: str, args: Optional[Dict[str, Any]] = None) -> Optional[Verdict]:
        """What would happen if she did this — before she does it.

        The reversibility class, the blast radius, what git would actually do, and the tail
        risk across bounded rollouts. Her own foresight: it asks nobody for permission.
        """
        try:
            if self.consequence is None:
                return None
            return self.consequence.check(action, args, registry=getattr(self, "tools", None))
        except Exception:  # noqa: BLE001
            return None

    def author_code(self, spec: str, *, name: str = "", load: Optional[bool] = None,
                    oversight: Any = None) -> Optional[Authored]:
        """Write code from a spec, through the full gauntlet — or refuse and name why.

        L-OMNI lowers functions she has already measured as slow; this writes ones that did
        not exist. Bounded by what the synthesiser can *derive*: outside those families the
        result carries a refusal naming what was not derived, never a plausible-looking file.
        """
        try:
            if self.author is None:
                return None
            return self.author.write(spec, name=name, load=load, oversight=oversight)
        except Exception:  # noqa: BLE001
            return None

    def act(self, request: str, *, tool: str = "", args: Optional[Dict[str, Any]] = None,
            oversight: Any = None, dry_run: bool = False) -> Optional[Reach]:
        """Reach for a tool, through the registry's unchanged pipeline. Never around it.

        Judges first whether the turn wants a tool at all — a question of fact does not — and
        stops dead when oversight has paused or scrammed her.
        """
        try:
            if self.hands is None:
                return None
            return self.hands.reach(request, tool=tool, args=args,
                                    intent=self.intent_of(request),
                                    oversight=oversight, dry_run=dry_run)
        except Exception:  # noqa: BLE001
            return None

    def reason_about(self, question: str) -> Optional[Chain]:
        """Work down the reasoning tiers and return the chain, with its honest label.

        Outside the four checkable domains this is a *plausible* chain, not a derivation, and
        :meth:`~nyxara.nyx.reason.Chain.label` says so on every answer. That is the thing V.01
        could not do: it had no tier below "derived", so it produced silence instead.
        """
        try:
            return self.reason.solve(question) if self.reason is not None else None
        except Exception:  # noqa: BLE001
            return None

    def intent_of(self, text: str) -> Optional[Intent]:
        """What was actually asked for: mood, actions, ordering, negation, what is unclear.

        The structure V.01 had no way to represent. Read ``ambiguous`` and
        ``clarifying_question()`` before acting on ``goal`` — a live second reading means she
        should ask, not choose.
        """
        try:
            return self.intent.read(text) if self.intent is not None else None
        except Exception:  # noqa: BLE001
            return None

    def learn_from(self, text: str, *, name: str = "") -> Optional[Learned]:
        """Induce a procedure from demonstrations in this turn, and answer its probe.

        The other kind of learning: not a weight, not a memory write, but a *program* she did
        not have before the turn started. Outside her operation set the result carries a
        refusal in plain words rather than a plausible-looking guess.
        """
        try:
            return self.icl.learn(text, name=name) if self.icl is not None else None
        except Exception:  # noqa: BLE001
            return None

    def meaning(self, a: str, b: str = "") -> Any:
        """What she can honestly say about a concept's meaning — or about a pair of them.

        With one argument: the nearest concepts she can justify, each labelled with the rung
        that justified it. With two: a :class:`~nyxara.nyx.semantics.Similarity`, whose
        ``known`` flag distinguishes *I do not know* from *unrelated* — the distinction V.01
        could not make.
        """
        try:
            if self.semantics is None:
                return None
            return self.semantics.similarity(a, b) if b else \
                self.semantics.similar(a, candidates=list(self.graph.nodes) or None)
        except Exception:  # noqa: BLE001
            return None

    def teach_meaning(self, a: str, b: str, *, kind: str = "synonym",
                      weight: float = 0.9) -> bool:
        """Tell her two words are the same thing. The relational rung, filled by hand."""
        try:
            if self.semantics is None:
                return False
            return bool(self.semantics.teach(a, b, kind=kind, weight=weight, source="taught"))
        except Exception:  # noqa: BLE001
            return False

    def consolidate_meaning(self, *, budget_s: Optional[float] = None) -> Dict[str, Any]:
        """Spend a real gradient budget on her own corpus, then fuse what turned out to be one.

        Called from a maintenance beat, never from a turn. Node fusion is bounded per call and
        written to the graph's merge ledger, so every structural change stays reversible.
        """
        out: Dict[str, Any] = {"trained": False}
        try:
            if self.semantics is None:
                return out
            out = dict(self.semantics.consolidate(budget_s=budget_s) or {})
            per_beat = int(getattr(self.config, "semantics_merge_per_beat", 1))
            if getattr(self.config, "semantics_merge_enabled", True) and per_beat > 0:
                out["merged"] = [list(p) for p in self.graph.auto_merge(limit=per_beat)]
            return out
        except Exception:  # noqa: BLE001
            return out

    def stats(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "graph": self.graph.stats().to_dict(), "memory": self.memory.stats(),
                "turns": self.turns,
                "as_reasoner": bool(getattr(self.config, "as_reasoner", False))}
            if self.lingua is not None:
                out["lingua"] = self.lingua.stats()
            if self.semantics is not None:
                out["semantics"] = self.semantics.stats()
            if self.intent is not None:
                out["intent"] = self.intent.stats()
            if self.reason is not None:
                out["reason"] = self.reason.stats()
            if self.hands is not None:
                out["hands"] = self.hands.stats()
            if self.author is not None:
                out["author"] = self.author.stats()
            if self.consequence is not None:
                out["consequence"] = self.consequence.stats()
            if self.axiom is not None:
                out["axiom"] = self.axiom.stats()
            if self.omega is not None:
                out["omega"] = self.omega.stats()
            if self.agenda is not None:
                out["agenda"] = self.agenda.stats()
            if self.telepathy is not None:
                out["telepathy"] = self.telepathy.stats()
            if self.prover is not None:
                out["prover"] = self.prover.stats()
            if self.vsa is not None:
                out["vsa"] = self.vsa.stats()
            if self.causal is not None:
                out["causal"] = self.causal.stats()
            if self.weaver is not None:
                out["weaver"] = self.weaver.stats()
            if self.dialectic is not None:
                out["dialectic"] = self.dialectic.stats()
            if self.stream is not None:
                out["stream"] = self.stream.stats()
            if self.goals is not None:
                out["goals"] = self.goals.stats()
            if self.icl is not None:
                out["icl"] = self.icl.stats()
            if self.workspace is not None:
                out["workspace"] = self.workspace.stats()
            if self.metacog is not None:
                out["metacog"] = self.metacog.stats()
            if self.ground is not None:
                out["ground"] = self.ground.stats()
            if self.dialogue is not None:
                out["dialogue"] = self.dialogue.stats()
            if self.chronos is not None:
                out["chronos"] = self.chronos.stats()
            if self.will is not None:
                out["will"] = self.will.stats()
            if self.aura is not None:
                out["aura"] = self.aura.stats()
            if self.episteme is not None:
                out["episteme"] = self.episteme.stats()
            if self.omni is not None:
                out["omni"] = self.omni.stats()
            if self.nexus is not None:
                out["nexus"] = self.nexus.stats()
            if self.synergy is not None:
                out["synergy"] = self.synergy.stats()
            if self.eternal is not None:
                out["eternal"] = self.eternal.stats()
            if self.car is not None:
                out["car"] = self.car.stats()
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out = {"graph": self.graph.to_dict(), "memory": self.memory.to_dict(),
               "turns": self.turns}
        if self.lingua is not None:
            out["lingua"] = self.lingua.to_dict()
        if self.semantics is not None:
            out["semantics"] = self.semantics.to_dict()
        if self.icl is not None:
            out["icl"] = self.icl.to_dict()
        if self.hands is not None:
            out["hands"] = self.hands.to_dict()
        if self.author is not None:
            out["author"] = self.author.to_dict()
        if self.axiom is not None:
            out["axiom"] = self.axiom.to_dict()
        if self.omega is not None:
            out["omega"] = self.omega.to_dict()
        if self.agenda is not None:
            out["agenda"] = self.agenda.to_dict()
        if self.telepathy is not None:
            out["telepathy"] = self.telepathy.to_dict()
        if self.vsa is not None:
            out["vsa"] = self.vsa.to_dict()
        if self.causal is not None:
            out["causal"] = self.causal.to_dict()
        if self.weaver is not None:
            out["weaver"] = self.weaver.to_dict()
        if self.dialectic is not None:
            out["dialectic"] = self.dialectic.to_dict()
        if self.stream is not None:
            out["stream"] = self.stream.to_dict()
        if self.goals is not None:
            out["goals"] = self.goals.to_dict()
        if self.metacog is not None:
            out["metacog"] = self.metacog.to_dict()
        if self.ground is not None:
            out["ground"] = self.ground.to_dict()
        if self.aura is not None:
            out["aura"] = self.aura.to_dict()
        if self.episteme is not None:
            out["episteme"] = self.episteme.to_dict()
        if self.omni is not None:
            out["omni"] = self.omni.to_dict()
        if self.nexus is not None:
            out["nexus"] = self.nexus.to_dict()
        if self.car is not None:
            out["car"] = self.car.to_dict()
        return out

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            if not isinstance(d, dict):
                return
            if d.get("graph"):
                self.graph.load_dict(d["graph"])
            if d.get("memory"):
                self.memory.load_dict(d["memory"])
            if d.get("lingua") and self.lingua is not None:
                self.lingua.load_dict(d["lingua"])
            if d.get("semantics") and self.semantics is not None:
                self.semantics.load_dict(d["semantics"])
            if d.get("icl") and self.icl is not None:
                self.icl.load_dict(d["icl"])
            if d.get("hands") and self.hands is not None:
                self.hands.load_dict(d["hands"])
            if d.get("author") and self.author is not None:
                self.author.load_dict(d["author"])
            if d.get("axiom") and self.axiom is not None:
                self.axiom.load_dict(d["axiom"])
            if d.get("omega") and self.omega is not None:
                self.omega.load_dict(d["omega"])
            if d.get("agenda") and self.agenda is not None:
                self.agenda.load_dict(d["agenda"])
            if d.get("telepathy") and self.telepathy is not None:
                self.telepathy.load_dict(d["telepathy"])
            if d.get("vsa") and self.vsa is not None:
                self.vsa.load_dict(d["vsa"])
            if d.get("causal") and self.causal is not None:
                self.causal.load_dict(d["causal"])
            if d.get("weaver") and self.weaver is not None:
                self.weaver.load_dict(d["weaver"])
            if d.get("dialectic") and self.dialectic is not None:
                self.dialectic.load_dict(d["dialectic"])
            if d.get("stream") and self.stream is not None:
                self.stream.load_dict(d["stream"])
            if d.get("goals") and self.goals is not None:
                self.goals.load_dict(d["goals"])
            if d.get("metacog") and self.metacog is not None:
                self.metacog.load_dict(d["metacog"])
            if d.get("ground") and self.ground is not None:
                self.ground.load_dict(d["ground"])
            if d.get("aura") and self.aura is not None:
                self.aura.load_dict(d["aura"])
            if d.get("episteme") and self.episteme is not None:
                self.episteme.load_dict(d["episteme"])
            if d.get("omni") and self.omni is not None:
                self.omni.load_dict(d["omni"])
            if d.get("nexus") and self.nexus is not None:
                self.nexus.load_dict(d["nexus"])
            if d.get("car") and self.car is not None:
                self.car.load_dict(d["car"])
            self.turns = int(d.get("turns", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            pass
