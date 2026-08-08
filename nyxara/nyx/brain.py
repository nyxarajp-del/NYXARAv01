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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nyxara.nyx.aura import AwarenessField
from nyxara.nyx.car import CarStep, ContinuousAutonomousReasoning
from nyxara.nyx.chronos import Futures, TemporalCausalMatrix
from nyxara.nyx.dialogue import Dialogue, Reply
from nyxara.nyx.graph import Activation, DynamicNeuralGraph
from nyxara.nyx.ground import Grounded, WorldGrounding
from nyxara.nyx.holomem import HoloMemory, Recall, Trace
from nyxara.nyx.hybrid import SymbolicSubsymbolicFusion, Verification
from nyxara.nyx.metacog import RecursiveMetaCognition
from nyxara.nyx.modules import (
    CreativeSpecialist,
    DerivationSpecialist,
    EthicsSpecialist,
    GraphSpecialist,
    MemorySpecialist,
    Proposal,
    Situation,
)
from nyxara.nyx.selfmodel import NyxSelfModel, SelfReport
from nyxara.nyx.superpose import Collapsed, SolutionSuperposition
from nyxara.nyx.will import Choice, SovereignWill
from nyxara.nyx.workspace import Deliberation, NyxWorkspace

__all__ = ["NyxPercept", "NyxThought", "NyxBrain"]

# Being *told* something and being *asked* something are different events, and conflating them
# is why a stored question can come back as its own answer. A question is not knowledge; a
# statement the Master makes is.
_QUESTION = re.compile(
    r"^\s*(what|why|how|who|whom|whose|when|where|which|is|are|was|were|do|does|did|can|could|"
    r"will|would|should|shall|may|might|have|has|had|am|tell me|explain)\b", re.I)


def is_question(text: str) -> bool:
    """Was this asked, rather than asserted? Deliberately simple and deterministic."""
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
        self.metacog = self._build_metacog(c)
        self.workspace = self._build_workspace(c)
        self.hybrid = self._build_hybrid(c)
        self.ground = self._build_ground(c)
        self.dialogue = self._build_dialogue(c)
        self.chronos = self._build_chronos(c)
        self.will = self._build_will(c)
        self.aura = self._build_aura(c)
        self.car = self._build_car(c)
        self.selfmodel = NyxSelfModel(self) if getattr(c, "selfmodel_enabled", True) else None
        self.turns = 0

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

    @staticmethod
    def _build_hybrid(c: Any) -> Optional[SymbolicSubsymbolicFusion]:
        if not getattr(c, "hybrid_enabled", True):
            return None
        try:
            return SymbolicSubsymbolicFusion(
                grounding=getattr(c, "grounding_check", True),
                min_grounding_overlap=getattr(c, "min_grounding_overlap", 0.5))
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
            out.percept = self.perceive(out.stimulus, remember=False)
            out.cycle_id = f"cycle-{self.turns}"
            if self.workspace is None:
                return out
            situation = Situation(
                stimulus=out.stimulus, concepts=out.percept.concepts,
                context=out.percept.context, recall=out.percept.recall,
                novelty=out.percept.novelty, brain=self)
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
                    out.futures = self.chronos.explore(out.deliberation.proposals)
                    evidence = self.chronos.evidence(out.futures)
                    if evidence:
                        state.observe(evidence)
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
                out.verification = self.hybrid.check_and_learn(self, out)

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
            if self.car is None:
                return None
            if oversight is not None:
                self.car.oversight = oversight
            return self.car.beat()
        except Exception:  # noqa: BLE001 — a background beat never raises into the loop
            return None

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

    def stats(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "graph": self.graph.stats().to_dict(), "memory": self.memory.stats(),
                "turns": self.turns,
                "as_reasoner": bool(getattr(self.config, "as_reasoner", False))}
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
            if self.car is not None:
                out["car"] = self.car.stats()
            return out
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out = {"graph": self.graph.to_dict(), "memory": self.memory.to_dict(),
               "turns": self.turns}
        if self.metacog is not None:
            out["metacog"] = self.metacog.to_dict()
        if self.ground is not None:
            out["ground"] = self.ground.to_dict()
        if self.aura is not None:
            out["aura"] = self.aura.to_dict()
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
            if d.get("metacog") and self.metacog is not None:
                self.metacog.load_dict(d["metacog"])
            if d.get("ground") and self.ground is not None:
                self.ground.load_dict(d["ground"])
            if d.get("aura") and self.aura is not None:
                self.aura.load_dict(d["aura"])
            if d.get("car") and self.car is not None:
                self.car.load_dict(d["car"])
            self.turns = int(d.get("turns", 0))
        except Exception:  # noqa: BLE001 — a corrupt sidecar must never block boot
            pass
