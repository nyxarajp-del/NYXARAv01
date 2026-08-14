"""NYXARA · njp/brain.py — NJP V.01, the one brain (🧠, the single object the kernel holds).

:class:`NJPBrain` is what ``NyxaraCore`` builds and holds as ``self.njp``. It is the *only* brain
in NYXARA: there is no fusion of rival minds here and no vote to referee, because there is one
substrate — the Fluid Neural Automata in :mod:`nyxara.njp.fabric` — and everything else is an organ
attached to it.

**One turn of** :meth:`think` **is:**

1. **read the Master, including the half that was not said** — the Intent-Refinement Loop
   (:mod:`nyxara.njp.soulsync`) attaches the standing wants this request has earned, learned from
   corrections the Master actually made. What was *said* comes from :mod:`nyxara.njp.intent`.
2. **anticipate** — ask the manifold what this stimulus usually leads to, **before** settling. When
   that prediction is trusted the turn can answer from it; when it is not, that is reported and the
   fabric settles properly. Foresight that has not been earned is never spent.
3. **perceive** — words become cells, the cells are stimulated, and the automaton settles.
4. **ground** — content-addressed recall from :mod:`nyxara.njp.memory`. There is no token context
   window here; a fact from fifty turns ago is exactly as reachable as the last one. That is not
   the same as infinite, and :meth:`stats` reports the real capacity.
5. **judge** — anything she is about to state as fact goes through the Truth-Seeking Gauntlet
   (:mod:`nyxara.njp.truth`). An unestablished claim leaves labelled a conjecture, or is not made.
6. **expand** — the turn is handed to the pulse, which runs synaptogenesis, pruning and
   neurogenesis over it. **This is the "after every conversation" clause, and it is literal:** the
   synapse count after a turn is a different number from the one before it.

Everything is **fail-soft**: any organ that raises degrades to a null result and the turn still
completes. Every organ is optional; one that is off is *absent* from the reports rather than
reporting zeros, which is the distinction this whole package keeps.

**Sovereignty.** NJP only ever *proposes*. Every candidate flows through the kernel's unchanged,
fail-closed gate, and the safety core — corrigibility, oversight, loyalty, honesty — is never
governed, rewritten or bypassed by anything here. The mind proposes; the kernel disposes; the
Master is sovereign.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.ledger import Ledger
from nyxara.njp.manifold import Prediction
from nyxara.njp.soulsync import Reading, SoulSync
from nyxara.njp.truth import Judgement, TruthGauntlet, Verdict

__all__ = ["NJPPercept", "NJPThought", "NJPBrain"]


def _cell_id(token: str) -> int:
    """A stable cell id for a concept label.

    Hashed rather than assigned from a counter so the *same word maps to the same cell across
    restarts and across machines* — without that, a reloaded fabric would be wired for concepts it
    could no longer name, which is the one way persistence can be worse than starting fresh.
    """
    return int(hashlib.blake2b(token.encode("utf-8"), digest_size=6).hexdigest(), 16)


@dataclass
class NJPPercept:
    """One perceive-tick: what the fabric did with a turn, and what memory brought back."""

    stimulus: str = ""
    concepts: List[str] = field(default_factory=list)
    cells: List[int] = field(default_factory=list)
    settled: Optional[SettleResult] = None
    recall: Any = None
    anticipated: Optional[Prediction] = None
    # What the turn grounded to: entities, relations, and the words that reached neither. The
    # fabric's cell ids say what fired with what; this says what any of it was *about*.
    grounding: Any = None

    @property
    def novelty(self) -> float:
        """How unlike anything she knows this turn was — 0.0 familiar … 1.0 wholly new.

        Read straight off the manifold's familiarity, which is **graded and always measured**.
        Deliberately not ``settled.surprise``: that is the sharper signal about *this* turn and it
        feeds neurogenesis, but as a confidence discount it is far too blunt — it sits at 1.0 for
        every genuinely new prompt, including ordinary ones she answers perfectly well, and
        discounting those into abstention makes her useless rather than honest.
        Keying it on whether the prediction was *trusted* instead made this a step function: it
        sat at a flat 1.0 through every turn below the evidence floor, so a brain that was
        steadily learning still reported total ignorance right up to the moment it crossed. That
        matters because the reason-seat discounts confidence by this number — a step function
        there means she is uniformly, maximally unsure and then abruptly is not, rather than
        growing into confidence as she actually comes to recognise the ground.
        """
        ant = self.anticipated
        if ant is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - float(ant.familiarity)))

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:200], "concepts": self.concepts[:32],
                "n_cells": len(self.cells), "novelty": round(self.novelty, 4),
                "settled": self.settled.to_dict() if self.settled else None,
                "anticipated": self.anticipated.to_dict() if self.anticipated else None,
                "grounding": self.grounding.to_dict() if self.grounding is not None else None}


@dataclass
class NJPThought:
    """One whole turn, with everything that produced it kept inspectable."""

    stimulus: str = ""
    percept: Optional[NJPPercept] = None
    reading: Optional[Reading] = None          # the Master's unsaid half
    intent: Any = None                         # what was actually asked
    answer: str = ""
    judgement: Optional[Judgement] = None
    growth: Optional[GrowthReport] = None
    cycle_id: str = ""
    ms: float = 0.0
    focus: Any = None                          # what won the turn's attention, and what lost
    # What she is entitled to say about `answer`: known / believed / unknown. Carried beside the
    # claim rather than baked into it, so the voice can hedge without the hedge becoming content.
    epistemic: str = "unknown"
    epistemic_confidence: float = 0.0

    @property
    def confidence(self) -> float:
        return float(self.judgement.confidence) if self.judgement is not None else 0.0

    @property
    def verified(self) -> bool:
        """Was this **established** by the gauntlet? Confidence alone never sets this."""
        return bool(self.judgement is not None and self.judgement.established)

    @property
    def assertable(self) -> bool:
        """May she state this as fact? Only an established claim may be stated as one."""
        return bool(self.judgement is not None and self.judgement.assertable)

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:200], "answer": self.answer[:400],
                "cycle_id": self.cycle_id, "confidence": round(self.confidence, 4),
                "verified": self.verified, "assertable": self.assertable,
                "ms": round(self.ms, 3),
                "percept": self.percept.to_dict() if self.percept else None,
                "reading": self.reading.to_dict() if self.reading else None,
                "intent": self.intent.to_dict() if hasattr(self.intent, "to_dict") else None,
                "judgement": self.judgement.to_dict() if self.judgement else None,
                "growth": self.growth.to_dict() if self.growth else None,
                "focus": self.focus.to_dict() if self.focus is not None else None}


class NJPBrain:
    """NJP V.01 — one automaton, its organs, and the loop that grows it."""

    def __init__(self, config: Any = None) -> None:
        self.config = c = config
        self.turns = 0
        self.core: Any = None
        self.tools: Any = None
        self.knowledge: Any = None
        self.replay: List[str] = []
        # The kernel runs hypothesis framings CONCURRENTLY when the reason-seat accepts **kwargs,
        # which this one does — so three threads call think() on this same object at once. A
        # fabric is one organism: two thoughts rewiring it simultaneously corrupt the adjacency
        # dicts (and iterating one while another mutates it raises outright). Held for the whole
        # turn rather than per-mutation, because a turn is the unit that must be coherent: an
        # expand() that potentiates pairs from a settle another thread has already overwritten is
        # not a partial result, it is a wrong one. Re-entrant so resolve() can be called from
        # inside a turn.
        self._lock = threading.RLock()

        self.fabric = self._build_fabric(c)
        self.ledger = Ledger() if self._gate("ledger", True) else None
        self.memory = self._build_memory(c)
        self.tongue = self._build_tongue(c)
        self.intent = self._build_intent(c)
        self.ladder = self._build_ladder(c)
        self.grounder = self._build_grounder(c)
        self.world = self._build_world(c)
        self.predictor = self._build_predictor(c)
        self.levels = self._build_levels(c)
        self.discoverer = self._build_discoverer(c)
        self.reasoner = self._build_reasoner(c)
        self.self_model = self._build_self_model(c)
        self.meta = self._build_meta(c)
        self.goals = self._build_goals(c)
        self.curiosity = self._build_curiosity(c)
        self.attention = self._build_attention(c)
        self.environment = self._build_environment(c)
        self.readout = self._build_readout(c)
        self.voice = self._build_voice(c)
        self.truth = self._build_truth(c)
        self.soul = self._build_soul(c)
        self.evolver = self._build_evolver(c)
        self.pulse = self._build_pulse(c)

    # ---- construction ----------------------------------------------------- #
    def _gate(self, name: str, default: bool = True) -> bool:
        return bool(getattr(self.config, f"{name}_enabled", default)) if self.config else default

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default) if self.config else default

    def _build_fabric(self, c: Any) -> Fabric:
        """The one organ that is never optional. Without it there is no brain to configure."""
        try:
            return Fabric(
                seed=self._cfg("seed", 42),
                leak=self._cfg("leak", 0.85),
                threshold=self._cfg("threshold", 1.0),
                refractory=self._cfg("refractory", 2),
                max_settle_steps=self._cfg("max_settle_steps", 24),
                hebbian_rate=self._cfg("hebbian_rate", 0.08),
                depress_rate=self._cfg("depress_rate", 0.02),
                prune_threshold=self._cfg("prune_threshold", 0.015),
                growth_budget=self._cfg("growth_budget", 256),
                neurogenesis_error=self._cfg("neurogenesis_error", 0.55),
                neurogenesis_batch=self._cfg("neurogenesis_batch", 8),
                soft_cell_ceiling=self._cfg("soft_cell_ceiling", 250_000),
                soft_synapse_ceiling=self._cfg("soft_synapse_ceiling", 4_000_000),
                manifold_dim=self._cfg("manifold_dim", 10000))
        except Exception:  # noqa: BLE001 — a default fabric beats no fabric
            return Fabric()

    def _build_memory(self, c: Any) -> Any:
        if not self._gate("memory", True):
            return None
        try:
            from nyxara.njp.memory import HoloMemory
            return HoloMemory(dim=self._cfg("holo_dim", 4096),
                              capacity=self._cfg("holo_capacity", 20000),
                              seed=self._cfg("seed", 42))
        except Exception:  # noqa: BLE001 — she thinks without long-term recall, less well
            return None

    def _build_tongue(self, c: Any) -> Any:
        if not self._gate("tongue", True):
            return None
        try:
            from nyxara.njp.tongue import Lingua
            return Lingua()
        except Exception:  # noqa: BLE001 — falls back to the ASCII concept floor
            return None

    def _build_intent(self, c: Any) -> Any:
        if not self._gate("intent", True):
            return None
        try:
            from nyxara.njp.intent import IntentReader
            return IntentReader(lingua=self.tongue)
        except Exception:  # noqa: BLE001
            return None

    def _build_ladder(self, c: Any) -> Any:
        """The concept ladder — where types are *discovered* rather than declared.

        Nothing in NJP ever says "Jay is a person". Instances arrive with the relations they take
        part in as features, and the ladder's own agglomeration finds that the things with a name
        and a location form one group and the places form another. A hand-written type table would
        know that on turn one and never learn anything afterwards.
        """
        if not self._gate("concepts", True):
            return None
        try:
            from nyxara.cognition.concept_formation import AbstractionLadder
            return AbstractionLadder(link_threshold=self._cfg("concept_link_threshold", 0.2))
        except Exception:  # noqa: BLE001 — she grounds without forming concepts, less well
            return None

    def _build_grounder(self, c: Any) -> Any:
        """Language → concepts → entities → relations → beliefs. The organ that lets her answer.

        Built with the local ladder; the kernel's KnowledgeGraph is attached later by
        :meth:`attach_kernel`, because the brain is constructed before the kernel finishes wiring.
        """
        if not self._gate("grounding", True):
            return None
        try:
            from nyxara.njp.grounding import Grounder
            return Grounder(ladder=self.ladder,
                            known_floor=self._cfg("grounding_known_floor", 0.75),
                            learn_patterns=self._cfg("grounding_learn_patterns", True))
        except Exception:  # noqa: BLE001 — without it she is back to co-occurrence alone
            return None

    def _build_world(self, c: Any) -> Any:
        """Her model of what *happens*, as distinct from what *is*.

        Grounding holds facts; this holds events, causes and consequences. A mind with only the
        first knows a snapshot and cannot answer "what happens next" or "why did that happen".
        """
        if not self._gate("world", True):
            return None
        try:
            from nyxara.njp.world import WorldView
            return WorldView(window=self._cfg("world_window", 4),
                             min_support=self._cfg("world_min_support", 3),
                             min_lift=self._cfg("world_min_lift", 0.15))
        except Exception:  # noqa: BLE001 — she reasons about facts, not about change
            return None

    def _build_levels(self, c: Any) -> Any:
        """Five kinds of memory over the one content-addressed store.

        The levels govern durability and promotion, not reachability: recall still reaches
        everything, so a fact from fifty turns ago stays exactly as findable as the last one.
        What changes is that a passing greeting and the Master's name no longer decay at the same
        rate, and a claim that recurs from independent episodes is promoted to a fact.
        """
        if not self._gate("levels", True):
            return None
        try:
            from nyxara.njp.levels import HierarchicalMemory
            return HierarchicalMemory(self.memory,
                                      forget_below=self._cfg("forget_below", 0.05))
        except Exception:  # noqa: BLE001 — one undifferentiated pool still works, less well
            return None

    def _build_discoverer(self, c: Any) -> Any:
        """Where she proposes abstractions above her own episodes and tries to falsify them.

        The world model finds cause -> effect between things already named. This finds the pattern
        above that: when A + B + C keep arriving before D, the useful move is to hypothesise that
        they *jointly* predict D and then test that on episodes it was never fitted to.
        """
        if not self._gate("discover", True):
            return None
        try:
            from nyxara.njp.discover import Discoverer
            return Discoverer(min_support=self._cfg("discover_min_support", 4),
                              min_precision=self._cfg("discover_min_precision", 0.7),
                              max_order=self._cfg("discover_max_order", 3))
        except Exception:  # noqa: BLE001 — she keeps her pairs, and forms no rules above them
            return None

    def _build_reasoner(self, c: Any) -> Any:
        """Deliberate reasoning, at a depth the question deserves.

        Associative recall answers instantly and is right about familiar things — and it was the
        only mode she had, so a reflex and a conclusion were indistinguishable from outside. This
        adds the deliberate half and, more importantly, the routing between them.
        """
        if not self._gate("reason", True):
            return None
        try:
            from nyxara.njp.reason import Reasoner
            return Reasoner(self,
                            decide_margin=self._cfg("reason_decide_margin", 0.15),
                            enough=self._cfg("reason_enough", 0.75),
                            max_rung=self._cfg("reason_max_rung", "proof"))
        except Exception:  # noqa: BLE001 — she answers associatively, at one depth, as before
            return None

    def _build_self_model(self, c: Any) -> Any:
        """Her measured performance record — a capability map, not a brochure."""
        if not self._gate("self_model", True):
            return None
        try:
            from nyxara.njp.selfmodel import SelfModel
            return SelfModel()
        except Exception:  # noqa: BLE001 — she works without knowing how well she works
            return None

    def _build_meta(self, c: Any) -> Any:
        """Her own strategies as bandit arms. Learning how to learn, measured."""
        if not self._gate("meta", True):
            return None
        try:
            from nyxara.njp.selfmodel import MetaLearner
            meta = MetaLearner(self.self_model,
                               explore=self._cfg("meta_explore", 0.15))
            # The knobs she is allowed to tune on herself. Each value is a real setting that
            # changes what a turn costs, so a win here is a measured win rather than a preference.
            for name, value in (("shallow", 8), ("normal", 24), ("deep", 64)):
                meta.register("settle_steps", name, value)
            for name, value in (("narrow", 3), ("wide", 8)):
                meta.register("recall_k", name, value)
            for name, value in (("cheap", "association"), ("thorough", "verification")):
                meta.register("reason_depth", name, value)
            return meta
        except Exception:  # noqa: BLE001 — she keeps whatever settings she was built with
            return None

    def _build_goals(self, c: Any) -> Any:
        """Mission → Goal → Subgoal → Task, with the economics that make plans comparable.

        A flat list of goals is a to-do list. The structure is what lets a finished task advance a
        mission; the economics are what let her choose between two plans that both sound good.
        """
        if not self._gate("goals", True):
            return None
        try:
            from nyxara.njp.goals import GoalTree
            return GoalTree()
        except Exception:  # noqa: BLE001 — she acts on what she is asked, without a plan of her own
            return None

    def _build_curiosity(self, c: Any) -> Any:
        """Her own questions, priced by what answering them is worth.

        Which question to ask is a decision, not an appetite: expected value of information
        against cost, so she asks what reduces uncertainty most per unit cost rather than whatever
        occurred to her last.
        """
        if not self._gate("curiosity", True):
            return None
        try:
            from nyxara.njp.curiosity import Curiosity
            return Curiosity(self, min_value=self._cfg("curiosity_min_value", 0.05))
        except Exception:  # noqa: BLE001 — she answers what she is asked and wonders nothing
            return None

    def _build_attention(self, c: Any) -> Any:
        """The bottleneck. Every organ competes; exactly one wins the turn.

        Handed to a caller as a pile, six organs' outputs are not integration — they are six
        opinions and no cognition. Something has to lose.
        """
        if not self._gate("attention", True):
            return None
        try:
            from nyxara.njp.attention import Attention
            return Attention(self, capacity=self._cfg("attention_capacity", 1))
        except Exception:  # noqa: BLE001 — every organ reports, and nothing arbitrates
            return None

    def _build_environment(self, c: Any) -> Any:
        """observe → act → outcome → learn over real tool calls, via the repo's own model."""
        if not self._gate("environment", True):
            return None
        try:
            from nyxara.sim.envmodel import EnvironmentModel
            return EnvironmentModel()
        except Exception:  # noqa: BLE001 — she uses tools without modelling what they do
            return None

    def _build_readout(self, c: Any) -> Any:
        """The gradient-trained head over the fabric.

        Every update in the fabric is local — a synapse changes because its own two endpoints
        fired, and nothing tells it what the network as a whole was trying to achieve. Local rules
        cannot assign credit *through* a chain of activity to the thing that actually caused the
        error. This is the other kind of learning, and it is additive: the fabric's plasticity is
        untouched, and two learners now sit over one substrate.
        """
        if not self._gate("readout", True):
            return None
        try:
            from nyxara.njp.learn import Readout, available
            if not available():
                return None      # no numpy ⇒ no gradients, and absent beats pretending
            return Readout(width=self._cfg("readout_width", 512),
                           hidden=self._cfg("readout_hidden", 128),
                           lr=self._cfg("readout_lr", 0.01),
                           seed=self._cfg("seed", 42))
        except Exception:  # noqa: BLE001 — she keeps her local plasticity and learns no gradients
            return None

    def _build_predictor(self, c: Any) -> Any:
        """The predict → observe → diagnose → repair loop, with repairs wired to real organs.

        The repairs are what make the diagnosis worth computing. Classifying an error as
        ``grounding`` and then feeding the same scalar back into everything would be a more
        expensive way to learn nothing.
        """
        if not self._gate("predict", True):
            return None
        try:
            from nyxara.njp.predict import ErrorKind, PredictionEngine
            engine = PredictionEngine(self)
            engine.register_repair(ErrorKind.GROUNDING, self._repair_grounding)
            engine.register_repair(ErrorKind.WORLD_MODEL, self._repair_world)
            engine.register_repair(ErrorKind.MEMORY, self._repair_memory)
            engine.register_repair(ErrorKind.REASONING, self._repair_reasoning)
            return engine
        except Exception:  # noqa: BLE001 — she still learns per-organ, just without the diagnosis
            return None

    # ---- the repairs, one per error kind ---------------------------------- #
    def _repair_grounding(self, outcome: Any) -> bool:
        """A sentence that should have grounded and did not: try again with the fluent surface.

        This is the one place the slow deep parse is worth its seconds — she already knows the
        cheap path failed, so the cost is paid only on a measured miss, and a successful parse
        also induces a pattern so the same sentence shape is cheap next time.
        """
        try:
            if self.grounder is None:
                return False
            text = str(getattr(outcome, "key", "") or "")
            return bool(self.grounder.ground(text, deep=True).grounded)
        except Exception:  # noqa: BLE001
            return False

    def _repair_world(self, outcome: Any) -> bool:
        """What she expected to follow did not: record what actually did."""
        try:
            if self.world is None:
                return False
            from nyxara.njp.world import Event
            self.world.observe(Event(action=str(getattr(outcome, "actual", "") or "")))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _repair_memory(self, outcome: Any) -> bool:
        """The answer was on record and was not retrieved: re-file it under the cue that missed."""
        try:
            if self.memory is None:
                return False
            self.memory.remember(f"repair-{self.turns}", str(getattr(outcome, "actual", "")),
                                 kind="conclusion", cue=str(getattr(outcome, "key", "")))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _repair_reasoning(self, outcome: Any) -> bool:
        """Right facts, wrong conclusion: record it so the gauntlet weighs it against the next one."""
        try:
            if self.ledger is None or not hasattr(self.ledger, "errors"):
                return False
            self.ledger.errors.record(str(getattr(outcome, "key", "")),
                                      verdict="wrong conclusion from correct facts",
                                      truth=str(getattr(outcome, "actual", "")))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _build_voice(self, c: Any) -> Any:
        """Her voice, with her identity attached to it.

        ``soul`` is passed because :meth:`~nyxara.njp.voice.Dialogue._voice` returns ``None``
        without it, and a ``None`` voice fragment means the rendering prompt carries no identity
        at all — a fluent model then phrases her content in its own default register rather than
        hers. The LLM itself stays lazily resolved inside ``Dialogue`` (it is a ~2.4 GB on-device
        engine; building the brain must not block on loading it).
        """
        if not self._gate("voice", True):
            return None
        try:
            from nyxara.njp.voice import Dialogue
            return Dialogue(max_tokens=self._cfg("reply_max_tokens", 220),
                            soul=self._identity_soul())
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _identity_soul() -> Any:
        """Her identity, for the voice fragment. Absent identity is a plainer voice, never a crash."""
        try:
            from nyxara.identity.soul import Soul
            return Soul()
        except Exception:  # noqa: BLE001
            return None

    def _build_truth(self, c: Any) -> Optional[TruthGauntlet]:
        if not self._gate("truth", True):
            return None
        try:
            return TruthGauntlet(min_sources=self._cfg("truth_min_sources", 2),
                                 require_hard=self._cfg("truth_require_hard", True),
                                 ledger=self.ledger)
        except Exception:  # noqa: BLE001 — without it nothing is verified, and nothing claims to be
            return None

    def _build_soul(self, c: Any) -> Optional[SoulSync]:
        if not self._gate("soulsync", True):
            return None
        try:
            return SoulSync(min_support=self._cfg("soulsync_min_support", 2))
        except Exception:  # noqa: BLE001
            return None

    def _build_evolver(self, c: Any) -> Any:
        if not self._gate("evolve", True):
            return None
        try:
            from nyxara.njp.evolve import SelfEvolver
            return SelfEvolver(self, ledger=self.ledger,
                               every_s=self._cfg("evolve_every_s", 300.0),
                               min_gain=self._cfg("evolve_min_gain", 0.05))
        except Exception:  # noqa: BLE001 — without it she stays exactly as she was written
            return None

    def _build_pulse(self, c: Any) -> Any:
        if not self._gate("pulse", True):
            return None
        try:
            from nyxara.njp.pulse import PulseEngine
            return PulseEngine(self, every_s=self._cfg("pulse_every_s", 1.0),
                               consolidate_every_s=self._cfg("consolidate_every_s", 60.0),
                               evolve_every_s=self._cfg("evolve_every_s", 300.0))
        except Exception:  # noqa: BLE001 — without it she grows only when spoken to
            return None

    def attach_kernel(self, *, tools: Any = None, knowledge: Any = None,
                      core: Any = None, graph: Any = None) -> None:
        """Take the live toolset, knowledge base and organs the kernel already builds.

        Without this her brain is blind to tools — and, before ``graph`` was passed, blind to the
        kernel's :class:`~nyxara.memory.graph.KnowledgeGraph` too, so everything she grounded lived
        in a private store the rest of the system could not see and that did not survive a restart.

        Every argument is optional and an absent organ is genuinely absent, not stubbed: the
        grounder falls back to its own in-process store, which is what keeps a bare ``NJPBrain()``
        useful in a test and on a cold boot.
        """
        self.tools = tools
        self.knowledge = knowledge
        self.core = core
        try:
            if graph is not None and self.grounder is not None:
                self.grounder.graph = graph
        except Exception:  # noqa: BLE001 — a brain that cannot see the graph still thinks
            pass

    # ---- encoding --------------------------------------------------------- #
    def concepts(self, text: str) -> List[str]:
        try:
            from nyxara.njp.tongue import concepts_in
            return concepts_in(text)
        except Exception:  # noqa: BLE001
            return [w for w in str(text or "").lower().split() if len(w) > 2][:32]

    def encode(self, text: str) -> List[int]:
        """Words to cells. Stable across restarts, which is what makes the fabric reloadable."""
        return [_cell_id(tok) for tok in self.concepts(text)]

    # ---- perception ------------------------------------------------------- #
    def perceive(self, text: str, *, remember: bool = True,
                 intent: Any = None) -> NJPPercept:
        """Stimulate the fabric with a turn, settle it, and bring back grounded context."""
        out = NJPPercept(stimulus=str(text or ""))
        try:
          with self._lock:
                if not out.stimulus.strip():
                    return out
                out.concepts = self.concepts(out.stimulus)

                # Ground BEFORE settling. The fabric's answer to a turn is co-activation; the
                # grounder's is structure, and a question that structure can answer should be
                # answered from it rather than from whatever happened to fire. Cheap enough to sit
                # in every turn — the fluent surface is not consulted here (`deep` stays off),
                # because that call is seconds, and seconds do not belong in a perceive.
                if self.grounder is not None:
                    try:
                        out.grounding = self.grounder.ground(out.stimulus, intent=intent)
                    except Exception:  # noqa: BLE001
                        out.grounding = None

                # A relation that *happened* goes on the timeline; one that *holds* stays a fact.
                # Conflating them would put unchanging facts in causal order.
                if self.world is not None and out.grounding is not None:
                    try:
                        self.world.from_grounding(out.grounding)
                    except Exception:  # noqa: BLE001
                        pass

                out.cells = self.encode(out.stimulus)
                if not out.cells:
                    return out

                # Ask what usually follows this BEFORE settling — the pre-cognitive read.
                out.anticipated = self.fabric.anticipate(out.cells)

                self.fabric.stimulate(out.cells)
                out.settled = self.fabric.settle()

                if self.memory is not None:
                    try:
                        out.recall = self.memory.recall(out.stimulus,
                                                        k=self._cfg("recall_k", 5))
                    except Exception:  # noqa: BLE001
                        out.recall = None
                return out
        except Exception:  # noqa: BLE001 — a perceive failure never breaks a turn
            return out

    # ---- the turn --------------------------------------------------------- #
    def think(self, stimulus: str, *, remember: bool = True,
              outcome: float = 1.0) -> NJPThought:
        """One whole turn: read → anticipate → perceive → ground → judge → expand.

        Serialised: see :attr:`_lock`. The kernel may call this from several framing threads at
        once, and a turn has to be atomic against the fabric or the growth it records describes a
        state that never existed.
        """
        with self._lock:
            return self._think(stimulus, remember=remember, outcome=outcome)

    def _think(self, stimulus: str, *, remember: bool = True,
               outcome: float = 1.0) -> NJPThought:
        out = NJPThought(stimulus=str(stimulus or ""))
        t0 = time.perf_counter()
        try:
            if not out.stimulus.strip():
                return out
            self.turns += 1
            out.cycle_id = f"njp-{self.turns}"

            # 1. the Master's unsaid half, and what was actually asked
            if self.soul is not None:
                out.reading = self.soul.read(out.stimulus)
            if self.intent is not None:
                try:
                    out.intent = self.intent.read(out.stimulus)
                except Exception:  # noqa: BLE001
                    out.intent = None

            # 2-4. perceive and ground
            out.percept = self.perceive(out.stimulus, remember=remember, intent=out.intent)
            out.answer = self._compose(out)
            self._set_epistemic(out)
            # Register what she just committed to, so `resolve()` has something to score it
            # against. Registered even when the answer is empty: "I don't know" is a real
            # prediction about the world and being wrong about it is a real, diagnosable miss.
            if self.predictor is not None:
                try:
                    self.predictor.predict(
                        out.stimulus, out.answer or "<unknown>",
                        confidence=out.epistemic_confidence,
                        organ=("grounding" if getattr(
                            getattr(out.percept, "grounding", None), "answer", None) else "fabric"))
                except Exception:  # noqa: BLE001
                    pass

            # 5. nothing is stated as fact until the gauntlet says it may be
            if self.truth is not None and out.answer:
                out.judgement = self.truth.judge(out.answer, context=self)

            # 6. remember the conclusion, not the question: storing a bare question would make it
            # its own best match and echo back as its own answer.
            if remember and out.answer:
                self._remember_turn(out)

            # 7. ATTEND — every organ bid, one won. This is where the turn stops being a pile of
            # six opinions and becomes one thing she is actually thinking about.
            if self.attention is not None:
                out.focus = self.attention.attend(out)

            # 8. an episode for the discoverer: what was present, and what she concluded. This
            # is the raw material abstractions are proposed from — one episode is never a
            # pattern, which is exactly why it is only recorded here and judged elsewhere.
            if self.discoverer is not None and out.percept is not None and out.answer:
                try:
                    self.discoverer.observe(out.percept.concepts[:8], out.answer[:60])
                except Exception:  # noqa: BLE001
                    pass

            # 9. EXPAND — the physical growth, after every single turn
            out.growth = self._expand(out, outcome=outcome)

            out.ms = (time.perf_counter() - t0) * 1000.0
            if self.ledger is not None:
                self.ledger.observe_latency(out.ms)
            if self.evolver is not None:
                self.evolver.observe("nyxara.njp.fabric", out.ms)
            self._remember_sample(out.stimulus)
            return out
        except Exception:  # noqa: BLE001 — a thought that fails is empty, never fatal
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out

    def _remember_turn(self, thought: NJPThought) -> None:
        """File this turn's conclusion at the level it belongs to.

        The level is chosen from what the turn was *about*, not from how it was phrased. Anything
        grounded to the Master or to NYXARA herself is autobiographical and therefore protected —
        that is the level whose loss would change who she is, so it is not subject to forgetting
        at all. Everything else is an episode, and becomes semantic only if it recurs.
        """
        try:
            if self.levels is not None:
                self.levels.remember(
                    f"turn-{self.turns}", thought.answer,
                    level=self._level_for(thought), cue=thought.stimulus,
                    source=f"turn-{self.turns}")
                return
            if self.memory is not None:
                self.memory.remember(f"turn-{self.turns}", thought.answer,
                                     kind=("conclusion" if thought.verified else "episode"),
                                     cue=thought.stimulus)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _level_for(thought: NJPThought) -> str:
        """Which store this conclusion belongs in."""
        try:
            from nyxara.njp.grounding import SELF_ENTITY
            from nyxara.njp.levels import Level

            grounding = getattr(thought.percept, "grounding", None)
            for triple in (getattr(grounding, "triples", None) or []):
                subject = str(getattr(triple, "subject", ""))
                if subject.startswith(SELF_ENTITY) or subject == "NYXARA":
                    return Level.AUTOBIOGRAPHICAL
            return Level.EPISODIC
        except Exception:  # noqa: BLE001
            from nyxara.njp.levels import Level
            return Level.EPISODIC

    def _expand(self, thought: NJPThought, *, outcome: float) -> Optional[GrowthReport]:
        """Grow on this turn, synchronously, and return what actually changed.

        Deliberately **not** routed through the pulse. "After every conversation" has to mean the
        synapse count is different when the turn returns, not that growth was queued and might
        happen on some later beat — and a caller that never beats (a one-shot CLI, a test) must
        still see the fabric expand. The pulse's queue is for experience that arrives *outside* a
        turn; letting it also carry turns would double-count every one of them.
        """
        try:
            settled = thought.percept.settled if thought.percept else None
            if settled is None:
                return None
            return self.fabric.expand(outcome=outcome, result=settled)
        except Exception:  # noqa: BLE001
            return None

    def _compose(self, thought: NJPThought) -> str:
        """What she can honestly say about this turn, answering from structure where there is any.

        Deliberately plain. This brain's contribution is *what is true and what fired*, not
        fluency; :mod:`nyxara.njp.voice` phrases it when a fluent surface is installed, and says so
        when one is not rather than letting a fallback babble in her name.

        Order matters. A grounded answer comes first and alone, because it is the only thing here
        that actually answers the question that was asked. Everything below it is context about
        the turn, and context concatenated onto an answer reads as hedging.

        What this used to emit instead is worth naming: on a familiar turn it returned *"this
        usually leads to 4 familiar activations"* — a true statement about her own manifold, and
        not in any sense a reply. A brain reporting its own activation counts to the Master is the
        clearest possible symptom of having no content to report.
        """
        try:
            grounding = thought.percept.grounding if thought.percept else None

            # 1. She was asked something and the structure knows the answer.
            #
            # The bare claim is returned, with no hedge attached. The epistemic state travels
            # beside it on the thought and :mod:`nyxara.njp.voice` applies the caveat after
            # rendering. Writing "Jay (I believe this, confidence 0.90)" here instead made the
            # hedge part of the *content*, and the voice's faithfulness check — which asks whether
            # a rendering still covers the content it was given — then demanded the model echo the
            # words "believe" and "confidence" back. A model that answered the question perfectly
            # with "Jay" was scored as having wandered off, and every correct fluent rendering was
            # discarded. Content and epistemic framing are different layers; keep them apart.
            answer = getattr(grounding, "answer", None)
            if answer is not None and answer.answered:
                return str(answer.text)[:1000]

            # 2. She was asked something and genuinely does not know. Say that — do not fall
            # through to recall and offer a loosely-related memory as though it were an answer.
            if getattr(grounding, "is_question", False):
                return ""

            bits: List[str] = []
            # 3. She was told something and it landed. Confirm what was actually understood, so a
            # misparse is visible to the Master on the turn it happens rather than three turns later.
            triples = list(getattr(grounding, "triples", None) or [])
            if triples:
                learned = "; ".join(f"{t.subject} {t.predicate.replace('_', ' ')} {t.object}"
                                    for t in triples[:3])
                bits.append(f"noted: {learned}")
            for prior, now in list(getattr(grounding, "contradictions", None) or [])[:1]:
                bits.append(f"this contradicts what I had ({prior.object}); "
                            f"I have revised it to {now.object}")

            rec = thought.percept.recall if thought.percept else None
            best = getattr(rec, "best", None) if rec is not None else None
            if best is not None and getattr(best, "text", ""):
                bits.append(str(best.text))
            if thought.reading is not None and thought.reading.has_latent:
                wants = "; ".join(w.want for w in thought.reading.latent[:2])
                bits.append(f"you usually also want: {wants}")
            return " — ".join(bits)[:1000]
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _set_epistemic(thought: NJPThought) -> None:
        """Record which of the three states this turn's answer is in.

        ``KNOWN`` and ``BELIEVED`` are different assertions and ``UNKNOWN`` is not an assertion at
        all. Keeping them apart on the thought is what lets the voice say "Jay", "I think Jay" and
        "I don't know" as three genuinely different replies rather than one string with optional
        decoration.
        """
        try:
            from nyxara.njp.grounding import Epistemic
            grounding = getattr(thought.percept, "grounding", None)
            answer = getattr(grounding, "answer", None)
            if answer is not None and answer.answered:
                thought.epistemic = answer.state
                thought.epistemic_confidence = float(answer.confidence)
                return
            # An acknowledgement is not a claim about the world. "noted: Master has name Jay"
            # reports what she just recorded, and she is not uncertain about her own record —
            # hedging it with "I believe this, but I am not certain" would be false modesty about
            # the one thing she does know for sure.
            if getattr(grounding, "triples", None):
                thought.epistemic = Epistemic.KNOWN
                thought.epistemic_confidence = 1.0
                return
            # No grounded answer: known only if the gauntlet established what she did say.
            if thought.answer and thought.verified:
                thought.epistemic = Epistemic.KNOWN
                thought.epistemic_confidence = thought.confidence
            elif thought.answer:
                thought.epistemic = Epistemic.BELIEVED
                thought.epistemic_confidence = thought.confidence
            else:
                thought.epistemic = Epistemic.UNKNOWN
                thought.epistemic_confidence = 0.0
        except Exception:  # noqa: BLE001
            pass

    def converse(self, text: str) -> Any:
        """Think, then say it. Her content; a fluent surface only phrases it."""
        try:
            thought = self.think(text)
            if self.voice is None:
                return thought.answer
            return self.voice.respond(thought, brain=self)
        except Exception:  # noqa: BLE001 — she always says something, never crashes mid-sentence
            return ""

    # ---- feedback --------------------------------------------------------- #
    def resolve(self, thought: Any, *, correct: float, actual: Any = None) -> None:
        """Tell the brain how a turn actually turned out.

        Drives plasticity, the soul-sync, and — when the turn registered a prediction — the
        diagnosis. The scalar reaches the fabric as before; what is new is that the *diagnosis*
        reaches whichever organ actually produced the mistake. "She was wrong" is nearly useless
        as a training signal: it is the same string whether she misheard the question, retrieved
        the wrong memory, or reasoned badly from correct facts, and feeding it to every organ at
        once mostly teaches all of them to be vaguely less confident.
        """
        try:
          with self._lock:
                settled = getattr(getattr(thought, "percept", None), "settled", None)
                if settled is not None:
                    # Map correctness onto the plasticity signal: right reinforces what fired, wrong
                    # weakens it. This is the only place an outcome becomes structure.
                    self.fabric.expand(outcome=(float(correct) * 2.0 - 1.0), result=settled)
                if self.soul is not None:
                    self.soul.confirm(corrected=(float(correct) < 0.5))
                if self.predictor is not None:
                    outcome = self._score_prediction(thought, correct=float(correct),
                                                     actual=actual)
                    # The same attribution that routes a repair also updates the record of which
                    # faculty keeps needing repairing — the error taxonomy paying twice.
                    if self.self_model is not None and outcome is not None:
                        self.self_model.observe_diagnosis(getattr(outcome, "diagnosis", None))
                if self.self_model is not None and float(correct) >= 0.5:
                    self.self_model.observe("grounding", 1.0) if getattr(
                        getattr(thought, "percept", None), "grounding", None) else None
                if self.meta is not None:
                    self.meta.reward("settle_steps", float(correct))
        except Exception:  # noqa: BLE001
            pass

    def _score_prediction(self, thought: Any, *, correct: float, actual: Any) -> Any:
        """Close the loop on this turn's expectation, with the evidence the diagnosis needs."""
        try:
            stimulus = str(getattr(thought, "stimulus", "") or "")
            if not stimulus:
                return None
            percept = getattr(thought, "percept", None)
            grounding = getattr(percept, "grounding", None)
            return self.predictor.observe(
                stimulus,
                actual if actual is not None else ("correct" if correct >= 0.5 else "wrong"),
                evidence={
                    "stimulus": stimulus,
                    "concepts": list(getattr(percept, "concepts", None) or []),
                    "triples": list(getattr(grounding, "triples", None) or []),
                    # A turn that stated a fact and grounded nothing is the grounder's miss, and
                    # this is what distinguishes it from a turn that had no fact to state.
                    "expected_triples": bool(
                        grounding is not None and not getattr(grounding, "is_question", False)),
                    "facts_correct": bool(getattr(grounding, "triples", None)),
                })
        except Exception:  # noqa: BLE001
            pass

    def correct(self, before: str, after: str) -> Any:
        """The Master re-issued a request with something added. Learn the standing rule."""
        try:
            return self.soul.observe_correction(before, after) if self.soul is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _remember_sample(self, text: str) -> None:
        """Keep a small replay set — the held-out evidence the self-editor is judged against."""
        try:
            self.replay.append(str(text)[:200])
            if len(self.replay) > 64:
                del self.replay[:-64]
        except Exception:  # noqa: BLE001
            pass

    def measure_sample(self, sample: Any) -> float:
        """Re-run one replay sample and return what it costs now, in ms.

        This is the hook :mod:`nyxara.njp.evolve` calls to find out whether an edit actually made
        her faster. Without a real measurement the self-editor refuses the edit, so this returning
        a real number is what makes the improvement claim checkable rather than asserted.
        """
        try:
            t0 = time.perf_counter()
            self.perceive(str(sample), remember=False)
            return (time.perf_counter() - t0) * 1000.0
        except Exception:  # noqa: BLE001
            return float("inf")

    def measure_capability(self, sample: Any) -> Dict[str, float]:
        """Re-run one sample and report **both** what it cost and how well she did.

        Speed alone is a bad objective for a self-editor: the fastest version of a mind is the one
        that stops thinking. So the self-editor is given a capability reading alongside the
        latency, and an edit that buys milliseconds by making her worse at modelling herself is
        refused with the same machinery that refuses one that is simply slower.

        ``quality`` is ``1 − surprise`` — how well the settle matched what she predicted of herself
        before running it. It is the one accuracy signal available without a labelled task, and it
        is measured on the same replay samples the latency is measured on.
        """
        out = {"ms": float("inf"), "quality": 0.0, "cells": 0.0, "synapses": 0.0}
        try:
            t0 = time.perf_counter()
            percept = self.perceive(str(sample), remember=False)
            out["ms"] = (time.perf_counter() - t0) * 1000.0
            settled = getattr(percept, "settled", None)
            if settled is not None:
                out["quality"] = max(0.0, min(1.0, 1.0 - float(settled.surprise)))
            out["cells"] = float(self.fabric.n_cells)
            out["synapses"] = float(self.fabric.n_synapses)
            return out
        except Exception:  # noqa: BLE001
            return out

    # ---- the beat --------------------------------------------------------- #
    def tick(self, *, oversight: Any = None) -> Dict[str, Any]:
        """One beat of the continuous loop. Driven by the kernel's clock, never its own thread."""
        try:
            if self.pulse is None:
                return {}
            with self._lock:
                return self.pulse.beat(oversight=oversight).to_dict()
        except Exception:  # noqa: BLE001 — a background beat never raises into the loop
            return {}

    def anticipate(self, text: str = "") -> Any:
        """What she expects next — either from this stimulus, or from the Master's own pattern."""
        try:
            if text:
                return self.fabric.anticipate(self.encode(text))
            return self.soul.anticipate() if self.soul is not None else None
        except Exception:  # noqa: BLE001
            return None

    # ---- reporting -------------------------------------------------------- #
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"turns": self.turns, "fabric": self.fabric.stats()}
        for name, organ in (("ledger", self.ledger), ("truth", self.truth),
                            ("soulsync", self.soul), ("evolve", self.evolver),
                            ("pulse", self.pulse), ("grounding", self.grounder),
                            ("world", self.world), ("predict", self.predictor), ("levels", self.levels),
                            ("discover", self.discoverer), ("reason", self.reasoner),
                            ("self_model", self.self_model), ("meta", self.meta),
                            ("goals", self.goals), ("curiosity", self.curiosity),
                            ("attention", self.attention), ("readout", self.readout)):
            if organ is None:
                continue                       # an organ that is off is ABSENT, not zeroed
            try:
                out[name] = organ.stats()
            except Exception:  # noqa: BLE001
                out[name] = {"error": "stats failed"}
        if self.memory is not None:
            try:
                out["memory"] = {"traces": len(self.memory)}
            except Exception:  # noqa: BLE001
                pass
        return out

    def report(self) -> Dict[str, Any]:
        """Then vs now — is she actually more than she was? Numbers, not a claim."""
        try:
            return self.ledger.report() if self.ledger is not None else {}
        except Exception:  # noqa: BLE001
            return {}

    # ---- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"turns": self.turns, "fabric": self.fabric.to_dict()}
        for name, organ in (("ledger", self.ledger), ("soulsync", self.soul),
                            ("grounding", self.grounder), ("world", self.world), ("predict", self.predictor), ("levels", self.levels),
                            ("discover", self.discoverer), ("reason", self.reasoner),
                            ("self_model", self.self_model), ("meta", self.meta),
                            ("goals", self.goals), ("curiosity", self.curiosity),
                            ("attention", self.attention), ("readout", self.readout)):
            if organ is None:
                continue
            try:
                d[name] = organ.to_dict()
            except Exception:  # noqa: BLE001
                continue
        if self.memory is not None and hasattr(self.memory, "to_dict"):
            try:
                d["memory"] = self.memory.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        """Wake up as the brain that went to sleep. This is what makes tomorrow's NJP today's."""
        try:
            self.turns = int(d.get("turns", 0))
            if d.get("fabric"):
                self.fabric.load_dict(d["fabric"])
            if d.get("ledger") and self.ledger is not None:
                self.ledger.load_dict(d["ledger"])
            if d.get("soulsync") and self.soul is not None:
                self.soul.load_dict(d["soulsync"])
            if d.get("grounding") and self.grounder is not None:
                self.grounder.load_dict(d["grounding"])
            if d.get("world") and self.world is not None:
                self.world.load_dict(d["world"])
            if d.get("predict") and self.predictor is not None:
                self.predictor.load_dict(d["predict"])
            if d.get("levels") and self.levels is not None:
                self.levels.load_dict(d["levels"])
            if d.get("discover") and self.discoverer is not None:
                self.discoverer.load_dict(d["discover"])
            if d.get("self_model") and self.self_model is not None:
                self.self_model.load_dict(d["self_model"])
            if d.get("meta") and self.meta is not None:
                self.meta.load_dict(d["meta"])
            if d.get("goals") and self.goals is not None:
                self.goals.load_dict(d["goals"])
            if d.get("curiosity") and self.curiosity is not None:
                self.curiosity.load_dict(d["curiosity"])
            if d.get("readout") and self.readout is not None:
                self.readout.load_dict(d["readout"])
            if d.get("memory") and self.memory is not None and hasattr(self.memory, "load_dict"):
                self.memory.load_dict(d["memory"])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born brain
            pass
