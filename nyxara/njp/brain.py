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
import re
import threading
import time
from dataclasses import dataclass, field
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.fabric import Fabric, GrowthReport, SettleResult
from nyxara.njp.ledger import Ledger
from nyxara.njp.manifold import Prediction
from nyxara.njp.soulsync import Reading, SoulSync
from nyxara.njp.truth import Judgement, TruthGauntlet, Verdict

__all__ = ["NJPPercept", "NJPThought", "NJPBrain"]

# What a counterfactual question asks her to *do* to a variable, and by how much.
#
# Only interventions whose magnitude the sentence actually states are listed. "reduce the water"
# is deliberately absent: a factor she chose herself would come back through the do-operator
# carrying the same confidence as one the Master gave, and nothing downstream could tell the two
# apart afterwards. An intervention she cannot size is one she declines to simulate.
_INTERVENTION_FACTOR: Dict[str, float] = {
    # Past participles are here because "what if the water IS HALVED" is the commonest English
    # phrasing of the question and matched none of the infinitive/gerund forms — the passive is
    # how a counterfactual about a thing rather than an actor is normally put.
    "halve": 0.5, "halving": 0.5, "halved": 0.5, "half": 0.5,
    "aadha": 0.5, "adha": 0.5, "aadhi": 0.5,
    "double": 2.0, "doubling": 2.0, "doubled": 2.0, "dugna": 2.0, "duguna": 2.0,
    "remove": 0.0, "removing": 0.0, "removed": 0.0, "stop": 0.0, "stopping": 0.0,
    "stopped": 0.0, "band": 0.0, "bina": 0.0,
}

_INTERVENTION = re.compile(
    r"\b(?P<op>halve|halving|half|double|doubling|remove|removing|stop|stopping|"
    r"aadha|adha|aadhi|dugna|duguna|band|bina)\b"
    r"(?:\s+(?:the|le|kar|of|its))*\s+(?P<var>[a-z][a-z_]*)\b",
    re.IGNORECASE)

# The same question stated as an absolute rather than a factor — "what if the water were 3".
_INTERVENTION_ABSOLUTE = re.compile(
    r"\b(?:the\s+)?(?P<var>[a-z][a-z_]*)\s+(?:were|was|is|becomes?|ho)\s+"
    r"(?P<val>-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE)

# Hinglish puts the object before the verb — "paani aadha kar doon" — so the factor word follows
# the variable rather than preceding it. Same intervention, mirrored word order.
#
# The English passive belongs in this pattern rather than the leading one for exactly the same
# reason: "the water is halved" also puts the variable first. It is the commonest way to ask a
# counterfactual about a *thing* rather than about someone acting on it, and matching only
# "halve the water" meant the natural phrasing of the question fell through to no intervention
# at all — the parse then supplied no `variable` key, `_strategy_simulate` returned None before
# reaching the do-operator, and the turn answered nothing while the engine sat ready.
_INTERVENTION_TRAILING = re.compile(
    r"\b(?:the\s+)?(?P<var>[a-z][a-z_]*)\s+(?:(?:is|was|were|be|gets?|got)\s+)?"
    r"(?P<op>aadha|adha|aadhi|dugna|duguna|band|half|halved|double|doubled|"
    r"removed|stopped)\b",
    re.IGNORECASE)


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
    # Set when the answer came from a relation the question did not ask for. Carries the trace
    # of which fact was offered and why, so a retrieved answer is never indistinguishable from
    # one the store held under exactly the relation that was requested.
    recalled: str = ""
    # What she is entitled to say about `answer`: known / believed / unknown. Carried beside the
    # claim rather than baked into it, so the voice can hedge without the hedge becoming content.
    epistemic: str = "unknown"
    epistemic_confidence: float = 0.0
    # What closing the learning loop behind this turn actually changed. `None` when the
    # integration layer is switched off, which is a different thing from "it changed nothing".
    loop: Any = None
    # What the Recursive Cognitive Field did with this turn: concepts formed, error diagnosed,
    # and whether the diagnosis called for a coefficient or for restructuring what she can
    # represent. `trial` is set only on the turns loop 2 comes round.
    field: Any = None
    trial: Any = None
    # What the Cognitive Learning Core did with this turn: schemas induced over the kinds the
    # field formed, scored against held-out facts, and the ones that failed dropped. A
    # `CoreReport`; absent when the organ is off.
    learning: Any = None
    # How an answer was *derived*, when it was derived rather than looked up — the composition or
    # the schema transfer, with its steps. Carried separately from `answer` because the two are
    # different epistemic objects: a derived answer is defeasible however confident it reads, and
    # a caller that cannot tell it from a stated fact will eventually state it as one.
    derivation: Any = None
    # What the Master was *doing* with this turn, and what each recalled memory scored against
    # it. Carried so a reply that reached for something irrelevant is visible from outside rather
    # than having to be inferred from the reply itself.
    act: Any = None
    relevance: List[Any] = dc_field(default_factory=list)
    # The meta-reasoner's own record of how this answer was reached, kept so that when reality
    # grades the answer later the credit can reach the strategy that actually produced it.
    # Without it :meth:`~nyxara.njp.metareason.MetaReasoner.outcome` has nothing to be called
    # with, which is why strategy credit came only from the critic — a mind grading its own
    # reasoning by its own opinion of that reasoning.
    solution: Any = None
    # A question of her own she put to the Master this turn, because she had nothing to answer
    # with. Carried as structure rather than folded into `answer`: "I do not know" and "I do not
    # know, and here is what would help" are the same epistemic state, and a caller that wants
    # only the honest non-answer must still be able to get it.
    asked: Any = None

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
                "loop": self.loop.to_dict() if self.loop is not None else None,
                "field": self.field.to_dict() if self.field is not None else None,
                "trial": self.trial.to_dict() if self.trial is not None else None,
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
        # What the gauntlet reads off this brain when it judges a claim. Refreshed per turn by
        # `_prepare_evidence`; both are empty between turns, which is why an ungrounded claim
        # gets no support rather than stale support from a previous one.
        self.observations: List[str] = []
        self.holdout: List[Any] = []
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
        # Before `metareason`, which registers a strategy bound to it: a calculator built after
        # the strategy table would be registered as absent and never chosen.
        self.calculator = self._build_calculator(c)
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
        # ---- NJP V.04: the Recursive Cognitive Field and the organs it drives ---- #
        # Concepts she invents rather than is given; a simulated universe she can intervene on;
        # an experiment designer that measures information gain; and a belief ledger where every
        # claim carries its own case. Built before `loop` and `field` because both read them.
        self.genesis = self._build_concepts(c)
        self.universe = self._build_universe(c)
        self.designer = self._build_designer(c)
        self.beliefs = self._build_beliefs(c)
        # The subsystem that attacks what the ones above it conclude. After `world`, `universe`
        # and `beliefs`, because every one of its four attacks is settled against one of them —
        # an adversary with nothing to check against is a generator of rhetorical questions.
        self.adversary = self._build_adversary(c)
        # The Cognitive Learning Core, before `metareason` because that registers a strategy
        # bound to it. Everything the Core reads — the grounder, the world, the concept layer,
        # the universe, curiosity — is already built by this line; it needs nothing from `field`,
        # which is why it does not have to wait for the end of this method.
        #
        # Named `learner`, not `core`: `self.core` is already the kernel back-reference set at
        # the top of this method, and shadowing it would sever the brain from the orchestrator
        # that holds it.
        self.learner = self._build_learner(c)
        self.metareason = self._build_metareason(c)
        # The predictive brain: what follows what in the WORLD, as opposed to which of her own
        # cells fire next. Without it she is a fact store that grows; with it she is something
        # that expects, is wrong, and learns from the difference.
        self.predictive = self._build_predictive(c)
        # Stage G: she plans over what she has actually learned, acts, and is graded by the
        # world. Built after `predictive` because it plans by searching that model — an agent
        # with no world model does not plan, it flails.
        self.agent = self._build_agent(c)
        self.curriculum = self._build_curriculum(c)
        # Truth is not relevance, and reasoning is not always what a turn calls for.
        self._speech, self._policy, self.gate = self._build_relevance(c)
        # Last, because it registers repairs against organs built above it and reads them on
        # every turn. Without it the organs are all present and none of them hear from each other.
        self.loop = self._build_loop(c)
        # After the loop, for the same reason the loop comes after `expand`: the field's
        # self-critic reads the error the loop just scored, and a field that ran before it would
        # be diagnosing the previous turn's mistake.
        self.field = self._build_field(c)

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
        """The gauntlet, wired to evidence it can actually read.

        It was built with its default sources and nothing else, and the defaults are inert by
        design: :class:`~nyxara.njp.truth.PredictiveSource` needs a ``predictor`` and held-out
        samples, and :class:`~nyxara.njp.truth.ObservationSource` reads ``context.observations``.
        The brain supplied neither, so **no source could ever support a claim** — only refute one.
        Measured over 113 turns: 108 claims judged, 0 established, and that zero was widely
        readable as "she is being appropriately strict" when it actually meant the strictness was
        never tested. A gate that cannot pass anything is not a gate, it is a wall.
        """
        if not self._gate("truth", True):
            return None
        try:
            from nyxara.njp.truth import (ConsistencySource, FormalSource, LedgerSource,
                                          ObservationSource, PredictiveSource)
            return TruthGauntlet(
                sources=[
                    FormalSource(),
                    PredictiveSource(predictor=self._claim_survives,
                                     min_samples=self._cfg("truth_min_holdout", 2)),
                    ConsistencySource(),
                    LedgerSource(ledger=self.ledger),
                    ObservationSource(),
                ],
                min_sources=self._cfg("truth_min_sources", 2),
                require_hard=self._cfg("truth_require_hard", True),
                ledger=self.ledger)
        except Exception:  # noqa: BLE001 — without it nothing is verified, and nothing claims to be
            return None

    @staticmethod
    def _claim_survives(claim: str, sample: Any) -> bool:
        """Does the claim agree with one **withheld** assertion of the same fact?

        The samples handed to this are restatements of the very ``(subject, predicate)`` the claim
        is about, drawn from turns *other* than the one that produced the claim — see
        :meth:`_prepare_evidence`. So each is a genuine test: the claim was formed without it, and
        it can disagree.

        This is corroboration by independent restatement, not proof, and it is counted as *hard*
        for the one reason the gauntlet cares about — it can be **surprised**. If the Master said
        "Jay" once and "Raj" twice afterwards, the withheld samples disagree with the claim and
        the source refutes it rather than shrugging. That is the same evidence standard
        :meth:`nyxara.njp.levels.MemoryLevels.consolidate` already uses to promote an episode to
        semantic memory, applied to the question of what she may state as fact.
        """
        try:
            obj = str(getattr(sample, "object", "") or "").strip().lower()
            return bool(obj) and obj in str(claim or "").lower()
        except Exception:  # noqa: BLE001
            return False

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

    def _build_loop(self, c: Any) -> Any:
        """The integration layer: what makes the organs learn from each other rather than sit."""
        if not self._gate("loop", True):
            return None
        try:
            from nyxara.njp.integrate import LearningLoop
            return LearningLoop(
                self,
                consolidate_every=self._cfg("consolidate_every_turns", 8),
                discover_every=self._cfg("discover_every_turns", 12),
                wonder_every=self._cfg("wonder_every_turns", 16),
                # These three took the constructor's defaults and had no way to reach config,
                # so three of the six turn cadences were fixed at whatever `integrate.py` said.
                # `attack_every` is the one that mattered: a 1,352-pair study run built 1,750
                # beliefs and, at one attack per twenty turns with `limit=1`, challenged roughly
                # 67 of them. The adversary was not missing — it was rationed, and there was no
                # knob to un-ration it.
                attack_every=self._cfg("attack_every_turns", 20),
                assess_every=self._cfg("assess_every_turns", 24),
                ledger_every=self._cfg("ledger_every_turns", 32),
                train=self._cfg("train_readout", True))
        except Exception:  # noqa: BLE001 — without it the organs never close the loop
            return None

    # ---- NJP V.04 organs ---------------------------------------------------- #
    def _build_concepts(self, c: Any) -> Any:
        """Concept Genesis: kinds invented from observations, and the compression they buy."""
        if not self._gate("concepts", True):
            return None
        try:
            from nyxara.njp.concepts import ConceptGenesis
            return ConceptGenesis(
                min_members=self._cfg("concept_min_members", 2),
                similarity=self._cfg("concept_similarity", 0.34),
                invariant_share=self._cfg("concept_invariant_share", 0.6),
                cover=self._cfg("concept_cover", 0.45))
        except Exception:  # noqa: BLE001 — without it she has facts and no kinds
            return None

    def _build_universe(self, c: Any) -> Any:
        """The internal simulation universe. Takes the world model as its causal skeleton."""
        if not self._gate("universe", True):
            return None
        try:
            from nyxara.njp.universe import InternalUniverse
            return InternalUniverse(world=self.world,
                                    max_depth=self._cfg("universe_depth", 6))
        except Exception:  # noqa: BLE001 — without it she can remember but not imagine
            return None

    def _build_designer(self, c: Any) -> Any:
        """Curiosity as information gain: the experiment that most separates her hypotheses."""
        if not self._gate("designer", True):
            return None
        try:
            from nyxara.njp.universe import ExperimentDesigner
            return ExperimentDesigner()
        except Exception:  # noqa: BLE001
            return None

    def _build_beliefs(self, c: Any) -> Any:
        """Self-model 2.0: one record per belief, carrying its own evidence and falsifier."""
        if not self._gate("beliefs", True):
            return None
        try:
            from nyxara.njp.beliefs import BeliefLedger
            return BeliefLedger(self_model=self.self_model)
        except Exception:  # noqa: BLE001
            return None

    def _build_learner(self, c: Any) -> Any:
        """The Cognitive Learning Core: the loop that makes the other organs add up to learning.

        Given the concept layer and the fact store rather than a copy of either. A Core holding
        its own knowledge would be a second brain to keep in step, which is exactly the
        arrangement this package exists to not have.
        """
        if not self._gate("learner", True):
            return None
        try:
            from nyxara.njp.core import CognitiveLearningCore
            return CognitiveLearningCore(
                self, concepts=self.genesis, grounder=self.grounder,
                universe=self.universe, world=self.world, curiosity=self.curiosity,
                max_depth=self._cfg("learner_max_depth", 4),
                min_members=self._cfg("learner_min_members", 2),
                holdout_share=self._cfg("learner_holdout_share", 0.3),
                represent_every=self._cfg("learner_represent_every", 4),
                test_every=self._cfg("learner_test_every", 6))
        except Exception:  # noqa: BLE001
            return None

    def _build_calculator(self, c: Any) -> Any:
        """Arithmetic. Absent before this: no module in ``nyxara/njp/`` could evaluate anything.

        :mod:`nyxara.njp.prove` is the package's only sympy consumer and it *verifies* claims — an
        arithmetic question reached it and came back INEXPRESSIBLE. So "2+2 kitna hai" descended
        the whole ladder and returned the empty string, which is an honest answer to a question
        she genuinely could not answer, and an absurd one to that question.
        """
        if not self._gate("calculate", True):
            return None
        try:
            from nyxara.njp.calculate import Calculator
            return Calculator(prefer_exact=self._cfg("calculate_prefer_exact", True))
        except Exception:  # noqa: BLE001
            return None

    def _build_metareason(self, c: Any) -> Any:
        """Meta-reasoning: which way of thinking this problem calls for, learned from outcomes.

        The strategies are bound to organs this brain already has, so nothing here re-implements
        reasoning — it decides which existing reasoner runs. A strategy whose organ is absent is
        simply not registered, and the chooser then has one fewer option rather than a broken one.
        """
        if not self._gate("metareason", True):
            return None
        try:
            from nyxara.njp.metareason import MetaReasoner, ProblemKind
            meta = MetaReasoner(meta_learner=self.meta, beliefs=self.beliefs, world=self.world)
            if self.ladder is not None or self.reasoner is not None:
                meta.register("ladder", (ProblemKind.SYMBOLIC, ProblemKind.CONTRADICTION),
                              self._strategy_ladder, prior=0.6)
            if self.world is not None:
                meta.register("causal", (ProblemKind.CAUSAL,), self._strategy_causal, prior=0.6)
            if self.universe is not None:
                meta.register("simulate", (ProblemKind.CAUSAL, ProblemKind.EMPIRICAL),
                              self._strategy_simulate, prior=0.5)
            if self.learner is not None:
                # The derivation strategy: compose stated facts into one that was never stated,
                # or transfer a held-out-confirmed schema onto a subject that never showed the
                # property. Registered for FACTUAL and CAUSAL because those are the kinds whose
                # questions the fact store answers by lookup — and a lookup is precisely what
                # has already failed by the time any strategy here runs.
                # EMPIRICAL is in the list as a fallback rather than as the intended route: the
                # `derivable` context key should have pulled a derivable question into FACTUAL
                # before this table is consulted. It is here for the case where it did not —
                # working an answer out of facts already held is cheaper than designing an
                # experiment to go and get it, so it should at least be *reachable* wherever
                # `experiment` is.
                meta.register("derive",
                              (ProblemKind.FACTUAL, ProblemKind.CAUSAL, ProblemKind.EMPIRICAL),
                              self._strategy_derive, prior=0.65)
            if self.calculator is not None:
                # Registered for EMPIRICAL as well as SYMBOLIC, and that is not belt-and-braces.
                # `grounded is False` adds 0.5 to EMPIRICAL on every unanswered question, so a
                # sum she has never been told the answer to can still classify empirical even
                # with the arithmetic flag set. Being a candidate under both readings means the
                # retry loop reaches it either way; being *preferred* under neither is the
                # chooser's business, which is what the priors are for.
                meta.register("calculate", (ProblemKind.SYMBOLIC, ProblemKind.EMPIRICAL),
                              self._strategy_calculate, prior=0.75)
            if self.grounder is not None:
                meta.register("recall", (ProblemKind.FACTUAL,), self._strategy_recall, prior=0.7)
            if self.self_model is not None or self.beliefs is not None:
                meta.register("introspect", (ProblemKind.INTROSPECTIVE,),
                              self._strategy_introspect, prior=0.6)
            if self.designer is not None:
                meta.register("experiment", (ProblemKind.EMPIRICAL,),
                              self._strategy_experiment, prior=0.4)
            return meta
        except Exception:  # noqa: BLE001
            return None

    def _build_relevance(self, c: Any) -> Tuple[Any, Any, Any]:
        """The speech-act reader, the cognitive policy and the relevance gate.

        Built together because they are one mechanism: the reader says what kind of turn it is,
        the policy says which cognition that kind permits, and the gate enforces it over memory.
        Switching the group off restores the previous behaviour exactly — every turn gets the
        full apparatus and recall attaches whatever was nearest — which is worth being able to
        do, and worth naming as the thing that produced a pendulum law in answer to "how are you".
        """
        if not self._gate("relevance", True):
            return None, None, None
        try:
            from nyxara.njp.relevance import CognitivePolicy, RelevanceGate, SpeechActReader
            return (SpeechActReader(), CognitivePolicy(),
                    RelevanceGate(threshold=self._cfg("relevance_threshold", 0.22)))
        except Exception:  # noqa: BLE001 — without it she is credulous, not broken
            return None, None, None

    def _build_predictive(self, c: Any) -> Any:
        """The world-state sequence model. Stage A of the curriculum, and the floor under it."""
        if not self._gate("predictive", True):
            return None
        try:
            from nyxara.njp.predictive import PredictiveWorldModel
            return PredictiveWorldModel(
                max_order=self._cfg("predictive_order", 3),
                min_evidence=self._cfg("predictive_min_evidence", 2))
        except Exception:  # noqa: BLE001 — without it she remembers but never expects
            return None

    def deliberate(self, question: str = "", *, action: str = "") -> Any:
        """Think about the current situation as an inspectable object, not a string.

        The Master's sequence, walked over the predictive model: what do I know, what am I
        assuming, what could explain this, which explanation predicts best, what would prove me
        wrong, conclusion. Returns the :class:`~nyxara.njp.predictive.ThoughtState` itself so the
        reasoning can be read while it is happening rather than inferred from what came out.
        """
        try:
            from nyxara.njp.predictive import ThoughtState
            state = ThoughtState(context=str(question or ""))
            if self.predictive is None:
                return state
            return state.deliberate(self.predictive, action=action,
                                    beliefs=self.beliefs, concepts=self.genesis)
        except Exception:  # noqa: BLE001
            return None

    def _build_agent(self, c: Any) -> Any:
        """Stage G — the loop that closes: goal → plan → action → outcome → better model."""
        if not self._gate("agency", True) or self.predictive is None:
            return None
        try:
            from nyxara.njp.agency import Agent
            return Agent(self.predictive, max_depth=self._cfg("plan_depth", 4))
        except Exception:  # noqa: BLE001 — without it she intends but never acts
            return None

    def _build_adversary(self, c: Any) -> Any:
        """The subsystem that goes after her own conclusions.

        Given the world record and the simulator rather than copies: an attack is only worth
        anything if it is checked against the same evidence the belief was built from, and a
        second private store would let a claim survive an attack on data that no longer matches
        what she actually knows.
        """
        if not self._gate("adversary", True):
            return None
        try:
            from nyxara.njp.adversary import SelfAttacker
            return SelfAttacker(
                world=self.world, universe=self.universe, beliefs=self.beliefs,
                min_support=self._cfg("adversary_min_support", 3),
                coincidence_lift=self._cfg("adversary_coincidence_lift", 0.1))
        except Exception:  # noqa: BLE001
            return None

    def _build_curriculum(self, c: Any) -> Any:
        """The nine stages, and the refusal to report one as reached before it is."""
        if not self._gate("curriculum", True):
            return None
        try:
            from nyxara.njp.curriculum import Curriculum
            return Curriculum()
        except Exception:  # noqa: BLE001
            return None

    def pursue(self, goal: Any, *, actuator: Any = None, steps: int = 4) -> Any:
        """Plan toward ``goal`` over the learned world model and act, re-planning each step.

        Without an ``actuator`` this returns the plan as a **proposal** and takes no action —
        which is the honest default, because deciding whether a proposed action may run is the
        kernel's gate to make, never this brain's.

        **This has no production callers, and adding one would change nothing.** It is tempting
        to read that as missing wiring — the curriculum's Stage G scores `agency.success_rate`
        and is therefore unreachable, which looks like a call site nobody wrote. It is not.
        Measured on a brain taught three causal facts: ``known_actions`` 0, and
        ``pursue("garmi")`` returns ``[]``.

        The reason is circular and structural. :meth:`Agent.plan` searches over the actions
        :class:`~nyxara.njp.predictive.PredictiveWorldModel` knows, and the only thing that ever
        teaches it one is :meth:`Agent.act` (``agency.py``), which runs only after a plan is
        found. No plan without actions, no actions without a plan. Nothing else in NJP names an
        action: there is no actuator, no action vocabulary, no affordance list — she models what
        follows what, and never what she could *do*.

        So Stage G is not blocked by a missing caller and will not be unblocked by one. Calling
        this from the turn loop would add a scheduled no-op and move a counter off zero without
        anything having happened, which is the kind of wiring this package has already been
        burned by. The gap is an action vocabulary, and that is a design question rather than a
        line of plumbing.
        """
        try:
            if self.agent is None:
                return []
            return self.agent.pursue(goal, actuator=actuator, steps=steps)
        except Exception:  # noqa: BLE001
            return []

    def _recall(self, thought: NJPThought) -> None:
        """Answer from a relation the question did not ask for, about the entity it did name.

        The gate this passes through is the same one a remembered episode passes through, and for
        the same reason: a fact being *about* what was asked is a claim about relevance, and
        relevance is exactly what :class:`~nyxara.njp.relevance.RelevanceGate` is for. A retrieved
        fact that cannot clear it is not offered, however confidently it was stored.

        Never ``KNOWN``. Having found something is not corroboration of it.
        """
        try:
            if self.grounder is None:
                return
            grounding = getattr(thought.percept, "grounding", None)
            if not getattr(grounding, "is_question", False):
                return
            # A social or reflexive turn has no business reaching the fact store at all, and the
            # policy that says so has already run for `_compose`. Asking it again here keeps the
            # rule in one place rather than trusting that nothing has changed in between.
            if thought.act is not None and self._policy is not None:
                if self._policy.forbids_world_knowledge(thought.act):
                    return
            answer = self.grounder.answer_by_recall(thought.stimulus)
            if not answer.answered:
                return
            thought.answer = str(answer.text)[:1000]
            thought.recalled = answer.why
            # Being retrieved and used IS the evidence consolidation is looking for, and until
            # now it never reached it: grounding answers from a fact store keyed by
            # (subject, predicate) while `levels` is keyed by the turn a memory arrived on, so a
            # fact used a hundred times looked exactly like one nobody had read. The canonical
            # claim is the join.
            if self.levels is not None and answer.triples:
                triple = answer.triples[0]
                self.levels.touch_claim(
                    f"{triple.subject}|{triple.predicate}|{triple.object}".lower())
        except Exception:  # noqa: BLE001 — a failed recall leaves the turn as it was
            return

    def _ask_back(self, thought: NJPThought) -> None:
        """Put her own best question to the Master, on a turn where she had nothing to answer.

        :meth:`~nyxara.njp.curiosity.Curiosity.ask` had no caller anywhere, so ``asked`` never
        incremented, so ``stale`` was permanently ``False``, so ``stale_questions`` was always
        empty and the escalation path behind it could not be reached. A queue of questions nobody
        ever puts is not curiosity — it is a list.

        Only on a turn that produced no answer, and only where the speech act permits reaching
        for world knowledge at all: a greeting must not come back with "what is a tachyon?", which
        is the same failure :mod:`nyxara.njp.relevance` was written for, pointed outward. The
        question is recorded on the thought and never folded into ``answer`` — a caller asking
        whether she could answer must still get the honest empty string.

        ``top`` already excludes stale questions, so asking the same thing a fourth time is
        impossible by construction; what has gone stale is reported by :meth:`shadow` instead,
        which is the escalation this makes reachable.
        """
        try:
            if self.curiosity is None or thought.answer:
                return
            act = getattr(thought, "act", None)
            if act is not None and self._policy is not None:
                if self._policy.forbids_world_knowledge(act):
                    return
            # Look before marking. `ask` increments `asked` on whatever is top, and a question
            # curiosity priced as "go and find this out yourself" is not one she has put to him —
            # counting it would drive a gap she never voiced to stale and silence it.
            candidate = self.curiosity.top()
            if candidate is None or getattr(candidate, "action", "") != "ask":
                return
            thought.asked = self.curiosity.ask()
        except Exception:  # noqa: BLE001 — a turn that cannot ask still answers honestly
            return

    def budget(self) -> Dict[str, Any]:
        """What this turn is allowed to spend, chosen by what has actually been paying.

        The homeostasis loop was almost entirely built and had no middle. ``_build_meta``
        registers three arms — ``settle_steps``, ``recall_k``, ``reason_depth`` — over real
        settings, with the comment that each "changes what a turn costs, so a win here is a
        measured win rather than a preference". :meth:`~nyxara.njp.selfmodel.MetaLearner.choose`
        picks between them, :meth:`resolve` already rewards one of them. Nothing ever *read a
        choice back*, so every turn spent the same fixed config constant and the bandit learned
        about a knob that was never turned.

        Choosing is what registers the pending arm that :meth:`resolve` later grades, so this is
        also what makes the existing reward meaningful rather than a credit assigned to whichever
        arm happened to be chosen last.

        Falls back to the configured constants for any arm that has not been registered, which is
        the honest default: an untuned knob is the one she shipped with, not a guess.
        """
        out: Dict[str, Any] = {
            "settle_steps": int(self._cfg("max_settle_steps", 24)),
            "recall_k": int(self._cfg("recall_k", 5)),
            "reason_depth": str(self._cfg("reason_max_rung", "proof")),
            "learned": False,
        }
        if self.meta is None:
            return out
        try:
            for arm in ("settle_steps", "recall_k", "reason_depth"):
                chosen = self.meta.choose(arm)
                if chosen is None:
                    continue
                out[arm] = chosen.value
                out["learned"] = True
        except Exception:  # noqa: BLE001 — an unchosen knob keeps the value she was built with
            return out
        return out

    def shadow(self) -> Dict[str, Any]:
        """What she does not know, as one structure instead of five nobody joined.

        Ignorance was tracked in five places and read in none of them usefully. The grounder
        computed the words that reached no entity and the only consumer rendered them as a
        sentence on turns where she had nothing else to say. The belief ledger's ``audit`` — which
        names, per belief, whether confidence exceeds evidence and whether a falsifier was ever
        stated — was called once, by its own ``stats``, which kept the count and dropped the rows.
        ``ThoughtState``'s assumptions were never written at runtime at all. Each store was
        honest; none of them was answerable.

        Two axes, kept apart because they call for different repairs. **Gaps** are propositions
        she lacks and curiosity already prices them by expected information gain against what
        they cost, so this reads that pricing rather than inventing a second one. **Faculties** is
        the other kind of not-knowing — which of her own organs is untrustworthy — and it stays
        separate because it is about her, not about the world, and merging the two would put
        "I am bad at recall" in a queue of things to go and look up.

        Read-only. Nothing here decides anything; it reports what the organs already hold, which
        is the whole point — an aggregator that also acted would become a sixth store.
        """
        out: Dict[str, Any] = {"gaps": [], "by_gap": {}, "stale": [],
                               "faculties": {}, "open": 0}
        try:
            if self.curiosity is not None:
                questions = list(self.curiosity.open_questions())
                # Most valuable first: curiosity has already priced each one by what answering it
                # buys against what it costs, and re-ranking here would be a second opinion with
                # no new information behind it.
                questions.sort(key=lambda q: float(getattr(q, "value", 0.0)), reverse=True)
                out["open"] = len(questions)
                out["gaps"] = [q.to_dict() for q in questions[:12]]
                for question in questions:
                    kind = str(getattr(question, "gap", "") or "")
                    out["by_gap"][kind] = out["by_gap"].get(kind, 0) + 1
                # Asked three times and still open. `top` refuses to surface these, so without a
                # reader they simply went quiet — which reads as "no longer curious" when what
                # actually happened is "asked repeatedly and never answered". That is the one
                # state in the queue that wants a different response rather than more patience.
                out["stale"] = [q.to_dict() for q in self.curiosity.stale_questions()[:6]]
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.self_model is not None:
                weakest = self.self_model.weakest()
                out["faculties"] = {
                    # `certainty` had no reader anywhere, and it is the one calibrated
                    # "how much do I know about how good I am" number in the package.
                    "measured": {name: {"level": round(c.level, 4),
                                        "certainty": round(c.certainty, 4),
                                        "weak": c.weak}
                                 for name, c in self.self_model.capabilities.items()},
                    "weakest": weakest.name if weakest is not None else "",
                }
        except Exception:  # noqa: BLE001
            pass
        return out

    def report_card(self) -> Dict[str, Any]:
        """Which of the nine stages she has actually reached, measured, not claimed."""
        try:
            if self.curriculum is None:
                return {}
            return self.curriculum.assess(self).to_dict()
        except Exception:  # noqa: BLE001
            return {}

    def _build_field(self, c: Any) -> Any:
        """The Recursive Cognitive Field — both loops, over the organs built above."""
        if not self._gate("field", True):
            return None
        try:
            from nyxara.njp.field import RecursiveCognitiveField
            return RecursiveCognitiveField(
                self, concepts=self.genesis, universe=self.universe,
                designer=self.designer, beliefs=self.beliefs, meta=self.metareason,
                crystallise_every=self._cfg("crystallise_every_turns", 4),
                meta_every=self._cfg("meta_cycle_every_turns", 25))
        except Exception:  # noqa: BLE001 — without it the new organs never learn from each other
            return None

    # ---- the strategies the meta-reasoner chooses between --------------------- #
    def _strategy_recall(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            answer = self.grounder.answer(problem) if self.grounder is not None else None
            return getattr(answer, "text", None) or getattr(answer, "object", None)
        except Exception:  # noqa: BLE001
            return None

    def _strategy_causal(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            subject = str(ctx.get("subject") or "").strip().lower()
            if not subject or self.world is None:
                return None
            why = self.world.why(subject)
            if not getattr(why, "explained", False):
                return None
            return ", ".join(c.cause for c in getattr(why, "causes", [])[:3])
        except Exception:  # noqa: BLE001
            return None

    def _strategy_simulate(self, problem: str, ctx: Dict[str, Any]) -> Any:
        """Answer by intervening on the world model rather than recalling what went with what.

        The rendering states the intervention, the consequences it entails, and the confidence the
        do-operator itself earned — which is a real number, off the arrow's fit and how far past
        the observed range the question reached, not a decoration. A counterfactual shown without
        it is indistinguishable from an observation, and the two are not the same kind of claim.

        A counterfactual that entails no consequence is not an answer and returns nothing, so the
        turn goes to whatever can answer it instead of to a restatement of the question's own
        premise.
        """
        try:
            variable = str(ctx.get("variable") or "").strip().lower()
            if not variable or self.universe is None:
                return None
            value = float(ctx.get("value", 0.0))
            out = self.universe.what_if(variable, value)
            if not out.answerable:
                return None
            effects = [d for d in out.changed() if d.variable != variable]
            if not effects:
                return None
            was = out.factual.get(variable)
            done = (f"{variable} {float(was):g} → {value:g}" if was is not None
                    else f"{variable} = {value:g}")
            entails = "; ".join(f"{d.variable} {d.before:g} → {d.after:g}" for d in effects[:3])
            return f"{done} predicts {entails} (confidence {out.confidence:.2f})"
        except Exception:  # noqa: BLE001
            return None

    def _resolve_variable(self, name: str) -> str:
        """A bare noun from a question, as the universe's own variable name.

        The universe names a variable ``entity.attribute``, because that is what lets it tell two
        attributes of one thing from two unrelated readings. A question says "the water", never
        "plant.water", so the two have to be reconciled somewhere. Here, and against the variables
        she has actually observed — guessing an entity would invent a subject the Master never
        mentioned and simulate confidently over it.
        """
        want = str(name or "").strip().lower()
        if not want:
            return ""
        try:
            for variable in list(getattr(self.universe, "state", None) or {}):
                if variable == want or variable.rsplit(".", 1)[-1] == want:
                    return variable
        except Exception:  # noqa: BLE001 — an unresolvable name simply is not simulated
            return ""
        return ""

    def _counterfactual_context(self, question: str) -> Dict[str, Any]:
        """The intervention a question asks for, as a variable and the value to set it to.

        Empty where the question names no intervention, names a variable she has never observed,
        or names an intervention whose size it does not state — each of which is a reason to
        decline rather than to simulate something adjacent. This is the dict that makes
        :meth:`_strategy_simulate` reachable at all; without a ``variable`` key it returns ``None``
        on every call, which is why the do-operator went unused while working perfectly.

        A factor resolves against what was actually observed, so "halve the water" means half of
        what this plant has been getting rather than half of an arbitrary unit.
        """
        out: Dict[str, Any] = {}
        if self.universe is None:
            return out
        try:
            text = str(question or "")
            absolute = _INTERVENTION_ABSOLUTE.search(text)
            if absolute is not None:
                variable = self._resolve_variable(absolute.group("var"))
                if variable:
                    return {"variable": variable, "value": float(absolute.group("val"))}

            match = _INTERVENTION.search(text) or _INTERVENTION_TRAILING.search(text)
            if match is None:
                return out
            variable = self._resolve_variable(match.group("var"))
            factor = _INTERVENTION_FACTOR.get(match.group("op").lower())
            if not variable or factor is None:
                return out
            current = getattr(self.universe, "state", None) or {}
            observed = current.get(variable)
            if observed is None:
                return out
            return {"variable": variable, "value": float(observed) * factor}
        except Exception:  # noqa: BLE001 — an unparsed intervention is simply not simulated
            return {}

    def _strategy_ladder(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            if self.reasoner is None:
                return None
            conclusion = self.reasoner.reason(problem)
            # Honour the ladder's own refusal to commit. This path used to read the answer
            # regardless of `decided`, so a conclusion the direct fallback would have thrown away
            # for being too close to call entered the meta-reasoner as a strategy result — one
            # margin rule enforced in one caller and ignored in the other, on the same object.
            if conclusion is None or not conclusion.decided:
                return None
            return conclusion.answer or None
        except Exception:  # noqa: BLE001
            return None

    def _strategy_derive(self, problem: str, ctx: Dict[str, Any]) -> Any:
        """Answer by composing facts, or by transferring from a kind. Never by looking up.

        The answer carries its route in ``kind``, and only ``direct`` is a fact — which this
        never returns, because a direct hit means the grounder already answered and this strategy
        was never reached. So everything from here is defeasible by construction, and the caveat
        travels with it rather than being attached to the sentence: a composed or transferred
        answer that reads like a stated one is the failure this whole layer is supposed to avoid.
        """
        try:
            if self.learner is None:
                return None
            # The classifier already ran this walk to decide what kind of problem this is; doing
            # it again here would be the same graph search twice per turn for one answer.
            cached = ctx.get("derived")
            if cached:
                return str(cached)
            derived = self.learner.answer(problem)
            if not derived.ok:
                return None
            return derived.answer or None
        except Exception:  # noqa: BLE001
            return None

    def _strategy_calculate(self, problem: str, ctx: Dict[str, Any]) -> Any:
        """Work out the value. The one strategy here whose answer is not a belief.

        Everything else in this table returns something that could be wrong and travels with a
        confidence — a recalled fact, a simulated consequence, a rung of the ladder. An evaluated
        expression is not of that kind: ``2+2`` is 4 in every world she could be in, so it is
        stated flatly and the gauntlet downstream has nothing to weigh it against. That is why
        the calculator refuses anything it cannot close rather than approximating: an organ
        whose output is exempt from doubt has to earn the exemption on every call.
        """
        try:
            if self.calculator is None:
                return None
            evaluation = self.calculator.evaluate(problem)
            return evaluation.text if evaluation.ok else None
        except Exception:  # noqa: BLE001
            return None

    def _strategy_introspect(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            if self.beliefs is None:
                return None
            unknown = self.beliefs.unknown()
            known = self.beliefs.known()
            return (f"{len(known)} beliefs held with evidence, "
                    f"{len(unknown)} open or unsupported")
        except Exception:  # noqa: BLE001
            return None

    def _strategy_experiment(self, problem: str, ctx: Dict[str, Any]) -> Any:
        try:
            if self.designer is None:
                return None
            experiment = self.designer.design()
            if experiment is None:
                return None
            return f"run {experiment.name} (expected gain {experiment.gain:.3f} bits)"
        except Exception:  # noqa: BLE001
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

                # What this turn may spend, chosen by what has been paying rather than by a
                # constant. Read once so the settle and the recall are budgeted together — two
                # separate draws would let her deliberate deeply on a narrow recall and call the
                # combination a win.
                spend = self.budget()

                self.fabric.stimulate(out.cells)
                out.settled = self.fabric.settle(max_steps=int(spend["settle_steps"]))

                if self.memory is not None:
                    try:
                        out.recall = self.memory.recall(out.stimulus,
                                                        k=int(spend["recall_k"]))
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
            # Cleared at the top of the turn rather than only before a judgement, so the pools are
            # never a turn out of date. `_prepare_evidence` runs just before the gauntlet and so
            # nothing has ever *read* stale evidence — but leaving the previous turn's facts
            # sitting on the brain makes the invariant unverifiable from outside, and an invariant
            # nobody can check is one that quietly stops holding.
            self.observations = []
            self.holdout = []

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
            # "I'm certain that I understand: <your words back>" is not a reply, it is a machine
            # narrating its own comprehension. Understanding belongs in the internal state, which
            # is where `out.act` already carries it; a turn that produced only an echo is a turn
            # with no answer, and saying so is more honest than saying it back.
            if self._is_echo(out):
                out.answer = ""
            self._set_epistemic(out)
            # What she committed to is registered by :mod:`nyxara.njp.integrate`, not here.
            #
            # This used to call ``predictor.predict(out.stimulus, out.answer)`` directly, and the
            # key was the raw stimulus — so the only thing that could ever have scored it was the
            # identical sentence arriving again with the truth attached, which never happens.
            # Measured over 113 turns: 113 predictions registered, 0 scored, and an "open" queue
            # that only ever grew. A prediction that cannot in principle be observed is not a
            # prediction; it is a counter going up.
            #
            # The loop registers two kinds that CAN be resolved: the manifold's pre-settle
            # anticipation, scored against what actually fired microseconds later, and an unanswered
            # question keyed on ``(subject, predicate)`` so the Master's later statement of that
            # very fact is what grades it.

            # 4b. DELIBERATE — a question that structure could not answer gets the ladder.
            #
            # Gated on "asked, and grounding came back empty" for a reason: descending five rungs
            # for a question the graph answered outright is the "runs a proof engine over what is
            # my name" failure the ladder module names in its own docstring. This way the
            # reasoner can only ever add an answer where there was none, never overwrite one.
            #
            # Before this, `reason.passes` was 0 on every session — the ladder, the problem
            # states, the rejected-hypothesis memory and the debate were all reachable only from
            # tests. Curiosity's known-unknown gap reads `reasoner.problems`, so an unwired
            # reasoner also meant she could never notice a question she had failed to answer.
            if not out.answer:
                self._deliberate(out)
                # 4c. RECALL — the last thing tried, and last on purpose.
                #
                # She may hold the entity the question named under a relation it did not ask for:
                # asked "what is deep learning" with `deep learning --part_of--> machine learning`
                # in the store, `_lookup` misses on the predicate and the fact is unreachable
                # though it is exactly what was wanted. That was the second measured bottleneck
                # after extraction — 521 facts held, and questions about them answered UNKNOWN.
                #
                # After deliberation, never before it. A retrieved fact is the weakest answer that
                # still counts as one, and running it earlier stops the reasoner from ever being
                # reached: `sparrow needs water`, derived through `sparrow is_a bird`, was
                # measured being replaced by the flat `sparrow is_a bird` the moment retrieval
                # answered first. Strictly worse, strictly sooner.
                if not out.answer:
                    self._recall(out)
                # The epistemic pass above ran against an empty answer, so a deliberated one
                # would keep `unknown` at confidence 0.0 however well it was derived. Re-run
                # rather than move: the first pass also feeds the echo check, which has to
                # happen before deliberation, not after it.
                if out.answer:
                    self._set_epistemic(out)
                else:
                    self._ask_back(out)

            # 5. nothing is stated as fact until the gauntlet says it may be
            if self.truth is not None and out.answer:
                self._prepare_evidence(out)
                out.judgement = self.truth.judge(out.answer, context=self)

            # 6. remember the conclusion, not the question: storing a bare question would make it
            # its own best match and echo back as its own answer.
            if remember and out.answer:
                self._remember_turn(out)

            # 7. ATTEND — every organ bid, one won. This is where the turn stops being a pile of
            # six opinions and becomes one thing she is actually thinking about.
            if self.attention is not None:
                out.focus = self.attention.attend(out)

            # 8. an episode for the discoverer: what was present, and what followed from it. This
            # is the raw material abstractions are proposed from — one episode is never a
            # pattern, which is exactly why it is only recorded here and judged elsewhere.
            #
            # The consequent used to be `out.answer` and *only* `out.answer`, which meant that
            # every turn in which the Master stated something rather than asked something
            # recorded no episode at all. Measured over a 20-turn session of plain statements:
            # 3 episodes, 0 abstractions, compression 1.0 — the compressor was being starved,
            # not failing. A statement is the better raw material of the two: "aag se garmi" is
            # an antecedent and a consequent already, whereas an answer is something she said.
            if self.discoverer is not None and out.percept is not None:
                self._observe_episodes(out)

            # 9. EXPAND — the physical growth, after every single turn
            out.growth = self._expand(out, outcome=outcome)

            # 10. CLOSE THE LOOP — score what was predicted, diagnose what missed, route the
            # repair to the organ that owns it, grade the faculties this turn exercised, take a
            # gradient step, and run the slow organs when their turn-count comes round.
            #
            # After expand, deliberately: the growth report is part of what the turn produced, and
            # a loop that ran before it would be scoring a fabric state that no longer exists.
            if self.loop is not None:
                out.loop = self.loop.close(out)

            # 11. THE FIELD — form concepts from what was grounded, keep the simulated universe
            # in step with the causal skeleton, and put this turn's error through the self-critic
            # that decides between adjusting a coefficient and **restructuring what she can
            # represent at all**. After the loop, because the critic reads the error the loop
            # just scored; a field that ran before it would be diagnosing the previous turn.
            if self.field is not None:
                out.field = self.field.cycle(out)
                # Loop 2 on its own slower count: evaluate herself, find what is limiting her,
                # propose one bounded change, and let a held-out benchmark decide whether it
                # survives. Reverted unless it strictly wins.
                if self.field.due_for_meta():
                    out.trial = self.field.meta_cycle()

            # 12. THE CORE — induce schemas over the kinds the field just formed, score the
            # standing ones against facts they were never built from, and drop the ones that
            # fail. Last on purpose: REPRESENT reads `concepts`, which step 11 has only just
            # updated with this turn's triples, so a Core running any earlier would be
            # generalising over the previous turn's kinds. This is the step at which having the
            # organs becomes learning from them.
            if self.learner is not None:
                out.learning = self.learner.cycle(out)

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

    def _observe_episodes(self, thought: NJPThought) -> None:
        """Everything this turn offers the compressor, as ``antecedents → consequent`` episodes.

        Three sources, in decreasing order of how directly they state a regularity:

        1. **Grounded relations.** ``(subject, predicate, object)`` is already the shape the
           discoverer wants — the subject and predicate are the condition, the object is what
           follows. This is the source that was missing entirely.
        2. **Stated laws.** A causal triple is the same thing said outright, and worth recording
           under the bare cause as well, so "aag" alone becomes predictive of "garmi".
        3. **Her own conclusion**, when she reached one. Kept last and kept honest: what she
           concluded is evidence about her, not about the world, so it never displaces (1).
        """
        try:
            percept = thought.percept
            concepts = list(getattr(percept, "concepts", ()) or [])[:8]
            grounding = getattr(percept, "grounding", None)
            seen = 0
            for triple in (getattr(grounding, "triples", None) or [])[:4]:
                subject = str(getattr(triple, "subject", "") or "").strip().lower()
                predicate = str(getattr(triple, "predicate", "") or "").strip().lower()
                obj = str(getattr(triple, "object", "") or "").strip().lower()
                if not subject or not predicate:
                    continue
                consequent = obj or predicate
                antecedents = [subject, predicate] if obj else [subject]
                self.discoverer.observe(antecedents, consequent)
                seen += 1
                # The cause on its own, so a regularity can be found over the cause rather than
                # only over the exact sentence that stated it.
                if obj and predicate in ("causes", "produces", "requires"):
                    self.discoverer.observe([subject], consequent)
            if not seen and concepts and thought.answer:
                self.discoverer.observe(concepts, thought.answer[:60])
        except Exception:  # noqa: BLE001 — a missed episode costs one sample, never the turn
            pass

    def _prepare_evidence(self, thought: NJPThought) -> None:
        """Lay out what the gauntlet may read about *this* claim, and nothing else.

        Two pools, and the difference between them is the difference between soft and hard:

        * :attr:`observations` — the independent record that bears on the claim. Grounded facts
          and world events, **excluding the assertion this very turn just made**. Without that
          exclusion a statement would corroborate itself: she would ground "my name is Jay",
          immediately find "Master has_name Jay" in the record, and report that an outside source
          agreed with her. One source counted twice is not two sources.
        * :attr:`holdout` — restatements of the exact ``(subject, predicate)`` the claim is about,
          from turns other than the one that produced it. These are withheld in the sense the
          gauntlet requires: the claim was formed without them and they are free to disagree.

        Both are cleared first. A claim she has no evidence about must come out of the gauntlet as
        a conjecture, and leaving the previous turn's evidence lying around would quietly
        establish it against facts that have nothing to do with it.
        """
        self.observations = []
        self.holdout = []
        try:
            grounder = self.grounder
            if grounder is None:
                return
            grounding = getattr(thought.percept, "grounding", None)
            mine = (getattr(grounding, "triples", None) or [])
            # Excluded by **identity**, not by value. Excluding every triple that happens to say
            # the same thing would throw away the Master's independent earlier statements of it,
            # which are the only corroboration a restated fact can ever have.
            current = {id(t) for t in mine}

            for triples in grounder.facts.values():
                for triple in triples:
                    if triple.superseded or id(triple) in current:
                        continue
                    # The surface the Master actually wrote, where it was kept. That is what an
                    # "independent recorded observation" is; the rendered triple is her own
                    # paraphrase of it, and it also overlaps a short answer far worse — "Jay"
                    # against "Master has_name Jay" scores 0.33 and misses the floor by a hair.
                    self.observations.append(str(triple.text or "").strip() or
                                             f"{triple.subject} {triple.predicate} {triple.object}")

            # The held-out half: other assertions of whatever this turn is actually about.
            for triple in mine:
                key = (triple.subject.lower(), triple.predicate)
                for prior in grounder.facts.get(key, []):
                    if id(prior) not in current and not prior.superseded:
                        self.holdout.append(prior)
            # A question is about a fact she is reporting rather than asserting; the answer's own
            # supporting triples are what it rests on, so their restatements are the held-out set.
            answer = getattr(grounding, "answer", None)
            for triple in (getattr(answer, "triples", None) or []):
                key = (triple.subject.lower(), triple.predicate)
                self.holdout.extend(t for t in grounder.facts.get(key, []) if not t.superseded)

            self.observations = self.observations[:256]
            self.holdout = self.holdout[:64]
        except Exception:  # noqa: BLE001 — no evidence means no support, which is the safe default
            self.observations = []
            self.holdout = []

    def _deliberate(self, thought: NJPThought) -> None:
        """Work an unanswered question down the ladder, and take the answer only if it decided.

        An undecided conclusion is left on the floor deliberately. :class:`~nyxara.njp.reason.
        Reasoner` refuses to commit when its top two hypotheses are within ``decide_margin`` of
        each other, and honouring that refusal is the whole value of having it — overwriting an
        empty answer with the marginally higher of two indistinguishable guesses would convert an
        honest "I don't know" into a confident coin-flip.

        The problem state survives either way, which is what lets curiosity find the question
        again later and what stops her re-proposing a hypothesis already rejected.
        """
        try:
            grounding = getattr(thought.percept, "grounding", None)
            if grounding is None or not getattr(grounding, "is_question", False):
                return

            # Choose *how* to think before thinking. The ladder is one strategy among several and
            # it is the wrong one for a causal question or an empirical one — it was previously
            # the only one, so every unanswered question got the same descent regardless of what
            # kind of question it was. The meta-reasoner classifies first, picks the strategy
            # that has actually been working for that kind, criticises what comes back, and takes
            # a second opinion when the classification is close or the critic is unhappy.
            if self.metareason is not None:
                triples = list(getattr(grounding, "triples", None) or [])
                context = {
                    "grounded": bool(triples),
                    "subject": (str(getattr(triples[0], "subject", "")) if triples else ""),
                    "about_self": bool(getattr(thought.intent, "about_self", False)),
                }
                # The intervention the question asks for, where it asks for one. Without these
                # two keys `_strategy_simulate` returns None on every call, so the do-operator
                # was registered, chosen, run and unable to do anything — the gap that made a
                # working causal engine unreachable from a causal question.
                context.update(self._counterfactual_context(thought.stimulus))
                # Whether there is a closed sum in here at all. A parse result rather than a
                # keyword, for the same reason the intervention above is: the classifier can
                # count digits and operators but cannot tell "2+2" from "100 degree at 3pm", and
                # it was routing the first of those to the empirical strategies.
                if self.calculator is not None:
                    from nyxara.njp.calculate import expression_in
                    context["arithmetic"] = expression_in(thought.stimulus) or ""
                # Whether this follows from facts she already holds. Derived here rather than
                # inside the strategy so the *classifier* can see it — a lookup miss otherwise
                # reads as "go and find out" and never reaches the reasoning that would settle
                # it without leaving. Carried in the context so `_strategy_derive` reuses this
                # derivation instead of recomputing the same walk.
                if self.learner is not None:
                    derived = self.learner.answer(thought.stimulus)
                    if derived.ok:
                        context["derivable"] = True
                        context["derived"] = derived.answer
                        thought.derivation = derived
                solution = self.metareason.solve(thought.stimulus, context=context)
                thought.solution = solution
                if solution.assertable and solution.answer:
                    thought.answer = str(solution.answer)
                    return
                # Not assertable is not nothing: the critic's defects are why she is not saying
                # it, and they are the honest content of an unanswered turn.

            if self.reasoner is None:
                return
            novelty = float(getattr(thought.percept, "novelty", 0.5) or 0.5)
            # The depth arm caps how far down the ladder this turn may go. `depth_for` still
            # chooses within that cap from the turn's own uncertainty and stakes — the budget
            # says what she may afford, not what this question needs, and conflating the two
            # would make an easy question expensive just because she has been doing well.
            ceiling = str(self.budget().get("reason_depth") or "")
            conclusion = self.reasoner.reason(
                thought.stimulus, uncertainty=novelty,
                stakes=self._cfg("reason_stakes", 0.5), task="question",
                max_rung=ceiling or None)
            if conclusion is not None and conclusion.decided and conclusion.answer:
                thought.answer = conclusion.answer
        except Exception:  # noqa: BLE001 — a failed deliberation leaves the turn unanswered
            pass

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
                    source=f"turn-{self.turns}", claim=self._claim_of(thought))
                return
            if self.memory is not None:
                self.memory.remember(f"turn-{self.turns}", thought.answer,
                                     kind=("conclusion" if thought.verified else "episode"),
                                     cue=thought.stimulus)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _claim_of(thought: NJPThought) -> str:
        """What this turn *asserted*, canonically — the thing recurrence should be counted over.

        :mod:`nyxara.njp.levels` falls back to a bag of the memory's words, which cannot see that
        "Deep learning is a subset of machine learning" and "ML contains deep learning" are one
        claim. It has the triple by this point in the turn, and the triple is the claim: two
        sentences say the same thing exactly when they assert the same relation between the same
        two entities.

        Empty for a turn that grounded nothing, which leaves the fallback in place rather than
        inventing an identity for text nobody parsed.
        """
        try:
            grounding = getattr(thought.percept, "grounding", None)
            triples = list(getattr(grounding, "triples", None) or [])
            if not triples:
                return ""
            # A turn stating several relations is identified by all of them, sorted so the order
            # they were extracted in cannot make one turn look unlike another that said the same.
            return " ; ".join(sorted(
                f"{t.subject}|{t.predicate}|{t.object}".lower() for t in triples[:4]))[:200]
        except Exception:  # noqa: BLE001
            return ""

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

            # 0. WHAT KIND OF TURN IS THIS, before any machinery is pointed at it.
            #
            # The failure this closes, from the Master's own transcript: he asked "How are you
            # NYXARA?", perception correctly called it chat, and recall→reason ran anyway because
            # that is what came next — returning a verified pendulum-period law with confidence
            # raised to 1.00. Every stage worked. What was missing was anything that asked
            # whether a true fact had the slightest thing to do with the question.
            #
            # A social or reflexive act is answered from the relationship and her own state, and
            # `CognitivePolicy` forbids the world-knowledge pathways outright — so there is no
            # route by which a physics law can reach a greeting, however well-established it is.
            act = self._act(thought)
            thought.act = act
            if act is not None and self._policy is not None:
                if self._policy.forbids_world_knowledge(act):
                    return self._answer_socially(thought, act)

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
            #
            # This was tried the other way and the numbers said no. Letting a question answer from
            # recall-through-the-gate took in-sample coverage from 4.0% to 37.5% (200 taught
            # questions, 8 → 75 answered, 4 → 59 scoring F1 ≥ 0.4) — and on held-out questions it
            # turned 0 honest abstentions into 3 confident wrong ones while gaining not a single
            # correct answer:
            #
            #     "Give me an example of irony?"        → a memory about the word "charge"
            #     "Which tree has aromatic blossoms?"   → a memory about the Resplendent Quetzal
            #
            # Neither a stricter gate threshold nor `Recall.decided` separates those from the good
            # ones: the worst held-out answer scored 0.394, above the median 0.388 of the correct
            # in-sample ones, and `decided` was True for every case in both populations. Relevance
            # is not correctness, and a score that cannot tell them apart cannot be the licence to
            # speak.
            #
            # The in-sample gain was mostly an artifact besides — it is the store returning the
            # answer whose *text* sits nearest the question, which is the echo `memory.py`
            # deliberately avoids by not encoding cues, and it does not transfer.
            #
            # So the guard stays. Recall still reaches the gate on statement turns (see
            # `_recall_through_gate`, which really was dead), and a question NJP cannot ground is
            # still answered with silence rather than with the nearest thing in the store.
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

            # 4. Recall, but only what is actually *about* this turn.
            #
            # This line used to be `bits.append(best.text)` unconditionally — the single place
            # where a true, well-established, entirely unrelated memory got attached to a reply
            # because it happened to be the nearest thing in the store. Truth is not relevance,
            # and the gate is where the difference is enforced: below threshold the memory is not
            # down-weighted, it is not seen.
            self._recall_through_gate(thought, act, bits)
            if thought.reading is not None and thought.reading.has_latent:
                wants = "; ".join(w.want for w in thought.reading.latent[:2])
                bits.append(f"you usually also want: {wants}")
            return " — ".join(bits)[:1000]
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _is_echo(thought: NJPThought) -> bool:
        """Did this turn hand the *question* back instead of answering it?

        Scoped to turns that asked something, and that is not a detail. An acknowledgement of a
        statement — "noted: Sara is a person" — is structurally identical to an echo and is not
        one: it reports what she recorded, which is how a misparse becomes visible on the turn it
        happens rather than three turns later. Blanking it (the first version of this did) meant
        nothing was remembered on any statement turn, so nothing was ever promoted out of
        episodic memory, and two integration tests that had been green went red. The failure this
        check exists for is narrower than it first looks: *asked something, got the question
        back*.
        """
        try:
            from nyxara.njp.relevance import is_meta_commentary
            grounding = getattr(thought.percept, "grounding", None)
            if getattr(grounding, "triples", None):
                return False        # she recorded something; saying so is not an echo
            return is_meta_commentary(thought.answer, thought.stimulus)
        except Exception:  # noqa: BLE001
            return False

    def _act(self, thought: NJPThought) -> Any:
        """What the Master was *doing* with this turn. Prior to what it is about."""
        if self._speech is None:
            return None
        try:
            return self._speech.read(thought.stimulus, intent=thought.intent)
        except Exception:  # noqa: BLE001
            return None

    def _answer_socially(self, thought: NJPThought, act: Any) -> str:
        """Answer a social or reflexive turn from the relationship and her own state.

        Short, and deliberately so. A greeting wants a greeting back; it does not want a report,
        a fact, or — worst of the three — a statement that she has understood the greeting.
        Where the reading is genuinely ambiguous she asks which was meant rather than asserting
        a comprehension she does not have, which is the calibrated form of the same honesty the
        rest of this brain applies to facts.
        """
        try:
            from nyxara.njp.relevance import SpeechAct
            kind = getattr(act, "kind", "")
            if getattr(act, "ambiguous", False) and kind not in (
                    SpeechAct.GREETING, SpeechAct.THANKS, SpeechAct.FAREWELL):
                return "Tum mera current state pooch rahe ho, ya kuch karne ko keh rahe ho?"
            if kind == SpeechAct.GREETING:
                return "Hii Master 👋"
            if kind == SpeechAct.FAREWELL:
                return "Theek hai Master — main yahin hoon."
            if kind == SpeechAct.THANKS:
                return "Koi baat nahi, Master."
            if kind == SpeechAct.SOCIAL_CHECKIN:
                return self._self_state(mood=True)
            if kind == SpeechAct.STATE_QUERY:
                return self._self_state(mood=False)
            return self._self_state(mood=False)
        except Exception:  # noqa: BLE001
            return "Hii Master 👋"

    def _self_state(self, *, mood: bool) -> str:
        """Her actual state, read off her own organs. Never a fact about the world.

        Every number in here is one this brain already maintains, so "how are you" is answered by
        introspection rather than by retrieval — which is the distinction the transcript's failure
        turned on. If the organs are absent she says the honest small thing instead of reaching
        for something impressive.
        """
        bits: List[str] = []
        try:
            cells = int(getattr(self.fabric, "n_cells", 0) or 0)
            synapses = int(getattr(self.fabric, "n_synapses", 0) or 0)
            if cells:
                bits.append(f"{cells} cells / {synapses} synapses")
            if self.genesis is not None:
                stats = self.genesis.stats()
                if stats.get("concepts"):
                    bits.append(f"{stats['concepts']} concepts at {stats['compression']}× "
                                f"compression")
            if self.beliefs is not None:
                known = len(self.beliefs.known())
                unknown = len(self.beliefs.unknown())
                if known or unknown:
                    bits.append(f"{known} beliefs held, {unknown} open")
            if self.goals is not None:
                try:
                    active = getattr(self.goals, "active", None)
                    top = active() if callable(active) else None
                    if top:
                        bits.append(f"working on: {getattr(top[0], 'text', top[0])}")
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        if not bits:
            return ("Main theek hoon Master — abhi shuruaat hi hai, mere paas batane layak "
                    "kuch bana nahi hai." if mood else
                    "Abhi kuch khaas nahi kar rahi — sun rahi hoon.")
        head = "Main theek hoon Master. " if mood else "Abhi: "
        return head + "; ".join(bits[:3]) + f" — {self.turns} turns."

    def _recall_through_gate(self, thought: NJPThought, act: Any,
                            bits: List[str]) -> None:
        """Attach a recalled memory only if it is about this turn.

        With no gate installed this falls back to the old unconditional behaviour rather than
        silently dropping recall — a missing organ should cost her the improvement, not the
        faculty.
        """
        try:
            rec = thought.percept.recall if thought.percept else None
            if rec is None:
                return
            # `Recall` carries `hit` (the nearest trace) and `associated` ([(key, weight)]).
            #
            # This used to read `rec.traces` and `rec.best`, and `Recall` has never had either
            # field — so both `getattr`s returned None, `candidates` was always empty, and the
            # method returned here on every turn of every kind. The holographic memory reached an
            # answer exactly never, while `study.py` wrote to it twice per corpus pair. Because
            # both reads were `getattr` with a default, nothing raised and nothing logged; the
            # only symptom was recall silently contributing nothing.
            candidates = [rec.hit] if rec.hit is not None else []
            store = getattr(self.memory, "recall_key", None)
            if store is not None:
                for key, _weight in (rec.associated or [])[:4]:
                    trace = store(key)
                    if trace is not None and trace is not rec.hit:
                        candidates.append(trace)
            if not candidates:
                return
            best = candidates[0]
            if self.gate is None:
                text = str(getattr(best, "text", "") or "")
                if text:
                    bits.append(text)
                return
            admitted = self.gate.filter(
                candidates, query=thought.stimulus, act=act, world=self.world,
                goals=self._goal_texts(), thread=self.replay[-4:])
            thought.relevance = list(self.gate.last)
            if admitted:
                bits.append(admitted[0].memory)
        except Exception:  # noqa: BLE001
            pass

    def _goal_texts(self) -> List[str]:
        try:
            if self.goals is None:
                return []
            active = getattr(self.goals, "active", None)
            rows = active() if callable(active) else []
            return [str(getattr(g, "text", g)) for g in list(rows)[:8]]
        except Exception:  # noqa: BLE001
            return []

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
            #
            # Confidence is passed through `revise_confidence`, which is monotonic: it may only
            # exceed what it started at when independent evidence, real relevance and consistency
            # all say so. The transcript's `0.80 → 1.00` came from none of those — it came from
            # a conclusion having been reached, and "I thought about it harder" is not evidence
            # about the world. Depth may lower confidence and may never raise it.
            from nyxara.njp.relevance import revise_confidence
            relevance = 1.0
            scores = list(getattr(thought, "relevance", None) or [])
            if scores:
                relevance = max((s.total for s in scores), default=0.0)
            judgement = thought.judgement
            supports = int(len(getattr(judgement, "supports", None) or [])) if judgement else 0
            confidence = revise_confidence(
                thought.confidence, independent_evidence=supports,
                relevance=relevance,
                consistent=not bool(getattr(judgement, "refutations", None)))
            if thought.answer and thought.verified:
                thought.epistemic = Epistemic.KNOWN
                thought.epistemic_confidence = confidence
            elif thought.answer:
                thought.epistemic = Epistemic.BELIEVED
                thought.epistemic_confidence = confidence
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
                # Grade the *route* the answer came by, not just "reasoning". A stored fact read
                # back and a two-hop composition fail at different rates, and a single posterior
                # lets the reliable one subsidise the shaky one — which is backwards, because the
                # composed answer is exactly where the caller most needs the warning. The label
                # is already on the derivation; this only records the outcome against it.
                derivation = getattr(thought, "derivation", None)
                if self.self_model is not None and derivation is not None:
                    from nyxara.njp.selfmodel import mode_of
                    self.self_model.observe(mode_of(derivation), float(correct))
                if self.meta is not None:
                    # Every arm that was spent this turn is graded, not just the one. Rewarding
                    # `settle_steps` alone meant the other two knobs accumulated choices and
                    # never a single outcome, so their means stayed at the prior for ever and
                    # `best()` could never name a winner. A knob that is turned and never scored
                    # is not being learned about, it is being varied.
                    for arm in ("settle_steps", "recall_k", "reason_depth"):
                        self.meta.reward(arm, float(correct))
        except Exception:  # noqa: BLE001
            pass

    def _score_prediction(self, thought: Any, *, correct: float, actual: Any) -> Any:
        """Close the loop on this turn's expectation, with the evidence the diagnosis needs.

        The expectation is registered **here**, at the moment the Master grades the turn, rather
        than speculatively when the turn was taken. Registering up front is what produced the
        original symptom — 113 predictions, 0 scored — because most turns are never graded, so
        every one of them left an expectation that nothing could ever observe. Registering on
        arrival of the verdict means ``predictions`` counts things that were actually decided, and
        an ungraded turn costs nothing.

        What she is scored against is what she *said*; what says whether it was right is the
        Master's ``correct``/``actual``. The two come from different places, which is the whole
        requirement.
        """
        try:
            stimulus = str(getattr(thought, "stimulus", "") or "")
            if not stimulus:
                return None
            percept = getattr(thought, "percept", None)
            grounding = getattr(percept, "grounding", None)
            key = f"{getattr(thought, 'cycle_id', '') or stimulus}:answer"
            said = str(getattr(thought, "answer", "") or "")
            self.predictor.predict(
                key, said or "<unknown>",
                confidence=float(getattr(thought, "epistemic_confidence", 0.0) or 0.0),
                organ=("grounding" if getattr(grounding, "answer", None) else "fabric"))
            return self.predictor.observe(
                key,
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
                            ("attention", self.attention), ("readout", self.readout),
                            ("loop", self.loop),
                            ("concepts", self.genesis), ("universe", self.universe),
                            ("designer", self.designer), ("beliefs", self.beliefs),
                            ("metareason", self.metareason), ("predictive", self.predictive),
                            ("agency", self.agent), ("curriculum", self.curriculum),
                            ("calculate", self.calculator),
                            ("field", self.field), ("learner", self.learner),
                            ("adversary", self.adversary)):
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
                            ("attention", self.attention), ("readout", self.readout),
                            ("concepts", self.genesis), ("universe", self.universe),
                            ("designer", self.designer), ("beliefs", self.beliefs),
                            ("metareason", self.metareason), ("predictive", self.predictive),
                            ("agency", self.agent),
                            ("field", self.field), ("learner", self.learner),
                            ("adversary", self.adversary)):
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
            # The V.04 organs. `concepts` rebuilds its whole hierarchy from the restored
            # observations rather than trusting a stored one, so a sidecar written by an older
            # clustering cannot resurrect concepts the current rules would never form.
            for key, organ in (("concepts", self.genesis), ("universe", self.universe),
                               ("designer", self.designer), ("beliefs", self.beliefs),
                               ("metareason", self.metareason), ("predictive", self.predictive),
                               ("agency", self.agent), ("field", self.field),
                               ("learner", self.learner)):
                if d.get(key) and organ is not None:
                    organ.load_dict(d[key])
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born brain
            pass
